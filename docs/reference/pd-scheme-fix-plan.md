# PD 分离方案局限性 —— 修复方案（仅设计，未改代码）

| 项 | 内容 |
|----|------|
| 日期 | 2026-06-13 |
| 范围 | 针对 [pd-scheme-limitations.md](pd-scheme-limitations.md) 的 L1/L3/L4/L5(+L2) 给可落地方案 |
| 原则 | 每条含 现状→方案→改动点→兼容门控→风险→验证→工时 |
| **实施状态（2026-06-13）** | **L4 / L3 / L2 / L5 已实现并验证**（`tests/pd_external_lb_verify.py` 新增层 F，总 **61 PASS / 0 FAIL**；非 PD 字节级不变）。**L1 暂缓**（真机依赖，hf-overrides 合并未做）。下文各条标注 ✅已实现 / ⏸暂缓。 |
| 统一约束 | ① 一切新逻辑**严格门控到 PD external-lb**（`_pd_external_lb` 命中），非 PD 字节级不变；② 注册表来自模块级缓存，凡读出再合并**必须 deepcopy**（L6 教训）；③ 每条都以 `dry_run.py --pd` + `tests/pd_external_lb_verify.py` 回归。 |

## 总览

| # | 局限 | 方案一句话 | 性质 | 工时 | 风险 |
|---|------|-----------|------|:----:|:----:|
| L4 | A2/A3 单条目装不下 | 注册表加 `platform_overrides` 子块，loader 按平台深合并 | schema+loader | 1d | 低 |
| L3 | 共用 env 笨拙叠加 | 加 `common_env` 槽 + PD 脚本对合并后 env 跑 dedupe | schema+1处 | 0.5d | 低 |
| L1 | IndexCache 在 PD 被丢 | `sparse_args` 传入 PD fork 并按 service 追加（须合并 hf-overrides） | 2处 | 1~2d | 中(真机) |
| L5 | 注入器回填覆盖注册表 | 加「注册表键全部存活」回归断言锁行为（治本可选改注入器门控） | 测试为主 | 0.5d | 低 |
| L2 | 角色级 KV extra 不可表达 | 加 `prefill/decode.extra_config` 角色位，合并进 kv extra | schema+1处 | 0.5d | 低 |

---

## L4. 注册表加平台维度（A2/A3） ✅已实现

**现状**：`entry = registry.get(arch) or registry.get("default")`（[config_loader.py:1037](../wings_control/core/config_loader.py)）只按架构取，无平台分支；平台由 `WINGS_ASCEND_PLATFORM` 独立解析，只驱动 env 块。

**方案（平台 overlay，最小重复 + 向后兼容）**：基条目放平台无关值，新增可选 `platform_overrides`，命中平台时深合并覆盖：
```jsonc
"DeepseekV4ForCausalLM": {
  "connector": "...", "kv_port": {...},
  "common":  { /* 平台无关 */ },
  "prefill": { "engine": { "max_num_batched_tokens": 8192 /* 默认/A3 */ } },
  "decode":  { "engine": { "max_num_batched_tokens": 120 } },
  "platform_overrides": {
    "a2": { "prefill": { "engine": { "max_num_batched_tokens": 4096 } },
            "decode":  { "engine": { "max_num_batched_tokens": 60, "max_num_seqs": 30 } } }
  }
}
```

**改动点**（`_apply_pd_external_lb`，[config_loader.py:1032+](../wings_control/core/config_loader.py)）：
1. `entry = copy.deepcopy(registry.get(arch) or registry.get("default"))`（deepcopy 防污染缓存）。
2. 解析平台：新增 `_resolve_ascend_platform()`（读 `WINGS_ASCEND_PLATFORM`/`ASCEND_PLATFORM` 等，复用现有第 ~1219 行的探测口径），得 `plat`。
3. `ov = entry.pop("platform_overrides", {}).get(plat)`；若有则对 `entry["common"/"prefill"/"decode"]` 做**深合并**（overlay 优先）。
4. 后续 `merged_engine`/kv/env 逻辑不变，自然吃到平台值。

**兼容/门控**：无 `platform_overrides` 的条目（GLM5/V3.2/Qwen3.5/default）行为**完全不变**；平台解析失败/为空 → 不应用 overlay（退化为基条目=A3）。

**风险**：低。深合并需对 dict 递归（`additional_config` 等）；务必 deepcopy。

**验证**：dry_run 给 `v4flash` 增设 `platform: "a2"` 场景，断言 P=4096 / D=60·30；harness 加「a2 overlay 生效 / a3 走基值」两条。

**工时**：~1 人日。

---

## L3. 加 `common_env` 槽 + PD 脚本对合并 env 去重 ✅已实现

**现状**：注册表 env 只有 `prefill.env`/`decode.env`，角色 env 在 PD 脚本里**追加在 common_env 之后**（能覆盖，bash 后者生效），但 ① 无共用槽（P/D 都要的须各写一遍）；② 角色 env 不参与 dedupe（base 值与覆盖值并存，实测 MLAPO 两次）。

**方案**：
1. **schema 加 `common_env`**（P/D 共用，角色 env 仍可覆盖）：
```jsonc
"GlmMoeDsaForCausalLM": {
  "common_env": { "HCCL_BUFFSIZE": "256", "ASCEND_AGGREGATE_ENABLE": "1",
                  "ACL_OP_INIT_MODE": "1", "ASCEND_A3_ENABLE": "1",
                  "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT": "480" },
  "prefill": { "env": { "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1", ... } },
  "decode":  { "env": { ... } }
}
```
2. **loader 合并**（`_apply_pd_external_lb`，[config_loader.py:1069](../wings_control/core/config_loader.py)）：
   `cmd_known_params["_pd_env"] = {**entry.get("common_env", {}), **entry.get(role_key, {}).get("env", {})}`（角色覆盖共用）。
3. **PD 脚本对合并后 env 去重**（`_build_vllm_pd_external_lb_script`，[vllm_adapter.py:2917-2920](../wings_control/engines/vllm_adapter.py)）：把角色 env 追加进 `env_lines` 后，对**整段** `env_lines` 跑一次 `dedupe_env_exports(env_lines)`，使覆盖值收口（保留最后一条=注册表值，丢 base 重复）。

**兼容/门控**：无 `common_env` 的条目不受影响；dedupe 对累加型（`LD_LIBRARY_PATH`）天然跳过（`classify_env_export`），最终值不变只去重。

**可选增强（unset 语义）**：约定 `common_env` 里值为 `null` → 渲染 `unset VAR`（对齐 engine_config 的 `null` 删键语义）。非必须。

**风险**：低。dedupe 已有 10 用例覆盖；这里只是把它的作用域从 common_env 扩到 common+role。

**验证**：给 GLM5 `common_env` 填 `HCCL_BUFFSIZE=256`，dry_run 断言脚本里只剩一条 `HCCL_BUFFSIZE=256`（无 1024 残留）。

**工时**：~0.5 人日。

---

## L1. IndexCache（sparse_args）接入 PD fork ⏸暂缓（真机依赖，未实现）

**现状**：`build_start_script` 算了 `sparse_args`，但 PD 分支 `_build_vllm_pd_external_lb_script(params, cmd, common_env_cmds, pd_ext)` **没传**（[vllm_adapter.py:2992](../wings_control/engines/vllm_adapter.py)），IndexCache `--hf-overrides` 丢失。

**方案**：
1. 形参加 `sparse_args`：`_build_vllm_pd_external_lb_script(params, cmd, common_env_cmds, pd_ext, sparse_args)`；调用处（2992）传入。
2. fork 循环里每个 `vllm serve` 末尾追加 `sparse_args`（IndexCache 的 `--hf-overrides` 各 service 相同，可在循环前并入 `svc_cmd`）。

**关键风险/必做**：**`--hf-overrides` 冲突**。PD 下 `_ensure_pd_head_dim`（[config_loader.py:1279](../wings_control/core/config_loader.py)）也会注 `--hf-overrides '{"head_dim":N}'`。若再 append IndexCache 的 `--hf-overrides`，会出现**两个 `--hf-overrides`**（vLLM 取其一 → 另一个丢）。故方案必须**合并两者的 hf_overrides JSON**为一个 flag，而非各 append。
- 落点：让 IndexCache 路径也走 `params['hf_overrides']`（与 head_dim 同一出口，由 `_build_vllm_cmd_parts` 统一渲染一个 `--hf-overrides`），而不是返回独立 flag 字符串。或在 PD 脚本组装时检测重复 key 合并。

**兼容/门控**：仅 `should_emit_sparse` 时追加；FP8 路径 `sparse_args=""`（其效果已在 engine_config/cmd），append 空串无副作用。

**真机依赖**：IndexCache + Mooncake PD 能否共存未知（属 dry-run 盲区 L15）；方案只解决"参数下发"，**生效性须真机验证**。

**验证**：dry_run 给 `glm5` 场景加 `enable_sparse=true`，断言每个 fork 的 serve 含合并后的单个 `--hf-overrides`（同时含 head_dim 与 IndexCache 键）。

**工时**：~1~2 人日（含 hf-overrides 合并 + 真机）。

---

## L5. 锁住「注册表键全部存活」（治本/防回退） ✅已实现（断言方案）

**现状**：注册表权威靠 `_prepare_engine_config` 末尾的「注入器后重申」（[本分支已加]）。**未来任一注入器新增 `_force_set_*` 仍可能再覆盖注册表值**，且不一定被现有断言抓到。

**方案（务实优先）—— 加回归断言锁行为**（`tests/pd_external_lb_verify.py`）：
- 对每个已注册架构跑一遍 PD 生成，断言：注册表 `common`+角色 `engine` 里的**每个键**都按注册值出现在最终 serve 命令（`null` 键则断言**不出现**）。
- 一旦未来注入器回填覆盖，CI/harness 立即红，定位到具体键。

**方案（治本/可选）—— 注入器 PD 感知**：在 `_apply_*_engine_defaults` 入口 `if params.get("_pd_external_lb"): return`（或仅跳过其中的 `_force_set_*`）。
- 收益：不再依赖"重申"补丁，注册表天然权威。
- 代价：注入器也设了注册表没覆盖的键（tokenizer_mode/block_size/api_server_count 等）→ 跳过会丢；故需**先把注册表补全**或保留这些"非冲突"默认。风险比断言方案高，建议二期。

**风险**：断言方案零风险（只读校验）；注入器门控方案中风险（需保证注册表完整）。

**工时**：断言 ~0.5 人日；注入器门控另算（建议二期）。

---

## L2. 角色级 KV `extra_config`（次要） ✅已实现

**现状**：注册表 `extra_config` 经 `_build_pd_external_lb_kv` 对 P/D 同时写（[config_loader.py:1007-1009](../wings_control/core/config_loader.py)）；官方 Qwen3.5 的 `kv_buffer_device:"npu"` 只在 consumer(D)。

**方案**：schema 加可选 `prefill.extra_config`/`decode.extra_config`；`_build_pd_external_lb_kv` 里 `extra.update(entry.get("extra_config", {}))` 后再 `extra.update(entry.get(role_key, {}).get("extra_config", {}))`（角色级覆盖/追加）。

**兼容**：无角色 extra 的条目不变。**工时**：~0.5 人日。

---

## 实施顺序建议

1. **L5 断言**（先锁住现有修复不回退，0.5d，零风险）→
2. **L4 平台维度**（A2 上线前必须，1d）→
3. **L3 common_env + dedupe**（对齐官方共用 env，0.5d）→
4. **L2 角色 extra**（顺带，0.5d）→
5. **L1 IndexCache**（依赖真机验证，单独排期 1~2d）。

> 统一验证闭环：每条改完跑 `python dry_run.py --pd glm5 && python dry_run.py --pd v4flash`（L4 另加 a2 场景）+ `python tests/pd_external_lb_verify.py`，并对照 [pd-dryrun-vs-official-report.md](pd-dryrun-vs-official-report.md) 的官方列确认无回退。所有改动严格门控 `_pd_external_lb`，保证非 PD 路径字节级不变。
