# DeepSeek-V4-Flash · NVIDIA Day0 特性适配设计（定稿）

> 范围：**仅 NVIDIA**（`engine == "vllm"`，H20 单机 8 卡，场景键 `v4flash-nv-h20-8`）
> 模型：`DeepseekV4ForCausalLM` / `DeepSeek-V4-Flash`
> 目标：让 wings-control 生成下方「目标启动命令」中相对当前产物**新增/变化的三项**——投机 method、IndexCache（KV 稀疏）、native KV 卸载。
> Ascend 路径本次**不动**。

---

## 1. 目标启动命令

```bash
vllm serve /usr/local/serving/models/ \
  --trust-remote-code \
  --port 18000 \
  --gpu-memory-utilization 0.9 \
  --served-model-name DeepSeek-V4-Flash \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --seed 42 \
  --speculative_config '{"method":"mtp","num_speculative_tokens":1}' \
  --kv_offloading_backend native \
  --kv_offloading_size 200 \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
```

基础参数（fp8 / block 256 / EP / TP=8 / tokenizer 等）已由 [nvidia_default.json](../../wings_control/config/defaults/nvidia_default.json#L321) 与现有逻辑产出，本设计只补三项差异。

---

## 1.1 实际生成命令（dry_run 实测产出）

> 命令：`python dry_run.py --scenario v4flash-nv-h20-8`
> 三开关全开（`enable_speculative_decode` + `enable_sparse` + `LMCACHE_OFFLOAD`，`LMCACHE_MAX_LOCAL_CPU_SIZE=25`、`device_count=8`）。

**引擎主命令（去除端口/路径等部署细节）：**
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --trust-remote-code --max-model-len 4096 --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel --tokenizer-mode deepseek_v4 \
  --served-model-name DeepSeek-V4-Flash --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 --max-num-batched-tokens 4096 --max-num-seqs 32 --seed 0 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --kv_offloading_backend native --kv_offloading_size 200 &
```

三项特性逐一对账：

| 特性 | 生成片段 | 说明 |
|---|---|---|
| 投机 method | `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'` | D1 ✓；裸 `mtp`，无 `deepseek_mtp` |
| IndexCache | `--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'` | D2 ✓；不装补丁 |
| native 卸载 | `--kv_offloading_backend native --kv_offloading_size 200` | D3/D4 ✓；`200 = device_count(8) × LMCACHE_MAX_LOCAL_CPU_SIZE(25)` |
| LMCache 互斥 | （无 `LMCacheConnectorV1` / `--kv-transfer-config`） | native 卸载命中 → 不注入 LMCache 连接器 |

**崩溃 fallback 行（自动剥除全部三特性，退回基线）：**
```bash
python3 -m vllm.entrypoints.openai.api_server ... --tensor-parallel-size 8 &
# ↑ 无 --speculative-config / --hf-overrides / --kv_offloading_backend
```

### 开关门控（非写死，逐特性跟随上层开关）

| 特性 | 上层开关 | 关闭时的产物 |
|---|---|---|
| IndexCache（`--hf-overrides`） | `enable_sparse`（CLI `--enable-sparse` / env `ENABLE_SPARSE`） | 不出 `--hf-overrides` |
| native KV 卸载 | `LMCACHE_OFFLOAD=true` | 不出 `--kv_offloading_*` |
| 投机推理 | `enable_speculative_decode`（`--enable-speculative-decode`） | 不出 `--speculative-config` |

实测：关掉 `enable_sparse` + `LMCACHE_OFFLOAD`（仅留投机）后，主命令只剩
`--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`，另两项如期消失。
对应回归见 [tests/test_v4flash_nv_day0.py](../../tests/test_v4flash_nv_day0.py) 的 `TestSparseSwitchGating`。

---

## 2. 设计决策（已全部定案）

| 编号 | 议题 | 决策 |
|---|---|---|
| D1 | 投机 method 字面值 | **NV V4-Flash 用 `mtp`**；Ascend V4-Flash 维持 `deepseek_mtp`（官方模板），按 engine 收口 |
| D2 | IndexCache 补丁 | **不安装 `indexcache` 补丁**；仅输出 `--hf-overrides`（引擎内置 IndexCache） |
| D3 | native 卸载触发开关 | **复用 `LMCACHE_OFFLOAD`**（沿用现有 offload 总开关，不引新开关） |
| D4 | `--kv_offloading_size` 取值 | **做乘法**：`device_count(本节点卡数) × LMCACHE_MAX_LOCAL_CPU_SIZE`；env 未设/非法 → 默认 `200`（不乘） |
| D5 | 适配范围 | **仅 NV**（`engine == "vllm"`）；Ascend 不动 |

---

## 3. 现状差距（NV V4-Flash）

| 特性 | 目标 | 当前产物 | 缺口 |
|---|---|---|---|
| 投机 method | `"method":"mtp"` | `"method":"deepseek_mtp"`（num=1 已对齐） | method 字面值不同 |
| IndexCache | `--hf-overrides '{"use_index_cache":true,"index_topk_freq":4}'` | 无（走 FP8 分支） | V4 不在 IndexCache 范围；`use_index_cache` 为新 key |
| native 卸载 | `--kv_offloading_backend native --kv_offloading_size N` | 无 | native flag 仓库零匹配 |

代码佐证：
- 投机 method 来自 [`_resolve_mtp_method`:2502](../../wings_control/engines/vllm_adapter.py#L2502)（`DeepseekV4ForCausalLM → deepseek_mtp`）+ [`_build_speculative_cmd`:2646](../../wings_control/engines/vllm_adapter.py#L2646)（V4-Flash num=1）。
- IndexCache：`DeepseekV4ForCausalLM` 不在 [`INDEXCACHE_ARCHS`:39](../../wings_control/utils/model_utils.py#L39)，故 [`_build_kv_sparse_cmd`:2780](../../wings_control/engines/vllm_adapter.py#L2780) 落 FP8 else 分支；载荷恒为 `{"index_topk_freq":N}`，无 `use_index_cache`。
- 卸载：`--kv_offloading_*` 零匹配；现有 [`_apply_deepseek_v4_cpu_offload`:2105](../../wings_control/engines/vllm_adapter.py#L2105) 是 ascend 专用（[`_is_deepseek_v4_cpu_offload_params`:1505](../../wings_control/engines/vllm_adapter.py#L1505) 首行 `engine!="vllm_ascend"→False`），用 `kv_transfer_config`，机制不同。

---

## 4. 详细设计

所有改动收敛在 `engine=="vllm"` + `_is_deepseek_v4_flash_params(params)`，不影响其它模型与 Ascend。

### 4.1 投机 method → `mtp`（D1）
- **不改** `_resolve_mtp_method` 全局映射（其 `deepseek_mtp` 是 Ascend 官方模板所需，见 [2506 注释](../../wings_control/engines/vllm_adapter.py#L2506)）。
- 在 [`_build_speculative_cmd`:2646-2651](../../wings_control/engines/vllm_adapter.py#L2646) 的 `*_mtp` 分支按 engine 收口覆盖：
  ```python
  if engine == "vllm" and _is_deepseek_v4_flash_params(params) and strategy.endswith("_mtp"):
      strategy = "mtp"
  ```
- 结果：NV V4-Flash → `'{"method":"mtp","num_speculative_tokens":1}'`；Ascend V4-Flash 不变。

### 4.2 IndexCache（D2，无补丁）
- 在 [`_build_kv_sparse_cmd`:2746](../../wings_control/engines/vllm_adapter.py#L2746) NV 分支、`INDEXCACHE_ARCHS` 判定**之前**插入 V4-Flash 专用 case：
  ```python
  if engine == "vllm" and _is_deepseek_v4_flash_params(params):
      return " --hf-overrides '{\"use_index_cache\": true, \"index_topk_freq\": 4}'"
  ```
- **不把 `DeepseekV4ForCausalLM` 加入 `INDEXCACHE_ARCHS`**——这样 [`_collect_indexcache_patch_features`:449](../../wings_control/core/wings_entry.py#L449) 因架构不在白名单天然返回 `[]`，**不装补丁**，正好满足 D2。
- fp8 KV 由 nvidia_default.json 基础参数提供，与 `--hf-overrides` 共存（目标命令两者都有），无需在此分支再注入 `kv_cache_dtype`。

### 4.3 native KV 卸载（D3 + D4）
1. **取值公共函数**（ascend 与 native 共用，避免数值漂移）：
   ```python
   def _resolve_v4_flash_offload_gb(params) -> int:
       raw = os.getenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "").strip()
       per_card = int(raw) if raw.isdigit() else None        # 非法/未设 → None
       if per_card is None:
           return 200                                        # 默认平铺，不乘
       if _is_deepseek_v4_flash_params(params):
           return (_safe_int(params.get("device_count")) or 1) * per_card  # ← 乘法(整节点口径)
       return per_card
   ```
   现有 [`_apply_deepseek_v4_cpu_offload`:2130-2147](../../wings_control/engines/vllm_adapter.py#L2130) 的同段逻辑改为调用它，保证 ascend `cpu_swap_space_gb` 与 NV `--kv_offloading_size` 同源同值。
2. **CLI 生成函数**：
   ```python
   def _build_kv_offload_cmd(params, engine) -> str:
       if engine != "vllm" or not _is_deepseek_v4_flash_params(params):
           return ""
       if not get_lmcache_env():        # D3：复用 LMCACHE_OFFLOAD 总开关
           return ""
       n = _resolve_v4_flash_offload_gb(params)
       return f" --kv_offloading_backend native --kv_offloading_size {n}"
   ```
3. **与 LMCache env 互斥**：在 [`_build_cache_env_commands`:665](../../wings_control/engines/vllm_adapter.py#L665) 的跳过 LMCache 守卫中，补上「NV V4-Flash + LMCACHE_OFFLOAD」分支——命中 native 卸载时不再导出 `LMCacheConnectorV1` 相关 env，二者只存其一。

### 4.4 拼装与 fallback
- 三段接入 [`_build_vllm_single_script`:2853](../../wings_control/engines/vllm_adapter.py#L2853) 的 `exec` 行，顺序：`speculative` → `kv_sparse(--hf-overrides)` → `kv_offload`。
- 高级特性崩溃后的 fallback 重启命令必须把 `--hf-overrides` 与 `--kv_offloading_*` 一并剥除（与现有剥 `--speculative-config` 同逻辑），退回基线命令。

---

## 5. 改动清单（已实现）

| 文件 | 改动 |
|---|---|
| [vllm_adapter.py](../../wings_control/engines/vllm_adapter.py) | ① `_build_speculative_cmd`：NV V4-Flash 覆盖 method=`mtp`<br>② `_build_kv_sparse_cmd`：NV V4-Flash IndexCache 分支（`use_index_cache`+`index_topk_freq`）<br>③ 新增 `_resolve_v4_flash_offload_gb` 并被 `_apply_deepseek_v4_cpu_offload` 复用<br>④ 新增 `_build_kv_offload_cmd`<br>⑤ `_build_cache_env_commands`：NV V4-Flash native 卸载时跳过 LMCache env<br>⑥ `_build_vllm_single_script`：拼接 + fallback 剥除<br>⑦ **（实现中补）** `resolve_speculative_strategy`：NV V4-Flash native 卸载与 MTP 共存，不被 LMCache 误降级为 suffix |
| [config_loader.py](../../wings_control/core/config_loader.py) | **（实现中补）** 新增 `_is_deepseek_v4_flash_nv`；`_set_kv_cache_config` 对 NV V4-Flash 跳过 `LMCacheConnectorV1` 注入（改用 native CLI flag） |
| [wings_entry.py](../../wings_control/core/wings_entry.py) | `_build_advanced_feature_fallback_cmd`：LMCACHE_OFFLOAD 时置 `_wings_fallback_no_kv_offload` 抑制 native 卸载；`INDEXCACHE_ARCHS` 不含 V4，补丁聚合天然不装（D2） |
| [dry_run.py](../../dry_run.py) | `v4flash-nv-h20-8` 场景开启 sparse+offload；`setup_env` 支持 `enable_kv_offload` / `lmcache_max_local_cpu_size` |
| `INDEXCACHE_ARCHS` | **不改**（D2，确保不装补丁） |
| [tests/test_v4flash_nv_day0.py](../../tests/test_v4flash_nv_day0.py) | 新增 11 条单测（method/IndexCache/无补丁/卸载乘法/默认值/开关/互斥/fallback/ascend 不回归） |

> **实现验证**：`python dry_run.py --scenario v4flash-nv-h20-8` 产出主命令含
> `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`、
> `--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'`、
> `--kv_offloading_backend native --kv_offloading_size 200`（=8×25），且无 `LMCacheConnectorV1`；
> fallback 行三特性全剥除。新增 11 单测全过，全量回归无新增失败（既有 11 项失败为 master 上预存）。

---

## 6. 测试要点
- UT：NV V4-Flash → `"method": "mtp"`；**Ascend V4-Flash → `"method": "deepseek_mtp"` 不回归**。
- UT：NV V4-Flash → `--hf-overrides` 含 `use_index_cache` 与 `index_topk_freq`；**不触发 indexcache 补丁安装**（advanced_features / install.py 不含 indexcache）。
- UT：NV V4-Flash + `LMCACHE_OFFLOAD=true` → 含 `--kv_offloading_backend native`，`--kv_offloading_size` = `device_count × LMCACHE_MAX_LOCAL_CPU_SIZE`；env 未设 → `200`。
- UT：native 卸载命中时，脚本**不含** LMCache env（`LMCACHE_OFFLOAD` export / `LMCACHE_CONFIG_FILE`）。
- 回归：现有 [test_config_loader_engine_selection.py:511-538](../../tests/test_config_loader_engine_selection.py#L511) 断言「V4-Flash 不带 `--hf-overrides`」将失效，**需同步更新为新预期**。
- 回归：GLM-5.1 / DeepseekV32 的 IndexCache 载荷不被 `use_index_cache` 污染；ascend `cpu_swap_space_gb` 数值不变。
- 端到端：dry_run `v4flash-nv-h20-8` 产物逐字段比对目标命令；fallback 段不含三特性。
