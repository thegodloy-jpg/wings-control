# PD external-lb 注册表权威化（registry-authoritative）设计方案

| 项 | 内容 |
|----|------|
| 状态 | **提案 / 待实现**（代码未改；本文为设计） |
| 日期 | 2026-06-17 |
| 适用 | 仅 PD external-lb 路径（`PD_ROLE∈{P,D}` 且 `DP_SIZE>1`）；引擎 `vllm` / `vllm_ascend` |
| 目标 | 让 `pd_config.json` 注册表对「自己声明的 engine 键」在 PD 场景**权威生效**，不被平台模板反射性灌入的 `--flag` 静默顶掉；同时保留运维「有意调优」的逃生舱 |
| 关联 | 注册表权威机制现状见 [设计文档 §13.2](../../xuqiu/deepseek-v32-pd-disaggregation.md)；字段对齐见 [pd-a3-official-alignment-report.md](pd-a3-official-alignment-report.md) |

---

## 1. 背景与问题

PD external-lb 的每模型调优值（`max_num_batched_tokens` / `max_num_seqs` / `gpu_memory_utilization` / `compilation_config` …）由 `pd_config.json` 注册表提供，是对齐官方 A3 的「唯一真相源」。

**现状优先级**：`用户显式 > 注册表 > base 默认`。其中「显式」由 [`_detect_explicit_cli_keys`](../../wings_control/core/config_loader.py#L2024) 判定 = **命令行出现 `--flag`（进 `sys.argv`）或对应 ENV 被设**（与值是否等于默认无关，见 `_COMMON_CLI_ENV_MAP`/`_VLLM_CLI_ENV_MAP`）。

**问题**：平台 serving 模板对所有部署**反射性灌一串通用 flag**（真机 jzow306/xlka343 实测）：
```
--gpu-memory-utilization 0.8 --max-num-seqs 256 --max-num-batched-tokens 4096 \
--block-size 16 --enable-chunked-prefill --enable-prefix-caching ...
```
这些在 PD external-lb 下全部被判为「显式」→ **静默顶掉注册表的 decode 调优**。

**验证证据**（`build/verify_registry_override.py`，Qwen3-30B-A3B D 角色）：

| flag | 注册表 | 平台灌入后实际 | 结论 |
|---|---|---|---|
| `--max-num-batched-tokens` | 120 | **4096** | ❌ 被顶掉 |
| `--max-num-seqs` | 60 | **256** | ❌ 被顶掉 |
| `--gpu-memory-utilization` | 0.88 | **0.8** | ❌ 被顶掉 |
| `--compilation-config` | FULL_DECODE_ONLY | FULL_DECODE_ONLY | ✅ 保住（平台没传） |
| `--async-scheduling` | 有 | 有 | ✅ 保住（平台没传） |

decode 被按 prefill 口径（batched 4096 / seqs 256）跑 → 显存/吞吐 profile 全错、可能 OOM。**注册表形同虚设**。

> 根因：注册表"不被覆盖"的充要条件 = 用户/平台「没显式传该键」。平台惯性灌值即等于显式覆盖。

---

## 2. 目标与原则

1. **PD external-lb 下注册表对其声明的 engine 键权威**：压过平台/用户的显式值。
2. **保留「有意调优」逃生舱**：运维确需自定义某键时，能显式放行（而非被一刀切锁死）。
3. **可见**：被注册表盖掉的显式值，**醒目日志**，不静默。
4. **零外溢**：非 PD / standalone PD / Ray / 单机部署**字节级不变**；默认不改变既有 external-lb 行为（opt-in）。
5. **区分「有意 vs 惯性」**：wings 无法从 `sys.argv` 自动区分（两者都是 `--flag`），故用一个**平台模板不会反射性设的 PD 专属信号**来表达「有意」。

---

## 3. 契约（新增两个 env）

| 变量 | 取值 | 语义 |
|------|------|------|
| `PD_REGISTRY_AUTHORITATIVE` | `1`/`true`（默认关） | 开 → PD external-lb 下注册表对其声明键权威（盖过显式） |
| `PD_KEEP_CLI_KEYS` | 逗号分隔 snake_case 键，如 `gpu_memory_utilization,max_num_seqs` | 列出的键**保留命令行/ENV 值**（逃生舱）；仅在 authoritative 开时有意义 |

> `PD_KEEP_CLI_KEYS` 是平台通用模板**不会反射性设**的 PD 专属变量 → 天然把「运维有意调优」与「平台惯性灌值」分开。

---

## 4. 判定逻辑（改造后）

```
前提:external-lb 命中(PD_ROLE∈{P,D} 且 DP_SIZE>1)

对「注册表为本角色声明的 engine 键 K」:
   IF PD_REGISTRY_AUTHORITATIVE 未开(默认):
       用户显式(K) > 注册表(K) > base 默认            ← 现状不变
   ELSE(已开):
       IF K ∈ PD_KEEP_CLI_KEYS 且用户显式传了 K:
           用户显式(K)                                ← 逃生舱
       ELSE:
           注册表(K)（盖过用户显式）+ logger.warning   ← 权威

对「注册表未声明的键」:        用户显式 > base 默认     ← 永不受注册表影响
对「非 external-lb 部署」:      完全不进此逻辑,字节级不变
```

**优先级对照**：

| 场景 | 改造前 | 改造后（authoritative 开） |
|------|--------|------|
| 注册表声明键、未放行 | 用户显式 > 注册表 > 默认 | **注册表 > 用户显式 > 默认** |
| 注册表声明键、`PD_KEEP_CLI_KEYS` 放行 | 同上 | 用户显式 > 注册表 > 默认（回到原序） |
| 注册表未声明键 | 用户显式 > 默认 | 用户显式 > 默认（不变） |

---

## 5. 改动点（2 处 + 2 helper）

### 5.1 `config_loader._apply_pd_external_lb`（[~L1108](../../wings_control/core/config_loader.py#L1108)）
把「哪些键强制用注册表」的决策**全压进 stash**：门控由 `explicit` 改成 `keep_cli`。

```python
# 现状：
for k, v in merged_engine.items():
    if k not in explicit:
        ec[k] = copy.deepcopy(v)
cmd_known_params["_pd_engine_overrides"] = {k: deepcopy(v) for k,v in merged_engine.items() if k not in explicit}

# 改为：
authoritative = _env_true("PD_REGISTRY_AUTHORITATIVE")
keep_cli = _pd_keep_cli_keys() if authoritative else explicit   # 关→keep_cli=explicit，等价原行为
overridden = []
for k, v in merged_engine.items():
    if k in keep_cli:
        continue                                # 放行：保留 CLI/base 值
    if authoritative and k in explicit:
        overridden.append(k)
    ec[k] = copy.deepcopy(v)                    # 否则注册表写入（authoritative 时无视 explicit）
if overridden:
    logger.warning("[PD external-lb] 注册表权威覆盖了 CLI/env 显式键 %s；"
                   "如需保留命令行值设 PD_KEEP_CLI_KEYS=<逗号分隔>", overridden)
cmd_known_params["_pd_engine_overrides"] = {k: deepcopy(v) for k,v in merged_engine.items() if k not in keep_cli}
```

### 5.2 `vllm_adapter._prepare_engine_config` 重申块（[~L1890](../../wings_control/engines/vllm_adapter.py#L1890)）
`_pd_engine_overrides` 现已按 `keep_cli` 排除 → 重申时**删掉 explicit 守卫**，无脑应用 stash。

```python
pd_overrides = params.get("_pd_engine_overrides")
if pd_overrides:
    for k, v in pd_overrides.items():
        # 删掉： if k in explicit_keys: continue   ← 决策已在 5.1 编码进 stash
        if v is None: engine_config.pop(k, None)   # None → 删 base 键（语义不变）
        else: engine_config[k] = v
```
> 删守卫对两种模式都安全：开关关时 stash 本就不含 explicit 键（删了无影响）；开关开时 forced 键必须无视 explicit 重申（删了才对）。

### 5.3 两个 helper（config_loader）
- `_env_true("PD_REGISTRY_AUTHORITATIVE")` — 真值判断。
- `_pd_keep_cli_keys()` — 读 `PD_KEEP_CLI_KEYS`，逗号分隔 → snake_case set。

---

## 6. 行为样例（Qwen3-30B-A3B **D 角色**；注册表 batched=120 / seqs=60 / gpu=0.88；`seed` 不在注册表）

| # | 部署 / 开关 | 传入关键 flag | 结果 | 为什么 |
|---|---|---|---|---|
| 1 | external-lb,开关**关** | 精简（不传调优） | batched=**120** seqs=**60** gpu=**0.88** | 非显式 → 注册表（现状） |
| 2 | external-lb,开关**关** | `--max-num-batched-tokens 4096 --max-num-seqs 256 --gpu-memory-utilization 0.8` | batched=**4096** seqs=**256** gpu=**0.8** | explicit 赢（现状，证不回归） |
| 3 | external-lb,开关**开** | 同 #2 | batched=**120** seqs=**60** gpu=**0.88** + ⚠️日志 | 注册表权威，盖平台 |
| 4 | external-lb,开关**开** + `PD_KEEP_CLI_KEYS=gpu_memory_utilization` | 同 #2 | batched=**120** seqs=**60** **gpu=0.8** | 逃生舱保 gpu，其余注册表 |
| 5 | external-lb,开关**开** | `--seed 42`（seed 不在注册表） | seed=**42** | 注册表没声明 → 不受影响 |
| 6 | **非 PD**(无 PD_ROLE),开关**开** | `--gpu-memory-utilization 0.8` | gpu=**0.8** | 不进 external-lb → 开关无效 |

**#3 日志**：
```
[PD external-lb] 注册表权威覆盖了 CLI/env 显式键 ['max_num_batched_tokens', 'max_num_seqs', 'gpu_memory_utilization']；
                 如需保留命令行值设 PD_KEEP_CLI_KEYS=<逗号分隔>
```

---

## 7. 边界与未覆盖

1. **注册表未声明的键关不掉**：平台多传的 `--enable-chunked-prefill`（decode 本不该开）若注册表 decode 里**没有**对应键，则注册表无从"盖"它 → 开关开也关不掉。
   - 处置（择一）：① 在注册表 decode 显式加 `"enable_chunked_prefill": false`（渲染器 `false`→不出 flag），让它进 `merged_engine` 被权威；② 另设「禁用键」机制。**本方案默认走 ①**（注册表声明 `false` 即可关）。
2. **`null`/`false` 抑制语义保留**：重申块 `None→pop`、渲染器 `False`/`None`→不出 flag，不变。
3. **无法自动区分「有意 vs 惯性」**：靠 `PD_KEEP_CLI_KEYS` 显式信号——设计取舍，非缺陷。
4. **gpu-mem 强制风险**：若注册表 0.88 而某节点显存紧需 0.8，开关开会强制 0.88 → 靠 ⚠️日志 + `PD_KEEP_CLI_KEYS=gpu_memory_utilization` 兜。

---

## 8. 回归隔离

- 只动 external-lb 路径（`_pd_engine_overrides` 仅 PD 命中时非空）。**非 PD / standalone PD（`_get_pd_config`）/ Ray / 单机**不进这两段 → 字节级不变。
- 开关默认**关**（方案 X）时，`keep_cli==explicit` → stash 内容与现状一致 → **连 external-lb 也字节级不变**。改动是「纯增量、可回退」。

---

## 9. 测试计划（`tests/pd_external_lb_verify.py` 加层）

1. external-lb + 平台灌 flag + `PD_REGISTRY_AUTHORITATIVE=1` → batched/seqs/gpu = **注册表值**（120/60/0.88）。
2. + `PD_KEEP_CLI_KEYS=gpu_memory_utilization` → gpu=**用户值**，其余注册表。
3. 开关**关** + 平台灌 flag → 平台值赢（**证不回归**）。
4. 非 PD explicit + 开关开 → 用户值（隔离）。
5.（若采纳 §7.1①）注册表 `enable_chunked_prefill:false` → 平台传的 `--enable-chunked-prefill` 被关。

---

## 10. 决策点

| 决策 | 选项 | 说明 |
|------|------|------|
| 触发模式 | **X：opt-in `PD_REGISTRY_AUTHORITATIVE`（推荐）** | 平台拨 1 个 env 即生效；默认零行为变化、可回退 |
| | Y：default-on（external-lb 一律权威） | 彻底免平台改动；但改变所有 external-lb 行为，风险中 |
| chunked-prefill 类「注册表未声明」键 | 走 §7.1①（注册表声明 `false` 关之）（推荐） | 复用现有渲染器语义，无需新机制 |
| | 另设「禁用键」机制 | 更通用但更复杂 |

---

## 11. 工作量

~半天：`config_loader` 2 处改 + 2 helper、`vllm_adapter` 删 1 行守卫、harness 加 §9 用例、本文档 + 设计文档 §13.x 落 as-built。**纯增量，opt-in 默认关，可随时回退。**
