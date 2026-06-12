# Qwen3.5 / Qwen3.6 · Day0 特性适配设计（定稿）

> 范围：**Ascend（主）+ NVIDIA**，dense 与 MoE 两条架构。
> 模型/架构：
> - `Qwen3_5ForConditionalGeneration` —— dense：Qwen3.5-27B / Qwen3.6-27B
> - `Qwen3_5MoeForConditionalGeneration` —— MoE：Qwen3.6-35B-A3B、Qwen3.5-397B-A17B
> 目标：让 wings-control 对齐 vLLM-Ascend 官方 tutorial / [recipes.vllm.ai](https://recipes.vllm.ai/) 的 Qwen3.5/3.6 启动命令——MTP 投机（含 `enforce_eager`）、MoE 启动字段、function call 解析器、混合模型 mamba 缓存。
> Qwen3.5 系列与 Qwen3.6 **共用同一套架构与 `qwen3_5_mtp` 方法**（Qwen3.6 只是发布名，config.json 仍注册为 `Qwen3_5*`）。

---

## 1. 目标启动命令（官方 tutorial / recipe）

**Qwen3.6-35B-A3B（MoE，A3 单机 2 卡，w8a8）：**
```bash
vllm serve Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
  --data-parallel-size 1 --tensor-parallel-size 2 --enable-expert-parallel \
  --quantization ascend --max-num-seqs 128 --max-model-len 262144 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}' \
  --async-scheduling
```

**Qwen3.5-397B-A17B（MoE，A3 单机 16 卡，w8a8）：** 同上，差异 `--tensor-parallel-size 16`、`--max-model-len 133000`、`additional-config` 无 `multistream_overlap_shared_expert`。

**Qwen3.5/3.6-27B（dense，A3 2 卡）：** 工具调用 `qwen3_coder`；其中 Qwen3.6-27B 官方命令带 `--enable-prefix-caching --mamba-cache-mode align`，Qwen3.5-27B-w8a8 官方命令则是 `--no-enable-prefix-caching` 且不带 mamba（见 §7 已知分歧）。

> 关键点：官方命令把 MTP 写成 `enforce_eager: true`（只让 MTP 头 eager），而**主模型仍走 `FULL_DECODE_ONLY` 全图**——这正是规避「全图 decode replay MTE 越界」崩溃的写法（参见 GLM-5 aclgraph 案例）。

---

## 1.1 实际生成命令（dry_run 实测产出）

> 命令：`python dry_run.py --scenario qwen36-35b-a3b` / `qwen35-397b-a17b` / `qwen36-27b`
> 场景已接通 `ENABLE_AUTO_TOOL_CHOICE` / `ENABLE_AUTO_THINK_CHOICE`，否则 `tool_call_parser` 会被 FC 门控剥掉。

**Qwen3.6-35B-A3B 主命令（节选）：**
```bash
python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 262144 \
  --quantization ascend --enable-expert-parallel --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 --enable-prefix-caching --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --tensor-parallel-size 2 \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' &
```

逐项对账：

| 项 | 35B-A3B（spec 开） | 397B-A17B（spec 关） | 27B dense（spec 开） |
|---|---|---|---|
| `--tool-call-parser` | `qwen3_coder` ✓ | `qwen3_coder` ✓ | `qwen3_coder` ✓ |
| `--speculative-config` | `…enforce_eager: true` ✓ | **无**（默认关）✓ | `…enforce_eager: true` ✓ |
| `--max-model-len` | `262144` ✓ | `133000` ✓ | `4096`（dense 默认，CLI 覆盖） |
| `--tensor-parallel-size` | `2`（device_count 推导） | `16` | `2` |
| `--enable-expert-parallel` | ✓ | ✓ | 无（dense 正确） |
| `--quantization ascend` | ✓ | ✓ | 无 |
| `--mamba-cache-mode align` | 无（MoE 不带）✓ | 无 ✓ | `align` ✓ |
| `additional-config` | 含 `multistream` | 不含 multistream | `{enable_cpu_binding}` |

---

## 2. 设计决策（已全部定案）

| 编号 | 议题 | 决策 |
|---|---|---|
| D1 | 投机开关 | **所有模型默认关，仅由上层 `--enable-speculative-decode` / `ENABLE_SPECULATIVE_DECODE` 控制**；Qwen3.5 不自动置真（曾短暂自动开，已撤销） |
| D2 | MTP `enforce_eager` | 投机开启且架构为 Qwen3.5 时，自动合成的 `--speculative-config` **追加 `"enforce_eager": true`**（spec 内部 eager，主模型仍全图，规避 MTE 崩溃） |
| D3 | dense prefix/mamba | dense `Qwen3_5ForConditionalGeneration` 默认 `enable_prefix_caching: true` + `mamba_cache_mode: align`（跟随 Qwen3.6-27B 工作命令） |
| D4 | MoE 启动字段 | 按官方命令补 MoE recipe；**MoE 不带 mamba**（官方 MoE 命令无）；TP/DP **不写死**，由 device_count 运行时推导 |
| D5 | MoE 双模型差异 | 用**模型名精确匹配**的专属 key 承载差异（`Qwen3.6-35B-A3B` → 262144+multistream；`Qwen3.5-397B-A17B` → 133000）；其余共用 arch `default` 基线 |
| D6 | function call 解析器 | `tool_call_parser` 由 `hermes` 改为 **`qwen3_coder`**（官方 Qwen3.5/3.6 recipe 明确要求；vLLM 已注册该 parser），sglang 侧维持 `qwen` |

---

## 3. 现状差距（适配前）

| 特性 | 目标 | 适配前产物 | 缺口 |
|---|---|---|---|
| MTP `enforce_eager` | spec config 含 `enforce_eager:true` | 自动合成只出 `method`+`num` | 缺 `enforce_eager`，MTP 头会被全图捕获 → MTE 崩溃风险 |
| MoE recipe | EP/quant/max_model_len/multistream 等 | `Qwen3_5MoeForConditionalGeneration` 仅一个极简 `default` | 双 MoE 模型无差异化、无 EP/quant 等 |
| tool_call_parser | `qwen3_coder` | `hermes` | 解析器不符合官方 recipe |
| mamba（dense）| `align` | 无 / `no_enable_prefix_caching` | 混合模型 prefix caching 需 mamba 对齐 |

---

## 4. 详细设计

### 4.1 投机默认关、仅上层控制（D1）
- **不**为 Qwen3.5 写任何「自动置 `enable_speculative_decode=True`」的 gate（与 V4-Flash a3 / GLM-4.7-W8A8 那两处运行时自动开**刻意不同**）。
- 投机由上层 `--enable-speculative-decode` 触发后，[`_should_append_auto_speculative_config`](../../wings_control/engines/vllm_adapter.py) 才合成 `--speculative-config`；JSON 里也**不写 `speculative_config`**，保持「开关一致」。

### 4.2 MTP `enforce_eager`（D2）
- 架构集合 [`_QWEN35_ARCHES`:2524](../../wings_control/engines/vllm_adapter.py#L2524) + 助手 [`_is_qwen35_arch`:2530](../../wings_control/engines/vllm_adapter.py#L2530)。
- `qwen3_5_mtp` 方法映射见 [`_resolve_mtp_method`:2535](../../wings_control/engines/vllm_adapter.py#L2535)（dense/MoE 同值）。
- 在 [`_build_speculative_cmd`:2646](../../wings_control/engines/vllm_adapter.py#L2646) 的 MTP 分支，`num_speculative_tokens` 之后追加：
  ```python
  if _is_qwen35_arch(model_info.model_architecture):   # L2709
      speculative_config_temp.append('"enforce_eager": true')
  ```
- 这是 **spec 内部 eager**（仅 MTP/草稿头），与顶层 `--enforce-eager`（`ASCEND_ENFORCE_EAGER`、整模型退 eager）是两个旋钮，互不影响。

### 4.3 dense prefix + mamba（D3）
- [ascend_default.json `Qwen3_5ForConditionalGeneration`:491](../../wings_control/config/defaults/ascend_default.json#L491)：`enable_prefix_caching: true` + `mamba_cache_mode: "align"`（去掉原 `no_enable_prefix_caching`）。

### 4.4 MoE recipe + 模型名专属 key（D4 + D5）
- [ascend_default.json `Qwen3_5MoeForConditionalGeneration`:519](../../wings_control/config/defaults/ascend_default.json#L519)：
  - `default`：公共基线（EP / `max_num_seqs 128` / `max_num_batched_tokens 16384` / `gpu_mem 0.9` / `enable_prefix_caching` / `async_scheduling` / `FULL_DECODE_ONLY` / `additional_config {enable_cpu_binding}`），**不带 quantization、不带 mamba、不写 TP**。
  - [`Qwen3.6-35B-A3B`:552](../../wings_control/config/defaults/ascend_default.json#L552)：`max_model_len 262144` + `quantization ascend` + `additional_config` 含 `multistream_overlap_shared_expert: true`。
  - [`Qwen3.5-397B-A17B`:586](../../wings_control/config/defaults/ascend_default.json#L586)：`max_model_len 133000` + `quantization ascend`。
- 命中规则：非-V4 模型按 **`--model-name` 精确匹配**专属 key（[`_match_model_engine_config`](../../wings_control/core/config_loader.py)），未匹配回落 `default`。两个产品名已登记到 [model_utils.py:197-198](../../wings_control/utils/model_utils.py#L197)。
- TP 不写死：dense/MoE 都让 `tensor_parallel_size` 随 `device_count` 推导，兼容单机 A3-TP16 与 2×A2 多机。

### 4.5 tool_call_parser → `qwen3_coder`（D6）
- ascend：dense [#491](../../wings_control/config/defaults/ascend_default.json#L491) 与 MoE [#519](../../wings_control/config/defaults/ascend_default.json#L519) 全部 `vllm_ascend` 块 `hermes → qwen3_coder`。
- nvidia：[`Qwen3_5ForConditionalGeneration`:406](../../wings_control/config/defaults/nvidia_default.json#L406) / [`Qwen3_5MoeForConditionalGeneration`:434](../../wings_control/config/defaults/nvidia_default.json#L434) 的 `vllm` 块同步；sglang 维持 `qwen`。
- 文档真值表 [function_call_support.yaml](../../wings_control/docs/features/function_call/function_call_support.yaml) 已同步。
- 同批 recipe 对齐（非 Qwen3.5，但同次审计）：DeepSeek-V3.2 `reasoning_parser` `deepseek_r1 → deepseek_v3`；Qwen3-Coder（nvidia vllm）`qwen3_xml → qwen3_coder`。

---

## 5. 改动清单（已实现）

| 文件 | 改动 |
|---|---|
| [vllm_adapter.py](../../wings_control/engines/vllm_adapter.py) | ① 新增 `_QWEN35_ARCHES` / `_is_qwen35_arch`；② `_build_speculative_cmd`：Qwen3.5 MTP 追加 `enforce_eager:true`；③ 撤销 Qwen3.5 自动开投机 |
| [ascend_default.json](../../wings_control/config/defaults/ascend_default.json) | dense 加 prefix+mamba+`qwen3_coder`；MoE 重构为 `default` + `Qwen3.6-35B-A3B` + `Qwen3.5-397B-A17B` 三块 recipe，`qwen3_coder` |
| [nvidia_default.json](../../wings_control/config/defaults/nvidia_default.json) | Qwen3.5/3.6 dense+MoE `hermes → qwen3_coder`；Qwen3-Coder `qwen3_xml → qwen3_coder`；V3.2 reasoning 修正 |
| [model_utils.py](../../wings_control/utils/model_utils.py#L197) | 登记 `Qwen3.5-397B-A17B`、`Qwen3.6-35B-A3B` |
| [function_call_support.yaml](../../wings_control/docs/features/function_call/function_call_support.yaml) | Qwen3.5/3.6、Qwen3-Coder 真值表同步 `qwen3_coder` |
| [dry_run.py](../../dry_run.py) | 新增 `qwen36-35b-a3b` / `qwen35-397b-a17b` / `qwen36-27b` 场景 + FC/think 开关接线 |
| [tests/test_unit_tcp.py](../../tests/test_unit_tcp.py) | tcp03/04/07 更新为新值 + 新增 Qwen3.5/3.6 → `qwen3_coder`（nvidia+ascend）2 例 |

> **实现验证**：三场景 dry_run 逐字段对齐目标命令（见 §1.1）；`test_unit_tcp.py` 50 passed；全量回归 21 failed / 531 passed，21 项失败均为 master 预存（V4-Pro / GLM-5 / MiniMax / Kimi / 快照等），**零新增失败**。

---

## 6. 测试要点
- UT：Qwen3.5 dense/MoE → `tool_call_parser = qwen3_coder`、`reasoning_parser = qwen3`（nvidia + ascend）。
- UT：投机开启时 Qwen3.5 `--speculative-config` 含 `"method": "qwen3_5_mtp"` 与 `"enforce_eager": true`；**DeepSeek 等其它 MTP 不带 `enforce_eager`**（无回归）。
- UT：未传 `--enable-speculative-decode` → 命令**不含** `--speculative-config`（默认关）。
- 端到端：`Qwen3.6-35B-A3B` 命中专属 recipe（262144 + multistream），`Qwen3.5-397B-A17B`（133000），未知名回落 `default`；MoE 命令**不带** `--mamba-cache-mode`，dense **带** `align`。
- 回归：普通 Qwen3 / Qwen3Moe / Qwen3Next 仍 `hermes`（未被误改）。

---

## 7. 已知分歧（待定）
1. **dense prefix/mamba 自相矛盾**：官方 **Qwen3.6-27B** 命令 `--enable-prefix-caching` + `--mamba-cache-mode align`；官方 **Qwen3.5-27B-w8a8** 命令 `--no-enable-prefix-caching` 且无 mamba。二者同属 `Qwen3_5ForConditionalGeneration`。当前默认跟 3.6（开 prefix + mamba），故对 Qwen3.5-27B-w8a8 这条**不完全对齐**。若需精确区分，应为 `Qwen3.5-27B` 加模型名专属 key（prefix off / 无 mamba / `quantization ascend`）。
2. **MoE quantization 写死 `ascend`**：模型名专属 recipe 默认带 `quantization: ascend`（官方为 w8a8）；若用 BF16 同名权重，需 CLI 覆盖或另开非量化 key。
3. **D6 docs 与 recipe 冲突**：vLLM tool_calling **文档表**把 Qwen3-Coder 标 `qwen3_xml`，而 **recipe 命令**用 `qwen3_coder`；本次按 recipe 取 `qwen3_coder`，如需跟文档可单独回退该项。

---

> 附（同次基础设施清理，非 Qwen3.5 专属）：start_command.sh 的重复 `export` 收口去重已抽到
> [utils/shell_env_utils.py `dedupe_env_exports`](../../wings_control/utils/shell_env_utils.py#L40)，由 [vllm_adapter.py:2915](../../wings_control/engines/vllm_adapter.py#L2915) 调用，保证每个环境变量每条执行路径只剩一条生效；用例见 [tests/test_env_dedup.py](../../tests/test_env_dedup.py)。
