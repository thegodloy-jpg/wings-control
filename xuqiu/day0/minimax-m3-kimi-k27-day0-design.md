# MiniMax-M3 / Kimi-K2.7-Code · Day0 适配报告（分析稿）

> 状态：**仅分析，未落代码**。本文给出两模型在主分支上的适配方案、改动清单、风险与测试要点。
> 范围分工（按本次诉求）：
> - **MiniMax-M3**：**仅 NVIDIA** 场景（`engine == "vllm"`）；**该单个模型开启 enforce-eager**（模型级，不影响他人）。
> - **Kimi-K2.7-Code**：**重点 DP/TP 策略**；其余（parser / 图编译 / 前缀缓存 / 多模态 / Ascend env）**由现有 default 模板承载**。
>
> 架构事实（取自各自 `config.json`，权威）：
> | 模型 | `architectures[0]` | `model_type` | 注意力 | MoE | 量化 | 平台 |
> |---|---|---|---|---|---|---|
> | MiniMax-M3-MXFP8 | `MiniMaxM3SparseForConditionalGeneration` | `minimax_m3_vl` | GQA + MSA 稀疏 | 128 选 4 + 1 shared | `mxfp8` | NVIDIA |
> | Kimi-K2.7-Code | `KimiK25ForConditionalGeneration` | `kimi_k25` | MLA | 384 选 8 + 1 shared | `compressed-tensors`(公版)/w4a8(Ascend) | Ascend |

---

## 0. 目标启动命令（用户给定）

**MiniMax-M3-MXFP8（NVIDIA）**
```bash
vllm serve /var/ai-model/MiniMax-M3-MXFP8 \
  --tensor-parallel-size 8 --block-size 128 --max-model-len 80000 \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 4096 --enforce-eager
# 差异点：上下文降配；并行 TP（DP 起不来）；去除投机
```

**Kimi-K2.7-Code（Ascend，含 host/env 前置）**
```bash
vllm serve /model/Kimi-K2.7-Code \
  --trust-remote-code --no-enable-prefix-caching --seed 1024 \
  --tensor-parallel-size 16 --enable-expert-parallel \
  --max-num-seqs 64 --max-model-len 102400 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,16,32,64,128,256],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --allowed-local-media-path /
# env：MLAPO / FLASHCOMM1 / BALANCE_SCHEDULING / HCCL_BUFFSIZE=800 ...
```

---

## 1. 设计决策

| 编号 | 议题 | 决策 |
|---|---|---|
| K1 | Kimi-K2.7-Code 是否需注册 | **架构已注册**（`KimiK25ForConditionalGeneration`），仅核对 DP/TP + 补名 |
| K2 | Kimi DP/TP 策略 | **复用现有 Ascend DP 规则**：`device_count∈{8,16}→TP=device_count`，DP=1（纯 TP），EP 非分布式自动开 |
| K3 | Kimi 其余参数 | **全部走 default 模板**（parser / FULL_DECODE_ONLY / no-prefix-cache / mm / ascend env），不新增逻辑 |
| M1 | MiniMax-M3 适配范围 | **仅 NV**（`engine == "vllm"`），新架构需完整注册 |
| M2 | MiniMax-M3 eager | **模型级**：在 `nvidia_default.json` 该架构下置 `"enforce_eager": true` → 渲染 `--enforce-eager`；**不动** `_need_enforce_eager`（那是 Ascend A+X 旋钮） |
| M3 | MiniMax-M3 parser | **待核实**官方 vLLM 是否提供 `minimax_m3` parser；不照搬 `minimax_m2`（见 §6） |

---

## 2. 现状差距（含代码佐证）

### 2.1 Kimi-K2.7-Code —— 架构已通，差"名 + DP/TP 核对"

| 维度 | 现状 | 佐证 |
|---|---|---|
| 架构识别 | `config.json → architectures[0]` 自动取到 `KimiK25ForConditionalGeneration` | [model_utils.py:321](../../wings_control/utils/model_utils.py#L321) |
| 模型名 | `_LLM_MODELS` 已含 `Kimi-K2.7` / `Kimi-K2.7-w4a8`（**但无 `Kimi-K2.7-Code`**） | [model_utils.py:149-154](../../wings_control/utils/model_utils.py#L149-L154) |
| Ascend defaults | `FULL_DECODE_ONLY` 图 / `no_enable_prefix_caching` / `kimi_k2` parser / `mm_encoder_tp_mode` / `async_scheduling` 全部就位 | [ascend_default.json:706-733](../../wings_control/config/defaults/ascend_default.json#L706-L733) |
| Ascend 专属 env | `_build_kimik25_ascend_env` 注入 `MLAPO` + `FLASHCOMM1` + `BALANCE_SCHEDULING` + `HCCL_OP_EXPANSION_MODE` + `PYTORCH_NPU_ALLOC_CONF` + `OMP_*` + `TASK_QUEUE_ENABLE` + `HCCL_BUFFSIZE=1024`（分布式感知）**【实测确认全部注入】** | [vllm_adapter.py:1142-1181](../../wings_control/engines/vllm_adapter.py#L1142-L1181) |
| 引擎自动选 | 架构属"强制 vllm_ascend"集合 | [config_loader.py:2142](../../wings_control/core/config_loader.py#L2142) |
| **DP/TP** | KimiK25 ∈ Ascend-DP 架构集；`device_count∈{8,16}→TP=device_count` | [vllm_adapter.py:214](../../wings_control/engines/vllm_adapter.py#L214) / [vllm_adapter.py:225](../../wings_control/engines/vllm_adapter.py#L225) |
| EP | 非分布式自动开 EP | [vllm_adapter.py:1786-1793](../../wings_control/engines/vllm_adapter.py#L1786-L1793) |

> 参照产物：snap03（Kimi-K2.5 单机 8 卡）已生成 `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --async-scheduling --no-enable-prefix-caching --compilation-config {...FULL_DECODE_ONLY} --mm-encoder-tp-mode data --tensor-parallel-size 8`，见 [tests/snapshots/snap03](../../tests/snapshots/snap03_vllm_ascend_kimik25_single.sh#L249)。K2.7-Code 同架构，复用同模板。

**缺口**：仅 ① `Kimi-K2.7-Code` 未在模型名表（不致命，`model_type=auto` 回退 `llm`）；② 需用 dry_run 实证 **16 卡 → TP16 / DP1 / EP** 与命令一致。

### 2.2 MiniMax-M3 —— 全新架构，完整缺失

| 维度 | 现状 | 缺口 |
|---|---|---|
| 架构注册 | `MiniMaxM3SparseForConditionalGeneration` 不在 `_LLM_MODELS`（只有 `MiniMaxM2ForCausalLM`） | `is_wings_supported()=False` |
| nvidia defaults | 无该架构条目（只有 `MiniMaxM2ForCausalLM`，[nvidia_default.json:526](../../wings_control/config/defaults/nvidia_default.json#L526)） | 无 `trust_remote_code` / parser / `block_size` / eager |
| enforce-eager | `_need_enforce_eager` 仅 `vllm_ascend`，NV 恒 False | NV 无自动 eager 通道 |
| trust_remote_code | 无 arch 默认 → 不注入 | **缺它 MSA/VL 自定义代码加载失败 → 起不来** |

代码佐证：
- 引擎命令渲染：`engine_config` 逐键 → `--{key.replace('_','-')}`，[vllm_adapter.py:2463-2478](../../wings_control/engines/vllm_adapter.py#L2463-L2478)；bool `True` 渲成裸 flag、`False` 省略，[vllm_helpers.py:114-115](../../wings_control/utils/vllm_helpers.py#L114-L115) → **`enforce_eager:true` ⇒ `--enforce-eager`** 成立。
- NV 无 eager 通道：[vllm_adapter.py:178-191](../../wings_control/engines/vllm_adapter.py#L178-L191)（`engine!="vllm_ascend"→False`）。
- 未注册不阻断、仅告警落 vllm：[config_loader.py:2085-2088](../../wings_control/core/config_loader.py#L2085-L2088)。

---

## 3. 详细设计

### 3.1 Kimi-K2.7-Code（DP/TP 为主，其余 default 承载）

**核心：DP/TP 策略（K2）**
- 目标命令是 **单机 TP16 + EP，无 DP**（差异点"开 DP 无法启动 → 纯 TP16"）。
- 现有规则即可达成：`device_count=16` 时 [`_default_deepseek_ascend_dp_tensor_parallel_size`](../../wings_control/engines/vllm_adapter.py#L225) 返回 `16` → `TP=16`；DP 走 `dp_deployment` 拓扑时 `dp_local = device_count / TP = 16/16 = 1`（即无 DP）；EP 由非分布式分支自动开。
- **结论**：无需新增 DP/TP 代码；**关键是用 dry_run 实证** `device_count=16` 产出 `--tensor-parallel-size 16` + `--enable-expert-parallel` + 不带 DP（或 dp_size_local=1），且不会误走 DP4 导致"起不来"。

**其余（K3）全部 default 承载**，不改：
- parser / `--no-enable-prefix-caching` / `--async-scheduling` / `--compilation-config FULL_DECODE_ONLY` / `--mm-encoder-tp-mode data` → ascend defaults 已有。
- `MLAPO` / `FLASHCOMM1` → `_build_kimik25_ascend_env` 已注入。
- `max_model_len`(102400) / `max_num_seqs`(64) / `max_num_batched_tokens`(16384) / `seed`(1024) → 用户经 CLI/env 覆盖 default（默认 4096/16/8192）。

**小改（可选）**：`_LLM_MODELS["KimiK25ForConditionalGeneration"]` 追加 `"Kimi-K2.7-Code"`（识别/矩阵更干净；不加也能跑）。

### 3.2 MiniMax-M3（仅 NV + 模型级 eager）

**M-a 架构注册**（必做）：`_LLM_MODELS` 新增
```python
"MiniMaxM3SparseForConditionalGeneration": [
    "MiniMax-M3", "MiniMax-M3-MXFP8",
],
```
（独立新键，**不可并入 `MiniMaxM2ForCausalLM`** —— 架构类名不同）

**M-b nvidia_default.json 新增该架构块**（必做，含 M2 的 eager）：
```jsonc
"MiniMaxM3SparseForConditionalGeneration": {
  "default": {
    "vllm": {
      "trust_remote_code": true,        // ← 关键，否则 MSA/VL 自定义代码不加载
      "max_model_len": 4096,            // 用户覆盖到 80000
      "block_size": 128,                // 对齐 config.sparse_block_size=128
      "enforce_eager": true,            // ← M2：模型级 eager，渲染 --enforce-eager
      "tool_call_parser": "<待核实>",   // 见 §6，勿照搬 minimax_m2
      "mm_encoder_tp_mode": "data"      // VL，多模态编码器 TP 模式（若需要）
    },
    "vllm_distributed": { /* 同上 */ }
  }
}
```
- `enforce_eager:true` 经 §2.2 渲染链 ⇒ 启动命令带 `--enforce-eager`，**仅对该架构生效**，完美对应"单个模型开启 eager"。
- TP/上下文/显存（TP8、80K、0.95、batched 4096）走用户 CLI，无需 default 写死。

---

## 4. 改动清单（分析口径，未实施）

| 文件 | Kimi-K2.7-Code | MiniMax-M3 |
|---|---|---|
| [model_utils.py](../../wings_control/utils/model_utils.py) `_LLM_MODELS` | （可选）加 `Kimi-K2.7-Code` 名 | **新增** `MiniMaxM3SparseForConditionalGeneration` 架构键 + 名 |
| [nvidia_default.json](../../wings_control/config/defaults/nvidia_default.json) | — | **新增** 架构块（trust_remote_code / block_size=128 / `enforce_eager:true` / parser / mm） |
| [ascend_default.json](../../wings_control/config/defaults/ascend_default.json) | **dedup** 重复 `KimiK25` 键（676 死键，706 生效） | — |
| [vllm_adapter.py](../../wings_control/engines/vllm_adapter.py) | 不改（DP/TP/EP/env 已覆盖） | 不改（eager 走 JSON；MSA 由模型代码处理） |
| [dry_run.py](../../dry_run.py) | 新增场景 `kimi-k27-ascend-16` | 新增场景 `minimax-m3-nv-8` |
| tests | snap/单测：16 卡 TP16/DP1/EP | snap/单测：含 `--enforce-eager` + trust_remote_code + block-size 128 |

---

## 5. 注意事项 / 风险

**Kimi-K2.7-Code**
1. ⚠️ **ascend_default.json 重复键**：`KimiK25ForConditionalGeneration` 出现两次（[676](../../wings_control/config/defaults/ascend_default.json#L676) / [706](../../wings_control/config/defaults/ascend_default.json#L706)），JSON last-wins → 676 死键。顺手 dedup（类比 `d0b0545` 对 Qwen 的处理）。
2. **env 已实测全覆盖（更正）**：`_build_kimik25_ascend_env` 已注入 `MLAPO` / `FLASHCOMM1` / `BALANCE_SCHEDULING` / `HCCL_OP_EXPANSION_MODE` / `PYTORCH_NPU_ALLOC_CONF` / `OMP_*` / `TASK_QUEUE_ENABLE`，通用块再补 jemalloc + perf tuning。**唯一差异：`HCCL_BUFFSIZE` —— kimi env 注入 `1024`，命令是 `800`**（在通用块之后注入，运行时 last-wins=1024）。1024≥800，通常安全；若需对齐 800（手工或为 100K 长上下文省显存），改 [`_build_kimik25_ascend_env`:1171](../../wings_control/engines/vllm_adapter.py#L1171) 一行即可（KimiK25 范围，K2.5/K2.6/K2.7 共用）。另：wings 多注一个无害的 `VLLM_ENGINE_READY_TIMEOUT_S=3600`。
   - ⚠️ **FLASHCOMM1 受分布式分支控制**：[`_is_kimik25_distributed`](../../wings_control/engines/vllm_adapter.py#L1126) 读 `DISTRIBUTED`；单机（`false`）→ 注 FLASHCOMM1（与手工一致），2 节点（`true`）→ 改注 `HCCL_INTRA_PCIE/ROCE`、不注 FLASHCOMM1。需确认 K2.7-Code 为单机 16 卡。该函数 `in [True,"true",1]` 判定对 `"True"`/`"1"` 字符串有边界漏判，建议顺手收紧。
3. **量化错配**：HF 公版 config 是 `compressed-tensors`（GPU），Ascend 实跑应是 w4a8 变体（`Kimi-K2.7-w4a8` 已注册）；确认权重与平台匹配。
4. **DP/TP 必须 dry_run 实证**：确认 16 卡不误入 DP4（差异点明示 DP 起不来），落到纯 TP16。

**MiniMax-M3**
1. ⚠️ **parser 对齐**（[[reasoning-parser-official-alignment]]）：务必核实部署 vLLM 是否有 `minimax_m3` 的 tool/reasoning parser；M3 是 GQA+MSA 新架构，**不能默认套 `minimax_m2`**。
2. **MXFP8 支持**：vLLM 由 `quantization_config.quant_method=mxfp8` 自动识别，需确认部署 vLLM 版本含 mxfp8 kernel；`--enforce-eager` 很可能正是绕 mxfp8 图编译兼容性（与 M2 决策一致）。
3. **trust_remote_code 必加**：缺它 MSA/VL 自定义建模代码不加载 → 启动失败（这是 M3 的真正硬门槛，而非 `is_wings_supported`）。
4. **MSA 稀疏注意力**：由模型自定义代码实现，**不要**套 DeepSeek 专用 IndexCache/KV-sparse；反向也确认不被项目里 MLA/DeepSeek 假设的分支误伤。
5. **多模态**：`served-model-name minimax-m3` 若只跑文本，确认 VL 分支不强制要求 vision 配置；`--allowed-local-media-path /`（开放整盘）安全面需评估是否收窄。

**横切**
- 两者都多模态（`ForConditionalGeneration`），注意 `task=generate` 与 mm 默认项一致性。
- 与近期 native 卸载修复无冲突（两命令均未开 KV offload）；若将来对它们开 `LMCACHE_OFFLOAD`，Kimi(MLA)/MiniMax(MSA) 走哪条卸载路径需另评（非 DeepSeek 的 CPUOffloadingConnector/native 假设）。

---

## 6. 待核实项（落代码前必须确认）

| 项 | 模型 | 如何确认 |
|---|---|---|
| `minimax_m3` tool/reasoning parser 是否存在 | MiniMax-M3 | 查部署 vLLM 版本 parser 注册表（对齐官方 Reasoning/Tool Outputs 表） |
| mxfp8 kernel 支持 | MiniMax-M3 | 部署 vLLM 版本 + 是否仍需 `--enforce-eager` |
| VL 多模态必填项（limit-mm / mm_encoder_tp_mode） | 两者 | 读 config + 官方部署文档 |
| 16 卡 DP/TP 实际产物 | Kimi-K2.7 | `dry_run.py` 新场景比对 |
| `enforce_eager:true` 是否被某些 NV 分支二次清洗 | MiniMax-M3 | dry_run 比对 exec 行确有 `--enforce-eager` |

---

## 7. 一句话结论

- **Kimi-K2.7-Code**：架构与模板**已就绪**，工作量集中在 **DP/TP 的 dry_run 实证** + 重复键 dedup + 补模型名；其余 default 承载。
- **MiniMax-M3**：**新架构注册**（`_LLM_MODELS` + nvidia_default.json）+ **模型级 `enforce_eager:true`**（仅该模型 NV 开 eager）；**parser 与 mxfp8 支持需先核实**再落地。

---

## 8. 实施定稿（已落地 + 强校验 100%）

> 下列为**实际实现**口径，若与上文分析稿有出入，以本节为准。

**MiniMax-M3（仅 NV）**
- `_LLM_MODELS` 新增 `MiniMaxM3SparseForConditionalGeneration`：`MiniMax-M3` / `MiniMax-M3-MXFP8`。
- `nvidia_default.json` 新增该架构块：`enforce_eager:true`（模型级 eager，bool→裸 flag 渲染）+ `block_size:128` + `max_model_len:4096`。
- **按手工命令对齐：不加 `trust_remote_code`、不加 parser**（部署 vLLM 原生支持 M3；§3/§5 中"trust_remote_code 必加"的早期假设被手工命令推翻，已不采纳）。reasoning/function_call 两矩阵置 null/omit。
- 强校验：注入部署值后 6/6 flag 全复现（含 `--enforce-eager`/`--block-size 128`/`--max-model-len 80000`/`--gpu-memory-utilization 0.95`）。

**Kimi-K2.7-Code（Ascend，default 模板承载）**
- `_LLM_MODELS` 补 `Kimi-K2.7-Code` / `Kimi-K2.7-Code-w4a8`；两矩阵同步 `kimi_k2`。
- DP/TP 实证：`device_count=16 → 纯 TP16、不走 DP、EP 开`（对齐"DP起不来→TP16"）。
- env 全由 `_build_kimik25_ascend_env` 承载（MLAPO/FLASHCOMM1/BALANCE_SCHEDULING/…）；**实测更正**：`BALANCE_SCHEDULING` 已注入（早期"未注入"为误判）。
- **数值决策（KimiK25 default，影响所有 K2.x）**：
  - `max_num_batched_tokens`：8192 → **16384**（随手工命令）。
  - `max_num_seqs`：**保持 16**（随"差异点：并发降到16"；与手工命令的 64 矛盾，取差异点；需 64 时部署注入 `MAX_NUM_SEQS=64`）。
  - `HCCL_BUFFSIZE`：**保持 1024**（不随命令的 800）。
  - `allowed_local_media_path: "/"`：**硬编码进 KimiK25 全部 4 引擎块（选项 B）** → 渲染 `--allowed-local-media-path /`。⚠️ 把整盘开放给媒体加载，安全面扩大，系用户明确选择；如需收窄可改为 env/CLI 旋钮（选项 A）。
- 强校验：注入部署值后手工命令 **11/11 flag 全复现**（含 `--allowed-local-media-path /`）。

**验证汇总**：单测 556 passed；dry-run 15/15 场景无回归；两模型字段+数值强校验 100% 复现（max-num-seqs 取 16 为主动决策，非缺口）。
