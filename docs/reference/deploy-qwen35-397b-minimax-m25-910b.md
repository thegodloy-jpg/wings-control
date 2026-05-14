# Qwen3.5-397B-A17B & MiniMax-M2.5 单机 8×910B 部署可行性深度分析

> 编制日期：2026-04-07  
> 硬件环境：单机 8×Ascend 910B (64 GB HBM2e/卡, 总计 512 GB)  
> 引擎版本：vllm-ascend ≥0.17.0rc1 / MindIE 2.3.0  
> 数据来源：wings-control 代码库配置 + vllm-ascend 官方文档 + 模型架构分析

---

## 目录

1. [两模型核心参数对比](#1-两模型核心参数对比)
2. [Qwen3.5-397B-A17B 部署分析](#2-qwen35-397b-a17b-部署分析)
3. [MiniMax-M2.5 部署分析](#3-minimax-m25-部署分析)
4. [部署命令与配置参考](#4-部署命令与配置参考)
5. [结论与建议](#5-结论与建议)

---

## 1. 两模型核心参数对比

| 参数 | Qwen3.5-397B-A17B | MiniMax-M2.5 |
|------|:---:|:---:|
| **架构** | Qwen3_5MoeForConditionalGeneration | MiniMaxM2ForCausalLM |
| **总参数量** | ~397B | ~456B |
| **激活参数** | ~17B | ~46B |
| **MoE 结构** | 是 | 是 |
| **Expert 数量** | 128 (估计) | 64×2 (估计) |
| **注意力机制** | GQA | Lightning Attention (线性注意力) |
| **BF16 权重大小** | ~794 GB | ~912 GB |
| **W8A8 权重大小** | ~397 GB | ~456 GB |
| **W4A16 权重大小** | ~199 GB | ~228 GB |
| **FP8 权重大小** | ~~397 GB~~ (910B 不支持 FP8) | ~~456 GB~~ (910B 不支持 FP8) |

### 硬件约束回顾

| 硬件参数 | 值 |
|---------|:-:|
| 单卡显存 | 64 GB HBM2e |
| 8 卡总显存 | **512 GB** |
| 卡间互联 | HCCS |
| 支持量化 | INT8 (W8A8), INT4 (W4A16), ~~FP8~~ |

---

## 2. Qwen3.5-397B-A17B 部署分析

### 2.1 显存需求估算

#### W8A8 (INT8) 量化方案

| 组件 | 显存需求 | 说明 |
|------|:---:|------|
| 模型权重 (W8A8) | ~397 GB | 每参数 1 Byte |
| KV Cache | ~30-50 GB | 取决于 max_model_len 和 batch_size |
| 激活值 + 框架开销 | ~15-25 GB | 包含 EP 通信 buffer 等 |
| **合计** | **~442-472 GB** | — |
| **剩余可用显存** | **~40-70 GB** | 从 512 GB 扣除 |

#### W4A16 (INT4) 量化方案

| 组件 | 显存需求 | 说明 |
|------|:---:|------|
| 模型权重 (W4A16) | ~199 GB | 每参数 0.5 Byte |
| KV Cache | ~30-80 GB | 余量更大，可用更大 max_model_len |
| 激活值 + 框架开销 | ~15-25 GB | — |
| **合计** | **~244-304 GB** | — |
| **剩余可用显存** | **~208-268 GB** | 充裕 |

### 2.2 可行性判定

| 方案 | 单机 8×910B | 可行性 | 风险等级 |
|------|:---:|:---:|:---:|
| **BF16 (无量化)** | ❌ 794 GB >> 512 GB | 不可行 | — |
| **W8A8 (INT8)** | ⚠️ ~450 GB < 512 GB | **勉强可行** | 🟡 中等 |
| **W4A16 (INT4/GPTQ)** | ✅ ~250 GB < 512 GB | **可行** | 🟢 低 |

### 2.3 W8A8 部署要点与风险

**可行但需严格管控的部署参数**：

| 参数 | 建议值 | 说明 |
|------|:---:|------|
| `tensor_parallel_size` | 8 | 必须使用全部 8 卡 |
| `enable_expert_parallel` | true | MoE 模型必须启用 EP |
| `max_model_len` | **2048-4096** | ❗ 不能设太大，否则 KV Cache 会超出剩余显存 |
| `max_num_batched_tokens` | ≤4096 | 限制单次 prefill 的 token 数 |
| `max_num_seqs` | ≤16 | 限制并发序列数，控制 KV Cache 峰值 |
| `gpu_memory_utilization` | 0.95 | 充分利用显存，但保留 5% 安全边际 |
| `enable_chunked_prefill` | true | 分块 prefill，降低峰值显存 |
| `enforce_eager` | true (可选) | 如果 CUDA graph 显存占用过大 |

**W8A8 风险点**：

1. **KV Cache 空间紧张**：剩余仅 ~40-70 GB，长序列或大 batch 会 OOM
2. **MoE EP 通信开销**：Expert Parallel 需要额外的 HCCS 通信 buffer
3. **910B 不支持 FP8**：无法使用社区常见的 FP8 MoE 优化
4. **量化模型可用性**：截至目前，Qwen3.5-397B 的 W8A8 官方量化版本可能未发布，需自行量化

### 2.4 W4A16 部署方案（推荐）

W4A16 (GPTQ-INT4 或 AWQ-INT4) 方案更适合单机 8×910B：

| 参数 | 建议值 | 说明 |
|------|:---:|------|
| `tensor_parallel_size` | 8 | 全部 8 卡 |
| `enable_expert_parallel` | true | EP 分散 Expert |
| `max_model_len` | **8192-16384** | 余量充足，可支持更长上下文 |
| `max_num_seqs` | ≤64 | 可支持更多并发 |
| `gpu_memory_utilization` | 0.90 | 有余量 |
| `quantization` | `gptq` 或 `awq` | vllm 原生支持 |

**W4A16 优势**：
- 权重仅 ~199 GB → 充裕的 KV Cache 空间
- 可支持更大 batch 和更长序列
- 质量损失略大于 W8A8，但对于 MoE 模型通常可接受

### 2.5 量化方法推荐

| 量化方法 | 工具 | 引擎兼容性 | 910B 支持 | 推荐度 |
|---------|------|:---:|:---:|:---:|
| **GPTQ-INT4** | AutoGPTQ | ✅ vllm, vllm-ascend, sglang | ✅ | ⭐⭐⭐⭐ |
| **GPTQ-INT8** | AutoGPTQ | ✅ vllm, vllm-ascend | ✅ | ⭐⭐⭐ |
| **AWQ-INT4** | AutoAWQ | ✅ vllm, vllm-ascend | ✅ | ⭐⭐⭐⭐ |
| **SmoothQuant (W8A8)** | AMCT / vllm 内置 | ✅ vllm, vllm-ascend, MindIE | ✅ | ⭐⭐⭐ |
| compressed-tensors | vllm built-in | ✅ vllm | ⚠️ 部分支持 | ⭐⭐ |

**自行量化 Qwen3.5-397B GPTQ-INT4 的方法**：

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_path = "/path/to/Qwen3.5-397B-A17B-Instruct"
quant_path = "/path/to/Qwen3.5-397B-A17B-Instruct-GPTQ-Int4"

quantize_config = BaseQuantizeConfig(
    bits=4,           # INT4 量化
    group_size=128,
    damp_percent=0.01,
    desc_act=False,
    sym=True,
    true_sequential=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoGPTQForCausalLM.from_pretrained(
    model_path, 
    quantize_config,
    max_memory={i: "60GB" for i in range(8)}  # 量化过程需要大显存
)

# 准备校准数据（~128 条对话样本）
data = [...]  # 格式化后的 input_ids tensor
model.quantize(data, cache_examples_on_gpu=False)
model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
```

> ⚠️ **量化 397B 模型本身需要大量 GPU 显存和时间**，建议在 NVIDIA A100/H100 集群上进行量化操作。

### 2.6 Wings 代码中的 Qwen3.5-397B 已有配置

`ascend_default.json` 中 `Qwen3_5MoeForConditionalGeneration` 的配置：

```json
{
  "vllm_ascend": {
    "trust_remote_code": true,
    "max_model_len": 4096,
    "task": "generate",
    "enable_expert_parallel": true,
    "tool_call_parser": "hermes",
    "reasoning_parser": "qwen3"
  },
  "mindie": {
    "maxSeqLen": 4096,
    "maxInputTokenLen": 2048,
    "maxIterTimes": 2048,
    "tp": 4, "dp": 4,
    "moe_ep": 4, "moe_tp": 4
  }
}
```

**vllm_ascend 环境变量**（`vllm_adapter.py`）：
```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
```

---

## 3. MiniMax-M2.5 部署分析

### 3.1 显存需求估算

#### W8A8 (INT8) 量化方案

| 组件 | 显存需求 | 说明 |
|------|:---:|------|
| 模型权重 (W8A8) | ~456 GB | 每参数 1 Byte |
| KV Cache | ~40-60 GB | M2.5 激活参数更多(46B)，KV Cache 更大 |
| 激活值 + 框架开销 | ~20-30 GB | Lightning Attention 额外开销 |
| **合计** | **~516-546 GB** | — |
| **剩余可用显存** | **❌ -4 ~ -34 GB** | **已超出 512 GB** |

#### W4A16 (INT4) 量化方案

| 组件 | 显存需求 | 说明 |
|------|:---:|------|
| 模型权重 (W4A16) | ~228 GB | 每参数 0.5 Byte |
| KV Cache | ~40-80 GB | — |
| 激活值 + 框架开销 | ~20-30 GB | — |
| **合计** | **~288-338 GB** | — |
| **剩余可用显存** | **~174-224 GB** | 充裕 |

### 3.2 可行性判定

| 方案 | 单机 8×910B | 可行性 | 风险等级 |
|------|:---:|:---:|:---:|
| **BF16 (无量化)** | ❌ 912 GB >> 512 GB | 不可行 | — |
| **W8A8 (INT8)** | ❌ ~530 GB > 512 GB | **不可行** | 🔴 高 |
| **W4A16 (INT4/GPTQ)** | ✅ ~310 GB < 512 GB | **可行** | 🟢 低 |
| **多机 2×8 卡 W8A8** | ✅ ~530 GB < 1024 GB | **可行** | 🟢 低 |

### 3.3 为什么 W8A8 单机不可行

**核心矛盾**：MiniMax-M2.5 总参 ~456B，W8A8 仅权重就占 ~456 GB，几乎耗尽 512 GB 显存。

详细拆解：

```
单卡可用显存:  64 GB
8 卡总量:     512 GB

模型权重 (W8A8, EP 分片): 
  - 非 Expert 参数 (共享层): ~30-40 GB (每卡都要加载全部共享参数)
  - Expert 参数 (EP 分片):   ~420 GB / 8 = ~52.5 GB/卡
  - 单卡权重合计: ~30 + 52.5 = ~82.5 GB/卡  ← 已超出 64 GB!

等等，这里有个关键点：使用 TP + EP 组合时——
  - TP=8: 共享层按 8 卡切分 → 每卡 ~5 GB
  - EP=8: Expert 按 8 卡切分 → 每卡 ~52.5 GB
  - 单卡权重合计: ~57.5 GB  ← 接近 64 GB上限
  - KV Cache 需要的空间: 几乎为 0  ← 无法推理
```

**结论**：即使 TP=8 + EP=8 最大化分片，W8A8 下每卡权重已 ~57.5 GB，仅剩 ~6.5 GB/卡用于 KV Cache + 激活值，**完全不够推理使用**。

### 3.4 W4A16 方案（单机 8×910B 可行）

W4A16 将权重减半至 ~228 GB，使单机部署成为可能：

```
单卡权重 (TP=8 + EP=8):
  - 共享层: ~20 GB / 8 = ~2.5 GB/卡
  - Expert: ~210 GB / 8 = ~26.3 GB/卡
  - 单卡合计: ~28.8 GB/卡
  - 剩余 KV Cache: ~35 GB/卡 ← 足够
```

| 参数 | 建议值 | 说明 |
|------|:---:|------|
| `tensor_parallel_size` | 8 | 全卡 |
| `enable_expert_parallel` | true | MoE EP |
| `max_model_len` | **4096-8192** | 取决于 KV Cache 余量 |
| `max_num_seqs` | ≤32 | 控制 KV Cache |
| `quantization` | `gptq` 或 `awq` | W4A16 |
| `gpu_memory_utilization` | 0.95 | — |

### 3.5 2 机 16 卡 W8A8 方案

如果客户有 2 台 8×910B 机器：

```
总显存: 2 × 512 = 1024 GB
W8A8 需求: ~530 GB
剩余: ~494 GB  ← 非常充裕

配置:
  TP=8 (机内), PP=2 (跨机) 或 TP=16 (跨机)
  enable_expert_parallel: true
```

| 参数 | 建议值 | 说明 |
|------|:---:|------|
| `tensor_parallel_size` | 8 或 16 | 跨机需要高速互联 |
| `pipeline_parallel_size` | 2 (如果 TP=8) | 流水线并行 |
| `enable_expert_parallel` | true | EP |
| `max_model_len` | **16384-32768** | 余量充足 |

### 3.6 MiniMax-M2.5 特殊注意事项

1. **Lightning Attention**：MiniMax-M2.5 使用混合注意力机制（Lightning Attention + Softmax Attention），vllm-ascend 需要对应的算子支持
2. **FlashComm**：Wings 代码为 MiniMax-M2.5 配置了 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 环境变量，说明通信优化是必要的
3. **trust_remote_code**：必须设为 `true`，因为 M2.5 使用自定义模型类
4. **MindIE 不支持**：MiniMax-M2.5 在 MindIE 2.3.0 中无原生适配，仅能使用 vllm-ascend

### 3.7 Wings 代码中的 MiniMax-M2.5 已有配置

`ascend_default.json` 中 `MiniMaxM2ForCausalLM` 的配置：

```json
{
  "vllm_ascend": {
    "trust_remote_code": true,
    "max_model_len": 4096,
    "enable_expert_parallel": true,
    "tool_call_parser": "minimax_m2",
    "reasoning_parser": "minimax_m2_reasoning"
  },
  "mindie": {
    "maxSeqLen": 4096,
    "maxInputTokenLen": 2048,
    "maxIterTimes": 2048,
    "tp": 4, "dp": 4,
    "moe_ep": 4, "moe_tp": 4
  }
}
```

**vllm_ascend 环境变量**（`vllm_adapter.py`）：
```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

---

## 4. 部署命令与配置参考

### 4.1 Qwen3.5-397B-A17B W8A8 — 单机 8×910B

```bash
# 环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

# 拉起服务
vllm serve /path/to/Qwen3.5-397B-A17B-Instruct-W8A8 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --task generate \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --port 8000
```

### 4.2 Qwen3.5-397B-A17B W4A16 — 单机 8×910B（推荐）

```bash
# 环境变量（同上）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

# 拉起服务
vllm serve /path/to/Qwen3.5-397B-A17B-Instruct-GPTQ-Int4 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 \
    --task generate \
    --trust-remote-code \
    --quantization gptq \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --port 8000
```

### 4.3 MiniMax-M2.5 W4A16 — 单机 8×910B

```bash
# 环境变量
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# 拉起服务
vllm serve /path/to/MiniMax-M2.5-GPTQ-Int4 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --quantization gptq \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_reasoning \
    --port 8000
```

### 4.4 MiniMax-M2.5 W8A8 — 2 机 16×910B

```bash
# Master 节点（Node 0）
# 环境变量
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# 使用 Ray 多机部署
ray start --head --port=6379

vllm serve /path/to/MiniMax-M2.5-W8A8 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --enable-expert-parallel \
    --max-model-len 16384 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_reasoning \
    --port 8000

# Worker 节点（Node 1）
ray start --address='<master_ip>:6379'
```

### 4.5 MindIE 配置参考（Qwen3.5-397B，如 MindIE 后续版本支持）

```json
{
    "BackendConfig": {
        "npuDeviceIds": [[0,1,2,3,4,5,6,7]],
        "ModelDeployConfig": {
            "maxSeqLen": 4096,
            "maxInputTokenLen": 2048,
            "maxIterTimes": 2048,
            "ModelConfig": [{
                "modelName": "qwen35_397b",
                "modelWeightPath": "/path/to/Qwen3.5-397B-A17B-W8A8",
                "worldSize": 8,
                "backendType": "atb",
                "npuMemSize": -1,
                "moeCfg": {
                    "moeEP": 4,
                    "moeTP": 4
                }
            }]
        }
    }
}
```

> ⚠️ MindIE 2.3.0 **不支持** Qwen3.5 和 MiniMax-M2.5 架构，以上配置仅为预留参考。

---

## 5. 结论与建议

### 5.1 结论汇总

| 模型 | W8A8 单机8×910B | W4A16 单机8×910B | 推荐方案 |
|------|:---:|:---:|------|
| **Qwen3.5-397B-A17B** | ⚠️ 勉强可行 (~450/512 GB) | ✅ 可行 (~250/512 GB) | **优先 W4A16**；W8A8 需严格限制 batch/seq |
| **MiniMax-M2.5** | ❌ 不可行 (~530/512 GB) | ✅ 可行 (~310/512 GB) | **单机必须 W4A16**；W8A8 需 2 机 16 卡 |

### 5.2 "单机 8 张 910B 是如何拉起的"

两个模型在昇腾 910B 上的拉起方式：

#### 引擎选择
- **首选 vllm-ascend ≥0.17.0**（推荐 v0.18.0rc1）
- MindIE 2.3.0 不支持这两个模型的架构
- sglang 对 Qwen3.5 有 `qwen25` parser，但对 MiniMax 无 parser

#### 核心拉起流程

```
1. 准备量化模型
   ├── Qwen3.5-397B: 下载/量化为 GPTQ-INT4 或 W8A8
   └── MiniMax-M2.5: 下载/量化为 GPTQ-INT4 (单机必须)

2. 准备容器环境
   └── docker pull quay.io/ascend/vllm-ascend:v0.18.0rc1
       └── 挂载 8 张 NPU (/dev/davinci0-7)

3. 配置环境变量
   ├── Qwen3.5: PYTORCH_NPU_ALLOC_CONF, HCCL_BUFFSIZE, TASK_QUEUE_ENABLE
   └── MiniMax: VLLM_ASCEND_ENABLE_FLASHCOMM1

4. vllm serve 启动
   ├── --tensor-parallel-size 8
   ├── --enable-expert-parallel       ← MoE 必须
   ├── --max-model-len 4096           ← 控制 KV Cache
   ├── --trust-remote-code            ← 自定义模型类
   └── --quantization gptq            ← 如果使用 GPTQ 模型

5. Wings 自动化
   └── wings_control 根据 ascend_default.json 中的配置
       自动设置环境变量 + 构建 CLI 参数 + 拉起 vllm serve
```

#### Wings 自动拉起方式

通过 Wings 框架，用户只需设置基本参数，Wings 会自动：

1. **识别模型架构**：`ModelIdentifier` 读取 `config.json` → 得到 `Qwen3_5MoeForConditionalGeneration` 或 `MiniMaxM2ForCausalLM`
2. **加载默认配置**：从 `ascend_default.json` 匹配架构 → 获取 TP/EP/max_model_len 等参数
3. **注入环境变量**：`_build_model_ascend_env_commands()` → 注入 TASK_QUEUE_ENABLE / FlashComm 等
4. **构建 vllm CLI**：`_build_vllm_cmd_parts()` → 生成完整 `vllm serve` 命令
5. **启动服务**：通过 subprocess / shell 执行

### 5.3 是否有对应的部署实施文档

| 文档来源 | Qwen3.5-397B | MiniMax-M2.5 | 状态 |
|---------|:---:|:---:|------|
| **华为 MindIE 官方** | ❌ | ❌ | 2.3.0 不支持这两个架构 |
| **vllm-ascend 官方文档** | ⚠️ 通用 MoE EP 文档 | ⚠️ 通用 MoE EP 文档 | 有通用 EP 部署教程，无专用模型文档 |
| **Qwen 官方** | ⚠️ GPTQ/AWQ 量化文档 | — | 有通用的 vllm 部署指南，未专门针对 397B |
| **MiniMax 官方** | — | ⚠️ README 有 vllm 部署示例 | GitHub 仓库有基本部署说明 |
| **Wings 代码库** | ✅ 已有完整配置 | ✅ 已有完整配置 | `ascend_default.json` + `vllm_adapter.py` |

**结论**：目前**没有专门的、端到端的部署实施文档**针对这两个模型在 8×910B 上的部署。最接近的参考是：

1. **vllm-ascend 的 Large Scale EP 教程**：`https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html`  
   — 演示了 MoE 模型在昇腾上的 Expert Parallel 部署流程

2. **Wings 代码库自身**：`ascend_default.json` 中的配置 + `vllm_adapter.py` 中的环境变量注入逻辑就是最完整的"部署实施方案"

3. **Qwen 官方 GPTQ 文档**：`https://qwen.readthedocs.io/en/latest/quantization/gptq.html`  
   — 提供量化方法，但尚未更新到 Qwen3.5 系列

### 5.4 行动建议

| 优先级 | 行动 | 详细说明 |
|:---:|------|---------|
| **P0** | 准备 Qwen3.5-397B GPTQ-INT4 量化模型 | 使用 AutoGPTQ 在 NVIDIA 集群上量化，得到可直接部署的 GPTQ 模型 |
| **P0** | 实测 Qwen3.5-397B W8A8 在 8×910B | 验证 max_model_len=4096 下是否 OOM |
| **P1** | 准备 MiniMax-M2.5 GPTQ-INT4 量化模型 | 同上 |
| **P1** | 实测 MiniMax-M2.5 W4A16 在 8×910B | 验证 vllm-ascend + EP + FlashComm 是否正常工作 |
| **P2** | 编写端到端部署 SOP | 基于实测结果，编写包含量化、容器准备、参数配置、验证的完整 SOP |
| **P2** | 关注 vllm-ascend v0.18.0 正式版 | 可能包含更完善的 MoE 多机支持 |
