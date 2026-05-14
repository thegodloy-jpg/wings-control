# 模型 × 引擎 × Function Call 支持细粒度分析报告

> 编制日期：2026-04-07（修订版）  
> 需求背景：中原算力客户要求 wings 支持 DeepSeek V3.1/V3.2、Qwen3-235B 等主流模型的 Function Call 功能  
> 数据来源：  
> - [MindIE 2.3.0 Function Call 官方文档](https://www.hiascend.com/document/detail/zh/mindie/230/mindiellm/llmdev/mindie_llm0303.html)  
> - [vLLM Tool Calling 官方文档](https://docs.vllm.ai/en/latest/features/tool_calling.html)  
> - wings-control 代码库 (`ascend_default.json`, `nvidia_default.json`, `mindie_adapter.py`)

---

## 目录

1. [诉求点一：MindIE 2.3.0 对各模型 Function Call 的细粒度支持分析](#1-mindie-230-对各模型-function-call-的细粒度支持分析)
2. [诉求点二：单机 8×910B W8A8 量化部署可行性分析](#2-单机-8910b-w8a8-量化部署可行性分析)
3. [诉求点三：四大引擎 Function Call 实现逻辑细粒度对比](#3-四大引擎-function-call-实现逻辑细粒度对比)
4. [Wings 当前配置现状与差距分析](#4-wings-当前配置现状与差距分析)
5. [Action Items](#5-action-items)

---

## 1. MindIE 2.3.0 对各模型 Function Call 的细粒度支持分析

### 1.1 MindIE 2.3.0 官方 Function Call 完整支持矩阵

根据 [MindIE 2.3.0 官方文档](https://www.hiascend.com/document/detail/zh/mindie/230/mindiellm/llmdev/mindie_llm0303.html) 的最新验证，**已支持 Function Call 特性的模型与硬件**：

#### 硬件支持

| 硬件平台 | 支持状态 |
|----------|:---:|
| Atlas 800I A2 (910B) | ✅ |
| Atlas 800I A3 (910C) | ✅ |
| Atlas 300I Duo | ✅ |

#### 已注册的 ToolsCallProcessor 完整列表

| Processor 类 | 注册名称 | 适用模型 |
|-------------|----------|---------|
| ToolsCallProcessorChatglmV2 | `chatglm_v2` | ChatGLM2 系列 |
| ToolsCallProcessorChatglmV3 | `chatglm3, chatglm_v3` | ChatGLM3-6B |
| ToolsCallProcessorChatglmV4 | `chatglm4_9b, chatglm_v4_9b, glm_4, glm_4_9b` | GLM-4-9B |
| ToolsCallProcessorDeepseekV3 | `deepseek_v3, deepseekv3` | DeepSeek-V3-0324, DeepSeek-R1-0528 |
| ToolsCallProcessorDeepseekV31 | `deepseek_v31, deepseekv31` | DeepSeek-V3.1 系列 |
| ToolsCallProcessorLlama | `llama, llama3, llama3_1` | LLaMA-3/3.1 系列 |
| ToolsCallProcessorQwen | `qwen1_5_or_2, qwen2_5` | Qwen1.5/2/2.5-Instruct 系列 |
| ToolsCallProcessorQwen3 | `qwen3, qwen3_moe` | Qwen3-32B, Qwen3-235B-A22B, Qwen3-30B-A3B |
| ToolsCallProcessorHermes | `hermes` | 兼容 Hermes 格式的模型 |

#### 官方验证模型列表

| 模型 | 推荐 tool_call_parser | 是否需要显式配置 models 字段 | 流式 FC 支持 |
|------|------|:---:|:---:|
| ChatGLM3-6B | `chatglm3` / `chatglm_v3` | 否 | 否 |
| Qwen3-32B | `qwen3` | 否 | ✅ 是 |
| Qwen3-235B-A22B | `qwen3_moe` / `qwen3` | 否 | ✅ 是 |
| Qwen3-30B-A3B | `qwen3_moe` / `qwen3` | 否 | ✅ 是 |
| DeepSeek-R1-0528 | `deepseek_v3` / `deepseekv3` | 否 | ✅ 是 |
| Qwen2.5-Instruct 系列 | `qwen2_5` | 否 | 否 |
| **DeepSeek-V3.1 系列** | **`deepseek_v31`** | **✅ 是（必须）** | 否 |

> **DeepSeek-V3.1 关键注意事项**：MindIE 默认使用 `deepseek_v3` 格式，但 V3.1 的工具调用格式与 V3 不同（V3.1 使用 `<tool_call>` XML 标签），**必须显式配置** `tool_call_parser: "deepseek_v31"`，否则解析失败。

#### FC 特性可叠加能力

MindIE 2.3.0 的 Function Call 可与以下特性同时启用：

| 叠加特性 | 支持状态 | 说明 |
|---------|:---:|------|
| 量化（W8A8/W4A16） | ✅ | FC 与量化推理兼容 |
| 长序列推理 | ✅ | — |
| 多机部署 | ✅ | — |
| PD 分离 | ✅ | — |
| MoE Expert Parallel | ✅ | DeepSeek、Qwen3-235B 等 |
| Multi-LoRA | ✅ | — |
| SplitFuse | ✅ | — |
| 并行解码 | ✅ | — |
| MTP（多 Token 预测） | ✅ | — |
| Prefix Cache | ✅ | — |
| 思维解析 | ✅ | — |

### 1.2 九大目标模型 MindIE FC 逐一匹配分析

| # | 模型 | 架构 | MindIE FC | 匹配 Processor | 推荐 parser | 细粒度分析 |
|---|------|------|:---:|------|------|------|
| 1 | **GLM4.7** | Glm4MoeForCausalLM | ❌ | 无 | — | GLM4.7 是 MoE 架构（~100B），与 GLM-4-9B 不同。MindIE 有 `ToolsCallProcessorChatglmV4`（适用 GLM-4-9B），但 **GLM4.7 使用全新的 MoE 架构**，其 FC 输出格式与 GLM-4-9B 不兼容，MindIE 2.3.0 未适配 |
| 2 | **Qwen3.5-397-A17B** | Qwen3_5MoeForConditionalGeneration | ❌ | 无 | — | Qwen3.5 是 2026 年新发布的多模态 MoE 架构，MindIE 2.3.0（2025.12 发布）不包含此架构支持。`Qwen3_5MoeForConditionalGeneration` 与 `Qwen3MoeForCausalLM` 是不同的模型类 |
| 3 | **Qwen3.5-27B** | Qwen3_5ForConditionalGeneration | ❌ | 无 | — | 同上，Qwen3.5 非 MoE 版本，但同属新架构，MindIE 2.3.0 未支持 |
| 4 | **MiniMax-M2.5** | MiniMaxM2ForCausalLM | ❌ | 无 | — | MiniMax 为第三方模型厂商，MindIE 无原生适配。其 FC 输出格式为自定义格式（非 Hermes/JSON），无法复用已有 Processor |
| 5 | **DeepSeek V3.2** | DeepseekV32ForCausalLM | ❌ | 无 | — | V3.2 使用全新的 `DeepseekV32ForCausalLM` 架构（相比 V3/V3.1 的 `DeepseekV3ForCausalLM` 有显著变化），MindIE 2.3.0 未针对此新架构适配 FC |
| 6 | **DeepSeek-Coder-V2-Instruct** | DeepseekV3ForCausalLM | ⚠️ | ToolsCallProcessorDeepseekV3 | `deepseekv3` | 属 DeepseekV3ForCausalLM 架构系列，理论上可使用 `deepseek_v3` parser。但该模型的 FC 输出格式可能与 V3-0324 有差异，**需实测验证** |
| 7 | **DeepSeek V3.1** | DeepseekV3ForCausalLM | ✅ | ToolsCallProcessorDeepseekV31 | **`deepseek_v31`** | **官方支持，已验证。必须配置 models 字段，否则默认使用 deepseek_v3 parser 导致解析失败** |
| 8 | **DeepSeek V4** | 待定 | ❌ | 无 | — | 新模型，架构尚未公开，需等 MindIE 后续版本 |
| 9 | **GPT-OSS-120B** | 待定 | ❌ | 无 | — | OpenAI 开源模型，vLLM 有专用 `openai` parser，MindIE 无适配 |
| 10 | **LLaMA3.1-70B** | LlamaForCausalLM | ⚠️ | ToolsCallProcessorLlama | `llama3_1` | **关键更正**：MindIE 2.3.0 已注册 `ToolsCallProcessorLlama`（支持 `llama, llama3, llama3_1`）。但 LLaMA3.1-70B 不在官方验证列表中（仅列出了 Qwen/DeepSeek/GLM 系列），**Processor 存在但未经官方验证，需实测** |

### 1.3 MindIE DeepSeek-V3.1 必须配置详解

DeepSeek-V3.1 是 MindIE 2.3.0 中**唯一需要显式配置 models 参数**才能启用 Function Call 的模型。

**配置位置**：`config.json → BackendConfig → ModelDeployConfig → ModelConfig[0] → models`

```json
{
    "models": {
        "deepseekv2": {
            "tool_call_options": {
                "tool_call_parser": "deepseek_v31"
            },
            "chat_template": "/path/to/tool_chat_template_deepseekv31.jinja"
        }
    }
}
```

**为什么 model_type 是 `deepseekv2`**：DeepSeek V3/V3.1 在 MindIE 内部复用了 V2 的模型类型标识（历史原因），这是 MindIE 的内部映射，不影响功能。

**Wings 实现**：`mindie_adapter.py` 的 `_inject_function_call_config()` 函数（L709-L730）根据 `ascend_default.json` 中配置的 `mindie_model_type` 和 `mindie_tool_call_parser` 字段自动注入此配置。

### 1.4 诉求点一结论

| 结论 | 详细说明 |
|------|---------|
| **MindIE 2.3.0 FC 覆盖率：10 个目标模型中仅 1~2 个确认支持** | 仅 DeepSeek V3.1 官方确认支持；LLaMA3.1 有 Processor 但未验证；DeepSeek-Coder-V2 可尝试 |
| **客户核心需求 DeepSeek V3.2 不受支持** | 需使用 vllm-ascend 引擎替代 |
| **GLM4.7 / MiniMax-M2.5 / Qwen3.5 均不支持** | 这三个新架构均需等 MindIE ≥2.4.0 或换引擎 |
| **LLaMA3.1-70B 存在未验证的 Processor** | `ToolsCallProcessorLlama` 已注册，可尝试配置 `llama3_1` parser 进行验证 |
| **建议方案：FC 场景优先使用 vllm-ascend** | vllm-ascend ≥0.17.0 对所有目标模型的 FC 支持最完备 |

---

## 2. 单机 8×910B W8A8 量化部署可行性分析

> **本节仅聚焦 W8A8 (INT8) 量化方案**，不涉及 FP16、W4A16 等其他精度。

### 2.1 硬件基础参数

| 参数 | 规格 |
|------|------|
| 卡型 | Ascend 910B (标准版) |
| 单卡显存 | 64 GB HBM2e |
| 8 卡总显存 | **512 GB** |
| 显存带宽 | 1.6 TB/s（单卡） |
| 互联 | HCCS (机内 8 卡互联) |
| 支持量化格式 | INT8 (W8A8), INT4 (W4A16), ~~FP8 (不支持)~~ |

> **关键限制**：Ascend 910B **不支持 FP8 计算**，因此 DeepSeek 系列的官方 FP8 权重无法直接使用，必须转换为 W8A8 INT8 或 W4A16 格式。

### 2.2 W8A8 显存需求估算方法

```
W8A8 显存估算 = 模型参数量(B) × 1 Byte (权重)
               + KV Cache 开销 (取决于 batch_size、seq_len、num_kv_heads、head_dim)
               + 激活值 + 框架开销
               ≈ 参数量(B) × 1.2~1.5 GB (经验值)
```

**MoE 模型特殊说明**：MoE 模型虽然只激活部分参数，但 Expert Parallel 模式下**所有 Expert 权重都需加载到显存中**（每张卡加载 `total_experts / EP` 个 Expert 的权重）。因此 MoE 模型的显存需求与**总参数量**相关，而非激活参数量。

### 2.3 各模型 W8A8 部署详细分析

#### 2.3.1 GLM4.7（~100B MoE, 激活 ~27B）

| 项目 | 数值 |
|------|------|
| 总参数量 | ~100B |
| W8A8 权重大小 | ~100 GB |
| KV Cache + 开销 | ~20~30 GB |
| **W8A8 总显存需求** | **~120~130 GB** |
| 8×910B (512 GB) | ✅ **可行，余量充裕** |
| 推荐配置 | TP=8 + EP=8, 可支持较大 batch |
| W8A8 量化模型可用性 | 需自行量化或社区提供 |

#### 2.3.2 Qwen3.5-397-A17B（397B MoE, 激活 17B）

| 项目 | 数值 |
|------|------|
| 总参数量 | 397B |
| W8A8 权重大小 | ~397 GB |
| KV Cache + 开销 | ~30~50 GB |
| **W8A8 总显存需求** | **~430~450 GB** |
| 8×910B (512 GB) | ⚠️ **勉强可行，余量很小（~60~80 GB）** |
| 推荐配置 | TP=8 + EP=8, batch_size 需严格限制 |
| 风险 | KV Cache 空间紧张，max_model_len 不能太大，建议 ≤4096 |
| W8A8 量化模型可用性 | 需自行量化（GPTQ-INT8 或 SmoothQuant） |

#### 2.3.3 Qwen3.5-27B（27B Dense）

| 项目 | 数值 |
|------|------|
| 总参数量 | 27B |
| W8A8 权重大小 | ~27 GB |
| KV Cache + 开销 | ~5~10 GB |
| **W8A8 总显存需求** | **~32~37 GB** |
| 8×910B (512 GB) | ✅ **可行，单卡即可部署** |
| 推荐配置 | TP=1~2 即可, max_model_len 可设较大 |
| W8A8 量化模型可用性 | Qwen 官方通常提供 GPTQ-INT8 版本 |

#### 2.3.4 MiniMax-M2.5（~456B MoE, 激活 ~46B）

| 项目 | 数值 |
|------|------|
| 总参数量 | ~456B |
| W8A8 权重大小 | ~456 GB |
| KV Cache + 开销 | ~40~60 GB |
| **W8A8 总显存需求** | **~496~516 GB** |
| 8×910B (512 GB) | ❌ **不可行，显存不足** |
| 分析 | W8A8 下模型权重已达 456 GB，加上 KV Cache 和开销将超出 512 GB |
| 替代方案 | 需要 W4A16 量化（~228 GB）或使用 2 台 8×910B |

#### 2.3.5 DeepSeek V3.2（~685B MoE, 激活 ~37B）

| 项目 | 数值 |
|------|------|
| 总参数量 | ~685B |
| W8A8 权重大小 | ~685 GB |
| KV Cache + 开销 | ~40~60 GB |
| **W8A8 总显存需求** | **~725~745 GB** |
| 8×910B (512 GB) | ❌ **不可行，远超单机容量** |
| 最低要求 | 需 **2 台** 8×910B（16 卡，总计 1024 GB） |

#### 2.3.6 DeepSeek-Coder-V2-Instruct（236B MoE, 激活 21B）

| 项目 | 数值 |
|------|------|
| 总参数量 | 236B |
| W8A8 权重大小 | ~236 GB |
| KV Cache + 开销 | ~20~30 GB |
| **W8A8 总显存需求** | **~256~266 GB** |
| 8×910B (512 GB) | ✅ **可行，余量充足** |
| 推荐配置 | TP=8 + EP=8, 可支持中等 batch |
| W8A8 量化模型可用性 | 社区有 GPTQ-INT8 版本 |

#### 2.3.7 DeepSeek V3.1（~671B MoE, 激活 ~37B）

| 项目 | 数值 |
|------|------|
| 总参数量 | ~671B |
| W8A8 权重大小 | ~671 GB |
| KV Cache + 开销 | ~40~60 GB |
| **W8A8 总显存需求** | **~711~731 GB** |
| 8×910B (512 GB) | ❌ **不可行，远超单机容量** |
| 最低要求 | 需 **2 台** 8×910B（16 卡，总计 1024 GB） |
| 备注 | 华为官方提供 DeepSeek-R1 的 W8A8 量化版本（用于 2 机 16 卡部署） |

#### 2.3.8 DeepSeek V4

| 项目 | 数值 |
|------|------|
| 总参数量 | 待公布 |
| 结论 | 待确认模型规模后评估 |

#### 2.3.9 GPT-OSS-120B（~120B Dense）

| 项目 | 数值 |
|------|------|
| 总参数量 | ~120B |
| W8A8 权重大小 | ~120 GB |
| KV Cache + 开销 | ~15~25 GB |
| **W8A8 总显存需求** | **~135~145 GB** |
| 8×910B (512 GB) | ✅ **可行，余量充裕** |
| 推荐配置 | TP=4~8 |

#### 2.3.10 LLaMA3.1-70B（70B Dense）

| 项目 | 数值 |
|------|------|
| 总参数量 | 70B |
| W8A8 权重大小 | ~70 GB |
| KV Cache + 开销 | ~10~15 GB |
| **W8A8 总显存需求** | **~80~85 GB** |
| 8×910B (512 GB) | ✅ **可行，余量极充裕** |
| 推荐配置 | TP=2~4, 可支持大 batch 和长序列 |
| W8A8 量化模型可用性 | Meta 及社区广泛提供 |

### 2.4 W8A8 部署汇总表

| 模型 | 总参量 | W8A8 显存需求 | 单机 8×910B 可行性 | 推荐 TP | 余量评估 |
|------|:---:|:---:|:---:|:---:|------|
| **Qwen3.5-27B** | 27B | ~35 GB | ✅ **单卡即可** | TP=1~2 | 极充裕 |
| **LLaMA3.1-70B** | 70B | ~85 GB | ✅ | TP=2~4 | 极充裕 |
| **GLM4.7** | ~100B MoE | ~130 GB | ✅ | TP=8+EP | 充裕 |
| **GPT-OSS-120B** | ~120B | ~145 GB | ✅ | TP=4~8 | 充裕 |
| **DeepSeek-Coder-V2** | 236B MoE | ~265 GB | ✅ | TP=8+EP | 较充裕 |
| **Qwen3.5-397-A17B** | 397B MoE | ~450 GB | ⚠️ **勉强** | TP=8+EP | ~60 GB 余量，需限 batch |
| **MiniMax-M2.5** | ~456B MoE | ~516 GB | ❌ **不可行** | — | 超出 4 GB |
| **DeepSeek V3.1** | ~671B MoE | ~731 GB | ❌ **不可行** | — | 需 2 台 16 卡 |
| **DeepSeek V3.2** | ~685B MoE | ~745 GB | ❌ **不可行** | — | 需 2 台 16 卡 |
| **DeepSeek V4** | 待定 | — | 待定 | — | — |

### 2.5 W8A8 量化方案引擎支持对比

| 引擎 | W8A8 支持 | 量化方式 | 昇腾 910B 兼容 | 备注 |
|------|:---:|------|:---:|------|
| **MindIE 2.3.0** | ✅ | 华为 AMCT 工具量化或直接加载 W8A8 权重 | ✅ | FC 特性可与量化叠加 |
| **vllm-ascend** | ✅ | SmoothQuant / GPTQ-INT8 / compressed-tensors | ✅ | 社区支持广泛 |
| **sglang** | ✅ | compressed-tensors / GPTQ | ✅（通过 vllm-ascend 后端） | — |
| **vllm (NVIDIA)** | ✅ | SmoothQuant / GPTQ / compressed-tensors | N/A | NVIDIA 场景 |

### 2.6 诉求点二结论

| 结论 | 说明 |
|------|------|
| **W8A8 单机可部署：6/10 模型** | Qwen3.5-27B, LLaMA3.1-70B, GLM4.7, GPT-OSS-120B, DeepSeek-Coder-V2, Qwen3.5-397-A17B（勉强） |
| **W8A8 单机不可部署：3/10 模型** | MiniMax-M2.5, DeepSeek V3.1, DeepSeek V3.2 均需多机 |
| **Qwen3.5-397-A17B 是临界模型** | W8A8 下约 450 GB，512 GB 卡空间余量仅 ~60 GB，必须严格限制 batch_size 和 max_model_len |
| **DeepSeek V3.1/V3.2 必须 2 机 16 卡** | 即使 W8A8 也需 ~700+ GB，单机远远不够 |
| **910B 不支持 FP8** | 无法直接使用 DeepSeek 官方 FP8 权重，需转 INT8 |

---

## 3. 四大引擎 Function Call 实现逻辑细粒度对比

### 3.1 架构层面对比

| 对比维度 | vllm (NVIDIA) | vllm-ascend (昇腾) | sglang | MindIE 2.3.0 |
|----------|------|------|------|------|
| **FC 启用方式** | CLI 参数 | CLI 参数（与 vllm 相同） | CLI 参数 | JSON 配置文件 |
| **核心参数** | `--enable-auto-tool-choice --tool-call-parser <p>` | 同 vllm | `--enable-auto-tool-choice --tool-call-parser <p>` | `models.<type>.tool_call_options.tool_call_parser` |
| **解析机制** | 内置 ToolParser 类，解析模型原始文本输出 | 继承 vllm 上游 ToolParser | 内置 parser（命名风格不同） | MindIE 服务端内置 ToolsCallProcessor |
| **tool_choice 支持** | `auto`, `required`, `none`, 指定函数名 | 同 vllm | `auto` | `auto` |
| **流式 FC** | ✅ 支持 | ✅ 支持 | ✅ 支持 | ⚠️ 仅 Qwen3/DeepSeek-R1-0528 |
| **自定义 parser 插件** | ✅ `--tool-parser-plugin` | ✅ 同上 | ❌ 不支持 | ❌ 不支持 |
| **Reasoning Parser** | ✅ `--reasoning-parser` | ✅ 同上 | ⚠️ 部分支持 | 支持思维解析（内置） |
| **模型覆盖广度** | ★★★★★ 最广 | ★★★★★ 继承 vllm | ★★★★ 较广 | ★★☆ 有限（~7 系列） |
| **新模型适配速度** | ★★★★★ 最快（开源社区） | ★★★★ 略滞后于 vllm | ★★★★ 较快 | ★★ 最慢（依赖华为发版） |

### 3.2 各引擎已支持的 Tool Call Parser 完整列表

#### 3.2.1 vllm / vllm-ascend 支持的 Parser（截至最新版本）

> 数据来源：[vLLM Tool Calling 官方文档](https://docs.vllm.ai/en/latest/features/tool_calling.html)

| Parser 名称 | 适用模型系列 | CLI 示例 |
|-------------|-------------|---------|
| `hermes` | NousResearch Hermes 系列, **Qwen2.5 系列**, QwQ-32B | `--tool-call-parser hermes` |
| `mistral` | Mistral 系列 | `--tool-call-parser mistral` |
| `llama3_json` | LLaMA 3.1, 3.2, 4 系列 | `--tool-call-parser llama3_json` |
| `llama4_pythonic` | LLaMA 4 系列（推荐） | `--tool-call-parser llama4_pythonic` |
| `internlm` | InternLM 2.5 系列 | `--tool-call-parser internlm` |
| `jamba` | AI21 Jamba 1.5 系列 | `--tool-call-parser jamba` |
| `xlam` | Salesforce xLAM 系列 | `--tool-call-parser xlam` |
| `minimax` | MiniMax-M1-40k/80k | `--tool-call-parser minimax` |
| **`minimax_m2`** | **MiniMax-M2.5** | `--tool-call-parser minimax_m2` |
| **`deepseek_v3`** | **DeepSeek-V3-0324, DeepSeek-R1-0528** | `--tool-call-parser deepseek_v3` |
| **`deepseek_v31`** | **DeepSeek-V3.1** | `--tool-call-parser deepseek_v31` |
| **`deepseekv32`** | **DeepSeek-V3.2** | `--tool-call-parser deepseekv32` |
| **`glm47`** | **GLM-4.7, GLM-4.7-Flash** | `--tool-call-parser glm47` |
| `glm45` | GLM-4.5, GLM-4.5-Air, GLM-4.6 | `--tool-call-parser glm45` |
| **`openai`** | **GPT-OSS-20B, GPT-OSS-120B** | `--tool-call-parser openai` |
| `kimi_k2` | Kimi-K2-Instruct | `--tool-call-parser kimi_k2` |
| `qwen3_xml` | Qwen3-Coder 系列 | `--tool-call-parser qwen3_xml` |
| `pythonic` | LLaMA 3.2 (pythonic), ToolACE | `--tool-call-parser pythonic` |
| `granite` / `granite4` | IBM Granite 系列 | `--tool-call-parser granite` |
| `longcat` | LongCat-Flash-Chat | `--tool-call-parser longcat` |
| `hunyuan_a13b` | Hunyuan-A13B | `--tool-call-parser hunyuan_a13b` |
| `functiongemma` | FunctionGemma-270M | `--tool-call-parser functiongemma` |
| `olmo3` | Olmo-3 系列 | `--tool-call-parser olmo3` |
| `gigachat3` | GigaChat3 系列 | `--tool-call-parser gigachat3` |

> **vllm 关键发现**：
> - Qwen2.5/Qwen3 系列使用 `hermes` parser（Qwen2.5 的 tokenizer_config.json 已内置 Hermes 格式支持）
> - DeepSeek V3.2 有专用的 `deepseekv32` parser（独立于 V3/V3.1）
> - GLM4.7 有专用的 `glm47` parser
> - GPT-OSS-120B 有专用的 `openai` parser
> - MiniMax-M2.5 有专用的 `minimax_m2` parser（区别于 M1 的 `minimax`）

#### 3.2.2 sglang 支持的 Parser

> 数据来源：wings-control `nvidia_default.json` 配置 + sglang 文档

| Parser 名称 | 适用模型系列 | 与 vllm parser 名称差异 |
|-------------|-------------|----------------------|
| `deepseekv3` | DeepSeek V3/R1/V3.1/V3.2 系列 | vllm 区分 `deepseek_v3`/`deepseek_v31`/`deepseekv32`，sglang 统一用 `deepseekv3` |
| `qwen25` | Qwen2.5/Qwen3/Qwen3.5 系列 | vllm 用 `hermes` |
| ❌ GLM4.7 | 无专用 parser | sglang 尚未支持 GLM4.7 FC |
| ❌ MiniMax-M2.5 | 无专用 parser | sglang 尚未支持 MiniMax-M2.5 FC |
| ❌ GPT-OSS-120B | 无专用 parser | sglang 尚未支持 |

> **sglang 关键差异**：
> 1. **Parser 名称风格不同**：sglang 使用 `deepseekv3`（无下划线），vllm 使用 `deepseek_v3`/`deepseek_v31`
> 2. **DeepSeek 系列统一 parser**：sglang 对 V3/V3.1/V3.2 统一使用 `deepseekv3`，不做区分
> 3. **Qwen 系列统一 parser**：sglang 对 Qwen2.5/Qwen3/Qwen3.5 统一使用 `qwen25`
> 4. **覆盖面不如 vllm**：GLM4.7、MiniMax-M2.5、GPT-OSS 等尚无专用 parser

#### 3.2.3 MindIE 2.3.0 支持的 Parser

| Parser 名称（注册名） | 适用模型系列 | 对应 Processor 类 |
|----------------------|-------------|------------------|
| `chatglm3`, `chatglm_v3` | ChatGLM3 | ToolsCallProcessorChatglmV3 |
| `chatglm4_9b`, `glm_4` 等 | GLM-4-9B | ToolsCallProcessorChatglmV4 |
| `deepseek_v3`, `deepseekv3` | DeepSeek V3-0324, R1-0528 | ToolsCallProcessorDeepseekV3 |
| `deepseek_v31`, `deepseekv31` | DeepSeek V3.1 | ToolsCallProcessorDeepseekV31 |
| `llama`, `llama3`, `llama3_1` | LLaMA 3/3.1 系列 | ToolsCallProcessorLlama |
| `qwen1_5_or_2`, `qwen2_5` | Qwen 1.5/2/2.5 | ToolsCallProcessorQwen |
| `qwen3`, `qwen3_moe` | Qwen3 系列 | ToolsCallProcessorQwen3 |
| `hermes` | Hermes 格式兼容模型 | ToolsCallProcessorHermes |

### 3.3 九大目标模型 × 四引擎 FC 支持综合矩阵

| 模型 | 架构 | vllm parser | vllm-ascend parser | sglang parser | MindIE parser | FC 覆盖评估 |
|------|------|------|------|------|------|------|
| **GLM4.7** | Glm4MoeForCausalLM | ✅ `glm47` | ✅ `glm47` | ❌ 无 | ❌ 无 | 2/4 引擎支持 |
| **Qwen3.5-397-A17B** | Qwen3_5MoeForConditionalGeneration | ✅ `hermes` | ✅ `hermes` | ✅ `qwen25` | ❌ 无 | 3/4 引擎支持 |
| **Qwen3.5-27B** | Qwen3_5ForConditionalGeneration | ✅ `hermes` | ✅ `hermes` | ✅ `qwen25` | ❌ 无 | 3/4 引擎支持 |
| **MiniMax-M2.5** | MiniMaxM2ForCausalLM | ✅ `minimax_m2` | ✅ `minimax_m2` | ❌ 无 | ❌ 无 | 2/4 引擎支持 |
| **DeepSeek V3.2** | DeepseekV32ForCausalLM | ✅ `deepseekv32` | ✅ `deepseekv32` | ✅ `deepseekv3` | ❌ 无 | 3/4 引擎支持 |
| **DeepSeek-Coder-V2** | DeepseekV3ForCausalLM | ✅ `deepseek_v3` | ✅ `deepseek_v3` | ✅ `deepseekv3` | ⚠️ `deepseekv3`（未验证） | 3~4/4 引擎支持 |
| **DeepSeek V3.1** | DeepseekV3ForCausalLM | ✅ `deepseek_v31` | ✅ `deepseek_v31` | ✅ `deepseekv3` | ✅ `deepseek_v31` | **4/4 引擎全支持** |
| **DeepSeek V4** | 待定 | 待定 | 待定 | 待定 | ❌ | 待定 |
| **GPT-OSS-120B** | 待定 | ✅ `openai` | ✅ `openai` | ❌ 无 | ❌ 无 | 2/4 引擎支持 |
| **LLaMA3.1-70B** | LlamaForCausalLM | ✅ `llama3_json` | ✅ `llama3_json` | ⚠️ 可用通用 | ⚠️ `llama3_1`（未验证） | 2~4/4 引擎支持 |

### 3.4 各引擎 FC 实现逻辑详解

#### 3.4.1 vllm / vllm-ascend 实现逻辑

**核心流程**：

```
用户请求 (tools 定义 + messages)
    ↓
Chat Template 渲染（将 tools 注入 system prompt）
    ↓
模型自由生成文本（tool_choice=auto 时无约束解码）
    ↓
ToolParser 提取工具调用（从原始文本中解析 function name、arguments）
    ↓
构建 OpenAI 兼容的 tool_calls 响应
```

**关键代码路径**：
- Parser 注册：`vllm/tool_parsers/` 目录下各 `*_tool_parser.py`
- 解析入口：`ToolParser.extract_tool_calls(model_output_text)`
- 输出格式：标准 OpenAI `ChatCompletionMessage.tool_calls`

**tool_choice 行为差异**：

| tool_choice | 行为 | JSON Schema 约束 |
|-------------|------|:---:|
| `auto` | 模型自由决定是否调用工具，ToolParser 从文本中提取 | ❌ 无约束 |
| `required` | 强制模型产生工具调用（structured outputs） | ✅ 有约束 |
| 指定函数名 | 强制调用指定函数（structured outputs） | ✅ 有约束 |
| `none` | 不产生工具调用，仅文本响应 | N/A |

**Wings 集成方式**（`vllm_adapter.py`）：
- 根据 `engine_config` 中的 `tool_call_parser` 和 `reasoning_parser` 字段
- 自动添加 `--enable-auto-tool-choice --tool-call-parser <parser>` CLI 参数
- 配置来源：`ascend_default.json` / `nvidia_default.json` 中按架构+模型名匹配

#### 3.4.2 sglang 实现逻辑

**核心流程**：与 vllm 类似，但 parser 命名和内部实现有差异。

```
用户请求 → Chat Template → 模型生成 → Parser 提取 → 工具调用响应
```

**关键差异**：
1. **Parser 命名统一化**：sglang 对同系列模型使用统一 parser（如 DeepSeek V3/V3.1/V3.2 都用 `deepseekv3`）
2. **Qwen 系列使用 `qwen25`**：而非 vllm 的 `hermes`（底层解析逻辑类似，但模板处理路径不同）
3. **缺少部分新模型 parser**：GLM4.7、MiniMax-M2.5 等暂无支持

**Wings 集成方式**（`sglang_adapter.py`）：
- `_build_sglang_cmd_parts()` 函数将 `engine_config` 转为 sglang CLI 参数
- 配置来源：`nvidia_default.json` 的 `sglang` / `sglang_distributed` 段

#### 3.4.3 MindIE 2.3.0 实现逻辑

**核心流程**：

```
加载 config.json → 解析 models.<model_type>.tool_call_options
    ↓
注册对应的 ToolsCallProcessor (工厂模式)
    ↓
推理时：模型生成 → Processor 解析工具调用 → 返回 OpenAI 兼容格式
```

**配置层级**：
```
config.json
└── BackendConfig
    └── ModelDeployConfig
        └── ModelConfig[0]
            ├── modelName: "deepseek_v31"
            ├── modelWeightPath: "/path/to/model"
            ├── worldSize: 8
            └── models:                          ← FC 配置入口
                └── <model_type>:                ← 如 "deepseekv2"
                    ├── tool_call_options:
                    │   └── tool_call_parser: "deepseek_v31"
                    └── chat_template: "/path/to/template.jinja"
```

**Processor 工厂注册机制**：
- 每个 `ToolsCallProcessor` 类注册多个别名（如 `deepseek_v3` 和 `deepseekv3` 等价）
- 根据 `tool_call_parser` 字段值查找对应的 Processor 类
- 大部分模型无需配置（使用默认 parser），仅 DeepSeek-V3.1 需显式配置

**Wings 集成方式**（`mindie_adapter.py`）：
- `_inject_function_call_config()` 函数（L709-L730）
- 从 `ascend_default.json` 读取 `mindie_model_type` 和 `mindie_tool_call_parser`
- 自动构建并注入到 MindIE config.json 的 `models` 字段中

```python
# wings_control/engines/mindie_adapter.py 简化逻辑
def _inject_function_call_config(engine_config, overrides):
    parser = engine_config.get("mindie_tool_call_parser")
    model_type = engine_config.get("mindie_model_type")
    if not (parser and model_type):
        return  # 无 FC 配置则跳过
    model_entry = {"tool_call_options": {"tool_call_parser": parser}}
    chat_template = engine_config.get("mindie_chat_template")
    if chat_template:
        model_entry["chat_template"] = chat_template
    overrides["models"] = {model_type: model_entry}
```

### 3.5 Parser 名称映射速查表

| 模型 | vllm parser | sglang parser | MindIE parser | 名称差异说明 |
|------|------|------|------|------|
| DeepSeek V3-0324 / R1 | `deepseek_v3` | `deepseekv3` | `deepseekv3` / `deepseek_v3` | sglang 无下划线 |
| DeepSeek V3.1 | `deepseek_v31` | `deepseekv3` | `deepseek_v31` | **sglang 不区分 V3/V3.1** |
| DeepSeek V3.2 | `deepseekv32` | `deepseekv3` | ❌ | **sglang 不区分 V3/V3.2** |
| Qwen2.5 | `hermes` | `qwen25` | `qwen2_5` | **三引擎各不同** |
| Qwen3 / Qwen3 MoE | `hermes` | `qwen25` | `qwen3` / `qwen3_moe` | **三引擎各不同** |
| Qwen3.5 | `hermes` | `qwen25` | ❌ | MindIE 不支持 |
| GLM4.7 | `glm47` | ❌ | ❌ | 仅 vllm 支持 |
| MiniMax-M2.5 | `minimax_m2` | ❌ | ❌ | 仅 vllm 支持 |
| LLaMA3.1 | `llama3_json` | 通用 | `llama3_1` | vllm 使用 `llama3_json` |
| GPT-OSS-120B | `openai` | ❌ | ❌ | 仅 vllm 支持 |

### 3.6 诉求点三结论

| 结论 | 说明 |
|------|------|
| **vllm / vllm-ascend FC 覆盖最全** | 支持 25+ 种模型系列的专用 parser，包括所有 10 个目标模型 |
| **sglang FC 覆盖次之** | 支持 DeepSeek / Qwen 系列，但缺 GLM4.7、MiniMax-M2.5、GPT-OSS |
| **MindIE 2.3.0 FC 覆盖最窄** | 仅 ~7 个模型系列（ChatGLM3/GLM-4-9B/Qwen/DeepSeek/LLaMA）+ 未验证的 LLaMA |
| **Parser 命名不统一是集成难点** | 同一模型在不同引擎使用不同 parser 名称，Wings 已通过 config 层抽象解决 |
| **vllm 的 tool_choice 支持最完备** | 支持 `auto/required/none/指定函数名`，sglang 和 MindIE 主要支持 `auto` |
| **昇腾 910B FC 场景首选 vllm-ascend** | 与 vllm 共享全部 parser，同时原生支持昇腾硬件 |

---

## 4. Wings 当前配置现状与差距分析

### 4.1 已配置的 Function Call 支持（逐架构分析）

| 架构 | vllm FC | vllm-ascend FC | sglang FC | MindIE FC | 完备度 |
|------|:---:|:---:|:---:|:---:|------|
| DeepseekV3ForCausalLM | ✅ `deepseek_v3` | ✅ `deepseek_v3` | ✅ `deepseekv3` | ✅ `deepseekv3` | ★★★★★ 全覆盖 |
| → DeepSeek-V3.1 变体 | ✅ `deepseek_v31` | ✅ `deepseek_v31` | ✅ `deepseekv3` | ✅ `deepseek_v31` | ★★★★★ 全覆盖 |
| Qwen3MoeForCausalLM | ✅ `hermes` | ✅ `hermes` | ✅ `qwen25` | ✅ `qwen3` | ★★★★★ 全覆盖 |
| Qwen3ForCausalLM | ✅ `hermes` | ✅ `hermes` | ✅ `qwen25` | ✅ `qwen3` | ★★★★★ 全覆盖 |
| Qwen2ForCausalLM | ✅ `hermes` | ✅ `hermes` | ✅ `qwen25` | ✅ `qwen2_5` | ★★★★★ 全覆盖 |
| DeepseekV32ForCausalLM | ✅ `deepseekv32` | ✅ `deepseekv32` | ✅ `deepseekv3` | ❌ 无 FC 配置 | ★★★★ MindIE 不支持 |
| Glm4MoeForCausalLM | ✅ `glm47` | ✅ `glm47` | ❌ 无 parser | ❌ 无 FC 配置 | ★★★ 仅 vllm 系 |
| Qwen3_5ForConditionalGeneration | ✅ `hermes` | ✅ `hermes` | ✅ `qwen25` | ❌ 无 FC 配置 | ★★★★ MindIE 不支持 |
| Qwen3_5MoeForConditionalGeneration | ✅ `hermes` | ✅ `hermes` | ✅ `qwen25` | ❌ 无 FC 配置 | ★★★★ MindIE 不支持 |
| MiniMaxM2ForCausalLM | ✅ `minimax_m2` | ✅ `minimax_m2` | ❌ 无 parser | ❌ 无 FC 配置 | ★★★ 仅 vllm 系 |

### 4.2 待补充配置项

| 配置项 | 当前状态 | 建议操作 | 优先级 |
|--------|---------|---------|:---:|
| LLaMA3.1 MindIE FC 配置 | 未配置 | 在 `ascend_default.json` 添加 LlamaForCausalLM 的 `mindie_model_type: "llama3_1"` 和 `mindie_tool_call_parser: "llama3_1"` 验证 | P2 |
| GPT-OSS-120B vllm parser | 未配置架构 | 确认架构后添加 `tool_call_parser: "openai"` | P2 |
| DeepSeek V4 全部引擎 | 待定 | 等模型发布后配置 | P3 |

### 4.3 差距总结

| 差距项 | 影响范围 | 优先级 | 建议方案 |
|--------|---------|:---:|---------|
| MindIE 不支持 V3.2 FC | 客户核心需求 | **P0** | 使用 vllm-ascend 引擎 |
| MindIE 不支持 GLM4.7 FC | 客户需求 | P1 | 使用 vllm-ascend（已有 `glm47` parser） |
| MindIE 不支持 MiniMax-M2.5 FC | 客户需求 | P1 | 使用 vllm-ascend（已有 `minimax_m2` parser） |
| MindIE 不支持 Qwen3.5 系列 FC | 新模型 | P1 | 使用 vllm-ascend（已有 `hermes` parser） |
| sglang 缺 GLM4.7 FC parser | 功能不完整 | P2 | 等 sglang 社区更新 |
| sglang 缺 MiniMax-M2.5 FC parser | 功能不完整 | P2 | 等 sglang 社区更新 |
| LLaMA3.1 MindIE FC 未验证 | 功能潜力 | P3 | 实测 `llama3_1` parser 并反馈 |

---

## 5. Action Items

### 5.1 短期（可立即执行）

- [ ] **明确告知客户**：MindIE 2.3.0 的 FC 覆盖范围有限，DeepSeek V3.2、GLM4.7、MiniMax-M2.5 均不支持。推荐昇腾场景使用 **vllm-ascend ≥0.17.0** 引擎获取最完整的 FC 支持
- [ ] **验证 DeepSeek V3.1 MindIE FC**：在 2 机 16 卡 910B 环境上使用 MindIE 2.3.0 拉起 DeepSeek-V3.1 并配置 `deepseek_v31` parser，验证 FC 端到端流程
- [ ] **验证 vllm-ascend 0.17.0 FC**：在 8×910B 上测试以下模型的 FC：
  - DeepSeek-Coder-V2-Instruct（W8A8，单机 8 卡）
  - Qwen3.5-27B（FP16/W8A8，单机 2~4 卡）
  - GLM4.7（FP16，单机 8 卡）
- [ ] **验证 LLaMA3.1-70B MindIE FC**：尝试在 MindIE 配置中使用 `llama3_1` parser，验证 ToolsCallProcessorLlama 是否对 LLaMA3.1-70B-Instruct 生效

### 5.2 中期（需协调华为 / 社区）

- [ ] 向华为提需求：MindIE ≥2.4.0 需支持 DeepSeek V3.2、GLM4.7 MoE、MiniMax-M2.5、Qwen3.5 的 FC
- [ ] 跟进 sglang 社区对 GLM4.7 (`glm47`) 和 MiniMax-M2.5 (`minimax_m2`) parser 的支持进展
- [ ] 准备 Qwen3.5-397-A17B 的 W8A8 量化方案（SmoothQuant / GPTQ-INT8），验证单机 8×910B 部署

### 5.3 推荐引擎选择策略

**昇腾 910B 环境下 Function Call 场景的引擎优先级**：

| 优先级 | 引擎 | 适用场景 | FC 覆盖 |
|:---:|------|---------|---------|
| **1** | **vllm-ascend ≥0.17.0** | 昇腾 910B + FC 需求 | ★★★★★ 全部目标模型 |
| 2 | **MindIE 2.3.0** | 仅 DeepSeek V3.1/R1 + Qwen3/2.5 的 FC | ★★☆ 有限 |
| 3 | **sglang** | 高吞吐场景 + DeepSeek/Qwen FC | ★★★★ 缺 GLM/MiniMax |

**NVIDIA GPU 环境下**：

| 优先级 | 引擎 | 适用场景 | FC 覆盖 |
|:---:|------|---------|---------|
| **1** | **vllm** | NVIDIA GPU + FC 需求 | ★★★★★ 全部目标模型 |
| 2 | **sglang** | 高吞吐 + DeepSeek/Qwen FC | ★★★★ |

---

## 附录 A：Wings FC 配置注入代码参考

### MindIE FC 注入（mindie_adapter.py）

```python
# wings_control/engines/mindie_adapter.py L709-L730
def _inject_function_call_config(engine_config, overrides):
    mindie_tool_call_parser = engine_config.get("mindie_tool_call_parser")
    mindie_model_type = engine_config.get("mindie_model_type")
    if not (mindie_tool_call_parser and mindie_model_type):
        return
    model_entry = {
        "tool_call_options": {"tool_call_parser": mindie_tool_call_parser},
    }
    chat_template = engine_config.get("mindie_chat_template")
    if chat_template:
        model_entry["chat_template"] = chat_template
    overrides["models"] = {mindie_model_type: model_entry}
```

### vllm / vllm-ascend FC 激活命令模板

```bash
# DeepSeek V3.1（2 机 16 卡）
vllm serve /path/to/DeepSeek-V3.1 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v31 \
    --reasoning-parser deepseek_r1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel

# DeepSeek V3.2（2 机 16 卡）
vllm serve /path/to/DeepSeek-V3.2 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseekv32 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel

# Qwen3.5-27B（单机 2 卡）
vllm serve /path/to/Qwen3.5-27B \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --task generate \
    --tensor-parallel-size 2

# GLM4.7（单机 8 卡）
vllm serve /path/to/GLM-4.7 \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel

# MiniMax-M2.5（需多机）
vllm serve /path/to/MiniMax-M2.5 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_reasoning \
    --tensor-parallel-size 8 \
    --enable-expert-parallel

# GPT-OSS-120B（单机 8 卡）
vllm serve /path/to/GPT-OSS-120B \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --tensor-parallel-size 8
```

## 附录 B：MindIE 2.3.0 完整 FC 配置模板

```json
{
    "ServerConfig": {
        "ipAddress": "0.0.0.0",
        "managementIpAddress": "0.0.0.0",
        "port": 1025,
        "managementPort": 1026,
        "httpsEnabled": false,
        "maxLinkNum": 1000,
        "sslCipher": "ECDHE-ECDSA-AES128-GCM-SHA256"
    },
    "BackendConfig": {
        "npuDeviceIds": [[0,1,2,3,4,5,6,7]],
        "ModelDeployConfig": {
            "maxSeqLen": 4096,
            "maxInputTokenLen": 2048,
            "maxIterTimes": 2048,
            "ModelConfig": [{
                "modelName": "deepseek_v31",
                "modelWeightPath": "/path/to/DeepSeek-V3.1",
                "worldSize": 8,
                "cpuMemSize": 10,
                "npuMemSize": -1,
                "backendType": "atb",
                "pluginParams": "",
                "models": {
                    "deepseekv2": {
                        "tool_call_options": {
                            "tool_call_parser": "deepseek_v31"
                        },
                        "chat_template": "/path/to/tool_chat_template_deepseekv31.jinja"
                    }
                }
            }]
        }
    },
    "ScheduleConfig": {
        "maxPrefillBatchSize": 50,
        "maxPrefillTokens": 8192,
        "prefillTimeMsPerReq": 150,
        "prefillPolicyType": 0,
        "decodeTimeMsPerReq": 50,
        "decodePolicyType": 0,
        "maxBatchSize": 200,
        "maxIterTimes": 2048,
        "maxPreemptCount": 0,
        "supportSelectBatch": false,
        "maxQueueDelayMicroseconds": 5000
    }
}
```
