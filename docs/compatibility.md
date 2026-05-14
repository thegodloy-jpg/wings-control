# 兼容性矩阵

本文档定义产品文档中的兼容性口径。兼容性不是单个维度，而是 `芯片 × 引擎 × 引擎版本 × 模型 × 特性` 的组合结果。

## 状态口径

| 状态 | 含义 |
|------|------|
| 已验证 | 仓库配置中有验证记录或明确示例，推荐用户优先使用 |
| 实验性 | 可通过 experimental 配方或专项文档启用，需要显式接受风险 |
| 理论支持 | 参数映射或上游能力存在，但当前仓库未给出完整验证证据 |
| 不支持 | 当前适配器、引擎或模型组合不支持 |
| 待验证 | 信息不足，不能作为交付承诺 |

## 芯片

当前硬件 profile 覆盖以下芯片 ID：

| 芯片 ID | 厂商 | 显存 | 互联 | 备注 |
|---------|------|------|------|------|
| `910b-32` | Ascend | 32GB | HCCL | 大模型长上下文需量化或并行切分 |
| `910b-64` | Ascend | 64GB | HCCL | Ascend 通用路径 |
| `910c` | Ascend | 待补充 | HCCL | 当前存在 experimental 配方示例 |
| `h20-96` | NVIDIA | 96GB | NVLink | NVIDIA 通用路径 |
| `h20-141` | NVIDIA | 141GB | NVLink | 大显存 NVIDIA 路径 |
| `l20` | NVIDIA | 48GB | PCIe | 中等显存场景 |
| `rtx-pro-5000` | NVIDIA | 32GB | PCIe | 小规模验证或开发场景 |

## 引擎

| 引擎值 | 当前 manifest 版本 | 参数承载方式 | 典型硬件 | 状态 |
|--------|--------------------|--------------|----------|------|
| `vllm` | `0.11.0` | CLI | NVIDIA | 已验证路径 |
| `vllm_ascend` | `0.11.0` | CLI + Ascend 环境变量 | Ascend | 已验证路径 |
| `sglang` | `0.4.0` | CLI | NVIDIA | 需结合模型验证 |
| `mindie` | `2.0.0` | 生成配置文件 / additional config | Ascend | 需结合模型验证 |

## 模型

当前模型兼容性不是完备的模型 × 引擎 × 芯片矩阵。运行时代码首先依赖
`wings_control/utils/model_utils.py` 的 `ModelIdentifier`：

1. 读取模型目录下的 `config.json`。
2. 解析 `architectures[0]`、量化字段和用户传入的 `model_type`。
3. 通过 `_LLM_MODELS`、`_EMBEDDING_MODELS`、`_RERANK_MODELS` 判断模型类型和 Wings 已知模型集合。
4. 再由 `wings_control/core/config_loader.py` 结合硬件、引擎和默认配置做自动引擎选择与参数合并。

因此，本节只能描述“当前代码可识别的模型/架构口径”，不能替代实测兼容性结论。没有出现在
`model_utils.py` 映射表中的模型仍可能按 `llm` 默认路径尝试启动，但应标为“待验证”。

| 模型类型 | 架构 | `model_utils.py` 中的已知模型示例 | 兼容性口径 |
|----------|------|------------------------------------|------------|
| LLM | `DeepseekV3ForCausalLM` | `DeepSeek-R1`、`DeepSeek-V3.1`、`DeepSeek-R1-w8a8` | 已知架构，具体引擎/芯片组合需看 defaults、示例和实测 |
| LLM | `DeepseekV32ForCausalLM` | `DeepSeek-V3.2`、`DeepSeek-V3.2-Exp` | 已知架构，Sparse/IndexCache 相关能力由代码分支控制 |
| LLM | `Glm4ForCausalLM` | `GLM-4-9B-0414` | 已知架构，需结合引擎验证 |
| LLM | `GlmMoeDsaForCausalLM` | `GLM-5`、`GLM-5.1-FP8` | 已知架构，属于混合 KV / IndexCache 相关重点架构 |
| LLM | `Glm4MoeForCausalLM` | `GLM-4.7`、`GLM-4.7-w8a8` | 已知架构，需结合引擎验证 |
| LLM | `Qwen2ForCausalLM` | `DeepSeek-R1-Distill-Qwen-32B`、`Qwen2.5-32B-Instruct`、`QwQ-32B` | 已知架构，需结合引擎验证 |
| LLM | `Qwen3ForCausalLM` | `Qwen3-32B` | 已知架构，Embedding 场景存在同架构特殊分类 |
| LLM | `Qwen3MoeForCausalLM` | `Qwen3-30B-A3B`、`Qwen3-235B-A22B` | 已知 MoE 架构，Expert Parallel 需结合引擎/硬件 |
| LLM | `Qwen3NextForCausalLM` | `Qwen3-Next-80B-A3B-Instruct` | 已知架构，需结合引擎验证 |
| LLM | `Qwen3_5ForConditionalGeneration` | `Qwen3.5-27B`、`Qwen3.5-27B-Instruct` | 已知架构，当前示例主线模型 |
| LLM | `Qwen3_5MoeForConditionalGeneration` | `Qwen3.5-397-A17B`、`Qwen3.5-397-A17B-w8a8` | 已知 MoE 架构，需结合引擎/硬件 |
| LLM | `MiniMaxM2ForCausalLM` | `MiniMax-M2.5`、`MiniMax-M2.7-w8a8` | 已知架构，需结合引擎验证 |
| LLM | `LlamaForCausalLM` | `LLaMA3.1-70B-Instruct`、`DeepSeek-R1-Distill-Llama-70B` | 已知架构，需结合引擎验证 |
| Embedding | `XLMRobertaModel` | `bge-m3` | 已知 embedding 类型 |
| Embedding | `BertModel` | `bge-large-zh-v1.5` | 已知 embedding 类型 |
| Embedding | `Qwen3ForCausalLM` | `Qwen3-Embedding-0.6B` | 同架构按模型名识别为 embedding |
| Rerank | `XLMRobertaForSequenceClassification` | `bge-reranker-v2-m3`、`bge-reranker-large` | 已知 rerank 类型 |

维护规则：

1. 新增模型兼容性时，先确认 `model_utils.py` 是否能正确识别模型类型和架构。
2. 新增模型名称时，同步更新 `_LLM_MODELS`、`_EMBEDDING_MODELS` 或 `_RERANK_MODELS`。
3. 新增架构时，同时检查 `wings_control/config/defaults/*.json`、`config_loader.py`、引擎适配器和测试。
4. 文档中的“已知模型”只表示代码识别范围；只有示例、测试或部署记录覆盖的组合才能写成“已验证”。

## 特性

| 特性 | 主要开关 | 支持口径 |
|------|----------|----------|
| Prefix Cache | `--enable-prefix-caching` | vLLM / vLLM-Ascend / SGLang 参数映射存在，需结合模型和硬件限制 |
| Chunked Prefill | `--enable-chunked-prefill` | vLLM / vLLM-Ascend 支持，SGLang 通过 chunked prefill size 映射 |
| Expert Parallel | `--enable-expert-parallel` | MoE 模型场景，需结合引擎和模型 |
| Speculative Decoding | `--enable-speculative-decode` | 主要由 vLLM / vLLM-Ascend 适配器生成 `--speculative-config` |
| Sparse KV | `--enable-sparse` | 依赖模型架构、补丁和引擎支持 |
| LMCache Offload | `LMCACHE_OFFLOAD=true` | 依赖引擎、补丁目标和运行时库 |
| PD Disaggregation | `PD_ROLE=P/D` | 特性专题，部署仍落在 Compose 或 K8s |
| Function Call / Tool Choice | `--enable-auto-tool-choice` | 需结合模型 parser、chat template 和引擎支持 |
| RAG Acceleration | `--enable-rag-acc` | 需结合业务链路和运行时服务 |
| Wings Router | `WINGS_ROUTE_*` | 路由特性，部署仍落在 Compose 或 K8s |

## 维护规则

1. 新增芯片或设备型号时，先确认 `wings_control/core/hardware_detect.py` 和相关默认配置能表达该硬件。
2. 新增引擎或引擎版本时，先检查 `wings_control/engines/`、`wings_control/config/defaults/` 和参数映射。
3. 新增模型时，先补 `wings_control/utils/model_utils.py` 的模型名称或架构识别，再补 defaults、适配器分支和测试。
4. 产品文档只引用验证结论，不把临时调试结论写成支持承诺。
5. 没有验证证据的组合必须标为理论支持或待验证。
