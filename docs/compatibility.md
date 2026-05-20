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
| `vllm` | `0.19.0` | CLI | NVIDIA | 已验证路径 |
| `vllm_ascend` | `0.18.0` | CLI + Ascend 环境变量 | Ascend | 已验证路径 |
| `sglang` | `0.5.9` | CLI | NVIDIA | 需结合模型验证 |
| `mindie` | `2.3.0` | 生成配置文件 / additional config | Ascend | 需结合模型验证 |

## 模型

当前模型配置以 recipe 和 architecture recipe 为准。示例：

| 模型/架构 | 配置来源 | 已知状态 |
|-----------|----------|----------|
| `Qwen3-32B` | `wings_control/config/recipes/models/Qwen3-32B.yaml` | `vllm_ascend` + `910b-32` / `910b-64` 有验证记录 |
| `Qwen3-32B-910C-experimental` | `wings_control/config/recipes/models/_experimental/Qwen3-32B-910c.yaml` | `910c` 实验性 |
| `Qwen3ForCausalLM` | `wings_control/config/recipes/architectures/qwen3.yaml` | 架构级默认配置，需被具体模型验证收敛 |

完整模型矩阵应优先由 recipe、architecture recipe 和 `docs/model_engine_support_matrix.xlsx` 生成，避免手写表格与配置漂移。

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

1. 新增芯片时，先补 `wings_control/config/hardware_profiles/`。
2. 新增引擎版本时，先补 `wings_control/config/manifests/`。
3. 新增模型时，先补 model recipe 或 architecture recipe。
4. 产品文档只引用验证结论，不把临时调试结论写成支持承诺。
5. 没有验证证据的组合必须标为理论支持或待验证。
