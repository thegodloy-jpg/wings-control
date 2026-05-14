# Wings-Control 文档

本文档目录按使用者路径组织。部署形态只包含 Docker Compose 和 K8s；单机多卡、多机分布式、PD 分离、LMCache、Sparse KV 等归入特性专题。

## 产品文档

| 文档 | 说明 |
|------|------|
| [product-overview.md](product-overview.md) | 产品定位、运行链路、镜像职责、端口和共享卷 |
| [compatibility.md](compatibility.md) | 芯片、引擎、模型、特性的支持矩阵和状态口径 |

## 部署形态

| 文档 | 说明 |
|------|------|
| [deployment/docker-compose.md](deployment/docker-compose.md) | Docker Compose 三容器部署方式 |
| [deployment/k8s.md](deployment/k8s.md) | K8s initContainer + 同 Pod Sidecar/Engine 部署方式 |

## 特性专题

| 文档 | 说明 |
|------|------|
| [features/index.md](features/index.md) | 特性总览、支持边界和推荐阅读顺序 |
| [features/pd-disaggregation.md](features/pd-disaggregation.md) | Ascend PD 分离能力说明 |

## 示例

| 文档 | 说明 |
|------|------|
| [examples/qwen35-27b/README.md](examples/qwen35-27b/README.md) | Qwen3.5-27B Compose / K8s 示例文件 |
| [examples/pd-mooncake-vllm-ascend/README.md](examples/pd-mooncake-vllm-ascend/README.md) | PD + Mooncake 生成脚本示例 |

## 研发参考

设计分析、实现数据流、历史专项部署记录放在 [design/](design/) 和 [reference/](reference/) 下。它们用于研发追溯，不作为新用户的第一阅读入口。

| 目录 | 说明 |
|------|------|
| [design/](design/) | 设计方案、配置合并、参数映射、高级特性数据流 |
| [reference/](reference/) | 历史场景文档、安装记录、测试说明和环境记录 |

## 统一写法

1. 示例优先使用 `wings_control/wings_start.sh` 的 CLI 字段。
2. 只有脚本没有 CLI 字段的运行时变量才保留为环境变量。
3. 部署文档只描述 Compose 和 K8s 编排。
4. PD、分布式、LMCache、Sparse KV、Function Call、RAG 等按特性专题描述。
5. 兼容性必须标明状态：已验证、实验性、理论支持、不支持、待验证。
