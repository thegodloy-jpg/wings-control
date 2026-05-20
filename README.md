# Wings-Control

Wings-Control 是统一推理控制 Sidecar。它负责读取启动字段、合并默认配置和模型配置、生成 `/shared-volume/start_command.sh`，再由真实推理 Engine 容器执行该脚本。

项目当前按三类镜像交付：

| 镜像 | 职责 |
|------|------|
| `wings-control:<version>` | 控制面 Sidecar，生成引擎启动脚本，并托管 Proxy、Health、Monitor Proxy |
| `wings-accel:<version>` | 可选加速包镜像，通常作为 Compose 初始化容器或 K8s `initContainer` |
| Engine 镜像 | 实际推理引擎，例如 vLLM、vLLM-Ascend、SGLang、MindIE |

控制容器不直接运行推理引擎。它通过共享卷把启动脚本交给 Engine 容器执行。

## 快速入口

| 目标 | 文档 |
|------|------|
| 了解产品边界和运行链路 | [产品总览](docs/product-overview.md) |
| 查看芯片、模型、引擎、特性支持情况 | [兼容性矩阵](docs/compatibility.md) |
| 使用 Docker Compose 部署 | [Docker Compose 部署](docs/deployment/docker-compose.md) |
| 使用 K8s 部署 | [K8s 部署](docs/deployment/k8s.md) |
| 使用 PD 分离、分布式、LMCache、Sparse KV 等能力 | [特性索引](docs/features/index.md) |
| 查看完整文档目录 | [docs/README.md](docs/README.md) |

## 支持范围

部署形态只分为两类：

1. Docker Compose
2. K8s

单机多卡、多机分布式、PD 分离、LMCache、Sparse KV、投机推理、Function Call、RAG 加速、Wings Router 等都属于特性或能力场景，不再作为独立部署形态描述。

## 默认端口

| 服务 | 默认端口 | 覆盖方式 | 用途 |
|------|----------|----------|------|
| Engine | `17000` | `ENGINE_PORT` | Engine 容器内真实推理服务 |
| Proxy | `18000` | `--port` / `PROXY_PORT` / `PORT` | OpenAI 兼容 API 入口 |
| Health | `19000` | `HEALTH_PORT` | `/health` 与 `/v1/health` |
| Monitor Proxy | `19100` | `MONITOR_PROXY_PORT` | 透传 Engine 侧监控接口 |

## 最小启动字段

`wings_start.sh` 优先使用 CLI 字段；只有没有 CLI 字段的运行时变量才保留为环境变量。

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
ENABLE_REASON_PROXY="true" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --chip h20-96 \
  --device-count 2 \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --trust-remote-code
```

在 Compose 或 K8s 中，把可 CLI 化字段放入 `command` / `args`，把 `ENGINE_PORT`、`HEALTH_PORT`、`MONITOR_PROXY_PORT`、`WINGS_DEVICE`、`RANK_IP`、`PD_ROLE`、`LMCACHE_*`、`WINGS_ROUTE_*` 等运行时变量放入 `environment` / `env`。

## 启动后验证

查看生成的引擎启动脚本：

```bash
cat /shared-volume/start_command.sh
```

健康检查：

```bash
curl http://127.0.0.1:19000/health
curl http://127.0.0.1:19000/v1/health
curl http://127.0.0.1:18000/v1/models
```

常见问题优先检查：

| 现象 | 优先检查 |
|------|----------|
| `/shared-volume/start_command.sh` 未生成 | `wings-control` 日志、`MODEL_NAME`、`MODEL_PATH`、`ENGINE` |
| Health 未就绪 | Engine 容器是否执行脚本、`ENGINE_PORT` 是否一致 |
| Proxy 返回 502 | Engine 是否监听 `ENGINE_PORT`，Proxy 是否访问同一网络命名空间 |
| 分布式校验失败 | `RANK_IP`、`MASTER_IP`、`NODE_IPS`、`NNODES` 是否一致 |
| Ascend 启动失败 | NPU 设备、驱动挂载、CANN 环境、`WINGS_DEVICE=ascend`、`--engine vllm_ascend` 或 `--engine mindie` |
