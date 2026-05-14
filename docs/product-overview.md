# 产品总览

Wings-Control 是推理服务控制面 Sidecar。它不替代 vLLM、vLLM-Ascend、SGLang 或 MindIE，而是统一处理启动字段、配置合并、端口规划、特性开关和启动脚本生成。

## 运行链路

```text
用户配置 / CLI / 环境变量
        |
        v
wings-control
  - 解析 wings_start.sh 字段
  - 合并默认配置、模型配置、硬件配置和用户配置
  - 生成 /shared-volume/start_command.sh
  - 启动 Proxy / Health / Monitor Proxy
        |
        v
Engine 容器
  - 等待 start_command.sh
  - 执行真实推理引擎命令
```

## 镜像职责

| 镜像 | 是否常驻 | 职责 |
|------|----------|------|
| `wings-accel:<version>` | 否 | 准备可选加速包，通常作为 Compose 初始化容器或 K8s `initContainer` |
| `wings-control:<version>` | 是 | 生成启动脚本，提供 Proxy、Health、Monitor Proxy |
| Engine 镜像 | 是 | 运行 vLLM、vLLM-Ascend、SGLang 或 MindIE |

## 支持引擎

| 引擎值 | 典型硬件 | 说明 |
|--------|----------|------|
| `vllm` | NVIDIA | 默认通用路径 |
| `vllm_ascend` | Ascend | Ascend 通用路径 |
| `sglang` | NVIDIA | 高性能路径，需结合模型支持情况使用 |
| `mindie` | Ascend | MindIE 服务化路径 |

## 部署形态

部署形态只包含两类：

| 部署形态 | 入口 |
|----------|------|
| Docker Compose | [deployment/docker-compose.md](deployment/docker-compose.md) |
| K8s | [deployment/k8s.md](deployment/k8s.md) |

单机多卡、多机分布式、PD 分离、LMCache、Sparse KV、投机推理、Function Call、RAG 加速、Wings Router 属于特性专题，不再作为独立部署形态。

## 共享卷

| 路径 | 生产方 | 消费方 | 说明 |
|------|--------|--------|------|
| `/shared-volume/start_command.sh` | `wings-control` | Engine 容器 | 最终引擎启动脚本 |
| `/shared-volume/progress.jsonl` | Engine / log analyzer | `wings-control` / 运维侧 | 启动进度记录 |
| `/shared-volume/accel_features.jsonl` | Engine / log analyzer | `wings-control` / 运维侧 | 加速特性记录 |
| `/accel-volume` | `wings-accel` | `wings-control` / Engine 容器 | 可选加速包目录 |

## 端口

| 服务 | 默认端口 | 覆盖方式 | 用途 |
|------|----------|----------|------|
| Engine | `17000` | `ENGINE_PORT` | Engine 容器内真实推理服务 |
| Proxy | `18000` | `--port` / `PROXY_PORT` / `PORT` | OpenAI 兼容 API 入口 |
| Health | `19000` | `HEALTH_PORT` | `/health` 与 `/v1/health` |
| Monitor Proxy | `19100` | `MONITOR_PROXY_PORT` | 透传 Engine 侧监控接口 |

## 启动字段原则

`wings_start.sh` 既支持 CLI 参数，也支持环境变量。文档示例统一优先使用 CLI 字段；只有脚本没有对应 CLI 的运行时变量才保留为环境变量，例如 `ENGINE_PORT`、`HEALTH_PORT`、`MONITOR_PROXY_PORT`、`WINGS_DEVICE`、`RANK_IP`、`PD_ROLE`、`LMCACHE_*`、`WINGS_ROUTE_*`。
