# 特性索引

本文档汇总 Wings-Control 的特性入口。部署形态只包含 Docker Compose 和 K8s；以下内容都是可叠加在两种部署形态上的能力。

## 特性总览

| 特性 | 开启方式 | 说明 | 入口 |
|------|----------|------|------|
| 长上下文优化 | `--enable-prefix-caching`、`--enable-chunked-prefill`、`LMCACHE_OFFLOAD=true` | 降低重复前缀和长上下文场景成本 | 本页 |
| MoE Expert Parallel | `--enable-expert-parallel` | MoE 模型专家并行 | 本页 |
| 分布式 | `--distributed`、`RANK_IP`、`MASTER_IP`、`NODE_IPS`、`NNODES` | 多节点引擎启动能力 | 本页 |
| PD 分离 | `PD_ROLE=P/D` | Prefill / Decode 分离 | [pd-disaggregation.md](pd-disaggregation.md) |
| Sparse KV | `--enable-sparse` | 稀疏 KV Cache | [../design/advanced-features-dataflow.md](../design/advanced-features-dataflow.md) |
| 投机推理 | `--enable-speculative-decode`、`--speculative-decode-model-path` | 生成 `--speculative-config` | [../design/advanced-features-dataflow.md](../design/advanced-features-dataflow.md) |
| Function Call / Tool Choice | `--enable-auto-tool-choice` | 自动工具选择与 tool call parser | [../design/model-engine-function-call-analysis.md](../design/model-engine-function-call-analysis.md) |
| RAG 加速 | `--enable-rag-acc` | RAG 相关加速能力 | 本页 |
| Wings Router | `WINGS_ROUTE_*` | 多实例路由能力 | 本页 |

## 分布式字段

分布式不是独立部署形态。它是在 Compose 或 K8s 中额外启用的特性。

Master 节点完整启动命令：

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
RANK_IP="192.168.1.10" \
MASTER_IP="192.168.1.10" \
NODE_IPS="192.168.1.10,192.168.1.11" \
NNODES="2" \
HEAD_NODE_ADDR="192.168.1.10" \
RAY_HEAD_IP="192.168.1.10" \
DISTRIBUTED_EXECUTOR_BACKEND="ray" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --device-count 2 \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --model-type llm \
  --trust-remote-code \
  --distributed
```

Worker 节点完整启动命令：

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
RANK_IP="192.168.1.11" \
MASTER_IP="192.168.1.10" \
NODE_IPS="192.168.1.10,192.168.1.11" \
NNODES="2" \
HEAD_NODE_ADDR="192.168.1.10" \
RAY_HEAD_IP="192.168.1.10" \
DISTRIBUTED_EXECUTOR_BACKEND="ray" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --device-count 2 \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --model-type llm \
  --trust-remote-code \
  --distributed
```

要求：

- 每个节点设置自己的 `RANK_IP`。
- 所有节点使用同一个 `MASTER_IP`。
- `NODE_IPS` 包含完整节点地址，逗号分隔。
- `NNODES` 与 `NODE_IPS` 数量一致。
- `RANK_IP == MASTER_IP` 的节点作为 master。

## 长上下文优化

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
LMCACHE_OFFLOAD="true" \
LMCACHE_LOCAL_DISK="/tmp/lmcache" \
LMCACHE_MAX_LOCAL_DISK_SIZE="200GiB" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --device-count 2 \
  --port 18000 \
  --input-length 32768 \
  --output-length 4096 \
  --model-type llm \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192
```

具体是否生效取决于当前引擎、模型架构、硬件和运行时依赖。

## MoE

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="8" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3-235B-A22B" \
  --model-path "/models/Qwen3-235B-A22B" \
  --engine vllm \
  --device-count 8 \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --model-type llm \
  --trust-remote-code \
  --enable-expert-parallel
```

MoE 特性需要结合模型架构和引擎支持情况使用。

## RAG 加速

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --device-count 2 \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --model-type llm \
  --trust-remote-code \
  --enable-rag-acc
```

RAG 加速依赖 `wings_control/rag_acc` 相关服务和业务链路，使用前需要确认模型类型、请求路径和模板配置。

## Wings Router

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
WINGS_ROUTE_ENABLE="true" \
WINGS_ROUTE_INSTANCE_GROUP_NAME="qwen35-group" \
WINGS_ROUTE_INSTANCE_NAME="qwen35-0" \
WINGS_ROUTE_NATS_PATH="nats://127.0.0.1:4222" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --device-count 2 \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --model-type llm \
  --trust-remote-code
```

Router 是流量路由特性，不改变部署形态。Compose 和 K8s 都需要额外提供路由依赖服务和实例标识。

## 维护规则

1. 每个特性文档都要说明适用引擎、适用芯片、开启方式、验证方式和限制。
2. 特性示例中可 CLI 化字段写入 `command` / `args`。
3. 无 CLI 字段的运行时变量写入 `environment` / `env`。
4. 不把特性命名为部署形态。
