# 特性索引

本文档汇总 Wings-Control 的特性入口。部署形态只包含 Docker Compose 和 K8s；以下内容都是可叠加在两种部署形态上的能力。

## 特性总览

| 特性 | 开启方式 | 说明 | 入口 |
|------|----------|------|------|
| 长上下文优化 | `--enable-prefix-caching`、`--enable-chunked-prefill`、`LMCACHE_OFFLOAD=true` | 降低重复前缀和长上下文场景成本 | 本页 |
| MoE Expert Parallel | `--enable-expert-parallel` | MoE 模型专家并行 | 本页 |
| 分布式 | `--distributed`、`--node-ips`、`--master-ip`、`RANK_IP` | 多节点引擎启动能力 | 本页 |
| PD 分离 | `PD_ROLE=P/D` | Prefill / Decode 分离 | [pd-disaggregation.md](pd-disaggregation.md) |
| Sparse KV | `--enable-sparse` | 稀疏 KV Cache | [../design/advanced-features-dataflow.md](../design/advanced-features-dataflow.md) |
| 投机推理 | `--enable-speculative-decode`、`--speculative-decode-model-path` | 生成 `--speculative-config` | [../design/advanced-features-dataflow.md](../design/advanced-features-dataflow.md) |
| Function Call / Tool Choice | `--enable-auto-tool-choice` | 自动工具选择与 tool call parser | [../design/model-engine-function-call-analysis.md](../design/model-engine-function-call-analysis.md) |
| RAG 加速 | `--enable-rag-acc` | RAG 相关加速能力 | 本页 |
| Wings Router | `WINGS_ROUTE_*` | 多实例路由能力 | 本页 |

## 分布式字段

分布式不是独立部署形态。它是在 Compose 或 K8s 中额外启用的特性。

```bash
RANK_IP="192.168.1.10" \
--distributed \
--master-ip "192.168.1.10" \
--node-ips "192.168.1.10,192.168.1.11" \
--head-node-addr "192.168.1.10" \
--ray-head-ip "192.168.1.10" \
--nnodes 2 \
--distributed-executor-backend ray
```

要求：

- 每个节点设置自己的 `RANK_IP`。
- 所有节点使用同一个 `--master-ip`。
- `--node-ips` 包含完整节点地址，逗号分隔。
- `--nnodes` 与 `--node-ips` 数量一致。
- `RANK_IP == MASTER_IP` 的节点作为 master。

## 长上下文优化

```bash
--enable-prefix-caching \
--enable-chunked-prefill \
--max-num-batched-tokens 8192 \
LMCACHE_OFFLOAD="true" \
LMCACHE_LOCAL_DISK="/tmp/lmcache" \
LMCACHE_MAX_LOCAL_DISK_SIZE="200GiB"
```

具体是否生效取决于当前引擎、模型架构、硬件和运行时依赖。

## MoE

```bash
--enable-expert-parallel \
--device-count 8
```

MoE 特性需要结合模型架构和引擎支持情况使用。

## RAG 加速

```bash
--enable-rag-acc
```

RAG 加速依赖 `wings_control/rag_acc` 相关服务和业务链路，使用前需要确认模型类型、请求路径和模板配置。

## Wings Router

```yaml
WINGS_ROUTE_ENABLE: "true"
WINGS_ROUTE_INSTANCE_GROUP_NAME: "qwen35-group"
WINGS_ROUTE_INSTANCE_NAME: "qwen35-0"
WINGS_ROUTE_NATS_PATH: "nats://127.0.0.1:4222"
```

Router 是流量路由特性，不改变部署形态。Compose 和 K8s 都需要额外提供路由依赖服务和实例标识。

## 维护规则

1. 每个特性文档都要说明适用引擎、适用芯片、开启方式、验证方式和限制。
2. 特性示例中可 CLI 化字段写入 `command` / `args`。
3. 无 CLI 字段的运行时变量写入 `environment` / `env`。
4. 不把特性命名为部署形态。
