# Ascend NPU PD 分离特性

> 基于 vLLM-Ascend + Wings-Control 的 Prefill-Decode 分离能力说明。PD 分离不是独立部署形态；它通过 Docker Compose 或 K8s 编排落地，`wings_start.sh` 支持的启动项统一使用 CLI 字段，环境变量只保留无 CLI 的运行时变量。

## 1. 概述

### 1.1 PD 分离架构

```
┌─────────────────────────────────────────────────────────────┐
│                     同一台物理机                              │
├────────────────────────┬────────────────────────────────────┤
│   Prefill (P) 节点      │       Decode (D) 节点              │
│  ┌──────────────────┐  │  ┌──────────────────┐             │
│  │ control-p        │  │  │ control-d        │             │
│  │ (wings_control)  │  │  │ (wings_control)  │             │
│  │ PORT=18200       │  │  │ PORT=18100       │             │
│  │ HEALTH=19400     │  │  │ HEALTH=19200     │             │
│  └────────┬─────────┘  │  └────────┬─────────┘             │
│           │            │           │                        │
│  ┌────────▼─────────┐  │  ┌────────▼─────────┐             │
│  │ engine-p         │  │  │ engine-d         │             │
│  │ (vLLM API Server)│  │  │ (vLLM API Server)│             │
│  │ ENGINE_PORT=17200│  │  │ ENGINE_PORT=17100│             │
│  │ NPU 2            │  │  │ NPU 3            │             │
│  │ kv_producer      │◄─┼──►kv_consumer       │             │
│  └──────────────────┘  │  └──────────────────┘             │
│                        │                                    │
│     Mooncake RDMA KV Transfer (ens65f1np1)                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 关键组件

| 组件 | 版本 | 作用 |
|------|------|------|
| vllm-ascend | v0.17.0rc1 | Ascend NPU 推理引擎 |
| mooncake-transfer-engine | 0.3.10.post1 | KV Cache RDMA 传输 |
| wings_control | - | 容器编排和配置管理 |
| MooncakeConnectorV1 | vllm_ascend 内置 | Ascend 专用 KV 连接器 |

## 2. 前置条件

### 2.1 硬件要求

- Ascend 910B2C NPU × 2（或更多）
- RDMA 网卡（如 ens65f1np1）
- HBM 内存: 每 NPU ≥ 32GB

### 2.2 软件要求

- Docker 20.10+ 与 Docker Compose V2，或可用的 K8s 集群
- CANN 8.5.1+
- 容器镜像: `swr.cn-south-1.myhuaweicloud.com/ascendhub/vllm-ascend:v0.17.0rc1`

### 2.3 网络要求

- P/D 节点可通过 RDMA 网卡互通
- 需要知道正确的网络接口名（如 `ens65f1np1`，非 `eth0`）

## 3. 环境变量配置

### 3.1 通用环境变量（P/D 共用）

| 字段 | 示例值 | 说明 |
|------|--------|------|
| `WINGS_DEVICE` | `ascend` | 硬件类型 |
| `WINGS_DEVICE_COUNT` | `1` | 每个 P/D 实例使用的 NPU 数 |
| `RANK_IP` | `7.6.52.170` | 当前节点 IP |
| `NETWORK_INTERFACE` | `ens65f1np1` | RDMA 网卡名 |
| `ASCEND_ENFORCE_EAGER` | `true` | A+X 环境跳过图编译 |
| `PD_PREFILL_TP_SIZE` / `PD_PREFILL_DP_SIZE` | `1` / `1` | Prefill 并行配置 |
| `PD_DECODE_TP_SIZE` / `PD_DECODE_DP_SIZE` | `1` / `1` | Decode 并行配置 |

### 3.2 P 节点专属环境变量

| 字段 | 示例值 | 说明 |
|------|--------|------|
| `PD_ROLE` | `P` | Prefill 角色 |
| `ENGINE_PORT` | `17200` | Engine 端口，避免与 D 冲突 |
| `--port` | `18200` | Proxy 端口，通过 CLI 传入 |
| `HEALTH_PORT` | `19400` | 健康检查端口 |
| `MONITOR_PROXY_PORT` | `19500` | 监控代理端口 |
| `VLLM_LLMDD_RPC_PORT` | `5569` | vLLM LLMDD RPC 端口 |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | `23000` | Mooncake bootstrap 端口 |

### 3.3 D 节点专属环境变量

| 字段 | 示例值 | 说明 |
|------|--------|------|
| `PD_ROLE` | `D` | Decode 角色 |
| `ENGINE_PORT` | `17100` | Engine 端口 |
| `--port` | `18100` | Proxy 端口，通过 CLI 传入 |
| `HEALTH_PORT` | `19200` | 健康检查端口 |
| `MONITOR_PROXY_PORT` | `19300` | 监控代理端口 |
| `VLLM_LLMDD_RPC_PORT` | `5570` | vLLM LLMDD RPC 端口 |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | `23100` | Mooncake bootstrap 端口 |

## 4. Compose / K8s 使用方式

### 4.1 Control 容器完整启动命令

Prefill 节点控制容器完整启动命令：

```bash
WINGS_DEVICE="ascend" \
WINGS_DEVICE_COUNT="1" \
PD_ROLE="P" \
ENGINE_PORT="17200" \
HEALTH_PORT="19400" \
MONITOR_PROXY_PORT="19500" \
RANK_IP="7.6.52.170" \
NETWORK_INTERFACE="ens65f1np1" \
ASCEND_ENFORCE_EAGER="true" \
PD_PREFILL_TP_SIZE="1" \
PD_PREFILL_DP_SIZE="1" \
PD_DECODE_TP_SIZE="1" \
PD_DECODE_DP_SIZE="1" \
VLLM_LLMDD_RPC_PORT="5569" \
VLLM_MOONCAKE_BOOTSTRAP_PORT="23000" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3-8B" \
  --model-path "/models/Qwen3-8B" \
  --engine vllm_ascend \
  --device-count 1 \
  --port 18200 \
  --trust-remote-code
```

Decode 节点控制容器完整启动命令：

```bash
WINGS_DEVICE="ascend" \
WINGS_DEVICE_COUNT="1" \
PD_ROLE="D" \
ENGINE_PORT="17100" \
HEALTH_PORT="19200" \
MONITOR_PROXY_PORT="19300" \
RANK_IP="7.6.52.170" \
NETWORK_INTERFACE="ens65f1np1" \
ASCEND_ENFORCE_EAGER="true" \
PD_PREFILL_TP_SIZE="1" \
PD_PREFILL_DP_SIZE="1" \
PD_DECODE_TP_SIZE="1" \
PD_DECODE_DP_SIZE="1" \
VLLM_LLMDD_RPC_PORT="5570" \
VLLM_MOONCAKE_BOOTSTRAP_PORT="23100" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3-8B" \
  --model-path "/models/Qwen3-8B" \
  --engine vllm_ascend \
  --device-count 1 \
  --port 18100 \
  --trust-remote-code
```

在 Compose/K8s 中，把上述 CLI 字段放入 `command` / `args`，把前置环境变量放入 `environment` / `env`。

### 4.2 Engine 容器

```bash
# 1. 安装 mooncake
pip install mooncake-transfer-engine

# 2. 设置 LD_LIBRARY_PATH（ascend_transport.so）
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

# 3. 等待 start_command.sh 生成
while [ ! -f /shared-volume/start_command.sh ]; do sleep 2; done

# 4. 执行启动脚本
bash /shared-volume/start_command.sh
```

### 4.3 启动

PD 场景建议沿用根目录 [../README.md](../README.md) 的三容器模板：每个 P/D 实例都是一组独立的 `wings-control` + Engine，共享卷、端口和设备号必须互相隔离。

```bash
docker compose -f docker-compose.pd-qwen3-8b.yml up -d
docker compose -f docker-compose.pd-qwen3-8b.yml logs -f
```

K8s 场景建议把 P/D 分别建成独立 Pod 或独立 Deployment，并使用不同 Service 暴露 Proxy 端口。

## 5. 关键问题与解决方案

### 5.1 问题清单（10 次迭代总结）

| # | 问题 | 错误信息 | 解决方案 |
|---|------|---------|---------|
| 1 | Docker Compose 不可用 | `docker compose: command not found` | 安装 Docker Compose V2，或改用 K8s 编排 |
| 2 | 端口冲突 | `Address already in use` | 使用不同 ENGINE_PORT/PORT |
| 3 | Shell 转义 | Python SyntaxWarning | 使用 `\\$` 替代 `\$` |
| 4 | NPU 内存满 | HBM 占用 >90% | 使用空闲 NPU |
| 5 | 设备访问失败 | `rtGetDeviceCount drvRetCode=87` | 使用 `--privileged` + `ASCEND_RT_VISIBLE_DEVICES` |
| 6 | 网卡名错误 | `Unable to find address for: eth0` | 设置 `NETWORK_INTERFACE=ens65f1np1` |
| 7 | 图编译失败 | `'vllm' has no attribute 'qkv_rmsnorm_rope'` | A+X 环境设 `ASCEND_ENFORCE_EAGER=true` |
| 8 | KV Connector 错误 | `'tuple' object has no attribute 'shape'` | 使用 `MooncakeConnectorV1`（Ascend 版） |
| 9-10 | 并行配置缺失 | `assert "tp_size" in prefill_parallel_config` | 配置 `PD_PREFILL_TP_SIZE` 等环境变量 |

### 5.2 A+X 环境说明

**A+X 环境**指 Ascend NPU 与 X86 CPU 混合部署的环境。此环境下 `triton` 和 `triton-ascend` 版本冲突，导致 `qkv_rmsnorm_rope` 等融合算子无法正确注册（参见 vllm-ascend issue [#6737](https://github.com/vllm-project/vllm-ascend/issues/6737)、[#6578](https://github.com/vllm-project/vllm-ascend/issues/6578)）。

**解决方案**: 设置 `ASCEND_ENFORCE_EAGER=true`，强制使用 eager mode 跳过图编译。

**性能影响**: 无 CUDA/NPU graph 优化，推理性能略低但功能完全正常。

### 5.3 MooncakeConnectorV1 vs MooncakeConnector

| 连接器 | 来源 | 支持 Ascend tuple KV cache |
|--------|------|---------------------------|
| `MooncakeConnector` | 上游 vllm | ❌ 不支持 |
| `MooncakeConnectorV1` | vllm_ascend | ✅ 支持 |

vllm_ascend 的 KV cache 是 tuple 格式 `(key_cache, value_cache)`，上游 vllm 的 MooncakeConnector 无法正确处理。必须使用 vllm_ascend 注册的 `MooncakeConnectorV1`。

## 6. 验证与测试

### 6.1 健康检查

```bash
# P 节点
curl -s http://127.0.0.1:19400/health
# 预期: {"status":"healthy"}

# D 节点
curl -s http://127.0.0.1:19200/health
# 预期: {"status":"healthy"}
```

### 6.2 推理测试

```bash
# 通过 P 节点 Proxy 发送请求
curl -s http://127.0.0.1:18200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "Say hello in Chinese"}],
    "max_tokens": 32
  }'
```

### 6.3 日志检查

```bash
# 查看引擎日志
docker logs pd-engine-p 2>&1 | tail -50
docker logs pd-engine-d 2>&1 | tail -50

# 成功标志
# "Application startup complete."
# "INFO:     Started server process"
```

## 7. 实现参考

本次部署对 wings_control 进行了以下关键修改：

### 7.1 port_plan.py

从环境变量读取 `ENGINE_PORT`，支持同机多实例：

```python
_engine_port = int(os.environ.get("ENGINE_PORT", "17000"))
```

### 7.2 config_loader.py

1. 使用 `MooncakeConnectorV1` 替代 `MooncakeConnector`
2. 添加 `prefill`/`decode` 并行配置到 `kv_connector_extra_config`

```python
config = {
    "kv_connector": "MooncakeConnectorV1",
    "kv_role": kv_role,
    "kv_connector_extra_config": {
        "mooncake_protocol": "rdma",
        "prefill": {"tp_size": prefill_tp, "dp_size": prefill_dp, "pp_size": prefill_pp},
        "decode": {"tp_size": decode_tp, "dp_size": decode_dp, "pp_size": decode_pp},
    },
}
```

### 7.3 vllm_adapter.py

1. 将 `--enforce-eager` 改为通过 `ASCEND_ENFORCE_EAGER` 环境变量控制
2. 添加 mooncake LD_LIBRARY_PATH 到 PD 环境变量

```python
# A+X 环境控制
def _need_enforce_eager(engine: str) -> bool:
    if engine != "vllm_ascend":
        return False
    return os.getenv("ASCEND_ENFORCE_EAGER", "").lower() in ("true", "1", "yes")

# LD_LIBRARY_PATH 注入
'export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"',
```

## 8. 参考资料

- [vllm-ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [Issue #6737: qkv_rmsnorm_rope triton conflict](https://github.com/vllm-project/vllm-ascend/issues/6737)
- [Issue #6578: 910B A+X qkv_rmsnorm_rope](https://github.com/vllm-project/vllm-ascend/issues/6578)
- [mooncake-transfer-engine PyPI](https://pypi.org/project/mooncake-transfer-engine/)
