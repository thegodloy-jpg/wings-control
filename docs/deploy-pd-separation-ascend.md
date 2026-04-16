# Ascend NPU PD 分离部署指南

> 基于 vllm-ascend v0.17.0rc1 + wings_control 的 Prefill-Decode 分离部署实践

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

- Docker 20.10+（无需 docker-compose）
- CANN 8.5.1+
- 容器镜像: `swr.cn-south-1.myhuaweicloud.com/ascendhub/vllm-ascend:v0.17.0rc1`

### 2.3 网络要求

- P/D 节点可通过 RDMA 网卡互通
- 需要知道正确的网络接口名（如 `ens65f1np1`，非 `eth0`）

## 3. 环境变量配置

### 3.1 通用环境变量（P/D 共用）

```bash
# 基础配置
DEVICE=ascend
DEVICE_COUNT=1                    # 每个节点使用的 NPU 数
ENGINE=vllm_ascend
MODEL_NAME=Qwen3-8B
MODEL_PATH=/models/Qwen3-8B

# 网络配置
RANK_IP=7.6.52.170               # 当前节点 IP
NETWORK_INTERFACE=ens65f1np1      # RDMA 网卡名（关键！）

# A+X 环境配置（triton/triton-ascend 版本冲突）
ASCEND_ENFORCE_EAGER=true         # A+X 环境必须设为 true

# PD 并行配置（MooncakeConnectorV1 要求）
PD_PREFILL_TP_SIZE=1
PD_PREFILL_DP_SIZE=1
PD_DECODE_TP_SIZE=1
PD_DECODE_DP_SIZE=1
```

### 3.2 P 节点专属环境变量

```bash
PD_ROLE=P
ENGINE_PORT=17200                 # 引擎端口（避免与 D 冲突）
PORT=18200                        # Proxy 端口
HEALTH_PORT=19400                 # 健康检查端口
MONITOR_PROXY_PORT=19500
VLLM_LLMDD_RPC_PORT=5569
VLLM_MOONCAKE_BOOTSTRAP_PORT=23000
```

### 3.3 D 节点专属环境变量

```bash
PD_ROLE=D
ENGINE_PORT=17100
PORT=18100
HEALTH_PORT=19200
MONITOR_PROXY_PORT=19300
VLLM_LLMDD_RPC_PORT=5570
VLLM_MOONCAKE_BOOTSTRAP_PORT=23100
```

## 4. 容器部署

### 4.1 Control 容器

```bash
# 安装依赖
pip install orjson fastapi uvicorn pydantic pydantic-settings httpx python-dotenv

# 启动 wings_control
python3 -m wings_control
```

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

### 4.3 Docker Run 完整示例

```bash
# P 节点 - Control
docker run -d --name pd-control-p --network host \
  -e PD_ROLE=P -e ENGINE_PORT=17200 -e PORT=18200 \
  -e HEALTH_PORT=19400 -e NETWORK_INTERFACE=ens65f1np1 \
  -e ASCEND_ENFORCE_EAGER=true \
  -v pd-shared-vol-p:/shared-volume \
  -v /data/zhanghui/wings-control:/opt/wings-control:ro \
  vllm-ascend:v0.17.0rc1 \
  python3 -m wings_control

# P 节点 - Engine
docker run -d --name pd-engine-p --network host --privileged \
  -e ASCEND_RT_VISIBLE_DEVICES=2 \
  -v pd-shared-vol-p:/shared-volume \
  -v /data/xqr/Qwen3-8B:/models/Qwen3-8B:ro \
  vllm-ascend:v0.17.0rc1 \
  /bin/sh -c "pip install mooncake-transfer-engine && \
    export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH && \
    while [ ! -f /shared-volume/start_command.sh ]; do sleep 2; done && \
    bash /shared-volume/start_command.sh"
```

## 5. 关键问题与解决方案

### 5.1 问题清单（10 次迭代总结）

| # | 问题 | 错误信息 | 解决方案 |
|---|------|---------|---------|
| 1 | Docker 版本太旧 | `docker compose: command not found` | 使用原生 `docker run` |
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

## 7. wings_control 代码修改点

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
