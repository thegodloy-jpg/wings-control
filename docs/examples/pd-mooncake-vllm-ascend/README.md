# vllm-ascend + Mooncake PD 分离 start_command.sh 示例

本目录给出当前 `wings_control` 项目在 vllm-ascend 场景下，使用 Mooncake 拉起 PD 分离时，Prefill(P) 节点和 Decode(D) 节点对应的完整 `start_command.sh` 示例。

## 示例假设

- 模型：`Qwen3-8B`
- 模型路径：`/models/Qwen3-8B`
- 引擎：`vllm_ascend`
- Connector：`MooncakeConnectorV1`
- 传输协议：`rdma`
- 并行配置：Prefill/Decode 均为 `tp=1, dp=1, pp=1`
- 网卡：`ens65f1np1`
- 本机业务 IP：`7.6.52.170`
- P 节点端口：engine `17000`、`VLLM_LLMDD_RPC_PORT=5569`、`VLLM_MOONCAKE_BOOTSTRAP_PORT=23000`
- D 节点端口：engine `17100`、`VLLM_LLMDD_RPC_PORT=5570`、`VLLM_MOONCAKE_BOOTSTRAP_PORT=23100`

## 文件说明

- [start_command_p.sh](start_command_p.sh)：Prefill(P) 节点，`kv_role=kv_producer`。
- [start_command_d.sh](start_command_d.sh)：Decode(D) 节点，`kv_role=kv_consumer`。

## 关键差异

| 项 | P 节点 | D 节点 |
| --- | --- | --- |
| `PD_ROLE` 对应语义 | Prefill | Decode |
| `kv_role` | `kv_producer` | `kv_consumer` |
| engine port | `17000` | `17100` |
| `VLLM_LLMDD_RPC_PORT` | `5569` | `5570` |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | `23000` | `23100` |

## 生成逻辑对应代码

- PD/Mooncake 环境变量来自 [../../../wings_control/engines/vllm_adapter.py](../../../wings_control/engines/vllm_adapter.py) 中 `_build_pd_role_env_commands()`。
- `kv_transfer_config` 来自 [../../../wings_control/core/config_loader.py](../../../wings_control/core/config_loader.py) 中 `_get_pd_config()`。
- Ascend PD 不走 `dp_deployment`，而是 P/D 独立 vLLM 进程 + Mooncake RDMA KV Transfer；该分支在 [../../../wings_control/core/config_loader.py](../../../wings_control/core/config_loader.py) 中 `_handle_vllm_distributed()`。

## 注意

这里是本地可读示例，不会直接修改远端容器或服务。实际部署时，`HCCL_IF_IP`、`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME`、端口和模型路径应按目标机器环境替换。
