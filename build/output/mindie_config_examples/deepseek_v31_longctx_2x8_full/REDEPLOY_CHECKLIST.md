# DeepSeek-V3.1 2x8 长上下文现场重部署核对清单

用于确认现场 Pod、镜像、ConfigMap 和启动脚本已经使用包含修复的版本，避免继续运行旧配置。

## 1. 代码版本

- 最新修复提交应至少包含：
  - `b0237bf Add 910B HCCL env override template`
  - `4b7a86c Add MindIE HCCL diagnostics for 910B`
  - `44342d6 Skip implicit MindIE GPU memory default`
  - `de0462c Remove MindIE default NPU memory override`
  - `772c8b8 Add DeepSeek V3.1 long-context start scripts`
  - `46ccc08 Align MindIE CP SP parallel defaults`
- 如果部署镜像内能看到源码，确认仓库 HEAD 不早于 `b0237bf`。

## 2. 启动脚本来源

- 长上下文 2x8 场景必须使用本目录脚本：
  - `start_command_node0.sh`
  - `start_command_node1.sh`
- 不要使用旧的非长上下文目录：
  - `../deepseek_v31_2x8_full/`
- 旧目录是普通 2x8 baseline，可能仍显示 `tp=16`；它不是 CP/SP 长上下文示例。

## 3. config.json 关键字段

现场最终打印的 MindIE `config.json` 必须满足：

| 字段 | 期望值 |
| --- | --- |
| `BackendConfig.ModelDeployConfig.maxSeqLen` | `12048` |
| `BackendConfig.ModelDeployConfig.maxInputTokenLen` | `10000` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].worldSize` | `8` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].dp` | `1` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].tp` | `8` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].sp` | `8` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].cp` | `2` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].moe_tp` | `1` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].moe_ep` | `16` |
| 根级 `enable_ep_moe` | `true` |

如果看到 `tp=16` 且同时看到 `cp=2`，说明仍在运行旧逻辑或旧脚本。

## 4. 环境变量关键字段

现场 engine 日志中应只出现默认：

- `NPU_MEMORY_FRACTION=0.96`

不应再出现：

- `NPU_MEMORY_FRACTION=0.9`
- `NPU_MEMORY_FRACTION=0.8`

如果仍看到 `0.9`，优先检查 Pod 是否仍挂载旧 `start_command.sh` 或旧 ConfigMap。
如果看到 `0.8`，优先检查镜像内源码是否未包含 `de0462c`。
如果现场显式设置了 `GPU_MEMORY_UTILIZATION` 或 `--gpu-memory-utilization`，则会按用户显式值覆盖 `0.96`；未显式设置时不应由 argparse 默认值 `0.9` 自动覆盖。

## 5. 分布式环境关键字段

两个节点都应打印：

| 字段 | node0 | node1 |
| --- | --- | --- |
| `MASTER_ADDR` | master 节点 IP | master 节点 IP |
| `RANK` | `0` | `1` |
| `WORLD_SIZE` | `16` | `16` |
| `MINDIE_MODEL_WORLD_SIZE` | `16` | `16` |
| `RANK_TABLE_FILE` | `/shared-volume/hccl_ranktable.json` | `/shared-volume/hccl_ranktable.json` |

## 6. rank table

- `rank_table_all.json` 必须存在于 `RANK_TABLE_PATH` 指向的位置，默认 `/workspace/rank_table_all.json`。
- adapter 只校验文件存在；复制到共享卷后会把每个 server 内的 `device_id` 归一化为 `0..N-1`。

## 7. 判定结论

- 若最终 config 为 `dp=1,tp=8,sp=8,cp=2` 且 `enable_ep_moe=true`：CP/SP 配置正确。
- 若 `NPU_MEMORY_FRACTION` 只剩 `0.96`：环境默认值没有二次覆盖。
- 若仍出现旧值，优先重建镜像、刷新 ConfigMap，并重启两个 Pod，确保启动脚本来自最新提交。

## 8. Kubernetes 现场排查命令

以下命令以 Bash 写法为例；执行前替换命名空间、Pod 名和容器名。

```bash
NS=<namespace>
POD0=<node0-pod-name>
POD1=<node1-pod-name>
ENGINE_CONTAINER=engine
CONTROL_CONTAINER=wings-control
```

### 8.1 确认 Pod、节点和镜像

```bash
kubectl -n "$NS" get pod "$POD0" "$POD1" -o wide
kubectl -n "$NS" get pod "$POD0" -o jsonpath='{range .spec.containers[*]}{.name}{"\t"}{.image}{"\n"}{end}'
kubectl -n "$NS" get pod "$POD1" -o jsonpath='{range .spec.containers[*]}{.name}{"\t"}{.image}{"\n"}{end}'
```

重点确认：两个 Pod 使用的新镜像必须包含 `8cf2e47` 或更晚提交。

### 8.2 确认挂载和 ConfigMap 没有使用旧脚本

```bash
kubectl -n "$NS" describe pod "$POD0" | sed -n '/Volumes:/,/QoS Class:/p'
kubectl -n "$NS" describe pod "$POD1" | sed -n '/Volumes:/,/QoS Class:/p'
kubectl -n "$NS" get configmap | grep -Ei 'mindie|wings|start|engine|deepseek'
```

重点确认：如果启动脚本来自 ConfigMap，必须重新生成并滚动重启 Pod；只更新镜像但仍挂载旧 ConfigMap 会继续复现旧值。

### 8.3 直接检查共享卷中的 start_command.sh

```bash
kubectl -n "$NS" exec "$POD0" -c "$ENGINE_CONTAINER" -- sh -lc \
  'grep -nE "NPU_MEMORY_FRACTION|\"tp\"|\"sp\"|\"cp\"|\"moe_ep\"|enable_ep_moe" /shared-volume/start_command.sh || true'

kubectl -n "$NS" exec "$POD1" -c "$ENGINE_CONTAINER" -- sh -lc \
  'grep -nE "NPU_MEMORY_FRACTION|\"tp\"|\"sp\"|\"cp\"|\"moe_ep\"|enable_ep_moe" /shared-volume/start_command.sh || true'
```

期望看到：

- `export NPU_MEMORY_FRACTION=0.96`
- `"tp": 8`
- `"sp": 8`
- `"cp": 2`
- `"moe_ep": 16`
- `"enable_ep_moe": true`

不应看到：

- `NPU_MEMORY_FRACTION=0.9`
- `NPU_MEMORY_FRACTION=0.8`
- CP/SP 长上下文场景下的 `"tp": 16`

### 8.4 检查 MindIE 最终 config.json

```bash
kubectl -n "$NS" exec "$POD0" -c "$ENGINE_CONTAINER" -- sh -lc \
  'grep -nE "\"worldSize\"|\"dp\"|\"tp\"|\"sp\"|\"cp\"|\"moe_tp\"|\"moe_ep\"|enable_ep_moe|maxSeqLen|maxInputTokenLen" /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json || true'

kubectl -n "$NS" exec "$POD1" -c "$ENGINE_CONTAINER" -- sh -lc \
  'grep -nE "\"worldSize\"|\"dp\"|\"tp\"|\"sp\"|\"cp\"|\"moe_tp\"|\"moe_ep\"|enable_ep_moe|maxSeqLen|maxInputTokenLen" /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json || true'
```

如果这里仍是旧值，但 `/shared-volume/start_command.sh` 已经是新值，说明 MindIE 进程没有重启或配置文件没有被重新 merge。

### 8.5 检查 launcher 和 engine 日志

```bash
kubectl -n "$NS" logs "$POD0" -c "$CONTROL_CONTAINER" --tail=300 | grep -Ei 'start_command|globalWorldSize|configWorldSize|cp|sp|tp|NPU_MEMORY_FRACTION|enable_ep_moe'
kubectl -n "$NS" logs "$POD1" -c "$CONTROL_CONTAINER" --tail=300 | grep -Ei 'start_command|globalWorldSize|configWorldSize|cp|sp|tp|NPU_MEMORY_FRACTION|enable_ep_moe'

kubectl -n "$NS" logs "$POD0" -c "$ENGINE_CONTAINER" --tail=500 | grep -Ei 'mindie-env|config.json|World size|NPU_MEMORY_FRACTION|enable_ep_moe|moe_ep|\"tp\"|\"sp\"|\"cp\"'
kubectl -n "$NS" logs "$POD1" -c "$ENGINE_CONTAINER" --tail=500 | grep -Ei 'mindie-env|config.json|World size|NPU_MEMORY_FRACTION|enable_ep_moe|moe_ep|\"tp\"|\"sp\"|\"cp\"'
```

若 engine 日志仍报 `World size must equal to attention's dp_size * attention's cp_size * attention's tp_size`，优先看同一段日志上方打印的最终 `config.json`，确认是否仍为旧的 `dp=1,tp=16,cp=2`。

### 8.6 强制刷新建议

如果上述检查发现旧脚本或旧 config，建议按顺序处理：

1. 重新构建并推送包含 `8cf2e47` 或更晚提交的镜像。
2. 重新生成或更新挂载 start script 的 ConfigMap。
3. 删除或滚动重启两个 Pod，确保共享卷中的 `/shared-volume/start_command.sh` 重新生成。
4. 重启后先检查 `/shared-volume/start_command.sh`，再看 MindIE 最终 `config.json` 和 engine 日志。

## 9. 910B 双机 HCCL 建组失败专项检查

如果 engine 日志报：

```text
External Comm Manager: Create the hccl communication group failed
```

优先按 910B HCCN/RDMA 网络排查，而不是先怀疑 MindIE config 的 `tp/dp`。910B 双机通常比 910C 更依赖正确的 HCCN device IP 和 `/etc/hccn.conf` 挂载。

### 9.1 必查现象

从日志中确认以下字段：

- `WORLD_SIZE` 是否等于全局 rank 数。
- `RANK` 在两个节点是否分别为 `0` 和 `1`。
- `RANK_TABLE_FILE` 是否两个节点都指向 `/shared-volume/hccl_ranktable.json`。
- rank table 中每个 `device_ip` 是否为 HCCN/RDMA IP，而不是 Pod IP、宿主业务 IP、`server_id` 或 `container_ip`。
- engine 容器内是否能看到 `/etc/hccn.conf`。

### 9.2 直接检查 rank table 和 hccn

```bash
kubectl -n "$NS" exec "$POD0" -c "$ENGINE_CONTAINER" -- sh -lc \
  'cat /shared-volume/hccl_ranktable.json; echo; cat /etc/hccn.conf 2>/dev/null || true; ip -o -4 addr show | grep -E "hccn|eth|bond|en|ib" || true'

kubectl -n "$NS" exec "$POD1" -c "$ENGINE_CONTAINER" -- sh -lc \
  'cat /shared-volume/hccl_ranktable.json; echo; cat /etc/hccn.conf 2>/dev/null || true; ip -o -4 addr show | grep -E "hccn|eth|bond|en|ib" || true'
```

如果 rank table 的 `device_ip` 等于 `112.x.x.x` 这类业务网络 IP，而不是 hccn/RDMA IP，910B 上很容易在 ATB warmup 阶段建 HCCL group 失败。

### 9.3 用 HCCL_DEVICE_IPS 强制覆盖 rank table device_ip

新版本启动脚本支持通过 `HCCL_DEVICE_IPS` 在 engine 容器内覆盖 rank table 的 `device_ip`。格式：

```bash
export HCCL_DEVICE_IPS='node0_card0_hccn_ip,node0_card1_hccn_ip;node1_card0_hccn_ip,node1_card1_hccn_ip'
```

单卡双机场景示例：

```bash
export HCCL_DEVICE_IPS='192.168.100.10;192.168.101.10'
```

两机八卡示例：

```bash
export HCCL_DEVICE_IPS='192.168.100.10,192.168.100.11,192.168.100.12,192.168.100.13,192.168.100.14,192.168.100.15,192.168.100.16,192.168.100.17;192.168.101.10,192.168.101.11,192.168.101.12,192.168.101.13,192.168.101.14,192.168.101.15,192.168.101.16,192.168.101.17'
```

本目录提供 `910b_hccl_env_override.env` 模板，并提供 `910b_hccl_env_override_configmap.yaml` 作为 Kubernetes 挂载示例。可通过 env override ConfigMap 注入到 `wings_control/config/env_overrides/*.env` 或 `*.sh`。新脚本会打印：

```text
[mindie] Updated N rank table device_ip value(s) from HCCL_DEVICE_IPS
```

### 9.4 保留显式 HCCL_IF_IP / HCCL_SOCKET_IFNAME

新版本启动脚本不会再强行覆盖用户显式设置的：

- `HCCL_IF_IP`
- `HCCL_SOCKET_IFNAME`
- `GLOO_SOCKET_IFNAME`

如果 910B 环境要求指定 HCCN 网卡或 RDMA 网卡，可通过 env override 显式注入。

### 9.5 与当前日志对应的判断

当前日志中出现：

- `NPU_MEMORY_FRACTION=0.96` 后又出现 `NPU_MEMORY_FRACTION=0.8`：说明现场镜像仍旧，或显式配置了 `gpu_memory_utilization/npu_memory_fraction=0.8`。
- `WORLD_SIZE=2`、`worldSize=1`、`tp=2`：双机单卡普通 TP 语义本身成立。
- 真正 fatal 是 `Create the hccl communication group failed`：优先检查 rank table 的 `device_ip` 是否为 910B HCCN/RDMA IP，以及 `/etc/hccn.conf` 是否挂载进 engine 容器。
