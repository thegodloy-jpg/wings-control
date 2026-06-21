# Shell 取默认路由网卡（`_SH_IF_DETECT`）使用逻辑

## 1. 它是什么

`_SH_IF_DETECT` 是一段内嵌到启动脚本里的 **shell 命令替换**，作用是在容器/节点真正启动时，自动探测出本机**默认路由对应的网卡名**，用于设置各类 `*_SOCKET_IFNAME` 通信环境变量。

定义位置：[wings_control/utils/vllm_helpers.py:26](../../wings_control/utils/vllm_helpers.py#L26)

```python
_SH_IF_DETECT = "$(awk '$2==\"00000000\"{print $1;exit}' /proc/net/route 2>/dev/null || echo eth0)"
```

## 2. 探测逻辑

| 步骤 | 说明 |
| --- | --- |
| 读取路由表 | 解析 `/proc/net/route`，每行第 2 列是目的网段（十六进制） |
| 匹配默认路由 | `$2 == "00000000"` 即目的地址为 `0.0.0.0` 的那条，也就是默认路由 |
| 取网卡名 | 打印该行第 1 列（网卡名，如 `eth0`/`eno1`），`exit` 只取第一条 |
| 兜底 | `awk` 失败（无文件/无默认路由）时，`|| echo eth0` 回退为 `eth0` |

> 注意：它取的是**默认路由出口网卡**，不是「按某个 IP 反查网卡」。在单网卡或默认路由即业务网的环境下，二者结果一致。

## 3. 为什么用 shell 而不是 Python

- 探测发生在**生成的启动脚本运行时**，而非 wings-control 进程内。控制面 sidecar 容器可能无 GPU/NPU、网络拓扑也与目标节点不同，**只有目标节点自己执行时探到的网卡才准确**。
- 纯文本命令替换可直接拼进 `export` 语句，无需在目标镜像里额外装 `netifaces` 等依赖。

相关对照：Python 侧 [vllm_adapter.py:2837](../../wings_control/engines/vllm_adapter.py#L2837) 不做动态探测，改用环境变量 `NETWORK_INTERFACE` → `GLOO_SOCKET_IFNAME` → `eth0` 兜底。

## 4. 在哪里被使用

集中在 Ray / 分布式脚本生成处：[wings_control/engines/vllm_distributed.py](../../wings_control/engines/vllm_distributed.py)

| 行号 | 用途 |
| --- | --- |
| L38-39 | Ray 模式 Ascend：`HCCL_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` |
| L120 | `GLOO_SOCKET_IFNAME` |
| L158-159 | `HCCL_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` |
| L180 | `GLOO_SOCKET_IFNAME` |

典型生成结果（写入 `start_command.sh`）：

```bash
export HCCL_SOCKET_IFNAME=$(awk '$2=="00000000"{print $1;exit}' /proc/net/route 2>/dev/null || echo eth0)
export TP_SOCKET_IFNAME=$(awk '$2=="00000000"{print $1;exit}' /proc/net/route 2>/dev/null || echo eth0)
```

脚本执行时命令替换被展开，例如默认路由走 `eno1`，则等价于 `export HCCL_SOCKET_IFNAME=eno1`。

## 5. 配套：取本机 IP

同文件还有一段对偶逻辑 [vllm_helpers.py:20-25](../../wings_control/utils/vllm_helpers.py#L20-L25)，用于探测本机 IP（而非网卡名）：

```python
_SH_DETECT_IP = (
    "$(python3 -c \"import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
    "s.connect(('8.8.8.8',80));print(s.getsockname()[0]);s.close()\""
    " 2>/dev/null || hostname -i)"
)
_SH_VLLM_HOST = "export VLLM_HOST_IP=${POD_IP:-${RANK_IP:-" + _SH_DETECT_IP + "}}"
```

逻辑：优先用 `POD_IP`，其次 `RANK_IP`，都没有时用 UDP「连」`8.8.8.8` 反查出口 IP，失败再回退 `hostname -i`。它与 `_SH_IF_DETECT` 一个管 IP、一个管网卡，共同支撑分布式通信地址的自动配置。

## 6. 运维排查

- **网卡探错**：在目标节点执行 `awk '$2=="00000000"{print $1;exit}' /proc/net/route` 看是否与预期一致；多默认路由时只取第一条。
- **想强制指定网卡**：通过环境变量覆盖（如设置 `NETWORK_INTERFACE` / `GLOO_SOCKET_IFNAME`），优先于自动探测。
- **回退到 `eth0`**：说明 `/proc/net/route` 不可读或无默认路由，需检查容器网络。
