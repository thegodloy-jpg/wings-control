# Qwen3-30B-A3B PD 分离部署 —— Decode 做成 DP 组（§6 方案）

> 本文是 [deploy-qwen3-30b-pd-1p4c-2d2c.md](./deploy-qwen3-30b-pd-1p4c-2d2c.md) §6 的展开。
> 拓扑：**P 节点 1 个（TP=4，standalone）+ D 节点 2 个组成一个 `dp_size=2` 的数据并行组**。
>
> 与主文档的根本区别：两个 D **不是独立副本，而是同一个 decode 引擎的 rank0 / rank1**，
> 必须互相 rendezvous 才能起来。走的是 external-lb（mode A）代码路径
> （[config_loader.py:903](../../wings_control/core/config_loader.py#L903) `_get_pd_external_lb_params`、
> [vllm_adapter.py:2860](../../wings_control/engines/vllm_adapter.py#L2860) `_build_vllm_pd_external_lb_script`）。

---

## ⚠️ 0. 选它之前必须知道的代价

external-lb 的 fork 脚本里写死了这条语义（[vllm_adapter.py:2909](../../wings_control/engines/vllm_adapter.py#L2909)）：

> **任一 service 退出 → 整 pod `exit 1` → 编排层把整个 DP 组一起重启**
> （EP all-to-all 下单 rank 缺失会让整域 hang，所以必须整组重启）。

也就是说:
- **D-1 挂 → D-2 也被拖着重启**,反之亦然;
- rank1 启动要等 rank0 的 `--data-parallel-address` rendezvous,**启动有强时序依赖**;
- 这正是你之前遇到的**级联重启风暴**的放大版。

**结论先行**:除非单个 D 实例(2 卡 TP=2)的吞吐/显存确实扛不住、必须靠 DP 横向扩成一个大 decode,
否则**优先用主文档 §2/§3 的独立副本方案**。下面是确实要用 DP 组时的正确配法。

---

## 1. 角色划分（注意 P 和 D 走不同路径）

| | P 节点 | D 节点 ×2 |
|---|---|---|
| 实例形态 | **standalone 单实例**（DP_SIZE=1，不进 external-lb） | **一个 dp_size=2 的 DP 组**（external-lb mode A） |
| 卡数 / TP | 4 卡，TP=4 | 各 2 卡，TP=2 |
| 代码路径 | `_build_ascend_pd_kv_config`（标准） | `_get_pd_external_lb_params` + fork 脚本 |
| 拓扑变量来源 | `PD_PREFILL_*` / `PD_DECODE_*` | `DP_SIZE/TP_SIZE/...`（本角色权威）+ `PD_PREFILL_*`（对端） |

> 关键非对称点:P 只有 1 个 pod、dp=1 → `DP_SIZE` 触发不了 external-lb（[config_loader.py:941](../../wings_control/core/config_loader.py#L941) `dp_size<=1` 直接返回 standalone），所以 **P 仍用标准路径**，只是要把 decode 的拓扑告诉它做 KV 映射。

---

## 2. P 节点（4 卡，TP=4，standalone kv_producer）

### 2.1 环境变量

> ⚠️ **裸跑必须 `export`**，否则变量传不进 `wings_start.sh`，tp/dp 会回退成本地 device_count（见 §10）。
> K8s 部署放进容器 `env`（YAML 不写 `export`）。

```bash
export WINGS_DEVICE="ascend"
export WINGS_DEVICE_COUNT="4"
export PD_ROLE="P"
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"     # 按本机实际可见卡

# 全局 PD 拓扑：prefill 是自己，decode 要写 DP 组的真实拓扑（dp=2!）
export PD_PREFILL_TP_SIZE="4"
export PD_PREFILL_DP_SIZE="1"
export PD_DECODE_TP_SIZE="2"
export PD_DECODE_DP_SIZE="2"                    # ← 和独立方案的唯一区别：decode dp=2

export PD_CONNECTOR_TYPE="MooncakeConnectorV1"
export RANK_IP="<P节点RDMA_IP>"
export NETWORK_INTERFACE="<RDMA网卡名>"
export ASCEND_ENFORCE_EAGER="true"

export ENGINE_PORT="17000"
export HEALTH_PORT="19000"
export MONITOR_PROXY_PORT="19100"
export VLLM_LLMDD_RPC_PORT="5569"
export VLLM_MOONCAKE_BOOTSTRAP_PORT="23000"
```

### 2.2 启动命令（与主文档 §2 完全一致）

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend \
  --model-name Qwen3-30B-A3B \
  --model-path /usr/local/serving/models/ \
  --device-count 4 \
  --port 18000 \
  --input-length 4096 \
  --output-length 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code --seed 42
```

KV 段（核对）：`kv_role=kv_producer`，`decode:{tp_size:2,dp_size:2}`。

---

## 3. D 节点 ×2（一个 dp_size=2 的 DP 组）

D 走 external-lb 契约变量（**角色域命名，不是 `PD_DECODE_*`**），见
[config_loader.py:908-914](../../wings_control/core/config_loader.py#L908-L914)。
两个 D pod **大部分变量相同**，靠 `HOST_IP` 在 `NODE_IPS` 中的位置自动算出各自的 dp_rank。

### 3.1 共用契约变量（D-1 / D-2 都要设，且值一致）

> ⚠️ **裸跑必须 `export`**（见 §10）。

```bash
export WINGS_DEVICE="ascend"
export PD_ROLE="D"

# —— external-lb DP 组契约（决定 D 是一个分布式整体）——
export DP_SIZE="2"                 # decode 全局 DP = 2（两个 D 合成一个实例）
export TP_SIZE="2"                 # 单实例 TP = 2
export DP_SIZE_LOCAL="1"           # 每个 pod fork 1 个 service（一机一 rank）
export Master_IP="<D-1的RDMA_IP>"  # = --data-parallel-address，指向 rank0（D-1）
export VLLM_LLMDD_RPC_PORT="5570"  # DP RPC 端口，组内一致
export NODE_IPS="<D-1_IP>,<D-2_IP>"  # 顺序即 rank 顺序：D-1=rank0, D-2=rank1（两 pod 必须写一样）

# —— 对端（prefill）拓扑，给 KV 映射用 ——
export PD_PREFILL_DP_SIZE="1"
export PD_PREFILL_TP_SIZE="4"

export PD_CONNECTOR_TYPE="MooncakeConnectorV1"
export NETWORK_INTERFACE="<RDMA网卡名>"
export ASCEND_ENFORCE_EAGER="true"

export ENGINE_PORT="17000"
export HEALTH_PORT="19000"
export MONITOR_PROXY_PORT="19100"
export VLLM_MOONCAKE_BOOTSTRAP_PORT="23100"
```

### 3.2 每个 pod 各自不同的两个变量

| 变量 | D-1（rank0） | D-2（rank1） |
|------|-------------|-------------|
| `HOST_IP`（=`RANK_IP`） | `<D-1_IP>` | `<D-2_IP>` |
| `ASCEND_RT_VISIBLE_DEVICES` | 按本机实际（你的：`0,4`） | 按本机实际（你的：`3,7`） |

> rank 不用手填：`dp_rank_start = NODE_IPS.index(HOST_IP) × DP_SIZE_LOCAL`
> （[config_loader.py:954-957](../../wings_control/core/config_loader.py#L954-L957)）。
> D-1 的 HOST_IP 排在 NODE_IPS 第 0 位 → rank0；D-2 → rank1。
> **所以 `NODE_IPS` 在两个 pod 上必须字符串完全一致、顺序一致。**

### 3.3 启动命令（D-1 / D-2 相同）

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend \
  --model-name Qwen3-30B-A3B \
  --model-path /usr/local/serving/models/ \
  --device-count 2 \
  --port 18000 \
  --input-length 4096 \
  --output-length 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code --seed 42
```

> 不用手写 `--data-parallel-*`。fork 脚本会按契约自动追加：
> `--tensor-parallel-size 2 --data-parallel-size 2 --data-parallel-rank <rank>
> --data-parallel-size-local 1 --data-parallel-address <Master_IP>
> --data-parallel-rpc-port 5570 --data-parallel-external-lb`
> （[vllm_adapter.py:2917-2922](../../wings_control/engines/vllm_adapter.py#L2917-L2922)）。

---

## 4. 启动时序（DP 组的强依赖，必看）

```
1. D-1(rank0) 起 vllm serve，在 Master_IP:RPC 上开 DP rendezvous，等齐 dp_size=2
2. D-2(rank1) 起 vllm serve，连 Master_IP:RPC 上报
3. 两个 rank 都到齐 → decode 引擎才算 ready → /health 才转 200
4. P 端 prefill 把 KV 推给该 decode 组
```

含义:
- **D-1 必须先于(或同时于) D-2 起来**;D-1 没起,D-2 连不上 Master_IP 一直等。
- 任一 rank 中途死 → 整 pod `exit 1` → **两个 D pod 一起重启**。
- 所以 **startupProbe 的宽限期要按"两个 rank 都加载完 + rendezvous"来给**,比独立方案更长。

---

## 5. K8s 探针（DP 组场景更要放宽）

给 P 和两个 D 的 engine / wings-control 容器都加 startupProbe；D 组因为多一层 rendezvous，
`failureThreshold` 给得比独立方案更宽：

```yaml
startupProbe:
  httpGet: { path: /health, port: 19000 }
  periodSeconds: 10
  failureThreshold: 120       # ~20 分钟，覆盖 30B 加载 + DP rendezvous
  timeoutSeconds: 5
livenessProbe:
  httpGet: { path: /health, port: 19000 }
  initialDelaySeconds: 0
  periodSeconds: 20
  failureThreshold: 6
```

> 改 Deployment（非 Pod），平台 `backend-serving` 生成的需回填模板才持久。

---

## 6. 验证

```bash
# 两个 D 都 ready 后，DP 组才健康
curl -s http://<D1_POD_IP>:19000/health     # 200
curl -s http://<D2_POD_IP>:19000/health     # 200

# 经 P proxy 推理
curl -s http://<P_POD_IP>:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-30B-A3B","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'

# D engine 日志成功标志
#   "data parallel" rendezvous 完成、两个 rank 都 "Application startup complete."
#   kv_consumer 与 P 的 kv_producer 握手成功
```

---

## 7. 独立副本 vs DP 组 速查

| 维度 | 独立副本（主文档 §2/§3，推荐） | DP 组（本文） |
|------|------------------------------|--------------|
| D 之间关系 | 互不通信,各自完整实例 | 同一引擎 rank0/rank1,必须 rendezvous |
| `PD_DECODE_DP_SIZE` | `1` | `2` |
| D 契约变量 | `PD_DECODE_*` | `DP_SIZE/TP_SIZE/DP_SIZE_LOCAL/Master_IP/NODE_IPS/HOST_IP` |
| 一个 D 挂掉 | 另一个照常服务 | **整组一起重启** |
| 启动依赖 | 无互锁 | 强时序(rank1 等 rank0) |
| 适用场景 | 默认 | 单实例吞吐/显存不够,必须横向扩 |
| 对重启风暴 | 缓解 | **放大** |

## 8. 占位替换清单

| 占位 | 含义 |
|------|------|
| `<RDMA网卡名>` | RDMA 网卡（非 eth0） |
| `<P节点RDMA_IP>` / `<D-1_IP>` / `<D-2_IP>` | 各节点 RDMA IP；`Master_IP` 必须 = `<D-1_IP>` |
| `NODE_IPS` | `"<D-1_IP>,<D-2_IP>"`，两 D pod 写法完全一致 |
| `ASCEND_RT_VISIBLE_DEVICES` | 各机实际可见卡（P=1,2,3,5 / D1=0,4 / D2=3,7） |
| `--model-path` | 按实际 volume |
| `--input-length` / `--output-length` | 上下文长度（`max_model_len = 两者之和`），无 `--max-model-len` flag |

---

## 9. 附：可直接粘贴的 export + 启动块（裸跑自测用）

> ⚠️ 同样：裸跑前必须先 export 这组变量。注意 **P 和 D 的契约变量不同** —— P 走标准
> `PD_DECODE_*`，D 走 external-lb 的 `DP_SIZE/TP_SIZE/Master_IP/NODE_IPS/HOST_IP`。
> 不要传 `--tensor-parallel-size` / `--max-model-len`（同主文档）。

### 9.1 P 节点（4 卡，standalone kv_producer，decode 拓扑 dp=2）

```bash
export WINGS_DEVICE=ascend WINGS_DEVICE_COUNT=4
export PD_ROLE=P
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3            # 按本机实际（你这台 1,2,3,5）
export PD_PREFILL_TP_SIZE=4 PD_PREFILL_DP_SIZE=1
export PD_DECODE_TP_SIZE=2  PD_DECODE_DP_SIZE=2     # ← DP 组：decode dp=2
export PD_CONNECTOR_TYPE=MooncakeConnectorV1
export RANK_IP=<P节点RDMA_IP> NETWORK_INTERFACE=<RDMA网卡名>
export ASCEND_ENFORCE_EAGER=true
export ENGINE_PORT=17000 HEALTH_PORT=19000 MONITOR_PROXY_PORT=19100
export VLLM_LLMDD_RPC_PORT=5569 VLLM_MOONCAKE_BOOTSTRAP_PORT=23000
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name Qwen3-30B-A3B \
  --model-path /usr/local/serving/models/ \
  --device-count 4 --port 18000 \
  --input-length 4096 --output-length 4096 \
  --gpu-memory-utilization 0.9 --trust-remote-code --seed 42
```

### 9.2 D-1 节点（rank0，DP 组 head）

```bash
export WINGS_DEVICE=ascend WINGS_DEVICE_COUNT=2
export PD_ROLE=D
# —— external-lb DP 组契约 ——
export DP_SIZE=2 TP_SIZE=2 DP_SIZE_LOCAL=1
export Master_IP=<D-1_IP>                           # = data-parallel-address，指向自己(rank0)
export NODE_IPS=<D-1_IP>,<D-2_IP>                   # 顺序即 rank：D-1=0, D-2=1
export HOST_IP=<D-1_IP>                             # 本 pod IP → 落在 NODE_IPS 第0位 → rank0
export VLLM_LLMDD_RPC_PORT=5570
# —— 对端 prefill 拓扑（KV 映射用）——
export PD_PREFILL_TP_SIZE=4 PD_PREFILL_DP_SIZE=1
export PD_CONNECTOR_TYPE=MooncakeConnectorV1
export NETWORK_INTERFACE=<RDMA网卡名> ASCEND_ENFORCE_EAGER=true
export ENGINE_PORT=17000 HEALTH_PORT=19000 MONITOR_PROXY_PORT=19100
export VLLM_MOONCAKE_BOOTSTRAP_PORT=23100
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name Qwen3-30B-A3B \
  --model-path /usr/local/serving/models/ \
  --device-count 2 --port 18000 \
  --input-length 4096 --output-length 4096 \
  --gpu-memory-utilization 0.9 --trust-remote-code --seed 42
```

### 9.3 D-2 节点（rank1）

与 D-1 **只差 `HOST_IP`**（`Master_IP` / `NODE_IPS` 必须和 D-1 一模一样）：

```bash
export WINGS_DEVICE=ascend WINGS_DEVICE_COUNT=2
export PD_ROLE=D
export DP_SIZE=2 TP_SIZE=2 DP_SIZE_LOCAL=1
export Master_IP=<D-1_IP>                           # 仍指向 D-1（rank0）
export NODE_IPS=<D-1_IP>,<D-2_IP>                   # 与 D-1 完全一致
export HOST_IP=<D-2_IP>                             # 本 pod IP → 落在第1位 → rank1
export VLLM_LLMDD_RPC_PORT=5570
export PD_PREFILL_TP_SIZE=4 PD_PREFILL_DP_SIZE=1
export PD_CONNECTOR_TYPE=MooncakeConnectorV1
export NETWORK_INTERFACE=<RDMA网卡名> ASCEND_ENFORCE_EAGER=true
export ENGINE_PORT=17000 HEALTH_PORT=19000 MONITOR_PROXY_PORT=19100
export VLLM_MOONCAKE_BOOTSTRAP_PORT=23100
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name Qwen3-30B-A3B \
  --model-path /usr/local/serving/models/ \
  --device-count 2 --port 18000 \
  --input-length 4096 --output-length 4096 \
  --gpu-memory-utilization 0.9 --trust-remote-code --seed 42
```

### 9.4 跑完核对

```bash
# D 端应进 external-lb fork 脚本（含 --data-parallel-* 与 fork 循环）
grep -E 'data-parallel-(rank|address|size)|for i in \$\(seq' /shared-volume/start_command.sh
# D-1 期望 rank=0，D-2 期望 rank=1（--data-parallel-rank）；address 都指向 <D-1_IP>
grep -o '"kv_role":"[^"]*"' /shared-volume/start_command.sh   # P=kv_producer, D=kv_consumer
```

> `ASCEND_RT_VISIBLE_DEVICES` 在 D 的 external-lb 路径里由 fork 脚本按 `[i*tp,(i+1)*tp)` 自动设置
> （[vllm_adapter.py:2916-2917](../../wings_control/engines/vllm_adapter.py#L2916-L2917)），
> 所以 D 块不用手 export 它，只要 pod 实际挂了 2 张卡即可。

---

## 10. 排查：kv-config 里 prefill/decode 的 tp 填错

和独立方案同一类坑（详见 [主文档 §10](./deploy-qwen3-30b-pd-1p4c-2d2c.md#10-排查kv-config-里-prefilldecode-的-tp-填错最常见)）：
不注入对端拓扑变量时，节点会把对端 tp 按自己的卡数瞎填，两边 KV 映射算不一致。

本方案（DP 组）的对端拓扑变量：
- **P 节点**（标准路径）：靠 `PD_DECODE_TP_SIZE=2` + **`PD_DECODE_DP_SIZE=2`** 告诉它 decode 是 dp=2 的组；
- **D 节点**（external-lb）：本角色 tp/dp 来自 `TP_SIZE`/`DP_SIZE`（权威）；对端 prefill 靠
  `PD_PREFILL_TP_SIZE=4` + `PD_PREFILL_DP_SIZE=1`，缺失会打 `peer(prefill) topology unknown` 告警并回退本角色
  （[config_loader.py:1000-1004](../../wings_control/core/config_loader.py#L1000-L1004)），KV 映射就会错。

核对：P 端 kv-config 应是 `prefill{tp:4,dp:1}` + `decode{tp:2,dp:2}`；
D 端日志若出现 `[PD external-lb] peer(prefill) topology unknown` 即说明 `PD_PREFILL_*` 没设。
