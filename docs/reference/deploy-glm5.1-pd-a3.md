# GLM-5.1 PD 分离部署（A3 / dp>1 / external-lb / mooncake）

> 目标拓扑：**P 节点 2 个（各 16 卡）+ D 节点 4 个（各 16 卡）**，**强制 dp>1**，走
> vLLM-Ascend external-lb（DP fork）+ MooncakeConnectorV1。
>
> 并行：**P = DP2×TP16**（每 P pod fork 1 个 prefill、TP16）；**D = DP16×TP4**（每 D pod fork 4 个、各 TP4）。
> 全局拓扑 `prefill={dp:2,tp:16}` / `decode={dp:16,tp:4}`。合计 **6 pod / 96 卡**。
>
> proxy 放 wings 外面自己调度（见 §5）。同款 V4-Flash 见 [deploy-v4flash-pd-a3.md](./deploy-v4flash-pd-a3.md)；字段逐项对齐官方见 [pd-a3-official-alignment-report.md](./pd-a3-official-alignment-report.md)；官方 [GLM5](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.html)。

---

## 0. 关键事实（先读）

1. **必须走 external-lb（dp>1）**：触发 = `PD_ROLE∈{P,D}` **且 `DP_SIZE>1`**（[config_loader.py:935/943](../../wings_control/core/config_loader.py#L935)）。dp=1 退回 standalone，不读 pd_config.json。
2. **`GlmMoeDsaForCausalLM` 已注册** [pd_config.json](../../wings_control/config/defaults/pd_config.json)：命中后自动补 max-model-len(P=131072/D=200000)、enforce-eager(P)、FULL_DECODE_ONLY+capture(D)、tool/reasoning parser、`use_ascend_direct`、`kv_port`、`engine_id`、共用/角色 env 等（逐项对齐官方 A3）。
3. **平台 a3**：给**任一** a3 信号即可——`WINGS_ASCEND_PLATFORM=a3` / a3 镜像的 `ENGINE_VERSION` 带 `-a3` 后缀 / `ASCEND_A3_ENABLE=1` / `hardware_info.json` 含 910c。**全无信号会回退 a2**（HCCL_BUFFSIZE/算子块都会错）。
4. **本机 IP 只设 `RANK_IP`**（本 pod 唯一 IP）：`get_local_ip()` 读它（[env_utils.py:65](../../wings_control/utils/env_utils.py#L65)），`HCCL_IF_IP` 与 rank_start 派生都回退到它，**不必再设 `POD_IP`/`HOST_IP`**。`RANK_IP` 须**逐字**在 `NODE_IPS` 内。
5. **`DP_SIZE`/`TP_SIZE` 可省**：本角色 dp/tp 由全局拓扑 `PD_{ROLE}_*` 派生（P→`PD_PREFILL_*`，D→`PD_DECODE_*`）。本文显式给以便核对；省略亦可。
6. **不要传** `--tensor-parallel-size` / `--max-model-len`（`parse_launch_args` 不认，会报错）。TP 由 `TP_SIZE` 决定，max-model-len 由注册表给。
7. **mooncake 不会引擎一起来就自动传 KV**：靠 §5 外部 PD proxy 在请求层把 P→D 串起来才触发。

---

## 1. 拓扑与全局并行

| | P 节点（2 pod，各 16 卡） | D 节点（4 pod，各 16 卡） |
|---|---|---|
| 角色 | `PD_ROLE=P` → kv_producer | `PD_ROLE=D` → kv_consumer |
| DP / TP | `DP_SIZE=2 TP_SIZE=16` | `DP_SIZE=16 TP_SIZE=4` |
| 本 pod fork 数 | `DP_SIZE_LOCAL=1` | `DP_SIZE_LOCAL=4` |
| rank | P-0=0 / P-1=1 | D-0=0-3 / D-1=4-7 / D-2=8-11 / D-3=12-15 |
| kv-config `prefill` / `decode` | `{dp:2,tp:16}` / `{dp:16,tp:4}` | 同（两端必须完全一致） |

> **拓扑一致性铁律**：P 与 D 最终 kv-config 的 `prefill{}`、`decode{}` 必须分别完全相同，否则 mooncake KV rank 映射两端不一致 → 握手失败（自查见 §8）。

---

## 2. 端口规划（proxy 按此表填）

| 节点/rank | 引擎 HTTP（`ENGINE_PORT+i`） | kv_port（`base+i`） | bootstrap | 可见卡（`i*TP..`） |
|-----------|---------------|---------|-----------|--------|
| P-0（rank0） | `17000` | 30000 | 23000 | 0-15 |
| P-1（rank1） | `17000` | 30000 | 23000 | 0-15 |
| D-0（rank0-3） | `17000-17003` | 30100-30103 | 23100-23103 | 0-3 / 4-7 / 8-11 / 12-15 |
| D-1（rank4-7）/ D-2（rank8-11）/ D-3（rank12-15） | `17000-17003` | 30100-30103 | 23100-23103 | 同上 |

> 端口算法（fork 脚本）：`PORT=ENGINE_PORT+i`、`kv_port=base+i`（P base 30000 / D base 30100）、`bootstrap=BOOTSTRAP+i`（P 23000 / D 23100）、`卡=[i*TP,(i+1)*TP)`。不同 pod 同端口不冲突。

---

## 3. P 节点 ×2（各 16 卡，DP2×TP16，kv_producer）

P-0 / P-1 **只差 `RANK_IP`**（`Master_IP`/`NODE_IPS` 两端写法完全一致）。

### 3.1 环境变量（注入 wings-control 容器 env）

```bash
export PD_ROLE=P
export WINGS_DEVICE=ascend  WINGS_DEVICE_COUNT=16
export WINGS_ASCEND_PLATFORM=a3            # 或 a3 镜像 ENGINE_VERSION 带 -a3 即可省
# —— external-lb fork 编排 ——
export DP_SIZE=2 TP_SIZE=16 DP_SIZE_LOCAL=1     # DP_SIZE/TP_SIZE 可省（由 PD_PREFILL_* 派生）
export Master_IP=<P0_IP>                        # = data-parallel-address，指向 P-0（rank0）
export NODE_IPS=<P0_IP>,<P1_IP>                 # 顺序即 rank：P-0=0、P-1=1
export RANK_IP=<本P_IP>                         # P-0 填 <P0_IP> / P-1 填 <P1_IP>，须 ∈ NODE_IPS
export VLLM_LLMDD_RPC_PORT=10521  ENGINE_PORT=17000
# —— 全局拓扑（KV 映射；两端一致）——
export PD_PREFILL_DP_SIZE=2  PD_PREFILL_TP_SIZE=16
export PD_DECODE_DP_SIZE=16  PD_DECODE_TP_SIZE=4
```

### 3.2 wings-control 启动命令

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name glm-5.1-chat \
  --model-path /usr/local/serving/models/ \
  --device-count 16 --port 18000 \
  --gpu-memory-utilization 0.95 --trust-remote-code --seed 1024
```

### 3.3 wings 生成的引擎命令（dry-run 实测；仅核对结构，**不手动执行**）

```bash
for i in $(seq 0 0); do            # DP_SIZE_LOCAL=1 → fork 1
  RANK=$((<rank_start> + i)); PORT=$((17000 + i)); KVPORT=$((30000 + i)); BOOTSTRAP=$((23000 + i))
  LO=$((i * 16)); HI=$((LO + 15)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /usr/local/serving/models/ --served-model-name glm-5.1-chat \
    --max-model-len 131072 --quantization ascend --seed 1024 \
    --max-num-batched-tokens 4096 --max-num-seqs 64 --gpu-memory-utilization 0.95 \
    --enable-expert-parallel --enable-chunked-prefill --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"ascend_compilation_config":{"enable_npugraph_ex":true},"enable_dsa_cp":true,"layer_sharding":["q_b_proj","o_proj"]}' \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}' \
    --port $PORT --tensor-parallel-size 16 --data-parallel-size 2 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <P0_IP> --data-parallel-rpc-port 10521 --data-parallel-external-lb \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":16},"decode":{"dp_size":16,"tp_size":4},"use_ascend_direct":true}}' &
done; wait -n
```
> `<rank_start>` = `RANK_IP` 在 `NODE_IPS` 的位置 ×`DP_SIZE_LOCAL`：P-0→0、P-1→1（自动派生）。角色 env：P 加 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`+`FUSED_MC2`。

---

## 4. D 节点 ×4（各 16 卡，组成 DP16，kv_consumer）

D-0..D-3 **只差 `RANK_IP`**。

### 4.1 环境变量

```bash
export PD_ROLE=D
export WINGS_DEVICE=ascend  WINGS_DEVICE_COUNT=16
export WINGS_ASCEND_PLATFORM=a3
export DP_SIZE=16 TP_SIZE=4 DP_SIZE_LOCAL=4      # DP_SIZE/TP_SIZE 可省（由 PD_DECODE_* 派生）
export Master_IP=<D0_IP>                         # 指向 D-0（rank0）
export NODE_IPS=<D0_IP>,<D1_IP>,<D2_IP>,<D3_IP>  # 顺序即 rank 段：D-0=0、D-1=4、D-2=8、D-3=12
export RANK_IP=<本D_IP>                          # 须 ∈ NODE_IPS
export VLLM_LLMDD_RPC_PORT=10523  ENGINE_PORT=17000
export PD_PREFILL_DP_SIZE=2  PD_PREFILL_TP_SIZE=16
export PD_DECODE_DP_SIZE=16  PD_DECODE_TP_SIZE=4
```

### 4.2 wings-control 启动命令（D-0..D-3 相同）

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name glm-5.1-chat \
  --model-path /usr/local/serving/models/ \
  --device-count 16 --port 18000 \
  --gpu-memory-utilization 0.92 --trust-remote-code --seed 1024
```

### 4.3 wings 生成的引擎命令（每 D pod fork 4；仅核对）

```bash
for i in $(seq 0 3); do            # DP_SIZE_LOCAL=4 → fork 4
  RANK=$((<rank_start> + i)); PORT=$((17000 + i)); KVPORT=$((30100 + i)); BOOTSTRAP=$((23100 + i))
  LO=$((i * 4)); HI=$((LO + 3)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /usr/local/serving/models/ --served-model-name glm-5.1-chat \
    --max-model-len 200000 --quantization ascend --seed 1024 \
    --max-num-batched-tokens 32 --max-num-seqs 8 --gpu-memory-utilization 0.92 \
    --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}' \
    --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}' \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}' \
    --port $PORT --tensor-parallel-size 4 --data-parallel-size 16 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <D0_IP> --data-parallel-rpc-port 10523 --data-parallel-external-lb \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":16},"decode":{"dp_size":16,"tp_size":4},"use_ascend_direct":true}}' &
done; wait -n
```
> D 无 `--enforce-eager`/`--enable-prefix-caching`/`--enable-chunked-prefill`（decode 走 cudagraph）。角色 env：D 加 `VLLM_ASCEND_ENABLE_MLAPO=1`+`TASK_QUEUE_ENABLE=1`+`FUSED_MC2`。`<rank_start>`：D-0→0、D-1→4、D-2→8、D-3→12。

---

## 5. ⭐ 在 wings 外起 PD proxy（触发 mooncake，必做）

`--prefiller-*` 指 §2 的 2 个 P 服务、`--decoder-*` 指 4 个 D pod 的 16 个 service（每 pod 4 个端口 17000-17003）：

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 18000 --host 0.0.0.0 \
  --prefiller-hosts <P0_IP> <P1_IP> \
  --prefiller-ports 17000 17000 \
  --decoder-hosts  <D0_IP> <D0_IP> <D0_IP> <D0_IP>  <D1_IP> <D1_IP> <D1_IP> <D1_IP> ... \
  --decoder-ports  17000 17001 17002 17003  17000 17001 17002 17003 ...   # 每 D pod 4 端口
```
> 压测/curl 打 **proxy :18000**，不是引擎 17000。

---

## 6. K8s 探针（冷启动别被打死）

GLM-5.1 16 卡冷加载 + DP rendezvous 久。engine、wings-control 两容器都加 startupProbe：

```yaml
startupProbe: { httpGet: { path: /health, port: 19000 }, periodSeconds: 10, failureThreshold: 120, timeoutSeconds: 5 }
livenessProbe: { httpGet: { path: /health, port: 19000 }, periodSeconds: 20, failureThreshold: 6 }
```

---

## 7. 验证（mooncake 触发判据）

```bash
# 1) external-lb 触发（无此行 = dp>1 没生效）
kubectl logs <P-0-pod> -c wings-control | grep "PD external-lb"
#   期望: [PD external-lb] arch=GlmMoeDsaForCausalLM role=P connector=MooncakeConnectorV1 dp_size=2 local=1 rank_start=0
# 2) 各引擎 health：P 每 pod 1 端口(17000)，D 每 pod 4 端口(17000-17003)
curl -s http://<P0_IP>:17000/health ; curl -s http://<D0_IP>:1700{0..3}/health
# 3) 打 proxy 出预测
curl -s http://<proxy_IP>:18000/v1/chat/completions -H 'Content-type: application/json' \
  -d '{"model":"glm-5.1-chat","messages":[{"role":"user","content":"你是谁"}],"max_tokens":64,"temperature":0}'
# 4) ★ 两端都要有 mooncake 传输日志
kubectl logs <P-0-pod> -c engine | grep -iE "kv_producer|mooncake|transfer"
kubectl logs <D-0-pod> -c engine | grep -iE "kv_consumer|mooncake|pull"
```

---

## 8. 拓扑一致性自查（四值对齐）

| kv-config 块 | P 端来源 | D 端来源 | 本拓扑值 |
|------|------|------|------|
| `prefill` | P 的 `DP_SIZE/TP_SIZE`（或 `PD_PREFILL_*`） | D 的 `PD_PREFILL_*` | `{dp:2,tp:16}` |
| `decode` | P 的 `PD_DECODE_*` | D 的 `DP_SIZE/TP_SIZE`（或 `PD_DECODE_*`） | `{dp:16,tp:4}` |

> 即所有 pod 都设 `PD_PREFILL_DP_SIZE=2 PD_PREFILL_TP_SIZE=16 PD_DECODE_DP_SIZE=16 PD_DECODE_TP_SIZE=4`，两端 kv-config 必然相同。

---

## 9. 排障

| 现象 | 原因 | 处理 |
|------|------|------|
| 无 `[PD external-lb]` | `DP_SIZE` 没读成 >1 | 进容器 `printenv DP_SIZE`；确认镜像读 `DP_SIZE` 还是 `PD_DP_SIZE` |
| 命令是单进程 `--tensor-parallel-size 16` 无 fork | 走了 standalone | 同上 |
| HCCL_BUFFSIZE=512 / 无 ASCEND_A3_ENABLE | 平台被判成 a2 | 给 a3 信号（§0.3），`printenv WINGS_ASCEND_PLATFORM ENGINE_VERSION` |
| 多节点 rank 撞车 / DP 组不起来 | `RANK_IP` 不在 `NODE_IPS` 内（逐字） | 对齐 `RANK_IP`/`NODE_IPS` 文本 |
| 预测通但 D engine 无 mooncake 日志 | 没起 §5 proxy / 端口错 | 起 proxy，端口对齐 §2 |
| KV 映射错/握手失败 | `prefill`/`decode` 两端不一致 | 对齐 §8 四值 |

---

## 10. 占位替换清单

| 占位 | 含义 |
|------|------|
| `<P0_IP>` / `<P1_IP>` | 2 个 P pod IP（`NODE_IPS` 顺序即 rank） |
| `<D0_IP>`..`<D3_IP>` | 4 个 D pod IP；`Master_IP=<D0_IP>`，4 端写法一致 |
| `<proxy_IP>` | 跑 `load_balance_proxy_server_example.py` 的机器 |
| `<rank_start>` | 自动派生（`RANK_IP` 在 `NODE_IPS` 位置 ×`DP_SIZE_LOCAL`），不手填 |
