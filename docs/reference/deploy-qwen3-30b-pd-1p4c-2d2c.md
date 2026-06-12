# Qwen3-30B-A3B PD 分离部署（dp>1 / external-lb / mooncake 验证版）

> 目标拓扑：**P 节点 1 个（4 卡）+ D 节点 2 个（各 2 卡）**，**强制 dp>1**，走
> vLLM-Ascend external-lb（DP fork）+ MooncakeConnectorV1。本文保证：①wings 下发命令准确 ②mooncake 真正触发可验证。
>
> 并行：**P = DP2×TP2**（1 pod fork 2 个 prefill，各 TP2）；**D = DP2**（2 pod 各 1 rank、TP2）。
> 全局拓扑 `prefill={dp:2,tp:2}` / `decode={dp:2,tp:2}`。
>
> proxy 放 wings 外面自己调度（见 §5）。端到端原理另见 [qwen3-wings-mooncake-pd-verify.md](./qwen3-wings-mooncake-pd-verify.md)。

---

## 0. 关键事实（先读）

1. **必须走 external-lb（dp>1）**：触发条件 = `PD_ROLE∈{P,D}` **且 `DP_SIZE>1`**
   （[config_loader.py:934/941](../../wings_control/core/config_loader.py#L941)）。dp=1 会退回 standalone，
   **不读 pd_config.json、不补 kv_port/engine_id**，本文不走那条。
2. **`Qwen3MoeForCausalLM` 已注册** [pd_config.json](../../wings_control/config/defaults/pd_config.json)：
   external-lb 命中后自动补 `--enable-expert-parallel` / `--additional-config` / `kv_port` / `engine_id` /
   `HCCL_INTRA_ROCE_ENABLE` 等（对齐官方 `run_dp_template.sh`）。
3. **mooncake 不是引擎一起来就自动传 KV**：靠 §5 的外部 PD proxy 在请求层把 P→D 串起来才触发。
   没有 proxy → mooncake 永不触发（请求被 P 单独吃下）。**这是"dp>1 也不通"的真正原因。**
4. **不要传** `--tensor-parallel-size` / `--max-model-len`（wings_start.sh 不认）。TP 由 external-lb 的
   `TP_SIZE` 决定；上下文用 `--input-length` + `--output-length`。

---

## 1. 拓扑与全局并行

| | P 节点（1 pod，4 卡） | D 节点（2 pod，各 2 卡） |
|---|---|---|
| 角色 | `PD_ROLE=P` → kv_producer | `PD_ROLE=D` → kv_consumer |
| DP / TP | `DP_SIZE=2 TP_SIZE=2` | `DP_SIZE=2 TP_SIZE=2` |
| 本 pod fork 数 | `DP_SIZE_LOCAL=2`（fork 2 服务） | `DP_SIZE_LOCAL=1`（每 pod 1 rank） |
| rank | rank 0,1（同 pod） | D-1=rank0 / D-2=rank1 |
| kv-config `prefill` | `{dp:2,tp:2}` | `{dp:2,tp:2}` |
| kv-config `decode` | `{dp:2,tp:2}` | `{dp:2,tp:2}` |

> **拓扑一致性铁律**：P 和 D 最终 kv-config 里 `prefill{}` 必须完全相同、`decode{}` 必须完全相同，
> 否则 mooncake KV rank 映射两端算不一致 → 传错/握手失败。

---

## 2. 端口规划（proxy 要按这个表填）

| 节点/rank | 引擎 HTTP 端口 | kv_port | bootstrap | 可见卡 |
|-----------|---------------|---------|-----------|--------|
| P rank0 | `17000` | 30000 | 23000 | 0,1 |
| P rank1 | `17001` | 30001 | 23001 | 2,3 |
| D-1（rank0） | `17000` | 30100 | 23100 | 0,1 |
| D-2（rank1） | `17000` | 30100 | 23100 | 0,1 |

> 端口算法（external-lb fork 脚本）：`PORT=ENGINE_PORT+i`、`kv_port=base+i`、`bootstrap=BOOTSTRAP+i`、
> `卡=[i*TP, (i+1)*TP)`。P 的 `i∈{0,1}`，D 的 `i=0`。不同 pod 同端口不冲突。

---

## 3. P 节点（4 卡，DP2×TP2，kv_producer）

### 3.1 环境变量（注入 **wings-control 容器** env；裸跑加 `export`）

```bash
export PD_ROLE=P
export WINGS_DEVICE=ascend WINGS_DEVICE_COUNT=4
# —— external-lb fork 编排（dp>1 的命脉）——
export DP_SIZE=2 TP_SIZE=2 DP_SIZE_LOCAL=2
export Master_IP=<P_IP> NODE_IPS=<P_IP> HOST_IP=<P_IP>     # 单 pod，三者同 = P 自己
export VLLM_LLMDD_RPC_PORT=12321 VLLM_MOONCAKE_BOOTSTRAP_PORT=23000
export ENGINE_PORT=17000
# —— 对端（decode）拓扑，给 mooncake KV 映射；必须和 D 实际一致 ——
export PD_DECODE_DP_SIZE=2 PD_DECODE_TP_SIZE=2
export PD_CONNECTOR_TYPE=MooncakeConnectorV1
export ASCEND_ENFORCE_EAGER=true
```

### 3.2 wings-control 启动命令

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name Qwen3-30B-A3B \
  --model-path /usr/local/serving/models/ \
  --device-count 4 --port 18000 \
  --input-length 4096 --output-length 4096 \
  --gpu-memory-utilization 0.9 --trust-remote-code --seed 42
```

### 3.3 wings 生成的引擎命令（仅核对，**不用手动执行**）

> 由 wings 写进 `/shared-volume/start_command.sh`、**engine 容器自动执行**。起来后
> `cat /shared-volume/start_command.sh` 对照下面核对结构即可（已 dry_run 验证）。

```bash
for i in $(seq 0 1); do
  RANK=$((0 + i)); PORT=$((17000 + i)); KVPORT=$((30000 + i)); BOOTSTRAP=$((23000 + i))
  LO=$((i * 2)); HI=$((LO + 1)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /usr/local/serving/models/ --served-model-name Qwen3-30B-A3B \
    --port $PORT --tensor-parallel-size 2 \
    --data-parallel-size 2 --data-parallel-rank $RANK --data-parallel-size-local 1 \
    --data-parallel-address <P_IP> --data-parallel-rpc-port 12321 --data-parallel-external-lb \
    --enable-expert-parallel --enable-prefix-caching --enforce-eager \
    --additional-config '{"enable_cpu_binding":"True"}' \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer",
      "kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'",
      "kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}' &
done; wait -n
```

---

## 4. D 节点 ×2（各 2 卡，组成 DP2，kv_consumer）

D-1 / D-2 **只差 `HOST_IP`**（`Master_IP`/`NODE_IPS` 两端写法完全一致）。

### 4.1 环境变量

```bash
export PD_ROLE=D
export WINGS_DEVICE=ascend WINGS_DEVICE_COUNT=2
export DP_SIZE=2 TP_SIZE=2 DP_SIZE_LOCAL=1
export Master_IP=<D1_IP>                 # = data-parallel-address，指向 D-1（rank0）
export NODE_IPS=<D1_IP>,<D2_IP>          # 顺序即 rank：D-1=0, D-2=1（两 pod 写法一致）
export HOST_IP=<本D_IP>                  # D-1 填 <D1_IP> / D-2 填 <D2_IP>
export VLLM_LLMDD_RPC_PORT=12321 VLLM_MOONCAKE_BOOTSTRAP_PORT=23100
export ENGINE_PORT=17000
# —— 对端（prefill）拓扑；必须和 P 实际一致 ——
export PD_PREFILL_DP_SIZE=2 PD_PREFILL_TP_SIZE=2
export PD_CONNECTOR_TYPE=MooncakeConnectorV1
export ASCEND_ENFORCE_EAGER=true
```

### 4.2 wings-control 启动命令（D-1 / D-2 相同）

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name Qwen3-30B-A3B \
  --model-path /usr/local/serving/models/ \
  --device-count 2 --port 18000 \
  --input-length 4096 --output-length 4096 \
  --gpu-memory-utilization 0.9 --trust-remote-code --seed 42
```

### 4.3 wings 生成的引擎命令（每个 D pod fork 1 个；仅核对，**不用手动执行**）

> 同 §3.3：engine 容器自动执行 `/shared-volume/start_command.sh`，这里只供对照。

```bash
for i in $(seq 0 0); do
  RANK=$((<rank_start> + i)); PORT=$((17000 + i)); KVPORT=$((30100 + i)); BOOTSTRAP=$((23100 + i))
  CARDS=0,1
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /usr/local/serving/models/ --served-model-name Qwen3-30B-A3B \
    --port $PORT --tensor-parallel-size 2 \
    --data-parallel-size 2 --data-parallel-rank $RANK --data-parallel-size-local 1 \
    --data-parallel-address <D1_IP> --data-parallel-rpc-port 12321 --data-parallel-external-lb \
    --enable-expert-parallel --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer",
      "kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'",
      "kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}' &
done; wait -n
```

> `<rank_start>` = `HOST_IP` 在 `NODE_IPS` 中的位置 ×`DP_SIZE_LOCAL`：D-1→0、D-2→1（自动派生，不手填）。

---

## 5. ⭐ 在 wings 外面起 PD proxy（触发 mooncake 的那块，必做）

引擎起好后，**这一步才让 mooncake 真正传 KV**。`--prefiller-*` 指 §2 表里 P 的两个服务、`--decoder-*` 指两个 D：

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 18000 --host 0.0.0.0 \
  --prefiller-hosts <P_IP> <P_IP> \
  --prefiller-ports 17000 17001 \
  --decoder-hosts  <D1_IP> <D2_IP> \
  --decoder-ports  17000 17000
```

> 压测/curl 打 **proxy 的 :18000**，不是引擎的 17000，也不是 wings 自己的 18000。

---

## 6. K8s 探针（冷启动别被打死）

Qwen3-30B 多卡冷加载 + DP rendezvous 常 > liveness 默认 ~160s 宽限。给 engine、wings-control 两容器都加 startupProbe：

```yaml
startupProbe:
  httpGet: { path: /health, port: 19000 }
  periodSeconds: 10
  failureThreshold: 120       # ~20 分钟，覆盖加载 + DP rendezvous
  timeoutSeconds: 5
livenessProbe:
  httpGet: { path: /health, port: 19000 }
  initialDelaySeconds: 0
  periodSeconds: 20
  failureThreshold: 6
```

> 改 Deployment（非 Pod）；平台 `backend-serving` 生成的需回填模板才持久。

---

## 7. 验证（mooncake 触发判据）

```bash
# 1) external-lb 是否触发（没这行 = dp>1 没生效，查 §9 排障）
kubectl logs <P-pod> -n namespace-0 -c wings-control | grep "PD external-lb"
#   期望: [PD external-lb] arch=Qwen3MoeForCausalLM role=P ... dp_size=2 local=2

# 2) 各引擎 health
curl -s http://<P_IP>:17000/health ; curl -s http://<P_IP>:17001/health
curl -s http://<D1_IP>:17000/health ; curl -s http://<D2_IP>:17000/health

# 3) 打 proxy 出预测
curl -H 'Content-type: application/json' -X POST http://<proxy_IP>:18000/v1/chat/completions \
  -d '{"model":"Qwen3-30B-A3B","messages":[{"role":"user","content":"你是谁"}],"max_tokens":64,"temperature":0}'

# 4) ★ 确认 mooncake 真传了 KV（核心判据：两端都要有）
kubectl logs <P-pod>  -c engine | grep -iE "kv_producer|mooncake|transfer|remote_decode"
kubectl logs <D1-pod> -c engine | grep -iE "kv_consumer|mooncake|remote_prefill|pull"
#   两端都有 mooncake transfer/握手 = PD 真互通；只有 P 有 = 还是 P 单干
```

---

## 8. 拓扑一致性自查（必须四个值对齐）

| kv-config 块 | P 端来源 | D 端来源 | 本拓扑值 |
|------|------|------|------|
| `prefill` | P 的 `DP_SIZE/TP_SIZE` | D 的 `PD_PREFILL_*` | `{dp:2,tp:2}` |
| `decode` | P 的 `PD_DECODE_*` | D 的 `DP_SIZE/TP_SIZE` | `{dp:2,tp:2}` |

> 即：P 设 `DP_SIZE=2 TP_SIZE=2 PD_DECODE_DP_SIZE=2 PD_DECODE_TP_SIZE=2`；
> D 设 `DP_SIZE=2 TP_SIZE=2 PD_PREFILL_DP_SIZE=2 PD_PREFILL_TP_SIZE=2`。四值一致 → 两端 kv-config 完全相同。

---

## 9. 排障

| 现象 | 原因 | 处理 |
|------|------|------|
| wings 日志无 `[PD external-lb]` | `DP_SIZE` 没被读成 >1（名字/镜像版本/生成时机） | 进容器 `printenv DP_SIZE`；确认部署镜像读 `DP_SIZE` 还是 `PD_DP_SIZE`（见 [dp-decode-group §10](./deploy-qwen3-30b-pd-dp-decode-group.md)） |
| 命令是 `--tensor-parallel-size 4` 单进程 | 走了 standalone（`DP_SIZE` 未生效） | 同上，先让 `DP_SIZE=2` 真正生效 |
| 预测通但 D engine 无 mooncake 日志 | **没起 §5 proxy** 或端口填错 | 起 proxy，端口对齐 §2 表 |
| `tuple object has no attribute shape` | 用了上游 `MooncakeConnector` | 用 `MooncakeConnectorV1`（注册表已是） |
| KV 映射错/握手失败 | `prefill`/`decode` 两端不一致 | 对齐 §8 四个值 |
| 冷启动反复重启 | liveness 宽限太短 | §6 startupProbe |

---

## 10. 占位替换清单

| 占位 | 含义 | 你的值 |
|------|------|------|
| `<P_IP>` | P 节点 IP（=`HOST_IP`/`Master_IP`） | 94.254.84.66 |
| `<D1_IP>` / `<D2_IP>` | 两个 D pod IP；`NODE_IPS` 两端一致 | 按实际 |
| `<proxy_IP>` | 跑 `load_balance_proxy_server_example.py` 的机器 | 按实际 |
| `--input-length` / `--output-length` | 上下文（`max_model_len=两者之和`） | 各 4096 → 8192 |
