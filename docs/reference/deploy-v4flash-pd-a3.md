# DeepSeek-V4-Flash PD 分离部署（A3 1P1D / dp>1 / external-lb / mooncake）

> 目标拓扑：**P 节点 1 个（16 卡）+ D 节点 1 个（16 卡）**，**强制 dp>1**，走
> vLLM-Ascend external-lb（DP fork）+ **MooncakeHybridConnector**。
>
> 并行：**P = DP4×TP4**（1 P pod fork 4 个 prefill、各 TP4）；**D = DP16×TP1**（1 D pod fork 16 个、各 TP1）。
> 全局拓扑 `prefill={dp:4,tp:4}` / `decode={dp:16,tp:1}`。合计 **2 pod / 32 卡**。
>
> proxy 放 wings 外面（见 §5）。同款 GLM-5.1 见 [deploy-glm5.1-pd-a3.md](./deploy-glm5.1-pd-a3.md)；字段逐项对齐官方见 [pd-a3-official-alignment-report.md](./pd-a3-official-alignment-report.md)；官方 [DeepSeek-V4-Flash](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html)。**A2(4P1D) 见 §0.8**。

---

## 0. 关键事实（先读）

1. **必须走 external-lb（dp>1）**：触发 = `PD_ROLE∈{P,D}` **且 `DP_SIZE>1`**（[config_loader.py:935/943](../../wings_control/core/config_loader.py#L935)）。
2. **`DeepseekV4ForCausalLM` 已注册** [pd_config.json](../../wings_control/config/defaults/pd_config.json)：命中后自动补 max-model-len=1048576、seed=1024、enforce-eager(P)、async+FULL_DECODE_ONLY(D)、`no-enable-prefix-caching`、**`no-disable-hybrid-kv-cache-manager`**、`reasoning/tool parser=deepseek_v4`、`model-loader-extra-config`、`speculative {1,mtp,enforce_eager}`、`kv_port`、`engine_id` 等（逐项对齐官方 A3）。
3. **连接器 = `MooncakeHybridConnector`**（非 V1）；KV producer 30000 / consumer 30100（按 service 偏移）。
4. **平台 a3**：给**任一** a3 信号即可——`WINGS_ASCEND_PLATFORM=a3` / a3 镜像 `ENGINE_VERSION` 带 `-a3` / `ASCEND_A3_ENABLE=1` / `hardware_info.json` 含 910c。**全无信号回退 a2 → 拿到 A2(4P1D) 的 batched 值，A3 部署会错**（见 §0.8）。
5. **本机 IP 只设 `RANK_IP`**：`get_local_ip()` 读它，`HCCL_IF_IP`/rank_start 都回退到它，不必再设 `POD_IP`/`HOST_IP`。
6. **`DP_SIZE`/`TP_SIZE` 可省**：由 `PD_{ROLE}_*` 派生（P→`PD_PREFILL_*`，D→`PD_DECODE_*`）。
7. **不要传** `--tensor-parallel-size`/`--max-model-len`。
8. **A2(4P1D)**：注册表 `platform_overrides.a2` 仅 overlay 高置信 `batched/seqs`（P 4096/16、D 60/30），其余继承 A3；A2 部署须让平台信号解析为 a2，且 A2 值落地前逐项核官方。本文主体对齐 **A3**。

---

## 1. 拓扑与全局并行

| | P 节点（1 pod，16 卡） | D 节点（1 pod，16 卡） |
|---|---|---|
| 角色 | `PD_ROLE=P` → kv_producer | `PD_ROLE=D` → kv_consumer |
| DP / TP | `DP_SIZE=4 TP_SIZE=4` | `DP_SIZE=16 TP_SIZE=1` |
| 本 pod fork 数 | `DP_SIZE_LOCAL=4` | `DP_SIZE_LOCAL=16` |
| rank | rank 0-3（同 pod） | rank 0-15（同 pod） |
| kv-config `prefill` / `decode` | `{dp:4,tp:4}` / `{dp:16,tp:1}` | 同（两端必须完全一致） |

---

## 2. 端口规划（proxy 按此表填）

| 节点/rank | 引擎 HTTP（`ENGINE_PORT+i`） | kv_port（`base+i`） | bootstrap | 可见卡（`i*TP..`） |
|-----------|---------------|---------|-----------|--------|
| P rank0-3 | `17000-17003` | 30000-30003 | 23000-23003 | 0-3 / 4-7 / 8-11 / 12-15 |
| D rank0-15 | `17000-17015` | 30100-30115 | 23100-23115 | 每 service 1 卡（0,1,…,15） |

> 端口算法：`PORT=ENGINE_PORT+i`、`kv_port=base+i`（P 30000 / D 30100）、`bootstrap=BOOTSTRAP+i`（P 23000 / D 23100）、`卡=[i*TP,(i+1)*TP)`。

---

## 3. P 节点（16 卡，DP4×TP4，kv_producer）

### 3.1 环境变量

```bash
export PD_ROLE=P
export WINGS_DEVICE=ascend  WINGS_DEVICE_COUNT=16
export WINGS_ASCEND_PLATFORM=a3            # 或 a3 镜像 ENGINE_VERSION 带 -a3 即可省
export DP_SIZE=4 TP_SIZE=4 DP_SIZE_LOCAL=4      # DP_SIZE/TP_SIZE 可省（由 PD_PREFILL_* 派生）
export Master_IP=<P_IP> NODE_IPS=<P_IP> RANK_IP=<P_IP>   # 单 pod，三者同 = P 自己
export VLLM_LLMDD_RPC_PORT=10521  ENGINE_PORT=17000
export PD_PREFILL_DP_SIZE=4  PD_PREFILL_TP_SIZE=4
export PD_DECODE_DP_SIZE=16  PD_DECODE_TP_SIZE=1
```

### 3.2 wings-control 启动命令

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name DeepSeek-V4-Flash \
  --model-path /usr/local/serving/models/ \
  --device-count 16 --port 18000 \
  --gpu-memory-utilization 0.9 --trust-remote-code --seed 1024
```

### 3.3 wings 生成的引擎命令（dry-run 实测；仅核对结构）

```bash
for i in $(seq 0 3); do            # DP_SIZE_LOCAL=4 → fork 4
  RANK=$((0 + i)); PORT=$((17000 + i)); KVPORT=$((30000 + i)); BOOTSTRAP=$((23000 + i))
  LO=$((i * 4)); HI=$((LO + 3)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    vllm serve /usr/local/serving/models/ --served-model-name DeepSeek-V4-Flash \
    --max-model-len 1048576 --quantization ascend --seed 1024 --block-size 128 \
    --max-num-batched-tokens 8192 --max-num-seqs 16 --gpu-memory-utilization 0.9 \
    --enforce-eager --no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --safetensors-load-strategy prefetch \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
    --enable-expert-parallel --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}' \
    --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_dsa_cp":true}' \
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' \
    --api-server-count 1 --port $PORT --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <P_IP> --data-parallel-rpc-port 10521 --data-parallel-external-lb \
    --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}}' &
done; wait -n
```
> 角色 env：P 加 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`+`FUSED_MC2`。

---

## 4. D 节点（16 卡，DP16×TP1，kv_consumer）

### 4.1 环境变量

```bash
export PD_ROLE=D
export WINGS_DEVICE=ascend  WINGS_DEVICE_COUNT=16
export WINGS_ASCEND_PLATFORM=a3
export DP_SIZE=16 TP_SIZE=1 DP_SIZE_LOCAL=16     # DP_SIZE/TP_SIZE 可省（由 PD_DECODE_* 派生）
export Master_IP=<D_IP> NODE_IPS=<D_IP> RANK_IP=<D_IP>   # 单 pod，三者同
export VLLM_LLMDD_RPC_PORT=10523  ENGINE_PORT=17000
export PD_PREFILL_DP_SIZE=4  PD_PREFILL_TP_SIZE=4
export PD_DECODE_DP_SIZE=16  PD_DECODE_TP_SIZE=1
```

### 4.2 wings-control 启动命令

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name DeepSeek-V4-Flash \
  --model-path /usr/local/serving/models/ \
  --device-count 16 --port 18000 \
  --gpu-memory-utilization 0.9 --trust-remote-code --seed 1024
```

### 4.3 wings 生成的引擎命令（fork 16；仅核对）

```bash
for i in $(seq 0 15); do           # DP_SIZE_LOCAL=16 → fork 16
  RANK=$((0 + i)); PORT=$((17000 + i)); KVPORT=$((30100 + i)); BOOTSTRAP=$((23100 + i))
  LO=$((i * 1)); HI=$LO; CARDS=$LO
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    vllm serve /usr/local/serving/models/ --served-model-name DeepSeek-V4-Flash \
    --max-model-len 1048576 --quantization ascend --seed 1024 --block-size 128 \
    --max-num-batched-tokens 120 --max-num-seqs 60 --gpu-memory-utilization 0.9 \
    --async-scheduling --no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --safetensors-load-strategy prefetch \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}' \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
    --enable-expert-parallel --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}' \
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' \
    --api-server-count 1 --port $PORT --tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <D_IP> --data-parallel-rpc-port 10523 --data-parallel-external-lb \
    --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}}' &
done; wait -n
```
> D 无 `--enforce-eager`（走 cudagraph）。角色 env：D 加 `VLLM_ASCEND_ENABLE_FUSED_MC2=1`+`MLAPO`。

---

## 5. ⭐ 在 wings 外起 PD proxy（触发 mooncake，必做）

`--prefiller-*` 指 P pod 的 4 个 service（17000-17003）、`--decoder-*` 指 D pod 的 16 个 service（17000-17015）：

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 18000 --host 0.0.0.0 \
  --prefiller-hosts <P_IP> <P_IP> <P_IP> <P_IP> \
  --prefiller-ports 17000 17001 17002 17003 \
  --decoder-hosts  $(yes <D_IP> | head -16) \
  --decoder-ports  $(seq -s' ' 17000 17015)
```
> 压测/curl 打 **proxy :18000**。

---

## 6. K8s 探针

```yaml
startupProbe: { httpGet: { path: /health, port: 19000 }, periodSeconds: 10, failureThreshold: 120, timeoutSeconds: 5 }
livenessProbe: { httpGet: { path: /health, port: 19000 }, periodSeconds: 20, failureThreshold: 6 }
```

---

## 7. 验证（mooncake 触发判据）

```bash
# 1) external-lb 触发
kubectl logs <P-pod> -c wings-control | grep "PD external-lb"
#   期望: [PD external-lb] arch=DeepseekV4ForCausalLM role=P connector=MooncakeHybridConnector dp_size=4 local=4 rank_start=0
# 2) health：P 4 端口(17000-17003)，D 16 端口(17000-17015)
curl -s http://<P_IP>:1700{0..3}/health ; for p in $(seq 17000 17015); do curl -s http://<D_IP>:$p/health; done
# 3) 打 proxy；4) 两端 mooncake 日志
kubectl logs <P-pod> -c engine | grep -iE "kv_producer|mooncake|transfer"
kubectl logs <D-pod> -c engine | grep -iE "kv_consumer|mooncake|pull"
```

---

## 8. 拓扑一致性自查（四值对齐）

| kv-config 块 | P 端来源 | D 端来源 | 本拓扑值 |
|------|------|------|------|
| `prefill` | P 的 `DP_SIZE/TP_SIZE`（或 `PD_PREFILL_*`） | D 的 `PD_PREFILL_*` | `{dp:4,tp:4}` |
| `decode` | P 的 `PD_DECODE_*` | D 的 `DP_SIZE/TP_SIZE`（或 `PD_DECODE_*`） | `{dp:16,tp:1}` |

> 所有 pod 都设 `PD_PREFILL_DP_SIZE=4 PD_PREFILL_TP_SIZE=4 PD_DECODE_DP_SIZE=16 PD_DECODE_TP_SIZE=1`。

---

## 9. 排障 / 待确认

| 现象 / 项 | 原因 | 处理 |
|------|------|------|
| 无 `[PD external-lb]` | `DP_SIZE` 没读成 >1 | `printenv DP_SIZE`；确认镜像读 `DP_SIZE`/`PD_DP_SIZE` |
| P batched=4096（应 8192）/ HCCL_BUFFSIZE=512 | 平台被判成 a2 → 走了 A2 overlay | 给 a3 信号（§0.4），`printenv WINGS_ASCEND_PLATFORM ENGINE_VERSION` |
| **engine_id**（待真机） | 官方 Hybrid 示例固定 `0/1`，wings 按 dp_rank 注入 | 多 service 下按 rank 更合理；真机确认 Mooncake Hybrid 期望 |
| **hybrid-kv**（待真机） | V4-Flash 须 `--no-disable-hybrid-kv-cache-manager`（保留 HMA）；注册表已在 guard 之后注入 | 确认生成命令含该 flag，未被 `_guard_pd_hybrid_kv_cache` 吃掉 |
| 预测通但 D 无 mooncake 日志 | 没起 §5 proxy / 端口错 | 起 proxy，端口对齐 §2 |
| KV 映射错/握手失败 | `prefill`/`decode` 两端不一致 | 对齐 §8 四值 |

---

## 10. 占位替换清单

| 占位 | 含义 |
|------|------|
| `<P_IP>` | P pod IP（=`RANK_IP`/`Master_IP`/`NODE_IPS`，单 pod 三者同） |
| `<D_IP>` | D pod IP（单 pod 三者同） |
| `<proxy_IP>` | 跑 `load_balance_proxy_server_example.py` 的机器 |
