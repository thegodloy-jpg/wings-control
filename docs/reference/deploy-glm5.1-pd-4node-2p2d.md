# GLM-5.1 PD 分离部署（4 机 / 64 卡：2P×TP16 + 2D×TP4，external-lb + mooncake）

> 目标拓扑：**P 节点 2 个（各 16 卡，DP2×TP16）+ D 节点 2 个（各 16 卡，DP8×TP4）**，共 **4 机 / 64 卡**。
> 命中注册表 `GlmMoeDsaForCausalLM`，走 vLLM-Ascend external-lb（DP fork）+ MooncakeConnectorV1。
>
> 这是官方满规模（P:DP2×TP16=2节点 + D:DP16×TP4=4节点 = 6节点/96卡）的 **4 机缩规模版**：P 与官方一致，D 取 DP8（官方 DP16 的一半，TP4 不变）。
> 字段级对齐见 [pd-a3-official-alignment-report.md](pd-a3-official-alignment-report.md)；机制见 [deepseek-v32-pd-disaggregation.md](../../xuqiu/deepseek-v32-pd-disaggregation.md)。

---

## 0. 关键事实（先读）

1. **命中 GLM 注册表靠架构,不靠机器数**:权重 `config.json` 的 `architectures=["GlmMoeDsaForCausalLM"]` + `PD_ROLE∈{P,D}` 且 `DP_SIZE>1` → `_apply_pd_external_lb` 命中 GLM5 条目,自动下发其全部调优(连接器/kv_port/parser/compilation/additional-config/common_env)。
2. **⚠️ 内存(EP-MoE 首要可行性)**:专家按 `EP=DP×TP` 切。官方 D EP64;本拓扑 D EP32(DP8×TP4)→ **每卡专家权重 ≈ 官方 D 的 2×**。P 是 EP32(= 官方 P,已证装得下),故权重在 EP32 下放得下;但 D 的 `max_model_len=200000` KV 很吃显存,**首跑重点看 D 是否 OOM**,必要时临时调小注册表 D 的 `max_model_len`/`max_num_seqs`。
3. **少传参数 = 注册表生效**:wings_start.sh 只下发 PD 契约 env + `--model-name/--model-path/--engine/--device-count`,**不要传** `--gpu-memory-utilization`/`--max-num-seqs`/`--max-num-batched-tokens`/`--block-size`/`--enable-chunked-prefill`/`--enable-prefix-caching`（及同名 ENV）——传了就顶掉注册表(见 [pd-registry-authoritative-design.md](pd-registry-authoritative-design.md)）。
4. **上层 proxy 必做**:wings 不做负载均衡;mooncake 靠 §5 外部 proxy 把 P→D 串起来才真传 KV。

---

## 1. 拓扑与全局并行

| | P 节点 ×2（各 16 卡） | D 节点 ×2（各 16 卡） |
|---|---|---|
| 角色 | `PD_ROLE=P` → kv_producer | `PD_ROLE=D` → kv_consumer |
| 并行 | **DP2×TP16** | **DP8×TP4** |
| `DP_SIZE` / `TP_SIZE`（派生） | 2 / 16 | 8 / 4 |
| `DP_SIZE_LOCAL`（每 pod fork） | **1**（16÷16） | **4**（16÷4） |
| pod 内 service 数 | 1（TP16 占满 16 卡） | 4（各 TP4） |
| rank 分布 | P-0=rank0 / P-1=rank1 | D-0=rank0-3 / D-1=rank4-7 |
| kv-config `prefill` | `{dp:2, tp:16}` | `{dp:2, tp:16}` |
| kv-config `decode` | `{dp:8, tp:4}` | `{dp:8, tp:4}` |

> **派生公式**（上层算好下发，wings 不自算）：`DP_SIZE=节点数×16÷TP`、`DP_SIZE_LOCAL=16÷TP`、`dp_rank_start=角色内节点序×DP_SIZE_LOCAL`（wings 由 `RANK_IP` 在 `NODE_IPS` 的位置派生）、`dp_address=角色 node0 IP`。
> **拓扑一致性铁律**：P、D 两边 kv-config 的 `prefill{}` 必须完全相同、`decode{}` 必须完全相同（本表 `{2,16}`/`{8,4}`），否则 mooncake KV rank 映射两端算不一致 → 传错/握手失败。

---

## 2. 端口规划（proxy 按此表填）

| 节点/rank | 引擎 HTTP | kv_port | bootstrap | 可见卡 |
|---|---|---|---|---|
| P-0（rank0） | `17000` | 30000 | 23000 | 0-15 |
| P-1（rank1） | `17000` | 30000 | 23000 | 0-15 |
| D-0（rank0-3） | `17000`~`17003` | 30100~30103 | 23100~23103 | 0-3 · 4-7 · 8-11 · 12-15 |
| D-1（rank4-7） | `17000`~`17003` | 30100~30103 | 23100~23103 | 0-3 · 4-7 · 8-11 · 12-15 |

> fork 脚本算法：`PORT=ENGINE_PORT+i`、`kv_port=base+i`、`bootstrap=BOOTSTRAP+i`、卡=`[i*TP,(i+1)*TP)`。P 的 `i∈{0}`，D 的 `i∈{0..3}`。不同 pod 同端口不冲突（各自 pod IP）。

---

## 3. 逐 Pod 下发（环境变量契约）

### 3.0 公共 env（4 个 pod 都设）
```bash
export WINGS_DEVICE=ascend  WINGS_ASCEND_PLATFORM=a3  DEVICE_COUNT=16
# 4 个全局拓扑（单一真相源；本角色 DP_SIZE/TP_SIZE 由此派生，不必单独下发）
export PD_PREFILL_DP_SIZE=2  PD_PREFILL_TP_SIZE=16
export PD_DECODE_DP_SIZE=8   PD_DECODE_TP_SIZE=4
```

### 3.1 P 节点（P-0 / P-1，只差 RANK_IP）
```bash
export PD_ROLE=P
export DP_SIZE_LOCAL=1
export Master_IP=<P0_IP>                 # = data-parallel-address，指向 P-0
export NODE_IPS=<P0_IP>,<P1_IP>          # 顺序即 rank：P-0=0, P-1=1（两 pod 写法一致）
export RANK_IP=<本P_IP>                  # P-0 填 <P0_IP> / P-1 填 <P1_IP>（须逐字在 NODE_IPS 内）
export VLLM_LLMDD_RPC_PORT=10521  ENGINE_PORT=17000
```

### 3.2 D 节点（D-0 / D-1，只差 RANK_IP）
```bash
export PD_ROLE=D
export DP_SIZE_LOCAL=4
export Master_IP=<D0_IP>                 # 指向 D-0
export NODE_IPS=<D0_IP>,<D1_IP>          # D-0=rank_start 0, D-1=rank_start 4
export RANK_IP=<本D_IP>                  # D-0 填 <D0_IP> / D-1 填 <D1_IP>
export VLLM_LLMDD_RPC_PORT=10523  ENGINE_PORT=17000
```

> 本机 IP **只设 `RANK_IP`**（`HCCL_IF_IP`/`rank_start` 都回退到它，且 rank 派生取 RANK_IP 优先，不必再设 `POD_IP`/`HOST_IP`）。
> ⚠️ **不要传 `--distributed`**（PD external-lb 走对等 standalone 启动器；`--distributed` 会误入 Ray master/worker，见设计 §13.7）。

### 3.3 wings_start.sh（4 个 pod 同款,仅"少参数"）
```bash
bash /opt/wings-control/wings_start.sh --engine vllm_ascend \
  --model-name GLM-5.1 --model-path /usr/local/serving/models/ \
  --device-count 16 --port 18000 --trust-remote-code
# 不传 gpu-mem / max-num-seqs / max-num-batched-tokens / block-size / chunked-prefill / prefix-caching
# 也不传 input/output-length（GLM5 注册表已定 max-model-len：P=131072 / D=200000）
```

---

## 4. wings 生成的引擎命令（仅核对，引擎容器自动执行）

### 4.1 P 节点（fork 1 个 service / pod）
```bash
for i in $(seq 0 0); do
  RANK=$((0 + i)); PORT=$((17000+i)); KVPORT=$((30000+i)); BOOTSTRAP=$((23000+i)); CARDS=$(seq -s, 0 15)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server --trust-remote-code \
    --max-model-len 131072 --quantization ascend --seed 1024 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.95 --enable-chunked-prefill \
    --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"ascend_compilation_config":{"enable_npugraph_ex":true},"enable_dsa_cp":true}' \
    --served-model-name GLM-5.1 --model /usr/local/serving/models/ --dtype auto --kv-cache-dtype auto --block-size 16 \
    --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}' --enforce-eager \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":16},"decode":{"dp_size":8,"tp_size":4},"use_ascend_direct":true}}' \
    --port $PORT --tensor-parallel-size 16 --data-parallel-size 2 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <P0_IP> --data-parallel-rpc-port 10521 --data-parallel-external-lb &
done; wait -n
```
> P 公共 env 另注入 `HCCL_BUFFSIZE=256 ASCEND_AGGREGATE_ENABLE=1 ACL_OP_INIT_MODE=1 ASCEND_A3_ENABLE=1 VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480`（common_env）+ 角色 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1 VLLM_ASCEND_ENABLE_FUSED_MC2=1`。

### 4.2 D 节点（fork 4 个 service / pod）
```bash
for i in $(seq 0 3); do
  RANK=$((<rank_start> + i)); PORT=$((17000+i)); KVPORT=$((30100+i)); BOOTSTRAP=$((23100+i))
  LO=$((i*4)); HI=$((LO+3)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server --trust-remote-code \
    --max-model-len 200000 --quantization ascend --seed 1024 \
    --max-num-seqs 8 --max-num-batched-tokens 32 --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}' \
    --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}' \
    --served-model-name GLM-5.1 --model /usr/local/serving/models/ --dtype auto --kv-cache-dtype auto --block-size 16 \
    --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}' \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":16},"decode":{"dp_size":8,"tp_size":4},"use_ascend_direct":true}}' \
    --port $PORT --tensor-parallel-size 4 --data-parallel-size 8 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <D0_IP> --data-parallel-rpc-port 10523 --data-parallel-external-lb &
done; wait -n
```
> `<rank_start>`：D-0→0、D-1→4（= RANK_IP 在 NODE_IPS 位置 ×4，自动派生，不手填）。
> D 角色 env 另注入 common_env + `VLLM_ASCEND_ENABLE_MLAPO=1 TASK_QUEUE_ENABLE=1 VLLM_ASCEND_ENABLE_FUSED_MC2=1`。

---

## 5. ⭐ 外部 PD proxy（触发 mooncake，必做）
```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 18000 --host 0.0.0.0 \
  --prefiller-hosts <P0_IP> <P1_IP> \
  --prefiller-ports 17000 17000 \
  --decoder-hosts  <D0_IP> <D0_IP> <D0_IP> <D0_IP> <D1_IP> <D1_IP> <D1_IP> <D1_IP> \
  --decoder-ports  17000 17001 17002 17003 17000 17001 17002 17003
```
> prefiller = 2 个 P service（各 :17000）；decoder = 8 个 D service（D-0/D-1 各 4 个，:17000-3）。压测打 proxy 的 `:18000`。

---

## 6. K8s 探针（冷启动别被打死）
GLM5.1 多卡冷加载 + DP rendezvous 久。engine、wings-control 两容器都加 startupProbe（`/health`），`failureThreshold≈120 periodSeconds=10`（~20 分钟）。
> ⚠️ PD external-lb 下每 pod 走 standalone 启动器，health 都在 **19000**（无 master/worker 偏移）。

---

## 7. 验证（逐层）
```bash
# 1) external-lb 触发（每个 pod 的 wings-control 容器）
kubectl logs <pod> -c wings-control | grep "PD external-lb"
#   期望 P: ...role=P dp_size=2 local=1 rank_start=0/1   D: ...role=D dp_size=8 local=4 rank_start=0/4
# 2) 进程/端口：P 每 pod 1 个、D 每 pod 4 个 vllm
kubectl exec <D-pod> -- bash -lc 'pgrep -af "vllm.*api_server" | wc -l'   # D 期望 4
# 3) 注册表参数确实生效（没被平台 flag 顶掉）
kubectl exec <D-pod> -- bash -lc 'grep -oE "max-num-batched-tokens [0-9]+|max-num-seqs [0-9]+|gpu-memory-utilization [0-9.]+" /shared-volume/start_command.sh'
#   D 期望 32 / 8 / 0.92；P 期望 4096 / 64 / 0.95
# 4) ★ mooncake 真传 KV（两端都要有）
kubectl logs <P-pod> -c engine | grep -iE "kv_producer|mooncake|transfer"
kubectl logs <D-pod> -c engine | grep -iE "kv_consumer|mooncake|pull"
# 5) 冒烟推理（打 proxy）
curl -s http://<proxy>:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.1","messages":[{"role":"user","content":"你是谁"}],"max_tokens":64}'
```

---

## 8. 拓扑一致性自查（四个值对齐）
| kv-config 块 | P 端来源 | D 端来源 | 本拓扑值 |
|---|---|---|---|
| `prefill` | P 的 `DP_SIZE/TP_SIZE` | D 的 `PD_PREFILL_*` | `{dp:2,tp:16}` |
| `decode` | P 的 `PD_DECODE_*` | D 的 `DP_SIZE/TP_SIZE` | `{dp:8,tp:4}` |
> 即四个全局值 `PD_PREFILL_DP_SIZE=2 PD_PREFILL_TP_SIZE=16 PD_DECODE_DP_SIZE=8 PD_DECODE_TP_SIZE=4` 在 P、D 所有 pod 上**完全一致**。

---

## 9. 排障
| 现象 | 原因 | 处置 |
|---|---|---|
| wings 日志无 `[PD external-lb]` | `DP_SIZE` 没读成 >1 / 镜像旧（无 external-lb） | 进容器 `printenv DP_SIZE PD_DECODE_DP_SIZE`；确认 sidecar 镜像 ≥ 含 external-lb |
| 命令变 `--tensor-parallel-size 16` 单进程、无 fork | 走了 standalone（external-lb 未触发） | 同上；确认 `PD_ROLE`+`DP_SIZE>1` |
| 多节点 rank 撞车 / DP 域起不来 | `RANK_IP` 未逐字在 `NODE_IPS` 内 / 用了 `--distributed` | 修 RANK_IP/NODE_IPS 文本一致；去掉 `--distributed` |
| D OOM | EP32 每卡专家 2× + max-model-len 200000 KV 过大 | 临时调小注册表 D 的 `max_model_len`/`max_num_seqs`；或扩到官方 6 节点 |
| 注册表 batched/seqs/gpu 被改成别的值 | 平台经 CLI/ENV 灌了 tuning flag | `grep "Starting wings application" wings_start.log` + `printenv`；让平台对 PD 停发这些 |
| 预测通但 D engine 无 mooncake 日志 | 没起 §5 proxy / 端口填错 | 起 proxy，端口对齐 §2 |
| FULL_DECODE_ONLY 编译崩（AICPU `open so failed`） | 引擎/镜像 AICPU 算子问题 | 临时 enforce-eager 验证；查 CANN/AICPU 包（见 [pd-a3-deploy-verify-guide §8](pd-a3-deploy-verify-guide.md)） |

---

## 10. 占位替换清单
| 占位 | 含义 | 你的值 |
|---|---|---|
| `<P0_IP>` / `<P1_IP>` | 2 个 P pod IP；`NODE_IPS` 两端一致、顺序即 rank | 按实际 |
| `<D0_IP>` / `<D1_IP>` | 2 个 D pod IP；同上 | 按实际 |
| `<proxy_IP>` | 跑 load_balance_proxy 的机器 | 按实际 |
| `VLLM_LLMDD_RPC_PORT` | DP RPC 端口（角色内一致；P/D 可各一常量） | P=10521 / D=10523（示例） |

---

## 附：与官方 6 节点的差异（缩规模点）
| | 官方满规模 | 本 4 机版 |
|---|---|---|
| P | DP2×TP16（2 节点） | **同** |
| D | DP16×TP4（4 节点，EP64） | **DP8×TP4（2 节点，EP32）** |
| 合计 | 6 节点 / 96 卡 | 4 节点 / 64 卡 |
| 注册表条目 | GlmMoeDsaForCausalLM | **同**（按架构命中，与规模无关） |
| 字段下发 | 同一套注册表参数 | **同**（dp/tp 拓扑不同不影响注册表引擎参数） |
| 唯一风险 | — | D 每卡专家内存 ×2（EP32 vs EP64）→ 盯 OOM |
