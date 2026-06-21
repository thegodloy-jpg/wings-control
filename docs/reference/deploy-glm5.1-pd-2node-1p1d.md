# GLM-5.1 PD 分离部署（2 机 / 32 卡：1P + 1D，仅功能验证）

> 目标拓扑：**P 节点 1 个（16 卡）+ D 节点 1 个（16 卡）**，共 **2 机 / 32 卡**。
> 用途：**在最小机器上把 GLM5.1 这个模型在 wings PD 流程里跑通验证**（命中注册表、external-lb fork、mooncake P→D、推理正确性）。
> ⚠️ **先读 §0：GLM-5.1 是 754B，2 机 1P1D（每侧 EP16）在 64GB/卡上放不下权重——能不能用这套取决于你的 HBM。** 要满规模/性能请用 [deploy-glm5.1-pd-4node-2p2d.md](deploy-glm5.1-pd-4node-2p2d.md) 或官方 6 节点。

---

## 0. 关键事实（先读：显存是否够，决定走哪条路）

> 实测自 [zai-org/GLM-5.1 config.json](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json)：`GlmMoeDsaForCausalLM`，**754B**，78 层（`first_k_dense_replace=3` → 75 层 MoE），`n_routed_experts=256` / `num_experts_per_tok=8` / `n_shared_experts=1`，`hidden=6144` / `moe_intermediate_size=2048`，MLA（`q_lora_rank=2048,kv_lora_rank=512`）+ DSA（`index_topk=2048`）。w8a8 ≈ **1 byte/参数**。

1. **⚠️ 权重账（首要可行性）**：**96% 参数在路由专家**（256×75 层×3×6144×2048 ≈ **725B**），按 `EP=DP×TP` 切；注意力/embedding/dense 按 TP 切。每卡权重：

   | 配置 | 路由专家/卡 | 其余/卡 | **权重合计/卡** | 64GB 卡余量 |
   |---|---|---|---|---|
   | **本 2 机 1P1D（每侧 EP16/TP4）** | 725/16 ≈ **45.3 GB** | ~5 | **≈ 50 GB** | ~14 GB ❌ |
   | 单实例 EP32（TP16×DP2 或 TP8×DP4，32 卡） | 725/32 ≈ 22.7 | ~2 | ≈ 25 GB | ~39 GB ✅ |
   | 官方 P（EP32/TP16） | 22.7 | ~1 | ≈ 24 GB | ~40 GB ✅ |
   | 官方 D（EP64/TP4） | 11.3 | ~5 | ≈ 16 GB | ~48 GB ✅ |

2. **⚠️ 路线判定（按你 910C 的 HBM）**：
   - **64GB/卡** → **1P1D 不可行**：单卡光权重 ~50GB，D 还要 `max_model_len=200000` 的巨量 KV（MLA+DSA 每 token ~100KB），必然 OOM。**验证模型请走 §11 单实例 EP32**（仍是这 2 机 32 卡，但作为一个实例跑，不分 PD），或验 PD 流程上 4 机版（P/D 各 EP32）。
   - **≥96GB/卡（如 910C 128GB SKU）** → 1P1D **可行**：~50GB 权重 + 调小后的 KV 放得下，可按 §1–§9 跑通完整 PD。首跑仍建议先调小 KV（下条）。
3. **KV 缓解（走 1P1D 时必做）**：临时把**注册表** GLM5 的 `decode.max_model_len`(200000→如 32768)、`prefill.max_model_len`(131072→如 16384)、`max_num_seqs` 调小（纯功能验证；max_model_len 注册表权威，**改注册表而非传 CLI**，否则顶掉注册表）。
4. **命中 GLM 注册表靠架构**：`architectures=["GlmMoeDsaForCausalLM"]` + `DP_SIZE>1` → 自动下发 GLM5 全套调优(连接器/parser/compilation/additional-config/common_env)，**与机器数无关**。
5. **少传参数 = 注册表生效**：只下发 PD 契约 env + `--model-name/--model-path/--engine/--device-count`，**不要传** tuning flag（gpu-mem/seqs/batched/block-size/chunked/prefix）及其同名 ENV。
6. **不要传 `--distributed`**（PD external-lb 走对等 standalone；`--distributed` 会误入 Ray master/worker，见设计 §13.7）。
7. **proxy 必做**（§5）才真传 mooncake KV。

---

## 1. 拓扑与全局并行

| | P 节点 ×1（16 卡） | D 节点 ×1（16 卡） |
|---|---|---|
| 角色 | `PD_ROLE=P` → kv_producer | `PD_ROLE=D` → kv_consumer |
| 并行 | **DP4×TP4** | **DP4×TP4** |
| `DP_SIZE` / `TP_SIZE`（派生） | 4 / 4 | 4 / 4 |
| `DP_SIZE_LOCAL`（每 pod fork） | **4**（16÷4） | **4** |
| pod 内 service 数 | 4（各 TP4） | 4（各 TP4） |
| rank | rank0-3（同 pod） | rank0-3（同 pod） |
| EP（=DP×TP） | 16 | 16 |
| kv-config `prefill` | `{dp:4, tp:4}` | `{dp:4, tp:4}` |
| kv-config `decode` | `{dp:4, tp:4}` | `{dp:4, tp:4}` |

> P、D 各单节点 → `dp_rank_start=0`（无跨节点 rank 派生）。TP4 与官方 D 一致（注意力/MLA 已证可切）；P 用 TP4（非官方 TP16）只影响 prefill 吞吐,功能不受影响。
> **一致性铁律**：四个全局值 `PD_PREFILL_DP_SIZE=4 PD_PREFILL_TP_SIZE=4 PD_DECODE_DP_SIZE=4 PD_DECODE_TP_SIZE=4` 在 P、D 两 pod 上**完全一致**。

---

## 2. 端口规划（proxy 按此填）

| 节点/rank | 引擎 HTTP | kv_port | bootstrap | 可见卡 |
|---|---|---|---|---|
| P（rank0-3） | `17000`~`17003` | 30000~30003 | 23000~23003 | 0-3 · 4-7 · 8-11 · 12-15 |
| D（rank0-3） | `17000`~`17003` | 30100~30103 | 23100~23103 | 0-3 · 4-7 · 8-11 · 12-15 |

> `PORT=ENGINE_PORT+i`、`kv_port=base+i`、`bootstrap=BOOTSTRAP+i`、卡=`[i*4,(i+1)*4)`。P/D 各 `i∈{0..3}`。两 pod 同端口不冲突（各自 pod IP）。

---

## 3. 逐 Pod 下发（环境变量契约）

### 3.0 公共 env（两个 pod 都设）
```bash
export WINGS_DEVICE=ascend  WINGS_ASCEND_PLATFORM=a3  DEVICE_COUNT=16
export PD_PREFILL_DP_SIZE=4  PD_PREFILL_TP_SIZE=4
export PD_DECODE_DP_SIZE=4   PD_DECODE_TP_SIZE=4
```

### 3.1 P 节点
```bash
export PD_ROLE=P  DP_SIZE_LOCAL=4
export Master_IP=<P_IP>  NODE_IPS=<P_IP>  RANK_IP=<P_IP>   # 单节点,三者同
export VLLM_LLMDD_RPC_PORT=10521  ENGINE_PORT=17000
```

### 3.2 D 节点
```bash
export PD_ROLE=D  DP_SIZE_LOCAL=4
export Master_IP=<D_IP>  NODE_IPS=<D_IP>  RANK_IP=<D_IP>
export VLLM_LLMDD_RPC_PORT=10523  ENGINE_PORT=17000
```
> 本机 IP 只设 `RANK_IP`（`HCCL_IF_IP`/rank 派生都取它）。

### 3.3 wings_start.sh（两 pod 同款，少参数）
```bash
bash /opt/wings-control/wings_start.sh --engine vllm_ascend \
  --model-name GLM-5.1 --model-path /usr/local/serving/models/ \
  --device-count 16 --port 18000 --trust-remote-code
# 不传 gpu-mem/seqs/batched/block-size/chunked/prefix；也不传 input/output-length（注册表已定 max-model-len）
```

---

## 4. wings 生成的引擎命令（仅核对）

### 4.1 P 节点（fork 4 个 TP4 service）
```bash
for i in $(seq 0 3); do
  RANK=$((0 + i)); PORT=$((17000+i)); KVPORT=$((30000+i)); BOOTSTRAP=$((23000+i))
  LO=$((i*4)); HI=$((LO+3)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server --trust-remote-code \
    --max-model-len 131072 --quantization ascend --seed 1024 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.95 --enable-chunked-prefill \
    --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"ascend_compilation_config":{"enable_npugraph_ex":true},"enable_dsa_cp":true}' \
    --served-model-name GLM-5.1 --model /usr/local/serving/models/ --dtype auto --kv-cache-dtype auto --block-size 16 \
    --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}' --enforce-eager \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":4,"tp_size":4},"use_ascend_direct":true}}' \
    --port $PORT --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <P_IP> --data-parallel-rpc-port 10521 --data-parallel-external-lb &
done; wait -n
```
> P 另注入 common_env（`HCCL_BUFFSIZE=256` 等 5 项）+ 角色 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1 VLLM_ASCEND_ENABLE_FUSED_MC2=1`。

### 4.2 D 节点（fork 4 个 TP4 service）
```bash
for i in $(seq 0 3); do
  RANK=$((0 + i)); PORT=$((17000+i)); KVPORT=$((30100+i)); BOOTSTRAP=$((23100+i))
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
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":4,"tp_size":4},"use_ascend_direct":true}}' \
    --port $PORT --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <D_IP> --data-parallel-rpc-port 10523 --data-parallel-external-lb &
done; wait -n
```
> D 另注入 common_env + 角色 `VLLM_ASCEND_ENABLE_MLAPO=1 TASK_QUEUE_ENABLE=1 VLLM_ASCEND_ENABLE_FUSED_MC2=1`。

---

## 5. ⭐ 外部 PD proxy（必做）
```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 18000 --host 0.0.0.0 \
  --prefiller-hosts <P_IP> <P_IP> <P_IP> <P_IP> \
  --prefiller-ports 17000 17001 17002 17003 \
  --decoder-hosts  <D_IP> <D_IP> <D_IP> <D_IP> \
  --decoder-ports  17000 17001 17002 17003
```
> prefiller = P 的 4 个 service；decoder = D 的 4 个 service。压测打 proxy 的 `:18000`。

---

## 6. K8s 探针
engine、wings-control 两容器加 startupProbe（`/health`，`failureThreshold≈120 periodSeconds=10`）。PD external-lb 下每 pod 走 standalone，health 都在 **19000**（无偏移）。

---

## 7. 验证（逐层）
```bash
# 1) external-lb 触发
kubectl logs <pod> -c wings-control | grep "PD external-lb"
#   期望 P: role=P dp_size=4 local=4 rank_start=0   D: role=D dp_size=4 local=4 rank_start=0
# 2) 进程数：P、D 各 4 个 vllm
kubectl exec <pod> -- bash -lc 'pgrep -af "vllm.*api_server" | wc -l'   # 期望 4
# 3) 注册表参数没被顶掉
kubectl exec <D-pod> -- bash -lc 'grep -oE "max-num-batched-tokens [0-9]+|max-num-seqs [0-9]+|gpu-memory-utilization [0-9.]+|max-model-len [0-9]+" /shared-volume/start_command.sh'
#   D 期望 32 / 8 / 0.92 / 200000（或你临时调小后的值）
# 4) ★ mooncake 真传 KV（两端都要有）
kubectl logs <P-pod> -c engine | grep -iE "kv_producer|mooncake|transfer"
kubectl logs <D-pod> -c engine | grep -iE "kv_consumer|mooncake|pull"
# 5) 冒烟推理
curl -s http://<proxy>:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.1","messages":[{"role":"user","content":"你是谁"}],"max_tokens":64}'
# 6) ★ 看有没有 OOM（本方案最大风险）
kubectl logs <D-pod> -c engine | grep -iE "out of memory|NPU out of|HBM|OOM"
```

---

## 8. 排障
| 现象 | 原因 | 处置 |
|---|---|---|
| **D（或 P）OOM** | EP16 每卡路由专家 ~45GB（官方 P 的 ~4×，见 §0）+ KV | 64GB/卡：1P1D 本就放不下，退 §11 单实例 EP32 或上 4 机版；≥96GB/卡：临时调小注册表 `max_model_len`/`max_num_seqs` |
| wings 日志无 `[PD external-lb]` | `DP_SIZE` 没读成 >1 / 镜像旧 | `printenv DP_SIZE PD_PREFILL_DP_SIZE`；确认 sidecar 含 external-lb |
| 命令变单进程 TP4、无 fork | 走了 standalone | 确认 `PD_ROLE`+`DP_SIZE>1`；别传 `--distributed` |
| 注册表值被改 | 平台灌了 tuning flag | `grep "Starting wings application" wings_start.log` + `printenv` 自查 |
| 预测通但无 mooncake 日志 | 没起 §5 proxy / 端口错 | 起 proxy，端口对齐 §2 |
| FULL_DECODE_ONLY 编译崩（AICPU `open so failed`） | 引擎/镜像 AICPU 问题 | 临时 enforce-eager；查 CANN/AICPU 包 |

---

## 9. 占位替换清单
| 占位 | 含义 | 你的值 |
|---|---|---|
| `<P_IP>` | P pod IP | 按实际 |
| `<D_IP>` | D pod IP | 按实际 |
| `<proxy_IP>` | 跑 proxy 的机器 | 按实际 |

---

## 10. 与 4 机/官方的差异
| | 官方 | 4 机版 | **本 2 机 1P1D** |
|---|---|---|---|
| P | DP2×TP16（2 节点，EP32） | 同官方 | **DP4×TP4（1 节点，EP16）** |
| D | DP16×TP4（4 节点，EP64） | DP8×TP4（2 节点，EP32） | **DP4×TP4（1 节点，EP16）** |
| 合计 | 6 节点/96 卡 | 4 节点/64 卡 | **2 节点/32 卡** |
| 每卡路由专家权重（w8a8） | P 22.7 / D 11.3 GB | P 22.7 / D 22.7 GB | **P/D 各 45.3 GB（最紧）** |
| 64GB/卡可行性 | ✅ | ✅ | **❌（权重就 ~50GB）；需 ≥96GB/卡才行** |
| 注册表命中 / 字段 | GlmMoeDsaForCausalLM，同一套 | 同 | **同**（按架构，与规模无关） |
| 验证覆盖 | 全 | 跨节点 DP + mooncake | external-lb fork + **mooncake** + 推理（**无跨节点 DP**） |

---

## 11. ⭐ 推荐（64GB/卡时唯一可行）：单实例 EP32 验模型，不走 PD
你说的目的是**验证 GLM-5.1 这个模型**（能加载、推理输出正常）。在 64GB/卡上这是**唯一放得下**的 32 卡跑法：**单实例非 PD，EP32**（每卡路由专家 ~22.7GB，= 官方 P 档位，宽松）。

拓扑选一种能整除 64 个注意力头的并行：
- **DP4×TP8**（每节点 16 卡=2 个 TP8 DP rank，DP 跨节点）—— TP 不跨机，最稳，推荐。
- **DP2×TP16**（每节点 1 个 TP16 rank，TP 跨节点）—— 等同官方 P 的并行。

```bash
# 不设 PD_ROLE / 不设 PD_*_DP_SIZE → 不进 PD external-lb，按普通 DP 分布式起 1 个 GLM5.1 实例
# 仍少传 tuning flag（让模型默认/通用注册表生效）；验证期可把 max-model-len 设小（如 8192）省 KV
# DP 跨节点的 IP/rank 下发沿用本仓库 DP 部署约定（NODE_IPS/RANK_IP/Master_IP）
```
> 注意：这条**不命中 PD 注册表**（无 `PD_ROLE`），GLM5 的 PD 专属调优（连接器/kv/parser via PD entry）不会下发——它只验"模型本身能否加载+推理"。要验 **PD 流程**，64GB/卡需上 [4 机版](deploy-glm5.1-pd-4node-2p2d.md)（P/D 各 EP32）。

---

## 一句话（结论按 HBM 分叉）
- **910C 64GB/卡**：1P1D **放不下**（权重就 ~50GB/卡）。**验模型 → §11 单实例 EP32**（这 2 机 32 卡当一个实例）；**验完整 PD 流程 → 上 4 机版**。
- **910C ≥96GB/卡（如 128GB SKU）**：1P1D **可行**，按 §1–§9 跑，首跑先按 §0.3 调小 `max_model_len`。
- 共性铁律：少传 tuning flag、`max_model_len` 改注册表不传 CLI、不传 `--distributed`、proxy 必做。
