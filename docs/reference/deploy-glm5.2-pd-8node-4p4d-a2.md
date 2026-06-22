# GLM-5.2 PD 分离部署（8× Atlas 800 A2 / 4P4D / external-lb / mooncake）

> 目标拓扑：**P 节点 4 个（各 8 卡）+ D 节点 4 个（各 8 卡）**，**强制 dp>1**，走
> vLLM-Ascend external-lb（DP fork）+ MooncakeConnectorV1。
>
> 并行：**P = DP4×TP8**（每 P pod fork 1 个 prefill、TP8）；**D = DP8×TP4**（每 D pod fork 2 个、各 TP4）。
> 全局拓扑 `prefill={dp:4,tp:8}` / `decode={dp:8,tp:4}`。合计 **8 pod / 64 卡 / 4P4D**。
>
> **环境变量驱动**：用户只下发 env，连接器 JSON / CLI 全部由 wings-control 自动推导（机制见 [pd-separation-unified.md](../features/pd-separation-unified.md)，external-lb fork 等同官方 `launch_online_dp.py`，**不改代码、直接复用**）。同模型 A3 形态见 [deploy-glm5.1-pd-a3.md](./deploy-glm5.1-pd-a3.md)；官方 [GLM5.2](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html)。**以下 IP / 网卡名 / 端口 / 权重路径均为占位符。**

---

## 0. 关键事实（先读）

1. **必须走 external-lb（dp>1）**：触发 = `PD_ROLE∈{P,D}` **且 `DP_SIZE>1`**（[config_loader.py:935/943](../../wings_control/core/config_loader.py#L935)）。dp=1 退回 standalone，不读 pd_config.json。external-lb 的 fork 循环就是官方 `launch_online_dp.py` 的等价物——本文「保持不变直接复用」即指此路径无需改码，只换 env。
2. **架构同 GLM-5.1**：GLM-5.2-w8a8 的 `architectures[0]` = `GlmMoeDsaForCausalLM`，命中 [pd_config.json](../../wings_control/config/defaults/pd_config.json) 同一条目。但该条目当前是 **GLM-5.1 / A3 口径**，5.2 / A2 有差量需覆盖（见 §0.4 与 §6），否则会注入错值。
3. **平台 = A2**：**给 a2 信号 / 不给 a3 信号**——`WINGS_ASCEND_PLATFORM=a2`（或镜像 `ENGINE_VERSION` 不带 `-a3`）。全无信号也回退 a2。⚠️ 但注册表 `common_env` 里**硬写了 `ASCEND_A3_ENABLE=1`**，A2 必须覆盖掉（§6 差量①），`HCCL_BUFFSIZE` 走 A2 平台值（项目 1024），不按 A3 注册表的 256 硬编码。
4. **GLM-5.2 相对 5.1 的差量（必须照做）**：
   - **前缀缓存两端都开**（注册表 P/D 当前都是 `enable_prefix_caching:false`；5.2 按用户口径要 true）。
   - **Decode 开 `DYNAMIC_EPLB=1`**（注册表 decode.env 当前无）。
   - **双机 `VLLM_ASCEND_BALANCE_SCHEDULING=0`**（单机才 1；本部署全多机，两端 0）。
   - **保留双机 `additional_config`**（5.1 在 A3 双机会剥除以避 aclgraph MTE 崩溃；5.2 已稳定，豁免剥除）。
   - **MTP `num_speculative_tokens=3`**（`method=deepseek_mtp`，与 5.1 注册表一致）。
5. **本机 IP 只设 `RANK_IP`**：`get_local_ip()` 读它，`HCCL_IF_IP` 与 rank_start 派生都回退到它。`RANK_IP` 须**逐字**在 `NODE_IPS` 内。
6. **`DP_SIZE`/`TP_SIZE` 可省**：本角色 dp/tp 由全局拓扑 `PD_{ROLE}_*` 派生（P→`PD_PREFILL_*`，D→`PD_DECODE_*`）。本文显式给以便核对。
7. **不要传** `--tensor-parallel-size` / `--max-model-len` 给 `wings_start.sh`（`parse_launch_args` 不认）。TP 由 `TP_SIZE` 决定，max-model-len 由注册表给。
8. **rpc-port 现按角色硬编码** `P=12890 / D=12777`（[最近提交](#)，刻意不读 `VLLM_LLMDD_RPC_PORT`），网络策略须放行这两个固定口。
9. **mooncake 不会引擎一起来就自动传 KV**：靠 §5 外部 PD proxy 在请求层把 P→D 串起来才触发。

---

## 1. 拓扑与全局并行

| | P 节点（4 pod，各 8 卡） | D 节点（4 pod，各 8 卡） |
|---|---|---|
| 角色 | `PD_ROLE=P` → kv_producer | `PD_ROLE=D` → kv_consumer |
| DP / TP | `DP_SIZE=4 TP_SIZE=8` | `DP_SIZE=8 TP_SIZE=4` |
| 本 pod fork 数 | `DP_SIZE_LOCAL=1`（8÷8） | `DP_SIZE_LOCAL=2`（8÷4） |
| rank | P-0=0 / P-1=1 / P-2=2 / P-3=3 | D-0=0,1 / D-1=2,3 / D-2=4,5 / D-3=6,7 |
| 本 pod 可见卡 | `0-7` | `0-3` / `4-7` |
| kv-config `prefill` / `decode` | `{dp:4,tp:8}` / `{dp:8,tp:4}` | 同（两端必须完全一致） |

> **拓扑一致性铁律**：P 与 D 最终 kv-config 的 `prefill{}`、`decode{}` 必须分别完全相同，否则 mooncake KV rank 映射两端不一致 → 握手失败（自查见 §7）。每 pod `dp-rank-start = role_node_rank × DP_SIZE_LOCAL`：P 步长 1（0,1,2,3）、D 步长 2（0,2,4,6），由 wings 自动派生。

---

## 2. 端口规划（proxy 按此表填）

| 节点/rank | 引擎 HTTP（`ENGINE_PORT+i`） | kv_port（`base+i`） | bootstrap | dp-rpc-port | 可见卡（`i*TP..`） |
|---|---|---|---|---|---|
| P-0..P-3（各 rank=节点序号） | `17000` | `30000` | `23000` | `12890` | `0-7` |
| D-0..D-3（各 2 rank） | `17000-17001` | `30100-30101` | `23100-23101` | `12777` | `0-3 / 4-7` |

> 端口算法（fork 脚本）：`PORT=ENGINE_PORT+i`、`kv_port=base+i`（P base 30000 / D base 30100）、`bootstrap=BOOTSTRAP+i`（P 23000 / D 23100）、`卡=[i*TP,(i+1)*TP)`。不同 pod 同端口不冲突（不同 IP）。`dp-rpc-port` 按角色硬编码（§0.8）。

---

## 3. P 节点 ×4（各 8 卡，DP4×TP8，kv_producer，FlashComm1 + DSA CP）

P-0..P-3 **只差 `RANK_IP`**（`Master_IP`/`NODE_IPS` 四端写法完全一致）。

### 3.1 环境变量（注入 wings-control 容器 env）

```bash
export PD_ROLE=P
export WINGS_DEVICE=ascend  WINGS_DEVICE_COUNT=8
export WINGS_ASCEND_PLATFORM=a2                 # A2：给 a2 信号 / 不给 a3
# —— external-lb fork 编排 ——
export DP_SIZE=4 TP_SIZE=8 DP_SIZE_LOCAL=1       # DP_SIZE/TP_SIZE 可省（由 PD_PREFILL_* 派生）
export Master_IP=<P0_IP>                         # = data-parallel-address，指向 P-0（rank0）
export NODE_IPS=<P0_IP>,<P1_IP>,<P2_IP>,<P3_IP>  # 顺序即 rank：P-0=0 … P-3=3
export RANK_IP=<本P_IP>                          # 须 ∈ NODE_IPS
export ENGINE_PORT=17000
export NETWORK_INTERFACE=<RDMA_NIC>              # RDMA 网卡名，非 eth0
# —— 全局拓扑（KV 映射；两端一致）——
export PD_PREFILL_DP_SIZE=4  PD_PREFILL_TP_SIZE=8
export PD_DECODE_DP_SIZE=8   PD_DECODE_TP_SIZE=4
# —— GLM-5.2 / A2 差量（覆盖注册表 5.1/A3 口径，见 §6）——
export VLLM_ASCEND_BALANCE_SCHEDULING=0          # 双机=0
# （前缀缓存 / 去 ASCEND_A3_ENABLE 经注册表 a2 overlay 落，见 §6；角色 env FlashComm1 由注册表给）
```

### 3.2 wings-control 启动命令

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name glm-5.2-chat \
  --model-path <WEIGHT_PATH> \
  --device-count 8 --port 18000 \
  --gpu-memory-utilization 0.95 --trust-remote-code --seed 1024
```

### 3.3 wings 生成的引擎命令（dry-run 核对；**不手动执行**）

```bash
for i in $(seq 0 0); do            # DP_SIZE_LOCAL=1 → fork 1
  RANK=$((<rank_start> + i)); PORT=$((17000 + i)); KVPORT=$((30000 + i)); BOOTSTRAP=$((23000 + i))
  LO=$((i * 8)); HI=$((LO + 7)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server \
    --model <WEIGHT_PATH> --served-model-name glm-5.2-chat \
    --max-model-len 131072 --quantization ascend --seed 1024 \
    --max-num-batched-tokens 4096 --max-num-seqs 64 --gpu-memory-utilization 0.95 \
    --enable-expert-parallel --enable-chunked-prefill --enable-prefix-caching --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"ascend_compilation_config":{"enable_npugraph_ex":true},"enable_dsa_cp":true}' \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}' \
    --port $PORT --tensor-parallel-size 8 --data-parallel-size 4 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <P0_IP> --data-parallel-rpc-port 12890 --data-parallel-external-lb \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":8},"decode":{"dp_size":8,"tp_size":4},"use_ascend_direct":true}}' &
done; wait -n
```
> `<rank_start>` = `RANK_IP` 在 `NODE_IPS` 的位置 ×`DP_SIZE_LOCAL`：P-0→0 … P-3→3（自动派生）。角色 env：P 加 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`+`VLLM_ASCEND_ENABLE_FUSED_MC2=1`。`--enable-prefix-caching` 为 5.2 差量（注册表当前关，需 §6 覆盖）。

---

## 4. D 节点 ×4（各 8 卡，组成 DP8，kv_consumer，MLAPO + DYNAMIC_EPLB + FULL_DECODE_ONLY）

D-0..D-3 **只差 `RANK_IP`**，每 pod fork 2 个 service。

### 4.1 环境变量

```bash
export PD_ROLE=D
export WINGS_DEVICE=ascend  WINGS_DEVICE_COUNT=8
export WINGS_ASCEND_PLATFORM=a2
export DP_SIZE=8 TP_SIZE=4 DP_SIZE_LOCAL=2        # DP_SIZE/TP_SIZE 可省（由 PD_DECODE_* 派生）
export Master_IP=<D0_IP>                          # 指向 D-0（rank0）
export NODE_IPS=<D0_IP>,<D1_IP>,<D2_IP>,<D3_IP>   # 顺序即 rank 段：D-0=0、D-1=2、D-2=4、D-3=6
export RANK_IP=<本D_IP>                           # 须 ∈ NODE_IPS
export ENGINE_PORT=17000
export NETWORK_INTERFACE=<RDMA_NIC>
export PD_PREFILL_DP_SIZE=4  PD_PREFILL_TP_SIZE=8
export PD_DECODE_DP_SIZE=8   PD_DECODE_TP_SIZE=4
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export DYNAMIC_EPLB=1                             # GLM-5.2 差量：动态专家负载均衡
```

### 4.2 wings-control 启动命令（D-0..D-3 相同）

```bash
bash /opt/wings-control/wings_start.sh \
  --engine vllm_ascend --model-name glm-5.2-chat \
  --model-path <WEIGHT_PATH> \
  --device-count 8 --port 18000 \
  --gpu-memory-utilization 0.92 --trust-remote-code --seed 1024
```

### 4.3 wings 生成的引擎命令（每 D pod fork 2；仅核对）

```bash
for i in $(seq 0 1); do            # DP_SIZE_LOCAL=2 → fork 2
  RANK=$((<rank_start> + i)); PORT=$((17000 + i)); KVPORT=$((30100 + i)); BOOTSTRAP=$((23100 + i))
  LO=$((i * 4)); HI=$((LO + 3)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server \
    --model <WEIGHT_PATH> --served-model-name glm-5.2-chat \
    --max-model-len 200000 --quantization ascend --seed 1024 \
    --max-num-batched-tokens 32 --max-num-seqs 8 --gpu-memory-utilization 0.92 \
    --enable-expert-parallel --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}' \
    --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}' \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}' \
    --port $PORT --tensor-parallel-size 4 --data-parallel-size 8 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address <D0_IP> --data-parallel-rpc-port 12777 --data-parallel-external-lb \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":8},"decode":{"dp_size":8,"tp_size":4},"use_ascend_direct":true}}' &
done; wait -n
```
> D 无 `--enforce-eager`（走 `FULL_DECODE_ONLY` 图）。角色 env：D 加 `VLLM_ASCEND_ENABLE_MLAPO=1`+`DYNAMIC_EPLB=1`+`TASK_QUEUE_ENABLE=1`+`VLLM_ASCEND_ENABLE_FUSED_MC2=1`。`<rank_start>`：D-0→0、D-1→2、D-2→4、D-3→6。
> ⚠️ **真机待核**：①GLM-5.1 在 A3 对 D **关**前缀缓存，5.2 这里**按用户口径开** + `FULL_DECODE_ONLY`——全图 decode replay 有 MTE 越界历史（[[glm5-aclgraph-mte-crash]]），首跑务必真机验证，必要时回退 `--enforce-eager`。②`DYNAMIC_EPLB` 若官方 5.2 还要求 `additional-config` 内 EPLB 旋钮（`dynamic_eplb`/`num_iterations_eplb_update`/`expert_map_path`），从官方 5.2 EPLB 段补齐。③D 的 batched/seqs/util/max-model-len 为 A3 起点，A2 按真机吞吐重调。

---

## 5. ⭐ 在 wings 外起 PD proxy（触发 mooncake，必做）

`--prefiller-*` 指 4 个 P 服务、`--decoder-*` 指 4 个 D pod 的 8 个 service（每 pod 2 端口 17000-17001）：

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 18000 --host 0.0.0.0 \
  --prefiller-hosts <P0_IP> <P1_IP> <P2_IP> <P3_IP> \
  --prefiller-ports 17000 17000 17000 17000 \
  --decoder-hosts  <D0_IP> <D0_IP>  <D1_IP> <D1_IP>  <D2_IP> <D2_IP>  <D3_IP> <D3_IP> \
  --decoder-ports  17000 17001  17000 17001  17000 17001  17000 17001
```
> 压测/curl 打 **proxy :18000**，不是引擎 17000。

---

## 6. ⚠️ 注册表差量：`GlmMoeDsaForCausalLM` 是 5.1/A3 口径，落 5.2/A2 需覆盖

external-lb 的 P/D 差异化参数全来自 [pd_config.json](../../wings_control/config/defaults/pd_config.json) 的 `GlmMoeDsaForCausalLM`，当前对齐 GLM-5.1 / A3。落本部署需补一个 `platform_overrides.a2`（或 5.2 专属覆盖）：

| # | 注册表现状（5.1/A3） | 本部署（5.2/A2）需改 |
|---|---|---|
| 1 | `prefill.enable_prefix_caching=false` / `decode.enable_prefix_caching=false` | 两端改 **true** |
| 2 | `decode.env` 无 EPLB | 加 **`DYNAMIC_EPLB=1`**（必要时配 `additional_config` EPLB 旋钮） |
| 3 | `common_env.ASCEND_A3_ENABLE=1`（硬写） | A2 **去掉**；`HCCL_BUFFSIZE` 走 A2 平台值 |
| 4 | 无 `VLLM_ASCEND_BALANCE_SCHEDULING` | 双机两端加 **`=0`** |
| 5 | DP/TP 不在注册表（env 决定） | `DP_SIZE/TP_SIZE`：P=`4/8`、D=`8/4`；`DP_SIZE_LOCAL`：P=`1`、D=`2` |

> 其余（MTP `num=3` / `enable_dsa_cp` / FlashComm1 / MLAPO / 双机保留 additional_config）现有条目已覆盖。差量 ①③④ 建议落注册表 `platform_overrides.a2`（一次改、所有 8 pod 生效）；②⑤ 可经 env（如上）下发。落地前对 ①②③ 做 dry-run 快照逐行核对。

---

## 7. 验证（mooncake 触发判据）

```bash
# 1) external-lb 触发（无此行 = dp>1 没生效）
kubectl logs <P-0-pod> -c wings-control | grep "PD external-lb"
#   期望: [PD external-lb] arch=GlmMoeDsaForCausalLM role=P connector=MooncakeConnectorV1 dp_size=4 local=1 rank_start=0
# 2) 各引擎 health：P 每 pod 1 端口(17000)，D 每 pod 2 端口(17000-17001)
curl -s http://<P0_IP>:17000/health ; curl -s http://<D0_IP>:1700{0..1}/health
# 3) 打 proxy 出预测
curl -s http://<proxy_IP>:18000/v1/chat/completions -H 'Content-type: application/json' \
  -d '{"model":"glm-5.2-chat","messages":[{"role":"user","content":"你是谁"}],"max_tokens":64,"temperature":0}'
# 4) ★ 两端都要有 mooncake 传输日志
kubectl logs <P-0-pod> -c engine | grep -iE "kv_producer|mooncake|transfer"
kubectl logs <D-0-pod> -c engine | grep -iE "kv_consumer|mooncake|pull"
# 5) 一致性自查：所有 pod 设 PD_PREFILL_DP_SIZE=4 PD_PREFILL_TP_SIZE=8 PD_DECODE_DP_SIZE=8 PD_DECODE_TP_SIZE=4
#    → 两端 kv-config prefill{4,8}+decode{8,4} 必然相同
```

---

## 8. 占位替换清单

| 占位 | 含义 | 约定 |
|---|---|---|
| `<P0_IP>`..`<P3_IP>` | 4 个 P pod IP（`NODE_IPS` 顺序即 rank）；`Master_IP=<P0_IP>` | — |
| `<D0_IP>`..`<D3_IP>` | 4 个 D pod IP；`Master_IP=<D0_IP>`，4 端写法一致 | — |
| `<RDMA_NIC>` | RDMA 网卡名（`NETWORK_INTERFACE`），非 `eth0` | — |
| `<WEIGHT_PATH>` | GLM-5.2 权重目录 | 如 `/usr/local/serving/models/` |
| `<proxy_IP>` | 跑 `load_balance_proxy_server_example.py` 的机器 | — |
| `<rank_start>` | 自动派生（`RANK_IP` 在 `NODE_IPS` 位置 ×`DP_SIZE_LOCAL`），不手填 | P:0-3 / D:0/2/4/6 |
| rpc-port | 按角色硬编码 | P=`12890` / D=`12777` |
