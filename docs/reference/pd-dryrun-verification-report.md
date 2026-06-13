# PD 分离 dry-run 验证测试报告（接收参数 → 启动参数 → 最终输出 → 问题）

| 项 | 内容 |
|----|------|
| 日期 | 2026-06-13（修复后复跑验证） |
| 命令 | `python dry_run.py --pd glm5` / `python dry_run.py --pd v4flash` |
| 产物 | `build/output/start_command_pd-<scenario>-<P\|D>_node<N>.sh` |
| 配置源 | `wings_control/config/defaults/pd_config.json`（已修复条目）+ 模型 base 默认 |
| 验证范围 | GLM-5（MooncakeConnectorV1）、DeepSeek-V4-Flash A3（MooncakeHybridConnector） |
| 结论 | 两场景 P/D 下发字段**逐项对齐官方**；`tests/pd_external_lb_verify.py` **45 PASS / 0 FAIL**；非 PD 部署字节级不变。本次**未改任何代码**。 |

> 数据流：上层契约（env）→ `_get_pd_external_lb_params` 解析拓扑 → `_apply_pd_external_lb` 注入注册表 → `_prepare_engine_config` 重申注册表（修复点）→ fork 脚本按 `dp_size_local` 展开 N 个 `vllm serve`。
> 注：dry-run 用 mock 模型目录（仅 config.json），`--model` 指向临时目录；真机为实际权重路径。

---

## 1. GLM-5（场景 `glm5`，P:DP2×TP16 / D:DP16×TP4）

### 1.1 Prefill（P-node0）

**A. 接收参数（上层契约 env）**

| 字段 | 值 |
|------|-----|
| PD_ROLE | P |
| DP_SIZE / TP_SIZE / DP_SIZE_LOCAL | 2 / 16 / 1 |
| Master_IP（=dp-address） | 7.0.0.1 |
| VLLM_LLMDD_RPC_PORT | 10521 |
| NODE_IPS / HOST_IP | 7.0.0.1,7.0.0.2 / 7.0.0.1 |
| PD_PREFILL_* / PD_DECODE_*（KV 全局拓扑） | 2×16 / 16×4 |
| DEVICE_COUNT / 平台 | 16（local×tp）/ a3 |

**B. 解析结果（日志实证）**
`[PD external-lb] arch=GlmMoeDsaForCausalLM role=P connector=MooncakeConnectorV1 dp_size=2 local=1 rank_start=0 addr=7.0.0.1`
→ fork **1** service：`RANK=0+i`，`PORT=18000+i`，`KVPORT=30000+i`，`CARDS=i*16..` → rank0 / 18000 / kv30000 / 卡 0-15。

**C. 启动参数（关键下发字段 vs 官方）**

| 字段 | 下发 | 官方 | |
|------|------|------|---|
| tp / dp | 16 / 2 | 16 / 2 | ✅ |
| max-model-len | **131072** | 131072 | ✅ |
| max-num-batched-tokens / seqs | 4096 / 64 | 4096 / 64 | ✅ |
| gpu-memory-utilization | 0.95 | 0.95 | ✅ |
| enforce-eager | **有** | 有 | ✅ |
| enable-prefix-caching | **无** | 无 | ✅ |
| compilation-config | **无** | 无 | ✅ |
| enable-chunked-prefill | 有 | 有 | ✅ |
| enable-auto-tool-choice / tool-call-parser / reasoning-parser | 有 / glm47 / glm45 | 同 | ✅ |
| additional-config | `{fuse_muls_add,multistream_overlap_shared_expert,recompute_scheduler_enable,ascend_compilation_config{enable_npugraph_ex},enable_dsa_cp,layer_sharding:[q_b_proj,o_proj]}` | 同 | ✅ |
| speculative-config | `{3,deepseek_mtp}` | 同 | ✅ |
| seed / quant / EP | 1024 / ascend / 有 | 同 | ✅ |
| kv-transfer | V1 / producer / kv_port 30000 / extra{P2×16,D16×4,use_ascend_direct} / engine_id=$RANK | 同（engine_id 见 §4） | ✅ |

**D. 最终输出**（`start_command_pd-glm5-P_node0.sh`，fork 模板）
```bash
for i in $(seq 0 0); do
  RANK=$((0+i)); PORT=$((18000+i)); KVPORT=$((30000+i)); BOOTSTRAP=$((23000+i))
  LO=$((i*16)); HI=$((LO+16-1)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
  python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 131072 \
    --quantization ascend --seed 1024 --max-num-seqs 64 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 --enable-chunked-prefill \
    --additional-config '{...recompute_scheduler_enable...enable_dsa_cp...layer_sharding...}' \
    --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}' --enforce-eager \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"'"$KVPORT"'",...,"engine_id":"'"$RANK"'"}' \
    --port $PORT --tensor-parallel-size 16 --data-parallel-size 2 --data-parallel-rank $RANK \
    --data-parallel-size-local 1 --data-parallel-address 7.0.0.1 --data-parallel-rpc-port 10521 \
    --data-parallel-external-lb & ...
```

### 1.2 Decode（D-node0 / D-node1）

**A. 接收参数**：PD_ROLE=D，DP_SIZE/TP/LOCAL=16/4/4，Master_IP=7.0.1.1，rpc=10523，NODE_IPS=7.0.1.1..7.0.1.4，HOST_IP=node0=7.0.1.1 / node1=7.0.1.2，DEVICE_COUNT=16。

**B. 解析结果**：
- D-node0：`dp_size=16 local=4 rank_start=0 addr=7.0.1.1` → fork **4**（rank0-3，port 18000-18003，kv 30100-30103，卡 0-3/4-7/8-11/12-15）。
- D-node1：`rank_start=4`（`RANK=4+i`，HCCL_IF_IP=7.0.1.2）→ fork **4**（rank4-7）。**`rank_start` 由 HOST_IP 在 NODE_IPS 的位置×local 派生，验证通过**。

**C. 启动参数（vs 官方）**

| 字段 | 下发 | 官方 | |
|------|------|------|---|
| tp / dp | 4 / 16 | 4 / 16 | ✅ |
| max-model-len | 200000 | 200000 | ✅ |
| batched / seqs / gpu-mem | 32 / 8 / 0.92 | 32 / 8 / 0.92 | ✅ |
| compilation-config | `{FULL_DECODE_ONLY, cudagraph_capture_sizes:[4,8,12,16,20,24,28,32]}` | 同 | ✅ |
| enable-prefix-caching / chunked-prefill | 无 / 无 | 无 / 无 | ✅ |
| additional-config | `{...recompute_scheduler_enable...}` | 同 | ✅ |
| tool/reasoning parser | glm47 / glm45 | 同 | ✅ |
| kv-transfer | V1 / consumer / kv_port 30100 | 同 | ✅ |

---

## 2. DeepSeek-V4-Flash（场景 `v4flash`，A3 1P1D，P:DP4×TP4 / D:DP16×TP1）

### 2.1 Prefill（P-node0）

**A. 接收参数**：PD_ROLE=P，DP_SIZE/TP/LOCAL=4/4/4，Master_IP=8.0.0.1，rpc=10521，NODE_IPS/HOST_IP=8.0.0.1，PD_PREFILL=4×4 / PD_DECODE=16×1，DEVICE_COUNT=16，平台 a3。

**B. 解析结果**：`[PD external-lb] arch=DeepseekV4ForCausalLM role=P connector=MooncakeHybridConnector dp_size=4 local=4 rank_start=0 addr=8.0.0.1` → fork **4**（rank0-3，port 18000-18003，kv 30000-30003，卡 i*4：0-3/4-7/8-11/12-15）。

**C. 启动参数（vs 官方 A3）**

| 字段 | 下发 | 官方(A3) | |
|------|------|---------|---|
| tp / dp | 4 / 4 | 4 / 4 | ✅ |
| max-num-batched-tokens / seqs | **8192 / 16** | 8192 / 16 | ✅ |
| gpu-memory-utilization | **0.9** | 0.9 | ✅ |
| max-model-len | **1048576** | 1048576 | ✅ |
| seed | **1024** | 1024 | ✅ |
| enforce-eager | **有** | 有 | ✅ |
| async-scheduling | **无** | 无 | ✅ |
| compilation-config | **无** | 无 | ✅ |
| no-enable-prefix-caching | **有** | 有 | ✅ |
| enable-chunked-prefill | **无** | 无 | ✅ |
| additional-config | `{enable_cpu_binding,enable_shared_expert_dp,enable_dsa_cp}` | 同 | ✅ |
| speculative-config | `{1,mtp,enforce_eager:true}` | 同 | ✅ |
| reasoning-parser | deepseek_v4 | deepseek_v4 | ✅ |
| model-loader-extra-config | `{enable_multithread_load,num_threads:128}` | 同 | ✅ |
| no-disable-hybrid-kv-cache-manager | **有** | 有 | ✅ |
| tokenizer-mode / tool-call-parser / block-size | deepseek_v4 / deepseek_v4 / 128 | 同 | ✅ |
| kv-transfer | Hybrid / producer / kv_port 30000 / extra{P4×4,D16×1} / engine_id=$RANK | 同（engine_id 见 §4） | ✅* |

**D. 最终输出**（`start_command_pd-v4flash-P_node0.sh`，fork 模板）
```bash
for i in $(seq 0 3); do
  RANK=$((0+i)); PORT=$((18000+i)); KVPORT=$((30000+i)); BOOTSTRAP=$((23000+i))
  LO=$((i*4)); HI=$((LO+4-1)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP vllm serve <model> \
    --trust-remote-code --max-model-len 1048576 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 \
    --max-num-seqs 16 --enable-expert-parallel --quantization ascend --block-size 128 \
    --safetensors-load-strategy prefetch \
    --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_dsa_cp":true}' \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --seed 1024 \
    --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer",...,"engine_id":"'"$RANK"'"}' \
    --no-enable-prefix-caching --reasoning-parser deepseek_v4 --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}' \
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' --enforce-eager \
    --api-server-count 1 --port $PORT --tensor-parallel-size 4 --data-parallel-size 4 \
    --data-parallel-rank $RANK --data-parallel-size-local 1 --data-parallel-address 8.0.0.1 \
    --data-parallel-rpc-port 10521 --data-parallel-external-lb & ...
```

### 2.2 Decode（D-node0）

**A. 接收参数**：PD_ROLE=D，DP_SIZE/TP/LOCAL=16/1/16，Master_IP=8.0.1.1，rpc=10523，NODE_IPS/HOST_IP=8.0.1.1，DEVICE_COUNT=16。

**B. 解析结果**：`dp_size=16 local=16 rank_start=0 addr=8.0.1.1` → fork **16**（rank0-15，port 18000-18015，kv 30100-30115，卡 i*1：每 service 1 卡）。

**C. 启动参数（vs 官方 A3）**

| 字段 | 下发 | 官方(A3) | |
|------|------|---------|---|
| tp / dp | 1 / 16 | 1 / 16 | ✅ |
| batched / seqs / gpu-mem | **120 / 60 / 0.9** | 120 / 60 / 0.9 | ✅ |
| max-model-len / seed | 1048576 / 1024 | 同 | ✅ |
| async-scheduling | 有 | 有 | ✅ |
| compilation-config | `{FULL_DECODE_ONLY}` | 同 | ✅ |
| no-enable-prefix-caching / chunked | 有 / 无 | 有 / 无 | ✅ |
| additional-config | `{ascend_compilation_config{enable_npugraph_ex,enable_static_kernel:false},enable_cpu_binding,multistream_overlap_shared_expert:true,recompute_scheduler_enable}` | 同 | ✅ |
| speculative-config | `{1,mtp,enforce_eager:true}` | 同 | ✅ |
| no-disable-hybrid-kv / model-loader-extra-config / reasoning-parser | 有 / 有 / deepseek_v4 | 同 | ✅ |
| kv-transfer | Hybrid / consumer / kv_port 30100 | 同 | ✅* |

---

## 3. 端口 / 卡组 / rank 自洽性验证（全场景）

| 场景-角色 | fork 数 | rank | port | kv_port | bootstrap | 卡组 |
|-----------|:-----:|------|------|---------|-----------|------|
| glm5-P | 1 | 0 | 18000 | 30000 | 23000 | 0-15 |
| glm5-D node0 | 4 | 0-3 | 18000-18003 | 30100-30103 | 23100-23103 | 0-3/4-7/8-11/12-15 |
| glm5-D node1 | 4 | **4-7** | 18000-18003 | 30100-30103 | 23100-23103 | 同上（每节点本地切卡） |
| v4flash-P | 4 | 0-3 | 18000-18003 | 30000-30003 | 23000-23003 | 0-3/4-7/8-11/12-15 |
| v4flash-D | 16 | 0-15 | 18000-18015 | 30100-30115 | 23100-23115 | 每 service 1 卡 |

- P/D rpc 端口分离（P=10521 / D=10523）✅；kv_port producer/consumer 错开（300xx/301xx）✅；端口块（18xxx）与 health/monitor 不撞 ✅。
- `dp_rank_start` 由 HOST_IP 位置派生（D-node1=4）✅，与上层无需显式下发 rank-start 一致。

---

## 4. 问题清单

| # | 级别 | 问题 | 说明 / 建议 |
|---|------|------|------------|
| 1 | **待真机** | **engine_id（V4-Flash Hybrid）** | wings 对 V1 与 Hybrid 一律按 `dp_rank` 注入 `engine_id=$RANK`；官方 V4-Flash(Hybrid) 示例为固定 `0/1`。多 service 下按 rank 更合理（避免同 pod 冲突），但 Mooncake Hybrid 是否要求 role 级常量需真机确认。表中标 ✅* 即此项。属代码路径（`_build_pd_external_lb_kv`），本次未改。 |
| 2 | P2 | **env 段冗余/抖动** | 生成脚本 env 段有重复 export 与抖动（`OMP_NUM_THREADS` 1→100→1、`HCCL_BUFFSIZE` 多次覆盖）。**最终生效值以最后一次为准**，功能正确但可读性差。 |
| 3 | P2 | **GLM5 共用 env 与官方有差** | `HCCL_BUFFSIZE=1024`（官方 GLM5=256）；缺官方 `ASCEND_AGGREGATE_ENABLE/ACL_OP_INIT_MODE/ASCEND_A3_ENABLE/VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`。属 base env 注入层（非 pd_config），需确认注入位置后再调。 |
| 4 | 提示 | **Qwen3.5 未经 dry-run 验证** | `dry_run.py` 无 Qwen3.5 PD 场景，其注册表补齐为静态修改，未生成命令验证。 |
| 5 | 非问题 | wings 基础设施字段 | `--dtype auto`、`--kv-cache-dtype auto`、`--block-size 16`(GLM5)、`--chat-template`、`--default-chat-template-kwargs`(GLM5 关思考)、`--api-server-count 1`(V4) 为 wings 约定/平台默认，官方命令未列，属预期差异，非缺陷。 |

---

## 5. 总体结论

1. **下发字段对齐**：GLM-5、V4-Flash 的 P/D 引擎下发字段（拓扑/批量/显存/seed/max-model-len/compilation/prefix/parser/additional-config/speculative/kv-transfer）**逐项与官方一致**（§1、§2 表）。
2. **拓扑/端口/卡组/rank** 自洽（§3），`dp_rank_start` 由 HOST_IP 派生正确。
3. **自动化回归**：`python tests/pd_external_lb_verify.py` → **45 PASS / 0 FAIL**（含「非 PD → 不触发 external-lb」回归、V3.2/default 条目）。
4. **非 PD 不受影响**：standalone V4-Flash 仍为 base 行为（`max-model-len 1024000`/`async`/`prefix` 不变），证明修复严格收窄于 PD external-lb。
5. **唯一待真机项**：engine_id（Hybrid）。其余问题均为 P2 可读性/env 层，不影响字段对齐。

> 复现：`python dry_run.py --pd glm5 && python dry_run.py --pd v4flash`，产物在 `build/output/`；字段修复明细见 [pd-dryrun-vs-official-report.md](pd-dryrun-vs-official-report.md) §0。
