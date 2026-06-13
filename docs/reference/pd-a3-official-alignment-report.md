# PD 分离 A3 场景 dry-run 验证报告 —— 对齐官方（GLM-5.1 / DeepSeek-V4-Flash）

| 项 | 内容 |
|----|------|
| 日期 | 2026-06-13 |
| 模型/平台 | GLM-5.1（`GlmMoeDsaForCausalLM`）、DeepSeek-V4-Flash（`DeepseekV4ForCausalLM`）；**均 A3（910C）** |
| 命令 | `python dry_run.py --pd glm5` / `python dry_run.py --pd v4flash` |
| 官方基准 | [GLM5](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.html) / [DeepSeek-V4-Flash](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html)（A3 1P1D） |
| 结论 | 两模型 P/D 下发字段**逐项对齐官方 A3** ✅；差异仅 wings 基础设施字段（`--dtype`/`--block-size`/`--chat-template` 等，非偏离）与 1 项真机待确认（`engine_id`）。 |
| 操作指南 | 真实部署/验证 runbook 见 [pd-a3-deploy-verify-guide.md](pd-a3-deploy-verify-guide.md)（本报告为其字段级证据附录） |

---

## 0. 输入分类（CLI vs 环境变量）与精简

dry-run = **模拟「上层编排① + 平台② + 用户④」对 wings 的一次调用**。wings 入参有**两条通道且互为兜底**——`build_parser` 给每个 `--arg` 都配了 `_env(同名大写)` 默认（[start_args_compat.py:259-315](../wings_control/core/start_args_compat.py)）：传了 CLI 用 CLI，否则取同名 env。

### 0.1 CLI 入参（dry-run 显式传 6 个 → 同名 env 冗余）

| CLI | 值（V4-P / GLM5-P） | env 兜底名 | 备注 |
|-----|------|------|------|
| `--model-name` | DeepSeek-V4-Flash / glm-5.1-chat | `MODEL_NAME` | — |
| `--model-path` | <权重路径> | `MODEL_PATH` | env 未被直接读取 |
| `--engine` | vllm_ascend | `ENGINE` | parse 后 `os.environ["ENGINE"]=` 同步回写 |
| `--device-count` | 16 | `DEVICE_COUNT` | 见 §0.4 |
| `--nnodes` | 1 | `NNODES` | — |
| `--node-rank` | 0 | （无 env，Master 注入） | — |

### 0.2 必填环境变量（**12 个**：无默认 / 默认不对 / 无 CLI 等价）

| 组 | 变量 | 来源 |
|----|------|:--:|
| PD 契约（无 CLI，必须 env） | `PD_ROLE` `DP_SIZE_LOCAL` `Master_IP` `NODE_IPS` `RANK_IP` | ① |
| KV 全局拓扑（两边一致，**本角色 dp/tp 也由此派生**） | `PD_PREFILL_DP_SIZE` `PD_PREFILL_TP_SIZE` `PD_DECODE_DP_SIZE` `PD_DECODE_TP_SIZE` | ① |
| 平台/硬件 | `WINGS_DEVICE`(缺省 nvidia→必填 ascend) `WINGS_ASCEND_PLATFORM`(⚠️**缺省回退 a2**) `DEVICE_COUNT`(缺省 1) | ② |

> `Master_IP`/`NODE_IPS` 虽有 CLI 双胞胎（`--master-ip`/`--node-ips`），但 PD 路径经 `get_master_ip()`/`get_node_ips()` **只读 env**，故归 env。
> ⚠️ **`RANK_IP` 是本机 IP 的唯一真相源**：`get_local_ip()` 读它（[env_utils.py:65](../wings_control/utils/env_utils.py)）；`current_ip`(→HCCL_IF_IP，`os.getenv("POD_IP", get_local_ip())`)、PD 的 `host_ip`(→rank_start，`_first_env("HOST_IP","RANK_IP") or get_local_ip()`) **均回退到它**。故**只设 `RANK_IP`，不另设 `POD_IP`/`HOST_IP`**（重复）；`RANK_IP` 须精确匹配 `NODE_IPS` 中一项。
> ⚠️ **A3 平台信号（任一即可，全无则兜底 `a2`，非 a3！）**：`WINGS_ASCEND_PLATFORM=a3` / `ASCEND_PLATFORM` / `ENGINE_IMAGE_FLAVOR` / **`ENGINE_VERSION` 带 `-a3` 后缀**（如 `0.13.0rc3-a3`，a3 镜像构建版本号通常自带 → **常可省显式 `WINGS_ASCEND_PLATFORM`**）/ `ASCEND_A3_ENABLE=1` / `hardware_info.json` 含 910c / `WINGS_DEVICE_NAME`。源码：`_get_engine_config_platform`（[vllm_adapter.py:1236](../wings_control/engines/vllm_adapter.py)）与 `_resolve_ascend_platform`（[config_loader.py:1033](../wings_control/core/config_loader.py)，本轮已对齐 `ENGINE_VERSION`/`-a3`）。探针证：无信号 → HCCL_BUFFSIZE 1024(a3)→512(a2)，命令偏离。

### 0.2.1 派生 / 可选环境变量（**可省略**，源码核实）

| 变量 | 缺省时取 | 依据 |
|------|---------|------|
| `DP_SIZE` / `TP_SIZE` | **派生自本角色全局拓扑 `PD_{ROLE}_*`**（P→`PD_PREFILL_*`，D→`PD_DECODE_*`） | `_first_env("DP_SIZE","PD_DP_SIZE",f"PD_{role}_DP_SIZE")`（[config_loader.py:935](../wings_control/core/config_loader.py)）；探针证派生 == 显式。**`DP_SIZE_LOCAL` 不可派生（=卡/节点÷tp），仍必填** |
| `VLLM_LLMDD_RPC_PORT` | **P=`12890` / D=`12777`**（按角色 fork 脚本兜底） | `rpc = pd_ext.rpc_port or ("12890" if P else "12777")`（[vllm_adapter.py:2896](../wings_control/engines/vllm_adapter.py)）；探针删它 → rpc=12890 |
| `SHARED_VOLUME_PATH` | `/shared-volume` | `os.getenv("SHARED_VOLUME_PATH","/shared-volume")`（[vllm_adapter.py:464](../wings_control/engines/vllm_adapter.py)），仅 LMCache 用；探针删它 → 命令字节级不变 |

> **PD 拓扑单一真相源 = 4 个全局变量**（`PD_PREFILL_DP_SIZE/TP_SIZE` + `PD_DECODE_DP_SIZE/TP_SIZE`），P/D 互相感知对方；本角色 `DP_SIZE`/`TP_SIZE` 由 `PD_{ROLE}_*` 派生，无需重复下发。

### 0.3 已移除的冗余/默认 env（探针验证「生成命令字节级不变」）

- **CLI 承载 / argparse 默认（9）**：`ENGINE`·`MODEL_NAME`·`MODEL_PATH`·`MODEL_TYPE`·`NNODES`·`NODE_RANK`（CLI 承载）、`DISTRIBUTED`(=false 默认)、`PORT`/`ENGINE_PORT`(=18000 默认)。
- **本机 IP 合并为 `RANK_IP`（2）**：`POD_IP`/`HOST_IP` 回退到它（§0.2 注）。
- **派生/有默认（4）**：`DP_SIZE`/`TP_SIZE`（派生自 `PD_{ROLE}_*`）、`SHARED_VOLUME_PATH`（默认 /shared-volume）；`VLLM_LLMDD_RPC_PORT` 给例值。
**验证**：固定 mock 目录，对比【全量 env】vs【精简 env】生成的 `vllm serve` 行 → **完全一致**；已落地到 `dry_run.run_pd_dry_run._one`（精简后注入 **13 个** env：12 必填 + `VLLM_LLMDD_RPC_PORT`(给例值，可省)）。

### 0.4 DEVICE_COUNT 的准确性说明（**双消费者**，纠正"env 必然冗余"的直觉）

`--device-count`（→`launch_args.device_count`，喂 launcher 的并行/device 规划）与 `DEVICE_COUNT` env（→`hardware_detect`/`device_utils` 的**硬件 count**，[hardware_detect.py:194](../wings_control/core/hardware_detect.py)）是**两个不同消费者**：
- 探针证实：**仅就生成的 PD 命令而言**，`--device-count` 已承载，删 `DEVICE_COUNT` env 命令不变（sidecar 模式 VRAM 校验被跳过）；
- 但 `DEVICE_COUNT` env 仍是 `hardware_detect` 硬件 count 的来源（真机 VRAM 校验/日志），且 `--device-count` 的默认值**本就取自 `DEVICE_COUNT`**。故单一真相源宜为 `DEVICE_COUNT` env（保留之）；真正可省的反而是重复的 `--device-count` CLI（此处保留以贴合真机显式传参）。

**来源图例**：`①上层下发` · `②平台/K8s` · `④用户/模型`；`⑤权重 config.json`（决定注册表命中）见各模型 §A。

---

## 1. DeepSeek-V4-Flash（A3，P:DP4×TP4 / D:DP16×TP1）

### 1.A 全部输入（dry-run 模拟的「用户/上层」调用）

**P（Prefill，node0）与 D（Decode，node0）共用的输入**（除标注外 P/D 相同）：

| 输入（env / CLI / config） | P 值 | D 值 | 来源 | 真机由谁下发 |
|---|---|---|:--:|---|
| `PD_ROLE` | `P` | `D` | ① | 编排层 |
| `DP_SIZE` / `TP_SIZE`（**派生，不下发**） | `4`/`4` | `16`/`1` | — | =本角色 `PD_{ROLE}_*`（见下 KV 拓扑行） |
| `DP_SIZE_LOCAL`（本节点 fork 数，**不可派生**） | `4` | `16` | ① | 编排层（=卡/节点÷tp） |
| `Master_IP`（=`--data-parallel-address`） | `8.0.0.1` | `8.0.1.1` | ① | 编排层（角色域 node0 IP） |
| `VLLM_LLMDD_RPC_PORT`（DP RPC 死值） | `10521` | `10523` | ① | 编排层（P/D 各一常量） |
| `NODE_IPS`（角色域全部节点 IP） | `8.0.0.1` | `8.0.1.1` | ① | 编排层 |
| `RANK_IP`（本 pod 唯一 IP；`POD_IP`/`HOST_IP` 回退到它，派生 rank_start，**须在 NODE_IPS 内**） | `8.0.0.1` | `8.0.1.1` | ① | 上层（MaaS） |
| `PD_PREFILL_DP_SIZE`/`PD_PREFILL_TP_SIZE` | `4`/`4` | `4`/`4` | ① | 编排层（KV 全局拓扑，两边一致） |
| `PD_DECODE_DP_SIZE`/`PD_DECODE_TP_SIZE` | `16`/`1` | `16`/`1` | ① | 编排层（同上） |
| `WINGS_ASCEND_PLATFORM` | `a3` | `a3` | ② | 平台/镜像（⚠️缺省回退 a2） |
| `WINGS_DEVICE` | `ascend` | `ascend` | ② | 平台 |
| `DEVICE_COUNT`（整 pod 卡数，=local×tp） | `16` | `16` | ② | 平台（硬件探测/下发） |
| **CLI 入参（6 项，见 §0.1）** | `--model-name --model-path --engine vllm_ascend --device-count 16 --nnodes 1 --node-rank 0` | 同 | ③④ | 同名 env 已移除（冗余） |
| **已移除冗余 env（9，见 §0.3）** | ENGINE/MODEL_NAME/MODEL_PATH/MODEL_TYPE/DISTRIBUTED/NNODES/NODE_RANK/PORT/ENGINE_PORT | — | — | 探针证生成命令不变 |
| **权重 `config.json`（⑤）** | `architectures=["DeepseekV4ForCausalLM"]`, `model_type="deepseek_v4"`, `quantization_config={"quant_method":"ascend"}` | 同 | ⑤ | `architectures` 决定命中注册表条目 |

> **真机区别**：dry-run 用 mock 目录（仅 `config.json`）；真机 `MODEL_PATH` 指向实际权重，`DEVICE_COUNT`/`RANK_IP`/`NODE_IPS` 由平台/编排注入，`ASCEND_RT_VISIBLE_DEVICES`（整 pod 卡）由平台映射。

### 1.B 解析结果（wings 识别，不算拓扑）

| 角色 | 日志实证 | fork 计划 |
|------|---------|----------|
| **P** | `connector=MooncakeHybridConnector dp_size=4 local=4 rank_start=0 addr=8.0.0.1` | fork **4**：rank 0-3 / port 18000-18003 / kv_port 30000-30003 / bootstrap 23000-23003 / 卡组 0-3·4-7·8-11·12-15（`i*tp`，tp=4） |
| **D** | `connector=MooncakeHybridConnector dp_size=16 local=16 rank_start=0 addr=8.0.1.1` | fork **16**：rank 0-15 / port 18000-18015 / kv_port 30100-30115 / bootstrap 23100-23115 / 卡组 每 service 1 卡（tp=1） |

### 1.C 启动参数 vs 官方 A3（逐项）

| 字段 | P 下发 / 官方 | D 下发 / 官方 | |
|------|---------------|---------------|:--:|
| tp / dp | 4 / 4 ＝ 4 / 4 | 1 / 16 ＝ 1 / 16 | ✅ |
| max-num-batched-tokens | 8192 ＝ 8192 | 120 ＝ 120 | ✅ |
| max-num-seqs | 16 ＝ 16 | 60 ＝ 60 | ✅ |
| gpu-memory-utilization | 0.9 ＝ 0.9 | 0.9 ＝ 0.9 | ✅ |
| max-model-len | 1048576 ＝ 1048576 | 1048576 ＝ 1048576 | ✅ |
| seed | 1024 ＝ 1024 | 1024 ＝ 1024 | ✅ |
| enforce-eager | 有 ＝ 有 | 无 ＝ 无 | ✅ |
| async-scheduling | 无 ＝ 无 | 有 ＝ 有 | ✅ |
| compilation-config | 无 ＝ 无 | `{FULL_DECODE_ONLY}` ＝ 同 | ✅ |
| no-enable-prefix-caching | 有 ＝ 有 | 有 ＝ 有 | ✅ |
| block-size | 128 ＝ 128 | 128 ＝ 128 | ✅ |
| tokenizer-mode / tool-call-parser / reasoning-parser | deepseek_v4 ×3 ＝ 同 | 同 | ✅ |
| safetensors-load-strategy | prefetch ＝ prefetch | 同 | ✅ |
| model-loader-extra-config | `{multithread,128}` ＝ 同 | 同 | ✅ |
| no-disable-hybrid-kv-cache-manager | 有 ＝ 有 | 有 ＝ 有 | ✅ |
| additional-config | `{cpu_binding,shared_expert_dp,dsa_cp}` ＝ 同 | `{ascend_compilation{npugraph,static_kernel:false},cpu_binding,multistream_overlap_shared_expert:true,recompute}` ＝ 同 | ✅ |
| speculative-config | `{1,mtp,enforce_eager}` ＝ 同 | 同 | ✅ |
| kv-transfer | Hybrid/producer/30000/extra{P4×4,D16×1} ＝ 同 | Hybrid/consumer/30100 ＝ 同 | ✅ |

### 1.D 最终输出（fork 模板，`start_command_pd-v4flash-P_node0.sh`）

```bash
for i in $(seq 0 3); do
  RANK=$((0+i)); PORT=$((18000+i)); KVPORT=$((30000+i)); BOOTSTRAP=$((23000+i))
  LO=$((i*4)); HI=$((LO+4-1)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP vllm serve <model> \
    --max-model-len 1048576 --max-num-batched-tokens 8192 --max-num-seqs 16 --gpu-memory-utilization 0.9 \
    --enable-expert-parallel --quantization ascend --block-size 128 --safetensors-load-strategy prefetch \
    --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_dsa_cp":true}' \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --seed 1024 \
    --no-enable-prefix-caching --reasoning-parser deepseek_v4 --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}' \
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' --enforce-eager \
    --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"'"$KVPORT"'","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}},"engine_id":"'"$RANK"'"}' \
    --api-server-count 1 --port $PORT --tensor-parallel-size 4 --data-parallel-size 4 \
    --data-parallel-rank $RANK --data-parallel-size-local 1 --data-parallel-address 8.0.0.1 \
    --data-parallel-rpc-port 10521 --data-parallel-external-lb & ...
done; wait -n || true; kill "${pids[@]}"; exit 1
```

---

## 2. GLM-5.1（A3，P:DP2×TP16 / D:DP16×TP4）

### 2.A 全部输入

| 输入 | P 值 | D 值（node0 / node1） | 来源 | 真机由谁下发 |
|---|---|---|:--:|---|
| `PD_ROLE` | `P` | `D` | ① | 编排层 |
| `DP_SIZE` / `TP_SIZE`（**派生，不下发**） | `2`/`16` | `16`/`4` | — | =本角色 `PD_{ROLE}_*` |
| `DP_SIZE_LOCAL`（**不可派生**） | `1` | `4` | ① | 编排层 |
| `Master_IP`（dp-address） | `7.0.0.1` | `7.0.1.1` | ① | 编排层（角色域 node0） |
| `VLLM_LLMDD_RPC_PORT` | `10521` | `10523` | ① | 编排层 |
| `NODE_IPS` | `7.0.0.1,7.0.0.2` | `7.0.1.1,7.0.1.2,7.0.1.3,7.0.1.4` | ① | 编排层 |
| `RANK_IP`（本 pod 唯一 IP；`POD_IP`/`HOST_IP` 回退到它，派生 rank_start，**须在 NODE_IPS 内**） | `7.0.0.1` | `7.0.1.1`（node0）/ `7.0.1.2`（node1） | ① | 上层（MaaS） |
| `PD_PREFILL_*` / `PD_DECODE_*` | `2`/`16` · `16`/`4` | 同 | ① | 编排层（KV 全局拓扑） |
| `WINGS_ASCEND_PLATFORM` / `WINGS_DEVICE` | `a3` / `ascend` | 同 | ② | 平台（⚠️platform 缺省回退 a2） |
| `DEVICE_COUNT`（=local×tp） | `16` | `16` | ② | 平台 |
| **CLI 入参（6 项，见 §0.1）** | `--model-name glm-5.1-chat --model-path --engine vllm_ascend --device-count 16 --nnodes 1 --node-rank 0` | 同 | ③④ | 同名 env 已移除（冗余） |
| **已移除冗余 env（9，见 §0.3）** | ENGINE/MODEL_NAME/MODEL_PATH/MODEL_TYPE/DISTRIBUTED/NNODES/NODE_RANK/PORT/ENGINE_PORT | — | — | 探针证生成命令不变 |
| **权重 `config.json`（⑤）** | `architectures=["GlmMoeDsaForCausalLM"]`, `model_type="glm4"`, `quantization_config={"quant_method":"ascend"}` | 同 | ⑤ | 命中 `GlmMoeDsaForCausalLM` 条目 |

### 2.B 解析结果

| 角色 | 日志实证 | fork 计划 |
|------|---------|----------|
| **P** | `connector=MooncakeConnectorV1 dp_size=2 local=1 rank_start=0 addr=7.0.0.1` | fork **1**：rank 0 / port 18000 / kv_port 30000 / 卡组 0-15（tp=16） |
| **D node0** | `dp_size=16 local=4 rank_start=0 addr=7.0.1.1` | fork **4**：rank 0-3 / port 18000-18003 / kv_port 30100-30103 / 卡组 0-3·4-7·8-11·12-15（tp=4） |
| **D node1** | `dp_size=16 local=4 rank_start=4 addr=7.0.1.1` | fork **4**：rank **4-7**（`RANK=4+i`，由 `RANK_IP`=7.0.1.2 在 NODE_IPS 的位置×local 派生）✅ |

### 2.C 启动参数 vs 官方 A3（逐项）

| 字段 | P 下发 / 官方 | D 下发 / 官方 | |
|------|---------------|---------------|:--:|
| tp / dp | 16 / 2 ＝ 16 / 2 | 4 / 16 ＝ 4 / 16 | ✅ |
| max-model-len | 131072 ＝ 131072 | 200000 ＝ 200000 | ✅ |
| max-num-batched-tokens / seqs | 4096 / 64 ＝ 同 | 32 / 8 ＝ 同 | ✅ |
| gpu-memory-utilization | 0.95 ＝ 0.95 | 0.92 ＝ 0.92 | ✅ |
| enforce-eager | 有 ＝ 有 | 无 ＝ 无 | ✅ |
| enable-chunked-prefill | 有 ＝ 有 | 无 ＝ 无 | ✅ |
| enable-prefix-caching | 无 ＝ 无 | 无 ＝ 无 | ✅ |
| compilation-config | 无 ＝ 无 | `{FULL_DECODE_ONLY,capture[4,8,12,16,20,24,28,32]}` ＝ 同 | ✅ |
| enable-auto-tool-choice / tool-call-parser / reasoning-parser | 有 / glm47 / glm45 ＝ 同 | 同 | ✅ |
| additional-config | `{fuse_muls_add,multistream_overlap_shared_expert,recompute,ascend_compilation{npugraph},enable_dsa_cp,layer_sharding}` ＝ 同 | `{...recompute...}`（无 dsa_cp/layer_sharding）＝ 同 | ✅ |
| speculative-config | `{3,deepseek_mtp}` ＝ 同 | 同 | ✅ |
| quantization / seed / EP | ascend / 1024 / 有 ＝ 同 | 同 | ✅ |
| kv-transfer | V1/producer/30000/extra{P2×16,D16×4,use_ascend_direct} ＝ 同 | V1/consumer/30100 ＝ 同 | ✅ |
| **共用 env** | `HCCL_BUFFSIZE=256`、`ASCEND_AGGREGATE_ENABLE=1`、`ACL_OP_INIT_MODE=1`、`ASCEND_A3_ENABLE=1`、`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480` ＝ 官方 | 同 | ✅ |
| **角色 env** | `FLASHCOMM1+FUSED_MC2` ＝ 官方 | `MLAPO+TASK_QUEUE+FUSED_MC2` ＝ 官方 | ✅ |

---

## 3. 偏离说明（与官方不同但属预期，非缺陷）

| 字段 | 下发 | 官方 | 说明 |
|------|------|------|------|
| `--dtype auto` / `--kv-cache-dtype auto` | 有 | 未列 | wings 默认；`auto` 不改变量化行为 |
| `--block-size 16`（GLM5） | 有 | 未列 | wings 默认；官方 GLM5 命令未显式指定 |
| `--default-chat-template-kwargs '{"enable_thinking":false}'`（GLM5） | 有 | 未列 | wings 启动关思考（`enable_auto_think_choice`），产品行为 |
| `--chat-template .../chat_template.jinja`（V4） | 有 | 未列 | wings 默认模板路径 |
| `--api-server-count 1`（V4） | 有 | 未列 | wings 默认，单 API server |

---

## 4. 唯一真机待确认项

| 项 | 现状 | 待确认 |
|----|------|--------|
| `engine_id` | wings 对 V1（GLM5）与 Hybrid（V4）一律按 `dp_rank` 注入 `engine_id=$RANK` | 官方 GLM5(V1) 命令**无** engine_id、V4(Hybrid) 示例**固定 `0/1`**；多 service 下按 rank 更合理（避免同 pod 冲突），但需真机验证 Mooncake Hybrid 是否要求 role 级常量。 |

---

## 5. 结论

1. **A3 下发字段逐项对齐官方**：GLM-5.1（§2.C）、DeepSeek-V4-Flash（§1.C）的 P/D engine flag + kv-transfer + env 全部 ✅。
2. **所有输入已显式列出并分类**（§0：6 个 CLI 入参 + **12 个必填 env** + 派生/可选 env，标注 ①上层 / ②平台 / ④用户 / ⑤权重 config）；**已精简**：9 CLI/默认 env + 本机 IP 合并为单一 `RANK_IP`(原 `POD_IP`/`HOST_IP` 回退到它) + `DP_SIZE`/`TP_SIZE` 派生自 4 个全局拓扑 `PD_{ROLE}_*`(P/D 互相感知，单一真相源) + `SHARED_VOLUME_PATH` 默认（探针验证生成命令字节级不变，已落地 `dry_run`）；另澄清 `DEVICE_COUNT` 双消费者（§0.4）、`WINGS_ASCEND_PLATFORM` 缺省回退 a2（§0.2）、`VLLM_LLMDD_RPC_PORT` 默认 12890/12777（§0.2.1）——真机部署照 §0 下发即可复现。
3. **拓扑/端口/卡组/rank 自洽**：P/D rpc 分离、kv_port producer/consumer 错开、`dp_rank_start` 由 `RANK_IP` 派生（GLM5 D-node1=4 已验证）。
4. **偏离仅 wings 基础设施字段**（§3，非缺陷）；**唯一真机待确认 = `engine_id`**（§4）。

> 复现：`python dry_run.py --pd glm5 && python dry_run.py --pd v4flash`，产物在 `build/output/`。字段修复与机制见 [pd-dryrun-vs-official-report.md](pd-dryrun-vs-official-report.md) §0、[pd-scheme-fix-plan.md](pd-scheme-fix-plan.md)。
