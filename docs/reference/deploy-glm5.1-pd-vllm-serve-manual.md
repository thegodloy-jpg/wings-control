# GLM-5.1 PD 分离手工启动命令（vllm serve · P 节点 / D 节点）

> 用途：把贴来的（OCR/复制错位的）`vllm serve` 命令**整理成准确、可直接运行**的版本。
> 拓扑（由命令内 `kv_connector_extra_config` 固定）：**Prefill = DP4×TP8（EP32）**、**Decode = DP8×TP4（EP32）**，约 **8×910b（4P4D-node）**。
> P=Prefill=`kv_producer`，D=Decode=`kv_consumer`。**P、D 命令不同**：D 是「微批 + FULL_DECODE_ONLY 图」，P 是「大批 + enforce-eager」（见 §4 / §5）。

---

## 1. 位置参数约定（外部编排器逐 service 下发）

| 变量 | 含义 | P 示例 | D 示例 |
|------|------|--------|--------|
| `$2` | HTTP 端口（`--port`） | `17000` | `17000` |
| `$3` | `--data-parallel-size`（角色全局 DP） | `4` | `8` |
| `$4` | `--data-parallel-rank`（本 service 全局 DP rank） | `0` | `0` |
| `$5` | `--data-parallel-address`（角色 DP rank0 的 IP） | `<P_rank0_IP>` | `<D_rank0_IP>` |
| `$6` | `--data-parallel-rpc-port`（DP RPC 端口） | `12890` | `12777` |
| `$7` | `--tensor-parallel-size` | `8` | `4` |
| `$8` | 本机 IP（`local_ip`，用于 `HCCL_IF_IP`） | `<本机IP>` | `<本机IP>` |

> `$1` 未用于 `vllm serve`（角色/句柄占位）。

---

## 2. 环境变量

### 2.1 公共部分（P、D 都设）
```bash
nic_name="enp189s0f0"      # 数据面网卡，按实际改
local_ip=$8                # 本机 IP（位置参数 $8）

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000000
export HCCL_INTRA_ROCE_ENABLE=1

# 本 service 独占卡：P（TP8）→ 8 张；D（TP4）→ 改成 4 张
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 2.2 仅 P 节点额外设
```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1     # 仅 P；D 不设
```

> 环境变量原文修正：
> - `HCCL_OP_EXPANSION MODE` → `HCCL_OP_EXPANSION_MODE`（P 段缺下划线；D 段本就正确）。
> - D 段 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:1True` → `...:True`（多写了 `1`）。
> - 末尾裸 `export ASCEND_RT_VISIBLE_DEVICES`（无取值）补成实际卡列表，否则进程会看到全部卡。
> - `VLLM_ASCEND_ENABLE_FLASHCOMM1` **只有 P 段有**，D 段没有 —— 保持这个区别。
> - `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` 两段都是 `30000000`（=30s），无需统一。

---

## 3. JSON 配置块说明（运行时用 §4/§5 的单行紧凑形式；这里是美化版便于核对）

原文里 JSON 错位较多，统一修正：`"""engine_id"": ""0"","` → `"engine_id":"0"`；`"fuse muls add"` → `"fuse_muls_add"`；`"use_ascend_ direct"` → `"use_ascend_direct"`；`\'{`、`}'{`、`}"` 等收尾/开头符还原为合法 `{` / `}`；嵌套大括号补全。

**`--kv-transfer-config`** —— P 与 D 仅 `kv_role`/`kv_port`/`engine_id` 不同；`kv_connector_extra_config` 描述整套拓扑，两端一致：

```json
// P 节点
{ "kv_connector":"MooncakeConnectorV1", "kv_role":"kv_producer", "kv_port":"30000", "engine_id":"0",
  "kv_connector_module_path":"vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config":{ "use_ascend_direct":true,
    "prefill":{"dp_size":4,"tp_size":8}, "decode":{"dp_size":8,"tp_size":4} } }

// D 节点
{ "kv_connector":"MooncakeConnectorV1", "kv_role":"kv_consumer", "kv_port":"30100", "engine_id":"1",
  "kv_connector_module_path":"vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config":{ "use_ascend_direct":true,
    "prefill":{"dp_size":4,"tp_size":8}, "decode":{"dp_size":8,"tp_size":4} } }
```

**`--additional-config`**（P、D 相同）：
```json
{ "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true,
  "fuse_qknorm_rope":true, "fuse_muls_add":true, "enable_npugraph_ex":true }
```

**`--compilation-config`**（**仅 D 节点**）：
```json
{ "cudagraph_capture_sizes":[4,8,12,16,20,24,28,32], "cudagraph_mode":"FULL_DECODE_ONLY" }
```

**`--speculative-config`**（P、D 相同）：
```json
{ "num_speculative_tokens":3, "method":"deepseek_mtp" }
```

> `engine_id` 按原文取 P=`0` / D=`1`（角色级常量）。若同一角色 fork 多个 service 且引擎要求每进程唯一，需改成按 `--data-parallel-rank` 取值。

---

## 4. P 节点完整命令（大批 + enforce-eager）

```bash
vllm serve /mnt/share/weights/glm-5.1-w8a8 \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --seed 1024 \
  --served-model-name glm5.1 \
  --max-model-len 135168 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --quantization ascend \
  --async-scheduling \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_module_path":"vllm_ascend.distributed.mooncake_connector","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":4,"tp_size":8},"decode":{"dp_size":8,"tp_size":4}}}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert":true,"fuse_qknorm_rope":true,"fuse_muls_add":true,"enable_npugraph_ex":true}' \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}'
```

---

## 5. D 节点完整命令（微批 + FULL_DECODE_ONLY，**无 `--enforce-eager`**）

```bash
vllm serve /mnt/share/weights/glm-5.1-w8a8 \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --seed 1024 \
  --served-model-name glm5.1 \
  --max-model-len 135168 \
  --max-num-batched-tokens 32 \
  --trust-remote-code \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"30100","engine_id":"1","kv_connector_module_path":"vllm_ascend.distributed.mooncake_connector","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":4,"tp_size":8},"decode":{"dp_size":8,"tp_size":4}}}' \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert":true,"fuse_qknorm_rope":true,"fuse_muls_add":true,"enable_npugraph_ex":true}' \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}'
```

---

## 6. P vs D 关键差异速查

| 项 | P 节点 | D 节点 |
|----|--------|--------|
| 角色 / `kv_role` | Prefill / `kv_producer` | Decode / `kv_consumer` |
| `kv_port` | `30000` | `30100` |
| `engine_id` | `0` | `1` |
| `--tensor-parallel-size`（`$7`） | 8 | 4 |
| `--data-parallel-size`（`$3`） | 4 | 8 |
| `--max-num-batched-tokens` | **4096** | **32** |
| `--max-num-seqs` | **256** | **8** |
| `--enforce-eager` | **有** | **无** |
| `--compilation-config` | 无 | **FULL_DECODE_ONLY + capture_sizes** |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 设 | 不设 |
| 其余（模型路径/parser/投机解码/`--additional-config`/拓扑/`gpu-mem 0.95`/`max-model-len 135168`） | 相同 | 相同 |

---

## 7. 原文 → 准确版 修正清单

| 原文（错） | 修正后 |
|-----------|--------|
| `vllm serve/mnt/share/...` | `vllm serve /mnt/share/...`（补空格） |
| 行尾 `\` 紧贴取值（`--port $2\` 等）/ 缺 `\`（`--seed 1024`、D 的 `--host 0.0.0.0`） | 续行 `\` 前补空格、缺的补上 |
| P 段 `--max-model-len 135168 |` | 删行尾误入的 `|` |
| `HCCL_OP_EXPANSION MODE`（P 段） | `HCCL_OP_EXPANSION_MODE` |
| D 段 `expandable_segments:1True` | `expandable_segments:True` |
| JSON：`"""engine_id"": ""0"","`、`"fuse muls add"`、`"use_ascend_ direct"`、`\'{` / `}'{` / `}"` 收尾错位、缺括号 | 见 §3 |
| D 段 `--kv-transfer-config\` 后跟 `}"`（开头多了 `}`） | 还原为 `'{ ... }'` |

---

## 8. 运行前必检

1. **`kv_role` 别搞反**：P=`kv_producer`、D=`kv_consumer`；`kv_port` 也错开（30000/30100）。搞反/同口则 KV 不流动、无输出。
2. **`ASCEND_RT_VISIBLE_DEVICES`** 每 service 给独占卡：P 的 TP8→8 张、D 的 TP4→4 张；同节点多 service 不能重叠。
3. **`--data-parallel-address`(`$5`)** = 该角色 DP rank0 的 IP；**`--data-parallel-rpc-port`(`$6`)** 同角色一致且网络放行（P/D 用不同口，见 §1 示例 12890/12777）。
4. **mooncake 端口**（kv_port 30000/30100 + bootstrap）需在 P↔D 间放行。
5. **需外部 PD proxy** 才能真把请求按 prefill→decode 串起来（见同目录 [deploy-glm5.1-pd-2node-1p1d.md](deploy-glm5.1-pd-2node-1p1d.md) §5）。

> 与仓库**已验证 recipe**（[pd_config.json](../../wings_control/config/defaults/pd_config.json) 的 `GlmMoeDsaForCausalLM`）仍存的差异，按需对齐：
> - 本文 connector 为 `MooncakeConnectorV1` + `vllm_ascend.distributed.mooncake_connector`；注册表已换成 `MooncakeConnector` + `...kv_transfer.kv_p2p.mooncake_connector`。以镜像里实际存在的模块为准。
> - 本文 env 含 `VLLM_ASCEND_BALANCE_SCHEDULING=1`；注册表**刻意不设**（PD 分离下 vLLM-Ascend≥0.20.2 会 ValidationError，`vllm_adapter._filter_pd_incompatible_env` 会剔除）。真机若用裸命令启动需留意。
> - decode 批量：本文 `32/8` vs 注册表 `164/48`；`max_model_len` 本文两端 `135168` vs 注册表 P=131072/D=200000；`additional-config` 本文 `enable_npugraph_ex` 在顶层且含 `fuse_qknorm_rope`，注册表把它嵌在 `ascend_compilation_config` 下、改用 `enable_sparse_c8`/`enable_dsa_cp`。择一，以引擎版本接受的为准。
```

