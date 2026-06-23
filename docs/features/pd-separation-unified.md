# Wings-Control PD 分离全场景讲解

> 面向使用与二次开发 Wings-Control 的工程师。讲解当前分支**所有 PD（Prefill-Decode）分离场景**，每个场景给出「输入 → Wings-Control 自动生成的产物（kv_transfer_config / 环境变量 / 最终命令）」的完整走查。
>
> 文中提到的函数只写名字，**精确文件:行号统一集中在文末 §9 代码索引**（源码频繁变动，正文不散落行号；以函数名为准）。

---

## 1. 总览：四个场景与判定

PD 分离总开关是环境变量 **`PD_ROLE ∈ {P, D}`**。命中后由「设备 + DP_SIZE + 是否叠加 LMCache」决定走哪个场景：

```
PD_ROLE ∈ {P,D} ?
 └─否 → 非 PD（不注入任何 PD 配置）
 └─是 ┬─ DP_SIZE>1 且 架构命中 pd_config.json(或 default) → 【场景二】external-lb / DP fork（Ascend）
      ├─ engine=vllm_ascend 且 DP_SIZE≤1            → 【场景一】Ascend Standalone PD
      └─ engine=vllm（NVIDIA）                       → 【场景三】NVIDIA PD
  叠加：LMCACHE_OFFLOAD=true（作用于场景一/三）       → 【场景四】MultiConnector
```

进程拓扑分流在 `build_start_script()`：

```python
if _pd_external_lb:           → _build_vllm_pd_external_lb_script   # 场景二：pod 内 fork 多进程
elif distributed and nnodes>1: → _build_vllm_distributed_script     # 场景三 NVIDIA PD（及 Ray/Ascend DP）
else:                         → _build_vllm_single_script           # 场景一：单进程
```

### 1.1 处理时序（一次启动经过的 PD 相关步骤）

PD 配置不是一次成型，而是「先按 standalone 写好，再在末尾被 external-lb 覆盖」。完整顺序：

```
① 配置合并  _merge_vllm_params
     _set_kv_cache_config     → 先按 standalone 写 kv_transfer_config（_get_pd_config / MultiConnector）
     _guard_pd_hybrid_kv_cache→ 移除不兼容的 hybrid-kv 开关
     _ensure_pd_head_dim      → config.json 缺 head_dim 时注入 --hf-overrides（对所有 PD 生效）
② 分布式注入 _handle_distributed → _handle_vllm_distributed
     Ascend PD: return（不加 dp）/ NVIDIA PD: dp_deployment / 其它: Ray
③ 最终合并后 _apply_pd_external_lb（step 6，在 explicit_keys 终定之后）
     若 DP_SIZE>1 且命中注册表 → 用注册表覆盖 engine_config、**重写 kv_transfer_config**、
                                  置 _pd_external_lb / _pd_env / distributed=False
④ 命令生成  _prepare_engine_config
     注入器（_force_set_* 等）之后**重申** _pd_engine_overrides（注册表为唯一真相源；value=None 表示删键）
⑤ 脚本分派  build_start_script → 见上方三分支
```

> 关键点：**场景二会覆盖第①步写入的 standalone kv**——所以 DP_SIZE>1 时 `_get_pd_config` 的产物只是中间态，最终以注册表为准。

| 场景 | 触发 | 连接器 | 进程模型 |
| --- | --- | --- | --- |
| 一 Ascend Standalone | Ascend + `DP_SIZE≤1` | `MooncakeConnectorV1` + RDMA | 每 P/D 一个独立进程 |
| 二 external-lb（DP fork） | `DP_SIZE>1` + 注册表命中 | 注册表（Mooncake 系列） | pod 内 fork `dp_size_local` 个进程 |
| 三 NVIDIA PD | NVIDIA | `NixlConnector` | dp_deployment (NIXL) |
| 四 PD + LMCache | 场景一/三 + `LMCACHE_OFFLOAD` | `MultiConnector` | 同上 |

> 注：external-lb 门控 `_get_pd_external_lb_params()` 只看 `PD_ROLE + DP_SIZE`（设备无关），但注册表条目均为 Ascend/Mooncake，故实际是 Ascend 场景。

---

## 2. 场景一：Ascend Standalone PD（最常用，1P1D）

**何时用**：单个 Prefill、单个 Decode，各占一个（或一组 TP）NPU，`DP_SIZE=1`。KV 靠 Mooncake + RDMA 点对点互传。

### 2.1 输入（P 节点）

```bash
WINGS_DEVICE=ascend  WINGS_DEVICE_COUNT=1  PD_ROLE=P \
RANK_IP=7.6.52.170  NETWORK_INTERFACE=ens65f1np1  ASCEND_ENFORCE_EAGER=true \
PD_PREFILL_TP_SIZE=1  PD_DECODE_TP_SIZE=1 \
ENGINE_PORT=17200  HEALTH_PORT=19400 \
VLLM_LLMDD_RPC_PORT=5569  VLLM_MOONCAKE_BOOTSTRAP_PORT=23000 \
bash wings_start.sh --model-name Qwen3-8B --model-path /models/Qwen3-8B \
  --engine vllm_ascend --device-count 1 --port 18200 --trust-remote-code
```

D 节点只改：`PD_ROLE=D`、`ENGINE_PORT=17100`、`HEALTH_PORT=19200`、`VLLM_LLMDD_RPC_PORT=5570`、`VLLM_MOONCAKE_BOOTSTRAP_PORT=23100`、`--port 18100`。

### 2.2 生成的 `kv_transfer_config`（`_get_pd_config`）

P 节点：

```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "mooncake_protocol": "rdma",
    "prefill": {"tp_size": 1, "dp_size": 1, "pp_size": 1},
    "decode":  {"tp_size": 1, "dp_size": 1, "pp_size": 1}
  }
}
```

D 节点仅 `"kv_role": "kv_consumer"` 不同。连接器类型可由 `PD_CONNECTOR_TYPE` 覆盖；prefill/decode 拓扑来自 `PD_PREFILL_*`/`PD_DECODE_*`（tp 缺省取 `device_count`，dp/pp 缺省 1）。

### 2.3 生成的环境变量（`_build_pd_role_env_commands`）

`HCCL_IF_IP` 取本节点 IP（`RANK_IP`，未设则探测本机）：

```bash
export HCCL_IF_IP=7.6.52.170
export GLOO_SOCKET_IFNAME=ens65f1np1
export TP_SOCKET_IFNAME=ens65f1np1
export HCCL_SOCKET_IFNAME=ens65f1np1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100                              # 可由 OMP_NUM_THREADS 覆盖
export VLLM_USE_V1=1
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export CLOSE_MATMUL_K_SHIFT=1
export VLLM_LLMDD_RPC_PORT=5569
export VLLM_MOONCAKE_BOOTSTRAP_PORT=23000
export ASCEND_CONNECT_TIMEOUT=${ASCEND_CONNECT_TIMEOUT:-120000}   # PD 软默认，防 ADXL connect 超时(#2970)
export ASCEND_TRANSFER_TIMEOUT=${ASCEND_TRANSFER_TIMEOUT:-120000} # 同上，平台已设则其值优先
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256     # 可由 NPU_MAX_SPLIT_SIZE_MB 覆盖
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
```

> `current_ip` / `network_interface` 由 `build_start_script` 上游传入（分别取 `RANK_IP`/本机 IP 与 `NETWORK_INTERFACE`）。CANN/ATB 的 `set_env.sh` 由基础环境段（`_build_base_env_commands`）注入，此处不重复。

### 2.4 最终命令（P 节点）

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-8B --port 17200 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 --max-model-len 32768 --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma","prefill":{"tp_size":1,"dp_size":1,"pp_size":1},"decode":{"tp_size":1,"dp_size":1,"pp_size":1}}}'
```

> 要点：`_handle_vllm_distributed` 在 Ascend PD 分支**提前 return**，所以这里**没有任何 `--data-parallel-*`**——P/D 就是两个普通单进程，靠 KV 连接器互传。

---

## 3. 场景二：external-lb / DP fork（DP_SIZE > 1）

**何时用**：一个角色需要起**多个 DP rank** 组成同一 EP/DP all-to-all 域（大 MoE 模型常见）。对齐官方 `launch_online_dp.py`：**一个 pod 内 fork `dp_size_local` 个独立 `vllm serve`**，外部 LB 前置。

**已注册架构（`pd_config.json`，取不到条目且无 `default` → 回退场景一）：**

| 架构 | connector | 备注 |
| --- | --- | --- |
| `default`（兜底） | `MooncakeConnectorV1` | 仅通用安全项；建议专属条目覆盖 |
| `Qwen3MoeForCausalLM` | `MooncakeConnectorV1` | Qwen3-30B-A3B |
| `DeepseekV32ForCausalLM` | `MooncakeLayerwiseConnector` | DeepSeek-V3.2 |
| `GlmMoeDsaForCausalLM` | `MooncakeConnectorV1` | GLM5；P/D 角色级 max_model_len；`common_env` |
| `Qwen3_5MoeForConditionalGeneration` | `MooncakeLayerwiseConnector` | D 角色级 `kv_buffer_device=npu` |
| `DeepseekV4ForCausalLM` | `MooncakeHybridConnector` | 含 `platform_overrides.a2`（A2 4P1D） |

下面用 **Qwen3-30B-A3B（`Qwen3MoeForCausalLM`）的 Prefill 角色，单节点 8 卡，DP_SIZE=2 / TP_SIZE=2 / DP_SIZE_LOCAL=2** 完整走查。

### 3.1 输入（P pod）

```bash
WINGS_DEVICE=ascend  PD_ROLE=P \
DP_SIZE=2  TP_SIZE=2  DP_SIZE_LOCAL=2 \
Master_IP=10.0.0.1  NODE_IPS=10.0.0.1  RANK_IP=10.0.0.1 \
PD_DECODE_DP_SIZE=1  PD_DECODE_TP_SIZE=4 \
NETWORK_INTERFACE=ens65f1np1 \
bash wings_start.sh --model-name Qwen3-30B-A3B --model-path /models/Qwen3-30B-A3B \
  --engine vllm_ascend --device-count 8 --port 18000
```

`_get_pd_external_lb_params()` 解析出：

```
role=P  dp_size=2  tp_size=2  dp_size_local=2
dp_address=10.0.0.1                      # Master_IP
rpc_port=12890                           # P 硬编码（D=12777），不读 env
dp_rank_start = RANK_IP在NODE_IPS的位置(0) × dp_size_local(2) = 0
```

入参支持多级回退（上层平台契约 → 旧名 → 全局拓扑）：

| 字段 | 读取顺序（首个非空生效） |
| --- | --- |
| dp_size | `DP_SIZE` → `PD_DP_SIZE` → `PD_{PREFILL\|DECODE}_DP_SIZE`（≤1 则不走 external-lb） |
| tp_size | `TP_SIZE` → `PD_TP_SIZE` → `PD_{PREFILL\|DECODE}_TP_SIZE` |
| dp_size_local | `DP_SIZE_LOCAL` → `PD_DP_SIZE_LOCAL` |
| dp_address | `Master_IP` → `MASTER_IP` → `PD_DP_ADDRESS` → `get_master_ip()` |
| dp_rank_start | 显式 `PD_DP_RANK_START`，否则由 `RANK_IP`(→`HOST_IP`→本机) 在 `NODE_IPS` 的位置 × `dp_size_local` 派生 |

### 3.2 命中注册表条目（`pd_config.json` → `Qwen3MoeForCausalLM`）

`_apply_pd_external_lb()` 把 `common + prefill.engine` 合并进 engine_config（不覆盖用户显式键），并在所有模型默认注入器**之后重申**（pd_config.json 是 external-lb 唯一真相源）：

```
connector = MooncakeConnectorV1     kv_port.P = 30000
engine: enable_expert_parallel=true, enable_prefix_caching=true,
        disable_hybrid_kv_cache_manager=false, trust_remote_code=true,
        max_num_batched_tokens=8192, max_num_seqs=4, gpu_memory_utilization=0.9,
        enforce_eager=true, additional_config={enable_cpu_binding:"True"}
env  : HCCL_INTRA_ROCE_ENABLE=1, USE_MULTI_GROUPS_KV_CACHE=1, ASCEND_BUFFER_POOL=4:8,
        HCCL_BUFFSIZE=2560, VLLM_RPC_TIMEOUT=3600000, HCCL_EXEC_TIMEOUT=204, ...
```

### 3.3 生成的 `kv_transfer_config`（`_build_pd_external_lb_kv`）

连接器/拓扑来自注册表 + 上层 DP/TP；含两个**占位符**，由 fork 脚本按 rank 替换：

```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "__PD_KVPORT__",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": 2, "tp_size": 2},
    "decode":  {"dp_size": 1, "tp_size": 4}
  },
  "engine_id": "__PD_RANK__"
}
```

（`prefill` 取本角色权威值；`decode` 取 `PD_DECODE_*` 对端拓扑，缺失会告警并回退本角色。注意 external-lb 的 extra 不含 `mooncake_protocol`，与场景一不同。）

### 3.4 生成的 fork 脚本（`_build_vllm_pd_external_lb_script`）

占位符 `__PD_RANK__`/`__PD_KVPORT__` 替换为 bash 变量 `$RANK`/`$KVPORT`，循环 fork 2 个 service：

```bash
# ...注册表 env（HCCL_INTRA_ROCE_ENABLE=1 等）已 export...
(
  pids=()
  for i in $(seq 0 1); do
    RANK=$((0 + i)); PORT=$((18000 + i))
    KVPORT=$((30000 + i)); BOOTSTRAP=$((23000 + i))
    LO=$((i * 2)); HI=$((LO + 2 - 1)); CARDS=$(seq -s, $LO $HI)
    ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
      python3 -m vllm.entrypoints.openai.api_server --model /models/Qwen3-30B-A3B \
      --enable-expert-parallel --max-num-batched-tokens 8192 ... \
      --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":'"$KVPORT"',...,"engine_id":'"$RANK"'}' \
      --port $PORT \
      --tensor-parallel-size 2 --data-parallel-size 2 \
      --data-parallel-rank $RANK --data-parallel-size-local 1 \
      --data-parallel-address 10.0.0.1 --data-parallel-rpc-port 12890 \
      --data-parallel-external-lb &
    pids+=($!)
  done
  wait -n || true
  echo "[pd] a service exited, tearing down pod" >&2
  kill "${pids[@]}" 2>/dev/null || true
  exit 1
)
```

展开后两个 service：

| service | RANK | PORT | KVPORT | BOOTSTRAP | 卡组 |
| --- | --- | --- | --- | --- | --- |
| i=0 | 0 | 18000 | 30000 | 23000 | 0,1 |
| i=1 | 1 | 18001 | 30001 | 23001 | 2,3 |

> 容错：`wait -n` 任一 service 退出即 `kill` 全部 + `exit 1` → 编排层整 pod 重启（EP all-to-all 下单 rank 缺失会让整域 hang）。

### 3.5 要点

- **环境变量 = 基础 PD 角色环境 + 注册表 env**：fork 脚本的 env 先取 `common_env_cmds`（含场景一那套 `HCCL_IF_IP`/sockets/超时/`LD_LIBRARY_PATH`），再叠加注册表 `common_env` + 角色 `env`，最后整段去重。
- **engine 参数以注册表为唯一真相源**：注册表值在所有模型默认注入器之后由 `_pd_engine_overrides` 重申一次，避免被 `_force_set_*` 回填覆盖。
- **rpc-port 硬编码** P=12890 / D=12777，同角色每 pod 各算同一常量 → DP 域天然一致；**网络策略须放行这两个固定口**。
- **`RANK_IP` 必须与 `NODE_IPS` 逐字一致**，否则 `dp_rank_start` 回退 0 → 多节点 rank 撞车。
- 多节点时每个 pod 设各自的 `RANK_IP`，`dp_rank_start` 自动错开（pod0→0、pod1→`dp_size_local`…）。
- `compilation_config: null`（注册表里）= **删除** base 注入的图捕获键。

---

## 4. 场景三：NVIDIA PD（NixlConnector + dp_deployment）

**何时用**：GPU（NVIDIA）上的 PD 分离，KV 经 NIXL side-channel 传输。

### 4.1 输入（P 节点）

```bash
WINGS_DEVICE=nvidia  PD_ROLE=P \
NETWORK_INTERFACE=eth0  VLLM_NIXL_SIDE_CHANNEL_PORT=12345 \
ENGINE_PORT=8100 \
bash wings_start.sh --model-name Qwen3-8B --model-path /models/Qwen3-8B \
  --engine vllm --device-count 1 --port 18100
```

### 4.2 生成的 `kv_transfer_config`（`_get_pd_config` 非 Ascend 分支）

```json
{ "kv_connector": "NixlConnector", "kv_role": "kv_both" }
```

### 4.3 生成的环境变量（分两部分，注意条件不同）

- **始终注入**（PD 角色环境 `_build_pd_role_env_commands` vllm 分支）：

  ```bash
  export VLLM_NIXL_SIDE_CHANNEL_HOST=10.1.0.5     # 本节点 IP
  ```

- **仅当走多节点 dp_deployment 脚本时追加**（`_build_nvidia_dp_env_commands`，单节点 `_build_vllm_single_script` 不产生）：

  ```bash
  export GLOO_SOCKET_IFNAME=eth0
  export TP_SOCKET_IFNAME=eth0
  export NCCL_SOCKET_IFNAME=eth0
  export VLLM_NIXL_SIDE_CHANNEL_PORT=12345
  export NCCL_IB_DISABLE=0
  export NCCL_CUMEM_ENABLE=0
  export NCCL_NET_GDR_LEVEL=SYS
  ```

### 4.4 分布式后端

`_handle_vllm_distributed()` 命中 `(pd_role and not is_ascend)` → 设：

```
distributed_executor_backend = dp_deployment
nixl_ip   = <本节点 IP>
nixl_port = 27070   (VLLM_DISTRIBUTED_PORT 可覆盖)
rpc_port  = 27071   (dp_rpc_port 优先取此值，否则 VLLM_DP_RPC_PORT / 13355)
```

最终 exec 行由 `_build_dp_deployment_commands` → `_build_dp_exec_command` 按 node_rank 生成（在 `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` 之外）：

```bash
# rank0（head 节点）
exec python3 -m vllm... --kv-transfer-config '{...NixlConnector...}' \
  --data-parallel-address <head_ip> --data-parallel-rpc-port 27071 \
  --data-parallel-size <N> --data-parallel-size-local <local>      # 可选 --data-parallel-start-rank

# rank>0（worker，剥离 --port/--host）
exec python3 -m vllm... --data-parallel-address <head_ip> --data-parallel-rpc-port 27071 \
  --data-parallel-size <N> --data-parallel-size-local <local> \
  --headless --data-parallel-start-rank <start>
```

> 该路径走 `_build_vllm_distributed_script`（需 `distributed and nnodes>1`）；单节点 NVIDIA PD 则走单进程脚本，只保留 §4.3 的 `VLLM_NIXL_SIDE_CHANNEL_HOST`。

> 区分：`is_ascend_deepseek`（DeepseekV3/V32/V4、GlmMoeDsa、KimiK25 on Ascend）即使不开 PD 也走 dp_deployment，但那是**纯 DP 部署、非 PD 分离**。

---

## 5. 场景四：PD + LMCache 叠加（MultiConnector）

**何时用**：在 PD 分离基础上再叠加 KV 卸载（`LMCACHE_OFFLOAD=true`）。`_set_kv_cache_config()` 把两者打包进 `MultiConnector`。

### 5.1 输入

在场景一/三的环境变量基础上加：`LMCACHE_OFFLOAD=true`（Ascend 可加 `LMCACHE_ENGINE_ID=lmca1`）。

### 5.2 生成的 `kv_transfer_config`（Ascend P 节点）

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
          "mooncake_protocol": "rdma",
          "prefill": {"tp_size": 1, "dp_size": 1, "pp_size": 1},
          "decode":  {"tp_size": 1, "dp_size": 1, "pp_size": 1}
        }
      },
      {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both",
        "engine_id": "lmca1",
        "kv_buffer_device": "npu"
      }
    ]
  }
}
```

> 决策矩阵：LMCache+PD→MultiConnector；仅 LMCache→LMCacheConnectorV1；仅 PD→场景一/三连接器；都无→不注入。该叠加作用于 standalone 路径；external-lb（场景二）的 kv 由注册表单独构建。

---

## 6. 跨场景通用机制

| 机制 | 函数 | 说明 |
| --- | --- | --- |
| 触发开关 | `get_pd_role_env` | 仅认 `P`/`D`，否则关闭 |
| hybrid-kv 保护 | `_guard_pd_hybrid_kv_cache` | 移除显式 `no_disable_hybrid_kv_cache_manager`（与连接器冲突） |
| head_dim 补全 | `_ensure_pd_head_dim` | config.json 缺 `head_dim`（如 Qwen2）→ 注入 `--hf-overrides '{"head_dim":N}'` |
| 崩溃回退 | wings_entry | 120s 内崩溃回退基础模式时 `pop(kv_transfer_config)` → PD 失效（属预期） |
| 逃生阀（场景二） | `PD_DISABLE_ASCEND_DIRECT` | 移 `use_ascend_direct`，绕开 mooncake ADXL 直传（vllm-ascend#2970） |

---

## 7. 部署与验证

- **Compose**：每个 P/D 一组三容器（control + engine + 共享卷）；同机 P/D 的 `ENGINE_PORT`/`HEALTH_PORT`/`--port`/RPC/Bootstrap 端口/设备号/共享卷必须互不冲突。
- **K8s**：P/D 各建独立 Pod/Deployment，用不同 Service 暴露 Proxy 端口；P/D 须经 RDMA 网卡互通。

```bash
curl -s http://127.0.0.1:19400/health   # P → {"status":"healthy"}
curl -s http://127.0.0.1:18200/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
# 日志成功标志：Application startup complete. / Started server process
```

---

## 8. 注意事项

| # | 场景 | 事项 |
| --- | --- | --- |
| 1 | 一/二/四 | 连接器须 `MooncakeConnectorV1`（Ascend KV 是 tuple，上游 `MooncakeConnector` 不兼容） |
| 2 | 一/二 | 并行参数不可缺，否则 `assert "tp_size" in prefill_parallel_config` |
| 3 | 全部 | 网卡用 RDMA 网卡（`ens65f1np1`），非 `eth0`；同机 P/D 资源隔离 |
| 4 | 二 | rpc-port 固定 12890/12777，网络策略放行；`RANK_IP` 必须与 `NODE_IPS` 逐字一致 |
| 5 | 二 | `FULL_DECODE_ONLY` 有 MTE 越界历史，新模型首跑真机验证，必要时条目改 `enforce_eager` |
| 6 | 全部 | 高并发堆损坏 → mooncake `0.3.10.post1` + jemalloc 兜底 |
| 7 | Ascend | A+X 环境设 `ASCEND_ENFORCE_EAGER=true` |

---

## 9. 代码索引

> 行号为当前快照；源码频繁变动，**以函数名为准**，行号仅辅助跳转。

| 环节 | 函数 | 位置 |
| --- | --- | --- |
| 总开关 | `get_pd_role_env` | [env_utils.py:223](../../wings_control/utils/env_utils.py#L223) |
| KV 配置注入 | `_set_kv_cache_config` | [config_loader.py:1267](../../wings_control/core/config_loader.py#L1267) |
| Standalone 连接器 | `_get_pd_config` | [config_loader.py:832](../../wings_control/core/config_loader.py#L832) |
| external-lb 门控 | `_get_pd_external_lb_params` | [config_loader.py:904](../../wings_control/core/config_loader.py#L904) |
| external-lb kv 构建 | `_build_pd_external_lb_kv` | [config_loader.py:997](../../wings_control/core/config_loader.py#L997) |
| external-lb 编排 | `_apply_pd_external_lb` | [config_loader.py:1087](../../wings_control/core/config_loader.py#L1087) |
| 平台 overlay | `_resolve_ascend_platform` | [config_loader.py:1064](../../wings_control/core/config_loader.py#L1064) |
| 注册表加载 | `_load_pd_config` | [config_loader.py:87](../../wings_control/core/config_loader.py#L87) |
| 注册表文件 | — | [pd_config.json](../../wings_control/config/defaults/pd_config.json) |
| hybrid-kv 保护 | `_guard_pd_hybrid_kv_cache` | [config_loader.py:1358](../../wings_control/core/config_loader.py#L1358) |
| head_dim 补全 | `_ensure_pd_head_dim` | [config_loader.py:1378](../../wings_control/core/config_loader.py#L1378) |
| 分布式分流 | `_handle_vllm_distributed` | [config_loader.py:2447](../../wings_control/core/config_loader.py#L2447) |
| 角色环境变量 | `_build_pd_role_env_commands` | [vllm_adapter.py:755](../../wings_control/engines/vllm_adapter.py#L755) |
| NVIDIA DP 环境 | `_build_nvidia_dp_env_commands` | [vllm_distributed.py:244](../../wings_control/engines/vllm_distributed.py#L244) |
| external-lb fork 脚本 | `_build_vllm_pd_external_lb_script` | [vllm_adapter.py:2907](../../wings_control/engines/vllm_adapter.py#L2907) |
| NVIDIA DP 脚本 | `_build_dp_deployment_commands` | [vllm_distributed.py:341](../../wings_control/engines/vllm_distributed.py#L341) |
| 单进程脚本 | `_build_vllm_single_script` | [vllm_adapter.py:2887](../../wings_control/engines/vllm_adapter.py#L2887) |
| 脚本生成入口 | `build_start_script` | [vllm_adapter.py:2985](../../wings_control/engines/vllm_adapter.py#L2985) |
| 崩溃回退清理 | (pop kv_transfer_config) | [wings_entry.py:1281](../../wings_control/core/wings_entry.py#L1281) |
