# PD 分离关键参数参考（TP / DP / 端口 / rank / 卡组 如何设置与计算）

> 本文只讲**参数**：每个关键量的**来源**（哪个 env / 注册表 / 派生公式）、**默认值**、**由谁设定**（上层下发 vs wings 派生），以及**计算规则**。不涉及实现细节。
> 适用范围：external-lb PD（`PD_ROLE∈{P,D}` 且 `DP_SIZE>1`，vLLM-Ascend DP fork + Mooncake）。
> 配套：拓扑装配机制见 [pd-registry-authoritative-design.md](pd-registry-authoritative-design.md)，整机示例见 [deploy-glm5.1-pd-4node-2p2d.md](deploy-glm5.1-pd-4node-2p2d.md)。

---

## 0. 三条总原则（先记住）

1. **P 和 D 是两个独立的 DP 域**。每个角色各有自己的 `DP_SIZE / TP_SIZE / Master_IP / RPC 端口 / NODE_IPS`。下面凡带"本角色"的量，P、D 各算各的。
2. **并行度（dp/tp/local/rank_start）由上层算好下发，wings 不自算**——唯一例外是 `dp_rank_start`，wings 用 `RANK_IP` 在 `NODE_IPS` 的位置派生。
3. **引擎调优参数（batched/seqs/gpu-mem/max-model-len/compilation…）由注册表 [pd_config.json](../../wings_control/config/defaults/pd_config.json) 按"模型架构 + 角色"给定**，上层**不要再传同名 flag**，传了就顶掉注册表（优先级：用户显式 > 注册表 > base 默认）。

---

## 1. 参数总表（来源 · 默认 · 谁设）

**必传图例**：✅ 必传（无默认，不传则 PD 不触发/不工作） · △ 条件必传（有 fallback，但多 pod/多卡/异构拓扑不传会出错或退化，见脚注） · ○ 选传（有安全默认，不传也正确） · — 不由上层传（wings 派生 / 注册表给）

| 参数 | env / 来源 | 默认 | 必传? | 作用 |
|---|---|---|---|---|
| **PD_ROLE** | `PD_ROLE` | 无 | ✅ | 角色门控 `P`/`D`；决定 kv_role、注册表角色块 |
| **DP_SIZE**（本角色全局 DP） | `DP_SIZE` →`PD_DP_SIZE` →`PD_{PREFILL\|DECODE}_DP_SIZE` | 无 | ✅ | `--data-parallel-size`；须 >1 才走 external-lb，否则退 standalone |
| **TP_SIZE**（单实例 TP） | `TP_SIZE` →`PD_TP_SIZE` →`PD_{PREFILL\|DECODE}_TP_SIZE` | `1` | △¹ | `--tensor-parallel-size`；卡组大小 |
| **DP_SIZE_LOCAL**（每 pod fork 数） | `DP_SIZE_LOCAL` →`PD_DP_SIZE_LOCAL` | `1` | △² | pod 内 fork 几个 service |
| **dp_rank_start**（本 pod 起始 rank） | `PD_DP_RANK_START`（显式）否则派生 | 派生 | — | 本 pod 第 0 个 service 的全局 rank（wings 派生；显式 env 可覆盖） |
| **Master_IP**（DP 域 head） | `Master_IP`→`MASTER_IP`→`PD_DP_ADDRESS`→`get_master_ip()` | fallback `get_master_ip()` | △³ | `--data-parallel-address`（= 本角色 node0 IP） |
| **NODE_IPS**（本角色全部节点 IP） | `NODE_IPS`（逗号分隔） | 无（缺则 node_rank=0） | △⁴ | 顺序即节点 rank 序，派生 rank_start 用 |
| **RANK_IP**（本 pod 唯一 IP） | `RANK_IP`→`HOST_IP`→`get_local_ip()` | fallback `HOST_IP`/本机 IP | △⁵ | 派生 rank_start；`--host` |
| **对端拓扑** | `PD_PREFILL_{DP,TP}_SIZE` / `PD_DECODE_{DP,TP}_SIZE` | 缺则回退本角色+告警 | △⁶ | KV extra_config 两端映射 |
| **RPC 端口** | `VLLM_LLMDD_RPC_PORT`→`PD_DP_RPC_PORT` | P=`12890` / D=`12777` | ○ | `--data-parallel-rpc-port` |
| **引擎 HTTP 端口**（base） | `ENGINE_PORT` → PortPlan.backend_port → base cmd `--port` | `17000` | ○ | 每 service `PORT = base + i` |
| **bootstrap 端口**（base） | `VLLM_MOONCAKE_BOOTSTRAP_PORT` | P=`23000` / D=`23100` | ○ | 每 service `BOOTSTRAP = base + i` |
| **proxy 端口** | 用户 `--port` / `PORT`(env) → PortPlan.proxy_port | `18000`（`settings.PORT`） | ○ | sidecar 对外（K8s Service） |
| **health 端口** | `settings.HEALTH_PORT` → PortPlan.health_port | `19000` | ○ | 探针；external-lb 无 +1 偏移 |
| **monitor 端口** | `settings.MONITOR_PROXY_PORT` | `19100` | ○ | 监控透传 |
| **kv_port**（base） | **注册表** `entry["kv_port"][role]`（逐模型） | 见 §3.1（代码兜底 30000/30100） | — | 每 service `KVPORT = base + i`（注册表给，上层不传） |

**脚注**（△ 条件必传的触发条件）：
1. **TP_SIZE**：默认 1（单卡）；任何多卡部署都须显式传（= 单实例占卡数）。
2. **DP_SIZE_LOCAL**：默认 1；当 `TP < 每 pod 卡数` 时**必须传**，否则只 fork 1 个 service、余卡闲置。
3. **Master_IP**：不传回退 `get_master_ip()`；多 pod DP 域必须显式指向**本角色 node0**，否则 rendezvous 找不到 head。
4. **NODE_IPS**：单节点（每角色 1 pod）可省；多节点**必传**，且**顺序 = rank 序**（rank_start 依赖位置）。
5. **RANK_IP**：不传回退 `HOST_IP`/本机 IP；多 pod **必须显式**且**逐字在 NODE_IPS 内**，否则同宿主多 pod rank 撞车。
6. **对端拓扑**：不传回退本角色+告警 → KV 映射两端不一致；**P/D 拓扑只要不同就必传**（实务上 PD 恒不同 → 视为必传）。

> `--data-parallel-size-local` 在 fork 出的每个 service 上**恒为 1**：external-lb 把 pod 内每个 service 当独立 DP 成员，本地并发靠 fork 而非 vLLM 的 local-dp。

---

## 2. 核心计算公式

### 2.1 并行度（上层算，三者互相约束）

```
DP_SIZE_LOCAL = 每 pod 卡数 ÷ TP_SIZE          # 一个 pod 能塞下几个 TP 组
DP_SIZE       = 本角色节点数 × DP_SIZE_LOCAL    # = 本角色节点数 × 卡数 ÷ TP_SIZE
EP（专家并行） = DP_SIZE × TP_SIZE              # MoE 专家切分数，决定每卡专家权重内存
```

**约束**：`DP_SIZE_LOCAL × TP_SIZE ≤ 每 pod 卡数`（卡不够会越界）。
**举例**（16 卡/pod）：TP16 → local=1；TP4 → local=4；TP8 → local=2。

### 2.2 rank_start（wings 唯一自算的量）

```
dp_rank_start = index(RANK_IP in NODE_IPS) × DP_SIZE_LOCAL
```

- `RANK_IP` 必须**逐字**出现在 `NODE_IPS` 里；不在 → 回退 0 并报错（多节点会 rank 撞车）。
- **必须用 RANK_IP，不能用 HOST_IP**：同宿主多 pod 共享 `HOST_IP`，用它会让多 pod 算出同一 rank_start → DP 域起不来。
- 显式 `PD_DP_RANK_START` 可覆盖派生。

### 2.3 逐 service 派生（pod 内 `for i in 0 .. DP_SIZE_LOCAL-1`）

| 量 | 公式 |
|---|---|
| 全局 rank | `RANK = dp_rank_start + i` |
| 引擎端口 | `PORT = base_port + i` |
| kv 端口 | `KVPORT = kv_base + i` |
| bootstrap 端口 | `BOOTSTRAP = bootstrap_base + i` |
| 可见卡组 | `[i×TP, (i+1)×TP)`，即 `LO=i×TP, HI=LO+TP-1` |
| engine_id | `= RANK`（MooncakeConnectorV1 / Hybrid 按 rank 唯一） |

> 端口 `+i` 是为了**单 pod 内多 service 不抢端口**；不同 pod 即使端口相同也不冲突（各自 pod IP）。
> **base 取值**（易误区）：`base_port` = 引擎 `--port`（= `ENGINE_PORT`，默认 17000）；`kv_base` = **注册表** `kv_port[role]`（逐模型，见 §3.1，**非定值**）；`bootstrap_base` = `VLLM_MOONCAKE_BOOTSTRAP_PORT`（缺省 P=23000/D=23100）。

### 2.4 KV 拓扑（两端必须一致）

`kv_connector_extra_config` 里同时写 `prefill{}` 和 `decode{}` 两块，每块 `{dp_size, tp_size}`：

| 块 | P 端取值 | D 端取值 |
|---|---|---|
| `prefill` | 本角色 `DP_SIZE/TP_SIZE` | 对端 `PD_PREFILL_{DP,TP}_SIZE` |
| `decode` | 对端 `PD_DECODE_{DP,TP}_SIZE` | 本角色 `DP_SIZE/TP_SIZE` |

**铁律**：P、D 两边算出的 `prefill{}` 必须完全相同、`decode{}` 必须完全相同，否则 mooncake KV rank 映射两端不一致 → 传错/握手失败。所以把 `PD_PREFILL_{DP,TP}_SIZE` + `PD_DECODE_{DP,TP}_SIZE` 这 4 个值作为**单一真相源**，在所有 P/D pod 上一字不差地下发；本角色的 `DP_SIZE/TP_SIZE` 其实就等于其中对应的那两项，可不必再单独下发。

---

## 3. 端口规划（来源 · 取值）

> **辨明"默认值" vs"取来的值"**：
> - **真·env 默认值**（不设 env 才用缺省）：引擎 backend(`ENGINE_PORT`17000)、bootstrap(23000/23100)、DP RPC(12890/12777)、health(19000)、monitor(19100)。
> - **非默认值**：`kv_port` 来自**注册表逐模型**（§3.1，30000/30100 只是代码异常兜底）；`proxy` 来自**用户 `--port`**（缺省才 18000）。
>
> **必传性**：本节所有端口**均非必传**（全部有默认或注册表/派生来源）——上层不传也能起。`proxy` 可由 `--port` 自定，其余按默认即可。

| 端口 | 取值 | 来源（代码） | 作用域 |
|---|---|---|---|
| 引擎 HTTP `PORT` | `17000 (+i)` | `ENGINE_PORT`(env,默认17000) → PortPlan.backend_port → 引擎 `--port` | 每 service |
| `kv_port` | **逐模型(+i)**，见 §3.1 | 注册表 `entry["kv_port"][role]`；异常兜底 P=30000/D=30100 | 每 service |
| `bootstrap` | P `23000` / D `23100` `(+i)` | `VLLM_MOONCAKE_BOOTSTRAP_PORT`（env 缺省） | 每 service |
| DP RPC | P `12890` / D `12777` | `VLLM_LLMDD_RPC_PORT`/`PD_DP_RPC_PORT`（env），缺省在 fork 脚本兜底 | 每角色 |
| proxy | 用户 `--port`，缺省 `18000` | `launch_args.port` → PortPlan.proxy_port（`settings.PORT=18000`） | 每 pod |
| health | `19000` | `settings.HEALTH_PORT` → PortPlan.health_port；external-lb **无 master/worker +1 偏移** | 每 pod |
| monitor | `19100` | `settings.MONITOR_PROXY_PORT` → PortPlan.monitor_proxy_port | 每 pod |

> proxy 的 `--prefiller-ports/--decoder-ports` 填的是**引擎 HTTP 端口**（17000+i），不是 kv_port/bootstrap。
> ⚠️ 代码细节：fork 脚本优先从 base cmd 的 `--port` 取 base（正常路径 = PortPlan.backend_port = 17000）；仅当 cmd 里**完全没有** `--port` 时才回退 `os.getenv("ENGINE_PORT","18000")`——此兜底默认 18000 与 port_plan 的 17000 不一致，但因 `--port` 恒由 PortPlan 注入，该兜底分支正常不会触发。

### 3.1 kv_port 逐模型（注册表 `kv_port`，**非定值**）

| 模型架构 | P（kv_producer） | D（kv_consumer） |
|---|---|---|
| `GlmMoeDsaForCausalLM` | 30000 | 30100 |
| `DeepseekV32ForCausalLM` | 30000 | 30100 |
| `DeepseekV4ForCausalLM` | 30000 | 30100 |
| `Qwen3MoeForCausalLM` | 30000 | 30100 |
| `Qwen3_5MoeForConditionalGeneration` | **23010** | **36010** |
| `default`（未注册兜底模板） | **30100** | **30400** |
| 代码异常兜底（注册表读不到 `kv_port`） | 30000 | 30100 |

> 多数模型恰为 30000/30100，但 `default` 模板（30100/30400）与 Qwen3.5（23010/36010）不同 → **别把 30000/30100 当通用默认**，以注册表条目为准。来源 [pd_config.json](../../wings_control/config/defaults/pd_config.json) 各条目 `kv_port` 字段。

---

## 4. 工作示例：GLM-5.1 · 2P2D · 4 机 64 卡

拓扑：**P×2（DP2×TP16）+ D×2（DP8×TP4）**，每 pod 16 卡。

| | P 节点（×2） | D 节点（×2） |
|---|---|---|
| `PD_ROLE` | `P` | `D` |
| `TP_SIZE` | 16 | 4 |
| `DP_SIZE`（=节点数×16÷TP） | 2 | 8 |
| `DP_SIZE_LOCAL`（=16÷TP） | 1 | 4 |
| pod 内 service 数 | 1 | 4 |
| `dp_rank_start`（=节点序×local） | P-0→0, P-1→1 | D-0→0, D-1→4 |
| 引擎端口 | 17000 | 17000~17003 |
| kv_port | 30000 | 30100~30103 |
| bootstrap | 23000 | 23100~23103 |
| 卡组 | 0–15 | 0–3 · 4–7 · 8–11 · 12–15 |
| kv-config `prefill` | `{dp:2, tp:16}` | `{dp:2, tp:16}` |
| kv-config `decode` | `{dp:8, tp:4}` | `{dp:8, tp:4}` |

4 个全局拓扑值（所有 pod 一致）：`PD_PREFILL_DP_SIZE=2 PD_PREFILL_TP_SIZE=16 PD_DECODE_DP_SIZE=8 PD_DECODE_TP_SIZE=4`。

---

## 5. 上层下发清单（✅必传 / △条件必传 / ○选传）

> 判定口径：**无默认 → 必传；有默认 → 选传**（不传用缺省）。△ 在多 pod/多卡/异构拓扑下等同必传（脚注见 §1）。

**公共**（所有 pod 一字不差一致）
- ✅ `PD_PREFILL_DP_SIZE` `PD_PREFILL_TP_SIZE` `PD_DECODE_DP_SIZE` `PD_DECODE_TP_SIZE`（KV 两端拓扑唯一真相源；本角色 `DP_SIZE/TP_SIZE` 由此派生，可不必单独下发）
- ✅ `DEVICE_COUNT`、平台标识（`WINGS_ASCEND_PLATFORM` 等）

**P / D 角色每 pod**
- ✅ `PD_ROLE`（`P` / `D`）
- △ `Master_IP`（指本角色 node0）、`NODE_IPS`（顺序=rank 序）、`RANK_IP`（逐字在 NODE_IPS 内）——**多 pod 三者必传**
- △ `DP_SIZE_LOCAL`（`TP < 每 pod 卡数` 时必传，否则默认 1 只 fork 1 个 service）
- △ `TP_SIZE`（多卡必传；或经公共的 `PD_{PREFILL|DECODE}_TP_SIZE` 派生）
- ○ `VLLM_LLMDD_RPC_PORT`（默认 P=12890/D=12777）、`ENGINE_PORT`（默认 17000）、`VLLM_MOONCAKE_BOOTSTRAP_PORT`（默认 23000/23100）、`--port`（proxy，默认 18000）

**— 不由上层传**：`dp_rank_start`（wings 派生）、`kv_port` 及引擎调优参数（注册表给）。

**🚫 禁止下发**：`--distributed`（会误入 Ray master/worker）；`--gpu-memory-utilization / --max-num-seqs / --max-num-batched-tokens / --block-size / --enable-chunked-prefill / --enable-prefix-caching / max-model-len`（顶掉注册表）。

---

## 6. 一致性自查（4 条）

1. `DP_SIZE_LOCAL × TP_SIZE ≤ 每 pod 卡数`。
2. `Σ(各 pod DP_SIZE_LOCAL) = DP_SIZE`（本角色所有 pod 的 local 之和 = 全局 DP）。
3. `RANK_IP` 逐字在 `NODE_IPS` 内；`NODE_IPS` 顺序 = 节点 rank 序。
4. P、D 两端 `prefill{}` / `decode{}` 完全一致（= 4 个全局拓扑值在所有 pod 一字不差）。
</content>
</invoke>
