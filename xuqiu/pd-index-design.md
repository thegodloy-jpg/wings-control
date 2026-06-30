# PD_INDEX 设计文档

> 状态：方案分析，未实施

---

## 1. 核心诉求

```
PD_INDEX 是跨 P/D 共用的全局连续实例编号。

  P 实例在前：PD_INDEX = 0, 1, …, P_count-1
  D 实例在后：PD_INDEX = P_count, …, P_count + D_count - 1

PD_ROLE + PD_INDEX 唯一标识任意实例。
engine_id 和 kv_port 都从 PD_INDEX 派生。
```

**三个要点：**

| 要点 | 含义 |
|------|------|
| **跨角色共用** | P 和 D 在同一套序号空间，不各自从 0 数 |
| **连续唯一** | 0, 1, 2, … 不间断、不重复 |
| **同源派生** | engine_id 和 kv_port 来自同一个 PD_INDEX |

---

## 2. 派生规则

```
PD_INDEX  = pd_index_offset + dp_rank_start + i

  pd_index_offset:  P = 0,  D = PD_PREFILL_DP_SIZE
  dp_rank_start:    本节点在角色内的起始 rank
  i:                节点内 fork 序号（0 … dp_size_local-1）

engine_id = str(PD_INDEX)
kv_port   = 30000 + PD_INDEX          # 统一基址，不再分 P/D
```

**与实例数无关——1P1D 到多 P 多 D，同一条公式：**

```
1P1D（P dp=1, D dp=1）：
  P: PD_INDEX = 0 + (0 + 0) = 0
  D: PD_INDEX = 1 + (0 + 0) = 1

多P多D（P dp=2, D dp=4）：
  P: PD_INDEX = 0 + (start + i)  →  0, 1
  D: PD_INDEX = 2 + (start + i)  →  2, 3, 4, 5
```

**MooncakeConnector(kv_p2p) 除外**——它的 engine_id/kv_port 是官方 role 常量（`"0"/"1"`, `30000/30100`），不走 PD_INDEX。

---

## 3. 具体场景实例

### 场景 1：qwen3-1p1d（1P1D）

P: dp1×tp4 / D: dp1×tp4

```
PD_ROLE  PD_INDEX  engine_id  kv_port
───────  ────────  ────────   ───────
  P         0        "0"       30000
  D         1        "1"       30001
```

### 场景 2：qwen3（P: dp2×tp2 / D: dp4×tp1）

P 1 节点 2 实例，D 2 节点每节点 2 实例（共 6 实例）

```
PD_ROLE  PD_INDEX  所在节点    engine_id  kv_port
───────  ────────  ──────────  ────────   ───────
  P         0       9.0.0.1      "0"       30000
  P         1       9.0.0.1      "1"       30001
  D         2       9.0.1.1      "2"       30002
  D         3       9.0.1.1      "3"       30003
  D         4       9.0.1.2      "4"       30004
  D         5       9.0.1.2      "5"       30005
```

### 场景 3：glm5（P: dp2×tp16 / D: dp16×tp4）

P 2 节点 2 实例，D 4 节点 16 实例（共 18 实例）

```
PD_ROLE  PD_INDEX  所在节点    engine_id  kv_port
───────  ────────  ──────────  ────────   ───────
  P         0       7.0.0.1      "0"       30000
  P         1       7.0.0.2      "1"       30001
  D         2       7.0.1.1      "2"       30002
  D         3       7.0.1.1      "3"       30003
  D         4       7.0.1.1      "4"       30004
  D         5       7.0.1.1      "5"       30005
  D         6       7.0.1.2      "6"       30006
  …         …          …          …          …
  D        17       7.0.1.4     "17"       30017
```

### 场景 4：glm52-a2（P: dp4×tp8 / D: dp8×tp4）

P 4 节点 4 实例，D 4 节点 8 实例（共 12 实例）

```
PD_ROLE  PD_INDEX  所在节点    engine_id  kv_port
───────  ────────  ──────────  ────────   ───────
  P         0       7.0.0.1      "0"       30000
  P         1       7.0.0.2      "1"       30001
  P         2       7.0.0.3      "2"       30002
  P         3       7.0.0.4      "3"       30003
  D         4       7.0.1.1      "4"       30004
  D         5       7.0.1.1      "5"       30005
  D         6       7.0.1.2      "6"       30006
  D         7       7.0.1.2      "7"       30007
  D         8       7.0.1.3      "8"       30008
  D         9       7.0.1.3      "9"       30009
  D        10       7.0.1.4     "10"       30010
  D        11       7.0.1.4     "11"       30011
```

### 场景 5：v4flash（P: dp4×tp4 / D: dp16×tp1）

P 1 节点 4 实例，D 1 节点 16 实例（共 20 实例）

```
PD_ROLE  PD_INDEX  所在节点    engine_id  kv_port
───────  ────────  ──────────  ────────   ───────
  P         0       8.0.0.1      "0"       30000
  P         1       8.0.0.1      "1"       30001
  P         2       8.0.0.1      "2"       30002
  P         3       8.0.0.1      "3"       30003
  D         4       8.0.1.1      "4"       30004
  D         5       8.0.1.1      "5"       30005
   ⋮         ⋮          ⋮          ⋮          ⋮
  D        18       8.0.1.1     "18"       30018
  D        19       8.0.1.1     "19"       30019
```

---

## 4. 代码改造

涉及 2 个文件、7 处变更。

### 4.1 config_loader.py

#### 变更 1：`_get_pd_external_lb_params` — 新增 `pd_index_offset`

**文件**：[config_loader.py:1005-1013](wings_control/core/config_loader.py#L1005-L1013)

```python
# 改前
return {
    "role": role,
    "dp_size": dp_size,
    "tp_size": tp_size,
    "dp_size_local": dp_size_local,
    "dp_rank_start": dp_rank_start,
    "dp_address": dp_address,
    "rpc_port": str(rpc_port),
}

# 改后
# PD_INDEX 偏移：P 从 0 开始，D 从 P 总实例数之后开始
try:
    p_total = int(os.getenv("PD_PREFILL_DP_SIZE", "0") or "0")
except (ValueError, TypeError):
    p_total = 0
pd_index_offset = 0 if role == "P" else p_total

return {
    "role": role,
    "dp_size": dp_size,
    "tp_size": tp_size,
    "dp_size_local": dp_size_local,
    "dp_rank_start": dp_rank_start,
    "dp_address": dp_address,
    "rpc_port": str(rpc_port),
    "pd_index_offset": pd_index_offset,   # 新增
}
```

#### 变更 2：`_apply_pd_external_lb` — 移除 per-role kv_port_base

**文件**：[config_loader.py:1187-1196](wings_control/core/config_loader.py#L1187-L1196)

```python
# 改前
# 端口偏移基址：fork 脚本按 base + 本地 i 给每个 service 算独立 kv_port / bootstrap_port
ext["kv_port_base"] = int(entry["kv_port"][role])
ext["bootstrap_base"] = int(
    os.getenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", "23000" if role == "P" else "23100")
)
ext["connector"] = entry["connector"]

# 改后
# kv_port 统一从 30000 + PD_INDEX 派生，不再按 P/D 分基址。
# pd_config.json 中 kv_port 字段保留但不再读取（向后兼容注册表格式）。
ext["bootstrap_base"] = int(
    os.getenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", "23000" if role == "P" else "23100")
)
ext["connector"] = entry["connector"]
```

#### 变更 3：`_build_pd_external_lb_kv` — `__PD_RANK__` → `__PD_INDEX__`

**文件**：[config_loader.py:1087-1094](wings_control/core/config_loader.py#L1087-L1094)

```python
# 改前
if entry["connector"] == "MooncakeConnector":
    cfg["engine_id"] = "0" if role == "P" else "1"
elif entry["connector"] in ("MooncakeConnectorV1", "MooncakeHybridConnector"):
    cfg["engine_id"] = "__PD_RANK__"

# 改后
if entry["connector"] == "MooncakeConnector":
    cfg["engine_id"] = "0" if role == "P" else "1"
else:
    cfg["engine_id"] = "__PD_INDEX__"
```

注释一并更新：
```python
# 改前
# kv_port 按 service 偏移（base + 本地 i），避免单 pod 多 service 抢同一端口；
# 占位符由 fork 脚本（vllm_adapter）按 base + i 替换。base 见 _apply_pd_external_lb。

# 改后
# kv_port 占位符由 fork 脚本按 30000 + PD_INDEX 替换（全局连续唯一）。
```

### 4.2 vllm_adapter.py

#### 变更 4：读取 `pd_index_offset`

**文件**：[vllm_adapter.py:2920-2945](wings_control/engines/vllm_adapter.py#L2920-L2945)

```python
# 改前
start = pd_ext["dp_rank_start"]
# ...
kv_base = pd_ext.get("kv_port_base", 30000)

# 改后
start = pd_ext["dp_rank_start"]
pd_index_offset = pd_ext.get("pd_index_offset", 0)   # 新增
# kv_base 不再需要，改为固定 30000
```

#### 变更 5：占位符替换 — `__PD_RANK__` → `__PD_INDEX__`，kv_port 字面值固定 30000

**文件**：[vllm_adapter.py:2942-2952](wings_control/engines/vllm_adapter.py#L2942-L2952)

```python
# 改前
# 占位符 → 让 bash 在单引号 JSON 内展开 shell 变量（engine_id 按 rank，kv_port 按连接器分叉）：
#   MooncakeConnector(kv_p2p): kv_port 是 per-role 标识符，用字面值不引入 shell 变量
#   V1 / Hybrid: kv_port 按 base+i 自增，每 service 独立端口
kv_base = pd_ext.get("kv_port_base", 30000)
bootstrap_base = pd_ext.get("bootstrap_base", 23000)
svc_cmd = svc_cmd.replace("__PD_RANK__", "'\"$RANK\"'")
connector = pd_ext.get("connector", "")
if connector == "MooncakeConnector":
    svc_cmd = svc_cmd.replace("__PD_KVPORT__", str(kv_base))
else:
    svc_cmd = svc_cmd.replace("__PD_KVPORT__", "'\"$KVPORT\"'")

# 改后
# 占位符 → 让 bash 在单引号 JSON 内展开 shell 变量：
#   engine_id 按 PD_INDEX（跨 P/D 全局连续）
#   kv_port 按 30000 + PD_INDEX 统一派生
#   MooncakeConnector(kv_p2p): 保持 role 级字面值，不走 PD_INDEX
bootstrap_base = pd_ext.get("bootstrap_base", 23000)
svc_cmd = svc_cmd.replace("__PD_INDEX__", "'\"$PD_INDEX\"'")
connector = pd_ext.get("connector", "")
if connector == "MooncakeConnector":
    svc_cmd = svc_cmd.replace("__PD_KVPORT__", "30000" if pd_ext.get("role") == "P" else "30100")
else:
    svc_cmd = svc_cmd.replace("__PD_KVPORT__", "'\"$KVPORT\"'")
```

#### 变更 6：fork 循环引入 `$PD_INDEX`，`$KVPORT` 改为 `30000 + PD_INDEX`

**文件**：[vllm_adapter.py:2993-3003](wings_control/engines/vllm_adapter.py#L2993-L3003)

```bash
# 改前
fork_body = [
    "(",
    "  pids=()",
    f"  for i in $(seq 0 {local - 1}); do",
    f"    RANK=$(({start} + i)); PORT=$(({base_port} + i))",
]
if connector == "MooncakeConnector":
    fork_body.append(f"    BOOTSTRAP=$(({bootstrap_base} + i))")
else:
    fork_body.append(f"    KVPORT=$(({kv_base} + i)); BOOTSTRAP=$(({bootstrap_base} + i))")

# 改后
fork_body = [
    "(",
    "  pids=()",
    f"  for i in $(seq 0 {local - 1}); do",
    f"    RANK=$(({start} + i)); PORT=$(({base_port} + i))",
    f"    PD_INDEX=$(({pd_index_offset} + RANK))",
]
if connector == "MooncakeConnector":
    fork_body.append(f"    BOOTSTRAP=$(({bootstrap_base} + i))")
else:
    fork_body.append(f"    KVPORT=$((30000 + PD_INDEX)); BOOTSTRAP=$(({bootstrap_base} + i))")
```

#### 变更 7：dp_size=1 路径

**文件**：[vllm_adapter.py:2984-2991](wings_control/engines/vllm_adapter.py#L2984-L2991)

```python
# 改前
if dp_size == 1:
    svc_cmd = svc_cmd.replace("'\"$RANK\"'", "0")
    svc_cmd = svc_cmd.replace("'\"$KVPORT\"'", str(kv_base))
    # ...

# 改后
if dp_size == 1:
    # PD_INDEX = pd_index_offset + 0（单实例，无 fork 循环，直接字面值）
    _pd_idx = str(pd_index_offset)
    _kvp = str(30000 + pd_index_offset)
    svc_cmd = svc_cmd.replace("'\"$PD_INDEX\"'", _pd_idx)
    if connector != "MooncakeConnector":
        svc_cmd = svc_cmd.replace("'\"$KVPORT\"'", _kvp)
    # ...
```

---

## 5. 兼容性

**实际部署模式**：V1/Hybrid 只有 1P1D，多 P 多 D 走 MooncakeConnector（不动）。

| 场景 | connector | 改造前 engine_id | 改造后 engine_id | 改造前 kv_port | 改造后 kv_port |
|------|-----------|-----------------|-----------------|---------------|---------------|
| 1P1D P | V1/Hybrid | `"0"` | `"0"` | `30000` | `30000` |
| 1P1D D | V1/Hybrid | `"0"` | `"1"` | `30100` | `30001` |
| 多P多D | Mooncake  | `"0"`/`"1"` 字面值 | 不变 | `30000`/`30100` 字面值 | 不变 |

**1P1D D 实例变化**：engine_id 从 `"0"` 变为 `"1"`，kv_port 从 `30100` 变为 `30001`——因为 PD_INDEX 跨角色连续（P=0, D=1），基址统一为 30000。

---

## 6. MooncakeConnector 不变

| 字段 | 值 | 原因 |
|------|-----|------|
| `engine_id` | `"0"`(P) / `"1"`(D) | 官方 kv_p2p 的 producer/consumer 标签 |
| `kv_port` | `30000`(P) / `30100`(D) | role 级常量 |

与实例数无关，不引入 PD_INDEX。

---

## 7. 不改造的部分

| 组件 | 理由 |
|------|------|
| `--data-parallel-rank` | 仍用 `$RANK`（角色内 DP rank），与 PD_INDEX 独立 |
| `--data-parallel-rpc-port` | role 级硬编码 P=12890/D=12777 |
| `--data-parallel-address` | role 级 DP 域 head |
| `--port`（API 端口） | 节点内唯一即可，用 `base_port + i` |
| `$RANK` 变量 | 保留，供 `--data-parallel-rank` 使用 |
| pd_config.json | 格式不变，`kv_port` 字段保留但不再读取 |
| `_build_pd_role_env_commands` | role 级 env，不涉及 per-instance |

---

## 8. 实施步骤

1. **config_loader.py**：3 处变更（pd_index_offset / kv_port_base 移除 / __PD_INDEX__ 占位符）
2. **vllm_adapter.py**：4 处变更（offset 读取 / 占位符替换 / fork 循环 / dp_size=1 路径）
3. **dry_run 验证**：对比改造前后 start_command.sh diff
4. **真机回归**：1P1D GLM5.2 / V4-Flash / Qwen3 验证
