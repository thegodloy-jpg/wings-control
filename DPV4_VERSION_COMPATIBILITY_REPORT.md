# DPv4-Flash/Pro 版本兼容性完备检查报告

**报告生成日期**: 2026-05-21  
**检查范围**: wings-control 项目完整版本、配置、拓扑、功能特性  
**适用硬件**: Ascend 910B（A2）、910C（A3）  
**检查深度**: 代码实现 + 配置 + 测试用例 + 文档

---

## 目录
1. [项目版本信息](#项目版本信息)
2. [DPv4-Flash 兼容性分析](#dpv4-flash-兼容性分析)
3. [DPv4-Pro 兼容性分析](#dpv4-pro-兼容性分析)
4. [硬件平台识别与映射](#硬件平台识别与映射)
5. [单双机拓扑支持](#单双机拓扑支持)
6. [功能特性支持矩阵](#功能特性支持矩阵)
7. [版本变化影响评估](#版本变化影响评估)
8. [已知问题与限制](#已知问题与限制)

---

## 项目版本信息

### 当前版本标识

| 项目 | 值 |
|------|-----|
| **项目名称** | wings-control |
| **类型** | Sidecar Launcher（无 pyproject.toml/setup.py） |
| **主要模块** | `wings_control.py` + 适配器 + 配置合并 |
| **vLLM 引擎版本** | 0.11.0+ (通用 vllm_default.json) |
| **Ascend 特定版本** | vLLM-Ascend（同一约束） |
| **默认引擎版本tuple** | `(0, 17)` (version_util.py:22) |
| **配置方式** | 基于硬件 + 模型 + 用户三层合并 |

### 配置版本查询机制

```python
# core/version_util.py (L20-22)
DEFAULT_VERSION_TUPLE = (0, 17)  # 未设置时的默认值
parse_engine_version_tuple()      # 从 ENGINE_VERSION 环境变量解析 (major, minor)
normalize_engine_version()        # 规范化为 "major.minor.0" 格式
```

**影响**: 版本号影响默认配置集合的选择，但 **V4-Flash/Pro 配置独立于版本号**，写死在模型架构层。

---

## DPv4-Flash 兼容性分析

### 配置位置
- **配置文件**: [wings_control/config/defaults/ascend_default.json](../wings_control/config/defaults/ascend_default.json#L201-L251)
- **处理逻辑**: [wings_control/engines/vllm_adapter.py](../wings_control/engines/vllm_adapter.py) 行号范围 `1245-1894`
- **测试覆盖**: [tests/test_dp_topology_sync.py](../tests/test_dp_topology_sync.py) L94-L250

### 身份识别规则

**优先级顺序**（vllm_adapter.py:1289-1306）：
1. 模型名称中包含 `deepseek-v4-flash` / `deepseek_v4_flash` / `deepseekv4flash`
2. 架构验证：必须是 `DeepseekV4ForCausalLM` 或 `DeepSeekV4ForCausalLM`
3. 量化指纹（w8a8 推断）
4. 模型路径或 config.json 中的标识文本

```python
# vllm_adapter.py:1289-1306
def _is_deepseek_v4_flash_params(params, model_info):
    text = _deepseek_v4_identity_text(params, model_info)
    arch_match = _deepseek_v4_arch_matches(params, model_info)
    if "deepseek-v4-flash" in text:
        return arch_match is not False  # 架构非否决
    if arch_match is True:
        return "v4" in text and "flash" in text
    return False
```

### 平台自动检测

**A2 vs A3 判断** (vllm_adapter.py:1245-1287)：

| 条件 | 判定 |
|------|------|
| 环境变量 `WINGS_ASCEND_PLATFORM` 明确指定 | 优先使用 |
| `hardware_info.json` details 中包含 "a2" 或 "910b" | A2 |
| `hardware_info.json` details 中包含 "a3" | A3 |
| V4-Pro 身份确认 | A3（Pro 仅 A3） |
| 默认值 | **A2**（910B 保守选择） |

```python
# vllm_adapter.py:1281-1286
if "a2" in device_name or "910b" in device_name:
    return "a2"
if _is_deepseek_v4_pro_params(params, model_info):
    logger.info("[DeepSeek-V4-Pro] Ascend platform not detected; assuming A3")
    return "a3"
return "a2"  # 默认 A2
```

### 容量与拓扑配置

#### A2 单机（8×910B-64 或 16×910B-32）

| 参数 | 值 | 来源 |
|------|-----|------|
| **tensor_parallel_size** | **8** （锁定） | vllm_adapter.py:1834-1836 |
| **data_parallel_size** | max(1, device_count // 8) | 计算：_compute_deepseek_v4_flash_data_parallel_size |
| **max_model_len** | 65536 | _DEEPSEEK_V4_FLASH_CAPACITY_DEFAULTS |
| **max_num_batched_tokens** | 8192 | 同上 |
| **max_num_seqs** | 16 | 同上 |
| **gpu_memory_utilization** | 0.9 | 同上 |
| **enable_expert_parallel** | **True** （强制） | vllm_adapter.py:1840，MoE 必需 |
| **quantization** | "ascend" | vllm_adapter.py:1852 |

**A2 8卡单机结果**:
```
device_count=8 → DP = max(1, 8/8) = 1
启动参数: --tensor-parallel-size 8 --data-parallel-size 1
```

**A2 16卡单机结果**:
```
device_count=16 → DP = max(1, 16/8) = 2
启动参数: --tensor-parallel-size 8 --data-parallel-size 2
```

#### A3 单机（16×910C）

```
device_count=16 → DP = max(1, 16/8) = 2
启动参数: --tensor-parallel-size 8 --data-parallel-size 2
```

#### A3 双机（16×910C per node）

```
device_count=16, nnodes=2, distributed=True
total_cards = 16 * 2 = 32 → DP = max(1, 32/8) = 4
启动参数: --tensor-parallel-size 8 --data-parallel-size 4
```

### 环境变量注入

**共同环境变量**（vllm_adapter.py:1450-1457）：
```bash
export USE_MULTI_BLOCK_POOL=1
export OMP_PROC_BIND=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export USE_MULTI_GROUPS_KV_CACHE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

**A3 特定环境变量**（vllm_adapter.py:1458-1465）：
```bash
export OMP_NUM_THREADS=10
export ASCEND_A3_ENABLE=1
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
```

**A2 特定环境变量**（vllm_adapter.py:1468）：
```bash
export OMP_NUM_THREADS=8
```

**KV 缓存处理**：
- 启用 CPU Offload 时触发：`LMCACHE_OFFLOAD=true`
- 使用 CPUOffloadingConnector（不是 LMCache）
- CPU 交换空间：`LMCACHE_MAX_LOCAL_CPU_SIZE`，默认 200GB

### JSON 配置差异

#### A2 特定 additional_config
```json
{
  "enable_cpu_binding": true,
  "multistream_overlap_shared_expert": false
}
```

#### A3 特定 additional_config
```json
{
  "enable_cpu_binding": true,
  "multistream_overlap_shared_expert": false,
  "multistream_dsa_preprocess": false,
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_static_kernel": false
  }
}
```

---

## DPv4-Pro 兼容性分析

### 配置位置
- **配置文件**: [ascend_default.json](../wings_control/config/defaults/ascend_default.json#L253-L282)
- **处理逻辑**: [vllm_adapter.py](../wings_control/engines/vllm_adapter.py) 行号 `1368-1982`
- **测试覆盖**: [test_dp_topology_sync.py](../tests/test_dp_topology_sync.py) L117-L200+

### 身份识别规则

**严格限制版本**（vllm_adapter.py:1368-1400）：

| 识别源 | 判断逻辑 |
|-------|---------|
| 模型名称 | `deepseek-v4-pro` / `deepseek_v4_pro` / `deepseekv4pro` 任一匹配 |
| 架构 | **必须** `DeepseekV4ForCausalLM` 或 `DeepSeekV4ForCausalLM` |
| 量化指纹 | **w4a8** 量化（V4-Flash 是 w8a8） |
| 互斥性 | 名称含 "flash" 时优先返回 False，交给 Flash 处理 |

```python
# vllm_adapter.py:1378-1399
text = _deepseek_v4_identity_text(params, model_info)
if "flash" in text:
    return False  # Flash 优先，不进 Pro 路径

if "deepseek-v4-pro" in text:
    return _deepseek_v4_arch_matches(...) is not False

# 量化指纹兜底
if arch_is_deepseek_v4:
    quantize = model_info.model_quantize or _extract_quantize_from_config(...)
    if _is_w4a8_quantize(quantize):
        return True
return False
```

### 适配范围闸门

**V4-Pro 仅在以下场景注入专属默认**（vllm_adapter.py:1309-1326）：

```python
def _is_deepseek_v4_pro_adapted_scope(params):
    # 1. 引擎必须是 vllm_ascend
    if params.get("engine") != "vllm_ascend":
        return False
    # 2. 模型身份必须识别为 V4-Pro
    if not _is_deepseek_v4_pro_params(params):
        return False
    # 3. 平台必须是 A3
    if _resolve_deepseek_v4_flash_platform(...) != "a3":
        return False
    # 4. 必须是分布式（distributed=True）
    if not bool(params.get("distributed")):
        return False
    # 5. 节点数必须恰好是 2（nnodes==2）
    nnodes = _safe_int(params.get("nnodes")) or 0
    return nnodes == 2
```

### 容量与拓扑配置

**V4-Pro 双机 A3 固定配置**（vllm_adapter.py:1896-1945）：

| 参数 | 值 | 用途 |
|------|-----|------|
| **tensor_parallel_size** | **16** | 两机共 32 卡切分 |
| **data_parallel_size** | **2** | 两个节点并行 |
| **data_parallel_size_local** | **1** | 单节点单 DP rank |
| **data_parallel_start_rank** | node_rank | 节点 0→start=0，节点 1→start=1 |
| **max_model_len** | 135000 | 长上下文支持 |
| **max_num_batched_tokens** | 4096 | 受限于 w4a8 显存 |
| **max_num_seqs** | 16 | 并发请求数 |
| **gpu_memory_utilization** | 0.9 | 显存利用率 |
| **enable_expert_parallel** | True | MoE 必需 |
| **quantization** | "ascend" | W4A8 量化 |

**启动参数示例**：
```bash
node0: --tensor-parallel-size 16 --data-parallel-size 2 --data-parallel-start-rank 0
node1: --tensor-parallel-size 16 --data-parallel-size 2 --data-parallel-start-rank 1
```

### 推测解码配置

```json
{
  "speculative_config": {
    "num_speculative_tokens": 1,
    "method": "mtp"  // vLLM 0.18+ 要求（deepseek_mtp 被静默忽略）
  }
}
```

### additional_config

```json
{
  "enable_cpu_binding": true,
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_static_kernel": false
  }
}
```

### 限制声明

**明确不支持的场景**：
- ❌ V4-Pro 单机（任何硬件）
- ❌ V4-Pro + A2（910B）
- ❌ V4-Pro + Ray 单机
- ❌ V4-Pro 三节点或更多
- ❌ V4-Pro + NVIDIA GPU

**错误处理**（test_dp_topology_sync.py）：
```python
# 缺少硬件信号时的识别
def test_v4_pro_identity_can_come_from_quantize_field(self):
    """当 model_name/hardware_info 都不包含 Pro 标记，
    但 config.json 有 w4a8 量化时，应正确识别为 V4-Pro"""
    # 否则 DP resolver 会抛 "DeepSeek Ascend DP requires positive tensor_parallel_size"
```

---

## 硬件平台识别与映射

### 硬件探测机制

**优先级**（hardware_detect.py）：
1. **JSON 文件** → `WINGS_HARDWARE_FILE` 或 `/shared-volume/hardware_info.json`
2. **环境变量** → `WINGS_DEVICE` / `DEVICE` / `HARDWARE_TYPE`
3. **默认值** → 无法探测时回退 `nvidia` (GPU)

### 910B vs 910C 识别

| 硬件 | Ascend 平台 | 标准检测字符串 | 显存 | 互联 |
|------|-----------|--------------|------|------|
| **Ascend 910B** | **A2** | "910b" / "a2" | 32/64GB | HCCL |
| **Ascend 910C** | **A3** | "910c" / "a3" | TBD | HCCL |

### 硬件识别代码

```python
# vllm_adapter.py:1280-1286
device_name = (detail.get("name", "") or "").lower() if device_details else ""

if "a2" in device_name or "910b" in device_name:
    return "a2"
if _is_deepseek_v4_pro_params(params):
    return "a3"
return "a2"  # 默认保守值
```

### 配置融合顺序

```
硬件检测 → 设备类型确定(ascend)
       ↓
模型识别 → DeepSeek-V4-Flash/Pro?
       ↓
引擎选择 → vllm_ascend（自动）
       ↓
平台判定 → A2 or A3?
       ↓
专属默认注入 → TP/DP/additional_config/env vars
       ↓
拓扑同步 → 回写 params["engine_config"]
```

---

## 单双机拓扑支持

### 拓扑同步机制

**核心函数**（vllm_adapter.py:1692-1719）：

```python
def _prepare_engine_config(params):
    """执行顺序必须固定：
    1. 删除内部字段
    2. 注入 V4/GLM/DeepSeek 专属默认
    3. CPU offload 处理
    4. DP 拓扑回写（最后）
    """
    engine_config = dict(params.get("engine_config", {}))
    _strip_internal_engine_config_keys(params, engine_config)
    explicit_keys = set(params.get("_explicit_cli_keys") or [])
    
    # V4 Flash 优先于 Pro（Flash 在前）
    _apply_deepseek_v4_flash_engine_defaults(params, engine_config, explicit_keys)
    _apply_deepseek_v4_pro_engine_defaults(params, engine_config, explicit_keys)
    _apply_deepseek_v4_cpu_offload(engine_config, explicit_keys)
    _apply_glm5_ascend_engine_defaults(params, engine_config, explicit_keys)
    _apply_generic_deepseek_ascend_dp_defaults(params, engine_config, explicit_keys)
    
    _writeback_dp_topology_to_params(params, engine_config)  # 关键步骤
    return engine_config
```

### V4-Flash 多机场景

**DP 计算公式**（vllm_adapter.py:1806-1818）：

```python
def _compute_deepseek_v4_flash_data_parallel_size(params):
    device_count = params.get("device_count") or 8
    is_distributed = bool(params.get("distributed"))
    nnodes = params.get("nnodes") or (2 if is_distributed else 1)
    
    total_cards = device_count * (nnodes if is_distributed else 1)
    return max(1, total_cards // 8)  # TP=8 固定，剩余全 DP
```

**示例计算**:

| 场景 | device_count | distributed | nnodes | total_cards | DP 结果 |
|------|-------------|------------|--------|------------|---------|
| A2 单机 | 8 | False | 1 | 8 | 1 |
| A3 单机 | 16 | False | 1 | 16 | 2 |
| A3 双机 | 16 | True | 2 | 32 | 4 |
| A3 三机 | 16 | True | 3 | 48 | 6 |

### V4-Pro 双机限制

**测试覆盖**（test_dp_topology_sync.py:117-200）：

```python
def test_v4_pro_dual_node_a3_syncs_tp_and_dp(self):
    """V4-Pro 双机 A3：TP=16、DP=2 必须被回写到 params["engine_config"]"""
    params = {
        "model_name": "DeepSeek-V4-Pro-w4a8-mtp1",
        "engine": "vllm_ascend",
        "distributed": True,
        "nnodes": 2,
        "node_rank": 0,  # 节点 0
        "device_count": 16,
        "device_details": [{"name": "910c"}],
        "distributed_executor_backend": "dp_deployment",
        "engine_config": {},
    }
    _prepare_engine_config(params)
    
    # 验证
    ec = params["engine_config"]
    assert ec.get("tensor_parallel_size") == 16
    assert ec.get("data_parallel_size") == 2
    assert ec.get("data_parallel_start_rank") == 0  # node_rank=0
```

### 分布式拓扑同步故障排查

**历史问题**（test_dp_topology_sync.py 注释 L16-22）：

```
v4_proto_single_machine_a3_16card_crashed:
  ValueError: DeepSeek Ascend DP requires a positive tensor_parallel_size

根因：
  _apply_deepseek_v4_pro_engine_defaults() 只写局部 engine_config，
  但 _resolve_dp_deployment_topology() 从 params["engine_config"]["tensor_parallel_size"] 读，
  导致 None → DP resolver 崩溃。
  
修复：
  _writeback_dp_topology_to_params() 必须在 _prepare_engine_config 最后调用，
  确保 TP/DP 被同步回 params["engine_config"]。
```

---

## 功能特性支持矩阵

### V4-Flash 功能特性

| 特性 | A2 单机 | A2 多机 | A3 单机 | A3 双机+ | 备注 |
|------|---------|---------|---------|----------|------|
| **Tensor Parallel** | ✅ TP=8 | ✅ TP=8 | ✅ TP=8 | ✅ TP=8 | 硬锁定，MoE 切分 |
| **Data Parallel** | ✅ DP=1 | ✅ DP>=1 | ✅ DP=2 | ✅ DP=4+ | 自动计算 (cards/8) |
| **Expert Parallel** | ✅ True | ✅ True | ✅ True | ✅ True | 强制开启，MoE 必需 |
| **Speculative Decoding** | ✅ MTP-1 | ✅ MTP-1 | ✅ MTP-1 | ✅ MTP-1 | method="mtp" |
| **Prefix Caching** | ✅ 开 | ✅ 开 | ✅ 开 | ✅ 开 | A3+ 推荐开启 |
| **Chunked Prefill** | ✅ 开 | ✅ 开 | ✅ 开 | ✅ 开 | 8192 token 批 |
| **KV Offload** | ✅ CPU | ✅ CPU | ✅ CPU | ✅ CPU | CPUOffloadingConnector |
| **Chat Template** | ✅ 有 | ✅ 有 | ✅ 有 | ✅ 有 | /usr/local/serving/models/chat_template.jinja |
| **Tool Call** | ✅ deepseek_v4 | ✅ 同 | ✅ 同 | ✅ 同 | enable_auto_tool_choice=True |

### V4-Pro 功能特性

| 特性 | A3 双机 | 备注 |
|------|---------|------|
| **Tensor Parallel** | ✅ TP=16 | 2×16 卡配置 |
| **Data Parallel** | ✅ DP=2 | 节点间 1 rank/卡 |
| **Expert Parallel** | ✅ True | MoE 必需 |
| **Speculative Decoding** | ✅ MTP-1 | method="mtp" |
| **Max Context** | ✅ 135000 | vs Flash 的 65536 |
| **Quantization** | ✅ W4A8 | Flash 是 W8A8 |
| **KV Offload** | ✅ CPU | CPUOffloadingConnector |
| **Tool Call** | ✅ deepseek_v4 | 同 Flash |

### 版本约束

| 特性 | 约束 | 实现位置 |
|------|------|---------|
| vLLM 版本 | 0.11.0+ | README.md 官方支持 |
| vLLM 推测解码 | 0.18+ 仅支持 method="mtp" | vllm_adapter.py:1885-1889 |
| Ascend 平台库 | A2/A3 通用 HCCL | 环境变量注入 |
| Python | 3.9+ | 项目默认 |

---

## 版本变化影响评估

### 历史变更轨迹

**最近 20 次 commit 中涉及 V4 的关键提交**：

| Commit | 描述 | 影响范围 |
|--------|------|----------|
| c907684 | Add DeepSeek-V4 Flash vllm ascend | ✅ 初始 V4-Flash 支持 |
| 585929e | Guard V4-Pro Ascend scope | ✅ V4-Pro 双机 A3 限制 |
| 2307788 | Align DeepSeek-V4 Ascend defaults | ✅ 配置对齐 |
| ebb8abf | Refactor V4 identity resolution，fix V4-Pro MTP1 | ✅ 身份识别完善 |
| 758a83d | Unify DeepSeek-V4 Flash/Pro CPU KV offload | ✅ KV 管理统一 |

**关键改进**：
1. ✅ V4-Flash A2/A3 环境变量差异化（OMP_NUM_THREADS、A3_ENABLE 等）
2. ✅ V4-Pro 双机拓扑同步（TP 回写到 engine_config）
3. ✅ V4 身份识别容错（量化指纹、config.json 备用）
4. ✅ CPU KV offload 统一为 CPUOffloadingConnector

### 不同版本影响

**当前版本(0.17) vs 旧版的差异**:

| 场景 | 旧版 | 当前(0.17) | 影响 |
|------|------|-----------|------|
| V4-Pro 双机单卡 | ❌ 无 | ✅ 支持（DP=2） | 显著提升 |
| V4-Flash A2/A3 差异 | ⚠️ 环境变量不一致 | ✅ 完整差异化 | 稳定性提升 |
| 硬件自动检测 | ⚠️ 需 WINGS_HARDWARE_FILE | ✅ 备用环境变量 | 易用性提升 |
| V4 身份识别 | ⚠️ 容错不足 | ✅ 量化指纹识别 | 可靠性提升 |

**升级建议**：
- ✅ 已在 0.17 版本支持 V4-Flash/Pro
- ✅ 无需升级即可使用全部功能
- ⚠️ 若遇到 V4-Pro 拓扑识别问题，确认是否运行了最新 vllm_adapter.py

---

## 已知问题与限制

### 硬件平台限制

**910B（A2）硬件**：
- ✅ V4-Flash 完全支持
- ❌ V4-Pro **不支持**（需 A3 平台）
- ⚠️ V4-Flash 单机最多 16 卡（device_count=16 → DP=2）

**910C（A3）硬件**：
- ✅ V4-Flash 全支持
- ✅ V4-Pro 双机支持
- ⚠️ V4-Pro 单机 A3 **不进入** Pro 路径（需 distributed=True + nnodes=2）

### 拓扑限制

**V4-Flash 拓扑**：
- ✅ 任意卡数（8 的倍数最优：TP=8 整除）
- ✅ 单机/多机通用
- ⚠️ TP=8 硬锁定（TP=16 导致 MTP 维度 0 崩溃）

**V4-Pro 拓扑**：
- ✅ 仅 A3 双机（nnodes==2）
- ✅ TP=16、DP=2、DP_local=1
- ❌ 单机（任何卡数）
- ❌ 三节点+
- ❌ A2 平台
- ❌ Ray 单机模式

**错误示例**：
```python
# ❌ V4-Pro 单机 A3（16 卡）
params = {
    "model_name": "DeepSeek-V4-Pro-w4a8",
    "distributed": False,  # 单机
    "engine": "vllm_ascend",
}
# 结果：Pro defaults 不被注入，TP 为空 → DP resolver 崩溃

# ✅ 改为 Flash（单机配置）
params = {
    "model_name": "DeepSeek-V4-Flash",
    "distributed": False,
    "engine": "vllm_ascend",
}
# 结果：自动识别 Flash，TP=8、DP=2（16/8）
```

### 功能特性限制

**MoE（Expert Parallel）**：
- ✅ V4-Flash/Pro 都强制 enable_expert_parallel=True
- ⚠️ 用户显式 `--enable-expert-parallel=false` 会被覆盖
- ❌ 禁用 EP 会导致 MoE+MTP 路径崩溃（KV cache spec 不一致）

**推测解码（Speculative Decoding）**：
- ✅ MTP-1（单 token 推测）
- ❌ vLLM 0.17 不支持更高级别 MTP（vLLM 0.18+ 可 MTP-N）

**量化**：
- ✅ V4-Flash：W8A8（默认）
- ✅ V4-Pro：W4A8（w4a8 指纹识别）
- ⚠️ BF16 权重：无特殊优化，走通用路径

### 配置冲突

**LMCache vs CPU Offload**：
```python
# 互斥关系（vllm_adapter.py:1994-1995）
if LMCACHE_OFFLOAD=true and V4-Flash/Pro:
    # 使用 CPUOffloadingConnector，跳过 LMCache YAML
    # 两者不会同时生效
```

**用户显式参数**：
```python
# explicit_keys 中已有的参数优先保留
if "tensor_parallel_size" in explicit_keys:
    # 用户显式指定 TP，不覆盖
else:
    # Flash: TP=8（强制）
    # Pro: TP=16（强制）
```

### 硬件信息缺失时的降级

**缺失硬件 details**：
```python
# test_dp_topology_sync.py:145-156
device_details = []  # 无硬件信息

# V4-Flash：默认 A2，计算 TP/DP ✅
# V4-Pro：应识别为 Pro（量化指纹 w4a8），但平台兜底为 A2
#        → _is_deepseek_v4_pro_adapted_scope 检查 A3 失败 ❌
#        → Pro defaults 不注入，DP resolver 抛错

# 修复：显式声明 WINGS_ASCEND_PLATFORM=a3 或提供 hardware_info.json
```

### GLM-5.1 特殊行为

**GLM-5.1 在 Ascend 路径的强制 EP 关闭**（vllm_adapter.py:2077-2098）：
```python
# GLM-5.1 与 V4 无直接关联，但同属 MoE 模型
# 因社区已知不稳定（vllm-ascend#8015），EP 被强制关闭
enable_expert_parallel = False  # 覆盖用户配置
```

---

## 总结与建议

### 当前状态评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **V4-Flash 支持** | ⭐⭐⭐⭐⭐ | 完整支持 A2/A3 单双机 |
| **V4-Pro 支持** | ⭐⭐⭐⭐ | 完整支持 A3 双机，限制明确 |
| **硬件自动识别** | ⭐⭐⭐⭐ | 容错完善，量化指纹识别 |
| **拓扑自动计算** | ⭐⭐⭐⭐⭐ | TP/DP 回写、DP 公式清晰 |
| **文档完整性** | ⭐⭐⭐ | 代码注释详尽，文档需补充 |

### 部署清单

#### V4-Flash 单机（910B A2，8 卡）

```bash
# 环境变量
export WINGS_DEVICE="ascend"
export WINGS_DEVICE_COUNT="8"
export WINGS_DEVICE_NAME="Ascend910B"

# 启动参数
python -m wings_control \
  --model-name "DeepSeek-V4-Flash" \
  --model-path /models/DeepSeek-V4-Flash \
  --engine vllm_ascend

# 预期：TP=8, DP=1
```

#### V4-Flash 单机（910C A3，16 卡）

```bash
export WINGS_DEVICE="ascend"
export WINGS_DEVICE_COUNT="16"
export WINGS_DEVICE_NAME="Ascend910C"

# 预期：TP=8, DP=2
```

#### V4-Pro 双机（910C A3，16 卡/节点）

```bash
# 节点 0
export WINGS_DEVICE="ascend"
export WINGS_DEVICE_COUNT="16"
export NODE_RANK="0"
export NNODES="2"

python -m wings_control \
  --model-name "DeepSeek-V4-Pro-w4a8-mtp1" \
  --model-path /models/DeepSeek-V4-Pro-w4a8-mtp1 \
  --engine vllm_ascend \
  --distributed

# 节点 1：NODE_RANK="1"

# 预期：TP=16, DP=2, DP_start_rank={0|1}
```

### 常见错误排查

| 错误 | 原因 | 解决 |
|------|------|------|
| V4-Pro 拓扑崩溃 | nnodes=1（单机） | 改为双机或用 V4-Flash |
| DP resolver "positive TP" | 硬件 details 缺失 + 量化无识别 | 提供 hardware_info.json 或 WINGS_ASCEND_PLATFORM |
| MoE+MTP 'list' merge 失败 | EP=False（被覆盖） | 确保 enable_expert_parallel=True |
| A3 环境变量未注入 | 平台判定为 A2 | 检查 device_details 或显式 WINGS_ASCEND_PLATFORM=a3 |

### 版本升级影响

**当前版本(0.17) 到未来版本的预期变更**：

- ⏳ vLLM 0.18 升级：推测解码 MTP-N 支持（目前 MTP-1）
- ⏳ 910C 完整验证：当前为 experimental，需生产级测试
- ⏳ A2 性能优化：OMP 线程数、HCCL buffer 大小调优
- ⏳ V4 w4a8 路径完善：当前基于量化指纹，未来或有官方权重标记

---

**报告维护者**: wings-control 开发组  
**更新频率**: 每月或版本变更时  
**反馈渠道**: Git Issue / Pull Request
