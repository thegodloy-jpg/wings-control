# 高级特性完整数据流文档

> 本文档详细描述 wings-control 中各**高级特性**（Advanced Features）从环境变量传入、
> 配置合并、脚本拼接、运行时补丁，到启动使能和故障回退的完整链路。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [投机推理 (Speculative Decoding)](#2-投机推理-speculative-decoding)
3. [KV 稀疏 (Sparse KV)](#3-kv-稀疏-sparse-kv)
4. [KV 卸载 / PD 分离 (LMCache Offload / PD Disaggregation)](#4-kv-卸载--pd-分离-lmcache-offload--pd-disaggregation)
5. [Soft FP8 / Soft FP4 量化](#5-soft-fp8--soft-fp4-量化)
6. [wings-accel 运行时补丁](#6-wings-accel-运行时补丁)
7. [故障回退与重试机制](#7-故障回退与重试机制)
8. [start_command.sh 最终组装顺序](#8-start_commandsh-最终组装顺序)

---

## 1. 整体架构概览

所有高级特性共享相同的 **4 层管线**：

```
┌──────────────────────────────────────────────────────────────────────┐
│ 第 1 层  平台环境变量注入 (K8s Deployment / ConfigMap)              │
│   SD_ENABLE, LMCACHE_OFFLOAD, ENABLE_SPARSE, PD_ROLE, ...          │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│ 第 2 层  config_loader 配置合并                                      │
│   load_and_merge_configs() → _merge_vllm_params()                   │
│   → _set_kv_cache_config(), _set_spec_decoding_config()                  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│ 第 3 层  引擎适配器脚本生成                                           │
│   vllm_adapter.build_start_script()                                 │
│   → _build_speculative_cmd(), _build_sparse_cmd(), etc.             │
│   → 拼接为 exec python3 -m vllm ... --speculative-config ... 命令  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│ 第 4 层  wings_entry 组装包装                                        │
│   build_launcher_plan()                                              │
│   → preamble (triton_patch / accel_install / env_overrides)         │
│   → script_body (引擎启动命令 + PID 跟踪)                           │
│   → monitor_script (快速失败回退 / 崩溃重试)                         │
│   → 输出最终 start_command.sh                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 关键文件索引

| 文件 | 职责 |
|------|------|
| `core/config_loader.py` | 多层配置合并、参数注入 |
| `engines/vllm_adapter.py` | vLLM/vLLM-Ascend 启动脚本生成 |
| `core/wings_entry.py` | 脚本包装、回退逻辑、accel 补丁 |
| `utils/env_utils.py` | 环境变量读取工具函数 |
| `utils/model_utils.py` | 模型架构识别 |
| `core/version_util.py` | 引擎版本解析 |
| `config/settings.py` | 全局配置 |

---

## 2. 投机推理 (Speculative Decoding)

### 2.1 环境变量

| 环境变量 | 来源 | 读取位置 | 说明 |
|----------|------|----------|------|
| `ENABLE_SPECULATIVE_DECODE` | K8s ConfigMap | CLI 参数解析 → `cmd_known_params` | 用户开关 |
| `SD_ENABLE` | `_set_spec_decoding_config()` 设置 | `env_utils.get_speculative_decoding_env()` | 运行时标志 |
| `SPECULATIVE_DECODE_MODEL_PATH` | K8s ConfigMap | CLI → `cmd_known_params` / `wings_entry._collect_enabled_features()` | 草稿模型路径 |
| `SPECULATIVE_TOKEN_RANGE` | K8s ConfigMap | CLI → `cmd_known_params` | 自适应草稿长度候选，如 `"1,3,5"` |
| `DRAFT_CONFIDENCE_THRESHOLD` | K8s ConfigMap | CLI → `cmd_known_params` | 草稿置信度阈值 (0.0~1.0) |
| `ENABLE_ACCEL` | `config/settings.py` | `settings.ENABLE_ACCEL` | wings-accel 总开关 |

### 2.2 配置合并链路

```
CLI 参数
  │
  ▼
_build_engine_cmd_parameter()                    # config_loader.py L234
  提取 keys: enable_speculative_decode,
             speculative_decode_model_path,
             speculative_token_range,
             draft_confidence_threshold
  │
  ▼
_set_spec_decoding_config(cmd_known_params)       # config_loader.py L121
  if params.get("enable_speculative_decode"):
      os.environ['SD_ENABLE'] = 'true'
  else:
      os.environ['SD_ENABLE'] = 'false'
  │
  ▼
_merge_vllm_params()                              # config_loader.py L284
  _set_common_params() → 通过 engine_parameter_mapping.json 映射参数名
  → 投机推理参数保留在 merged dict 中
```

### 2.3 脚本生成：自动策略选取

`_build_speculative_cmd(params, engine)` — `vllm_adapter.py L979`

**自动选取决策树：**

```
engine ∈ {vllm, vllm_ascend}?
├── 否 → 返回 ""（不支持）
└── 是 → 继续判断
          │
          ├── speculative_decode_model_path 存在?
          │   └── 是 → 读取 ModelIdentifierDraft 识别架构
          │         ├── "eagle3" in architecture → eagle3 方法
          │         │   method="eagle3", num_speculative_tokens=4
          │         │   speculative_token_range=[1,2,4]
          │         │   draft_confidence_threshold=0.8
          │         └── 否 → draft_model 方法
          │             method="draft_model", num_speculative_tokens=4
          │             disable_padded_drafter_batch=true
          │
          ├── Qwen3NextForCausalLM + vllm_ascend?
          │   └── 是 → suffix 方法
          │         method="suffix", num_speculative_tokens=5
          │         suffix_decoding_max_cached_requests=1000
          │
          ├── MTP 支持模型匹配?
          │   ├── DeepseekV3/V32ForCausalLM → "deepseek_mtp"
          │   ├── Qwen3NextForCausalLM → "qwen3_next_mtp"
          │   ├── Glm4MoeForCausalLM → "glm4_moe_mtp"
          │   └── Qwen3_5For*/Qwen3_5MoeFor* → "qwen3_5_mtp"
          │   num_speculative_tokens=1
          │
          └── 全部不匹配 → 兜底 suffix 方法
```

### 2.4 输出格式

```bash
 --speculative-config '{"method": "eagle3", "model": "/path/to/draft",
   "draft_tensor_parallel_size": 1, "num_speculative_tokens": 4,
   "speculative_token_range": [1, 2, 4], "draft_confidence_threshold": 0.8}'
```

### 2.5 拼接到启动命令

**单机模式** — `_build_vllm_single_script()` (vllm_adapter.py L1724):
```python
speculative_extra = _build_speculative_cmd(params, engine) \
    if params.get("enable_speculative_decode") else ""
return env_prefix + f"exec {cmd}{speculative_extra}{sparse_args}\n"
```

**分布式 Ray Head** — `_build_ray_head_commands()` (vllm_adapter.py L1452):
```python
speculative_extra = _build_speculative_cmd(params, ctx.engine) \
    if params.get("enable_speculative_decode") else ""
```

### 2.6 wings-accel 自适应草稿模型集成

当 `ENABLE_ACCEL=true` 且有 `SPECULATIVE_TOKEN_RANGE` 或 `DRAFT_CONFIDENCE_THRESHOLD` 时：

1. `_should_inject_adaptive_draft_fields()` 返回 `True`
2. `_inject_adaptive_draft_fields()` 将 `speculative_token_range`（JSON 列表）和 `draft_confidence_threshold`（浮点数）注入到 `speculative-config` JSON 中
3. wings-accel 的 `adaptive_draft_model_patch` 在 vLLM import 阶段拦截并读取这些字段（在原生解析器之前剥离）

---

## 3. KV 稀疏 (Sparse KV)

### 3.1 环境变量

| 环境变量 | 来源 | 读取位置 | 说明 |
|----------|------|----------|------|
| `ENABLE_SPARSE` | CLI 参数 | `_set_sparse_config()` → `os.environ['SPARSE_ENABLE']` | 用户开关 |
| `SPARSE_ENABLE` | sidecar 设置 | `env_utils.get_sparse_env()` | 运行时标志 |
| `SPARSE_LIB_PATH` | K8s ConfigMap | `vllm_adapter._build_cache_env_commands()` | vsparse 原生库路径 |
| `WINGS_DEVICE_MEMORY` | 硬件探测 | `_estimate_gpu_total_memory_gb()` | GPU 总显存 (GB) |

### 3.2 配置合并

```
_set_sparse_config(params)              # config_loader.py L134
  if params.get("enable_sparse"):
      os.environ['SPARSE_ENABLE'] = 'true'
```

CLI 参数提取的 keys:
- `enable_sparse`, `lc_sparse_threshold`, `total_budget`, `local_kvstore_capacity`

### 3.3 脚本生成：`_build_sparse_cmd()`

`vllm_adapter.py L1239`

**限制条件：**
- 仅 `engine == "vllm"` (NVIDIA)，Ascend 不支持
- 需要 `total_budget > 0`

**生成的参数：**

```bash
 --kv-transfer-config '{"kv_connector": "SparseConnector", "kv_role": "kv_both",
   "kv_connector_module_path": "vsparse.connectors.sparse_connector",
   "kv_connector_extra_config": {"sparse_connectors": [
     {"connector_name": "LocalStoreKVStore", "connector_config": {"capacity": 8192}}
   ]}}'
 --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
 --sparse-config '{"enable_sparse": true, "sparse_algo_type": "BMSA",
   "total_budget": 0.8, "max_num_seqs": 256, "lc_sparse_threshold": 0.3}'
```

### 3.4 容量自动计算

`_resolve_sparse_capacity()` 优先级：
1. 用户显式设置 `local_kvstore_capacity` → 直接使用
2. 根据 `max_model_len / block_size` 估算，再用 GPU 显存约束上限
3. 兜底值 `8192`

`_estimate_memory_limit_capacity()` 逻辑：
```
localstore_budget_gb = gpu_total_gb × total_budget
per_block_gb = (block_size / 16) × (2 / 1024) / tp_size
capacity = localstore_avail_gb / per_block_gb
```

### 3.5 与 LMCache 的冲突处理

`_prepare_engine_config()` (vllm_adapter.py L787)：
当 `enable_sparse=true` 时，从 `engine_config` 中移除 `kv_transfer_config`，避免与 sparse 的 `--kv-transfer-config` 冲突。

---

## 4. KV 卸载 / PD 分离 (LMCache Offload / PD Disaggregation)

### 4.1 环境变量

| 环境变量 | 来源 | 读取位置 | 说明 |
|----------|------|----------|------|
| `LMCACHE_OFFLOAD` | K8s ConfigMap | `env_utils.get_lmcache_env()` | KV 卸载开关 |
| `LMCACHE_QAT` | K8s ConfigMap | `env_utils.get_qat_env()` | QAT 压缩开关 |
| `LMCACHE_LOCAL_CPU` | K8s ConfigMap | `env_utils.get_lmcache_cpu_env()` | CPU 内存 |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | K8s ConfigMap | 日志记录 | 最大 CPU 内存 |
| `LMCACHE_LOCAL_DISK` | K8s ConfigMap | `env_utils.get_lmcache_disk_env()` | 磁盘路径 |
| `LMCACHE_MAX_LOCAL_DISK_SIZE` | K8s ConfigMap | 校验 | 最大磁盘大小 |
| `LMCACHE_QAT_LOSS_LEVEL` | K8s ConfigMap | 日志记录 | QAT 损失等级 |
| `LMCACHE_QAT_INSTANCE_NUM` | K8s ConfigMap | 日志记录 | QAT 实例数 |
| `KV_AGENT_LIB_PATH` | K8s ConfigMap | `_build_cache_env_commands()` | vLLM (NVIDIA) kv_agent 原生库路径 |
| `LMCACHE_LIB_PATH` | K8s ConfigMap | `_build_cache_env_commands()` | vLLM-Ascend lmcache 原生库路径 |
| `PD_ROLE` | K8s ConfigMap / setEnv.sh | `env_utils.get_pd_role_env()` | "P" (Prefill) 或 "D" (Decode) |
| `PD_CONNECTOR_TYPE` | K8s ConfigMap / setEnv.sh | `_get_pd_config()` | Ascend PD connector 类型，默认 "MooncakeConnectorV1"，可选 "MooncakeConnector" |
| `PYTHONHASHSEED` | sidecar 设置 | `_build_cache_env_commands()` | 设为 0，跨实例哈希一致性 |

### 4.2 `_set_kv_cache_config` 配置注入

`config_loader.py L675`

**决策矩阵：**

| LMCache | PD Role | 生成的 Connector |
|---------|---------|------------------|
| ✓ | ✓ | **MultiConnector**（包裹 PD connector + LMCacheConnectorV1） |
| ✓ | ✗ | **LMCacheConnectorV1** |
| ✗ | ✓ | **MooncakeConnector** (Ascend) 或 **NixlConnector** (NVIDIA) |
| ✗ | ✗ | 跳过（不注入） |

### 4.3 PD Connector 配置详情

`_get_pd_config(ctx, pd_role)` — `config_loader.py L677`

**Ascend 设备：**

通过 `PD_CONNECTOR_TYPE` 环境变量选择 connector（默认 `MooncakeConnectorV1`）：

| Connector | 说明 | 适用场景 |
|-----------|------|---------|
| `MooncakeConnectorV1` | vllm-ascend 原生实现，支持 tuple KV cache 和 MLA | 默认，主流场景 |
| `MooncakeConnector` | 新版连接器，改进了 KV cache 注册逻辑 | MooncakeConnectorV1 运行时崩溃时的替代方案 |

```json
{
    "kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "mooncake_protocol": "rdma",
        "prefill": {"tp_size": 4, "dp_size": 1, "pp_size": 1},
        "decode": {"tp_size": 4, "dp_size": 1, "pp_size": 1}
    }
}
```
- `pd_role="P"` → `kv_role="kv_producer"`
- `pd_role="D"` → `kv_role="kv_consumer"`

**非 Ascend 设备 (NVIDIA)：**
```json
{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_both"
}
```

### 4.4 MultiConnector 组合配置

当 LMCache + PD 同时启用：
```json
{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "connectors": [
            {"kv_connector": "MooncakeConnector", ...},
            {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        ]
    }
}
```

### 4.5 LD_LIBRARY_PATH 注入

`_build_cache_env_commands()` (vllm_adapter.py L267)：
- `vllm` (NVIDIA) → `export LD_LIBRARY_PATH=$KV_AGENT_LIB_PATH:$LD_LIBRARY_PATH`
- `vllm_ascend` → `export LD_LIBRARY_PATH=$LMCACHE_LIB_PATH:$LD_LIBRARY_PATH`

### 4.6 分布式场景下的 PD 处理

`_handle_vllm_distributed()` (config_loader.py L1587)：
- Ascend + PD → 直接返回（standalone 模式，不走 Ray/DP）
- NVIDIA + PD → 使用 `dp_deployment` 后端

---

## 5. Soft FP8 / Soft FP4 量化

> **已移除**：Soft FP8 / Soft FP4 自动量化整体删除——含 `_set_soft_fp8` / `_set_soft_fp4`
> 及其 helper、检测函数（`is_qwen3_series_fp8` / `is_deepseek_series_fp8` /
> `is_qwen3_32b_nvfp4` / `is_deepseek_series_modelslim_quant` / `_is_deepseek_v3_modelslim_layout`）、
> 开关 `ENABLE_SOFT_FP4`、运行时 env `_build_deepseek_fp8_env_commands`、以及官方 W8A8 路径
> `_set_deepseek_v3_family_ascend_quant_params`。
>
> 现状：wings 不再为「config.json 不带 `quantization_config`」的裸布局权重自动注入
> `quantization='ascend'`。此类 Ascend 量化权重需用户**显式传 `--quantization`**，或改用
> 自带量化声明的权重。`ENABLE_SOFT_FP8` 仅余引擎自动选择中的路由作用。

---

## 6. wings-accel 运行时补丁

### 6.1 总体机制

wings-accel 在引擎启动前注入 Python monkey-patch，修改 vLLM 运行时行为以支持高级特性。

### 6.2 特性与补丁名称映射

`_FEATURE_SWITCH_MAP` — `wings_entry.py L59`

```python
# 现已为空：投机推理改走 --install-runtime-deps；IndexCache(sparse) 通过动态 feature 聚合安装；
# LMCache 走 --lmcache-target；Soft FP8/FP4 已整体移除。
_FEATURE_SWITCH_MAP: dict[str, str] = {}
```

### 6.3 引擎到补丁 key 的映射

`_ENGINE_PATCH_KEY_MAP` — `wings_entry.py L54`

```python
_ENGINE_PATCH_KEY_MAP = {
    "vllm":       "vllm",
    "vllm_ascend": "vllm",    # 复用 vllm 的补丁体系
}
```

仅 vLLM 系列引擎支持 accel 补丁。

### 6.4 特性收集逻辑

`_collect_enabled_features()` — `wings_entry.py L197`

遍历 `_FEATURE_SWITCH_MAP`，检查每个环境变量是否为 `"true"`。

**特殊规则 — adaptive_draft_model：**
需要同时满足：
1. `ENABLE_SPECULATIVE_DECODE=true`
2. `SPECULATIVE_DECODE_MODEL_PATH` 非空

无草稿模型路径时跳过（MTP/suffix 方法不需要 adaptive_draft_model 补丁）。

### 6.5 补丁安装脚本生成

`_build_accel_preamble(engine)` — `wings_entry.py L267`

**路径 A — 用户覆盖：**
如果 `WINGS_ENGINE_PATCH_OPTIONS` 环境变量已设置（有效 JSON dict），直接使用用户提供的值。

**路径 B — 自动生成：**

```
1. patch_key = _ENGINE_PATCH_KEY_MAP[engine]        (e.g. "vllm")
2. features = _collect_enabled_features()            (e.g. ["adaptive_draft_model"])
3. engine_version = normalize_engine_version()       (e.g. "0.17.0")
4. WINGS_ENGINE_PATCH_OPTIONS = JSON 序列化
```

生成的 `WINGS_ENGINE_PATCH_OPTIONS` 示例：
```json
{
    "vllm": {
        "version": "0.17.0",
        "features": ["adaptive_draft_model"]
    }
}
```

### 6.6 两级容错安装

生成的 bash 脚本:

```bash
# 第 1 级：批量安装
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.17.0", "features": [...]}}'
if [ -f "/accel-volume/install.py" ]; then
    set +e
    python3 /accel-volume/install.py --features "$WINGS_ENGINE_PATCH_OPTIONS"
    ACCEL_RC=$?
    set -e
    if [ $ACCEL_RC -ne 0 ]; then
        # 第 2 级：逐特性安装
        echo "[wings-accel] Batch install failed, trying per-feature fallback..."
        # 每个特性独立安装，失败只打 WARNING 不退出
        for each feature:
            set +e
            python3 /accel-volume/install.py --features '{"vllm": {"version":"0.17.0","features":["<name>"]}}'
            set -e
    fi
fi
# 无论结果如何，继续拉起引擎服务
```

### 6.7 版本解析

`normalize_engine_version()` — `version_util.py L81`：
- 从 `ENGINE_VERSION` 环境变量读取（如 `v0.17.0-202603231535`）
- `parse_engine_version_tuple()` 通过正则 `r'(\d+)\.(\d+)'` 提取 → `(0, 17)`
- 格式化为 `"0.17.0"`

install.py 内部有 `future_fallback` 逻辑，未知版本号会自动回退到默认版本。

---

## 7. 故障回退与重试机制

### 7.1 高级特性判定

`_has_advanced_features(merged)` — `wings_entry.py L561`

以下三个条件满足任一即视为启用了高级特性：
1. `merged.get("enable_speculative_decode")` → True
2. `merged.get("enable_sparse")` → True
3. `os.getenv("LMCACHE_OFFLOAD") == "true"`

### 7.2 回退 vs 重试的选择

`build_launcher_plan()` — `wings_entry.py L893`:

```python
has_advanced_feature = _has_advanced_features(merged)

if has_advanced_feature:
    fallback_cmd = _build_advanced_feature_fallback_cmd(merged)  # 禁用高级特性的备用命令
    retry_cmd = ""                                                # 有 fallback 就不生成 retry
else:
    fallback_cmd = ""
    retry_cmd = _build_engine_retry_cmd(merged)                   # 普通重试（相同参数）
```

| 场景 | fallback_cmd | retry_cmd | 行为 |
|------|-------------|-----------|------|
| 有高级特性 | ✓ 填充 | ✗ 空 | 快速失败 → 禁用高级特性重启 |
| 无高级特性 | ✗ 空 | ✓ 填充 | 崩溃 → 相同参数重试一次 |
| 都为空 | ✗ | ✗ | 基础模式：崩溃 → 直接退出 |

### 7.3 快速失败检测阈值

```python
_ADV_FEATURE_FAST_FAIL_THRESHOLD_SEC = 120    # wings_entry.py L558
```

引擎在 120 秒内退出 → 视为启动阶段失败 → 触发回退。

### 7.4 高级特性回退命令生成

`_build_advanced_feature_fallback_cmd(merged)` — `wings_entry.py L800`

```python
merged_no_features = dict(merged)
merged_no_features["enable_speculative_decode"] = False   # 禁用投机推理
merged_no_features["enable_sparse"] = False               # 禁用 KV 稀疏

# 移除 kv_transfer_config（LMCache 场景）
if LMCACHE_OFFLOAD == "true":
    ec_copy = dict(merged_no_features["engine_config"])
    ec_copy.pop("kv_transfer_config", None)
    merged_no_features["engine_config"] = ec_copy

# 重新生成无高级特性的启动命令
fallback_body = start_engine_service(merged_no_features)
fallback_cmd = _strip_exec_and_backgroundify(fallback_body) + "ENGINE_PID=$!\n"
```

### 7.5 普通重试命令生成

`_build_engine_retry_cmd(merged)` — `wings_entry.py L864`

```python
retry_body = start_engine_service(merged)    # 使用完全相同的参数
retry_cmd = _strip_exec_and_backgroundify(retry_body) + "ENGINE_PID=$!\n"
```

### 7.6 监控脚本逻辑

`_build_monitor_script(fallback_cmd, retry_cmd, active_features)` — `wings_entry.py L581`

**高级特性回退版** (fallback_cmd 非空):

```
引擎启动并后台运行
  │
  ▼
wait $ENGINE_PID → 正常退出? → 是 → 清理退出
  │
  └── 否 → 获取 EXIT_CODE 和 ENGINE_DURATION
       │
       ├── ENGINE_DURATION < 120s → "启动阶段失败"
       │     sleep 5 → 执行 fallback_cmd (禁用高级特性) → wait
       │     ├── 正常退出 → 成功
       │     └── 再次失败 → 写进度文件 + exit
       │
       └── ENGINE_DURATION >= 120s → "运行时崩溃，非高级特性问题"
             → 写进度文件 + exit（不回退）
```

**普通重试版** (retry_cmd 非空):

```
引擎启动并后台运行
  │
  ▼
wait $ENGINE_PID → 正常退出? → 是 → 清理退出
  │
  └── 否 → 获取 EXIT_CODE 和 ENGINE_DURATION
       sleep 5 → 执行 retry_cmd (相同参数) → wait
       ├── 正常退出 → 成功
       └── 再次失败 → 写进度文件 + exit
```

---

## 8. start_command.sh 最终组装顺序

`build_launcher_plan()` — `wings_entry.py L905`

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /var/log/wings

# ① analyzer_preamble — 日志解析器后台启动（仅 Master 节点）
#    启动 log_analyzer.py 子进程，实时分析引擎日志

# ② PYTHONUNBUFFERED + 日志过滤
export PYTHONUNBUFFERED=1
exec > >(tee -a /var/log/wings/engine-full.log | grep ... | tee -a engine.log) 2>&1

# ③ faulthandler_patch — SGLang OOM 补丁（仅 sglang 引擎）
#    注入 sitecustomize.py monkey-patch

# ④ triton_patch — Ascend NPU Triton 驱动补丁（仅 vllm_ascend 引擎）
#    python3 << 'TRITON_PATCH_EOF' ... TRITON_PATCH_EOF

# ⑤ env_overrides — 用户自定义环境变量
#    读取 env_overrides/ 目录下的 .env/.sh 文件

# ⑥ accel_preamble — wings-accel 补丁安装
#    python3 /accel-volume/install.py --features ... (两级容错)

# ⑦ script_body — 引擎启动命令
#    ENGINE_START_EPOCH=$(date +%s)
#    python3 -m vllm.entrypoints.openai.api_server \
#      --model ... --speculative-config '...' --kv-transfer-config '...' &
#    ENGINE_PID=$!

# ⑧ monitor_script — 进程监控 + 回退/重试
#    wait $ENGINE_PID → fallback / retry / exit
```

### 设计要点

| 模块 | 放在 preamble 的原因 |
|------|----------------------|
| triton_patch | 包含 heredoc，不能被 `_build_monitor_script` 缩进（会破坏 heredoc 闭合标记） |
| accel_preamble | 是一次性安装操作，重试/回退时不需要重复安装 |
| env_overrides | 环境变量只需设置一次 |
| faulthandler_patch | sitecustomize.py 只需创建一次 |

| 模块 | 放在 retry/fallback 命令中的原因 |
|------|--------------------------------|
| script_body | 引擎启动命令可能因参数不同需要重新生成 |
| monitor_script | 包含完整的 wait + 错误处理逻辑 |

---

## 附录：完整参数数据流图

```
                          K8s Deployment
                              │
     ┌────────────────────────┼─────────────────────────┐
     │                        │                         │
     ▼                        ▼                         ▼
  SD_ENABLE=true        LMCACHE_OFFLOAD=true      ENABLE_SPARSE=true
  SPEC_MODEL_PATH=...   PD_ROLE=P
     │                        │                         │
     ▼                        ▼                         ▼
  ┌──────────── config_loader._merge_vllm_params() ──────────────┐
  │                                                               │
  │  _set_spec_decoding_config()  → SD_ENABLE env                │
  │  _set_sparse_config()         → SPARSE_ENABLE env            │
  │  _set_kv_cache_config()       → params['kv_transfer_config'] │
  │                                                               │
  └──────────────────────┬────────────────────────────────────────┘
                         │ merged dict
                         ▼
  ┌──────────── vllm_adapter.build_start_script() ───────────────┐
  │                                                               │
  │  _build_speculative_cmd()  → " --speculative-config '{...}'" │
  │  _build_sparse_cmd()       → " --kv-transfer-config '{...}'" │
  │  _format_cli_arg()         → " --kv-transfer-config '{...}'" │
  │  (quantization 通过 engine_config 注入)                        │
  │                                                               │
  │  输出: exec python3 -m vllm ... [所有参数拼接]                │
  └──────────────────────┬────────────────────────────────────────┘
                         │ script_body
                         ▼
  ┌──────────── wings_entry.build_launcher_plan() ───────────────┐
  │                                                               │
  │  ④ triton_patch         ← build_triton_patch_preamble()      │
  │  ⑥ accel_preamble       ← _build_accel_preamble()           │
  │  ⑦ script_body          ← _build_pid_tracked_script()       │
  │  ⑧ monitor_script       ← _build_monitor_script(            │
  │                              fallback_cmd, retry_cmd)         │
  │                                                               │
  │  输出: start_command.sh 完整脚本                               │
  └───────────────────────────────────────────────────────────────┘
```
