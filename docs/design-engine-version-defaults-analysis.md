# 引擎稳定版本默认配置方案 — 完整分析与补充设计

> **源方案**: `D:\project\wings-k8s-260323\...\docs\design-engine-version-defaults.md`  
> **分析基线**: `D:\project\wings-k8s-260325\...\wings_control\` 当前代码  
> **日期**: 2026-03-25

---

## 目录

1. [方案概述](#1-方案概述)
2. [现有代码架构映射](#2-现有代码架构映射)
3. [方案亮点评价](#3-方案亮点评价)
4. [风险点识别与补充设计](#4-风险点识别与补充设计)
5. [可行性分析（基于当前代码）](#5-可行性分析基于当前代码)
6. [实施影响与工作量评估](#6-实施影响与工作量评估)
7. [架构级版本配置扩展方案](#7-架构级版本配置扩展方案)
8. [结论与建议](#8-结论与建议)

---

## 关联文档：模型级文件化配置系列

在版本级配置的基础上，进一步扩展为 **模型级文件化配置方案**，以下为系列文档索引：

| 文档 | 说明 |
|------|------|
| [design-model-file-config.md](design-model-file-config.md) | **主设计文档** — 目录结构、查找链、合并流程、支持模型列表 |
| [design-model-file-config-schema.md](design-model-file-config-schema.md) | **Schema 与命名规范** — JSON Schema 定义、文件命名规则、校验策略、完整示例 |
| [design-model-file-config-integration.md](design-model-file-config-integration.md) | **代码集成与迁移** — 代码插入点、新增函数设计、env_vars 注入、测试策略 |
| [design-model-file-config-mindie.md](design-model-file-config-mindie.md) | **MindIE 专项** — 统一扁平格式决策、flat→nested 映射关系、MindIE 配置示例 |

---

## 1. 方案概述

### 1.1 解决的问题

当前配置合并链路存在三个缺陷：

| # | 问题 | 影响 |
|---|------|------|
| 1 | **无版本维度** — `vllm_default.json` 版本无关，但 vLLM 0.16 与 0.17 参数语义不同 | 版本升级可能导致默认值不适配 |
| 2 | **无环境变量注入层** — 部分版本需特定 env（如 `VLLM_USE_V1=1`），仅靠用户手动设置 | 容易遗漏，导致启动失败 |
| 3 | **最小参数场景无保障** — 用户只传最小必要参数时，通用默认值未必适配特定版本 | 服务拉起不可靠 |

> **关于"最小参数"的修正**: 原方案描述"仅传 3 个参数（input_length / gpu_memory_utilization / tp_size）"不够准确。实际上用户还需传入 `output_length`（与 `input_length` 一起通过 `_set_sequence_length()` 计算为 `max_model_len`），以及 `MODEL_NAME`、`MODEL_PATH`、`ENGINE` 等基础参数。"最小参数"应理解为：用户只需传入**业务必要参数**（模型标识 + 序列长度 + 资源配额），引擎级别的参数和环境变量由版本配置自动补全。

### 1.2 核心设计

- 新增 `config/stable_versions/{engine}/{major}.{minor}.json` 目录结构
- 三段式 JSON 格式：`engine_params` + `env_vars` + `meta`
- 版本匹配：`ENGINE_VERSION` → 去前缀 → 提取 major.minor → 精确匹配 → 回退 `latest.json`
- 合并优先级：硬件默认 < **版本默认（新增）** < 模型专属 < 用户配置 < CLI
- env_vars 注入到 `start_command.sh`，位于 `user_overrides.env` 之前

---

## 2. 现有代码架构映射

### 2.1 配置合并链路（config_loader.py）

```
load_and_merge_configs()                         ← 主入口 (L1732)
  ├─ _process_cmd_args()                          ← 提取 CLI 参数
  ├─ _check_vram_requirements()                   ← 显存检查
  ├─ _auto_select_engine()                        ← 引擎自动选择/校验
  ├─ _load_user_config()                          ← 用户 config-file
  ├─ _translate_user_config_for_engine()           ← 参数名翻译
  ├─ _get_model_specific_config()                  ← 【插入点】多层模型配置查找
  │   ├─ _load_default_config()                   ← vllm_default.json (硬件默认)
  │   ├─ model_deploy_config 分层查找             ← 精确模型→架构→类型→兜底
  │   └─ _merge_cmd_params()                      ← 三层合并
  ├─ _merge_configs(engine_specific, user_config)  ← 合并用户配置
  ├─ _apply_cli_overrides()                        ← CLI 最高优先级
  └─ _merge_final_config()                         ← 包装 engine_config 子字典
```

**方案的版本默认层应插入 `_get_model_specific_config()` 内部**，在 `_load_default_config()` 之后、model_deploy_config 查找之前。

### 2.2 启动脚本组装（wings_entry.py）

```python
# build_launcher_plan() 当前脚本组装顺序 (L480-490)
command = (
    "#!/usr/bin/env bash\nset -euo pipefail\n"
    "mkdir -p /var/log/wings\n"
    + analyzer_preamble          # 1. log_analyzer（仅 master）
    + "exec > >(tee -a ...)\n"
    + env_overrides              # 2. 用户 env_overrides
    + accel_preamble             # 3. Accel 补丁安装
    + script_body                # 4. 引擎启动
    + monitor_script             # 5. 进程监控
)
```

方案需在 `env_overrides` 之前插入 `version_env_block`，为纯字符串拼接，零侵入。

### 2.3 ENGINE_VERSION 现有使用

| 文件 | 位置 | 用途 |
|------|------|------|
| `engines/vllm_adapter.py` L56 | `_parse_engine_version()` | 解析为 (major, minor) 元组，控制 Ascend NPU 资源声明 |
| `core/wings_entry.py` L124 | `_build_accel_env_line()` | Accel 补丁版本号传递 |
| `proxy/health_service.py` L291 | 健康检查接口 | 返回引擎版本号 |

方案不改变 `ENGINE_VERSION` 语义，仅新增读取逻辑。`vllm_adapter.py` 中的版本解析逻辑可复用。

### 2.4 现有配置文件格式

```
config/
├── defaults/
│   ├── vllm_default.json           # 14 个参数，扁平 key-value
│   ├── sglang_default.json         # 13 个参数，扁平 key-value
│   ├── mindie_default.json         # 嵌套结构
│   ├── nvidia_default.json         # 含 model_deploy_config 分层结构
│   ├── ascend_default.json
│   ├── distributed_config.json
│   └── engine_parameter_mapping.json
├── env_overrides/                  # .gitkeep + README
├── settings.py
└── __init__.py
```

`stable_versions/` 完全不冲突，可安全新增。

---

## 3. 方案亮点评价

| 方面 | 评价 |
|------|------|
| **目录结构** | `stable_versions/{engine}/{major}.{minor}.json` 命名与 `ENGINE` 环境变量一致，直观可查找 |
| **三段式 JSON** | `engine_params` + `env_vars` + `meta` 职责分离，meta 不参与合并仅供审计 |
| **版本匹配** | 去前缀 → major.minor → 精确匹配 → latest.json → 跳过，三级降级策略健壮 |
| **合并优先级** | 在现有链中插入新层，不改变现有行为，向后完全兼容 |
| **兜底安全** | 无版本配置文件时返回空字典，等价于当前行为，零副作用 |
| **K8s 集成** | 支持镜像内置 + ConfigMap subPath 热替换，部署灵活 |

---

## 4. 风险点识别与补充设计

### 4.1 数据流解耦 — env_vars 传递方式 [P0]

**问题**：原方案计划将 `env_vars` 通过 `merged["_version_env_vars"]` side-channel 传递，在 `build_launcher_plan()` 中 `pop` 取出。当前代码中 `merged` 字典会被 `start_engine_service(merged)` 消费，adapter 未做未知 key 过滤，`_version_env_vars` 会被误传入 adapter。

**补充方案**：引入 `VersionDefaults` dataclass，env_vars 独立于 merged 传递：

```python
@dataclass
class VersionDefaults:
    engine_params: dict[str, Any]       # 参与 _merge_configs
    env_vars: dict[str, str]            # 注入 start_command.sh
    deprecated_params: list[str]        # 废弃参数
    meta: dict[str, Any]                # 仅供日志

def load_and_merge_configs(...) -> tuple[dict, VersionDefaults]:
    version_defaults = _load_engine_version_defaults(engine, engine_version)
    # engine_params 参与合并，env_vars 独立返回
    ...
```

**影响**：`_prepare_merged_params()` 和 `build_launcher_plan()` 需调整签名。当前 `load_and_merge_configs` 仅在 `_prepare_merged_params()` 中被调用（唯一调用者），影响链可控（3 个函数）。

### 4.2 配置文件 Schema 校验 [P0]

**问题**：`stable_versions/*.json` 格式错误（如 `engine_params` 拼写为 `engine_param`）时，系统会静默使用空字典。现有 `load_json_config()` 只做 JSON 解析无结构校验。

**补充方案**：加载时校验三个必需顶层字段的存在性和类型，env_vars 值必须为字符串：

```python
_VERSION_CONFIG_REQUIRED_KEYS = {"engine_params", "env_vars", "meta"}

def _validate_version_config(config: dict, file_path: str) -> None:
    missing = _VERSION_CONFIG_REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Version config '{file_path}' missing keys: {missing}")
    if not isinstance(config["engine_params"], dict):
        raise TypeError(...)
    for key, value in config["env_vars"].items():
        if not isinstance(value, str):
            raise TypeError(f"env_vars['{key}'] must be str, got {type(value).__name__}")
```

支持 strict/non-strict 模式，通过 `WINGS_STRICT_VERSION_CONFIG` 环境变量控制。

### 4.3 deprecated_params 从可选提升为必选 [P0]

**问题**：引擎版本升级时废弃参数直接导致启动失败（如 vLLM 0.17 不再支持 `swap_space`）。当前代码中 `vllm_adapter.py` 仅有 Ascend 版本分支逻辑，**无通用废弃参数处理机制**。

**补充方案**：将 deprecated_params 扩展为分级结构并纳入必经路径：

```jsonc
"deprecated_params": {
    "removed": ["swap_space"],              // 直接删除（传了必崩）
    "ignored": ["enable_chunked_prefill"],  // 引擎忽略（删除并 warning）
    "renamed": {                             // 参数改名（自动迁移）
        "old_param_name": "new_param_name"
    }
}
```

向后兼容：若 `deprecated_params` 为列表，自动视为全部 `removed`。

### 4.4 版本匹配增强与 Patch 级预警 [P1]

**问题**：只取 `major.minor`，同一 minor 版本的 patch 差异被忽略。

**补充方案**：
- 匹配粒度保持 major.minor（足够简洁）
- 加载成功后对 `meta.version_range` 做运行时校验，不匹配时发 WARNING
- 预留 patch 级扩展路径：`{major}.{minor}.{patch}.json` → `{major}.{minor}.json` → `latest.json`

`_parse_engine_version()` 已有的解析逻辑可直接复用：
```python
# vllm_adapter.py 现有实现
match = re.match(r"(\d+)\.(\d+)", ver_str)
→ (int(match.group(1)), int(match.group(2)))
```

### 4.5 K8s 多版本并行部署 ConfigMap 命名 [P1]

**问题**：同一集群多引擎版本并行时 ConfigMap 可能命名冲突。

**补充方案**：
```
命名格式: wings-stable-config-{engine}-{major}-{minor}
示例: wings-stable-config-vllm-0-17
```

Pod spec 中 `configMap.optional: true`，确保无 ConfigMap 时不阻断启动。

### 4.6 环境变量注入顺序明确化 [P0]

**问题**：方案需明确 5 层脚本注入顺序及覆盖关系。

当前代码中 Accel 在 env_overrides 之后（这是设计意图：Accel 补丁安装需在用户 env 之后执行）。

**补充方案**：

```bash
# 层级 0: 系统初始化 (log_analyzer)
# 层级 1: 引擎版本默认环境变量 (最低优先级)  ← 新增
# 层级 2: 用户环境变量覆盖 (可覆盖层级1)
# 层级 3: Accel 补丁安装 (可覆盖层级1+2)
# 层级 4: 引擎启动命令
# 层级 5: 进程监控
```

覆盖关系：**Accel(3) > 用户(2) > 版本默认(1)**

新增 `_build_version_env_block()` 函数，使用 `shlex.quote()` 防止环境变量值中的特殊字符导致 shell 注入。

---

## 5. 可行性分析（基于当前代码）

### 5.1 配置插入点

| 评估项 | 结论 |
|--------|------|
| **合并链插入** | `_get_model_specific_config()` L1617 内部，`_load_default_config()` 之后、model_deploy_config 查找之前 merge 一层 |
| **`_merge_configs` 兼容性** | 深度合并函数天然支持多层叠加，新增一层不改变下游逻辑 |
| **脚本组装** | 纯字符串拼接，插入 `version_env_block` 零侵入 |

### 5.2 返回值变更影响

```
load_and_merge_configs()  → 返回 tuple[dict, VersionDefaults]
  ↑ 调用者
_prepare_merged_params()  → 需同步调整返回值
  ↑ 调用者
build_launcher_plan()     → 接收 VersionDefaults，传入 _build_version_env_block()
```

**影响范围**：3 个函数，改动可控。`load_and_merge_configs` 在整个代码库中仅有一处调用。

### 5.3 settings.py 需新增字段

```python
ENGINE_VERSION: str = ""                           # 引擎版本号
STABLE_VERSIONS_DIR: str = "config/stable_versions" # 版本配置目录
WINGS_STRICT_VERSION_CONFIG: bool = True            # 是否严格校验
```

### 5.4 目录结构兼容性

`config/stable_versions/` 与现有目录完全不冲突。`DEFAULT_CONFIG_DIR` 指向 `config/defaults/`，新目录需要独立路径解析：

```python
def _resolve_stable_versions_dir() -> str:
    env_dir = os.getenv("WINGS_STABLE_VERSIONS_DIR", "").strip()
    if env_dir:
        return env_dir
    bundled_dir = Path(__file__).resolve().parents[1] / "config" / "stable_versions"
    if bundled_dir.exists():
        return str(bundled_dir)
    return ""  # 不存在则跳过
```

### 5.5 ENGINE_VERSION 复用

`vllm_adapter.py` 中的 `_parse_engine_version()` 可提取为公共工具函数放到 `utils/` 中，供 config_loader 和 vllm_adapter 共用。

---

## 6. 实施影响与工作量评估

### 6.1 文件级变更

| 文件 | 变更类型 | 改动量 | 说明 |
|------|----------|--------|------|
| `config_loader.py` | 修改 | ~130 行新增 | `VersionDefaults`, `_load_engine_version_defaults()`, Schema 校验, deprecated_params 处理, 合并链调整 |
| `wings_entry.py` | 修改 | ~30 行新增 | `_build_version_env_block()`, 签名调整, 脚本组装插入 |
| `config/settings.py` | 修改 | ~3 行新增 | 新增 3 个设置字段 |
| `config/stable_versions/` | 新增 | 目录 + 示例文件 | vLLM 0.17 配置示例 |
| `utils/version_utils.py`（可选） | 新增 | ~20 行 | 公共版本解析函数 |
| **总计** | **3 改 + 1~2 新增** | **~180 行** | 纯新增逻辑，不修改现有行为 |

### 6.2 风险矩阵

| 变更 | 风险 | 缓解措施 |
|------|------|----------|
| `load_and_merge_configs` 返回值 | 中 — 函数签名变更 | 唯一调用者，改动可控 |
| 版本配置文件格式 | 低 — 新增文件 | Schema 校验 + 兜底空返回 |
| 脚本注入顺序 | 低 — 字符串拼接 | 与现有层级隔离，可独立回滚 |
| deprecated_params 自动删除 | 中 — 修改最终参数 | 仅删除已验证废弃的参数，有日志审计 |

### 6.3 测试策略

| 场景 | 测试内容 |
|------|----------|
| 版本配置存在 | 验证 engine_params 正确合并、env_vars 注入脚本 |
| 版本配置不存在 | 验证行为等价于当前（零副作用） |
| 版本配置格式错误 | 验证 strict 模式报错、non-strict 模式跳过 |
| deprecated_params | 验证 removed/ignored/renamed 三类分别处理 |
| 用户覆盖版本默认 | 验证 CLI > 用户配置 > 版本默认 优先级 |
| 最小参数场景 | 仅传最小必要参数（input_length + output_length + gpu_memory_utilization + MODEL_NAME + MODEL_PATH 等），验证服务可拉起 |

---

## 7. 架构级版本配置扩展方案

### 7.1 需求背景

原方案的版本配置粒度为 **引擎级**（`stable_versions/vllm/0.17.json`），所有模型架构共用同一组版本默认参数。但实际场景中，不同架构在同一引擎版本下的最优参数可能不同：

- `DeepseekV3ForCausalLM` 在 vLLM 0.17 上需要 `enable_expert_parallel=true`
- `Qwen3MoeForCausalLM` 在 vLLM 0.17 上需要 `enable_expert_parallel=true` + `tool_call_parser=hermes`
- `Qwen3ForCausalLM` 在 vLLM 0.17 上无需 MoE Expert Parallel

当前 `nvidia_default.json` / `ascend_default.json` 已有按架构分层的 `model_deploy_config`，但它是**版本无关**的。如果某个架构在 vLLM 0.17 上的最优参数与 0.16 不同，现有机制无法区分。

### 7.2 当前已支持的架构

从 `nvidia_default.json` 和 `ascend_default.json` 中提取：

| 架构 | NVIDIA 引擎 | Ascend 引擎 |
|------|-------------|-------------|
| `DeepseekV3ForCausalLM` | vllm, sglang (含 H20 卡型细分) | vllm_ascend, mindie |
| `Qwen3MoeForCausalLM` | vllm, sglang | vllm_ascend, mindie |
| `Qwen3ForCausalLM` | vllm, sglang | vllm_ascend, mindie |
| `Qwen2ForCausalLM` | vllm, sglang | vllm_ascend, mindie |

架构识别来源：`ModelIdentifier.identify_model_architecture()` 从模型目录 `config.json` 的 `architectures[0]` 字段读取。

### 7.3 推荐方案：JSON 内嵌架构覆盖段

在现有版本配置 JSON 中新增可选的 `architecture_overrides` 段落：

```jsonc
// stable_versions/vllm/0.17.json
{
  "engine_params": {
    // 引擎级默认（所有架构共用）
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
    "enable_prefix_caching": true,
    "trust_remote_code": true,
    "dtype": "auto",
    "block_size": 16,
    "max_num_seqs": 256,
    "seed": 42
  },

  "env_vars": {
    "VLLM_USE_V1": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn"
  },

  // 【新增】架构级参数覆盖（可选段落）
  "architecture_overrides": {
    "DeepseekV3ForCausalLM": {
      "engine_params": {
        "enable_expert_parallel": true,
        "enable_auto_tool_choice": true,
        "tool_call_parser": "deepseek_v3"
      },
      "env_vars": {
        "VLLM_MLA_DISABLE": "0"
      }
    },
    "Qwen3MoeForCausalLM": {
      "engine_params": {
        "enable_expert_parallel": true,
        "enable_auto_tool_choice": true,
        "tool_call_parser": "hermes"
      }
    },
    "Qwen3ForCausalLM": {
      "engine_params": {
        "enable_auto_tool_choice": true,
        "tool_call_parser": "hermes"
      }
    }
    // 其他架构未列出 → 使用引擎级 engine_params
  },

  "deprecated_params": {
    "removed": ["swap_space"],
    "ignored": [],
    "renamed": {}
  },

  "meta": {
    "engine": "vllm",
    "version_range": ">=0.17.0, <0.18.0",
    "tested_date": "2025-07-01",
    "notes": "vLLM 0.17 默认启用 V1 Runtime"
  }
}
```

**合并逻辑**：

```python
def _load_engine_version_defaults(engine, engine_version, model_architecture=None):
    config = _load_version_config(engine, engine_version)  # 加载 JSON

    base_params = config.get("engine_params", {})
    base_env = config.get("env_vars", {})

    # 架构覆盖（若有）
    if model_architecture:
        arch_overrides = config.get("architecture_overrides", {}).get(model_architecture, {})
        arch_params = arch_overrides.get("engine_params", {})
        arch_env = arch_overrides.get("env_vars", {})

        # 架构参数覆盖引擎默认
        final_params = _merge_configs(base_params, arch_params)
        final_env = {**base_env, **arch_env}
    else:
        final_params = base_params
        final_env = base_env

    return VersionDefaults(
        engine_params=final_params,
        env_vars=final_env,
        deprecated_params=config.get("deprecated_params", {}),
        meta=config.get("meta", {}),
    )
```

**优势**：
- 单文件管理一个引擎版本的所有架构配置，便于整体审查和版本发布
- 架构覆盖是可选的，不影响原有引擎级默认逻辑
- 与 `nvidia_default.json` 的 `model_deploy_config` 模式一致，维护者零学习成本

### 7.4 备选方案：独立架构文件

```
stable_versions/vllm/
├── 0.17.json                                # 引擎级默认
├── 0.17.DeepseekV3ForCausalLM.json          # 架构级覆盖
├── 0.17.Qwen3MoeForCausalLM.json
└── latest.json
```

查找顺序：`{version}.{architecture}.json` → `{version}.json` → `latest.json`

优势是 Git diff 更清晰，劣势是文件膨胀（4 引擎 × N 版本 × M 架构）。**不推荐**。

### 7.5 合并优先级链（含架构维度）

```
1. 硬件默认                vllm_default.json                                       (最低)
2. 版本默认(引擎级)        stable_versions/vllm/0.17.json → engine_params
3. 版本默认(架构级)        stable_versions/vllm/0.17.json → architecture_overrides[arch]
4. 模型专属                nvidia_default.json → model_deploy_config[...][engine]
5. 用户配置                --config-file JSON
6. CLI / 环境变量           --gpu-memory-utilization 0.9                             (最高)
```

版本层的架构覆盖 **低于** `model_deploy_config`。这确保：
- 版本配置提供"该版本 + 该架构的合理默认"
- 模型专属配置仍可覆盖版本默认（如 DeepSeek-R1 的 H20 特定参数）

### 7.6 与现有 model_deploy_config 的关系

```
                    版本无关                               版本相关
                    ─────────                              ──────────
架构级参数    nvidia/ascend_default.json            stable_versions/vllm/0.17.json
              model_deploy_config                    architecture_overrides
              [model_type][arch][engine]              [arch].engine_params

              → 已验证的模型+架构+引擎组合            → 该版本下该架构的必要调整
              → 长期稳定，少变更                      → 随引擎版本更新
              → 优先级更高                            → 优先级更低
```

**两者不冲突、不重复**：
- `model_deploy_config` 关注"这个模型在这个引擎上的最佳配置"（版本无关）
- `architecture_overrides` 关注"这个引擎版本对该架构引入了什么变化"（版本相关增量）

### 7.7 走查示例：DeepSeek-R1 + vLLM 0.17 + NVIDIA

```
用户输入: ENGINE=vllm, ENGINE_VERSION=v0.17.0, MODEL_NAME=DeepSeek-R1
         INPUT_LENGTH=8192, OUTPUT_LENGTH=4096

合并过程:

1. vllm_default.json
   → max_model_len=4096, gpu_memory_utilization=0.8, enable_prefix_caching=true

2. stable_versions/vllm/0.17.json → engine_params
   → gpu_memory_utilization=0.85 (覆盖), block_size=16, seed=42
   → env: VLLM_USE_V1=1, VLLM_WORKER_MULTIPROC_METHOD=spawn

3. architecture_overrides["DeepseekV3ForCausalLM"]
   → enable_expert_parallel=true (新增), tool_call_parser="deepseek_v3" (新增)

4. nvidia_default.json → model_deploy_config["llm"]["DeepseekV3ForCausalLM"]["DeepSeek-R1"]["vllm"]
   → trust_remote_code=true, max_model_len=4096, enable_expert_parallel=true,
     enable_auto_tool_choice=true, tool_call_parser="deepseek_v3"

5. 无 --config-file

6. CLI/ENV → _set_sequence_length(8192+4096) → max_model_len=12288

最终: max_model_len=12288, gpu_memory_utilization=0.85, enable_expert_parallel=true,
      tool_call_parser="deepseek_v3", env: VLLM_USE_V1=1
```

---

## 8. 结论与建议

### 8.1 总体评价

**方案设计质量高**，与现有代码架构高度兼容，实施风险低。核心设计决策（目录结构、三段式 JSON、合并优先级链、兜底策略）均合理。

> **关于"最小参数"的修正**: 原方案描述"仅传 3 个参数"不够准确。实际上用户还需传入 `output_length`（与 `input_length` 一起通过 `_set_sequence_length()` 计算为 `max_model_len`），以及 `MODEL_NAME`、`MODEL_PATH`、`ENGINE` 等基础参数。"最小参数"应理解为：用户只需传入**业务必要参数**（模型标识 + 序列长度 + 资源配额），引擎级别的参数和环境变量由版本配置自动补全。

### 8.2 建议优先级

| 优先级 | 补充项 | 理由 |
|--------|--------|------|
| **P0** | 数据流解耦（VersionDefaults） | 架构层决策，应在编码前确定 |
| **P0** | Schema 校验 | 防止静默错误，排查成本极高 |
| **P0** | deprecated_params 必选化 | 直接影响启动稳定性 |
| **P0** | 注入顺序明确化 | 与数据流解耦一起落地 |
| **P0** | architecture_overrides 扩展 | 架构粒度是配置精确性的关键 |
| P1 | 版本匹配增强（version_range 校验） | 增强健壮性，非阻塞 |
| P1 | K8s ConfigMap 命名规范 | 运维规范，可后续补充 |

### 8.3 实施路线

```
Phase 1 (P0): config_loader 核心改造
  → VersionDefaults + _load_engine_version_defaults + Schema 校验
  → 合并链插入 + deprecated_params 处理
  → architecture_overrides 支持
  → 单元测试覆盖

Phase 2 (P0): wings_entry 脚本注入
  → _build_version_env_block + 签名调整
  → 集成测试：最小参数场景走查

Phase 3 (P1): 增强与规范
  → version_range 运行时校验
  → K8s ConfigMap 命名规范
  → vLLM 0.17 + SGLang 0.4 配置文件编写与验证
  → 各架构 architecture_overrides 配置编写
```
