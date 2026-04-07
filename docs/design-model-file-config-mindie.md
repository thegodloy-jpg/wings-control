# 模型文件化配置 — MindIE 专项说明

> 关联文档: [design-model-file-config.md](design-model-file-config.md) — 主设计文档  
> 关联文档: [design-model-file-config-integration.md](design-model-file-config-integration.md) — 代码集成方案

---

## 1. MindIE 与 CLI 引擎的根本差异

| 维度 | vLLM / SGLang | MindIE |
|------|--------------|--------|
| 参数传递方式 | CLI `--arg value` | JSON `config.json` 多层嵌套 |
| 配置结构 | 扁平 key-value | 5 层嵌套对象 |
| 参数生效机制 | 直接命令行 | merge-update 脚本覆写 `config.json` |
| 模型路径 | `--model` | `ModelConfig[0].modelWeightPath` |

### MindIE 的 5 层配置结构

```
config.json
├── ServerConfig        → ipAddress, port, httpsEnabled, inferMode ...
├── BackendConfig
│   ├── npuDeviceIds, multiNodesInferEnabled ...
│   ├── ModelDeployConfig
│   │   ├── maxSeqLen, maxInputTokenLen, truncation ...
│   │   └── ModelConfig[0]
│   │       ├── modelName, modelWeightPath, worldSize ...
│   │       ├── tp, dp, moe_tp, moe_ep (MOE 模型)
│   │       ├── sp, cp (特殊并行)
│   │       └── plugin_params (MTP 投机采样)
│   └── ScheduleConfig
│       └── cacheBlockSize, maxBatchSize, maxIterTimes ...
```

---

## 2. 统一扁平格式 (Route A) 的决策

### 2.1 方案对比

| 方案 | 格式 | 优点 | 缺点 |
|------|------|------|------|
| **Route A: 统一扁平** | `{"engine_params": {"maxSeqLen": 2560, "tp": 8}}` | 所有引擎同一 schema; config_loader 无需感知嵌套 | MindIE 适配器需做 flat→nested 映射 |
| Route B: 引擎原生 | `{"ServerConfig": {...}, "BackendConfig": {...}}` | MindIE 零映射 | 每个引擎 schema 不同; config_loader 需要字段分发 |
| Route C: 混合 | 通用参数扁平 + MindIE 嵌套选项 | 灵活 | 复杂; 边界模糊 |

**选择 Route A 的理由**:

1. **适配器已有映射能力** — `mindie_adapter.py` 中 `_build_server_overrides()`、`_build_model_config_overrides()` 等 5 个函数**已经在做 flat→nested 映射**，这正是它们存在的意义
2. **用户心智模型统一** — 配置文件编写者只需知道参数名 + 值，不需要知道 MindIE 的 config.json 嵌套结构
3. **`config_loader.py` 引擎无关** — 合并逻辑对所有引擎一致，不出现 `if engine == "mindie": ...` 分支

### 2.2 映射关系

文件中的扁平 `engine_params` 如何映射到 MindIE 5 层结构:

| 扁平参数 | 目标层 | 目标字段 | 映射函数 |
|---------|--------|---------|---------|
| `ipAddress` | ServerConfig | ipAddress | `_build_server_overrides()` |
| `port` | ServerConfig | port | `_build_server_overrides()` |
| `httpsEnabled` | ServerConfig | httpsEnabled | `_build_server_overrides()` |
| `inferMode` | ServerConfig | inferMode | `_build_server_overrides()` |
| `tokenTimeout` | ServerConfig | tokenTimeout | `_build_server_overrides()` |
| `npuDeviceIds` | BackendConfig | npuDeviceIds | `_build_backend_overrides()` |
| `multiNodesInferEnabled` | BackendConfig | multiNodesInferEnabled | `_build_backend_overrides()` |
| `maxSeqLen` | ModelDeployConfig | maxSeqLen | `_build_model_deploy_overrides()` |
| `maxInputTokenLen` | ModelDeployConfig | maxInputTokenLen | `_build_model_deploy_overrides()` |
| `modelWeightPath` | ModelConfig[0] | modelWeightPath | `_build_model_config_overrides()` |
| `worldSize` | ModelConfig[0] | worldSize | `_build_model_config_overrides()` |
| `tp` | ModelConfig[0] | tp | `_build_model_config_overrides()` |
| `dp` | ModelConfig[0] | dp | `_build_model_config_overrides()` |
| `moe_tp` | ModelConfig[0] | moe_tp | `_build_model_config_overrides()` (isMOE=true) |
| `moe_ep` | ModelConfig[0] | moe_ep | `_build_model_config_overrides()` (isMOE=true) |
| `sp` | ModelConfig[0] | sp | `_build_model_config_overrides()` |
| `cp` | ModelConfig[0] | cp | `_build_model_config_overrides()` |
| `isMOE` | 控制标志 | — | 触发 MOE 参数分支 |
| `isMTP` | 控制标志 | — | 触发 plugin_params 注入 |
| `cacheBlockSize` | ScheduleConfig | cacheBlockSize | `_build_schedule_overrides()` |
| `maxBatchSize` | ScheduleConfig | maxBatchSize | `_build_schedule_overrides()` |
| `maxIterTimes` | ScheduleConfig | maxIterTimes | `_build_schedule_overrides()` |

### 2.3 适配器无需修改

`mindie_adapter.py` 接收的 `engine_config` 字典来自 `config_loader.py` 合并后的结果。无论参数来自 `model_deploy_config`（旧）还是文件化配置（新），到达适配器时都是同一个扁平字典。适配器完全不感知参数来源。

---

## 3. MindIE 模型配置文件示例

### 3.1 DeepSeek-R1-w8a8 on MindIE 2.3

```jsonc
// config/models/mindie/2.3/DeepseekV3ForCausalLM/DeepSeek-R1-w8a8.json
{
  "engine_params": {
    "maxSeqLen": 4096,
    "maxInputTokenLen": 4000,
    "maxIterTimes": 4096,
    "truncation": true,
    "isMOE": true,
    "tp": 16,
    "dp": -1,
    "moe_tp": 16,
    "moe_ep": -1,
    "cacheBlockSize": 128,
    "maxPrefillBatchSize": 50,
    "maxPrefillTokens": 8192,
    "maxBatchSize": 200,
    "trustRemoteCode": true
  },
  "distributed_overrides": {
    "engine_params": {
      "tp": 16,
      "dp": -1,
      "maxSeqLen": 8192,
      "maxIterTimes": 8192
    }
  },
  "env_vars": {
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  },
  "meta": {
    "architecture": "DeepseekV3ForCausalLM",
    "model_type": "llm",
    "source": "ascend_default.json mindie 配置段",
    "tested_hardware": "Ascend 910B x16",
    "notes": "W8A8 量化版本，tp=16 全卡部署"
  }
}
```

### 3.2 Qwen3-235B-A22B on MindIE 2.3

```jsonc
// config/models/mindie/2.3/Qwen3MoeForCausalLM/Qwen3-235B-A22B.json
{
  "engine_params": {
    "maxSeqLen": 4096,
    "maxInputTokenLen": 4000,
    "maxIterTimes": 4096,
    "truncation": true,
    "isMOE": true,
    "tp": 16,
    "dp": -1,
    "moe_tp": 16,
    "moe_ep": -1,
    "cacheBlockSize": 128,
    "maxPrefillBatchSize": 50,
    "maxPrefillTokens": 8192,
    "maxBatchSize": 200,
    "trustRemoteCode": true,
    "isMTP": true
  },
  "meta": {
    "architecture": "Qwen3MoeForCausalLM",
    "model_type": "llm",
    "source": "ascend_default.json mindie 配置段",
    "tested_hardware": "Ascend 910B x16",
    "notes": "Qwen3 MoE 架构, isMTP=true 启用投机采样"
  }
}
```

---

## 4. MindIE 特有的控制标志参数

MindIE 配置中有几个参数不是直接传递给 `config.json` 的字段，而是**控制适配器行为的标志**:

| 标志 | 类型 | 作用 |
|------|------|------|
| `isMOE` | bool | 触发 `_build_model_config_overrides()` 中 MOE 参数分支 (tp/dp/moe_tp/moe_ep) |
| `isMTP` | bool | 触发 `plugin_params` 注入 (MTP 投机采样) |
| `sp` | int/null | 非 None 时设置 `sp` 序列并行参数 |
| `cp` | int/null | 非 None 时设置 `cp` 上下文并行参数 |

这些标志在文件化配置中同样作为 `engine_params` 的 key 传递。适配器通过 `engine_config.get("isMOE", False)` 读取，与现有行为完全一致。

---

## 5. MindIE config.json 合并流程（不变）

文件化配置的引入**不改变** MindIE 的最终配置生成流程:

```
engine_params (来自文件化配置或 model_deploy_config)
  ↓
mindie_adapter._build_server_overrides()      → server dict
mindie_adapter._build_backend_overrides()     → backend dict
mindie_adapter._build_model_deploy_overrides() → model_deploy dict
mindie_adapter._build_model_config_overrides() → model_config dict
mindie_adapter._build_schedule_overrides()    → schedule dict
  ↓
overrides_json = json.dumps({
    "server": server,
    "backend": backend,
    "model_deploy": model_deploy,
    "model_config": model_config,
    "schedule": schedule,
})
  ↓
_build_config_merge_script()  → 生成内联 Python 合并脚本
  ↓
start_command.sh 执行:
  1. 读取镜像内原始 config.json
  2. 用 overrides dict.update() 逐层覆盖
  3. 写回 config.json
  4. 启动 mindieservice_daemon
```

---

## 6. 从旧格式迁移 MindIE 模型

### 6.1 旧格式 (ascend_default.json) 中的 MindIE 模型

```json
{
  "model_deploy_config": {
    "llm": {
      "DeepseekV3ForCausalLM": {
        "DeepSeek-R1-w8a8": {
          "mindie": {
            "maxSeqLen": 4096,
            "maxInputTokenLen": 4000,
            "maxIterTimes": 4096,
            "truncation": true,
            "isMOE": true,
            "tp": 16,
            "dp": -1,
            "moe_tp": 16,
            "moe_ep": -1,
            ...
          }
        }
      }
    }
  }
}
```

### 6.2 新格式（文件化配置）

将上述 `"mindie": {...}` 内容直接复制为新文件的 `engine_params`，**参数名和值完全不变**:

```json
{
  "engine_params": {
    "maxSeqLen": 4096,
    "maxInputTokenLen": 4000,
    "maxIterTimes": 4096,
    "truncation": true,
    "isMOE": true,
    "tp": 16,
    "dp": -1,
    "moe_tp": 16,
    "moe_ep": -1,
    ...
  }
}
```

### 6.3 迁移验证

```python
# 旧路径: ascend_default.json → model_deploy_config[llm][DeepseekV3...][DeepSeek-R1-w8a8][mindie]
# 新路径: config/models/mindie/2.3/DeepseekV3ForCausalLM/DeepSeek-R1-w8a8.json → engine_params

# 两者输出到 mindie_adapter 的 engine_config 应完全一致
assert old_engine_config == new_engine_config
```

---

## 7. MindIE 分布式配置

MindIE 的分布式行为由以下参数控制:
- `multiNodesInferEnabled: true` (自动由 nnodes > 1 触发)
- `worldSize` (自动计算: nnodes × nproc_per_node)
- `interCommTLSEnabled`, `interNodeTLSEnabled` (安全通信)

这些参数中:
- `multiNodesInferEnabled` 和 `worldSize` 由 `mindie_adapter.py` 根据运行时 nnodes 自动计算，**不应放入 `distributed_overrides`**
- `maxSeqLen`, `maxIterTimes` 等可能在分布式场景需要调大，**适合放入 `distributed_overrides`**

```jsonc
{
  "engine_params": {
    "maxSeqLen": 4096,
    ...
  },
  "distributed_overrides": {
    "engine_params": {
      "maxSeqLen": 8192,       // 分布式场景可用更长序列
      "maxIterTimes": 8192,
      "maxBatchSize": 400      // 分布式可支持更大 batch
    }
  }
}
```
