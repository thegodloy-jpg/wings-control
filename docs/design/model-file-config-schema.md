# 模型文件化配置方案 Schema 与命名规范

> 关联文档: [model-file-config.md](model-file-config.md)

---

## 1. 命名与路径规范

### 1.1 目录层级

```
config/models/{engine}/{version}/{architecture}/{model_name}.json
```

其中四个作用域分别表示：

| 层级 | 规则 | 示例 |
|------|------|------|
| `engine` | 与 `ENGINE` 保持一致 | `vllm`, `sglang`, `vllm_ascend`, `mindie` |
| `version` | `{major}.{minor}`，不带 `v`，不带 patch | `0.17`, `0.4`, `2.3` |
| `architecture` | 与模型 `config.json` 中 `architectures[0]` 一致 | `DeepseekV3ForCausalLM` |
| `model_name` | 与 `MODEL_NAME` 对应，扩展名固定为 `.json` | `DeepSeek-R1.json` |

另外允许以下默认文件：

```text
config/models/{engine}/_default.json
config/models/{engine}/{version}/_default.json
config/models/{engine}/{version}/{architecture}/_default.json
```

### 1.2 大小写策略

1. `engine`、`version`、`architecture` 目录名采用精确匹配。
2. `model_name` 文件允许大小写不敏感匹配。
3. loader 命中后必须记录真实文件名，避免日志里只看到用户输入而看不到规范文件。

原因是目录层级属于稳定命名空间，必须保持确定性；模型名则要兼容当前大小写混用的现实情况。

### 1.3 特殊字符约束

| 字符类别 | 是否允许 | 说明 |
|----------|----------|------|
| 字母、数字、连字符、下划线、点号 | 允许 | 如 `DeepSeek-V3.1.json` |
| 空格 | 不建议 | 容易引入运维和脚本兼容性问题 |
| `/`、`\`、`..` | 禁止 | 必须在加载前做路径安全校验 |

---

## 2. 顶层 Schema

首版配置文件允许的顶层字段如下：

```jsonc
{
  "engine_params": {
    "key": "value"
  },
  "env_vars": {
    "VLLM_USE_V1": "1"
  },
  "distributed_overrides": {
    "engine_params": {},
    "env_vars": {}
  },
  "hardware_variants": {
    "H20-141G": {
      "engine_params": {},
      "env_vars": {}
    }
  },
  "meta": {
    "architecture": "DeepseekV3ForCausalLM",
    "model_type": "llm",
    "source": "manual tuning",
    "tested_hardware": "H20-141G",
    "notes": "optional"
  }
}
```

### 为什么没有 `env_scripts`

`env_scripts` 不纳入首版 schema。原因如下：

1. 它会把配置文件从“声明式参数”变成“可执行入口”，安全边界明显变复杂。
2. 这些脚本究竟应当位于 `wings_entry` 前缀、还是 adapter 环境链中，顺序很难统一。
3. 当前已有 `ENV_OVERRIDES_DIR` 承担脚本注入职责，首版没有必要重复造一条执行链。

---

## 3. 字段约束

| 字段 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `engine_params` | `Dict[str, Any]` | 否 | `{}` | 引擎原生参数名，不做统一翻译 |
| `env_vars` | `Dict[str, str]` | 否 | `{}` | 仅用于声明式环境变量 |
| `distributed_overrides` | `Dict[str, Dict]` | 否 | `{}` | 仅在 `distributed=true` 时叠加 |
| `hardware_variants` | `Dict[str, Dict]` | 否 | `{}` | 仅在识别到特定硬件变体时叠加 |
| `meta` | `Dict[str, str]` | 否 | `{}` | 审计信息，不参与合并 |

### 3.1 文件最小有效内容

配置文件至少要对运行结果产生一种贡献，即以下四类字段中至少有一个非空：

- `engine_params`
- `env_vars`
- `distributed_overrides`
- `hardware_variants`

不再强制要求 `engine_params` 非空。原因是版本级文件可能只承担 env 差异，例如只设置 `VLLM_USE_V1=1`。

### 3.2 值类型约束

1. `env_vars` 的 value 必须是字符串。
2. `engine_params` 的 value 允许 `string / int / float / bool / null`。
3. 若某个引擎确实需要 list 或浅层 dict，必须在该引擎适配器已支持的前提下显式说明；否则默认不鼓励使用复杂嵌套。
4. `meta.architecture` 若存在，必须与所在目录名一致。
5. 禁止未知顶层 key，防止 `enigine_params` 这类拼写错误静默生效。

---

## 4. 合并规则

### 4.1 跨文件合并

固定按以下顺序从低到高叠加：

```text
engine/_default.json
  → version/_default.json
  → architecture/_default.json
  → model.json
```

每一层都只覆盖自己声明的字段，未声明的字段继续继承下层结果。

### 4.2 单文件内部合并

在文件链合并完成后，再应用当前文件内的运行时分支：

```text
base
  → distributed_overrides         (if distributed=true)
  → hardware_variants[matched]    (if matched)
```

`hardware_variants` 的优先级高于 `distributed_overrides`，因为它表示更细粒度的“最后一跳调优”。

### 4.3 旧逻辑兜底

只有在四层文件全部未命中时，才回退到旧 `model_deploy_config`。一旦至少命中一层文件，就不再叠加旧逻辑，避免新旧来源混杂。

---

## 5. 示例

### 5.1 版本级默认

```jsonc
// config/models/vllm/0.17/_default.json
{
  "engine_params": {
    "gpu_memory_utilization": 0.85,
    "enable_prefix_caching": true,
    "block_size": 16,
    "max_num_seqs": 256,
    "seed": 42
  },
  "env_vars": {
    "VLLM_USE_V1": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn"
  },
  "meta": {
    "source": "vLLM 0.17 baseline"
  }
}
```

### 5.2 架构级默认

```jsonc
// config/models/vllm/0.17/DeepseekV3ForCausalLM/_default.json
{
  "engine_params": {
    "trust_remote_code": true,
    "enable_expert_parallel": true,
    "enable_auto_tool_choice": true,
    "tool_call_parser": "deepseek_v3"
  },
  "meta": {
    "architecture": "DeepseekV3ForCausalLM",
    "model_type": "llm"
  }
}
```

### 5.3 模型级 delta

```jsonc
// config/models/vllm/0.17/DeepseekV3ForCausalLM/DeepSeek-R1.json
{
  "engine_params": {
    "max_model_len": 4096
  },
  "meta": {
    "architecture": "DeepseekV3ForCausalLM",
    "source": "model-specific delta"
  }
}
```

这个模型文件只保留自己的差异项，不再重复书写 `VLLM_USE_V1`、`block_size` 等已在低层默认里出现的字段。

### 5.4 SGLang 硬件变体

```jsonc
// config/models/sglang/0.4/DeepseekV3ForCausalLM/DeepSeek-R1.json
{
  "engine_params": {
    "trust_remote_code": true,
    "context_length": 4096,
    "enable_ep_moe": true,
    "tool_call_parser": "deepseekv3"
  },
  "hardware_variants": {
    "H20-96G": {
      "engine_params": {
        "mem_fraction_static": 0.9
      }
    },
    "H20-141G": {
      "engine_params": {
        "mem_fraction_static": 0.95,
        "max_running_requests": 256,
        "enable_dp_attention": true
      }
    }
  },
  "meta": {
    "architecture": "DeepseekV3ForCausalLM",
    "tested_hardware": "H20-96G, H20-141G"
  }
}
```

---

## 6. 校验建议

推荐同时做三层校验：

1. 运行时轻量校验：字段类型、未知 key、路径安全。
2. CI 校验：对 `config/models/**.json` 做 schema 检查。
3. 迁移校验：对关键模型比对“新文件链结果”和“旧 `model_deploy_config` 结果”是否一致。
