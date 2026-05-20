# 按模型文件化配置方案设计

> 关联文档:
> - [model-file-config-schema.md](model-file-config-schema.md)
> - [model-file-config-integration.md](model-file-config-integration.md)
> - [model-file-config-mindie.md](model-file-config-mindie.md)
> - [engine-version-defaults-analysis.md](engine-version-defaults-analysis.md)

---

## 1. 背景与目标

当前模型启动参数主要依赖 `model_deploy_config`，它内嵌在 `vllm_default.json`、`nvidia_default.json`、`ascend_default.json` 等大文件里。这个结构在模型数量和引擎版本都增加后暴露出几个问题：

1. 单文件持续膨胀，Git 冲突频繁。
2. 版本差异无处安放，`vLLM 0.16` 和 `0.17` 很难分别维护。
3. 迁移粒度过粗，新增一个模型往往需要修改多个共享大文件。
4. 模型级环境变量缺少统一管理入口。

本方案的目标不是简单把旧 JSON“拆散”，而是把配置拆成可组合的层级，让“版本默认”“架构默认”“模型差异”分别落在不同文件里。

---

## 2. 评审结论与优化点

| 主题 | 原方案问题 | 优化后结论 | 原因 |
|------|-----------|-----------|------|
| 文件查找 | 命中任一层即停止，不做横向 merge | 改为固定四层分层继承 | 命中即停会迫使模型文件重复书写版本默认和架构默认，维护成本会重新膨胀 |
| env 传递 | 通过 `_model_env_vars` 挂在 `merged` 上传递 | 解析结果与 `merged` 分离，单独携带 `env_vars` | `merged` 还会继续流向 adapter，side-channel 方式不利于边界清晰和后续扩展 |
| env 能力范围 | 同时支持 `env_vars` 和 `env_scripts` | 首版只支持 `env_vars` | `env_scripts` 会引入执行顺序、安全校验和适配器耦合问题，收益低于复杂度 |
| 文件命名 | 文档一处说大小写不敏感，一处又按精确文件名查找 | 目录精确匹配，模型文件名大小写不敏感查找 | 目录层级需要稳定，模型名则要兼容当前大小写混用的实际使用方式 |
| 迁移方式 | 迁移一个模型时容易变成“全量复制原配置” | 允许先落版本/架构默认，再追加模型 delta 文件 | 这样迁移是增量的，模型文件只保留差异项 |

---

## 3. 目录结构

```
config/models/
├── vllm/
│   ├── _default.json
│   └── 0.17/
│       ├── _default.json
│       ├── DeepseekV3ForCausalLM/
│       │   ├── _default.json
│       │   └── DeepSeek-R1.json
│       └── Qwen3MoeForCausalLM/
│           ├── _default.json
│           └── Qwen3-235B-A22B.json
├── sglang/
│   └── 0.4/
│       └── DeepseekV3ForCausalLM/
│           ├── _default.json
│           └── DeepSeek-R1.json
├── vllm_ascend/
│   └── 0.17/
│       └── DeepseekV3ForCausalLM/
│           └── DeepSeek-R1-w8a8.json
└── mindie/
    └── 2.3/
        └── DeepseekV3ForCausalLM/
            └── DeepSeek-R1-w8a8.json
```

### 为什么保留 `engine/_default.json`

这是最低层的兜底配置，负责承接“该引擎所有版本都成立”的通用默认值。这样在 `ENGINE_VERSION` 缺失、版本无法解析，或只是想给某个引擎提供一组基础默认时，都有稳定落点。

### 为什么暂不引入 `model_type` 目录层

旧逻辑有 `llm / embedding / rerank` 的类型层，但从当前已验证模型看，真正高频变化的是“版本差异”和“架构差异”，而不是类型差异。首版先聚焦 `engine / version / architecture / model` 四维；如果后续出现大量同类型、跨架构共用的默认项，再评估是否补一个 `model_type` 层，而不是一开始就把目录复杂度抬高。

---

## 4. 查找与合并规则

### 4.1 文件层级

从低优先级到高优先级，固定按以下四层尝试：

1. `config/models/{engine}/_default.json`
2. `config/models/{engine}/{version}/_default.json`
3. `config/models/{engine}/{version}/{architecture}/_default.json`
4. `config/models/{engine}/{version}/{architecture}/{model_name}.json`

### 4.2 核心规则

不是“命中即停”，而是“按层收集、按顺序叠加”：

```text
L4 engine/_default.json                  ← 基础引擎默认
  → L3 engine/version/_default.json      ← 版本默认覆盖
  → L2 engine/version/arch/_default.json ← 架构默认覆盖
  → L1 engine/version/arch/model.json    ← 模型精确覆盖
```

只有在四层全部未命中时，才回退到现有 `model_deploy_config` 逻辑。

### 4.3 为什么不能用“命中即停”

以 `DeepSeek-R1` 为例，如果命中 `DeepSeek-R1.json` 就停止，那么以下内容都必须在每个模型文件里重复写一遍：

- `VLLM_USE_V1=1`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `gpu_memory_utilization`
- `block_size`
- `seed`

这等于把原来大文件里的重复搬到了更多小文件里，不但没有减复杂度，反而更容易出现漂移。

### 4.4 文件命中后的边界

一旦四层里任意一层命中，最终结果只来自“文件化配置链”本身，不再继续叠加旧 `model_deploy_config`。原因是：

1. 避免同一个字段同时来自新旧两套来源，审计时无法判断来源。
2. 渐进迁移时，配置拥有者必须明确，不能一半来自新目录、一半来自旧大文件。
3. 文件链已经提供了默认层和差异层，不需要再用旧逻辑补齐。

---

## 5. 文件内容模型

首版配置文件只保留声明式字段：

```jsonc
{
  "engine_params": {},
  "env_vars": {},
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
  "meta": {}
}
```

### 为什么首版不支持 `env_scripts`

`env_scripts` 会让文档方案立刻引入三类问题：

1. 这些脚本应该在 `wings_entry` 注入，还是在各 adapter 的环境链里注入，顺序难统一。
2. 路径安全、容器路径存在性、是否依赖基础环境脚本，都需要额外约束。
3. 当前系统已经有 `env_overrides` 目录用于脚本型扩展，重复建设收益不高。

因此首版只允许 `env_vars`。如果后续确实需要脚本型能力，优先在 `env_overrides` 机制上扩展，而不是在模型配置里直接开放执行入口。

---

## 6. 运行时合并顺序

在文件链解析完成后，最终参数按下面顺序合并：

```text
1. 文件链基础层: engine/_default → version/_default → arch/_default → model.json
2. distributed_overrides            (仅 distributed=true 时叠加)
3. hardware_variants[matched]       (仅命中硬件变体时叠加)
4. _merge_cmd_params()              (运行时派生参数，如 host/port、tp 等)
5. user_config                      (--config-file)
6. CLI / ENV 显式覆盖              (最高优先级)
```

### `distributed_overrides` 与 `hardware_variants` 的先后顺序

先分布式、后硬件变体。原因是硬件变体通常代表更精确的“最后一跳调优”，例如 H20-141G 的专属吞吐参数，它应该比通用分布式默认更高一层。

---

## 7. 环境变量处理原则

### 7.1 首版支持范围

模型文件只支持 `env_vars`，用于表达和模型或引擎版本直接相关的叶子变量，例如：

```bash
VLLM_USE_V1=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_ASCEND_ENABLE_NZ=0
HCCL_OP_EXPANSION_MODE=AIV
```

### 7.2 与现有脚本注入机制的关系

`env_vars` 是“模型配置能力”；`set_vllm_ascend_env.sh`、`set_mindie_env.sh` 这类基础环境脚本是“引擎启动前提”。两者不是同一层概念。

因此文档口径定义为：

1. 模型文件只负责模型/版本差异变量，不负责 toolkit/bootstrap 级变量。
2. operator 级脚本和临时环境修正继续走 `ENV_OVERRIDES_DIR`。
3. 如果某个环境变量实际上属于引擎基础初始化而不是模型差异，就不应该写进模型文件。

这样可以避免“模型配置是否能覆盖基础环境脚本”的语义争议。

---

## 8. 版本与名称匹配

### 8.1 版本目录

```
ENGINE_VERSION=v0.17.2  → 0.17
ENGINE_VERSION=0.4.1    → 0.4
ENGINE_VERSION=v2.3.0   → 2.3
```

规则：

1. 只解析 `major.minor`。
2. 目录严格精确匹配，不做范围匹配，不做 `latest` 软链。
3. 版本缺失或解析失败时，只使用 `engine/_default.json`。

### 8.2 名称匹配

1. `engine`、`version`、`architecture` 目录名精确匹配。
2. 模型文件名允许大小写不敏感匹配，但 loader 必须记录实际命中的规范文件名。
3. `model_name` 必须先做路径安全校验，禁止出现 `/`、`\`、`..`。

---

## 9. 迁移策略

### 9.1 渐进迁移原则

迁移不是“先把旧 JSON 全量拷贝一遍”，而是按层落文件：

1. 先补 `engine/_default.json`。
2. 再补版本级 `_default.json`。
3. 再补架构级 `_default.json`。
4. 最后只为有差异的模型补 `model.json`。

这样一个模型迁移过来后，模型文件通常只保留真正的 delta，而不是整个旧配置副本。

### 9.2 新旧方案边界

```text
若四层文件全部未命中:
    走旧 model_deploy_config

若至少命中一层文件:
    只走文件链
```

这是为了保持配置来源单一、便于审计和回滚。

---

## 10. 初始迁移建议

优先迁移以下高价值组合：

1. `vllm / 0.17 / DeepseekV3ForCausalLM / DeepSeek-R1`
2. `sglang / 0.4 / DeepseekV3ForCausalLM / DeepSeek-R1`
3. `vllm_ascend / 0.17 / DeepseekV3ForCausalLM / DeepSeek-R1-w8a8`
4. `mindie / 2.3 / DeepseekV3ForCausalLM / DeepSeek-R1-w8a8`

原因：

1. 这几条链同时覆盖版本差异、架构差异、硬件变体和 Ascend 特有 env。
2. 它们是现有方案里最复杂、最能暴露设计问题的路径。
3. 先把最复杂路径跑通，后续迁移 Qwen 系列时复用性最高。
