# 模型级文件化配置方案导读

本系列文档描述 `model_deploy_config` 的演进方向。经过本轮评审后，方案从“文件查找替换旧逻辑”收敛为“文件链分层继承 + 旧逻辑兜底”，这样既能减少重复配置，也更适合渐进迁移。

---

## 1. 本轮评审后的四个关键决策

| 决策 | 结论 | 原因 |
|------|------|------|
| 文件查找策略 | 由“命中即停”改为“低层默认 + 高层覆盖” | 否则版本默认和架构默认必须在每个模型文件里重复拷贝，和“降低重复、按版本管理”的目标冲突 |
| env 传递方式 | 不再通过 `_model_env_vars` 之类的私有键挂在 `merged` 上，改为结构化返回值 | 现有启动链路里 `merged` 会继续流向 adapter；把运行时 env 作为 side-channel 携带，可读性和边界都不够清晰 |
| 首版 env 能力 | 首版只支持 `env_vars`，不支持 `env_scripts` | `env_scripts` 会引入执行顺序、安全校验和引擎耦合问题，首版收益不够高；脚本能力已有 `env_overrides` 可承接 |
| 文件命名与匹配 | 目录精确匹配，模型文件大小写不敏感匹配 | 目录层级需要稳定、可审计；模型名则要兼容当前大小写混用的实际情况 |

---

## 2. 推荐阅读顺序

1. [design-model-file-config.md](design-model-file-config.md)
   先看主设计，明确目录结构、查找链、合并顺序和迁移边界。
2. [design-model-file-config-schema.md](design-model-file-config-schema.md)
   再看 Schema、命名规则和示例，确认配置文件怎么写。
3. [design-model-file-config-integration.md](design-model-file-config-integration.md)
   最后看集成方案，确认代码接入点和返回值设计。
4. [design-model-file-config-mindie.md](design-model-file-config-mindie.md)
   仅在涉及 MindIE 时阅读，关注扁平参数与嵌套 `config.json` 的映射。

---

## 3. 文档索引

| 文档 | 内容 | 定位 |
|------|------|------|
| [design-model-file-config.md](design-model-file-config.md) | 主设计：目录结构、分层继承、env 原则、迁移策略 | 方案主文档 |
| [design-model-file-config-schema.md](design-model-file-config-schema.md) | Schema、命名规范、校验规则、示例 | 配置编写规范 |
| [design-model-file-config-integration.md](design-model-file-config-integration.md) | 集成点、辅助函数、返回值结构、测试建议 | 开发落地说明 |
| [design-model-file-config-mindie.md](design-model-file-config-mindie.md) | MindIE 参数映射与迁移方式 | MindIE 专项 |
| [design-engine-version-defaults-analysis.md](design-engine-version-defaults-analysis.md) | 版本默认配置前置分析 | 背景材料 |

---

## 4. 一句话方案概要

最终方案是在 `config/models/{engine}/{version}/{architecture}/{model}.json` 目录下维护可组合的配置层：

- `engine/_default.json`
- `engine/{version}/_default.json`
- `engine/{version}/{architecture}/_default.json`
- `engine/{version}/{architecture}/{model}.json`

加载时按上述顺序从低到高叠加；如果四层都未命中，再回退到现有 `model_deploy_config` 逻辑。
