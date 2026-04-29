# wings_control 设计文档

本目录存放 `wings_control` 的设计文档与方案分析。当前推荐以“模型级文件化配置”系列为准；`design-engine-version-defaults-analysis.md` 保留为前置分析和设计背景，不作为最终落地口径。

## 推荐入口

| 文档 | 说明 |
|------|------|
| [design-model-file-config-readme.md](design-model-file-config-readme.md) | 系列导读，包含本轮评审后的关键决策与阅读顺序 |
| [design-model-file-config.md](design-model-file-config.md) | 主设计文档，定义目录结构、分层继承、迁移边界和 env 处理原则 |
| [design-model-file-config-schema.md](design-model-file-config-schema.md) | Schema、命名规则、校验规则和配置示例 |
| [design-model-file-config-integration.md](design-model-file-config-integration.md) | 集成方案，说明如何接入 `config_loader.py` 和 `wings_entry.py` |
| [design-model-file-config-mindie.md](design-model-file-config-mindie.md) | MindIE 专项说明，解释扁平参数到嵌套 `config.json` 的映射 |
| [design-engine-version-defaults-analysis.md](design-engine-version-defaults-analysis.md) | 背景分析文档，记录版本默认配置方案的评审与演进路径 |
| [vllm-advanced-start-command.md](vllm-advanced-start-command.md) | vLLM / vLLM-Ascend 高级特性开启后 `start_command.sh` 关键片段验证 |

## 本轮评审后的统一结论

1. 文件查找不再采用“命中即停”的独占模式，而是采用“低层默认 + 高层覆盖”的分层继承。
2. 模型配置解析结果需要与 `merged` 参数分离，避免通过私有键做 side-channel 传递。
3. `env_scripts` 不纳入首版模型配置能力，首版只支持声明式 `env_vars`，脚本型注入继续走现有 `env_overrides` 机制。
4. 目录名保持精确匹配，模型文件名允许大小写不敏感查找，以兼容当前 `MODEL_NAME` 使用习惯并减少迁移摩擦。
