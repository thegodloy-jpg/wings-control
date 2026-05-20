# wings_control 包内文档

产品使用文档以仓库根目录的 [../../docs/README.md](../../docs/README.md) 为准。

本目录不再作为用户文档入口。新的文档归属规则：

| 内容类型 | 位置 |
|----------|------|
| 产品总览、兼容性、快速入口 | [../../docs/](../../docs/) |
| Docker Compose / K8s 部署 | [../../docs/deployment/](../../docs/deployment/) |
| PD、分布式、LMCache、Sparse KV 等特性 | [../../docs/features/](../../docs/features/) |
| 设计分析和实现数据流 | [../../docs/design/](../../docs/design/) |
| 历史场景和环境记录 | [../../docs/reference/](../../docs/reference/) |

如果需要新增面向用户的文档，请优先放入仓库根目录 `docs/` 下对应分类，避免包内文档和产品文档重复维护。
