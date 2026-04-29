# DeepSeek-V3.1 MindIE 1×16 长上下文最终配置示例

本目录保存 DeepSeek-V3.1 在 MindIE 单机 16 卡长上下文场景下，经过 `wings-control` 参数合并和 MindIE adapter 配置覆盖后的最终 `config.json` 示例。

## 场景

- 引擎：MindIE
- 模型：DeepSeek-V3.1
- 拓扑：1 节点 × 16 NPU
- 长上下文示例：`input_length=10000`，`output_length=2048`
- `maxSeqLen=12048`
- `maxInputTokenLen=2048`
- MindIE 服务端口：`17000`
- 多节点开关：`multiNodesInferEnabled=false`

## 文件

- `config.json`：合并 MindIE 模板后的最终配置示例。
- `start_command_1x16.sh`：engine 容器侧执行的启动脚本示例，包含环境初始化、配置覆盖合并和 `mindieservice_daemon` 启动流程。

## 关键并行参数

| 字段 | 值 |
| --- | --- |
| `worldSize` | `16` |
| `dp` | `1` |
| `tp` | `8` |
| `sp` | `8` |
| `cp` | `2` |
| `npuDeviceIds` | `[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]` |

## Function Call

DeepSeek-V3.1 会注入 MindIE Function Call parser：

- `BackendConfig.ModelDeployConfig.ModelConfig[0].models.deepseekv2.tool_call_options.tool_call_parser=deepseek_v31`

启动时仍需要打开自动工具选择开关，例如：

- CLI：`--enable-auto-tool-choice`
- 或环境变量：`ENABLE_AUTO_TOOL_CHOICE=true`

## CP/SP 自动触发规则

当前自动 CP/SP 只支持总 16 卡长上下文场景：

- 1×16：自动生成 `dp=1, cp=2, tp=8, sp=8`
- 2×8：自动生成 `dp=1, cp=2, tp=8, sp=8`

2×16 不再自动触发 CP/SP；如果确实需要 2×16 的特殊并行参数，需要显式传入。
