# DeepSeek-V3.1 2x8 长上下文现场重部署核对清单

用于确认现场 Pod、镜像、ConfigMap 和启动脚本已经使用包含修复的版本，避免继续运行旧配置。

## 1. 代码版本

- 最新修复提交应至少包含：
  - `de0462c Remove MindIE default NPU memory override`
  - `772c8b8 Add DeepSeek V3.1 long-context start scripts`
  - `46ccc08 Align MindIE CP SP parallel defaults`
- 如果部署镜像内能看到源码，确认仓库 HEAD 不早于 `de0462c`。

## 2. 启动脚本来源

- 长上下文 2x8 场景必须使用本目录脚本：
  - `start_command_node0.sh`
  - `start_command_node1.sh`
- 不要使用旧的非长上下文目录：
  - `../deepseek_v31_2x8_full/`
- 旧目录是普通 2x8 baseline，可能仍显示 `tp=16`；它不是 CP/SP 长上下文示例。

## 3. config.json 关键字段

现场最终打印的 MindIE `config.json` 必须满足：

| 字段 | 期望值 |
| --- | --- |
| `BackendConfig.ModelDeployConfig.maxSeqLen` | `12048` |
| `BackendConfig.ModelDeployConfig.maxInputTokenLen` | `10000` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].worldSize` | `8` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].dp` | `1` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].tp` | `8` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].sp` | `8` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].cp` | `2` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].moe_tp` | `1` |
| `BackendConfig.ModelDeployConfig.ModelConfig[0].moe_ep` | `16` |
| 根级 `enable_ep_moe` | `true` |

如果看到 `tp=16` 且同时看到 `cp=2`，说明仍在运行旧逻辑或旧脚本。

## 4. 环境变量关键字段

现场 engine 日志中应只出现默认：

- `NPU_MEMORY_FRACTION=0.96`

不应再出现：

- `NPU_MEMORY_FRACTION=0.9`
- `NPU_MEMORY_FRACTION=0.8`

如果仍看到 `0.9`，优先检查 Pod 是否仍挂载旧 `start_command.sh` 或旧 ConfigMap。
如果看到 `0.8`，优先检查镜像内源码是否未包含 `de0462c`。

## 5. 分布式环境关键字段

两个节点都应打印：

| 字段 | node0 | node1 |
| --- | --- | --- |
| `MASTER_ADDR` | master 节点 IP | master 节点 IP |
| `RANK` | `0` | `1` |
| `WORLD_SIZE` | `16` | `16` |
| `MINDIE_MODEL_WORLD_SIZE` | `16` | `16` |
| `RANK_TABLE_FILE` | `/shared-volume/hccl_ranktable.json` | `/shared-volume/hccl_ranktable.json` |

## 6. rank table

- `rank_table_all.json` 必须存在于 `RANK_TABLE_PATH` 指向的位置，默认 `/workspace/rank_table_all.json`。
- adapter 只校验文件存在；复制到共享卷后会把每个 server 内的 `device_id` 归一化为 `0..N-1`。

## 7. 判定结论

- 若最终 config 为 `dp=1,tp=8,sp=8,cp=2` 且 `enable_ep_moe=true`：CP/SP 配置正确。
- 若 `NPU_MEMORY_FRACTION` 只剩 `0.96`：环境默认值没有二次覆盖。
- 若仍出现旧值，优先重建镜像、刷新 ConfigMap，并重启两个 Pod，确保启动脚本来自最新提交。
