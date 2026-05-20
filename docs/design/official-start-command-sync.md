# 官方启动命令来源与定时同步设计

> 状态：方案 2 落地文档。本文只定义官方来源、同步边界和与本项目启动字段拼接链路的关系。运行时不访问远程文档。
>
> 机器可读源目录：`wings_control/config/official_sources.yaml`
>
> 最近校验日期：2026-05-15

## 目标

本机制解决两个问题：

1. 保存 vLLM、SGLang、vLLM-Ascend、MindIE 的官方启动命令和配置说明来源，便于定时抓取。
2. 把抓取结果转换为本项目已有的启动字段维护入口，而不是直接保存一段 shell 字符串。

定时同步的最终产物应该是候选差异报告。只有人工确认后的字段才进入 `manifests`、`mappings`、`deviations`、`env_policies`、`recipes` 或 MindIE 模板。

## 当前启动链路

本项目当前的启动命令不是从官方命令整段复制生成，而是由字段逐层合并后拼接：

```text
wings_start.sh / CLI / env
  -> core/start_args_compat.py
  -> core/config_loader.py::load_and_merge_configs()
  -> config/manifests, mappings, deviations, env_policies, recipes
  -> engines/*_adapter.py
  -> core/wings_entry.py::build_launcher_plan()
  -> /shared-volume/start_command.sh
```

关键约束：

- vLLM 与 vLLM-Ascend 官方命令通常以 `vllm serve` 表达；本项目当前适配器用 `python3 -m vllm.entrypoints.openai.api_server` 拼接。同步时比较的是参数字段和语义，不比较入口命令字符串本身。
- SGLang 当前适配器拼接 `python3 -m sglang.launch_server`。官方 cookbook 和 server arguments 中的参数需要落到 canonical 字段或 SGLang native 字段。
- MindIE 的命令基本固定为 `mindieservice_daemon`。模型和场景差异主要落在 `config.json` 的配置补丁，不应只抓取启动命令字符串。

## 官方源目录

机器可读源保存在 `wings_control/config/official_sources.yaml`。它按引擎记录以下信息：

| 字段 | 含义 |
|------|------|
| `source_type` | 与现有 tooling 兼容，使用 `official_doc`、`vendor_doc`、`vendor_cookbook`、`engine_source` 等类型 |
| `url` | 可抓取的官方页面、索引页或源码入口 |
| `crawl_mode` | 建议抓取方式，例如 HTML code block、CLI table、plain text index |
| `extract_targets` | 抽取目标，例如命令示例、CLI 参数、env、MindIE config patch |
| `target_config_files` | 人工确认后可能更新的本地配置文件 |

当前源口径：

| 引擎 | 主要来源 | 抽取重点 | 本地落点 |
|------|----------|----------|----------|
| vLLM | OpenAI-compatible server、serve args、CLI reference | `vllm serve` 参数、配置文件优先级、模型/特性示例 | `manifests/vllm`、`mappings`、`recipes` |
| SGLang | `llms.txt`、server arguments、cookbook | 模型 cookbook、`launch_server` 参数、TP/DP/多节点示例 | `manifests/sglang`、`mappings`、`recipes` |
| vLLM-Ascend | tutorials、model tutorials、env vars | Ascend 模型场景、NPU env、DP/PD/CP/Ray 场景 | `manifests/vllm_ascend`、`env_policies/vllm_ascend.yaml`、`deviations/vllm_ascend.yaml`、`recipes` |
| MindIE | quickstart、daemon command、service config params | `config.json` 字段、模型部署参数、daemon 启动方式 | `templates/mindie_service_config.json`、`deviations/mindie.yaml`、`env_policies/mindie.yaml` |

## 归一化记录

抓取工具不应直接修改运行时配置。它先生成如下候选记录：

```json
{
  "engine": "vllm_ascend",
  "source_id": "vllm_ascend-latest-tutorials",
  "scenario_id": "deepseek_v31_multi_node_dp",
  "source_type": "official_doc",
  "source_url": "https://docs.vllm.ai/projects/ascend/en/latest/tutorials/",
  "model_family": "DeepSeek",
  "hardware": ["Ascend"],
  "official_entrypoint": "vllm serve",
  "env": {
    "ASCEND_RT_VISIBLE_DEVICES": "inherit"
  },
  "cli_args": {
    "model": "model repo or local path",
    "tensor_parallel_size": 8
  },
  "config_patch": {},
  "target_updates": [
    "wings_control/config/recipes/models/DeepSeek-V3.1.yaml",
    "wings_control/config/env_policies/vllm_ascend.yaml"
  ]
}
```

MindIE 候选记录以 `config_patch` 为主：

```json
{
  "engine": "mindie",
  "source_id": "mindie-service-config-params",
  "scenario_id": "text_generation_service",
  "official_entrypoint": "./bin/mindieservice_daemon",
  "cli_args": {},
  "env": {},
  "config_patch": {
    "BackendConfig.ModelDeployConfig.maxSeqLen": 2560,
    "BackendConfig.ModelDeployConfig.ModelConfig[0].modelWeightPath": "/path/to/model"
  },
  "target_updates": [
    "wings_control/config/templates/mindie_service_config.json",
    "wings_control/config/deviations/mindie.yaml"
  ]
}
```

## 字段进入本项目的规则

| 官方抽取对象 | 进入位置 | 规则 |
|--------------|----------|------|
| CLI 参数存在性、类型、默认值 | `config/manifests/<engine>/<version>.yaml` | 用于 orphan check 和版本差异，不直接代表 Wings 默认值 |
| 通用概念到引擎原生字段 | `config/mappings/canonical_to_engines.yaml` | 例如 `max_model_len` 到 SGLang `context_length`、MindIE `maxSeqLen` |
| Wings 与上游不同的安全默认 | `config/deviations/*.yaml` | 必须记录 `source_type`、`rationale`、`decision_date`，不能由爬虫自动覆盖 |
| 环境变量默认值和覆盖策略 | `config/env_policies/*.yaml` | 区分 `inherit`、`idempotent`、`force_override` |
| 架构或模型级推荐启动字段 | `config/recipes/architectures`、`config/recipes/models` | 只保存可复用默认和模型场景，不保存部署私有路径 |
| MindIE 服务化 JSON 字段 | `config/templates/mindie_service_config.json`、`deviations/mindie.yaml` | 命令固定，差异落到配置 patch |

## 定时同步流程

建议的定时任务每周执行一次：

```text
1. 读取 wings_control/config/official_sources.yaml
2. 抓取 enabled=true 的官方源
3. 抽取代码块、CLI 参数表、文档索引和 MindIE 配置表
4. 归一化为 candidate records
5. 与当前 manifests/mappings/deviations/env_policies/recipes 做差异比对
6. 输出报告到 build/official_start_command_refs/<date>/
7. 人工确认后再修改 config carrier
8. 执行 lint、render 和 start_command 快照验证
```

建议的校验命令：

```powershell
python tools/lint_mappings.py
python tools/lint_deviations.py
python tools/lint_env_policies.py
python tools/lint_recipes.py
python tools/lint_cross_refs.py
python tools/render_effective_config.py --model DeepSeek-V3.1 --engine vllm_ascend --json
pytest tests/test_phase_d_tooling_v36.py tests/test_script_snapshots.py
```

建议的 GitHub Actions 定时入口：

```yaml
on:
  schedule:
    - cron: "0 2 * * 1"
  workflow_dispatch: {}
```

该 workflow 应只生成差异报告或 PR，不应在主分支静默改写运行时配置。

## 风险控制

- 不在 `start_command.sh` 生成路径中访问远程文档。
- 不把官方示例中的模型路径、容器镜像、私有端口直接写为默认值。
- 不自动覆盖 `source_type: wings_decision` 或 `source_type: wings_validation_run` 的字段。
- 不把 MindIE 的 daemon 命令当成全部信息，必须抓取 `config.json` 字段。
- vLLM/vLLM-Ascend 的 `vllm serve` 与本项目 Python module 入口只做参数语义对齐。
- SGLang cookbook 是模型场景参考，进入 `recipes` 前需要确认模型架构、字段名和版本范围。

## 后续实现接口

后续可新增 `tools/sync_official_start_commands.py`，但接口应保持只读和可审计：

```powershell
python tools/sync_official_start_commands.py `
  --source-catalog wings_control/config/official_sources.yaml `
  --output-dir build/official_start_command_refs `
  --mode report
```

可选输出：

- `candidates.jsonl`：归一化候选记录。
- `source_fetch_log.json`：URL、状态码、hash、抓取时间。
- `config_diff.md`：面向人工 review 的差异说明。
- `manifest_seed/<engine>/<version>.json`：可传给 `tools/build_manifest.py --from-json` 的种子。

只有当 `--mode apply` 被显式启用，并且本地校验通过时，工具才允许生成配置修改补丁。
