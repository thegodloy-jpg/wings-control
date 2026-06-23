---
name: model-day0-adapt
description: >-
  端到端「新模型/新版本 Day0 适配」向导，用于把 wings-control 产出的启动命令对齐到官方
  recipe 标准。强约束：产出的「CLI flag 名 ∪ JSON 嵌套 key 路径 ∪ 环境变量名」三集合必须
  与契约精确相等（数值可差异）；新逻辑只能挂在已有结构上，不开并行路径。
  TRIGGER when: 要把某模型/某引擎版本适配/接入 wings-control；要让 vllm / vllm_ascend
  启动命令对齐官方 recipe/tutorial；提到 day0 / 启动命令对齐 / 三方对比 / 新增模型架构或
  模型名识别 / 对齐启动字段或环境变量。
  SKIP when: 与适配无关的纯 bug 修复；仅询问已有模型当前行为且不改动。
---

# 新模型 Day0 适配向导

> 本仓库（wings-control）专用。把「让启动命令对齐官方标准」固化成可执行流程。
> 任何「适配新模型 / 对齐启动命令 / 新版本接入」的任务，**先按本 skill 的 Phase 0→6 走**。

## 0. 核心不变量（任何一步都不得违反）

**让 wings 产出的「CLI flag 名 ∪ 配置 JSON 嵌套 key 路径 ∪ 环境变量名」三集合与契约精确相等；数值（max_model_len、gpu_util、TP/DP、HCCL_BUFFSIZE 等）允许由页面/平台/device 驱动而不同；新增逻辑只能挂在已有结构/函数上，不开并行代码路径。**

判定粒度（"字段"的定义）：
- CLI flag 名，如 `--enable-expert-parallel`、`--async-scheduling`
- **配置 JSON 内的嵌套 key 路径**，如 `additional_config.enable_npugraph_ex`、`speculative_config.num_speculative_tokens`（嵌套 key 也算字段）
- 环境变量名，如 `VLLM_ASCEND_BALANCE_SCHEDULING`、`VLLM_ASCEND_ENABLE_FLASHCOMM1`

「字段名出现/缺失」= 违约；交集内「值不同」= 允许。

---

## Phase 0 — 抓契约（三方对比定标准）

三个来源，角色不同，**不可等价混用**：

| 代号 | 来源 | 角色 | 权威性 |
|---|---|---|---|
| **甲** | 用户提供的官方启动脚本 | 部署口径真值（强约束的"标准"就是它） | 最高 |
| **乙** | 联网抓取的 recipe（vllm-ascend tutorial / recipes.vllm.ai / HF 模型卡） | 官方公开口径，交叉佐证 | 参考 |
| **丙** | wings dry_run 当前产出 | 现状基线（被改造对象） | — |

**动作**：
1. 接收/索取 甲（用户脚本）。**同时**用 WebFetch/WebSearch 抓 乙（必抓，不可省）。
2. 把 甲、乙 各自解析成三集合：CLI flag 名 ∪ JSON 嵌套 key 路径 ∪ env 名（并记录各自数值，供回落默认）。
3. **甲 ⟷ 乙 判定**：
   - 一致 → 契约稳固。
   - **字段名/env 名集合分歧**（一方有某字段另一方无）→ ⚠ **必须停下来 escalate 给用户拍板**取哪方，不得自行猜测（强约束是字段集合精确相等，集合本身不同就无法定契约）。
   - **仅数值分歧**（字段名相同、值不同）→ 两个值都记录，契约取 **甲**（部署真值），乙的值作回落/备注。
   - **乙 抓取失败**（429 / 版本页缺失）→ 降级为「仅甲」，并在契约上显式标注"未在线交叉核对"，转入 Phase 6 待确认。
4. 产出：契约三集合 + 数值表 + 出处链接（甲来源 + 乙 URL）。

---

## Phase 1 — 基线追踪

跑 `python dry_run.py --scenario <近邻场景>`（或临时加一个场景）得到 wings 当前对该模型/架构的产出 = **丙**，解析成三集合。
参考既有场景写法见 [dry_run.py](../../../dry_run.py) 与 day0 文档里的「实际生成命令」章节。

---

## Phase 2 — 双向 diff + 三方矩阵（核心工件）

逐字段建一张矩阵，这是整个适配的中枢：

| 字段/env 名（含嵌套路径） | 甲(脚本) | 乙(recipe) | 丙(现状) | 甲乙判定 | 契约 vs 丙 | 落点 |
|---|---|---|---|---|---|---|
| `speculative_config.num_speculative_tokens` | 3 | 5 | 1 | 分歧·escalate→取甲 | 值差 | 代码 gate |
| `additional_config.enable_npugraph_ex` | ✓ | ✓ | ✓(嵌套) | 一致 | 结构对齐 | arch default |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | =1 | 无 | 无 | 分歧·取甲 | 缺 env | DP env builder |

- `契约 − 丙` = **缺字段**（违约，必须补）
- `丙 − 契约` = **多字段**（违约，必须删）
- 交集内值差异 → 忽略（数值自由）

---

## Phase 3 — 承载层决策树（每个差量都过一遍）

**先做识别**：同架构内的派生版本用名称标识切分，仿 [model_utils.py](../../../wings_control/utils/model_utils.py) 里 `is_glm51_model` / `is_glm52_model` 范式：
- 新增 `_XXX_NAME_MARKERS` + `_contains_xxx_marker` + `is_xxx_model`
- 标识集与邻近版本**严格互斥**（不误命中基座 / 上一版本）

**再判落点**：

```
某字段名 / 嵌套 key / env 名「是否出现」？
├─ 依赖运行时条件吗？（平台 a2/a3、单/双机、spec 开关、device_count…）
│   ├─ 是 → 代码 gate：在【已有】gate 函数里挂 is_xxx 子判定
│   │        典型函数：vllm_adapter._build_speculative_cmd（spec num）、
│   │                  _apply_glm5_dsa_distributed_fixups（A3 双机 additional_config 去留）、
│   │                  _apply_glm5_ascend_engine_defaults（单机 TP/DP、必产字段）
│   └─ 否（该架构恒定结构）
│        → 架构级 default 命名组承载「完整字段名 + 嵌套 key 结构」
│           （务必写在 arch 级 default，保证复杂实模型名回落 default 也带齐 → 不违约；
│            精确模型名组只覆盖【数值】，不负责结构）
│           载体：config/defaults/ascend_default.json 等
└─ env 名 → DP/arch env builder 分支（last-wins，后置 export 覆盖前序）
            载体：vllm_distributed._build_ascend_dp_env_commands；值交平台 env 覆盖
```

**结构对齐特例**：当契约的 JSON 结构（扁平 vs 嵌套）≠ 现有 default 结构时，必须在模板层把结构对齐到契约，且**不破坏既有 deep-merge**（扁平叠到嵌套会致 key 重复）。这一步不能跳。

---

## Phase 4 — 改动清单 + 非破坏审查

按文件列改动清单（仿 day0 文档 §5）。提交前自检：
- [ ] 差量都挂在**已有** gate 函数 / env builder 分支上，**未开并行函数路径**
- [ ] 识别标识与邻近版本**互斥**，不污染其它模型的字段集合
- [ ] 凡"页面/平台/device 可驱动的数值"**不写死**（只兜默认，允许覆盖）
- [ ] 嵌套结构落 **arch 级 default**，精确名组只放数值
- [ ] 模板结构与 deep-merge 兼容（无重复 key）

---

## Phase 5 — 验证闭环

- **三方对账归零**：`python dry_run.py --scenario <新场景>`，与契约逐字段比，字段名/env 名集合差为空（值除外）。用复杂实模型名 + `ENGINE_VERSION=…-a3` 等真实驱动跑，验证芯片/平台解析。
- **互斥测试**：`is_xxx_model` 不误判邻近版本（如不把 5.1 判成 5.2）。
- **不回归回归**：老模型的字段集合不变（如 GLM-5.1 的 A3 双机 additional_config 仍被剥除、num 仍为 1）。

---

## Phase 6 — 归档

写 `xuqiu/day0/{model}-day0-design.md`，沿用既有 7 章模板（见 [glm52 设计](../../../xuqiu/day0/glm52-ascend-910c-day0-design.md)）：
1. 目标启动命令（甲，附出处） 2. 现状追踪（丙） 3. 设计决策表（D1…）
4. 现状差距表（G1…，每行带代码佐证链接） 5. 改动清单（带状态）
6. 仍需确认（含 Phase 0 escalate 项 / 乙未核对项） 7. 测试要点

---

## 关键代码锚点（复用入口，勿新开）

- 识别：[wings_control/utils/model_utils.py](../../../wings_control/utils/model_utils.py)（`is_glm51_model` / `is_glm52_model` 范式、`INDEXCACHE_ARCHS`、`_LLM_MODELS` 登记）
- 模板：[wings_control/config/defaults/ascend_default.json](../../../wings_control/config/defaults/ascend_default.json)（架构级 default + 精确名命名组）
- 代码 gate：[wings_control/engines/vllm_adapter.py](../../../wings_control/engines/vllm_adapter.py)
- env builder：[wings_control/engines/vllm_distributed.py](../../../wings_control/engines/vllm_distributed.py)
- 验证：[dry_run.py](../../../dry_run.py)
- 文档模板：[xuqiu/day0/](../../../xuqiu/day0/)
