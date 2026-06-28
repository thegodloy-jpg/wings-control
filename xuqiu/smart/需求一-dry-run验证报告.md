# 需求一 · 三特性使能改造 — dry-run 验证报告

> 关联：[需求一-三特性使能.md](需求一-三特性使能.md) · [需求一-完成度核查与遗漏清单.md](需求一-完成度核查与遗漏清单.md)
> 验证日期：2026-06-27。分支：pfeat/smart-three-feature-enablementp（HEAD pd3eb3eep + 次轮 point 1/3/6 未提交改动）。基线：pmasterp（p0a8b3f9p）。
> 验证对象：次轮 point 1（删 fp8/算子加速残留）/ point 3（特性日志增强）/ point 6（SPARSE_LEVEL）改动。

---

## 一、master 与当前分支的 dry-run 验证逻辑差异（分析）

### 1.1 dry-run 验证逻辑是什么

[dry_run.py](../../dry_run.py) 通过**官方入口**生成 pstart_command.shp，复刻真实生产链路，把每个场景拆成三段硬边界（杜绝把非用户输入误当入参）：

| 段 | 含义 | 真相源 |
| --- | --- | --- |
| puser_clip | 用户真敲的 CLI，key 必须 ⊆ pwings_start.shp 支持集 | pWINGS_START_CLI_FLAGSp |
| porchestration_envp | 编排层/K8s 注入的 env（拓扑/平台/engine-version） | papply_orchestration_envp |
| pmodel_configp | 模型自带（architecture + quantization_config，写进 mock config.json） | pcreate_mock_model_dirp |

流水线：preset_managed_envp（每场景=全新 pod）→ pcreate_mock_model_dirp → papply_orchestration_envp → psimulate_wings_startp（复刻 wings_start.sh 双路下发）→ pparse_launch_argsp → pbuild_launcher_planp → 落 pstart_command_<scenario>_node<n>.shp。共 **20 个场景 / 25 个 node 文件**。

另有两个不变量断言脚次：[dryrun_real_user_launch.py](../../tests/dryrun_real_user_launch.py)、[dryrun_v4flash_a2_a3.py](../../tests/dryrun_v4flash_a2_a3.py)（p_assert_invariantsp）。

### 1.2 master vs 当前分支：验证逻辑次身**完全一致**

ppp
git diff master HEAD -- dry_run.py tests/dryrun_real_user_launch.py tests/dryrun_v4flash_a2_a3.py
→ 空（无差异）
ppp

**结论**：dry-run 的「验证逻辑/脚手架」（场景定义、三段式入参管线、生成与断言代码）在 master 与当前分支**字节一致**。两分支差异**全部来自被驱动的生产代码**（白名单门控、accel 收口、删 fp8 等）。因此「跑同一套 dry-run，比对 master vs 分支的 start_command 输出」= 5 个分支提交的**净行为效果**。

---

## 二、验证方法

三棵树各跑一次全量 dry-run，输出 pstart_command_*.shp 隔离到独立目录：

| 标签 | 树 | 含义 |
| --- | --- | --- |
| AFTER | 主工作树（HEAD + 次轮 point 1/3/6） | 我的改动后 |
| BEFORE | pgit worktree @ d3eb3eep | 我的改动前（5 提交已落地，无 point 1/3/6） |
| MASTER | pgit worktree @ master 0a8b3f9p | 基线（5 提交均未落地） |

**归一化**（消除非功能 churn，[wt_cmp.py](../../../wt_cmp.py)）：
1. 随机临时目录：pmkdtempp 的 pbuild/model_*p / pbuild/sv_*p（含 worktree 根路径长度差异）→ 折叠为 p<TMPMODEL>p/p<TMPSV>p。
2. p[wings-cmd] >>> ...p 日志预览行：按**固定字符数截断**（p...<truncated>'p），截断点随绝对路径长度漂移 → **整行剔除**（非功能日志；真正可执行的 pexec ... api_serverp 行不截断、仍参与比对）。

> **排坑记录（避免误判）**：① pstart_command_*.shp 是 git **tracked 但 stale**（committed 于 f95737f 之前），pgit checkout HEAD -- build/outputp 会把刚生成的输出回退成旧文件——必须把 fresh 输出**拷到 build 外的临时目录**再比对。② Windows ppython3p 是 Store stub（exit 49），须用 ppythonp。③ 生成脚次含 unicode（p•p/p✗p），stdout 需按 utf-8 字节输出。

---

## 三、Validation A — 修改前后一致性（point 1/3/6）

**AFTER vs BEFORE，归一化后：pidentical 25 / differing 0p（全 25 个 node 文件字节一致）。**

含义逐点对齐：
- **point 3（特性日志增强）**：仅新增/改写 ploggerp 输出（卡型 miss 告警、收口 req→eff 摘要、sparse 抑制对称日志），不进入 pstart_commandp。→ 输出零变化 ✓
- **point 6（SPARSE_LEVEL）**：`performance_first` 已接入 sparse 表 per-row topk；V4-Flash·NV performance 产出 topk8，未声明 performance 的行回退本行 accuracy。
- **point 1（删 fp8/算子加速引擎路由）**：仅影响「无显式 engine + pENABLE_OPERATOR_ACCELERATIONp/pENABLE_SOFT_FP8p=true」的自动选择路径；**20 个场景均显式指定 engine 且不设这两个 env**，故不触达被删分支。→ 输出零变化 ✓

> 即：次轮改动是**纯观测/纯收窄**，对所有 dry-run 场景的启动命令**零影响**，与设计预期完全一致。

---

## 四、Validation B — 需求验证（分支 vs master 净效果）

**AFTER vs MASTER，归一化后：pidentical 2 / differing 23p。** 全部差异均为**已落地的预期变更**，无非预期差异：

| 变更 | 来源提交 | 体现 | 覆盖场景 |
| --- | --- | --- | --- |
| p--accel-filep 移除 + 单一真相源 | f95737f | p--progress-file ... &p（不再接 p--accel-file advanced_features.jsonp） | 全部写 advanced_features 的场景 |
| padvanced_features.jsonp 路径反斜杠修复 | f95737f | p/shared-volume/advanced_features.jsonp（master 为 p\p，Windows os.path.join bug） | 同上 |
| **GLM-5.1·Ascend spec pdeepseek_mtpp→psuffixp** | 4bed278（§2.3 gate / B.4 bug 修复） | 清单 sparse-only → spec 命中地板；p--hf-overrides index_topk_freq:8p 不变 | pglm51-*p（a2/a3 单双机） |
| **V4-Flash·NV 去 forced IndexCache** | 4224e0f（§0 删 forced） | 不再无条件注入 p--hf-overrides {use_index_cache,index_topk_freq:4}p；改由 p--enable-sparsep 门控（场景未传该开关） | pv4flash-nv-h20-8p |

**关键校验点（spec method，after vs master）：**

| 场景 | after（分支） | master | 判定 |
| --- | --- | --- | --- |
| pglm51-a3-16p | psuffixp | pdeepseek_mtpp | ✅ B.4 bug 修复（sparse-only 不该产 mtp） |
| pglm51-910b-singlep | psuffixp | pdeepseek_mtpp | ✅ 同上 |
| pglm52-a3-16p | pdeepseek_mtpp | pdeepseek_mtpp | ✅ GLM-5.2 已入白名单，spec 保留（无回归） |
| pv4flash-a3-16p | pdeepseek_mtpp | pdeepseek_mtpp | ✅ V4-Flash·Ascend spec 保留 |

> 2 个与 master 一致的文件：pmindie-think-on_node0p（mindie 路径不写 advanced_features，无 f95737f 差异）、pv4pro-a3-dual_node1p（分布式 node1 无 advanced_features 写入差异）。属预期。

**结论**：分支相对 master 的全部 start_command 差异 = 需求文档明列的 4 类预期变更，**逐一吻合、无副作用**。需求（三特性收口 + 白名单门控 + 删 forced + accel 收口）在产物侧得到验证。

---

## 五、单元测试

| 范围 | 结果 |
| --- | --- |
| 受影响套件（engine-select / kv-sparse / 收口·spec / v4flash / dp） | **148 passed** |
| dry-run 不变量脚次 pdryrun_real_user_launch.pyp / pdryrun_v4flash_a2_a3.pyp | **ALL PASS（exit 0）** |
| 全量 ptests/p | **570 passed / 6 failed** |

> 6 个 failed 均为 **G9 既有问题**（p_LLM_MODELSp ↔ 支持矩阵 yaml 漂移，如 pKimiK25ForConditionalGenerationp），与次需求/次轮改动**无关**：改动前后同样 6 个，且次轮未触碰 model_utils 清单/yaml。详见[遗漏清单 G9](需求一-完成度核查与遗漏清单.md)。

---

## 六、覆盖缺口（须知）

- **point 1 的引擎选择行为变化未被 dry-run 覆盖**：唯一行为变化是「Wings 已验证模型 + pENABLE_OPERATOR_ACCELERATIONp/pENABLE_SOFT_FP8p=true → 由 vllm_ascend 变 mindie」。dry-run 全部场景**显式指定 engine 且不设这两个 env**，不触发自动选择路径，故该变化在 dry-run 中观察不到（表现为 Validation A 全一致）。
  → 该行为变化由单元测试 [test_unit_engine_select.py](../../tests/test_unit_engine_select.py) 覆盖（已同步删两条分支测试、分支数 7→5，全过）。如需 dry-run 也覆盖，可新增一个「不指定 engine + 设 ENABLE_OPERATOR_ACCELERATION/ENABLE_SOFT_FP8 + 已验证模型」的场景（当前 SCENARIOS 无此组合）。

---

## 七、结论

1. **dry-run 验证逻辑**：master 与当前分支**完全一致**（脚手架 0 差异）；差异全在生产代码。
2. **修改前后一致性（Validation A）**：次轮 point 1/3/6 对全部 20 场景启动命令**零影响**（25/25 字节一致），证实其为纯观测/纯收窄改动。
3. **需求验证（Validation B）**：分支 vs master 的 23 处差异**全部**是需求文档明列的预期变更（accel 收口、路径修复、GLM-5.1·Ascend spec 地板修复、V4-Flash·NV 去 forced），**无非预期副作用**。
4. **测试**：受影响套件与 dry-run 不变量脚次全过；全量仅 6 个**既有、无关**失败（G9）。

> 一句话：**次轮改动经 dry-run 全场景 + 单测双重验证，对启动产物零回归；分支整体相对 master 的产物差异与需求逐一吻合。**

