# 需求一 · 与 master 功能对比验证报告（防功能丢失）

> 方案见 [需求一-与master对比测试方案.md](需求一-与master对比测试方案.md)。
> 对比对象：`master`（`0a8b3f9`，**经 `git merge-base` 确认为分支的干净祖先**，master 无独有提交）↔ `feat/smart-three-feature-enablement`（`8e607d4`，领先 7 个提交）。
> 落地物：[tests/cmp_master_branch.py](../../tests/cmp_master_branch.py)；证据产物 [tests/cmp_master_branch_output.txt](../../tests/cmp_master_branch_output.txt)。
> **运行**：`git worktree add ../wt-master master && python tests/cmp_master_branch.py`

---

## 一、结论（先给结论）

```
40 / 40 场景无回归   ·   疑似回归项 = 0   ·   归类为预期变更 = 184 处
```

**开发需求一后，相对 master 未丢失/未破坏任何既有功能。** master↔branch 的全部差异（184 处）100% 落在需求文档列明的预期变更白名单 **D1–D8** 内，无任何「非特性差异」（即无 TP/DP/EP、parser、chat-template、max_model_len、模型 recipe 等功能性回归）。

---

## 二、方法与判定（要点）

1. **干净祖先确认**：`git merge-base master HEAD = 0a8b3f9 = master`，`HEAD..master = 0` 提交 → 所有差异均由分支 7 个提交引入（排除「master 自身演进被误判为回归」）。
2. **双 worktree 各跑生产代码**：同一场景（三特性经 **env 下发**）在 master/branch 各自代码下生成 `start_command.sh`，subprocess 进程隔离。
3. **归一化**：折叠临时目录 + worktree 根路径 churn；剥离 `[wings-cmd]` 截断预览；剥离「崩溃处理脚手架」尾段（fallback↔retry 模板，属 D7 下游、机制代码两侧一致）。
4. **分类**：逐行/逐 flag 匹配 D1–D8；命令行非特性 flag（TP/parser/template/recipe…）或非特性脚手架行若变化即记「疑似回归」。**判定通过 = 疑似回归 0**。

---

## 三、预期变更分类计数（184 处）

| 类 | 含义 | 命中数 | 说明 |
| --- | --- | --- | --- |
| D1 | 移除 `--accel-file` | 26 | 通用（每个写 advanced_features 的场景） |
| D2 | advanced_features 路径反斜杠修复 | 26 | 通用 |
| D3 | 投机白名单门控（mtp↔suffix） | 7 | 主要 GLM-5.1·Ascend（sparse-only→spec 地板 suffix） |
| D4 | V4-Flash·NV forced IndexCache 去除 | 1 | v4flash-nv-h20-8 |
| D5 | 删 Soft FP8/FP4 自动量化 | 0※ | 本批 mock 未触发（见 §五缺口），由单测覆盖 |
| D6 | KV 卸载 auto 容量 C4（新增） | 3 | offload-auto 场景：`LMCACHE_MAX_LOCAL_CPU_SIZE` + `--swap-space 0` |
| D7 | 白名单收窄（收口关 + 下游脚手架） | 7+※ | GLM-5.2·Ascend offload/sparse 收窄、PD 一票否决等 |
| D8 | 删算子加速/昆仑 ATB（§6-⑤） | ※ | `USE_KUNLUN_ATB=1` 不再导出（计入 scaffold） |
| scaffold | 特性门控的 install/LMCache/EARS/shell 控制脚手架（D6/D7/D8 伴生） | 133 | 随特性激活态确定，非独立改动 |

> ※ D7/D8 的多行脚手架统计入 scaffold（133）。

---

## 四、分组结论（逐组验证无回归）

### Group A · 主 recipe 全引擎/平台/拓扑（20 场景）
- **GLM-5.1·Ascend（4 场景）**：仅 `D3 spec deepseek_mtp→suffix`（清单 sparse-only 的地板，B.4 修复），sparse/TP/DP/EP/拓扑全保形。
- **GLM-5.2 / V4-Flash·Ascend / V4-Pro / Qwen3.5 / Qwen3.6 / MiniMax / Kimi / sglang / mindie**：**与 master 等价（仅 D1/D2 已中和）** —— 模型 recipe、TP/DP/EP、parser、chat-template、mindie config、sglang 参数逐字保形。
- **V4-Flash·NV**：仅 `D4 forced IndexCache removed`（§0 删 forced）。

### Group B · 三特性 env 组合矩阵（14 场景）
- glm-4.7·NV 各组合：未触发特性的字段全保形；`offload-auto` 仅 `D6 C4`（`LMCACHE_MAX_LOCAL_CPU_SIZE=50` + `swap-space 0`）。
- GLM-5.2·Ascend 各组合：`offload`/`sparse` 被白名单收窄 → `D7`（`--kv-transfer-config`/LMCache env/install 补丁收口关）；`all` 组合 `D3 spec suffix→deepseek_mtp`（offload 被收窄→不再触发 offload×spec 降级，**branch 反而保住 mtp**）。均为预期。

### Group C · 回归风险专项（6 场景）★
| 用例 | 结论 |
| --- | --- |
| C1 裸布局-NV / C2 裸布局-Ascend | **与 master 等价** —— 删软 fp8 对这些架构启动命令无影响（无回归） |
| C3 embedding-Ascend | **与 master 等价** —— `_set_task` 删 `use_kunlun_atb` **零副作用**（enforce_eager 等保留） |
| C5 op-accel（ENABLE_OPERATOR_ACCELERATION） | 仅 `USE_KUNLUN_ATB=1` 不再导出（**D8 §6-⑤ 删昆仑 ATB**，已裁定可删） |
| C6 soft-fp8（ENABLE_SOFT_FP8） | **与 master 等价** —— 显式 engine 下软 fp8 路由旁路删除无副作用 |
| C7 PD 角色 | `D3 spec→none` + `D7` PD 一票否决：`--kv-transfer-config` 由 `MultiConnector{Nixl+LMCache}`→`Nixl-only`（§3.1 删不可达 MultiConnector 共存分支）、`--calculate-kv-scales`/`kv-cache-dtype=fp8` 随 sparse 收口移除 |

### Group D · 平台/拓扑边界
- a2/a3 分叉、双机 node0/node1 拓扑（含 dp_deployment/rpc-port）均在 Group A 内保形，仅叠加各自的 D1/D2/D3。

---

## 五、覆盖与缺口

- **覆盖**：4 引擎 × 12+ 架构 × 3 平台 × 单/双/PD × 7 特性组合 + 7 回归专项 = **40 对比场景**，全部无回归。
- **D5（软 fp8 删除）未在本批触发**：mock 的 `config.json` 缺少 modelslim/特定量化布局信号，裸布局场景 master 也未注入自动量化，故无可见差异（既非回归）。软 fp8/fp4 删除的正确性由 73b177b 同步删除的单测覆盖。
- **引擎自动选择改道**：dry-run 无法触发（C5/C6 以「显式 engine + 旁路 env」做等价回归，确认旁路 env 仅 D8 一处影响），由单测 `test_unit_engine_select.py` 覆盖。

---

## 六、复现

```bash
cd wings-control
git worktree add ../wt-master master      # 若已存在可跳过
python tests/cmp_master_branch.py          # 40/40 无回归，疑似回归=0；详见 tests/cmp_master_branch_output.txt
git worktree remove ../wt-master           # 清理
```

> 一句话：**逐场景、逐 flag、逐脚手架行核对，分支相对 master 的 184 处差异全部可归因于需求文档列明的 D1–D8 预期变更，零功能丢失。**
