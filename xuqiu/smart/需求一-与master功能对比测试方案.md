# 需求一 · 与 master 功能对比测试方案（防功能丢失回归）

> 关联：[需求一-三特性使能.md](需求一-三特性使能.md) · [需求一-需求点dry-run测试方案与验证报告.md](需求一-需求点dry-run测试方案与验证报告.md)
> 目标：证明 `feat/smart-three-feature-enablement` 当前工作树相对 `master`（`0a8b3f9`）**只产生需求文档列明的预期变更，未丢失/未破坏任何既有功能**。
> 本文件＝**方案 + 用例矩阵设计**；执行结果见同目录《需求一-与master功能对比验证报告.md》。

---

## 一、对比目标与判定标准

**判定**：对每个场景，在 master 与 branch 上各生成 `start_command.sh`（+ `advanced_features.json`），归一化后逐行 diff，把每条差异归类：

- **预期变更（Intended，白名单 D1–D5）** —— 需求文档列明的改动；
- **疑似回归（Regression）** —— 不属 D1–D5 的任何差异。

> **通过条件：全场景「疑似回归」计数 = 0。** 即 master↔branch 的差异 100% 落在 D1–D5 内。

---

## 二、预期变更白名单（D1–D5，精确）

> 来源：分支 5 个功能提交。任何 diff 行必须能归入下列之一，否则记为回归。

| 编号 | 预期变更 | 出处提交 | 命令侧特征（master → branch） |
| --- | --- | --- | --- |
| **D1** | 移除 `--accel-file`（accel_features.jsonl 链路下线） | f95737f | log_analyzer 参数 `… --accel-file /shared-volume/advanced_features.json`（master 有）→ branch 无 |
| **D2** | advanced_features.json 路径反斜杠修复 | f95737f | `'/shared-volume\advanced_features.json'`（master，Windows os.path.join）→ `'/shared-volume/advanced_features.json'`（branch），出现在 feature-disable 片段与 cat heredoc |
| **D3** | 投机白名单门控（无 forced，地板 suffix） | 4bed278 / 4224e0f | `--speculative-config` 的 `method`：白名单外/sparse-only 由 `deepseek_mtp/mtp`→`suffix`（典型 GLM-5.1·Ascend）；新白名单模型保持 mtp |
| **D4** | V4-Flash·NV forced IndexCache 去除 | 4224e0f | 未开 sparse 时 master 强制 `--hf-overrides '{"use_index_cache":true,...}'`→ branch 无 |
| **D5** | 删 Soft FP8/FP4 自动量化全链路 | 73b177b | 裸布局（无 quantization_config）模型：master 自动注入 `--quantization ascend` / `kv_cache_dtype=fp8` 等 → branch 无（改由用户显式或权重自带声明） |
| **D6** | KV 卸载 auto 容量反向预算（C4，新增能力） | 4bed278 | auto 模式 branch 计算并写回 `KV_MEM_OFFLOAD_SIZE=<均卡>`/`cpu_swap_space_gb`，且 vLLM 新版本不再生成 `--swap-space 0`；master 无此自动计算 |
| **D7** | 三特性白名单收窄（未命中→收口关，及其下游脚手架） | 4bed278/4224e0f | 模型不在某特性白名单时：master 产出（`--kv-transfer-config`/LMCache env/install 补丁/`kv_cache_dtype=fp8`+`--calculate-kv-scales`）→ branch 收口关；崩溃处理由「advanced-feature-fallback」转「crash-retry」模板（**两套模板代码均未改**，仅因特性激活态变化而切换）。典型：GLM-5.2·Ascend offload/sparse 收窄、PD 一票否决（MultiConnector{Nixl+LMCache}→Nixl-only） |
| **D8** | 删算子加速/昆仑 ATB（§6-⑤，已裁定可删） | afb9a76 | `ENABLE_OPERATOR_ACCELERATION=true` 时 master 导出 `USE_KUNLUN_ATB=1` → branch 不导出（路由旁路 + 死代码全删） |

> 另：变体名差异属 D3 同源（DeepSeek-V3.2→`mtp`、GLM-5.x→`deepseek_mtp`）；offload×spec 降级豁免亦属 D3/D7 交互（offload 被白名单收窄→spec 不再被降级，反而保 mtp）。
> 归一化时「崩溃处理脚手架」尾段（fallback↔retry 模板）整体剥离——其切换是 D7 的下游后果，机制代码两侧一致（`git log -S` 证实「Advanced Feature Fallback」「Engine Crash Retry」两串在 master/branch 都存在、未被 7 个提交改动）。

---

## 三、对比方法（双 worktree + 自包含生成器 + 归一化 + 分类）

```
master worktree (0a8b3f9) ─┐                          ┌─ 归一化(去临时路径churn + [wings-cmd]截断预览)
                           ├─ 同一场景 → 各自生产代码 ─┤
branch 工作树 ─────────────┘   生成 start_command.sh   └─ 逐行 diff → 按 D1–D8 分类 → 回归计数
```

1. **双 worktree**：`git worktree add ../wt-master master`；branch = 当前工作树。两侧 `dry_run.py` 字节一致（已验证 `git diff` 为空），差异只来自各自的**生产代码**。
2. **生成器（自包含，跨版本可跑）**：`import dry_run`（各 worktree 自带），复刻真实下发三段式 + 三特性 **env 下发**（见对应方案 §〇），输出 `start_command.sh` + `advanced_features.json`。用 subprocess 在各 worktree 下各跑一遍（进程隔离，避免双版本模块缓存冲突）。KV offload 对比同时下发 master 旧键 `LMCACHE_OFFLOAD/LMCACHE_*` 与当前新键 `ENABLE_KV_OFFLOAD/KV_MEM_OFFLOAD_SIZE/AVAILABLE_POD_MEM_SIZE`，避免把环境变量改名误判成功能丢失。
3. **归一化**：① 折叠随机临时目录 `build/model_*`、`build/sv_*`；② 丢弃 `echo '[wings-cmd] >>> …'` 按字符截断的预览行（截断点随绝对路径长度漂移，非功能差异）。
4. **分类**：对每条差异行匹配 D1–D8 正则；不匹配即记入「疑似回归」清单（含场景名 + 原始行），供人工复核。

---

## 四、对比测试用例矩阵（详细 · 完备）

> 三特性开关一律经 **env**（orchestration_env：`ENABLE_SPECULATIVE_DECODE/ENABLE_SPARSE/ENABLE_KV_OFFLOAD`），`user_cli` 不含特性 CLI 标志。

### Group A · 主 recipe 全引擎/平台/拓扑（复用 dry_run.py 20 场景，作回归底座）

| 覆盖维度 | 场景（dry_run.py 内置） |
| --- | --- |
| GLM-5.1 (GlmMoeDsa) | glm51-910b-dual / glm51-910b-single / glm51-a3-dual / glm51-a3-16 |
| GLM-5.2 (复杂名/MLAPO/num=3) | glm52-910c-single / glm52-a3-16 / glm52-a3-dual / glm52-910b-single / glm52-910b-dual |
| DeepSeek-V4-Flash/Pro | v4flash-a3-16 / v4flash-a2-8 / v4flash-nv-h20-8 / v4pro-a3-dual |
| Qwen3.5 / Qwen3.6 | qwen35-397b-a17b / qwen36-35b-a3b / qwen36-27b |
| 其它引擎/架构 | minimax-m3-nv-8 / kimi-k27-ascend-16 / sglang-think-on / mindie-think-on |

> 覆盖：vllm / vllm_ascend / sglang / mindie × a2/a3/NV × 单机/双机 × FC/think/spec 默认开关。验证主流程（TP/DP/EP/max_model_len/parser/chat-template/env 脚本/PD 拓扑）整体保形。

### Group B · 三特性组合矩阵（同模型 env 全组合，验证特性开关不波及主命令）

代表模型 ×（7 组 env 开关组合）：

| 模型/平台 | none | spec | sparse | offload-auto | offload-custom | spec+sparse+offload | +SPARSE_LEVEL=perf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glm-4.7 · NV（白名单 spec,sparse,offload 全） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GLM-5.2 · Ascend a3（白名单 spec） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

> 每格与 master 同输入对比。预期：未触三特性的字段（模型 recipe/TP/DP/parser…）逐字保形；差异仅落 D1–D5。

### Group C · 回归风险专项（分支改动直接影响面）★

| 编号 | 场景 | 验证点 | 预期 |
| --- | --- | --- | --- |
| C1 裸布局-NV | Qwen3·NV，model_config **无 quantization_config** | D5 软 fp8 删除 | master 自动 quant → branch 无；其余保形 |
| C2 裸布局-Ascend | DeepSeek-V3.1·Ascend，**无 quantization_config** | D5 + 官方 W8A8 路径删除 | 同上 |
| C3 embedding-Ascend | Qwen3-Embedding·vllm_ascend | `_set_task` 删 `use_kunlun_atb` 无副作用 | 除 D1/D2 外字节一致（含 enforce_eager 保留） |
| C4 rerank-Ascend | XLMRoberta·vllm_ascend | 同 C3 | 同上 |
| C5 算子加速旁路 | Qwen3.5·vllm_ascend + `ENABLE_OPERATOR_ACCELERATION=true` | 引擎路由旁路删除 | 显式 engine 下：除 D1/D2 字节一致；不导出 USE_KUNLUN_ATB |
| C6 软fp8旁路 | 同 C5 但 `ENABLE_SOFT_FP8=true` | 同上 | 同上 |
| C7 PD 角色 | glm-4.7·NV，`PD_ROLE=P` + 三特性 env 全开 | PD 一票否决 | 三特性收口关；PD connector 保形 |
| C8 V4-Flash 显式关 spec | DeepSeek-V4-Flash·Ascend A3，`ENABLE_SPECULATIVE_DECODE=false` | 用户关闭优先于模型默认推荐 | branch 不生成 `--speculative-config` |
| C9 GLM-4.7 W8A8 显式关 spec | GLM-4.7·NV W8A8，`ENABLE_SPECULATIVE_DECODE=false` | 用户关闭优先于 W8A8 注入逻辑 | branch 不生成 `--speculative-config` |

### Group D · 平台/拓扑边界（保形回归）

| 编号 | 场景 | 验证点 |
| --- | --- | --- |
| D-a2a3 | GLM-5.2 同模型 a2 vs a3 | 平台 env-block 分叉（TP/DP/additional_config）保形 |
| D-dual | GLM-5.1 双机 node0/node1 | dp_deployment 拓扑/rank/rpc-port 保形 |

---

## 五、覆盖与缺口

- **覆盖**：4 引擎 × 12+ 架构 × 3 平台 × 单/双/PD 拓扑 × 7 特性组合 + 9 回归专项 + 2 边界 = **43 个对比场景**。
- **缺口**：引擎「自动选择」改道（不给 engine → Ascend mindie/vllm_ascend）dry-run 无法触发，由单测 `test_unit_engine_select.py` 覆盖（C5/C6 以「显式 engine + 旁路 env 无效」做等价回归）。

---

## 六、执行与复现

```bash
# 1) 建 master worktree（如 ../wt-master 不存在）
git worktree add ../wt-master master
# 2) 运行对比（双 worktree 各生成 + 归一化 + 分类）
python tests/cmp_master_branch.py        # 输出回归清单；回归=0 即无功能丢失
```
> 结果与逐场景差异分类见《需求一-与master功能对比验证报告.md》。
