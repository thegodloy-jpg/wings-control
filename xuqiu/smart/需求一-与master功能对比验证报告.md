# 需求一 · 与 master 功能对比验证报告（防白名单缺失 / 功能丢失）

> 关联：[需求一-三特性使能.md](需求一-三特性使能.md) · [需求一-与master功能对比测试方案.md](需求一-与master功能对比测试方案.md) · [需求一-需求点dry-run测试方案与验证报告.md](需求一-需求点dry-run测试方案与验证报告.md)

## 一、结论

截至 2026-07-02，本分支相对 `master` 的功能对比结论：

- `master`：`0a8b3f9`，为白名单生成前基线。
- 当前分支：`08941b1` + 本地修复工作树。
- 分叉点：`merge-base master HEAD = 0a8b3f9`。
- 提交关系：`HEAD..master = 0`，`master..HEAD = 30`。
- 对比结果：`43/43` 场景无回归。
- 疑似回归项：`0`。
- 已归类预期变更：`290`。

结论：未发现白名单缺失导致的功能丢失。master 与当前分支的差异均可归入需求一的预期变更或白名单收口；用户显式关闭特性时，当前分支能关闭，不再被模型默认推荐或 W8A8 注入逻辑强制打开。

## 二、本轮加强点

本轮在原 master 对比基础上补强了三类风险：

1. 新增 `C8:v4flash-a3-spec-off`：DeepSeek-V4-Flash Ascend A3 下发 `ENABLE_SPECULATIVE_DECODE=false`，验证当前分支不再输出 `--speculative-config`。
2. 新增 `C9:glm47-w8a8-spec-off`：GLM-4.7 W8A8 下发 `ENABLE_SPECULATIVE_DECODE=false`，验证当前分支不再输出 `--speculative-config`。
3. KV offload 对比同时兼容 master 旧键和当前新键：`LMCACHE_OFFLOAD/LMCACHE_*` 与 `ENABLE_KV_OFFLOAD/KV_MEM_OFFLOAD_SIZE/AVAILABLE_POD_MEM_SIZE` 同时下发，避免把环境变量改名误判成功能缺失。

## 三、白名单完整性核查

### 1. Speculative Decode

当前 spec 白名单共 10 行，覆盖：

| 引擎 | 模型/平台 | 状态 |
| --- | --- | --- |
| vllm | Qwen3.5-397B、GLM-4.7、MiniMax-M2.7、DeepSeek-V4-Flash | 保留 |
| vllm_ascend | GLM-4.7 910B/910C、MiniMax-M2.5 910B/910C、DeepSeek-V3.2 910C、Qwen3.6 910C、DeepSeek-V4-Flash 910B/910C、GLM-5.2 910B/910C | 保留 |

核查结论：

- GLM-5.1 Ascend 不在 spec 白名单，当前分支降为 suffix 地板能力，属于预期收口。
- V4-Flash Ascend A3 和 GLM-4.7 W8A8 显式关闭 spec 时，当前分支不再强制打开，符合“用户关闭优先”诉求。
- 白名单内模型在开启时仍可生成对应投机配置。

### 2. Sparse Attention

当前 sparse 白名单共 7 行，覆盖：

| 引擎 | 模型/平台 | 档位 |
| --- | --- | --- |
| vllm | Qwen3.5-397B、GLM-4.7、GLM-5.1、MiniMax-M2.7 | accuracy topk=4 |
| vllm | DeepSeek-V4-Flash | accuracy topk=4；performance topk=8 |
| vllm_ascend | DeepSeek-V4-Flash 910B/910C | 当前 sparse 白名单命中但产出口为 noop；630 后切 use_index_cache topk4/8 |
| vllm_ascend | GLM-5.1 910B/910C | accuracy/performance topk=8 |

核查结论：

- GLM-4.7 Ascend、GLM-5.2 Ascend 不在 sparse 白名单，关闭为预期收口。
- 白名单内 sparse 场景仍能生成 `--kv-cache-dtype fp8`、`--calculate-kv-scales` 或 IndexCache 相关配置。
- DeepSeek-V4-Flash·NV 的 accuracy/performance 档位均已覆盖；DeepSeek-V4-Flash·Ascend 当前按需求文档记录为 sparse noop，白名单先保留，待 630 引擎产出口切 use_index_cache topk4/8。

### 3. KV Offload

当前 offload 白名单共 7 行，覆盖：

| 引擎 | 模型/平台 | 状态 |
| --- | --- | --- |
| vllm | GLM-4.7、MiniMax-M2.7、DeepSeek-V4-Flash | 保留 |
| vllm_ascend | GLM-4.7 910B/910C、MiniMax-M2.5 910B/910C、DeepSeek-V3.2 910C、DeepSeek-V4-Flash 910B/910C | 保留 |

核查结论：

- GLM-5.2 Ascend offload 被收口关闭，属于白名单预期行为。
- GLM-5.1 NV offload 仍按现有逻辑硬关闭，未被误加入白名单。
- 当前新 env `ENABLE_KV_OFFLOAD=true` 路径由需求 dry-run 覆盖；master 对比脚本同时下发新旧 env，证明不是改名导致的假阴性。

## 四、master 差异解释

以下差异已核查为预期，不属于功能丢失：

| 场景 | master 行为 | 当前分支行为 | 结论 |
| --- | --- | --- | --- |
| GLM-5.1 Ascend | 生成 `deepseek_mtp` | 降为 `suffix` | 不在 spec 白名单，预期收口 |
| V4-Flash A3 默认/显式关 spec | master 仍强制 `deepseek_mtp` | 当前可关闭到无 `--speculative-config` | 满足用户关闭优先 |
| GLM-4.7 W8A8 显式关 spec | master 仍强制 `mtp` | 当前可关闭到无 `--speculative-config` | 满足用户关闭优先 |
| GLM-5.2 Ascend offload | master 生成 LMCache connector | 当前关闭 offload | 不在 offload 白名单，预期收口 |
| PD role | master 可混入 LMCache/MultiConnector | 当前三特性一票否决，仅保留 PD connector | PD 隔离预期行为 |
| V4-Flash NV sparse 未开启 | master 强制 IndexCache | 当前不强制 | 用户开关优先，预期收口 |

## 五、验证命令与产物

执行命令：

```powershell
# 如 ..\wt-master 不存在，先创建 master 基线 worktree
git worktree add ..\wt-master master
python .\tests\cmp_master_branch.py
python .\tests\dryrun_requirement_coverage.py
```

关键产物：

- `tests/cmp_master_branch.py`：master 对比脚本，43 场景。
- `tests/cmp_master_branch_output.txt`：对比输出，`43/43` 无回归。
- `tests/dryrun_requirement_coverage.py`：需求点 dry-run 覆盖脚本，包含显式关 spec、KV offload 新 env、sparse 档位等要求。
- `tests/dryrun_requirement_coverage_output.txt`：需求 dry-run 输出。

## 六、最终判断

当前实现满足需求核心诉求：

- 页面/编排侧可以全量下发三特性变量。
- 后端通过白名单决定特性是否真正生效。
- 用户显式关闭部分特性时，关闭优先级高于模型默认推荐、W8A8 注入和历史 forced 逻辑。
- 与 master 对比未发现白名单遗漏或既有功能丢失。
