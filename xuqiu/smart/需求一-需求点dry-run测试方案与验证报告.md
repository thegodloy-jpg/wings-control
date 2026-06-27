# 需求一 · 三特性使能改造 — 八需求点 dry-run 测试方案与验证报告

> 关联：[需求一-完成度核查与遗漏清单.md](需求一-完成度核查与遗漏清单.md) · [需求一-dry-run验证报告.md](需求一-dry-run验证报告.md) · [需求一-三特性使能.md](需求一-三特性使能.md)
> 日期：2026-06-27。分支：`feat/smart-three-feature-enablement`（`afb9a76`）。
> 落地物：测试驱动器 [tests/_dryrun_req_harness.py](../../tests/_dryrun_req_harness.py) + 用例集 [tests/dryrun_requirement_coverage.py](../../tests/dryrun_requirement_coverage.py)（22 检查），产物 [tests/dryrun_requirement_coverage_output.txt](../../tests/dryrun_requirement_coverage_output.txt)。
> **运行**：`python tests/dryrun_requirement_coverage.py`（exit 0 = 全过）。

---

## 一、测试方案（必须模拟用户真实下发）

### 1.1 真实下发口径（三段式硬边界）

完全复用 [dry_run.py](../../dry_run.py) 的真实链路，**不 mock 模型识别**，每个用例拆三段，杜绝把「非用户输入」误当入参：

| 段 | 含义 | 约束 |
| --- | --- | --- |
| `user_cli` | 用户真敲的 CLI | key **必须 ⊆ `wings_start.sh` 支持集**（`simulate_wings_start` 会校验，越界即报错） |
| `orchestration_env` | 编排层/K8s/MaaS 注入的 env | 拓扑（NNODES/IP）、平台（WINGS_ASCEND_PLATFORM）、`ENGINE_VERSION`、`LMCACHE_*`、页面开关 env（`SPARSE_LEVEL`/`PD_ROLE`/`LMCACHE_OFFLOAD`…） |
| `model_config` | 模型自带 config.json | `architecture` + `quantization_config` |

链路：`reset_managed_env`（每例=全新 pod）→ `create_mock_model_dir` → `apply_orchestration_env` → `simulate_wings_start`（复刻 wings_start.sh 双路下发）→ `parse_launch_args` → `build_launcher_plan` → `start_command.sh`。

### 1.2 三类可观测产物 = 断言对象

| 产物 | 来源 | 断言到 |
| --- | --- | --- |
| `start_command.sh` 可执行命令行 | `plan.command`（取真实 exec 行，非 echo 预览） | `--speculative-config`/`--hf-overrides`/`--kv-cache-dtype`/`LMCACHE_*`/`--swap-space`/`cpu_swap_space_gb`/`install.py --features` |
| `advanced_features.json` | 生成期写到 `settings.SHARED_VOLUME_PATH`（真实文件，非 start_command 里的崩溃回退 heredoc） | `features.{spec/sparse/offload}` bool + `variants.{...}` 字符串 |
| 生产代码日志 | 捕获 `core.config_loader` / `engines.vllm_adapter` 的 INFO/WARNING | 收口摘要 / 卡型 miss 告警 / 抑制日志 / SPARSE_LEVEL 回落告警 |

> 关键点：`start_command.sh` 内嵌的 `cat > advanced_features.json <<FEATURES_EOF` 是**崩溃回退默认块（全 false、无 variants）**；特性真相源是 `settings.SHARED_VOLUME_PATH/advanced_features.json`（含 variants），由 `_write_advanced_features_json` 在生成期落盘——本方案读后者。

---

## 二、八需求点 · 用例矩阵与断言

> 共 **22 个断言点 / 8 需求点**，全部基于真实下发三段式入参。

### P1 · 删 fp4/fp8 + 引擎路由删除
| 用例 | 真实下发关键 | 断言 | 结果 |
| --- | --- | --- | --- |
| 旁路已删·env 无效 | `engine=vllm_ascend` + 对照组设/不设 `ENABLE_OPERATOR_ACCELERATION=true`+`ENABLE_SOFT_FP8=true` | 两组 engine 一致 + `start_command` 归一化字节一致 | ✅ |
| 死代码已删 | 同上（设算子加速 env） | 命令**不导出** `USE_KUNLUN_ATB` | ✅ |

> 覆盖说明：引擎「自动选择」改道（已验证模型+开关→mindie）**无法经 dry-run 真实下发触发**——dry_run 在未给 engine 时按 nvidia 推断设备，不进 Ascend 自动选择路径。故此处验证**等价命题**：这两个 env 对产物**完全无效**（旁路确已删）。自动选择改道由单测 [test_unit_engine_select.py](../../tests/test_unit_engine_select.py) 覆盖。

### P2 · 对外接口 advanced_features.json（features + variants）
| 用例 | 真实下发关键 | 断言 | 结果 |
| --- | --- | --- | --- |
| GLM-5.2·Ascend spec | `GLM-5.2-w8a8` + `enable-speculative-decode` + `ENGINE_VERSION=0.21.0-a3` | `features` 含四特性 bool；`speculative_decode=true`；`variants.speculative_decode=deepseek_mtp`；`sparse_kv/kv_offload=false` | ✅ |
| GLM-5.1·Ascend sparse | `glm-5.1` + `enable-sparse` + `WINGS_ASCEND_PLATFORM=a2` | `features.sparse_kv=true`；`variants.sparse_kv=indexcache_topk8` | ✅ |

### P3 · 对内特性日志（本轮新增）
| 用例 | 真实下发关键 | 断言 | 结果 |
| --- | --- | --- | --- |
| 卡型 miss 告警 | `glm-5.1`·`vllm_ascend`，**无** platform/engine-version/device-name | WARNING `card_token unresolved on Ascend` + 收口摘要存在 | ✅ |
| sparse 抑制对称日志 | `Llama-3-70B`·NV + `enable-sparse`+`enable-spec` | INFO `sparse requested but not in whitelist`；摘要含 `sparse True->False` | ✅ |
| offload 抑制对称日志 | `Llama-3-70B`·NV + `LMCACHE_OFFLOAD=true` | INFO `offload requested but not in whitelist` | ✅ |

### P4 · 三特性依赖白名单（含 PD 一票否决）
| 用例 | 真实下发关键 | 断言 | 结果 |
| --- | --- | --- | --- |
| 命中 → mtp | `GLM-5.2`·a3 + `enable-spec` | spec method == `deepseek_mtp` | ✅ |
| sparse-only → spec 地板 | `glm-5.1`·a2 + spec+sparse（B.4） | spec == `suffix`；sparse 仍产 `index_topk_freq:8` | ✅ |
| offload 白名单外 → 关 | `qwen3.5-397b`·NV + `LMCACHE_OFFLOAD` | `features.kv_offload=false`；不导出 `LMCACHE_OFFLOAD=true` | ✅ |
| PD 一票否决 | `glm-4.7`·NV + 三开关全开 + `PD_ROLE=P` | PD veto 日志；三特性全 false；无 mtp | ✅ |

### P5 · 内存自动计算 C4（反向预算）
> 公式 `M_offload = M_container − (7×TP×DP+3) − 10%margin`；POD=512、8 卡、TP8/DP1 → `512−59−51=401`。

| 用例 | 真实下发关键 | 断言 | 结果 |
| --- | --- | --- | --- |
| auto LMCache（均卡） | `glm-4.7`·NV + `LMCACHE_OFFLOAD`+`LMCACHE_POD_MEMORY=512` | `LMCACHE_MAX_LOCAL_CPU_SIZE=50`（=401÷8）+ 强制 `--swap-space 0` | ✅ |
| auto native（整节点） | `V4-Flash`·Ascend a2 + `LMCACHE_POD_MEMORY=512` | `cpu_swap_space_gb=401`（**不除卡数**） | ✅ |
| custom（透传） | `glm-4.7`·NV + `LMCACHE_MAX_LOCAL_CPU_SIZE=200` | `LMCACHE_MAX_LOCAL_CPU_SIZE=200`（不计算） | ✅ |
| 熔断 | `glm-4.7`·NV + `LMCACHE_POD_MEMORY=100`（→31<100） | 无 auto 写回 CPU 容量 + 熔断告警日志 | ✅ |

### P6 · 稀疏多模式 SPARSE_LEVEL
| 用例 | 真实下发关键 | 断言 | 结果 |
| --- | --- | --- | --- |
| 缺省 | `glm-5.1`·NV + `enable-sparse`（无 SPARSE_LEVEL） | 日志 `effective SPARSE_LEVEL=accuracy_first`；无 performance 告警 | ✅ |
| performance_first | + `SPARSE_LEVEL=performance_first` | 告警 `performance_first not implemented` + 回落 accuracy_first；**命令与缺省一致** | ✅ |
| 非法值 | + `SPARSE_LEVEL=turbo` | 回落 accuracy_first；不触发 performance 告警 | ✅ |

### P7 · 硬件信息 / ENGINE-VERSION 卡型解析
| 用例 | 真实下发关键 | 断言 | 结果 |
| --- | --- | --- | --- |
| engine-version 定卡型 | `DeepSeek-V3.2`（仅 910c 白名单），`ENGINE_VERSION=…-a3` vs `…-a2` | a3→910c 命中→spec `mtp` 族；a2→910b 不命中→`suffix`；两者不同 | ✅ |
| device-name 定卡型 | `glm-5.1` + `WINGS_DEVICE_NAME=ascend910b3` | 卡型 910b 命中 → `features.sparse_kv=true` | ✅ |

### P8 · 打补丁逻辑保持现状
| 用例 | 真实下发关键 | 断言 | 结果 |
| --- | --- | --- | --- |
| NV 装补丁 | `glm-5.1`·**vllm** + `enable-sparse`（GlmMoeDsa） | 命令含 `install.py --features {...indexcache...}` | ✅ |
| Ascend 不装补丁 | `glm-5.1`·**vllm_ascend** + `enable-sparse` | **不含** install.py indexcache；仅 `--hf-overrides index_topk_freq:8` | ✅ |

---

## 三、验证结果

```
python tests/dryrun_requirement_coverage.py
→ 总计：22 PASS / 0 FAIL   (exit 0)
```

**八个需求点全部经 dry-run 真实下发覆盖、断言通过。** 逐点小结：

| 点 | 结论 |
| --- | --- |
| P1 | 两条引擎路由 env 旁路对产物零影响、`USE_KUNLUN_ATB` 不再导出（残留确已删） |
| P2 | advanced_features.json 如实透出 features + variants（GLM-5.2 spec→deepseek_mtp、GLM-5.1 sparse→indexcache_topk8） |
| P3 | 卡型 miss WARNING / req→eff 摘要 / sparse·offload 抑制对称日志均按设计触发 |
| P4 | 白名单命中→mtp、sparse-only→suffix 地板、白名单外→收口关、PD→一票否决，全部正确 |
| P5 | auto 均卡(50) / native 整节点(401) / custom 透传(200) / 熔断 四态全对，swap_space=0 原子绑定 |
| P6 | accuracy_first 缺省、performance_first 告警回落且命令不变、非法值回落，三态全对 |
| P7 | ENGINE_VERSION 后缀（a2/a3）与 WINGS_DEVICE_NAME 两条兜底链都能定卡型、驱动白名单命中 |
| P8 | NV 装 indexcache 补丁、Ascend day0 仅 `--hf-overrides` 不装补丁，现状保持 |

---

## 四、覆盖缺口与说明

1. **P1 引擎自动选择改道**：dry-run 真实下发**无法触发**（不给 engine 时框架按 nvidia 推断设备，不进 Ascend 自动选择分支）。本方案以「env 旁路对产物无效」做等价验证；自动选择改道（已验证模型+开关→mindie）由单测 [test_unit_engine_select.py](../../tests/test_unit_engine_select.py) 覆盖。
2. **P2/部分点用 advanced_features.json 真实文件**：写到 `settings.SHARED_VOLUME_PATH`（缺省 `/shared-volume`，进程级固定），用例每次跑后即读，不依赖 start_command 内的回退 heredoc。
3. **spec 变体名差异**：DeepSeek-V3.2→`mtp`、GLM-5.2→`deepseek_mtp`、GLM-5.1→`deepseek_mtp`（均 mtp 族），断言按各自真实 variant 校验。

---

## 五、复现

```bash
cd wings-control
python tests/dryrun_requirement_coverage.py     # 22 PASS / 0 FAIL，产物见 tests/dryrun_requirement_coverage_output.txt
```

> 驱动器 [_dryrun_req_harness.py](../../tests/_dryrun_req_harness.py) 复用 `dry_run.py` 的三段式管线，额外：① 清理需求点专用 env（SPARSE_LEVEL/PD_ROLE/ENABLE_*/LMCACHE_POD_MEMORY…）防串味；② 捕获生产代码日志；③ 读取 advanced_features.json 真实 features+variants。
