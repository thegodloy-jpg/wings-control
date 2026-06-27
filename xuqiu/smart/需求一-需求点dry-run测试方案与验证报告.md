# 需求一 · 三特性使能改造 — 八需求点 dry-run 测试方案与验证报告

> 关联：[需求一-完成度核查与遗漏清单.md](需求一-完成度核查与遗漏清单.md) · [需求一-dry-run验证报告.md](需求一-dry-run验证报告.md) · [需求一-三特性使能.md](需求一-三特性使能.md)
> 日期：2026-06-27。分支：`feat/smart-three-feature-enablement`。
> 落地物：驱动器 [tests/_dryrun_req_harness.py](../../tests/_dryrun_req_harness.py) + 用例集 [tests/dryrun_requirement_coverage.py](../../tests/dryrun_requirement_coverage.py)；证据产物 [tests/dryrun_requirement_coverage_output.txt](../../tests/dryrun_requirement_coverage_output.txt)（逐用例打印入参三段 + 下发通道 + 出参期望/实际）。
> **结果：29 用例 / 60 断言，全部 PASS（exit 0）。** 运行：`python tests/dryrun_requirement_coverage.py`

---

## 〇、★ 核心口径：三特性仅经「环境变量」下发，不走 CLI

投机（spec）/ 稀疏（sparse）/ 卸载（offload）三特性的使能，**由 MaaS 页面开关 → 编排层注入环境变量**触发，**不是 `wings_start.sh` 的用户 CLI 标志**：

| 特性 | 下发环境变量（编排注入） | 真实链路依据 |
| --- | --- | --- |
| 投机 SmartDecoding | `ENABLE_SPECULATIVE_DECODE=true` | `wings_start.sh:299` 读该 env → `:345` 传播进 APP_ARGS |
| 稀疏 SmartKVSparse | `ENABLE_SPARSE=true` | `wings_start.sh:300` 读该 env → `:347` 传播 |
| 卸载 SmartKVCache | `LMCACHE_OFFLOAD=true`（纯 env，无 CLI 标志） | `config_loader.get_lmcache_env()` 直接读 |

**因此本方案所有用例：三特性开关一律置于 `orchestration_env`（env），`user_cli` 不含任何 `--enable-*` 特性标志。** 驱动器复刻 `wings_start.sh(299-300/345-348)` 的 env→APP_ARGS 传播，与真实链路等价。P0 节专门验证「env 下发即生效、user_cli 无 CLI 标志」。

---

## 一、测试方案（模拟用户真实下发）

### 1.1 真实下发口径 —— 三段式硬边界（不 mock 模型识别）

| 入参段 | 代表谁 | 内容 |
| --- | --- | --- |
| **user_cli** | 用户在 `wings_start.sh` 真敲的 CLI | `--model-name/--engine/--device-count/--distributed`…（**不含三特性开关**） |
| **orchestration_env** | 编排层/K8s/MaaS 进程启动前注入的 env | **三特性开关**（`ENABLE_SPECULATIVE_DECODE/ENABLE_SPARSE/LMCACHE_OFFLOAD`）+ 拓扑/平台/`ENGINE_VERSION`/`SPARSE_LEVEL`/`LMCACHE_POD_MEMORY`/`PD_ROLE`… |
| **model_config** | 模型权重 `config.json` | `architecture` + `quantization_config` |

链路：`reset`（每例=全新 pod）→ mock config.json → 注入编排 env → `simulate_wings_start` +（复刻 wings_start.sh）env→APP_ARGS 传播 → `parse_launch_args` → `build_launcher_plan` → `start_command.sh`。

### 1.2 出参（三类可观测产物 = 断言对象）

| 出参 | 来源 | 断言示例 |
| --- | --- | --- |
| **start_command.sh** 可执行命令行 | `plan.command`（真实 exec 行） | `--speculative-config`/`--hf-overrides`/`--swap-space 0`/`cpu_swap_space_gb`/`LMCACHE_MAX_LOCAL_CPU_SIZE=N`/`install.py --features` |
| **advanced_features.json** | 生成期写到 `settings.SHARED_VOLUME_PATH`（真相源，含 variants） | `features.{spec/sparse/offload}` bool + `variants.{...}` 字符串 |
| **生产代码日志** | 捕获 `core.config_loader`/`engines.vllm_adapter` 的 INFO/WARNING | 收口摘要 / 卡型 miss 告警 / 抑制日志 / SPARSE_LEVEL 回落告警 |

> ⚠ `start_command.sh` 内嵌的 `cat > advanced_features.json <<FEATURES_EOF` 是**崩溃回退默认块（全 false、无 variants）**；真相源是 `settings.SHARED_VOLUME_PATH/advanced_features.json`（生成期落盘，含 variants）。

### 1.3 设计原则

- **三特性 env 下发**：开关一律 env（见 §〇），杜绝把「页面开关」误模型成「用户 CLI」。
- **正/反双向**：命中→产出 与 关/未命中→不产 都设例。
- **开关正交分层**：使能 = **开关 env(switch) AND 白名单(whitelist)**；稀疏额外叠 **档位(SPARSE_LEVEL)**。逐层独立设例（见 §三 P6）。

---

## 二、稀疏「开关」专题（三层门控，正面回应「有没有验证稀疏开关」）★

稀疏是否产出，由**三层门控**决定，缺一层即不产；开关本身经 `ENABLE_SPARSE` env 下发：

```
SmartKVSparse 产出 ⇔  层1 开关 ENABLE_SPARSE=true（env，非 CLI）
                  AND 层2 (engine,model,卡) ∈ sparse 白名单
                  AND 层3 档位 SPARSE_LEVEL（performance_first 暂回落 accuracy_first）
```

| 层 | 用例 | 入参要点（三特性=env） | 出参（实际） |
| --- | --- | --- | --- |
| **下发通道** | TC-P0-01 | `ENABLE_SPARSE=true`（env），user_cli **无** `--enable-sparse` | `features.sparse_kv=True`、`variants=indexcache_topk4`（仅 env 驱动即生效） |
| **层1 开关 OFF** | TC-P6-01 | 不设 `ENABLE_SPARSE` | `--hf-overrides` 不出现；`sparse_kv=False`；**无** `effective SPARSE_LEVEL` 日志 |
| **层1 优先层3** | TC-P6-02 | 不设 `ENABLE_SPARSE` + `SPARSE_LEVEL=performance_first` | `sparse_kv=False`；**无** performance_first 告警 → 开关 OFF 直接门控档位 |
| **层2 白名单** | TC-P6-03 | `ENABLE_SPARSE=true` + glm-4.7·Ascend（白名单无 sparse） | `sparse_kv=False`；日志 `sparse requested but not in whitelist → suppressed` |
| **层3 档位·缺省** | TC-P6-04 | `ENABLE_SPARSE=true` + glm-5.1·NV，无 SPARSE_LEVEL | `variants=indexcache_topk4`；日志 `effective SPARSE_LEVEL=accuracy_first` |
| **层3 档位·perf** | TC-P6-05 | + `SPARSE_LEVEL=performance_first` | 告警 `performance_first not implemented`；回落 accuracy_first；**命令与缺省字节一致** |
| **层3 档位·非法** | TC-P6-06 | + `SPARSE_LEVEL=turbo` | 回落 accuracy_first；不触发 performance 告警 |

> 结论：稀疏开关（env ON/OFF）、开关对档位的门控优先级、开关×白名单×档位三层正交，**全部验证通过**。

---

## 三、逐用例规格与结果（入参 → 期望出参 → 实际）

> 完整逐项证据见 [dryrun_requirement_coverage_output.txt](../../tests/dryrun_requirement_coverage_output.txt)。三特性开关均在 `orchestration_env`（env）。

### P0 · 下发通道：三特性仅经 env，不走 CLI

| TC | 入参（关键） | 出参（期望→实际） | 判定 |
| --- | --- | --- | --- |
| TC-P0-01 | user_cli=`{glm-5.1, vllm, 8}`（无 --enable-sparse）；orch=`{…, ENABLE_SPARSE: true}` | user_cli 含 CLI 标志=否；features.sparse_kv=True；variants=indexcache_topk4 | ✅ |
| TC-P0-02 | orch=`{…, ENGINE_VERSION:0.21.0-a3, ENABLE_SPECULATIVE_DECODE: true}` | user_cli 含 CLI 标志=否；variants.speculative_decode=deepseek_mtp | ✅ |
| TC-P0-03 | orch=`{…, LMCACHE_OFFLOAD: true, LMCACHE_POD_MEMORY:512}` | features.kv_offload=True；variants.kv_offload=lmcache_cpu+auto | ✅ |

### P1 · 删 fp4/fp8 + 引擎路由删除

**TC-P1-01** — 旁路已删，开关 env 对产物无效
- user_cli：`{model-name: Qwen3.5-397B-A17B, engine: vllm_ascend, device-count: 16}`
- orchestration_env：`{DISTRIBUTED_EXECUTOR_BACKEND: mp, WINGS_ASCEND_PLATFORM: a3, ENABLE_OPERATOR_ACCELERATION: true, ENABLE_SOFT_FP8: true}`（对照组去掉后两个）
- model_config：`{architecture: Qwen3_5MoeForConditionalGeneration, quant=ascend}`

| 出参 | 期望 | 实际 | 判定 |
| --- | --- | --- | --- |
| engine（设两开关后） | vllm_ascend（与不设一致） | vllm_ascend | ✅ |
| start_command 归一化 | 与不设两 env 字节一致 | 一致 | ✅ |
| 命令导出 USE_KUNLUN_ATB | 不出现 | 不出现 | ✅ |

### P2 · 对外接口 advanced_features.json

**TC-P2-01** — GLM-5.2·Ascend spec → features+variants
- user_cli：`{GLM-5.2-w8a8, vllm_ascend, 16}` · orch：`{dp_deployment, ENGINE_VERSION:0.21.0-a3, ENABLE_SPECULATIVE_DECODE:true}` · model：`GlmMoeDsa, quant=ascend`

| 出参 | 期望 | 实际 | 判定 |
| --- | --- | --- | --- |
| features 键集合 | {spec,sparse,offload,rag_acc} | 四键齐 | ✅ |
| features.speculative_decode | True | True | ✅ |
| variants.speculative_decode | deepseek_mtp | deepseek_mtp | ✅ |
| features.sparse_kv / kv_offload | False / False | False / False | ✅ |

**TC-P2-02** — GLM-5.1·Ascend sparse（orch `ENABLE_SPARSE:true, WINGS_ASCEND_PLATFORM:a2`）：`features.sparse_kv=True`✅；`variants.sparse_kv=indexcache_topk8`✅。

### P3 · 对内特性日志（本轮新增）

| TC | 入参要点（三特性=env） | 出参（期望→实际） | 判定 |
| --- | --- | --- | --- |
| TC-P3-01 | glm-5.1·vllm_ascend，orch `ENABLE_SPARSE:true`、**无** platform/engine-version/device-name | 日志 `card_token unresolved on Ascend`（出现）；收口摘要存在 | ✅ |
| TC-P3-02 | Llama-3-70B·NV，orch `ENABLE_SPARSE:true, ENABLE_SPECULATIVE_DECODE:true` | 日志 `sparse requested but not in whitelist`；摘要含 `sparse True->False` | ✅ |
| TC-P3-03 | Llama-3-70B·NV，orch `LMCACHE_OFFLOAD:true` | 日志 `offload requested but not in whitelist` | ✅ |

### P4 · 白名单门控 + PD + 开关基线

| TC | 入参要点（三特性=env） | 出参（期望→实际） | 判定 |
| --- | --- | --- | --- |
| TC-P4-01 | GLM-5.2·a3，`ENABLE_SPECULATIVE_DECODE:true` | variants.speculative_decode = deepseek_mtp | ✅ |
| TC-P4-02 | glm-5.1·a2，`ENABLE_SPECULATIVE_DECODE:true + ENABLE_SPARSE:true` | spec=suffix（地板, B.4）；sparse=indexcache_topk8 | ✅ |
| TC-P4-03 | qwen3.5-397b·NV，`ENABLE_SPARSE:true + LMCACHE_OFFLOAD:true + POD=512` | features.kv_offload=False；不导出 LMCACHE_OFFLOAD=true | ✅ |
| TC-P4-04 | glm-4.7·NV，`ENABLE_*` 三特性全开 + `PD_ROLE=P` | 日志 PD veto；features 三特性全 False | ✅ |
| TC-P4-05 | GLM-5.2·a3，**不设 ENABLE_SPECULATIVE_DECODE** | `--speculative-config` 不出现；features.speculative_decode=False | ✅ |
| TC-P4-06 | glm-4.7·NV，**不设 LMCACHE_OFFLOAD** | 不导出 LMCACHE_OFFLOAD=true；features.kv_offload=False | ✅ |

### P5 · 内存自动计算 C4（`M_offload = POD − (7×TP×DP+3) − 10%`）

| TC | 入参要点（三特性=env） | 出参（期望→实际） | 判定 |
| --- | --- | --- | --- |
| TC-P5-01 | glm-4.7·NV，`LMCACHE_OFFLOAD:true + POD=512`，8卡(TP8/DP1) | `LMCACHE_MAX_LOCAL_CPU_SIZE=50`（=401÷8，均卡）；`--swap-space 0` | ✅ |
| TC-P5-02 | V4-Flash·Ascend a2，`LMCACHE_OFFLOAD:true + POD=512`，8卡 | `cpu_swap_space_gb=401`（整节点，**不除卡数**） | ✅ |
| TC-P5-03 | glm-4.7·NV，`LMCACHE_OFFLOAD:true + LMCACHE_MAX_LOCAL_CPU_SIZE=200`（custom） | `LMCACHE_MAX_LOCAL_CPU_SIZE=200`（透传不算） | ✅ |
| TC-P5-04 | glm-4.7·NV，`LMCACHE_OFFLOAD:true + POD=100`（→31<100 熔断） | 无 auto 写回 CPU 容量；命中熔断告警日志 | ✅ |

### P6 · 稀疏三层门控（详见 §二）

TC-P6-01 ~ TC-P6-06，6 例全 PASS。覆盖开关 env ON/OFF、开关门控档位、白名单抑制、accuracy_first 缺省、performance_first 告警回落（命令不变）、非法值回落。

### P7 · 硬件信息 / ENGINE-VERSION 卡型解析

**TC-P7-01** — ENGINE_VERSION 后缀定卡型（DeepSeek-V3.2 仅 910c 白名单）
- user_cli：`{DeepSeek-V3.2, vllm_ascend, 16}` · model：`DeepseekV32ForCausalLM, quant=ascend`
- orch：`ENABLE_SPECULATIVE_DECODE:true` + 仅改 `ENGINE_VERSION`（`0.21.0-a3` vs `0.21.0-a2`），无 platform/device-name

| 出参 | 期望 | 实际 | 判定 |
| --- | --- | --- | --- |
| `-a3` → variants.speculative_decode | mtp 族（910c 命中） | mtp | ✅ |
| `-a2` → variants.speculative_decode | suffix（910b 不命中→地板） | suffix | ✅ |
| a2 vs a3 | 决策不同 | 不同 | ✅ |

**TC-P7-02** — orch `WINGS_DEVICE_NAME=ascend910b3 + ENABLE_SPARSE:true` → 卡型 910b 命中 GLM-5.1 sparse：`features.sparse_kv=True`✅。

### P8 · 打补丁逻辑保持现状

| TC | 入参要点（三特性=env） | 出参（期望→实际） | 判定 |
| --- | --- | --- | --- |
| TC-P8-01 | glm-5.1·**vllm**，`ENABLE_SPARSE:true`（GlmMoeDsa） | 命令含 `install.py --features {...indexcache...}` | ✅ |
| TC-P8-02 | glm-5.1·**vllm_ascend**，`ENABLE_SPARSE:true` | **不含** install.py indexcache；仅 `--hf-overrides index_topk_freq:8` | ✅ |

---

## 四、汇总

```
python tests/dryrun_requirement_coverage.py
→ 用例：29 个（29 PASS / 0 FAIL）  ·  断言：60 PASS / 0 FAIL   (exit 0)
```

| 点 | 用例数 | 结论 |
| --- | --- | --- |
| P0 下发通道 | 3 | 三特性**仅经 env**（ENABLE_SPECULATIVE_DECODE/ENABLE_SPARSE/LMCACHE_OFFLOAD）即生效，user_cli 无 CLI 标志 |
| P1 删 fp8/算子加速 | 1 | 两条引擎路由 env 旁路对产物零影响、USE_KUNLUN_ATB 不导出 |
| P2 对外接口 | 2 | advanced_features features+variants 如实透出 |
| P3 对内日志 | 3 | 卡型 miss / req→eff 摘要 / sparse·offload 抑制日志均触发 |
| P4 白名单+PD+开关基线 | 6 | 命中→变体、地板、收口关、PD 否决、env 开关 OFF→不产 |
| P5 内存自动计算 | 4 | auto 均卡/native 整节点/custom/熔断 四态 + swap_space=0 |
| P6 稀疏三层门控 | 6 | 开关 env ON/OFF、开关门控档位、白名单抑制、档位三态 |
| P7 卡型解析 | 2 | ENGINE_VERSION(a2/a3) + WINGS_DEVICE_NAME 两条兜底链 |
| P8 打补丁 | 2 | NV 装补丁 / Ascend day0 不装 |

---

## 五、覆盖缺口与说明

1. **P1 引擎自动选择改道**：dry-run 真实下发无法触发（未给 engine 时框架按 nvidia 推断设备，不进 Ascend 自动选择分支）。以「两个 env 旁路对产物无效」做等价验证；自动选择改道由单测 [test_unit_engine_select.py](../../tests/test_unit_engine_select.py) 覆盖。
2. **三特性 env 下发的链路保真**：驱动器复刻 `wings_start.sh(299-300/345-348)` 把 MaaS 注入的 `ENABLE_*` env 传播进 APP_ARGS，与真实 `python -m wings_control` 收到的最终 argv 一致。
3. **真相源**：features/variants 读 `settings.SHARED_VOLUME_PATH/advanced_features.json`，非 start_command 内的崩溃回退 heredoc。
4. **变体名按架构真实值断言**：DeepSeek-V3.2→`mtp`、GLM-5.x→`deepseek_mtp`，均 mtp 族；地板→`suffix`。

---

## 六、复现

```bash
cd wings-control
python tests/dryrun_requirement_coverage.py     # 29 PASS / 0 FAIL
#   产物（逐用例入参三段 + 下发通道 + 出参期望/实际）：tests/dryrun_requirement_coverage_output.txt
```

> 驱动器 [_dryrun_req_harness.py](../../tests/_dryrun_req_harness.py) 复用 `dry_run.py` 三段式管线，额外：① 复刻 wings_start.sh 的 `ENABLE_*` env→APP_ARGS 传播（三特性 env 下发）；② 清需求点专用 env 防串味；③ 捕获生产日志；④ 读 advanced_features.json 真实 features+variants。
