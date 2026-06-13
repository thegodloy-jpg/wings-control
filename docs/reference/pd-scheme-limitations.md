# PD 分离方案的局限性与真实场景受限情况（验证过程归纳）

| 项 | 内容 |
|----|------|
| 日期 | 2026-06-13 |
| 来源 | dry-run 验证（`--pd glm5 / v4flash`）+ 代码走查 + 官方对比 + `pd_external_lb_verify.py` |
| 性质 | 本文**归纳局限**；修复方案见 [pd-scheme-fix-plan.md](pd-scheme-fix-plan.md) |
| **修复状态（2026-06-13）** | **L2 / L3 / L4 / L5 已实现并验证**（harness 层 F，61 PASS / 0 FAIL，非 PD 字节级不变）；**L1 暂缓**（真机依赖）；其余（L8 engine_id、L9-L11 真机契约、L12-L17 真机/盲区）维持现状。 |
| 读法 | 每条标注：**性质**（实现可修 / 方案固有 / 真机依赖）、**证据**（代码位置或 dry-run 观察）、**真实场景影响** |

> 验证方法本身的边界先说在前：dry-run 用 **mock 模型目录（只有 config.json）**、离线生成脚本，**不跑真机**。因此本文「确认」级别仅限「生成的命令/分支逻辑」，凡涉及 HCCL/Mooncake 实际连通、算子执行、显存、吞吐的均属**真机依赖**，dry-run 给不了结论。

---

## 一、功能受限（部分开关/配置在 PD external-lb 下不可用或不可表达）

### L1. IndexCache（KV 稀疏）开关在 PD external-lb 下被丢弃 —— 实现可修
- **证据**：`build_start_script` 算了 `sparse_args = _build_kv_sparse_cmd(...)`，但 PD 分支 `_build_vllm_pd_external_lb_script(params, cmd, common_env_cmds, pd_ext)` **没接收 sparse_args**（[vllm_adapter.py:2992](../wings_control/engines/vllm_adapter.py)）；单机/分布式分支才传。
- **影响**：GLM-5.1(ascend) 的 IndexCache（`--hf-overrides`）在 PD fork 命令里**不会下发**，`--enable-sparse` 在 PD 形同失效。FP8 KV-sparse 因就地改 engine_config 会留在 cmd（但那不是 IndexCache）。**真实场景**：需要 IndexCache 长上下文/稀疏 KV 的模型一旦走 PD 分离，该能力静默丢失。

### L2. 角色级 `kv_connector_extra_config` 无法表达 —— 实现受限
- **证据**：注册表 `extra_config` 经 `_build_pd_external_lb_kv` 对 P/D **同时**写入（[config_loader.py:1007-1009](../wings_control/core/config_loader.py)）；官方 Qwen3.5 的 `kv_buffer_device:"npu"` 只在 **consumer(D)** 出现。
- **影响**：凡「只该出现在 P 或只该出现在 D」的 KV extra 字段都无法精确表达，只能两边都下发或都不下发。**真实场景**：Qwen3.5 等模型的消费端专属 KV 配置无法对齐官方。

### L3. 共用环境变量只能经角色 env 笨拙叠加，无干净的"共用 env"控制 —— 实现分层局限

> ⚠️ 修正：原表述「pd_config 改不动」**不准确**。实测注册表角色 env **能覆盖/新增**任意 env（bash 后者生效）；真正的局限是「只能笨拙叠加」，下述。

**env 分三层，注册表只直接摸得到第 3 层：**

| 层 | 产出 | 是否注册表可控 |
|----|------|:------------:|
| ① base/通用 | `_build_base_env_commands` + `_build_vllm_ascend_forced_env_commands`（软默认 `export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}`，[vllm_adapter.py:423](../wings_control/engines/vllm_adapter.py)） | ❌ |
| ② 架构/平台块 | `_build_model_env_commands` → 各架构硬编码块；V4-Flash 经 `_build_deepseek_v4_flash_env` 按 `WINGS_ASCEND_PLATFORM` 选 **a2(`HCCL_BUFFSIZE=512`) / a3(`=1024`)** 整块（[vllm_adapter.py:1529-1560](../wings_control/engines/vllm_adapter.py)）；GLM5 的 1024 亦出自此层 | ❌（块的选择不可控） |
| ③ 注册表角色 env | `pd_config` 的 `prefill.env` / `decode.env`（`_pd_env`），在 PD fork 脚本里**追加在 common_env 之后**（[vllm_adapter.py:2917-2920](../wings_control/engines/vllm_adapter.py)） | ✅ |

**实测证据（`start_command_pd-glm5-D_node0.sh`）**：base `export HCCL_BUFFSIZE=1024`@L237；注册表角色 env `MLAPO/TASK_QUEUE`@L243-247 **在其后** → bash 后者生效。故注册表角色 env **可覆盖/新增**任意 env（HCCL_BUFFSIZE=256、补 ASCEND_AGGREGATE_ENABLE 等都做得到）。

**真正的 4 条局限（"能改但不干净"）：**
1. **无"共用 env"槽**：注册表 env 只有 `prefill`/`decode` 两个角色位，没有 `common.env`。P/D 都要的 env（HCCL_BUFFSIZE / Mooncake 超时）须**在两个角色里各写一遍**，漏一个就只覆盖一个角色。
2. **角色 env 不参与去重**：`dedupe_env_exports` 只在 `_build_vllm_common_env_cmds` 内跑（角色 env 是其后追加的），覆盖后脚本里 **base 值与覆盖值并存**（实测 `MLAPO` 出现两次 @243/245，最终值对但更噪）。
3. **不能切换"平台/架构块"**：a2/a3 env 整块由 `WINGS_ASCEND_PLATFORM` 决定，注册表只能在选定块之后逐项覆盖，改不了"选哪块"。
4. **不能 unset**：engine_config 可用 `null` 删键（如 P 的 `compilation_config=null`），但 env **无删除语义**，只能覆盖成别的值。

- **真实场景**：让 GLM5 PD 的 `HCCL_BUFFSIZE` 对齐官方 256、补 4 个 Mooncake 共用 env —— **做得到**，但要在 `prefill.env` 与 `decode.env` 各写一遍，且脚本里 1024/256 并存。想要"单点 + 可删 + 去重"，需加 `common.env` 槽并让 PD 脚本对角色 env 也跑 dedupe（实现改动）。

### L4. A2 / A3 双平台无法同条目表达（注册表无平台维度）—— 方案受限

**机制**：注册表 key = **纯模型架构**，无平台子键。
- `entry = registry.get(arch) or registry.get("default")`（[config_loader.py:1037](../wings_control/core/config_loader.py)），`merged_engine = {**common, **role.engine}` 也无平台分支。
- 平台（a2/a3）是另一条独立线：`_resolve_deepseek_v4_flash_platform`（取 `WINGS_ASCEND_PLATFORM`，[vllm_adapter.py:1333](../wings_control/engines/vllm_adapter.py)），只驱动 **env 块**（L3 的 `_build_deepseek_v4_flash_env`）和模型默认注入器的少量分支，**不进注册表**。

**为什么一个条目装不下两套** —— 官方 V4-Flash 两平台的**引擎数值**不同：

| | A3 (1P1D) | A2 (4P1D) |
|--|:--:|:--:|
| P max-num-batched-tokens / seqs | 8192 / 16 | 4096 / 16 |
| D max-num-batched-tokens / seqs | 120 / 60 | 60 / 30 |
| 拓扑（来自上层，不归注册表） | P DP4×TP4 / D DP16×TP1 | 4P:1D 比例不同 |

注册表 `DeepseekV4ForCausalLM.prefill.engine.max_num_batched_tokens` 只能填**一个**值（本次填 A3 的 8192）。一旦该 pod 实际是 **A2**，`_apply_pd_external_lb` 仍注入 8192（A2 应 4096），D 的 60/30 也拿不到 —— **平台错配但不报错**。
> 注：错配的是**每角色 engine 数值**（batch/mem 等）；dp/tp 拓扑来自上层 `DP_SIZE/TP_SIZE`，不受此影响。但数值偏大同样会 OOM / 调度不匹配。

**支持双平台的几条路径（均需改动）：**
1. **注册表加平台维度**：如 `DeepseekV4ForCausalLM: {"a2":{...}, "a3":{...}}`，并让 `_apply_pd_external_lb` 用 `_resolve_*_platform` 选子条目（代码 + schema 改动）。
2. **平台相关数值交还模型默认注入器**（它已 platform-aware），注册表只放平台无关项 —— 但与「注册表权威」(L5) 冲突，且会再触发 L5 的注入器回填问题。
3. **拆两个架构 key**（不优雅，arch 实际相同）。

- **真实场景**：当前条目只在 **A3 安全**；A2 上跑 V4-Flash PD 会拿到 A3 的 batch/mem（偏大）→ 可能 OOM 或调度不匹配。且 dry-run 也测不出（`PD_SCENARIOS` 固定 `platform=a3`，见 L18）。

---

## 二、架构脆弱性（"能对齐"靠补丁式机制，易回退）

### L5. 注册表"权威"靠「深拷贝 + 注入器后重申」补丁，非干净设计 —— 方案脆弱
- **证据**：`_apply_pd_external_lb`（config_load 最后一步）写入的注册表值，会被随后 `_prepare_engine_config` 里的模型默认注入器（`_apply_*_engine_defaults`）用 `_force_set_*` / `_merge_dict_default_*` **回填覆盖**。本次靠新增 `_pd_engine_overrides` 深拷贝 + 在注入器后重申才压住（[vllm_adapter.py:_prepare_engine_config](../wings_control/engines/vllm_adapter.py) 末尾）。
- **影响**：**任何新加的注入器 `_force_set_*` 都可能再次静默覆盖注册表值**，且 dry-run 不一定立刻暴露（取决于键是否被某模型注入器命中）。**真实场景**：未来加模型/改默认时，PD 字段可能悄悄回退到 base，需回归 `pd_external_lb_verify.py` 才能发现。

### L6. 注册表缓存被就地深合并污染过 —— 已修但暴露设计风险
- **证据**：`_load_pd_config` 用模块级 `_PD_CONFIG_CACHE`（[config_loader.py:92-97](../wings_control/core/config_loader.py)）；V4-Flash 注入器对 `additional_config` **就地深合并**，曾污染缓存与下游（本次已用 deepcopy 修）。
- **影响**：共享可变结构 + 就地改 = 跨次调用泄漏。**真实场景**：一个 pod 内若复用进程/多次构建，未深拷贝会让前一次的合并结果串到下一次。

### L7. 注册表数值为人工维护、易与官方漂移 —— 方案固有
- **证据**：V4-Flash 原条目批量/显存/seed/max-len/speculative 全是「摘要待核」占位值，与官方 A3 全差（本次对齐）；`_comment` 自标「落地前需逐项核对」。
- **影响**：官方文档更新或新增模型时，注册表需人工追平，**无自动校验**。**真实场景**：版本演进后 PD 参数可能落后于官方最优配置而无人察觉。

---

## 三、正确性风险（真实多机/多 service 场景下可能出问题）

### L8. engine_id 对 Hybrid 连接器的注入策略未经真机确认 —— 真机依赖
- **证据**：`_build_pd_external_lb_kv` 对 V1 **和** Hybrid 一律按 `dp_rank` 注 `engine_id`（[config_loader.py:1018-1019](../wings_control/core/config_loader.py)）；官方 V4-Flash(Hybrid) 示例为固定 `0/1`。
- **影响**：多 service 下按 rank 更合理（避免同 pod 冲突），但 Mooncake Hybrid 是否要求 role 级常量未知。**真实场景**：若 Hybrid 期望 role 级 engine_id，KV 建连可能异常。

### L9. KV 全局拓扑依赖上层下发，缺失则静默回退 —— 方案契约风险
- **证据**：`_build_pd_external_lb_kv` 对端拓扑取 `PD_PREFILL_*/PD_DECODE_*`，缺失时**回退本角色拓扑并仅告警**（[config_loader.py:993-1004](../wings_control/core/config_loader.py)）。
- **影响**：上层漏发对端 dp/tp → KV 的 TP/DP 映射算错，**仅一条 warning，不阻断启动**。**真实场景**：跨角色 KV 传输错配，可能表现为 decode 输出错乱而非显式报错。

### L10. dp_rank_start 由 HOST_IP 在 NODE_IPS 的位置派生 —— 脆弱
- **证据**：`node_rank = node_ips.index(host_ip) if host_ip in node_ips else 0`（[config_loader.py:954-957](../wings_control/core/config_loader.py)）。
- **影响**：HOST_IP 与 NODE_IPS 文本不完全一致（如带/不带端口、大小写、顺序错位）就**回退 rank_start=0**，多节点会 rank 撞车。**真实场景**：IP 列表与实际网卡 IP 不一致时，多个 D 节点抢同一 rank，DP 域组不起来。

### L11. hybrid-kv guard 与 Hybrid 连接器语义相反 —— 设计耦合
- **证据**：`_guard_pd_hybrid_kv_cache` 为 V1/Nixl（HMA 不兼容）设计、PD 下无条件移除 `no_disable_hybrid_kv_cache_manager`（[config_loader.py:1259](../wings_control/core/config_loader.py)）；但 V4-Flash(Hybrid) 官方要保留 HMA。本次靠注册表**晚于 guard 注入**绕过。
- **影响**：guard 的「PD 一律去 HMA」假设对 Hybrid 连接器不成立，靠时序绕过而非显式按连接器分流。**真实场景**：若 guard 逻辑调整或时序变化，V4-Flash 的 `--no-disable-hybrid-kv-cache-manager` 可能又被吃掉。

---

## 四、可用性 / 运维受限

### L12. 任一 service 失败 → 整 pod 拆除 —— 方案固有（EP 语义）
- **证据**：fork 子 shell `wait -n || true; kill "${pids[@]}"; exit 1`（[vllm_adapter.py:2940-2944](../wings_control/engines/vllm_adapter.py)）。
- **影响**：EP all-to-all 下单 rank 缺失会让整域 hang，故设计为整 pod 重启。**真实场景**：大 DP（如 D dp16 单 pod 16 service）下，任一 service OOM/崩溃即整 pod 16 卡全重启，**故障粒度粗、抖动放大**。

### L13. 启动脚本 env 段跨 exec path 重复 —— 可读性（已部分缓解）
- **证据**：本次已加 `dedupe_env_exports` 做**每 exec path 内**去重；但脚本含「初始 + crash-retry」两个 exec path，env 整块仍重复一遍。
- **影响**：日志/脚本体量大、排查时需注意是哪个 path。**真实场景**：retry 触发时两份 env 都在日志里，肉眼 diff 易混。

### L14. rpc 死值 / 端口块自洽依赖约定 —— 真机依赖
- **证据**：P/D rpc 用两个常量（10521/10523），端口块 18000+i、kv 300xx/301xx、bootstrap 230xx/231xx 按 i 偏移。
- **影响**：P/D **同机共置**时单一 rpc 常量可能撞（设计文档 R4）；平台对容器做卡映射时 `ASCEND_RT_VISIBLE_DEVICES` 基址需对齐（R5）。**真实场景**：混部/共置部署需额外校验端口与卡组不重叠。

---

## 五、验证盲区（dry-run 给不了结论的部分）

| # | 盲区 | 为什么 dry-run 看不到 |
|---|------|----------------------|
| L15 | 真机 HCCL/Mooncake 多机连通、P→D KV 实际传输 | 离线只生成命令，不建链 |
| L16 | `FULL_DECODE_ONLY` 全图 decode replay 的 MTE 越界历史（设计文档 R6、[[glm5-aclgraph-mte-crash]]） | 需真机捕获图执行 |
| L17 | mock 模型无真实权重 → 量化探测 / `head_dim` 注入（`_ensure_pd_head_dim`）/ tokenizer 行为 | config.json 占位，无 safetensors |
| L18 | Qwen3.5-397B / Qwen3-30B 无 `PD_SCENARIOS`，**无法生成命令验证** | dry_run 仅注册 glm5/v4flash 两个 PD 场景 |
| L19 | reasoning_parser 与 `enable_auto_think_choice` 的交互（[[reasoning-parser-official-alignment]]） | 需运行时观察思考开关行为 |

---

## 六、小结：当前方案适用边界

**适用良好**：已注册架构（GLM-5 / V4-Flash / V3.2 / Qwen3.5）、上层完整下发 5 个 DP 参数 + 对端拓扑、单一平台（A3）、单节点或 IP 列表精确的多节点、不依赖 IndexCache 的 PD 分离。

**受限/需谨慎**：
1. 需要 **IndexCache** 的 PD 场景（L1，当前不生效）；
2. **A2 平台** 或同架构多平台（L4）；
3. **角色级 KV extra 字段**（L2）/共用 env 模型差异（L3）；
4. 上层**对端拓扑/IP 列表不规整**（L9、L10，静默错配风险）；
5. **P/D 同机共置**（L14 端口/rpc 冲突）；
6. 大 DP 单 pod 的**可用性**（L12 整 pod 重启粒度）；
7. 一切**真机相关**（L15-L17，必须 bring-up 验证）。

**根因型脆弱点（优先收敛）**：L5（注入器回填覆盖注册表，靠重申补丁压住）—— 这是「pd_config 是否真权威」的命门，新增模型/改默认时务必回归 `tests/pd_external_lb_verify.py`。

> 关联：字段对齐明细 [pd-dryrun-vs-official-report.md](pd-dryrun-vs-official-report.md)；逐场景验证 [pd-dryrun-verification-report.md](pd-dryrun-verification-report.md)；机制记录见记忆 [[pd-registry-authority-mechanism]]。
