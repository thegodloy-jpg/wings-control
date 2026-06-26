# PD 分离实际部署日志 × 官方 GLM-5.2 命令 — 环境变量与引擎字段详细对比报告

> **对比对象（仅此两份，不涉及分支代码/配置文件）**
> - **A · 官方**：用户提供的官方 GLM-5.2 A2 **4P4D** 手工 P/D 启动脚本（`vllm serve … --kv-transfer-config …`）。
> - **B · 我们**：实际部署运行日志中引擎自打印的 `[wings-env] export …`（环境变量）与 `non-default args: {…}`（引擎入参）。日志时间 2026-06-25，权重为 **GLM-5.1**，套用共享架构键 `GlmMoeDsaForCausalLM` 的 5.2 对齐配置。
> - 报告日期：2026-06-26。判定图例：✅ 完全一致 ｜ ⚠️ 实质偏差 ｜ ➕ wings 额外（无害/运维）｜ △ 标签/外观差异 ｜ 🔧 external-lb 机制差异（设计内）。

---

## 0. 一图结论

| 维度 | Prefill (P) | Decode (D) |
|---|---|---|
| **环境变量** | ✅ 官方 18 项全命中、值一致 | ✅ 官方 19 项全命中、值一致 |
| **引擎字段** | ✅ 仅 `max_model_len` + `seed` 偏差 | ⚠️ **3 处实质偏差** + `seed` |
| 角色专属隔离 | ✅ 无串台 | ✅ 无串台 |

**一句话：问题不在环境变量（P/D 全对齐），全部集中在 Decode 的几个标量引擎字段。**

拓扑（两份一致）：P = `tp8 × dp4` = 32 卡 = 4 机；D = `tp4 × dp8` = 32 卡 = 4 机 → 4P4D。

---

## 1. 环境变量对比

### 1.1 Prefill 节点（日志 IP 10.254.27.226，pid=742）

| # | 变量 | 官方 P | 日志 P | 判定 |
|---|---|---|---|---|
| 1 | HCCL_IF_IP | =本机IP | 10.254.27.226 | ✅ 同义 |
| 2 | VLLM_HOST_IP | =本机IP | 10.254.27.226 | ✅ |
| 3 | GLOO_SOCKET_IFNAME | =网卡名 | eth0 | ✅ |
| 4 | TP_SOCKET_IFNAME | =网卡名 | eth0 | ✅ |
| 5 | HCCL_SOCKET_IFNAME | =网卡名 | eth0 | ✅ |
| 6 | LD_LIBRARY_PATH | `$LD:/usr/local/lib`（lib 在**尾**） | `/usr/local/lib:$LD`（lib 在**头**） | △ 顺序 |
| 7 | HCCL_OP_EXPANSION_MODE | AIV | AIV | ✅ |
| 8 | OMP_PROC_BIND | false | false | ✅ |
| 9 | OMP_NUM_THREADS | 1 | 1 | ✅ |
| 10 | HCCL_BUFFSIZE | **256** | **256** | ✅ |
| 11 | PYTORCH_NPU_ALLOC_CONF | expandable_segments:True | 同 | ✅ |
| 12 | ASCEND_AGGREGATE_ENABLE | 1 | 1 | ✅ |
| 13 | ASCEND_TRANSPORT_PRINT | 1 | 1 | ✅ |
| 14 | ACL_OP_INIT_MODE | 1 | 1 | ✅ |
| 15 | VLLM_NIXL_ABORT_REQUEST_TIMEOUT | 300000 | 300000 | ✅ |
| 16 | VLLM_VERSION | 0.21.0 | 0.21.0 | ✅ |
| 17 | VLLM_ASCEND_ENABLE_FLASHCOMM1 | 1 | 1 | ✅ |
| 18 | ASCEND_RT_VISIBLE_DEVICES | =$1 | 内联 `=$CARDS python3 …` | ✅ 同义 |
| ➕ | PROMETHEUS_MULTIPROC_DIR | （无） | /var/log/wings/… | ➕ 运维 |
| ➕ | PYTHONUNBUFFERED | （无） | 1 | ➕ 运维 |

**P 专属隔离核对**：含 `FLASHCOMM1 / NIXL_ABORT / BUFFSIZE=256`；不含任何 D 专属变量（无 `MLAPO / TASK_QUEUE / DYNAMIC_EPLB`）。✅

### 1.2 Decode 节点（日志 IP 10.254.198.227，pid=731）

| # | 变量 | 官方 D | 日志 D | 判定 |
|---|---|---|---|---|
| 1 | HCCL_IF_IP | =本机IP | 10.254.198.227 | ✅ |
| 2 | VLLM_HOST_IP | =本机IP | 10.254.198.227 | ✅ |
| 3 | GLOO_SOCKET_IFNAME | =网卡名 | eth0 | ✅ |
| 4 | TP_SOCKET_IFNAME | =网卡名 | eth0 | ✅ |
| 5 | HCCL_SOCKET_IFNAME | =网卡名 | eth0 | ✅ |
| 6 | LD_LIBRARY_PATH | `$LD:/usr/local/lib` | `/usr/local/lib:$LD` | △ 顺序 |
| 7 | VLLM_ASCEND_ENABLE_MLAPO | 1 | 1 | ✅ |
| 8 | HCCL_OP_EXPANSION_MODE | AIV | AIV | ✅ |
| 9 | OMP_PROC_BIND | false | false | ✅ |
| 10 | OMP_NUM_THREADS | 1 | 1 | ✅ |
| 11 | HCCL_BUFFSIZE | **500** | **500** | ✅ |
| 12 | PYTORCH_NPU_ALLOC_CONF | expandable_segments:True | 同 | ✅ |
| 13 | TASK_QUEUE_ENABLE | 1 | 1 | ✅ |
| 14 | ASCEND_AGGREGATE_ENABLE | 1 | 1 | ✅ |
| 15 | ASCEND_TRANSPORT_PRINT | 1 | 1 | ✅ |
| 16 | ACL_OP_INIT_MODE | 1 | 1 | ✅ |
| 17 | VLLM_VERSION | 0.21.0 | 0.21.0 | ✅ |
| 18 | DYNAMIC_EPLB | 1 | 1 | ✅ |
| 19 | ASCEND_RT_VISIBLE_DEVICES | =$1 | 内联 | ✅ 同义 |
| ➕ | PROMETHEUS_MULTIPROC_DIR | （无） | /var/log/wings/… | ➕ 运维 |
| ➕ | PYTHONUNBUFFERED | （无） | 1 | ➕ 运维 |

**D 专属隔离核对**：含 `MLAPO / TASK_QUEUE / DYNAMIC_EPLB / BUFFSIZE=500`；不含 P 专属变量（无 `FLASHCOMM1`，无 `NIXL_ABORT`）。✅

### 1.3 环境变量差异汇总

- **值差异：0 个。** 官方声明的每个 env，名称与取值在对应角色都完全一致。
- **顺序差异：`LD_LIBRARY_PATH`。** 官方把 `/usr/local/lib` 拼在**尾部**（优先级最低），wings 拼在**头部**（优先级最高）。仅当 `/usr/local/lib` 与 CANN 目录有**同名 .so** 时才会改变实际加载库；`/usr/local/lib` 一般放 Mooncake 传输库，置前反而更利于优先加载。**低风险，记录备查。**
- **wings 额外 2 个（运维，无害）**：`PROMETHEUS_MULTIPROC_DIR`（指标多进程目录）、`PYTHONUNBUFFERED=1`（日志不缓冲）。官方裸脚本无 sidecar 才没有，**不影响引擎/算子行为**。

### 1.4 `strip_env` 生效证据

官方不设、wings 默认会带的内部 env，在日志里**全部缺席**，证明 a2 `strip_env` 运行时真实生效：

`VLLM_USE_V1`、`VLLM_LLMDD_RPC_PORT`、`VLLM_MOONCAKE_BOOTSTRAP_PORT`、`ASCEND_CONNECT_TIMEOUT`、`ASCEND_TRANSFER_TIMEOUT`、`ASCEND_A3_ENABLE`，以及 **P 专属剔除的 `TASK_QUEUE_ENABLE`**（P 无、D 保留）。

> 硬证据（日志原文）：
> `742 - ServiceProfiler - INFO - VLLM_USE_V1 not set, auto-detected via vLLM 0.21.0+empty: default 1`
> —— 引擎确认 `VLLM_USE_V1` 未被导出（剔除成功，值仍自动判定为 1，行为不变）。

---

## 2. 引擎字段对比

### 2.1 Prefill 节点

| 字段 | 官方 P | 日志 P | 判定 |
|---|---|---|---|
| max_model_len | **115168** | **131072** | ⚠️ |
| seed | **1024** | **42** | ⚠️ |
| max_num_batched_tokens | 4096 | 4096 | ✅ |
| max_num_seqs | 64 | 64 | ✅ |
| gpu_memory_utilization | 0.95 | 0.95 | ✅ |
| trust_remote_code | 有 | True | ✅ |
| quantization | ascend | ascend | ✅ |
| enable_expert_parallel | 有 | True | ✅ |
| async_scheduling | 有 | True | ✅ |
| enable_chunked_prefill | 有 | True | ✅ |
| enable_prefix_caching | 有 | True | ✅ |
| enforce_eager | 有 | True | ✅ |
| compilation_config | 无（走 eager） | 无 | ✅ |
| enable_auto_tool_choice | 有 | True | ✅ |
| tool_call_parser | glm47 | glm47 | ✅ |
| reasoning_parser | glm45 | glm45 | ✅ |
| additional_config | 6 项（含 `enable_dsa_cp`） | **逐字一致** | ✅ |
| speculative_config | num=3 / deepseek_mtp | 同 | ✅ |
| kv-transfer（role/port/module/extra） | producer / 30000 / kv_p2p / dp4tp8·dp8tp4·use_ascend_direct | **同** | ✅ |
| kv engine_id | 0 | 3（=dp_rank） | 🔧 机制 |
| block_size | 不设 | 16 | ➕ |
| default_chat_template_kwargs | 无 | `{enable_thinking:false}` | ➕ |
| served_model_name | glm5.2 | GLM-5.1 | △ 标签 |

**P 结论**：除 `max_model_len`(131072 vs 115168) 与 `seed`(42 vs 1024)，全部一致；`additional_config` / connector / speculative / parser 逐字对齐。

### 2.2 Decode 节点

| 字段 | 官方 D | 日志 D | 判定 |
|---|---|---|---|
| max_model_len | 135168 | 135168 | ✅ |
| **max_num_batched_tokens** | **164** | **4096** | ⚠️⚠️ |
| **gpu_memory_utilization** | **0.92** | **0.95** | ⚠️ |
| **enable_chunked_prefill** | 不设（关） | **True（开）** | ⚠️ |
| seed | **1024** | **42** | ⚠️ |
| max_num_seqs | 48 | 48 | ✅ |
| trust_remote_code | 有 | True | ✅ |
| quantization | ascend | ascend | ✅ |
| enable_expert_parallel | 有 | True | ✅ |
| async_scheduling | 有 | True | ✅ |
| enable_prefix_caching | 有 | True | ✅ |
| enforce_eager | 无 | 无 | ✅ |
| compilation_config | FULL_DECODE_ONLY | FULL_DECODE_ONLY | ✅ |
| enable_auto_tool_choice | 有 | True | ✅ |
| tool_call_parser | glm47 | glm47 | ✅ |
| reasoning_parser | glm45 | glm45 | ✅ |
| additional_config | 5 项（无 `enable_dsa_cp`） | **逐字一致** | ✅ |
| speculative_config | num=3 / deepseek_mtp | 同 | ✅ |
| kv-transfer | consumer / 30100 / kv_p2p | consumer / **30101** / 同 | ✅（端口 base+偏移） |
| kv engine_id | 1 | 3（=dp_rank） | 🔧 机制 |
| block_size | 不设 | 16 | ➕ |
| default_chat_template_kwargs | 无 | `{enable_thinking:false}` | ➕ |
| served_model_name | glm5.2 | GLM-5.1 | △ 标签 |

**D 结论**：`max_model_len` 反而对上了（135168），但有 **3 处实质偏差**。

### 2.3 引擎字段差异汇总

**⚠️ Decode 三处实质偏差（最该处理）：**

| 字段 | 官方 D | 日志 D | 影响 |
|---|---|---|---|
| max_num_batched_tokens | 164 | 4096 | 差 ~25 倍。解码端被当预填充档批量跑，浪费显存、偏离官方调优 |
| enable_chunked_prefill | 关 | 开 | 解码端不应开分块预填充 |
| gpu_memory_utilization | 0.92 | 0.95 | 显存水位偏高 |

> **共性根因（已澄清）**：这三项**不是当前 `pd_config.json` 的问题**——经 dry-run 复验（见下），当前配置的 Decode 已是官方口径 `164 / 0.92 / 无 chunked-prefill`。本节日志来自一次**旧/陈旧配置的部署**，故出现 4096/0.95/on。即"漏了 decode 覆盖"的是当时那套部署，不是现行注册表。

> **✅ dry-run 复验（当前配置，含 engine_id 修复后）**：`python dry_run.py --pd glm52-a2` 重生成 + `python tests/pd_compare.py` →
> **glm52-a2 P/D 各 20 项全 OK（含 `engine_id role 级 P=0/D=1`）**。逐字段：P/D 的 `max_num_batched_tokens`(4096/164)、`gpu_memory_utilization`(0.95/0.92)、`enable_chunked_prefill`(开/关)、`seed`(1024)、`engine_id`(0/1) **全部对齐官方**；仅剩 `max_model_len`（131072/200000，用户可控刻意 delta）与无害 wings 附加项（`block_size=16`/`dtype auto`/`default_chat_template_kwargs`）。结论：**§2 的 Decode 偏差与 seed=42 仅存于那次旧部署日志，现行配置 + engine_id 修复后已逐字段对齐官方。**

**⚠️ 两端共有偏差：**
- `seed` = 42（官方 1024）。两端均 42，说明部署期存在 seed 覆盖（注册表本意为 1024）。影响采样可复现性，风险低。
- `max_model_len`：P 131072（官方 115168）；D 135168（官方一致）。

**🔧 设计内差异（external-lb 机制，非偏差）：**
- `engine_id` 用 `dp_rank`（日志为 3），官方写死 P=0 / D=1。⚠️ 注意：官方 P/D 用**不同** engine_id（0 vs 1），我们同一 rank 的 P 与 D 共用同值 —— KV 配对依赖此编号，建议真机确认 Mooncake 配对正常。
- `kv_port` 30101 = base 30100 + 多 service 偏移；`data_parallel_external_lb=True` + `rpc_port` P=12890/D=12777 + `data_parallel_size_local=1` 为 wings 拓扑表达，与官方位置参数功能等价。

**➕ wings 额外（无害/产品策略）：**
- `block_size=16`（官方不设，用引擎默认）。建议确认 GLM 在 0.21.0 下的引擎默认是否 128，若是则属一处隐性差异。
- `default_chat_template_kwargs={enable_thinking:false}`（wings 默认关思考）。
- `served_model_name=GLM-5.1`（本次跑 5.1 权重，仅名称）。

---

## 3. 总结论

| 层面 | 结论 |
|---|---|
| **环境变量** | ✅ **P/D 与官方逐条对齐**（值全同、角色隔离正确、strip_env 生效）。唯一可记录项：`LD_LIBRARY_PATH` 顺序 + 2 个运维变量，均无害。 |
| **引擎字段** | ⚠️ **Decode 三处实质偏差**：`max_num_batched_tokens`(4096→应 164)、`gpu_memory_utilization`(0.95→应 0.92)、`enable_chunked_prefill`(开→应关)；外加两端 `seed`(42→1024)、P `max_model_len`(131072→115168)。 |

**待办（如需对齐官方解码调优口径）：** 为 Decode 角色补上 `max_num_batched_tokens=164`、`gpu_memory_utilization=0.92`、`enable_chunked_prefill=false` 三项角色级覆盖；其余（seed / max_model_len）按是否要逐字对齐官方再定。

---

## 4. P 侧崩溃分析（运行期 KV 连接器 bug → DP 集合通信级联）

> **修正说明（据 P 主 Master / `EngineCore_DP0` 日志重写）**：先前基于 13:56 的 `connectFullMesh` 日志判为“**启动期**握手失败”，**该判断已被推翻**。Master 日志显示集群**成功启动并正常服务过**，是**运行期**才崩；13:56 的启动失败是这次崩溃后**重启的余震**。与 §1–§3 配置对比无关。

### 4.1 修正后的时间线

| 时刻 | 事件 | 证据（日志原文） |
|---|---|---|
| ~13:35 | P/D 启动 | 启动日志 |
| **13:54:04** | **Master 健康服务中** | `Engine 000: Avg prompt throughput 3276.9 tokens/s, Running: 1, Waiting: 6, GPU KV cache 50.0%`；`POST /v1/completions 200 OK`；Mooncake 正常 `Delaying free of 1 blocks` |
| **13:54:09** | **DP2（10.254.255.37）先崩 ← 真·第一现场** | `Worker_DP2_TP*` 在 KV 连接器 `start_load_kv` 抛 `IndexError`/`AssertionError`（**非 gloo**）→ DP2 EngineCore 死。崩前同样健康（13:53:54 仍 1638 tok/s）|
| **13:54:14**（+5s） | **其余 P DP rank 被级联拖死** | DP0/DP1/DP3 在每步 `all_reduce` 收到 DP2 已关闭的连接 → `Connection closed by peer` → `EngineDeadError` → 请求转 500。三者崩前都在健康服务（DP0 3276 / DP1 1638 tok/s）|
| 13:56:49 | 重启余震 | 另一节点重启时 `connectFullMesh` 收 0 字节（即本节旧版误判的那段） |

> **关键：集群起得来、真服务过（200 OK + 3000+ tok/s）。** 故“端口 12890 不通 / 设备初始化失败 / 启动握手失败”等假设**全部排除**——网络与设备在启动期是好的。

### 4.2 第一现场错误链（运行期，非启动期）

```
worker_busy_loop → execute_model                      (worker.py:510)
  → model_runner.execute_model
    → _determine_batch_execution_and_padding           (model_runner_v1.py:2060 / 2907)
      → _sync_metadata_across_dp                        (model_runner_v1.py:658)
        → dist.all_reduce(packed_tensor, group=group)   ← 每步跨 DP 的元数据 all_reduce（gloo CPU 组）
          → work.wait()
            → RuntimeError: [gloo/.../tcp/pair.cc:547] Connection closed by peer [10.254.200.100]:59538
```

是 `execute_model` 里的**每步 DP 同步**（运行期），不是旧版写的 `init_world_group`（启动期）。

### 4.3 真正的耦合机制：external-lb 并不解耦 DP

`_sync_metadata_across_dp` **每个 step 都对 4 个 P 的 DP rank 做一次 gloo `all_reduce`**（同步 `num_tokens_across_dp` / cudagraph 模式，使各 DP rank 对齐批次与图模式；EP 全对全也要求 DP 锁步）。

> ⚠️ **重要架构事实**：即便加了 `--data-parallel-external-lb`，4 个 P 的 DP 副本**也不是相互独立的**——它们每步在 gloo 上锁步同步。**任一 DP rank 崩溃 → 该 all_reduce 无法完成 → 全部存活 rank 一起报错 → 整个 P 侧塌掉**（无故障隔离）。即 P 可用性 = 4 个 DP rank 的逻辑“与”。

### 4.4 “closed by peer” 是环邻居假象，不能据此定凶

补入 **10.254.200.100（DP1）** 日志后，上一版“200.100 先崩”的判断**也被推翻**：**DP1 自己同样是受害者**——崩前一直健康服务（`Engine 001: 1638 tok/s, 200 OK`，KV 17→20%），13:54:14 才在同一个 `all_reduce` 崩，且它报的对端是**另一个** IP `10.254.255.37`。

两台对照暴露关键规律——**每个 rank 报的 `Connection closed by peer` 是它的 gloo 环（ring）下一跳邻居，不是真凶**：

- DP0 报 → DP1（10.254.200.100）
- DP1 报 → DP2（10.254.255.37）

> **逻辑证明**：DP0 指认 DP1，但我们手上有 DP1 的日志、它崩前一直健康——故“被指认的对端”≠ 真凶。`Connection closed by peer` 只表示“我环上的下一跳 socket 断了”；某 rank 死亡会沿环向后级联，人人都怪自己的邻居。

**P 的 4 个 DP rank ↔ IP（按 rank 推定）：**

| DP rank | IP | 本次状态 | 依据 |
|---|---|---|---|
| DP0 / Master | 10.254.83.167 | 受害者 | 健康 3276 tok/s；崩时指认 DP1 |
| DP1 | 10.254.200.100 | 受害者 | 健康 1638 tok/s；崩时指认 DP2 |
| **DP2** | **10.254.255.37** | **✅ 第一死者（已确认）** | 13:54:09 先崩、早 5 秒、且是**非 gloo 的真错**（KV 连接器）；详见 §4.6 |
| DP3 | 10.254.27.226 | 受害者 | 即 §4.1 中 13:56 重启时报 connectFullMesh 的节点（前文 `data_parallel_rank=3`） |

→ 按环级联回溯（DP2 死 → DP1→DP0→DP3 收不到→崩）。**DP2 日志已坐实这一点**：它 13:54:09 先崩、早 5 秒、且是**非 gloo 的真错**（§4.6）。

### 4.5 排除项（噪音，非根因）

| 日志行 | 性质 |
|---|---|
| `AscendConfig.enable_flashcomm1 falls back to … VLLM_ASCEND_ENABLE_FLASHCOMM1 … will be removed` | 弃用**警告**，来自 13:56 重启那次启动，非崩溃原因 |
| `DeepseekV4ForCausalLM is already registered … will be overwritten` | vllm-ascend 插件正常噪音 |
| `Dynamic EPLB is False` | P 本就不设 `DYNAMIC_EPLB`（D 专属），符合预期 |
| `VLLM_USE_V1 not set, auto-detected` | `strip_env` 的预期结果，正常 |

### 4.6 根因（已定位到 KV 连接器）

DP2 的首错**不是** gloo，而是 Mooncake KV 连接器在 `start_load_kv` 里抛的应用层异常（同一现场并发出现两种）：

```
execute_model → kv_connector_no_forward → _get_kv_connector_output
  → MooncakeConnector.start_load_kv               (mooncake_connector.py:1333)
    → _get_kv_split_metadata                       (:2368 → :1964)
      → _get_remote_rank                           (:2452)
          return self._get_remote_ranks_for_req(req_id, prefill_tp_size)[self.tp_rank]
          → IndexError: list index out of range          ← 返回的 list 长度 < self.tp_rank
    或  start_load_kv (:2378)  assert self.kv_recv_thread is not None  → AssertionError
```

本质是**远端 rank 解析越界**：`_get_remote_ranks_for_req(...)` 返回的列表比 `self.tp_rank` 短，按 `[self.tp_rank]` 取下标即 IndexError；另一些 worker 因 `kv_recv_thread is None` 触发断言。两者都指向**连接器在本 PD 拓扑下把 P↔D rank 配对算错了**。

**关键判别证据：低并发不报、高并发才报。** 这基本排除静态配置错——静态错（engine_id/拓扑）会在**第一个请求**就崩。故触发因素是**负载相关的竞态 / 每请求元数据缺失**。据此初判负载触发；但**用户随后实测：严格按官方 `engine_id=0/1` 部署即可正常拉起、且不再崩溃** → 重排如下：

| 机制 | 评级（实测后） | 说明 |
|---|---|---|
| **A engine_id per-rank（应 role 0/1）** | ✅ **确诊真因（已修复）** | 静态冲突（P-rank-k 与 D-rank-k 撞号），但**高并发把请求路由到撞号对才暴露**，故"低并发不报"。改 role 0/1 后不崩——见 §4.9 |
| B 非对称 TP（tp8↔tp4）映射 | 次要 / 可能放大 | 每请求经 `kv_transfer_params` 传 `prefill_tp_size`（#5822）|
| C 负载触发（prefix-cache load / recompute / MTP 竞态）| 次要 / 纵深排查 | A 修复后未再复现；保留作排查 |

> **判别修正**：上一版据"低并发不报"把 A 降级，是把"静态配置"误等同于"必在首请求崩"。实际是**静态冲突 + 高并发路由才命中**，两者不矛盾。这是本次调查的第 4 次修正（前三次见 §4.8）。

**C 的三条路径（A 修复后若仍偶发，按此纵深排查），都踩在我们开着的开关上：**
1. **prefix caching 跨实例 load**：a2 overlay 把两端 `enable_prefix_caching` 开了（GLM-5.2 口径，基条目本为 false）。高并发前缀命中 → 触发 `start_load_kv` 跨 rank 拉缓存 KV → 撞越界。
2. **抢占 / recompute**：`recompute_scheduler_enable=true`，dump 里正是 `RecomputeSchedulerOutput`。高并发抢占后重算重入 KV load，远端元数据未复原 → 越界。
3. **MTP + 高并发**：`deepseek_mtp num=3`；upstream #7489 已知「MTP + KV + 高并发」脆弱组合（症状是另一个 ValueError，但同区）。

> 关联 upstream（均与本场景开关/平台重叠）：#5822（`prefill_tp_size` 经请求元数据传递）、#6498（**A2** 上 `head_or_tp_rank` 初始化 bug）、#7489（MTP+高并发+mooncake）、#2970（`use_ascend_direct` 传输失败）。
> 负载特征供复现：崩点在重负载 chunked-prefill（DP0 `8192+4096`、DP1 `12288+4096`，DP1 侧 `prefix_cache queries=16384` 一条 16384-token 请求）。

### 4.7 小结（根因链已闭合）

```
DP2(10.254.255.37) start_load_kv 抛 IndexError/AssertionError       @13:54:09
   └─ KV 连接器远端 rank 解析越界（疑因 P tp8↔D tp4 非对称 / engine_id=dp_rank 配对口径）
      └─ DP2 EngineCore 死 → gloo sockets 关
         └─ @13:54:14(+5s) DP1/DP0/DP3 每步 all_reduce 撞到关闭连接 → "Connection closed by peer"
            └─ 整组 P EngineDeadError → 500 → 重启 → 13:56 connectFullMesh 余震
```

**根因不是硬件 / OOM / 网络 / 启动，而是 Mooncake KV 连接器在本 PD 拓扑下的 rank 解析 bug**（§4.6）；所有 `Connection closed by peer` 都是 gloo 环邻居假象。两个相互独立的待办：

- **修复/规避连接器越界**：本次为**高并发触发**（§4.6），首试**关 prefix caching + 降并发**隔离 C1，再按 §4.9 缓解表逐项；engine_id 口径（§4.9）虽非本次触发，但应一并对回官方。
- **架构韧性**（§4.3）：external-lb 不隔离 DP 故障，单 rank 崩即拖垮整组 P——即使修了本 bug，也应评估 DP 故障隔离 / 重调度。

### 4.8 调查演进（三次判断纠正）

根因是分四份日志逐步逼近的，过程中纠正了三次判断，留作复盘：

| 拿到的证据 | 当时判断 | 被下一份推翻为 |
|---|---|---|
| 13:56 某节点 `connectFullMesh` 失败 | P **启动期** gloo 握手失败 | 实为崩溃后**重启的余震**；集群其实成功服务过 |
| Master(DP0) 13:54:14 `all_reduce` 撞 `closed by peer [200.100]` | **200.100 先崩**、Master 受害 | 200.100(DP1) 自己也健康服务到 13:54:14，**也是受害者** |
| DP1：`closed by peer [255.37]` | 被指认的 IP = 真凶 | IP 只是 **gloo 环邻居**；真凶是唯一无“健康/受害”日志的 DP2 |
| DP2(255.37)：13:54:09 `IndexError`（早 5s、非 gloo） | — | **确诊**：KV 连接器 rank 解析越界 |

**四节点证据汇总：**

| rank | IP | 角色 | 崩溃时刻 | 首错 |
|---|---|---|---|---|
| DP0 / Master | 10.254.83.167 | 受害者 | 13:54:14 | gloo `closed by peer [200.100]`（崩前 3276 tok/s）|
| DP1 | 10.254.200.100 | 受害者 | 13:54:14 | gloo `closed by peer [255.37]`（崩前 1638 tok/s）|
| **DP2** | **10.254.255.37** | **第一死者** | **13:54:09** | **`IndexError` @ `_get_remote_rank`（Mooncake 连接器）** |
| DP3 | 10.254.27.226 | 受害者 | 13:54:14（推定）→ 13:56 | gloo；13:56 重启 `connectFullMesh` 余震为直接证据 |

> **复盘教训**：分布式 gloo 崩溃里，`Connection closed by peer` 的 IP **不能当真凶**（它是 ring 下一跳邻居）。定位第一死者要靠**最早时间戳 + 唯一的非 gloo 真错**，而非被指认的 IP——本例靠这两点把矛头从 200.100 校正到 255.37。

### 4.9 engine_id 设计逻辑（全网核实）+ 可验证缓解项

**两种 Mooncake 连接器的 engine_id 约定相反**，这解释了 wings 为何没对齐官方：

| 连接器 | engine_id 约定 | 出处 |
|---|---|---|
| `MooncakeConnectorV1`（旧 / 手工 GLM-5.1）| **每节点唯一** | 多节点 Mooncake 教程 |
| `MooncakeConnector`（kv_p2p / 官方 GLM-5.2）| **role 级：P=0 / D=1** | kv_p2p 部署指南 + GLM-5.2 样例 |

- kv_p2p 里 engine_id 是**角色标签**，物理唯一性来自 **IP:kv_port**；源码有 `Conflict engine id … with local engine id` 校验（只需 local≠remote）。所以全 P=0、全 D=1 不冲突、可扩展到 NP+MD——**官方"简单样例"的 0/1 就是完整约定，不是简化**。
- wings 曾在 [config_loader.py:1064](wings_control/core/config_loader.py#L1064) 把三种连接器一并套 V1 的 per-rank 规则 → 对 kv_p2p 错：P 得 `{0,1,2,3}`、D 得 `{0..7}`，且 P-rank-k 与 D-rank-k **engine_id 撞号**。
- ✅ **实测 + 已修复**：严格按官方 `engine_id=0/1` 部署可正常拉起且不崩 → engine_id 冲突即本次崩溃**真因**（§4.6 的 A 已升回确诊）。修复：[config_loader.py:1064](wings_control/core/config_loader.py#L1064) 现**按连接器区分**——kv_p2p `MooncakeConnector`→role 0/1（P=0/D=1），`MooncakeConnectorV1`/`MooncakeHybridConnector`→per-rank 不变；`tests/pd_external_lb_verify.py` 加断言、`dry_run.py --pd glm52-a2/nosig` 重生成确认 P=`engine_id:0`/D=`1`。

**可验证缓解项（低成本，逐个隔离 §4.6 的 C）：**

| 试验 | 目的 | 代价 |
|---|---|---|
| 关 `enable_prefix_caching`（回基条目 false）| 隔离 C1（跨实例前缀 load）| 失部分缓存命中 |
| 关 MTP（`speculative_config`）| 隔离 C3 / #7489 | 失投机加速 |
| 关 `recompute_scheduler_enable` | 隔离 C2（抢占重算）| 调度退化 |
| 设 `PD_DISABLE_ASCEND_DIRECT`（已编码于 :1045）| 排查 #2970 ADXL 直传 | 直传退化 |
| 降 `max_num_seqs` / 客户端限流 | 确认并发阈值 | 吞吐降 |
| ✅ **已做** engine_id 按连接器区分（:1064）| 消除真因 A | 已实施 + 测试 + 重生成（P=0/D=1）|

> **建议**：A 修复（engine_id role 0/1）已落地，应优先部署验证；若仍偶发，再按上表关 prefix caching + 降并发做纵深排查（C）。

**Sources：** [kv_p2p 部署指南](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md) ｜ [多节点 Mooncake 教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/multi_node_pd_disaggregation_mooncake.html) ｜ [PR #5822](https://github.com/vllm-project/vllm-ascend/pull/5822) ｜ [PR #6498](https://github.com/vllm-project/vllm-ascend/pull/6498) ｜ [Issue #7489](https://github.com/vllm-project/vllm-ascend/issues/7489) ｜ [Issue #2970](https://github.com/vllm-project/vllm-ascend/issues/2970) ｜ [mooncake_connector.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)
