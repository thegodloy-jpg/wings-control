# PD 分离 A3 部署与验证操作指南（GLM-5.1 / DeepSeek-V4-Flash）

| 项 | 内容 |
|----|------|
| 类型 | **可操作的部署 + 验证 runbook**（真实场景） |
| 适用 | GLM-5.1（`GlmMoeDsaForCausalLM`）、DeepSeek-V4-Flash（`DeepseekV4ForCausalLM`）；**A3（910C，16 卡/节点）** |
| 官方基准 | [GLM5](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.html) · [DeepSeek-V4-Flash](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html)（A3 1P1D） |
| 字段级证据 | 逐字段对齐见 [pd-a3-official-alignment-report.md](pd-a3-official-alignment-report.md)；机制见 [deepseek-v32-pd-disaggregation.md](../../xuqiu/deepseek-v32-pd-disaggregation.md) |
| 约定 | `<X>` 为占位需替换；命令示例用 `kubectl`/`curl`，按实际编排替换 |

---

## 0. 前置条件（Checklist，缺一不可）

- [ ] **硬件**：A3（910C）节点，每节点 16 NPU；P/D 节点间 RDMA 可达（Mooncake KV 传输）。
- [ ] **网卡**：P/D 全部节点 `nic_name` 一致且互通（否则 KV 跨节点连不通，见 §7-B2）。
- [ ] **镜像**：engine 镜像（预装 CANN + vllm-ascend，含 Mooncake 连接器）；wings-control sidecar 镜像。
- [ ] **权重**：ascend 量化权重（GLM-5.1-w8a8 / DeepSeek-V4-Flash-w8a8-mtp），`config.json` 的 `architectures` 必须为上表架构名（决定命中 PD 注册表条目）。
- [ ] **驱动挂载**：宿主 `/usr/local/Ascend/driver` 挂入容器（缺失启动脚本会 FATAL 退出）。
- [ ] **共享卷**：`SHARED_VOLUME_PATH` 挂载（log_analyzer / 产物）。
- [ ] **上层 layerwise proxy**：负责把请求路由到 P/D（wings 不做负载均衡）。

---

## 1. 拓扑规划（A3）

| 模型 | 角色 | 并行 | 卡/节点 | 节点数 | service 数 | pod 数 | `DP_SIZE_LOCAL` |
|------|------|------|:------:|:------:|:----------:|:------:|:----:|
| **GLM-5.1** | Prefill | DP2×TP16 | 16 | 2 | 2（每节点 1）| 2 | 1 |
| | Decode | DP16×TP4 | 16 | 4 | 16（每节点 4）| 4 | 4 |
| **V4-Flash** | Prefill | DP4×TP4 | 16 | 1 | 4 | 1 | 4 |
| | Decode | DP16×TP1 | 16 | 1 | 16 | 1 | 16 |

> GLM-5.1 合计 **6 pod / 96 卡**；V4-Flash 合计 **2 pod / 32 卡**。

**派生公式（上层算好后下发，wings 不自算）**：
```
DP_SIZE        = 角色节点数 × 16 ÷ TP_SIZE
DP_SIZE_LOCAL  = 16 ÷ TP_SIZE                  # 单节点 fork 几个 service
dp_rank_start  = 角色内节点序 × DP_SIZE_LOCAL   # wings 由 RANK_IP（HOST_IP 缺省时回退它）在 NODE_IPS 的位置自动派生
dp_address     = 角色域 node0 IP（= Master_IP）
```

---

## 2. 下发参数（每 Pod 的环境变量契约）——【操作核心】

### 2.1 必填环境变量（12 个，无默认 / 默认不对 / 无 CLI 等价）

| 组 | 变量 | 取值规则 | 来源 |
|----|------|----------|:--:|
| 平台 | `WINGS_DEVICE` | `ascend`（缺省默认 `nvidia` → **必填**） | 平台/镜像 |
| 平台 | A3 平台信号（**任一即可**） | ⚠️ **全无信号则回退 `a2`（非 a3！）**。A3 须给一个：`WINGS_ASCEND_PLATFORM=a3` / `ASCEND_PLATFORM` / `ENGINE_IMAGE_FLAVOR` / **`ENGINE_VERSION` 带 `-a3` 后缀**（如 `0.13.0rc3-a3`，a3 镜像通常自带 → 常可省显式设置）/ `ASCEND_A3_ENABLE=1` / `hardware_info.json` 含 910c / `WINGS_DEVICE_NAME` 含 a3 | 平台/镜像 |
| 平台 | `DEVICE_COUNT` | `16`（整 pod 卡数；缺省默认 1） | 平台 |
| ① 本机IP | `RANK_IP` | **本 pod 唯一 IP**（上层 MaaS 下发）；`get_local_ip()` 读它，`POD_IP`(→HCCL_IF_IP)、`HOST_IP`(→rank_start) 均回退到它 → **只设 RANK_IP，勿重复设 POD_IP/HOST_IP**。**须精确匹配 NODE_IPS 中一项** | 上层 |
| PD 契约 | `PD_ROLE` | `P` 或 `D`（触发 PD） | 上层 |
| PD 契约 | `DP_SIZE_LOCAL` | 本节点 fork 数（=卡/节点÷tp，**不可派生**） | 上层 |
| PD 契约 | `Master_IP` | 本角色 node0 IP（= dp-address） | 上层 |
| PD 契约 | `NODE_IPS` | 本角色全部节点 IP（逗号分隔；本 pod 的 `RANK_IP` 须在其中） | 上层 |
| KV 全局拓扑 | `PD_PREFILL_DP_SIZE` / `PD_PREFILL_TP_SIZE` | 见 §2.2 / §2.3；**P/D 互相感知，且本角色 dp/tp 由此派生** | 上层 |
| KV 全局拓扑 | `PD_DECODE_DP_SIZE` / `PD_DECODE_TP_SIZE` | 同上 | 上层 |

### 2.1.1 派生 / 可选环境变量（可省略）

| 变量 | 缺省时取 | 说明 |
|------|---------|------|
| `DP_SIZE` / `TP_SIZE` | **派生自本角色 `PD_{ROLE}_*`**（P→`PD_PREFILL_*`，D→`PD_DECODE_*`） | 4 个全局拓扑变量即单一真相源，本角色 dp/tp 无需重复下发；`DP_SIZE_LOCAL` 例外（不可派生，见 §2.1） |
| `VLLM_LLMDD_RPC_PORT` | **P=`12890` / D=`12777`**（按角色，fork 脚本兜底） | 自定义才设；§2.2/§2.3 示例用 10521/10523 |
| `SHARED_VOLUME_PATH` | `/shared-volume` | 仅 LMCache 用，**不进 PD 命令**；探针证删除后命令字节级不变 |

**模型/引擎入参（CLI 或同名 env 兜底，二选一）**：`--model-name <name>` `--model-path <weights>` `--engine vllm_ascend`（`--device-count`/`--nnodes`/`--node-rank` 可省，默认从 `DEVICE_COUNT`/`NNODES`/`0` 取；其同名 env 属冗余，见 [对齐报告 §0.3](pd-a3-official-alignment-report.md)）。
> ⚠️ **`RANK_IP` 必须与 `NODE_IPS` 中的某一项逐字相同**，否则 `dp_rank_start` 回退 0 → 多节点 rank 撞车（见 §7-限制 L10）。

### 2.2 GLM-5.1 逐 Pod 下发表（6 pod）

公共：`PD_PREFILL_DP_SIZE=2 PD_PREFILL_TP_SIZE=16 PD_DECODE_DP_SIZE=16 PD_DECODE_TP_SIZE=4`，`WINGS_ASCEND_PLATFORM=a3 WINGS_DEVICE=ascend DEVICE_COUNT=16`。
> 下表 `DP_SIZE`/`TP_SIZE` 列为**派生值**（=本角色 `PD_{ROLE}_*`，见公共行），**不单独下发**；每 pod 实际下发 = `PD_ROLE`/`DP_SIZE_LOCAL`/`Master_IP`/`NODE_IPS`/`RANK_IP` + 公共行。

| Pod | `PD_ROLE` | `DP_SIZE` | `TP_SIZE` | `DP_SIZE_LOCAL` | `Master_IP` | `VLLM_LLMDD_RPC_PORT` | `NODE_IPS` | `RANK_IP` | →rank_start | pod 内 service |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| P-0 | P | 2 | 16 | 1 | `<P0>` | 10521 | `<P0>,<P1>` | `<P0>` | 0 | rank0 / :18000 / 卡0-15 / kv30000 |
| P-1 | P | 2 | 16 | 1 | `<P0>` | 10521 | `<P0>,<P1>` | `<P1>` | 1 | rank1 / :18000 / 卡0-15 / kv30000 |
| D-0 | D | 16 | 4 | 4 | `<D0>` | 10523 | `<D0>,<D1>,<D2>,<D3>` | `<D0>` | 0 | rank0-3 / :18000-3 / 卡0-3·4-7·8-11·12-15 / kv30100-3 |
| D-1 | D | 16 | 4 | 4 | `<D0>` | 10523 | `<D0>,<D1>,<D2>,<D3>` | `<D1>` | 4 | rank4-7 / :18000-3 |
| D-2 | D | 16 | 4 | 4 | `<D0>` | 10523 | `<D0>,<D1>,<D2>,<D3>` | `<D2>` | 8 | rank8-11 / :18000-3 |
| D-3 | D | 16 | 4 | 4 | `<D0>` | 10523 | `<D0>,<D1>,<D2>,<D3>` | `<D3>` | 12 | rank12-15 / :18000-3 |

### 2.3 DeepSeek-V4-Flash 逐 Pod 下发表（2 pod）

公共：`PD_PREFILL_DP_SIZE=4 PD_PREFILL_TP_SIZE=4 PD_DECODE_DP_SIZE=16 PD_DECODE_TP_SIZE=1`，`WINGS_ASCEND_PLATFORM=a3 WINGS_DEVICE=ascend DEVICE_COUNT=16`。
> 下表 `DP_SIZE`/`TP_SIZE` 列为**派生值**（=本角色 `PD_{ROLE}_*`），**不单独下发**。

| Pod | `PD_ROLE` | `DP_SIZE` | `TP_SIZE` | `DP_SIZE_LOCAL` | `Master_IP` | `VLLM_LLMDD_RPC_PORT` | `NODE_IPS` | `RANK_IP` | →rank_start | pod 内 service |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| P | P | 4 | 4 | 4 | `<P0>` | 10521 | `<P0>` | `<P0>` | 0 | rank0-3 / :18000-3 / 卡0-3·4-7·8-11·12-15 / kv30000-3 |
| D | D | 16 | 1 | 16 | `<D0>` | 10523 | `<D0>` | `<D0>` | 0 | rank0-15 / :18000-15 / 每 service 1 卡 / kv30100-15 |

> 平台另需给每 pod 注入整 pod 卡 `ASCEND_RT_VISIBLE_DEVICES`（0-15）；wings fork 时按 `i*tp` 再切给每个 service。

---

## 3. 部署步骤

1. **备好权重/镜像/网络/驱动挂载**（§0 checklist 全绿）。
2. **按 §2 给每个 pod 注入 env**（K8s `env:` / Downward API 把本 pod IP 注入 `RANK_IP`），并设 `MODEL_NAME`/`MODEL_PATH`（或等价 CLI）、`ENGINE=vllm_ascend`。
3. **启动各 pod 的 wings-control**：wings 识别 `PD_ROLE`+`DP_SIZE>1` → 进 external-lb 分支 → pod 内 fork `DP_SIZE_LOCAL` 个 `vllm serve`（每 service 独立 rank/port/卡组/kv_port）。
4. **起上层 layerwise proxy**，按 rank 路由（prefill→P 集群，decode→D 集群）。
5. **顺序建议**：先全部 P/D pod 起到健康，再接 proxy 引流。

---

## 4. 启动前预检（dry-run，强烈建议）——【上线前必做】

**目的**：在不占真卡的前提下，离线生成将要执行的 `vllm serve` 命令，逐字对照官方 A3（§6）。

**用真实拓扑预览**（dry_run 的 PD 场景在 [dry_run.py:331](../../dry_run.py) `PD_SCENARIOS`）：
- 内置 `glm5` / `v4flash` 已是 A3 官方拓扑，IP 为占位；**把 `nodes`/`rpc` 改成你的真实 IP/端口**即可：
```bash
# 1) 编辑 dry_run.py PD_SCENARIOS 里对应模型的 prefill/decode 的 nodes、rpc
# 2) 生成
python dry_run.py --pd glm5      # 或 --pd v4flash
# 3) 查看产物（每角色 node0；D 多节点再出 node1 展示 rank 派生）
ls build/output/start_command_pd-*-{P,D}_node*.sh
```
- **校验点**：日志出现 `[PD external-lb] arch=… role=… connector=… dp_size=… local=… rank_start=… addr=…`，且与 §1 规划一致；脚本里 `for i in $(seq 0 N-1)`（N=`DP_SIZE_LOCAL`）、`KVPORT=$((<base>+i))`、卡组 `i*tp..` 正确。

> 输入清单（12 必填 env + 派生/可选 + CLI）详见 [pd-a3-official-alignment-report.md](pd-a3-official-alignment-report.md) §0。

---

## 5. 启动后验证（逐层）

### 5.1 进程/端口（每 pod）
```bash
# pod 内应有 DP_SIZE_LOCAL 个 vllm 进程，端口 18000..18000+local-1
kubectl exec <pod> -- bash -lc 'pgrep -af "vllm.*api_server|vllm serve" | wc -l'   # 期望 = DP_SIZE_LOCAL
kubectl exec <pod> -- bash -lc 'for p in $(seq 18000 $((18000+LOCAL-1))); do curl -sf localhost:$p/health && echo " :$p OK"; done'
```

### 5.2 角色内 DP 域组建（同角色跨 pod）
```bash
# 同角色各 service 经相同 dp-address(Master_IP)+rpc-port 握手组 DP/EP 域
kubectl logs <pod> | grep -iE "data.?parallel|DP rank|coordinator|EngineCore"
# 确认 rank 连续：P 集群 rank 0..DP-1；D 集群 rank 0..DP-1（跨 pod 由 rank_start 拼接）
```

### 5.3 KV 传输通路（P→D，Mooncake）
```bash
# P 为 producer(kv_port 30000+)，D 为 consumer(30100+)；确认 KV 建连无 timeout
kubectl logs <D-pod> | grep -iE "mooncake|kv_?transfer|kv_?connector|bootstrap"
# 关注：无 "connection refused"/"timeout"；producer/consumer 端口与 §2 一致
```

### 5.4 集群就绪 + 冒烟推理
```bash
# 经上层 proxy 发一条短请求（或直连一个 decode service 做连通性冒烟）
curl -s http://<proxy>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-model-name>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

### 5.5 与官方基线对齐
- 小并发跑通后，按官方 A3 文档的输入/输出长度做基准，比对吞吐/TPOT 量级（具体数值以官方文档为准）。

---

## 6. 对齐官方 A3 核对清单（运行命令 / dry-run 产物逐项核对）

**GLM-5.1**：

| 字段 | P 期望 | D 期望 |
|------|--------|--------|
| tp / dp | 16 / 2 | 4 / 16 |
| max-model-len | **131072** | **200000** |
| max-num-batched-tokens / seqs | 4096 / 64 | 32 / 8 |
| gpu-memory-utilization | 0.95 | 0.92 |
| enforce-eager / chunked-prefill | 有 / 有 | 无 / 无 |
| enable-prefix-caching | **无** | **无** |
| compilation-config | **无** | `FULL_DECODE_ONLY + capture[4,8,12,16,20,24,28,32]` |
| tool-call-parser / reasoning-parser | glm47 / glm45 | glm47 / glm45 |
| speculative-config | `{3,deepseek_mtp}` | 同 |
| kv-transfer | V1 / producer / **30000** / `use_ascend_direct` | V1 / consumer / **30100** |
| 共用 env | `HCCL_BUFFSIZE=256`+`ASCEND_AGGREGATE_ENABLE`+`ACL_OP_INIT_MODE`+`ASCEND_A3_ENABLE`+`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480` | 同 |
| 角色 env | `FLASHCOMM1`+`FUSED_MC2` | `MLAPO`+`TASK_QUEUE`+`FUSED_MC2` |

**DeepSeek-V4-Flash**：

| 字段 | P 期望 | D 期望 |
|------|--------|--------|
| tp / dp | 4 / 4 | 1 / 16 |
| max-num-batched-tokens / seqs | **8192 / 16** | **120 / 60** |
| gpu-memory-utilization / max-model-len / seed | 0.9 / 1048576 / 1024 | 0.9 / 1048576 / 1024 |
| enforce-eager / async-scheduling | 有 / 无 | 无 / 有 |
| compilation-config | **无** | `FULL_DECODE_ONLY` |
| no-enable-prefix-caching / no-disable-hybrid-kv-cache-manager | 有 / 有 | 有 / 有 |
| tokenizer/tool/reasoning-parser | deepseek_v4 ×3 | deepseek_v4 ×3 |
| model-loader-extra-config | `{multithread,128}` | 同 |
| speculative-config | `{1,mtp,enforce_eager}` | 同 |
| additional-config | `{cpu_binding,shared_expert_dp,enable_dsa_cp}` | `{ascend_compilation{npugraph,static_kernel:false},cpu_binding,multistream_overlap_shared_expert:true,recompute}` |
| kv-transfer | Hybrid / producer / **30000** | Hybrid / consumer / **30100** |

> 快速核对：`kubectl exec <pod> -- bash -lc 'pgrep -af "vllm serve" | head -1'`，或对 dry-run 产物 `grep -oE "\-\-[a-z0-9-]+( '\''[^'\'']*'\''| [^ -][^ ]*)?"`。

---

## 7. 故障排查（含本项目已知限制）

| 现象 | 排查 | 处置 |
|------|------|------|
| 多节点 rank 撞车 / DP 域组不起来 | `RANK_IP` 是否逐字在 `NODE_IPS` 内（L10） | 修正 `RANK_IP`/`NODE_IPS` 文本一致 |
| KV 跨节点连不通 | P/D 网卡名/网段是否一致（B2） | 统一 `nic_name`，确认 RDMA 可达 |
| 同 pod 多 service KV 冲突 | `engine_id` 是否按 rank 唯一 | wings 已按 `dp_rank` 注入；**Hybrid 连接器官方示例为固定 0/1，需真机确认**（[待确认项](#8-待确认项)） |
| HCCL 建连/执行误判失败 | P/D 超时量级是否一致（B4） | 对齐 `HCCL_EXEC/CONNECT_TIMEOUT` |
| decode 输出错乱但不报错 | `PD_PREFILL_*`/`PD_DECODE_*` 对端拓扑是否下发且两边一致（L9，缺失只告警） | 补齐对端拓扑 env |
| 任一 service 崩 → 整 pod 退出 | EP all-to-all 语义，wings 设计为整 pod 重启（L12） | 编排层做整组重启；查崩溃 service 日志 |
| 需要 IndexCache（`--enable-sparse`） | **PD external-lb 下当前不下发**（L1，暂未实现） | 暂不可用，见 [fix-plan](pd-scheme-fix-plan.md) |
| 跑在 A2 而非 A3 | 注册表当前条目对齐 A3；A2 经 `platform_overrides.a2` overlay | 确认 `WINGS_ASCEND_PLATFORM`；A2 值需真机核（L4 `_confirm`） |

> 完整限制清单：[pd-scheme-limitations.md](pd-scheme-limitations.md)；脚本自查表：设计文档[附录 B](../../xuqiu/deepseek-v32-pd-disaggregation.md)。

---

## 8. 待确认项（上线真机时重点验证）

| 项 | 说明 |
|----|------|
| `engine_id`（Hybrid） | wings 对 V1/Hybrid 一律按 `dp_rank` 注入；官方 V4-Flash(Hybrid) 示例固定 `0/1` —— 多 service 下按 rank 更合理，但需真机确认 Mooncake Hybrid 期望。 |
| `FULL_DECODE_ONLY` 全图 decode | 历史曾触发 MTE 越界（GLM5 aclgraph）；D 角色首跑重点观察是否崩溃，必要时临时 `enforce_eager` 验证。 |
| A2 平台值 | `DeepseekV4ForCausalLM.platform_overrides.a2` 仅 overlay 高置信 batched/seqs，其余继承 A3；A2 上线前逐项核官方。 |

---

## 9. 回滚 / 降级

- **降级到 standalone（非 external-lb）**：去掉 `DP_SIZE`（或设 `DP_SIZE=1`）→ wings 走原 1P1D standalone 路径（字节级与改造前一致）。
- **整组重启**：任一 service 异常退出，子 shell `exit 1` → 编排层 crash-retry 整 pod（EP 语义要求整域同时在位）。
- **观测**：`/shared-volume/progress.jsonl` 有阶段/失败记录；`/var/log/wings/engine.log` 为引擎日志。

---

## 附：一句话 SOP

> 按 §1 规划拓扑 → §2 逐 pod 注入 12 个必填 env（本机 IP 只设 `RANK_IP`；`DP_SIZE`/`TP_SIZE` 由 4 个全局拓扑派生；rpc/共享卷/模型名路径引擎按需）→ §4 dry-run 预检对照官方 → 部署 → §5 逐层验证（端口/DP 域/KV/冒烟）→ §6 清单核对 → 异常查 §7。
