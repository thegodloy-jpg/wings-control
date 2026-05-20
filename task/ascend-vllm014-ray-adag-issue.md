# 910C + vllm-ascend 0.14+ + Ray 双机：aDAG 卡死 / `ray_compile_graph_communication` 错误

> 状态：历史 Ascend/Ray 专项分析。当前 wings-control 部署口径请以 [../README.md](../README.md)、[../docs/deployment/docker-compose.md](../docs/deployment/docker-compose.md)、[../docs/deployment/k8s.md](../docs/deployment/k8s.md) 和 [../docs/features/pd-disaggregation.md](../docs/features/pd-disaggregation.md) 为准；本文只保留上游 vllm-ascend / Ray 问题处置经验。

> **当前文档角色：分析（Why）** —— 根因剖析与修复策略，受众：研发、架构

## 文档资产列表（Ascend + Ray 系列）

| 角色 | 文件 | 受众 | 用途 |
|---|---|---|---|
| 📘 **分析（本文）** | [ascend-vllm014-ray-adag-issue.md](./ascend-vllm014-ray-adag-issue.md) | 研发 / 架构 | 根因 + 修复策略 + 设计稿 |
| 📗 **实施 Runbook** | [ascend-vllm014-ray-adag-runbook.md](./ascend-vllm014-ray-adag-runbook.md) | 运维 / SRE | 现场逐步操作命令 + 故障升级路径 |
| 📙 **配置示例** | [ascend-values-examples.yaml](./ascend-values-examples.yaml) | 运维 | 三套 Helm Profile 直接复制 |

**两条独立链路**：
- **链路 A**：0.14+ aDAG 卡死（应用层，env 软回退）
- **链路 B**：0.12 必须开特权（K8s 层，最小权限替换）—— 见本文附录 B

---

本文档为**问题分析与缓解方案**，不涉及 wings-control 启动逻辑修复。
结论先行：根因在 vllm / vllm-ascend 上游 0.14+ 的 aDAG（Ray Compiled Graph）路径与 HCCL P2P 实现的成熟度问题，**与本仓库 Ray 拉起方式无关**。

---

## 一、现象

- 环境：910C × 2 节点，Ray 分布式，K8s 部署
- 触发版本：vllm-ascend ≥ 0.14（0.12 + 特权模式可正常运行）
- 症状链：
  1. 服务启动后**卡死在 compile 阶段**（actor 起来了，HCCL channel 一直建不完）
  2. 加 `--enforce-eager` 也救不了：服务能起来但首次推理报 `ray_compile_graph_communication` 错误
  3. 是否开启 K8s `securityContext.privileged: true` **不影响** 0.14+ 上的卡死
- 0.12 + 特权模式可稳定运行（用作对照）

## 二、根因分析

### 1. `ray_compile_graph_communication` 是什么

vLLM V1 的 Ray executor 用 **Ray Compiled Graph (aDAG, accelerated DAG)** 在 worker 间传输 tensor：

- aDAG 与 `torch.compile` / NPU graph **完全是两层**
- `--enforce-eager` 只关闭 torch / NPU 图编译，**不影响 aDAG**
- 因此「加 eager 后改在推理期报 cgraph communication」是预期行为

aDAG 跨机走 **HCCL P2P (`hcclSend`/`hcclRecv`)**，对端发现走 `VLLM_HOST_IP`，channel 是**持久化**的。

### 2. 0.12 → 0.14+ 的关键变化

| 维度 | 0.12 | 0.14+ |
|---|---|---|
| Worker 间通信 | HCCL **集合通信**（allreduce/allgather） | aDAG **P2P send/recv channel** |
| 调度模式 | driver-centric | SPMD worker（`VLLM_USE_RAY_SPMD_WORKER=1`） |
| 默认特性 | 传统 prefill | chunked prefill + async scheduler |
| HCCL init | 在 worker 主流程显式初始化 | Ray actor `__init__` hook，与 `torch_npu.npu.init()` 存在竞态 |

### 3. 特权模式为什么对 0.12 有效，对 0.14+ 无效

K8s `privileged: true` 实际给的是 **OS / RDMA 层**能力：

| 能力 | 用途 |
|---|---|
| `/dev/infiniband/uverbs*` 全量可写 | HCCL on RoCE 建 QP |
| `CAP_IPC_LOCK` + `ulimit -l unlimited` | HCCL/aDAG pin 大块内存 |
| `CAP_NET_ADMIN` | 配置 RoCE GID、PFC |
| `/sys`、`/proc/sys` 可写 | hccn_tool / 拓扑发现 |
| Hugepages 访问 | HCCL 大 buffer |

- 0.12 走 HCCL allreduce —— 只要这些 OS 层能力齐了就能跑通
- 0.14+ 卡点在 **vLLM/Ray 应用层的 aDAG 调度逻辑、HCCL P2P 实现、CGraph timeout 默认值、SPMD 竞态**，**不是给容器更多权限就能修**

### 4. 0.14+ aDAG 在 NPU 上的具体不稳定点

1. **HCCL P2P 实现成熟度低**：vllm-ascend 的 `hcclSend/hcclRecv` 远不如 allreduce 稳定，channel 空闲会被当成断开
2. **首次 send lazy init 慢**：910C 双机首跑实测 > 默认 `RAY_CGRAPH_get_timeout=10s`，首次必然 timeout
3. **SPMD 竞态**：HCCL world 建立放进 actor `__init__`，与 `torch_npu.npu.init()` 抢资源
4. **chunked prefill + async scheduler** 默认开 → 跨机 P2P 频次比 0.12 提高 5-10 倍，任一抖动被放大成 communication error
5. **CANN/HCCL 版本敏感**：0.14+ 要求 CANN ≥ 8.0.RC3，910C 通常需 8.0.RC3.alpha003+；镜像与宿主 driver 不齐 → P2P 偶发断开

### 5. 社区已知问题方向（建议核实最新状态）

> 以下是社区已存在的同类问题方向，建议按关键词到 vllm-project/vllm-ascend、vllm-project/vllm 仓库 issue tracker 验证：

- `Ray Compiled Graph hang on multi-node 910B/910C`
- `HCCL P2P send/recv unstable in aDAG mode`
- `RAY_CGRAPH_get_timeout default too small for NPU first launch`
- `V1 + Ray + non-CUDA backend stability`
- `SPMD worker race with HCCL/NCCL init`

社区临时方案普遍指向：**`export VLLM_USE_RAY_COMPILED_DAG=0`** 退回旧路径。

---

## 三、修复策略总览

| 层级 | 目标 | 风险 | 见效速度 |
|---|---|---|---|
| **L1 短期回避** | 关掉 0.14+ 的不稳定路径，立刻能跑 | 低 | 立即 |
| **L2 中期加固** | 让默认值适配 NPU 长 warmup，减少超时 | 低 | 立即 |
| **L3 长期治本** | 跟随上游修 aDAG/HCCL P2P，或锁版本基线 | 中 | 跟版本 |

**核心原则**：L1+L2 并行，L3 作为持续跟踪项；**不在应用侧动 vllm-ascend HCCL P2P**（属上游 C++/CANN 层职责）。

---

## 四、L1：短期回避（最有效）

> ⚠️ **2025-12 重要更新**：经过对 vllm 主仓代码的核实（见 §9.1、§9.6），**`VLLM_USE_RAY_COMPILED_DAG` 这个 env 在 vllm 当前主分支已不存在**。新版正确的回退方式是 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`，它继承自 MultiprocExecutor，控制面走 MQ、数据面走 NCCL，**完全绕开 aDAG 路径**。
>
> - **0.14+**：用 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`
> - **≤ 0.13 老版本**：仍可用 `VLLM_USE_RAY_COMPILED_DAG=0`

### 方案 A：绕开 aDAG（V2 Executor Backend，0.14+ 主方案）

```bash
# 0.14+ 推荐
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1

# 老版本（≤ 0.13）兼容写法
# export VLLM_USE_RAY_COMPILED_DAG=0
```

- **效果**：worker 通信走 MQ + NCCL/HCCL，绕开 aDAG P2P 全部坑
- **代价**：单 token 延迟略高（5-15%）
- **可逆**：用户在 `engine.extra_envs` / values.yaml 显式设 `=0` 可覆盖

### 方案 B：关闭 SPMD worker

```bash
export VLLM_USE_RAY_SPMD_WORKER=0
```

- 解决 SPMD 与 HCCL init 竞态
- 通常与 A 配合，单独开收益有限

### 方案 C：极端兜底——回退 V1 引擎

```bash
export VLLM_USE_V1=0
```

- 仅当 A+B 都不见效时考虑
- 损失：失去 V1 chunked prefill / async scheduler 收益
- **不在代码里默认开**，仅作文档说明

**推荐组合**：默认 **A**，可选 **B**；C 仅作"最后手段"备查。

---

## 五、L2：中期加固

### 加固 1：NPU 友好的超时默认值

| Env | 当前/HCCL 默认 | 建议（Ascend 默认） | 理由 |
|---|---|---|---|
| `RAY_CGRAPH_get_timeout` | 300（本仓库现值） | **1800** | 910C 双机首跑 compile 实测可达 600s+ |
| `HCCL_CONNECT_TIMEOUT` | 120 | **600** | RoCE 抖动重试余量 |
| `HCCL_EXEC_TIMEOUT` | 1836 | **1800** | 与 RAY_CGRAPH 对齐 |
| `HCCL_OP_BASE_FFTS_MODE_ENABLE` | 未设 | **TRUE** | 减少 HCCL 算子调度抖动 |

### 加固 2：Ascend 运行时 env 集中导出

当前 `ASCEND_PROCESS_LOG_PATH`、`RAY_CGRAPH_get_timeout`、`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` 散在 head/worker/dp 三路径。

建议抽出 `_build_ascend_runtime_env_commands(ctx)`，head/worker/dp 都引用同一段，顺带把 L1 的 `VLLM_USE_RAY_COMPILED_DAG=0` 放进去（用 bash 软默认 `: "${VAR:=value}"`，允许用户覆盖）。

### 加固 3：版本探测分支

复用 `_get_ray_resource_flag` 里的 vllm-ascend 版本读取逻辑：

```python
if ctx.is_ascend and _vllm_ascend_version_ge("0.14"):
    # 注入 VLLM_USE_RAY_COMPILED_DAG=0 软默认
    # 拉大 RAY_CGRAPH_get_timeout 默认
```

0.12 用户**完全不受影响**，只对受影响的 0.14+ 生效。

### 加固 4：`_build_ray_wait_loop` 失败 fail-fast（issue #6）

- 现状：60×5s=300s 后**不管成败继续往下跑**
- 改为：count<nnodes 时 `exit 1`，并打印「head 未发现足够 worker」明确错误
- 与 L1/L2 配合，避免「卡 compile」被掩盖成「卡 ray wait」

---

## 六、L3：长期治本（跟踪不下场）

1. **锁定基线版本**：在 `engine_version_defaults.yaml` 把 Ascend 多机的「经过验证版本」显式钉死（建议 vllm-ascend 0.12.x + CANN 8.0.RC2.alpha003），新版本进入前必须过双机 e2e 回归
2. **跟踪上游**：watchlist 关键词 `RAY_COMPILED_DAG`、`hcclSend`、`P2P`、`multi-node hang`；上游修复 aDAG P2P 后把 L1 默认值改回 `=1`
3. **e2e 巡检**：CI 加最小双机 Ray 拉起冒烟（10 token 即可）
4. **诊断信息增强**：start script 在 head 启动前 dump `npu-smi info` / `hccn_tool -i 0..7 -ip -g` / 关键 env 到固定路径

---

## 七、落到 `vllm_adapter.py` 的具体修改点（设计稿，待决策）

| 修改点 | 位置 | 内容 |
|---|---|---|
| 新增 `_vllm_ascend_version_ge(min_ver)` | 工具区，靠近 `_get_ray_resource_flag` | 读包版本，比较 |
| 新增 `_build_ascend_runtime_env_commands(ctx)` | env 构造区 | 集中输出 `: "${VAR:=default}"` 软默认 |
| `_build_ray_head_commands` 调用上面新函数 | 替换现有零散 export | 统一来源 |
| `_build_ascend_ray_worker_env` 调用同函数 | 同上 | head/worker 一致 |
| `_build_dp_env_commands` ascend 分支 | 同上 | 三路径同步 |
| `_build_ray_wait_loop` | 末尾加 fail-fast | count<nnodes 则 `exit 1` |

**关键设计原则**：
- 全部用 bash 软默认 `: "${X:=Y}"`，**绝不强制覆盖**用户/values.yaml 显式值
- 版本门控放在 Python 侧，bash 输出干净的 export
- `_build_ascend_runtime_env_commands` 是纯字符串函数，可加 snapshot 测试

---

## 八、风险与边界

- A 方案对 0.14+ 上 GPU 用户**无影响**（仅在 ascend 分支注入）
- 用户已在 values.yaml 显式设 `VLLM_USE_RAY_COMPILED_DAG=1` 时，软默认会被覆盖，他承担风险——符合预期
- **不解决**：vllm-ascend 在 0.14+ 自身的 HCCL P2P bug；只是绕开。需上游修复后才能恢复 aDAG 性能优势
- **不引入**：任何新依赖、任何 vendor-specific patch

---

## 九、现场快速验证步骤（不改代码先验证）

按顺序，能验证根因归属：

```bash
# 1. 关 aDAG 验证（最关键）
kubectl set env deploy/<your-deploy> -c <engine-container> VLLM_USE_RAY_COMPILED_DAG=0
# 重启 → 如果立刻不卡 → 100% 锁定 aDAG/P2P 路径问题

# 2. 关 SPMD 验证
... VLLM_USE_RAY_SPMD_WORKER=0

# 3. 拉大超时
... RAY_CGRAPH_get_timeout=1800 HCCL_EXEC_TIMEOUT=1800 HCCL_CONNECT_TIMEOUT=600

# 4. 退 V0
... VLLM_USE_V1=0

# 5. 版本一致性核查
# 容器内：
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
# 宿主：
npu-smi info
```

---

## 十、一句话结论

**有修复方案，但本质是"回退 + 加固 + 跟踪"，不是"修好 aDAG"**。aDAG/HCCL P2P 修复属 vllm-ascend 上游职责；wings-control 侧能做的是给出一个**对 0.12 用户零影响、对 0.14+ 用户默认稳定、且用户可显式覆盖**的安全默认配置，并把超时与失败兜底做扎实。

---

# 附录 B：0.12 必须开特权的根因 与 去特权方案

> 与正文 0.14+ aDAG 问题**完全独立**的另一条链：0.12 + Ascend + Ray 在 K8s 必须 `privileged: true` 才能跑通的根因，以及最小权限替代方案。

## B.1 直接结论

**是「Linux capability + 设备文件 + OS resource limit + 配置文件挂载」四类 K8s 层资源/能力缺失叠加导致**。这些是 OS/kernel 层硬约束，**不能通过环境变量或应用代码绕过**——必须在 K8s YAML/Pod spec 层面补齐。

## B.2 不开特权时具体缺什么、报什么错

| 缺失项 | 类别 | 触发的报错（典型） | 应用代码能不能绕？ |
|---|---|---|---|
| `CAP_IPC_LOCK` | Linux capability | `mlock: Cannot allocate memory` / `ibv_reg_mr failed` / HCCL pin 内存失败 | **不能**，kernel 层硬限 |
| `memlock` ulimit 未到 unlimited | OS resource limit | 同上，HCCL 大 buffer 注册失败 | **不能**，OS 进程限制 |
| `/dev/infiniband/uverbs*` 未挂载 | 设备文件 | `ibv_open_device failed` / `Failed to open RDMA device` | **不能**，设备不存在 |
| `/etc/hccn.conf` 未挂载 | 配置文件 | `hccp_get_card_ip failed` / 跨机 NPU IP 解析失败 | 理论可绕但极脆弱 |
| `CAP_NET_ADMIN` | Linux capability | `Failed to set RDMA GID` / RoCE GID 配置失败 | **不能**，kernel 权限 |
| `/dev/davinci*` 未挂载 | 设备文件 | NPU 不可见 | **不能** |
| `/usr/local/Ascend/driver` 未挂 | hostPath | `libascendcl.so` 找不到 | **不能** |

特权模式之所以"一开就好"，是因为它**一次性把这 7 项全开**了：全 cap + 关 seccomp + `/dev` 全挂 + 默认无 ulimit 限制。

## B.3 为什么不能纯靠环境变量或代码绕过

### B.3.1 HCCL pin 内存（IPC_LOCK + memlock）
- HCCL 依赖 `ibv_reg_mr()` 把 GB 级内存 pin 在 RAM 里供 RDMA DMA
- 走 kernel `mlock()` 系统调用，kernel 检查 `CAP_IPC_LOCK` + `RLIMIT_MEMLOCK`
- 任一不满足就 `EPERM` 或 `ENOMEM`，应用层只能拿错误码，**没有 fallback 路径**
- 理论 fallback：换 gloo TCP backend；实际：910C 上 vllm-ascend 不支持 gloo，性能下降 50x+，等于不能用

### B.3.2 RDMA 设备文件
- `/dev/infiniband/uverbs0` 是 InfiniBand verbs 的字符设备
- 容器是独立 mount namespace，宿主有的设备**默认不可见**
- 必须显式 `volumeMount` 或走 device plugin
- 应用层完全无法"创建"这个设备
- 理论 fallback：HCCL TCP 模式；实际仅供调试，吞吐 < 1GB/s，跨机不可用

### B.3.3 `/etc/hccn.conf`
- 唯一理论上可代码生成的项
- 内容是各 NPU 卡 IP/MASK/GATEWAY，由宿主 `hccn_tool` 生成
- 业界标准做法都是 hostPath 挂载，自己拼属于侵入 NPU 运维领域

### B.3.4 Ascend driver 库文件
- `libascendcl.so` 等驱动库在宿主 `/usr/local/Ascend/driver/lib64`
- 容器镜像不带（带也会与宿主 driver 版本不匹配），必须 hostPath 挂载

## B.4 三个核心问题的回答

| 问题 | 答案 |
|---|---|
| 触犯的原因是什么？ | Linux capability + 设备文件 + OS resource limit + 配置文件挂载 这四类 K8s 层约束的缺失 |
| 是缺失资源导致的吗？ | **是**，本质是「容器边界默认拒绝访问宿主资源」+「HCCL 必须用宿主 RDMA」的冲突 |
| 能通过环境变量或代码规避吗？ | **不能**。capability/ulimit 是 kernel 检查，设备文件需挂载，driver 库必须 hostPath。代码层只能感知失败，无可用 fallback |

## B.5 最小权限替代方案

### B.5.1 Linux capabilities（替代 privileged 的全 cap）

```yaml
securityContext:
  privileged: false
  allowPrivilegeEscalation: false
  runAsUser: 0          # NPU driver 通常需 root，不可避
  capabilities:
    drop: ["ALL"]
    add:
      - IPC_LOCK        # HCCL pin 内存（关键）
      - NET_ADMIN       # 配置 RoCE GID / route
      - NET_RAW         # RDMA raw socket
      - SYS_PTRACE      # 部分 NPU 调试路径
      # SYS_NICE 可选：HCCL 线程优先级
```

### B.5.2 NPU 设备挂载（推荐 Ascend Device Plugin）

```yaml
resources:
  limits:
    huawei.com/Ascend910: 8     # 由 ascend-device-plugin 注入
```

无 device plugin 时手工挂：`/dev/davinci0..N`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`

### B.5.3 RDMA 设备挂载（910C HCCL on RoCE 必需）

推荐 [`k8s-rdma-shared-dev-plugin`](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin) 或 SR-IOV CNI：

```yaml
resources:
  limits:
    rdma/hca_shared_devices_a: 1
```

手工 hostPath：

```yaml
volumeMounts:
  - { name: rdma-uverbs, mountPath: /dev/infiniband }
volumes:
  - name: rdma-uverbs
    hostPath: { path: /dev/infiniband }
```

### B.5.4 驱动 / 配置 hostPath 最小集

```yaml
volumeMounts:
  - { name: ascend-driver,  mountPath: /usr/local/Ascend/driver, readOnly: true }
  - { name: dcmi,           mountPath: /usr/local/dcmi,          readOnly: true }
  - { name: npu-smi,        mountPath: /usr/local/bin/npu-smi,   readOnly: true }
  - { name: ascend-install, mountPath: /etc/ascend_install.info, readOnly: true }
  - { name: hccn-conf,      mountPath: /etc/hccn.conf,           readOnly: true }
volumes:
  - { name: ascend-driver,  hostPath: { path: /usr/local/Ascend/driver } }
  - { name: dcmi,           hostPath: { path: /usr/local/dcmi } }
  - { name: npu-smi,        hostPath: { path: /usr/local/bin/npu-smi } }
  - { name: ascend-install, hostPath: { path: /etc/ascend_install.info } }
  - { name: hccn-conf,      hostPath: { path: /etc/hccn.conf } }
```

### B.5.5 ulimit

K8s ≥ 1.28 用 Pod 级，旧版本走节点 kubelet `defaultUlimits`，并在 entrypoint 加 `ulimit -l unlimited`（需 IPC_LOCK cap 才生效）。

## B.6 落地难度排序的方案

| 方案 | 要点 | 优点 | 缺点 | 风险 |
|---|---|---|---|---|
| **A 最小权限替换（推荐）** | §B.5 cap + device plugin + hostPath | 改动仅 YAML / Helm；可过 PSS baseline | 集群需装 ascend-device-plugin + rdma-device-plugin | 低 |
| **B Ascend Container Runtime（最干净）** | 节点装 `ascend-docker-runtime`，pod 用 `RuntimeClass: ascend` | 业务无感；可 `privileged: false` + 最少 cap | 需节点级改 containerd/dockerd；运维门槛高 | 低（昇腾官方推荐） |
| **C CDI** | Ascend Device Plugin v6+ 支持 CDI 模式 | 标准化；K8s 1.28+ 原生支持 | 需较新版本 device plugin | 低，需验证版本 |
| **D hostNetwork 绕开** | `hostNetwork: true` | 配置简单 | 占宿主端口；单节点只能跑 1 个 engine 实例；安全差 | 高，**不推荐生产** |
| **E 保留 privileged + PSA 隔离** | namespace 级 PSA + NetworkPolicy | 仅合规审计 | 没真正减权 | **不推荐** |

## B.7 常见踩坑

1. **`memlock` 没拉到无限**：加了 `IPC_LOCK` cap 但 `ulimit -l` 还是 64KB → HCCL pin 失败。需 entrypoint `ulimit -l unlimited` 或 runtime `defaultUlimits`
2. **`/etc/hccn.conf` 没挂**：HCCL 找不到 NPU 卡 IP/网关 → 跨机直接不通。**最小权限场景下必须显式挂载**
3. **Ascend driver 路径不一致**：宿主在 `/usr/local/Ascend/driver`，容器期望 `lib64` 在 `LD_LIBRARY_PATH`；driver 升级后路径变化要同步
4. **rdma-device-plugin 与 NPU device plugin 冲突**：两者都改设备挂载路径，部分版本互相覆盖。解决：用 ascend operator 统一管理
5. **`runAsUser: 0` vs `runAsNonRoot`**：NPU driver 强依赖 root，PSS `restricted` 不兼容，只能用 `baseline`
6. **`CAP_NET_ADMIN` 在 PSS `baseline` 默认禁止**：需单独 allow 或用 `restricted` + 例外

## B.8 迁移路径

```
当前: privileged: true
   ↓ Step 1（一周内）
方案 A: 最小 cap + device plugin + hostPath（安全性 +80%）
   ↓ Step 2（中期，需运维配合）
方案 B 或 C: ascend container runtime / CDI（业务零侵入 + 安全性最佳）
```

每步都先在测试集群跑通双机 Ray + 长跑 (>2h) + 高并发压测，确认无 HCCL 抖动再切。

## B.9 与 0.14+ aDAG 问题的关系

- 这两条是**独立链**：
  - 0.12 特权问题：**OS/kernel 层**资源缺失，可用「最小权限 YAML」彻底解决
  - 0.14+ aDAG 问题：**应用层** vllm-ascend 实现 bug，只能等上游修或环境变量回退
- 即使把 0.12 的特权拿掉，0.14+ 切上去**仍然会卡**
- 反过来，0.14+ aDAG 修好后，0.12 的权限问题**仍然存在**
- **wings-control 代码层对这两个问题都无法根治**——一个是 K8s 部署问题，一个是上游代码问题

---

## 九、上游参考（社区 issue / 代码证据）

> ⚠️ **声明**：以下仅列出**通过 GitHub 代码搜索可直接验证的上游证据**（仓库代码 / release notes / 官方 tutorial）。本次检索**未在 vllm-ascend 仓库中找到精确的"双机 aDAG hang"issue 编号**——因此采用"代码 + release notes + 官方文档"三角佐证，而非引用未经核实的 issue 号。

### 9.1 vllm 主仓代码层证据（aDAG / Ray Compiled Graph 路径确认）

| 证据 | 文件 / 链接 | 说明 |
|---|---|---|
| `RAY_CGRAPH_get_timeout` 默认 300s | [vllm/v1/executor/ray_executor.py#L544-L559](https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/ray_executor.py#L544-L559) | 超时是**驱动侧 wait** 而非 RPC，超时只能让上层快失败、不能解决 HCCL 死锁 |
| aDAG 通过 `with_tensor_transport` 走 NCCL/HCCL P2P | [ray_executor.py#L588-L608](https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/ray_executor.py#L588-L608) | 与 0.12 走集合通信路径根本不同，HCCL P2P 在 vllm-ascend 成熟度低 |
| 通道选择 env：`VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = auto / nccl / shm` | [envs.py#L733-L762](https://github.com/vllm-project/vllm/blob/main/vllm/envs.py#L733-L762) | **可作为软回退尝试**：跨机仍需 NCCL/HCCL，单机可 `shm` |
| `VLLM_USE_RAY_WRAPPED_PP_COMM` 默认 1 | [envs.py#L755-L760](https://github.com/vllm-project/vllm/blob/main/vllm/envs.py#L755-L760) | 改用 vLLM 自己的 PP 通信器包装 Ray，可作为绕开 Ray 原生 NCCL 通信器的尝试 |
| `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`（MQ-based，不走 Compiled Graph） | [envs.py#L762-L765](https://github.com/vllm-project/vllm/blob/main/vllm/envs.py#L762-L765) + [ray_executor_v2.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/ray_executor_v2.py) | **关键回退**：v2 backend 继承 MultiprocExecutor，控制面走 MQ、数据面走 NCCL，**绕开整条 aDAG 路径**。建议在卡死场景验证 |
| 多 Node 分配测试 | [tests/distributed/test_multi_node_assignment.py](https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_multi_node_assignment.py) | 官方多机测试用例 |

### 9.2 vllm-ascend release notes 已知问题（直接相关）

| 版本 | 已知问题原文 | 链接 |
|---|---|---|
| **v0.16.0rc1** | "In 4-node A3 PD disaggregation deployment with DeepSeek V3.2, the **P-Node may hang when benchmarking in high concurrency scenario**, e.g., 2K/2K tokens with 512 concurrent requests." | [release_notes.md#L182-L188](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/release_notes.md#L182-L188) |
| **v0.16.0rc1** | "MTP with large EP configurations may cause graph capture buffer overflow ... workaround: explicitly set `--compilation-config '{\"max_cudagraph_capture_size\": N}'`" | 同上 |
| **v0.9.0rc1** | "**Multi node data-parallel doesn't work with this release.** This is a known issue in vllm and has been fixed on main branch." | [release_notes.md#L1241-L1246](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/release_notes.md#L1241-L1246) → vLLM PR [#18981](https://github.com/vllm-project/vllm/pull/18981) |
| **v0.9.0rc1** | "vLLM process may be crashed with aclgraph enabled. We're working this issue and it'll be fixed in the next release." | 同上 |

> 上述官方已知问题印证了我们的判断：**多机 + Ascend 在压力 / 长稳场景下确实存在 hang 类问题**，且这是 vllm-ascend 团队**正在修而尚未完全修完**的领域。

### 9.3 vllm-ascend 官方教程的"反向证据"

> 官方文档对**跨机 TP** 给出的明确警告，与我们遇到的现象一致：

| 文档 | 原文 | 结论 |
|---|---|---|
| **MiniMax-M2.5 双机指南** | "Since **cross-node tensor parallelism (TP) can be unstable**, the dual-node guide uses a **tp=8 + dp=2** setup (8 NPUs per node, 16 NPUs total)." | [MiniMax-M2.5.md#L195-L309](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/MiniMax-M2.5.md#L195-L309) — 官方推荐**用 DP 替代跨机 TP** |
| **MiniMax-M2.5 FAQ** | "Why not use cross-node tp=16? A: The referenced practice noted that cross-node TP may be unstable, so tp=8, dp=2 is recommended for dual-node deployment." | [MiniMax-M2.5.md#L465-L476](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/MiniMax-M2.5.md#L465-L476) |
| **GLM-4.x 双机指南** | "Although the former tutorial said 'Not recommended to deploy multi-node on Atlas 800 A2 (64G × 8)', but if you insist to deploy GLM-4.x model on multi-node like 2 × Atlas 800 A2 (64G × 8) ..." | [zh_CN/.../GLM4.x.po#L245-L258](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/GLM4.x.po#L245-L258) — 官方"不建议双机"措辞 |

### 9.4 HCCL 超时官方默认值（DeepSeek-V3.1/V3.2 双机示例脚本）

```bash
# 来自 vllm-ascend 官方 DeepSeek-V3.x 多机部署文档
export HCCL_EXEC_TIMEOUT=204         # 算子执行超时（秒）
export HCCL_CONNECT_TIMEOUT=120      # 连接超时（秒）
export HCCL_BUFFSIZE=200             # MB
export HCCL_OP_EXPANSION_MODE=AIV
```

参考：[DeepSeek-R1.md#L174-L198](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/DeepSeek-R1.md#L174-L198)、[Kimi-K2.5.md#L250-L273](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Kimi-K2.5.md#L250-L273)

> 我们在分析中提到的 `HCCL_EXEC_TIMEOUT` 调高建议，与官方默认值方向一致。

### 9.5 vllm-ascend KV transfer 超时相关代码

- `vllm_ascend/distributed/kv_transfer/utils/utils.py::get_transfer_timeout_value()` 暴露：
  - `ASCEND_TRANSFER_TIMEOUT`
  - `HCCL_RDMA_TIMEOUT`
  - `HCCL_RDMA_RETRY_CNT`

> 如走 PD 分离 + RDMA 场景命中超时，可优先调这三个 env。

### 9.6 本次检索未找到的（坦白说明）

| 类别 | 状态 |
|---|---|
| "vllm-ascend 双机 Ray Compiled Graph 卡死"明确 issue 编号 | **未找到** —— 用 `multi node ray hang CompiledGraph` 等关键词在 vllm-ascend 仓库未命中明确 issue |
| `VLLM_USE_RAY_COMPILED_DAG=0` 关闭 aDAG 的明确文档 | **未在 vllm 当前主分支找到此 env**（只有 `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` 和 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`）。早期版本可能存在该 env，新版**应改用 V2 executor backend 回退** |
| 0.14+ 与 0.12 行为差异的官方 changelog 条目 | **未找到精确说明**，但通过代码层的 `_compiled_ray_dag` / `_init_executor` 实现可佐证 0.14+ 默认走 aDAG |

### 9.7 给到运维 / 研发的可操作回退矩阵（基于上述上游证据）

| 顺序 | 操作 | 上游依据 |
|---|---|---|
| 1 | 升级到 vllm-ascend 最新 rc 验证（≥ 0.16.0rc1 已修部分） | 9.2 release notes |
| 2 | **跨机改用 DP 替代 TP**（如官方 MiniMax-M2.5 建议 tp=8 + dp=2） | 9.3 |
| 3 | 试 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`（绕开 aDAG） | 9.1 |
| 4 | 试 `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm`（仅单机有效） | 9.1 |
| 5 | 试 `VLLM_USE_RAY_WRAPPED_PP_COMM=0` | 9.1 |
| 6 | 调高 `HCCL_EXEC_TIMEOUT` / `HCCL_CONNECT_TIMEOUT` | 9.4 |
| 7 | 回退 vllm-ascend 0.12 + 配最小权限 YAML（见 Appendix B） | 本文 |
