# 910C + vllm-ascend + Ray 双机：现场实施验证 Runbook

> **当前文档角色：实施（How）** —— 现场可执行操作手册，受众：运维 / SRE

## 文档资产列表（Ascend + Ray 系列）

| 角色 | 文件 | 受众 | 用途 |
|---|---|---|---|
| 📘 分析 | [ascend-vllm014-ray-adag-issue.md](./ascend-vllm014-ray-adag-issue.md) | 研发 / 架构 | 根因 + 修复策略 + 设计稿 |
| 📗 **实施 Runbook（本文）** | [ascend-vllm014-ray-adag-runbook.md](./ascend-vllm014-ray-adag-runbook.md) | 运维 / SRE | 现场逐步操作命令 + 故障升级路径 |
| 📙 配置示例 | [ascend-values-examples.yaml](./ascend-values-examples.yaml) | 运维 | 三套 Helm Profile 直接复制 |

**两条独立链路**：
- **链路 A**：0.14+ aDAG 卡死（应用层，env 软回退）
- **链路 B**：0.12 去特权模式（K8s 层，最小权限替换）

---

本文档是**可直接复制粘贴执行**的现场操作手册。每个步骤都包含「执行命令」「预期结果」「失败处理」三段，可由运维独立完成。

---

## 0. 通用前置准备

### 0.1 确认环境信息（开始前必填）

| 项 | 值 | 备注 |
|---|---|---|
| 集群入口 | `kubectl config current-context` 输出 | |
| Namespace | 例：`namespace-0` | |
| Deployment 名 | 例：`serving-rn-axtq207` | |
| Engine 容器名 | 例：`engine` | |
| Wings-control 容器名 | 例：`wings-control` | |
| Master Pod 名 | `kubectl -n <ns> get pod -l role=master` | |
| Worker Pod 名 | 同上 `role=worker` | |
| vllm-ascend 版本 | 容器内 `pip show vllm-ascend` | 决定走链路 A 还是只看链路 B |
| CANN 版本 | 容器内 `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` | |
| 节点数 | 例：2 | |
| 每节点 NPU 数 | 例：8 | |
| 模型 | 例：Qwen3-32B | |
| TP / PP / DP | 例：TP=8 PP=1 | |

### 0.2 设环境变量便于后续命令引用

```bash
export NS=namespace-0
export DEP=serving-rn-axtq207
export ENGINE_CTR=engine
export WINGS_CTR=wings-control
export MASTER_POD=$(kubectl -n $NS get pod -l app=$DEP,role=master -o jsonpath='{.items[0].metadata.name}')
export WORKER_POD=$(kubectl -n $NS get pod -l app=$DEP,role=worker -o jsonpath='{.items[0].metadata.name}')
echo "MASTER=$MASTER_POD  WORKER=$WORKER_POD"
```

> 标签 key 视实际部署调整（可能是 `role` / `wings.io/role` / `app.kubernetes.io/component` 等）

---

# 链路 A：0.14+ aDAG 卡死验证（应用层）

## A.1 验证当前是否处于 aDAG 卡死状态

### A.1.1 看 head Pod 当前状态

```bash
kubectl -n $NS describe pod $MASTER_POD | grep -E "Status|Restart|Reason|Message"
kubectl -n $NS get pod $MASTER_POD -o jsonpath='{.status.containerStatuses[*].restartCount}'
```

**预期**：CrashLoopBackOff 或 Running 但日志卡住

### A.1.2 看 head 日志关键字

```bash
kubectl -n $NS logs $MASTER_POD -c $ENGINE_CTR --tail=500 | \
  grep -iE "compiled.*graph|cgraph|ray.*timeout|hccl|p2p|aDAG|ChannelOutput|hcclSend|hcclRecv"
```

**关键词命中即说明是 aDAG 路径问题**：
- `Failed to get communicator from compiled graph`
- `RayCompiledGraph timeout`
- `aDAG channel`
- `hcclSend/hcclRecv` 错误

### A.1.3 看 worker 日志

```bash
kubectl -n $NS logs $WORKER_POD -c $ENGINE_CTR --tail=500 | \
  grep -iE "compiled.*graph|cgraph|hccl|aDAG"
```

## A.2 现场快速回退测试（推荐第一步做这个）

> ⚠️ **2025-12 更新**：上游 vllm 主分支已不存在 `VLLM_USE_RAY_COMPILED_DAG=0` 这个 env（详见分析文档 §9.6）。**当前正确的回退方式是启用 V2 executor backend**，它继承自 MultiprocExecutor，控制面走 MQ、数据面走 NCCL，**完全绕开 aDAG 路径**。
>
> 如果您部署的是较老版本（≤ 0.13）仍可尝试 `VLLM_USE_RAY_COMPILED_DAG=0`，但 **0.14+ 必须用 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`**。

### A.2.1 注入 V2 Executor Backend（替代 aDAG）

> **方法 1**：通过 wings-control 的 `extra_envs` / values.yaml 加（推荐，可持久）
>
> **方法 2**：直接 `kubectl set env` 临时覆盖（快速验证）

```bash
# 方法 2 快速验证（0.14+ 推荐）
kubectl -n $NS set env deploy/$DEP -c $ENGINE_CTR VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1

# 旧版本（≤ 0.13）兼容写法
# kubectl -n $NS set env deploy/$DEP -c $ENGINE_CTR VLLM_USE_RAY_COMPILED_DAG=0

# 等待 Pod 重启
kubectl -n $NS rollout status deploy/$DEP --timeout=10m
```

### A.2.2 验证 env 已生效

```bash
kubectl -n $NS exec $MASTER_POD -c $ENGINE_CTR -- env | grep -E 'VLLM_USE_RAY_V2_EXECUTOR_BACKEND|VLLM_USE_RAY_COMPILED_DAG'
kubectl -n $NS exec $WORKER_POD -c $ENGINE_CTR -- env | grep -E 'VLLM_USE_RAY_V2_EXECUTOR_BACKEND|VLLM_USE_RAY_COMPILED_DAG'
```

**预期**：两个 Pod 都输出 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`（或老版本下的 `VLLM_USE_RAY_COMPILED_DAG=0`）

### A.2.3 观察启动日志

```bash
kubectl -n $NS logs -f $MASTER_POD -c $ENGINE_CTR | tee /tmp/head-after-fix.log
```

**预期**：
- 不再看到 `RayCompiledGraph` / `aDAG channel` 关键字
- 出现 `vLLM API server started` / `Uvicorn running` 表示启动成功
- 时间从「卡住 >5min」降到「<2min 启动完成」

### A.2.4 推理冒烟

```bash
# 端口转发到本地
kubectl -n $NS port-forward $MASTER_POD 8000:8000 &
PF_PID=$!

# 发起一次推理
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "prompt": "你好",
    "max_tokens": 20,
    "temperature": 0
  }'

kill $PF_PID
```

**预期**：返回 JSON，无 `ray_compile_graph_communication` 错误

### A.2.5 验证结果判定

| 现象 | 结论 | 下一步 |
|---|---|---|
| 启动 + 推理都 OK | 100% 锁定 aDAG 路径问题 | 进入 A.3 持久化方案 |
| 启动 OK 推理仍报 cgraph 错 | 还有 SPMD 竞态 | 进入 A.6 加 SPMD 关闭 |
| 启动仍卡 | aDAG 不是唯一原因 | 进入 A.4 拉超时 + A.5 看 HCCL |

## A.3 持久化方案：写进 values.yaml / wings-control 配置

> 验证有效后，把环境变量固化下来，避免每次部署重新 set。

```yaml
# Helm values.yaml 示例（0.14+ 推荐）
engine:
  extraEnvs:
    # 关键：启用 V2 executor backend，绕开 aDAG（vllm 0.14+ 主流方案）
    - name: VLLM_USE_RAY_V2_EXECUTOR_BACKEND
      value: "1"
    # 备选：通信通道改 shm（仅单机生效，跨机仍需 nccl）
    # - name: VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
    #   value: "shm"
    # 备选：换用 vLLM 自己的 PP 通信器包装（绕开 Ray 原生 NCCL 通信器）
    # - name: VLLM_USE_RAY_WRAPPED_PP_COMM
    #   value: "0"
    # 兜底：拉高超时（即使死锁也至少能快速失败而非永久 hang）
    - name: RAY_CGRAPH_get_timeout
      value: "1800"
    - name: HCCL_CONNECT_TIMEOUT
      value: "600"
    - name: HCCL_EXEC_TIMEOUT
      value: "1800"
    # ── 老版本（≤ 0.13）兼容（新版 vllm 已无此 env，留作历史参考） ──
    # - name: VLLM_USE_RAY_COMPILED_DAG
    #   value: "0"
```

应用：

```bash
helm upgrade $DEP <chart> -f values.yaml -n $NS
kubectl -n $NS rollout status deploy/$DEP --timeout=10m
```

## A.4 如果 A.2 不见效：拉大所有超时

```bash
kubectl -n $NS set env deploy/$DEP -c $ENGINE_CTR \
  RAY_CGRAPH_get_timeout=1800 \
  HCCL_CONNECT_TIMEOUT=600 \
  HCCL_EXEC_TIMEOUT=1800

kubectl -n $NS rollout status deploy/$DEP --timeout=10m
```

观察是否「变慢但能跑完」—— 如果是，说明只是 timeout 默认值过小。

## A.5 检查 HCCL 网络层

```bash
# 在 master pod 内执行
kubectl -n $NS exec $MASTER_POD -c $ENGINE_CTR -- bash -c '
  for i in 0 1 2 3 4 5 6 7; do
    hccn_tool -i $i -ip -g
    hccn_tool -i $i -net_health -g
  done
'
```

**预期**：
- 每张 NPU 卡都有 IP，且与 worker pod 上的 NPU IP 同子网
- `net_health` 显示 `Healthy`

**异常处理**：
- IP 缺失 → 节点 `hccn.conf` 没配置好，找运维
- `net_health: Unhealthy` → 物理 RoCE 链路问题，找网络组

## A.6 加 SPMD 关闭

```bash
# 0.14+ 推荐写法
kubectl -n $NS set env deploy/$DEP -c $ENGINE_CTR \
  VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 \
  VLLM_USE_RAY_SPMD_WORKER=0

# 老版本（≤ 0.13）兼容
# kubectl -n $NS set env deploy/$DEP -c $ENGINE_CTR \
#   VLLM_USE_RAY_COMPILED_DAG=0 \
#   VLLM_USE_RAY_SPMD_WORKER=0

kubectl -n $NS rollout status deploy/$DEP --timeout=10m

> 仅当 A.2 ~ A.6 全部不见效再做

```bash
kubectl -n $NS set env deploy/$DEP -c $ENGINE_CTR VLLM_USE_V1=0
```

## A.8 链路 A 完成判据

- [ ] master pod 启动 < 3 min，状态 Running
- [ ] worker pod 状态 Running
- [ ] head 日志最后出现 `Uvicorn running on 0.0.0.0:8000` 或 `vLLM API server started`
- [ ] curl 推理返回正常 JSON
- [ ] 长跑 30min 无 cgraph 错误
- [ ] 配置已固化进 values.yaml

---

# 链路 B：0.12 去特权模式验证（K8s 最小权限）

## B.1 前置：确认集群已具备替代能力

### B.1.1 检查 Ascend Device Plugin

```bash
kubectl get ds -A | grep -i ascend
kubectl get nodes -o json | jq '.items[].status.allocatable' | grep -i ascend
```

**预期**：节点上能看到 `huawei.com/Ascend910` 资源；如无，先装 [`ascend-device-plugin`](https://gitee.com/ascend/ascend-device-plugin)

### B.1.2 检查 RDMA Device Plugin

```bash
kubectl get ds -A | grep -iE "rdma|sriov"
kubectl get nodes -o json | jq '.items[].status.allocatable' | grep -iE "rdma|hca"
```

**预期**：能看到 `rdma/hca_shared_devices_a` 或类似资源；如无，装 [`k8s-rdma-shared-dev-plugin`](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin)

### B.1.3 检查节点 ulimit 配置

```bash
# 在某个节点上 SSH 执行
cat /etc/containerd/config.toml | grep -A5 default_ulimits
# 或 docker
cat /etc/docker/daemon.json | grep -A3 default-ulimits
```

**预期**：包含 `memlock = -1`；如无，需配置：

```toml
# /etc/containerd/config.toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
  default_ulimits = ["memlock=-1:-1", "stack=67108864:67108864"]
```

修改后 `systemctl restart containerd` 并 cordon/drain 节点。

## B.2 准备最小权限 Pod 模板

### B.2.1 完整可粘贴的 securityContext + volumes 片段

```yaml
# 替换原 deployment 的 securityContext + volumes
spec:
  template:
    spec:
      containers:
      - name: engine
        securityContext:
          privileged: false                     # 关键：关掉特权
          allowPrivilegeEscalation: false
          runAsUser: 0
          capabilities:
            drop: ["ALL"]
            add:
              - IPC_LOCK
              - NET_ADMIN
              - NET_RAW
              - SYS_PTRACE
        resources:
          limits:
            huawei.com/Ascend910: 8
            rdma/hca_shared_devices_a: 1
        volumeMounts:
        - { name: ascend-driver,  mountPath: /usr/local/Ascend/driver, readOnly: true }
        - { name: dcmi,           mountPath: /usr/local/dcmi,          readOnly: true }
        - { name: npu-smi,        mountPath: /usr/local/bin/npu-smi,   readOnly: true }
        - { name: ascend-install, mountPath: /etc/ascend_install.info, readOnly: true }
        - { name: hccn-conf,      mountPath: /etc/hccn.conf,           readOnly: true }
        - { name: shm,            mountPath: /dev/shm }
        command: ["/bin/bash", "-c"]
        args:
        - |
          ulimit -l unlimited        # 配合 IPC_LOCK 生效
          ulimit -n 65535
          exec /your/entrypoint.sh
      volumes:
      - { name: ascend-driver,  hostPath: { path: /usr/local/Ascend/driver } }
      - { name: dcmi,           hostPath: { path: /usr/local/dcmi } }
      - { name: npu-smi,        hostPath: { path: /usr/local/bin/npu-smi } }
      - { name: ascend-install, hostPath: { path: /etc/ascend_install.info } }
      - { name: hccn-conf,      hostPath: { path: /etc/hccn.conf } }
      - { name: shm,            emptyDir: { medium: Memory, sizeLimit: 32Gi } }
```

### B.2.2 应用前先在测试 namespace 试

```bash
# 复制原 deployment 到测试 ns
kubectl -n $NS get deploy $DEP -o yaml > /tmp/deploy.yaml
# 改 namespace + 改 securityContext + apply
sed -i "s/namespace: $NS/namespace: namespace-test/" /tmp/deploy.yaml
# 手工编辑 /tmp/deploy.yaml 套上 §B.2.1 模板
kubectl apply -f /tmp/deploy.yaml
```

## B.3 启动后逐项验证

### B.3.1 Pod 拉起成功

```bash
kubectl -n namespace-test get pod -l app=$DEP -w
```

**预期**：所有 pod 5 分钟内进入 Running

**失败处理**：

| 现象 | 原因 | 解决 |
|---|---|---|
| Pending: insufficient `huawei.com/Ascend910` | device plugin 未注册 | 检查 §B.1.1 |
| Pending: insufficient `rdma/hca_shared_devices_a` | rdma plugin 未注册 | 检查 §B.1.2 |
| CrashLoopBackOff | driver 路径错 / hccn.conf 错 | 进入 §B.3.2 |

### B.3.2 容器内 NPU 可见性

```bash
kubectl -n namespace-test exec $MASTER_POD -c $ENGINE_CTR -- npu-smi info
```

**预期**：列出 8 张 NPU，状态 `OK`

**失败**：检查 `/usr/local/Ascend/driver` 挂载路径是否与节点一致

### B.3.3 容器内 RDMA 可见性

```bash
kubectl -n namespace-test exec $MASTER_POD -c $ENGINE_CTR -- ls /dev/infiniband/
kubectl -n namespace-test exec $MASTER_POD -c $ENGINE_CTR -- ibv_devices
```

**预期**：能列出 `uverbs0` 等设备，`ibv_devices` 输出 HCA 列表

**失败**：rdma device plugin 未生效，回退用 hostPath 挂 `/dev/infiniband`

### B.3.4 容器内 ulimit / capability 验证

```bash
kubectl -n namespace-test exec $MASTER_POD -c $ENGINE_CTR -- bash -c '
  echo "=== capabilities ==="
  capsh --print | grep "Current:"
  echo "=== ulimit -l ==="
  ulimit -l
  echo "=== /proc/self/status Cap ==="
  grep ^Cap /proc/self/status
'
```

**预期**：
- `Current:` 包含 `cap_ipc_lock,cap_net_admin,cap_net_raw`
- `ulimit -l` 输出 `unlimited`

**失败**：
- cap 缺 → securityContext 没生效，检查 `allowPrivilegeEscalation: false` 是否冲突
- ulimit 不是 unlimited → 节点 containerd 没配 default_ulimits，回到 §B.1.3

### B.3.5 HCCL 跨机连通性测试

```bash
# 在 master pod 内
kubectl -n namespace-test exec $MASTER_POD -c $ENGINE_CTR -- bash -c '
  for i in 0 1 2 3 4 5 6 7; do
    hccn_tool -i $i -ip -g
  done
'
# 在 worker pod 内做同样事
kubectl -n namespace-test exec $WORKER_POD -c $ENGINE_CTR -- bash -c '
  for i in 0 1 2 3 4 5 6 7; do
    hccn_tool -i $i -ip -g
  done
'
# 两边的 NPU IP 应该同子网
```

### B.3.6 HCCL 实际打洞测试（HCCL ping）

```bash
# master 上拿 worker NPU 0 的 IP，例如 192.168.100.10
kubectl -n namespace-test exec $MASTER_POD -c $ENGINE_CTR -- \
  hccn_tool -i 0 -ping -g address=192.168.100.10 count=10
```

**预期**：10/10 success；如失败，物理网络问题，找网络组

### B.3.7 启动 vLLM 推理冒烟

跟 §A.2.3 / §A.2.4 同样的方法。

## B.4 长跑稳定性验证

```bash
# 后台跑 30min 持续推理
for i in $(seq 1 600); do
  curl -s http://127.0.0.1:8000/v1/completions \
    -d '{"model":"qwen3-32b","prompt":"测试","max_tokens":50}' \
    -o /tmp/resp_$i.json
  sleep 3
done &

# 同时观察日志
kubectl -n namespace-test logs -f $MASTER_POD -c $ENGINE_CTR | \
  grep -iE "error|fail|hccl|timeout" | tee /tmp/longrun-errors.log
```

**判定**：30min 内 `/tmp/longrun-errors.log` 无 HCCL/timeout 类错误

## B.5 灰度推到生产

```
Step 1: 测试 namespace 验证通过（§B.3 + §B.4）
   ↓
Step 2: 生产灰度 1 个副本（保留 1 个 privileged 副本作回滚）
   ↓
Step 3: 观察 24h，监控 HCCL 错误率 / 推理 P99 延迟
   ↓
Step 4: 全量切换；保留回滚 PR
```

## B.6 回滚预案

```bash
# 一行命令回滚到 privileged 版本
kubectl -n $NS rollout undo deploy/$DEP
# 或
helm rollback $DEP -n $NS
```

## B.7 链路 B 完成判据

- [ ] Pod `securityContext.privileged: false`
- [ ] `capsh --print` 只显示明确加的 4 个 cap
- [ ] `ulimit -l` 输出 `unlimited`
- [ ] HCCL ping 跨机 100% success
- [ ] 30min 长跑无 HCCL 错误
- [ ] 推理 P99 延迟与 privileged 版本相比 ±5%
- [ ] 灰度通过，已写回滚预案

---

# 总览 Checklist

## 适用场景判定

```
你的现象是？
├── 0.14+ 卡 compile / 报 cgraph 错 → 走链路 A
├── 0.12 必须开特权才能跑 → 走链路 B
├── 同时存在 → 先 A 后 B（A 是应用层验证更快，B 涉及集群改动）
└── 都没有但想预防 → 至少把 A.3 的超时配置写进 values.yaml
```

## 一键执行命令包（链路 A 快速验证）

```bash
# 复制粘贴一段直接验证（0.14+ 推荐）
export NS=<your-ns> DEP=<your-dep> ENGINE_CTR=<engine-ctr>
kubectl -n $NS set env deploy/$DEP -c $ENGINE_CTR \
  VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 \
  RAY_CGRAPH_get_timeout=1800 \
  HCCL_CONNECT_TIMEOUT=600 \
  HCCL_EXEC_TIMEOUT=1800
kubectl -n $NS rollout status deploy/$DEP --timeout=10m
sleep 30
kubectl -n $NS logs -l app=$DEP -c $ENGINE_CTR --tail=200 | grep -iE "started|error|fail"

# 老版本（≤ 0.13）改用：
#   VLLM_USE_RAY_COMPILED_DAG=0 \
```

## 故障升级路径

| 链路 A 仍卡 | 链路 B 起不来 |
|---|---|
| 1. 加 SPMD 关闭 (§A.6) | 1. 检查 device plugin (§B.1.1/B.1.2) |
| 2. 退 V0 (§A.7) | 2. 检查节点 ulimit (§B.1.3) |
| 3. 抓 head + worker 完整日志报上游 issue | 3. 抓 `dmesg \| grep -i hccl` 找 kernel 层错误 |
| 4. 临时回退 0.12 + 特权 | 4. 临时回退 privileged: true |

## 联系人 / 上游 issue

- vllm-ascend issue tracker: https://github.com/vllm-project/vllm-ascend/issues
- vllm V1 + Ray 通用问题: https://github.com/vllm-project/vllm/issues
- 上报 issue 必带信息：vllm 版本 / vllm-ascend 版本 / CANN 版本 / NPU 型号 / 完整 head+worker 日志 / 启动脚本

---

## 修订记录

| 日期 | 变更 |
|---|---|
| 2026-04-23 | 初版，配套 ascend-vllm014-ray-adag-issue.md |
