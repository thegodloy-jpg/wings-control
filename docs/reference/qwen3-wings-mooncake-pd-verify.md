# Qwen3-30B + wings + mooncake PD 验证手册

> 目标:用 **wings 这个项目** 把 Qwen3-30B-A3B 的 PD 分离跑起来,**正常触发 mooncake KV 传输 + 出预测**。
>
> 适用拓扑:P 节点(external-lb, DP fork)+ D 节点 ×N。proxy 放 wings 外面自己调度。

---

## 0. 先认清:为什么"当前版本即使 dp>1 也不通"

**wings 只负责生成每个节点的引擎启动命令,它本身不做 PD 请求路由。** mooncake 的 KV 传输**不是引擎一起来就自动发生**的,而是靠一个 **PD proxy 在请求层把 P→D 串起来**触发的:

```
client ──> [PD proxy]  ──(1)请求发 prefiller(P)──> P 引擎:产 KV,经 mooncake 注册
              │         <──(2)拿到 kv 传输元数据──
              └─────────(3)带元数据转发 decoder(D)──> D 引擎:经 mooncake 拉 KV,出 token
```

**关键**:第 (1)(2)(3) 步的编排,是 vllm-ascend 的 `load_balance_proxy_server_example.py` 干的。
wings 的 proxy(:18000)只是把请求转给**本地**引擎,**没有 P→D 拆流**。

> 所以无论 standalone 还是 dp>1,只要**没有这个外部 proxy**,mooncake 就永远不触发——请求要么打到 P 被它单独 prefill+decode 全干了(你看到的"预测通了但节点不互通"),要么直接挂。
> **这不是 wings 的 bug,是 PD 架构本来就需要的"分发大脑"不在 wings 里。** 你已决定 proxy 放 wings 外面,本文就按这个来。

---

## 1. 完整链路(三件套,缺一不可)

| # | 组件 | 谁提供 | 作用 |
|---|------|--------|------|
| 1 | P/D 引擎(带 mooncake kv-config) | **wings 生成 start_command.sh** | 产/收 KV |
| 2 | mooncake RDMA 传输 | 引擎内 vllm-ascend + 网络 | 真正搬 KV 字节 |
| 3 | **PD proxy** | **你在 wings 外面跑** `load_balance_proxy_server_example.py` | 请求层 P→D 编排,**触发 mooncake** |

---

## 2. 用哪条路径:external-lb(dp>1)—— 推荐

wings 有两条 PD 生成路径,**验证 mooncake 推荐 external-lb**,因为它会自动补 `kv_port` + 唯一 `engine_id`(mooncake 跨实例需要),且能读 `pd_config.json` 注册表带上 MoE/性能参数:

| | external-lb(dp>1,推荐) | standalone(dp=1) |
|---|---|---|
| 触发条件 | `PD_ROLE` + **`DP_SIZE>1`** | `PD_ROLE` + 不设 `DP_SIZE` |
| 生成结构 | `vllm ... --data-parallel-external-lb` fork(= 官方 `launch_online_dp.py`) | 单进程 `--tensor-parallel-size N` |
| `kv_port`/`engine_id` | ✅ 自动补(`30000+i` / `$RANK`) | ❌ 无(靠 vLLM 默认) |
| 读 `pd_config.json` | ✅(`Qwen3MoeForCausalLM` 已注册) | ❌ 不读 |
| 引擎参数(EP/additional-config/HCCL env) | ✅ 来自注册表 | ❌ 只能 CLI 补部分 |

> `Qwen3MoeForCausalLM` 已在 [pd_config.json](../../wings_control/config/defaults/pd_config.json) 注册(external-lb 专用)。

---

## 3. wings 环境变量(注入 **wings-control 容器** env)

### 3.1 P 节点(4 卡,DP2×TP2 fork)

```bash
PD_ROLE=P
WINGS_DEVICE=ascend
# external-lb fork 编排(平台或你注入)
DP_SIZE=2 TP_SIZE=2 DP_SIZE_LOCAL=2
Master_IP=<P_RDMA_IP> NODE_IPS=<P_RDMA_IP> HOST_IP=<P_RDMA_IP>
VLLM_LLMDD_RPC_PORT=12321 VLLM_MOONCAKE_BOOTSTRAP_PORT=23000
# 对端(decode)拓扑——给 mooncake KV 映射,必须和 D 实际一致
PD_DECODE_DP_SIZE=2 PD_DECODE_TP_SIZE=2
ASCEND_ENFORCE_EAGER=true
```

### 3.2 D 节点(2 卡 ×N;若组成 DP 组则 DP_SIZE>1)

```bash
PD_ROLE=D
WINGS_DEVICE=ascend
DP_SIZE=2 TP_SIZE=2 DP_SIZE_LOCAL=1          # 每个 D pod fork 1 个,N 个 pod 组成 dp
Master_IP=<D-rank0_IP> NODE_IPS=<D1_IP>,<D2_IP> HOST_IP=<本D_IP>
VLLM_LLMDD_RPC_PORT=12321 VLLM_MOONCAKE_BOOTSTRAP_PORT=23100
# 对端(prefill)拓扑——必须和 P 实际一致
PD_PREFILL_DP_SIZE=2 PD_PREFILL_TP_SIZE=2
ASCEND_ENFORCE_EAGER=true
```

> ⚠️ **拓扑一致性铁律**:P 和 D 最终 kv-config 里的 `prefill{}` 必须完全相同、`decode{}` 必须完全相同。
> external-lb 下:P 的 `prefill`=P 的 `DP_SIZE/TP_SIZE`,P 的 `decode`=`PD_DECODE_*`;
> D 的 `decode`=D 的 `DP_SIZE/TP_SIZE`,D 的 `prefill`=`PD_PREFILL_*`。四个值要对齐。

---

## 4. wings 生成的引擎命令(已 dry_run 验证)

P 节点(`DP_SIZE_LOCAL=2`)生成的 fork 体:

```bash
for i in $(seq 0 1); do
  RANK=$((0 + i)); PORT=$((17000 + i))
  KVPORT=$((30000 + i)); BOOTSTRAP=$((23000 + i))
  LO=$((i * 2)); HI=$((LO + 1)); CARDS=$(seq -s, $LO $HI)
  ASCEND_RT_VISIBLE_DEVICES=$CARDS VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /usr/local/serving/models/ --served-model-name Qwen3-30B-A3B \
    --port $PORT --tensor-parallel-size 2 \
    --data-parallel-size 2 --data-parallel-rank $RANK --data-parallel-size-local 1 \
    --data-parallel-address <P_IP> --data-parallel-rpc-port 12321 --data-parallel-external-lb \
    --enable-expert-parallel --enable-prefix-caching --enforce-eager \
    --additional-config '{"enable_cpu_binding":"True"}' \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer",
      "kv_port":"'"$KVPORT"'","engine_id":"'"$RANK"'",
      "kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}' &
done; wait -n
```

**端口表(proxy 要按这个填):**

| 节点 | 服务 HTTP 端口 | kv_port | bootstrap | 可见卡 |
|------|---------------|---------|-----------|--------|
| P rank0 | `17000` | 30000 | 23000 | 0,1 |
| P rank1 | `17001` | 30001 | 23001 | 2,3 |
| D(每 pod,local=1) | `17000` | 30100 | 23100 | 0,1 |

---

## 5. ⭐ 关键:在 wings 外面起 PD proxy(触发 mooncake 的那块)

引擎起好后,**这一步才让 mooncake 真正传 KV**。`--prefiller-*` 指向 P 的服务端口,`--decoder-*` 指向 D 的:

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 18000 --host 0.0.0.0 \
  --prefiller-hosts <P_IP> <P_IP> \
  --prefiller-ports 17000 17001 \
  --decoder-hosts <D1_IP> <D2_IP> \
  --decoder-ports 17000 17000
```

> `--prefiller-hosts/ports` = §4 端口表里 P 的两个服务(17000/17001);
> `--decoder-hosts/ports` = 两个 D pod 的引擎端口(各 17000)。
> proxy 监听 18000——**压测/curl 打这个 18000,不是打引擎的 17000,也不是 wings 的 18000。**

---

## 6. mooncake / 网络要求

- 引擎里 `HCCL_IF_IP` = 本节点 IP(wings 已注入);`*_SOCKET_IFNAME=eth0` + `HCCL_INTRA_ROCE_ENABLE=1` 即可(你的自定义版用 eth0 也能跑,eth0 不是阻塞点)。
- P/D 之间 RDMA 互通;`VLLM_MOONCAKE_BOOTSTRAP_PORT` 在节点间可达。
- `pd_config.json` 的 Qwen3 条目已带 `USE_MULTI_GROUPS_KV_CACHE` / `USE_MULTI_BLOCK_POOL` / `ASCEND_BUFFER_POOL` / `VLLM_RPC_TIMEOUT` 等(对齐你模板)。

---

## 7. 验证步骤

```bash
# 1) 各引擎起来(看 wings-control 日志确认 external-lb 触发)
kubectl logs <P-pod> -c wings-control | grep "PD external-lb"
#   期望: [PD external-lb] arch=Qwen3MoeForCausalLM role=P ... dp_size=2 local=2

# 2) 引擎 health（每个服务端口）
curl -s http://<P_IP>:17000/health   # 200
curl -s http://<D1_IP>:17000/health  # 200

# 3) 打 PD proxy(:18000),发预测
curl -H 'Content-type: application/json' -X POST http://<proxy_IP>:18000/v1/chat/completions \
  -d '{"model":"Qwen3-30B-A3B","messages":[{"role":"user","content":"你是谁"}],"max_tokens":64,"temperature":0}'

# 4) 确认 mooncake 真传了 KV(关键判据)
kubectl logs <P-pod> -c engine | grep -iE "kv_producer|mooncake|transfer|remote_decode"
kubectl logs <D1-pod> -c engine | grep -iE "kv_consumer|mooncake|remote_prefill|pull"
#   两端都有 mooncake transfer/握手日志 = PD 真正互通,而非 P 单干
```

---

## 8. 排障对照

| 现象 | 原因 | 处理 |
|------|------|------|
| wings 日志无 `[PD external-lb]` | `DP_SIZE` 没被读成 >1(名字/版本/生成时机) | 见 [deploy-qwen3-30b-pd-dp-decode-group.md](./deploy-qwen3-30b-pd-dp-decode-group.md) §10;确认部署镜像读 `DP_SIZE` 还是 `PD_DP_SIZE` |
| 预测通但 D engine 无 mooncake 日志 | **没起 PD proxy** 或 proxy 端口填错 | 起 §5 的 proxy,端口对齐 §4 表 |
| `tuple object has no attribute shape` | 用了上游 `MooncakeConnector` | 用 `MooncakeConnectorV1`(注册表已是) |
| P/D KV 映射错/握手失败 | `prefill`/`decode` 块两端不一致 | 对齐 §3.2 铁律的四个值 |

---

## 9. 占位替换

| 占位 | 含义 |
|------|------|
| `<P_IP>` | P 节点 RDMA IP(= `HOST_IP`/`Master_IP`) |
| `<D1_IP>` / `<D2_IP>` | 两个 D pod IP;`NODE_IPS` 两端写法一致 |
| `<proxy_IP>` | 你跑 `load_balance_proxy_server_example.py` 的机器 |
