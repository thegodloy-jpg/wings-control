# 启动流程顺序：proxy(18000) / health(19000) / engine(17000)

> 范围：launcher 入口 `wings_control.py`，standalone 主路径。
> 核心：launcher **不直接启动引擎**；引擎跑在同 Pod 的另一个容器，靠共享卷脚本异步拉起。

## 一、三个端口角色

| 端口 | 角色 | 进程 | 容器 |
|------|------|------|------|
| 17000 | backend 推理服务 | vllm/sglang/mindie | engine 容器 |
| 18000 | 对外反向代理 | `proxy.gateway:app` | launcher 容器 |
| 19000 | 独立健康探针 | `proxy.health_service:app` | launcher 容器 |

（附带 19100 `monitor_proxy`，透传引擎监控接口。）

## 二、启动顺序（`_launch_attempt`，wings_control.py:1172-1204）

```
① build_launcher_plan()      生成引擎启动脚本文本（含 17000 配置）
② _write_start_command()     写 /shared-volume/start_command.sh
        └─► engine 容器读脚本 → 加载权重 → 监听 17000（分钟级，异步）
③ _build_processes()         构造 [proxy, health, monitor_proxy]
        └─► node_rank>0 时过滤掉 proxy（仅 rank0 暴露）
④ for proc: _start(proc)     proxy(18000) → health(19000) → monitor(19100)
⑤ sleep(2) + 崩溃检查         立即退出则抛错，进入重试
⑥ _daemon_loop()             守护循环，子进程崩溃自动重启
```

**关键点：代码顺序 ≠ 端口就绪顺序**

- 步骤 ② 脚本最先写，所以引擎"逻辑上"最早开始启动；
- 但 launcher **不阻塞等待 17000 就绪**，立即起 18000/19000；
- 实际端口可用先后：**18000、19000 数秒就绪；17000 等模型加载完才 listening（分钟级）**。

## 三、时序图

```
时间轴 ──────────────────────────────────────────────►

launcher  ②写脚本   ④起 proxy/health
容器        │          │ 18000 ✔  19000 ✔
            │          │ (立即就绪，不等引擎)
            │          │
engine      └─读脚本──► 加载模型权重… ───────► 17000 ✔
容器           (同卷)        (数分钟)         (引擎 ready)

           ◄──── 此空窗期 ────►
           18000 收到请求 → 502（后端未就绪）
           19000 探 17000/health → 返回 201 starting
                  （STARTUP_GRACE_MS 默认 1 小时宽限）
```

## 四、运行时数据流（单向汇聚到 17000）

```
外部流量 ─► proxy(18000) ──转发──► engine(17000)
                              BACKEND_URL=http://{RANK_IP|127.0.0.1}:17000

K8s 探针 ─► health(19000) ──轮询──► engine(17000)/health  驱动状态机
                          ──TCP──► 17000               engine_alive 诊断
```

env 注入见 `_build_child_env`（wings_control.py:345）：proxy/health/monitor 共享
`BACKEND_URL`(→17000)、`PORT`(18000)、`HEALTH_PORT`(19000)。

health 状态码映射（`map_http_code_from_state`，health_router.py:419）：
- 引擎未就绪且在宽限期内 → **201 starting**
- 引擎 ready → **200**
- 退化 → **503**；宽限超时 → **502**

## 五、分布式差异（一图概括）

| | proxy(18000) | health | BACKEND_URL → | 引擎脚本 |
|---|---|---|---|---|
| standalone | ✔ | 19000 | 本地 17000 | 启动时自写 |
| master | ✔ | 19000 | 本地 17000 | 启动时写 rank0 |
| worker | ✘ 不起 | **19001**（+1 避冲突） | **master:17000** | master 分发后才写 |

> worker 只保留 health 且端口偏移 +1，探 master 的 17000 间接代表 Ray 集群健康。

## 六、一句话总结

> 先写引擎脚本（17000 在另一容器异步拉起）→ 立刻起 proxy(18000)、health(19000)，
> 不等引擎就绪。18000 转发流量到 17000，19000 轮询 17000/health 把"启动中/就绪/退化"
> 翻译给 K8s 探针，靠 201 + 1 小时宽限兜底引擎慢启动。
