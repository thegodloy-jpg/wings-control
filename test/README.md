# PROXY_WORKERS (\_MAX_PROXY_WORKERS=128) 性能影响验证

## 1. 测试目标

验证 `proxy_config.py` 中 `_MAX_PROXY_WORKERS = 128` 对 proxy 代理性能的影响，包括：

1. 不同 `PROXY_WORKERS` 值（1, 4, 16, 64, 128）对串行延迟的影响
2. 不同 `PROXY_WORKERS` 值对并发吞吐 (QPS) 的影响
3. 不同 `PROXY_WORKERS` 值的内存消耗
4. 配额分配 (`_split_strict`) 的实际效果

## 2. 代码分析

### 2.1 关键代码路径

```python
# proxy_config.py L91-95
_MAX_PROXY_WORKERS = 128
WORKERS = min(int(os.getenv("PROXY_WORKERS", "128")), _MAX_PROXY_WORKERS)
WORKER_INDEX = int(os.getenv("WORKER_INDEX", "-1"))
```

### 2.2 Worker 对配额分配的影响

`_split_strict(total, workers, idx)` 将全局配额均分给每个 worker：

| Workers | LOCAL_PASS_THROUGH (每worker) | LOCAL_QUEUE (每worker) | GATE0_LOCAL | GATE1_LOCAL | 总有效并发 | 内存估算 |
|---------|-------------------------------|------------------------|-------------|-------------|-----------|---------|
| 1       | 1024                          | 1024                   | 1           | 1023        | 1024      | ~65MB   |
| 4       | 256                           | 256                    | 1           | 255         | 1024      | ~260MB  |
| 16      | 64                            | 64                     | 1           | 63          | 1024      | ~1GB    |
| 64      | 16                            | 16                     | 1           | 15          | 1024      | ~4GB    |
| 128     | 8                             | 8                      | 1           | 7           | 1024      | ~8GB    |

> **注意**: 总有效并发 = WORKERS × LOCAL_PASS_THROUGH_LIMIT，理论上应恒等于 GLOBAL_PASS_THROUGH_LIMIT=1024

### 2.3 已发现的默认值不一致问题

| 文件 | 变量 | 默认值 | 影响 |
|------|------|--------|------|
| `proxy_config.py` L94 | `PROXY_WORKERS` env 默认 | `"128"` | 配额分配用 128 分 |
| `wings_control.py` L427 | launcher 默认 | `"4"` | 实际启动 4 个 worker |

当 `PROXY_WORKERS` 未显式设置时，launcher 启动 4 个 worker，但每个 worker 以为有 128 个，
导致每个 worker 只分到 `1024/128=8` 的配额，4 个 worker 总共只有 32 的有效并发。

## 3. 测试环境

- **机器**: 7.6.52.148 (a100)
- **硬件**: 64 CPU (Intel Xeon Gold 6444Y), 219GB RAM, 1× A100-PCIE-40GB
- **OS**: Ubuntu 22.04.5 LTS, kernel 5.15.0-171-generic
- **容器运行时**: containerd 1.7.28
- **k8s**: v1.28.2 (单节点)
- **镜像**: `wings-control:test_new`

## 4. 测试方法

### 4.1 架构

```
  bench_workers.py  ──→  proxy (k8s Pod)  ──→  mock_backend.py (host)
  (host 进程)            (hostNetwork)          (port 17000)
                         (port 18000)
```

- **mock backend**: 轻量 FastAPI 服务，返回固定响应，消除后端引擎的变量
- **proxy**: 以不同 `PROXY_WORKERS` 值部署在 k8s Pod 中（使用 hostNetwork）
- **bench client**: 从 host 发送请求，串行测延迟、并发测吞吐

### 4.2 测试场景

| 场景 | 类型 | 说明 |
|------|------|------|
| normal | 非流式 | 标准 2 条消息的 chat completion |
| stream | 流式 | 标准流式请求 |
| large_messages | 非流式 | 50 轮对话 (101 条消息) |
| tool_calls | 非流式 | assistant content=null + tool_calls |

### 4.3 并发级别

- 串行: 1 请求接 1 请求，测量 RTT
- 并发: 1, 10, 50, 100 并发

### 4.4 K8s 资源

- 命名空间: `workers-perf-zhanghui`
- Deployment 命名: `proxy-w{N}-zhanghui` (N = worker 数)
- 使用 `hostNetwork: true` 避免 kube-proxy 开销

## 5. 执行测试

### 5.1 快速测试（约 10 分钟）

```bash
cd /home/zhanghui/workers-perf-test
bash run_workers_perf_test.sh quick
```

### 5.2 完整测试（约 30 分钟）

```bash
cd /home/zhanghui/workers-perf-test
bash run_workers_perf_test.sh
```

### 5.3 指定 worker 列表

```bash
WORKERS_LIST="1 4 128" bash run_workers_perf_test.sh quick
```

## 6. 测试文件说明

| 文件 | 说明 |
|------|------|
| `mock_backend.py` | Mock vLLM backend，固定返回，消除引擎变量 |
| `bench_workers.py` | 性能测试客户端，串行延迟 + 并发吞吐 |
| `gen_workers_report.py` | 对比报告生成器 |
| `run_workers_perf_test.sh` | 主测试脚本，自动部署 k8s + 运行 benchmark |
| `results/` | 测试结果 JSON 和报告 |

## 7. 预期结论

1. **串行延迟**: 各 worker 配置下差异不大（均 < 5ms），因为串行场景不触发排队
2. **并发吞吐**:
   - Worker 数 1→4 有显著 QPS 提升（多进程并行处理）
   - Worker 数 4→16 中等提升
   - Worker 数 16→64→128 提升递减（CPU 竞争增加）
3. **内存**: 线性增长，128 workers ≈ 8GB
4. **128 上限合理性**: 作为定义上限值本身合理，但实际部署不应默认 128，而应按需配置
