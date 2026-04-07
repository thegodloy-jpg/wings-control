# Wings-Control 统一推理控制 Sidecar

> **引擎**: vLLM · vLLM-Ascend · SGLang · MindIE · Wings  
> **硬件**: NVIDIA GPU · Ascend 910B NPU  
> **模式**: 单机 · 分布式 (Ray/HCCL/nnodes) · Master-Worker  
> **兼容**: 与 wings/wings_start.sh 100% CLI 兼容

---

## 快速开始

```bash
# 1. 构建镜像
cd infer-control-sidecar-unified/
docker build -t wings-control:latest wings-control/

# 2. 单容器测试（仅生成启动脚本）
docker run --rm -it \
  -e WINGS_SKIP_PID_CHECK=true \
  -p 18000:18000 -p 19000:19000 \
  wings-control:latest \
  --model-name test-model --model-path /weights

# 3. 验证
curl http://localhost:19000/health
```

更多场景参见 [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

## 架构

```
┌─ K8s Pod ──────────────────────────────────────────────┐
│  (可选) initContainer: accel-init → /accel-volume/     │
│                                                         │
│  wings-control (Sidecar)          engine 容器             │
│  ┌────────────────────┐   ┌──────────────────────┐     │
│  │ wings_start.sh     │   │ 等待 start_command.sh│     │
│  │  → python -m app.main │ │  → bash 执行         │     │
│  │  → 生成脚本 ───────┼──→│  → serve :17000      │     │
│  │  → proxy :18000    │   │                      │     │
│  │  → health :19000   │   │                      │     │
│  └────────────────────┘   └──────────────────────┘     │
│              共享卷: /shared-volume/                     │
└─────────────────────────────────────────────────────────┘
```

**数据流**: CLI/环境变量 → `wings_start.sh` → `app.main` → 配置合并(4层) → 引擎适配器 → `start_command.sh` → engine 容器执行

**配置优先级**: CLI 参数 > 环境变量 > 用户配置文件 > 模型特定配置 > 硬件默认配置

---

## 支持矩阵

| 引擎 | 单机 | 分布式 | 硬件 | K8s Overlay |
|------|------|--------|------|-------------|
| vllm | ✅ | ✅ Ray/DP | NVIDIA GPU | `vllm-single/` · `vllm-distributed/` |
| vllm_ascend | ✅ | ✅ Ray | Ascend 910B | `vllm-ascend-single/` · `vllm-ascend-distributed/` |
| sglang | ✅ | ✅ nnodes | NVIDIA GPU | `sglang-single/` · `sglang-distributed/` |
| mindie | ✅ | ✅ HCCL | Ascend 910B | `mindie-single/` · `mindie-distributed/` |
| wings | ✅ | — | GPU/NPU | — |

**自动引擎选择**: Ascend + vllm → `vllm_ascend` · mmgm 模型 → `wings` · embedding/rerank + Ascend → `vllm_ascend`

---

## 项目结构

```
infer-control-sidecar-unified/
├── .env.example                      # 环境变量模板
├── wings_control/                    # 后端控制服务 (Python 包)
│   ├── Dockerfile                    # Sidecar 镜像
│   ├── wings_start.sh                # 启动入口 (ENTRYPOINT)
│   ├── wings_control.py              # 主入口 (角色分发 / 进程守护)
│   ├── health.py                     # 健康检查 (独立 uvicorn :19000)
│   ├── requirements.txt
│   ├── config/                       # 配置 (settings.py)
│   │   └── defaults/                 # 引擎默认 JSON 配置
│   ├── core/                         # 核心 (config_loader · wings_entry · engine_manager · hardware_detect · port_plan · start_args_compat)
│   ├── engines/                      # 适配器 (vllm_adapter · sglang_adapter · mindie_adapter)
│   ├── distributed/                  # 分布式 (master · worker · monitor · scheduler)
│   ├── proxy/                        # 当前代理 (gateway · health_router · health_service · http_client · queueing · proxy_config · monitor_proxy · tags)
│   ├── proxy-new/                    # 重构版代理 (新增 priority_queue · metrics_poller · prefix_affinity · request_preprocessor · token_estimator · stream_compress)
│   │                                 # 详见 [wings_control/proxy-new/README.md](wings_control/proxy-new/README.md)
│   ├── rag_acc/                      # RAG 加速 (rag_app · stream_collector · document_processor · templates)
│   └── utils/                        # 工具 (env_utils · file_utils · model_utils · noise_filter · log_config · process_utils)
│                                     # ⚠️ device_utils.py 为遗留代码，未被调用
│                                     #    实际硬件检测由 core/hardware_detect.py 通过环境变量完成
├── wings-accel/                      # 加速包 (可选 initContainer)
│   └── build-accel-image.sh          # Accel 镜像构建脚本
├── k8s/{base,overlays/}              # Kustomize (8 个部署 overlay)
└── docs/                             # 详细文档 (deploy/ · verify/)
```

---

## 部署

### Docker Compose (单机)

```yaml
services:
  wings-control:
    image: wings-control:latest
    ports: ["18000:18000", "19000:19000"]
    environment:
      ENGINE: vllm
      MODEL_NAME: DeepSeek-R1-Distill-Qwen-1.5B
      MODEL_PATH: /models/DeepSeek-R1-Distill-Qwen-1.5B
      WINGS_SKIP_PID_CHECK: "true"
      BACKEND_URL: http://engine:17000
    volumes: [shared-vol:/shared-volume, /path/to/models:/models:ro]

  engine:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    command: /bin/sh -c "while [ ! -f /shared-volume/start_command.sh ]; do sleep 2; done; bash /shared-volume/start_command.sh"
    volumes: [shared-vol:/shared-volume, /path/to/models:/models:ro]

volumes:
  shared-vol:
```

### K8s (Kustomize)

```bash
# 单机
kubectl apply -k k8s/overlays/vllm-single/

# 分布式
kubectl apply -k k8s/overlays/vllm-distributed/
```

### Docker 命令行

```bash
# 单机
docker run --runtime nvidia -p 18000:18000 -p 19000:19000 \
  -v /models:/models:ro wings-control:latest \
  --model-name Qwen2-7B --model-path /models/Qwen2-7B --engine vllm

# 分布式 rank-0
# 角色判定: RANK_IP == MASTER_IP → master，RANK_IP != MASTER_IP → worker
docker run --network host -e DISTRIBUTED=true -e NNODES=2 \
  -e RANK_IP=192.168.1.100 -e MASTER_IP=192.168.1.100 \
  -e HEAD_NODE_ADDR=192.168.1.100 \
  wings-control:latest --model-name DeepSeek-R1 --model-path /models/DeepSeek-R1 --distributed
```

---

## 端口规划

| 端口 | 用途 | 暴露 |
|------|------|------|
| 17000 | 推理引擎 | Pod 内部 |
| 18000 | API 代理 (OpenAI 兼容) | NodePort/LB |
| 19000 | 健康检查 (K8s 探针) | 探针 |

分布式端口: Ray `28020` (env `RAY_PORT`) · SGLang `28030` (env `SGLANG_DIST_PORT`) · vLLM DP `13355` · MindIE `27070` · NIXL `5759`

---

## 健康检查

```bash
curl http://<host>:19000/health         # 200=就绪 201=启动中 502=失败 503=降级
curl http://<host>:19000/health/detail  # JSON 详情
```

```yaml
readinessProbe:
  httpGet: { path: /health, port: 19000 }
  initialDelaySeconds: 60
  periodSeconds: 10
  failureThreshold: 36
livenessProbe:
  httpGet: { path: /health, port: 19000 }
  initialDelaySeconds: 120
  periodSeconds: 30
  failureThreshold: 5
```

---

## CLI 参数

| 参数 | 环境变量 | 默认 | 说明 |
|------|----------|------|------|
| `--model-name` | `MODEL_NAME` | **必填** | 模型名 |
| `--model-path` | `MODEL_PATH` | `/weights` | 模型路径 |
| `--engine` | `ENGINE` | `vllm` | vllm/vllm_ascend/sglang/mindie/wings |
| `--port` | `PORT` | `18000` | 监听端口 |
| `--input-length` | `INPUT_LENGTH` | `4096` | 最大输入 |
| `--output-length` | `OUTPUT_LENGTH` | `1024` | 最大输出 |
| `--gpu-memory-utilization` | `GPU_MEMORY_UTILIZATION` | `0.9` | 显存利用率 |
| `--max-num-seqs` | `MAX_NUM_SEQS` | `32` | 最大并发序列 |
| `--dtype` | `DTYPE` | `auto` | 数据类型 |
| `--model-type` | `MODEL_TYPE` | `auto` | auto/llm/embedding/rerank/mmum/mmgm |
| `--distributed` | `DISTRIBUTED` | `false` | 分布式模式 |
| `--trust-remote-code` | `TRUST_REMOTE_CODE` | `true` | 信任远程代码 |
| `--enable-prefix-caching` | `ENABLE_PREFIX_CACHING` | `false` | 前缀缓存 |
| `--enable-chunked-prefill` | `ENABLE_CHUNKED_PREFILL` | `false` | 分块预填充 |
| `--enable-expert-parallel` | `ENABLE_EXPERT_PARALLEL` | `false` | MoE 专家并行 |
| `--enable-speculative-decode` | `ENABLE_SPECULATIVE_DECODE` | `false` | 推测解码 |
| `--enable-rag-acc` | `ENABLE_RAG_ACC` | `false` | RAG 加速 |
| `--enable-auto-tool-choice` | `ENABLE_AUTO_TOOL_CHOICE` | `false` | 函数调用 |
| `--config-file` | `CONFIG_FILE` | — | 自定义配置 |
| `--device-count` | `DEVICE_COUNT` | `1` | 设备数 |
| `--save-path` | `SAVE_PATH` | `/opt/wings/outputs` | 输出目录 |

完整 30+ 参数详见 `wings_start.sh --help`

---

## 环境变量速查

### Sidecar

| 变量 | 默认 | 说明 |
|------|------|------|
| `SHARED_VOLUME_PATH` | `/shared-volume` | 共享卷 |
| `WINGS_SKIP_PID_CHECK` | `false` | 跳过 PID 检查 (sidecar 必须 true) |
| `ENABLE_REASON_PROXY` | `true` | 启用代理 |
| `BACKEND_URL` | `http://127.0.0.1:17000` | 后端 URL |

### 分布式

| 变量 | 说明 |
|------|------|
| `DISTRIBUTED` | 是否分布式 |
| `NNODES` | 节点总数 |
| `RANK_IP` | 当前节点 IP（由 MaaS 上层传入，每个 Pod 唯一） |
| `MASTER_IP` | Master 节点 IP（角色判定: RANK_IP == MASTER_IP → master） |
| `HEAD_NODE_ADDR` | Head 节点 IP |
| `NODE_IPS` | 所有节点 IP (逗号分隔) |

### 硬件

| 变量 | 说明 |
|------|------|
| `WINGS_DEVICE` | nvidia/ascend |
| `WINGS_DEVICE_COUNT` | 设备数 |
| `WINGS_DEVICE_MEMORY` | 显存 GB (cuda_graph_sizes 计算) |

### 加速

| 变量 | 说明 |
|------|------|
| `ENABLE_ACCEL` | 启用 Accel 补丁注入 |
| `WINGS_ENGINE_PATCH_OPTIONS` | 覆盖补丁选项 (JSON) |

完整模板: [.env.example](.env.example)

---

## Accel 加速包

可选的 initContainer，将 `wings_engine_patch` 注入 engine 容器：

```bash
bash wings-accel/build-accel-image.sh  # 构建 wings-accel:latest
```

启用: 设置 `ENABLE_ACCEL=true`，sidecar 自动注入 `WINGS_ENGINE_PATCH_OPTIONS`

详情: [docs/deploy/deploy-accel.md](docs/deploy/deploy-accel.md)

---

## Master-Worker 分布式

`main.py` 自动判断角色:

| 条件 | 角色 | 行为 |
|------|------|------|
| `DISTRIBUTED=false` | standalone | 直接生成脚本 + proxy/health |
| `DISTRIBUTED=true` + `MASTER_IP=本机` | master | FastAPI 协调 (注册/心跳/调度) |
| `DISTRIBUTED=true` + `MASTER_IP≠本机` | worker | 注册 → 等待指令 → 生成脚本 |

调度策略: `least_load` (默认) · `round_robin` · `random`

---

## 从 wings 迁移

```yaml
# 原版 wings — 单容器
containers:
  - name: wings
    image: wings:latest
    args: ["--model-name", "DeepSeek-R1", "--engine", "vllm"]

# 迁移后 unified — 双容器 (参数不变)
volumes:
  - name: shared-volume
    emptyDir: {}
containers:
  - name: wings-control
    image: wings-control:latest
    args: ["--model-name", "DeepSeek-R1", "--engine", "vllm"]
    env: [{name: WINGS_SKIP_PID_CHECK, value: "true"}]
    volumeMounts: [{name: shared-volume, mountPath: /shared-volume}]
  - name: engine
    image: vllm/vllm-openai:latest
    command: ["/bin/sh", "-c", "while [ ! -f /shared-volume/start_command.sh ]; do sleep 2; done; bash /shared-volume/start_command.sh"]
    volumeMounts: [{name: shared-volume, mountPath: /shared-volume}]
```

---

## 故障排查

| 症状 | 解决 |
|------|------|
| health 持续 201 | 引擎启动慢，增大 failureThreshold |
| health 502 | 引擎启动失败，查 engine 容器日志 |
| proxy 502 | 确认 BACKEND_URL 和 ENGINE_PORT |
| start_command.sh 未生成 | 查 wings-control 日志 |
| 分布式卡住 | 检查 HEAD_NODE_ADDR 可达性 |
| Ascend 用了 vllm | 设置 WINGS_DEVICE=ascend |

```bash
# 常用调试
kubectl exec -it deploy/infer -c wings-control -- cat /shared-volume/start_command.sh
kubectl exec -it deploy/infer -c wings-control -- curl localhost:19000/health/detail
```

详情: [docs/troubleshooting.md](docs/troubleshooting.md)

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [快速上手](docs/QUICKSTART.md) | 6 步构建到推理 |
| [架构详解](docs/architecture.md) | 模块·端口·状态机 |
| [故障排查](docs/troubleshooting.md) | 9 类问题 |
| [vLLM](docs/deploy/deploy-vllm.md) · [vLLM-Ascend](docs/deploy/deploy-vllm-ascend.md) · [SGLang](docs/deploy/deploy-sglang.md) · [MindIE](docs/deploy/deploy-mindie.md) | 引擎部署 |
| [Accel](docs/deploy/deploy-accel.md) | 加速包 |
| [版本差异](docs/version-diff-report.md) | wings vs unified |
| [Bug 修复](docs/BUG_FIX_REPORT.md) | 9 Bug 详情 |
| [安全审计R4](docs/security-audit-fix-report.md) | 第四轮安全专项 |
| [质量+分布式审计R5](docs/code-quality-distributed-audit-r5.md) | 第五轮代码质量与分布式逻辑 |

---

## Changelog

### [Unreleased] — 移除 fschat 依赖，消除 422 问题

**背景**

当 `RAG_ACC_ENABLED=true` 时，代理在 `handle_rag_scenario()` 中使用 fastchat 的
`ChatCompletionRequest(**payload_dict)` 对请求进行 Pydantic 校验。fastchat 的 `content`
字段类型为 `str`，导致以下两类合法请求被拒绝并返回 422：

- 多模态消息：`content: [{"type": "text", "text": "..."}]`（数组格式）
- tool_calls 场景：`content: null`（assistant 消息）

这些格式在 **vLLM v0.12.0** 的原生 `ChatCompletionRequest`（基于 openai Python 库的
TypedDict union，`ConfigDict(extra="allow")`）中完全合法，422 仅由 sidecar 自身的
fastchat 校验触发。

**变更**

| 文件 | 变更内容 |
|------|------|
| `proxy/gateway.py` | 新增 `_DictAttrView`；`handle_rag_scenario()` 由 `ChatCompletionRequest(**d)` 改为 `_DictAttrView(d)`；删除 fastchat import |
| `rag_acc/rag_app.py` | 删除 fastchat import；放宽函数参数类型注解 |
| `rag_acc/extract_dify_info.py` | 删除 fastchat import；放宽函数参数类型注解 |
| `requirements.txt` | 删除 `fschat>=0.2.36` |

**`_DictAttrView` 设计**

```python
class _DictAttrView:
    """Dict wrapper allowing attribute-style access (obj.key → dict[key]).
    Missing keys return None, matching Pydantic optional field defaults."""
    def __init__(self, data: dict) -> None: self._data = data
    def __getattr__(self, name: str): return self._data.get(name)
```

`rag_acc` 内部代码已有 `isinstance(msg, dict)` / `_get_msg_field()` 兼容逻辑，
无需修改任何 rag_acc 业务逻辑，存量 RAG 加速功能完全不受影响。

**行为对比**

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| 普通请求，RAG_ACC=false | ✅ 正常 | ✅ 正常 |
| 普通请求，RAG_ACC=true | ✅ 正常 | ✅ 正常 |
| multimodal content，RAG_ACC=false | ✅ 正常（vLLM 原生接受） | ✅ 正常 |
| multimodal content，RAG_ACC=true | ❌ 422（fastchat 拒绝） | ✅ 正常 |
| content: null，RAG_ACC=true | ❌ 422（fastchat 拒绝） | ✅ 正常 |

---

### [Unreleased] — 性能优化：byte pre-scan + 消除重复 JSON 解析

**背景**

在高并发短输入场景（LLM 推理延迟约 50-200ms），代理层的固定开销占总延迟比例更大。
性能测试数据（gateway-perf-test）显示：引入 `_normalize_messages()` 后，普通请求
P99 延迟上升约 35%，null_content 请求 P99 上升约 102%。

两处根因：
1. `RequestPreprocessor.preprocess()` 对每个请求无条件遍历 messages 列表，查找需要归一化的 content，而实际上绝大多数请求 content 为纯字符串，无需处理。
2. `RAG_ACC_ENABLED=True` 时，`handle_rag_scenario()` 内对同一 body 执行第二次 `json.loads()`，而 `RequestPreprocessor` 已完成首次解析。

**变更**

| 文件 | 变更内容 |
|------|------|
| `proxy/request_preprocessor.py` | 新增 `_needs_normalization(body_bytes)` 静态方法；在调用 `_normalize_messages()` 前做字节级预扫描，命中 `"content":[` / `"content":null` 等模式才进入遍历 |
| `proxy/gateway.py` | `handle_rag_scenario()` 新增 `preparse_payload: dict \| None = None` 参数；`chat_completions()` 传入 `result.payload`，消除第二次 `json.loads()` |

**优化原理与量化**

```
普通请求（content: str）:
  旧路径: json.loads → _normalize_messages 遍历（空跑）→ ...
  新路径: json.loads → _needs_normalization 字节扫描 (~1-5μs) → 跳过遍历

RAG 流式请求:
  旧路径: preprocess json.loads → handle_rag_scenario json.loads（重复）
  新路径: preprocess json.loads → handle_rag_scenario 直接使用 result.payload
```

Python 内置 `in` 字节搜索使用 C 层 Boyer-Moore 算法。对于 1KB body 约 1μs，10KB body 约 5μs，远低于一次 `json.loads()`（约 50-500μs）。

**行为不变性**

- 字节扫描为误报安全（false positive → 进入 `_normalize_messages()` 再精确判断）
- false negative 理论上极小（仅当 `"content":[` 等字节串恰好出现在 string 值内，且 content 字段本身也是数组时才会误判为不需归一化，实际场景几乎不存在）
- `handle_rag_scenario()` 在没有 `preparse_payload` 时（直接调用路径）仍回退到本地 `json.loads()`，向后兼容

---

### [Unreleased] — 多引擎适配性修复：ENGINE 环境变量、SGLang 指标、abort 路径

**背景**

代理层的多个内置优化模块（KV Cache 准入控制、客户端断连 abort、健康状态机）在设计时以 vLLM 为基准，对新引擎适配存在三个问题：
1. **MindIE** 不支持 `/v1/abort`，`asyncio.create_task` 裸调导致 task 内 404 异常得不到捕获，产生 `Task exception was never retrieved` 日志噪音。
2. **SGLang** 指标名称为 `sglang:token_usage`，与现有正则 `vllm:gpu_cache_usage_perc` 不匹配，`cache_pct` 始终为 0，准入控制对 SGLang 完全失效。
3. **SGLang** abort 端点为 `POST /cancel_batch`，现有代码发送到 `/v1/abort` 和 vLLM 共用发送方式，导致 404。
4. **MindIE sidecar 模式**下 PID 文件对 control 容器不可见，健康探测的 `_is_mindie()` / `_is_sglang()` 判断失效，无法切换到正确的远端健康端口。

**变更**

| 文件 | 变更内容 |
|------|------|
| `proxy/proxy_config.py` | 新增 `ENGINE = os.getenv("ENGINE", "vllm")` 全局引擎类型常量 |
| `proxy/gateway.py` | `_abort_backend_request()` 新增 ENGINE 分支：MindIE 直接跳过；SGLang 使用 `/cancel_batch`；task 内部包装独立 `_do_abort()` 协程带 try/except |
| `proxy/health_router.py` | `_is_mindie()` / `_is_sglang()` 优先读取 `ENGINE` 环境变量，无法获取时降级到 PID 文件 |
| `proxy/metrics_poller.py` | 新增 SGLang 指标正则；`_parse_metrics()` 优先匹配 vLLM，未命中时降级到 SGLang；`ENABLED` 自动屏蔽 MindIE |
| `proxy/README.md` | 新建，包含目录、环境变量、请求流程图、引擎适配矩阵、快速启动示例 |

**ENGINE 环境变量设计**

```
ENGINE 可选值: vllm | vllm-ascend | sglang | mindie
未设置时默认为: vllm
```

**引擎适配矩阵（修复后）**

| 功能 | vllm | vllm-ascend | sglang | mindie |
|---|---|---|---|---|
| KV Cache 准入控制 | ✅ | ✅ | ✅ | ❌ 自动禁用 |
| abort 断连释放 GPU | ✅ `/v1/abort` | ✅ `/v1/abort` | ✅ `/cancel_batch` | ❌ 自动跳过 |
| 健康探测端口切换 | — | — | SGLang 専项参数 | ✅ `MINDIE_HEALTH_PORT` |
| SJF 调度 / gzip 压缩 / 预处理 | ✅ | ✅ | ✅ | ✅ |

### [Unreleased] — 代码质量/安全修复 + 分布式逻辑加固

**代码质量与安全 (13 项)**

| 级别 | 修复 | 文件 |
|------|------|------|
| CRITICAL | HTTP 请求超时、Shell 注入防护、JSON Schema 校验 | scheduler.py, wings_entry.py |
| HIGH | bare except 清理、全局状态锁、调度器竞态、socket 泄漏、RAG JSON 校验、路径穿越 | worker.py, scheduler.py, vllm_adapter.py, stream_collector.py, config_loader.py |
| MEDIUM | 统一超时配置常量 | settings.py |

**分布式逻辑加固 (4 项修复 + 3 项误报确认)**

| # | 修复 | 文件 |
|---|------|------|
| C1 | 添加 node_ips 去重校验 + nnodes 一致性检查 | distributed/master.py |
| C3 | MindIE 分布式 nnodes ≤ 1 警告 | core/config_loader.py |
| C4 | 修正 RAY_PORT 默认值文档 (6379→28020) | engines/vllm_adapter.py |
| H1 | Ascend Ray Worker 优先尝试已知 head_addr | engines/vllm_adapter.py |

误报确认：SGLang --host 绑定方向正确、Worker LaunchArgs 设计合理、Monitor 已有锁保护。

---

## License

MIT
