# MaaS ↔ wings_control 交互契约清单

> MaaS（上层编排）通过 **三类机制** 向 sidecar 形态的 wings_control 注入运行时上下文：CLI 参数、环境变量、共享卷 JSON 文件。CLI > 环境变量 > 代码默认值；硬件信息反过来 JSON > 环境变量 > 默认。

## 来源标识（贯穿全文）

| 标签 | 含义 |
|---|---|
| **页面** | MaaS UI / 上层编排显式下发（用户或平台运营可配置） |
| **内部** | wings-control 自身默认值或工程调优项（不暴露页面，仅运维通过环境变量临时覆盖） |
| **混合** | 页面可下发；wings 同时保留合理默认值 |

---

## 一、参数字段（CLI）— MaaS 调用 wings_start.sh 时下发

入口：[wings_control/wings_start.sh](../wings_start.sh)。该脚本同时 `export` 为同名环境变量并以 `python -m wings_control <APP_ARGS>` 透传到内部 argparse。

> 本节 **全部为「页面」来源**（CLI 由 MaaS 编排时拼接）。

### 1. 模型与运行基础

| CLI 参数 | 同步环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `--model-name` | `MODEL_NAME` | (必填) | 模型逻辑名 |
| `--model-path` | `MODEL_PATH` | `/weights` | 权重目录 |
| `--save-path` | `SAVE_PATH` | `/opt/wings/outputs` | 输出目录 |
| `--engine` | `ENGINE` | — | vllm/sglang/mindie/vllm-ascend |
| `--model-type` | `MODEL_TYPE` | — | llm / embedding / rerank / mmum / mmgm |
| `--host` / `--port` | `HOST` / `PORT` | host 空 / 18000 | 对外监听 |
| `--config-file` | `CONFIG_FILE` | — | 用户自定义引擎配置 |

### 2. 推理参数透传

`--dtype`、`--kv-cache-dtype`、`--quantization`、`--quantization-param-path`、`--gpu-memory-utilization`、`--block-size`、`--max-num-seqs`、`--max-num-batched-tokens`、`--seed`、`--input-length`、`--output-length`、`--trust-remote-code`、`--enable-chunked-prefill`、`--enable-prefix-caching`、`--enable-expert-parallel`、`--enable-auto-tool-choice`

### 3. 资源 / 拓扑

| CLI | 环境变量 | 含义 |
|---|---|---|
| `--device-count` | `DEVICE_COUNT` | 卡数（与 hardware_info.json 的 `count` 互为补充） |
| `--gpu-usage-mode` | `GPU_USAGE_MODE` | GPU 使用模式 |
| `--distributed` | `DISTRIBUTED=true` | 多机模式开关 |

### 4. 加速特性

| CLI | 环境变量 |
|---|---|
| `--enable-speculative-decode` | `ENABLE_SPECULATIVE_DECODE=true` |
| `--speculative-decode-model-path` | `SPECULATIVE_DECODE_MODEL_PATH` |
| `--enable-sparse` | `ENABLE_SPARSE=true` |
| `--enable-rag-acc` | `ENABLE_RAG_ACC=true`（同步置 `RAG_ACC_ENABLED`） |

---

## 二、环境变量 — K8s ConfigMap / Pod env 直接注入

不通过 CLI，wings_control 在 [utils/env_utils.py](../utils/env_utils.py)、[config/settings.py](../config/settings.py) 及各 adapter 内读取。下表覆盖 wings_control 全部 `os.getenv` 调用点。

### 1. 引擎与版本（基础元数据）

| 变量 | 来源 | 默认 | 说明 |
|---|---|---|---|
| `ENGINE` | 页面 | `vllm` | 引擎类型，CLI `--engine` 同步导出 |
| `ENGINE_VERSION` | 页面 | `""` | 引擎版本字符串，MaaS 部署时下发，用于稳定版本回退匹配；[version_util.py:48](../core/version_util.py#L48) |
| `ENGINE_HOST` | 内部 | `127.0.0.1` | engine 容器监听地址（sidecar 同 Pod） |
| `ENGINE_PORT` | 内部 | `17000` | engine 容器真实监听端口 |
| `BACKEND_URL` | 内部 | `http://127.0.0.1:17000` | proxy / monitor 透传目标 |
| `BACKEND_HOST` / `BACKEND_PORT` | 内部 | `127.0.0.1` / `17000` | log_analyzer 探测 engine |
| `BACKEND_PID_FILE` | 内部 | `/var/log/wings/wings.txt` | 写引擎 PID/版本 |
| `WINGS_VERSION` | 内部 | `25.0.0.1` | wings-control 自身版本（构建时烤入） |
| `WINGS_BUILD_DATE` | 内部 | `2025-08-30` | wings-control 构建日期 |
| `WINGS_ENGINE_PATCH_OPTIONS` | 混合 | `""` | wings-accel 补丁选项（投机/稀疏/卸载等），由页面特性开关派生或运维覆盖 |
| `WINGS_CONFIG_DIR` | 混合 | `""` | 用户自定义 recipe/config 目录（K8s 挂卷可配） |
| `WINGS_STABLE_VERSIONS_DIR` | 内部 | `""` | 稳定版本配置目录覆盖 |

### 2. 分布式拓扑（核心契约） [wings_control.py:529-570](../wings_control.py#L529-L570)

| 变量 | 来源 | 默认 | 用途 |
|---|---|---|---|
| `DISTRIBUTED` | 页面 | `false` | 角色判定总开关 |
| `RANK_IP` | 页面 | hostname 反查 | **本 Pod IP，MaaS 每 Pod 唯一下发** |
| `POD_IP` | 页面 | `""` | 当前 Pod IP（K8s downward API），RANK_IP 备选 |
| `MASTER_IP` | 页面 | — | Master Pod IP / DNS 名 |
| `NODE_IPS` | 页面 | — | 全集群节点 IP，逗号分隔 |
| `MASTER_PORT` | 页面 | — | HCCL / Ray master 端口 |
| `SERVER_PORT` / `WORKER_PORT` | 页面 | — | 推理服务 / Worker 端口 |
| `COORDINATOR_PORT` | 内部 | 取 `MASTER_PORT` | 分布式协调端口，避免与 HCCL MASTER_PORT 语义冲突 |
| `VLLM_DISTRIBUTED_PORT` | 混合 | — | vLLM Ray worker 端口 |
| `SGLANG_DISTRIBUTED_PORT` / `SGLANG_DIST_PORT` | 混合 | `28030` | SGLang 节点间端口 |
| `RAY_PORT` | 内部 | `28020` | Ray head 端口 |

### 3. 硬件信息回退（JSON 不存在时）

| 变量 | 来源 | 默认 | 说明 |
|---|---|---|---|
| `WINGS_HARDWARE_FILE` | 内部 | `/shared-volume/hardware_info.json` | 路径覆盖 |
| `WINGS_DEVICE` / `DEVICE` / `HARDWARE_TYPE` | 页面 | `nvidia` | 设备类型 |
| `WINGS_DEVICE_COUNT` / `DEVICE_COUNT` | 页面 | `1` | 卡数 |
| `WINGS_DEVICE_NAME` | 页面 | — | 设备型号名 |
| `WINGS_DEVICE_MEMORY` | 页面 | `""` | 单卡显存（GB），用于 VRAM 校验 |
| `WINGS_H20_MODEL` | 页面 | `""` | H20 型号提示，影响默认 recipe |

### 4. 加速特性总闸（页面开关）

| 变量 | 来源 | 默认 |
|---|---|---|
| `ENABLE_SPECULATIVE_DECODE` / `SD_ENABLE` | 页面 | `false` |
| `ENABLE_SPARSE` / `SPARSE_ENABLE` | 页面 | `false` |
| `LMCACHE_OFFLOAD` | 页面 | `false` |
| `ENABLE_RAG_ACC` / `RAG_ACC_ENABLED` | 页面 | `false` |
| `ENABLE_OPERATOR_ACCELERATION` | 页面 | `false` |
| `ENABLE_SOFT_FP8` / `ENABLE_SOFT_FP4` | 页面 | `false` |
| `CONFIG_FORCE` | 页面 | `false` |

### 5. LMCache（KV 卸载）细化

| 变量 | 来源 | 默认 |
|---|---|---|
| `LMCACHE_QAT` | 页面 | `false`（依赖 `LMCACHE_OFFLOAD=true`） |
| `LMCACHE_COLD_START` | 页面 | `false` |
| `LMCACHE_LOCAL_CPU` / `LMCACHE_MAX_LOCAL_CPU_SIZE` | 页面 | — |
| `LMCACHE_LOCAL_DISK` / `LMCACHE_MAX_LOCAL_DISK_SIZE` | 页面 | —（QAT 必填） |
| `LMCACHE_QAT_INSTANCE_NUM` | 混合 | `2` |
| `LMCACHE_QAT_LOSS_LEVEL` | 混合 | `0` |
| `LMCACHE_QAT_LOG_ENABLED` | 内部 | `0` |
| `LMCACHE_QAT_MODULE` | 内部 | `kv_agent`（vllm 强制） |
| `LMCACHE_CHUNK_SIZE` | 内部 | `256` |
| `LMCACHE_PRE_CACHING_HASH` | 内部 | `sha256_cbor` |
| `LMCACHE_MANIFEST_WRITE_INTERVAL` | 内部 | `1` |
| `LMCACHE_ENGINE_ID` | 内部 | `lmca1` |

### 6. PD 分离（Prefill/Decode） [config_loader.py:819-830](../core/config_loader.py#L819-L830)

| 变量 | 来源 | 默认 |
|---|---|---|
| `PD_ROLE` | 页面 | `""`（`P`/`D`） |
| `PD_CONNECTOR_TYPE` | 混合 | `MooncakeConnectorV1` |
| `PD_PREFILL_TP_SIZE` | 页面 | TP 默认值 |
| `PD_PREFILL_DP_SIZE` | 页面 | `1` |
| `PD_PREFILL_PP_SIZE` | 页面 | `1` |
| `PD_DECODE_TP_SIZE` | 页面 | TP 默认值 |
| `PD_DECODE_DP_SIZE` | 页面 | `1` |
| `PD_DECODE_PP_SIZE` | 页面 | `1` |

### 7. 引擎适配器调优

**MindIE** [engines/mindie_adapter.py](../engines/mindie_adapter.py)

| 变量 | 来源 | 默认 |
|---|---|---|
| `MINDIE_MASTER_PORT` | 内部 | `27070` |
| `HCCL_IP_EXCHANGE_PORT` | 内部 | `27071` |
| `MINDIE_LONG_CONTEXT_THRESHOLD` | 内部 | `8192` |
| `MINDIE_HEALTH_HOST` / `MINDIE_HEALTH_PORT` | 内部 | `127.0.0.2` / `1026` |
| `MINDIE_NPU_DEVICE_IDS` | 混合 | `""` |
| `MINDIE_DISTRIBUTED_ENV_SCRIPT_PATH` | 内部 | `""` |
| `MINDIE_DISTRIBUTED_ENV_DEFAULTS_PATH` | 内部 | `""` |
| `RANK_TABLE_PATH` | 混合 | `""` |

**vLLM / vLLM-ascend** [engines/vllm_adapter.py](../engines/vllm_adapter.py)

| 变量 | 来源 | 默认 |
|---|---|---|
| `VLLM_LLMDD_RPC_PORT` | 内部 | `5569` |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | 内部 | `23000` |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | 内部 | `12345` |
| `VLLM_DP_RPC_PORT` | 内部 | `13355` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 内部 | `7200` |
| `RAY_RESOURCE_FLAG` | 内部 | `""` |
| `RAY_CGRAPH_get_timeout` | 内部 | `3600` |
| `ASCEND_ENFORCE_EAGER` | 内部 | `false` |
| `NPU_MAX_SPLIT_SIZE_MB` | 内部 | `256` |

### 8. 集合通信与网络（HCCL / NCCL / GLOO）

| 变量 | 来源 | 默认 |
|---|---|---|
| `HCCL_SOCKET_IFNAME` | 混合 | `eth0` |
| `GLOO_SOCKET_IFNAME` | 混合 | `eth0` |
| `NCCL_SOCKET_IFNAME` | 混合 | `eth0` |
| `NETWORK_INTERFACE` | 混合 | 同 GLOO_SOCKET_IFNAME |
| `HCCL_CONNECT_TIMEOUT` | 内部 | `1800` |
| `HCCL_EXEC_TIMEOUT` | 内部 | `7200` |
| `HCCL_BUFFSIZE` | 内部 | `1024` |
| `OMP_NUM_THREADS` | 内部 | `10` / `100`（按场景） |

### 9. Wings Router（NATS 路由元数据）

| 变量 | 来源 | 默认 |
|---|---|---|
| `WINGS_ROUTE_ENABLE` | 页面 | `false` |
| `WINGS_ROUTE_INSTANCE_GROUP_NAME` | 页面 | — |
| `WINGS_ROUTE_INSTANCE_NAME` | 页面 | —（启用 group 时必填） |
| `WINGS_ROUTE_NATS_PATH` | 页面 | —（启用 group 时必填） |

### 10. 容器 / 代理 / 端口

| 变量 | 来源 | 默认 |
|---|---|---|
| `SHARED_VOLUME_PATH` | 内部 | `/shared-volume`（K8s 挂卷路径） |
| `ENABLE_REASON_PROXY` | 混合 | `true` |
| `PROXY_PORT` / `PORT` | 页面 | `18000` |
| `HEALTH_PORT` / `HEALTH_SERVICE_PORT` | 内部 | `19000` |
| `MONITOR_PROXY_PORT` | 内部 | `19100` |
| `PROXY_WORKERS` | 内部 | `4`（封顶） |
| `WORKER_INDEX` | 内部 | `-1`（uvicorn worker 编号） |
| `APP_WORKDIR` | 内部 | `/opt/wings-control` |

### 11. Proxy 调优（HTTP/2、超时、队列、Warmup）

> 本节 **全部为「内部」来源**（工程调优项，不暴露页面）。位于 [proxy/proxy_config.py](../proxy/proxy_config.py) 与 [proxy/health_router.py](../proxy/health_router.py)。

| 类别 | 变量 |
|---|---|
| HTTPX | `HTTPX_MAX_CONNECTIONS` / `HTTPX_MAX_KEEPALIVE` / `HTTPX_KEEPALIVE_EXPIRY` / `HTTPX_CONNECT_TIMEOUT` / `HTTPX_WRITE_TIMEOUT` / `HTTPX_POOL_TIMEOUT` |
| HTTP/2 | `HTTP2_ENABLED` / `HTTP2_MAX_STREAMS` |
| 流式 | `FAST_PATH_BYTES` / `FIRST_FLUSH_BYTES` / `FIRST_FLUSH_MS` / `STREAM_FLUSH_BYTES` / `STREAM_FLUSH_MS` / `NONSTREAM_PIPE_THRESHOLD` / `STREAM_BACKEND_CONNECT_TIMEOUT` / `ENABLE_DELIM_FLUSH` |
| 重试 / 限流 | `RETRY_TRIES` / `RETRY_INTERVAL_MS` / `GLOBAL_PASS_THROUGH_LIMIT` / `GLOBAL_QUEUE_MAXSIZE` / `QUEUE_TIMEOUT` / `QUEUE_REJECT_POLICY` / `QUEUE_OVERFLOW_MODE` |
| 全局栅 | `USE_GLOBAL_GATE` / `GATE_SOCK` / `GATE_EARLY_RELEASE` |
| Warmup | `WARMUP_CONN` / `WARMUP_PROMPT` / `WARMUP_ROUNDS` / `WARMUP_TIMEOUT` / `WARMUP_CONNECT_TIMEOUT` / `WARMUP_REQUEST_TIMEOUT` |
| Backend 探测 | `BACKEND_PROBE_TIMEOUT` / `METRICS_CONNECT_TIMEOUT` / `STATUS_CONNECT_TIMEOUT` / `STATUS_READ_TIMEOUT` |
| 健康轮询 | `HEALTH_TIMEOUT_MS` / `PRE_READY_POLL_MS` / `POLL_INTERVAL_MS` / `HEALTH_CACHE_MS` / `STARTUP_GRACE_MS` / `FAIL_THRESHOLD` / `FAIL_GRACE_MS` / `HEALTH_JITTER_PCT` / `ENGINE_TCP_TIMEOUT` |
| SGLang 健康 | `SGLANG_FAIL_BUDGET` / `SGLANG_PID_GRACE_MS` / `SGLANG_DECAY` / `SGLANG_SILENCE_MAX_MS` / `SGLANG_CONSEC_TIMEOUT_MAX` |

### 12. 日志

| 变量 | 来源 | 默认 |
|---|---|---|
| `LOG_DIR` | 混合 | `/var/log/wings` |
| `LOG_FILE_PATH` | 内部 | `/var/log/wings/wings_control.log` |
| `LOG_LEVEL` | 混合 | `INFO` |
| `LOG_STDERR_LEVEL` | 内部 | 同 `LOG_LEVEL` |
| `LOG_DEDUP_WINDOW_SEC` | 内部 | `60` |
| `LOG_PATCH_DISABLE` | 内部 | `false` |
| `LOG_SPEAKER_INDEXES` | 内部 | `""` |
| `HEALTH_PATH_REGEX` / `HEALTH_ACCESS_DROP_REGEX` / `OUTBOUND_HEALTH_DROP_REGEX` / `BATCH_NOISE_REGEX` / `PYNVML_NOISE_REGEX` | 内部 | 噪音过滤正则 |

### 一致性校验 [check_env()](../utils/env_utils.py#L401-L433)

- `LMCACHE_QAT=true` ⇒ 必须 `LMCACHE_OFFLOAD=true` 且配 `LMCACHE_LOCAL_DISK` + `LMCACHE_MAX_LOCAL_DISK_SIZE`
- 配 `WINGS_ROUTE_INSTANCE_GROUP_NAME` ⇒ 必须配 `WINGS_ROUTE_INSTANCE_NAME` + `WINGS_ROUTE_NATS_PATH`
- `DISTRIBUTED=true` 且 `MASTER_IP` 缺失 ⇒ 回退 `standalone` 并 warn

---

## 三、JSON 文件 — `/shared-volume/` 共享卷交换

### 1. `hardware_info.json`（**页面** → wings-control，输入）

- **来源**：页面（由 MaaS 编排层在 Pod 启动前写入共享卷）
- **路径**：`/shared-volume/hardware_info.json`（可由 `WINGS_HARDWARE_FILE` 覆盖）
- **读取**：[core/hardware_detect.py:_load_hardware_from_file](../core/hardware_detect.py#L100-L140) / [utils/device_utils.py:_get_hardware_info](../utils/device_utils.py#L58-L114)
- **实际内容**（Pod `serving-rn-ajst72-uummtvwi-64569f5f87-n7gnv:/shared-volume`）：

```
root@serving-rn-ajst72-uummtvwi-64569f5f87-n7gnv:/shared-volume# cat hardware_info.json
{"count":1,"details":[],"units":"GB","device":"ascend"}
```

- **解析后内存表示**：

```json
{"count":1,"details":[],"units":"GB","device":"ascend"}
```

- **字段处理**（[hardware_detect.py:120-140](../core/hardware_detect.py#L120-L140)）：
  - `device` 必填，别名归一（`gpu`/`cuda` → `nvidia`，`npu` → `ascend`，未识别 → `nvidia`）
  - `count` 必填正整数；非法回退到 `len(details)`，再回退 `1`
  - `details` 缺省 `[]`
  - `units` 缺省 `"GB"`
  - 缺 `device` / `count` 抛 `ValueError`；解析失败不崩溃，回退环境变量

### 2. `advanced_features.json`（wings-control → **页面**，输出 / 状态汇报）

- **来源**：内部（由 wings-control 启动流程写出，供页面读取）
- **路径**：`/shared-volume/advanced_features.json`
- **写入**：[core/wings_entry.py:_write_advanced_features_json](../core/wings_entry.py#L906-L929)（`json.dump(..., indent=4)`）
- **实际内容**（按代码逻辑，全特性默认未启用 + engine=vllm 时）：

```json
{
    "engine": "vllm",
    "features": {
        "speculative_decode": false,
        "sparse_kv": false,
        "kv_offload": false,
        "rag_acc": false
    }
}
```

- 启用对应 CLI / env 后，相应布尔位为 `true`；补丁安装失败时由 shell 单行脚本回写为 `false`

---

## 总览：数据流（标注来源）

```
┌──── MaaS 编排层（页面） ────┐
│                             │
│  1) CLI args（页面） ─────────────────┐
│     wings_start.sh                   ▼
│  2) Pod env（页面 / 混合）──► wings-control 容器（含内部默认）
│     DISTRIBUTED / RANK_IP            │  读：CLI / env / hardware_info.json
│     MASTER_IP / NODE_IPS             │  写：advanced_features.json
│     ENGINE / ENGINE_VERSION          │
│     LMCACHE_OFFLOAD 等特性开关        │
│     WINGS_ROUTE_*                    │
│                                      ▼
│  3) hardware_info.json（页面）─┐
│              /shared-volume/ (K8s 共享卷)
│                                ▲  │
│                                │  ▼
│  ◄── advanced_features.json ───│ wings-engine 容器
└────────────────────────────────┘
```

**关键规则**

- 优先级：**CLI > 环境变量 > 代码默认值**；硬件信息反过来：**JSON > 环境变量 > 默认**
- 「页面」标签 = 真正属于 MaaS / 用户配置面板的字段
- 「内部」标签 = wings-control 自身固化的默认 / 工程调优项，**不应该出现在 MaaS 页面**
- 「混合」标签 = 默认值合理可直接跑，但页面也允许用户/运维覆盖
- 跨容器数据交换全部经过 `SHARED_VOLUME_PATH`，无直连网络协议
- MaaS 不回调 wings-control HTTP；状态汇报通过共享卷文件 + health 端口 `19000` 轮询
