# Wings-Control

Wings-Control 是统一推理控制 Sidecar。它不直接运行 vLLM、vLLM-Ascend、SGLang 或 MindIE，而是负责解析启动字段、探测或读取硬件上下文、合并默认配置和用户配置、生成 `/shared-volume/start_command.sh`，并托管 Proxy、Health、Monitor Proxy 等控制面服务。真实推理 Engine 容器等待并执行这个启动脚本。

## 运行链路

```text
CLI / env / config-file / hardware_info.json
        |
        v
wings_start.sh
        |
        v
python -m wings_control
  - 解析启动参数
  - 读取硬件信息
  - 自动选择或校验引擎
  - 合并默认配置、模型配置和用户配置
  - 写出 /shared-volume/start_command.sh
  - 启动 Proxy / Health / Monitor Proxy
        |
        v
Engine 容器执行 /shared-volume/start_command.sh
```

## 交付组件

| 组件 | 职责 |
|------|------|
| 控制面镜像 | 运行 `bash /opt/wings-control/wings_start.sh`，生成 Engine 启动脚本并托管控制面服务。部署文档通常写作 `wings-control:<version>`；当前构建脚本的历史产物名为 `fusionregistry:5000/wings-infer:<version>` |
| `wings-accel:<version>` | 可选加速包镜像，通常作为 Compose 初始化容器或 K8s `initContainer`，把加速包准备到 `/accel-volume` |
| Engine 镜像 | 真实推理运行时，例如 vLLM、vLLM-Ascend、SGLang、MindIE |

## 支持范围

正式部署形态只分为两类：

1. Docker Compose
2. K8s

单机多卡、多机分布式、PD 分离、LMCache、Sparse KV、投机推理、Function Call、RAG 加速、Wings Router 等属于特性或能力场景，不作为独立部署形态描述。

支持的 `--engine` 值来自 `wings_control/core/start_args_compat.py`：

| 引擎值 | 典型硬件 | 适配器 |
|--------|----------|--------|
| `vllm` | NVIDIA | `wings_control/engines/vllm_adapter.py` |
| `vllm_ascend` | Ascend | 复用 vLLM 适配器，追加 Ascend 环境和参数 |
| `sglang` | NVIDIA | `wings_control/engines/sglang_adapter.py` |
| `mindie` | Ascend | `wings_control/engines/mindie_adapter.py` |

## 快速入口

| 目标 | 文档 |
|------|------|
| 了解产品边界和运行链路 | [产品总览](docs/product-overview.md) |
| 查看兼容性口径和人工维护矩阵 | [兼容性矩阵](docs/compatibility.md) |
| 使用 Docker Compose 部署 | [Docker Compose 部署](docs/deployment/docker-compose.md) |
| 使用 K8s 部署 | [K8s 部署](docs/deployment/k8s.md) |
| 使用 PD 分离、分布式、LMCache、Sparse KV 等能力 | [特性索引](docs/features/index.md) |
| 查看完整文档目录 | [docs/README.md](docs/README.md) |

兼容性结论应以当前 `wings_control/utils/model_utils.py`、`wings_control/config/defaults/`、`tests/`、`docs/examples/` 和实际部署验证为准；手写矩阵需要随代码和配置同步维护。

## 关键目录

| 路径 | 说明 |
|------|------|
| `wings_control/wings_start.sh` | 兼容历史 Wings 启动入口的 Shell 包装脚本 |
| `wings_control/wings_control.py` | Sidecar launcher 主流程，负责写脚本和守护子服务 |
| `wings_control/core/start_args_compat.py` | `python -m wings_control` 的 CLI 契约和环境变量回退 |
| `wings_control/core/config_loader.py` | 默认配置、模型配置、用户配置和 CLI 覆盖的合并逻辑 |
| `wings_control/core/hardware_detect.py` | 从 `/shared-volume/hardware_info.json` 或环境变量读取硬件上下文 |
| `wings_control/engines/` | vLLM、SGLang、MindIE 启动脚本生成适配器 |
| `wings_control/proxy/` | OpenAI 兼容代理、健康检查和监控代理 |
| `wings_control/config/defaults/` | 引擎、硬件和模型架构默认配置 |
| `build/` | `wings-accel` 与控制面镜像构建脚本 |

## 配置优先级

`wings_start.sh` 层的优先级是：

```text
CLI 参数 > 环境变量 > 脚本默认值
```

进入 Python launcher 后，配置合并的有效优先级是：

```text
引擎/硬件/模型默认配置 < --config-file 指定的用户 JSON < CLI/环境变量启动字段 < 适配器生成的运行时参数
```

硬件信息优先从 `WINGS_HARDWARE_FILE` 指向的 JSON 读取，默认路径为 `/shared-volume/hardware_info.json`。文件不存在或读取失败时，回退到 `WINGS_DEVICE` / `DEVICE` / `HARDWARE_TYPE`、`WINGS_DEVICE_COUNT` / `DEVICE_COUNT`、`WINGS_DEVICE_NAME` 等环境变量。

## 默认端口

| 服务 | 默认端口 | 覆盖方式 | 用途 |
|------|----------|----------|------|
| Engine | `17000` | `ENGINE_PORT` | Engine 容器内真实推理服务 |
| Proxy | `18000` | `--port` / `PROXY_PORT` / `PORT` | OpenAI 兼容 API 入口 |
| Health | `19000` | `HEALTH_PORT` | `/health` 与 `/v1/health` |
| Monitor Proxy | `19100` | `MONITOR_PROXY_PORT` | 透传 Engine 侧监控接口 |

当前 `wings_control.py` 的 v4 MVP 要求启用 Proxy，`ENABLE_REASON_PROXY=false` 会直接返回错误。

## 共享卷契约

| 路径 | 生产方 | 消费方 | 说明 |
|------|--------|--------|------|
| `/shared-volume/start_command.sh` | `wings-control` | Engine 容器 | 最终引擎启动脚本 |
| `/shared-volume/progress.jsonl` | Engine / log analyzer | 控制面和运维侧 | 启动进度记录 |
| `/shared-volume/advanced_features.json` | `wings-control` | 控制面和运维侧 | 加速特性使能状态（使能 + 变体），`/v1/startup/accel` 数据源 |
| `/shared-volume/lmcache_config.yaml` | vLLM 适配器 | Engine 容器 | LMCache Offload 开启时生成的配置 |
| `/accel-volume` | `wings-accel` | 控制面和 Engine 容器 | 可选加速包目录 |

## 最小启动示例

`wings_start.sh` 支持的参数以脚本内 `usage()` 和 `case` 分支为准；当前没有 `--chip` 参数。卡型或硬件详情通过环境变量或 `hardware_info.json` 传入。

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --device-count 2 \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --trust-remote-code
```

在 Compose 或 K8s 中，优先把有 CLI 形式的字段放入 `command` / `args`，把运行时环境放入 `environment` / `env`，例如 `ENGINE_PORT`、`HEALTH_PORT`、`MONITOR_PROXY_PORT`、`WINGS_DEVICE`、`WINGS_DEVICE_COUNT`、`WINGS_DEVICE_NAME`、`WINGS_HARDWARE_FILE`、`RANK_IP`、`MASTER_IP`、`NODE_IPS`、`PD_ROLE`、`LMCACHE_*`、`WINGS_ROUTE_*`。

通过 `wings_start.sh` 传递的常用 CLI 字段：

| 参数 | 环境变量回退 | 是否必填 | 默认值 | 说明 |
|------|--------------|----------|--------|------|
| `--model-name` | `MODEL_NAME` | 是 | 无 | API 暴露的模型名；`wings_start.sh` 未提供时会退出 |
| `--model-path` | `MODEL_PATH` | 否 | `/weights` | 模型权重目录；目录下的 `config.json` 会被 `model_utils.py` 读取 |
| `--engine` | `ENGINE` | 否 | `vllm` | 推理引擎，支持 `vllm`、`vllm_ascend`、`sglang`、`mindie` |
| `--port` | `PORT` / `PROXY_PORT` | 否 | `18000` | Proxy 对外端口；Engine 端口由 `ENGINE_PORT` 控制 |
| `--host` | `HOST` | 否 | 空 / `0.0.0.0` | 服务监听地址，通常不需要显式传入 |
| `--model-type` | `MODEL_TYPE` | 否 | `auto` | `auto`、`llm`、`embedding`、`rerank` 等；`auto` 时由 `model_utils.py` 推断 |
| `--save-path` | `SAVE_PATH` | 否 | `/opt/wings/outputs` | 生成输出目录，主要用于多模态/扩展场景 |
| `--input-length` | `INPUT_LENGTH` | 否 | `4096` | 最大输入长度 |
| `--output-length` | `OUTPUT_LENGTH` | 否 | `1024` | 最大输出长度 |
| `--max-num-seqs` | `MAX_NUM_SEQS` | 否 | `32` | 最大并发序列数 |
| `--max-num-batched-tokens` | `MAX_NUM_BATCHED_TOKENS` | 否 | `4096` | 最大批处理 token 数 |
| `--dtype` | `DTYPE` | 否 | `auto` | 权重/计算 dtype |
| `--kv-cache-dtype` | `KV_CACHE_DTYPE` | 否 | `auto` | KV cache dtype |
| `--quantization` | `QUANTIZATION` | 否 | 空 | 量化方式 |
| `--quantization-param-path` | `QUANTIZATION_PARAM_PATH` | 否 | 空 | 量化参数路径 |
| `--gpu-memory-utilization` | `GPU_MEMORY_UTILIZATION` | 否 | `0.9` | 显存利用率上限 |
| `--block-size` | `BLOCK_SIZE` | 否 | `16` | KV block size；Ascend prefix cache 场景可能被代码修正 |
| `--seed` | `SEED` | 否 | `0` | 随机种子 |
| `--trust-remote-code` | `TRUST_REMOTE_CODE` | 否 | 关闭 | 信任模型仓库自定义代码 |
| `--enable-chunked-prefill` | `ENABLE_CHUNKED_PREFILL` | 否 | 关闭 | 开启 chunked prefill |
| `--enable-prefix-caching` | `ENABLE_PREFIX_CACHING` | 否 | 关闭 | 开启 prefix cache |
| `--enable-expert-parallel` | `ENABLE_EXPERT_PARALLEL` | 否 | 关闭 | MoE 专家并行 |
| `--enable-speculative-decode` | `ENABLE_SPECULATIVE_DECODE` | 否 | 关闭 | 开启投机推理 |
| `--speculative-decode-model-path` | `SPECULATIVE_DECODE_MODEL_PATH` | 否 | 空 | 草稿模型路径；只在投机推理草稿模型策略中需要 |
| `--enable-sparse` | `ENABLE_SPARSE` | 否 | 关闭 | 开启 Sparse KV 相关逻辑 |
| `--enable-rag-acc` | `ENABLE_RAG_ACC` | 否 | 关闭 | 开启 RAG 加速相关链路 |
| `--enable-auto-tool-choice` | `ENABLE_AUTO_TOOL_CHOICE` | 否 | 关闭 | 开启 Function Call / Tool Choice |
| `--distributed` | `DISTRIBUTED` | 否 | 关闭 | 开启分布式入口；拓扑用 `RANK_IP`、`MASTER_IP`、`NODE_IPS`、`NNODES` 等环境变量 |
| `--device-count` | `DEVICE_COUNT` / `WINGS_DEVICE_COUNT` | 否 | `1` | 当前实例使用的 GPU/NPU 数 |
| `--config-file` | `CONFIG_FILE` | 否 | 空 | 用户自定义 JSON 配置文件 |
| `--gpu-usage-mode` | `GPU_USAGE_MODE` | 否 | `full` | GPU 使用模式，供配置选择逻辑使用 |

`python -m wings_control` 的内部 parser 还支持 `--nnodes`、`--node-ips`、`--nodes`、`--master-ip`、`--head-node-addr`、`--distributed-executor-backend`、`--ray-head-ip`、`--compilation-config` 等字段；默认镜像入口经过 `wings_start.sh` 时，分布式拓扑仍建议用 `RANK_IP`、`MASTER_IP`、`NODE_IPS`、`NNODES` 等环境变量传入。

## 启动后验证

查看生成的引擎启动脚本：

```bash
cat /shared-volume/start_command.sh
```

健康和 API 检查：

```bash
curl http://127.0.0.1:19000/health
curl http://127.0.0.1:19000/v1/health
curl http://127.0.0.1:18000/v1/models
```

查看控制面日志：

```bash
cat /var/log/wings/wings_start.log
```

## 构建

本地调试可使用根目录的 `Dockerfile.simple` 构建控制面镜像：

```bash
docker build -f Dockerfile.simple -t wings-control:test .
```

交付构建的设计入口在 `build/build.sh`，默认版本为 `26.0.0`，也可显式传入版本：

```bash
bash build/build.sh 26.0.0
```

注意：当前 `build/build.sh` 中调用的控制面构建脚本名和实际文件名存在历史命名不一致，`build/wings-control/` 下实际脚本为 `build_wings_accel.sh`，但其内容是在构建控制面镜像。直接使用交付构建脚本前需要先修正这个命名差异，或单独进入 `build/wings-control/` 调用现有脚本。

## 常见问题

| 现象 | 优先检查 |
|------|----------|
| `/shared-volume/start_command.sh` 未生成 | `wings-control` 日志、`MODEL_NAME`、`MODEL_PATH`、`ENGINE`、`--config-file` 路径 |
| `--chip` 报未知参数 | 当前脚本没有 `--chip`；改用 `WINGS_DEVICE_NAME` 或 `/shared-volume/hardware_info.json` |
| Health 未就绪 | Engine 容器是否执行脚本、`ENGINE_PORT` 是否一致、Engine 是否监听成功 |
| Proxy 返回 502 | Engine 是否监听 `ENGINE_PORT`，Proxy 与 Engine 是否共享网络命名空间或可互通 |
| `ENABLE_REASON_PROXY=false` 后退出 | 当前 v4 MVP 不支持关闭 Proxy |
| 分布式校验失败 | `DISTRIBUTED=true`、`RANK_IP`、`MASTER_IP`、`NODE_IPS`、`NNODES` 是否一致且节点间互通 |
| Ascend 启动失败 | NPU 设备、驱动挂载、CANN 环境、`WINGS_DEVICE=ascend`、`WINGS_DEVICE_NAME`、`--engine vllm_ascend` 或 `--engine mindie` |
| LMCache 未生效 | `LMCACHE_OFFLOAD=true`、容量相关 `LMCACHE_*` 变量、`/shared-volume/lmcache_config.yaml` 是否生成 |
