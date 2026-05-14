# Docker Compose 部署

Docker Compose 是 Wings-Control 的两种正式部署形态之一。本文面向当前 `wings-control` 交付形态：

- `wings-control:<version>`：控制面镜像，默认入口为 `bash /opt/wings-control/wings_start.sh`。
- `wings-accel:<version>`：独立加速包镜像，作为一次性初始化容器。
- Engine 镜像：实际推理引擎镜像，例如 vLLM、vLLM-Ascend、SGLang、MindIE。

示例统一使用 `Qwen3.5-27B`。启动字段优先使用 `wings_start.sh` CLI；只有无 CLI 字段的运行时变量使用 `environment`。

---

## 1. 前提条件

| 项 | 要求 |
|----|------|
| Docker Engine | 20.10+ |
| Docker Compose | V2 / Compose Spec；如不支持 `depends_on.condition`，请先单独运行 init 容器 |
| NVIDIA 场景 | NVIDIA Container Toolkit 已安装 |
| Ascend 场景 | 已按环境挂载 NPU 设备、驱动、CANN 与 HCCN 配置 |
| 模型路径 | 宿主机存在 `/data/models/Qwen3.5-27B`，容器内挂载为 `/models/Qwen3.5-27B` |

---

## 2. 可直接启动的 Compose 文件

保存为 `docker-compose.qwen35-27b.yml`：

```yaml
version: "3.8"

x-images:
  wings-control: &wings-control-image "wings-control:26.0.0"
  wings-accel: &wings-accel-image "wings-accel:26.0.0"
  engine: &engine-image "vllm/vllm-openai:latest"

x-model:
  model-name: &model-name "Qwen3.5-27B"
  model-path: &model-path "/models/Qwen3.5-27B"
  model-volume: &model-volume "/data/models/Qwen3.5-27B:/models/Qwen3.5-27B:ro"

services:
  accel-init:
    image: *wings-accel-image
    restart: "no"
    entrypoint: ["/bin/sh", "-c"]
    command:
      - |
        set -e
        mkdir -p /accel-volume
        cp -r /opt/packages/. /accel-volume/ 2>/dev/null || true
        echo "[accel-init] accel packages prepared"
    volumes:
      - accel-vol:/accel-volume

  wings-control:
    image: *wings-control-image
    depends_on:
      accel-init:
        condition: service_completed_successfully
    network_mode: "host"
    restart: unless-stopped
    environment:
      WINGS_DEVICE: nvidia
      WINGS_DEVICE_COUNT: "2"
      WINGS_DEVICE_NAME: H20-96G
      ENGINE_PORT: "17000"
      HEALTH_PORT: "19000"
      MONITOR_PROXY_PORT: "19100"
      ENABLE_REASON_PROXY: "true"
    command:
      - bash
      - /opt/wings-control/wings_start.sh
      - --model-name
      - *model-name
      - --model-path
      - *model-path
      - --engine
      - vllm
      - --port
      - "18000"
      - --input-length
      - "8192"
      - --output-length
      - "2048"
      - --device-count
      - "2"
      - --model-type
      - llm
      - --trust-remote-code
      - --gpu-memory-utilization
      - "0.9"
      - --max-num-seqs
      - "32"
      - --max-num-batched-tokens
      - "8192"
      - --enable-prefix-caching
      - --enable-chunked-prefill
    volumes:
      - shared-vol:/shared-volume
      - accel-vol:/accel-volume:ro
      - *model-volume

  engine:
    image: *engine-image
    depends_on:
      - wings-control
    network_mode: "host"
    shm_size: "16g"
    restart: unless-stopped
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: "0,1"
    entrypoint: ["/bin/sh", "-c"]
    command:
      - |
        set -e
        timeout=3600
        elapsed=0
        echo "[engine] waiting for /shared-volume/start_command.sh"
        while [ ! -f /shared-volume/start_command.sh ]; do
          sleep 2
          elapsed=$$((elapsed + 2))
          if [ $$elapsed -ge $$timeout ]; then
            echo "[engine] ERROR: timeout waiting for /shared-volume/start_command.sh"
            exit 1
          fi
        done
        cat /shared-volume/start_command.sh
        exec bash /shared-volume/start_command.sh
    volumes:
      - shared-vol:/shared-volume
      - accel-vol:/accel-volume:ro
      - *model-volume

volumes:
  shared-vol:
  accel-vol:
```

---

## 3. 启动与验证

```bash
docker compose -f docker-compose.qwen35-27b.yml up -d
docker compose -f docker-compose.qwen35-27b.yml logs -f wings-control engine
```

检查生成的引擎启动脚本：

```bash
docker compose -f docker-compose.qwen35-27b.yml exec wings-control cat /shared-volume/start_command.sh
```

健康检查：

```bash
curl http://127.0.0.1:19000/health
curl http://127.0.0.1:19000/v1/health
curl http://127.0.0.1:18000/v1/models
```

停止：

```bash
docker compose -f docker-compose.qwen35-27b.yml down
```

---

## 4. 特性字段示例：分布式

分布式不是独立部署形态。它是在 Compose 中额外启用的特性。

以下示例是控制容器内可直接执行的完整启动命令。落到 Compose 时，把命令前的环境变量合并到 `wings-control.environment`，把 `bash /opt/wings-control/wings_start.sh ...` 合并到 `wings-control.command`。

Master 节点：

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
RANK_IP="192.168.1.10" \
MASTER_IP="192.168.1.10" \
NODE_IPS="192.168.1.10,192.168.1.11" \
NNODES="2" \
HEAD_NODE_ADDR="192.168.1.10" \
RAY_HEAD_IP="192.168.1.10" \
DISTRIBUTED_EXECUTOR_BACKEND="ray" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --device-count 2 \
  --model-type llm \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --distributed
```

Worker 节点：

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
WINGS_DEVICE_NAME="H20-96G" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
RANK_IP="192.168.1.11" \
MASTER_IP="192.168.1.10" \
NODE_IPS="192.168.1.10,192.168.1.11" \
NNODES="2" \
HEAD_NODE_ADDR="192.168.1.10" \
RAY_HEAD_IP="192.168.1.10" \
DISTRIBUTED_EXECUTOR_BACKEND="ray" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --port 18000 \
  --input-length 8192 \
  --output-length 2048 \
  --device-count 2 \
  --model-type llm \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --distributed
```

---

## 5. 注意事项

- 需要替换镜像仓库、镜像版本、模型宿主机路径、设备 ID 和端口规划。
- `ENABLE_REASON_PROXY` 当前必须保持 `true`。
- `ENGINE_PORT`、`HEALTH_PORT`、`MONITOR_PROXY_PORT` 当前作为运行时环境变量配置。
- NVIDIA 新版 Compose 可将 `runtime: nvidia` 改为 `gpus` 字段。
- Ascend 场景需要把 `WINGS_DEVICE=nvidia` 改为 `ascend`，并补充 NPU 设备、驱动、CANN、HCCN 等挂载。
