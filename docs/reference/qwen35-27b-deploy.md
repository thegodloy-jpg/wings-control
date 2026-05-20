# Qwen3.5-27B 部署示例

> 状态：历史参考副本。当前正式入口请以 [../../README.md](../../README.md)、[../deployment/docker-compose.md](../deployment/docker-compose.md)、[../deployment/k8s.md](../deployment/k8s.md) 和 [../examples/qwen35-27b/README.md](../examples/qwen35-27b/README.md) 为准。

本文是 `Qwen3.5-27B` 的专项部署参考，统一遵循 [../../README.md](../../README.md) 的当前口径：

- 不再以单独 `docker run` 作为主线。
- 使用 Docker Compose 或 K8s 编排 `wings-accel`、`wings-control`、Engine 三类容器。
- `wings_start.sh` 支持的启动项统一以 CLI 字段展示。
- 仅无 CLI 字段的运行时变量保留为环境变量。

---

## 1. 模型与镜像

| 项 | 示例值 | 说明 |
|----|--------|------|
| 模型名 | `Qwen3.5-27B` | API 请求中的 served model name |
| 宿主机路径 | `/data/models/Qwen3.5-27B` | 按实际环境替换 |
| 容器路径 | `/models/Qwen3.5-27B` | 与 `--model-path` 保持一致 |
| control 镜像 | `wings-control:26.0.0` | 独立控制面镜像 |
| accel 镜像 | `wings-accel:26.0.0` | 独立加速包镜像 |
| engine 镜像 | `vllm/vllm-openai:latest` | 可替换为实际 vLLM / vLLM-Ascend / SGLang / MindIE 镜像 |

---

## 2. Docker Compose 直接启动

推荐直接使用 [../examples/qwen35-27b/docker-compose.qwen35-27b.yml](../examples/qwen35-27b/docker-compose.qwen35-27b.yml) 模板。

启动：

```bash
docker compose -f docs/examples/qwen35-27b/docker-compose.qwen35-27b.yml up -d
docker compose -f docs/examples/qwen35-27b/docker-compose.qwen35-27b.yml logs -f wings-control engine
```

检查启动脚本：

```bash
docker compose -f docs/examples/qwen35-27b/docker-compose.qwen35-27b.yml exec wings-control cat /shared-volume/start_command.sh
```

验证服务：

```bash
curl http://127.0.0.1:19000/health
curl http://127.0.0.1:18000/v1/models
```

---

## 3. K8s 直接启动

推荐直接使用 [../examples/qwen35-27b/qwen35-27b.yaml](../examples/qwen35-27b/qwen35-27b.yaml) 模板，并按实际集群替换镜像、模型卷、GPU/NPU 资源名和调度规则。

启动：

```bash
kubectl apply -f docs/examples/qwen35-27b/qwen35-27b.yaml
kubectl rollout status deploy/qwen35-27b
kubectl logs -f deploy/qwen35-27b -c wings-control
```

验证：

```bash
kubectl port-forward svc/qwen35-27b 18000:18000 19000:19000
curl http://127.0.0.1:19000/health
curl http://127.0.0.1:18000/v1/models
```

---

## 4. 控制容器内 CLI 启动字段

以下命令用于调试 `wings-control` 镜像内的启动逻辑。实际 Compose/K8s 部署中，请把 CLI 字段写入 `command` / `args`，把环境变量写入 `environment` / `env`。

```bash
WINGS_DEVICE="nvidia" \
WINGS_DEVICE_COUNT="2" \
ENGINE_PORT="17000" \
HEALTH_PORT="19000" \
MONITOR_PROXY_PORT="19100" \
ENABLE_REASON_PROXY="true" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --chip h20-96 \
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
  --enable-chunked-prefill
```

---

## 5. 分布式补充字段

Master 节点示例：

```bash
RANK_IP="192.168.1.10" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --device-count 2 \
  --port 18000 \
  --distributed \
  --master-ip "192.168.1.10" \
  --node-ips "192.168.1.10,192.168.1.11" \
  --head-node-addr "192.168.1.10" \
  --ray-head-ip "192.168.1.10" \
  --nnodes 2 \
  --distributed-executor-backend ray
```

Worker 节点只需要替换 `RANK_IP`，其余分布式字段保持一致：

```bash
RANK_IP="192.168.1.11" \
bash /opt/wings-control/wings_start.sh \
  --model-name "Qwen3.5-27B" \
  --model-path "/models/Qwen3.5-27B" \
  --engine vllm \
  --device-count 2 \
  --port 18000 \
  --distributed \
  --master-ip "192.168.1.10" \
  --node-ips "192.168.1.10,192.168.1.11" \
  --head-node-addr "192.168.1.10" \
  --ray-head-ip "192.168.1.10" \
  --nnodes 2 \
  --distributed-executor-backend ray
```