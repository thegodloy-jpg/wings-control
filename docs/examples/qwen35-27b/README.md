# Qwen3.5-27B 示例文件

本目录提供可复制修改的 Qwen3.5-27B 部署示例：

- [docker-compose.qwen35-27b.yml](docker-compose.qwen35-27b.yml)：Docker Compose 三容器模板。
- [qwen35-27b.yaml](qwen35-27b.yaml)：K8s Deployment + Service 模板。

使用前必须按实际环境替换镜像仓库、镜像版本、模型宿主机路径、GPU/NPU 资源名和端口规划。

## Docker Compose

```bash
docker compose -f docs/examples/qwen35-27b/docker-compose.qwen35-27b.yml up -d
docker compose -f docs/examples/qwen35-27b/docker-compose.qwen35-27b.yml logs -f wings-control engine
```

## K8s

```bash
kubectl apply -f docs/examples/qwen35-27b/qwen35-27b.yaml
kubectl rollout status deploy/qwen35-27b
kubectl logs -f deploy/qwen35-27b -c wings-control
```