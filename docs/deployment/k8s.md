# K8s 部署

K8s 是 Wings-Control 的两种正式部署形态之一。推荐模式是 `wings-accel` 作为 `initContainer`，`wings-control` 和 Engine 容器放在同一个 Pod 内，通过 `emptyDir` 共享 `/shared-volume` 和 `/accel-volume`。

## 编排原则

1. `wings-accel` 初始化 `/accel-volume` 后退出。
2. `wings-control` 常驻运行，生成 `/shared-volume/start_command.sh`。
3. Engine 容器等待 `start_command.sh` 出现后执行。
4. Service 暴露 Proxy 端口，Health 探针使用 Health 端口。

## 最小结构

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen35-27b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qwen35-27b
  template:
    metadata:
      labels:
        app: qwen35-27b
    spec:
      volumes:
        - name: shared-vol
          emptyDir: {}
        - name: accel-vol
          emptyDir: {}
        - name: model-vol
          hostPath:
            path: /data/models/Qwen3.5-27B
            type: Directory
      initContainers:
        - name: wings-accel
          image: wings-accel:26.0.0
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -e
              mkdir -p /accel-volume
              cp -r /opt/packages/. /accel-volume/ 2>/dev/null || true
          volumeMounts:
            - name: accel-vol
              mountPath: /accel-volume
      containers:
        - name: wings-control
          image: wings-control:26.0.0
          env:
            - name: WINGS_DEVICE
              value: nvidia
            - name: WINGS_DEVICE_COUNT
              value: "2"
            - name: WINGS_DEVICE_NAME
              value: H20-96G
            - name: ENGINE_PORT
              value: "17000"
            - name: HEALTH_PORT
              value: "19000"
            - name: MONITOR_PROXY_PORT
              value: "19100"
            - name: ENABLE_REASON_PROXY
              value: "true"
          command: ["bash", "/opt/wings-control/wings_start.sh"]
          args:
            - --model-name
            - Qwen3.5-27B
            - --model-path
            - /models/Qwen3.5-27B
            - --engine
            - vllm
            - --device-count
            - "2"
            - --port
            - "18000"
            - --trust-remote-code
          volumeMounts:
            - name: shared-vol
              mountPath: /shared-volume
            - name: accel-vol
              mountPath: /accel-volume
              readOnly: true
            - name: model-vol
              mountPath: /models/Qwen3.5-27B
              readOnly: true
        - name: engine
          image: vllm/vllm-openai:latest
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -e
              while [ ! -f /shared-volume/start_command.sh ]; do sleep 2; done
              exec bash /shared-volume/start_command.sh
          volumeMounts:
            - name: shared-vol
              mountPath: /shared-volume
            - name: accel-vol
              mountPath: /accel-volume
              readOnly: true
            - name: model-vol
              mountPath: /models/Qwen3.5-27B
              readOnly: true
```

## 启动与验证

```bash
kubectl apply -f qwen35-27b.yaml
kubectl rollout status deploy/qwen35-27b
kubectl logs -f deploy/qwen35-27b -c wings-control
kubectl logs -f deploy/qwen35-27b -c engine
```

检查启动脚本：

```bash
kubectl exec deploy/qwen35-27b -c wings-control -- cat /shared-volume/start_command.sh
```

健康检查可通过 Service、端口转发或 Pod 内执行完成：

```bash
kubectl port-forward deploy/qwen35-27b 18000:18000 19000:19000
curl http://127.0.0.1:19000/health
curl http://127.0.0.1:18000/v1/models
```

## 注意事项

- NVIDIA 场景需要配置 GPU Device Plugin 或对应资源申请。
- Ascend 场景需要配置 NPU Device Plugin、驱动、CANN、HCCN 和必要的设备挂载。
- PD、分布式、LMCache、Sparse KV 等能力通过环境变量和 CLI 字段开启，但仍然使用本 K8s 部署形态。
- 多实例部署必须显式错开 `ENGINE_PORT`、`--port`、`HEALTH_PORT`、`MONITOR_PROXY_PORT`。
