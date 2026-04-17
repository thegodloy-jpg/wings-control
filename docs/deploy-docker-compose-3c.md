# 三容器 Docker Compose 部署指南

> **架构：** `accel-init` (一次性) → `wings-control` (sidecar) + `engine` (推理引擎)
>
> **配置文件：** `docker-compose-3c.yml`

---

## 目录

1. [前提条件](#1-前提条件)
2. [镜像准备](#2-镜像准备)
3. [模型文件准备](#3-模型文件准备)
4. [配置修改](#4-配置修改)
5. [部署执行](#5-部署执行)
6. [验证与测试](#6-验证与测试)
7. [日志与排障](#7-日志与排障)
8. [停止与清理](#8-停止与清理)
9. [Ascend NPU 适配](#9-ascend-npu-适配)

---

## 1. 前提条件

### 软件环境

| 依赖 | 最低版本 | 检查命令 |
|------|---------|---------|
| Docker Engine | 20.10+ | `docker --version` |
| Docker Compose | v2.20+ | `docker compose version` |
| NVIDIA Container Toolkit (GPU) | 1.14+ | `nvidia-ctk --version` |

### 硬件要求

| 模型 | 显存需求 | 推荐配置 |
|------|---------|---------|
| Qwen3.5-27B (TP=2) | ~60GB | 2× A100/H100 80GB |
| Qwen3-8B (TP=1) | ~20GB | 1× A100 40GB |

### 检查 NVIDIA runtime 是否可用

```bash
docker info | grep -i runtime
# 应看到: Runtimes: nvidia runc
# 若没有 nvidia，需安装 NVIDIA Container Toolkit：
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

---

## 2. 镜像准备

需要准备 3 个独立镜像：

### 2.1 wings-accel 镜像

accel 是加速补丁包，以 busybox 为基础镜像，体积小。

```bash
# 方式一：使用构建脚本（需要在构建机上执行）
cd build/
export WINGS_VERSION=26.0.0
bash build.sh ${WINGS_VERSION}
# 产物在 build/output/ 下：
#   Wings-Accel_26.0.0_x86_64.tar  (x86)
#   Wings-Accel_26.0.0_aarch64.tar (ARM)

# 方式二：直接加载已构建的 tar 包
docker load -i Wings-Accel_26.0.0_x86_64.tar
# 加载后镜像名: fusionregistry:5000/wings-accel:26.0.0
```

> **镜像内容：** `/opt/packages/` 目录下包含 `install.py` 和各引擎优化库

### 2.2 wings-control 镜像

control 是 Python sidecar 控制面。

```bash
# 方式一：使用构建脚本（build.sh 自动调用）
# 产物: Wings-Infer_26.0.0_x86_64.tar

# 方式二：加载已构建的 tar 包
docker load -i Wings-Infer_26.0.0_x86_64.tar
# 加载后镜像名: fusionregistry:5000/wings-infer:26.0.0
```

> **镜像内容：** python:3.10-slim 基础 + fastapi/uvicorn/httpx + wings_control 代码

### 2.3 engine 推理引擎镜像

根据设备类型选择：

| 设备 | 镜像 | 拉取命令 |
|------|------|---------|
| NVIDIA GPU | `vllm/vllm-openai:v0.17.0` | `docker pull vllm/vllm-openai:v0.17.0` |
| Ascend NPU | `quay.io/ascend/vllm-ascend:v0.17.0rc1` | `docker pull quay.io/ascend/vllm-ascend:v0.17.0rc1` |

### 2.4 确认镜像就绪

```bash
docker images | grep -E "wings-accel|wings-infer|wings-control|vllm"
# 应看到三个镜像各一行
```

---

## 3. 模型文件准备

```bash
# 确认模型目录存在且文件完整
ls -la /data/models/Qwen3.5-27B/
# 应包含:
#   config.json
#   tokenizer.json / tokenizer_config.json
#   model-*.safetensors (或 pytorch_model-*.bin)
#   generation_config.json

# 检查磁盘空间（模型文件 + 运行时缓存）
df -h /data/models/
```

---

## 4. 配置修改

编辑 `docker-compose-3c.yml` 中的 `x-user-config` 和 `x-model-volume`：

### 4.1 修改镜像名

根据实际加载的镜像名称调整（注意构建脚本产出的名称带 registry 前缀）：

```yaml
x-user-config:
  accel-image:   &accel-image   "fusionregistry:5000/wings-accel:26.0.0"
  control-image: &control-image "fusionregistry:5000/wings-infer:26.0.0"
  engine-image:  &engine-image  "vllm/vllm-openai:v0.17.0"
```

### 4.2 修改模型配置

```yaml
  model-name:      &model-name    "Qwen3.5-27B"       # 模型名称（API 请求中使用）
  model-path:      &model-path    "/models/Qwen3.5-27B" # 容器内模型路径
  host-model-dir:  "/data/models/Qwen3.5-27B"           # 宿主机模型路径
```

**同步修改 `x-model-volume`：**

```yaml
x-model-volume: &model-volume "/data/models/Qwen3.5-27B:/models/Qwen3.5-27B:ro"
#                               ↑ 宿主机路径              ↑ 容器内路径（与 model-path 一致）
```

### 4.3 修改设备配置

```yaml
  device-count:    &device-count  "2"         # GPU/NPU 数量（= TP size）
  visible-devices: &visible-devices "0,1"     # 使用哪些 GPU/NPU
```

### 4.4 修改引擎类型（可选）

如果使用 SGLang 或 MindIE 引擎，修改公共环境变量：

```yaml
x-common-env: &common-env
  ENGINE: vllm          # 可选: vllm, vllm_ascend, sglang, mindie
```

---

## 5. 部署执行

### 5.1 启动

```bash
cd /path/to/wings-control/

# 前台启动（首次建议前台观察日志）
docker compose -f docker-compose-3c.yml up

# 后台启动
docker compose -f docker-compose-3c.yml up -d
```

### 5.2 预期启动时序

```
时间线:
  T+0s    accel-init 启动
  T+1~2s  accel-init 完成拷贝 → 容器退出 (exit 0)
  T+2s    wings-control 启动（depends_on: service_completed_successfully）
  T+2s    engine 启动（并行于 control）
  T+3~8s  control 解析参数, 生成 /shared-volume/start_command.sh
  T+4~10s engine 检测到 start_command.sh, 开始执行
  T+30~120s  引擎加载模型权重（取决于模型大小和 IO 性能）
  T+60~180s  引擎 ready, 开始服务请求
```

### 5.3 预期容器状态

```bash
docker compose -f docker-compose-3c.yml ps
```

| NAME | STATUS | 说明 |
|------|--------|------|
| `wings-accel-init` | Exited (0) | 正常退出（一次性 init 容器） |
| `wings-control` | Up | 持续运行 proxy:18000 + health:19000 |
| `wings-engine` | Up | 持续运行推理引擎:17000 |

---

## 6. 验证与测试

### 6.1 健康检查

```bash
# 检查 health 端口
curl -s http://localhost:19000/health | python3 -m json.tool

# 预期: 引擎加载中时返回 "initializing"，就绪后返回 "ready" 或 "healthy"
```

### 6.2 查看生成的启动脚本

```bash
docker exec wings-control cat /shared-volume/start_command.sh
```

预期输出类似：

```bash
#!/bin/bash
# 引擎启动命令（由 wings_control 自动生成）
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": [...]}'
python3 /accel-volume/install.py --features "$WINGS_ENGINE_PATCH_OPTIONS"
python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3.5-27B \
    --tensor-parallel-size 2 \
    --port 17000 \
    ...
```

### 6.3 功能测试

```bash
# 非流式请求
curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "你好，请做个自我介绍"}],
    "max_tokens": 128
  }'

# 流式请求
curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "写一首五言绝句"}],
    "max_tokens": 128,
    "stream": true
  }'

# 模型列表
curl http://localhost:18000/v1/models
```

### 6.4 预期响应

非流式请求成功时返回 JSON：

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "Qwen3.5-27B",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "你好！我是..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 45, "total_tokens": 57}
}
```

---

## 7. 日志与排障

### 7.1 查看日志

```bash
# 全部日志
docker compose -f docker-compose-3c.yml logs -f

# 仅 control 日志
docker compose -f docker-compose-3c.yml logs -f wings-control

# 仅 engine 日志
docker compose -f docker-compose-3c.yml logs -f engine

# accel-init 日志（已退出，不带 -f）
docker compose -f docker-compose-3c.yml logs accel-init
```

### 7.2 常见问题

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| accel-init 退出码非 0 | 镜像中 `/opt/packages/` 为空或损坏 | `docker run --rm <accel-image> ls -la /opt/packages/` |
| engine 容器 `Timeout waiting for start_command.sh` | control 启动失败，未生成脚本 | 查看 control 日志：`docker logs wings-control` |
| engine 退出 | 模型路径错误 / GPU 显存不足 / CUDA 版本不匹配 | 查看 engine 日志；`docker exec wings-engine nvidia-smi` |
| control 启动但 curl 18000 超时 | 引擎尚未 ready，proxy 等待后端 | 查看 health:19000 状态；等待引擎加载完成 |
| `nvidia runtime not found` | 未安装 NVIDIA Container Toolkit | 安装 nvidia-container-toolkit 并重启 Docker |

### 7.3 进入容器调试

```bash
# 进入 control 容器
docker exec -it wings-control bash

# 进入 engine 容器
docker exec -it wings-engine bash

# 检查共享卷内容
docker exec wings-control ls -la /shared-volume/
docker exec wings-engine ls -la /accel-volume/
```

---

## 8. 停止与清理

```bash
# 停止所有容器
docker compose -f docker-compose-3c.yml down

# 停止并删除 volume（下次启动会重新初始化）
docker compose -f docker-compose-3c.yml down -v

# 仅重启 engine（不重启 control）
docker compose -f docker-compose-3c.yml restart engine
```

---

## 9. Ascend NPU 适配

在 Ascend 910B 设备上部署时，需修改 `docker-compose-3c.yml` 的以下部分：

### 9.1 修改镜像和引擎

```yaml
x-user-config:
  engine-image:  &engine-image  "quay.io/ascend/vllm-ascend:v0.17.0rc1"

x-common-env: &common-env
  ENGINE: vllm_ascend      # 改为 vllm_ascend
```

### 9.2 修改 engine 服务

1. **删除** `runtime: nvidia` 行
2. **替换** environment 块：

```yaml
  engine:
    # runtime: nvidia         ← 删除此行
    environment:
      ASCEND_VISIBLE_DEVICES: "8,9"        # NPU 设备 ID
      ASCEND_RT_VISIBLE_DEVICES: "8,9"
```

3. **取消注释** Ascend 卷挂载，注释掉原 volumes 块：

```yaml
    volumes:
      - shared-vol:/shared-volume
      - accel-vol:/accel-volume:ro
      - *model-volume
      - /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
      - /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
      - /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
      - /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
      - /usr/local/dcmi:/usr/local/dcmi:ro
      - /etc/ascend_install.info:/etc/ascend_install.info:ro
      - /etc/hccn.conf:/etc/hccn.conf:ro
```

### 9.3 Ascend 环境检查

```bash
# 进入 engine 容器后检查 NPU 状态
docker exec -it wings-engine npu-smi info
```
