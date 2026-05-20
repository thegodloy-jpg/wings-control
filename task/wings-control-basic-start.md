# wings_control 启动命令示例

> 状态：历史任务记录。当前正式部署文档请优先阅读 [../README.md](../README.md)、[../docs/deployment/docker-compose.md](../docs/deployment/docker-compose.md)、[../docs/deployment/k8s.md](../docs/deployment/k8s.md) 和 [../docs/examples/qwen35-27b/README.md](../docs/examples/qwen35-27b/README.md)。
>
> 当前统一口径：`wings-control`、`wings-accel`、Engine 独立镜像交付；推荐 Docker Compose / K8s 编排；`wings_start.sh` 支持的启动项统一使用 CLI 字段，只有无 CLI 字段的运行时变量保留为环境变量。

本文档覆盖三个可选模型：

- GLM-5.1-FP8：8 张 H20-141G
- MiniMax-M2.7：4 张 H20
- DeepSeek-V3.2：8 张 H20-141G

实际使用时每次只启动其中一个模型服务，不存在三个服务同时启动。端口和共享卷路径使用代码默认值即可，命令里不重复声明默认配置。

以下命令字段与生产常用 `wings_start.sh` 参数集对齐，但统一使用 `python -m wings_control` 和 `--engine vllm`。模型路径需要按实际环境替换。

## 运行前置

源码目录本地运行时，需要设置 `PYTHONPATH`：

```powershell
cd D:\project\wings-k8s-260417\wings-control
$env:PYTHONPATH="D:\project\wings-k8s-260417\wings-control\wings_control"
```

容器镜像中通常已经设置好 `PYTHONPATH`，不需要手动设置。

默认值不在命令中重复声明：backend `17000`，proxy `18000`，health `19000`，monitor `19100`，shared volume `/shared-volume`，device `nvidia`，reason proxy enabled。

## GLM-5.1-FP8

硬件：8 张 H20-141G。

基础参数：

- `device-count`: `8`
- `input-length`: `4096`
- `output-length`: `4096`
- `block-size`: `64`
- `tensor_parallel_size`: 由 `--device-count 8` 自动生成

### 基础命令

不启用 KV 卸载、KV 稀疏、投机推理、工具调用。

```bash
export WINGS_SKIP_PID_CHECK=true
export ENGINE_VERSION=v0.19.0
export SAFETENSORS_FAST_GPU=1
export WINGS_H20_MODEL=H20-141G
export WINGS_DEVICE_COUNT=8
export WINGS_DEVICE_MEMORY=141

python -m wings_control \
  --engine vllm \
  --model-name GLM-5.1-FP8 \
  --model-path /models/GLM-5.1-FP8 \
  --port 18000 \
  --seed 42 \
  --trust-remote-code \
  --dtype auto \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --output-length 4096 \
  --kv-cache-dtype auto \
  --max-num-seqs 256 \
  --block-size 64 \
  --gpu-memory-utilization 0.95 \
  --input-length 4096 \
  --enable-chunked-prefill \
  --gpu-usage-mode full \
  --device-count 8
```

### 全部高级特性

通过环境变量开启 KV 内存卸载、KV 稀疏、无草稿模型投机推理，并触发对应补丁安装。KV 卸载只做内存卸载，不设置 `LMCACHE_LOCAL_DISK` 或 `LMCACHE_MAX_LOCAL_DISK_SIZE`。

```bash
export WINGS_SKIP_PID_CHECK=true
export ENGINE_VERSION=v0.19.0
export SAFETENSORS_FAST_GPU=1
export WINGS_H20_MODEL=H20-141G
export WINGS_DEVICE_COUNT=8
export WINGS_DEVICE_MEMORY=141
export ENABLE_ACCEL=true
export WINGS_ACCEL_DIR=/accel-volume
export ENABLE_SPARSE=true
export ENABLE_SPECULATIVE_DECODE=true
export LMCACHE_OFFLOAD=true
export LMCACHE_LOCAL_CPU=true
export LMCACHE_MAX_LOCAL_CPU_SIZE=64

python -m wings_control \
  --engine vllm \
  --model-name GLM-5.1-FP8 \
  --model-path /models/GLM-5.1-FP8 \
  --port 18000 \
  --seed 42 \
  --trust-remote-code \
  --dtype auto \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --output-length 4096 \
  --kv-cache-dtype auto \
  --max-num-seqs 256 \
  --block-size 64 \
  --gpu-memory-utilization 0.95 \
  --input-length 4096 \
  --enable-chunked-prefill \
  --gpu-usage-mode full \
  --device-count 8
```

GLM-5.1-FP8 的 `ENABLE_SPARSE=true` 会走 IndexCache 路径，当前代码会自动追加：

- `--block-size 64`
- `--hf-overrides '{"index_topk_freq": 4}'`

### Docker Compose 对照测试

GLM-5.1 提供两份 compose 文件用于同机对照测试。两份文件都使用 8 张 H20-141G，但 GPU 编号、端口、容器名和 volume 名互相错开。端口规则固定为：基础版使用 `17000/18000/19000/19100`，高级版在此基础上全部加一。compose 只负责拉起容器，`wings-control` 使用 `command: ["sleep", "infinity"]` 保持运行，需要进入 control 容器后手动执行上面的基础或高级启动命令。

基础测试版：

- 文件：`docker-compose-3c-glm51.yml`
- GPU：`0,1,2,3,4,5,6,7`
- OpenAI API 端口：`18000`
- engine backend 端口：`17000`
- health 端口：`19000`
- monitor proxy 端口：`19100`
- 高级特性：全部关闭；compose 不设置启动环境变量，进入 `wings-control` 后手动执行 GLM 基础命令

```bash
docker compose -f docker-compose-3c-glm51.yml up -d

curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-5.1","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

高级特性测试版：

- 文件：`docker-compose-3c-glm51-advanced.yml`
- GPU：`8,9,10,11,12,13,14,15`
- OpenAI API 端口：`18001`
- engine backend 端口：`17001`
- health 端口：`19001`
- monitor proxy 端口：`19101`
- 高级特性：compose 不设置启动环境变量，进入 `wings-control` 后手动执行 GLM 全部高级特性命令
- KV 卸载：只启用内存卸载，`LMCACHE_LOCAL_CPU=true`，`LMCACHE_MAX_LOCAL_CPU_SIZE=64`，不设置磁盘卸载

```bash
docker compose -f docker-compose-3c-glm51-advanced.yml up -d

curl http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-5.1","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

手动拉起后查看生成的启动脚本：

```bash
docker compose -f docker-compose-3c-glm51.yml exec wings-control-glm51-basic \
  cat /shared-volume/start_command.sh

docker compose -f docker-compose-3c-glm51-advanced.yml exec wings-control-glm51-advanced \
  cat /shared-volume/start_command.sh
```

## MiniMax-M2.7

硬件：4 张 H20。

基础参数：

- `device-count`: `4`
- `input-length`: `196000`
- `output-length`: `4096`
- `block-size`: `16`
- `tensor_parallel_size`: 由 `--device-count 4` 自动生成

### 基础命令

不启用 KV 卸载、KV 稀疏、投机推理、工具调用。

```bash
export WINGS_SKIP_PID_CHECK=true
export ENGINE_VERSION=v0.19.0
export SAFETENSORS_FAST_GPU=1
export WINGS_DEVICE_COUNT=4

python -m wings_control \
  --engine vllm \
  --model-name MiniMax-M2.7 \
  --model-path /models/MiniMax-M2.7 \
  --port 18000 \
  --seed 42 \
  --trust-remote-code \
  --dtype auto \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --output-length 4096 \
  --kv-cache-dtype auto \
  --max-num-seqs 256 \
  --block-size 16 \
  --gpu-memory-utilization 0.95 \
  --input-length 196000 \
  --enable-chunked-prefill \
  --gpu-usage-mode full \
  --device-count 4
```

### 全部高级特性

通过环境变量开启 KV 内存卸载、KV 稀疏、无草稿模型投机推理，并触发对应补丁安装。KV 卸载只做内存卸载，不设置 `LMCACHE_LOCAL_DISK` 或 `LMCACHE_MAX_LOCAL_DISK_SIZE`。

```bash
export WINGS_SKIP_PID_CHECK=true
export ENGINE_VERSION=v0.19.0
export SAFETENSORS_FAST_GPU=1
export WINGS_DEVICE_COUNT=4
export ENABLE_ACCEL=true
export WINGS_ACCEL_DIR=/accel-volume
export ENABLE_SPARSE=true
export ENABLE_SPECULATIVE_DECODE=true
export LMCACHE_OFFLOAD=true
export LMCACHE_LOCAL_CPU=true
export LMCACHE_MAX_LOCAL_CPU_SIZE=64

python -m wings_control \
  --engine vllm \
  --model-name MiniMax-M2.7 \
  --model-path /models/MiniMax-M2.7 \
  --port 18000 \
  --seed 42 \
  --trust-remote-code \
  --dtype auto \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --output-length 4096 \
  --kv-cache-dtype auto \
  --max-num-seqs 256 \
  --block-size 16 \
  --gpu-memory-utilization 0.95 \
  --input-length 196000 \
  --enable-chunked-prefill \
  --gpu-usage-mode full \
  --device-count 4
```

MiniMax-M2.7 不属于当前 IndexCache 架构列表，`ENABLE_SPARSE=true` 会按当前代码走 FP8 KV Cache 路径；上线前建议先小流量验证输出质量和稳定性。

### Docker Compose 对照测试

MiniMax-M2.7 提供两份 compose 文件用于同机对照测试。两份文件都使用 4 张卡，但 GPU 编号、端口、容器名和 volume 名互相错开，可以同时启动。端口规则固定为：基础版使用 `17000/18000/19000/19100`，高级版在此基础上全部加一。compose 只负责拉起容器，`wings-control` 使用 `command: ["sleep", "infinity"]` 保持运行，需要进入 control 容器后手动执行上面的基础或高级启动命令。

两份 MiniMax compose 的 service、container 和 volume 名都带 `minimax-m27-basic` 或 `minimax-m27-advanced` 后缀，不存在同名资源。并行测试时不要对其中一个文件执行带 `--remove-orphans` 的 `docker compose down/up`，避免同一目录默认 project 下误清理另一个版本。

基础测试版：

- 文件：`docker-compose-3c-minimax-m27.yml`
- GPU：`0,1,2,3`
- OpenAI API 端口：`18000`
- engine backend 端口：`17000`
- health 端口：`19000`
- monitor proxy 端口：`19100`
- 高级特性：全部关闭；compose 不设置启动环境变量，进入 `wings-control` 后手动执行 MiniMax 基础命令

```bash
docker compose -f docker-compose-3c-minimax-m27.yml up -d

curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MiniMax-M2.7","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

高级特性测试版：

- 文件：`docker-compose-3c-minimax-m27-advanced.yml`
- GPU：`4,5,6,7`
- OpenAI API 端口：`18001`
- engine backend 端口：`17001`
- health 端口：`19001`
- monitor proxy 端口：`19101`
- 高级特性：compose 不设置启动环境变量，进入 `wings-control` 后手动执行 MiniMax 全部高级特性命令
- KV 卸载：只启用内存卸载，`LMCACHE_LOCAL_CPU=true`，`LMCACHE_MAX_LOCAL_CPU_SIZE=64`，不设置磁盘卸载

```bash
docker compose -f docker-compose-3c-minimax-m27-advanced.yml up -d

curl http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MiniMax-M2.7","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

手动拉起后查看生成的启动脚本：

```bash
docker compose -f docker-compose-3c-minimax-m27.yml exec wings-control-minimax-m27-basic \
  cat /shared-volume/start_command.sh

docker compose -f docker-compose-3c-minimax-m27-advanced.yml exec wings-control-minimax-m27-advanced \
  cat /shared-volume/start_command.sh
```

## DeepSeek-V3.2

硬件：8 张 H20-141G。

基础参数：

- `device-count`: `8`
- `input-length`: `131072`
- `output-length`: `4096`
- `block-size`: `128`
- `tokenizer_mode`: `deepseek_v32`
- `tensor_parallel_size`: 由 `--device-count 8` 自动生成

### 基础命令

不启用 KV 卸载、KV 稀疏、投机推理、工具调用。

```bash
export WINGS_SKIP_PID_CHECK=true
export ENGINE_VERSION=v0.19.0
export SAFETENSORS_FAST_GPU=1
export WINGS_H20_MODEL=H20-141G
export WINGS_DEVICE_COUNT=8
export WINGS_DEVICE_MEMORY=141

python -m wings_control \
  --engine vllm \
  --model-name DeepSeek-V3.2 \
  --model-path /models/DeepSeek-V3.2 \
  --port 18000 \
  --config-file '{"tokenizer_mode":"deepseek_v32"}' \
  --seed 42 \
  --trust-remote-code \
  --dtype auto \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --output-length 4096 \
  --kv-cache-dtype auto \
  --max-num-seqs 256 \
  --block-size 128 \
  --gpu-memory-utilization 0.95 \
  --input-length 131072 \
  --enable-chunked-prefill \
  --gpu-usage-mode full \
  --device-count 8
```

### 全部高级特性

通过环境变量开启 KV 内存卸载、KV 稀疏、无草稿模型投机推理，并触发对应补丁安装。KV 卸载只做内存卸载，不设置 `LMCACHE_LOCAL_DISK` 或 `LMCACHE_MAX_LOCAL_DISK_SIZE`。

```bash
export WINGS_SKIP_PID_CHECK=true
export ENGINE_VERSION=v0.19.0
export SAFETENSORS_FAST_GPU=1
export WINGS_H20_MODEL=H20-141G
export WINGS_DEVICE_COUNT=8
export WINGS_DEVICE_MEMORY=141
export ENABLE_ACCEL=true
export WINGS_ACCEL_DIR=/accel-volume
export ENABLE_SPARSE=true
export ENABLE_SPECULATIVE_DECODE=true
export LMCACHE_OFFLOAD=true
export LMCACHE_LOCAL_CPU=true
export LMCACHE_MAX_LOCAL_CPU_SIZE=64

python -m wings_control \
  --engine vllm \
  --model-name DeepSeek-V3.2 \
  --model-path /models/DeepSeek-V3.2 \
  --port 18000 \
  --config-file '{"tokenizer_mode":"deepseek_v32"}' \
  --seed 42 \
  --trust-remote-code \
  --dtype auto \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --output-length 4096 \
  --kv-cache-dtype auto \
  --max-num-seqs 256 \
  --block-size 128 \
  --gpu-memory-utilization 0.95 \
  --input-length 131072 \
  --enable-chunked-prefill \
  --gpu-usage-mode full \
  --device-count 8
```

DeepSeek-V3.2 的 `ENABLE_SPARSE=true` 会走 IndexCache 路径，当前代码会自动追加：

- `--block-size 64`
- `--hf-overrides '{"index_topk_freq": 4}'`

### Docker Compose 对照测试

DeepSeek-V3.2 提供两份 compose 文件用于同机对照测试。两份文件都使用 8 张 H20-141G，但 GPU 编号、端口、容器名和 volume 名互相错开。端口规则固定为：基础版使用 `17000/18000/19000/19100`，高级版在此基础上全部加一。compose 只负责拉起容器，`wings-control` 使用 `command: ["sleep", "infinity"]` 保持运行，需要进入 control 容器后手动执行上面的基础或高级启动命令。

基础测试版：

- 文件：`docker-compose-3c-deepseek-v32.yml`
- GPU：`0,1,2,3,4,5,6,7`
- OpenAI API 端口：`18000`
- engine backend 端口：`17000`
- health 端口：`19000`
- monitor proxy 端口：`19100`
- 高级特性：全部关闭；compose 不设置启动环境变量，进入 `wings-control` 后手动执行 DeepSeek 基础命令
- DeepSeek tokenizer：手动启动命令中使用 `--config-file '{"tokenizer_mode":"deepseek_v32"}'`

```bash
docker compose -f docker-compose-3c-deepseek-v32.yml up -d

curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V3.2","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

高级特性测试版：

- 文件：`docker-compose-3c-deepseek-v32-advanced.yml`
- GPU：`8,9,10,11,12,13,14,15`
- OpenAI API 端口：`18001`
- engine backend 端口：`17001`
- health 端口：`19001`
- monitor proxy 端口：`19101`
- 高级特性：compose 不设置启动环境变量，进入 `wings-control` 后手动执行 DeepSeek 全部高级特性命令
- KV 卸载：只启用内存卸载，`LMCACHE_LOCAL_CPU=true`，`LMCACHE_MAX_LOCAL_CPU_SIZE=64`，不设置磁盘卸载
- DeepSeek tokenizer：手动启动命令中使用 `--config-file '{"tokenizer_mode":"deepseek_v32"}'`

```bash
docker compose -f docker-compose-3c-deepseek-v32-advanced.yml up -d

curl http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V3.2","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

手动拉起后查看生成的启动脚本：

```bash
docker compose -f docker-compose-3c-deepseek-v32.yml exec wings-control-deepseek-v32-basic \
  cat /shared-volume/start_command.sh

docker compose -f docker-compose-3c-deepseek-v32-advanced.yml exec wings-control-deepseek-v32-advanced \
  cat /shared-volume/start_command.sh
```

## 通用说明

- `LMCACHE_MAX_LOCAL_CPU_SIZE` 不是 launcher 启动必填项，但建议生产环境显式指定。
- 如果只设置 `LMCACHE_LOCAL_CPU=true` 而不设置 `LMCACHE_MAX_LOCAL_CPU_SIZE`，当前代码会生成 `local_cpu: {}`，CPU 缓存上限将交给 LMCache 默认行为，不利于资源控制。
- `64` 表示传给 LMCache YAML 的 `local_cpu.max_size`，具体单位按当前 LMCache 版本的配置语义解释。
- 投机推理不传 `--speculative-decode-model-path`。当前代码会在无草稿模型路径时按模型架构选择内置策略：匹配 MTP 支持的模型走 MTP；不匹配时回退到 suffix。
- 基础命令和高级特性命令都默认不启用工具调用。如果需要 function calling，再追加 `--enable-auto-tool-choice`。
- 默认 tool parser：GLM-5.1-FP8 为 `glm47`，MiniMax-M2.7 为 `minimax_m2`，DeepSeek-V3.2 为 `deepseekv32`。

## 验证

查看生成的 engine 启动脚本：

```bash
cat /shared-volume/start_command.sh
```

健康检查：

```bash
curl http://127.0.0.1:19000/health
```

推理验证：

```bash
curl http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3.2",
    "messages": [{"role": "user", "content": "你好，请简单介绍一下自己"}],
    "max_tokens": 128
  }'
```
