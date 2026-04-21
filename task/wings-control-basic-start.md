# wings_control 启动命令示例

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
