# MiniMax-M2.5 / MiniMax-M2.7 vLLM Ascend 启动命令说明

> 适用范围：Wings Control 使用 `vllm_ascend` 引擎拉起 `MiniMax-M2.5` 或 `MiniMax-M2.7`。
> 说明日期：2026-05-09。
> 代码依据：`wings_control/core/config_loader.py`、`wings_control/engines/vllm_adapter.py`、`wings_control/config/defaults/ascend_default.json`。

---

## 1. 结论

`MiniMax-M2.5` 和 `MiniMax-M2.7` 在当前代码中都会识别为同一个模型架构：

```text
MiniMaxM2ForCausalLM
```

因此使用 `vllm_ascend` 拉起时，两者最终 `start_command.sh` 的环境变量、默认 vLLM 参数和脚本结构完全同构；差异仅为：

| 模型 | `--served-model-name` | `--model` |
|---|---|---|
| MiniMax-M2.5 | `MiniMax-M2.5` | `/models/MiniMax-M2.5` |
| MiniMax-M2.7 | `MiniMax-M2.7` | `/models/MiniMax-M2.7` |

默认情况下不会注入以下 vLLM 高级字段：

- `--async-scheduling`
- `--speculative-config`
- `--enable-expert-parallel`

当前 `ascend_default.json` 对 MiniMax-M2 的默认值偏向长上下文、较高并发和图解码优化：`max_model_len=34816`、`gpu_memory_utilization=0.92`、`max_num_seqs=64`、`compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}`、`additional_config={"enable_cpu_binding":true}`。这些值会显著提高 KV cache 和显存压力，建议在真实 910B 环境完成稳定性、OOM 边界和吞吐验证后再作为生产默认值使用。

---

## 2. 默认 engine_config

以下示例基于：

- `ENGINE=vllm_ascend`
- `DEVICE_COUNT=8`
- 单机模式：`DISTRIBUTED=false`
- 未显式设置 `INPUT_LENGTH` / `OUTPUT_LENGTH`
- 未启用 speculative / sparse / LMCache / PD / QAT

### 2.1 MiniMax-M2.5

```json
{
  "trust_remote_code": true,
  "max_model_len": 34816,
  "host": "0.0.0.0",
  "port": 18000,
  "served_model_name": "MiniMax-M2.5",
  "model": "/models/MiniMax-M2.5",
  "dtype": "auto",
  "kv_cache_dtype": "auto",
  "quantization": "",
  "quantization_param_path": "",
  "gpu_memory_utilization": 0.92,
  "enable_chunked_prefill": false,
  "max_num_batched_tokens": 4096,
  "block_size": 16,
  "max_num_seqs": 64,
    "compilation_config": {
        "cudagraph_mode": "FULL_DECODE_ONLY"
    },
    "additional_config": {
        "enable_cpu_binding": true
    },
    "tool_call_parser": "minimax_m2",
    "reasoning_parser": "minimax_m2_reasoning",
  "seed": 0,
  "enable_expert_parallel": false,
  "enable_prefix_caching": false,
  "tensor_parallel_size": 8
}
```

### 2.2 MiniMax-M2.7

```json
{
  "trust_remote_code": true,
  "max_model_len": 34816,
  "host": "0.0.0.0",
  "port": 18000,
  "served_model_name": "MiniMax-M2.7",
  "model": "/models/MiniMax-M2.7",
  "dtype": "auto",
  "kv_cache_dtype": "auto",
  "quantization": "",
  "quantization_param_path": "",
  "gpu_memory_utilization": 0.92,
  "enable_chunked_prefill": false,
  "max_num_batched_tokens": 4096,
  "block_size": 16,
  "max_num_seqs": 64,
    "compilation_config": {
        "cudagraph_mode": "FULL_DECODE_ONLY"
    },
    "additional_config": {
        "enable_cpu_binding": true
    },
    "tool_call_parser": "minimax_m2",
    "reasoning_parser": "minimax_m2_reasoning",
  "seed": 0,
  "enable_expert_parallel": false,
  "enable_prefix_caching": false,
  "tensor_parallel_size": 8
}
```

生成 vLLM 命令时，空字符串和 `false` 布尔值会被跳过，因此不会出现在最终命令中。

---

## 3. start_command.sh 环境变量

`vllm_ascend` 会先内联 `wings_control/config/set_vllm_ascend_env.sh`，再追加 MiniMax 架构专属环境变量。

### 3.1 CANN / ATB 环境初始化

```bash
set +u
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/ascend-toolkit/set_env.sh ascend-toolkit/set_env.sh; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; } || echo 'WARN: ascend-toolkit/set_env.sh not found'
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/nnal/atb/set_env.sh nnal/atb/set_env.sh; else source /usr/local/Ascend/nnal/atb/set_env.sh; fi; } || echo 'WARN: nnal/atb/set_env.sh not found'
set -u
```

### 3.2 Ascend 驱动检查与库路径

```bash
if [ -d /usr/local/Ascend/driver/lib64/driver ]; then
    export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
else
    echo 'FATAL: Ascend driver not found!'
    exit 1
fi

if [ ! -f /usr/local/Ascend/driver/lib64/driver/libascend_hal.so ]; then
    echo 'FATAL: libascend_hal.so not found at /usr/local/Ascend/driver/lib64/driver/'
    echo 'HINT: Ensure the host Ascend driver is mounted into the container (hostPath: /usr/local/Ascend/driver)'
    exit 1
fi
```

### 3.3 jemalloc 预加载

```bash
if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "INFO: jemalloc preloaded from /usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
fi
```

### 3.4 Ascend 性能调优

```bash
case "${WINGS_ASCEND_PERF_TUNING:-true}" in
    false|False|FALSE|0|no|No|NO)
        echo "INFO: WINGS_ASCEND_PERF_TUNING=${WINGS_ASCEND_PERF_TUNING}; skip Ascend performance tuning"
        ;;
    *)
        echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true
        sysctl -w vm.swappiness=0 || true
        sysctl -w kernel.numa_balancing=0 || true
        sysctl -w kernel.sched_migration_cost_ns=50000 || true
        ;;
esac
```

### 3.5 通用 Ascend 变量

```bash
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
```

### 3.6 MiniMaxM2ForCausalLM 专属变量

```bash
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_USE_GRAPH=1
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_TORCH_COMPILE=0
```

注意：`HCCL_OP_EXPANSION_MODE=AIV`、`HCCL_BUFFSIZE=1024`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`、jemalloc 预加载和系统性能调优均由通用 Ascend 环境脚本注入，MiniMax 专属段不再重复注入；`OMP_NUM_THREADS=1` 会覆盖通用默认值 `${OMP_NUM_THREADS:-10}`。

---

## 4. 最终 vLLM 启动命令

### 4.1 MiniMax-M2.5

```bash
exec python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 34816 --host 0.0.0.0 --port 18000 --served-model-name MiniMax-M2.5 --model /models/MiniMax-M2.5 --dtype auto --kv-cache-dtype auto --gpu-memory-utilization 0.92 --max-num-batched-tokens 4096 --block-size 16 --max-num-seqs 64 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"enable_cpu_binding":true}' --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_reasoning --seed 0 --tensor-parallel-size 8
```

### 4.2 MiniMax-M2.7

```bash
exec python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 34816 --host 0.0.0.0 --port 18000 --served-model-name MiniMax-M2.7 --model /models/MiniMax-M2.7 --dtype auto --kv-cache-dtype auto --gpu-memory-utilization 0.92 --max-num-batched-tokens 4096 --block-size 16 --max-num-seqs 64 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"enable_cpu_binding":true}' --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_reasoning --seed 0 --tensor-parallel-size 8
```

---

## 5. 参数变化规则

### 5.1 tensor_parallel_size

`tensor_parallel_size` 由 `DEVICE_COUNT` 推导：

| DEVICE_COUNT | 常规最终参数 |
|---:|---:|
| 1 | `--tensor-parallel-size 1` |
| 4 | `--tensor-parallel-size 4` |
| 8 | `--tensor-parallel-size 8` |

特殊情况：如果检测到 300I A2 PCIe 标卡，且设备数为 4 或 8，则会强制 `tensor_parallel_size=4`。

### 5.2 max_model_len

默认配置中 `max_model_len=34816`。
只有用户显式传入 `--input-length` / `--output-length` 或设置 `INPUT_LENGTH` / `OUTPUT_LENGTH` 时，才会重新计算：

```text
max_model_len = input_length + output_length
```

### 5.3 enforce_eager

默认不会追加 `--enforce-eager`。
如果设置：

```bash
export ASCEND_ENFORCE_EAGER=true
```

则最终 vLLM 命令末尾会追加：

```bash
--enforce-eager
```

### 5.4 日志 echo 注入

`vllm_ascend` 会自动给导出的环境变量和关键启动命令注入日志打印，例如：

```bash
echo "[wings-env] export VLLM_USE_GRAPH=${VLLM_USE_GRAPH:-}"
echo '[wings-cmd] >>> exec python3 -m vllm.entrypoints.openai.api_server ...'
```

这些 echo 不改变启动语义，只用于排查 `engine.log`。

---

## 6. 完整关键片段

### 6.1 MiniMax-M2.5

```bash
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_USE_GRAPH=1
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_TORCH_COMPILE=0

exec python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 34816 --host 0.0.0.0 --port 18000 --served-model-name MiniMax-M2.5 --model /models/MiniMax-M2.5 --dtype auto --kv-cache-dtype auto --gpu-memory-utilization 0.92 --max-num-batched-tokens 4096 --block-size 16 --max-num-seqs 64 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"enable_cpu_binding":true}' --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_reasoning --seed 0 --tensor-parallel-size 8
```

### 6.2 MiniMax-M2.7

```bash
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_USE_GRAPH=1
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_TORCH_COMPILE=0

exec python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 34816 --host 0.0.0.0 --port 18000 --served-model-name MiniMax-M2.7 --model /models/MiniMax-M2.7 --dtype auto --kv-cache-dtype auto --gpu-memory-utilization 0.92 --max-num-batched-tokens 4096 --block-size 16 --max-num-seqs 64 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"enable_cpu_binding":true}' --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_reasoning --seed 0 --tensor-parallel-size 8
```
