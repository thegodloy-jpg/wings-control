#!/usr/bin/env bash
# =============================================================================
# 示例: vllm-ascend + Mooncake PD 分离 Decode(D) 节点 start_command.sh
# 来源: wings_control.engines.vllm_adapter.build_start_script 生成逻辑
# 场景: Qwen3-8B, 单卡 TP=1, MooncakeConnectorV1, RDMA, D 节点 kv_consumer
# =============================================================================

# CANN/ATB 环境初始化脚本在 engine 容器内执行；若 helper 已注入，则打印 source 前后的环境变量差异。
set +u
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/ascend-toolkit/set_env.sh ascend-toolkit/set_env.sh; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; } || echo 'WARN: ascend-toolkit/set_env.sh not found'
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/nnal/atb/set_env.sh nnal/atb/set_env.sh; else source /usr/local/Ascend/nnal/atb/set_env.sh; fi; } || echo 'WARN: nnal/atb/set_env.sh not found'
set -u

# Ascend 驱动库路径（libascend_hal.so 等位于此目录）
# 如果驱动目录不存在，说明宿主机驱动未挂载到容器
if [ -d /usr/local/Ascend/driver/lib64/driver ]; then
    export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
    echo "[wings-env] export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
else
    echo '================================================================'
    echo 'FATAL: Ascend driver not found!'
    echo '  • /usr/local/Ascend/driver/lib64/driver directory is missing'
    echo '  • libascend_hal.so is required by torch_npu / vllm_ascend'
    echo ''
    echo 'FIX: Mount the host Ascend driver into the container:'
    echo '  volumeMounts:'
    echo '    - name: ascend-driver'
    echo '      mountPath: /usr/local/Ascend/driver'
    echo '  volumes:'
    echo '    - name: ascend-driver'
    echo '      hostPath:'
    echo '        path: /usr/local/Ascend/driver'
    echo '        type: Directory'
    echo '================================================================'
    exit 1
fi

# jemalloc 预加载: 防止 Mooncake Transfer Engine 并发堆损坏
if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[wings-env] export LD_PRELOAD=${LD_PRELOAD:-}"
    echo "INFO: jemalloc preloaded from /usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
fi

# 通用性能调优：默认开启；可通过 WINGS_ASCEND_PERF_TUNING=false/0/no 关闭。
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

# 昇腾通用环境变量
export HCCL_BUFFSIZE=1024
echo "[wings-env] export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"
export OMP_PROC_BIND=false
echo "[wings-env] export OMP_PROC_BIND=${OMP_PROC_BIND:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
echo "[wings-env] export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-}"
export HCCL_OP_EXPANSION_MODE=AIV
echo "[wings-env] export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-}"

# Pre-flight: verify Ascend driver is accessible
if [ ! -f /usr/local/Ascend/driver/lib64/driver/libascend_hal.so ]; then
    echo 'FATAL: libascend_hal.so not found at /usr/local/Ascend/driver/lib64/driver/'
    echo 'HINT: Ensure the host Ascend driver is mounted into the container (hostPath: /usr/local/Ascend/driver)'
    exit 1
fi

# PD / Mooncake 环境。D 节点为 kv_consumer。
export HCCL_IF_IP=7.6.52.170
echo "[wings-env] export HCCL_IF_IP=${HCCL_IF_IP:-}"
export GLOO_SOCKET_IFNAME=ens65f1np1
echo "[wings-env] export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
export TP_SOCKET_IFNAME=ens65f1np1
echo "[wings-env] export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-}"
export HCCL_SOCKET_IFNAME=ens65f1np1
echo "[wings-env] export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-}"
export OMP_PROC_BIND=false
echo "[wings-env] export OMP_PROC_BIND=${OMP_PROC_BIND:-}"
export OMP_NUM_THREADS=100
echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
export VLLM_USE_V1=1
echo "[wings-env] export VLLM_USE_V1=${VLLM_USE_V1:-}"
export LCCL_DETERMINISTIC=1
echo "[wings-env] export LCCL_DETERMINISTIC=${LCCL_DETERMINISTIC:-}"
export HCCL_DETERMINISTIC=true
echo "[wings-env] export HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-}"
export CLOSE_MATMUL_K_SHIFT=1
echo "[wings-env] export CLOSE_MATMUL_K_SHIFT=${CLOSE_MATMUL_K_SHIFT:-}"
export VLLM_LLMDD_RPC_PORT=5570
echo "[wings-env] export VLLM_LLMDD_RPC_PORT=${VLLM_LLMDD_RPC_PORT:-}"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=23100
echo "[wings-env] export VLLM_MOONCAKE_BOOTSTRAP_PORT=${VLLM_MOONCAKE_BOOTSTRAP_PORT:-}"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
echo "[wings-env] export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-}"
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
echo "[wings-env] export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

# Qwen3ForCausalLM vllm-ascend 架构专用环境变量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
echo "[wings-env] export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-}"
export HCCL_BUFFSIZE=512
echo "[wings-env] export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"
export OMP_PROC_BIND=false
echo "[wings-env] export OMP_PROC_BIND=${OMP_PROC_BIND:-}"
export OMP_NUM_THREADS=1
echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
export TASK_QUEUE_ENABLE=1
echo "[wings-env] export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-}"

echo '[wings-cmd] >>> exec python3 -m vllm.entrypoints.openai.api_server --model '\''/models/Qwen3-8B'\'' --served-model-name Qwen3-8B --host 0.0.0.0 --port 17100 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 32768 --trust-remote-code --disable-log-requests --kv-transfer-config '\''{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma","prefill":{"tp_size":1,"dp_size":1,"pp_size":1},"decode":{"tp_size":1,"dp_size":1,"pp_size":1}}}'\'''
exec python3 -m vllm.entrypoints.openai.api_server \
  --model '/models/Qwen3-8B' \
  --served-model-name Qwen3-8B \
  --host 0.0.0.0 \
  --port 17100 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --trust-remote-code \
  --disable-log-requests \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma","prefill":{"tp_size":1,"dp_size":1,"pp_size":1},"decode":{"tp_size":1,"dp_size":1,"pp_size":1}}}'
