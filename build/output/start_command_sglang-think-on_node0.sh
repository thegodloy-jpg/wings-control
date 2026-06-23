#!/usr/bin/env bash
set -euo pipefail
mkdir -p /var/log/wings
rm -rf /var/log/wings/prometheus_multiproc
mkdir -p /var/log/wings/prometheus_multiproc
# --- wings: env echo helpers ---
wings_source_env_with_diff() {
    local script_path="$1"
    local label="${2:-$1}"
    if [ "$#" -ge 2 ]; then
        shift 2
    else
        shift 1
    fi
    if [ ! -f "$script_path" ]; then
        echo "[wings-env-source] WARN: $label not found: $script_path"
        return 0
    fi

    local before_file after_file
    before_file="$(mktemp)"
    after_file="$(mktemp)"
    env | sort > "$before_file" || true

    set +u
    # shellcheck disable=SC1090
    source "$script_path" "$@"
    local source_rc=$?
    set -u

    env | sort > "$after_file" || true
    comm -13 "$before_file" "$after_file" | sed "s|^|[wings-env-source] $label |" || true
    rm -f "$before_file" "$after_file"
    return "$source_rc"
}
# --- end wings env echo helpers ---
export PROMETHEUS_MULTIPROC_DIR=/var/log/wings/prometheus_multiproc
echo "[wings-env] export PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-}"

# --- log_analyzer: 启动部署进度监控（仅master节点） ---
# 清空旧的日志文件，确保 log_analyzer 只分析新的日志（避免残留内容触发误判）
rm -f /var/log/wings/engine.log
rm -f /var/log/wings/engine-full.log
rm -f /shared-volume/progress.jsonl

# 记录脚本开始时间（用于计算耗时）
SCRIPT_START_EPOCH=$(date +%s)

ANALYZER_CONFIG='{"engine": "sglang", "deployment_mode": "single", "hardware": "nvidia", "nnodes": 1, "node_rank": 0, "distributed_backend": "mp", "tensor_parallel_size": 8, "model_name": "Qwen3.6-27B", "model_path": "D:/project/inference/wings-control/wings-control-0730/wings-control/build/model_27t67h34", "backend_port": 17000}'
echo "[log_analyzer] 配置信息: $ANALYZER_CONFIG"

# 启动日志分析器（后台）
# 清除旧 __pycache__，防止跨 Python 版本的 pyc magic number 不匹配
find /shared-volume/log_analyzer -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cd /shared-volume && python3 -B -m log_analyzer.log_analyzer \
    --config "$ANALYZER_CONFIG" \
    --log-file /var/log/wings/engine.log \
    --progress-file /shared-volume/progress.jsonl \
    --accel-file /shared-volume/advanced_features.json &
LOG_ANALYZER_PID=$!
echo "[log_analyzer] 分析器PID: $LOG_ANALYZER_PID"

# 注册清理函数（等待分析器完全退出）
cleanup_analyzer() {
    local exit_code=$?
    echo "[log_analyzer] 停止分析器..."
    if [ -n "$LOG_ANALYZER_PID" ]; then
        kill $LOG_ANALYZER_PID 2>/dev/null || true
        # 等待分析器进程完全退出，确保完成收尾工作
        wait $LOG_ANALYZER_PID 2>/dev/null || true
    fi

    if [ -n "${ENGINE_PID:-}" ]; then
        echo "[cleanup] 发送 SIGTERM 给引擎进程..."
        kill -TERM "$ENGINE_PID" 2>/dev/null || true
    else
        # ENGINE_PID 未设置说明引擎启动前脚本就失败了（如 ray: command not found）
        # 写入失败进度，让上层感知到部署失败
        if [ "$exit_code" -ne 0 ]; then
            echo "[cleanup] 引擎启动前脚本异常退出，退出码: $exit_code"
            local curr_time
            curr_time=$(date -Iseconds)
            local start_time
            start_time=$(date -Iseconds -d "@${SCRIPT_START_EPOCH}")
            local elapsed
            elapsed=$(( $(date +%s) - SCRIPT_START_EPOCH ))
            cat >> "/shared-volume/progress.jsonl" <<EARLY_FAIL_EOF
{"progress": 0, "phase_code": "script_error", "phase_name": "启动脚本执行失败", "status": "failed", "key_log": "引擎启动前脚本异常退出，退出码: $exit_code", "curr_time": "$curr_time", "start_time": "$start_time", "elapsed_time_s": $elapsed}
EARLY_FAIL_EOF
        fi
    fi
}
trap cleanup_analyzer EXIT  SIGTERM SIGINT

export PYTHONUNBUFFERED=1
echo "[wings-env] export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"
exec > >(tee -a /var/log/wings/engine-full.log | grep --line-buffered -vE '"GET\s+/(health|metrics)\s|\b(Prefill|Decode) batch\b' | tee -a /var/log/wings/engine.log) 2>&1
# --- wings: SGLang faulthandler.enable() OOM workaround ---
mkdir -p /tmp/wings_sitecustomize
cat > /tmp/wings_sitecustomize/sitecustomize.py << 'WINGS_FAULTHANDLER_PATCH'
import faulthandler as _fh
_original_enable = _fh.enable
def _safe_enable(*args, **kwargs):
    try:
        return _original_enable(*args, **kwargs)
    except OSError:
        pass  # /dev/shm tmpfs counted against cgroup memory limit
_fh.enable = _safe_enable
WINGS_FAULTHANDLER_PATCH
export PYTHONPATH="/tmp/wings_sitecustomize:${PYTHONPATH:-}"
echo "[wings-env] export PYTHONPATH=${PYTHONPATH:-}"
echo "[wings] Injected faulthandler.enable() OOM patch for SGLang"
# --- end faulthandler patch ---
ENGINE_START_EPOCH=$(date +%s)
echo '[wings-cmd] >>> exec python3 -m sglang.launch_server --trust-remote-code --context-length 5120 --tool-call-parser qwen --host 192.168.1.100 --port 17000 --served-model-name Qwen3.6-27B --model-path D:/project/inference/wings-control/wings-control-0730/wings-control/build/model_27t67h34 --dtype auto --kv-cache-dtype auto --mem-fraction-static 0.9 --chunked-prefill-size 4096 --max-running-requests 32 --random-seed 0 --disable-chunked-prefix-cache --tp-size 8'
python3 -m sglang.launch_server --trust-remote-code --context-length 5120 --tool-call-parser qwen --host 192.168.1.100 --port 17000 --served-model-name Qwen3.6-27B --model-path D:/project/inference/wings-control/wings-control-0730/wings-control/build/model_27t67h34 --dtype auto --kv-cache-dtype auto --mem-fraction-static 0.9 --chunked-prefill-size 4096 --max-running-requests 32 --random-seed 0 --disable-chunked-prefix-cache --tp-size 8 &
ENGINE_PID=$!
echo "[Engine] Engine PID: $ENGINE_PID"

# --- Engine process wait and exception handling (with crash retry) ---
echo "[Engine] Engine process monitor started, PID=$ENGINE_PID"
if wait "$ENGINE_PID"; then
  echo "[Engine] Engine process exited normally"
  echo "[引擎] 停止日志解析进程..."
  [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
  trap - EXIT
else
  EXIT_CODE=$?
  ENGINE_DURATION=$(( $(date +%s) - ENGINE_START_EPOCH ))
  echo "[Engine] Engine process exited abnormally, exit_code=$EXIT_CODE, runtime=${ENGINE_DURATION}s"
  echo "[Engine] ┌── Engine Crash Retry ──"
  echo "[Engine] │ Reason: Engine crashed (exit_code=$EXIT_CODE, runtime=${ENGINE_DURATION}s)"
  echo "[Engine] │ Action: Retrying engine startup with same parameters (attempt 2/2)"
  echo "[Engine] └── Retry command about to execute..."
  # 清理上一次启动残留：ray head/worker 进程 + 端口占用
  if command -v ray >/dev/null 2>&1; then
    echo "[Engine] Stopping leftover Ray cluster before retry..."
    echo '[wings-cmd] >>> ray stop --force >/dev/null 2>&1 || true'
    ray stop --force >/dev/null 2>&1 || true
  fi
  # 兜底：杀掉残留的 vLLM EngineCore / WorkerProc（父进程已死但子进程可能还在）
  pkill -9 -f 'vllm.*EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm.*WorkerProc' 2>/dev/null || true
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  # 一刀切：unset 所有补丁/加速层使能环境变量，退到最基本的启动命令
  # （不动 VLLM_ASCEND_ENABLE_* / VLLM_USE_V1 等常规性能 flag，它们不是补丁）
  echo "[Engine] Unsetting patch/accel env vars for retry: WINGS_ENGINE_PATCH_OPTIONS VLLM_EARS_TOLERANCE"
  unset WINGS_ENGINE_PATCH_OPTIONS
  unset VLLM_EARS_TOLERANCE
  echo "[Engine] Waiting 5s for port release before retry..."
  sleep 5
  ENGINE_START_EPOCH=$(date +%s)
echo '[wings-cmd] >>> exec python3 -m sglang.launch_server --trust-remote-code --context-length 5120 --tool-call-parser qwen --host 192.168.1.100 --port 17000 --served-model-name Qwen3.6-27B --model-path D:/project/inference/wings-control/wings-control-0730/wings-control/build/model_27t67h34 --dtype auto --kv-cache-dtype auto --mem-fraction-static 0.9 --chunked-prefill-size 4096 --max-running-requests 32 --random-seed 0 --disable-chunked-prefix-cache --tp-size 8'
python3 -m sglang.launch_server --trust-remote-code --context-length 5120 --tool-call-parser qwen --host 192.168.1.100 --port 17000 --served-model-name Qwen3.6-27B --model-path D:/project/inference/wings-control/wings-control-0730/wings-control/build/model_27t67h34 --dtype auto --kv-cache-dtype auto --mem-fraction-static 0.9 --chunked-prefill-size 4096 --max-running-requests 32 --random-seed 0 --disable-chunked-prefix-cache --tp-size 8 &
ENGINE_PID=$!
echo "[Engine] Engine PID: $ENGINE_PID (retry mode)"
  echo "[Engine] Retry engine started, waiting for process exit..."
  if wait "$ENGINE_PID"; then
    echo "[Engine] Engine process exited normally (retry mode)"
      echo "[引擎] 停止日志解析进程..."
      [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
      trap - EXIT
  else
    EXIT_CODE=$?
    echo "[Engine] Retry also failed, exit_code=$EXIT_CODE — unrecoverable"

      CURR_TIME=$(date -Iseconds)
      SCRIPT_START_EPOCH="${SCRIPT_START_EPOCH:-$(date +%s)}"
      START_TIME=$(date -Iseconds -d "@${SCRIPT_START_EPOCH}")
      ELAPSED_TIME=$(( $(date +%s) - SCRIPT_START_EPOCH ))

      cat >> "/shared-volume/progress.jsonl" <<EOF
{"progress": 0, "phase_code": "engine_crash", "phase_name": "引擎进程异常退出", "status": "failed", "key_log": "引擎进程异常退出，退出码: $EXIT_CODE", "curr_time": "$CURR_TIME", "start_time": "$START_TIME", "elapsed_time_s": $ELAPSED_TIME}
EOF

      echo "[引擎] 停止日志解析进程..."
      [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
      trap - EXIT

    exit "$EXIT_CODE"
  fi
fi

