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
export PYTHONUNBUFFERED=1
echo "[wings-env] export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"
exec > >(tee -a /var/log/wings/engine-full.log | grep --line-buffered -vE '"GET\s+/(health|metrics)\s|\b(Prefill|Decode) batch\b' | tee -a /var/log/wings/engine.log) 2>&1
ENGINE_START_EPOCH=$(date +%s)
# =============================================================================
# MindIE 单机引擎环境初始化脚本
# 用途: 被 _build_base_env_commands() 读取并内联到 start_command.sh
# 来源: 参考 wings/config/set_mindie_single_env.sh，适配 sidecar 架构
#
# 注意: 此脚本在 engine 容器内执行，不是在 wings-control 容器内。
# =============================================================================

# set +u: CANN 环境脚本引用未绑定变量
set +u
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/ascend-toolkit/set_env.sh ascend-toolkit/set_env.sh; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; } || echo 'WARN: ascend-toolkit/set_env.sh not found'
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/nnal/atb/set_env.sh nnal/atb/set_env.sh; else source /usr/local/Ascend/nnal/atb/set_env.sh; fi; } || echo 'WARN: nnal/atb/set_env.sh not found'
[ -f /usr/local/Ascend/mindie/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/mindie/set_env.sh mindie/set_env.sh --backend=atb; else source /usr/local/Ascend/mindie/set_env.sh --backend=atb; fi; } || echo 'WARN: mindie/set_env.sh not found'
[ -f /opt/atb-models/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /opt/atb-models/set_env.sh atb-models/set_env.sh; else source /opt/atb-models/set_env.sh; fi; } || echo 'WARN: atb-models/set_env.sh not found'
set -u

export NPU_MEMORY_FRACTION=0.96
printf '[mindie-env] NPU_MEMORY_FRACTION=%s\n' "${NPU_MEMORY_FRACTION:-}"
export ASCEND_GLOBAL_LOG_LEVEL=1
printf '[mindie-env] ASCEND_GLOBAL_LOG_LEVEL=%s\n' "${ASCEND_GLOBAL_LOG_LEVEL:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=0
printf '[mindie-env] ASCEND_SLOG_PRINT_TO_STDOUT=%s\n' "${ASCEND_SLOG_PRINT_TO_STDOUT:-}"
# ── Merge-update MindIE config.json (preserve original, override changed) ──
export _MINDIE_CONFIG_PATH='/usr/local/Ascend/mindie/latest/mindie-service\conf/config.json'
echo "[wings-env] export _MINDIE_CONFIG_PATH=${_MINDIE_CONFIG_PATH:-}"
export _LOCAL_TEMPLATE=/opt/wings-control/config/templates/mindie_service_config.json
echo "[wings-env] export _LOCAL_TEMPLATE=${_LOCAL_TEMPLATE:-}"

cat > /tmp/_mindie_overrides.json << 'OVERRIDES_EOF'
{
  "server": {
    "ipAddress": "10.0.0.1",
    "port": 17000,
    "httpsEnabled": false,
    "inferMode": "standard",
    "openAiSupport": "vllm",
    "tokenTimeout": 3600,
    "e2eTimeout": 65535,
    "allowAllZeroIpListening": false
  },
  "backend": {
    "npuDeviceIds": [
      [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
      ]
    ],
    "multiNodesInferEnabled": false
  },
  "model_deploy": {
    "maxSeqLen": 8192,
    "maxInputTokenLen": 2048,
    "truncation": false
  },
  "model_config": {
    "modelName": "Qwen3-235B-A22B",
    "modelWeightPath": "/models/Qwen3-235B",
    "worldSize": 8,
    "cpuMemSize": 5,
    "npuMemSize": -1,
    "trustRemoteCode": true
  },
  "schedule": {
    "cacheBlockSize": 128,
    "maxPrefillBatchSize": 50,
    "maxPrefillTokens": 8192,
    "prefillTimeMsPerReq": 150,
    "prefillPolicyType": 0,
    "decodeTimeMsPerReq": 50,
    "decodePolicyType": 0,
    "maxBatchSize": 200,
    "maxIterTimes": 2048,
    "maxPreemptCount": 0,
    "supportSelectBatch": false,
    "maxQueueDelayMicroseconds": 5000,
    "bufferResponseEnabled": false,
    "decodeExpectedTime": 50,
    "prefillExpectedTime": 1500
  },
  "extra": {
    "enable_ep_moe": false
  }
}
OVERRIDES_EOF

python3 << 'MERGE_SCRIPT_EOF'
import json, os, sys

CONFIG_PATH = os.environ['_MINDIE_CONFIG_PATH']
LOCAL_TEMPLATE = os.environ.get('_LOCAL_TEMPLATE', '/opt/wings-control/config/templates/mindie_service_config.json')
OVERRIDES_PATH = '/tmp/_mindie_overrides.json'
BACKUP_PATH = CONFIG_PATH + '.orig'

# 1. Load base config (idempotent: always merge from original/template, never from already-merged file)
template_path = LOCAL_TEMPLATE if os.path.isfile(LOCAL_TEMPLATE) else None

if template_path:
    with open(template_path, 'r') as f:
        config = json.load(f)
    for meta_key in ('_comment', '_usage'):
        config.pop(meta_key, None)
    print(f'[mindie] Loaded local template config from {template_path} ({len(json.dumps(config))} chars)')
elif os.path.isfile(BACKUP_PATH):
    # Idempotent: re-merge from the original backup, not the already-merged file
    with open(BACKUP_PATH, 'r') as f:
        config = json.load(f)
    print(f'[mindie] Loaded original backup config ({BACKUP_PATH}, {len(json.dumps(config))} chars)')
else:
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        # First merge: save a backup for future idempotent re-merges
        import shutil
        shutil.copy2(CONFIG_PATH, BACKUP_PATH)
        print(f'[mindie] Loaded original config.json ({len(json.dumps(config))} chars), '
              f'backup saved to {BACKUP_PATH}')
    except Exception as e:
        print(f'[mindie] ERROR: Cannot read {CONFIG_PATH}: {e}', file=sys.stderr)
        sys.exit(1)

# 2. Load overrides
with open(OVERRIDES_PATH, 'r') as f:
    ov = json.load(f)

# 3. Merge (update only specified keys; keep all other original fields intact)
if 'ServerConfig' in config:
    config['ServerConfig'].update(ov['server'])

if 'BackendConfig' in config:
    bc = config['BackendConfig']
    bc.update(ov['backend'])

    if 'ModelDeployConfig' in bc:
        bc['ModelDeployConfig'].update(ov['model_deploy'])
        mc = bc['ModelDeployConfig'].get('ModelConfig')
        if isinstance(mc, list) and len(mc) > 0 and isinstance(mc[0], dict):
            mc[0].update(ov['model_config'])
        elif mc is not None:
            print(f'[mindie] WARNING: ModelConfig has unexpected type/value: '
                  f'{type(mc).__name__} = {mc}', file=sys.stderr)
            # Force-create a valid ModelConfig array with our overrides
            bc['ModelDeployConfig']['ModelConfig'] = [ov['model_config']]
            print('[mindie] Created ModelConfig[0] from overrides')
        else:
            bc['ModelDeployConfig']['ModelConfig'] = [ov['model_config']]
            print('[mindie] ModelConfig was missing/None, created from overrides')

    if 'ScheduleConfig' in bc:
        bc['ScheduleConfig'].update(ov['schedule'])

# 4. Apply extra pass-through keys to config root level
extra = ov.get('extra', {})
if extra:
    config.update(extra)
    print(f'[mindie] Applied {len(extra)} extra pass-through keys: {list(extra.keys())}')

# 5. Write back (atomic: write to tmp then rename)
tmp_out = CONFIG_PATH + '.tmp'
with open(tmp_out, 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
os.chmod(tmp_out, 0o640)
os.replace(tmp_out, CONFIG_PATH)

print('[mindie] config.json merge-updated successfully')
print(json.dumps(config, indent=2, ensure_ascii=False))
MERGE_SCRIPT_EOF

# ── Start MindIE daemon (background + wait, per official boot.sh) ────────────
cd /usr/local/Ascend/mindie/latest/mindie-service
echo '[wings-cmd] >>> ./bin/mindieservice_daemon'
./bin/mindieservice_daemon &
MINDIE_PID=$!
echo "[mindie] Daemon started as PID $MINDIE_PID"
wait $MINDIE_PID
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "[mindie] ERROR: daemon exited with code $exit_code"
fi
exit $exit_code &
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
    # =============================================================================
    # MindIE 单机引擎环境初始化脚本
    # 用途: 被 _build_base_env_commands() 读取并内联到 start_command.sh
    # 来源: 参考 wings/config/set_mindie_single_env.sh，适配 sidecar 架构
    #
    # 注意: 此脚本在 engine 容器内执行，不是在 wings-control 容器内。
    # =============================================================================

    # set +u: CANN 环境脚本引用未绑定变量
    set +u
    [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/ascend-toolkit/set_env.sh ascend-toolkit/set_env.sh; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; } || echo 'WARN: ascend-toolkit/set_env.sh not found'
    [ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/nnal/atb/set_env.sh nnal/atb/set_env.sh; else source /usr/local/Ascend/nnal/atb/set_env.sh; fi; } || echo 'WARN: nnal/atb/set_env.sh not found'
    [ -f /usr/local/Ascend/mindie/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/mindie/set_env.sh mindie/set_env.sh --backend=atb; else source /usr/local/Ascend/mindie/set_env.sh --backend=atb; fi; } || echo 'WARN: mindie/set_env.sh not found'
    [ -f /opt/atb-models/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /opt/atb-models/set_env.sh atb-models/set_env.sh; else source /opt/atb-models/set_env.sh; fi; } || echo 'WARN: atb-models/set_env.sh not found'
    set -u

    export NPU_MEMORY_FRACTION=0.96
    printf '[mindie-env] NPU_MEMORY_FRACTION=%s\n' "${NPU_MEMORY_FRACTION:-}"
    export ASCEND_GLOBAL_LOG_LEVEL=1
    printf '[mindie-env] ASCEND_GLOBAL_LOG_LEVEL=%s\n' "${ASCEND_GLOBAL_LOG_LEVEL:-}"
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    printf '[mindie-env] ASCEND_SLOG_PRINT_TO_STDOUT=%s\n' "${ASCEND_SLOG_PRINT_TO_STDOUT:-}"
    # ── Merge-update MindIE config.json (preserve original, override changed) ──
    export _MINDIE_CONFIG_PATH='/usr/local/Ascend/mindie/latest/mindie-service\conf/config.json'
    echo "[wings-env] export _MINDIE_CONFIG_PATH=${_MINDIE_CONFIG_PATH:-}"
    export _LOCAL_TEMPLATE=/opt/wings-control/config/templates/mindie_service_config.json
    echo "[wings-env] export _LOCAL_TEMPLATE=${_LOCAL_TEMPLATE:-}"

    cat > /tmp/_mindie_overrides.json << 'OVERRIDES_EOF'
    {
      "server": {
        "ipAddress": "10.0.0.1",
        "port": 17000,
        "httpsEnabled": false,
        "inferMode": "standard",
        "openAiSupport": "vllm",
        "tokenTimeout": 3600,
        "e2eTimeout": 65535,
        "allowAllZeroIpListening": false
      },
      "backend": {
        "npuDeviceIds": [
          [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7
          ]
        ],
        "multiNodesInferEnabled": false
      },
      "model_deploy": {
        "maxSeqLen": 8192,
        "maxInputTokenLen": 2048,
        "truncation": false
      },
      "model_config": {
        "modelName": "Qwen3-235B-A22B",
        "modelWeightPath": "/models/Qwen3-235B",
        "worldSize": 8,
        "cpuMemSize": 5,
        "npuMemSize": -1,
        "trustRemoteCode": true
      },
      "schedule": {
        "cacheBlockSize": 128,
        "maxPrefillBatchSize": 50,
        "maxPrefillTokens": 8192,
        "prefillTimeMsPerReq": 150,
        "prefillPolicyType": 0,
        "decodeTimeMsPerReq": 50,
        "decodePolicyType": 0,
        "maxBatchSize": 200,
        "maxIterTimes": 2048,
        "maxPreemptCount": 0,
        "supportSelectBatch": false,
        "maxQueueDelayMicroseconds": 5000,
        "bufferResponseEnabled": false,
        "decodeExpectedTime": 50,
        "prefillExpectedTime": 1500
      },
      "extra": {
        "enable_ep_moe": false
      }
    }
    OVERRIDES_EOF

    python3 << 'MERGE_SCRIPT_EOF'
    import json, os, sys

    CONFIG_PATH = os.environ['_MINDIE_CONFIG_PATH']
    LOCAL_TEMPLATE = os.environ.get('_LOCAL_TEMPLATE', '/opt/wings-control/config/templates/mindie_service_config.json')
    OVERRIDES_PATH = '/tmp/_mindie_overrides.json'
    BACKUP_PATH = CONFIG_PATH + '.orig'

    # 1. Load base config (idempotent: always merge from original/template, never from already-merged file)
    template_path = LOCAL_TEMPLATE if os.path.isfile(LOCAL_TEMPLATE) else None

    if template_path:
        with open(template_path, 'r') as f:
            config = json.load(f)
        for meta_key in ('_comment', '_usage'):
            config.pop(meta_key, None)
        print(f'[mindie] Loaded local template config from {template_path} ({len(json.dumps(config))} chars)')
    elif os.path.isfile(BACKUP_PATH):
        # Idempotent: re-merge from the original backup, not the already-merged file
        with open(BACKUP_PATH, 'r') as f:
            config = json.load(f)
        print(f'[mindie] Loaded original backup config ({BACKUP_PATH}, {len(json.dumps(config))} chars)')
    else:
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            # First merge: save a backup for future idempotent re-merges
            import shutil
            shutil.copy2(CONFIG_PATH, BACKUP_PATH)
            print(f'[mindie] Loaded original config.json ({len(json.dumps(config))} chars), '
                  f'backup saved to {BACKUP_PATH}')
        except Exception as e:
            print(f'[mindie] ERROR: Cannot read {CONFIG_PATH}: {e}', file=sys.stderr)
            sys.exit(1)

    # 2. Load overrides
    with open(OVERRIDES_PATH, 'r') as f:
        ov = json.load(f)

    # 3. Merge (update only specified keys; keep all other original fields intact)
    if 'ServerConfig' in config:
        config['ServerConfig'].update(ov['server'])

    if 'BackendConfig' in config:
        bc = config['BackendConfig']
        bc.update(ov['backend'])

        if 'ModelDeployConfig' in bc:
            bc['ModelDeployConfig'].update(ov['model_deploy'])
            mc = bc['ModelDeployConfig'].get('ModelConfig')
            if isinstance(mc, list) and len(mc) > 0 and isinstance(mc[0], dict):
                mc[0].update(ov['model_config'])
            elif mc is not None:
                print(f'[mindie] WARNING: ModelConfig has unexpected type/value: '
                      f'{type(mc).__name__} = {mc}', file=sys.stderr)
                # Force-create a valid ModelConfig array with our overrides
                bc['ModelDeployConfig']['ModelConfig'] = [ov['model_config']]
                print('[mindie] Created ModelConfig[0] from overrides')
            else:
                bc['ModelDeployConfig']['ModelConfig'] = [ov['model_config']]
                print('[mindie] ModelConfig was missing/None, created from overrides')

        if 'ScheduleConfig' in bc:
            bc['ScheduleConfig'].update(ov['schedule'])

    # 4. Apply extra pass-through keys to config root level
    extra = ov.get('extra', {})
    if extra:
        config.update(extra)
        print(f'[mindie] Applied {len(extra)} extra pass-through keys: {list(extra.keys())}')

    # 5. Write back (atomic: write to tmp then rename)
    tmp_out = CONFIG_PATH + '.tmp'
    with open(tmp_out, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    os.chmod(tmp_out, 0o640)
    os.replace(tmp_out, CONFIG_PATH)

    print('[mindie] config.json merge-updated successfully')
    print(json.dumps(config, indent=2, ensure_ascii=False))
    MERGE_SCRIPT_EOF

    # ── Start MindIE daemon (background + wait, per official boot.sh) ────────────
    cd /usr/local/Ascend/mindie/latest/mindie-service
    echo '[wings-cmd] >>> ./bin/mindieservice_daemon'
    ./bin/mindieservice_daemon &
    MINDIE_PID=$!
    echo "[mindie] Daemon started as PID $MINDIE_PID"
    wait $MINDIE_PID
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "[mindie] ERROR: daemon exited with code $exit_code"
    fi
    exit $exit_code &
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

