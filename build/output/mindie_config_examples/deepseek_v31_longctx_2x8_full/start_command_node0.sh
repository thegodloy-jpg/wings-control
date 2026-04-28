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

# ── Ascend HCCL distributed env vars (multi-node TP) ──
export MASTER_ADDR=10.254.13.90
printf '[mindie-env] MASTER_ADDR=%s\n' "${MASTER_ADDR:-}"
export MASTER_PORT=27070
printf '[mindie-env] MASTER_PORT=%s\n' "${MASTER_PORT:-}"
export RANK=0
printf '[mindie-env] RANK=%s\n' "${RANK:-}"
export WORLD_SIZE=16
printf '[mindie-env] WORLD_SIZE=%s\n' "${WORLD_SIZE:-}"
export MINDIE_MODEL_WORLD_SIZE=16
printf '[mindie-env] MINDIE_MODEL_WORLD_SIZE=%s\n' "${MINDIE_MODEL_WORLD_SIZE:-}"
export HCCL_WHITELIST_DISABLE=1
printf '[mindie-env] HCCL_WHITELIST_DISABLE=%s\n' "${HCCL_WHITELIST_DISABLE:-}"
export HCCL_IF_IP=10.254.13.90
printf '[mindie-env] HCCL_IF_IP=%s\n' "${HCCL_IF_IP:-}"
export HCCL_SOCKET_IFNAME=eth0
printf '[mindie-env] HCCL_SOCKET_IFNAME=%s\n' "${HCCL_SOCKET_IFNAME:-}"
export GLOO_SOCKET_IFNAME=eth0
printf '[mindie-env] GLOO_SOCKET_IFNAME=%s\n' "${GLOO_SOCKET_IFNAME:-}"
export MIES_CONTAINER_IP=10.254.13.90
printf '[mindie-env] MIES_CONTAINER_IP=%s\n' "${MIES_CONTAINER_IP:-}"
export RANK_TABLE_FILE=/shared-volume/hccl_ranktable.json
printf '[mindie-env] RANK_TABLE_FILE=%s\n' "${RANK_TABLE_FILE:-}"

# ── MindIE distributed env defaults ──
# MindIE 多节点分布式环境变量默认值。
# 保持仅包含 export 命令；adapter 会将这些命令内联到 start_command.sh。
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
printf '[mindie-env] PYTORCH_NPU_ALLOC_CONF=%s\n' "${PYTORCH_NPU_ALLOC_CONF:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
printf '[mindie-env] OMP_NUM_THREADS=%s\n' "${OMP_NUM_THREADS:-}"
export HCCL_DETERMINISTIC="${HCCL_DETERMINISTIC:-false}"
printf '[mindie-env] HCCL_DETERMINISTIC=%s\n' "${HCCL_DETERMINISTIC:-}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
printf '[mindie-env] HCCL_OP_EXPANSION_MODE=%s\n' "${HCCL_OP_EXPANSION_MODE:-}"
export ATB_LLM_HCCL_ENABLE="${ATB_LLM_HCCL_ENABLE:-1}"
printf '[mindie-env] ATB_LLM_HCCL_ENABLE=%s\n' "${ATB_LLM_HCCL_ENABLE:-}"
export ATB_LLM_COMM_BACKEND="${ATB_LLM_COMM_BACKEND:-hccl}"
printf '[mindie-env] ATB_LLM_COMM_BACKEND=%s\n' "${ATB_LLM_COMM_BACKEND:-}"
export INF_NAN_MODE_ENABLE="${INF_NAN_MODE_ENABLE:-0}"
printf '[mindie-env] INF_NAN_MODE_ENABLE=%s\n' "${INF_NAN_MODE_ENABLE:-}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
printf '[mindie-env] HCCL_CONNECT_TIMEOUT=%s\n' "${HCCL_CONNECT_TIMEOUT:-}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-600}"
printf '[mindie-env] HCCL_EXEC_TIMEOUT=%s\n' "${HCCL_EXEC_TIMEOUT:-}"
export MINDIE_LOG_TO_STDOUT="${MINDIE_LOG_TO_STDOUT:-1}"
printf '[mindie-env] MINDIE_LOG_TO_STDOUT=%s\n' "${MINDIE_LOG_TO_STDOUT:-}"
# ── Merge-update MindIE config.json (preserve original, override changed) ──
export _MINDIE_CONFIG_PATH='/usr/local/Ascend/mindie/latest/mindie-service\conf/config.json'
_LOCAL_TEMPLATE=/opt/wings-control/config/defaults/mindie_service_config.json

cat > /tmp/_mindie_overrides.json << 'OVERRIDES_EOF'
{
  "server": {
    "ipAddress": "10.254.13.90",
    "port": 17000,
    "httpsEnabled": false,
    "inferMode": "standard",
    "openAiSupport": "vllm",
    "tokenTimeout": 3600,
    "e2eTimeout": 65535,
    "allowAllZeroIpListening": false,
    "interCommTLSEnabled": false
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
    "multiNodesInferEnabled": true,
    "interNodeTLSEnabled": false
  },
  "model_deploy": {
    "maxSeqLen": 12048,
    "maxInputTokenLen": 10000,
    "truncation": false
  },
  "model_config": {
    "modelName": "DeepSeek-V3.1",
    "modelWeightPath": "/usr/local/serving/models/",
    "worldSize": 8,
    "cpuMemSize": 5,
    "npuMemSize": -1,
    "trustRemoteCode": true,
    "tp": 8,
    "dp": 1,
    "sp": 8,
    "cp": 2,
    "moe_ep": 16,
    "moe_tp": 1,
    "models": {
      "deepseekv2": {
        "tool_call_options": {
          "tool_call_parser": "deepseek_v31"
        }
      }
    }
  },
  "schedule": {
    "cacheBlockSize": 128,
    "maxPrefillBatchSize": 50,
    "maxPrefillTokens": 10000,
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
    "enable_ep_moe": true
  }
}
OVERRIDES_EOF

python3 << 'MERGE_SCRIPT_EOF'
import json, os, sys

CONFIG_PATH = os.environ['_MINDIE_CONFIG_PATH']
LOCAL_TEMPLATE = os.environ.get('_LOCAL_TEMPLATE', '/opt/wings-control/config/defaults/mindie_service_config.json')
OVERRIDES_PATH = '/tmp/_mindie_overrides.json'
BACKUP_PATH = CONFIG_PATH + '.orig'

# 1. Load base config (idempotent: always merge from original/template, never from already-merged file)
if os.path.isfile(LOCAL_TEMPLATE):
    with open(LOCAL_TEMPLATE, 'r') as f:
        config = json.load(f)
    for meta_key in ('_comment', '_usage'):
        config.pop(meta_key, None)
    print(f'[mindie] Loaded LOCAL template config ({len(json.dumps(config))} chars)')
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
./bin/mindieservice_daemon &
MINDIE_PID=$!
echo "[mindie] Daemon started as PID $MINDIE_PID"
wait $MINDIE_PID
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "[mindie] ERROR: daemon exited with code $exit_code"
fi
exit $exit_code
