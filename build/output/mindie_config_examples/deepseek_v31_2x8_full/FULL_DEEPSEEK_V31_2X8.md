# DeepSeek-V3.1 2x8 dual-node full generated view

This document focuses only on one scenario: DeepSeek-V3.1, 2 nodes x 8 NPUs. DeepSeek-V3.1 is treated as MoE, so moe_tp/moe_ep are included without enable_expert_parallel.

## config.json

Source: config.json

```json
{
  "Version": "1.0.0",
  "ServerConfig": {
    "ipAddress": "112.254.176.114",
    "managementIpAddress": "127.0.0.2",
    "port": 17000,
    "managementPort": 1026,
    "metricsPort": 1027,
    "allowAllZeroIpListening": false,
    "maxLinkNum": 1000,
    "httpsEnabled": false,
    "fullTextEnabled": false,
    "tlsCaPath": "security/ca/",
    "tlsCaFile": [
      "ca.pem"
    ],
    "tlsCert": "security/certs/server.pem",
    "tlsPk": "security/keys/server.key.pem",
    "tlsPkPwd": "security/pass/key_pwd.txt",
    "tlsCrlPath": "security/certs/",
    "tlsCrlFiles": [
      "server_crl.pem"
    ],
    "managementTlsCaFile": [
      "management_ca.pem"
    ],
    "managementTlsCert": "security/certs/management/server.pem",
    "managementTlsPk": "security/keys/management/server.key.pem",
    "managementTlsPkPwd": "security/pass/management/key_pwd.txt",
    "managementTlsCrlPath": "security/management/certs/",
    "managementTlsCrlFiles": [
      "server_crl.pem"
    ],
    "metricsTlsCaFile": [
      "metrics_ca.pem"
    ],
    "metricsTlsCert": "security/certs/metrics/server.pem",
    "metricsTlsPk": "security/keys/metrics/server.key.pem",
    "metricsTlsPkPwd": "security/pass/metrics/key_pwd.txt",
    "metricsTlsCrlPath": "security/metrics/certs/",
    "metricsTlsCrlFiles": [
      "server_crl.pem"
    ],
    "kmcKsfMaster": "tools/pmt/master/ksfa",
    "kmcKsfStandby": "tools/pmt/standby/ksfb",
    "inferMode": "standard",
    "interCommTLSEnabled": false,
    "interCommPort": 1121,
    "interCommTlsCaPath": "security/grpc/ca/",
    "interCommTlsCaFiles": [
      "ca.pem"
    ],
    "interCommTlsCert": "security/grpc/certs/server.pem",
    "interCommPk": "security/grpc/keys/server.key.pem",
    "interCommPkPwd": "security/grpc/pass/key_pwd.txt",
    "interCommTlsCrlPath": "security/grpc/certs/",
    "interCommTlsCrlFiles": [
      "server_crl.pem"
    ],
    "openAiSupport": "vllm",
    "tokenTimeout": 3600,
    "e2eTimeout": 65535,
    "distDPServerEnabled": false
  },
  "BackendConfig": {
    "backendName": "mindieservice_llm_engine",
    "modelInstanceNumber": 1,
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
    "tokenizerProcessNumber": 8,
    "multiNodesInferEnabled": true,
    "multiNodesInferPort": 1120,
    "interNodeTLSEnabled": false,
    "interNodeTlsCaPath": "security/grpc/ca/",
    "interNodeTlsCaFiles": [
      "ca.pem"
    ],
    "interNodeTlsCert": "security/grpc/certs/server.pem",
    "interNodeTlsPk": "security/grpc/keys/server.key.pem",
    "interNodeTlsPkPwd": "security/grpc/pass/mindie_server_key_pwd.txt",
    "interNodeTlsCrlPath": "security/grpc/certs/",
    "interNodeTlsCrlFiles": [
      "server_crl.pem"
    ],
    "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
    "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb",
    "kvPoolConfig": {
      "backend": "",
      "configPath": ""
    },
    "ModelDeployConfig": {
      "maxSeqLen": 2560,
      "maxInputTokenLen": 2048,
      "truncation": false,
      "ModelConfig": [
        {
          "modelInstanceType": "Standard",
          "modelName": "DeepSeek-V3.1",
          "modelWeightPath": "/usr/local/serving/models/",
          "worldSize": 8,
          "cpuMemSize": 5,
          "npuMemSize": -1,
          "backendType": "atb",
          "trustRemoteCode": true,
          "async_scheduler_wait_time": 120,
          "kv_trans_timeout": 10,
          "kv_link_timeout": 1080,
          "tp": 16,
          "dp": 1,
          "moe_tp": 1,
          "moe_ep": 16,
          "models": {
            "deepseekv2": {
              "tool_call_options": {
                "tool_call_parser": "deepseek_v31"
              }
            }
          }
        }
      ]
    },
    "ScheduleConfig": {
      "templateType": "Standard",
      "templateName": "Standard_LLM",
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
      "maxFirstTokenWaitTime": 2500,
      "bufferResponseEnabled": false,
      "decodeExpectedTime": 50,
      "prefillExpectedTime": 1500
    }
  },
  "LogConfig": {
    "dynamicLogLevel": "",
    "dynamicLogLevelValidHours": 2,
    "dynamicLogLevelValidTime": ""
  },
  "EnableDynamicAdjustTimeoutConfig": false,
  "enable_ep_moe": false
}
```

## start_command.sh explicit environment variables: node0

Source: start_command_env_node0.sh

```bash
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/ascend-toolkit/set_env.sh ascend-toolkit/set_env.sh; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; } || echo 'WARN: ascend-toolkit/set_env.sh not found'
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/nnal/atb/set_env.sh nnal/atb/set_env.sh; else source /usr/local/Ascend/nnal/atb/set_env.sh; fi; } || echo 'WARN: nnal/atb/set_env.sh not found'
[ -f /usr/local/Ascend/mindie/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/mindie/set_env.sh mindie/set_env.sh --backend=atb; else source /usr/local/Ascend/mindie/set_env.sh --backend=atb; fi; } || echo 'WARN: mindie/set_env.sh not found'
[ -f /opt/atb-models/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /opt/atb-models/set_env.sh atb-models/set_env.sh; else source /opt/atb-models/set_env.sh; fi; } || echo 'WARN: atb-models/set_env.sh not found'
export NPU_MEMORY_FRACTION=0.96
printf '[mindie-env] NPU_MEMORY_FRACTION=%s\n' "${NPU_MEMORY_FRACTION:-}"
export ASCEND_GLOBAL_LOG_LEVEL=1
printf '[mindie-env] ASCEND_GLOBAL_LOG_LEVEL=%s\n' "${ASCEND_GLOBAL_LOG_LEVEL:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=0
printf '[mindie-env] ASCEND_SLOG_PRINT_TO_STDOUT=%s\n' "${ASCEND_SLOG_PRINT_TO_STDOUT:-}"
export MASTER_ADDR=112.254.176.114
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
export HCCL_IF_IP=112.254.176.114
printf '[mindie-env] HCCL_IF_IP=%s\n' "${HCCL_IF_IP:-}"
export HCCL_SOCKET_IFNAME=eth0
printf '[mindie-env] HCCL_SOCKET_IFNAME=%s\n' "${HCCL_SOCKET_IFNAME:-}"
export GLOO_SOCKET_IFNAME=eth0
printf '[mindie-env] GLOO_SOCKET_IFNAME=%s\n' "${GLOO_SOCKET_IFNAME:-}"
export MIES_CONTAINER_IP=112.254.176.114
printf '[mindie-env] MIES_CONTAINER_IP=%s\n' "${MIES_CONTAINER_IP:-}"
export RANK_TABLE_FILE=/shared-volume/hccl_ranktable.json
printf '[mindie-env] RANK_TABLE_FILE=%s\n' "${RANK_TABLE_FILE:-}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
printf '[mindie-env] PYTORCH_NPU_ALLOC_CONF=%s\n' "${PYTORCH_NPU_ALLOC_CONF:-}"
export NPU_MEMORY_FRACTION="${NPU_MEMORY_FRACTION:-0.9}"
printf '[mindie-env] NPU_MEMORY_FRACTION=%s\n' "${NPU_MEMORY_FRACTION:-}"
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
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
printf '[mindie-env] ASCEND_GLOBAL_LOG_LEVEL=%s\n' "${ASCEND_GLOBAL_LOG_LEVEL:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
printf '[mindie-env] ASCEND_SLOG_PRINT_TO_STDOUT=%s\n' "${ASCEND_SLOG_PRINT_TO_STDOUT:-}"
export MINDIE_LOG_TO_STDOUT="${MINDIE_LOG_TO_STDOUT:-1}"
printf '[mindie-env] MINDIE_LOG_TO_STDOUT=%s\n' "${MINDIE_LOG_TO_STDOUT:-}"
export _MINDIE_CONFIG_PATH='/usr/local/Ascend/mindie/latest/mindie-service\conf/config.json'
```

## start_command.sh explicit environment variables: node1

Source: start_command_env_node1.sh

```bash
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/ascend-toolkit/set_env.sh ascend-toolkit/set_env.sh; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; } || echo 'WARN: ascend-toolkit/set_env.sh not found'
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/nnal/atb/set_env.sh nnal/atb/set_env.sh; else source /usr/local/Ascend/nnal/atb/set_env.sh; fi; } || echo 'WARN: nnal/atb/set_env.sh not found'
[ -f /usr/local/Ascend/mindie/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/mindie/set_env.sh mindie/set_env.sh --backend=atb; else source /usr/local/Ascend/mindie/set_env.sh --backend=atb; fi; } || echo 'WARN: mindie/set_env.sh not found'
[ -f /opt/atb-models/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /opt/atb-models/set_env.sh atb-models/set_env.sh; else source /opt/atb-models/set_env.sh; fi; } || echo 'WARN: atb-models/set_env.sh not found'
export NPU_MEMORY_FRACTION=0.96
printf '[mindie-env] NPU_MEMORY_FRACTION=%s\n' "${NPU_MEMORY_FRACTION:-}"
export ASCEND_GLOBAL_LOG_LEVEL=1
printf '[mindie-env] ASCEND_GLOBAL_LOG_LEVEL=%s\n' "${ASCEND_GLOBAL_LOG_LEVEL:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=0
printf '[mindie-env] ASCEND_SLOG_PRINT_TO_STDOUT=%s\n' "${ASCEND_SLOG_PRINT_TO_STDOUT:-}"
export MASTER_ADDR=112.254.176.114
printf '[mindie-env] MASTER_ADDR=%s\n' "${MASTER_ADDR:-}"
export MASTER_PORT=27070
printf '[mindie-env] MASTER_PORT=%s\n' "${MASTER_PORT:-}"
export RANK=1
printf '[mindie-env] RANK=%s\n' "${RANK:-}"
export WORLD_SIZE=16
printf '[mindie-env] WORLD_SIZE=%s\n' "${WORLD_SIZE:-}"
export MINDIE_MODEL_WORLD_SIZE=16
printf '[mindie-env] MINDIE_MODEL_WORLD_SIZE=%s\n' "${MINDIE_MODEL_WORLD_SIZE:-}"
export HCCL_WHITELIST_DISABLE=1
printf '[mindie-env] HCCL_WHITELIST_DISABLE=%s\n' "${HCCL_WHITELIST_DISABLE:-}"
export HCCL_IF_IP=112.254.176.115
printf '[mindie-env] HCCL_IF_IP=%s\n' "${HCCL_IF_IP:-}"
export HCCL_SOCKET_IFNAME=eth0
printf '[mindie-env] HCCL_SOCKET_IFNAME=%s\n' "${HCCL_SOCKET_IFNAME:-}"
export GLOO_SOCKET_IFNAME=eth0
printf '[mindie-env] GLOO_SOCKET_IFNAME=%s\n' "${GLOO_SOCKET_IFNAME:-}"
export MIES_CONTAINER_IP=112.254.176.115
printf '[mindie-env] MIES_CONTAINER_IP=%s\n' "${MIES_CONTAINER_IP:-}"
export RANK_TABLE_FILE=/shared-volume/hccl_ranktable.json
printf '[mindie-env] RANK_TABLE_FILE=%s\n' "${RANK_TABLE_FILE:-}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
printf '[mindie-env] PYTORCH_NPU_ALLOC_CONF=%s\n' "${PYTORCH_NPU_ALLOC_CONF:-}"
export NPU_MEMORY_FRACTION="${NPU_MEMORY_FRACTION:-0.9}"
printf '[mindie-env] NPU_MEMORY_FRACTION=%s\n' "${NPU_MEMORY_FRACTION:-}"
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
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
printf '[mindie-env] ASCEND_GLOBAL_LOG_LEVEL=%s\n' "${ASCEND_GLOBAL_LOG_LEVEL:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
printf '[mindie-env] ASCEND_SLOG_PRINT_TO_STDOUT=%s\n' "${ASCEND_SLOG_PRINT_TO_STDOUT:-}"
export MINDIE_LOG_TO_STDOUT="${MINDIE_LOG_TO_STDOUT:-1}"
printf '[mindie-env] MINDIE_LOG_TO_STDOUT=%s\n' "${MINDIE_LOG_TO_STDOUT:-}"
export _MINDIE_CONFIG_PATH='/usr/local/Ascend/mindie/latest/mindie-service\conf/config.json'
```

## Full start_command.sh files

- start_command_node0.sh
- start_command_node1.sh
