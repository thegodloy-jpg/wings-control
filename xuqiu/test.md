# Qwen3-30B-A3B P/D 双机分布式 - vLLM Ascend 真实下发命令验证

> 基于真实容器下发命令：
>
> ```bash
> exec bash /opt/wings-control/wings_start.sh --engine vllm_ascend --model-name Qwen3-30B-A3B --model-path /usr/local/serving/models/ --port 18000 --distributed --seed 42 --trust-remote-code --dtype auto --output-length 2048 --enable-prefix-caching --max-num-batched-tokens 4096 --max-num-seqs 256 --kv-cache-dtype auto --block-size 16 --gpu-memory-utilization 0.95 --input-length 2048 --enable-chunked-prefill --gpu-usage-mode full --device-count 4
> ```
>
> 真实 env 中 `PD_PREFILL_DP_SIZE=2`、`PD_DECODE_DP_SIZE=2`、`PD_PREFILL_TP_SIZE=4`、`PD_DECODE_TP_SIZE=4`、`DP_SIZE_LOCAL=1`。因此这不是 1P1D 两实例，而是 P 角色 2 个 DP 成员 + D 角色 2 个 DP 成员，共 4 个实例。

## 必须下发的拓扑 env

真实下发片段里已看到：

```bash
PD_ROLE=P
ENGINE=vllm_ascend
ENGINE_VERSION=v0.21.0rc1-a3
PD_INDEX=0
PD_PREFILL_DP_SIZE=2
PD_PREFILL_TP_SIZE=4
PD_DECODE_DP_SIZE=2
PD_DECODE_TP_SIZE=4
DP_SIZE_LOCAL=1
ENGINE_TYPE=1
```

同角色 DP=2 时，还必须按角色下发 `Master_IP` / `NODE_IPS` / `RANK_IP`。否则当前代码会回退到本机 `RANK_IP`，worker 侧会出现 rank/address 错位。

```bash
# P0
export PD_ROLE=P
export PD_INDEX=0
export Master_IP=10.254.0.1
export NODE_IPS=10.254.0.1,10.254.0.3
export RANK_IP=10.254.0.1

# P1
export PD_ROLE=P
export PD_INDEX=1
export Master_IP=10.254.0.1
export NODE_IPS=10.254.0.1,10.254.0.3
export RANK_IP=10.254.0.3

# D0
export PD_ROLE=D
export PD_INDEX=2
export Master_IP=10.254.0.2
export NODE_IPS=10.254.0.2,10.254.0.4
export RANK_IP=10.254.0.2

# D1
export PD_ROLE=D
export PD_INDEX=3
export Master_IP=10.254.0.2
export NODE_IPS=10.254.0.2,10.254.0.4
export RANK_IP=10.254.0.4

# 四个 pod 公共
export PD_PREFILL_DP_SIZE=2
export PD_PREFILL_TP_SIZE=4
export PD_DECODE_DP_SIZE=2
export PD_DECODE_TP_SIZE=4
export DP_SIZE_LOCAL=1
export WINGS_DEVICE=ascend
export ENGINE=vllm_ascend
export ENGINE_VERSION=v0.21.0rc1-a3
export ENGINE_TYPE=1
export DEVICE_COUNT=4
```

## 最终命令：P0

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_MOONCAKE_BOOTSTRAP_PORT=23000 \
python3 -m vllm.entrypoints.openai.api_server \
  --trust-remote-code --max-model-len 4096 --host 10.254.0.1 \
  --served-model-name Qwen3-30B-A3B --model /usr/local/serving/models/ \
  --dtype auto --kv-cache-dtype auto --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill --max-num-batched-tokens 8192 --block-size 16 \
  --max-num-seqs 4 --seed 42 --enable-expert-parallel --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --kv-transfer-config '{"kv_connector":"MooncakeLayerwiseConnector","kv_role":"kv_producer","kv_port":"30000","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":4},"decode":{"dp_size":2,"tp_size":4}},"kv_buffer_device":"npu","engine_id":"0"}' \
  --enforce-eager \
  --additional-config '{"enable_cpu_binding":"True"}' \
  --port 17000 --tensor-parallel-size 4 \
  --data-parallel-size 2 --data-parallel-rank 0 --data-parallel-size-local 1 \
  --data-parallel-address 10.254.0.1 --data-parallel-rpc-port 12890 \
  --data-parallel-external-lb
```

## 最终命令：P1

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_MOONCAKE_BOOTSTRAP_PORT=23000 \
python3 -m vllm.entrypoints.openai.api_server \
  --trust-remote-code --max-model-len 4096 --host 10.254.0.3 \
  --served-model-name Qwen3-30B-A3B --model /usr/local/serving/models/ \
  --dtype auto --kv-cache-dtype auto --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill --max-num-batched-tokens 8192 --block-size 16 \
  --max-num-seqs 4 --seed 42 --enable-expert-parallel --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --kv-transfer-config '{"kv_connector":"MooncakeLayerwiseConnector","kv_role":"kv_producer","kv_port":"30100","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":4},"decode":{"dp_size":2,"tp_size":4}},"kv_buffer_device":"npu","engine_id":"1"}' \
  --enforce-eager \
  --additional-config '{"enable_cpu_binding":"True"}' \
  --port 17000 --tensor-parallel-size 4 \
  --data-parallel-size 2 --data-parallel-rank 1 --data-parallel-size-local 1 \
  --data-parallel-address 10.254.0.1 --data-parallel-rpc-port 12890 \
  --data-parallel-external-lb
```

## 最终命令：D0

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_MOONCAKE_BOOTSTRAP_PORT=23100 \
python3 -m vllm.entrypoints.openai.api_server \
  --trust-remote-code --max-model-len 4096 --host 10.254.0.2 \
  --served-model-name Qwen3-30B-A3B --model /usr/local/serving/models/ \
  --dtype auto --kv-cache-dtype auto --gpu-memory-utilization 0.88 \
  --enable-chunked-prefill --max-num-batched-tokens 120 --block-size 16 \
  --max-num-seqs 60 --seed 42 --enable-expert-parallel --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --kv-transfer-config '{"kv_connector":"MooncakeLayerwiseConnector","kv_role":"kv_consumer","kv_port":"30200","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":4},"decode":{"dp_size":2,"tp_size":4}},"kv_buffer_device":"npu","engine_id":"2"}' \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 17000 --tensor-parallel-size 4 \
  --data-parallel-size 2 --data-parallel-rank 0 --data-parallel-size-local 1 \
  --data-parallel-address 10.254.0.2 --data-parallel-rpc-port 12777 \
  --data-parallel-external-lb
```

## 最终命令：D1

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_MOONCAKE_BOOTSTRAP_PORT=23100 \
python3 -m vllm.entrypoints.openai.api_server \
  --trust-remote-code --max-model-len 4096 --host 10.254.0.4 \
  --served-model-name Qwen3-30B-A3B --model /usr/local/serving/models/ \
  --dtype auto --kv-cache-dtype auto --gpu-memory-utilization 0.88 \
  --enable-chunked-prefill --max-num-batched-tokens 120 --block-size 16 \
  --max-num-seqs 60 --seed 42 --enable-expert-parallel --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --kv-transfer-config '{"kv_connector":"MooncakeLayerwiseConnector","kv_role":"kv_consumer","kv_port":"30300","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":4},"decode":{"dp_size":2,"tp_size":4}},"kv_buffer_device":"npu","engine_id":"3"}' \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 17000 --tensor-parallel-size 4 \
  --data-parallel-size 2 --data-parallel-rank 1 --data-parallel-size-local 1 \
  --data-parallel-address 10.254.0.2 --data-parallel-rpc-port 12777 \
  --data-parallel-external-lb
```

## 验证结论

本地按上述真实下发拓扑生成了 4 个脚本：

```text
build/output/start_command_pd-qwen3-real-dp2-tp4-P_node0.sh
build/output/start_command_pd-qwen3-real-dp2-tp4-P_node1.sh
build/output/start_command_pd-qwen3-real-dp2-tp4-D_node0.sh
build/output/start_command_pd-qwen3-real-dp2-tp4-D_node1.sh
```

生成日志中的关键值：

| 节点 | PD_INDEX | rank_start | data-parallel-address | rpc |
|------|----------|------------|-----------------------|-----|
| P0 | 0 | 0 | 10.254.0.1 | 12890 |
| P1 | 1 | 1 | 10.254.0.1 | 12890 |
| D0 | 2 | 0 | 10.254.0.2 | 12777 |
| D1 | 3 | 1 | 10.254.0.2 | 12777 |

因此 D 侧最终命令必须是 `--data-parallel-address 10.254.0.2`，不是 P 侧的 `10.254.0.1`。
