# DeepSeek-V3.1 MindIE 2x8 long-context generated example

- node IPs: 10.254.13.90, 10.254.13.91
- input_length: 10000
- output_length: 2048
- total sequence length: 12048
- globalWorldSize: 16
- local worldSize: 8
- dp/tp/sp/cp: 1 / 8 / 8 / 2
- moe_tp/moe_ep: 1 / 16
- enable_ep_moe: True

Generated start scripts:

- start_command_node0.sh
- start_command_node1.sh
- config.json
- REDEPLOY_CHECKLIST.md
- 910b_hccl_env_override.env

Use `REDEPLOY_CHECKLIST.md` to verify the live Pod is not still running an old
start script or ConfigMap. In this long-context scenario, the final MindIE
config must keep `dp/tp/sp/cp = 1/8/8/2` and should only print
`NPU_MEMORY_FRACTION=0.96`.

Use `910b_hccl_env_override.env` as a template when 910B dual-node HCCL group
creation fails and the rank table needs real hccn/RDMA device IPs.
