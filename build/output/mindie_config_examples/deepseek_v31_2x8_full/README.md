# DeepSeek-V3.1 dual-node 2x8 full generated view

- Scenario: DeepSeek-V3.1, 2 nodes x 8 NPU, global world size 16.
- config.json is identical for node0 and node1.
- DeepSeek-V3.1 is treated as MoE, so moe_tp/moe_ep are included even when enable_expert_parallel is not used.
- start_command_env_node*.sh contains every explicit env/source line extracted from start_command_node*.sh.
- Rank table input path before startup: /workspace/rank_table_all.json.
- Rank table path used by MindIE engine: /shared-volume/hccl_ranktable.json.

Files:

- config.json
- start_command_node0.sh
- start_command_node1.sh
- start_command_env_node0.sh
- start_command_env_node1.sh
