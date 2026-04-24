# -*- coding: utf-8 -*-
"""端到端验证：模拟 GLM-4.7-W8A8 经 wings-control 流水线生成的最终 vLLM CLI 命令。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from engines.vllm_adapter import _inject_glm47_w8a8_engine_config, _build_vllm_cmd_parts  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "glm47_w8a8"


def run(label: str, params: dict) -> None:
    print(f"\n========== {label} ==========")
    print(f"[input engine_config] {params['engine_config']}")
    _inject_glm47_w8a8_engine_config(params)
    print(f"[after-inject engine_config keys] {list(params['engine_config'].keys())}")
    cmd = _build_vllm_cmd_parts(params)
    print(f"[final cmd]\n{cmd}\n")


# 场景 1：用户只给基础 CLI（模拟你贴的官方推荐起点之前的状态）
run("Scenario 1: user only sets basic CLI", {
    "engine": "vllm_ascend",
    "model_name": "GLM-4.7-W8A8",
    "model_path": str(FIXTURE),
    "model_type": "auto",
    "engine_config": {
        "host": "0.0.0.0",
        "port": 17000,
        "model": str(FIXTURE),
        "tensor_parallel_size": 8,
        "trust_remote_code": True,
        "max_model_len": 122048,
        "served_model_name": "GLM-4.7-W8A8",
    },
})

# 场景 3：你的真实 case，且 enable_speculative_decode=true（验证去重）
run("Scenario 3: your real case + enable_speculative_decode=true (dedup check)", {
    "engine": "vllm_ascend",
    "model_name": "GLM-4.7-W8A8-floatmtp",
    "model_path": str(FIXTURE),
    "model_type": "auto",
    "enable_speculative_decode": True,
    "engine_config": {
        "host": "56.254.84.39",
        "port": 17000,
        "model": "/usr/local/serving/models/",
        "trust_remote_code": True,
        "seed": 42,
        "max_model_len": 122048,
        "served_model_name": "GLM-4.7-W8A8-floatmtp",
        "tensor_parallel_size": 8,
        "enable_expert_parallel": True,
        "block_size": 128,
        "enable_prefix_caching": True,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 256,
        "enable_chunked_prefill": True,
        "async_scheduling": True,
        "speculative_config": {"method": "glm4_moe_mtp", "num_speculative_tokens": 1},
        "compilation_config": None,  # 显式 None，验证 None == empty 的注入
        "additional_config": {
            "ascend_scheduler_config": {"enabled": True},
            "expert_tensor_parallel_size": 1,
        },
    },
})

# 场景 4：compilation_config / quantization 是 None / 空串，验证仍能注入
run("Scenario 4: compilation_config=None / quantization='' should still be injected", {
    "engine": "vllm_ascend",
    "model_name": "GLM-4.7-W8A8",
    "model_path": str(FIXTURE),
    "model_type": "auto",
    "engine_config": {
        "host": "0.0.0.0",
        "port": 17000,
        "model": str(FIXTURE),
        "tensor_parallel_size": 8,
        "trust_remote_code": True,
        "max_model_len": 122048,
        "served_model_name": "GLM-4.7-W8A8",
        "compilation_config": None,
        "quantization": "",
        "additional_config": {},
    },
})
