# -*- coding: utf-8 -*-
"""Real-user vllm-ascend launch alignment checks.

These tests intentionally simulate user CLI input instead of calling adapter
helpers directly:

    parse_launch_args -> load_and_merge_configs -> build_start_script

The expected fragments are derived from the official vllm-ascend model
tutorials for the concrete models below. This file is verification-only: it
does not patch production behavior.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.config_loader import load_and_merge_configs  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines.vllm_adapter import build_start_script  # noqa: E402


class OfficialVllmAscendAlignmentTest(unittest.TestCase):
    def _build_script(
        self,
        *,
        architecture: str,
        model_name: str,
        model_config: dict | None = None,
        argv_extra: list[str] | None = None,
        hardware: dict | None = None,
        env: dict | None = None,
    ) -> str:
        with tempfile.TemporaryDirectory() as model_dir:
            config = {"architectures": [architecture]}
            config.update(model_config or {})
            Path(model_dir, "config.json").write_text(
                json.dumps(config, ensure_ascii=False),
                encoding="utf-8",
            )
            device_count = str((hardware or {}).get("count", 8))
            argv = [
                "--engine", "vllm_ascend",
                "--model-name", model_name,
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", device_count,
                "--trust-remote-code",
            ]
            argv.extend(argv_extra or [])
            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, env or {}, clear=True):
                    launch_args = parse_launch_args(argv).to_namespace()
                    merged = load_and_merge_configs(
                        hardware or {"device": "ascend", "count": 8, "details": []},
                        launch_args,
                    )
                    return build_start_script(merged)

    def _assert_official_fragments(self, script: str, fragments: list[str]) -> None:
        missing = [fragment for fragment in fragments if fragment not in script]
        self.assertFalse(
            missing,
            "Official vllm-ascend fragment(s) missing:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\n\nFinal exec line:\n"
            + next(line for line in script.splitlines() if line.startswith("exec ")),
        )

    def test_deepseek_v4_flash_a3_real_user_launch_matches_official_single_node(self):
        """Official DeepSeek-V4-Flash A3 example uses DP4/TP4 and 1M context."""
        script = self._build_script(
            architecture="DeepseekV4ForCausalLM",
            model_name="DeepSeek-V4-Flash-w8a8-mtp",
            hardware={"device": "ascend", "count": 16, "details": [{"name": "910c"}]},
            env={"WINGS_ASCEND_PLATFORM": "A3"},
        )

        self._assert_official_fragments(script, [
            "export OMP_PROC_BIND=false",
            "export OMP_NUM_THREADS=10",
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
            "export ACL_OP_INIT_MODE=1",
            "export ASCEND_A3_ENABLE=1",
            "export USE_MULTI_GROUPS_KV_CACHE=1",
            "export USE_MULTI_BLOCK_POOL=1",
            "export HCCL_BUFFSIZE=1024",
            "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
            "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
            "exec vllm serve",
            "--enable-prefix-caching",
            "--max-model-len 1024000",
            "--max-num-batched-tokens 8192",
            "--gpu-memory-utilization 0.9",
            "--api-server-count 1",
            "--max-num-seqs 16",
            "--data-parallel-size 4",
            "--tensor-parallel-size 4",
            "--enable-expert-parallel",
            "--tokenizer-mode deepseek_v4",
            "--tool-call-parser deepseek_v4",
            "--enable-auto-tool-choice",
            "--reasoning-parser deepseek_v4",
            "--safetensors-load-strategy prefetch",
            "--quantization ascend",
            "--block-size 128",
            "--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'",
            "--async-scheduling",
            "\"enable_npugraph_ex\":true",
            "\"enable_static_kernel\":false",
            "\"multistream_overlap_shared_expert\":false",
            "\"multistream_dsa_preprocess\":false",
            "--speculative-config",
            "\"num_speculative_tokens\": 1",
            "deepseek_mtp",
        ])

    def test_deepseek_v4_pro_a3_real_user_launch_matches_official_two_node_rank0(self):
        """Official DeepSeek-V4-Pro A3 rank0 example uses TP16/DP2 and A3 env."""
        script = self._build_script(
            architecture="DeepseekV4ForCausalLM",
            model_name="DeepSeek-V4-Pro-w4a8-mtp",
            argv_extra=[
                "--distributed",
                "--nnodes", "2",
                "--node-rank", "0",
                "--node-ips", "10.0.0.1,10.0.0.2",
                "--master-ip", "10.0.0.1",
            ],
            hardware={"device": "ascend", "count": 16, "details": [{"name": "910c"}]},
            env={"RANK_IP": "10.0.0.1", "WINGS_ASCEND_PLATFORM": "A3"},
        )

        self._assert_official_fragments(script, [
            "export HCCL_BUFFSIZE=2048",
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
            "export OMP_PROC_BIND=false",
            "export OMP_NUM_THREADS=10",
            "export TASK_QUEUE_ENABLE=1",
            "export USE_MULTI_GROUPS_KV_CACHE=1",
            "export USE_MULTI_BLOCK_POOL=1",
            "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
            "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
            "export HCCL_OP_EXPANSION_MODE=AIV",
            "export HCCL_CONNECT_TIMEOUT=7200",
            "export ASCEND_CONNECT_TIMEOUT=10000",
            "export ASCEND_TRANSFER_TIMEOUT=10000",
            "export VLLM_RPC_TIMEOUT=1800000",
            "export ASCEND_BUFFER_POOL=4:8",
            "export VLLM_USE_V1=1",
            "export VLLM_VERSION=0.18.0",
            "exec vllm serve",
            "--safetensors-load-strategy prefetch",
            "--max-model-len 135000",
            "--max-num-batched-tokens 4096",
            "--gpu-memory-utilization 0.9",
            "--max-num-seqs 16",
            "--data-parallel-size 2",
            "--data-parallel-size-local 1",
            "--data-parallel-start-rank 0",
            "--data-parallel-rpc-port 13399",
            "--tensor-parallel-size 16",
            "--enable-expert-parallel",
            "--quantization ascend",
            "--block-size 128",
            "--async-scheduling",
            "--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'",
            "--tokenizer-mode deepseek_v4",
            "--tool-call-parser deepseek_v4",
            "--enable-auto-tool-choice",
            "--reasoning-parser deepseek_v4",
            "--speculative-config",
            "\"num_speculative_tokens\": 1",
            "deepseek_mtp",
        ])
        # V4-Pro 路径不应注入 ``--api-server-count``（仅 V4-Flash A3 才会）。
        self.assertNotIn(
            "--api-server-count",
            script,
            "V4-Pro should NOT inject --api-server-count (that is V4-Flash A3 only)",
        )

    def test_deepseek_v4_pro_a3_real_user_launch_matches_official_two_node_rank1(self):
        """V4-Pro worker (node_rank=1) must inject --data-parallel-start-rank 1."""
        script = self._build_script(
            architecture="DeepseekV4ForCausalLM",
            model_name="DeepSeek-V4-Pro-w4a8-mtp",
            argv_extra=[
                "--distributed",
                "--nnodes", "2",
                "--node-rank", "1",
                "--node-ips", "10.0.0.1,10.0.0.2",
                "--master-ip", "10.0.0.1",
            ],
            hardware={"device": "ascend", "count": 16, "details": [{"name": "910c"}]},
            env={"RANK_IP": "10.0.0.2", "WINGS_ASCEND_PLATFORM": "A3"},
        )

        self._assert_official_fragments(script, [
            "export HCCL_BUFFSIZE=2048",
            "export VLLM_USE_V1=1",
            "export VLLM_RPC_TIMEOUT=1800000",
            "--data-parallel-size 2",
            "--data-parallel-size-local 1",
            "--data-parallel-start-rank 1",
            "--data-parallel-rpc-port 13399",
            "--tensor-parallel-size 16",
        ])
        self.assertNotIn("--api-server-count", script)

    def test_glm47_w8a8_real_user_launch_matches_official_single_node(self):
        """Official GLM-4.7-W8A8 single-node example uses vllm serve + MTP."""
        script = self._build_script(
            architecture="Glm4MoeForCausalLM",
            model_name="GLM-4.7-W8A8-floatmtp",
            model_config={"quantize": "w8a8"},
            hardware={"device": "ascend", "count": 8, "details": [{"name": "910b"}]},
        )

        self._assert_official_fragments(script, [
            "export HCCL_BUFFSIZE=512",
            "export OMP_PROC_BIND=false",
            "export OMP_NUM_THREADS=1",
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
            "export HCCL_OP_EXPANSION_MODE=AIV",
            "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
            "export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1",
            "exec vllm serve",
            "--data-parallel-size 2",
            "--tensor-parallel-size 8",
            "--enable-expert-parallel",
            "--seed 1024",
            "--max-model-len 133000",
            "--max-num-batched-tokens 8192",
            "--max-num-seqs 16",
            "--async-scheduling",
            "--quantization ascend",
            "--trust-remote-code",
            "--gpu-memory-utilization 0.9",
            "--speculative-config",
            "\"num_speculative_tokens\": 3",
            "\"method\": \"mtp\"",
            "256,512",
            "\"enable_shared_expert_dp\":true",
            "\"fusion_ops_gmmswigluquant\":false",
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
