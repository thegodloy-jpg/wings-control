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
        """Official DeepSeek-V4-Flash A3 example uses DP4/TP4 and 1M context.

        max_model_len 完全由用户控制（不再按 A3 平台注入长上下文默认），因此官方
        1M 上下文需用户显式给出：这里通过 ``--input-length`` + ``--output-length``
        合成 1024000（512000 + 512000），由 _set_sequence_length 计算为 max_model_len。
        """
        script = self._build_script(
            architecture="DeepseekV4ForCausalLM",
            model_name="DeepSeek-V4-Flash-w8a8-mtp",
            hardware={"device": "ascend", "count": 16, "details": [{"name": "910c"}]},
            env={"WINGS_ASCEND_PLATFORM": "A3"},
            # 官方 V4-Flash 启动命令带 --reasoning-parser，需显式开启 reasoning 开关；
            # 1M 上下文由用户显式传入（input+output=1024000），不再依赖平台默认。
            argv_extra=[
                "--enable-auto-think-choice",
                "--input-length", "512000",
                "--output-length", "512000",
            ],
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

    def test_deepseek_v4_pro_a3_dp_env_final_values_match_reference(self):
        """V4-Pro DP env 的"最终值"必须与官方参考脚本一致。

        ``_build_ascend_dp_env_commands`` 在 V4-Pro 模型 env 之后再次 export
        HCCL_BUFFSIZE / OMP_NUM_THREADS / HCCL_CONNECT_TIMEOUT；本测试锁定
        这些变量"最后一次 export" 必须命中 V4-Pro 专属默认 2048 / 10 / 7200，
        而不是通用 DeepSeek DP 的 1024 / 100 / 1800。
        """
        import re as _re
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
        finals: dict[str, str] = {}
        for line in script.splitlines():
            m = _re.match(r'^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.+)$', line.strip())
            if m:
                finals[m.group(1)] = m.group(2)
        expected_finals = {
            "HCCL_BUFFSIZE": "2048",
            "OMP_NUM_THREADS": "10",
            "HCCL_CONNECT_TIMEOUT": "7200",
        }
        for var, want in expected_finals.items():
            got = finals.get(var, "<MISSING>")
            self.assertEqual(
                got, want,
                f"V4-Pro final {var} must be {want} (got {got}); "
                f"likely a later export in dp_deployment env overrode the model-specific value",
            )

    def test_deepseek_v4_pro_a3_speculative_decode_off_by_default(self):
        """未传 --enable-speculative-decode 时 V4-Pro 不应注入 --speculative-config。"""
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
        self.assertNotIn(
            "--speculative-config",
            script,
            "V4-Pro should NOT default-enable speculative decoding; "
            "it must be opted in via --enable-speculative-decode",
        )
        self.assertNotIn("deepseek_mtp", script)

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
