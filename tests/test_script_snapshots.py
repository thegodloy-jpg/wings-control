# -*- coding: utf-8 -*-
"""E2E 快照测试：验证各场景下生成的 start_command.sh 内容。

运行：
  python -m pytest tests/test_script_snapshots.py -v

首次运行时自动生成 golden 文件到 tests/snapshots/*.sh
后续运行与 golden 文件对比，不一致则失败。

更新 golden 文件：
  UPDATE_SNAPSHOTS=1 python -m pytest tests/test_script_snapshots.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(TESTS_DIR))

from core.wings_entry import build_launcher_plan  # noqa: E402
from snapshot_framework import (  # noqa: E402
    FakeModelIdentifier,
    ScriptGenContext,
    SnapshotTestMixin,
    ascend_hardware,
    make_launch_args,
    make_port_plan,
    nvidia_hardware,
)


class TestScriptSnapshots(SnapshotTestMixin, unittest.TestCase):
    """10 个关键场景的 E2E 脚本生成快照测试。"""

    # ------------------------------------------------------------------
    # SNAP-01  vLLM + NVIDIA + Qwen3（单节点，最小配置）
    # ------------------------------------------------------------------
    def test_snap01_vllm_nvidia_qwen3_single(self):
        hw = nvidia_hardware(device_count=8, memory_gb=80)
        fake_model = FakeModelIdentifier(
            architecture="Qwen3ForCausalLM",
            model_name="Qwen3-32B-Instruct",
            model_path="/models/Qwen3-32B",
        )
        args = make_launch_args(
            engine="vllm",
            model_name="Qwen3-32B-Instruct",
            model_path="/models/Qwen3-32B",
            device_count=8,
            enable_auto_tool_choice=True,
            enable_auto_think_choice=True,
        )
        port_plan = make_port_plan()

        with ScriptGenContext(hw, fake_model):
            plan = build_launcher_plan(args, port_plan)

        self.assert_script_structure(
            plan.command,
            engine_cmd="python3 -m vllm.entrypoints.openai.api_server",
            model_path="/models/Qwen3-32B",
            expected_params=("--tool-call-parser hermes", "--reasoning-parser qwen3"),
        )
        self.assert_snapshot("snap01_vllm_nvidia_qwen3_single", plan.command)

    # ------------------------------------------------------------------
    # SNAP-02  vLLM-distributed + NVIDIA + DeepseekV3（分布式，2 节点）
    # ------------------------------------------------------------------
    def test_snap02_vllm_nvidia_deepseek_distributed(self):
        hw = nvidia_hardware(device_count=8, memory_gb=80)
        fake_model = FakeModelIdentifier(
            architecture="DeepseekV3ForCausalLM",
            model_name="DeepSeek-V3",
            model_path="/models/DeepSeek-V3",
        )
        args = make_launch_args(
            engine="vllm",
            model_name="DeepSeek-V3",
            model_path="/models/DeepSeek-V3",
            device_count=8,
            distributed=True,
            nnodes=2,
            node_rank=0,
            head_node_addr="10.0.0.1",
            node_ips="10.0.0.1,10.0.0.2",
            nodes="10.0.0.1,10.0.0.2",
            ray_head_ip="10.0.0.1",
            enable_auto_tool_choice=True,
            enable_auto_think_choice=True,
        )
        port_plan = make_port_plan()

        with ScriptGenContext(hw, fake_model):
            plan = build_launcher_plan(args, port_plan)

        self.assert_script_structure(
            plan.command,
            engine_cmd="python3 -m vllm.entrypoints.openai.api_server",
            model_path="/models/DeepSeek-V3",
            expected_params=(
                "--tool-call-parser deepseek_v3",
                "--reasoning-parser deepseek_v3",
            ),
        )
        self.assert_snapshot("snap02_vllm_nvidia_deepseek_distributed", plan.command)

    # ------------------------------------------------------------------
    # SNAP-03  vLLM-Ascend + Ascend NPU + KimiK25（单节点）
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # SNAP-04  MindIE + Ascend + Qwen3MoE（单节点，验证 MindIE 参数生成）
    # ------------------------------------------------------------------
    def test_snap04_mindie_ascend_qwen3moe_single(self):
        hw = ascend_hardware(device_count=8)
        fake_model = FakeModelIdentifier(
            architecture="Qwen3MoeForCausalLM",
            model_name="Qwen3-235B-A22B",
            model_path="/models/Qwen3-235B",
        )
        args = make_launch_args(
            engine="mindie",
            model_name="Qwen3-235B-A22B",
            model_path="/models/Qwen3-235B",
            device_count=8,
        )
        port_plan = make_port_plan()

        with ScriptGenContext(hw, fake_model):
            plan = build_launcher_plan(args, port_plan)

        self.assert_script_structure(
            plan.command,
            engine_cmd="mindieservice_daemon",
            model_path="/models/Qwen3-235B",
        )
        self.assert_snapshot("snap04_mindie_ascend_qwen3moe_single", plan.command)

    # ------------------------------------------------------------------
    # SNAP-05  SGLang + NVIDIA + Qwen3（Embedding 任务）
    # ------------------------------------------------------------------
    def test_snap05_sglang_nvidia_embedding(self):
        hw = nvidia_hardware(device_count=4, memory_gb=80)
        fake_model = FakeModelIdentifier(
            architecture="Qwen3ForCausalLM",
            model_type="embedding",
            model_name="Qwen3-Embedding",
            model_path="/models/Qwen3-Embedding",
        )
        args = make_launch_args(
            engine="sglang",
            model_name="Qwen3-Embedding",
            model_path="/models/Qwen3-Embedding",
            model_type="embedding",
            device_count=4,
        )
        port_plan = make_port_plan()

        with ScriptGenContext(hw, fake_model):
            plan = build_launcher_plan(args, port_plan)

        self.assert_script_structure(
            plan.command,
            engine_cmd="python3 -m sglang.launch_server",
            model_path="/models/Qwen3-Embedding",
        )
        self.assert_snapshot("snap05_sglang_nvidia_embedding", plan.command)

    # ------------------------------------------------------------------
    # SNAP-06  vLLM + NVIDIA + Qwen3 + 投机推理开启
    # ------------------------------------------------------------------
    def test_snap06_vllm_nvidia_speculative(self):
        hw = nvidia_hardware(device_count=8, memory_gb=80)
        fake_model = FakeModelIdentifier(
            architecture="Qwen3ForCausalLM",
            model_name="Qwen3-72B",
            model_path="/models/Qwen3-72B",
        )
        args = make_launch_args(
            engine="vllm",
            model_name="Qwen3-72B",
            model_path="/models/Qwen3-72B",
            device_count=8,
            enable_speculative_decode=True,
            speculative_decode_model_path="/models/Qwen3-0.6B",
        )
        port_plan = make_port_plan()

        with ScriptGenContext(hw, fake_model, mock_accel=False,
                              env_overrides={"ENABLE_ACCEL": "false"}):
            plan = build_launcher_plan(args, port_plan)

        self.assert_script_structure(
            plan.command,
            engine_cmd="python3 -m vllm.entrypoints.openai.api_server",
            model_path="/models/Qwen3-72B",
            expected_params=("speculative",),
        )
        self.assert_snapshot("snap06_vllm_nvidia_speculative", plan.command)

    # ------------------------------------------------------------------
    # SNAP-07  vLLM-Ascend + Ascend + GlmMoeDsa（含大量 NPU 优化参数）
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # SNAP-08  vLLM + NVIDIA + env_overrides 注入合法变量
    # ------------------------------------------------------------------
    def test_snap08_vllm_nvidia_env_overrides(self):
        hw = nvidia_hardware(device_count=8, memory_gb=80)
        fake_model = FakeModelIdentifier(
            architecture="LlamaForCausalLM",
            model_name="Llama-3-70B",
            model_path="/models/Llama-3-70B",
        )
        args = make_launch_args(
            engine="vllm",
            model_name="Llama-3-70B",
            model_path="/models/Llama-3-70B",
            device_count=8,
            enable_auto_tool_choice=True,
        )
        port_plan = make_port_plan()

        controlled_override = (
            "# --- wings: user env overrides ---\n"
            "export MY_CUSTOM_VAR='hello'\n"
            "export NCCL_DEBUG='INFO'\n"
            "# --- end env overrides ---\n"
        )

        with ScriptGenContext(hw, fake_model):
            with patch(
                "core.wings_entry._build_env_overrides_preamble",
                return_value=controlled_override,
            ):
                plan = build_launcher_plan(args, port_plan)

        self.assert_script_structure(
            plan.command,
            engine_cmd="python3 -m vllm.entrypoints.openai.api_server",
            model_path="/models/Llama-3-70B",
            expected_params=(
                "MY_CUSTOM_VAR",
                "NCCL_DEBUG",
                "--tool-call-parser llama3_json",
            ),
        )
        self.assert_snapshot("snap08_vllm_nvidia_env_overrides", plan.command)

    # ------------------------------------------------------------------
    # SNAP-09  vLLM-Ascend + Ascend + DeepseekV3 + RAG 加速
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # SNAP-10  vLLM + NVIDIA + 未知模型架构（降级 llm.default）
    # ------------------------------------------------------------------
    def test_snap10_vllm_nvidia_unknown_arch(self):
        hw = nvidia_hardware(device_count=4, memory_gb=40)
        fake_model = FakeModelIdentifier(
            architecture="UnknownArchForCausalLM",
            model_name="custom-model-v1",
            model_path="/models/custom-model",
        )
        args = make_launch_args(
            engine="vllm",
            model_name="custom-model-v1",
            model_path="/models/custom-model",
            device_count=4,
        )
        port_plan = make_port_plan()

        with ScriptGenContext(hw, fake_model):
            plan = build_launcher_plan(args, port_plan)

        self.assert_script_structure(
            plan.command,
            engine_cmd="python3 -m vllm.entrypoints.openai.api_server",
            model_path="/models/custom-model",
            # 未知架构不应注入模型专属 parser
            absent_params=("--tool-call-parser deepseek", "--reasoning-parser deepseek"),
        )
        self.assert_snapshot("snap10_vllm_nvidia_unknown_arch", plan.command)


if __name__ == "__main__":
    unittest.main(verbosity=2)
