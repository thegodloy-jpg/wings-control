# -*- coding: utf-8 -*-
"""Real-world launcher dry-run.

构造两组贴近平台真实下发的 LaunchArgs：

  1) 单机 910B3 64GB × 8 卡：Qwen3-32B-Instruct，vllm_ascend
  2) 双机 910B3 64GB × 2 × 8 卡：GLM-5.1 (GlmMoeDsaForCausalLM)，vllm_ascend, dp_deployment

调用 build_launcher_plan 端到端生成 start_command.sh，验证整链路是否能
跑通并产出完整的启动脚本。

运行：
    cd wings-control
    python tests/dryrun_real_user_launch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(ROOT / "wings_control" / "engines"))
sys.path.insert(0, str(TESTS_DIR))

from core.wings_entry import build_launcher_plan  # noqa: E402
from snapshot_framework import (  # noqa: E402
    FakeModelIdentifier,
    ScriptGenContext,
    ascend_hardware,
    make_launch_args,
    make_port_plan,
)


SEP = "=" * 88

OUTPUT_PATH = TESTS_DIR / "dryrun_real_user_launch_output.txt"
_OUT_FP = None


def _emit(line: str = "") -> None:
    if _OUT_FP is not None:
        _OUT_FP.write(line + "\n")
    try:
        print(line)
    except UnicodeEncodeError:
        # Windows console (gbk) can't encode some chars; fall back to safe ascii.
        print(line.encode("ascii", errors="replace").decode("ascii"))


def _print_command(label: str, command: str) -> None:
    _emit("\n" + SEP)
    _emit(f" {label}")
    _emit(SEP)
    _emit(command)
    _emit(SEP + "\n")


def _print_meta(label: str, plan) -> None:
    merged = plan.merged_params
    keys = (
        "engine", "model_name", "model_path", "device_count",
        "tensor_parallel_size", "pipeline_parallel_size",
        "data_parallel_size", "max_model_len",
        "distributed", "nnodes", "node_rank", "head_node_addr",
        "distributed_executor_backend", "host", "port",
        "node_ips", "master_ip", "ray_head_ip",
        "enable_expert_parallel", "enable_sparse",
        "enable_speculative_decode", "enable_chunked_prefill",
        "enable_prefix_caching",
    )
    _emit(f"\n[{label}] merged_params key fields:")
    for k in keys:
        if k in merged:
            _emit(f"  {k:35s} = {merged[k]!r}")
    _emit(f"  hardware.device                     = {plan.hardware_env.get('device')!r}")
    _emit(f"  hardware.count                      = {plan.hardware_env.get('count')!r}")
    _emit(f"  command_bytes                       = {len(plan.command)}")


# ---------------------------------------------------------------------------
# 场景 1: 单机 910B3 64GB × 8 卡 + Qwen3-32B-Instruct + vllm_ascend
# ---------------------------------------------------------------------------
def case_single_node_qwen3():
    label = "CASE-A  单机 910B3 × 8卡 / Qwen3-32B / vllm_ascend"

    hw = ascend_hardware(device_count=8)
    fake_model = FakeModelIdentifier(
        architecture="Qwen3ForCausalLM",
        model_name="Qwen3-32B-Instruct",
        model_path="/usr/local/serving/models/Qwen3-32B-Instruct",
    )
    args = make_launch_args(
        host="0.0.0.0",
        port=18000,
        engine="vllm_ascend",
        model_name="Qwen3-32B-Instruct",
        model_path="/usr/local/serving/models/Qwen3-32B-Instruct",
        model_type="llm",
        input_length=8192,
        output_length=4096,
        device_count=8,
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.9,
        block_size=128,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enable_auto_tool_choice=True,
        distributed=False,
        nnodes=1,
        node_rank=0,
        head_node_addr="",
        distributed_executor_backend="ray",
    )
    port_plan = make_port_plan()

    with ScriptGenContext(hw, fake_model):
        plan = build_launcher_plan(args, port_plan)

    _print_command(label, plan.command)
    _print_meta(label, plan)
    return plan


# ---------------------------------------------------------------------------
# 场景 2: 双机 910B3 64GB × 2 × 8 卡 + GLM-5.1 MoE + vllm_ascend / dp_deployment
# ---------------------------------------------------------------------------
def case_dual_node_glm51(node_rank: int):
    label = (
        f"CASE-B[rank={node_rank}]  双机 910B3 × 2×8卡 / GLM-5.1 GlmMoeDsa / "
        f"vllm_ascend / dp_deployment"
    )

    hw = ascend_hardware(device_count=8)
    fake_model = FakeModelIdentifier(
        architecture="GlmMoeDsaForCausalLM",
        model_name="GLM-5.1",
        model_path="/usr/local/serving/models/GLM-5.1",
    )

    head_ip = "10.254.124.178"
    worker_ip = "10.254.13.111"
    node_ips_csv = f"{head_ip},{worker_ip}"

    args = make_launch_args(
        host="0.0.0.0",
        port=18000,
        engine="vllm_ascend",
        model_name="GLM-5.1",
        model_path="/usr/local/serving/models/GLM-5.1",
        model_type="llm",
        input_length=16384,
        output_length=8192,
        device_count=8,
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.9,
        block_size=128,
        max_num_seqs=128,
        max_num_batched_tokens=16384,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enable_expert_parallel=True,
        distributed=True,
        nnodes=2,
        node_rank=node_rank,
        head_node_addr=head_ip,
        master_ip=head_ip,
        ray_head_ip=head_ip,
        node_ips=node_ips_csv,
        nodes=node_ips_csv,
        distributed_executor_backend="dp_deployment",
    )
    port_plan = make_port_plan()

    rank_ip = head_ip if node_rank == 0 else worker_ip
    extra_env = {
        "RANK_IP": rank_ip,
        "POD_IP": rank_ip,
        "MASTER_IP": head_ip,
        "NODE_IPS": node_ips_csv,
        "DISTRIBUTED": "true",
        "NNODES": "2",
        "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
    }

    with ScriptGenContext(hw, fake_model, env_overrides=extra_env):
        plan = build_launcher_plan(args, port_plan)

    _print_command(label, plan.command)
    _print_meta(label, plan)
    return plan


# ---------------------------------------------------------------------------
# 简单校验：每个生成的脚本都要满足的最低不变式
# ---------------------------------------------------------------------------
def _assert_invariants(label: str, command: str, *, expect_distributed: bool):
    errors = []
    if not (command.startswith("#!/bin/bash") or command.startswith("#!/usr/bin/env bash")):
        errors.append("缺少 shebang")
    if "set -euo pipefail" not in command:
        errors.append("缺少 strict mode")
    if "exec " not in command:
        errors.append("没有最终 exec 行")
    if expect_distributed:
        if "10.254.124.178" not in command:
            errors.append("分布式场景缺少 head_node_addr=10.254.124.178")
        if ("dp_deployment" not in command) and ("--data-parallel-size" not in command):
            errors.append("分布式场景缺少 dp_deployment / data-parallel-size 痕迹")
    _emit(f"\n[{label}] invariant check: " + ("PASS" if not errors else "FAIL"))
    for e in errors:
        _emit("    - " + e)
    return not errors


def main() -> int:
    global _OUT_FP
    _OUT_FP = open(OUTPUT_PATH, "w", encoding="utf-8")
    try:
        ok_all = True

        plan_a = case_single_node_qwen3()
        ok_all &= _assert_invariants(
            "CASE-A  Qwen3 single-node", plan_a.command, expect_distributed=False,
        )

        plan_b0 = case_dual_node_glm51(node_rank=0)
        ok_all &= _assert_invariants(
            "CASE-B  rank0", plan_b0.command, expect_distributed=True,
        )

        plan_b1 = case_dual_node_glm51(node_rank=1)
        ok_all &= _assert_invariants(
            "CASE-B  rank1", plan_b1.command, expect_distributed=True,
        )

        _emit("\n" + SEP)
        _emit(f" Overall result: {'ALL PASS' if ok_all else 'HAS FAILURES'}")
        _emit(f" Output written to: {OUTPUT_PATH}")
        _emit(SEP)
        return 0 if ok_all else 1
    finally:
        if _OUT_FP is not None:
            _OUT_FP.close()


if __name__ == "__main__":
    sys.exit(main())
