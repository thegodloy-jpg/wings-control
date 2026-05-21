"""End-to-end smoke test for build_launcher_plan.

Simulates two realistic user invocations (single-node 1×8, dual-node 2×8) on a
910B3 cluster with vllm_ascend, drives the full launcher pipeline, and dumps the
generated start_command.sh so we can eyeball it.

Run:
    cd d:/project/inference/wings/wings-control/wings_control
    python _smoke_launcher.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# Step 1: prepare env BEFORE importing wings_control modules.                 #
# --------------------------------------------------------------------------- #
_TMP = Path(tempfile.gettempdir()) / "wings_smoke"
_TMP.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("WINGS_DEVICE", "ascend")
os.environ.setdefault("WINGS_DEVICE_COUNT", "8")
os.environ.setdefault("WINGS_DEVICE_NAME", "Ascend910B3")
# Plan-build writes the selected-engine marker; redirect off /var/log on Windows.
os.environ.setdefault("BACKEND_PID_FILE", str(_TMP / "wings.txt"))
# Hardware JSON path: point at a non-existent file so detect_hardware falls back
# to env-var strategy. (default /shared-volume/... does not exist on Windows.)
os.environ.setdefault("WINGS_HARDWARE_FILE", str(_TMP / "nonexistent_hw.json"))
# Disable accel preamble to keep generated script focused on engine launch.
os.environ.setdefault("ENABLE_ACCEL", "false")

# Make the package importable: imports inside use `from core.x import ...`
# style, so cwd / sys.path must contain wings_control/.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))


def _build_launch_args(argv: list[str]):
    from core.start_args_compat import parse_launch_args
    return parse_launch_args(argv)


def _build_port_plan(port: int):
    from core.port_plan import derive_port_plan
    return derive_port_plan(port=port, enable_reason_proxy=True)


def _run(label: str, argv: list[str], port: int = 18000) -> None:
    print("=" * 80)
    print(f"[smoke] {label}")
    print("=" * 80)
    print(f"[smoke] argv = {argv}")

    launch_args = _build_launch_args(argv)
    port_plan = _build_port_plan(port)

    print(f"[smoke] LaunchArgs.engine          = {launch_args.engine}")
    print(f"[smoke] LaunchArgs.model_name      = {launch_args.model_name}")
    print(f"[smoke] LaunchArgs.device_count    = {launch_args.device_count}")
    print(f"[smoke] LaunchArgs.distributed     = {launch_args.distributed}")
    print(f"[smoke] LaunchArgs.nnodes          = {launch_args.nnodes}")
    print(f"[smoke] LaunchArgs.node_rank       = {launch_args.node_rank}")
    print(f"[smoke] LaunchArgs.head_node_addr  = {launch_args.head_node_addr}")
    print(f"[smoke] PortPlan                   = {port_plan}")

    from core.wings_entry import build_launcher_plan
    plan = build_launcher_plan(launch_args, port_plan)

    print(f"[smoke] hardware_env               = {plan.hardware_env}")
    print(f"[smoke] merged_params.engine       = {plan.merged_params.get('engine')}")
    print(f"[smoke] merged_params.host         = {plan.merged_params.get('host')}")
    print(f"[smoke] merged_params.port         = {plan.merged_params.get('port')}")
    print(f"[smoke] merged_params.distributed  = {plan.merged_params.get('distributed')}")
    print(f"[smoke] merged_params.nnodes       = {plan.merged_params.get('nnodes')}")
    print(f"[smoke] merged_params.node_rank    = {plan.merged_params.get('node_rank')}")
    print(f"[smoke] merged_params.master_ip    = {plan.merged_params.get('master_ip')}")
    print(f"[smoke] script length              = {len(plan.command)} bytes")
    print(f"[smoke] script lines               = {plan.command.count(chr(10))}")

    out_path = _TMP / f"start_command.{label}.sh"
    out_path.write_text(plan.command, encoding="utf-8")
    print(f"[smoke] wrote {out_path}")

    # Sanity markers: must be a real bash script driving vLLM/Ascend
    expected_markers = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "vllm",
    ]
    missing = [m for m in expected_markers if m not in plan.command]
    if missing:
        print(f"[smoke] WARN missing expected markers: {missing}")
    else:
        print("[smoke] OK: shebang + set -euo pipefail + vllm command present")
    print()


def main() -> None:
    # ----------------------------------------------------------------------- #
    # Case A: single node, 8x910B3, dense LLM, vllm_ascend                    #
    # ----------------------------------------------------------------------- #
    single_node_argv = [
        "--model-name", "Qwen2.5-7B-Instruct",
        "--model-path", "/weights/Qwen2.5-7B-Instruct",
        "--engine", "vllm_ascend",
        "--device-count", "8",
        "--input-length", "4096",
        "--output-length", "1024",
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "32",
        "--dtype", "auto",
        "--port", "18000",
        "--host", "0.0.0.0",
        "--trust-remote-code", "true",
        "--enable-chunked-prefill", "true",
    ]
    _run("single_node_qwen25_7b_ascend", single_node_argv, port=18000)

    # ----------------------------------------------------------------------- #
    # Case B: dual node, 2x8 910B3, large MoE, vllm_ascend, ray backend       #
    # head rank (node_rank=0) — this is what most users care about            #
    # ----------------------------------------------------------------------- #
    dual_node_head_argv = [
        "--model-name", "DeepSeek-V3",
        "--model-path", "/weights/DeepSeek-V3",
        "--engine", "vllm_ascend",
        "--device-count", "8",
        "--input-length", "8192",
        "--output-length", "4096",
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "64",
        "--dtype", "auto",
        "--port", "18000",
        "--host", "0.0.0.0",
        "--trust-remote-code", "true",
        "--enable-chunked-prefill", "true",
        "--enable-expert-parallel", "true",
        # Distributed
        "--distributed", "true",
        "--nnodes", "2",
        "--node-rank", "0",
        "--head-node-addr", "10.0.0.1",
        "--node-ips", "10.0.0.1,10.0.0.2",
        "--master-ip", "10.0.0.1",
        "--ray-head-ip", "10.0.0.1",
        "--distributed-executor-backend", "ray",
    ]
    # Pretend this pod is the head node so engine listener gets bound.
    os.environ["POD_IP"] = "10.0.0.1"
    os.environ["RANK_IP"] = "10.0.0.1"
    os.environ["MASTER_IP"] = "10.0.0.1"
    _run("dual_node_rank0_deepseek_v3_ascend", dual_node_head_argv, port=18000)

    # ----------------------------------------------------------------------- #
    # Case C: dual node worker (node_rank=1) — must NOT bind engine port      #
    # ----------------------------------------------------------------------- #
    dual_node_worker_argv = list(dual_node_head_argv)
    idx = dual_node_worker_argv.index("--node-rank")
    dual_node_worker_argv[idx + 1] = "1"
    os.environ["POD_IP"] = "10.0.0.2"
    os.environ["RANK_IP"] = "10.0.0.2"
    os.environ["MASTER_IP"] = "10.0.0.1"
    _run("dual_node_rank1_deepseek_v3_ascend", dual_node_worker_argv, port=18000)


if __name__ == "__main__":
    main()
