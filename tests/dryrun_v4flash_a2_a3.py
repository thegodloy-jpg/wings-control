# -*- coding: utf-8 -*-
"""DeepSeek-V4-Flash A2 / A3 单机 dry-run.

目的：触发 wings 真实下发链路（build_launcher_plan → build_start_script），
分别在模拟 A2 (Ascend910B, 8 卡) 和 A3 (Ascend910C, 16 卡) 硬件下生成
完整启动脚本，便于与 vllm-ascend v0.18.0 官方 DeepSeek-V4-Flash.md 单机模板逐字段对比。

运行：
    cd wings-control
    python tests/dryrun_v4flash_a2_a3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(ROOT / "wings_control" / "engines"))
sys.path.insert(0, str(TESTS_DIR))

from core.wings_entry import build_launcher_plan  # noqa: E402
from snapshot_framework import (  # noqa: E402
    FakeModelIdentifier,
    ScriptGenContext,
    make_launch_args,
    make_port_plan,
)


SEP = "=" * 88
OUTPUT_PATH = TESTS_DIR / "dryrun_v4flash_a2_a3_output.txt"
_OUT_FP = None


def _emit(line: str = "") -> None:
    if _OUT_FP is not None:
        _OUT_FP.write(line + "\n")
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"))


def _print_command(label: str, command: str) -> None:
    _emit("\n" + SEP)
    _emit(f" {label}")
    _emit(SEP)
    _emit(command)
    _emit(SEP + "\n")


def ascend_a2_hardware(device_count=8):
    return {
        "device": "ascend",
        "count": device_count,
        "details": [
            {
                "device_id": i,
                "name": "Ascend910B3",
                "total_memory": 64.0,
                "free_memory": 64.0,
                "used_memory": 0.0,
                "vendor": "Ascend",
            }
            for i in range(device_count)
        ],
        "units": "GB",
    }


def ascend_a3_hardware(device_count=16):
    return {
        "device": "ascend",
        "count": device_count,
        "details": [
            {
                "device_id": i,
                "name": "Ascend910C",
                "total_memory": 128.0,
                "free_memory": 128.0,
                "used_memory": 0.0,
                "vendor": "Ascend",
            }
            for i in range(device_count)
        ],
        "units": "GB",
    }


def _v4flash_args(device_count: int):
    return make_launch_args(
        host="0.0.0.0",
        port=18000,
        engine="vllm_ascend",
        model_name="DeepSeek-V4-Flash",
        model_path="/usr/local/serving/models/DeepSeek-V4-Flash-w8a8-mtp",
        model_type="llm",
        input_length=8192,
        output_length=4096,
        device_count=device_count,
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.9,
        block_size=128,
        max_num_seqs=16,
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


def case_v4flash_a2():
    label = "CASE-V4Flash-A2  单机 Ascend910B3 × 8卡 / DeepSeek-V4-Flash / vllm_ascend"
    hw = ascend_a2_hardware(device_count=8)
    fake_model = FakeModelIdentifier(
        architecture="DeepseekV4ForCausalLM",
        model_name="DeepSeek-V4-Flash",
        model_path="/usr/local/serving/models/DeepSeek-V4-Flash-w8a8-mtp",
        model_type="llm",
        model_quantize="w8a8",
    )
    args = _v4flash_args(device_count=8)
    port_plan = make_port_plan()

    with ScriptGenContext(hw, fake_model):
        plan = build_launcher_plan(args, port_plan)

    _print_command(label, plan.command)
    return plan


def case_v4flash_a3():
    label = "CASE-V4Flash-A3  单机 Ascend910C × 16卡 / DeepSeek-V4-Flash / vllm_ascend"
    hw = ascend_a3_hardware(device_count=16)
    fake_model = FakeModelIdentifier(
        architecture="DeepseekV4ForCausalLM",
        model_name="DeepSeek-V4-Flash",
        model_path="/usr/local/serving/models/DeepSeek-V4-Flash-w8a8-mtp",
        model_type="llm",
        model_quantize="w8a8",
    )
    args = _v4flash_args(device_count=16)
    port_plan = make_port_plan()

    with ScriptGenContext(hw, fake_model):
        plan = build_launcher_plan(args, port_plan)

    _print_command(label, plan.command)
    return plan


def main() -> int:
    global _OUT_FP
    _OUT_FP = open(OUTPUT_PATH, "w", encoding="utf-8")
    try:
        case_v4flash_a2()
        case_v4flash_a3()
        _emit("\n" + SEP)
        _emit(f" Output written to: {OUTPUT_PATH}")
        _emit(SEP)
        return 0
    finally:
        if _OUT_FP is not None:
            _OUT_FP.close()


if __name__ == "__main__":
    sys.exit(main())
