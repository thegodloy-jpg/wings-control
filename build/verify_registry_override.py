#!/usr/bin/env python3
"""验证：PD external-lb 下，平台灌的 `--xxx` flag 会不会顶掉注册表(pd_config.json)调优值。
A=精简 argv（只 model/engine/device-count）；B=平台全套 flag。同一模型同一 PD env，比生成命令。
"""
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wings_control"))
sys.path.insert(0, ROOT)

from core.start_args_compat import parse_launch_args      # noqa: E402
from core.port_plan import derive_port_plan               # noqa: E402
from core.wings_entry import build_launcher_plan          # noqa: E402
from core.config_loader import _detect_explicit_cli_keys  # noqa: E402
from config.settings import settings                      # noqa: E402
from dry_run import create_mock_model_dir                 # noqa: E402

ARCH = "Qwen3MoeForCausalLM"
MODEL = "Qwen3-30B-A3B"
PD_ENV = {
    "PD_ROLE": "D", "DP_SIZE_LOCAL": "2",
    "Master_IP": "9.0.1.1", "NODE_IPS": "9.0.1.1,9.0.1.2", "RANK_IP": "9.0.1.1",
    "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "2",
    "PD_DECODE_DP_SIZE": "4", "PD_DECODE_TP_SIZE": "1",
}
# 平台真机模板里灌的一串(取自 jzow306/xlka343 真机 wings_start.sh)
PLATFORM_FLAGS = [
    "--gpu-memory-utilization", "0.8",
    "--max-num-seqs", "256",
    "--max-num-batched-tokens", "4096",
    "--block-size", "16",
    "--enable-chunked-prefill",
    "--enable-prefix-caching",
    "--seed", "42",
]

_PLATFORM_ENV_KEYS = ["GPU_MEMORY_UTILIZATION", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS",
                      "BLOCK_SIZE", "ENABLE_CHUNKED_PREFILL", "ENABLE_PREFIX_CACHING", "SEED"]


def _clear():
    for k in list(os.environ):
        if k.startswith(("PD_", "DP_", "TP_")) or k in (
                "NODE_IPS", "HOST_IP", "Master_IP", "MASTER_IP", "RANK_IP", "POD_IP",
                "DISTRIBUTED", "VLLM_LLMDD_RPC_PORT", *_PLATFORM_ENV_KEYS):
            os.environ.pop(k, None)


def gen(extra_args):
    _clear()
    model_dir = create_mock_model_dir(ARCH, {"quantization_config": {"quant_method": "ascend"}})
    os.environ.update({"WINGS_DEVICE": "ascend", "WINGS_ASCEND_PLATFORM": "a3",
                       "ENGINE": "vllm_ascend", **PD_ENV})
    base = ["--model-name", MODEL, "--model-path", model_dir, "--engine", "vllm_ascend",
            "--device-count", "2", "--nnodes", "1", "--node-rank", "0"]
    argv = base + extra_args
    sys.argv = ["wings_control", *argv]          # explicit 检测读 sys.argv
    explicit = _detect_explicit_cli_keys("vllm_ascend")
    la = parse_launch_args(argv)
    pp = derive_port_plan(port=la.port, enable_reason_proxy=settings.ENABLE_REASON_PROXY,
                          health_port=settings.HEALTH_PORT)
    cmd = build_launcher_plan(la, pp).command
    shutil.rmtree(model_dir, ignore_errors=True)
    return cmd, explicit


def fval(cmd, flag):
    m = re.search(rf"{re.escape(flag)}\s+('[^']*'|\S+)", cmd)
    if m:
        return m.group(1)
    return "✔(flag在)" if flag in cmd else "—(无)"


FLAGS = ["--max-num-batched-tokens", "--max-num-seqs", "--gpu-memory-utilization",
         "--block-size", "--enable-chunked-prefill", "--enable-prefix-caching",
         "--compilation-config", "--async-scheduling"]

cmd_a, exp_a = gen([])                       # A：精简
cmd_b, exp_b = gen(PLATFORM_FLAGS)           # B：平台全套

print(f"模型: {ARCH} (D 角色, dp4×tp1)   注册表期望: batched=120 seqs=60 gpu=0.88 async FULL_DECODE_ONLY\n")
print(f"A 精简 argv 显式键: {sorted(exp_a)}")
print(f"B 平台 argv 显式键: {sorted(exp_b)}\n")
print(f"{'flag':<28} {'A=精简(注册表)':<22} {'B=平台全套':<22} 结论")
print("-" * 92)
for fl in FLAGS:
    a, b = fval(cmd_a, fl), fval(cmd_b, fl)
    verdict = "一致" if a == b else "[!] 被平台顶掉" if a not in ("—(无)",) else "平台新增(注册表本无)"
    print(f"{fl:<28} {a:<22} {b:<22} {verdict}")
