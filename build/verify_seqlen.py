#!/usr/bin/env python3
"""只传 --input-length/--output-length/--model-name/--model-path（+PD契约+engine+device-count）
是否影响注册表字段。GLM5 D(注册表 max_model_len=200000) vs qwen3 D(注册表无 max_model_len)。"""
import os, re, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wings_control"))
sys.path.insert(0, ROOT)
from core.start_args_compat import parse_launch_args            # noqa: E402
from core.port_plan import derive_port_plan                     # noqa: E402
from core.wings_entry import build_launcher_plan                # noqa: E402
from config.settings import settings                            # noqa: E402
from dry_run import create_mock_model_dir                       # noqa: E402

CASES = {
    "qwen3 D (注册表无 max_model_len)": dict(
        arch="Qwen3MoeForCausalLM", name="Qwen3-30B-A3B",
        pd=dict(PD_DECODE_DP_SIZE="4", PD_DECODE_TP_SIZE="1", DP_SIZE_LOCAL="2",
                NODE_IPS="9.0.1.1,9.0.1.2"),
        reg="batched=120 seqs=60 gpu=0.88  max_model_len=未声明"),
    "glm5 D (注册表 max_model_len=200000)": dict(
        arch="GlmMoeDsaForCausalLM", name="glm-5.1-chat",
        pd=dict(PD_DECODE_DP_SIZE="16", PD_DECODE_TP_SIZE="4", DP_SIZE_LOCAL="4",
                NODE_IPS="9.0.2.1,9.0.2.2,9.0.2.3,9.0.2.4"),
        reg="batched=32 seqs=8 gpu=0.92  max_model_len=200000"),
}


def _clear():
    for k in list(os.environ):
        if k.startswith(("PD_", "DP_", "TP_")) or k in (
                "NODE_IPS", "HOST_IP", "Master_IP", "MASTER_IP", "RANK_IP", "POD_IP",
                "DISTRIBUTED", "VLLM_LLMDD_RPC_PORT", "INPUT_LENGTH", "OUTPUT_LENGTH"):
            os.environ.pop(k, None)


def fval(cmd, flag):
    m = re.search(rf"{re.escape(flag)}\s+('[^']*'|\S+)", cmd)
    return m.group(1) if m else "—(无)"


for label, c in CASES.items():
    _clear()
    md = create_mock_model_dir(c["arch"], {"quantization_config": {"quant_method": "ascend"}})
    nodes = c["pd"]["NODE_IPS"].split(",")
    os.environ.update({
        "WINGS_DEVICE": "ascend", "WINGS_ASCEND_PLATFORM": "a3", "ENGINE": "vllm_ascend",
        "PD_ROLE": "D", "Master_IP": nodes[0], "RANK_IP": nodes[0],
        "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "2", **c["pd"],
    })
    # 只传这四个（+ engine/device-count 触发 PD 必需）
    argv = ["--model-name", c["name"], "--model-path", md, "--engine", "vllm_ascend",
            "--device-count", "4"]   # 不传 input/output-length
    sys.argv = ["wings_control", *argv]
    la = parse_launch_args(argv)
    pp = derive_port_plan(port=la.port, enable_reason_proxy=settings.ENABLE_REASON_PROXY,
                          health_port=settings.HEALTH_PORT)
    cmd = build_launcher_plan(la, pp).command
    shutil.rmtree(md, ignore_errors=True)
    print(f"\n## {label}")
    print(f"   注册表期望: {c['reg']}")
    print(f"   生成命令实际: max-model-len={fval(cmd,'--max-model-len')}  "
          f"batched={fval(cmd,'--max-num-batched-tokens')}  "
          f"seqs={fval(cmd,'--max-num-seqs')}  gpu={fval(cmd,'--gpu-memory-utilization')}")
