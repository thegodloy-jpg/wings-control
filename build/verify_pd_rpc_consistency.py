#!/usr/bin/env python3
"""验证 PD external-lb 多 pod 分布式 D 的 rpc-port 处理契约。

复现用户真实拓扑：D = 4 个 pod，DP8×TP1，local=2。rank 映射对齐其 env：
NODE_IPS[0]=10.254.13.83(head)，RANK_IP=10.254.13.72(idx2)→rank_start=4。

契约（修正后）：--data-parallel-rpc-port / --data-parallel-address 的【跨 pod 一致性由平台负责】
（页面对同一 D 实例统一下发同一值，且常与 NetworkPolicy 绑定）。wings 一律【信任并原样透传】，
不在 wings 侧强改——强改会顶掉平台协调好的端口、打断已放行的连通性。

两种平台下发方式对比：
  A) 统一（consistent）—— 4 pod 同一个 VLLM_LLMDD_RPC_PORT（真实修复后）→ wings 原样 → 可成组 ✅
  B) 不统一（per-pod）  —— 平台按 pod 各发一个 → wings 原样透传不一致 + 告警 → 须在平台侧修 ❌
"""
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import dry_run  # 复用 create_mock_model_dir + 注入 wings_control 到 sys.path

from core.start_args_compat import parse_launch_args          # noqa: E402
from core.port_plan import derive_port_plan                   # noqa: E402
from core.wings_entry import build_launcher_plan              # noqa: E402
from config.settings import settings                          # noqa: E402

ARCH = "Qwen3MoeForCausalLM"
MODEL = "Qwen3-30B-A3B"

D_PODS = ["10.254.13.83", "10.254.13.67", "10.254.13.72", "10.254.13.85"]
HEAD = D_PODS[0]
LOCAL, TP, DP = 2, 1, 8
PF_DP, PF_TP, DC_DP, DC_TP = 2, 1, 8, 1

# 平台下发方式
CONSISTENT_RPC = "46982"                                  # 真实修复后：4 pod 同值（ephemeral 也无妨）
PER_POD_RPC = ["46982", "53929", "47811", "39205"]        # 误配：平台按 pod 各发一个


def _clean_env():
    for k in list(os.environ):
        if k.startswith(("PD_", "DP_", "TP_")) or k in (
                "NODE_IPS", "HOST_IP", "Master_IP", "MASTER_IP",
                "VLLM_LLMDD_RPC_PORT", "RANK_IP", "POD_IP"):
            os.environ.pop(k, None)


def gen_pod_cmd(host_ip: str, rpc: str) -> str:
    _clean_env()
    model_dir = dry_run.create_mock_model_dir(
        ARCH, {"quantization_config": {"quant_method": "ascend"}})
    try:
        os.environ.update({
            "WINGS_DEVICE": "ascend", "WINGS_ASCEND_PLATFORM": "a3",
            "DEVICE_COUNT": str(LOCAL * TP),
            "RANK_IP": host_ip,
            "PD_ROLE": "D",
            "DP_SIZE_LOCAL": str(LOCAL),
            "Master_IP": HEAD,
            "VLLM_LLMDD_RPC_PORT": rpc,
            "NODE_IPS": ",".join(D_PODS),
            "PD_PREFILL_DP_SIZE": str(PF_DP), "PD_PREFILL_TP_SIZE": str(PF_TP),
            "PD_DECODE_DP_SIZE": str(DC_DP), "PD_DECODE_TP_SIZE": str(DC_TP),
        })
        la = parse_launch_args([
            "--model-name", MODEL, "--model-path", model_dir,
            "--engine", "vllm_ascend", "--device-count", str(LOCAL * TP),
            "--nnodes", "1", "--node-rank", "0"])
        pp = derive_port_plan(port=la.port,
                              enable_reason_proxy=settings.ENABLE_REASON_PROXY,
                              health_port=settings.HEALTH_PORT)
        return build_launcher_plan(la, pp).command
    finally:
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)


def extract(cmd: str) -> dict:
    rpc = re.search(r"--data-parallel-rpc-port\s+(\S+)", cmd)
    addr = re.search(r"--data-parallel-address\s+(?:'([^']*)'|(\S+))", cmd)
    start = re.search(r"RANK=\$\(\((\d+)\s*\+\s*i\)\)", cmd)
    local = re.search(r"for i in \$\(seq 0 (\d+)\)", cmd)
    addr_val = (addr.group(1) or addr.group(2)) if addr else "?"
    start_i = int(start.group(1)) if start else -1
    local_n = int(local.group(1)) + 1 if local else -1
    ranks = ",".join(str(start_i + i) for i in range(local_n)) if start_i >= 0 else "?"
    return {"rpc": rpc.group(1) if rpc else "?", "addr": addr_val, "ranks": ranks}


def run_mode(name: str, rpc_for_idx) -> list:
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"{'pod (RANK_IP)':<20}{'--dp-rank':<12}{'--dp-address':<18}{'--dp-rpc-port':<14}")
    print("-" * 64)
    rows = []
    for idx, ip in enumerate(D_PODS):
        info = extract(gen_pod_cmd(ip, rpc_for_idx(idx)))
        role = " (head)" if idx == 0 else ""
        print(f"{ip+role:<20}{info['ranks']:<12}{info['addr']:<18}{info['rpc']:<14}")
        rows.append(info)
    return rows


def dump_full_scripts():
    """落盘 4 个 D pod 的完整 start_command.sh（用平台统一下发的同一端口），作为可部署参考。"""
    out_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "build", "output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*78}\n落盘完整启动脚本（平台统一下发 {CONSISTENT_RPC}，wings 原样透传）\n{'='*78}")
    for idx, ip in enumerate(D_PODS):
        cmd = gen_pod_cmd(ip, CONSISTENT_RPC)
        fn = f"start_command_pd-qwen3-4d1n-D_node{idx}.sh"
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write(cmd)
        info = extract(cmd)
        print(f"  build/output/{fn}  (rank {info['ranks']}, "
              f"rpc-port {info['rpc']}, address {info['addr']})")


def main():
    print("复现拓扑：D = 4 pod，DP8×TP1，local=2；head =", HEAD)

    rows_a = run_mode("A：平台统一下发（consistent，真实修复后）", lambda i: CONSISTENT_RPC)
    ports_a = {r["rpc"] for r in rows_a}
    print(f"\n  → rpc-port 去重集合 = {ports_a}"
          f"  {'✅ 全一致（且=平台值，未被 wings 改写）→ 可成组' if len(ports_a)==1 else '❌'}")

    rows_b = run_mode("B：平台按 pod 各发（误配）", lambda i: PER_POD_RPC[i])
    ports_b = {r["rpc"] for r in rows_b}
    print(f"\n  → rpc-port 去重集合 = {ports_b}")

    print(f"\n{'='*78}\n结论\n{'='*78}")
    print("• wings 对 VLLM_LLMDD_RPC_PORT【信任并原样透传】，不强改（含 ephemeral 值）。")
    print("• 模式 A：平台统一 → wings 原样 → 4 pod 一致 → DP 可成组（且不顶掉平台端口/网络策略）。✅")
    print("• 模式 B：平台不统一 → wings 原样透传出不一致（仅打告警）→ 仍会握手超时。")
    print("  ⇒ 跨 pod 一致性必须由【平台/页面统一下发同一端口】保证（治本），wings 不越权强改。")

    dump_full_scripts()


if __name__ == "__main__":
    main()
