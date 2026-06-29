#!/usr/bin/env python3
"""Wings-infer Dry-Run 脚本：通过官方入口生成 start_command.sh。

使用方法：
  # GLM-5.1 + 910B(A2) 双机 dp_deployment
  python dry_run.py --scenario glm51-910b-dual

  # DeepSeek-V4-Flash + 910C(A3) 单机16卡
  python dry_run.py --scenario v4flash-a3-16

  # 自定义场景
  python dry_run.py --model-name "MyModel" --arch "GlmMoeDsaForCausalLM" \
      --engine vllm_ascend --device-count 8 --nnodes 1

输出目录: build/output/
"""
import argparse
import json
import logging
import os
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("dry_run")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINGS_CONTROL = os.path.join(SCRIPT_DIR, "wings_control")
sys.path.insert(0, WINGS_CONTROL)

# ── 预置场景 ──
SCENARIOS = {
    "glm51-910b-dual": {
        "description": "GLM-5.1 + 910B(A2) 双机 dp_deployment",
        "architecture": "GlmMoeDsaForCausalLM",
        "model_name": "glm-5.1-32b-chat",
        "engine": "vllm_ascend",
        "device_count": 8,
        "nnodes": 2,
        "distributed": True,
        "distributed_executor_backend": "dp_deployment",
        "head_node_addr": "192.168.1.100",
        "node_ips": "192.168.1.100,192.168.1.101",
        "enable_speculative_decode": True,
        "enable_sparse": True,
        "platform": "a2",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "v4flash-a3-16": {
        "description": "DeepSeek-V4-Flash + 910C(A3) 单机16卡",
        "architecture": "DeepseekV4ForCausalLM",
        "model_name": "DeepSeek-V4-Flash",
        "engine": "vllm_ascend",
        "device_count": 16,
        "nnodes": 1,
        "distributed": False,
        "distributed_executor_backend": "dp_deployment",
        "head_node_addr": "127.0.0.1",
        "node_ips": "192.168.1.100",
        "enable_speculative_decode": False,  # A3 会自动开启
        "enable_sparse": False,
        "platform": "a3",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm51-910b-single": {
        "description": "GLM-5.1 + 910B(A2) 单机8卡",
        "architecture": "GlmMoeDsaForCausalLM",
        "model_name": "glm-5.1-32b-chat",
        "engine": "vllm_ascend",
        "device_count": 8,
        "nnodes": 1,
        "distributed": False,
        "distributed_executor_backend": "ray",
        "head_node_addr": "127.0.0.1",
        "node_ips": "192.168.1.100",
        "enable_speculative_decode": True,
        "enable_sparse": True,
        "platform": "a2",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "v4flash-a2-8": {
        "description": "DeepSeek-V4-Flash + 910B(A2) 单机8卡",
        "architecture": "DeepseekV4ForCausalLM",
        "model_name": "DeepSeek-V4-Flash",
        "engine": "vllm_ascend",
        "device_count": 8,
        "nnodes": 1,
        "distributed": False,
        "distributed_executor_backend": "dp_deployment",
        "head_node_addr": "127.0.0.1",
        "node_ips": "192.168.1.100",
        "enable_speculative_decode": False,
        "enable_sparse": False,
        "platform": "a2",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "v4flash-nv-h20-8": {
        "description": "DeepSeek-V4-Flash + NVIDIA H20 单机8卡 (投机推理开)",
        "architecture": "DeepseekV4ForCausalLM",
        "model_name": "DeepSeek-V4-Flash",
        "engine": "vllm",
        "device_count": 8,
        "nnodes": 1,
        "distributed": False,
        "distributed_executor_backend": "mp",
        "head_node_addr": "127.0.0.1",
        "node_ips": "192.168.1.100",
        "enable_speculative_decode": True,
        "enable_sparse": False,
        "platform": "",
        "config_json_extra": {},
    },
    "glm51-a3-dual": {
        "description": "GLM-5.1 + 910C(A3) 双机32卡 dp_deployment",
        "architecture": "GlmMoeDsaForCausalLM",
        "model_name": "glm-5.1-32b-chat",
        "engine": "vllm_ascend",
        "device_count": 16,
        "nnodes": 2,
        "distributed": True,
        "distributed_executor_backend": "dp_deployment",
        "head_node_addr": "192.168.1.100",
        "node_ips": "192.168.1.100,192.168.1.101",
        "enable_speculative_decode": True,
        "enable_sparse": True,
        "platform": "a3",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm51-a3-16": {
        "description": "GLM-5.1 + 910C(A3) 单机16卡",
        "architecture": "GlmMoeDsaForCausalLM",
        "model_name": "glm-5.1-32b-chat",
        "engine": "vllm_ascend",
        "device_count": 16,
        "nnodes": 1,
        "distributed": False,
        "distributed_executor_backend": "ray",
        "head_node_addr": "127.0.0.1",
        "node_ips": "192.168.1.100",
        "enable_speculative_decode": True,
        "enable_sparse": True,
        "platform": "a3",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "v4pro-a3-dual": {
        "description": "DeepSeek-V4-Pro + 910C(A3) 双机32卡 dp_deployment (DP=2)",
        "architecture": "DeepseekV4ForCausalLM",
        "model_name": "DeepSeek-V4-Pro",
        "engine": "vllm_ascend",
        "device_count": 16,
        "nnodes": 2,
        "distributed": True,
        "distributed_executor_backend": "dp_deployment",
        "head_node_addr": "192.168.1.100",
        "node_ips": "192.168.1.100,192.168.1.101",
        "enable_speculative_decode": False,  # A3 会自动开启
        "enable_sparse": False,
        "platform": "a3",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
            "quantize": "w4a8_dynamic",
        },
    },
}


def create_mock_model_dir(architecture: str, extra_config: dict = None) -> str:
    """创建模拟模型目录（含 config.json）。"""
    build_dir = os.path.join(SCRIPT_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    model_dir = tempfile.mkdtemp(prefix="model_", dir=build_dir).replace("\\", "/")
    config = {
        "architectures": [architecture],
        "model_type": "deepseek_v4" if "Deepseek" in architecture else "glm4",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 64,
    }
    if extra_config:
        config.update(extra_config)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return model_dir


def setup_env(scenario: dict, model_dir: str) -> None:
    """设置环境变量模拟 K8s Pod 环境。"""
    shared_vol = tempfile.mkdtemp(prefix="sv_", dir=os.path.join(SCRIPT_DIR, "build")).replace("\\", "/")
    # 根据引擎推断设备类型：vllm_ascend → ascend，否则 nvidia
    device_type = "ascend" if "ascend" in scenario["engine"] else "nvidia"
    env = {
        "ENGINE": scenario["engine"],
        "MODEL_NAME": scenario["model_name"],
        "MODEL_PATH": model_dir,
        "MODEL_TYPE": "auto",
        "DEVICE_COUNT": str(scenario["device_count"]),
        "DISTRIBUTED": str(scenario["distributed"]).lower(),
        "NNODES": str(scenario["nnodes"]),
        "NODE_RANK": "0",
        "HEAD_NODE_ADDR": scenario["head_node_addr"],
        "MASTER_IP": scenario["head_node_addr"],
        "NODE_IPS": scenario["node_ips"],
        "DISTRIBUTED_EXECUTOR_BACKEND": scenario["distributed_executor_backend"],
        "ENABLE_SPECULATIVE_DECODE": str(scenario["enable_speculative_decode"]).lower(),
        "ENABLE_SPARSE": str(scenario["enable_sparse"]).lower(),
        "POD_IP": "192.168.1.100",
        "RANK_IP": "192.168.1.100",
        "NETWORK_INTERFACE": "eth0",
        "PORT": "18000",
        "ENABLE_ACCEL": "false",
        "LMCACHE_OFFLOAD": "false",
        "SHARED_VOLUME_PATH": shared_vol,
        # 硬件设备类型（决定加载 ascend_default.json 还是 nvidia_default.json）
        "WINGS_DEVICE": device_type,
        # 平台标识（A2/A3 细分）
        "WINGS_ASCEND_PLATFORM": scenario.get("platform", ""),
    }
    os.environ.update(env)


def run_dry_run(scenario_name: str, scenario: dict) -> None:
    """执行 dry-run 并输出 start_command.sh。"""
    from core.start_args_compat import parse_launch_args
    from core.port_plan import derive_port_plan
    from core.wings_entry import build_launcher_plan
    from config.settings import settings

    model_dir = create_mock_model_dir(
        scenario["architecture"],
        scenario.get("config_json_extra"),
    )
    setup_env(scenario, model_dir)

    logger.info("=" * 80)
    logger.info("场景: %s — %s", scenario_name, scenario["description"])
    logger.info("  架构: %s | 引擎: %s | 卡数: %d | 节点: %d | 平台: %s",
                scenario["architecture"], scenario["engine"],
                scenario["device_count"], scenario["nnodes"],
                scenario.get("platform", "auto"))
    logger.info("=" * 80)

    # 构建 CLI 参数
    cli_args = [
        "--model-name", scenario["model_name"],
        "--model-path", model_dir,
        "--engine", scenario["engine"],
        "--device-count", str(scenario["device_count"]),
        "--nnodes", str(scenario["nnodes"]),
        "--node-rank", "0",
        "--head-node-addr", scenario["head_node_addr"],
        "--distributed-executor-backend", scenario["distributed_executor_backend"],
    ]
    if scenario["distributed"]:
        cli_args.append("--distributed")
    if scenario["enable_speculative_decode"]:
        cli_args.append("--enable-speculative-decode")
    if scenario["enable_sparse"]:
        cli_args.append("--enable-sparse")

    launch_args = parse_launch_args(cli_args)
    port_plan = derive_port_plan(
        port=launch_args.port,
        enable_reason_proxy=settings.ENABLE_REASON_PROXY,
        health_port=settings.HEALTH_PORT,
    )

    # Node 0
    plan = build_launcher_plan(launch_args, port_plan)

    output_dir = os.path.join(SCRIPT_DIR, "build", "output")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"start_command_{scenario_name}_node0.sh"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(plan.command)
    logger.info("Node 0 → %s (%d bytes)", filename, len(plan.command))

    # 多节点场景: 也生成 Node 1
    if scenario["nnodes"] > 1:
        os.environ["NODE_RANK"] = "1"
        os.environ["RANK_IP"] = scenario["node_ips"].split(",")[1] if "," in scenario["node_ips"] else "192.168.1.101"
        os.environ["POD_IP"] = os.environ["RANK_IP"]
        cli_args_n1 = list(cli_args)
        idx = cli_args_n1.index("--node-rank")
        cli_args_n1[idx + 1] = "1"
        la1 = parse_launch_args(cli_args_n1)
        plan1 = build_launcher_plan(la1, port_plan)
        filename1 = f"start_command_{scenario_name}_node1.sh"
        path1 = os.path.join(output_dir, filename1)
        with open(path1, "w", encoding="utf-8", newline="\n") as f:
            f.write(plan1.command)
        logger.info("Node 1 → %s (%d bytes)", filename1, len(plan1.command))

    # 打印关键参数
    ec = plan.merged_params.get("engine_config", {})
    logger.info("  engine_config 关键字段:")
    for k in ["tensor_parallel_size", "data_parallel_size", "max_model_len",
              "enable_expert_parallel", "quantization", "enable_prefix_caching",
              "enable_chunked_prefill", "compilation_config", "additional_config",
              "speculative_config"]:
        if k in ec:
            logger.info("    %s = %s", k, ec[k])
    logger.info("  enable_speculative_decode = %s", plan.merged_params.get("enable_speculative_decode"))

    # 提取最终 vllm 命令
    for line in plan.command.splitlines():
        stripped = line.strip()
        if stripped.startswith("vllm serve") or stripped.startswith("exec python3 -m vllm") or stripped.startswith("exec vllm"):
            print(f"\n【{scenario_name} Node 0 最终命令】")
            print(stripped)
            break

    # 清理临时模型目录
    import shutil
    shutil.rmtree(model_dir, ignore_errors=True)


# ── PD external-lb 场景（上层契约下发 + pod 内 fork）──
PD_SCENARIOS = {
    "qwen3": {
        "description": "Qwen3-30B-A3B PD 分离 (P:dp2×tp2 / D:dp4×tp1，2 D 节点)",
        "architecture": "Qwen3MoeForCausalLM",
        "model_name": "Qwen3-30B-A3B",
        "prefill": {"dp": 2, "tp": 2, "local": 2, "nodes": ["9.0.0.1"], "rpc": "12321"},
        "decode": {"dp": 4, "tp": 1, "local": 2,
                   "nodes": ["9.0.1.1", "9.0.1.2"], "rpc": "12321"},
    },
    "qwen3-1p1d": {
        "description": "Qwen3-30B-A3B 1P1D (P:dp1×tp4 / D:dp1×tp4，单节点 P/D 分离)",
        "architecture": "Qwen3MoeForCausalLM",
        "model_name": "Qwen3-30B-A3B",
        "prefill": {"dp": 1, "tp": 4, "local": 1, "nodes": ["10.254.0.1"], "rpc": "12890"},
        "decode": {"dp": 1, "tp": 4, "local": 1, "nodes": ["10.254.0.2"], "rpc": "12777"},
    },
    "glm5": {
        "description": "GLM-5 PD 分离 (P:dp2×tp16 / D:dp16×tp4)",
        "architecture": "GlmMoeDsaForCausalLM",
        "model_name": "glm-5.1-chat",
        "prefill": {"dp": 2, "tp": 16, "local": 1,
                    "nodes": ["7.0.0.1", "7.0.0.2"], "rpc": "10521"},
        "decode": {"dp": 16, "tp": 4, "local": 4,
                   "nodes": ["7.0.1.1", "7.0.1.2", "7.0.1.3", "7.0.1.4"], "rpc": "10523"},
    },
    "glm52-a2": {
        "description": "GLM-5.2 PD 分离 8×A2 4P4D (P:dp4×tp8 local1 / D:dp8×tp4 local2)",
        "architecture": "GlmMoeDsaForCausalLM",
        "model_name": "glm-5.2-chat",
        "platform": "a2",
        "prefill": {"dp": 4, "tp": 8, "local": 1,
                    "nodes": ["7.0.0.1", "7.0.0.2", "7.0.0.3", "7.0.0.4"], "rpc": "12890"},
        "decode": {"dp": 8, "tp": 4, "local": 2,
                   "nodes": ["7.0.1.1", "7.0.1.2", "7.0.1.3", "7.0.1.4"], "rpc": "12777"},
    },
    "glm52-nosig": {
        "description": "GLM-5.2 PD 8×A2 但【不设平台信号】→ 应靠 default_platform=a2 退到 A2",
        "architecture": "GlmMoeDsaForCausalLM",
        "model_name": "glm-5.2-chat",
        "platform": "",
        "prefill": {"dp": 4, "tp": 8, "local": 1,
                    "nodes": ["7.0.0.1", "7.0.0.2", "7.0.0.3", "7.0.0.4"], "rpc": "12890"},
        "decode": {"dp": 8, "tp": 4, "local": 2,
                   "nodes": ["7.0.1.1", "7.0.1.2", "7.0.1.3", "7.0.1.4"], "rpc": "12777"},
    },
    "v4flash": {
        "description": "DeepSeek-V4-Flash A3 PD 分离 (P:dp4×tp4 / D:dp16×tp1)",
        "architecture": "DeepseekV4ForCausalLM",
        "model_name": "DeepSeek-V4-Flash",
        "prefill": {"dp": 4, "tp": 4, "local": 4, "nodes": ["8.0.0.1"], "rpc": "10521"},
        "decode": {"dp": 16, "tp": 1, "local": 16, "nodes": ["8.0.1.1"], "rpc": "10523"},
    },
}


def run_pd_dry_run(name: str, scenario: dict) -> None:
    """生成 PD external-lb 场景的 P/D 启动脚本（每角色 node0；D 多节点再出 node1 展示 rank 派生）。"""
    from core.start_args_compat import parse_launch_args
    from core.port_plan import derive_port_plan
    from core.wings_entry import build_launcher_plan
    from config.settings import settings

    arch = scenario["architecture"]
    model_name = scenario["model_name"]
    pf, dc = scenario["prefill"], scenario["decode"]
    output_dir = os.path.join(SCRIPT_DIR, "build", "output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PD 场景: %s — %s", name, scenario["description"])
    logger.info("=" * 80)

    def _one(role_key, topo, node_idx):
        role = "P" if role_key == "prefill" else "D"
        node_ips = ",".join(topo["nodes"])
        host_ip = topo["nodes"][node_idx]
        model_dir = create_mock_model_dir(arch, {"quantization_config": {"quant_method": "ascend"}})
        for k in list(os.environ):
            if k.startswith(("PD_", "DP_", "TP_")) or k in (
                    "NODE_IPS", "HOST_IP", "Master_IP", "MASTER_IP", "VLLM_LLMDD_RPC_PORT",
                    "RANK_IP", "POD_IP"):
                os.environ.pop(k, None)
        # 精简 env：只保留「无 CLI 等价 / 无默认（或默认不对）」的真·环境变量。探针验证移除后命令字节级不变：
        #   - CLI 承载/argparse 默认：ENGINE/MODEL_NAME/MODEL_PATH/MODEL_TYPE/DISTRIBUTED/NNODES/NODE_RANK/PORT/ENGINE_PORT
        #   - 有默认且不影响 PD 命令：SHARED_VOLUME_PATH（默认 /shared-volume，仅 LMCache 用）
        os.environ.update({
            # —— ② 平台/硬件（无 CLI 等价，真机由 K8s/镜像注入）——
            # ⚠️ WINGS_ASCEND_PLATFORM 必填：缺省时 _resolve_deepseek_v4_flash_platform 回退 a2（非 a3！），
            #    A3 部署须显式 a3，或靠硬件探测（/shared-volume/hardware_info.json 含 910c / ASCEND_A3_ENABLE=1）。
            "WINGS_DEVICE": "ascend", "WINGS_ASCEND_PLATFORM": scenario.get("platform", "a3"),
            # DEVICE_COUNT：hardware_detect/device_utils 直读取硬件 count（--device-count 另喂 launcher）
            "DEVICE_COUNT": str(topo["local"] * topo["tp"]),
            # RANK_IP：上层(MaaS)下发的本 pod 唯一 IP，是标识本机的唯一真相源。
            # get_local_ip() 读它；current_ip(→HCCL_IF_IP) 与 PD 的 HOST_IP 均回退到它，
            # 故不单独设 POD_IP / HOST_IP（重复设置）。
            "RANK_IP": host_ip,
            # —— ① 上层 PD 契约（无 CLI，必须 env）——
            # DP_SIZE/TP_SIZE 不再单独下发：从本角色全局拓扑 PD_{ROLE}_* 派生（见 _get_pd_external_lb_params）。
            # DP_SIZE_LOCAL 不可派生（=卡/节点÷tp，依赖节点数），仍须下发。
            # VLLM_LLMDD_RPC_PORT 可省（默认 P=12890/D=12777）；此处显式给例值以贴合自定义部署。
            "PD_ROLE": role,
            "DP_SIZE_LOCAL": str(topo["local"]), "Master_IP": topo["nodes"][0],
            "VLLM_LLMDD_RPC_PORT": topo["rpc"], "NODE_IPS": node_ips,
            # —— ① KV 全局拓扑（P/D 互相感知对方；本角色 dp/tp 亦由此派生）——
            "PD_PREFILL_DP_SIZE": str(pf["dp"]), "PD_PREFILL_TP_SIZE": str(pf["tp"]),
            "PD_DECODE_DP_SIZE": str(dc["dp"]), "PD_DECODE_TP_SIZE": str(dc["tp"]),
        })
        # CLI 入参（承载 model/engine/device-count/nnodes/node-rank；其 env 兜底名见 build_parser）
        la = parse_launch_args(["--model-name", model_name, "--model-path", model_dir,
                                "--engine", "vllm_ascend", "--device-count",
                                str(topo["local"] * topo["tp"]), "--nnodes", "1", "--node-rank", "0"])
        pp = derive_port_plan(port=la.port, enable_reason_proxy=settings.ENABLE_REASON_PROXY,
                              health_port=settings.HEALTH_PORT)
        cmd = build_launcher_plan(la, pp).command
        fn = f"start_command_pd-{name}-{role}_node{node_idx}.sh"
        with open(os.path.join(output_dir, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write(cmd)
        logger.info("  %s-node%d → %s (%d bytes)", role, node_idx, fn, len(cmd))
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)

    _one("prefill", pf, 0)
    _one("decode", dc, 0)
    if len(dc["nodes"]) > 1:
        _one("decode", dc, 1)  # 展示 dp_rank_start 由 HOST_IP 派生


def main():
    parser = argparse.ArgumentParser(description="Wings-infer Dry-Run: 生成 start_command.sh")
    parser.add_argument("--scenario", "-s", choices=list(SCENARIOS.keys()),
                        help="预置场景名称")
    parser.add_argument("--pd", choices=list(PD_SCENARIOS.keys()),
                        help="PD external-lb 场景（glm5 / v4flash）")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有预置场景")
    args = parser.parse_args()

    if args.pd:
        run_pd_dry_run(args.pd, PD_SCENARIOS[args.pd])
        logger.info("PD DRY RUN COMPLETE — 输出目录: build/output/")
        return

    if args.list:
        print("可用场景:")
        for name, cfg in SCENARIOS.items():
            print(f"  {name:20s} — {cfg['description']}")
        return

    if not args.scenario:
        # 默认跑所有场景
        for name, cfg in SCENARIOS.items():
            run_dry_run(name, cfg)
    else:
        run_dry_run(args.scenario, SCENARIOS[args.scenario])

    logger.info("=" * 80)
    logger.info("DRY RUN COMPLETE — 输出目录: build/output/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
