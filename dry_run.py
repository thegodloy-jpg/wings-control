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
        "description": "DeepSeek-V4-Flash + NVIDIA H20 单机8卡 (投机推理 + IndexCache默认强制开 + native KV 卸载)",
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
        # [V4-Flash-NV-Day0] 不传 --enable-sparse：验证 IndexCache 由强制闸默认开（方案 A）
        "enable_sparse": False,
        "enable_kv_offload": True,
        "lmcache_max_local_cpu_size": 25,
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
    "qwen36-35b-a3b": {
        "description": "Qwen3.6-35B-A3B(MoE) + 910C(A3) 单机2卡 (FC+think+spec, 验证 qwen3_coder/enforce_eager/MoE recipe)",
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "model_name": "Qwen3.6-35B-A3B",
        "engine": "vllm_ascend",
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
        "distributed_executor_backend": "mp",
        "head_node_addr": "127.0.0.1",
        "node_ips": "192.168.1.100",
        "enable_speculative_decode": True,
        "enable_sparse": False,
        "enable_auto_tool_choice": True,
        "enable_auto_think_choice": True,
        "platform": "a3",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "qwen35-397b-a17b": {
        "description": "Qwen3.5-397B-A17B(MoE) + 910C(A3) 单机16卡 (FC+think, spec默认关 → 验证无 speculative_config)",
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "model_name": "Qwen3.5-397B-A17B",
        "engine": "vllm_ascend",
        "device_count": 16,
        "nnodes": 1,
        "distributed": False,
        "distributed_executor_backend": "mp",
        "head_node_addr": "127.0.0.1",
        "node_ips": "192.168.1.100",
        "enable_speculative_decode": False,
        "enable_sparse": False,
        "enable_auto_tool_choice": True,
        "enable_auto_think_choice": True,
        "platform": "a3",
        "config_json_extra": {
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "qwen36-27b": {
        "description": "Qwen3.6-27B(dense) + 910C(A3) 单机2卡 (FC+think+spec, 验证 mamba_cache_mode=align + enforce_eager)",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_name": "Qwen3.6-27B",
        "engine": "vllm_ascend",
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
        "distributed_executor_backend": "mp",
        "head_node_addr": "127.0.0.1",
        "node_ips": "192.168.1.100",
        "enable_speculative_decode": True,
        "enable_sparse": False,
        "enable_auto_tool_choice": True,
        "enable_auto_think_choice": True,
        "platform": "a3",
        "config_json_extra": {},
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
        # Function call / reasoning 解耦开关：默认关闭（与生产一致），场景可显式打开
        # 以便在生成的 start_command 中观察 tool_call_parser / reasoning_parser。
        "ENABLE_AUTO_TOOL_CHOICE": str(scenario.get("enable_auto_tool_choice", False)).lower(),
        "ENABLE_AUTO_THINK_CHOICE": str(scenario.get("enable_auto_think_choice", False)).lower(),
        "POD_IP": "192.168.1.100",
        "RANK_IP": "192.168.1.100",
        "NETWORK_INTERFACE": "eth0",
        "PORT": "18000",
        "ENABLE_ACCEL": "false",
        "LMCACHE_OFFLOAD": "true" if scenario.get("enable_kv_offload") else "false",
        "SHARED_VOLUME_PATH": shared_vol,
        # 硬件设备类型（决定加载 ascend_default.json 还是 nvidia_default.json）
        "WINGS_DEVICE": device_type,
        # 平台标识（A2/A3 细分）
        "WINGS_ASCEND_PLATFORM": scenario.get("platform", ""),
    }
    # KV 卸载容量（每卡 GB，V4-Flash 乘本节点卡数；仅在开启卸载时注入）
    if scenario.get("enable_kv_offload") and scenario.get("lmcache_max_local_cpu_size"):
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(scenario["lmcache_max_local_cpu_size"])
    os.environ.update(env)
    # 未开启卸载的场景需清除上一轮残留，避免跨场景串味
    if not scenario.get("enable_kv_offload"):
        os.environ.pop("LMCACHE_MAX_LOCAL_CPU_SIZE", None)


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


def main():
    parser = argparse.ArgumentParser(description="Wings-infer Dry-Run: 生成 start_command.sh")
    parser.add_argument("--scenario", "-s", choices=list(SCENARIOS.keys()),
                        help="预置场景名称")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有预置场景")
    args = parser.parse_args()

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
