# -*- coding: utf-8 -*-
"""模拟 GLM-5.1 (GlmMoeDsaForCausalLM) Ascend 双机 ×8 卡 dp_deployment 部署，
打印 rank0 / rank1 两个节点的最终启动脚本（含环境变量与 vllm CLI 字段）。
"""
# pyright: reportMissingImports=false

import json
import sys
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

import engines.vllm_adapter as vllm_adapter  # noqa: E402


class _FakeGlm51Model:
    def __init__(self, *args, **kwargs):
        self.model_architecture = "GlmMoeDsaForCausalLM"
        self.model_quantize = ""


def _load_glm51_default_cfg():
    p = ROOT / "wings_control" / "config" / "defaults" / "ascend_default.json"
    return json.loads(p.read_text(encoding="utf-8-sig"))[
        "model_deploy_config"]["llm"]["GlmMoeDsaForCausalLM"]["default"]["vllm_ascend"]


def _build_params(node_rank: int) -> dict:
    """模拟 wings 上层 (config_loader + 修复后的 _set_parallelism_params)
    对 GLM-5.1 双机 ×8 卡 dp_deployment 算出的 params。"""
    cfg = dict(_load_glm51_default_cfg())
    cfg.update({
        "model": "/usr/local/serving/models/",
        "host": "10.254.124.178",
        "port": 18000,
        "served_model_name": "GLM-5.1",
        "dtype": "auto",
        "kv_cache_dtype": "auto",
        "block_size": 128,
        # tensor_parallel_size 不预填 —— 模拟修复后:
        # _set_parallelism_params 早返回，等 vllm_adapter._prepare_engine_config
        # 通过 _default_deepseek_ascend_dp_tensor_parallel_size 兜底成 tp=8
    })
    return {
        "engine": "vllm_ascend",
        "distributed": True,
        "nnodes": 2,
        "node_rank": node_rank,
        "head_node_addr": "10.254.124.178",
        "master_ip": "10.254.124.178",
        "node_ips": "10.254.124.178,10.254.13.111",
        "distributed_executor_backend": "dp_deployment",
        "rpc_port": 27071,
        "nixl_port": 27070,
        "device_count": 8,
        "model_name": "GLM-5.1",
        "model_path": "/usr/local/serving/models/",
        "model_type": "",
        "enable_speculative_decode": False,
        "engine_config": cfg,
    }


def _split_exports_and_cmd(script: str):
    """从 shell 脚本里抽出 export 行和 exec vllm 行。"""
    exports, exec_cmd = [], None
    for line in script.splitlines():
        s = line.strip()
        if s.startswith("export ") and "=" in s:
            exports.append(s[len("export "):])
        elif s.startswith("exec vllm") or s.startswith("exec python"):
            exec_cmd = s
    return exports, exec_cmd


def _format_cli(cmd: str):
    """按 --flag 切分一行 vllm CLI，便于阅读。"""
    if not cmd:
        return ""
    parts, cur = [], []
    for tok in cmd.split():
        if tok.startswith("--") and cur:
            parts.append(" ".join(cur))
            cur = [tok]
        else:
            cur.append(tok)
    if cur:
        parts.append(" ".join(cur))
    return "\n  ".join(parts)


def print_node_script(node_rank: int):
    params = _build_params(node_rank)
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(vllm_adapter, "ModelIdentifier", _FakeGlm51Model):
        script = vllm_adapter.build_start_script(params)
    exports, exec_cmd = _split_exports_and_cmd(script)

    print("\n" + "=" * 80)
    print(f" Node rank = {node_rank}   (GLM-5.1 / GlmMoeDsaForCausalLM, vllm_ascend, dp_deployment)")
    print("=" * 80)

    print("\n[环境变量 export]")
    for e in exports:
        print(f"  export {e}")

    print("\n[vLLM 启动命令]")
    print("  " + _format_cli(exec_cmd))


if __name__ == "__main__":
    print_node_script(0)
    print_node_script(1)
