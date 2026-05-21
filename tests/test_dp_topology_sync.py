# -*- coding: utf-8 -*-
"""验证 `_prepare_engine_config` 把 DP 拓扑键回写到 ``params["engine_config"]``。

回归用例：

- V4-Pro 双机 A3 16 卡/节点：``_apply_deepseek_v4_pro_engine_defaults`` 注入
  TP=16/DP=2/DP_local=1/DP_start_rank=node_rank 必须经 sync 后被
  ``_resolve_dp_deployment_topology`` 读到。
- V4-Flash 双机 A3 16 卡/节点：``_apply_deepseek_v4_flash_engine_defaults`` 注入
  TP=8/DP=4 同样必须被回写。
- V3 / GLM-5 双机 dp_deployment：走通用 DP 块默认表，TP 也必须落到
  ``params["engine_config"]``。
- 用户显式 ``tensor_parallel_size`` 不被 applier 覆盖（仍走显式优先）。
- NVIDIA vllm 非 DP 路径：``params["engine_config"]`` 不应被注入额外 DP 键。

历史背景：``_apply_deepseek_v4_pro_engine_defaults`` 此前只写局部 engine_config
+ 顶层 ``params["tensor_parallel_size"]``，但 ``_resolve_dp_deployment_topology``
从 ``params["engine_config"]["tensor_parallel_size"]`` 读，导致 V4-Pro 单机
（A3 16 卡）/ V4-Flash 多机崩溃：
``ValueError: DeepSeek Ascend DP requires a positive tensor_parallel_size``。
V3/V32/GLM-5 此前未崩是因为 ``_default_deepseek_ascend_dp_tensor_parallel_size``
兜底表覆盖了它们；V4 不在表里，直接暴露 sync 缺陷。
"""
# pyright: reportMissingImports=false

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from engines.vllm_adapter import (  # noqa: E402
    DistScriptCtx,
    _prepare_engine_config,
    _resolve_dp_deployment_topology,
)
from utils.model_utils import ModelIdentifier  # noqa: E402


def _make_model_dir(parent: Path, name: str, architecture: str) -> Path:
    model_dir = parent / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": [architecture]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return model_dir


def _ctx(nnodes: int, node_rank: int = 0) -> DistScriptCtx:
    return DistScriptCtx(
        engine="vllm_ascend",
        cmd="placeholder",
        is_ascend=True,
        node_rank=node_rank,
        nnodes=nnodes,
        head_addr="127.0.0.1",
        ray_port="28020",
        node_ips="127.0.0.1",
    )


class TestDpTopologySync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir = Path(cls._tmpdir_ctx.name)
        cls.v4_pro = _make_model_dir(tmpdir, "DeepSeek-V4-Pro-w4a8-mtp1", "DeepseekV4ForCausalLM")
        cls.v4_flash = _make_model_dir(tmpdir, "DeepSeek-V4-Flash", "DeepseekV4ForCausalLM")
        cls.v3 = _make_model_dir(tmpdir, "DeepSeek-V3", "DeepseekV3ForCausalLM")
        cls.glm51 = _make_model_dir(tmpdir, "glm-5.1-w8a8", "GlmMoeDsaForCausalLM")

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir_ctx.cleanup()

    def _base_v4_pro_params(self, node_rank: int = 0):
        return {
            "model_name": "DeepSeek-V4-Pro-w4a8-mtp1",
            "model_path": str(self.v4_pro),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": node_rank,
            "device_count": 16,
            "device_details": [{"name": "910c"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }

    # ──────────────── V4-Pro 双机 A3 ────────────────
    def test_v4_pro_dual_node_a3_syncs_tp_and_dp(self):
        params = self._base_v4_pro_params()
        _prepare_engine_config(params)
        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 16, ec)
        self.assertEqual(ec.get("data_parallel_size"), 2, ec)
        self.assertEqual(ec.get("data_parallel_size_local"), 1, ec)
        self.assertEqual(ec.get("data_parallel_start_rank"), 0, ec)

        model_info = ModelIdentifier(
            params["model_name"], params["model_path"], params["model_type"])
        dp_size, dp_local, dp_start = _resolve_dp_deployment_topology(
            params, _ctx(nnodes=2, node_rank=0), model_info)
        self.assertEqual((dp_size, dp_local, dp_start), ("2", "1", "0"))

    def test_v4_pro_rank1_dp_start_rank_follows_node_rank(self):
        params = self._base_v4_pro_params(node_rank=1)
        _prepare_engine_config(params)
        self.assertEqual(params["engine_config"].get("data_parallel_start_rank"), 1)

        model_info = ModelIdentifier(
            params["model_name"], params["model_path"], params["model_type"])
        dp_size, dp_local, dp_start = _resolve_dp_deployment_topology(
            params, _ctx(nnodes=2, node_rank=1), model_info)
        # 注：DistScriptCtx-level dp_start = node_rank * dp_size_local = 1
        self.assertEqual((dp_size, dp_local, dp_start), ("2", "1", "1"))

    # ──────────────── V4-Flash 双机 A3 ────────────────
    def test_v4_flash_dual_node_a3_syncs_tp_and_dp(self):
        params = {
            "model_name": "DeepSeek-V4-Flash",
            "model_path": str(self.v4_flash),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 16,
            "device_details": [{"name": "910c"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 8, ec)
        # V4-Flash applier: dp = device_count * nnodes // 8 = 16*2//8 = 4
        self.assertEqual(ec.get("data_parallel_size"), 4, ec)

        model_info = ModelIdentifier(
            params["model_name"], params["model_path"], params["model_type"])
        dp_size, dp_local, dp_start = _resolve_dp_deployment_topology(
            params, _ctx(nnodes=2), model_info)
        # 拓扑 resolver 用 device_count // tp 重新算 dp_size_local：16//8 = 2
        # dp_size = dp_size_local * nnodes = 4
        self.assertEqual((dp_size, dp_local, dp_start), ("4", "2", "0"))

    # ──────────────── V3 通用 DP 块 ────────────────
    def test_v3_dual_node_generic_dp_block_syncs_tp(self):
        params = {
            "model_name": "DeepSeek-V3",
            "model_path": str(self.v3),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 16,
            "device_details": [{"name": "910c"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        # 通用 DP 块按 _default_deepseek_ascend_dp_tensor_parallel_size 默认表：
        # DeepseekV3ForCausalLM, device_count=16 → 4
        self.assertEqual(ec.get("tensor_parallel_size"), 4, ec)

        model_info = ModelIdentifier(
            params["model_name"], params["model_path"], params["model_type"])
        dp_size, dp_local, dp_start = _resolve_dp_deployment_topology(
            params, _ctx(nnodes=2), model_info)
        # dp_size_local = 16 // 4 = 4；dp_size = 4 * 2 = 8
        self.assertEqual((dp_size, dp_local, dp_start), ("8", "4", "0"))

    # ──────────────── GLM-5.1 通用 DP 块 ────────────────
    def test_glm51_dual_node_generic_dp_block_syncs_tp(self):
        params = {
            "model_name": "glm-5.1-w8a8",
            "model_path": str(self.glm51),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 16,
            "device_details": [{"name": "910c"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        # GlmMoeDsaForCausalLM, device_count=16 → 16
        self.assertEqual(ec.get("tensor_parallel_size"), 16, ec)

        model_info = ModelIdentifier(
            params["model_name"], params["model_path"], params["model_type"])
        dp_size, dp_local, dp_start = _resolve_dp_deployment_topology(
            params, _ctx(nnodes=2), model_info)
        # dp_size_local = 16 // 16 = 1；dp_size = 1 * 2 = 2
        self.assertEqual((dp_size, dp_local, dp_start), ("2", "1", "0"))

    # ──────────────── 用户显式 TP 不被 applier 覆盖 ────────────────
    def test_v4_pro_explicit_user_tp_preserved(self):
        params = self._base_v4_pro_params()
        params["engine_config"]["tensor_parallel_size"] = 32
        params["_explicit_cli_keys"] = ["tensor_parallel_size"]
        _prepare_engine_config(params)
        self.assertEqual(params["engine_config"].get("tensor_parallel_size"), 32)

    # ──────────────── NVIDIA vllm 非 DP 路径回归 ────────────────
    def test_nvidia_vllm_non_dp_path_no_dp_keys_injected(self):
        params = {
            "model_name": "llama-3-8b",
            "model_path": str(self.v3),  # arch 不重要，只需可加载的 config.json
            "model_type": "auto",
            "engine": "vllm",
            "distributed": False,
            "nnodes": 1,
            "node_rank": 0,
            "device_count": 1,
            "distributed_executor_backend": "ray",
            "engine_config": {"tensor_parallel_size": 1},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 1)
        self.assertNotIn("data_parallel_size", ec)
        self.assertNotIn("data_parallel_size_local", ec)
        self.assertNotIn("data_parallel_start_rank", ec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
