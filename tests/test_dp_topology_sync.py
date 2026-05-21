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


def _make_model_dir(parent: Path, name: str, architecture: str, extra_config: dict | None = None) -> Path:
    model_dir = parent / name
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {"architectures": [architecture]}
    if extra_config:
        config.update(extra_config)
    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False),
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
        cls.v4_pro_from_config = _make_model_dir(
            tmpdir,
            "opaque-serving-model",
            "DeepseekV4ForCausalLM",
            {"_name_or_path": "DeepSeek-V4-Pro-w4a8-mtp1"},
        )
        # 名字 / config 字符串都不含 "pro"，只能靠 quantize 字段识别（w4a8 = V4-Pro）。
        cls.v4_pro_by_quantize = _make_model_dir(
            tmpdir,
            "served-model-anonymous",
            "DeepseekV4ForCausalLM",
            {"quantize": "w4a8"},
        )
        cls.v4_pro_by_quant_config = _make_model_dir(
            tmpdir,
            "served-model-anonymous-2",
            "DeepseekV4ForCausalLM",
            {"quantization_config": {"quant_method": "ascend-w4a8"}},
        )
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

    def test_v4_pro_unknown_platform_uses_model_requirement_not_a2_fallback(self):
        """V4-Pro 权重/名称已明确时，硬件 details 为空不能默默按 A2 跳过 Pro 默认。"""
        params = self._base_v4_pro_params()
        params["device_details"] = []

        _prepare_engine_config(params)

        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 16, ec)
        self.assertEqual(ec.get("data_parallel_size"), 2, ec)
        self.assertEqual(ec.get("data_parallel_size_local"), 1, ec)

    def test_v4_pro_identity_can_come_from_model_config(self):
        """当外部 model_name/path 不含 Pro 标记时，可从 config.json 名称字段识别 V4-Pro。"""
        params = {
            "model_name": "served-model",
            "model_path": str(self.v4_pro_from_config),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 16,
            "device_details": [],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }

        _prepare_engine_config(params)

        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 16, ec)
        self.assertEqual(ec.get("data_parallel_size"), 2, ec)
        self.assertEqual(ec.get("data_parallel_size_local"), 1, ec)

    def test_v4_pro_identity_can_come_from_quantize_field(self):
        """名字 / served_name 都不含 "pro"，但 config.json 量化指纹是 w4a8 → 应识别为 V4-Pro。

        回归崩溃场景：hardware_info.json 没有 details 字段；如果连模型名也被改成
        通用 served-model，identity_text 无法包含 "pro"，platform 兜底为 a2，
        scope 闸门拒绝注入 TP/DP → DP resolver 抛 "positive tensor_parallel_size"。
        加上 quantize 权重指纹后，应直接识别为 V4-Pro → 注入 TP=16/DP=2。
        """
        params = {
            "model_name": "served-model-anonymous",
            "model_path": str(self.v4_pro_by_quantize),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 16,
            "device_details": [],  # 缺失硬件信号
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }

        _prepare_engine_config(params)

        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 16, ec)
        self.assertEqual(ec.get("data_parallel_size"), 2, ec)
        self.assertEqual(ec.get("data_parallel_size_local"), 1, ec)

    def test_v4_pro_identity_can_come_from_quantization_config_quant_method(self):
        """config.json 用 quantization_config.quant_method 形式标注 w4a8 也应识别。"""
        params = {
            "model_name": "served-model-anonymous-2",
            "model_path": str(self.v4_pro_by_quant_config),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 16,
            "device_details": [],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }

        _prepare_engine_config(params)

        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 16, ec)
        self.assertEqual(ec.get("data_parallel_size"), 2, ec)

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

    # ──────────────── V4-Flash A2 (910B) 双机 2×8 ────────────────
    def test_v4_flash_a2_dual_2x8_syncs_tp_and_dp(self):
        params = {
            "model_name": "DeepSeek-V4-Flash",
            "model_path": str(self.v4_flash),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 8,
            "device_details": [{"name": "910b"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 8, ec)
        # V4-Flash A2: total_cards = 8*2 = 16, dp = 16//8 = 2
        self.assertEqual(ec.get("data_parallel_size"), 2, ec)

        model_info = ModelIdentifier(
            params["model_name"], params["model_path"], params["model_type"])
        dp_size, dp_local, dp_start = _resolve_dp_deployment_topology(
            params, _ctx(nnodes=2), model_info)
        # dp_size_local = 8 // 8 = 1; dp_size = 1 * 2 = 2
        self.assertEqual((dp_size, dp_local, dp_start), ("2", "1", "0"))

    # ──────────────── V3 A2 双机 2×8 通用 DP 块 ────────────────
    def test_v3_a2_dual_2x8_generic_dp_block_syncs_tp(self):
        params = {
            "model_name": "DeepSeek-V3",
            "model_path": str(self.v3),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 8,
            "device_details": [{"name": "910b"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        # V3 device_count>=4 → TP=4
        self.assertEqual(ec.get("tensor_parallel_size"), 4, ec)

        model_info = ModelIdentifier(
            params["model_name"], params["model_path"], params["model_type"])
        dp_size, dp_local, dp_start = _resolve_dp_deployment_topology(
            params, _ctx(nnodes=2), model_info)
        # dp_size_local = 8 // 4 = 2；dp_size = 2 * 2 = 4
        self.assertEqual((dp_size, dp_local, dp_start), ("4", "2", "0"))

    # ──────────────── GLM-5.1 A2 双机 2×8 通用 DP 块 ────────────────
    def test_glm51_a2_dual_2x8_generic_dp_block_syncs_tp(self):
        params = {
            "model_name": "glm-5.1-w8a8",
            "model_path": str(self.glm51),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 8,
            "device_details": [{"name": "910b"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        # GlmMoeDsa device_count in (8,16) → device_count
        self.assertEqual(ec.get("tensor_parallel_size"), 8, ec)

        model_info = ModelIdentifier(
            params["model_name"], params["model_path"], params["model_type"])
        dp_size, dp_local, dp_start = _resolve_dp_deployment_topology(
            params, _ctx(nnodes=2), model_info)
        # dp_size_local = 8 // 8 = 1; dp_size = 1 * 2 = 2
        self.assertEqual((dp_size, dp_local, dp_start), ("2", "1", "0"))

    # ──────────────── V4-Flash A3 单机 1×16 sync ────────────────
    def test_v4_flash_a3_single_1x16_sync(self):
        """单机不走 dp_deployment 拓扑解析（build_start_script 控制），
        但 _prepare_engine_config 仍跑，sync 必须把 TP/DP 写回 params。"""
        params = {
            "model_name": "DeepSeek-V4-Flash",
            "model_path": str(self.v4_flash),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": False,
            "nnodes": 1,
            "node_rank": 0,
            "device_count": 16,
            "device_details": [{"name": "910c"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 8, ec)
        # A3 单机: total_cards = 16 (is_distributed=False 路径), dp = 16//8 = 2
        self.assertEqual(ec.get("data_parallel_size"), 2, ec)

    # ──────────────── V4-Flash A2 单机 1×8 sync ────────────────
    def test_v4_flash_a2_single_1x8_sync(self):
        params = {
            "model_name": "DeepSeek-V4-Flash",
            "model_path": str(self.v4_flash),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": False,
            "nnodes": 1,
            "node_rank": 0,
            "device_count": 8,
            "device_details": [{"name": "910b"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        self.assertEqual(ec.get("tensor_parallel_size"), 8, ec)
        # A2 单机: total_cards = 8, dp = 8//8 = 1
        self.assertEqual(ec.get("data_parallel_size"), 1, ec)

    # ──────────────── V4-Pro scope-guard：A2 闸门拒绝 ────────────────
    def test_v4_pro_a2_scope_guard_skips_injection(self):
        """V4-Pro 适配范围限定 A3 双机；A2 路径下应不注入 TP/DP（闸门排除），
        同时通用 DP 块也因 _is_deepseek_v4_pro_params 而跳过。"""
        params = {
            "model_name": "DeepSeek-V4-Pro-w4a8-mtp1",
            "model_path": str(self.v4_pro),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 8,
            "device_details": [{"name": "910b"}],  # ← A2，闸门拒绝
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        self.assertNotIn("tensor_parallel_size", ec, ec)
        self.assertNotIn("data_parallel_size", ec, ec)
        self.assertNotIn("data_parallel_size_local", ec, ec)

    # ──────────────── V4-Pro scope-guard：单机 nnodes=1 闸门拒绝 ────────────────
    def test_v4_pro_single_node_scope_guard_skips_injection(self):
        """V4-Pro 单机部署不在适配范围（nnodes==2 才进入），不应注入 V4-Pro 默认。"""
        params = {
            "model_name": "DeepSeek-V4-Pro-w4a8-mtp1",
            "model_path": str(self.v4_pro),
            "model_type": "auto",
            "engine": "vllm_ascend",
            "distributed": False,
            "nnodes": 1,
            "node_rank": 0,
            "device_count": 16,
            "device_details": [{"name": "910c"}],
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        _prepare_engine_config(params)
        ec = params["engine_config"]
        self.assertNotIn("tensor_parallel_size", ec, ec)
        self.assertNotIn("data_parallel_size", ec, ec)

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
