# -*- coding: utf-8 -*-
"""验证 _set_parallelism_params 对 Ascend DeepSeek dp_deployment 的早返回精确性。

回归用例:
- Ascend DeepSeek dp_deployment 多机: 不再写入错误的 tp=device_count×nnodes
- NVIDIA PD dp_deployment: 仍按原 PD 分支写入 tp=device_count（不被误伤）
- Ray 多节点 (Ascend 非 DeepSeek): 仍按全局公式写入 tp=device_count×nnodes
- 单机: 仍按 device_count 写入
- 用户显式 CLI 指定 tp: 不被覆盖
"""
# pyright: reportMissingImports=false

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.config_loader import _set_parallelism_params  # noqa: E402


class TestDpDeploymentTpFix(unittest.TestCase):
    def setUp(self):
        # 清理可能影响 _adjust_tensor_parallelism 的环境变量
        env_patch = patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    # --- 主目标场景:Ascend DeepSeek 双机 ×8 卡 dp_deployment ---
    def test_ascend_deepseek_dp_deployment_multinode_skips_tp(self):
        """触发原 bug 的场景:tp 必须未被写入,等 vllm_adapter 兜底。"""
        params = {}
        ctx = {
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_ips": "10.0.0.1,10.0.0.2",
            "device_count": 8,
            "distributed_executor_backend": "dp_deployment",
        }
        _set_parallelism_params(params, ctx)
        self.assertNotIn(
            "tensor_parallel_size", params,
            "Ascend DeepSeek dp_deployment 多机场景下 tp 应早返回不写入,"
            "实际写入了 %r" % params.get("tensor_parallel_size"),
        )

    def test_ascend_deepseek_dp_deployment_singlenode_skips_tp(self):
        """单机 dp_deployment(DeepSeek + 1×16 卡):同样早返回。"""
        params = {}
        ctx = {
            "engine": "vllm_ascend",
            "distributed": False,
            "nnodes": 1,
            "node_ips": "",
            "device_count": 16,
            "distributed_executor_backend": "dp_deployment",
        }
        _set_parallelism_params(params, ctx)
        self.assertNotIn("tensor_parallel_size", params)

    # --- 回归保护:NVIDIA PD 不被误伤 ---
    def test_nvidia_pd_dp_deployment_keeps_tp_device_count(self):
        """NVIDIA PD 同样写 dp_deployment,但不能被早返回短路。"""
        params = {}
        ctx = {
            "engine": "vllm",
            "distributed": True,
            "nnodes": 1,
            "node_ips": "",
            "device_count": 4,
            "distributed_executor_backend": "dp_deployment",
        }
        with patch("core.config_loader.get_pd_role_env", return_value="P"):
            _set_parallelism_params(params, ctx)
        # PD 分支:tp = device_count(不是 ×nnodes)
        self.assertEqual(params.get("tensor_parallel_size"), 4)

    def test_nvidia_pd_dp_deployment_singlenode_keeps_tp(self):
        """NVIDIA PD 单机(if_distributed=False)依然 tp=device_count。"""
        params = {}
        ctx = {
            "engine": "vllm",
            "distributed": False,
            "nnodes": 1,
            "node_ips": "",
            "device_count": 8,
            "distributed_executor_backend": "dp_deployment",
        }
        _set_parallelism_params(params, ctx)
        self.assertEqual(params.get("tensor_parallel_size"), 8)

    # --- 不影响 Ray 全局 TP 路径 ---
    def test_ascend_non_deepseek_ray_uses_global_tp(self):
        """Ascend + 非 DeepSeek(如 Llama)是 Ray 后端,公式不变。"""
        params = {}
        ctx = {
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 2,
            "node_ips": "10.0.0.1,10.0.0.2",
            "device_count": 8,
            "distributed_executor_backend": "ray",
        }
        _set_parallelism_params(params, ctx)
        # 非 PD 分支:tp = device_count × nnodes = 16
        self.assertEqual(params.get("tensor_parallel_size"), 16)

    def test_nvidia_ray_multinode_uses_global_tp(self):
        """NVIDIA + Ray 多节点:全局 TP 公式不变。"""
        params = {}
        ctx = {
            "engine": "vllm",
            "distributed": True,
            "nnodes": 2,
            "node_ips": "10.0.0.1,10.0.0.2",
            "device_count": 4,
            "distributed_executor_backend": "ray",
        }
        _set_parallelism_params(params, ctx)
        self.assertEqual(params.get("tensor_parallel_size"), 8)

    # --- 单机非 distributed 场景 ---
    def test_ascend_single_node_non_distributed(self):
        """单机非 distributed:tp = device_count。"""
        params = {}
        ctx = {
            "engine": "vllm_ascend",
            "distributed": False,
            "nnodes": 1,
            "node_ips": "",
            "device_count": 8,
            "distributed_executor_backend": None,  # _handle_distributed 未触发
        }
        _set_parallelism_params(params, ctx)
        self.assertEqual(params.get("tensor_parallel_size"), 8)

    # --- 用户显式 tp 保护 ---
    def test_explicit_tp_preserved_on_ray_path(self):
        """Ray 路径下,用户预设的 tp 不被覆盖。"""
        params = {"tensor_parallel_size": 2}
        ctx = {
            "engine": "vllm",
            "distributed": True,
            "nnodes": 2,
            "node_ips": "10.0.0.1,10.0.0.2",
            "device_count": 4,
            "distributed_executor_backend": "ray",
        }
        _set_parallelism_params(params, ctx)
        self.assertEqual(params.get("tensor_parallel_size"), 2)

    # --- backend=None 兜底 ---
    def test_missing_backend_does_not_trigger_early_return(self):
        """distributed_executor_backend 缺失(Ascend PD 早返回场景):走原逻辑。"""
        params = {}
        ctx = {
            "engine": "vllm_ascend",
            "distributed": True,
            "nnodes": 1,
            "node_ips": "",
            "device_count": 8,
            # 不设 distributed_executor_backend
        }
        with patch("core.config_loader.get_pd_role_env", return_value="P"):
            _set_parallelism_params(params, ctx)
        # PD 分支:tp = device_count
        self.assertEqual(params.get("tensor_parallel_size"), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
