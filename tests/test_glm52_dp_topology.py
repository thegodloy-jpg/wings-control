# -*- coding: utf-8 -*-
"""GLM-5.2 TP/DP 拓扑回归 —— 杜绝单机 TP 超订(over-subscription)再次出现。

历史 bug(commit 0c906eb 修复):GLM-5.2 单机(nnodes==1)生成
``--tensor-parallel-size device_count --data-parallel-size 2``,即 TP×DP=device_count×2,
在一个只有 device_count 卡的节点上请求 2×device_count 卡(16 卡 → 请求 32)。根因是
``config_loader._set_parallelism_params`` 对单机非-dp_deployment 路径把 TP 钉成
device_count,下游 ``vllm_adapter`` 的 ``_set_if_not_explicit`` 只填空值、覆盖不掉。

本测试**端到端**(parse_launch_args → load_and_merge_configs → build_start_script)断言:
  * 核心不变式:单机 ``TP × DP == device_count``(精确占满本节点,绝不超订);
  * GLM-5.2 单机配方:``TP == device_count//2`` 且 ``DP == 2``;
  * 双机仍是每节点整卡 TP(``TP == device_count``,``DP_local == 1``);
  * GLM-5.1(对照)单机不减半(``TP == device_count``),证明 carve-out 仅作用于 5.2。
"""
# pyright: reportMissingImports=false

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.config_loader import load_and_merge_configs  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines.vllm_adapter import build_start_script  # noqa: E402

_GLM52_NAME = "GLM-5.2-w8a8"
_GLM51_NAME = "glm-5.1-chat"
_ARCH = "GlmMoeDsaForCausalLM"


def _gen_script(model_name, device_count, *, nnodes=1, platform="a3", argv_extra=None, extra_env=None):
    """跑完整启动链路,返回生成的 start_command.sh 文本。"""
    with tempfile.TemporaryDirectory() as _raw_dir:
        # Windows 临时路径含反斜杠(C:\Users → \U)会被下游 dp_deployment 脚本的 re.sub 当转义符。
        # 生产为 POSIX 路径,这里与 dry_run 一致做正斜杠归一,纯属测试环境处理。
        model_dir = _raw_dir.replace("\\", "/")
        Path(model_dir, "config.json").write_text(
            json.dumps({
                "architectures": [_ARCH],
                "quantization_config": {"quant_method": "ascend"},
            }),
            encoding="utf-8",
        )
        argv = [
            "--engine", "vllm_ascend",
            "--model-name", model_name,
            "--model-path", model_dir,
            "--host", "0.0.0.0",
            "--port", "18000",
            "--device-count", str(device_count),
            "--nnodes", str(nnodes),
            "--node-rank", "0",
            "--trust-remote-code",
        ]
        argv += (argv_extra or [])
        env = {"WINGS_DEVICE": "ascend", "WINGS_ASCEND_PLATFORM": platform}
        env.update(extra_env or {})
        with patch.object(sys, "argv", ["wings-launcher"] + argv):
            with patch.dict(os.environ, env, clear=True):
                launch_args = parse_launch_args(argv).to_namespace()
                merged = load_and_merge_configs(
                    {"device": "ascend", "count": device_count, "details": []},
                    launch_args,
                )
                return build_start_script(merged)


def _flag_int(script, flag):
    m = re.search(rf"--{flag}\s+(\d+)", script)
    return int(m.group(1)) if m else None


class Glm52SingleNodeTopologyTest(unittest.TestCase):
    """单机:TP=device_count//2 / DP=2 / TP×DP==device_count(无超订)。"""

    def test_single_node_matrix_no_oversubscription(self):
        for device_count in (8, 16):
            for platform in ("a2", "a3"):
                with self.subTest(device_count=device_count, platform=platform):
                    script = _gen_script(_GLM52_NAME, device_count, nnodes=1, platform=platform)
                    tp = _flag_int(script, "tensor-parallel-size")
                    dp = _flag_int(script, "data-parallel-size")
                    self.assertIsNotNone(tp, "缺少 --tensor-parallel-size")
                    self.assertIsNotNone(dp, "缺少 --data-parallel-size")
                    # 配方
                    self.assertEqual(tp, device_count // 2,
                                     f"GLM-5.2 单机 TP 应为 device_count//2={device_count//2}")
                    self.assertEqual(dp, 2, "GLM-5.2 单机 DP 应为 2")
                    # ★核心不变式:精确占满本节点,绝不超订(历史 bug 是 TP×DP=2×device_count)
                    self.assertEqual(
                        tp * dp, device_count,
                        f"单机 TP×DP={tp*dp} 必须 == device_count={device_count}（否则超订）",
                    )

    def test_single_node_emits_dp2_recipe(self):
        """显式锁住 16 卡单机的最终命令片段(用户报障原型)。"""
        script = _gen_script(_GLM52_NAME, 16, nnodes=1, platform="a3")
        self.assertIn("--tensor-parallel-size 8", script)
        self.assertIn("--data-parallel-size 2", script)
        self.assertNotIn("--tensor-parallel-size 16", script)


class Glm52DualNodeTopologyTest(unittest.TestCase):
    """双机:每节点整卡 TP、DP_local=1、全局 DP=nnodes(不进单机 carve-out)。"""

    def _dual_script(self, device_count):
        return _gen_script(
            _GLM52_NAME, device_count, nnodes=2, platform="a3",
            argv_extra=[
                "--distributed",
                "--distributed-executor-backend", "dp_deployment",
                "--head-node-addr", "10.0.0.1",
            ],
            extra_env={
                "NODE_IPS": "10.0.0.1,10.0.0.2",
                "RANK_IP": "10.0.0.1",
                "MASTER_IP": "10.0.0.1",
                "POD_IP": "10.0.0.1",
            },
        )

    def test_dual_node_full_node_tp(self):
        for device_count in (8, 16):
            with self.subTest(device_count=device_count):
                script = self._dual_script(device_count)
                tp = _flag_int(script, "tensor-parallel-size")
                dp_local = _flag_int(script, "data-parallel-size-local")
                dp = _flag_int(script, "data-parallel-size")
                self.assertEqual(tp, device_count, "双机每节点应整卡 TP=device_count")
                self.assertEqual(dp_local, 1, "双机 DP_local 应为 1(每节点 1 副本)")
                self.assertEqual(dp, 2, "双机全局 DP 应为 nnodes=2")
                # 本节点占用 = TP × DP_local 必须 == device_count(每节点也不超订)
                self.assertEqual(tp * dp_local, device_count,
                                 f"双机本节点 TP×DP_local={tp*dp_local} 必须 == {device_count}")


class Glm52EnvTest(unittest.TestCase):
    """GLM-5.2 专属 env:单/双机均须 ``export VLLM_VERSION=0.21.0``;GLM-5.1 不得注入。"""

    def test_glm52_single_pins_vllm_version(self):
        for platform in ("a2", "a3"):
            with self.subTest(platform=platform):
                script = _gen_script(_GLM52_NAME, 16, nnodes=1, platform=platform)
                self.assertIn("export VLLM_VERSION=0.21.0", script)

    def test_glm52_dual_pins_vllm_version(self):
        script = _gen_script(
            _GLM52_NAME, 16, nnodes=2, platform="a3",
            argv_extra=[
                "--distributed",
                "--distributed-executor-backend", "dp_deployment",
                "--head-node-addr", "10.0.0.1",
            ],
            extra_env={
                "NODE_IPS": "10.0.0.1,10.0.0.2", "RANK_IP": "10.0.0.1",
                "MASTER_IP": "10.0.0.1", "POD_IP": "10.0.0.1",
            },
        )
        self.assertIn("export VLLM_VERSION=0.21.0", script)

    def test_glm51_does_not_pin_vllm_version(self):
        script = _gen_script(_GLM51_NAME, 16, nnodes=1, platform="a3")
        self.assertNotIn("VLLM_VERSION", script)


class Glm51ControlTest(unittest.TestCase):
    """对照:GLM-5.1 单机【不】减半,证明 carve-out 仅命中 GLM-5.2。"""

    def test_glm51_single_node_not_halved(self):
        for device_count in (8, 16):
            with self.subTest(device_count=device_count):
                script = _gen_script(_GLM51_NAME, device_count, nnodes=1, platform="a3")
                tp = _flag_int(script, "tensor-parallel-size")
                self.assertEqual(tp, device_count,
                                 "GLM-5.1 单机 TP 应等于 device_count（不进 5.2 减半分支）")
                self.assertNotIn("--data-parallel-size 2", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
