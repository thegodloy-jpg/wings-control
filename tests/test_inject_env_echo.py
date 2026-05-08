# -*- coding: utf-8 -*-
"""_inject_env_echo 行为单测。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from engines.vllm_adapter import _inject_env_echo  # noqa: E402
import engines.vllm_adapter as vllm_adapter  # noqa: E402
import engines.sglang_adapter as sglang_adapter  # noqa: E402


class TestInjectEnvEcho(unittest.TestCase):
    def test_all_exported_variables_are_echoed(self):
        script = (
            "export LD_LIBRARY_PATH=/tmp/lib:${LD_LIBRARY_PATH:-}\n"
            "export OMP_NUM_THREADS=8\n"
            "export HCCL_BUFFSIZE=512\n"
        )

        rendered = _inject_env_echo(script)

        self.assertIn(
            'echo "[wings-env] export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"\n',
            rendered,
        )
        self.assertIn(
            'echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"\n',
            rendered,
        )
        self.assertIn(
            'echo "[wings-env] export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"\n',
            rendered,
        )

    def test_existing_export_echo_is_not_duplicated(self):
        script = (
            "export OMP_NUM_THREADS=10\n"
            'echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"\n'
            "export HCCL_CONNECT_TIMEOUT=7200\n"
            "printf '[mindie-env] HCCL_CONNECT_TIMEOUT=%s\\n' \"${HCCL_CONNECT_TIMEOUT:-}\"\n"
        )

        rendered = _inject_env_echo(script)

        self.assertEqual(rendered.count("[wings-env] export OMP_NUM_THREADS="), 1)
        self.assertEqual(rendered.count("[mindie-env] HCCL_CONNECT_TIMEOUT="), 1)
        self.assertNotIn("[wings-env] export HCCL_CONNECT_TIMEOUT=", rendered)

    def test_indented_export_inside_if_block_is_echoed(self):
        """if/elif/else 块内的缩进 export 也应被注入 [wings-env] echo（模拟 mindie HCCL_IF_IP 场景）。"""
        script = (
            'if [ -n "${HCCL_IF_IP:-}" ]; then\n'
            '    export HCCL_IF_IP="${HCCL_IF_IP}"\n'
            'elif [ -n "${HCCL_DEVICE_IPS:-}" ]; then\n'
            '    export HCCL_IF_IP=$(hostname -i | awk \'{print $1}\')\n'
            'fi\n'
        )
        rendered = _inject_env_echo(script)
        # 每个分支里的 export 都应追加 echo
        self.assertEqual(rendered.count('[wings-env] export HCCL_IF_IP='), 2,
                         "两个分支的 export 均应被注入一行 echo")
        # echo 行应保留与 export 相同的缩进（4 空格）
        self.assertIn('    echo "[wings-env] export HCCL_IF_IP=', rendered)

    def test_existing_command_echo_is_not_duplicated(self):
        script = (
            "echo '[wings-cmd] >>> exec python3 -m vllm.entrypoints.openai.api_server'\n"
            "exec python3 -m vllm.entrypoints.openai.api_server\n"
        )

        rendered = _inject_env_echo(script)

        self.assertEqual(rendered.count("[wings-cmd] >>>"), 1)


class _FakeModelInfo:
    """轻量 ModelIdentifier 存根，暴露 arch 和 model_architecture 属性。"""
    def __init__(self, arch: str):
        self.arch = arch
        self.model_architecture = arch
        self.draft_arch = None


class TestVllmNonAscendStandaloneEcho(unittest.TestCase):
    """vllm (非 ascend) 适配器独立调用时也应注入环境变量和命令 echo。"""

    def _build_script(self, engine: str = "vllm"):
        params = {
            "engine": engine,
            "engine_config": {
                "model": "/models/qwen3",
                "port": 8000,
                "tensor_parallel_size": 1,
                "host": "0.0.0.0",
            },
        }
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("Qwen3ForCausalLM"),
        ):
            return vllm_adapter.build_start_script(params)

    def test_vllm_engine_script_injects_cmd_echo_standalone(self):
        """vllm (非 ascend) build_start_script 不经 wings_entry 也应包含 [wings-cmd] echo。"""
        script = self._build_script("vllm")
        self.assertIn("[wings-cmd] >>>", script)
        self.assertIn("exec python3 -m vllm.entrypoints.openai.api_server", script)

    def test_vllm_ascend_and_vllm_behave_consistently(self):
        """vllm 与 vllm_ascend 的 [wings-cmd] echo 注入行为应一致。"""
        script_vllm = self._build_script("vllm")
        script_ascend = self._build_script("vllm_ascend")
        self.assertIn("[wings-cmd] >>>", script_vllm)
        self.assertIn("[wings-cmd] >>>", script_ascend)

    def test_vllm_with_lmcache_echoes_env_vars(self):
        """开启 LMCache 时 vllm 脚本应包含 [wings-env] export PYTHONHASHSEED= echo。"""
        params = {
            "engine": "vllm",
            "engine_config": {
                "model": "/models/qwen3",
                "port": 8000,
                "tensor_parallel_size": 1,
                "host": "0.0.0.0",
            },
        }
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("Qwen3ForCausalLM"),
        ), patch.dict("os.environ", {"LMCACHE_OFFLOAD": "true"}, clear=False):
            script = vllm_adapter.build_start_script(params)
        self.assertIn("[wings-env] export PYTHONHASHSEED=", script)
        self.assertIn("[wings-cmd] >>>", script)

    def test_wings_entry_second_pass_does_not_duplicate_cmd_echo(self):
        """wings_entry 对已注入 [wings-cmd] echo 的脚本再次调用 _inject_env_echo 不产生重复行。"""
        script = self._build_script("vllm")
        second_pass = _inject_env_echo(script)
        self.assertEqual(second_pass.count("[wings-cmd] >>>"), 1)


class TestSglangEcho(unittest.TestCase):
    """sglang 适配器 build_start_script 中的环境变量和启动命令 echo 行为。"""

    def test_sglang_single_node_has_cmd_echo(self):
        """单机模式下 sglang 脚本包含 [wings-cmd] >>> exec python3 ... 行。"""
        params = {
            "distributed": False,
            "nnodes": 1,
            "engine_config": {
                "model": "/models/qwen3",
                "port": 8000,
                "tp_size": 1,
                "host": "0.0.0.0",
            },
        }
        script = sglang_adapter.build_start_script(params)
        self.assertIn("[wings-cmd] >>>", script)
        self.assertIn("exec python3 -m sglang.launch_server", script)

    def test_sglang_distributed_has_env_echo(self):
        """多节点分布式模式下 sglang 脚本对三个通信接口变量都有 [wings-env] echo。"""
        params = {
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "head_node_addr": "192.168.1.1",
            "engine_config": {
                "model": "/models/qwen3",
                "port": 8000,
                "host": "0.0.0.0",
            },
        }
        with patch.dict("os.environ", {"NETWORK_INTERFACE": "bond0"}, clear=False):
            script = sglang_adapter.build_start_script(params)

        self.assertIn("[wings-env] export GLOO_SOCKET_IFNAME=", script)
        self.assertIn("[wings-env] export TP_SOCKET_IFNAME=", script)
        self.assertIn("[wings-env] export NCCL_SOCKET_IFNAME=", script)
        self.assertIn("[wings-cmd] >>>", script)


class TestMindieCommandEcho(unittest.TestCase):
    """_inject_env_echo 对 MindIE 守护进程命令的 echo 注入行为。"""

    def test_mindie_daemon_background_gets_cmd_echo(self):
        """后台运行的 ./bin/mindieservice_daemon & 应被注入 [wings-cmd] echo。"""
        script = (
            "cd /usr/local/Ascend/mindie/latest/mindie-service\n"
            "./bin/mindieservice_daemon &\n"
            "MINDIE_PID=$!\n"
        )
        rendered = _inject_env_echo(script)
        self.assertIn("[wings-cmd] >>> ./bin/mindieservice_daemon", rendered)

    def test_mindie_exec_daemon_gets_cmd_echo(self):
        """exec ./bin/mindieservice_daemon（单行 exec 形式）应被注入 [wings-cmd] echo。"""
        script = "exec ./bin/mindieservice_daemon\n"
        rendered = _inject_env_echo(script)
        self.assertIn("[wings-cmd] >>> exec ./bin/mindieservice_daemon", rendered)

    def test_shell_script_dot_slash_sh_still_echoed(self):
        """原有的 ./set_env.sh 形式（带 .sh）同样应被注入。"""
        script = "./set_env.sh\n"
        rendered = _inject_env_echo(script)
        self.assertIn("[wings-cmd] >>> ./set_env.sh", rendered)

    def test_no_duplicate_when_echo_already_present(self):
        """已有 [wings-cmd] echo 的行不应被重复注入。"""
        script = (
            "echo '[wings-cmd] >>> ./bin/mindieservice_daemon'\n"
            "./bin/mindieservice_daemon &\n"
        )
        rendered = _inject_env_echo(script)
        self.assertEqual(rendered.count("[wings-cmd] >>>"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
