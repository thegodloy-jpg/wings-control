# -*- coding: utf-8 -*-
"""MindIE 分布式环境默认值单测。"""

import os
import sys
import tempfile
import unittest
import importlib
import json
from pathlib import Path
from unittest.mock import patch

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

mindie_adapter = importlib.import_module("engines.mindie_adapter")
_append_export_echoes = mindie_adapter._append_export_echoes
_build_distributed_env_commands = mindie_adapter._build_distributed_env_commands
_build_mindie_distributed_env_default_commands = mindie_adapter._build_mindie_distributed_env_default_commands
_build_model_config_overrides = mindie_adapter._build_model_config_overrides
_build_server_overrides = mindie_adapter._build_server_overrides


class TestMindieDistributedEnvDefaults(unittest.TestCase):
    def _extract_overrides_json(self, script: str):
        start = "cat > /tmp/_mindie_overrides.json << 'OVERRIDES_EOF'"
        lines = script.splitlines()
        start_idx = lines.index(start) + 1
        end_idx = lines.index("OVERRIDES_EOF", start_idx)
        return json.loads("\n".join(lines[start_idx:end_idx]))

    def _merge_mindie_config(self, overrides):
        template_path = ROOT / "wings_control" / "config" / "defaults" / "mindie_service_config.json"
        config = json.loads(template_path.read_text(encoding="utf-8"))
        for meta_key in ("_comment", "_usage"):
            config.pop(meta_key, None)

        config["ServerConfig"].update(overrides["server"])
        backend = config["BackendConfig"]
        backend.update(overrides["backend"])
        model_deploy = backend["ModelDeployConfig"]
        model_deploy.update(overrides["model_deploy"])
        model_deploy["ModelConfig"][0].update(overrides["model_config"])
        backend["ScheduleConfig"].update(overrides["schedule"])
        config.update(overrides.get("extra", {}))
        return config

    def _build_final_multinode_config(
        self,
        node_rank: int,
        *,
        model_name: str = "Qwen3-32B",
        device_count: int = 2,
        global_world_size: int = 4,
        is_moe: bool = False,
    ):
        engine_config = {
            "modelName": model_name,
            "modelWeightPath": "/usr/local/serving/models/",
            "worldSize": global_world_size,
            "mindie_model_type": "qwen3",
            "mindie_tool_call_parser": "qwen3",
        }
        if is_moe:
            engine_config["isMOE"] = True
        if node_rank == 0:
            engine_config.update({"ipAddress": "112.254.176.114", "port": 17000})
        params = {
            "engine_config": engine_config,
            "distributed": True,
            "nnodes": 2,
            "node_rank": node_rank,
            "device_count": device_count,
            "node_ips": "112.254.176.114,112.254.176.115",
            "worldSize": global_world_size,
            "mindie_master_addr": "112.254.176.114",
        }
        with patch(
            "engines.mindie_adapter._resolve_external_rank_table_path",
            return_value="/tmp/rank_table.json",
        ):
            with patch(
                "engines.mindie_adapter._resolve_rank_table",
                return_value=([], "/shared-volume/hccl_ranktable.json"),
            ):
                script = mindie_adapter.build_start_script(params)
        return self._merge_mindie_config(self._extract_overrides_json(script))

    def _write_script(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8")
        with tmp:
            tmp.write(content)
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.remove(tmp.name))
        return tmp.name

    def test_defaults_only_render_for_multinode_distributed(self):
        path = self._write_script('export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"\n')
        env = {"MINDIE_DISTRIBUTED_ENV_SCRIPT_PATH": path}
        with patch.dict(os.environ, env, clear=False):
            single = _build_mindie_distributed_env_default_commands({"distributed": False, "nnodes": 2})
            one_node = _build_mindie_distributed_env_default_commands({"distributed": True, "nnodes": 1})
            multi = _build_mindie_distributed_env_default_commands({"distributed": True, "nnodes": 2})

        self.assertEqual(single, [])
        self.assertEqual(one_node, [])
        self.assertIn('export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"', multi)

    def test_shell_defaults_preserve_runtime_env_override_expression(self):
        path = self._write_script('export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"\n')
        env = {"MINDIE_DISTRIBUTED_ENV_SCRIPT_PATH": path}
        with patch.dict(os.environ, env, clear=False):
            commands = _build_mindie_distributed_env_default_commands({"distributed": True, "nnodes": 2})

        self.assertIn('export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"', commands)

    def test_dynamic_topology_keys_are_not_rendered_from_defaults(self):
        path = self._write_script("\n".join([
            "export RANK=9",
            "export WORLD_SIZE=999",
            "export RANK_TABLE_FILE=/tmp/bad.json",
            "export ATB_LLM_HCCL_ENABLE=1",
            "unset HTTPS_PROXY",
        ]))
        env = {"MINDIE_DISTRIBUTED_ENV_SCRIPT_PATH": path}
        with patch.dict(os.environ, env, clear=False):
            commands = _build_mindie_distributed_env_default_commands({"distributed": True, "nnodes": 2})

        rendered = "\n".join(commands)
        self.assertIn("export ATB_LLM_HCCL_ENABLE=1", commands)
        self.assertNotIn("export RANK=9", rendered)
        self.assertNotIn("export WORLD_SIZE=999", rendered)
        self.assertNotIn("export RANK_TABLE_FILE=/tmp/bad.json", rendered)
        self.assertNotIn("unset HTTPS_PROXY", rendered)

    def test_static_hccl_timeouts_are_not_rendered_by_dynamic_block(self):
        params = {
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "device_count": 8,
            "node_ips": "10.0.0.1,10.0.0.2",
            "mindie_master_addr": "10.0.0.1",
            "worldSize": 16,
        }
        with patch("engines.mindie_adapter._resolve_external_rank_table_path", return_value="/tmp/rank_table.json"):
            with patch("engines.mindie_adapter._resolve_rank_table", return_value=([], "/shared-volume/hccl_ranktable.json")):
                commands = _build_distributed_env_commands(params)

        rendered = "\n".join(commands)
        self.assertNotIn("HCCL_CONNECT_TIMEOUT", rendered)
        self.assertNotIn("HCCL_EXEC_TIMEOUT", rendered)
        self.assertIn("export RANK_TABLE_FILE=/shared-volume/hccl_ranktable.json", rendered)

    def test_multinode_rank0_server_ip_uses_master_ip(self):
        overrides = _build_server_overrides(
            {"ipAddress": "10.0.0.99"},
            is_distributed=True,
            node_rank=0,
            nnodes=2,
            master_addr="10.0.0.1",
        )

        self.assertEqual(overrides["ipAddress"], "10.0.0.1")

    def test_multinode_worker_server_ip_uses_master_ip(self):
        overrides = _build_server_overrides(
            {"ipAddress": "10.0.0.99"},
            is_distributed=True,
            node_rank=1,
            nnodes=2,
            master_addr="10.0.0.1",
        )

        self.assertEqual(overrides["ipAddress"], "10.0.0.1")

    def test_mindie_server_port_defaults_to_backend_port(self):
        overrides = _build_server_overrides({}, is_distributed=True, node_rank=1, nnodes=2)

        self.assertEqual(overrides["port"], 17000)

    def test_mindie_function_call_skips_non_deepseek_v31(self):
        overrides = _build_model_config_overrides(
            {"mindie_model_type": "qwen3", "mindie_tool_call_parser": "qwen3"},
            is_distributed=True,
            world_size=2,
        )

        self.assertNotIn("models", overrides)

    def test_mindie_function_call_keeps_deepseek_v31(self):
        overrides = _build_model_config_overrides(
            {"mindie_model_type": "deepseekv2", "mindie_tool_call_parser": "deepseek_v31"},
            is_distributed=True,
            world_size=2,
        )

        self.assertEqual(
            overrides["models"],
            {"deepseekv2": {"tool_call_options": {"tool_call_parser": "deepseek_v31"}}},
        )

    def test_mindie_multinode_tp_is_global_world_size_and_dp_is_one(self):
        overrides = _build_model_config_overrides(
            {},
            is_distributed=True,
            world_size=2,
            global_world_size=4,
            nnodes=2,
        )

        self.assertEqual(overrides["worldSize"], 2)
        self.assertEqual(overrides["dp"], 1)
        self.assertEqual(overrides["tp"], 4)

    def test_mindie_multinode_tp_ignores_pp_and_uses_global_world_size(self):
        overrides = _build_model_config_overrides(
            {"pp": 2},
            is_distributed=True,
            world_size=8,
            global_world_size=8,
            nnodes=2,
        )

        self.assertEqual(overrides["dp"], 1)
        self.assertEqual(overrides["tp"], 8)

    def test_mindie_multinode_accepts_global_world_size_without_dp_divisibility(self):
        overrides = _build_model_config_overrides(
            {},
            is_distributed=True,
            world_size=2,
            global_world_size=3,
            nnodes=2,
        )

        self.assertEqual(overrides["worldSize"], 2)
        self.assertEqual(overrides["dp"], 1)
        self.assertEqual(overrides["tp"], 3)

    def test_mindie_multinode_dp_tp_override_explicit_parallel_values(self):
        overrides = _build_model_config_overrides(
            {"dp": 1, "tp": 4},
            is_distributed=True,
            world_size=2,
            global_world_size=4,
            nnodes=2,
        )

        self.assertEqual(overrides["worldSize"], 2)
        self.assertEqual(overrides["dp"], 1)
        self.assertEqual(overrides["tp"], 4)

    def test_qwen3_multinode_final_config_is_identical_on_two_nodes(self):
        node0 = self._build_final_multinode_config(node_rank=0)
        node1 = self._build_final_multinode_config(node_rank=1)

        self.assertEqual(node0, node1)
        self.assertEqual(node0["ServerConfig"]["ipAddress"], "112.254.176.114")
        self.assertEqual(node0["ServerConfig"]["port"], 17000)
        self.assertEqual(node0["BackendConfig"]["npuDeviceIds"], [[0, 1]])
        self.assertEqual(node0["BackendConfig"]["multiNodesInferEnabled"], True)

        model_config = node0["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]
        self.assertEqual(model_config["modelName"], "Qwen3-32B")
        self.assertEqual(model_config["modelWeightPath"], "/usr/local/serving/models/")
        self.assertEqual(model_config["worldSize"], 2)
        self.assertEqual(model_config["tp"], 4)
        self.assertEqual(model_config["dp"], 1)
        self.assertNotIn("models", model_config)

    def test_qwen3_multinode_2x8_final_config_is_identical_and_expected(self):
        node0 = self._build_final_multinode_config(
            node_rank=0,
            device_count=8,
            global_world_size=16,
        )
        node1 = self._build_final_multinode_config(
            node_rank=1,
            device_count=8,
            global_world_size=16,
        )

        self.assertEqual(node0, node1)
        self.assertEqual(node0["BackendConfig"]["npuDeviceIds"], [list(range(8))])

        model_config = node0["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]
        self.assertEqual(model_config["worldSize"], 8)
        self.assertEqual(model_config["tp"], 16)
        self.assertEqual(model_config["dp"], 1)
        self.assertNotIn("models", model_config)

    def test_qwen3_moe_multinode_2x8_final_config_includes_moe_parallel_fields(self):
        node0 = self._build_final_multinode_config(
            node_rank=0,
            model_name="Qwen3-30B-A3B",
            device_count=8,
            global_world_size=16,
            is_moe=True,
        )
        node1 = self._build_final_multinode_config(
            node_rank=1,
            model_name="Qwen3-30B-A3B",
            device_count=8,
            global_world_size=16,
            is_moe=True,
        )

        self.assertEqual(node0, node1)
        model_config = node0["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]
        self.assertEqual(model_config["modelName"], "Qwen3-30B-A3B")
        self.assertEqual(model_config["worldSize"], 8)
        self.assertEqual(model_config["tp"], 16)
        self.assertEqual(model_config["dp"], 1)
        self.assertEqual(model_config["moe_tp"], 8)
        self.assertEqual(model_config["moe_ep"], -1)
        self.assertNotIn("models", model_config)

    def test_export_commands_print_values_when_rendered(self):
        commands = _append_export_echoes([
            "export OMP_NUM_THREADS=10",
            "echo done",
        ])

        self.assertEqual(commands, [
            "export OMP_NUM_THREADS=10",
            "printf '[mindie-env] OMP_NUM_THREADS=%s\\n' \"${OMP_NUM_THREADS:-}\"",
            "echo done",
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
