# -*- coding: utf-8 -*-
"""快照测试基础设施。

用法：
  - 首次运行：自动在 tests/snapshots/ 下生成 .sh golden 文件
  - 后续运行：与 golden 文件逐字符对比
  - 更新快照：UPDATE_SNAPSHOTS=1 python -m pytest tests/test_script_snapshots.py
"""

from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.port_plan import PortPlan  # noqa: E402
from core.start_args_compat import LaunchArgs  # noqa: E402

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
UPDATE_SNAPSHOTS = os.getenv("UPDATE_SNAPSHOTS", "0") == "1"


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

class FakeModelIdentifier:
    """可配置的 ModelIdentifier stub，避免读取真实 config.json。"""

    def __init__(self, architecture="Qwen3ForCausalLM", model_type="llm",
                 model_name="test-model", model_path="/models/test",
                 model_quantize="bfloat16", num_hidden_layers=32):
        self.model_architecture = architecture
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model_quantize = model_quantize
        self.num_hidden_layers = num_hidden_layers
        self.config = {}

    def identify_model_type(self):
        return self.model_type

    def identify_model_architecture(self):
        return self.model_architecture

    def is_wings_supported(self):
        return True

    def identify_model_quantize(self):
        return self.model_quantize


def nvidia_hardware(device_count=8, memory_gb=80):
    """返回模拟 NVIDIA 硬件信息字典（与 detect_hardware() 输出格式一致）。"""
    return {
        "device": "nvidia",
        "count": device_count,
        "details": [
            {
                "device_id": i,
                "name": f"NVIDIA H800 {memory_gb}G",
                "total_memory": float(memory_gb),
                "free_memory": float(memory_gb),
                "used_memory": 0.0,
                "util": 0,
                "vendor": "Nvidia",
            }
            for i in range(device_count)
        ],
        "units": "GB",
    }


def ascend_hardware(device_count=8):
    """返回模拟 Ascend NPU 硬件信息字典（与 detect_hardware() 输出格式一致）。"""
    return {
        "device": "ascend",
        "count": device_count,
        "details": [
            {
                "device_id": i,
                "name": "Ascend910B",
                "total_memory": 64.0,
                "free_memory": 64.0,
                "used_memory": 0.0,
                "vendor": "Ascend",
            }
            for i in range(device_count)
        ],
        "units": "GB",
    }


def make_port_plan(proxy_port=18000, backend_port=17000,
                   health_port=19000, monitor_proxy_port=19100):
    return PortPlan(
        enable_proxy=True,
        backend_port=backend_port,
        proxy_port=proxy_port,
        health_port=health_port,
        monitor_proxy_port=monitor_proxy_port,
    )


def make_launch_args(**overrides) -> LaunchArgs:
    """构建 LaunchArgs，使用最小化默认值，允许按场景覆盖。"""
    defaults = dict(
        host="0.0.0.0",
        port=18000,
        model_name="test-model",
        model_path="/models/test",
        engine="vllm",
        input_length=4096,
        output_length=4096,
        config_file="",
        gpu_usage_mode="full",
        device_count=8,
        model_type="",
        save_path="",
        trust_remote_code=True,
        dtype="auto",
        kv_cache_dtype="auto",
        quantization="",
        quantization_param_path="",
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=False,
        block_size=16,
        max_num_seqs=32,
        seed=0,
        enable_expert_parallel=False,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        enable_speculative_decode=False,
        speculative_decode_model_path="",
        enable_rag_acc=False,
        enable_auto_tool_choice=False,
        enable_sparse=False,
        distributed=False,
        nnodes=1,
        node_rank=0,
        head_node_addr="",
        distributed_executor_backend="ray",
        node_ips="",
        nodes="",
        master_ip="",
        ray_head_ip="",
    )
    defaults.update(overrides)
    return LaunchArgs(**defaults)


# ---------------------------------------------------------------------------
# Snapshot mixin
# ---------------------------------------------------------------------------

class SnapshotTestMixin:
    """为 unittest.TestCase 提供快照对比和结构断言能力。"""

    # 子类可覆盖以过滤脚本中的易变行（如构建时间戳）
    _snapshot_filter_lines: list[str] = []

    def _filter_script(self, content: str) -> str:
        if not self._snapshot_filter_lines:
            return content
        result = []
        for line in content.splitlines(keepends=True):
            if not any(kw in line for kw in self._snapshot_filter_lines):
                result.append(line)
        return "".join(result)

    def assert_snapshot(self, name: str, content: str):
        """与 golden 文件对比。golden 文件不存在时自动创建。"""
        SNAPSHOTS_DIR.mkdir(exist_ok=True)
        snapshot_path = SNAPSHOTS_DIR / f"{name}.sh"
        filtered = self._filter_script(content)

        if UPDATE_SNAPSHOTS or not snapshot_path.exists():
            snapshot_path.write_text(filtered, encoding="utf-8")
            print(f"\n  [SNAP] {'Updated' if snapshot_path.exists() else 'Created'}: {snapshot_path.name}")
        else:
            expected = snapshot_path.read_text(encoding="utf-8")
            if filtered != expected:
                diff = "\n".join(difflib.unified_diff(
                    expected.splitlines(),
                    filtered.splitlines(),
                    fromfile=f"golden/{name}.sh",
                    tofile=f"actual/{name}.sh",
                    lineterm="",
                    n=5,
                ))
                self.fail(  # type: ignore[attr-defined]
                    f"Snapshot mismatch for '{name}':\n{diff}\n\n"
                    f"Regenerate: UPDATE_SNAPSHOTS=1 python -m pytest tests/test_script_snapshots.py"
                )

    def assert_script_structure(
        self,
        command: str,
        *,
        engine_cmd: str,
        model_path: str,
        expected_params: tuple = (),
        absent_params: tuple = (),
    ):
        """基础结构断言：每个场景必须满足的不变式。"""
        self.assertTrue(  # type: ignore[attr-defined]
            command.startswith("#!/bin/bash") or command.startswith("#!/usr/bin/env bash"),
            "缺少 shebang"
        )
        self.assertIn("set -euo pipefail", command, "缺少 strict mode")  # type: ignore[attr-defined]
        self.assertIn(engine_cmd, command, f"缺少引擎命令: {engine_cmd}")  # type: ignore[attr-defined]
        self.assertIn(model_path, command, f"缺少模型路径: {model_path}")  # type: ignore[attr-defined]
        for param in expected_params:
            self.assertIn(param, command, f"期望参数缺失: {param}")  # type: ignore[attr-defined]
        for param in absent_params:
            self.assertNotIn(param, command, f"不应出现的参数: {param}")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Context manager: 标准 mock 组合
# ---------------------------------------------------------------------------

class ScriptGenContext:
    """封装快照测试所需的标准 mock 组合，用于 with 语句。

    mock 列表：
      - detect_hardware        → 返回固定 hardware dict
      - ModelIdentifier        → 返回 FakeModelIdentifier
      - _build_env_overrides_preamble → 返回 "" 以确保确定性
      - _build_accel_preamble  → 可选，默认透传（允许测试 accel 逻辑）
    """

    def __init__(
        self,
        hardware: dict,
        fake_model: FakeModelIdentifier,
        env_overrides: dict | None = None,
        mock_accel: bool = True,
    ):
        self._hardware = hardware
        self._fake_model = fake_model
        self._env_overrides = env_overrides or {}
        self._mock_accel = mock_accel
        self._patches: list = []

    def __enter__(self):
        env = {
            # 清除会影响脚本生成的关键环境变量
            "ENGINE_VERSION": "",
            "WINGS_H20_MODEL": "",
            "WINGS_ENGINE_PATCH_OPTIONS": "",
            "ENGINE_PORT": "17000",
            "LMCACHE_USE_NIXCACHE": "",
            "LMCACHE_OFFLOAD": "",
            # 固定 POD_IP，避免快照中出现机器相关 IP
            "POD_IP": "10.0.0.1",
            **self._env_overrides,
        }

        fake_model = self._fake_model

        def _make_fake_model(name, path, mtype):
            return fake_model

        p_hw = patch("core.wings_entry.detect_hardware", return_value=self._hardware)
        p_mi = patch("core.config_loader.ModelIdentifier", side_effect=_make_fake_model)
        p_env = patch("core.wings_entry._build_env_overrides_preamble", return_value="")
        p_os = patch.dict(os.environ, env, clear=False)
        # 固定 get_local_ip 返回值，确保脚本中的 host/IP 不随机器变化
        p_ip1 = patch("engines.vllm_adapter.get_local_ip", return_value="10.0.0.1")
        p_ip2 = patch("core.config_loader.get_local_ip", return_value="10.0.0.1")
        # MindIE 要求 config.json 存在并检查权限；测试中跳过实际文件检查
        p_perm = patch("core.config_loader.check_permission_640", return_value=True)
        # MindIE 分布式要求 rank_table 文件存在；测试中跳过文件存在性校验
        p_rank = patch(
            "engines.mindie_adapter._resolve_external_rank_table_path",
            return_value="",
        )

        self._patches = [p_hw, p_mi, p_env, p_os, p_ip1, p_ip2, p_perm, p_rank]
        if self._mock_accel:
            self._patches.append(
                patch("core.wings_entry._build_accel_preamble", return_value="")
            )

        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.stop()
