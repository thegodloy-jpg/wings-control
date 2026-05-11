# -*- coding: utf-8 -*-
"""UT-BASH：Shell 脚本语法验证测试。

通过 `bash -n` 验证所有快照文件语法合法，对应 TEST_PLAN.md GAP-6。

运行：
  python -m pytest tests/test_unit_bash_syntax.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"


def _bash_available() -> bool:
    return shutil.which("bash") is not None


@unittest.skipUnless(_bash_available(), "bash 不可用（跳过语法验证）")
class TestSnapshotBashSyntax(unittest.TestCase):
    """验证所有 golden 快照脚本的 bash 语法合法性（bash -n）。"""

    @classmethod
    def setUpClass(cls):
        cls.snapshot_files = sorted(SNAPSHOTS_DIR.glob("*.sh"))

    def _check_syntax(self, path: Path):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"bash -n 语法检查失败: {path.name}\n"
            f"stderr: {result.stderr}"
        )

    def test_snapshots_directory_has_files(self):
        """确认 snapshots/ 目录非空，防止误报全部通过。"""
        self.assertGreater(
            len(self.snapshot_files), 0,
            "snapshots/ 目录为空，请先运行 test_script_snapshots.py 生成 golden 文件"
        )

    def test_snap01_vllm_nvidia_qwen3_single_syntax(self):
        p = SNAPSHOTS_DIR / "snap01_vllm_nvidia_qwen3_single.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap02_vllm_nvidia_deepseek_distributed_syntax(self):
        p = SNAPSHOTS_DIR / "snap02_vllm_nvidia_deepseek_distributed.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap03_vllm_ascend_kimik25_single_syntax(self):
        p = SNAPSHOTS_DIR / "snap03_vllm_ascend_kimik25_single.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap04_mindie_ascend_qwen3moe_single_syntax(self):
        p = SNAPSHOTS_DIR / "snap04_mindie_ascend_qwen3moe_single.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap05_sglang_nvidia_embedding_syntax(self):
        p = SNAPSHOTS_DIR / "snap05_sglang_nvidia_embedding.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap06_vllm_nvidia_speculative_syntax(self):
        p = SNAPSHOTS_DIR / "snap06_vllm_nvidia_speculative.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap07_vllm_ascend_glmmoe_dsa_syntax(self):
        p = SNAPSHOTS_DIR / "snap07_vllm_ascend_glmmoe_dsa.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap08_vllm_nvidia_env_overrides_syntax(self):
        p = SNAPSHOTS_DIR / "snap08_vllm_nvidia_env_overrides.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap09_vllm_ascend_deepseek_rag_syntax(self):
        p = SNAPSHOTS_DIR / "snap09_vllm_ascend_deepseek_rag.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")

    def test_snap10_vllm_nvidia_unknown_arch_syntax(self):
        p = SNAPSHOTS_DIR / "snap10_vllm_nvidia_unknown_arch.sh"
        if p.exists():
            self._check_syntax(p)
        else:
            self.skipTest(f"快照不存在: {p.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
