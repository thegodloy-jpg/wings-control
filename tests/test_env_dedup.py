# -*- coding: utf-8 -*-
"""UT：环境变量去重 _dedupe_env_exports / _classify_env_export。

校验多 builder 重复导出同名变量时，去重后每个变量只剩一条 export 且最终值不变；
累加型（LD_LIBRARY_PATH）、缩进块内导出、单次出现一律不动。

运行：
  python -m pytest tests/test_env_dedup.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from utils.shell_env_utils import (  # noqa: E402
    classify_env_export as _classify_env_export,
    dedupe_env_exports as _dedupe_env_exports,
)


class TestClassifyEnvExport(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(_classify_env_export("HCCL_BUFFSIZE", "512"), "plain")

    def test_soft_default(self):
        self.assertEqual(
            _classify_env_export("HCCL_BUFFSIZE", "${HCCL_BUFFSIZE:-1024}"), "soft_default"
        )
        self.assertEqual(_classify_env_export("FOO", "${FOO}"), "soft_default")

    def test_accumulation(self):
        # 自身被嵌在更大的串里（追加路径）→ 累加，绝不能去重
        self.assertEqual(
            _classify_env_export("LD_LIBRARY_PATH", '"/a:/b:${LD_LIBRARY_PATH:-}"'),
            "accumulation",
        )
        self.assertEqual(_classify_env_export("PATH", "$PATH:/x"), "accumulation")

    def test_other_var_reference_is_plain(self):
        # 引用别的变量不算自引用
        self.assertEqual(_classify_env_export("A", "$B/x"), "plain")


class TestDedupeEnvExports(unittest.TestCase):
    def test_last_plain_wins_and_softdefault_dropped(self):
        cmds = [
            "export HCCL_BUFFSIZE=1024",          # base
            "export HCCL_BUFFSIZE=512",           # arch（最后一个纯赋值 → 生效）
            "export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}",  # forced 软默认 → 丢
        ]
        out = _dedupe_env_exports(cmds)
        self.assertEqual(out, ["export HCCL_BUFFSIZE=512"])

    def test_accumulation_preserved(self):
        cmds = [
            'export LD_LIBRARY_PATH="/a:${LD_LIBRARY_PATH:-}"',
            'export LD_LIBRARY_PATH="/b:${LD_LIBRARY_PATH:-}"',
        ]
        # 两条都是累加 → 全部保留
        self.assertEqual(_dedupe_env_exports(cmds), cmds)

    def test_indented_block_exports_untouched(self):
        cmds = [
            "if [ -d /x ]; then",
            "    export FOO=1",   # 缩进（块内）→ 不动
            "else",
            "    export FOO=2",
            "fi",
            "export FOO=9",       # 顶层单次 → 保留（与块内不冲突）
        ]
        self.assertEqual(_dedupe_env_exports(cmds), cmds)

    def test_only_soft_default_keeps_first(self):
        cmds = [
            "export Y=${Y:-1}",
            "export Y=${Y:-2}",
        ]
        self.assertEqual(_dedupe_env_exports(cmds), ["export Y=${Y:-1}"])

    def test_single_occurrence_and_nonexport_untouched(self):
        cmds = [
            "set +u",
            "export ONLY=7",
            "# comment",
            "source /x/set_env.sh",
        ]
        self.assertEqual(_dedupe_env_exports(cmds), cmds)

    def test_realistic_three_block_sequence(self):
        cmds = [
            "# base",
            "export HCCL_BUFFSIZE=1024",
            "export OMP_NUM_THREADS=1",
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
            "# arch",
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
            "export HCCL_BUFFSIZE=512",
            "export OMP_NUM_THREADS=1",
            "# forced",
            "export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}",
            "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
        ]
        out = _dedupe_env_exports(cmds)
        # 每个变量恰好一条
        self.assertEqual(out.count("export HCCL_BUFFSIZE=512"), 1)
        self.assertEqual(len([l for l in out if l.startswith("export HCCL_BUFFSIZE=")]), 1)
        self.assertEqual(len([l for l in out if l.startswith("export OMP_NUM_THREADS=")]), 1)
        self.assertEqual(len([l for l in out if l.startswith("export PYTORCH_NPU_ALLOC_CONF=")]), 1)
        # 最终值正确（arch 的 512 / 1 / True 生效）
        self.assertIn("export HCCL_BUFFSIZE=512", out)
        self.assertIn("export OMP_NUM_THREADS=1", out)
        self.assertIn("export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True", out)
        self.assertNotIn("export HCCL_BUFFSIZE=1024", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
