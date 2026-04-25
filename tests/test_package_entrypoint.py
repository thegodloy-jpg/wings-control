# -*- coding: utf-8 -*-
"""包级入口 (`python -m wings_control`) 兼容性测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]  # wings-control/
if str(ROOT / "wings_control") not in sys.path:
    sys.path.append(str(ROOT / "wings_control"))


def _load_package_entry_module():
    """Load the package ``__init__`` under an isolated module name."""
    init_path = ROOT / "wings_control" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "wings_control_pkg_entry_test",
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPackageEntrypoint(unittest.TestCase):
    def test_entry_module_path_resolves_to_launcher_file(self):
        wings_control_pkg = _load_package_entry_module()
        entry_path = wings_control_pkg._resolve_entry_module_path()
        self.assertEqual(entry_path.name, "wings_control.py")
        self.assertTrue(entry_path.is_file())

    def test_package_run_delegates_to_loaded_entry_module(self):
        wings_control_pkg = _load_package_entry_module()
        fake_entry = SimpleNamespace(run=Mock(return_value=7))
        with patch.object(wings_control_pkg, "_load_entry_module", return_value=fake_entry):
            result = wings_control_pkg.run(["--engine", "vllm"])

        self.assertEqual(result, 7)
        fake_entry.run.assert_called_once_with(["--engine", "vllm"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
