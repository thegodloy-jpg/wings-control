# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6 Phase A recipe shadow loader tests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.recipe_loader import (  # noqa: E402
    build_recipe_shadow_diff,
    find_recipe_candidates,
    load_structured_file,
    select_best_recipe,
    validate_recipe_document,
    write_recipe_shadow_diff,
    RecipeRuntimeContext,
)


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class TestV36RecipeLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "config"
        self.model_dir = Path(self.tmp.name) / "models" / "Qwen3-32B"
        self.model_dir.mkdir(parents=True)
        _write_json(self.model_dir / "config.json", {"architectures": ["Qwen3ForCausalLM"]})
        _write_json(
            self.root / "recipes" / "architectures" / "qwen3.yaml",
            {
                "architecture": "Qwen3ForCausalLM",
                "applies_to_versions": {"vllm": ">=0.10.0"},
                "defaults": {"tool_call_parser": "hermes"},
                "hardware_overlays": [
                    {"applies_to_hardware": ["h20-141"], "defaults": {}},
                ],
            },
        )
        _write_json(
            self.root / "recipes" / "models" / "Qwen3-32B.yaml",
            {
                "model_id": "Qwen3-32B",
                "applies_to_versions": {"vllm_ascend": ">=0.10.0"},
                "matches": {"model_paths": ["Qwen3-32B"], "model_names": ["Qwen3-32B"]},
                "defaults": {"max_num_seqs": 32},
                "hardware_overlays": [
                    {"applies_to_hardware": ["910b-32"], "defaults": {}},
                ],
            },
        )
        _write_json(
            self.root / "recipes" / "models" / "_experimental" / "Qwen3-32B-910c.yaml",
            {
                "model_id": "Qwen3-32B-910C",
                "matches": {"model_paths": ["Qwen3-32B"], "model_names": ["Qwen3-32B"]},
                "defaults": {"max_num_seqs": 128},
                "hardware_overlays": [
                    {"applies_to_hardware": ["910c"], "defaults": {}},
                ],
            },
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_scan_excludes_experimental(self):
        ctx = RecipeRuntimeContext(
            engine="vllm_ascend",
            version="0.10.0",
            chip_id="910c",
            model_name="Qwen3-32B",
            model_path=str(self.model_dir),
            architecture="Qwen3ForCausalLM",
            allow_experimental=False,
        )
        _, model_candidates = find_recipe_candidates(ctx, config_root=self.root)
        selected = select_best_recipe(model_candidates).selected
        self.assertIsNotNone(selected)
        self.assertFalse(selected.experimental)
        self.assertEqual(selected.defaults["max_num_seqs"], 32)
        self.assertIn("specificity", selected.score_breakdown)

    def test_allow_experimental_can_win_on_exact_hardware(self):
        ctx = RecipeRuntimeContext(
            engine="vllm_ascend",
            version="0.10.0",
            chip_id="910c",
            model_name="Qwen3-32B",
            model_path=str(self.model_dir),
            architecture="Qwen3ForCausalLM",
            allow_experimental=True,
        )
        _, model_candidates = find_recipe_candidates(ctx, config_root=self.root)
        selected = select_best_recipe(model_candidates).selected
        self.assertIsNotNone(selected)
        self.assertTrue(selected.experimental)
        self.assertEqual(selected.defaults["max_num_seqs"], 128)
        self.assertEqual(selected.score_breakdown["hardware_match"], 100.0)

    def test_shadow_diff_does_not_mutate_merged_params(self):
        args = SimpleNamespace(
            engine="vllm_ascend",
            model_name="Qwen3-32B",
            model_path=str(self.model_dir),
            allow_experimental=False,
        )
        merged = {"engine": "vllm_ascend", "engine_version": "0.10.0", "max_num_seqs": 8}
        with patch.dict(os.environ, {}, clear=True):
            shadow = build_recipe_shadow_diff(args, {"chip_id": "910b-32"}, merged, config_root=self.root)
        self.assertEqual(merged["max_num_seqs"], 8)
        self.assertEqual(shadow["effective_mode"], "shadow")
        self.assertEqual(shadow["summary"]["selected_count"], 2)
        self.assertEqual(shadow["summary"]["would_change_count"], 2)
        self.assertTrue(shadow["diff"]["max_num_seqs"]["would_change"])
        self.assertEqual(shadow["proposed_params"]["max_num_seqs"], 32)
        self.assertIn("model_candidates", shadow["diagnostics"])

    def test_write_recipe_shadow_diff(self):
        args = SimpleNamespace(
            engine="vllm_ascend",
            model_name="Qwen3-32B",
            model_path=str(self.model_dir),
            allow_experimental=False,
        )
        out = Path(self.tmp.name) / "shared"
        out.mkdir()
        with patch.dict(os.environ, {}, clear=True):
            shadow = write_recipe_shadow_diff(
                args,
                {"chip_id": "910b-32"},
                {"engine": "vllm_ascend", "engine_version": "0.10.0"},
                config_root=self.root,
                output_dir=out,
            )
        written = load_structured_file(out / "recipe_shadow_diff.json")
        self.assertEqual(shadow["status"], "ok")
        self.assertEqual(written["effective_mode"], "shadow")
        self.assertEqual(written["summary"]["proposed_param_count"], 2)

    def test_lint_flags_arch_missing_architecture_field(self):
        path = self.root / "recipes" / "architectures" / "Bad.yaml"
        diagnostics = validate_recipe_document(path, {"defaults": {}})
        self.assertTrue(any(level == "error" and "architecture" in message for level, message in diagnostics))

    def test_lint_flags_unknown_chip(self):
        path = self.root / "recipes" / "architectures" / "qwen3.yaml"
        data = {
            "architecture": "Qwen3ForCausalLM",
            "hardware_overlays": [{"applies_to_hardware": ["mystery-chip"], "defaults": {}}],
        }
        diagnostics = validate_recipe_document(path, data)
        self.assertTrue(any("mystery-chip" in message for _, message in diagnostics))


if __name__ == "__main__":
    unittest.main()
