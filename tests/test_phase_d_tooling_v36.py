import importlib.util
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name):
    import sys

    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhaseDToolingTests(unittest.TestCase):
    def test_deviation_validator_rejects_same_default(self):
        lint_deviations = load_tool("lint_deviations")
        diagnostics = lint_deviations.validate_deviation_document(
            "deviations/vllm.yaml",
            {
                "deviations": [
                    {
                        "field": "max_model_len",
                        "target": "cli",
                        "wings_value": 4096,
                        "upstream_default": 4096,
                        "applies_to_versions": ">=0.11.0",
                        "rationale": "test",
                        "source_type": "wings_validated",
                        "decision_date": "2026-01-01",
                    }
                ]
            },
        )
        errors = [message for level, message in diagnostics if level == "error"]
        warnings = [message for level, message in diagnostics if level == "warning"]
        self.assertTrue(any("equals upstream_default" in item for item in errors))
        self.assertTrue(any("has no applies_to_hardware" in item for item in warnings))

    def test_env_policy_force_override_requires_metadata(self):
        lint_env_policies = load_tool("lint_env_policies")
        diagnostics = lint_env_policies.validate_env_policy_document(
            "env_policies/vllm.yaml",
            {"env_policies": [{"name": "VLLM_TEST", "mode": "force_override", "applies_to_versions": ">=0.11.0"}]},
        )
        errors = [message for level, message in diagnostics if level == "error"]
        self.assertTrue(any("force_override requires rationale or derive_from" in item for item in errors))

    def test_env_policy_allows_same_name_for_different_runtime_predicates(self):
        lint_env_policies = load_tool("lint_env_policies")
        diagnostics = lint_env_policies.validate_env_policy_document(
            "env_policies/vllm_ascend.yaml",
            {
                "env_policies": [
                    {
                        "name": "HCCL_IF_IP",
                        "mode": "force_override",
                        "value_template": "$VLLM_HOST_IP",
                        "applies_when": {"deployment_mode": "ray"},
                        "applies_to_versions": ">=0.11.0",
                        "rationale": "runtime network env",
                        "source_type": "wings_decision",
                        "decision_date": "2026-05-15",
                    },
                    {
                        "name": "HCCL_IF_IP",
                        "mode": "force_override",
                        "value_template": "$VLLM_HOST_IP",
                        "applies_when": {"deployment_mode": "dp_deployment"},
                        "applies_to_versions": ">=0.11.0",
                        "rationale": "runtime network env",
                        "source_type": "wings_decision",
                        "decision_date": "2026-05-15",
                    },
                ]
            },
        )

        errors = [message for level, message in diagnostics if level == "error"]
        self.assertEqual(errors, [])

    def test_build_manifest_seed_and_diff(self):
        build_manifest = load_tool("build_manifest")
        diff_manifest = load_tool("diff_manifest")
        old = build_manifest.build_manifest(
            "vllm",
            "0.11.0",
            seed={"cli_params": {"model": {"type": "str", "upstream_default": None}}},
        )
        new = build_manifest.build_manifest(
            "vllm",
            "0.11.1",
            seed={
                "cli_params": {
                    "model": {"type": "str", "upstream_default": None},
                    "max_model_len": {"type": "int", "upstream_default": None},
                }
            },
        )
        diff = diff_manifest.diff_manifests(old, new)
        self.assertEqual(diff["sections"]["cli_params"]["added"], ["max_model_len"])
        self.assertEqual(diff["summary"]["added"], 1)

    def test_mapping_warns_when_manifest_missing_field(self):
        lint_mappings = load_tool("lint_mappings")
        config_tooling = load_tool("_config_tooling")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_dir = root / "manifests" / "vllm"
            manifest_dir.mkdir(parents=True)
            config_tooling.write_structured_file(
                manifest_dir / "0.11.0.yaml",
                {"engine": "vllm", "version": "0.11.0", "cli_params": {"model": {}}},
            )
            diagnostics = lint_mappings.validate_mapping_document(
                "mappings/canonical_to_engines.yaml",
                {
                    "canonical": {
                        "model_name": {
                            "vllm": {"field": "served_model_name", "target": "cli", "applies_to_versions": ">=0.11.0"}
                        }
                    }
                },
                config_root=str(root),
            )
        errors = [message for level, message in diagnostics if level == "error"]
        warnings = [message for level, message in diagnostics if level == "warning"]
        self.assertEqual(errors, [])
        self.assertTrue(any("not found in manifests/vllm" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
