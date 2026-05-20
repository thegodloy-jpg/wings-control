# -*- coding: utf-8 -*-
"""Tests for env_policies idempotent / force_override / inherit rendering (AC-005, AC-006).

Spec §6.5 三模式：
  idempotent    → proposed_env 只在 ENV 未设置时写入 default_value
  force_override → proposed_env 无条件写入 default_value
  inherit       → 不出现在 proposed_env（不覆盖）
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "wings_control"))

from core.hardware_detect import normalize_chip_id
from core.phase_d_loader import PhaseDRuntimeContext, build_phase_d_resolution


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ctx(engine: str = "vllm_ascend", version: str = "0.18.0", chip: str = "910b-32") -> PhaseDRuntimeContext:
    chip_id, _ = normalize_chip_id(chip)
    return PhaseDRuntimeContext(engine=engine, version=version, chip_id=chip_id or chip)


def _policies_root(tmp_path: Path, engine: str, policies: list[dict]) -> Path:
    """Write a minimal env_policies/<engine>.yaml and return config root."""
    import json
    ep_dir = tmp_path / "env_policies"
    ep_dir.mkdir(parents=True)
    (ep_dir / f"{engine}.yaml").write_text(
        json.dumps({"engine": engine, "env_policies": policies}),
        encoding="utf-8",
    )
    (tmp_path / "manifests").mkdir(exist_ok=True)
    (tmp_path / "deviations").mkdir(exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# idempotent mode
# ---------------------------------------------------------------------------

class TestIdempotentMode:
    def test_sets_default_when_env_not_set(self, tmp_path):
        """AC-005: idempotent → export X="${X:-Y}" semantics."""
        root = _policies_root(tmp_path, "vllm_ascend", [
            {"name": "VLLM_USE_MODELSCOPE", "mode": "idempotent", "default_value": "False",
             "applies_to_versions": "*", "rationale": "test", "source_type": "wings_decision",
             "decision_date": "2026-01-01"},
        ])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_USE_MODELSCOPE", None)
            result = build_phase_d_resolution(_ctx(), {}, config_root=root)
        assert result["proposed_env"].get("VLLM_USE_MODELSCOPE") == "False"

    def test_does_not_overwrite_when_env_already_set(self, tmp_path):
        """AC-006: idempotent → user-set env var must survive."""
        root = _policies_root(tmp_path, "vllm_ascend", [
            {"name": "VLLM_USE_MODELSCOPE", "mode": "idempotent", "default_value": "False",
             "applies_to_versions": "*", "rationale": "test", "source_type": "wings_decision",
             "decision_date": "2026-01-01"},
        ])
        with patch.dict(os.environ, {"VLLM_USE_MODELSCOPE": "True"}):
            result = build_phase_d_resolution(_ctx(), {}, config_root=root)
        # idempotent: when env is already set, must NOT appear in proposed_env
        assert "VLLM_USE_MODELSCOPE" not in result["proposed_env"]


# ---------------------------------------------------------------------------
# force_override mode
# ---------------------------------------------------------------------------

class TestForceOverrideMode:
    def test_always_sets_value(self, tmp_path):
        """force_override always writes to proposed_env regardless of current env."""
        root = _policies_root(tmp_path, "vllm_ascend", [
            {"name": "HCCL_BUFFSIZE", "mode": "force_override", "default_value": "1024",
             "applies_to_versions": "*", "rationale": "required for performance", "source_type": "wings_decision",
             "decision_date": "2026-01-01"},
        ])
        # Even when user has set it to 2048, force_override overwrites
        with patch.dict(os.environ, {"HCCL_BUFFSIZE": "2048"}):
            result = build_phase_d_resolution(_ctx(), {}, config_root=root)
        assert result["proposed_env"].get("HCCL_BUFFSIZE") == "1024"

    def test_sets_value_when_env_not_set(self, tmp_path):
        root = _policies_root(tmp_path, "vllm_ascend", [
            {"name": "HCCL_BUFFSIZE", "mode": "force_override", "default_value": "1024",
             "applies_to_versions": "*", "rationale": "required for performance", "source_type": "wings_decision",
             "decision_date": "2026-01-01"},
        ])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HCCL_BUFFSIZE", None)
            result = build_phase_d_resolution(_ctx(), {}, config_root=root)
        assert result["proposed_env"].get("HCCL_BUFFSIZE") == "1024"


# ---------------------------------------------------------------------------
# inherit mode
# ---------------------------------------------------------------------------

class TestInheritMode:
    def test_never_appears_in_proposed_env(self, tmp_path):
        """AC-005: inherit → must not appear in proposed_env."""
        root = _policies_root(tmp_path, "vllm_ascend", [
            {"name": "ASCEND_RT_VISIBLE_DEVICES", "mode": "inherit",
             "applies_to_versions": "*", "rationale": "managed by orchestration",
             "source_type": "wings_decision", "decision_date": "2026-01-01"},
        ])
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}):
            result = build_phase_d_resolution(_ctx(), {}, config_root=root)
        assert "ASCEND_RT_VISIBLE_DEVICES" not in result["proposed_env"]

    def test_never_appears_even_when_no_env(self, tmp_path):
        root = _policies_root(tmp_path, "vllm_ascend", [
            {"name": "ASCEND_RT_VISIBLE_DEVICES", "mode": "inherit",
             "applies_to_versions": "*", "rationale": "managed by orchestration",
             "source_type": "wings_decision", "decision_date": "2026-01-01"},
        ])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            result = build_phase_d_resolution(_ctx(), {}, config_root=root)
        assert "ASCEND_RT_VISIBLE_DEVICES" not in result["proposed_env"]


# ---------------------------------------------------------------------------
# version filtering
# ---------------------------------------------------------------------------

class TestVersionFiltering:
    def test_policy_skipped_when_version_too_old(self, tmp_path):
        root = _policies_root(tmp_path, "vllm_ascend", [
            {"name": "NEW_FLAG", "mode": "idempotent", "default_value": "1",
             "applies_to_versions": ">=0.12.0", "rationale": "new in 0.12",
             "source_type": "wings_decision", "decision_date": "2026-01-01"},
        ])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEW_FLAG", None)
            result = build_phase_d_resolution(_ctx(version="0.11.0"), {}, config_root=root)
        assert "NEW_FLAG" not in result["proposed_env"]

    def test_policy_applied_when_version_matches(self, tmp_path):
        root = _policies_root(tmp_path, "vllm_ascend", [
            {"name": "NEW_FLAG", "mode": "idempotent", "default_value": "1",
             "applies_to_versions": ">=0.11.0", "rationale": "new in 0.11",
             "source_type": "wings_decision", "decision_date": "2026-01-01"},
        ])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEW_FLAG", None)
            result = build_phase_d_resolution(_ctx(version="0.11.0"), {}, config_root=root)
        assert result["proposed_env"].get("NEW_FLAG") == "1"


# ---------------------------------------------------------------------------
# real vllm_ascend.yaml integration
# ---------------------------------------------------------------------------

class TestRealEnvPoliciesFile:
    def test_real_file_idempotent_vllm_use_modelscope(self):
        """Verify the shipped vllm_ascend.yaml VLLM_USE_MODELSCOPE is idempotent."""
        ctx = _ctx()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_USE_MODELSCOPE", None)
            result = build_phase_d_resolution(ctx, {})
        # VLLM_USE_MODELSCOPE should be proposed with default False
        assert result["proposed_env"].get("VLLM_USE_MODELSCOPE") == "False"

    def test_real_file_inherit_ascend_rt_visible_devices(self):
        """Verify the shipped vllm_ascend.yaml ASCEND_RT_VISIBLE_DEVICES is inherit."""
        ctx = _ctx()
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0"}):
            result = build_phase_d_resolution(ctx, {})
        assert "ASCEND_RT_VISIBLE_DEVICES" not in result["proposed_env"]

    def test_summary_counts(self):
        ctx = _ctx()
        result = build_phase_d_resolution(ctx, {})
        assert result["summary"]["env_policy_count"] >= 2
