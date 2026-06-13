# -*- coding: utf-8 -*-
"""reasoning_parser_support.yaml 一致性护栏。

把「YAML ↔ _LLM_MODELS 严格一致 + 无死配置 + default.json 无残留 + 解析端契约」
固化为回归测试，防止支持矩阵再次漂移。覆盖：
  - utils.model_utils._LLM_MODELS（权威模型清单）
  - core.config_loader._load_reasoning_parser_support / _resolve_reasoning_parser_support（YAML 唯一消费者）
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from utils.model_utils import _LLM_MODELS  # noqa: E402
from core import config_loader as cl  # noqa: E402

ENGINES = ("vllm", "vllm_ascend")

# config 段允许出现的、未登记到 _LLM_MODELS 的「幽灵模型」专属键（为同架构未登记模型兜底）。
# 这些键在运行时唯一可达的 config 精确键；新增时同步更新此处与 YAML 注释。
GHOST_CONFIG_KEYS = {
    "Qwen3MoeForCausalLM": {
        "Qwen3-Coder-480B-A35B-Instruct",
        "Qwen3-Coder-30B-A3B-Instruct",
    },
}


class TestReasoningParserSupportYaml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = cl._load_reasoning_parser_support()
        cls.assertTrue(cls.support, "reasoning_parser_support.yaml 加载为空")

    # ---- 完备性：架构集合 & 每架构 models 清单（集合 + 顺序）严格等于 _LLM_MODELS ----
    def test_architecture_set_matches_llm_models(self):
        self.assertEqual(set(self.support), set(_LLM_MODELS),
                         "YAML 架构集合与 _LLM_MODELS 不一致")

    def test_models_block_matches_llm_models_exactly(self):
        for arch, models in _LLM_MODELS.items():
            ym = self.support[arch].get("models") or {}
            self.assertEqual(list(ym.keys()), list(models),
                             f"{arch}: models 清单（集合或顺序）与 _LLM_MODELS 不一致")

    def test_every_model_declares_both_engines(self):
        # 每个已登记模型两引擎键齐全 → 恒命中 models 块、永不回落 config（死配置不可达的前提）。
        for arch, models in _LLM_MODELS.items():
            ym = self.support[arch].get("models") or {}
            for m in models:
                ev = ym.get(m)
                self.assertIsInstance(ev, dict, f"{arch}/{m}: 取值非 mapping")
                for eng in ENGINES:
                    self.assertIn(eng, ev, f"{arch}/{m}: 缺少 engine 键 {eng}")

    # ---- 无死配置：config 非 default 键不得镜像已登记模型；只允许声明的幽灵键 ----
    def test_config_has_no_dead_registered_keys(self):
        for arch, item in self.support.items():
            registered = set(_LLM_MODELS.get(arch, []))
            allowed_ghosts = GHOST_CONFIG_KEYS.get(arch, set())
            cfg = item.get("config") or {}
            for eng, mp in cfg.items():
                if not isinstance(mp, dict):
                    continue
                for key in mp:
                    if key == "default":
                        continue
                    self.assertNotIn(
                        key, registered,
                        f"{arch}.{eng}: config 精确键 {key!r} 镜像了已登记模型（死配置，会被 models 块屏蔽）")
                    self.assertIn(
                        key, allowed_ghosts,
                        f"{arch}.{eng}: config 精确键 {key!r} 既非 default 也非已声明幽灵键")

    # ---- 解析端契约：每个已登记模型的解析结果 == 其 models 块取值 ----
    def test_resolve_matches_models_block(self):
        for arch, models in _LLM_MODELS.items():
            ym = self.support[arch].get("models") or {}
            for m in models:
                for eng in ENGINES:
                    exp = (ym.get(m) or {}).get(eng)
                    found, val = cl._resolve_reasoning_parser_support(arch, m, eng)
                    got = val if found else None
                    self.assertEqual(got, exp, f"{arch}/{m}/{eng}: 解析={got!r} != models={exp!r}")

    # ---- 回归锚点：A1/A2（QwQ 与 R1-Distill 统一 null）、C1（幽灵键兜底 null）----
    def test_always_on_distill_models_are_null(self):
        for arch, name in (
            ("Qwen2ForCausalLM", "QwQ-32B"),
            ("Qwen2ForCausalLM", "DeepSeek-R1-Distill-Qwen-32B"),
            ("LlamaForCausalLM", "DeepSeek-R1-Distill-Llama-70B"),
        ):
            for eng in ENGINES:
                found, val = cl._resolve_reasoning_parser_support(arch, name, eng)
                self.assertIsNone(val, f"{name}/{eng} 应为 null（不启用思维解析），实得 {val!r}")

    def test_ghost_coder_keys_fallback_null(self):
        for name in ("Qwen3-Coder-480B-A35B-Instruct", "Qwen3-Coder-30B-A3B-Instruct"):
            for eng in ENGINES:
                found, val = cl._resolve_reasoning_parser_support("Qwen3MoeForCausalLM", name, eng)
                self.assertTrue(found, f"{name}/{eng} 应被 config 幽灵键命中")
                self.assertIsNone(val, f"{name}/{eng} 幽灵键应兜底 null，实得 {val!r}")


class TestDefaultJsonHasNoReasoningParser(unittest.TestCase):
    """nvidia_default.json / ascend_default.json 不得再存放 reasoning_parser。"""

    def _iter_keys(self, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from self._iter_keys(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from self._iter_keys(it)

    def test_no_reasoning_parser_residue(self):
        defaults_dir = ROOT / "wings_control" / "config" / "defaults"
        for fname in ("nvidia_default.json", "ascend_default.json"):
            path = defaults_dir / fname
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            self.assertNotIn("reasoning_parser", set(self._iter_keys(data)),
                             f"{fname} 残留 reasoning_parser（应仅由 YAML 承载）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
