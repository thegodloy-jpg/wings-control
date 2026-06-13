# -*- coding: utf-8 -*-
"""function_call_support.yaml 一致性护栏。

function_call_support.yaml 是 default.json 的【文档镜像】(运行时不加载，仅供展示)，
故无运行时校验、易漂移。本测试把它的不变量固化：
  - models 块严格等于 _LLM_MODELS（集合 + 顺序）——完备性，防再次缺模型 / 留 stale 名。
  - models 取值 == config 按匹配规则解析的结果——内部自洽（config 是 JSON 真值镜像）。
  - config 非 default 键均为合法模型名（已登记或已声明幽灵键）——防 typo / bogus 键。
覆盖 utils.model_utils._LLM_MODELS 与 docs/features/function_call/function_call_support.yaml。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from utils.model_utils import _LLM_MODELS  # noqa: E402

FC_YAML = (ROOT / "wings_control" / "docs" / "features"
           / "function_call" / "function_call_support.yaml")
ENGINES = ("vllm", "vllm_ascend", "sglang", "mindie")
# 仅这两个 V4 键按 token 子串匹配（与 _match_model_engine_config 对齐）。
_V4_SUBSTR_KEYS = ("deepseek-v4-flash", "deepseek-v4-pro")
# config 段允许出现的、未登记到 _LLM_MODELS 的幽灵键。
GHOST_CONFIG_KEYS = {
    "Qwen3MoeForCausalLM": {
        "Qwen3-Coder-480B-A35B-Instruct",
        "Qwen3-Coder-30B-A3B-Instruct",
    },
}


def _resolve(arch_item, model, engine):
    """复刻文档匹配规则：精确键 → V4 子串 → default → 省略。

    Returns (status, value)，status ∈ {"hit","default","omit"}；omit 表示该引擎
    配置缺失（models 条目应省略该 engine 键）。
    """
    cfg = (arch_item.get("config") or {}).get(engine)
    if not isinstance(cfg, dict):
        return ("omit", None)
    ml = model.lower()
    for key, val in cfg.items():
        if key != "default" and key.lower() == ml:
            return ("hit", val)
    for key, val in cfg.items():
        if key.lower() in _V4_SUBSTR_KEYS and key.lower() in ml:
            return ("hit", val)
    if "default" in cfg:
        return ("default", cfg["default"])
    return ("omit", None)


class TestFunctionCallSupportYaml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with FC_YAML.open("r", encoding="utf-8-sig") as fh:
            doc = yaml.safe_load(fh)
        cls.arches = {a["name"]: a for a in doc["architectures"]}

    def test_architecture_set_matches_llm_models(self):
        self.assertEqual(set(self.arches), set(_LLM_MODELS),
                         "function_call yaml 架构集合与 _LLM_MODELS 不一致")

    def test_models_block_matches_llm_models_exactly(self):
        for arch, models in _LLM_MODELS.items():
            ym = list((self.arches[arch].get("models") or {}).keys())
            self.assertEqual(ym, list(models),
                             f"{arch}: models 清单（集合或顺序）与 _LLM_MODELS 不一致")

    def test_models_values_match_resolved_config(self):
        for arch, models in _LLM_MODELS.items():
            item = self.arches[arch]
            ym = item.get("models") or {}
            for m in models:
                entry = ym.get(m) or {}
                for eng in ENGINES:
                    status, val = _resolve(item, m, eng)
                    if status == "omit":
                        self.assertNotIn(
                            eng, entry,
                            f"{arch}/{m}: 多余 {eng} 键（config 无该引擎段）")
                    else:
                        self.assertEqual(
                            entry.get(eng), val,
                            f"{arch}/{m}/{eng}: models={entry.get(eng)!r} != config解析={val!r}")

    def test_config_keys_are_known_models(self):
        for arch, item in self.arches.items():
            registered = set(_LLM_MODELS.get(arch, []))
            allowed = registered | GHOST_CONFIG_KEYS.get(arch, set())
            cfg = item.get("config") or {}
            for eng, mp in cfg.items():
                if not isinstance(mp, dict):
                    continue
                for key in mp:
                    if key == "default":
                        continue
                    self.assertIn(
                        key, allowed,
                        f"{arch}.{eng}: config 精确键 {key!r} 既非已登记模型也非已声明幽灵键")


if __name__ == "__main__":
    unittest.main(verbosity=2)
