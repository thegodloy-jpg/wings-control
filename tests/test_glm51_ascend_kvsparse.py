# -*- coding: utf-8 -*-
"""[GLM5.1-Ascend-Tmp] GLM-5.1 + vllm_ascend KV-Sparse 临时白名单单测。

覆盖范围：
  1. vllm_ascend + GLM-5.1（单机/双机） → 仅追加 --hf-overrides，不写 engine_config
  2. vllm_ascend + 非 GLM-5.1（含 GLM-5、Qwen、DeepseekV32） → no-op
  3. vllm（NVIDIA）回归：GLM-5.1/DeepseekV32 仍走 IndexCache，其他走 FP8
  4. rag→sparse 转译：vllm_ascend + GLM-5.1 + enable_rag_acc → enable_sparse=True
  5. 转译边界：非 GLM-5.1、非 ascend、enable_sparse 已为 True 时不转译
  6. _collect_indexcache_patch_features：vllm_ascend 永远返回 []（补丁不安装）
  7. is_glm51_ascend_kvsparse_tmp_scope 谓词单测

运行：
  python -m pytest tests/test_glm51_ascend_kvsparse.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(TESTS_DIR))

from engines import vllm_adapter  # noqa: E402
from engines.vllm_adapter import _build_kv_sparse_cmd  # noqa: E402
from utils.model_utils import is_glm51_ascend_kvsparse_tmp_scope  # noqa: E402
from core import config_loader  # noqa: E402
from core import wings_entry  # noqa: E402


class _FakeModelInfo:
    """ModelIdentifier stub —— 单测只关心 model_architecture。"""

    def __init__(self, architecture: str, config: dict | None = None):
        self.model_architecture = architecture
        self.config = config or {}


def _glm51_params(engine: str, *, enable_sparse: bool = False,
                  enable_rag_acc: bool = False) -> dict:
    """构造 GLM-5.1（路径含 5.1 标记）参数字典。"""
    params = {
        "model_name": "GLM-5.1",
        "model_path": "/models/glm-5.1",
        "model_type": "llm",
        "engine": engine,
    }
    if enable_sparse:
        params["enable_sparse"] = True
    if enable_rag_acc:
        params["enable_rag_acc"] = True
    return params


def _glm5_params(engine: str, *, enable_sparse: bool = False) -> dict:
    """构造 GLM-5（非 5.1）参数字典：同架构但路径不含 5.1 标记。"""
    params = {
        "model_name": "GLM-5",
        "model_path": "/models/glm-5",
        "model_type": "llm",
        "engine": engine,
    }
    if enable_sparse:
        params["enable_sparse"] = True
    return params


# ──────────────────────────────────────────────────────────────────────────────
# 1. 谓词 is_glm51_ascend_kvsparse_tmp_scope
# ──────────────────────────────────────────────────────────────────────────────


class TestIsGlm51AscendKvSparseTmpScope(unittest.TestCase):
    """谓词单测：仅 (vllm_ascend + GlmMoeDsaForCausalLM + name/path 含 5.1) 才命中。"""

    def test_ascend_glm51_path_marker_returns_true(self):
        info = _FakeModelInfo("GlmMoeDsaForCausalLM")
        self.assertTrue(is_glm51_ascend_kvsparse_tmp_scope(
            info, "vllm_ascend", model_name="GLM-5.1", model_path="/models/glm-5.1",
        ))

    def test_ascend_glm51_name_only_marker_returns_true(self):
        info = _FakeModelInfo("GlmMoeDsaForCausalLM")
        self.assertTrue(is_glm51_ascend_kvsparse_tmp_scope(
            info, "vllm_ascend", model_name="GLM-5.1", model_path="/models/foobar",
        ))

    def test_ascend_glm5_no_marker_returns_false(self):
        """vllm_ascend + GLM-5（架构相同但不含 5.1 标记） → False。"""
        info = _FakeModelInfo("GlmMoeDsaForCausalLM")
        self.assertFalse(is_glm51_ascend_kvsparse_tmp_scope(
            info, "vllm_ascend", model_name="GLM-5", model_path="/models/glm-5",
        ))

    def test_ascend_non_glm_arch_returns_false(self):
        info = _FakeModelInfo("DeepseekV32ForCausalLM")
        self.assertFalse(is_glm51_ascend_kvsparse_tmp_scope(
            info, "vllm_ascend", model_name="deepseek-v3.2",
            model_path="/models/deepseek-v3.2",
        ))

    def test_nvidia_glm51_returns_false(self):
        """vllm（NVIDIA）即便是 GLM-5.1 也不命中该谓词。"""
        info = _FakeModelInfo("GlmMoeDsaForCausalLM")
        self.assertFalse(is_glm51_ascend_kvsparse_tmp_scope(
            info, "vllm", model_name="GLM-5.1", model_path="/models/glm-5.1",
        ))

    def test_sglang_glm51_returns_false(self):
        info = _FakeModelInfo("GlmMoeDsaForCausalLM")
        self.assertFalse(is_glm51_ascend_kvsparse_tmp_scope(
            info, "sglang", model_name="GLM-5.1", model_path="/models/glm-5.1",
        ))


# ──────────────────────────────────────────────────────────────────────────────
# 2. _build_kv_sparse_cmd —— ascend 路径
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildKvSparseAscend(unittest.TestCase):
    """[GLM5.1-Ascend-Tmp] _build_kv_sparse_cmd 在 vllm_ascend 上的行为。"""

    def test_ascend_glm51_returns_hf_overrides(self):
        params = _glm51_params("vllm_ascend")
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            extra = _build_kv_sparse_cmd(params, "vllm_ascend")
        self.assertEqual(extra, " --hf-overrides '{\"index_topk_freq\": 4}'")

    def test_ascend_glm51_does_not_mutate_engine_config(self):
        """关键回归：ascend GLM-5.1 走 IndexCache 时不应触发 FP8 分支写 engine_config。"""
        params = _glm51_params("vllm_ascend")
        params["engine_config"] = {"tensor_parallel_size": 8}
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            _build_kv_sparse_cmd(params, "vllm_ascend")
        self.assertNotIn("kv_cache_dtype", params["engine_config"])
        self.assertNotIn("calculate_kv_scales", params["engine_config"])
        # 未触碰已有键
        self.assertEqual(params["engine_config"]["tensor_parallel_size"], 8)

    def test_ascend_glm5_not_glm51_returns_empty(self):
        """vllm_ascend + GLM-5（非 5.1）→ 空串，不进 IndexCache 分支。"""
        params = _glm5_params("vllm_ascend")
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            extra = _build_kv_sparse_cmd(params, "vllm_ascend")
        self.assertEqual(extra, "")

    def test_ascend_deepseek_v32_returns_empty(self):
        """vllm_ascend + DeepseekV32 → 空串（DeepseekV32 不在 ascend 白名单内）。"""
        params = {
            "model_name": "DeepSeek-V3.2",
            "model_path": "/models/deepseek-v3.2",
            "model_type": "llm",
            "engine": "vllm_ascend",
            "enable_sparse": True,
        }
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("DeepseekV32ForCausalLM"),
        ):
            extra = _build_kv_sparse_cmd(params, "vllm_ascend")
        self.assertEqual(extra, "")

    def test_ascend_qwen_returns_empty_no_fp8_mutation(self):
        """vllm_ascend + Qwen → 空串，绝对**不**触发 FP8 KV CACHE（FP8 仅限 vllm）。"""
        params = {
            "model_name": "Qwen3-32B",
            "model_path": "/models/qwen3-32b",
            "model_type": "llm",
            "engine": "vllm_ascend",
            "engine_config": {"kv_cache_dtype": "auto"},
        }
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("Qwen3ForCausalLM"),
        ):
            extra = _build_kv_sparse_cmd(params, "vllm_ascend")
        self.assertEqual(extra, "")
        # engine_config["kv_cache_dtype"] 必须保留为 "auto"，未被 FP8 改写
        self.assertEqual(params["engine_config"]["kv_cache_dtype"], "auto")
        self.assertNotIn("calculate_kv_scales", params["engine_config"])


# ──────────────────────────────────────────────────────────────────────────────
# 3. _build_kv_sparse_cmd —— NVIDIA vllm 回归
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildKvSparseNvidiaRegression(unittest.TestCase):
    """vllm（NVIDIA）回归：原有 IndexCache / FP8 行为不变。"""

    def test_nvidia_glm51_indexcache_unchanged(self):
        params = _glm51_params("vllm")
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            extra = _build_kv_sparse_cmd(params, "vllm")
        self.assertEqual(extra, " --hf-overrides '{\"index_topk_freq\": 4}'")

    def test_nvidia_deepseek_v32_indexcache_unchanged(self):
        params = {
            "model_name": "DeepSeek-V3.2",
            "model_path": "/models/deepseek-v3.2",
            "model_type": "llm",
            "engine": "vllm",
        }
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("DeepseekV32ForCausalLM"),
        ):
            extra = _build_kv_sparse_cmd(params, "vllm")
        self.assertEqual(extra, " --hf-overrides '{\"index_topk_freq\": 4}'")

    def test_nvidia_qwen_fp8_path_unchanged(self):
        params = {
            "model_name": "Qwen3-32B",
            "model_path": "/models/qwen3-32b",
            "model_type": "llm",
            "engine": "vllm",
            "engine_config": {},
        }
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("Qwen3ForCausalLM"),
        ):
            extra = _build_kv_sparse_cmd(params, "vllm")
        self.assertEqual(extra, "")
        self.assertEqual(params["engine_config"]["kv_cache_dtype"], "fp8")
        self.assertTrue(params["engine_config"]["calculate_kv_scales"])


# ──────────────────────────────────────────────────────────────────────────────
# 4. rag→sparse 转译
# ──────────────────────────────────────────────────────────────────────────────


class TestRagToSparseTranslation(unittest.TestCase):
    """[GLM5.1-Ascend-Tmp] _translate_glm51_ascend_rag_to_sparse 行为。"""

    def test_ascend_glm51_rag_translates_to_sparse_and_disables_rag(self):
        """转译后 enable_sparse=True 且 enable_rag_acc 被清零（rag 开关等价 kv 稀疏）。

        关键不变量：转译之后 params 中只有 sparse 是 True，rag 必须为 False，
        避免下游 RAG 加速链路（proxy 路由 / stream_collector）误启动。
        """
        params = _glm51_params("vllm_ascend", enable_rag_acc=True)
        with patch.object(
            config_loader, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            config_loader._translate_glm51_ascend_rag_to_sparse(params)
        self.assertTrue(params.get("enable_sparse"))
        # rag 开关被消化为 sparse，RAG 加速链路本身不启用
        self.assertFalse(params.get("enable_rag_acc"))

    def test_ascend_glm51_already_sparse_idempotent_no_logspam(self):
        """当 enable_sparse 已为 True，转译应直接跳过（不重复设置）。"""
        params = _glm51_params("vllm_ascend", enable_sparse=True, enable_rag_acc=True)
        with patch.object(
            config_loader, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ) as mock_ident:
            config_loader._translate_glm51_ascend_rag_to_sparse(params)
        # ModelIdentifier 不应被调用（早返回）
        mock_ident.assert_not_called()
        self.assertTrue(params["enable_sparse"])

    def test_ascend_glm5_rag_does_not_translate_or_disable_rag(self):
        """vllm_ascend + GLM-5（非 5.1）+ rag → 不转译，rag 保留原值。"""
        params = _glm5_params("vllm_ascend")
        params["enable_rag_acc"] = True
        with patch.object(
            config_loader, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            config_loader._translate_glm51_ascend_rag_to_sparse(params)
        # 非白名单 → rag 原样保留，不被改写
        self.assertTrue(params.get("enable_rag_acc"))
        self.assertFalse(params.get("enable_sparse"))

    def test_nvidia_glm51_rag_does_not_translate(self):
        """vllm（NVIDIA）+ GLM-5.1 + rag → 不转译（仅 ascend 临时白名单）。"""
        params = _glm51_params("vllm", enable_rag_acc=True)
        with patch.object(
            config_loader, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ) as mock_ident:
            config_loader._translate_glm51_ascend_rag_to_sparse(params)
        # NVIDIA 路径在 engine 门控就早返回，不应调用 ModelIdentifier
        mock_ident.assert_not_called()
        self.assertFalse(params.get("enable_sparse"))

    def test_ascend_glm51_without_rag_does_not_translate(self):
        """vllm_ascend + GLM-5.1 但未开 rag → 不转译。"""
        params = _glm51_params("vllm_ascend")
        with patch.object(
            config_loader, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ) as mock_ident:
            config_loader._translate_glm51_ascend_rag_to_sparse(params)
        mock_ident.assert_not_called()
        self.assertFalse(params.get("enable_sparse"))

    def test_model_identifier_failure_no_translation(self):
        """ModelIdentifier 抛异常 → 不转译，不抛出。"""
        params = _glm51_params("vllm_ascend", enable_rag_acc=True)
        with patch.object(
            config_loader, "ModelIdentifier",
            side_effect=RuntimeError("config.json missing"),
        ):
            # 不应抛出
            config_loader._translate_glm51_ascend_rag_to_sparse(params)
        self.assertFalse(params.get("enable_sparse"))

    def test_apply_engine_runtime_flags_end_to_end_env_vars(self):
        """端到端：_apply_engine_runtime_flags 执行后，RAG_ACC_ENABLED='false' 且 SPARSE_ENABLE='true'。

        关键不变量：转译路径不能让下游误以为 rag 是开的。
        SPARSE_ENABLE 由 _set_sparse_config 写入，
        RAG_ACC_ENABLED 由 _set_rag_acc_config 兜底写入，
        二者顺序：translate → sparse → rag。
        """
        import os
        params = _glm51_params("vllm_ascend", enable_rag_acc=True)
        # 模拟 wings_start.sh 跳过场景：先清掉 env 让 setdefault 路径生效
        env_backup = {
            "SPARSE_ENABLE": os.environ.pop("SPARSE_ENABLE", None),
            "RAG_ACC_ENABLED": os.environ.pop("RAG_ACC_ENABLED", None),
        }
        try:
            with patch.object(
                config_loader, "ModelIdentifier",
                return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
            ):
                config_loader._apply_engine_runtime_flags(params)
            self.assertTrue(params.get("enable_sparse"))
            self.assertFalse(params.get("enable_rag_acc"))
            self.assertEqual(os.environ.get("SPARSE_ENABLE"), "true")
            self.assertEqual(os.environ.get("RAG_ACC_ENABLED"), "false")
        finally:
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val


# ──────────────────────────────────────────────────────────────────────────────
# 5. _collect_indexcache_patch_features —— ascend 补丁安装屏蔽
# ──────────────────────────────────────────────────────────────────────────────


class TestIndexCachePatchCollectionAscend(unittest.TestCase):
    """vllm_ascend 永远不触发 indexcache 补丁安装（engine 门控天然屏蔽）。"""

    def test_ascend_glm51_sparse_returns_no_patch(self):
        params = _glm51_params("vllm_ascend", enable_sparse=True)
        with patch.object(
            wings_entry, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            features = wings_entry._collect_indexcache_patch_features("vllm_ascend", params)
        self.assertEqual(features, [])

    def test_ascend_deepseek_v32_sparse_returns_no_patch(self):
        params = {
            "model_name": "DeepSeek-V3.2",
            "model_path": "/models/deepseek-v3.2",
            "model_type": "llm",
            "enable_sparse": True,
        }
        with patch.object(
            wings_entry, "ModelIdentifier",
            return_value=_FakeModelInfo("DeepseekV32ForCausalLM"),
        ):
            features = wings_entry._collect_indexcache_patch_features("vllm_ascend", params)
        self.assertEqual(features, [])

    def test_nvidia_glm51_sparse_returns_indexcache_patch(self):
        """NVIDIA 回归：vllm + GLM-5.1 + sparse → indexcache 补丁仍正常采集。"""
        params = _glm51_params("vllm", enable_sparse=True)
        with patch.object(
            wings_entry, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            features = wings_entry._collect_indexcache_patch_features("vllm", params)
        self.assertEqual(features, ["indexcache"])


# ──────────────────────────────────────────────────────────────────────────────
# 6. 启动脚本端到端：sparse_args 进入单机/双机命令
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildStartScriptAscend(unittest.TestCase):
    """build_start_script 端到端验证：sparse_args 正确并入 ascend 命令。"""

    def test_single_node_ascend_glm51_script_contains_hf_overrides(self):
        params = _glm51_params("vllm_ascend", enable_sparse=True)
        params["engine_config"] = {"model": "/models/glm-5.1"}
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)
        self.assertIn("--hf-overrides '{\"index_topk_freq\": 4}'", script)

    def test_single_node_ascend_glm5_script_no_hf_overrides(self):
        params = _glm5_params("vllm_ascend", enable_sparse=True)
        params["engine_config"] = {"model": "/models/glm-5"}
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)
        self.assertNotIn("--hf-overrides", script)
        self.assertNotIn("--kv-cache-dtype fp8", script)

    def test_distributed_ray_head_ascend_glm51_contains_hf_overrides(self):
        """双机 GLM-5.1 (ray, rank=0) → head 节点命令包含 --hf-overrides。"""
        params = _glm51_params("vllm_ascend", enable_sparse=True)
        params.update({
            "distributed": True,
            "nnodes": 2,
            "node_rank": 0,
            "distributed_executor_backend": "ray",
            "master_ip": "192.168.1.1",
            "node_ips": "192.168.1.1,192.168.1.2",
            "engine_config": {"model": "/models/glm-5.1"},
        })
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)
        self.assertIn("--hf-overrides '{\"index_topk_freq\": 4}'", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
