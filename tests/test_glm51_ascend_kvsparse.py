# -*- coding: utf-8 -*-
"""[GLM5.1-Ascend-Tmp] GLM-5.1 + vllm_ascend KV-Sparse 临时白名单单测。

覆盖范围：
  1. vllm_ascend + GLM-5.1（单机/双机） → 仅追加 --hf-overrides，不写 engine_config
  2. vllm_ascend + 非 GLM-5.1（含 GLM-5、Qwen、DeepseekV32） → no-op
  3. vllm（NVIDIA）回归：GLM-5.1/DeepseekV32 仍走 IndexCache，其他走 FP8
  4. rag 与 sparse 开关在所有场景下独立（rag→sparse 翻译已下线）
  5. ascend + GLM-5.1 KV 稀疏走普通开关门控（§0 裁定1 删 forced）：enable_sparse on 才产，off/未设不产
  6. _collect_indexcache_patch_features：vllm_ascend 永远返回 []（补丁不安装）
  7. is_glm51_ascend_kvsparse_tmp_scope 谓词单测

运行：
  python -m pytest tests/test_glm51_ascend_kvsparse.py -v
"""

from __future__ import annotations

import os
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
# 4. rag 与 sparse 开关独立 + ascend+GLM-5.1 强制开启 KV 稀疏
# ──────────────────────────────────────────────────────────────────────────────


class TestRagAndSparseAreIndependent(unittest.TestCase):
    """rag→sparse 翻译已下线：两个开关在所有场景下完全独立。"""

    def setUp(self):
        # _apply_engine_runtime_flags 会写 env 变量；保证测试隔离。
        self._env_backup = {
            key: os.environ.pop(key, None)
            for key in ("SPARSE_ENABLE", "RAG_ACC_ENABLED")
        }

    def tearDown(self):
        for key, val in self._env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_ascend_glm51_rag_only_does_not_auto_enable_sparse(self):
        """vllm_ascend + GLM-5.1 + 仅 rag → enable_sparse 不被自动置 True，rag 保持。"""
        params = _glm51_params("vllm_ascend", enable_rag_acc=True)
        with patch.object(
            config_loader, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            config_loader._apply_engine_runtime_flags(params)
        self.assertTrue(params.get("enable_rag_acc"))
        self.assertFalse(params.get("enable_sparse"))
        # rag 链路独立生效
        self.assertEqual(os.environ.get("RAG_ACC_ENABLED"), "true")

    def test_ascend_glm51_sparse_only_does_not_touch_rag(self):
        """vllm_ascend + GLM-5.1 + 仅 sparse → enable_rag_acc 不被改动。"""
        params = _glm51_params("vllm_ascend", enable_sparse=True)
        with patch.object(
            config_loader, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            config_loader._apply_engine_runtime_flags(params)
        self.assertTrue(params.get("enable_sparse"))
        self.assertFalse(params.get("enable_rag_acc"))
        self.assertEqual(os.environ.get("SPARSE_ENABLE"), "true")
        # rag 未开 → 兜底 false
        self.assertEqual(os.environ.get("RAG_ACC_ENABLED"), "false")

    def test_ascend_glm51_both_enabled_both_stay_enabled(self):
        """vllm_ascend + GLM-5.1 + rag 与 sparse 同开 → 双链路独立生效。"""
        params = _glm51_params("vllm_ascend", enable_sparse=True, enable_rag_acc=True)
        with patch.object(
            config_loader, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            config_loader._apply_engine_runtime_flags(params)
        self.assertTrue(params.get("enable_sparse"))
        self.assertTrue(params.get("enable_rag_acc"))
        self.assertEqual(os.environ.get("SPARSE_ENABLE"), "true")
        self.assertEqual(os.environ.get("RAG_ACC_ENABLED"), "true")

    def test_nvidia_glm51_rag_unchanged(self):
        """vllm（NVIDIA）+ GLM-5.1 + rag → 仅 RAG_ACC_ENABLED=true，sparse 不被自动启用。"""
        params = _glm51_params("vllm", enable_rag_acc=True)
        config_loader._apply_engine_runtime_flags(params)
        self.assertTrue(params.get("enable_rag_acc"))
        self.assertFalse(params.get("enable_sparse"))
        self.assertEqual(os.environ.get("RAG_ACC_ENABLED"), "true")


class TestBuildStartScriptAscendSwitchGated(unittest.TestCase):
    """端到端：vllm_ascend + GLM-5.1 KV 稀疏走普通开关门控（§0 裁定1 删 forced）：
    enable_sparse=False/未传 → 脚本不含 --hf-overrides（不再强制开）。
    注：本套用例直接调 build_start_script（绕过 C14），enable_sparse 即「有效开关」。"""

    def _params(self, *, distributed: bool = False, with_sparse: bool = False,
                with_rag: bool = False, rank: int = 0) -> dict:
        engine_config = {
            "model": "/models/glm-5.1",
            "max_model_len": 131072 if distributed else 4096,
            "max_num_seqs": 2 if distributed else 8,
        }
        params = {
            "model_name": "GLM-5.1",
            "model_path": "/models/glm-5.1",
            "model_type": "llm",
            "engine": "vllm_ascend",
            "engine_config": engine_config,
        }
        if with_sparse:
            params["enable_sparse"] = True
        if with_rag:
            params["enable_rag_acc"] = True
        if distributed:
            params.update({
                "distributed": True,
                "nnodes": 2,
                "node_rank": rank,
                "distributed_executor_backend": "dp_deployment",
                "master_ip": "192.168.1.1",
                "node_ips": "192.168.1.1,192.168.1.2",
                "device_count": 8,
            })
        return params

    def _build(self, params, *, arch: str = "GlmMoeDsaForCausalLM") -> str:
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo(arch),
        ):
            return vllm_adapter.build_start_script(params)

    def test_single_node_sparse_missing_no_hf_overrides(self):
        """单机 vllm_ascend + GLM-5.1 + 未传 enable_sparse → 脚本不含 --hf-overrides（无 forced 复活）。"""
        params = self._params(with_sparse=False)
        params.pop("enable_sparse", None)
        script = self._build(params)
        self.assertNotIn("--hf-overrides", script)

    def test_single_node_sparse_on_emits_hf_overrides(self):
        """单机 vllm_ascend + GLM-5.1 + enable_sparse=True → 脚本含 --hf-overrides（开关 on 正常产）。"""
        script = self._build(self._params(with_sparse=True))
        self.assertIn("--hf-overrides", script)

    def test_dual_node_dp_head_sparse_false_no_hf_overrides(self):
        """双机 DP head + GLM-5.1 + enable_sparse=False → 脚本不含 --hf-overrides。"""
        script = self._build(self._params(distributed=True, with_sparse=False, rank=0))
        self.assertNotIn("--hf-overrides", script)

    def test_dual_node_dp_worker_sparse_false_no_hf_overrides(self):
        """双机 DP worker + GLM-5.1 + enable_sparse=False → 脚本不含 --hf-overrides。"""
        script = self._build(self._params(distributed=True, with_sparse=False, rank=1))
        self.assertNotIn("--hf-overrides", script)

    def test_dual_node_dp_head_rag_only_no_hf_overrides(self):
        """双机 DP head + GLM-5.1 + 仅 rag（sparse off）→ 脚本不含 --hf-overrides（sparse 与 rag 独立、无 forced）。"""
        script = self._build(
            self._params(distributed=True, with_sparse=False, with_rag=True, rank=0)
        )
        self.assertNotIn("--hf-overrides", script)

    def test_ascend_glm5_not_51_sparse_false_no_hf_overrides(self):
        """vllm_ascend + GLM-5（非 5.1） + enable_sparse=False → 脚本不含 --hf-overrides。"""
        params = self._params(with_sparse=False)
        params["model_name"] = "GLM-5"
        params["model_path"] = "/models/glm-5"
        script = self._build(params)
        self.assertNotIn("--hf-overrides", script)

    def test_nvidia_glm51_sparse_false_no_hf_overrides(self):
        """vllm（NVIDIA）+ GLM-5.1 + enable_sparse=False → 脚本不含 --hf-overrides（NVIDIA 仍由开关守门）。"""
        params = self._params(with_sparse=False)
        params["engine"] = "vllm"
        script = self._build(params)
        self.assertNotIn("--hf-overrides", script)


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

# ──────────────────────────────────────────────────────────────────────────────
# 8. _get_engine_config_platform —— ENGINE_VERSION 后缀检测
# ──────────────────────────────────────────────────────────────────────────────


class TestGetEngineConfigPlatform(unittest.TestCase):
    """ENGINE_VERSION 后缀 '-a3' 应作次级信号被识别。"""

    def setUp(self):
        for var in ("WINGS_ASCEND_PLATFORM", "ASCEND_PLATFORM", "ENGINE_IMAGE_FLAVOR",
                    "ENGINE_VERSION"):
            self.addCleanup(lambda v=var, old=os.environ.get(var):
                            os.environ.__setitem__(v, old) if old is not None
                            else os.environ.pop(v, None))
            os.environ.pop(var, None)

    def test_engine_version_a3_suffix_returns_a3(self):
        with patch.dict("os.environ", {"ENGINE_VERSION": "0.13.0rc3-a3"}, clear=False):
            self.assertEqual(vllm_adapter._get_engine_config_platform({}), "a3")

    def test_engine_version_uppercase_a3_suffix_returns_a3(self):
        with patch.dict("os.environ", {"ENGINE_VERSION": "0.13.0rc3-A3"}, clear=False):
            self.assertEqual(vllm_adapter._get_engine_config_platform({}), "a3")

    def test_engine_version_without_a3_suffix_returns_empty(self):
        """没有 -a3 后缀 → 不强推 a2，让下游 _resolve_*_platform 兜底。"""
        with patch.dict("os.environ", {"ENGINE_VERSION": "0.13.0rc3"}, clear=False):
            self.assertEqual(vllm_adapter._get_engine_config_platform({}), "")

    def test_explicit_declaration_overrides_engine_version(self):
        """显式声明优先级 > ENGINE_VERSION。"""
        with patch.dict("os.environ", {
            "WINGS_ASCEND_PLATFORM": "a2",
            "ENGINE_VERSION": "0.13.0rc3-a3",
        }, clear=False):
            self.assertEqual(vllm_adapter._get_engine_config_platform({}), "a2")

    def test_no_signal_returns_empty(self):
        self.assertEqual(vllm_adapter._get_engine_config_platform({}), "")


# ──────────────────────────────────────────────────────────────────────────────
# 9. 端到端：build_start_script 在 A2/A3 × 单机/双机下命令产出
# ──────────────────────────────────────────────────────────────────────────────


def _clear_platform_env(test_case):
    """清掉所有可能影响 platform 检测的 env，确保测试独立。"""
    for var in ("WINGS_ASCEND_PLATFORM", "ASCEND_PLATFORM", "ENGINE_IMAGE_FLAVOR",
                "ENGINE_VERSION", "ASCEND_A3_ENABLE", "WINGS_DEVICE_NAME"):
        test_case.addCleanup(
            lambda v=var, old=os.environ.get(var):
            os.environ.__setitem__(v, old) if old is not None
            else os.environ.pop(v, None)
        )
        os.environ.pop(var, None)


def _ascend_default_additional_config() -> dict:
    """模拟 ascend_default.json 中 GlmMoeDsaForCausalLM 默认带的 additional_config。"""
    return {
        "fuse_muls_add": True,
        "multistream_overlap_shared_expert": True,
        "ascend_compilation_config": {"enable_npugraph_ex": True},
    }


def _glm51_e2e_params(*, distributed: bool = False, rank: int = 0,
                       with_default_additional_config: bool = True) -> dict:
    """构造端到端 GLM-5.1 ascend 启动参数（模拟 JSON 默认已经合并进 engine_config）。"""
    engine_config = {
        "model": "/models/glm-5.1",
        "max_model_len": 131072 if distributed else 4096,
        "max_num_seqs": 2 if distributed else 8,
    }
    if with_default_additional_config:
        engine_config["additional_config"] = _ascend_default_additional_config()
    params = {
        "model_name": "GLM-5.1",
        "model_path": "/models/glm-5.1",
        "model_type": "llm",
        "engine": "vllm_ascend",
        "engine_config": engine_config,
    }
    if distributed:
        params.update({
            "distributed": True,
            "nnodes": 2,
            "node_rank": rank,
            "distributed_executor_backend": "dp_deployment",
            "master_ip": "192.168.1.1",
            "node_ips": "192.168.1.1,192.168.1.2",
            "device_count": 8,
        })
    return params


class TestEndToEndPlatformAwareScript(unittest.TestCase):
    """端到端：build_start_script 在 A2/A3 上的 --additional-config 与 env 输出。

    官方依据：vllm-ascend GLM-5 W8A8 双机命令
      * A2/A3 单机/双机：均传 ``--additional-config`` 三键
      * A3 额外环境变量：``VLLM_ASCEND_ENABLE_MLAPO=1``（A2 不带）
      * 显式 WINGS_ASCEND_PLATFORM 优先级高于 ENGINE_VERSION 后缀
    """

    def setUp(self):
        _clear_platform_env(self)

    def _build(self, params):
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("GlmMoeDsaForCausalLM"),
        ):
            return vllm_adapter.build_start_script(params)

    # ── A2 单机 ───────────────────────────────────────────────────────────
    def test_a2_single_node_script_contains_additional_config_three_keys(self):
        os.environ["WINGS_ASCEND_PLATFORM"] = "a2"
        script = self._build(_glm51_e2e_params())
        self.assertIn("--additional-config", script)
        self.assertIn("fuse_muls_add", script)
        self.assertIn("multistream_overlap_shared_expert", script)
        self.assertIn("ascend_compilation_config", script)
        self.assertIn("enable_npugraph_ex", script)
        # A2 不应注入 MLAPO
        self.assertNotIn("VLLM_ASCEND_ENABLE_MLAPO", script)

    def test_a2_single_node_unspecified_platform_falls_back_to_a2_keeps_additional_config(self):
        """无任何 platform 信号 → 兜底为 A2 → additional_config 保留，无 MLAPO。"""
        # 已由 setUp 清掉 env
        script = self._build(_glm51_e2e_params())
        self.assertIn("--additional-config", script)
        self.assertIn("fuse_muls_add", script)
        self.assertNotIn("VLLM_ASCEND_ENABLE_MLAPO", script)

    # ── A3 单机：保留 additional-config + 追加 MLAPO ─────────────────────
    def test_a3_single_node_via_wings_platform_keeps_additional_config_and_adds_mlapo(self):
        os.environ["WINGS_ASCEND_PLATFORM"] = "a3"
        script = self._build(_glm51_e2e_params())
        self.assertIn("--additional-config", script)
        self.assertIn("fuse_muls_add", script)
        self.assertIn("multistream_overlap_shared_expert", script)
        self.assertIn("enable_npugraph_ex", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_MLAPO=1", script)

    def test_a3_single_node_via_engine_version_suffix_keeps_additional_config_and_adds_mlapo(self):
        os.environ["ENGINE_VERSION"] = "0.13.0rc3-a3"
        script = self._build(_glm51_e2e_params())
        self.assertIn("--additional-config", script)
        self.assertIn("fuse_muls_add", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_MLAPO=1", script)

    # ── A2 双机（DP backend）─────────────────────────────────────────────
    def test_a2_dual_node_dp_head_contains_additional_config(self):
        os.environ["WINGS_ASCEND_PLATFORM"] = "a2"
        script = self._build(_glm51_e2e_params(distributed=True, rank=0))
        self.assertIn("--additional-config", script)
        self.assertIn("fuse_muls_add", script)
        # 同时验证 DP 后端关键参数到位
        self.assertIn("--data-parallel-size", script)
        self.assertNotIn("VLLM_ASCEND_ENABLE_MLAPO", script)

    def test_a2_dual_node_dp_worker_contains_additional_config(self):
        os.environ["WINGS_ASCEND_PLATFORM"] = "a2"
        script = self._build(_glm51_e2e_params(distributed=True, rank=1))
        # worker 节点也含 additional_config（vllm DP 模式 head/worker 命令相同）
        self.assertIn("--additional-config", script)

    # ── A3 双机：移除 additional-config（对齐官方 A3 多机模板）+ 追加 MLAPO ──
    # 官方 A3 双机模板特意省略 fuse_muls_add / multistream_overlap_shared_expert /
    # enable_npugraph_ex，这些图优化开关在长上下文 decode replay 时触发 MTE 越界崩溃。
    def test_a3_dual_node_drops_additional_config_and_adds_mlapo(self):
        os.environ["WINGS_ASCEND_PLATFORM"] = "a3"
        script = self._build(_glm51_e2e_params(distributed=True, rank=0))
        self.assertNotIn("--additional-config", script)
        self.assertNotIn("fuse_muls_add", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_MLAPO=1", script)
        # A3 双机 prefix caching 不再被强制关闭
        self.assertIn("--enable-prefix-caching", script)
        self.assertNotIn("--no-enable-prefix-caching", script)

    def test_a3_dual_node_via_engine_version_drops_additional_config_and_adds_mlapo(self):
        os.environ["ENGINE_VERSION"] = "0.13.0rc3-a3"
        script = self._build(_glm51_e2e_params(distributed=True, rank=0))
        self.assertNotIn("--additional-config", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_MLAPO=1", script)

    # ── 优先级 ───────────────────────────────────────────────────────────
    def test_wings_platform_a2_overrides_engine_version_a3(self):
        """显式 WINGS_ASCEND_PLATFORM=a2 应覆盖 ENGINE_VERSION=...-a3。"""
        os.environ["WINGS_ASCEND_PLATFORM"] = "a2"
        os.environ["ENGINE_VERSION"] = "0.13.0rc3-a3"
        script = self._build(_glm51_e2e_params())
        self.assertIn("--additional-config", script)
        self.assertIn("fuse_muls_add", script)
        # 显式 a2 → 不应追加 MLAPO
        self.assertNotIn("VLLM_ASCEND_ENABLE_MLAPO", script)

    def test_engine_version_non_a3_keeps_a2_default(self):
        """ENGINE_VERSION 不带 -a3 后缀 → 视作 A2 → 保留 additional_config，无 MLAPO。"""
        os.environ["ENGINE_VERSION"] = "0.13.0rc3"
        script = self._build(_glm51_e2e_params())
        self.assertIn("--additional-config", script)
        self.assertNotIn("VLLM_ASCEND_ENABLE_MLAPO", script)

    # ── 非 GLM 架构不受影响 ───────────────────────────────────────────────
    def test_a3_non_glm_arch_no_mlapo_added(self):
        """vllm_ascend + 非 GlmMoeDsa + A3 → 不进入 GLM-5 env 分支，不应注入 MLAPO。"""
        os.environ["WINGS_ASCEND_PLATFORM"] = "a3"
        params = _glm51_e2e_params()
        with patch.object(
            vllm_adapter, "ModelIdentifier",
            return_value=_FakeModelInfo("Qwen3ForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)
        # Qwen3 不进入 GLM 分支，env 不应注入 GLM-5 专属的 MLAPO
        self.assertNotIn("export VLLM_ASCEND_ENABLE_MLAPO=1", script)


# ──────────────────────────────────────────────────────────────────────────────
# 10. DSv4-Flash 回归：ENGINE_VERSION 信号扩展不破坏 DSv4-Flash 路径
# ──────────────────────────────────────────────────────────────────────────────


class TestDeepseekV4FlashRegression(unittest.TestCase):
    """ENGINE_VERSION 接入 _get_engine_config_platform 后，DSv4-Flash 路径行为不变。

    关键不变量：
      * `_resolve_deepseek_v4_flash_platform` 现在能感知 ENGINE_VERSION 后缀
      * 无任何信号时 → 仍兜底 a2（不破坏既有默认）
      * ENGINE_VERSION=...-a3 → 解析为 a3（这是 desired 扩展能力）
      * 显式 WINGS_ASCEND_PLATFORM 仍是最高优先级
    """

    def setUp(self):
        _clear_platform_env(self)

    def _params(self, **extra):
        params = {
            "model_name": "DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/deepseek-v4-flash-w8a8-mtp",
            "model_type": "llm",
            "engine": "vllm_ascend",
            "engine_config": {},
        }
        params.update(extra)
        return params

    def test_no_signal_resolves_to_a2(self):
        self.assertEqual(
            vllm_adapter._resolve_deepseek_v4_flash_platform(self._params()),
            "a2",
        )

    def test_explicit_wings_platform_a3_resolves_to_a3(self):
        os.environ["WINGS_ASCEND_PLATFORM"] = "a3"
        self.assertEqual(
            vllm_adapter._resolve_deepseek_v4_flash_platform(self._params()),
            "a3",
        )

    def test_engine_version_a3_suffix_resolves_to_a3(self):
        """新能力：DSv4-Flash 现在也能感知 ENGINE_VERSION 后缀。"""
        os.environ["ENGINE_VERSION"] = "0.13.0rc3-a3"
        self.assertEqual(
            vllm_adapter._resolve_deepseek_v4_flash_platform(self._params()),
            "a3",
        )

    def test_engine_version_no_suffix_falls_back_to_a2(self):
        os.environ["ENGINE_VERSION"] = "0.13.0rc3"
        self.assertEqual(
            vllm_adapter._resolve_deepseek_v4_flash_platform(self._params()),
            "a2",
        )

    def test_legacy_ascend_a3_enable_still_works(self):
        """legacy ASCEND_A3_ENABLE=1 信号仍生效（不破坏既有部署）。"""
        os.environ["ASCEND_A3_ENABLE"] = "1"
        self.assertEqual(
            vllm_adapter._resolve_deepseek_v4_flash_platform(self._params()),
            "a3",
        )

    def test_device_details_a3_still_works(self):
        """device_details 注入信号仍生效。"""
        params = self._params()
        params["device_details"] = [{"name": "Atlas-910C-A3"}]
        self.assertEqual(
            vllm_adapter._resolve_deepseek_v4_flash_platform(params),
            "a3",
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
