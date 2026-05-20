"""模型元数据解析和架构识别辅助函数。

在命令生成路径中被复用。

功能概述:
    本模块提供模型元信息提取，用于引擎自动选择和参数默认值决策:
    - ModelIdentifier 类: 读取 config.json 并解析模型架构/类型/量化方式
    - 模型架构映射表: 以架构名为 key，映射到已验证模型列表
    - 模型类型分类: llm/embedding/rerank

支持的模型架构:
    - LLM:       DeepseekV3ForCausalLM, DeepseekV32ForCausalLM,
                 GlmMoeDsaForCausalLM,
                 Glm4ForCausalLM, Glm4MoeForCausalLM,
                 Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM,
                 Qwen3NextForCausalLM, Qwen3_5ForConditionalGeneration,
                 Qwen3_5MoeForConditionalGeneration, MiniMaxM2ForCausalLM,
                 LlamaForCausalLM
    - Embedding: XLMRobertaModel, BertModel, Qwen3ForCausalLM(Embedding)
    - Rerank:    XLMRobertaForSequenceClassification

Sidecar 架构契约:
    - 模型识别必须保持确定性（同参数同结果）
    - 解析器行为向后兼容
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Optional

from utils.file_utils import load_json_config

try:
    from wings_control.core.phase_d_loader import load_structured_file
except ImportError:
    from core.phase_d_loader import load_structured_file  # noqa: F811

logger = logging.getLogger(__name__)

_MODELS_INVENTORY_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "models_inventory.yaml"
)


def _load_models_inventory(path: Path = _MODELS_INVENTORY_PATH) -> tuple[
    dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], frozenset[str]
]:
    """Load model SKU tables and IndexCache architectures from the yaml carrier."""
    data = load_structured_file(path)
    entries = data.get("inventory", []) or []
    llm: dict[str, list[str]] = {}
    embedding: dict[str, list[str]] = {}
    rerank: dict[str, list[str]] = {}
    indexcache: set[str] = set()
    buckets = {"llm": llm, "embedding": embedding, "rerank": rerank}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        arch = entry.get("architecture")
        kind = entry.get("type")
        models = list(entry.get("models") or [])
        if not arch or kind not in buckets:
            continue
        buckets[kind][arch] = models
        if kind == "llm" and entry.get("indexcache"):
            indexcache.add(arch)
    return llm, embedding, rerank, frozenset(indexcache)


_LLM_MODELS, _EMBEDDING_MODELS, _RERANK_MODELS, INDEXCACHE_ARCHS = _load_models_inventory()

_GLM51_NAME_MARKERS = (
    "glm-5.1",
    "glm5.1",
    "glm_5.1",
    "glm 5.1",
    "glm-51",
    "glm51",
)

_MODEL_NAME_CONFIG_KEYS = (
    "_name_or_path",
    "name_or_path",
    "model_name",
    "model_id",
    "hub_model_id",
)


def _contains_glm51_marker(value: Any) -> bool:
    """Return True when a free-form metadata value clearly names GLM-5.1."""
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _GLM51_NAME_MARKERS)


def is_glm51_model(model_name: Any = None, model_path: Any = None,
                   config: Optional[dict] = None) -> bool:
    """Best-effort GLM-5.1 variant detection from stable metadata.

    ``GLM-5`` and ``GLM-5.1`` both use ``GlmMoeDsaForCausalLM`` in
    ``config.json``.  Architecture alone cannot distinguish them, so this
    helper checks explicit model-name sources first: user ``model_name``, model
    path, and common HuggingFace name fields in ``config.json``.
    """
    if _contains_glm51_marker(model_name) or _contains_glm51_marker(model_path):
        return True
    if isinstance(config, dict):
        for key in _MODEL_NAME_CONFIG_KEYS:
            if _contains_glm51_marker(config.get(key)):
                return True
    return False


def is_glm_moe_dsa_glm51(model_info: Any, model_name: Any = None,
                         model_path: Any = None) -> bool:
    """Return True for the GLM-5.1 variant of ``GlmMoeDsaForCausalLM``."""
    if getattr(model_info, "model_architecture", None) != "GlmMoeDsaForCausalLM":
        return False
    return is_glm51_model(
        model_name if model_name is not None else getattr(model_info, "model_name", None),
        model_path if model_path is not None else getattr(model_info, "model_path", None),
        getattr(model_info, "config", None),
    )

class ModelIdentifier:
    """模型元信息识别器，从模型目录的 config.json 提取架构、类型、量化信息。

    Attributes:
        model_name:         模型名称（用户传入）
        model_path:         模型权重目录路径
        model_type:         模型类型（'auto' 时自动推断）
        config:             从 config.json 加载的配置字典
        model_architecture: 模型架构名（如 'DeepseekV3ForCausalLM'）
        model_quantize:     量化方式（如 'fp8'、'bfloat16'）
        num_hidden_layers:  隐藏层数量（用于 CUDA Graph 计算）
    """
    def __init__(self, model_name: str, model_path: str, model_type: str):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.config = load_json_config(self.model_path / "config.json")
        self.model_architecture = self.identify_model_architecture()
        self.model_quantize = self.identify_model_quantize()
        self.num_hidden_layers = self.config.get("num_hidden_layers")
        self.model_dict = {
                "llm": _LLM_MODELS,
                "embedding": _EMBEDDING_MODELS,
                "rerank": _RERANK_MODELS
            }

    def identify_model_architecture(self) -> Optional[str]:
        """从 config.json 中提取模型架构名称。

        读取 architectures 字段的第一个元素，如 ["DeepseekV3ForCausalLM"].

        Returns:
            str: 模型架构名称，未找到时返回 'unknown_architecture'
        """
        # Read the 'architectures' list from model config
        architectures = self.config.get("architectures", [])
        if architectures:
            return architectures[0]
        else:
            return "unknown_architecture"

    def identify_model_type(self) -> Optional[str]:
        """推断模型类型（llm/embedding/rerank）。

        当 model_type 为 'auto'、空字符串或 None 时，根据 model_name 与内置映射表匹配;
        否则直接返回用户指定值。

        Returns:
            str | None: 模型类型，无法推断时返回 None
        """
        if self.model_type in ('auto', '', None):
            model_name = self.model_name.lower()
            for model_type, models in self.model_dict.items():
                support_model_name = []
                for lst in models.values():
                    support_model_name += [name.lower() for name in lst]
                if model_name in support_model_name:
                    return model_type
            # llm
            return "llm"
        return self.model_type


    def identify_model_quantize(self) -> Optional[str]:
        model_quantize = ""
        if "quantize" in self.config:
            model_quantize = self.config["quantize"]
        elif "quantization_config" in self.config:
            model_quantize = self.config["quantization_config"].get("quant_method", "")
        if model_quantize:
            return model_quantize
        else:
            return self.config.get("torch_dtype", "")


    def is_wings_supported(self):
        support_model_architecture = []
        for models in self.model_dict.values():
            support_model_architecture += list(models.keys())
        if self.model_architecture in support_model_architecture:
            return True
        else:
            return False


class ModelIdentifierDraft:
    """草稿模型识别机制"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.config = load_json_config(self.model_path / "config.json")
        self.draft_model_architecture = self.identify_model_architecture()
        self.model_draft_vocab_size = self.identify_draft_vocab_size()

    def identify_model_architecture(self) -> Optional[str]:
        """识别模型类型"""
        architectures = self.config.get("architectures", [])
        if architectures:
            return architectures[0]
        else:
            return "unknown_architecture"

    def identify_draft_vocab_size(self) -> Optional[bool]:
        """识别eagle3模型特有特征"""
        draft_vocab_size = self.config.get("draft_vocab_size", 0)
        if draft_vocab_size:
            return True
        else:
            return False


def is_qwen3_32b_nvfp4(model_path: str) -> bool:
    """判断模型是否为 Qwen3-32B-NVFP4 模型

    判断标准：
    1. 模型架构 architectures 为 Qwen3ForCausalLM
    2. config.json 中没有 quantization_config 字段
    3. 权重路径下存在 quant_model_description.json 文件

    Args:
        model_path: 模型权重路径

    Returns:
        bool: 如果是 Qwen3-32B-NVFP4 模型返回 True，否则返回 False
    """
    try:
        model_path_obj = Path(model_path)
        config = load_json_config(model_path_obj / "config.json")

        architectures = config.get("architectures", [])
        if not architectures or architectures[0] != "Qwen3ForCausalLM":
            logger.warning("is_qwen3_32b_nvfp4: architectures check failed"
                           " - architectures=%s, expected=['Qwen3ForCausalLM']", architectures)
            return False

        if "quantization_config" in config:
            logger.warning("is_qwen3_32b_nvfp4: quantization_config check failed"
                           " - quantization_config exists in config.json")
            return False

        quant_model_desc_path = model_path_obj / "quant_model_description.json"
        if not quant_model_desc_path.exists():
            logger.warning("is_qwen3_32b_nvfp4: quant_model_description.json check failed"
                           " - file not found at %s", quant_model_desc_path)
            return False

        return True

    except Exception as e:
        logger.warning("Failed to check if model is Qwen3-32B-NVFP4: %s", e)
        return False


def _is_deepseek_v3_modelslim_layout(model_path_obj: Path) -> bool:
    """判断模型目录是否符合 DeepSeek V3 ModelSlim 量化权重布局。"""
    try:
        config = load_json_config(model_path_obj / "config.json")

        architectures = config.get("architectures", [])
        expected_architectures = ["DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"]
        if not architectures or architectures[0] not in expected_architectures:
            logger.warning("is_deepseek_v3_modelslim_layout: architectures check failed"
                           " - architectures=%s, expected=%s", architectures, expected_architectures)
            return False

        if "quantization_config" in config:
            logger.warning("is_deepseek_v3_modelslim_layout: quantization_config check failed"
                           " - quantization_config exists in config.json")
            return False

        quant_model_desc_path = model_path_obj / "quant_model_description.json"
        if not quant_model_desc_path.exists():
            logger.warning("is_deepseek_v3_modelslim_layout: quant_model_description.json check failed"
                           " - file not found at %s", quant_model_desc_path)
            return False

        return True

    except Exception as e:
        logger.warning("Failed to check if model is DeepSeek V3 ModelSlim layout: %s", e)
        return False


def is_deepseek_series_modelslim_quant(model_path: str) -> bool:
    """判断模型是否为 DeepSeek V3 ModelSlim/Ascend 量化权重。

    ModelSlim 导出的 DeepSeek V3.1 W8A8 权重也会带
    quant_model_description.json。该布局只说明模型是 Ascend 量化权重，
    不能直接等同于 Soft FP8。
    """
    return _is_deepseek_v3_modelslim_layout(Path(model_path))


def is_deepseek_series_fp8(model_path: str) -> bool:
    """判断模型是否为 DeepSeek 系列 Soft FP8 模型。

    判断标准：
    1. 模型目录符合 DeepSeek V3 ModelSlim 量化布局
    2. quant_model_description.json 中存在明确 FP8/Float8 标记
    3. W8A8/INT8 等 Ascend 量化标记优先排除，避免误走 Soft FP8 分支

    Args:
        model_path: 模型权重路径

    Returns:
        bool: 如果是 DeepSeek 系列 Soft FP8 模型返回 True，否则返回 False
    """
    try:
        model_path_obj = Path(model_path)
        if not _is_deepseek_v3_modelslim_layout(model_path_obj):
            return False

        quant_desc_text = (model_path_obj / "quant_model_description.json").read_text(
            encoding="utf-8", errors="ignore"
        ).lower()
        if any(marker in quant_desc_text for marker in ("w8a8", "int8", "int4")):
            logger.info("is_deepseek_series_fp8: detected non-FP8 quant marker in quant_model_description.json")
            return False

        if any(marker in quant_desc_text for marker in ("fp8", "float8")):
            return True

        logger.info("is_deepseek_series_fp8: no explicit FP8 marker in quant_model_description.json")
        return False

    except Exception as e:
        logger.warning("Failed to check if model is DeepSeek series Soft FP8: %s", e)
        return False


def is_qwen3_series_fp8(model_path: str, model_name: str) -> bool:
    """判断模型是否为 Qwen3 系列 FP8 模型

    判断标准：
    1. 模型架构为 Qwen3ForCausalLM 或 Qwen3MoeForCausalLM
    2. 如果是 Qwen3MoeForCausalLM，则模型名称中不包含 '235'
    3. config.json 中没有 quantization_config 字段
    4. 权重路径下存在 quant_model_description.json 文件

    Args:
        model_path: 模型权重路径
        model_name: 模型名称

    Returns:
        bool: 如果是 Qwen3 系列 FP8 模型返回 True，否则返回 False
    """
    try:
        model_path_obj = Path(model_path)
        config = load_json_config(model_path_obj / "config.json")

        architectures = config.get("architectures", [])
        if not architectures:
            logger.warning("is_qwen3_series_fp8: architectures check failed - architectures is empty")
            return False

        is_qwen3 = architectures[0] == "Qwen3ForCausalLM"
        is_qwen3_moe = architectures[0] == "Qwen3MoeForCausalLM" and '235' not in model_name

        if not is_qwen3 and not is_qwen3_moe:
            return False

        if "quantization_config" in config:
            logger.warning("is_qwen3_series_fp8: quantization_config check failed"
                           " - quantization_config exists in config.json")
            return False

        quant_model_desc_path = model_path_obj / "quant_model_description.json"
        if not quant_model_desc_path.exists():
            logger.warning("is_qwen3_series_fp8: quant_model_description.json check failed"
                           " - file not found at %s", quant_model_desc_path)
            return False

        return True

    except Exception as e:
        logger.warning("Failed to check if model is Qwen3 series FP8: %s", e)
        return False
