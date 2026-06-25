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
from core.version_util import engine_version_platform

logger = logging.getLogger(__name__)

# IndexCache 支持的模型架构列表
# 当 KV 稀疏开启时，这些架构使用 IndexCache 策略；其他架构使用 FP8 KV CACHE
INDEXCACHE_ARCHS: frozenset[str] = frozenset({
    "GlmMoeDsaForCausalLM",
    "DeepseekV32ForCausalLM",
})

# ── Smart 三特性白名单（投机 spec / 稀疏 sparse / 卸载 offload）────────────────
# 每条 4-tuple = (engine, (模型名标识...), (卡型标识...|"*"), 允许特性 frozenset)
#   无 forced 列：只开关不强制（需求一 §0 裁定1，白名单只收窄、永不强开）。
#   模型名/卡型 = 小写子串，任一命中即可；卡型 "*" = 任意卡。
#   未命中（miss）→ 三特性一律不产（§0 裁定2：不管老模型）。
# 来源：反串讲 0430 兼容性列表 + 0DAYS(26.0.3)。新增模型（含 0day 适配）按此 4-tuple 追加。
#   ⚠ 版本子串重叠：更具体的型号行须排在更泛的型号行之前（首个命中即返回）。
#     例：glm-5.1 行在 glm-5 基座行之前；deepseek-v3.2 行在 deepseek-v3.1 行之前。
SMART_FEATURE_WHITELIST: tuple = (
    # ── NVIDIA (vllm) ──  卡：NRP0500(72G) / NH02(141G)；NV 卡码是否进 key 待定，现用 "*"
    ("vllm",        ("qwen3.5-397b", "qwen3_5-397b"),   ("*",),           frozenset({"spec", "sparse"})),
    ("vllm",        ("glm-4.7",),                       ("*",),           frozenset({"spec", "sparse", "offload"})),
    ("vllm",        ("glm-5.1", "glm5.1"),              ("*",),           frozenset({"spec", "sparse"})),
    ("vllm",        ("glm-5", "glm5"),                  ("*",),           frozenset({"spec", "sparse", "offload"})),  # GLM-5 基座（5.1 已在上、先匹配）
    ("vllm",        ("minimax-m2.7", "minimax-m27"),    ("*",),           frozenset({"spec", "sparse", "offload"})),
    ("vllm",        ("deepseek-v4-flash", "v4-flash"),  ("*",),           frozenset({"spec", "sparse", "offload"})),  # V4-Flash·NV day0（原 §6-④）
    # ── Ascend (vllm_ascend) ──  卡：910B3(64G) / 910C(128G)
    ("vllm_ascend", ("glm-4.7",),                       ("910b", "910c"), frozenset({"spec", "offload"})),
    ("vllm_ascend", ("minimax-m2.5", "minimax-m25"),    ("910b", "910c"), frozenset({"spec", "offload"})),
    ("vllm_ascend", ("deepseek-v3.2", "deepseek_v3.2"), ("910c",),        frozenset({"spec", "offload"})),
    ("vllm_ascend", ("deepseek-v4-flash", "v4-flash"),  ("910b", "910c"), frozenset({"spec", "offload"})),
    ("vllm_ascend", ("deepseek-v4-pro", "v4-pro"),      ("910b", "910c"), frozenset({"spec", "offload"})),
    # DeepSeek-V3/V3.1/R1（DeepseekV3ForCausalLM）；token 不含裸 "deepseek-v3" 以免误伤 v3.2
    ("vllm_ascend", ("deepseek-v3.1", "deepseek_v3.1",
                     "deepseek-r1", "deepseek_r1"),     ("910b", "910c"), frozenset({"spec", "offload"})),
    # 曾 forced，已降为普通开关门控（§0 裁定1）
    ("vllm_ascend", ("glm-5.1", "glm5.1"),              ("910b", "910c"), frozenset({"sparse"})),
)


def resolve_feature_whitelist(engine, model_name, model_path, card_token):
    """返回 (engine, model, card) 命中的允许特性 frozenset；未命中返回空集。

    无 forced（需求一 §0 裁定1：只开关不强制，白名单只收窄、永不强开）。
    匹配维度：engine 精确 → 模型名/路径子串任一命中 → 卡型 "*" 或子串任一命中。
    """
    hay = " ".join(str(x).lower() for x in (model_name, model_path) if x)
    ct = (card_token or "").lower()
    for wl_engine, name_tokens, card_tokens, feats in SMART_FEATURE_WHITELIST:
        if wl_engine != engine:
            continue
        if not any(tok in hay for tok in name_tokens):
            continue
        if not (("*" in card_tokens) or any(c in ct for c in card_tokens)):
            continue
        return feats
    return frozenset()


def feature_allowed(engine, model_name, model_path, card_token, feature) -> bool:
    """白名单门控：该 (engine, model, card) 是否允许某特性（spec/sparse/offload）。"""
    return feature in resolve_feature_whitelist(engine, model_name, model_path, card_token)

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


# ── [GLM5.1-Ascend-Tmp] TEMPORARY: GLM-5.1 + vllm_ascend KV-Sparse Whitelist ──
# Scope: engine == "vllm_ascend" AND is_glm_moe_dsa_glm51(...)
# Behavior gated by this predicate:
#   - vllm_adapter._build_kv_sparse_cmd: 走 IndexCache --hf-overrides 分支
#     （不写 engine_config、不触发 indexcache 补丁安装）
# 注：原 _force_kv_sparse_for_glm51_ascend「强制开（即便 enable_sparse=False 也产）」
#    已按需求一 §0 裁定1 删除；GLM-5.1·Ascend(sparse) 现降为普通开关门控
#    （开关 on 且命中白名单才产，由 config_loader.apply_effective_feature_enablement 收口）。
# 范围说明：仅 GLM-5.1（架构 GlmMoeDsaForCausalLM + 名称/路径标记 5.1），
# GLM-5 (非 5.1)、DeepseekV32 在 ascend 上不进入 IndexCache。
# 移除时机：vllm-ascend 支持 indexcache 补丁安装时一次性拆除。
# 移除方法：grep "[GLM5.1-Ascend-Tmp]" 一次性定位所有触点。
def is_glm51_ascend_kvsparse_tmp_scope(model_info: Any, engine: Any,
                                       model_name: Any = None,
                                       model_path: Any = None) -> bool:
    """[GLM5.1-Ascend-Tmp] Return True for vllm_ascend + GLM-5.1（单机/双机均适用）."""
    if engine != "vllm_ascend":
        return False
    return is_glm_moe_dsa_glm51(model_info, model_name=model_name, model_path=model_path)


# ── GLM-5.2 识别（架构同 GlmMoeDsaForCausalLM，与 GLM-5/5.1 靠名称/路径区分）──
# GLM-5 / GLM-5.1 / GLM-5.2 的 config.json architectures[0] 同为 GlmMoeDsaForCausalLM，
# 架构维度无分辨力。GLM-5.2 在 wings 上需与 GLM-5/5.1 反向处理（MTP num=3、A3 双机保留
# additional_config），故用与 is_glm51_model 同范式、互斥的名称标识切出：
#   5.2 标识 {glm-5.2, glm5.2, glm_5.2, glm 5.2, glm-52, glm52} 与
#   5.1 标识 {glm-5.1, ..., glm-51, glm51} 不相交，亦不命中 GLM-5.0 基座（不含 5.2/52）。
_GLM52_NAME_MARKERS = (
    "glm-5.2",
    "glm5.2",
    "glm_5.2",
    "glm 5.2",
    "glm-52",
    "glm52",
)


def _contains_glm52_marker(value: Any) -> bool:
    """Return True when a free-form metadata value clearly names GLM-5.2."""
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _GLM52_NAME_MARKERS)


def is_glm52_model(model_name: Any = None, model_path: Any = None,
                   config: Optional[dict] = None) -> bool:
    """Best-effort GLM-5.2 variant detection from stable metadata.

    与 :func:`is_glm51_model` 同范式：先看 ``model_name`` / ``model_path``，再看
    ``config.json`` 常见名字段（``_name_or_path`` 等）。标识集与 GLM-5.1 严格互斥，
    且不会误命中 GLM-5.0 基座。
    """
    if _contains_glm52_marker(model_name) or _contains_glm52_marker(model_path):
        return True
    if isinstance(config, dict):
        for key in _MODEL_NAME_CONFIG_KEYS:
            if _contains_glm52_marker(config.get(key)):
                return True
    return False


def is_glm52_single_node_even(source: Optional[dict]) -> bool:
    """GLM-5.2 + vllm_ascend + 单机(nnodes==1) + 偶数卡 + **a3(910C)** 的统一判定（共享真相源）。

    config_loader._set_parallelism_params(用 ctx)与
    vllm_adapter._apply_generic_deepseek_ascend_dp_defaults(用 params)共用此判定，
    确保「config_loader 让位 ⇔ adapter 接管 TP=device_count//DP + DP2」的条件逐字一致，
    不会因两处各写一份布尔链而漂移。``source`` 须含 engine / nnodes / device_count /
    model_name / model_path 键（ctx 与 params 同构）。

    平台门控：单机 TP 减半 + DP2 是 **910C(a3) 官方 recipe**，仅 a3 触发；910B(a2) 上触发
    GLM-5.2 不进此分支，回落通用 GlmMoeDsa 单机整卡 TP（DP1）。平台**只按 engine-version
    后缀**判定（``-a3``），复用 :func:`core.version_util.engine_version_platform`——它读全局
    ``ENGINE_VERSION``，ctx / params 两侧取值一致，不依赖各自字典里是否带平台键。
    """
    src = source or {}
    if src.get("engine") != "vllm_ascend":
        return False
    try:
        nnodes = int(src.get("nnodes") or 1)
        device_count = int(src.get("device_count") or 0)
    except (TypeError, ValueError):
        return False
    if nnodes != 1 or device_count <= 0 or device_count % 2 != 0:
        return False
    if engine_version_platform() != "a3":
        return False
    return is_glm52_model(src.get("model_name"), src.get("model_path"))


#
_LLM_MODELS = {
    "DeepseekV3ForCausalLM": [
        "DeepSeek-R1",
        "DeepSeek-R1-0528",
        "DeepSeek-V3",
        "DeepSeek-V3-0324",
        "DeepSeek-V3.1",
        "DeepSeek-Coder-V2-Instruct",
        "DeepSeek-R1-w8a8",
        "DeepSeek-R1-0528-w8a8",
        "DeepSeek-V3-w8a8",
        "DeepSeek-V3-0324-w8a8",
        "DeepSeek-V3.1-w8a8",
        "DeepSeek-Coder-V2-Instruct-w8a8"
        ],
    # DeepSeek-V4 系列权重的 ``config.json -> architectures[0]`` 为
    # ``DeepseekV4ForCausalLM``。独立架构键保证 ascend_default.json 下的
    # V4-Flash / V4-Pro 默认值（``method: mtp``、``enable_cpu_binding: bool true``、
    # ``multistream_overlap_shared_expert: false`` 等）能够命中。
    "DeepseekV4ForCausalLM": [
        "DeepSeek-V4",
        "DeepSeek-V4-w8a8",
        "DeepSeek-V4-Flash",
        "DeepSeek-V4-Flash-w8a8-mtp",
        "DeepSeek-V4-Pro",
        "DeepSeek-V4-Pro-w4a8-mtp",
        ],
    "KimiK25ForConditionalGeneration": [
        "Kimi-K2.5",
        "Kimi-K2.5-w4a8",
        "Kimi-K2.7",
        "Kimi-K2.7-w4a8",
        "Kimi-K2.7-Code",
        "Kimi-K2.7-Code-w4a8",
        ],
    "DeepseekV32ForCausalLM": [
        "DeepSeek-V3.2",
        "DeepSeek-V3.2-w8a8",
        "DeepSeek-V3.2-Exp",
        ],
    "Glm4ForCausalLM": [
        "GLM-4-9B-0414"
        ],
    "GlmMoeDsaForCausalLM": [
        "GLM-5",
        "GLM-5-FP8",
        "GLM-5-w4a8",
        "GLM-5.1",
        "GLM-5.1-w8a8",
        "GLM-5.1-FP8",
        "GLM-5.2",
        "GLM-5.2-w8a8",
        "GLM-5.2-FP8"
        ],
    "Glm4MoeForCausalLM": [
        "GLM-4.7",
        "GLM-4.7-w8a8"
        ],
    "Qwen2ForCausalLM": [
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
        "DeepSeek-R1-Distill-Qwen-32B",
        "Qwen2.5-32B-Instruct",
        "QwQ-32B"
        ],
    "Qwen3ForCausalLM": [
        "Qwen3-32B"
        ],
    "Qwen3MoeForCausalLM": [
        "Qwen3-30B-A3B",
        "Qwen3-235B-A22B"
        ],
    "Qwen3NextForCausalLM": [
        "Qwen3-Next-80B-A3B-Instruct"
        ],
    "Qwen3_5ForConditionalGeneration": [
        "Qwen3.5-27B",
        "Qwen3.5-27B-w8a8",
        "Qwen3.6-27B",
        "Qwen3.6-27B-w8a8"
        ],
    "Qwen3_5MoeForConditionalGeneration": [
        "Qwen3.5-397B-A17B-w8a8",
        "Qwen3.5-397B-A17B",
        "Qwen3.6-35B-A3B",
        "Qwen3.6-35B-A3B-w8a8"
        ],
    "MiniMaxM2ForCausalLM": [
        "MiniMax-M2.5",
        "MiniMax-M2.5-w8a8",
        "MiniMax-M2.7",
        "MiniMax-M2.7-w8a8"
        ],
    "MiniMaxM3SparseForConditionalGeneration": [
        "MiniMax-M3",
        "MiniMax-M3-MXFP8",
        ],
    "LlamaForCausalLM": [
        "LLaMA3-8B",
        "LLaMA3.1-70B",
        "LLaMA3.1-70B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Llama-70B"
        ]
}

_EMBEDDING_MODELS = {
    "XLMRobertaModel": [
        "bge-m3"
        ],
    "BertModel": [
        "bge-large-zh-v1.5"
        ],
    "Qwen3ForCausalLM": [
        'Qwen3-Embedding-0.6B'
        ]
}

_RERANK_MODELS = {
    "XLMRobertaForSequenceClassification": [
        "bge-reranker-v2-m3",
        "bge-reranker-large"
        ]
}


# ---------------------------------------------------------------------------
# Thinking-mode 策略（供 --enable-auto-think-choice 在启动命令注入服务级默认思考状态）
# ---------------------------------------------------------------------------
# 背景：reasoning_parser 只控制服务端是否「解析」思维链，控制不了模型「是否思考」。
# 生成端（vllm/vllm_ascend）按开关在启动命令注入 --default-chat-template-kwargs 设服务级默认：
#   - 开关开 → {key: true}  强制默认【打开】思考；开关关 → {key: false} 强制默认【关闭】思考。
#   （见 config_loader._set_thinking_default；本函数只负责解析【键名】，布尔值由开关决定。）
# 注：这是服务级默认，请求级 chat_template_kwargs 优先级更高、可覆盖（客户端自负，不兜底）。
# 不同模型族的键名不同（已对齐各家官方/vLLM）：
#   - Qwen3 系列 / GLM-4.5+ MoE 系列 → enable_thinking
#   - DeepSeek-V3.1 / V3.2（混合推理）→ thinking
# 始终推理模型（R1 / QwQ / MiniMax-M2 等）无法切换思考，返回 THINKING_ALWAYS_ON 仅用于告警。

# resolve_thinking_off_policy 的 mode 取值（恒为 str，配合恒为 dict 的 off_kwargs，
# 使所有分支返回的类型与个数一致，符合 G.CTL.01）。
THINKING_HYBRID = "hybrid"          # 混合推理：off_kwargs 为关闭思考所需 chat_template_kwargs
THINKING_ALWAYS_ON = "always_on"    # 始终推理：无法关闭思考，off_kwargs 为空，仅告警
THINKING_NONE = "none"              # 非思考模型 / 无法识别：不介入，off_kwargs 为空
# 始终推理（无法关闭思考）模型名片段，优先于混合推理判断。
_ALWAYS_ON_THINKING_TOKENS = ("r1", "qwq", "minimax-m2")


def resolve_thinking_off_policy(model_name: str):
    """根据模型名解析思考策略，返回 (mode, off_kwargs)。

    Returns:
        tuple[str, dict]: 二元组，所有分支类型与个数一致——
            mode       : THINKING_HYBRID / THINKING_ALWAYS_ON / THINKING_NONE 之一；
            off_kwargs : 仅 THINKING_HYBRID 时非空，为强制关闭思考需注入/覆盖的
                         chat_template_kwargs（如 {"enable_thinking": False}）；
                         THINKING_ALWAYS_ON / THINKING_NONE 时恒为空 dict。

    注：取值按模型名子串匹配（忽略大小写、下划线归一为连字符），与各家官方
    chat 模板键名对齐；R1/蒸馏 R1/QwQ/MiniMax-M2 等优先判定为 THINKING_ALWAYS_ON。
    """
    if not model_name:
        return THINKING_NONE, {}
    name = model_name.lower().replace("_", "-")

    # 1) 始终推理模型优先（R1 / R1-Distill / QwQ / MiniMax-M2）
    if any(tok in name for tok in _ALWAYS_ON_THINKING_TOKENS):
        return THINKING_ALWAYS_ON, {}

    # 2) Qwen3 系列（混合推理）：enable_thinking
    #    排除 Qwen3-Coder-*：官方为非思考模型，reasoning_parser_support.yaml 显式置 null，
    #    不应注入思考默认（否则开关开时会把非思考的 Coder 误强制思考）。
    if "qwen3" in name and "coder" not in name:
        return THINKING_HYBRID, {"enable_thinking": False}

    # 3) GLM MoE 混合推理（GLM-4.5/4.6/4.7/5/5.1）：enable_thinking
    #    排除非思考的 glm-4-9b 等（不含 4.5+ 版本号则不匹配）。
    if any(tag in name for tag in ("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")):
        return THINKING_HYBRID, {"enable_thinking": False}

    # 4) DeepSeek V3.1/V3.2/V4 + Kimi-K2.x（混合推理）：thinking 键（注意不是 enable_thinking）
    #    - DeepSeek V4-Flash/-Pro：官方 vLLM Recipes 确认混合推理、键名 thinking；另有
    #      reasoning_effort=high/max 控制思考【深度】，属请求级、与本开/关正交，不在此注入。
    #    - Kimi-K2.x：moonshotai/Kimi-K2.5 chat_template.jinja 字面以
    #      `{% if thinking is defined and thinking is false %}` 关闭思考（键名 thinking，
    #      非 enable_thinking），混合推理（思考/即时双模）；与 reasoning_parser=kimi_k2 对应。
    if "deepseek-v3" in name or "deepseek-v4" in name or "kimi-k2" in name:
        return THINKING_HYBRID, {"thinking": False}

    # 其余（Qwen2.5 / Llama / GLM-4-9B / embedding / rerank 等非思考模型）→ 无需处理
    return THINKING_NONE, {}


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
