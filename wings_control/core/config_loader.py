"""多层配置加载与合并器 ── launcher 控制平面最大的单个模块。

Architecture:
    负责把多层配置源（硬件默认、引擎默认、模型专属、用户自定义、CLI 参数）
    合并为一份统一的参数字典，提供给 engine adapter 使用。

Config Merge Priority (low -> high):
    1. 硬件默认配置 (e.g., config/vllm_default.json)
    2. 模型专属配置 (model_deploy_config 匹配)
    3. 用户自定义配置 (--config-file 指定的 JSON)
    4. CLI 参数 / 环境变量覆盖

Key Responsibilities:
    - 引擎自动选择（_auto_select_engine）
    - 参数名映射（engine_parameter_mapping.json）
    - 张量并行度自动设置
    - PD 分离 / LMCache / Router / Soft FP8 等高级特性注入
    - 分布式参数注入（Ray / NIXL / HCCL）
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple
import argparse
import json
import logging
import math
import os
from pathlib import Path

import yaml

from utils.env_utils import get_master_ip, get_node_ips, get_lmcache_env, get_pd_role_env, \
    get_config_force_env, get_soft_fp8_env, get_soft_fp4_env, get_speculative_decoding_env, \
    get_vllm_distributed_port, get_sglang_distributed_port, get_router_env, \
    get_router_instance_group_name_env, get_router_instance_name_env, get_router_nats_path_env, \
    get_operator_acceleration_env, get_local_ip
from utils.file_utils import check_torch_dtype, get_directory_size, check_permission_640, load_json_config
from utils.model_utils import (ModelIdentifier, is_qwen3_32b_nvfp4, is_deepseek_series_fp8,
                               is_deepseek_series_modelslim_quant, is_qwen3_series_fp8,
                               is_glm_moe_dsa_glm51, resolve_thinking_off_policy,
                               THINKING_ALWAYS_ON, THINKING_HYBRID, THINKING_NONE)
from utils.device_utils import check_pcie_cards

logger = logging.getLogger(__name__)


# 解析默认配置目录路径（优先级：环境变量 > 包内自带 > 硬编码回退）
def _resolve_default_config_dir() -> str:
    """解析默认配置目录，放在此目录下的配置文件为引擎提供默认参数。

    查找顺序：
    1. WINGS_CONFIG_DIR 环境变量（支持部署时重定向）
    2. 包内的 app/config/ 目录（安装部署场景）
    3. "wings/config" 硬编码回退（兼容旧版目录结构）
    """
    env_dir = os.getenv("WINGS_CONFIG_DIR", "").strip()
    if env_dir:
        return env_dir
    bundled_dir = Path(__file__).resolve().parents[1] / "config" / "defaults"
    if bundled_dir.exists():
        return str(bundled_dir)
    return "wings/config"


# 配置目录单例（模块加载时解析一次）
DEFAULT_CONFIG_DIR = _resolve_default_config_dir()
REASONING_PARSER_SUPPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "features"
    / "reasoning_parser"
    / "reasoning_parser_support.yaml"
)

# 各设备类型和引擎对应的默认配置文件名映射
DEFAULT_CONFIG_FILES = {
    "nvidia": "vllm_default.json",
    "ascend": "vllm_default.json",
    "distributed": "distributed_config.json",
    "engine_parameter_mapping": "engine_parameter_mapping.json",
    # Engine-specific fallback defaults (used when vllm_default.json
    # has no model-level section for the selected engine)
    "sglang": "sglang_default.json",
    "mindie": "mindie_default.json",
}

SUPPORTED_DEVICE_TYPES = {"nvidia", "ascend"}


def _load_mapping(config_path: str, mapping_key: str) -> Dict[str, Any]:
    """从 JSON 配置文件中安全加载指定 key 下的映射字典。

    如果文件不存在、内容为空或格式不对，均返回空字典并打印警告。
    用于加载参数名映射表（如 default_to_vllm_parameter_mapping）。
    """
    cfg = load_json_config(config_path)
    mapping = cfg.get(mapping_key, {})
    if not isinstance(mapping, dict):
        logger.warning(
            "Invalid mapping format: key=%s file=%s type=%s; fallback to empty mapping",
            mapping_key,
            config_path,
            type(mapping).__name__,
        )
        return {}
    if not mapping:
        logger.warning("Missing/empty mapping key=%s in file=%s", mapping_key, config_path)
    return mapping


def _extract_node_ips(node_ips: str | None) -> list[str]:
    """Normalize a node IP CSV string into a clean list."""
    if not node_ips:
        return []
    return [ip.strip() for ip in node_ips.split(",") if ip.strip()]


def _resolve_distributed_node_count(node_ips: str | None, nnodes: int | None) -> int:
    """Prefer explicit node IP topology, then fall back to nnodes, then env."""
    explicit_node_ips = _extract_node_ips(node_ips)
    if explicit_node_ips:
        return len(explicit_node_ips)
    if nnodes and int(nnodes) > 0:
        return int(nnodes)
    env_node_ips = _extract_node_ips(get_node_ips())
    if env_node_ips:
        return len(env_node_ips)
    return 1


_VLLM_ADVANCED_FEATURE_ENGINES = {"vllm", "vllm_ascend"}


def _is_vllm_advanced_feature_engine(params: Dict[str, Any]) -> bool:
    """Return whether selected engine supports vLLM advanced feature switches."""
    return str(params.get("engine", "")).lower() in _VLLM_ADVANCED_FEATURE_ENGINES


def _set_spec_decoding_config(params):
    """配置推测解码（Speculative Decoding）相关环境变量。

    根据 params 中 enable_speculative_decode 的值设置 SD_ENABLE 环境变量，
    供引擎启动脚本和后续流程判断是否启用推测解码。
    """
    enabled = bool(params.get("enable_speculative_decode"))
    if enabled and not _is_vllm_advanced_feature_engine(params):
        logger.warning(
            "Spec Decoding is only wired for vllm/vllm_ascend; disabled for engine=%s",
            params.get("engine"),
        )
        params["enable_speculative_decode"] = False
        enabled = False

    if enabled:
        os.environ['SD_ENABLE'] = 'true'
        logger.info("Spec Decoding for vllm is enabled")
    else:
        os.environ['SD_ENABLE'] = 'false'


def _set_sparse_config(params):
    """配置稀疏 KV（Sparse KV Cache）相关环境变量。

    根据 params 中 enable_sparse 的值设置 SPARSE_ENABLE 环境变量，
    供引擎启动脚本和后续流程判断是否启用稀疏 KV。
    """
    enabled = bool(params.get("enable_sparse"))
    if enabled and not _is_vllm_advanced_feature_engine(params):
        logger.warning(
            "Sparse KV is only wired for vllm/vllm_ascend; disabled for engine=%s",
            params.get("engine"),
        )
        params["enable_sparse"] = False
        enabled = False

    if enabled:
        os.environ['SPARSE_ENABLE'] = 'true'
        logger.info("Sparse for vllm is enabled")
    else:
        os.environ['SPARSE_ENABLE'] = 'false'


def _set_rag_acc_config(params):
    """配置 RAG 加速相关环境变量。

    根据 params 中 enable_rag_acc 的值设置 RAG_ACC_ENABLED 环境变量，
    供 wings_entry（advanced_features.json）和 proxy 层判断是否启用 RAG 加速。

    兜底逻辑：当直接运行 python -m wings_control（跳过 wings_start.sh）时，
    wings_start.sh 中的 export RAG_ACC_ENABLED 不会被执行，此处确保 Python 层
    也能正确设置该环境变量，与 _set_spec_decoding_config / _set_sparse_config 对齐。
    """
    if params.get("enable_rag_acc"):
        os.environ['RAG_ACC_ENABLED'] = 'true'
        logger.info("RAG acceleration is enabled")
    else:
        os.environ.setdefault('RAG_ACC_ENABLED', 'false')


def _get_h20_model_hint() -> str:
    """获取 H20 GPU 型号提示（用于 DeepSeek 模型的卡型专属配置）。

    H20 GPU 有两种型号：H20-96G 和 H20-141G，显存不同导致最优参数不同。
    通过 WINGS_H20_MODEL 环境变量显式指定，返回空串表示未指定或无效。
    """
    hint = os.getenv("WINGS_H20_MODEL", "").strip()
    if not hint:
        return ""
    if hint in ("H20-96G", "H20-141G"):
        return hint
    logger.warning("Invalid WINGS_H20_MODEL=%s, expected H20-96G or H20-141G", hint)
    return ""


def _check_vram_requirements(weight_path: str, hardware_env: Dict[str, Any], nodes_count: int) -> None:
    """检查总可用显存是否足以加载模型权重。

    将模型目录总大小与全部节点的可用显存总和对比：
    - 总显存 < 模型大小                → WARNING （可能 OOM）
    - 总显存 < 模型大小 × 1.5            → WARNING （性能不佳）
    - 总显存 ≥ 模型大小 × 1.5            → INFO   （充裕）

    Args:
        weight_path:  模型权重目录路径
        hardware_env: 硬件环境信息（含 details 列表，每项含 free_memory）
        nodes_count:  分布式节点总数，用于计算跨节点总显存
    """
    if not os.path.exists(weight_path):
        logger.warning("Model weight path not found: %s", weight_path)
        return

    weight_size_bytes = get_directory_size(weight_path)
    weight_size_gb = weight_size_bytes / (1024 ** 3)

    if not hardware_env.get("details"):
        logger.info("No VRAM details available (expected in sidecar mode), skipping VRAM check")
        return

    # 如果 details 中缺少 free_memory 字段（只有 name），跳过 VRAM 检查
    if not all("free_memory" in d for d in hardware_env["details"]):
        logger.info("VRAM details lack free_memory field, skipping VRAM check")
        return

    # VRAM
    free_vram_per_node = sum(d["free_memory"] for d in hardware_env["details"])
    total_free_vram = free_vram_per_node * nodes_count

    if total_free_vram < weight_size_gb:
        logger.warning(
            f"Insufficient VRAM: Required {weight_size_gb:.2f}GB, "
            f"but only {total_free_vram:.2f}GB available "
            f"({nodes_count} nodes  {free_vram_per_node:.2f}GB each)"
        )
    elif total_free_vram < weight_size_gb * 1.5:
        logger.warning(
            f"Performance warning: Total VRAM ({total_free_vram:.2f}GB) is less than 1.5x "
            f"model weight size ({weight_size_gb:.2f}GB) "
            f"({nodes_count} nodes  {free_vram_per_node:.2f}GB each)"
        )
    else:
        logger.info(
            f"VRAM check: Total VRAM ({total_free_vram:.2f}GB) is more than 1.5x "
            f"model weight size ({weight_size_gb:.2f}GB) "
            f"({nodes_count} nodes  {free_vram_per_node:.2f}GB each)"
        )


def _build_common_context(hardware_env: Dict[str, Any],
                          cmd_known_params: Dict[str, Any],
                          model_info) -> Dict[str, Any]:
    """从硬件环境和 CLI 参数构建通用上下文字典。"""
    return {
        "device": hardware_env.get("device"),
        "device_details": hardware_env.get("details"),
        "device_count": cmd_known_params.get("device_count", 1),
        "engine": cmd_known_params.get("engine"),
        "distributed": cmd_known_params.get("distributed"),
        "nnodes": cmd_known_params.get("nnodes", 1),
        "node_ips": cmd_known_params.get("node_ips", ""),
        "model_type": model_info.identify_model_type(),
        "gpu_usage_mode": cmd_known_params.get("gpu_usage_mode", "full"),
        "distributed_executor_backend": cmd_known_params.get("distributed_executor_backend"),
    }


def _build_engine_cmd_parameter(cmd_known_params: Dict[str, Any]) -> Dict[str, Any]:
    """从 CLI 参数字典提取引擎级参数。"""
    keys = [
        "host", "port", "model_name", "model_path", "input_length", "output_length",
        "trust_remote_code", "dtype", "kv_cache_dtype", "quantization",
        "quantization_param_path", "gpu_memory_utilization", "enable_chunked_prefill",
        "block_size", "max_num_seqs", "seed", "enable_expert_parallel",
        "max_num_batched_tokens", "enable_prefix_caching", "enable_speculative_decode",
        "speculative_decode_model_path",
        "enable_rag_acc", "enable_auto_tool_choice",
        "enable_auto_think_choice",
        "enable_sparse",
    ]
    return {k: cmd_known_params.get(k) for k in keys}


def _merge_cmd_params(hardware_env, engine_specific_defaults, cmd_known_params, model_info):
    """将硬件上下文、引擎默认参数和用户 CLI 参数三层合并。

    该函数是配置合并的核心入口：
    1. 从 hardware_env 和 cmd_known_params 抽取通用上下文（device、分布式、模型类型等）
    2. 从 cmd_known_params 抽取引擎级参数（host、port、dtype、quantization 等）
    3. 根据 engine 类型分发到 _merge_vllm_params / _merge_mindie_params / _merge_sglang_params

    Args:
        hardware_env:             硬件环境信息
        engine_specific_defaults: 从默认配置文件加载的引擎参数
        cmd_known_params:         用户 CLI 参数
        model_info:               模型元信息对象

    Returns:
        合并后的引擎参数字典
    """
    common_context = _build_common_context(hardware_env, cmd_known_params, model_info)
    engine_cmd_parameter = _build_engine_cmd_parameter(cmd_known_params)

    # 根据引擎类型分发到不同的参数合并函数
    engine = common_context["engine"]
    # sglang / mindie 触发思考开关时仅提醒（启动期无法切换思考，详见函数说明）
    _warn_thinking_switch_unsupported_engine(engine, engine_cmd_parameter)
    # 将嵌套 dict 型配置值序列化为 JSON 字符串，便于作为 CLI 参数传递
    engine_specific_defaults = {
        k: json.dumps(v) if isinstance(v, dict) else v
        for k, v in engine_specific_defaults.items()
    }
    if engine in ("vllm", "vllm_ascend"):
        return _merge_vllm_params(engine_specific_defaults, common_context, engine_cmd_parameter, model_info)
    elif engine == "mindie":
        return _merge_mindie_params(engine_specific_defaults, common_context, engine_cmd_parameter, model_info)
    elif engine == "sglang":
        return _merge_sglang_params(engine_specific_defaults, common_context, engine_cmd_parameter)
    return engine_specific_defaults


def _merge_vllm_params(params, ctx, engine_cmd_parameter, model_info):
    """合并 vLLM / vLLM-Ascend 引擎专属参数。

    调用多个 setter 函数将硬件上下文、引擎配置和模型信息合并到 params 字典。

    调用链路:
        1. _set_common_params       → 根据参数映射表翻译 CLI 参数
        2. _set_sequence_length     → 合并序列长度（embedding/rerank 只用 input_length）
        3. _set_parallelism_params  → 设置张量并行度
        4. _set_kv_cache_config     → LMCache / PD 分离 KV Transfer 配置
        5. _guard_pd_hybrid_kv_cache → PD 模式移除不兼容的 hybrid KV flag
        6. _ensure_pd_head_dim      → PD 模式补全 config.json 缺失的 head_dim
        7. _set_router_config       → Wings Router NATS 配置
        8. _set_operator_acceleration → 昇腾算子加速
        9. _set_soft_fp8            → Soft FP8 量化配置
        10. _set_task               → embedding/rerank 任务类型

    Args:
        params:              当前引擎参数字典（会被原地修改）
        ctx:                 通用上下文（device、device_count、distributed 等）
        engine_cmd_parameter: 用户 CLI 传入的引擎参数
        model_info:          模型元信息对象

    Returns:
        Dict[str, Any]: 合并后的引擎参数字典
    """
    # 加载引擎参数名映射表
    engine_param_map_config_path = os.path.join(
        DEFAULT_CONFIG_DIR,
        DEFAULT_CONFIG_FILES.get("engine_parameter_mapping")
    )

    #
    _set_common_params(params, engine_cmd_parameter, engine_param_map_config_path)
    _set_function_call(params, engine_cmd_parameter)
    _set_reasoning_parser(params, engine_cmd_parameter)
    _set_thinking_default(params, engine_cmd_parameter, model_info)
    _set_sequence_length(params, engine_cmd_parameter, model_type=ctx.get("model_type", "llm"))
    _set_parallelism_params(params, ctx)
    _set_kv_cache_config(params, ctx, model_info)
    _guard_pd_hybrid_kv_cache(params)
    _ensure_pd_head_dim(params, model_info)
    _set_router_config(params)
    _set_operator_acceleration(params, ctx)
    if not _set_deepseek_v3_family_ascend_quant_params(params, ctx, model_info):
        _set_soft_fp8(params, ctx, model_info)
    _set_soft_fp4(params, ctx, model_info)
    _set_task(params, ctx)

    # 对于 embedding 和 rerank 模型，强制禁用 enable_chunked_prefill 和 enable_prefix_caching
    _validate_embedding_rerank_params(params, ctx)

    return params


def _set_deepseek_v3_family_ascend_quant_params(params, ctx, model_info) -> bool:
    """识别 DeepSeek V3-family Ascend W8A8/ModelSlim 官方量化路径。

    DeepSeek V3/V3.1/V3.2 W8A8/ModelSlim 属于官方 Ascend 量化路径，
    不是通用 Soft FP8 fallback。这里仅标记量化方式并阻断 Soft FP8
    误判，不再按官方示例逐项注入 vLLM 启动字段。
    """
    model_path = model_info.model_path
    model_name = model_info.model_name

    if ctx.get('device') != "ascend":
        return False
    if not _is_deepseek_v3_family_model(model_name, model_path):
        return False
    if not is_deepseek_series_modelslim_quant(model_path):
        return False

    explicit_keys = _detect_explicit_cli_keys()

    if 'quantization' not in explicit_keys:
        params['quantization'] = 'ascend'
    logger.info("DeepSeek V3-family Ascend W8A8/ModelSlim quantization path detected")
    return True


def _set_function_call(params, engine_cmd_parameter):
    """根据用户传入或模型默认配置中的 enable_auto_tool_choice 统一启用 function call。

    触发源：CLI 显式开关 **或** model_deploy_config 中明确写 ``enable_auto_tool_choice: true``
    （后者在 V4-Pro/V4-Flash 等模型默认中已配置，避免用户重复指定）。
    tool_call_parser 来自模型默认配置，不需要用户指定。

    逻辑（仅管 tool_call_parser / enable_auto_tool_choice，不再触碰 reasoning_parser）：
      - 触发源开启 + 模型配置了 tool_call_parser
        → 保留 tool_call_parser，注入 enable_auto_tool_choice
      - 触发源开启 但模型没有 tool_call_parser
        → 移除 enable_auto_tool_choice，打印警告
      - 触发源未开启
        → 移除 tool_call_parser 和 enable_auto_tool_choice，FC 不生效

    注意：reasoning_parser 已与 function call 解耦，改由 _set_reasoning_parser
    依据独立的 --enable-auto-think-choice 开关单独控制。
    """
    user_wants_fc = (
        engine_cmd_parameter.get("enable_auto_tool_choice")
        or params.get("enable_auto_tool_choice")
    )
    if user_wants_fc:
        if "tool_call_parser" in params:
            params["enable_auto_tool_choice"] = True
            logger.info(
                "Function Call enabled (parser=%s)",
                params["tool_call_parser"],
            )
        else:
            params.pop("enable_auto_tool_choice", None)
            logger.warning("enable_auto_tool_choice is set but model has no tool_call_parser configured")
    else:
        params.pop("tool_call_parser", None)
        params.pop("enable_auto_tool_choice", None)


def _set_reasoning_parser(params, engine_cmd_parameter):
    """解析端：独立控制 reasoning_parser（思维链解析），与 function call 解耦。

    由 --enable-auto-think-choice / ENABLE_AUTO_THINK_CHOICE 开关单独驱动（默认关闭）：
      - 开关开启 → 保留模型默认配置中已有的 reasoning_parser；
        若模型/引擎配置未定义 reasoning_parser 则仅打印警告（不凭空注入）。
      - 开关关闭 → 移除 reasoning_parser，启动命令不带思维链解析。

    适用引擎：vllm / vllm_ascend / sglang（三引擎一致；mindie 无 reasoning_parser 字段）。
    本函数仅作用于配置解析后 params 中已存在的字段；模型对应 parser 由
    docs/features/reasoning_parser/reasoning_parser_support.yaml 注入。
    """
    if engine_cmd_parameter.get("enable_auto_think_choice"):
        if params.get("reasoning_parser"):
            logger.info("Reasoning parser enabled (parser=%s)", params["reasoning_parser"])
        else:
            logger.warning(
                "enable_auto_think_choice is set but model/engine config has no reasoning_parser configured"
            )
    else:
        params.pop("reasoning_parser", None)


def _set_thinking_default(params, engine_cmd_parameter, model_info):
    """生成端：按 enable_auto_think_choice 注入服务级默认思考状态（对称开关）。

    仅 vllm / vllm_ascend 适配（复用 vLLM OpenAI server 的 --default-chat-template-kwargs；
    sglang 无对应启动参数，不在此路径调用）。注入的是引擎【服务级默认值】：
      - 请求不带 chat_template_kwargs → 按此默认；
      - 请求带 chat_template_kwargs → 由请求级覆盖（客户端自负，不兜底改写）。

    对【混合推理模型】(mode == THINKING_HYBRID) 按开关注入对应键的布尔值：
      - 开关开 → {key: True}  服务级默认【强制打开】思考；
      - 开关关 → {key: False} 服务级默认【强制关闭】思考。
    键名按模型族对齐官方：Qwen3 / GLM = enable_thinking、DeepSeek-V3.x = thinking。

    其余模型不注入：
      - THINKING_ALWAYS_ON（R1 / QwQ / MiniMax-M2 等天生必思考，开/关都改不了）→ 不注入，
        仅在开关关时告警一次（提示无法关闭）；
      - THINKING_NONE（本就不思考的模型）→ 不介入，并清除残留默认值。

    与解析端解耦：本函数仅依据 resolve_thinking_off_policy（模型族是否支持启动期思考切换），
    与是否存在 reasoning_parser 无关——模型没有 reasoning_parser 但支持思考切换时，生成端仍强制开/关。
    """
    enabled = bool(engine_cmd_parameter.get("enable_auto_think_choice"))
    model_name = getattr(model_info, "model_name", "") or ""
    mode, off_kwargs = resolve_thinking_off_policy(model_name)
    if mode == THINKING_NONE:
        params.pop("default_chat_template_kwargs", None)
        return
    if mode == THINKING_ALWAYS_ON:
        params.pop("default_chat_template_kwargs", None)
        if not enabled:
            logger.warning(
                "enable_auto_think_choice=false but model '%s' is an always-on reasoner "
                "(e.g. DeepSeek-R1 / QwQ / MiniMax-M2); thinking cannot be disabled at "
                "startup, only reasoning parsing is affected.", model_name)
        return
    key = next(iter(off_kwargs))  # 'enable_thinking' or 'thinking'
    kwargs = {key: enabled}
    params["default_chat_template_kwargs"] = kwargs
    logger.info(
        "[thinking] enable_auto_think_choice=%s; inject "
        "--default-chat-template-kwargs=%s for model '%s'", enabled, kwargs, model_name)


def _warn_thinking_switch_unsupported_engine(engine, engine_cmd_parameter):
    """sglang / mindie 无法在启动期按开关切换思考；用户触发开关时仅日志提醒。

    思考的"开启/关闭"靠引擎服务级默认 chat_template_kwargs 落地，仅 vllm / vllm_ascend
    支持（--default-chat-template-kwargs）。其余引擎无法在启动期切换：
      - sglang：官方仅在请求级暴露是否思考，无对应启动参数；
      - mindie：思维解析/思考为服务端内置，不受本开关控制。
    故 enable_auto_think_choice 对它们【无效】，触发时提醒改在请求级控制（不改变行为）。
    """
    if engine in ("sglang", "mindie") and engine_cmd_parameter.get("enable_auto_think_choice"):
        logger.warning(
            "[thinking] enable_auto_think_choice=true 但引擎 '%s' 不支持启动期思考开关"
            "（sglang 无 --default-chat-template-kwargs；mindie 思考为服务端内置），"
            "该开关对此引擎无效；请在请求级 chat_template_kwargs 控制思考。", engine)


def _resolve_gpu_total_memory(ctx: Dict[str, Any]) -> float:
    """从运行时上下文或环境变量中解析 GPU 总显存（GB）。

    优先级: device_details[0].total_memory → WINGS_DEVICE_MEMORY 环境变量 → 12 GB 硬编码。
    """
    if ctx["device_details"] and ctx["device_details"][0]:
        total_memory = ctx["device_details"][0].get("total_memory", 12)
        if total_memory is None:
            logger.warning("total_memory is None in device details, defaulting to 12G")
            return 12.0
        return float(total_memory)
    mem_env = os.getenv("WINGS_DEVICE_MEMORY", "").strip()
    if mem_env:
        try:
            mem_val = float(mem_env)
            logger.info("Using WINGS_DEVICE_MEMORY=%s GB for cuda-graph-sizes", mem_val)
            return mem_val
        except ValueError:
            logger.warning("Invalid WINGS_DEVICE_MEMORY='%s', fallback to 12GB", mem_env)
    else:
        logger.warning("Can't get device details and WINGS_DEVICE_MEMORY not set, fallback to 12GB")
    return 12.0


def _set_operator_acceleration(params, ctx):
    """当昇腾算子加速（USE_KUNLUN_ATB）启用时，注入 use_kunlun_atb=True 到参数字典。"""
    if get_operator_acceleration_env() and ctx["device"] == "ascend":
        params['use_kunlun_atb'] = True
    else:
        return


def _should_skip_soft_fp8_for_official_deepseek_v3(ctx, model_name: str, model_path: str) -> bool:
    """Return True for official DeepSeek V3-family Ascend quantized models."""
    return (
        ctx.get('device') == "ascend"
        and _is_deepseek_v3_family_model(model_name, model_path)
        and is_deepseek_series_modelslim_quant(model_path)
    )


def _is_soft_fp8_model(model_name: str, model_path: str) -> bool:
    """Detect whether the model should use Soft FP8 settings."""
    return is_qwen3_series_fp8(model_path, model_name) or is_deepseek_series_fp8(model_path)


def _log_soft_fp8_switch_state(model_name: str) -> None:
    """Log Soft FP8/FP4 switch state for an auto-detected FP8 model."""
    if get_soft_fp4_env():
        logger.warning("Model %s is detected as FP8 model, but Soft FP4 switch is enabled. "
                       "Automatically correcting to use Soft FP8 configuration.", model_name)
    if not get_soft_fp8_env():
        logger.info("Model %s is detected as FP8 model, automatically enabling Soft FP8 configuration", model_name)


def _apply_qwen3_soft_fp8(params, model_architecture: str, explicit_keys: set) -> None:
    """Apply Qwen3 Soft FP8 parameters while preserving explicit user values."""
    if 'quantization' not in explicit_keys:
        params['quantization'] = 'ascend'
    if model_architecture == "Qwen3MoeForCausalLM" and 'enable_expert_parallel' not in explicit_keys:
        params['enable_expert_parallel'] = False
        logger.info("Soft FP8 configured for Qwen3 MOE Series models")
    else:
        logger.info("Soft FP8 configured for Qwen3 Series models")


def _resolve_deepseek_soft_fp8_parallel(params) -> Tuple[int, int]:
    """Resolve recommended DeepSeek FP8 TP/DP values from device_count."""
    try:
        device_count_val = int(params.get('device_count', 0))
    except (ValueError, TypeError):
        logger.warning("Invalid device_count value, defaulting to 0")
        device_count_val = 0
    recommended_tp = min(4, device_count_val) if device_count_val > 0 else 4
    recommended_dp = min(4, device_count_val // recommended_tp) if device_count_val > 0 else 4
    return recommended_tp, recommended_dp


def _apply_deepseek_soft_fp8(params, explicit_keys: set) -> None:
    """Apply DeepSeek Soft FP8 parameters while preserving explicit user values."""
    if 'quantization' not in explicit_keys:
        params['quantization'] = 'ascend'
    if 'enforce_eager' not in explicit_keys:
        params["enforce_eager"] = True
    if 'no_enable_prefix_caching' not in explicit_keys and 'enable_prefix_caching' not in explicit_keys:
        params['no_enable_prefix_caching'] = True
    if 'enable_expert_parallel' not in explicit_keys:
        params['enable_expert_parallel'] = True
    if 'async_scheduling' not in explicit_keys:
        params['async_scheduling'] = True

    recommended_tp, recommended_dp = _resolve_deepseek_soft_fp8_parallel(params)
    if 'data_parallel_size' not in explicit_keys:
        params['data_parallel_size'] = recommended_dp
    if 'tensor_parallel_size' not in explicit_keys:
        params['tensor_parallel_size'] = recommended_tp
    params['use_kunlun_atb'] = False
    logger.info("Soft FP8 configured for Deekseek Series models")


def _set_soft_fp8(params, ctx, model_info):
    """FP8 特性的参数配置（支持 DeepSeek / Qwen3 系列 FP8 模型）。

    自动检测模型是否为 FP8 模型，并根据模型系列设置对应的量化参数：
      - Qwen3 系列：设置 quantization='ascend'，MOE 模型禁用专家并行
            - DeepSeek 系列：设置 quantization='ascend'，禁用 prefix caching，启用 EP/async，固定 TP=4/DP=4

    如果模型不是 FP8 模型，则跳过此函数，交由 _set_soft_fp4 处理。
    """
    model_architecture = model_info.model_architecture
    model_name = model_info.model_name
    model_path = model_info.model_path
    explicit_keys = _detect_explicit_cli_keys()

    if _should_skip_soft_fp8_for_official_deepseek_v3(ctx, model_name, model_path):
        logger.info("DeepSeek V3-family official Ascend quantized path, skip Soft FP8")
        return

    # 如果模型不是 FP8，则跳过此函数，让 FP4 函数处理
    if not _is_soft_fp8_model(model_name, model_path):
        return

    _log_soft_fp8_switch_state(model_name)

    if ctx['device'] != "ascend":
        logger.warning("Soft FP8 is only supported on Ascend devices")
        return
    if is_qwen3_series_fp8(model_path, model_name):
        _apply_qwen3_soft_fp8(params, model_architecture, explicit_keys)
    elif is_deepseek_series_fp8(model_path):
        _apply_deepseek_soft_fp8(params, explicit_keys)


def _is_deepseek_v3_family_model(model_name: str, model_path: str) -> bool:
    """Return True for DeepSeek V3-family identifiers, excluding R1/R2 names."""
    candidates = [str(item) for item in (model_name, model_path) if item]
    for item in candidates:
        normalized = item.lower().replace("_", "-")
        if "deepseek" in normalized and "v3" in normalized:
            return True
    return False


def _set_soft_fp4(params, ctx, model_info):
    """FP4 特性的参数配置（仅支持昇腾设备上的 Qwen3-32B NVFP4 模型）。

    检查模型是否为 FP4 模型，如果是则设置 quantization='ascend'。
    如果同时启用了 FP8 开关，会给出警告但继续使用 FP4 配置。
    """
    model_path = model_info.model_path
    model_name = model_info.model_name

    # 检查模型是否为 FP4 模型
    is_fp4_model = is_qwen3_32b_nvfp4(model_path)

    # 如果模型不是 FP4，则跳过此函数
    if not is_fp4_model:
        return

    # 检查是否启用了 FP8 开关，如果是则给出警告但继续使用 FP4 配置
    if get_soft_fp8_env():
        logger.warning("Model %s is detected as FP4 model, but Soft FP8 switch is enabled. "
                       "Automatically correcting to use Soft FP4 configuration.", model_name)

    # 检查是否启用了 FP4 开关，如果没有启用则给出提示但继续配置
    if not get_soft_fp4_env():
        logger.info("Model %s is detected as FP4 model, automatically enabling Soft FP4 configuration", model_name)

    if ctx['device'] != "ascend":
        logger.warning("Soft FP4 is only supported on Ascend (NPU) devices and will be ignored on current device")
        return

    logger.info("Will use Soft FP4 configuration")
    if 'quantization' not in _detect_explicit_cli_keys():
        params['quantization'] = 'ascend'


def _validate_embedding_rerank_params(params, ctx):
    """对于 embedding 和 rerank 模型，强制禁用 enable_chunked_prefill 和 enable_prefix_caching 参数。

    如果用户传入了这些参数，会记录警告日志后再取消这些参数。

    Args:
        params: 参数字典
        ctx: 包含模型类型等上下文信息的字典
    """
    model_type = ctx.get("model_type", "")

    # 仅对 embedding 和 rerank 模型进行处理
    if model_type not in ["embedding", "rerank"]:
        return

    # 检查并处理 enable_chunked_prefill 参数
    if "enable_chunked_prefill" in params:
        if params["enable_chunked_prefill"] not in [None, False, "False", 0, "0"]:
            logger.warning(
                f"Model type '{model_type}' does not support 'enable_chunked_prefill' parameter. "
                f"This parameter will be disabled."
            )
        params.pop("enable_chunked_prefill", None)

    # 检查并处理 enable_prefix_caching 参数
    if "enable_prefix_caching" in params:
        if params["enable_prefix_caching"] not in [None, False, "False", 0, "0"]:
            logger.warning(
                f"Model type '{model_type}' does not support 'enable_prefix_caching' parameter. "
                f"This parameter will be disabled."
            )
        params.pop("enable_prefix_caching", None)


def _set_common_params(params, engine_cmd_parameter, config_path):
    """根据参数映射表，将用户 CLI 参数翻译为引擎实际的参数键名并写入 params。

    优先级保护：仅当用户通过 CLI 参数或环境变量 **显式** 指定了某个参数时，
    才覆盖模型默认配置（nvidia_default.json / model_deploy_config）中的已有值。
    对于用户未显式指定的参数，若模型配置中已有值则保留，否则用 argparse 默认值补充。
    """
    vllm_param_map_config = _load_mapping(config_path, 'default_to_vllm_parameter_mapping')
    explicit_keys = _detect_explicit_cli_keys()
    for key, value in vllm_param_map_config.items():
        if not value:
            continue
        cli_val = engine_cmd_parameter.get(key)
        if cli_val is None:
            continue
        # 用户显式设置的参数（CLI 或环境变量）：始终覆盖
        if key in explicit_keys:
            params[value] = cli_val
        # 模型默认配置中不存在的参数：用 argparse 默认值补充
        elif value not in params:
            params[value] = cli_val
        # 否则：保留模型默认配置中的值，不被 argparse 默认值覆盖


def _set_sequence_length(params, engine_cmd_parameter, model_type: str = "llm"):
    """将序列长度合并为 max_model_len 并写入 params。

    - LLM：max_model_len = input_length + output_length
    - embedding / rerank：max_model_len = input_length（无生成阶段）
    """
    explicit_keys = _detect_explicit_cli_keys()
    if not explicit_keys.intersection({"input_length", "output_length"}):
        return

    input_len = engine_cmd_parameter.get("input_length")
    output_len = engine_cmd_parameter.get("output_length")

    # Default None values to 0 before summation; cast to int to prevent string concatenation
    input_len = int(input_len) if input_len is not None else 0
    output_len = int(output_len) if output_len is not None else 0

    if model_type in ("embedding", "rerank"):
        max_model_len = input_len
    else:
        max_model_len = input_len + output_len

    if max_model_len <= 0:
        return
    params['max_model_len'] = max_model_len


def _set_task(params, ctx):
    """根据模型类型（embedding/rerank）设置 vllm task 参数。

    昇腾设备上 embedding/rerank 模型需要强制启用 eager 模式并关闭 ATB 算子加速。
    """
    if ctx["model_type"] == "embedding":
        params["task"] = "embedding"
        if ctx["device"] == "ascend":
            params["enforce_eager"] = True
            params["use_kunlun_atb"] = False
    elif ctx["model_type"] == "rerank":
        params["task"] = "score"
        if ctx["device"] == "ascend":
            params["enforce_eager"] = True
            params["use_kunlun_atb"] = False
    else:
        return


def _set_parallelism_params(params, ctx):
    """根据设备数和分布式模式设置张量并行度（tensor_parallel_size）。"""
    # Ascend DeepSeek dp_deployment 后端 TP 语义是「节点内」，由 vllm_adapter 的
    # _default_deepseek_ascend_dp_tensor_parallel_size 按架构兜底；
    # 此处不能套用 Ray 全局 TP 公式 (device_count × nnodes)，否则
    # 双机 ×8 卡场景下会被算成 tp=16 触发 DP 拓扑校验失败。
    # 仅短路 Ascend DeepSeek 路径，避免误伤 NVIDIA PD（同样使用 dp_deployment
    # 但需要 _adjust_tensor_parallelism 的 PD 分支给 tp=device_count）。
    if (
        ctx.get("engine") == "vllm_ascend"
        and ctx.get("distributed_executor_backend") == "dp_deployment"
    ):
        return
    _adjust_tensor_parallelism(
        params,
        ctx["device_count"],
        TensorParallelConfig(
            tp_key='tensor_parallel_size',
            if_distributed=ctx['distributed'],
            node_ips=ctx.get("node_ips"),
            nnodes=ctx.get("nnodes"),
        ),
    )


def _get_pd_config(ctx, pd_role):
    """生成 PD（Prefill-Decode）分离部署所需的 KV Transfer 配置片段。

    参数:
        pd_role: PD 角色，"P" 表示 Prefill 节点，"D" 表示 Decode 节点
        ctx: 运行上下文，如 {'device': 'ascend', 'device_count': 2}

    返回:
        包含 KV Transfer 配置项的字典

    环境变量:
        PD_CONNECTOR_TYPE: 指定 Ascend PD 分离使用的 KV connector 类型。
            - "MooncakeConnectorV1"（默认）: vllm-ascend 的 Ascend 原生实现，
              支持 tuple KV cache 格式和 MLA，但部分模型存在 head_dim 兼容性问题
              和运行时内存损坏风险（详见 vllm-ascend #7352, #5660）。
            - "MooncakeConnector": vllm-ascend 注册的新版连接器，
              改进了 KV cache 注册逻辑，部分场景更稳定（详见 #7433）。
            - 也可设置为其他已注册的连接器名称（如自定义连接器）。
    """
    device = ctx.get('device', '')
    config = {}

    if device == "ascend":
        kv_role = "kv_producer" if pd_role == "P" else "kv_consumer"

        # 通过环境变量选择 connector 类型，默认 MooncakeConnectorV1
        connector_type = os.getenv("PD_CONNECTOR_TYPE", "MooncakeConnectorV1")

        # Ascend Mooncake 系列 connector 要求 kv_connector_extra_config 中包含
        # prefill 和 decode 双方的并行配置 (tp_size, dp_size, pp_size)，
        # 用于 KV cache 传输时的 TP/DP 映射计算。
        default_tp = int(ctx.get('device_count', 1))
        prefill_tp = int(os.getenv("PD_PREFILL_TP_SIZE", str(default_tp)))
        prefill_dp = int(os.getenv("PD_PREFILL_DP_SIZE", "1"))
        prefill_pp = int(os.getenv("PD_PREFILL_PP_SIZE", "1"))
        decode_tp = int(os.getenv("PD_DECODE_TP_SIZE", str(default_tp)))
        decode_dp = int(os.getenv("PD_DECODE_DP_SIZE", "1"))
        decode_pp = int(os.getenv("PD_DECODE_PP_SIZE", "1"))

        config = {
            "kv_connector": connector_type,
            "kv_role": kv_role,
            "kv_connector_extra_config": {
                "mooncake_protocol": "rdma",
                "prefill": {
                    "tp_size": prefill_tp,
                    "dp_size": prefill_dp,
                    "pp_size": prefill_pp,
                },
                "decode": {
                    "tp_size": decode_tp,
                    "dp_size": decode_dp,
                    "pp_size": decode_pp,
                },
            },
        }
        logger.info("[PD Config] Ascend device detected, connector=%s, role=%s, kv_role=%s, "
                     "prefill(tp=%d,dp=%d,pp=%d), decode(tp=%d,dp=%d,pp=%d)",
                     connector_type, pd_role, kv_role,
                     prefill_tp, prefill_dp, prefill_pp,
                     decode_tp, decode_dp, decode_pp)
    else:
        # AscendPD
        config = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both"
        }
        logger.info("[PD Config] non-ascend device (%s) detected, role=%s", device, pd_role)

    return config


def _is_glm51_nvidia_vllm(ctx, model_info) -> bool:
    """Return True for NVIDIA + vLLM/vLLM distributed GLM-5.1 deployments."""
    engine = ctx.get("engine", "")
    return (
        ctx.get("device") == "nvidia"
        and engine in {"vllm", "vllm_distributed"}
        and is_glm_moe_dsa_glm51(
            model_info,
            model_name=ctx.get("model_name"),
            model_path=ctx.get("model_path"),
        )
    )


_DEEPSEEK_V4_CPU_OFFLOAD_ARCHES = {
    "DeepseekV4ForCausalLM",
    "DeepSeekV4ForCausalLM",
}


def _v4_offload_identity_text(ctx, model_info) -> str:
    """拼接 V4 身份判定文本（模型名/路径，小写），供 offload 判定复用。"""
    candidates = [
        getattr(model_info, "model_name", ""),
        getattr(model_info, "model_path", ""),
        ctx.get("model_name", ""),
        ctx.get("model_path", ""),
    ]
    return " ".join(str(item).lower() for item in candidates if item)


def _is_deepseek_v4_cpu_offload(ctx, model_info) -> bool:
    """Return True when V4 KV offload should bypass LMCache in config merge."""
    if ctx.get("engine") != "vllm_ascend":
        return False
    text = _v4_offload_identity_text(ctx, model_info)
    is_v4_flash_or_pro = "v4" in text and ("flash" in text or "pro" in text)
    if not is_v4_flash_or_pro:
        return False
    arch = getattr(model_info, "model_architecture", "")
    return arch in _DEEPSEEK_V4_CPU_OFFLOAD_ARCHES


def _is_deepseek_v4_flash_offload(ctx, model_info) -> bool:
    """Return True when the V4 offload target is V4-Flash (vs V4-Pro)."""
    return "flash" in _v4_offload_identity_text(ctx, model_info)


def _is_deepseek_v4_flash_nv(ctx, model_info) -> bool:
    """[V4-Flash-NV-Day0] Return True for V4-Flash on NVIDIA/vllm.

    该路径用 native ``--kv_offloading_backend``（在 vllm_adapter 生成 CLI），
    因此 config 合并阶段不注入 LMCacheConnectorV1 kv_transfer_config。
    """
    if ctx.get("engine") != "vllm":
        return False
    text = _v4_offload_identity_text(ctx, model_info)
    if not ("v4" in text and "flash" in text):
        return False
    arch = getattr(model_info, "model_architecture", "")
    return arch in _DEEPSEEK_V4_CPU_OFFLOAD_ARCHES


def _resolve_v4_offload_device_count(params, ctx) -> int:
    """本节点卡数，用于 V4-Flash 卸载容量按卡放大。"""
    try:
        return int(ctx.get("device_count") or params.get("device_count") or 1)
    except (TypeError, ValueError):
        return 1


def _build_deepseek_v4_cpu_offload_config(params, ctx, model_info) -> Dict[str, Any]:
    """构建 V4 CPUOffloadingConnector 配置。

    cpu_swap_space_gb 取值:
      * V4-Flash: ``device_count(本节点卡数) × LMCACHE_MAX_LOCAL_CPU_SIZE``（每卡语义）;
      * V4-Pro / 其它: 直接等于 ``LMCACHE_MAX_LOCAL_CPU_SIZE``（不乘卡数）;
      * 未设置/非法: 一律缺省 200（不乘）。
    须与 ``vllm_adapter._apply_deepseek_v4_cpu_offload`` 公式保持一致。
    """
    raw_size = os.getenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "").strip()
    try:
        per_card_gb = int(raw_size) if raw_size else None
    except ValueError:
        logger.warning(
            "[DeepSeek-V4 KV Offload] Invalid LMCACHE_MAX_LOCAL_CPU_SIZE=%r; "
            "falling back to 200 GB.", raw_size,
        )
        per_card_gb = None
    if per_card_gb is None:
        cpu_swap_gb = 200
    elif _is_deepseek_v4_flash_offload(ctx, model_info):
        cpu_swap_gb = _resolve_v4_offload_device_count(params, ctx) * per_card_gb
    else:
        cpu_swap_gb = per_card_gb
    return {
        "kv_connector": "CPUOffloadingConnector",
        "kv_connector_module_path":
            "vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload.cpu_offload_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "swap_in_threshold": 1,
            "cpu_swap_space_gb": cpu_swap_gb,
        },
    }


def _set_kv_cache_config(params, ctx, model_info=None):
    """根据 LMCache Offload 和 PD 分离角色，生成 vllm kv_transfer_config 配置。

    优先级逻辑：
    - LMCache + PD 同时启用 → MultiConnector（同时承载 KV Offload 和 PD 传输）
    - 仅启用 LMCache → LMCacheConnectorV1
    - 仅启用 PD → 按设备类型选择 MooncakeConnector 或 NixlConnector
    - 两者都未启用 → 跳过不注入
    """
    lmcache_offload = get_lmcache_env()
    pd_role = get_pd_role_env()
    device = ctx.get('device', '')

    if lmcache_offload and model_info is not None and _is_deepseek_v4_cpu_offload(ctx, model_info):
        params["kv_transfer_config"] = json.dumps(
            _build_deepseek_v4_cpu_offload_config(params, ctx, model_info)
        )
        logger.info(
            "[KVCache Offload] DeepSeek-V4 Flash/Pro on vllm_ascend uses "
            "CPUOffloadingConnector; not injecting LMCacheConnectorV1."
        )
        return

    if lmcache_offload and model_info is not None and _is_deepseek_v4_flash_nv(ctx, model_info):
        logger.info(
            "[KVCache Offload] DeepSeek-V4-Flash on NVIDIA/vllm uses native "
            "--kv_offloading_backend; not injecting LMCacheConnectorV1."
        )
        return

    if lmcache_offload and model_info is not None and _is_glm51_nvidia_vllm(ctx, model_info):
        logger.warning(
            "[KVCache Offload] Forced disabled for GLM-5.1 on NVIDIA/vLLM; "
            "ignoring LMCACHE_OFFLOAD=true and not injecting LMCache kv_transfer_config."
        )
        lmcache_offload = False

    # Ascend NPU 需要额外的 engine_id 和 kv_buffer_device 字段
    is_ascend = (device == "ascend")

    def _build_lmcache_connector():
        """构建 LMCacheConnectorV1 配置，Ascend 场景追加 NPU 专属字段。"""
        connector = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        if is_ascend:
            connector["engine_id"] = os.getenv("LMCACHE_ENGINE_ID", "lmca1")
            connector["kv_buffer_device"] = "npu"
        return connector

    if lmcache_offload and pd_role:
        lmcache_connector = _build_lmcache_connector()
        config = {
            "kv_connector": 'MultiConnector',
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "connectors": [
                    _get_pd_config(ctx, pd_role),
                    lmcache_connector
                ]
            }
        }
        logger.info("[KVCache Offload] KVCache Offload feature is enabled and PD role is %s", pd_role)
    elif lmcache_offload:
        config = _build_lmcache_connector()
        logger.info("[KVCache Offload] KVCache Offload feature is enabled")
    elif pd_role:
        config = _get_pd_config(ctx, pd_role)
        logger.info("PD role is %s", pd_role)
    else:
        return  #

    params['kv_transfer_config'] = json.dumps(config)


def _enforce_glm51_nvidia_no_kv_offload(engine_config: Dict[str, Any],
                                        ctx: Dict[str, Any],
                                        model_info) -> None:
    """Final guard: GLM-5.1 on NVIDIA/vLLM must not use KV offload.

    This runs after user config, CLI overrides, and raw ``engine_config`` are
    merged, so even upper-layer injected ``kv_transfer_config`` is removed.
    """
    if not _is_glm51_nvidia_vllm(ctx, model_info):
        return

    removed = engine_config.pop("kv_transfer_config", None)
    if removed is not None:
        logger.warning(
            "[KVCache Offload] Forced disabled for GLM-5.1 on NVIDIA/vLLM; "
            "removed user/upstream kv_transfer_config=%s",
            removed,
        )
    elif get_lmcache_env():
        logger.warning(
            "[KVCache Offload] Forced disabled for GLM-5.1 on NVIDIA/vLLM; "
            "LMCACHE_OFFLOAD=true was requested but will be ignored."
        )


def _guard_pd_hybrid_kv_cache(params):
    """PD 分离模式下移除显式传入的 hybrid KV cache manager 开关。

    该函数只做 PD 保护，不再根据模型架构自动注入
    --no-disable-hybrid-kv-cache-manager。若用户或上层配置显式传入该字段，
    且当前开启 PD 分离，则移除它，避免 MooncakeConnectorV1 / NixlConnector
    等 KV 连接器与 HMA（Hybrid Memory Architecture）路径不兼容。
    """
    if not get_pd_role_env():
        return
    if "no_disable_hybrid_kv_cache_manager" not in params:
        return
    params.pop("no_disable_hybrid_kv_cache_manager", None)
    logger.warning(
        "[HybridKV] PD separation is enabled; removed "
        "--no-disable-hybrid-kv-cache-manager because current KV connectors "
        "do not support HMA."
    )


def _ensure_pd_head_dim(params, model_info):
    """PD 分离模式下，通过 --hf-overrides 注入缺失的 head_dim。

    vllm-ascend 的 MooncakeConnectorV1 在 KVCacheRecvingThread 初始化时
    直接读取 hf_text_config.head_dim，但部分模型架构（如 Qwen2ForCausalLM）
    的 config.json 中未显式声明 head_dim（模型代码中动态计算为
    hidden_size // num_attention_heads），导致 decode 侧 AttributeError 崩溃。

    通过直接设置 params['hf_overrides'] 注入，生成 --hf-overrides '{"head_dim": N}'
    CLI 参数，无需修改模型文件，兼容只读挂载。

    注意：此函数在 _merge_vllm_params 中调用，此时 params 是平坦的引擎参数字典
    （即 engine_config 的内容），不可使用 params.setdefault("engine_config", ...)
    否则会在 engine_config 内部创建嵌套的 engine_config 键，导致 vLLM 收到
    无法识别的 --engine-config CLI 参数。

    参考: https://github.com/vllm-project/vllm-ascend/issues/7352
    """
    if not get_pd_role_env():
        return

    if not model_info or not model_info.config:
        return

    config = model_info.config
    if "head_dim" in config:
        return

    hidden_size = config.get("hidden_size")
    num_attention_heads = config.get("num_attention_heads")
    if not hidden_size or not num_attention_heads:
        return

    head_dim = hidden_size // num_attention_heads
    params["hf_overrides"] = json.dumps({"head_dim": head_dim})
    logger.info(
        "[PD] Model config.json missing head_dim, injecting "
        "--hf-overrides '{\"head_dim\": %d}' (hidden_size=%d / num_attention_heads=%d)",
        head_dim, hidden_size, num_attention_heads)


def _set_router_config(params):
    """当 Wings Router 路由功能启用时，注入 KV 事件 NATS 发布配置。

    Wings Router 依赖 NATS 消息队列来感知各实例的 KV Cache 命中情况，
    从而做智能路由。此函数将 NATS 发布配置序列化为 JSON 并写入 params。
    """
    router_enable = get_router_env()
    router_instance_group_name = get_router_instance_group_name_env()

    if not router_enable or not router_instance_group_name:
        return

    router_instance_name = get_router_instance_name_env()
    router_nats_path = get_router_nats_path_env()

    kv_events_config = json.dumps({
        "enable_kv_cache_events": True,
        "publisher": "nats",
        "instance_id": f"{router_instance_group_name}:{router_instance_name}",
        "nats_servers": router_nats_path
    })

    params['kv_events_config'] = kv_events_config
    logger.info("Wings Router for vllm is enabled")


def _detect_mtp_moe_features(engine_cmd_parameter: Dict[str, Any],
                              params: Dict[str, Any]) -> None:
    """检测 MTP / MOE 特性并更新 params 中的 isMTP 和 isMOE 字段。"""
    is_mtp = False
    is_moe = False
    model_path = engine_cmd_parameter.get("model_path")
    if model_path and os.path.exists(model_path):
        mtp_file = os.path.join(model_path, "mtp.safetensors")
        is_mtp = os.path.exists(mtp_file)
    if params.get("enable_ep_moe"):
        is_moe = True
    params.update({'isMTP': is_mtp, 'isMOE': is_moe})


def _apply_us8_long_ctx_strategy(params: Dict[str, Any],
                                  ctx: Dict[str, Any],
                                  engine_cmd_parameter: Dict[str, Any],
                                  model_info) -> None:
    """US8: DeepSeek 满血模型 16 卡长上下文时注入 dp/sp/cp/tp 并行策略。"""
    long_ctx_threshold = int(os.getenv("MINDIE_LONG_CONTEXT_THRESHOLD", "8192"))
    model_architecture = getattr(model_info, "model_architecture", None) if model_info else None
    total_seq_len = (
        (int(engine_cmd_parameter.get("input_length") or 0))
        + (int(engine_cmd_parameter.get("output_length") or 0))
    )
    if not (model_architecture in ["DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"]
            and total_seq_len > long_ctx_threshold):
        return

    device_count = int(ctx.get("device_count", 1) or 1)
    if ctx.get('distributed'):
        nnodes_actual = _resolve_distributed_node_count(ctx.get("node_ips"), ctx.get("nnodes"))
    else:
        nnodes_actual = 1
    global_world_size = device_count * nnodes_actual
    if global_world_size != 16 or (nnodes_actual, device_count) not in [(1, 16), (2, 8)]:
        logger.info(
            "[US8] DeepSeek long-context CP/SP skipped: only 1x16 or 2x8 is supported "
            "(nnodes=%d, device_count=%d, globalWorldSize=%d)",
            nnodes_actual, device_count, global_world_size,
        )
        return

    def _safe_int_env(name: str, default: str) -> int:
        val = os.getenv(name, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            logger.warning("Invalid %s=%r, using default %s", name, val, default)
            return int(default)

    params['dp'] = _safe_int_env("MINDIE_DS_DP", "1")
    params['cp'] = _safe_int_env("MINDIE_DS_CP", "2")
    tp_default = max(1, global_world_size // max(1, params['dp'] * params['cp']))
    params['tp'] = _safe_int_env("MINDIE_DS_TP", str(tp_default))
    params['sp'] = _safe_int_env("MINDIE_DS_SP", str(params['tp']))
    # CPSP long-context reference requires the three length limits to stay
    # aligned; otherwise MindIE may prefill against a smaller legacy default.
    params['maxSeqLen'] = total_seq_len
    params['maxInputTokenLen'] = total_seq_len
    params['maxPrefillTokens'] = total_seq_len
    logger.info(
        "[US8] DeepSeek long-context enabled (seq=%d > %d): "
        "dp=%d, sp=%d, cp=%d, tp=%d",
        total_seq_len, long_ctx_threshold,
        params['dp'], params['sp'], params['cp'], params['tp'],
    )


def _set_mindie_distributed_params(params, ctx):
    """MindIE 分布式 / 单机场景下设置 worldSize 和 npuDeviceIds。"""
    if ctx.get('distributed'):
        node_ips = ctx.get("node_ips") or get_node_ips()
        node_ips_list = _extract_node_ips(node_ips)
        nnodes_actual = _resolve_distributed_node_count(node_ips, ctx.get("nnodes"))

        if nnodes_actual <= 1:
            logger.warning(
                "MindIE distributed mode requires nnodes > 1, got %d. "
                "HCCL multi-node initialization may fail.",
                nnodes_actual,
            )

        # MindIE multi-node TP: worldSize = total devices across ALL nodes.
        # rank table 含所有节点（Fix P-CD-1），server_count = nnodes_actual，
        # ConfigManager 校验 worldSize(total) % n_nodes == 0 → 通过。
        # npuDeviceIds 只列本节点本地设备 ID。
        params['worldSize'] = int(ctx["device_count"]) * nnodes_actual
        # multiNodesInferEnabled=True 开启 HCCL 跨节点通信。
        # 前提：worldSize 已设为全局总 rank 数，ConfigManager 不会再错误覆盖。
        params['multiNodesInferEnabled'] = nnodes_actual > 1
        params['node_ips'] = ",".join(node_ips_list) if node_ips_list else (node_ips or "")
        params['npuDeviceIds'] = [[i for i in range(ctx["device_count"])]]
    else:
        _adjust_tensor_parallelism(params, ctx["device_count"], TensorParallelConfig(tp_key='worldSize'))
        params['npuDeviceIds'] = [[i for i in range(ctx["device_count"])]]


def _set_mindie_function_call(params, engine_cmd_parameter):
    """MindIE Function Call 开关处理。

    用户通过 enable_auto_tool_choice 统一触发，映射到 MindIE 的
    mindie_tool_call_parser / mindie_model_type（由 mindie_adapter 注入）。
    """
    user_wants_fc = engine_cmd_parameter.get("enable_auto_tool_choice")
    if user_wants_fc:
        if "mindie_tool_call_parser" in params:
            logger.info(
                "Function Call enabled for MindIE (parser=%s, model_type=%s)",
                params.get("mindie_tool_call_parser"),
                params.get("mindie_model_type"),
            )
        else:
            logger.warning("enable_auto_tool_choice is set but MindIE model has no mindie_tool_call_parser configured")
    else:
        params.pop("mindie_tool_call_parser", None)
        params.pop("mindie_model_type", None)
        params.pop("mindie_chat_template", None)


def _set_mindie_common_params(params, engine_cmd_parameter):
    """MindIE 参数映射：翻译 CLI 参数名为 MindIE 原生键名。

    优先级保护（对齐 vLLM 的 _set_common_params）：
    仅当用户通过 CLI 或环境变量显式指定了某个参数时才覆盖模型默认配置。
    对于用户未显式指定的参数，若模型配置中已有值则保留，否则用 argparse 默认值补充。
    """
    engine_param_map_config_path = os.path.join(DEFAULT_CONFIG_DIR,
                                            DEFAULT_CONFIG_FILES.get("engine_parameter_mapping"))
    mindie_param_map_config = _load_mapping(engine_param_map_config_path, 'default_to_mindie_parameter_mapping')
    explicit_keys = _detect_explicit_cli_keys()
    for key, value in mindie_param_map_config.items():
        if not value:
            continue
        cli_val = engine_cmd_parameter.get(key)
        if cli_val is None:
            continue
        # 用户显式设置的参数（CLI 或环境变量）：始终覆盖
        if key in explicit_keys:
            params[value] = cli_val
        # MindIE 的 NPU_MEMORY_FRACTION 默认值由 set_mindie_env.sh 统一提供。
        # argparse 的 gpu_memory_utilization 默认值是 0.9；如果在这里按“缺省补充”
        # 翻译成 npu_memory_fraction，会在启动脚本中把 set_mindie_env.sh 的 0.96
        # 二次覆盖成 0.9。只有用户显式传入 --gpu-memory-utilization 或
        # GPU_MEMORY_UTILIZATION 时，才允许覆盖 MindIE 默认值。
        elif key == "gpu_memory_utilization" and value == "npu_memory_fraction":
            continue
        # 模型默认配置中不存在的参数：用 argparse 默认值补充
        elif value not in params:
            params[value] = cli_val
        # 否则：保留模型默认配置中的值，不被 argparse 默认值覆盖


def _merge_mindie_params(params, ctx, engine_cmd_parameter, model_info=None):
    """将通用参数合并为 MindIE config.json 所要求的字段格式。

    调用链路:
        1. _set_mindie_common_params    → 根据参数映射表翻译 CLI 参数
        2. _detect_mtp_moe_features     → MTP/MOE 特性检测
        3. maxSeqLen / maxPrefillTokens → 计算序列长度
        4. _apply_us8_long_ctx_strategy → DeepSeek 分布式长上下文策略
        5. _set_mindie_distributed_params → worldSize / npuDeviceIds
        6. _set_mindie_function_call    → Function Call 开关
    """
    _set_mindie_common_params(params, engine_cmd_parameter)
    _detect_mtp_moe_features(engine_cmd_parameter, params)

    if engine_cmd_parameter["input_length"] and engine_cmd_parameter["output_length"]:
        params.update({
            'maxSeqLen': int(engine_cmd_parameter["input_length"]) + int(engine_cmd_parameter["output_length"]),
            'maxPrefillTokens': max(8192, int(engine_cmd_parameter["input_length"]))
        })

    _apply_us8_long_ctx_strategy(params, ctx, engine_cmd_parameter, model_info)
    _set_mindie_distributed_params(params, ctx)
    _set_mindie_function_call(params, engine_cmd_parameter)

    return params


def _merge_sglang_params(params, ctx, engine_cmd_parameter):
    """将通用参数合并为 SGLang 启动参数格式。

    - 通过 sglang 参数映射表翻译 CLI 参数名；
    - 注意：sglang 的 enable_prefix_caching 语义与 vllm 相反，因此需要取反；
    - 合并 input_length + output_length 为 context_length；
    - 设置张量并行度（tp_size）；
    - sglang 4.10.0+ 中 --enable-ep-moe 已废弃，改为 ep_size = tp_size。

    优先级保护（对齐 vLLM 的 _set_common_params）：
    仅当用户通过 CLI 或环境变量显式指定了某个参数时才覆盖模型默认配置。
    对于用户未显式指定的参数，若模型配置中已有值则保留，否则用 argparse 默认值补充。
    """
    #
    engine_param_map_config_path = os.path.join(DEFAULT_CONFIG_DIR,
                                            DEFAULT_CONFIG_FILES.get("engine_parameter_mapping"))
    sglang_param_map_config = _load_mapping(engine_param_map_config_path, 'default_to_sglang_parameter_mapping')
    explicit_keys = _detect_explicit_cli_keys()
    for key, value in sglang_param_map_config.items():
        if not value or engine_cmd_parameter.get(key) is None:
            continue
        cli_val = engine_cmd_parameter.get(key)
        # sglang 的 enable_prefix_caching 语义与 vllm 相反，需要取反
        if key == "enable_prefix_caching":
            cli_val = not cli_val
        # 用户显式设置的参数（CLI 或环境变量）：始终覆盖
        if key in explicit_keys:
            params[value] = cli_val
        # 模型默认配置中不存在的参数：用 argparse 默认值补充
        elif value not in params:
            params[value] = cli_val
        # 否则：保留模型默认配置中的值，不被 argparse 默认值覆盖

    #
    if engine_cmd_parameter["input_length"] and engine_cmd_parameter["output_length"]:
        input_len = int(engine_cmd_parameter["input_length"])
        output_len = int(engine_cmd_parameter["output_length"])
        params['context_length'] = input_len + output_len

    #
    _adjust_tensor_parallelism(
        params,
        ctx["device_count"],
        TensorParallelConfig(
            tp_key='tp_size',
            if_distributed=ctx['distributed'],
            node_ips=ctx.get("node_ips"),
            nnodes=ctx.get("nnodes"),
        ),
    )

    # sglang 4.10.0--enable-ep-moe is deprecated
    # 注意：必须检查值为 truthy 而非仅检查 key 存在，
    # 因为 sglang_default.json 中 enable_ep_moe 默认为 null，
    # 且参数映射会将 enable_expert_parallel=False 写入为 enable_ep_moe=False，
    # 若仅检查 key 存在会导致 EP 被错误启用。
    ep_moe_val = params.pop("enable_ep_moe", None)
    if ep_moe_val:
        params['ep_size'] = params['tp_size']

    # 处理 tool parser 参数（function call 支持）
    # 用户通过 enable_auto_tool_choice 统一触发，映射到 SGLang 的 tool_call_parser。
    # reasoning_parser 不适用于 SGLang（已整体移出 reasoning，仅 vllm/vllm_ascend 注入），此处只处理 tool_call_parser。
    params.pop("enable_tool_choice", None)  # 清理旧参数名
    user_wants_fc = engine_cmd_parameter.get("enable_auto_tool_choice")
    if user_wants_fc:
        if "tool_call_parser" in params:
            logger.info(
                "Function Call enabled for SGLang (parser=%s)",
                params["tool_call_parser"],
            )
        else:
            logger.warning("enable_auto_tool_choice is set but SGLang model has no tool_call_parser configured")
    else:
        params.pop("tool_call_parser", None)
        logger.info("Function Call not enabled for SGLang")

    # SGLang 不处理 reasoning_parser（整体移出 reasoning）；思考开关若误用于 sglang，
    # 由 _warn_thinking_switch_unsupported_engine 在分发处统一提醒，此处不再触碰。
    return params


@dataclass
class TensorParallelConfig:
    """张量并行相关配置，用于封装 _adjust_tensor_parallelism 的多个参数。"""
    tp_key: str
    if_distributed: bool = False
    node_ips: str | None = None
    nnodes: int | None = None


def _adjust_tensor_parallelism(
    params,
    device_count,
    tp_config: TensorParallelConfig,
):
    """设置张量并行度（TP）参数。

    - 非分布式模式：TP = 当前节点设备数
    - 分布式模式（非 PD）：TP = 设备数 × 节点数，实现全局 TP
    - 若已有用户设置则不覆盖
    """
    tp_key = tp_config.tp_key
    if_distributed = tp_config.if_distributed
    node_ips = tp_config.node_ips
    nnodes = tp_config.nnodes
    default_tp = params.get(tp_key)
    if default_tp is not None:
        if not if_distributed and default_tp != int(device_count):
            logger.warning(
                "Detected %s devices in current environment, "
                "while default recommended TP is %s, "
                "keeping explicitly configured TP value",
                device_count, default_tp,
            )
        return
    if not if_distributed:
        # 300I A2 标卡为 4 张或 8 张时，强制 TP=4（PCIe 拓扑限制）
        try:
            is_pcie_300i, _ = check_pcie_cards("d802", "4000")
            if is_pcie_300i and int(device_count) in [4, 8]:
                params[tp_key] = 4
                logger.info("Detected 300I A2 PCIe card with %s devices, set TP=4", device_count)
                return
        except Exception as e:
            logger.debug("PCIe card detection skipped in sidecar: %s", e)  # Sidecar 环境可能无法访问 PCIe
        params[tp_key] = int(device_count)
    else:
        n_nodes = _resolve_distributed_node_count(node_ips, nnodes)
        # PD+
        if get_pd_role_env():
            params[tp_key] = int(device_count)
        else:
            params[tp_key] = int(device_count) * n_nodes


# 额外的通用参数名 → 引擎原生参数名映射（engine_parameter_mapping.json 中未覆盖的常见参数）
_EXTRA_KEY_TRANSLATION: Dict[str, Dict[str, str]] = {
    "sglang": {
        "max_model_len": "context_length",
    },
    "mindie": {
        "max_model_len": "maxInputTokenLen",
    },
}


def _translate_user_config_for_engine(user_config: Dict[str, Any],
                                      engine: str) -> Dict[str, Any]:
    """将用户配置文件中的通用参数名翻译为引擎原生参数名。

    用户可能在 config-file 中使用 vLLM 风格的参数名（如 gpu_memory_utilization、
    max_model_len）。对于 SGLang / MindIE 等引擎，需要先将这些键翻译为引擎原生
    参数名（如 mem_fraction_static、context_length），再参与配置合并。

    这确保 user_config 中的值会正确覆盖 engine_specific_defaults 中的同名键，
    而不是作为多余的未知键被引擎拒绝。

    Args:
        user_config: 用户配置文件解析后的字典
        engine:      引擎标识 (vllm / sglang / mindie / vllm-ascend)

    Returns:
        Dict[str, Any]: 翻译后的配置字典
    """
    if not user_config or engine in ("vllm", "vllm-ascend"):
        return user_config

    mapping_name = {
        "sglang": "default_to_sglang_parameter_mapping",
        "mindie": "default_to_mindie_parameter_mapping",
    }.get(engine)
    if not mapping_name:
        return user_config

    config_path = os.path.join(DEFAULT_CONFIG_DIR,
                               DEFAULT_CONFIG_FILES.get("engine_parameter_mapping"))
    param_map = _load_mapping(config_path, mapping_name)

    extra_map = _EXTRA_KEY_TRANSLATION.get(engine, {})

    translated: Dict[str, Any] = {}
    for key, value in user_config.items():
        if key in param_map and param_map[key]:
            translated[param_map[key]] = value
        elif key in extra_map:
            translated[extra_map[key]] = value
        else:
            translated[key] = value

    if translated != user_config:
        logger.info("Translated user config keys for engine '%s': %s → %s",
                     engine, list(user_config.keys()), list(translated.keys()))
    return translated


def _merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并多个配置字典（后者覆盖前者，嵌套 dict 递归合并）。

    合并规则：
    - 若同一个 key 在多个字典中都是 dict，则递归合并；
    - 否则后续字典的值直接覆盖之前的值。

    Args:
        *configs: 任意数量的字典，按顺序从低优先级到高优先级传入

    Returns:
        Dict[str, Any]: 深度合并后的字典
    """
    merged = {}
    for config in configs:
        if not isinstance(config, dict):
            continue #

        for key, value in config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                #
                merged[key] = _merge_configs(merged[key], value)
            else:
                #
                merged[key] = value
    return merged


def _load_default_config(hardware_env: Dict[str, Any]) -> Dict[str, Any]:
    """根据硬件类型（nvidia/ascend）加载对应的默认引擎配置文件。

    加载策略：
      1. 优先加载 vllm_default.json（新版统一配置）
      2. 若 vllm_default.json 不存在 → 回退到 <device>_default.json（旧版布局）
      3. 若 vllm_default.json 存在但缺少 model_deploy_config →
         尝试从旧版 <device>_default.json 中补充 model_deploy_config（兼容旧配置）

    兼容说明：
      旧版使用 nvidia_default.json / ascend_default.json，其中包含按模型细分的
      model_deploy_config 段落。新版统一使用 vllm_default.json，但如果部署环境
      中同时存在旧版配置文件，会自动合并其中的 model_deploy_config 到新版配置中。
    """
    device_key = 'device'
    device_type = hardware_env.get(device_key, "nvidia")
    if device_type not in SUPPORTED_DEVICE_TYPES:
        logger.warning("Unsupported device type '%s', fallback to 'nvidia'", device_type)
        device_type = "nvidia"
    default_file = DEFAULT_CONFIG_FILES.get(device_type)
    default_config_path = os.path.join(DEFAULT_CONFIG_DIR, default_file)
    if not os.path.exists(default_config_path) and default_file == "vllm_default.json":
        legacy_file = f"{device_type}_default.json"
        legacy_path = os.path.join(DEFAULT_CONFIG_DIR, legacy_file)
        if os.path.exists(legacy_path):
            logger.warning("Fallback to legacy default config: %s", legacy_path)
            default_config_path = legacy_path
    logger.info("Determined default config file for hardware environment '%s': %s", device_type, default_config_path)
    config = load_json_config(default_config_path)

    # 兼容旧版：若主配置缺少 model_deploy_config，尝试从旧版设备配置文件中补充
    if "model_deploy_config" not in config and default_file == "vllm_default.json":
        legacy_file = f"{device_type}_default.json"
        legacy_path = os.path.join(DEFAULT_CONFIG_DIR, legacy_file)
        if os.path.exists(legacy_path):
            legacy_config = load_json_config(legacy_path)
            if "model_deploy_config" in legacy_config:
                config["model_deploy_config"] = legacy_config["model_deploy_config"]
                logger.info(
                    "Supplemented model_deploy_config from legacy config: %s",
                    legacy_path,
                )
    return config


def _load_engine_fallback_defaults(engine: str) -> Dict[str, Any]:
    """加载特定引擎的兜底默认配置（sglang_default.json / mindie_default.json）。

    当 vllm_default.json 或 model_deploy_config 中没有该引擎的专属配置项时，
    从引擎专属默认文件加载参数。vllm/vllm_ascend 复用公共默认配置，无需此步骤。
    """
    fallback_file = DEFAULT_CONFIG_FILES.get(engine)
    if not fallback_file:
        logger.debug("No engine-level fallback config for engine='%s'", engine)
        return {}
    path = os.path.join(DEFAULT_CONFIG_DIR, fallback_file)
    if not os.path.exists(path):
        logger.warning(
            "Engine fallback config '%s' not found at '%s'; using empty defaults",
            fallback_file, path,
        )
        return {}
    cfg = load_json_config(path)
    logger.info("Loaded engine-level fallback defaults from '%s'", path)
    return cfg


def _normalize_user_config_keys(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize top-level user config keys from kebab-case to snake_case.

    ``--config-file`` users often copy CLI flag names such as
    ``tool-call-parser`` into JSON.  Internally, engine_config uses Python-style
    snake_case keys (``tool_call_parser``); without normalization the kebab key
    becomes a separate field and cannot override defaults.

    Only top-level keys are normalized.  Nested dict values may be engine-native
    JSON payloads and must remain untouched.
    """
    if not isinstance(user_config, dict):
        return user_config

    normalized: Dict[str, Any] = {}
    renamed: Dict[str, str] = {}
    for key, value in user_config.items():
        normalized_key = key.replace('-', '_') if isinstance(key, str) else key
        if normalized_key in normalized and normalized_key != key:
            logger.warning(
                "Duplicate user config key after normalization: %s -> %s; "
                "keeping existing value",
                key,
                normalized_key,
            )
            continue
        normalized[normalized_key] = value
        if normalized_key != key:
            renamed[str(key)] = str(normalized_key)

    if renamed:
        logger.info("Normalized user config keys: %s", renamed)
    return normalized


def _load_user_config(config) -> Dict[str, Any]:
    """加载用户自定义 JSON 配置文件，合并到默认配置之上。

    Args:
        config: 配置来源，支持两种格式：
            - 文件路径字符串（指向 JSON 文件）
            - 已反序列化的 JSON 字典对象
    """
    user_config = {}
    if not config:
        return user_config

    # 支持已反序列化的 dict 对象
    if isinstance(config, dict):
        logger.info("Config is already a dict, keys: %s", list(config.keys()))
        return _normalize_user_config_keys(config)

    if config.strip().startswith('{') and config.strip().endswith('}'):
        # JSON
        try:
            user_config = json.loads(config)
            logger.info("Successfully parsed config from JSON string, keys: %s", list(user_config.keys()))
            return _normalize_user_config_keys(user_config)
        except json.JSONDecodeError:
            logger.info("The config-file is not JSON string, will load it as a file")
    elif os.path.exists(config):
        # 路径规范化：解析符号链接，防止路径遍历攻击
        resolved = os.path.realpath(config)
        if not os.path.isfile(resolved):
            logger.warning("Config path is not a regular file: %s -> %s", config, resolved)
        else:
            logger.info("Loading user-specified config file: %s (resolved: %s)", config, resolved)
            user_config = load_json_config(resolved)
            if user_config:
                logger.info("User config loaded, keys: %s", list(user_config.keys()))
                user_config = _normalize_user_config_keys(user_config)
    else:
        logger.warning("User-specified config not found or invalid: %s", config)

    return user_config


def _process_cmd_args(known_args: argparse.Namespace) -> Dict[str, Any]:
    """将 argparse.Namespace 转为字典，过滤掉 None 值和 config_file 键。

    config_file 由 _load_user_config 单独处理，不参与引擎参数合并。
    """
    cmd_known_params = {
        k: v
        for k, v in vars(known_args).items()
        if v is not None and k not in ["config_file", "engine_config"]
    }
    return cmd_known_params


# 通用 launcher CLI/ENV 参数与其对应环境变量的映射表。
# 用于 _detect_explicit_cli_keys() 判断哪些参数是用户显式指定的。
#
# 约束：这里仅放 parser 层通用参数；vLLM 原生参数放到
# _VLLM_CLI_ENV_MAP，避免 MindIE/SGLang 被 vLLM 专属环境变量污染。
_COMMON_CLI_ENV_MAP: Dict[str, str] = {
    "gpu_memory_utilization": "GPU_MEMORY_UTILIZATION",
    "max_num_seqs": "MAX_NUM_SEQS",
    "block_size": "BLOCK_SIZE",
    "seed": "SEED",
    "dtype": "DTYPE",
    "kv_cache_dtype": "KV_CACHE_DTYPE",
    "quantization": "QUANTIZATION",
    "host": "HOST",
    "port": "PORT",
    "input_length": "INPUT_LENGTH",
    "output_length": "OUTPUT_LENGTH",
    "max_num_batched_tokens": "MAX_NUM_BATCHED_TOKENS",
    "trust_remote_code": "TRUST_REMOTE_CODE",
    "enable_chunked_prefill": "ENABLE_CHUNKED_PREFILL",
    "enable_prefix_caching": "ENABLE_PREFIX_CACHING",
    "enable_expert_parallel": "ENABLE_EXPERT_PARALLEL",
}


# vLLM / vLLM-Ascend 原生覆盖参数。
# 这些参数不是 MindIE/SGLang 的通用启动契约，只在最终 engine 为
# vllm/vllm_ascend 时参与显式覆盖判断。
_VLLM_CLI_ENV_MAP: Dict[str, str] = {
    "no_enable_prefix_caching": "NO_ENABLE_PREFIX_CACHING",
    "enforce_eager": "ENFORCE_EAGER",
    "data_parallel_size": "DATA_PARALLEL_SIZE",
    "tensor_parallel_size": "TENSOR_PARALLEL_SIZE",
}


def _resolve_explicit_env_map(engine: str | None = None) -> Dict[str, str]:
    """根据当前引擎返回允许参与显式覆盖判断的 ENV 映射。"""
    resolved_engine = (engine or os.environ.get("WINGS_ENGINE") or os.environ.get("ENGINE") or "").lower()
    env_map = dict(_COMMON_CLI_ENV_MAP)
    if resolved_engine in {"vllm", "vllm_ascend"}:
        env_map.update(_VLLM_CLI_ENV_MAP)
    return env_map


def _detect_explicit_cli_keys(engine: str | None = None) -> set:
    """检测用户通过 CLI 参数或环境变量显式设定的参数键集合。

    识别规则：
    1. 检查 sys.argv 中的 --xxx 参数名 → 转为 snake_case 加入集合
    2. 检查当前引擎允许的 ENV 映射中对应的环境变量是否被设置

    Returns:
        set: 用户显式设定的参数键名集合 (snake_case)
    """
    import sys as _sys
    explicit = set()

    for arg in _sys.argv:
        if arg.startswith("--"):
            key = arg.lstrip("-").split("=")[0].replace("-", "_")
            explicit.add(key)

    for param_key, env_var in _resolve_explicit_env_map(engine).items():
        if os.environ.get(env_var) is not None:
            explicit.add(param_key)

    return explicit


def _cast_env_value(env_val: str, cfg_type: type) -> Any:
    """将环境变量字符串值按目标类型做类型转换。"""
    try:
        if cfg_type == float:
            return float(env_val)
        if cfg_type == int:
            return int(env_val)
        if cfg_type == bool:
            return env_val.lower() in ('true', '1', 'yes')
        return env_val
    except (ValueError, TypeError):
        return env_val


def _apply_cli_overrides(engine_config: Dict[str, Any],
                         cmd_known_params: Dict[str, Any]) -> Dict[str, Any]:
    """将用户显式指定的 CLI/ENV 参数覆盖到 engine_config 中。

    遵循优先级规则：CLI/ENV > config-file > model_default > hardware_default。
    仅覆盖用户显式设定的参数（通过 _detect_explicit_cli_keys 检测），
    不会用 argparse 的默认值错误覆盖 config-file 中的有效值。

    Args:
        engine_config:    已合并 defaults + user_config 的引擎参数字典
        cmd_known_params: CLI 解析后的完整参数字典

    Returns:
        Dict[str, Any]: 应用 CLI 覆盖后的 engine_config（原地修改并返回）
    """
    env_map = _resolve_explicit_env_map(cmd_known_params.get("engine"))
    explicit_keys = _detect_explicit_cli_keys(cmd_known_params.get("engine"))

    for key in list(engine_config.keys()):
        if key not in explicit_keys:
            continue

        # 优先从 argparse 解析结果取值，若无则从环境变量回退读取
        if key in cmd_known_params:
            cli_val = cmd_known_params[key]
        elif key in env_map:
            env_val = os.environ.get(env_map[key])
            if env_val is None:
                continue
            # 按 engine_config 中已有值的类型做类型转换
            cfg_type = type(engine_config.get(key))
            cli_val = _cast_env_value(env_val, cfg_type)
        else:
            continue

        cfg_val = engine_config[key]
        if str(cli_val) != str(cfg_val):
            logger.info(
                "CLI/ENV override: engine_config[%s] = %s (was %s from config-file/defaults)",
                key, cli_val, cfg_val,
            )
            engine_config[key] = cli_val

    return engine_config


def _write_engine_second_line(path: str, engine: str) -> None:
    """将引擎名称写入标记文件的第 2 行。

    标记文件（如 /var/log/wings/wings.txt）用于进程间通信和运维排查。
    第 1 行预留给 PID，第 2 行记录当前使用的引擎名称。
    文件不存在时自动创建。
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        lines = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

        if len(lines) == 0:
            lines = ["", engine]
        elif len(lines) == 1:
            lines.append(engine)
        else:
            lines[1] = engine

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        logger.warning("Write engine marker file failed (%s): %s", path, e)



def _resolve_device_name(hardware_env: Dict[str, Any]) -> str:
    """Return the first detected device name, or a stable fallback."""
    if hardware_env.get('details'):
        return hardware_env.get('details')[0]['name']
    return 'unknown'


def _resolve_engine_choice(
    device_type: str,
    device_name: str,
    gpu_usage_mode: str,
    cmd_known_params: Dict[str, Any],
    model_info,
) -> str:
    """Select an engine when missing, otherwise validate the user-provided engine."""
    engine = cmd_known_params.get("engine")
    if not engine:
        if device_type == "nvidia" and (
            cmd_known_params.get("enable_sparse") or cmd_known_params.get("enable_speculative_decode")
        ):
            logger.info("vLLM-only runtime feature enabled, automatically selected engine: vllm")
            return "vllm"
        return _select_engine_automatically(device_type, device_name, gpu_usage_mode, model_info)
    return _validate_user_engine(engine, device_name, gpu_usage_mode, model_info)


def _prepare_mindie_model_config(cmd_known_params: Dict[str, Any], device_name: str) -> None:
    """Apply MindIE-specific model config permission and dtype checks."""
    config_json_file = os.path.join(cmd_known_params.get('model_path'), 'config.json')
    if not check_permission_640(config_json_file):
        try:
            os.chmod(config_json_file, 0o640)
            logger.info(
                "The permission setting for model config.json is not set to 640. "
                "Since MindIE only supports the 640 permission configuration, we will adjust it to 640."
            )
        except Exception as e:
            logger.warning("Failed to set permission for model config.json to 640: %s.", e)
    if "310" in device_name:
        check_torch_dtype(config_json_file)


def _apply_engine_runtime_flags(cmd_known_params: Dict[str, Any]) -> None:
    """Set feature environment variables driven by selected engine parameters."""
    _set_spec_decoding_config(cmd_known_params)
    _set_sparse_config(cmd_known_params)
    _set_rag_acc_config(cmd_known_params)


def _record_selected_engine(engine: str) -> None:
    """Persist the selected engine for sidecar coordination and diagnostics."""
    _write_engine_second_line(os.getenv("BACKEND_PID_FILE", "/var/log/wings/wings.txt"), engine)
    os.environ['WINGS_ENGINE'] = engine
    logger.info("Set global environment variable WINGS_ENGINE=%s", engine)


def _warn_unsupported_lmcache_engine(engine: str) -> None:
    """Warn when LMCache is enabled for an engine that will ignore it."""
    if get_lmcache_env() and engine not in {"vllm", "vllm_ascend"}:
        logger.warning(
            "[KVCache Offload] LMCACHE_OFFLOAD is enabled, but selected engine '%s' "
            "does not support LMCache offload. Offload settings will be ignored unless "
            "engine is explicitly set to vllm or vllm_ascend.",
            engine,
        )


def _set_final_device_count(
    hardware_env: Dict[str, Any],
    cmd_known_params: Dict[str, Any],
) -> None:
    """Resolve and validate final device_count."""
    if cmd_known_params.get("gpu_usage_mode") == "full":
        device_count = hardware_env.get("count", 1)
    else:
        device_count = cmd_known_params.get("device_count", 1)
    if device_count <= 0:
        raise ValueError(f"device_count must be an integer greater than 0. Current value: {device_count}")
    cmd_known_params['device_count'] = device_count


def _auto_select_engine(hardware_env: Dict[str, Any],
                       cmd_known_params: Dict[str, Any],
                       model_info) -> Dict[str, Any]:
    """Select and initialize the engine-related runtime parameters."""
    device_type = hardware_env['device']
    device_name = _resolve_device_name(hardware_env)
    gpu_usage_mode = cmd_known_params.get("gpu_usage_mode", "full")
    engine = _resolve_engine_choice(device_type, device_name, gpu_usage_mode, cmd_known_params, model_info)
    if engine == 'mindie':
        _prepare_mindie_model_config(cmd_known_params, device_name)

    cmd_known_params["model_type"] = model_info.identify_model_type()
    cmd_known_params["engine"] = engine
    _apply_engine_runtime_flags(cmd_known_params)

    final_engine = cmd_known_params.get("engine", engine)
    _warn_unsupported_lmcache_engine(final_engine)
    _record_selected_engine(final_engine)

    if cmd_known_params.get("distributed"):
        _handle_distributed(final_engine, cmd_known_params, model_info)

    _set_final_device_count(hardware_env, cmd_known_params)
    return cmd_known_params


def _select_engine_automatically(device_type: str,
                                 device_name: str,
                                 gpu_usage_mode: str,
                                 model_info) -> str:
    """根据设备类型、模型特征自动选择最合适的推理引擎。"""
    if device_type == "nvidia":
        return _select_nvidia_engine(gpu_usage_mode, model_info)
    elif device_type == "ascend":
        return _select_ascend_engine(device_name, model_info)
    else:
        logger.info("No engine specified, automatically selected engine: vllm")
        return 'vllm'


def _select_nvidia_engine(gpu_usage_mode: str, model_info) -> str:
    """NVIDIA GPU 场景下的引擎自动选择逻辑。

    优先级（由高到低）：
    1. 启用 PD 分离 → vllm
    2. 启用 Wings Router → vllm
    3. MIG 模式 → vllm
    4. embedding/rerank 模型 → vllm
    5. wings 已验证模型 → sglang（推荐高性能路径）
    6. 其他未验证架构 → vllm（兜底）
    """
    model_architecture = model_info.model_architecture
    model_type = model_info.identify_model_type()
    is_wings_supported = model_info.is_wings_supported()
    vllm = 'vllm'
    if get_pd_role_env():
        logger.info("PD enabled, automatically switched to VLLM engine")
        return vllm
    elif get_router_env():
        logger.info("Wings router enabled, automatically switched to VLLM engine")
        return vllm
    elif gpu_usage_mode == "mig":
        logger.info("Device is Mig, automatically switched to VLLM engine")
        return vllm
    elif model_type in ["embedding", "rerank"]:
        logger.info("model type is %s, automatically switched to VLLM engine", model_type)
        return vllm
    elif is_wings_supported:
        logger.info("No engine specified, automatically selected engine: sglang")
        return 'sglang'
    else:
        logger.warning("This model architecture %s has not been validated on Wings. "
                       "automatically switched to VLLM engine", model_architecture)
        return vllm


def _select_ascend_engine(device_name: str, model_info) -> str:
    """华为昇腾 NPU 场景下的引擎自动选择逻辑。

    优先级（由高到低）：
    1. Ascend310 → 强制 mindie（vllm_ascend 不支持 310 系列）
    2. embedding / rerank 模型 → vllm_ascend
    3. 算子加速（USE_KUNLUN_ATB）启用 → vllm_ascend
    4. Wings Router 启用 → vllm_ascend
    5. Soft FP8 量化启用 → vllm_ascend
    6. Wings 已验证模型 → mindie（Ascend 上的推荐引擎）
    7. 未验证架构 → vllm_ascend（兜底）

    Args:
        device_name: 设备型号名称，含 '310' 表示昇腾 310 系列
        model_info:  模型元信息对象

    Returns:
        str: 选定的引擎名称 'mindie' 或 'vllm_ascend'

    Raises:
        ValueError: 昇腾 310 不支持 embedding/rerank 模型
    """
    model_architecture = model_info.model_architecture
    model_type = model_info.identify_model_type()
    is_wings_supported = model_info.is_wings_supported()
    if "310" in device_name:
        if model_type in ["embedding", "rerank"]:
            raise ValueError(f"Ascend310 not support {model_type} model currenly")
        logger.info("Ascend310 not support vllm ascend, automatically selected engine: mindie")
        return 'mindie'
    elif model_type in ["embedding", "rerank"]:
        logger.info("model type is %s, automatically switched to VLLM engine", model_type)
        return "vllm_ascend"
    elif get_operator_acceleration_env():
        logger.warning("operator_acceleration is enabled, "
                       "automatically switched to VLLM_Ascend engine")
        return "vllm_ascend"
    elif get_router_env():
        logger.info("Wings router enabled, automatically switched to VLLM engine")
        return "vllm_ascend"
    elif get_soft_fp8_env():
        logger.warning("soft fp8 is enabled, "
                       "automatically switched to VLLM_Ascend engine")
        return "vllm_ascend"
    elif model_architecture in ["DeepseekV32ForCausalLM", "Qwen3NextForCausalLM",
                                  "DeepseekV4ForCausalLM",
                                  "Glm4MoeForCausalLM",
                                  "GlmMoeDsaForCausalLM",
                                  "Qwen3_5ForConditionalGeneration",
                                  "Qwen3_5MoeForConditionalGeneration",
                                  "MiniMaxM2ForCausalLM",
                                  "KimiK25ForConditionalGeneration"]:
        logger.info("Model architecture %s requires vllm_ascend, automatically selected", model_architecture)
        return "vllm_ascend"
    elif is_wings_supported:
        logger.info("No engine specified, automatically selected engine: mindie")
        return 'mindie'
    else:
        logger.warning("This model architecture %s has not been validated on Wings. "
                       "automatically switched to VLLM_Ascend engine", model_architecture)
        return "vllm_ascend"


def _validate_user_engine(engine: str, device_name: str, gpu_usage_mode: str, model_info) -> str:
    """校验用户指定的引擎名称是否合法，engine 参数具有最高优先级，不做任何自动覆盖。

    参数:
        engine (str): 用户指定引擎，支持 'mindie', 'vllm', 'vllm_ascend', 'sglang'
        device_name (str): 设备型号名称
        gpu_usage_mode (str): GPU 使用模式
        model_info: 模型信息对象

    返回:
        str: 用户传入的引擎名（原样返回）

    异常:
        ValueError: 引擎名不在支持列表中时抛出
    """
    if engine not in ['mindie', 'vllm', 'vllm_ascend', 'sglang']:
        raise ValueError(
            f"The engine {engine} is not supported yet! "
            "Please change to 'mindie', 'vllm', 'vllm_ascend' or 'sglang'"
        )
    return engine


def _handle_mindie_distributed(distributed_config: Dict[str, Any], cmd_params: Dict[str, Any]):
    """注入 MindIE 多节点分布式所需的 MASTER_ADDR / MASTER_PORT。

    从分布式配置文件读取 master 通信端口，并通过 get_master_ip() 获取当前节点 IP，
    将结果写入 cmd_params，供 mindie_adapter 构建启动脚本时使用。
    """
    mindie_cfg = distributed_config.get('mindie_distributed', {})
    master_port = mindie_cfg.get('master_port', 27070)
    cmd_params.update({
        'mindie_master_addr': get_master_ip(),
        'mindie_master_port': master_port,
    })


def _handle_distributed(engine: str, cmd_params: Dict[str, Any], model_info):
    """根据引擎类型将分布式参数注入 cmd_params。

    从默认配置目录加载 distributed.json，并根据 engine 分发到对应的处理函数：
    - vllm / vllm_ascend → _handle_vllm_distributed（Ray 或 PD 模式）
    - sglang             → _handle_sglang_distributed（dist_port）
    - mindie             → _handle_mindie_distributed（MASTER_ADDR/PORT）
    """
    distributed_config_path = os.path.join(
        DEFAULT_CONFIG_DIR,
        DEFAULT_CONFIG_FILES.get("distributed")
    )
    distributed_config = load_json_config(distributed_config_path)

    if engine in ['vllm', 'vllm_ascend']:
        _handle_vllm_distributed(distributed_config, cmd_params, model_info)
    elif engine == 'sglang':
        _handle_sglang_distributed(distributed_config, cmd_params)
    elif engine == 'mindie':
        _handle_mindie_distributed(distributed_config, cmd_params)


def _handle_vllm_distributed(distributed_config: Dict[str, Any], cmd_params: Dict[str, Any], model_info):
    """为 vLLM / vLLM-Ascend 配置分布式推理参数。

    部署策略:
    - Ascend PD (Prefill/Decode 分离): 使用 MooncakeConnector，各实例独立运行，
      不使用 dp_deployment / NIXL。KV 传输由 MooncakeConnector + RDMA 处理。
    - NVIDIA PD 或 Ascend DeepSeek DP: 使用 NIXL 协议 (dp_deployment)。
    - 其他: 使用 Ray 作为分布式执行后端。

    端口优先来自环境变量 VLLM_DISTRIBUTED_PORT，若未设置则回退到配置文件默认值。
    """
    vllm_distributed_port = get_vllm_distributed_port()
    pd_role = get_pd_role_env()
    model_architecture = model_info.model_architecture
    is_ascend = cmd_params.get("engine") == 'vllm_ascend'
    # V4 (Flash/Pro) 与 V3/V32 走同一条 Ascend DeepSeek dp_deployment 路径；
    # 缺失 V4 会导致分布式启动回退到 Ray，与 V4-Pro 双机 NIXL 拓扑不兼容。
    is_ascend_deepseek = (model_architecture in ["DeepseekV3ForCausalLM",
                                                  "DeepseekV32ForCausalLM",
                                                  "DeepseekV4ForCausalLM",
                                                  "GlmMoeDsaForCausalLM",
                                                  "KimiK25ForConditionalGeneration"]
                          and is_ascend)

    if pd_role in ['P', 'D'] and is_ascend:
        # Ascend PD: 使用 MooncakeConnector，各 P/D 实例作为独立 vllm 进程运行。
        # KV 传输由 MooncakeConnector + RDMA TransferEngine 自动处理，
        # 不需要 dp_deployment / NIXL 参数（那些是 NVIDIA PD 路径）。
        logger.info("[PD] Ascend PD mode: standalone instances with MooncakeConnector (role=%s)", pd_role)
        return

    if (pd_role in ['P', 'D'] and not is_ascend) or is_ascend_deepseek:
        # NVIDIA PD (NIXL) 或 Ascend DeepSeek DP: 使用 dp_deployment
        if not vllm_distributed_port:
            vllm_distributed_port = distributed_config.get('vllm_distributed', {}).get('nixl_port', 27070)

        rpc_port = distributed_config.get('vllm_distributed', {}).get('rpc_port', 27071)
        cmd_params.update({
            'distributed_executor_backend': 'dp_deployment',
            'nixl_ip': get_local_ip(),
            'nixl_port': vllm_distributed_port,
            'rpc_port': rpc_port
        })
    else:
        if not vllm_distributed_port:
            vllm_distributed_port = distributed_config.get('vllm_distributed', {}).get('ray_head_port', 27070)

        cmd_params.update({
            'distributed_executor_backend': 'ray',
            'ray_head_ip': get_master_ip(),
            'ray_head_port': vllm_distributed_port
        })


def _handle_sglang_distributed(distributed_config: Dict[str, Any], cmd_params: Dict[str, Any]):
    """为 SGLang 配置分布式通信端口 dist_port。

    端口优先来自环境变量 SGLANG_DISTRIBUTED_PORT，
    若未设置则回退到 distributed.json 中 sglang_distributed.dist_port 的默认值。
    """
    dist_port = get_sglang_distributed_port()
    if not dist_port:
        dist_port = distributed_config.get('sglang_distributed', {}).get('dist_port', 25565)

    cmd_params.update({
        'dist_port': dist_port
    })


_V4_PREFIX_MODEL_CONFIG_KEYS = {
    "deepseek-v4-flash",
    "deepseek-v4-pro",
}
_MODEL_NAME_TOKEN_BOUNDARY_CHARS = set("-_./:")

# W4A8 量化别名集 —— DeepSeek-V4-Pro 的权重指纹（与
# ``utils.vllm_helpers._W4A8_QUANT_METHOD_ALIASES`` 保持一致，避免在 config_loader
# 中再依赖 vllm_helpers，从而保留这层模块边界）。
_W4A8_QUANTIZE_ALIASES = {
    "w4a8",
    "ascend-w4a8",
    "ascend_w4a8",
    "w4a8_dynamic",
}


def _is_w4a8_quantize_token(quantize: Any) -> bool:
    if not quantize:
        return False
    q = str(quantize).strip().lower()
    if not q:
        return False
    if q in _W4A8_QUANTIZE_ALIASES:
        return True
    return "w4a8" in q


# V4-Pro 身份证据可能藏在 config.json 的这些字段里（与 adapter
# ``_DEEPSEEK_V4_IDENTITY_CONFIG_KEYS`` 保持一致）。生产环境会出现 served-model
# 改名导致 model_name 不含 "pro"，但权重目录里的 config.json 仍然保留原始名称。
_V4_PRO_IDENTITY_CONFIG_KEYS = (
    "_name_or_path",
    "name_or_path",
    "model_name",
    "model_id",
    "base_model_name",
    "source_model",
)


def _config_value_contains_v4_pro(value: Any) -> bool:
    if isinstance(value, str):
        text = value.lower()
        return "deepseek-v4-pro" in text or "deepseek_v4_pro" in text or "deepseekv4pro" in text
    if isinstance(value, (list, tuple, set)):
        return any(_config_value_contains_v4_pro(item) for item in value)
    if isinstance(value, dict):
        return any(_config_value_contains_v4_pro(item) for item in value.values())
    return False


def _fingerprint_model_config_keys(model_info) -> list:
    """根据 model_info 的架构 + 量化/identity 指纹补充 model_deploy_config 查找名。

    设计动机：JSON 的 ``DeepSeek-V4-Pro`` 等条目以模型名为 key，但生产环境会出现
    匿名 served-model（CLI model_name 不含 "pro"），其身份信号要么藏在 config.json
    的 _name_or_path 等字段，要么只能靠 w4a8 量化指纹识别。这里把这两类兜底
    集中到一处，以替代 adapter 内部的硬编码默认。
    """
    if model_info is None:
        return []
    try:
        architecture = model_info.identify_model_architecture()
    except Exception:  # noqa: BLE001
        architecture = None
    if architecture != "DeepseekV4ForCausalLM":
        return []

    config = getattr(model_info, "config", None) or {}
    for key in _V4_PRO_IDENTITY_CONFIG_KEYS:
        if _config_value_contains_v4_pro(config.get(key)):
            return ["deepseek-v4-pro"]

    quantize = getattr(model_info, "model_quantize", None)
    if _is_w4a8_quantize_token(quantize):
        return ["deepseek-v4-pro"]
    return []


def _model_name_contains_config_token(model_name_lower: str, config_key_lower: str) -> bool:
    """Return True when config_key appears as a token inside model_name."""
    start = 0
    while True:
        index = model_name_lower.find(config_key_lower, start)
        if index < 0:
            return False

        end = index + len(config_key_lower)
        before_ok = (
            index == 0
            or model_name_lower[index - 1] in _MODEL_NAME_TOKEN_BOUNDARY_CHARS
        )
        after_ok = (
            end == len(model_name_lower)
            or model_name_lower[end] in _MODEL_NAME_TOKEN_BOUNDARY_CHARS
        )
        if before_ok and after_ok:
            return True
        start = index + 1


def _model_config_key_matches_lookup_names(config_key_lower: str, lookup_names: list) -> bool:
    """Match model_deploy_config keys against CLI model names."""
    if config_key_lower in lookup_names:
        return True
    if config_key_lower not in _V4_PREFIX_MODEL_CONFIG_KEYS:
        return False
    return any(
        _model_name_contains_config_token(name, config_key_lower)
        for name in lookup_names
    )


def _match_model_engine_config(
    arch_dict: Dict[str, Any],
    model_name_lower: str,
    engine_key: str,
    is_deepseek_sglang_nvidia: bool,
    model_info=None,
) -> Dict[str, Any]:
    """在架构配置字典中按模型名查找引擎参数，支持 H20 卡型适配。

    遍历 arch_dict 中的模型条目，找到名称匹配项后返回对应的引擎参数。
    DeepSeek+SGLang+NVIDIA 场景下额外检测 H20 GPU 型号以选用专属配置。
    匿名模型可通过 ``model_info`` 提供的架构 + 量化指纹补充查找名（如 w4a8
    DeepseekV4 → ``deepseek-v4-pro``），以匹配 JSON 中的对应条目。

    Args:
        arch_dict:                 架构级配置（model_name → {engine_key: config}）
        model_name_lower:          待匹配的模型名（已小写化）
        engine_key:                引擎键名（如 'vllm'、'sglang_distributed'）
        is_deepseek_sglang_nvidia: 是否为 DeepSeek+SGLang+NVIDIA 特殊场景
        model_info:                可选，模型元信息对象（提供架构 + 量化指纹）

    Returns:
        匹配到的引擎参数字典；未匹配则返回空字典
    """
    h20_model = _get_h20_model_hint()

    lookup_names = [model_name_lower]
    if model_name_lower.startswith("deepseek-v4-pro-") and model_name_lower.endswith("-mtp1"):
        lookup_names.append(model_name_lower[:-1])
    for extra in _fingerprint_model_config_keys(model_info):
        if extra not in lookup_names:
            lookup_names.append(extra)

    for model, config in arch_dict.items():
        if not _model_config_key_matches_lookup_names(model.lower(), lookup_names):
            continue

        if is_deepseek_sglang_nvidia and h20_model in ("H20-96G", "H20-141G"):
            logger.info("Using dedicated config for model '%s' on %s", model, h20_model)
            return config.get(engine_key, {}).get(h20_model, {})

        if is_deepseek_sglang_nvidia:
            logger.info(
                "DeepSeek+SGLang+NVIDIA (non-H20): using engine-level config for '%s'",
                model,
            )
            return config.get(engine_key, {})

        logger.info("Using engine config for model '%s' (engine_key=%s)", model, engine_key)
        return config.get(engine_key, {})

    return {}


def _load_and_validate_models_dict(
    hardware_env: Dict[str, Any],
    model_type: str,
    engine: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """加载并校验 model_deploy_config 中对应 model_type 的配置字典。

    若结构异常则记录警告并降级为空字典。若 models_dict 最终为空，
    返回兜底 engine 默认配置供调用方直接使用。

    Args:
        hardware_env: 硬件环境信息（传递给 _load_default_config）。
        model_type:   模型类型键（如 'llm'）。
        engine:       引擎名称，用于兜底日志与查找。

    Returns:
        (models_dict, fallback)：fallback 非 None 时应直接作为引擎默认值使用。
    """
    config_model_key = "model_deploy_config"
    default_config = _load_default_config(hardware_env)
    model_deploy_config = default_config.get(config_model_key, {})
    if not isinstance(model_deploy_config, dict):
        logger.warning("Invalid default config structure: %s is not a dict", config_model_key)
        model_deploy_config = {}
    models_dict = model_deploy_config.get(model_type, {})
    if not isinstance(models_dict, dict):
        logger.warning("Invalid model config structure: model_type=%s is not a dict", model_type)
        models_dict = {}
    if not models_dict:
        logger.warning(
            "No model_deploy_config found for model_type=%s (engine=%s),"
            " try engine-level fallback defaults",
            model_type,
            engine,
        )
        return {}, _load_engine_fallback_defaults(engine)
    return models_dict, None


def _resolve_model_lookup_keys(cmd_known_params: Dict[str, Any]) -> Tuple[str, str, str]:
    """从 CLI 参数中提取模型查找所需的三元键：(model_name_lower, engine, engine_key)。"""
    model_name = cmd_known_params.get("model_name")
    if not model_name:
        logger.warning("model_name is None or empty, using empty string for config matching")
        model_name_lower = ""
    else:
        model_name_lower = model_name.lower()
    engine: str = cmd_known_params.get("engine", "")
    engine_key = f"{engine}_distributed" if cmd_known_params.get("distributed") else engine
    return model_name_lower, engine, engine_key


@lru_cache(maxsize=1)
def _load_reasoning_parser_support() -> Dict[str, Any]:
    """Load the reasoning parser matrix keyed by model architecture."""
    try:
        with REASONING_PARSER_SUPPORT_PATH.open("r", encoding="utf-8-sig") as stream:
            support = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "Failed to load reasoning parser support file %s: %s",
            REASONING_PARSER_SUPPORT_PATH,
            exc,
        )
        return {}

    architectures = support.get("architectures", [])
    if not isinstance(architectures, list):
        logger.warning(
            "Invalid reasoning parser support format: architectures must be a list"
        )
        return {}
    return {
        item["name"]: item
        for item in architectures
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _resolve_reasoning_parser_support(
    model_architecture: str,
    model_name: str,
    engine: str,
) -> Tuple[bool, Optional[str]]:
    """Resolve one parser value from reasoning_parser_support.yaml.

    Concrete model rows, including explicit ``null``, take precedence over the
    architecture config map. Distributed vLLM variants share their base engine
    mapping.
    """
    base_engine = engine.removesuffix("_distributed")
    if base_engine not in {"vllm", "vllm_ascend"}:
        return False, None

    architecture = _load_reasoning_parser_support().get(model_architecture)
    if not architecture:
        return False, None

    model_name_lower = (model_name or "").lower()
    models = architecture.get("models", {})
    if isinstance(models, dict):
        for supported_name, engine_values in models.items():
            if str(supported_name).lower() != model_name_lower:
                continue
            if isinstance(engine_values, dict) and base_engine in engine_values:
                return True, engine_values[base_engine]

    config = architecture.get("config", {})
    engine_config = config.get(base_engine, {}) if isinstance(config, dict) else {}
    if not isinstance(engine_config, dict):
        return False, None
    for configured_name, parser in engine_config.items():
        if configured_name != "default" and configured_name.lower() == model_name_lower:
            return True, parser
    if "default" in engine_config:
        return True, engine_config["default"]
    return False, None


def _apply_reasoning_parser_support(
    engine_specific_defaults: Dict[str, Any],
    model_architecture: str,
    model_name: str,
    engine_key: str,
) -> Dict[str, Any]:
    """Merge the YAML-backed reasoning parser into model defaults."""
    resolved = dict(engine_specific_defaults)
    found, parser = _resolve_reasoning_parser_support(
        model_architecture,
        model_name,
        engine_key,
    )
    if not found:
        return resolved
    if parser:
        resolved["reasoning_parser"] = parser
    else:
        resolved.pop("reasoning_parser", None)
    return resolved


def _get_model_specific_config(hardware_env: Dict[str, Any],
                             cmd_known_params: Dict[str, Any],
                             model_info) -> Dict[str, Any]:
    """获取并合并模型专属的默认部署配置（按查找链路逐级回退）。

    查找链路: 精确模型名 → 架构级默认 → 模型类型默认 → 引擎兜底默认。
    DeepSeek+SGLang+NVIDIA 场景支持按 H20 卡型选配（H20-96G / H20-141G）。

    Args:
        hardware_env:     硬件环境信息（device, gpu_memory 等）。
        cmd_known_params: 用户 CLI 已知参数（model_name, engine, distributed 等）。
        model_info:       模型元信息对象，提供 identify_model_architecture / type 方法。

    Returns:
        合并后的最终参数字典，可直接传递给各引擎适配器。
    """
    default_key = "default"
    model_name_lower, engine, engine_key = _resolve_model_lookup_keys(cmd_known_params)

    model_architecture = model_info.identify_model_architecture()
    model_type = model_info.identify_model_type()

    models_dict, fallback = _load_and_validate_models_dict(hardware_env, model_type, engine)
    if fallback is not None:
        return _merge_cmd_params(hardware_env, fallback, cmd_known_params, model_info)

    if model_architecture in models_dict:
        model_architecture_dict = models_dict[model_architecture]

        is_deepseek_sglang_nvidia = (
            model_architecture == "DeepseekV3ForCausalLM"
            and hardware_env.get("device") == "nvidia"
            and engine == "sglang"
            and not cmd_known_params.get("distributed")
        )

        engine_specific_defaults = _match_model_engine_config(
            model_architecture_dict, model_name_lower, engine_key,
            is_deepseek_sglang_nvidia, model_info,
        )
        if not engine_specific_defaults:
            logger.info("The default deploy configuration of the "
                        "model architecture %s will be used.", model_architecture)
            engine_specific_defaults = model_architecture_dict.get(default_key, {}).get(engine_key, {})
    else:
        engine_specific_defaults = models_dict.get(default_key, {}).get(engine_key, {})
        logger.info("The default deploy configuration of the model type %s will be used.", model_type)

    engine_specific_defaults = _apply_reasoning_parser_support(
        engine_specific_defaults,
        model_architecture,
        cmd_known_params.get("model_name", ""),
        engine_key,
    )
    return _merge_cmd_params(hardware_env, engine_specific_defaults, cmd_known_params, model_info)


def _merge_final_config(engine_config: Dict[str, Any],
                       cmd_known_params: Dict[str, Any]) -> Dict[str, Any]:
    """将引擎专属配置包装进最终参数字典并返回。

    把 engine_config（引擎参数子集）挂载到 cmd_known_params['engine_config'] 键下，
    作为后续引擎适配器（vllm_adapter / sglang_adapter 等）的输入。

    Args:
        engine_config:    引擎专属参数字典（由 _get_model_specific_config 生成）。
        cmd_known_params: 用户 CLI 已知参数字典（将原地修改并返回）。

    Returns:
        追加了 engine_config 字段的 cmd_known_params 字典。
    """
    cmd_known_params['engine_config'] = engine_config
    cmd_known_params['_explicit_cli_keys'] = sorted(_detect_explicit_cli_keys())

    return cmd_known_params


def load_and_merge_configs(
    hardware_env: Dict[str, Any],
    known_args: argparse.Namespace
) -> Dict[str, Any]:
    """配置加载与合并的主入口函数。

    将多层配置源从低优先级到高优先级合并：
        1. 硬件默认配置 (e.g., config/nvidia_default.json)
        2. 用户指定的配置文件 (--config-file)
        3. 用户 CLI 参数 (e.g., --model-path, --port)
        4. 引擎专属额外参数 (e.g., --tensor-parallel-size 2)

    处理流程:
        1. 提取 CLI 参数并检查 VRAM 需求
        2. 初始化 ModelIdentifier 对象获取模型元信息
        3. 自动选择/校验引擎，并处理 Ascend 专属逻辑
        4. 加载用户配置文件 (若指定)
        5. 通过 _get_model_specific_config 查找链获取引擎默认配置
        6. 合并所有配置层并返回最终参数字典

    Args:
        hardware_env: 硬件探测结果（device, count, details 等）
        known_args:   argparse.parse_known_args() 返回的已知参数

    Returns:
        Dict[str, Any]: 合并后的最终参数字典，包含 engine_config 子字典

    Raises:
        ValueError: 当 VRAM 不足、权重路径无效或引擎不兼容时抛出
    """
    logger.info("Starting config loading and merging...")
    raw_engine_config = getattr(known_args, "engine_config", None)
    raw_engine_config = raw_engine_config if isinstance(raw_engine_config, dict) else None
    inherited_explicit_keys = set(getattr(known_args, "_explicit_cli_keys", None) or [])
    # 1.
    # VRAM
    cmd_known_params = _process_cmd_args(known_args)
    if cmd_known_params.get("model_path"):
        if cmd_known_params.get("nnodes"):
            nodes_count = cmd_known_params.get("nnodes")
        else:
            nodes_count = 1
        _check_vram_requirements(cmd_known_params["model_path"], hardware_env, nodes_count)
    #
    model_info = ModelIdentifier(cmd_known_params.get("model_name"),
                                 cmd_known_params.get("model_path"),
                                 cmd_known_params.get("model_type"))



    # 2. 引擎自动选择/校验
    cmd_known_params = _auto_select_engine(hardware_env, cmd_known_params, model_info)
    if cmd_known_params.get("distributed"):
        _handle_distributed(cmd_known_params.get("engine"), cmd_known_params, model_info)

    # 3. 加载用户配置
    config = known_args.config_file
    user_config = _load_user_config(config)

    # 3.5 将 user_config 中的通用参数名翻译为引擎原生参数名
    #     (如 SGLang: gpu_memory_utilization → mem_fraction_static)
    engine_name = cmd_known_params.get("engine", "vllm")
    if user_config:
        user_config = _translate_user_config_for_engine(user_config, engine_name)

    if user_config and get_config_force_env():
        engine_config = user_config
        logger.info("CONFIG_FORCE=true: using user config exclusively, keys: %s", list(engine_config.keys()))
    else:
        engine_specific_defaults = _get_model_specific_config(hardware_env, cmd_known_params, model_info)
        engine_config = _merge_configs(engine_specific_defaults, user_config)

    # 4. CLI/ENV 参数覆盖 config-file（保证 CLI > config-file 优先级）
    engine_config = _apply_cli_overrides(engine_config, cmd_known_params)

    # 4.1 分布式 Master -> Worker 下发的 engine_config 是上层已经合并好的
    # vLLM 启动字段，应继续覆盖 Worker 本地默认值。否则
    # Worker 会只用本地环境/默认配置重建 engine_config，导致 master/worker
    # 在 seed、max_num_seqs、max_model_len、prefix-cache 等字段上不对齐。
    # 但不能把 engine_config 中所有字段都标记为显式 CLI/ENV，否则默认值也会
    # 绕过 adapter 后续的安全归一化（例如 DeepSeek DP 关闭 prefix-cache）。
    if raw_engine_config:
        engine_config = _merge_configs(engine_config, raw_engine_config)

    # 4.6 embedding/rerank 最终守卫：移除不兼容参数
    #     必须在所有合并（user_config、raw_engine_config、CLI 覆盖）之后执行，
    #     否则 user_config 或 --engine-config 中的 enable_prefix_caching /
    #     enable_chunked_prefill 会在 _merge_vllm_params 的清理之后被重新注入。
    _validate_embedding_rerank_params(engine_config, {"model_type": model_info.identify_model_type()})

    # 4.7 GLM-5.1 + NVIDIA/vLLM 硬约束：禁止 KVCache Offload。
    #     必须放在所有合并之后，确保 user_config、CLI 或 Master 下发的
    #     raw engine_config 中即使带 kv_transfer_config 也会被强制移除。
    _enforce_glm51_nvidia_no_kv_offload(
        engine_config,
        {**cmd_known_params, "device": hardware_env.get("device")},
        model_info,
    )

    # 5.
    final_engine_params = _merge_final_config(engine_config, cmd_known_params)
    if inherited_explicit_keys:
        explicit_keys = set(final_engine_params.get("_explicit_cli_keys") or [])
        explicit_keys.update(inherited_explicit_keys)
        final_engine_params["_explicit_cli_keys"] = sorted(explicit_keys)

    logger.info("Final engine_config keys: %s", list(engine_config.keys()))
    logger.info("Config merging completed.")
    return final_engine_params

