# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""Thinking-mode 强制关闭策略（proxy 层）。

背景：reasoning_parser 开关只控制服务端是否「解析」思维链，控制不了模型「是否思考」。
当 launcher 解析出 enable_reasoning=False 且模型为可关闭思考的混合推理模型时，会把
关闭策略写入环境变量 ``WINGS_THINKING_OFF``（见 wings_control._export_thinking_policy_env）。
本模块在 proxy 进程读取该环境变量，对每个 ``/v1/chat/completions`` 请求强制注入/覆盖
``chat_template_kwargs``，使客户端无法绕过，从而「保证关闭后模型不触发思考」。

WINGS_THINKING_OFF 取值：
  - JSON（如 ``{"enable_thinking": false}`` / ``{"thinking": false}``）：强制注入的 kwargs
  - ``"always_on"``：模型始终推理、无法关闭思考 → 仅打印一次告警，不改写请求
  - 不设置：proxy 不介入（enable_reasoning 开启，或模型本就不思考）
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# 与 utils.model_utils.THINKING_ALWAYS_ON 保持一致（proxy 不反向依赖 launcher 包，故复制常量）。
_ALWAYS_ON = "always_on"

# 解析结果类型：dict（强制 kwargs）/ "always_on"（仅告警）/ None（不介入）
_Policy = Union[Dict[str, Any], str, None]


def _parse_policy(raw: Optional[str]) -> _Policy:
    """解析 WINGS_THINKING_OFF 环境变量值。"""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw == _ALWAYS_ON:
        return _ALWAYS_ON
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[thinking] invalid WINGS_THINKING_OFF=%r; ignoring", raw)
        return None
    if not isinstance(parsed, dict):
        logger.warning("[thinking] WINGS_THINKING_OFF must be a JSON object, got %s; ignoring",
                       type(parsed).__name__)
        return None
    return parsed


# 模块加载时解析一次（proxy 进程内策略固定，无需每请求重复解析）。
_POLICY: _Policy = _parse_policy(os.getenv("WINGS_THINKING_OFF"))
_always_on_warned = False


def reload_policy() -> None:
    """重新从环境变量读取策略（主要供测试使用）。"""
    global _POLICY, _always_on_warned
    _POLICY = _parse_policy(os.getenv("WINGS_THINKING_OFF"))
    _always_on_warned = False


def is_active() -> bool:
    """是否存在需要改写请求的强制关闭策略（dict 型）。always_on / None 不改写。"""
    return isinstance(_POLICY, dict)


def _warn_always_on_once() -> None:
    global _always_on_warned
    if not _always_on_warned:
        logger.warning(
            "[thinking] enable_reasoning=false but model is an always-on reasoner "
            "(e.g. DeepSeek-R1 / QwQ / MiniMax-M2); thinking cannot be disabled, "
            "only reasoning parsing is affected."
        )
        _always_on_warned = True


def apply_to_chat_body(body_bytes: bytes, upstream_path: str) -> bytes:
    """对 /v1/chat/completions 请求体应用「关闭思考」策略，返回（可能改写后的）body。

    - 非 chat/completions 路径或无策略 → 原样返回。
    - always_on → 仅告警一次，不改写（不解析 body，零额外开销）。
    - dict 策略 → 解析 body，强制写入 chat_template_kwargs 后重新序列化返回。
      解析失败/非对象 → 安全回退原 body（绝不因策略导致请求失败）。
    """
    if _POLICY is None or "chat/completions" not in upstream_path:
        return body_bytes
    if _POLICY == _ALWAYS_ON:
        _warn_always_on_once()
        return body_bytes
    try:
        payload = json.loads(body_bytes)
    except (json.JSONDecodeError, ValueError):
        return body_bytes
    if not isinstance(payload, dict):
        return body_bytes
    if enforce_thinking_off(payload):
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return body_bytes


def enforce_thinking_off(payload: Dict[str, Any]) -> bool:
    """对单个 chat 请求体强制关闭思考。

    - dict 策略：把策略键强制写入 ``payload["chat_template_kwargs"]``，覆盖客户端值，
      使客户端无法重新开启思考。返回 True 表示已改写。
    - always_on：模型无法关闭思考，打印一次告警，不改写，返回 False。
    - None：不介入，返回 False。
    """
    global _always_on_warned
    if _POLICY is None:
        return False
    if _POLICY == _ALWAYS_ON:
        if not _always_on_warned:
            logger.warning(
                "[thinking] enable_reasoning=false but model is an always-on reasoner "
                "(e.g. DeepSeek-R1 / QwQ / MiniMax-M2); thinking cannot be disabled, "
                "only reasoning parsing is affected."
            )
            _always_on_warned = True
        return False

    if not isinstance(payload, dict):
        return False
    ctk = payload.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
    # 强制覆盖：即使客户端自带 enable_thinking/thinking=true 也压制为关闭。
    ctk.update(_POLICY)
    payload["chat_template_kwargs"] = ctk
    return True
