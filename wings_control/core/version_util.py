# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""ENGINE_VERSION 环境变量统一解析与规范化。

上层传入的版本号格式可能不规范，例如：
    v0.17.0-20260325, 0.14.rc1, V0.17, 0.12.0.post1, latest, nightly-20260301

本模块提供两个函数：
    - parse_engine_version_tuple(): 返回 (major, minor) 整数元组，用于版本比较
    - normalize_engine_version():   返回规范化的 "major.minor.patch" 字符串
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# 未设置或无法解析时的默认版本元组
# 与 supported_features.json 中的 is_default 版本对齐
DEFAULT_VERSION_TUPLE = (0, 17)


def parse_engine_version_tuple(ver_str: str | None = None) -> tuple[int, int]:
    """将版本字符串解析为 (major, minor) 元组。

    解析策略：
    1. 去除前导 v/V 前缀
    2. 提取前两段数字（以 '.' 分隔）
    3. 忽略后续内容（patch、预发布标签、日期后缀等）

    支持格式示例：
        "0.17"            → (0, 17)
        "0.17.0"          → (0, 17)
        "v0.17.0"         → (0, 17)
        "V0.17.0-20260325"→ (0, 17)
        "0.14.rc1"        → (0, 14)
        "0.12.0.post1"    → (0, 12)

    Args:
        ver_str: 版本字符串。若为 None，则从 ENGINE_VERSION 环境变量读取。

    Returns:
        (major, minor) 整数元组；无法解析时返回 DEFAULT_VERSION_TUPLE。
    """
    if ver_str is None:
        ver_str = os.getenv("ENGINE_VERSION", "").strip()
    else:
        ver_str = ver_str.strip()

    if not ver_str:
        return DEFAULT_VERSION_TUPLE

    # 去除前导 v/V
    cleaned = re.sub(r'^[vV]', '', ver_str)

    # 提取前两段数字
    match = re.match(r'(\d+)\.(\d+)', cleaned)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    # 可能只有一个数字（如 "17"）
    match_single = re.match(r'(\d+)', cleaned)
    if match_single:
        logger.warning(
            "ENGINE_VERSION='%s' only has major version, "
            "defaulting minor to 0",
            ver_str,
        )
        return (int(match_single.group(1)), 0)

    logger.warning(
        "ENGINE_VERSION='%s' format unrecognized, defaulting to %s",
        ver_str,
        ".".join(str(v) for v in DEFAULT_VERSION_TUPLE),
    )
    return DEFAULT_VERSION_TUPLE


def normalize_engine_version(ver_str: str | None = None) -> str:
    """将版本字符串规范化为 "major.minor.0" 格式。

    用于需要干净版本字符串的场景（如 JSON 构造、传递给 install.py 等）。
    输出 3 段格式以确保与 supported_features.json 中的版本 key（如 "0.17.0"）
    直接匹配，避免 PEP 440 字符串比较不一致问题。

    Args:
        ver_str: 版本字符串。若为 None，则从 ENGINE_VERSION 环境变量读取。

    Returns:
        规范化版本字符串，如 "0.17.0"；无法解析时返回 "0.0.0"。
    """
    major, minor = parse_engine_version_tuple(ver_str)
    return f"{major}.{minor}.0"
