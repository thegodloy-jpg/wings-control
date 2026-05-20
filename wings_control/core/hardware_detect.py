# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""硬件环境静态探测模块 (core/hardware_detect.py)。

为 config_loader 提供设备类型和数量信息。

工作原理:
    在 sidecar 架构中，推理引擎和 sidecar 分属不同容器，因此 sidecar
    无法直接访问 GPU/NPU 硬件。本模块改为从环境变量读取硬件信息，
    而不是直接调用 torch/pynvml 进行探测。

支持的环境变量:
    - WINGS_HARDWARE_FILE: 硬件信息 JSON 文件路径
      （优先，默认 /shared-volume/hardware_info.json）
    - WINGS_DEVICE / DEVICE / HARDWARE_TYPE:
      设备类型，支持 nvidia|ascend，默认 nvidia
    - WINGS_DEVICE_COUNT / DEVICE_COUNT: 设备数量，默认 1
    - WINGS_DEVICE_NAME: 设备型号名称（可选，如 "Ascend910B"）

探测优先级:
    JSON 文件 → 环境变量 → 默认值

输出格式示例::

    {
        "device": "nvidia" | "ascend",
        "count": int,
        "details": [
            {
                "device_id": int,
                "name": str,
                "total_memory": float,   # GB
                "used_memory": float,    # GB
                "free_memory": float,    # GB
                "util": int,             # GPU 利用率 % (仅 NVIDIA)
                "vendor": str            # "Nvidia" 或 "Ascend"
            }
        ],
        "units": "GB"
    }

Sidecar 契约:
    - 探测应使用最佳努力策略，不应因任何探测失败而崩溃
    - 避免破坏异构节点（混合 GPU/NPU 环境）的兼容性
"""

import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


CHIP_ALIASES = {
    "910b-32g": "910b-32",
    "910b_32g": "910b-32",
    "910b_32": "910b-32",
    "910b-64g": "910b-64",
    "910b_64g": "910b-64",
    "910b_64": "910b-64",
    "rtx_pro_5000": "rtx-pro-5000",
    "rtx-pro5000": "rtx-pro-5000",
    "h20_96": "h20-96",
    "h20_141": "h20-141",
}

KNOWN_CHIP_IDS = {
    "h20-96",
    "h20-141",
    "l20",
    "rtx-pro-5000",
    "910b-32",
    "910b-64",
    "910c",
}


def normalize_chip_id(raw: str) -> tuple[str, list[str]]:
    """Normalize hardware aliases to canonical chip IDs.

    Returns:
        tuple[str, list[str]]: canonical chip id and alias chain. Unknown input
        returns ("", aliases) so callers can keep legacy best-effort behavior.
    """
    value = (raw or "").strip().lower().replace(" ", "-")
    if not value:
        return "", []
    canonical = CHIP_ALIASES.get(value, value)
    if canonical in KNOWN_CHIP_IDS:
        aliases = [raw]
        if canonical != raw:
            aliases.append(canonical)
        return canonical, aliases
    return "", [raw]


def _memory_gb(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _infer_chip_id(device: str, details: list[dict[str, Any]], device_name: str = "") -> tuple[str, list[str]]:
    """Infer a canonical chip ID from static sidecar hardware hints."""
    candidates = []
    if device_name:
        candidates.append({"name": device_name})
    candidates.extend([d for d in details if isinstance(d, dict)])

    for detail in candidates:
        name = str(detail.get("name") or detail.get("model") or "").strip().lower()
        memory = _memory_gb(
            detail.get("total_memory")
            or detail.get("memory_total")
            or detail.get("hbm_gb")
        )
        normalized, aliases = normalize_chip_id(name)
        if normalized:
            return normalized, aliases
        if device == "ascend":
            if "910c" in name:
                return "910c", [name]
            if "910b" in name or "ascend910b" in name:
                if memory and memory <= 40:
                    return "910b-32", [name]
                if memory and memory > 40:
                    return "910b-64", [name]
                return "910b-64", [name]
        elif device == "nvidia":
            if "h20" in name:
                if memory and memory >= 120:
                    return "h20-141", [name]
                return "h20-96", [name]
            if "l20" in name:
                return "l20", [name]
            if "rtx" in name and "5000" in name:
                return "rtx-pro-5000", [name]
    return "", []


def _attach_chip_context(
    data: Dict[str, Any],
    *,
    explicit_chip: str = "",
    source: str = "inferred",
) -> Dict[str, Any]:
    """Attach canonical chip fields while preserving legacy keys."""
    aliases: list[str] = []
    chip_id = ""
    chip_source = source

    if explicit_chip:
        chip_id, aliases = normalize_chip_id(explicit_chip)
        chip_source = "cli"
    if not chip_id:
        file_chip = str(data.get("chip_id") or "")
        if file_chip:
            chip_id, aliases = normalize_chip_id(file_chip)
            chip_source = "hardware_file"
    if not chip_id:
        chip_id, aliases = _infer_chip_id(
            str(data.get("device") or ""),
            data.get("details") if isinstance(data.get("details"), list) else [],
            os.getenv("WINGS_DEVICE_NAME", "").strip(),
        )
        chip_source = "inferred" if chip_id else "unknown"

    data["chip_id"] = chip_id
    data["chip_source"] = chip_source
    data["chip_aliases"] = aliases
    return data


def _normalize_device(raw: str) -> str:
    """将用户输入的设备类型字符串标准化为内部统一格式。

    支持多种别名映射：
      - 'nvidia'/'gpu'/'cuda' -> 'nvidia'
      - 'ascend'/'npu'        -> 'ascend'
      - 其他未识别的值回退到 'nvidia'

    Args:
        raw: 原始设备类型字符串（大小写不敏感）

    Returns:
        str: 标准化后的设备类型 ('nvidia' 或 'ascend')
    """
    val = (raw or "").strip().lower()
    mapping = {
        "nvidia": "nvidia",
        "gpu": "nvidia",
        "cuda": "nvidia",
        "ascend": "ascend",
        "npu": "ascend",
    }
    return mapping.get(val, "nvidia")


def _parse_count(raw: str) -> int:
    """解析设备数量字符串，确保返回至少为 1 的正整数。

    异常输入（非数字字符串、负数、零）均回退到默认值 1，
    避免配置错误导致 launcher 崩溃。

    Args:
        raw: 原始设备数量字符串

    Returns:
        int: 解析后的设备数量（>= 1）
    """
    try:
        value = int((raw or "1").strip())
        return value if value > 0 else 1
    except Exception as _:
        return 1


def _load_hardware_from_file(file_path: str) -> Dict[str, Any]:
    """从 JSON 文件加载完整硬件信息。

    JSON 文件应包含与原始 wings 项目 detect_hardware() 输出一致的格式：
      - device:  设备类型 ('nvidia' 或 'ascend')
      - count:   设备数量
      - details: 每张卡的详细信息列表
      - units:   显存单位 ('GB')

    Args:
        file_path: JSON 文件的绝对路径

    Returns:
        Dict[str, Any]: 硬件信息字典

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误
        ValueError: 缺少必要字段或字段类型错误
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 校验必要字段
    if not isinstance(data, dict):
        raise ValueError("hardware info JSON must be a dict")
    if "device" not in data or "count" not in data:
        raise ValueError("hardware info JSON must contain 'device' and 'count' fields")

    # 标准化 device 字段
    data["device"] = _normalize_device(data["device"])
    data.setdefault("details", [])
    data.setdefault("units", "GB")

    # 确保 count 为正整数
    try:
        data["count"] = max(int(data["count"]), 1)
    except (TypeError, ValueError):
        data["count"] = len(data["details"]) if data["details"] else 1

    return data


def detect_hardware(chip: str = "") -> Dict[str, Any]:
    """探测硬件环境信息。

        按以下优先级获取硬件信息：
            1. 显式 chip 参数 / WINGS_CHIP（仅补充 canonical chip 上下文）
            2. JSON 文件（由 WINGS_HARDWARE_FILE 指定路径，默认 /shared-volume/hardware_info.json）
            3. 环境变量（WINGS_DEVICE / DEVICE, WINGS_DEVICE_COUNT / DEVICE_COUNT, WINGS_DEVICE_NAME）
            4. 默认值（nvidia, 1 卡）

    JSON 文件方式可提供每张卡的完整信息（型号、显存、利用率等），
    激活 VRAM 校验和 CUDA Graph 尺寸自动计算等下游功能。

    Returns:
        Dict[str, Any]: 硬件环境描述字典，包含以下字段：
            - device:  设备类型 ('nvidia' 或 'ascend')
            - count:   设备数量
            - details: 设备详情列表
            - units:   显存单位 (GB)
    """
    explicit_chip = chip or os.getenv("WINGS_CHIP", "")

    # ── 策略 1: 从 JSON 文件加载 ──────────────────────────────────────────
    hw_file = os.getenv(
        "WINGS_HARDWARE_FILE",
        "/shared-volume/hardware_info.json",
    )
    if os.path.isfile(hw_file):
        try:
            result = _attach_chip_context(
                _load_hardware_from_file(hw_file),
                explicit_chip=explicit_chip,
                source="hardware_file",
            )
            logger.info(
                "Loaded hardware info from file %s: device=%s, count=%d, chip=%s, cards=%d",
                hw_file, result["device"], result["count"], result.get("chip_id") or "unknown", len(result["details"]),
            )
            for i, detail in enumerate(result["details"]):
                logger.info(
                    "  Card %d: name=%s, total=%.2f GB, free=%.2f GB, used=%.2f GB%s",
                    detail.get("device_id", i),
                    detail.get("name", "unknown"),
                    detail.get("total_memory", 0),
                    detail.get("free_memory", 0),
                    detail.get("used_memory", 0),
                    ", util=%d%%" % detail["util"] if "util" in detail else "",
                )
            return result
        except Exception as e:
            logger.warning(
                "Failed to load hardware file %s: %s — falling back to env vars",
                hw_file, e,
            )

    # ── 策略 2: 回退到环境变量（原有逻辑） ────────────────────────────────
    # 设备类型优先级: WINGS_DEVICE → DEVICE → HARDWARE_TYPE → 默认 nvidia
    device_raw = os.getenv("WINGS_DEVICE") or os.getenv("DEVICE") or os.getenv("HARDWARE_TYPE", "nvidia")
    device = _normalize_device(device_raw)
    count = _parse_count(os.getenv("WINGS_DEVICE_COUNT", os.getenv("DEVICE_COUNT", "1")))
    device_name = os.getenv("WINGS_DEVICE_NAME", "").strip()

    details = []
    if device_name:
        details.append({"name": device_name})

    result = {
        "device": device,
        "count": count,
        "details": details,
        "units": "GB",
    }
    result = _attach_chip_context(result, explicit_chip=explicit_chip, source="inferred")
    logger.info("Using static hardware context (env vars): %s", result)
    return result

