# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""
vLLM 引擎适配器。

在 sidecar launcher 模式下，本模块仅负责命令拼装，不启动任何子进程。
生成的 shell 脚本将由 engine 容器读取并执行。

支持的引擎类型:
    - vllm:        NVIDIA GPU 版本，使用 NCCL 通信
    - vllm_ascend: 华为昇腾 NPU 版本，使用 HCCL 通信

分布式后端:
    - ray:           Ray 集群模式，支持多节点 TP
    - dp_deployment: 数据并行模式，支持多节点 DP
"""

import ast
import json
import logging
import os
import re
import shlex
import stat
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import yaml

from utils.model_utils import ModelIdentifier, ModelIdentifierDraft, is_deepseek_series_fp8, INDEXCACHE_ARCHS

from utils.env_utils import get_local_ip, get_lmcache_env, \
    get_pd_role_env, get_qat_env, get_cold_start_env
from utils.file_utils import safe_write_file, WriteOptions

try:
    from wings_control.core.version_util import parse_engine_version_tuple
except ImportError:
    from core.version_util import parse_engine_version_tuple  # noqa: F811


def _sanitize_shell_path(path: str) -> str:
    """对路径进行 shell 安全转义，防止命令注入攻击。

    使用 shlex.quote() 进行标准 POSIX shell 转义，
    相比简单的正则过滤更安全且不会破坏包含空格的合法路径。

    Args:
        path: 原始文件路径字符串

    Returns:
        str: 经过 shell 安全转义的路径
    """
    return shlex.quote(path)

logger = logging.getLogger(__name__)

# ── 引擎版本解析 ──────────────────────────────────────────────────────
# vllm-ascend 从 v0.14 起，Ray 集群使用自定义资源 --resources='{"NPU": 1}'
# 代替 --num-gpus，以正确声明 Ascend NPU 设备。
# 低版本 (< 0.14) 沿用 --num-gpus（兼容 V1 行为）。
# 同时，v0.14 需要 Triton NPU 补丁和 --enforce-eager 标志。
_ASCEND_NPU_RESOURCE_MIN_VERSION = (0, 14)


def _parse_engine_version() -> tuple:
    """解析 ENGINE_VERSION 环境变量为 (major, minor) 元组。

    委托给 version_util.parse_engine_version_tuple()，
    支持 v0.17.0-20260325 等非标准格式。

    Returns:
        (major, minor) 整数元组；若未设置或格式异常，返回 (0, 17)（与 supported_features.json 对齐）。
    """
    return parse_engine_version_tuple()


def _get_ray_resource_flag(engine: str, params: dict) -> str:
    """根据引擎类型和版本返回 Ray 节点资源声明标志。

    版本策略：
      - vllm (NVIDIA):         始终使用 --num-gpus=1
      - vllm_ascend >= 0.14:   使用 --resources='{"NPU": 1}'（NPU 自定义资源）
      - vllm_ascend < 0.14:    使用 --num-gpus {tp_size}（兼容 V1 行为）

    可通过 RAY_RESOURCE_FLAG 环境变量完全覆盖自动检测结果。
    """
    # 环境变量覆盖 — 允许用户完全自定义
    override = os.getenv("RAY_RESOURCE_FLAG", "").strip()
    if override:
        logger.info("[ray] Using RAY_RESOURCE_FLAG override: %s", override)
        return override

    if engine != "vllm_ascend":
        return "--num-gpus=1"

    ver = _parse_engine_version()
    if ver >= _ASCEND_NPU_RESOURCE_MIN_VERSION:
        # device_count 已经是每节点的设备数（DEVICE_COUNT 环境变量），
        # 全局 TP = device_count * nnodes 在 _adjust_tensor_parallelism 中计算。
        # 此处直接用 device_count 作为每节点 NPU 资源数。
        npu_per_node = max(1, params.get("device_count", 1))
        logger.info("[ray] Ascend engine version %s >= 0.14, using --resources NPU=%d", ver, npu_per_node)
        return f"--resources='{{\"NPU\": {npu_per_node}}}'"
    else:
        tp_size = params.get("device_count", 1)
        logger.info("[ray] Ascend engine version %s < 0.14, using --num-gpus=%d (V1 compat)", ver, tp_size)
        return f"--num-gpus={tp_size}"


def _need_triton_patch(engine: str) -> bool:
    """判断是否需要 Triton NPU 驱动补丁。

    仅 vllm_ascend >= 0.14 需要 Triton NPU 驱动补丁（解决 "0 active drivers" 崩溃）。
    此补丁是安全的一次性文件修改，不影响性能。
    """
    if engine != "vllm_ascend":
        return False
    ver = _parse_engine_version()
    return ver >= _ASCEND_NPU_RESOURCE_MIN_VERSION


def _need_enforce_eager(engine: str) -> bool:
    """判断是否需要 --enforce-eager 标志（跳过图编译）。

    A+X 环境（Ascend + NVIDIA GPU 混合部署）中，triton 和 triton-ascend
    版本冲突会导致 qkv_rmsnorm_rope 等算子无法正确注册
    (参见 vllm-ascend issue #6737, #6578)，需要通过 --enforce-eager 绕过。

    通过环境变量 ASCEND_ENFORCE_EAGER 控制：
      - true:  强制添加 --enforce-eager（用于 A+X 环境或遇到 triton 冲突时）
      - false: 不添加 --enforce-eager（默认，用于纯 Ascend 环境，可享受图编译性能优化）
    """
    if engine != "vllm_ascend":
        return False
    return os.getenv("ASCEND_ENFORCE_EAGER", "").lower() in ("true", "1", "yes")


def _need_deepseek_ascend_mla_eager_fallback() -> bool:
    """是否对 DeepSeek V3-family Ascend DP 启用 MLA 图编译兜底。

    vLLM-Ascend MLA 在 profile_run 阶段进入 torch.compile/Dynamo 图后，
    某些环境会在 mla_forward/output.fill_ 报 PTA acl api failed。该错误与
    Soft FP8 误判无关，是图编译路径的运行时兼容性问题；默认兜底为 eager，
    如确认本机 vLLM-Ascend 版本图编译稳定，可设置为 0/false 关闭。
    """
    return os.getenv("WINGS_DEEPSEEK_ASCEND_MLA_EAGER_FALLBACK", "0").lower() not in (
        "0",
        "false",
        "no",
    )


def _need_triton_patch_and_eager(engine: str) -> bool:
    """兼容旧接口：判断是否需要 Triton 补丁或 enforce-eager。

    此函数已拆分为 _need_triton_patch() 和 _need_enforce_eager()，
    保留此函数仅用于向后兼容。
    """
    return _need_triton_patch(engine) or _need_enforce_eager(engine)

# 模块根目录：用于定位配置文件和环境脚本
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _inline_ascend_env_script(config_dir: str, engine: str) -> List[str]:
    """读取 Ascend 引擎环境脚本并内联为 shell 命令列表。

    脚本映射:
      - vllm_ascend → config/set_vllm_ascend_env.sh
      - mindie      → config/set_mindie_env.sh
    如脚本不存在则返回 fallback 命令；非 Ascend 引擎返回空列表。

    Args:
        config_dir: 配置目录路径
        engine:     引擎类型

    Returns:
        List[str]: 内联后的 shell 命令列表
    """
    script_map = {
        "vllm_ascend": "set_vllm_ascend_env.sh",
        "mindie": "set_mindie_env.sh",
    }
    script_name = script_map.get(engine)
    if not script_name:
        return []

    script_path = os.path.join(config_dir, script_name)
    if os.path.exists(script_path):
        return _read_and_inline_script(script_path, engine)

    logger.warning("Env script %s not found, using fallback for %s", script_path, engine)
    return _build_ascend_fallback_env(engine)


def _read_and_inline_script(script_path: str, engine: str) -> List[str]:
    """读取脚本文件内容并转为内联 shell 命令，附加驱动预检查。

    Args:
        script_path: 脚本文件完整路径
        engine:      引擎类型

    Returns:
        List[str]: 内联后的 shell 命令列表
    """
    commands = []
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n\r")
            if stripped.startswith("#!"):
                continue
            commands.append(stripped)
    logger.info("Inlined env script %s for engine %s (%d lines)",
                script_path, engine, len(commands))

    if engine == "vllm_ascend":
        commands.extend(_build_ascend_driver_check())
    return commands


def _build_ascend_driver_check() -> List[str]:
    """生成 Ascend 驱动预检查 shell 命令，驱动缺失时 exit 1。

    Returns:
        List[str]: 预检查 shell 命令列表
    """
    return [
        "# Pre-flight: verify Ascend driver is accessible",
        "if [ ! -f /usr/local/Ascend/driver/lib64/driver/libascend_hal.so ]; then",
        "    echo 'FATAL: libascend_hal.so not found at "
        "/usr/local/Ascend/driver/lib64/driver/'",
        "    echo 'HINT: Ensure the host Ascend driver is mounted "
        "into the container (hostPath: /usr/local/Ascend/driver)'",
        "    exit 1",
        "fi",
    ]


def _build_ascend_fallback_env(engine: str) -> List[str]:
    """在环境脚本缺失时生成 Ascend fallback 环境命令。

    Args:
        engine: 引擎类型

    Returns:
        List[str]: fallback 环境命令列表
    """
    if engine not in ("vllm_ascend", "mindie"):
        return []
    return [
        # "set +u",
        # (
        #     "[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] "
        #     "&& source /usr/local/Ascend/ascend-toolkit/set_env.sh "
        #     "|| echo 'WARN: ascend-toolkit/set_env.sh not found'"
        # ),
        # (
        #     "[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] "
        #     "&& source /usr/local/Ascend/nnal/atb/set_env.sh "
        #     "|| echo 'WARN: nnal/atb/set_env.sh not found'"
        # ),
        # "set -u",
        "export LD_LIBRARY_PATH=\"/usr/local/Ascend/driver/lib64/driver"
        ":/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}\"",
    ]


def _build_vllm_ascend_extensions(params) -> List[str]:
    """生成 vllm_ascend 扩展环境命令（昆仑 ATB、Qwen3Next 支持）。

    Args:
        params: 参数字典

    Returns:
        List[str]: 扩展环境命令列表
    """
    commands = []
    if params.get("engine_config", {}).get("use_kunlun_atb"):
        commands.append("export USE_KUNLUN_ATB=1")
        logger.info("kunlun atb is used")
    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type")
    )
    if model_info.model_architecture == "Qwen3NextForCausalLM":
        commands.extend([
            "set +u",
            "source /usr/local/Ascend/ascend-toolkit/8.3.RC2/bisheng_toolkit/set_env.sh",
            "set -u",
        ])
        logger.info("Qwen3NextForCausalLM will source bisheng_toolkit")
    return commands


def _build_base_env_commands(params, engine: str, root: str) -> List[str]:
    """构建基础环境变量设置命令列表。

    仅 Ascend 引擎（vllm_ascend / mindie）需要环境初始化脚本，
    NVIDIA 引擎（vllm / sglang）无需额外设置。

    Args:
        params: 参数字典
        engine: 引擎类型 ('vllm', 'vllm_ascend', 'sglang', 'mindie')
        root:   项目根目录路径

    Returns:
        List[str]: shell 命令列表
    """
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
    )
    env_commands = _inline_ascend_env_script(config_dir, engine)
    if engine == "vllm_ascend":
        env_commands.extend(_build_vllm_ascend_extensions(params))
    return env_commands


# ── LMCache YAML 配置文件 ─────────────────────────────────────────────
# 当 cold_start 或 QAT 特性启用时，需要生成 YAML 配置文件供 LMCache 读取。
# 纯内存卸载场景不需要 YAML 文件（环境变量即可控制）。
_LMCACHE_CONFIG_FILENAME = "lmcache_config.yaml"
_LMCACHE_SHARED_VOLUME = os.getenv("SHARED_VOLUME_PATH", "/shared-volume")


def _build_lmcache_yaml_dict(engine: str) -> dict:
    """根据环境变量构建 LMCache 的 YAML 配置字典。

    配置结构参考 LMCache 官方 YAML schema，包含以下可选段：
    - chunk_size: KV 缓存分块大小（默认 256）
    - local_cpu:  CPU 内存缓存配置
    - local_disk: 本地磁盘缓存配置
    - pre_caching: 冷启动预热配置（仅 cold_start 启用）
    - qat:         QAT 硬件压缩配置（仅 QAT 启用）

    Args:
        engine: 引擎类型（vllm / vllm_ascend）

    Returns:
        dict: 可被 yaml.dump() 序列化的配置字典
    """
    config: dict = {}

    # ── chunk_size ──
    chunk_size_str = os.getenv("LMCACHE_CHUNK_SIZE", "256")
    try:
        config["chunk_size"] = int(chunk_size_str)
    except (ValueError, TypeError):
        config["chunk_size"] = 256

    # ── local_cpu ──
    local_cpu_enabled = os.getenv("LMCACHE_LOCAL_CPU", "").strip().lower() == "true"
    max_cpu_size = os.getenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "").strip()
    if local_cpu_enabled or max_cpu_size:
        config["local_cpu"] = True
        if max_cpu_size:
            config["max_local_cpu_size"] = float(max_cpu_size)

    # ── local_disk ──
    local_disk_path = os.getenv("LMCACHE_LOCAL_DISK", "").strip()
    max_disk_size = os.getenv("LMCACHE_MAX_LOCAL_DISK_SIZE", "").strip()
    if local_disk_path:
        config["local_disk"] = local_disk_path
        if max_disk_size:
            config["max_local_disk_size"] = float(max_disk_size)
    elif max_disk_size:
        config["max_local_disk_size"] = float(max_disk_size)

    # ── pre_caching（冷启动预热）──
    if get_cold_start_env():
        pre_caching: dict = {
            "hash_algorithm": os.getenv("LMCACHE_PRE_CACHING_HASH", "sha256_cbor"),
            "manifest_write_interval": int(os.getenv("LMCACHE_MANIFEST_WRITE_INTERVAL", "1")),
            "maintenance": {"enabled": False},
            "full_sync": {"enabled": False},
        }
        config["pre_caching"] = pre_caching
        logger.info("[LMCache YAML] Cold-start pre_caching section enabled")

    # ── qat（QAT 硬件压缩）──
    if get_qat_env():
        qat_module = "kv_agent" if engine == "vllm" else os.getenv("LMCACHE_QAT_MODULE", "kv_agent")
        qat_section: dict = {
            "module_name": qat_module,
            "instance_num": int(os.getenv("LMCACHE_QAT_INSTANCE_NUM", "2")),
            "loss_level": int(os.getenv("LMCACHE_QAT_LOSS_LEVEL", "0")),
            "log_enabled": int(os.getenv("LMCACHE_QAT_LOG_ENABLED", "0")),
        }
        config["qat"] = qat_section
        logger.info("[LMCache YAML] QAT section enabled (module=%s)", qat_module)

    return config


def _need_lmcache_config_yaml() -> bool:
    """判断是否需要生成 LMCache YAML 配置文件。

    触发条件（任一满足即生成）：
      1. cold_start 或 QAT 特性启用（功能性段落）
      2. 配置了 CPU 内存卸载相关 env：
         - LMCACHE_LOCAL_CPU=true
         - LMCACHE_MAX_LOCAL_CPU_SIZE 非空
      3. 配置了本地磁盘卸载相关 env：
         - LMCACHE_LOCAL_DISK 非空
         - LMCACHE_MAX_LOCAL_DISK_SIZE 非空

    说明：LMCache 的容量类字段（max_size 等）在多数版本下仅识别
    YAML 文件，不保证从同名 env 自动注入；因此只要用户传了容量
    配置，就必须落盘 YAML，否则会沉默失效（参数丢失 bug）。

    Returns:
        bool: 需要生成返回 True
    """
    if get_cold_start_env() or get_qat_env():
        return True
    if os.getenv("LMCACHE_LOCAL_CPU", "").strip().lower() == "true":
        return True
    if os.getenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "").strip():
        return True
    if os.getenv("LMCACHE_LOCAL_DISK", "").strip():
        return True
    if os.getenv("LMCACHE_MAX_LOCAL_DISK_SIZE", "").strip():
        return True
    return False


def _write_lmcache_config_yaml(engine: str) -> Optional[str]:
    """生成并写入 LMCache YAML 配置文件到共享卷。

    条件：见 _need_lmcache_config_yaml()，覆盖 cold_start / QAT /
    CPU 卸载 / 磁盘卸载 等所有需要落盘的场景。
    写入路径：/shared-volume/lmcache_config.yaml

    Args:
        engine: 引擎类型

    Returns:
        str | None: 写入成功返回文件路径，无需写入或失败返回 None
    """
    if not _need_lmcache_config_yaml():
        return None

    config = _build_lmcache_yaml_dict(engine)
    yaml_content = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    file_path = os.path.join(_LMCACHE_SHARED_VOLUME, _LMCACHE_CONFIG_FILENAME)
    os.makedirs(_LMCACHE_SHARED_VOLUME, exist_ok=True)

    ok = safe_write_file(
        file_path, yaml_content, is_json=False,
        options=WriteOptions(
            modes=stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
            atomic=True,
        ),
    )
    if ok:
        logger.info("[LMCache YAML] Config written to %s", file_path)
        return file_path
    else:
        logger.error("[LMCache YAML] Failed to write config to %s", file_path)
        return None


def _append_lmcache_env_export(env_commands: List[str], name: str, value: Optional[str] = None) -> None:
    """Append an explicit engine-side export for an LMCache environment variable."""
    if value is None:
        value = os.getenv(name, "").strip()
    if value:
        env_commands.append(f"export {name}={shlex.quote(value)}")


def _build_cache_env_commands(engine: str) -> List[str]:
    """构建 KVCache Offload 特性的环境变量设置命令。

    KVCache Offload 允许将 KV 缓存卸载到主机内存或远端存储。
    LMCache 所需的共享库已在 accel-volume 安装阶段通过
    ``install.py --lmcache-target`` 注入，无需再手动设置 LD_LIBRARY_PATH。

    只要传入了任何容量/路径类配置（CPU/Disk/cold_start/QAT），就会
    生成 LMCache YAML 配置文件并通过 ``LMCACHE_CONFIG_FILE`` 环境变量
    告知 LMCache，避免容量参数被沉默丢弃。

    Args:
        engine: 引擎类型

    Returns:
        List[str]: 环境变量设置命令列表，未启用时返回空列表

    环境变量:
        - LMCACHE_OFFLOAD: 是否启用 KVCache Offload (true/false)
        - LMCACHE_CONFIG_FILE: LMCache YAML 配置文件路径（自动生成）
    """
    env_commands = []
    if not get_lmcache_env():
        return env_commands

    # 跨实例Hash一致
    env_commands.append('export PYTHONHASHSEED=0')
    _append_lmcache_env_export(env_commands, "LMCACHE_OFFLOAD", "true")
    _append_lmcache_env_export(env_commands, "LMCACHE_CHUNK_SIZE")

    local_cpu_value = os.getenv("LMCACHE_LOCAL_CPU", "").strip()
    max_cpu_size = os.getenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "").strip()
    if local_cpu_value or max_cpu_size:
        _append_lmcache_env_export(env_commands, "LMCACHE_LOCAL_CPU", local_cpu_value or "true")
        _append_lmcache_env_export(env_commands, "LMCACHE_MAX_LOCAL_CPU_SIZE", max_cpu_size)

    _append_lmcache_env_export(env_commands, "LMCACHE_LOCAL_DISK")
    _append_lmcache_env_export(env_commands, "LMCACHE_MAX_LOCAL_DISK_SIZE")

    # 任何 LMCache 容量/功能段配置都会触发 YAML 生成并导出路径
    yaml_path = _write_lmcache_config_yaml(engine)
    if yaml_path:
        env_commands.append(f'export LMCACHE_CONFIG_FILE={shlex.quote(yaml_path)}')
        logger.info("[KVCache Offload] LMCACHE_CONFIG_FILE exported -> %s", yaml_path)
    else:
        logger.warning(
            "[KVCache Offload] LMCACHE_OFFLOAD enabled but no LMCache config "
            "yaml generated. Capacity envs (LMCACHE_MAX_LOCAL_CPU_SIZE / "
            "LMCACHE_MAX_LOCAL_DISK_SIZE) may not take effect. "
            "Set LMCACHE_LOCAL_CPU=true (or any capacity env) to enable."
        )

    return env_commands


def _build_qat_env_commands(engine) -> List[str]:
    """构建 KVCache QAT 压缩特性的环境变量设置命令。

    QAT (QuickAssist Technology) 是 Intel 的硬件压缩加速技术，
    可用于压缩 KV 缓存以减少内存占用和传输开销。

    注意:
        - 当前仅 vllm (NVIDIA) 支持 QAT 压缩
        - vllm_ascend 不支持，会自动禁用并打印警告

    Args:
        engine: 引擎类型

    Returns:
        List[str]: LMCACHE_QAT_ENABLED 设置命令列表

    环境变量:
        - LMCACHE_QAT: 是否启用 QAT 压缩 (true/false)
    """
    env_commands = []
    if not get_qat_env():
        return env_commands

    if engine == "vllm":
        env_commands.append('export LMCACHE_QAT_ENABLED=True')
    else:
        env_commands.append('export LMCACHE_QAT_ENABLED=False')
        logger.warning("[KVCache Offload] QAT compression feature is not supported by the current engine %s, "
                       "it has been automatically disabled", engine)
    return env_commands


def _build_pd_role_env_commands(engine: str, current_ip: str, network_interface: str) -> List[str]:
    """构建 PD 分离部署的环境变量设置命令。

    PD 分离 (Prefill-Decode Disaggregation) 是一种高级部署架构，
    将 Prefill 和 Decode 阶段分离到不同节点，以优化资源利用率。

    vllm (NVIDIA) 场景:
        - 使用 NIXL 协议进行 KV 传输
        - 设置 VLLM_NIXL_SIDE_CHANNEL_HOST

    vllm_ascend 场景:
        - 使用 HCCL 进行跨节点通信
        - 需要设置多个网络接口环境变量
        - 依赖 CANN 和 ATB 工具包

    Args:
        engine:           引擎类型 ('vllm' 或 'vllm_ascend')
        current_ip:       当前节点 IP 地址
        network_interface: 网络接口名称 (如 'eth0')

    Returns:
        List[str]: PD 分离所需的环境变量设置命令

    环境变量:
        - PD_ROLE: PD 角色 ('P' 或 'D')
        - VLLM_LLMDD_RPC_PORT: LLMDataDist RPC 端口号
    """
    env_commands = []
    if get_pd_role_env():
        if engine == "vllm":
            env_commands.append(f'export VLLM_NIXL_SIDE_CHANNEL_HOST={shlex.quote(current_ip)}')
        elif engine == "vllm_ascend":
            rpc_port = os.getenv('VLLM_LLMDD_RPC_PORT', "5569")
            mooncake_bootstrap_port = os.getenv('VLLM_MOONCAKE_BOOTSTRAP_PORT', "23000")
            # CANN 环境初始化已由 _build_base_env_commands() 完成，此处不再重复
            env_commands.extend([
                f"export HCCL_IF_IP={shlex.quote(current_ip)}",
                f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
                f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
                f"export HCCL_SOCKET_IFNAME={shlex.quote(network_interface)}",
                "export OMP_PROC_BIND=false",
                f"export OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', '100')}",
                "export VLLM_USE_V1=1",
                "export LCCL_DETERMINISTIC=1",
                "export HCCL_DETERMINISTIC=true",
                "export CLOSE_MATMUL_K_SHIFT=1",
                f"export VLLM_LLMDD_RPC_PORT={rpc_port}",
                f"export VLLM_MOONCAKE_BOOTSTRAP_PORT={mooncake_bootstrap_port}",
                f"export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:{os.getenv('NPU_MAX_SPLIT_SIZE_MB', '256')}",
                # mooncake-transfer-engine 的 Ascend 传输后端 (ascend_transport.so)
                # 安装在 /usr/local/lib，需追加到 LD_LIBRARY_PATH 以便运行时加载
                'export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"',
            ])
    return env_commands


def _build_distributed_env_commands(params: Dict[str, Any], current_ip: str,
                                    network_interface: str, engine: str) -> List[str]:
    """构建分布式环境变量设置命令（扩展点）。

    当前返回空列表，分布式 NCCL/HCCL 环境设置已在
    _build_pd_role_env_commands 和 build_start_script 内部的
    Ray 初始化块中处理。

    根据 distributed_executor_backend（ray / dp_deployment）和引擎类型
    （vllm / vllm_ascend）设置对应的网络通信环境变量。

    Args:
        params:            参数字典
        current_ip:        当前节点 IP
        network_interface: 网络接口名称
        engine:            引擎类型

    Returns:
        List[str]: 环境变量设置命令列表
    """
    env_commands = []
    if params.get("distributed", False):
        backend = params.get("distributed_executor_backend")
        if backend == "ray":
            if engine == "vllm":
                env_commands.extend([
                    f"export VLLM_HOST_IP={shlex.quote(current_ip)}",
                    f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    f"export NCCL_SOCKET_IFNAME={shlex.quote(network_interface)}",
                ])
            elif engine == "vllm_ascend":
                env_commands.extend([
                    f"export HCCL_IF_IP={shlex.quote(current_ip)}",
                    f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
                    "export ASCEND_PROCESS_LOG_PATH=/tmp/ray_vllm010",
                    # Ascend NPU 首次推理 JIT 编译算子耗时可能远超 Ray 编译DAG默认 300s 超时
                    "export RAY_CGRAPH_get_timeout=" + os.getenv('RAY_CGRAPH_get_timeout', '3600'),
                ])
        elif backend == "dp_deployment":
            if engine == "vllm":
                env_commands.extend([
                    f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    f"export NCCL_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    f"export VLLM_NIXL_SIDE_CHANNEL_PORT={params.get('nixl_port', '')}",
                    "export NCCL_IB_DISABLE=0",
                    "export NCCL_CUMEM_ENABLE=0",
                    "export NCCL_NET_GDR_LEVEL=SYS",
                ])
            elif engine == "vllm_ascend":
                is_deepseek_v3_family = _is_deepseek_v3_family_ascend_dp_deployment(params)
                omp_threads = os.getenv('OMP_NUM_THREADS', '1' if is_deepseek_v3_family else '10')
                hccl_buffsize = os.getenv('HCCL_BUFFSIZE', '200' if is_deepseek_v3_family else '1024')
                env_commands.extend([
                    f"export HCCL_IF_IP={shlex.quote(current_ip)}",
                    f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    f"export HCCL_SOCKET_IFNAME={shlex.quote(network_interface)}",
                    "export OMP_PROC_BIND=false",
                    f"export OMP_NUM_THREADS={omp_threads}",
                    f"export HCCL_BUFFSIZE={hccl_buffsize}",
                    'echo "[wings-env] final HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"',
                    "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
                ])
                if is_deepseek_v3_family:
                    env_commands.extend([
                        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
                        "export HCCL_INTRA_PCIE_ENABLE=1",
                        "export HCCL_INTRA_ROCE_ENABLE=0",
                    ])
                else:
                    env_commands.extend([
                        "export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/"
                        "latest/opp/deepseek-v32/vendors/customize:${ASCEND_CUSTOM_OPP_PATH:-}",
                        "export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/"
                        "opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH:-}",
                    ])
    return env_commands


def _build_deepseek_fp8_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 DeepSeek FP8 模型所需的环境变量命令。

    仅在满足以下条件时设置 DeepSeek FP8 专属环境变量：
    1. 引擎类型为 vllm_ascend
    2. 模型路径存在
    3. 模型是 DeepSeek 系列 FP8 模型

    Args:
        params: 参数字典，包含 model_path 等信息
        engine: 引擎类型

    Returns:
        List[str]: 环境变量导出命令列表
    """
    env_commands = []
    model_path = params.get("model_path")

    if _is_deepseek_v3_family_ascend_dp_deployment(params):
        logger.info(
            "[DeepSeek V3-family Ascend DP] Skip generic DeepSeek FP8 env vars; "
            "online dp_deployment follows official DeepSeek-V3/3.1 command envs."
        )
        return env_commands

    if engine == "vllm_ascend" and model_path and is_deepseek_series_fp8(model_path):
        env_commands.extend([
            "export VLLM_ASCEND_ENABLE_NZ=0",
            "export HCCL_OP_EXPANSION_MODE=AIV",
            "export VLLM_ASCEND_ENABLE_MLAPO=1",
            "export VLLM_ASCEND_BALANCE_SCHEDULING=1"
        ])
        logger.info("[DeepSeek FP8] Set environment variables for DeepSeek FP8 model")

    return env_commands


def _build_ascend910_9362_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 Ascend910_9362 设备特定环境变量命令。

    当满足以下条件时，添加特定的环境变量：
    1. 通过 torch_npu 检测设备名称为 Ascend910_9362
    2. 模型结构为 DeepseekV32ForCausalLM 或 DeepseekV3ForCausalLM
    3. 引擎为 vllm_ascend
    4. 不是 dp_deployment 分布式模式（避免与 _build_distributed_env_commands 重复）

    Args:
        params: 参数字典
        engine: 引擎类型

    Returns:
        List[str]: 环境变量导出命令列表
    """
    env_commands = []
    distributed_backend = params.get("distributed_executor_backend")

    # 从硬件信息 JSON 或环境变量中获取设备名称（不依赖 torch_npu SDK）
    device_name = None
    try:
        from core.hardware_detect import detect_hardware
        hw = detect_hardware()
        if hw.get("details"):
            device_name = hw["details"][0].get("name")
        if not device_name:
            device_name = os.getenv("WINGS_DEVICE_NAME", "").strip() or None
        if device_name:
            logger.info("[Ascend910_9362] Detected device from hardware info: %s", device_name)
    except Exception as e:
        logger.warning("[Ascend910_9362] Failed to get device name: %s", e)

    if device_name != "Ascend910_9362":
        return env_commands

    if engine != "vllm_ascend":
        return env_commands

    if distributed_backend == "dp_deployment":
        return env_commands

    if not params.get("model_path"):
        return env_commands

    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type")
    )

    if model_info.model_architecture in ["DeepseekV32ForCausalLM", "DeepseekV3ForCausalLM"]:
        env_commands.extend([
            "export OMP_PROC_BIND=false",
            "export OMP_NUM_THREADS=10",
            "export HCCL_BUFFSIZE=1024"
        ])
        logger.info("[Ascend910_9362] Set environment variables for %s", model_info.model_architecture)

    return env_commands


def _build_glm4moe_ascend_env(arch: str) -> List[str]:
    """构建 GLM-4.7 (Glm4MoeForCausalLM) Ascend 环境变量命令。"""
    logger.info("[GLM-4.7] Set Ascend environment variables for %s", arch)
    return [
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
        "export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
        "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
    ]


def _build_glm_moe_dsa_ascend_env(arch: str) -> List[str]:
    """构建 GLM-5/5.1 (GlmMoeDsaForCausalLM) Ascend 环境变量命令。"""
    logger.info("[GLM-5] Set Ascend environment variables for %s", arch)
    return [
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export HCCL_BUFFSIZE=200",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
    ]


def _build_qwen3_ascend_env(arch: str) -> List[str]:
    """构建 Qwen3 密集模型 (Qwen3ForCausalLM) Ascend 环境变量命令。

    适用于 Qwen3-32B 等密集架构。
    """
    logger.info("[Qwen3] Set Ascend environment variables for %s", arch)
    return [
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
    ]


def _build_qwen35_ascend_env(arch: str) -> List[str]:
    """构建 Qwen3.5 (Qwen3_5ForConditionalGeneration) Ascend 环境变量命令。"""
    logger.info("[Qwen3.5] Set Ascend environment variables for %s", arch)
    return [
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
    ]


def _build_qwen35moe_ascend_env(arch: str) -> List[str]:
    """构建 Qwen3.5-MoE (Qwen3_5MoeForConditionalGeneration) Ascend 环境变量命令。"""
    logger.info("[Qwen3.5-MoE] Set Ascend environment variables for %s", arch)
    return [
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
    ]


def _build_minimaxm2_ascend_env(arch: str) -> List[str]:
    """构建 MiniMax-M2.5 (MiniMaxM2ForCausalLM) Ascend 环境变量命令。

    注入 MiniMax-M2.5 在 Ascend 910B 上所需的环境变量：
    - VLLM_USE_GRAPH:                  启用 NPU Graph 加速
    - VLLM_USE_V1:                     启用 vLLM V1 多进程架构
    - VLLM_ASCEND_ENABLE_FLASHCOMM1:   启用 FlashComm 通信优化（EP 密集通信场景）
    - VLLM_TORCH_COMPILE:              关闭 torch.compile（Ascend 910B 兼容性）

    注意: HCCL_OP_EXPANSION_MODE=AIV 已由 set_vllm_ascend_env.sh 全局设置，
    此处不重复注入。
    """
    logger.info("[MiniMax-M2.5] Set Ascend environment variables for %s", arch)
    return [
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export VLLM_USE_GRAPH=1",
        "export VLLM_USE_V1=1",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
        "export VLLM_TORCH_COMPILE=0",
    ]


def _build_deepseekv32_ascend_env(arch: str) -> List[str]:
    """构建 DeepSeek V3.2 (DeepseekV32ForCausalLM) Ascend 环境变量命令。"""
    logger.info("[DeepSeek V3.2] Set Ascend environment variables for %s", arch)
    # DeepSeek V3.2 独有变量（不与 _build_deepseek_fp8_env_commands 重叠）
    return [
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=10",
        "export VLLM_USE_V1=1",
        "export HCCL_BUFFSIZE=512",
        "export VLLM_ASCEND_ENABLE_MLAPO=1",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
    ]


def _build_llama_ascend_env(arch: str) -> List[str]:
    """构建 LLaMA3.1 (LlamaForCausalLM) Ascend 环境变量命令。"""
    logger.info("[LLaMA3.1] Set Ascend environment variables for %s", arch)
    return [
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
    ]


def _build_kimik25_ascend_env(arch: str) -> List[str]:
    """构建 Kimi-K2.5 (KimiK25ForConditionalGeneration) Ascend 环境变量命令。"""
    logger.info("[Kimi-K2.5] Set Ascend environment variables for %s", arch)
    return [
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
        "export HCCL_BUFFSIZE=1024",
        "export VLLM_ASCEND_ENABLE_MLAPO=1",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
        "export VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ]


def _build_model_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建模型架构特定的环境变量命令（支持 NVIDIA 和 Ascend）。

    根据模型架构注入引擎官方文档推荐的环境变量。

    已覆盖的 Ascend 架构:
    - Glm4MoeForCausalLM (GLM-4.7): TOPK 优化, FlashComm, Fused MC2
    - GlmMoeDsaForCausalLM (GLM-5/5.1): DSA MTP 基础运行时变量
    - Qwen3_5ForConditionalGeneration (Qwen3.5-27B): TASK_QUEUE_ENABLE
    - Qwen3_5MoeForConditionalGeneration (Qwen3.5-397B): TASK_QUEUE_ENABLE
    - MiniMaxM2ForCausalLM (MiniMax-M2.5): FlashComm
    - DeepseekV32ForCausalLM (DeepSeek V3.2): MLAPO, FlashComm, VLLM_USE_V1
    - LlamaForCausalLM (LLaMA3.1-70B): 基础 NPU 内存/线程优化
    - KimiK25ForConditionalGeneration (Kimi-K2.5): MLAPO, FlashComm, Eagle3 超时加固

    Args:
        params: 参数字典
        engine: 引擎类型

    Returns:
        List[str]: 环境变量导出命令列表
    """
    if engine not in ("vllm", "vllm_ascend"):
        return []

    model_path = params.get("model_path")
    if not model_path:
        return []

    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type")
    )
    arch = model_info.model_architecture

    if _is_deepseek_v3_family_ascend_dp_deployment(params):
        logger.info(
            "[DeepSeek V3-family Ascend DP] Skip architecture-specific env vars; "
            "online dp_deployment reuses official DeepSeek-V3/3.1 envs."
        )
        return []

    if engine == "vllm_ascend":
        _arch_env_builders = {
            "Glm4MoeForCausalLM": _build_glm4moe_ascend_env,
            "GlmMoeDsaForCausalLM": _build_glm_moe_dsa_ascend_env,
            "Qwen3ForCausalLM": _build_qwen3_ascend_env,
            "Qwen3_5ForConditionalGeneration": _build_qwen35_ascend_env,
            "Qwen3_5MoeForConditionalGeneration": _build_qwen35moe_ascend_env,
            "MiniMaxM2ForCausalLM": _build_minimaxm2_ascend_env,
            "DeepseekV32ForCausalLM": _build_deepseekv32_ascend_env,
            "LlamaForCausalLM": _build_llama_ascend_env,
            "KimiK25ForConditionalGeneration": _build_kimik25_ascend_env,
        }
    else:
        _arch_env_builders = {}

    builder = _arch_env_builders.get(arch)
    return builder(arch) if builder else []


def _build_env_commands(params: Dict[str, Any], current_ip: str, network_interface: str, root: str) -> List[str]:
    """组装完整的环境变量设置命令列表。

    按顺序调用各子模块构建环境设置，创建完整的环境初始化流程：
    1. 基础环境（CANN/ATB 工具包）
    2. KVCache Offload 环境
    3. QAT 压缩环境
    4. PD 分离环境
    5. 分布式环境（扩展点）

    Args:
        params:            参数字典，包含 engine 等配置
        current_ip:        当前节点 IP 地址
        network_interface: 网络接口名称
        root:              项目根目录

    Returns:
        List[str]: 所有环境变量设置命令的有序列表
    """
    engine = params.get("engine")
    env_commands = []

    env_commands.extend(_build_base_env_commands(params, engine, root))
    env_commands.extend(_build_cache_env_commands(engine))
    env_commands.extend(_build_qat_env_commands(engine))
    env_commands.extend(_build_pd_role_env_commands(engine, current_ip, network_interface))
    env_commands.extend(_build_distributed_env_commands(params, current_ip, network_interface, engine))
    env_commands.extend(_build_deepseek_fp8_env_commands(params, engine))
    env_commands.extend(_build_ascend910_9362_env_commands(params, engine))
    env_commands.extend(_build_model_env_commands(params, engine))

    return env_commands


def _prepare_engine_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """准备 engine_config：移除内部参数、处理弃用字段、避免参数冲突。

    Args:
        params: 参数字典

    Returns:
        Dict[str, Any]: 清理后的 engine_config 浅拷贝
    """
    engine_config = dict(params.get("engine_config", {}))
    engine_config.pop("use_kunlun_atb", None)
    engine_config.pop("enable_sparse", None)  # consumed by _build_kv_sparse_cmd; not a vllm CLI arg
    explicit_keys = set(params.get("_explicit_cli_keys") or [])

    if _is_deepseek_ascend_dp_deployment(params):
        prefix_cache_explicit = bool(
            explicit_keys.intersection({"enable_prefix_caching", "no_enable_prefix_caching"})
        )
        if (
            not prefix_cache_explicit
            and engine_config.get("enable_prefix_caching") not in (None, False, "False", 0, "0")
        ):
            logger.warning(
                "[DeepSeek Ascend DP] prefix caching is incompatible with the "
                "dp_deployment path; forcing --no-enable-prefix-caching."
            )
        if not prefix_cache_explicit:
            engine_config.pop("enable_prefix_caching", None)
            engine_config["no_enable_prefix_caching"] = True

        if (
            "enable_expert_parallel" not in explicit_keys
            and engine_config.get("enable_expert_parallel") in (None, False, "False", 0, "0")
        ):
            logger.info(
                "[DeepSeek Ascend DP] enabling expert parallel to align with "
                "vLLM-Ascend DeepSeek multi-node launch examples."
            )
        if "enable_expert_parallel" not in explicit_keys:
            engine_config["enable_expert_parallel"] = True
        if "async_scheduling" not in explicit_keys:
            engine_config["async_scheduling"] = True

        if _is_deepseek_v3_family_ascend_dp_deployment(params):
            # DeepSeek V3-family W8A8 dynamic quant on Ascend can fail in
            # npu_quant_matmul when dtype=auto resolves to float16:
            # output_dtype=float16 requires per-token scale to be float32, but
            # vllm-ascend may produce float16 fake tensors during Dynamo checks.
            # Default to bfloat16 for this official online DP path while keeping
            # explicit --dtype / DTYPE overrides intact.
            if "dtype" not in explicit_keys or str(engine_config.get("dtype", "")).lower() == "auto":
                engine_config["dtype"] = "bfloat16"
            if "seed" not in explicit_keys:
                engine_config["seed"] = 1024
            if "max_num_seqs" not in explicit_keys:
                engine_config["max_num_seqs"] = 16
            if "gpu_memory_utilization" not in explicit_keys:
                engine_config["gpu_memory_utilization"] = 0.92
            if "max_model_len" not in explicit_keys and engine_config.get("max_model_len") in (None, 4096):
                engine_config["max_model_len"] = 16384
            if "compilation_config" not in explicit_keys and not engine_config.get("compilation_config"):
                engine_config["compilation_config"] = {
                    "cudagraph_capture_sizes": [4, 16, 32, 48, 64],
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                }
            # This is not the old Soft-FP8 misclassification path. If the MLA
            # profile run fails inside torch.compile/Dynamo, fall back to eager
            # only for this DeepSeek V3-family Ascend DP scenario. Explicit
            # --enforce-eager/--no-enforce-eager style overrides still win.
            if "enforce_eager" not in explicit_keys:
                if _need_deepseek_ascend_mla_eager_fallback():
                    engine_config["enforce_eager"] = True
                    if "compilation_config" not in explicit_keys:
                        engine_config.pop("compilation_config", None)
                else:
                    engine_config.pop("enforce_eager", None)

    # "task" 在旧版 vLLM (v0.7) 中为 --task 参数，新版改为 --runner
    removed_task = engine_config.pop("task", None)
    if removed_task and removed_task != "generate":
        logger.info("[vLLM] Mapping deprecated task=%s to --runner pooling", removed_task)
        engine_config.setdefault("runner", "pooling")

    return engine_config


def _is_deepseek_ascend_dp_deployment(params: Dict[str, Any]) -> bool:
    """判断当前启动是否为 DeepSeek Ascend dp_deployment 路径。"""
    if params.get("engine") != "vllm_ascend":
        return False
    if params.get("distributed_executor_backend") != "dp_deployment":
        return False
    model_path = params.get("model_path")
    if not model_path:
        return False
    model_info = ModelIdentifier(
        params.get("model_name"),
        model_path,
        params.get("model_type"),
    )
    return model_info.model_architecture in ["DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"]


def _is_deepseek_v31_ascend_dp_deployment(params: Dict[str, Any]) -> bool:
    """判断当前启动是否为 DeepSeek-V3.1 Ascend dp_deployment 路径。"""
    if not _is_deepseek_ascend_dp_deployment(params):
        return False
    candidates: List[str] = []
    for key in ("model_name", "model_path"):
        value = params.get(key)
        if value:
            candidates.append(str(value))
    served_name = params.get("engine_config", {}).get("served_model_name")
    if isinstance(served_name, list):
        candidates.extend(str(item) for item in served_name)
    elif served_name:
        candidates.append(str(served_name))

    for item in candidates:
        normalized = item.lower().replace("_", "-")
        if "deepseek" in normalized and ("v3.1" in normalized or "v31" in normalized):
            return True
    return False


def _is_deepseek_v3_family_ascend_dp_deployment(params: Dict[str, Any]) -> bool:
    """判断当前启动是否为 DeepSeek V3-family Ascend dp_deployment 路径。"""
    if not _is_deepseek_ascend_dp_deployment(params):
        return False
    candidates: List[str] = []
    for key in ("model_name", "model_path"):
        value = params.get(key)
        if value:
            candidates.append(str(value))
    served_name = params.get("engine_config", {}).get("served_model_name")
    if isinstance(served_name, list):
        candidates.extend(str(item) for item in served_name)
    elif served_name:
        candidates.append(str(served_name))

    for item in candidates:
        normalized = item.lower().replace("_", "-")
        if "deepseek" in normalized and "v3" in normalized:
            return True
    return False


def _strip_cli_flag(cmd: str, flag: str) -> str:
    """从已构建的 vLLM CLI 命令字符串中移除指定的 ``--xxx <value>`` 片段。

    用于 Ray 分布式 head 端在 MoE PP 自动注入路径下，避免与 engine_config
    生成的同名参数（如 ``--tensor-parallel-size``）形成重复传参。

    Args:
        cmd:  ``_build_vllm_cmd_parts`` 生成的命令字符串
        flag: 需要移除的 CLI 参数名（如 ``--tensor-parallel-size``）

    Returns:
        str: 移除该 flag 及其紧随值后的命令字符串（值若不存在则原样返回）
    """
    tokens = cmd.split()
    out: List[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == flag and i + 1 < len(tokens):
            i += 2  # 跳过 flag 与其值
            continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def _format_cli_arg(arg_name: str, value) -> List[str]:
    """将单个引擎参数值格式化为 CLI 参数片段。

    Args:
        arg_name: CLI 参数名（如 --tensor-parallel-size）
        value:    参数值

    Returns:
        List[str]: CLI 参数片段列表
    """
    if isinstance(value, bool):
        return [arg_name] if value else []
    if isinstance(value, list):
        str_items = [shlex.quote(str(item)) for item in value]
        return [arg_name] + str_items
    if isinstance(value, dict):
        # dict 透传：序列化为紧凑 JSON 后用 shlex.quote 做 shell 转义，
        # 避免 str(dict) 输出 Python repr（单引号 key），导致 vLLM JSON 解析失败。
        # shlex.quote 会自动用单引号包裹并转义其中的单引号，保证 shell 安全。
        return [arg_name, shlex.quote(json.dumps(value, ensure_ascii=False, separators=(',', ':')))]
    if isinstance(value, str):
        stripped = value.strip()
        # 防御性处理 dict/list 形态字符串：上游某些路径可能将 dict 误用 str() 转换，
        # 产生 Python repr（单引号 key），这种字符串不是合法 JSON，会导致 vLLM
        # --compilation-config / --speculative-config / --additional-config 等
        # 期望 JSON 的参数解析失败。这里统一兜底：
        #   1. 优先按 JSON 解析（已经合法则直接紧凑序列化）；
        #   2. 失败时按 Python literal 解析（兼容 str(dict) 单引号 key 形态）；
        #   3. 仍失败则原样透传。
        if (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']')):
            normalized: Optional[str] = None
            try:
                parsed = json.loads(stripped)
                normalized = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed = ast.literal_eval(stripped)
                    normalized = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
                    logger.warning(
                        "[vLLM] %s value is Python-repr str (single-quoted keys); "
                        "auto-normalized to JSON. Upstream stringification suspected.",
                        arg_name,
                    )
                except (ValueError, SyntaxError):
                    logger.warning(
                        "[vLLM] %s value looks like dict/list but neither JSON nor "
                        "Python literal; passing through as-is: %r",
                        arg_name, stripped[:120],
                    )
            if normalized is not None:
                return [arg_name, shlex.quote(normalized)]
            return [arg_name, shlex.quote(value)]
    return [arg_name, shlex.quote(str(value))]


# ── GLM-4.7-W8A8 引擎参数注入（仅针对量化变体，避免污染同架构 BF16 模型）──
# 触发条件：架构 == Glm4MoeForCausalLM 且 config.json 量化字段命中 W8A8 别名表
# 合并策略：
#   * 标量字段：用户已显式给出则不覆盖（user > injected）
#   * dict 字段（additional_config / speculative_config / compilation_config）：
#       做 **深合并**，用户给出的 sub-key 优先，未给出的 sub-key 注入
_GLM47_W8A8_ENGINE_DEFAULTS: Dict[str, Any] = {
    "enable_expert_parallel": True,
    "async_scheduling": True,
    "quantization": "ascend",
    "additional_config": {
        # 官方 GLM-4.7-W8A8 强推荐
        "enable_shared_expert_dp": True,
        "ascend_fusion_config": {"fusion_ops_gmmswigluquant": False},
    },
    # 推测解码：使用 vllm-ascend 专用 method 名 glm4_moe_mtp
    "speculative_config": {
        "method": "glm4_moe_mtp",
        "num_speculative_tokens": 3,
    },
    # 编译图：cudagraph 全量解码模式，覆盖常用并发档位
    "compilation_config": {
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
        "cudagraph_mode": "FULL_DECODE_ONLY",
    },
}

# 需要做 dict 深合并的字段（不能整体覆盖）
_GLM47_W8A8_DEEP_MERGE_KEYS = {
    "additional_config",
    "speculative_config",
    "compilation_config",
}

# W8A8 量化别名白名单 + 子串匹配（兼容 quantize / quantization_config.quant_method 字段值多样命名）
_W8A8_QUANT_METHOD_ALIASES = {
    "w8a8", "w8a8_int8", "w8a8int8",
    "smoothquant", "smooth_quant",
    "ascend_w8a8", "ascend-w8a8",
}


def _is_w8a8_quantize(quantize: Optional[str]) -> bool:
    """判定模型是否为 W8A8 量化变体（容忍命名差异）。"""
    if not quantize:
        return False
    q = str(quantize).strip().lower()
    if not q:
        return False
    if q in _W8A8_QUANT_METHOD_ALIASES:
        return True
    # 子串匹配：覆盖未来可能出现的 ascend-w8a8-int8 / xxx_w8a8_yyy 等命名
    return "w8a8" in q


def _deep_merge_user_priority(user: Any, default: Any) -> Any:
    """递归深合并：user 有则保留 user，user 没有的 sub-key 用 default 填充。

    仅对 dict 做递归；其他类型（含 list / 标量）用户优先。
    """
    if not isinstance(user, dict) or not isinstance(default, dict):
        return user if user is not None else default
    merged = dict(user)
    for k, v in default.items():
        if k not in merged or merged[k] is None:
            merged[k] = v
        else:
            merged[k] = _deep_merge_user_priority(merged[k], v)
    return merged


def _inject_glm47_w8a8_engine_config(params: Dict[str, Any]) -> None:
    """检测 GLM-4.7-W8A8 模型，**就地**向 engine_config 追加调优默认字段。

    设计要点：
      * 仅当 (架构 == Glm4MoeForCausalLM) 且 (quantize 命中 W8A8) 时触发
      * 标量字段：用户优先；dict 字段：深合并，用户的 sub-key 优先
      * BF16 / 同架构非量化变体（如 GLM-4.5）不会被影响
      * 仅对 vllm / vllm_ascend 引擎生效
    """
    engine = params.get("engine", "vllm")
    if engine not in ("vllm", "vllm_ascend"):
        return

    model_path = params.get("model_path")
    if not model_path:
        return

    try:
        info = ModelIdentifier(
            params.get("model_name", ""),
            model_path,
            params.get("model_type", "auto"),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[GLM-4.7-W8A8] Skip injection, ModelIdentifier failed: %s", e)
        return

    if info.model_architecture != "Glm4MoeForCausalLM":
        return
    if not _is_w8a8_quantize(info.model_quantize):
        return

    # 若上层已通过 enable_speculative_decode 走 _build_speculative_cmd 路径，
    # 则我们不再向 engine_config 注入 speculative_config，避免命令行出现两份
    # --speculative-config（一份来自 engine_config，一份来自 _build_speculative_cmd）
    suppress_speculative = bool(params.get("enable_speculative_decode"))

    engine_config = params.setdefault("engine_config", {})
    injected: List[str] = []
    deep_merged: List[str] = []
    skipped: List[str] = []
    for key, default_val in _GLM47_W8A8_ENGINE_DEFAULTS.items():
        if key == "speculative_config" and suppress_speculative:
            skipped.append(key + "(handled by _build_speculative_cmd)")
            # 同时，从 engine_config 中移除任何已存在的 speculative_config，
            # 避免 _build_vllm_cmd_parts 也输出一份。MTP 路径在尾部追加。
            engine_config.pop("speculative_config", None)
            continue
        existing = engine_config.get(key)
        # 把"空"等价于"未设置"：None / 空字符串 / 空 dict / 空 list 都视为未提供
        is_empty = (
            existing is None
            or (isinstance(existing, str) and not existing.strip())
            or (isinstance(existing, (dict, list)) and len(existing) == 0)
        )
        if key in _GLM47_W8A8_DEEP_MERGE_KEYS and isinstance(default_val, dict):
            if is_empty:
                engine_config[key] = dict(default_val) if isinstance(default_val, dict) else default_val
                injected.append(key)
            elif isinstance(existing, dict):
                merged = _deep_merge_user_priority(existing, default_val)
                if merged != existing:
                    engine_config[key] = merged
                    deep_merged.append(key)
            else:
                # 用户给了非 dict 非空值（如 JSON 字符串）：先尝试解析后深合并；
                # 若无法解析，保留用户原值，避免模型特化注入覆盖上层显式传参。
                parsed_existing = None
                if isinstance(existing, str):
                    try:
                        parsed_existing = json.loads(existing)
                    except (json.JSONDecodeError, ValueError):
                        try:
                            parsed_existing = ast.literal_eval(existing)
                        except (ValueError, SyntaxError):
                            parsed_existing = None
                if isinstance(parsed_existing, dict):
                    engine_config[key] = _deep_merge_user_priority(parsed_existing, default_val)
                    deep_merged.append(key)
                else:
                    logger.warning(
                        "[GLM-4.7-W8A8] %s already present as non-dict (%s); "
                        "keeping user value and skipping default injection for this key.",
                        key, type(existing).__name__,
                    )
                    skipped.append(key)
        else:
            if not is_empty:
                skipped.append(key)
                continue  # 标量字段：用户优先
            engine_config[key] = default_val
            injected.append(key)

    if injected or deep_merged:
        logger.info(
            "[GLM-4.7-W8A8] Engine config tuning for arch=%s quantize=%s | injected=%s | deep_merged=%s | user_kept=%s",
            info.model_architecture, info.model_quantize, injected, deep_merged, skipped,
        )
        # 摘要：打印 W8A8 影响的最终字段值，便于排查
        try:
            summary = {k: engine_config.get(k) for k in _GLM47_W8A8_ENGINE_DEFAULTS.keys()}
            logger.info(
                "[GLM-4.7-W8A8] Final engine_config for tuned keys:\n%s",
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("[GLM-4.7-W8A8] Skip summary dump: %s", e)


def _build_vllm_cmd_parts(params: Dict[str, Any]) -> str:
    """构建 vLLM 核心启动命令字符串。

    将 engine_config 字典转换为 vLLM CLI 参数格式：
    python3 -m vllm.entrypoints.openai.api_server --arg1 value1 ...

    Args:
        params: 参数字典，必须包含 engine_config 字典

    Returns:
        str: 完整的 vLLM 启动命令字符串
    """
    engine_config = _prepare_engine_config(params)
    cmd_parts = ["python3", "-m", "vllm.entrypoints.openai.api_server"]

    for arg, value in engine_config.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if arg == "max_num_batched_tokens":
            try:
                if int(value) <= 0:
                    logger.warning("Skip invalid max_num_batched_tokens=%s; vLLM requires >=1", value)
                    continue
            except (TypeError, ValueError):
                logger.warning("Skip non-integer max_num_batched_tokens=%s", value)
                continue

        arg_name = f"--{arg.replace('_', '-')}"
        cmd_parts.extend(_format_cli_arg(arg_name, value))

    return " ".join(cmd_parts)


# ── 推测解码 (Speculative Decoding) ──────────────────────────────────────


def _format_speculative_result(config_entries: List[str]) -> str:
    """将推测解码配置列表格式化为 --speculative-config 命令行参数。"""
    result = " --speculative-config '{" + ", ".join(config_entries) + "}'"
    logger.info("[AdvFeature-SpecDecode] Generated params: %s", result.strip())
    return result


def _handle_draft_model_case(params: Dict[str, Any], config: List[str]) -> None:
    """处理有草稿模型的推测解码配置"""
    draft_path = params.get("speculative_decode_model_path", "")
    # 对路径中的双引号和反斜杠进行 JSON 转义，防止 JSON-in-shell 注入
    safe_path = draft_path.replace('\\', '\\\\').replace('"', '\\"')
    config.append(f'"model": "{safe_path}"')
    config.append('"draft_tensor_parallel_size": 1')
    draft_model_info = ModelIdentifierDraft(params.get("speculative_decode_model_path"))

    if 'eagle3' in draft_model_info.draft_model_architecture.lower():
        logger.info('--- Using the Eagle3 speculative decoding approach ---')
        config.append('"method" : "eagle3"')
        num_spec_tokens = 4
        config.append(f'"num_speculative_tokens": {num_spec_tokens}')
    else:
        logger.info('--- Using the draft model speculative decoding approach ---')
        config.append('"method" : "draft_model"')
        num_spec_tokens = 4
        config.append(f'"num_speculative_tokens": {num_spec_tokens}')


def _handle_mtp_case(model_info: ModelIdentifier, mtp_support_models: List[Any],
                     mtp_types: List[str], config: List[str]) -> None:
    """处理 MTP 推测解码配置"""
    logger.info('--- Using the MTP speculative decoding approach ---')

    for i, model_group in enumerate(mtp_support_models):
        if model_info.model_architecture in model_group:
            config.append(f'"method": "{mtp_types[i]}"')
            break
    # MTP 强制 num_speculative_tokens=3（官方 GLM-4.7 / DeepSeek-V3 推荐值）
    config.append('"num_speculative_tokens": 3')


def _handle_suffix_case(config: List[str]) -> None:
    """处理 suffix 推测解码配置"""
    logger.info('--- Using the suffix speculative decoding approach ---')
    config.append('"method" : "suffix"')
    config.append('"num_speculative_tokens": 5')
    config.append('"suffix_decoding_max_cached_requests": 1000')


def _is_mtp_or_suffix_strategy(params: Dict[str, Any], engine: str) -> bool:
    """判断当前投机推理策略是否为 MTP 或 suffix（即非草稿模型方案）。

    当启用投机推理且未指定草稿模型路径时，策略一定是 MTP 或 suffix。
    vllm_ascend 不注入 VLLM_EARS_TOLERANCE，Ascend 侧无需该参数。
    """
    if not params.get("enable_speculative_decode"):
        return False
    if engine != "vllm":
        return False
    if params.get("speculative_decode_model_path"):
        return False
    return True


def _build_speculative_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 MTP / suffix 投机推理策略所需的环境变量命令。

    当投机推理采用 MTP 或 suffix 策略时，默认注入
    ``VLLM_EARS_TOLERANCE=0.5`` 环境变量以控制容忍度参数。

    Args:
        params: 参数字典
        engine: 引擎类型

    Returns:
        List[str]: 环境变量设置命令列表，未启用时返回空列表
    """
    if not _is_mtp_or_suffix_strategy(params, engine):
        return []
    logger.info("[AdvFeature-SpecDecode] MTP/suffix strategy detected, "
                "injecting VLLM_EARS_TOLERANCE=0.5")
    return ['export VLLM_EARS_TOLERANCE=0.5']


def _resolve_mtp_method(model_architecture: str) -> str:
    mtp_methods_by_arch = {
        "DeepseekV3ForCausalLM": "deepseek_mtp",
        "DeepseekV32ForCausalLM": "deepseek_mtp",
        "GlmMoeDsaForCausalLM": "deepseek_mtp",
        "Qwen3NextForCausalLM": "qwen3_next_mtp",
        "Glm4MoeForCausalLM": "glm4_moe_mtp",
        "Qwen3_5ForConditionalGeneration": "qwen3_5_mtp",
        "Qwen3_5MoeForConditionalGeneration": "qwen3_5_mtp",
    }
    return mtp_methods_by_arch.get(model_architecture, "")


def resolve_speculative_strategy(params: Dict[str, Any], engine: str) -> str:
    """Return the speculative decoding strategy selected for vLLM."""
    if engine not in ("vllm", "vllm_ascend"):
        return ""

    if params.get("speculative_decode_model_path"):
        draft_model_info = ModelIdentifierDraft(params.get("speculative_decode_model_path"))
        if 'eagle3' in draft_model_info.draft_model_architecture.lower():
            return "eagle3"
        return "draft_model"

    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type"),
    )
    if model_info.model_architecture == "Qwen3NextForCausalLM" and engine == "vllm_ascend":
        return "suffix"

    mtp_method = _resolve_mtp_method(model_info.model_architecture)
    if mtp_method:
        return "suffix" if get_lmcache_env() else mtp_method

    return "suffix"


def _build_speculative_cmd(params: Dict[str, Any], engine: str) -> str:
    """推测解码方案的自动选取。

    根据模型架构自动选择最优的推测解码策略：
    1. 如有草稿模型 → eagle3 / draft_model
    2. Qwen3NextForCausalLM + vllm_ascend → suffix
    3. DeepSeek/GLM-5/Qwen3Next/Glm4Moe → MTP
    4. 其他 → suffix

    Args:
        params: 参数字典
        engine: 引擎类型 ('vllm' 或 'vllm_ascend')

    Returns:
        str: --speculative-config 参数字符串，未启用时返回空字符串
    """
    model_info = ModelIdentifier(params.get("model_name"),
                                 params.get("model_path"),
                                 params.get("model_type"))
    logger.info("[AdvFeature-SpecDecode] Model architecture detection: %s (model_name=%s)",
                model_info.model_architecture, params.get("model_name"))

    speculative_config_temp = []

    strategy = resolve_speculative_strategy(params, engine)
    if not strategy:
        logger.info("[AdvFeature-SpecDecode] engine='%s' does not support speculative decode, skipping", engine)
        return ""

    if params.get("speculative_decode_model_path"):
        logger.info("[AdvFeature-SpecDecode] Draft model path detected: %s, using draft_model strategy",
                    params.get("speculative_decode_model_path"))
        _handle_draft_model_case(params, speculative_config_temp)
        return _format_speculative_result(speculative_config_temp)

    if strategy == "suffix":
        logger.info("[AdvFeature-SpecDecode] Architecture %s → suffix strategy",
                    model_info.model_architecture)
        _handle_suffix_case(speculative_config_temp)
        return _format_speculative_result(speculative_config_temp)

    if strategy.endswith("_mtp"):
        logger.info("[AdvFeature-SpecDecode] Architecture %s → MTP strategy (%s)",
                    model_info.model_architecture, strategy)
        speculative_config_temp.append(f'"method": "{strategy}"')
        # MTP 强制 num_speculative_tokens=3（官方 GLM-4.7 / GLM-5 / DeepSeek 推荐值）
        speculative_config_temp.append('"num_speculative_tokens": 3')
        return _format_speculative_result(speculative_config_temp)

    return ""


def _should_append_auto_speculative_config(params: Dict[str, Any]) -> bool:
    """Return True when launcher should synthesize speculative_config itself."""
    if not params.get("enable_speculative_decode"):
        return False
    engine_config = params.get("engine_config") or {}
    return not bool(engine_config.get("speculative_config"))


# ── KV Sparse（IndexCache / FP8 KV CACHE）───────────────────────────────

# 当 enable_sparse=true 时，根据模型架构决定 KV 稀疏策略：
#   - INDEXCACHE_ARCHS 中的架构 → IndexCache 加速
#   - 其他架构 → FP8 KV CACHE 量化


def _build_kv_sparse_cmd(params: Dict[str, Any], engine: str) -> str:
    """构建 KV 稀疏特性的启动命令参数。

    仅 vllm (NVIDIA) 支持 KV 稀疏特性。
    根据模型架构决定策略：
      - IndexCache 架构（GlmMoeDsa/DeepseekV32）：返回 --hf-overrides CLI 参数
      - 其他架构：直接修改 engine_config 注入 kv_cache_dtype=fp8，返回空字符串

    **必须在 _build_vllm_cmd_parts 之前调用**，以便 FP8 参数正确合入基础命令，
    避免与 engine_config 中已有的 kv_cache_dtype 产生重复。

    Args:
        params: 参数字典（FP8 路径会就地修改 engine_config）
        engine: 引擎类型

    Returns:
        str: 额外的 CLI 参数字符串（IndexCache 返回 --hf-overrides，FP8 返回空串）
    """
    if engine != "vllm":
        return ""

    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type"),
    )
    arch = model_info.model_architecture

    if arch in INDEXCACHE_ARCHS:
        logger.info("[KV Sparse] Architecture %s → IndexCache strategy (--hf-overrides)", arch)
        return " --hf-overrides '{\"index_topk_freq\": 4}'"
    else:
        logger.info("[KV Sparse] Architecture %s → FP8 KV CACHE strategy (kv_cache_dtype=fp8)", arch)
        engine_config = params.setdefault("engine_config", {})
        engine_config["kv_cache_dtype"] = "fp8"
        engine_config["calculate_kv_scales"] = True
        return ""


def build_start_command(params: Dict[str, Any]) -> str:
    """为 launcher 生成 vLLM 启动命令字符串（旧版接口）。

    此函数仅执行命令拼装，不启动任何子进程。
    返回的命令不包含环境变量设置，适合简单场景。

    Args:
        params: 参数字典

    Returns:
        str: vLLM 启动命令字符串

    Raises:
        ValueError: 分布式模式不支持此简化接口

    建议:
        推荐使用 build_start_script() 获取完整脚本
    """
    if params.get("distributed", False):
        raise ValueError("Launcher MVP does not support distributed mode for vLLM.")
    return _build_vllm_cmd_parts(params)


# ── Shell snippet constants for distributed scripts ────────────────────────
_SH_DETECT_IP = (
    "$(python3 -c \""
    "import socket;"
    "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
    "s.connect(('8.8.8.8',80));"
    "print(s.getsockname()[0]);"
    "s.close()\""
    " 2>/dev/null || hostname -i)"
)
# VLLM_HOST_IP 优先级：POD_IP（K8s downward API） > RANK_IP（上层调度注入）> 路由探测。
# 与 HCCL_IF_IP 走同一来源，避免多网卡场景下两者落到不同网卡。
_SH_VLLM_HOST = "export VLLM_HOST_IP=${POD_IP:-${RANK_IP:-" + _SH_DETECT_IP + "}}"
_SH_IF_DETECT = (
    "$(awk '$2==\"00000000\"{print $1;exit}'"
    " /proc/net/route 2>/dev/null || echo eth0)"
)


@dataclass
class DistScriptCtx:
    """分布式脚本生成共用上下文，减少函数参数传递。"""

    engine: str
    cmd: str
    is_ascend: bool
    node_rank: int
    nnodes: int
    head_addr: str
    ray_port: str
    node_ips: str


def build_triton_patch_preamble(engine: str) -> str:
    """返回 Triton NPU 补丁的 shell 脚本片段（用于注入到 start_command.sh preamble 层）。

    Triton 补丁是一次性文件修改操作，应在引擎启动前执行一次即可。
    将其放在 preamble 而非 build_start_script 中，可避免 retry/fallback
    命令因缩进 heredoc 闭合标记而导致 bash 语法错误。

    Args:
        engine: 引擎类型

    Returns:
        str: shell 脚本片段；非 vllm_ascend 或版本 < 0.14 时返回空字符串。
    """
    if not _need_triton_patch(engine):
        return ""
    lines = _build_triton_npu_patch_block()
    return "\n".join(lines) + "\n"


def _build_triton_npu_patch_block() -> List[str]:
    """返回 Ascend NPU Triton 驱动补丁的 shell 命令列表。"""
    return [
        "# Patch triton driver.py: Ascend NPU has no Triton backend, return dummy driver",
        "python3 << 'TRITON_PATCH_EOF'",
        "try:",
        "    import triton.runtime, os",
        "    drv_path = os.path.join(os.path.dirname(triton.runtime.__file__), 'driver.py')",
        "    with open(drv_path) as f:",
        "        src = f.read()",
        "    if 'raise RuntimeError' in src and 'PATCHED_NPU' not in src:",
        "        patch = '''",
        "        # PATCHED_NPU: Ascend NPU has no Triton backend, provide dummy driver",
        "        class _NpuDummyDrv:",
        "            def get_current_target(self):",
        "                import types; return types.SimpleNamespace("
        "backend='npu', arch='Ascend910B', warp_size=0)",
        "            def get_current_device(self): return 0",
        "            def get_device_capability(self, *a): return (0, 0)",
        "            def get_device_properties(self, device=0):",
        "                try:",
        "                    import torch_npu; "
        "n = torch_npu.npu.get_device_name(device); "
        "c = 20 if '910B' in str(n) else 30",
        "                except Exception: c = 20",
        "                return {'num_aicore': c, 'num_vectorcore': c}",
        "            def __getattr__(self, name): return _NpuDummyDrv()",
        "            def __call__(self, *a, **k): return self",
        "            def __repr__(self): return '<NpuDummy>'",
        "            def __int__(self): return 0",
        "            def __bool__(self): return False",
        "        return _NpuDummyDrv()'''",
        "        src = src.replace(",
        '            \'raise RuntimeError('
        'f"{len(active_drivers)} active drivers '
        '({active_drivers}). There should only be one.")\',',
        "            patch.strip()",
        "        )",
        "        with open(drv_path, 'w') as f:",
        "            f.write(src)",
        "        print('[triton-patch] Patched', drv_path, 'for Ascend NPU')",
        "    else:",
        "        print('[triton-patch] Already patched or not needed')",
        "except Exception as e:",
        "    print(f'[triton-patch] Skip: {e}')",
        "TRITON_PATCH_EOF",
    ]


def build_modelslim_quarot_patch_preamble(engine: str) -> str:
    """为 QuaRot 等非 modelslim 量化格式注入 modelslim_config.py 兼容性补丁。

    vllm-ascend 的 ``modelslim_config.ModelSlimConfig.is_layer_skipped_ascend``
    使用 ``self.quant_description[key]`` 直接访问字典，当模型使用 QuaRot 等
    非华为 AMCT 量化方案时，quant_description 中缺少各层独立权重 key
    (如 ``model.layers.0.self_attn.q_proj.weight``)，导致 KeyError 崩溃。

    本补丁将所有 ``self.quant_description[...]`` 替换为
    ``self.quant_description.get(...)``，缺失 key 时返回 None（非 "FLOAT"），
    即该层不被跳过、按量化处理——这是 W8A8 模型的安全默认行为。

    仅对 vllm_ascend 引擎生效。

    Args:
        engine: 引擎类型

    Returns:
        str: shell 脚本片段；非 vllm_ascend 时返回空字符串。
    """
    if engine != "vllm_ascend":
        return ""
    return (
        "# --- wings: modelslim_config.py QuaRot compatibility patch ---\n"
        "python3 << 'MODELSLIM_PATCH_EOF'\n"
        "try:\n"
        "    import importlib.util, pathlib\n"
        "    spec = importlib.util.find_spec("
        "'vllm_ascend.quantization.modelslim_config')\n"
        "    if spec and spec.origin:\n"
        "        p = pathlib.Path(spec.origin)\n"
        "        txt = p.read_text()\n"
        "        old = 'self.quant_description[shard_prefix + ' + '\"' + '.weight' + '\"' + ']'\n"
        "        new = 'self.quant_description.get(shard_prefix + ' + '\"' + '.weight' + '\"' + ')'\n"
        "        if old in txt:\n"
        "            p.write_text(txt.replace(old, new))\n"
        "            print('[modelslim-patch] Patched modelslim_config.py: "
        "dict[] -> dict.get() for QuaRot compatibility')\n"
        "        else:\n"
        "            print('[modelslim-patch] Already patched or pattern not found')\n"
        "    else:\n"
        "        print('[modelslim-patch] modelslim_config module not found, skipping')\n"
        "except Exception as e:\n"
        "    print(f'[modelslim-patch] Skip: {e}')\n"
        "MODELSLIM_PATCH_EOF\n"
        "# --- end modelslim patch ---\n"
    )


def _build_comm_env_commands(is_ascend: bool) -> List[str]:
    """返回 HCCL/NCCL 通信环境变量设置命令。"""
    if is_ascend:
        hccl_connect_timeout = os.getenv('HCCL_CONNECT_TIMEOUT', '1800')
        hccl_exec_timeout = os.getenv('HCCL_EXEC_TIMEOUT', '7200')
        # Ascend NPU 首次推理需 JIT 编译算子，耗时可能超过 Ray 编译DAG默认 300s 超时。
        # 设置较大值避免 RayChannelTimeoutError；可通过环境变量覆盖。
        ray_cgraph_get_timeout = os.getenv('RAY_CGRAPH_get_timeout', '3600')
        return [
            "export HCCL_WHITELIST_DISABLE=1",
            "export HCCL_IF_IP=$VLLM_HOST_IP",
            "export HCCL_SOCKET_IFNAME=" + _SH_IF_DETECT,
            "export TP_SOCKET_IFNAME=" + _SH_IF_DETECT,
            "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
            "export ASCEND_PROCESS_LOG_PATH=/tmp/ray_vllm010",
            f"export HCCL_CONNECT_TIMEOUT={hccl_connect_timeout}",
            f"export HCCL_EXEC_TIMEOUT={hccl_exec_timeout}",
            f"export RAY_CGRAPH_get_timeout={ray_cgraph_get_timeout}",
        ]
    nccl_if = os.getenv('NCCL_SOCKET_IFNAME', 'eth0')
    return [
        f"export NCCL_SOCKET_IFNAME={nccl_if}",
        f"export TP_SOCKET_IFNAME={nccl_if}",
    ]


def _build_ray_wait_loop(nnodes: int) -> List[str]:
    """返回等待所有 Ray 节点加入的 shell 循环命令。

    行为：最多轮询 60 次 × 5s = 300s，每次用 python 读取 ray.nodes() 中 alive
    节点数；达到 ``nnodes`` 立即跳出循环。

    Fail-fast：超时仍未达标时直接 ``exit 1``，避免后续 ``exec vllm`` 在
    ray 集群未就绪的情况下进入卡 compile 路径——那种隐性卡死非常难定位，
    显式 ``exit 1`` + 明确错误日志可以让 K8s 立刻 CrashLoopBackOff，
    运维通过 ``kubectl logs`` 1 秒就能看到根因。
    """
    return [
        "RAY_WAIT_OK=0",
        "for i in $(seq 1 60); do",
        "  COUNT=$(python3 -c \"import ray;"
        " ray.init(address='auto',"
        "ignore_reinit_error=True);"
        " print(len([n for n in ray.nodes()"
        " if n['alive']]));"
        " ray.shutdown()\""
        " 2>/dev/null || echo 0)",
        f"  if [ \"$COUNT\" -ge \"{nnodes}\" ]; then RAY_WAIT_OK=1; break; fi",
        f"  echo \"[ray-wait] iter=$i count=$COUNT expected={nnodes}, sleep 5s...\"",
        "  sleep 5",
        "done",
        "if [ \"$RAY_WAIT_OK\" != \"1\" ]; then",
        f"  echo \"[ray-wait] FATAL: only $COUNT/{nnodes} ray nodes joined after"
        " 300s. Check worker pod status / network / RAY_PORT reachability.\" >&2",
        "  exit 1",
        "fi",
        "echo \"[ray-wait] OK: $COUNT ray nodes joined.\"\n",
    ]


def _build_ray_head_start_commands(params: Dict[str, Any], ctx: DistScriptCtx) -> List[str]:
    """Build the Ray head startup command and matching diagnostics."""
    ray_head_resource = _get_ray_resource_flag(ctx.engine, params)
    ray_head_cmd = (
        f"ray start --head --port={ctx.ray_port}"
        f" --node-ip-address=$VLLM_HOST_IP"
        f" {ray_head_resource}"
        f" --dashboard-host=$VLLM_HOST_IP\n"
    )
    logger.info("[ray] head start command: %s", ray_head_cmd.strip())
    return [f'echo "[ray] head start command: {ray_head_cmd.strip()}"', ray_head_cmd]


def _build_ray_parallel_overrides(params: Dict[str, Any], ctx: DistScriptCtx) -> tuple[str, str]:
    """Override TP/PP for Ray MoE architectures that need per-node TP."""
    model_info_ray = ModelIdentifier(
        params.get("model_name"), params.get("model_path"), params.get("model_type"))
    ray_auto_pp_archs = {
        "Qwen3MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "MiniMaxM2ForCausalLM",
    }
    if getattr(model_info_ray, "model_architecture", None) not in ray_auto_pp_archs:
        return ctx.cmd, ""

    nodes_list = ctx.node_ips.split(",") if ctx.node_ips else []
    num_nodes = len(nodes_list) if nodes_list else 1
    tp_size = params.get("device_count", 1)
    cmd_for_exec = _strip_cli_flag(ctx.cmd, "--tensor-parallel-size")
    cmd_for_exec = _strip_cli_flag(cmd_for_exec, "--pipeline-parallel-size")
    logger.info(
        "[vllm_ascend ray] Set parallel parameters: pipeline_parallel_size=%s, tensor_parallel_size=%s",
        num_nodes,
        tp_size,
    )
    return cmd_for_exec, f" --pipeline-parallel-size {num_nodes} --tensor-parallel-size {tp_size}"


def _build_ray_backend_override(params: Dict[str, Any], cmd_for_exec: str) -> tuple[str, str]:
    """Ensure the generated vLLM command explicitly uses the Ray executor."""
    if params.get("distributed_executor_backend", "ray") != "ray":
        return cmd_for_exec, ""
    cmd_for_exec = _strip_cli_flag(cmd_for_exec, "--distributed-executor-backend")
    return cmd_for_exec, " --distributed-executor-backend ray"


def _build_ray_head_exec_command(
    params: Dict[str, Any],
    ctx: DistScriptCtx,
    sparse_args: str,
) -> str:
    """Build the final vLLM exec command for the Ray head node."""
    eager_flag = " --enforce-eager" if _need_enforce_eager(ctx.engine) else ""
    speculative_extra = _build_speculative_cmd(params, ctx.engine) if _should_append_auto_speculative_config(params) else ""
    cmd_for_exec, ray_pp_extra = _build_ray_parallel_overrides(params, ctx)
    cmd_for_exec, backend_extra = _build_ray_backend_override(params, cmd_for_exec)
    return (
        f"exec {cmd_for_exec}{eager_flag}"
        f"{speculative_extra}{sparse_args}"
        f"{ray_pp_extra}"
        f"{backend_extra}"
    )


def _build_ray_head_commands(
    params: Dict[str, Any],
    ctx: DistScriptCtx,
    sparse_args: str,
) -> List[str]:
    """Build shell commands for the Ray head node (rank 0)."""
    parts: List[str] = [_SH_VLLM_HOST]
    parts.extend(_build_comm_env_commands(ctx.is_ascend))
    parts.append("export GLOO_SOCKET_IFNAME=" + _SH_IF_DETECT + "\n")
    parts.extend(_build_ray_head_start_commands(params, ctx))
    parts.extend(_build_ray_wait_loop(ctx.nnodes))
    parts.append(_build_ray_head_exec_command(params, ctx, sparse_args))
    return parts


def _build_ascend_ray_worker_env(ray_port: str, node_ips: str, head_addr: str = "") -> List[str]:
    """构建 Ascend NPU Ray worker 的 HCCL 环境命令。

    优先尝试 Master 注入的 head_addr，失败后再扫描 node_ips 列表。
    """
    # 构建优先级排列的 IP 列表：head_addr 在前（如果存在且不在 node_ips 中已是首位）
    if head_addr:
        ip_list_expr = (
            f"KNOWN_HEAD=\"{head_addr}\"\n"
            f"NODE_IPS_LIST=\"{node_ips}\"\n"
            "# 优先尝试已知 head，再扫描其余节点\n"
            "CANDIDATE_IPS=\"$KNOWN_HEAD $(echo $NODE_IPS_LIST | tr ',' ' ' "
            "| grep -v \"^$KNOWN_HEAD$\")\""
        )
    else:
        ip_list_expr = (
            f"NODE_IPS_LIST=\"{node_ips}\"\n"
            "CANDIDATE_IPS=\"$(echo $NODE_IPS_LIST | tr ',' ' ')\""
        )
    return [
        "export HCCL_WHITELIST_DISABLE=1",
        # 先确定 VLLM_HOST_IP（POD_IP > RANK_IP > 路由探测），后续 HCCL_IF_IP 与 Ray
        # node-ip-address 都复用此值，保证多网卡场景下走同一张网卡。
        _SH_VLLM_HOST,
        ip_list_expr,
        "HEAD_IP=\"\"",
        f"echo \"[worker] Scanning for Ray head on port {ray_port}...\"",
        "for attempt in $(seq 1 120); do",
        "  for ip in $CANDIDATE_IPS; do",
        f"    if python3 -c \""
        f"import socket; s=socket.socket();"
        f" s.settimeout(2);"
        f" s.connect(('$ip',{ray_port}));"
        f" s.close()\""
        f" 2>/dev/null; then",
        "      HEAD_IP=$ip",
        f"      echo \"[worker] Found Ray head at $HEAD_IP:{ray_port}\"",
        "      break 2",
        "    fi",
        "  done",
        "  sleep 5",
        "done",
        "if [ -z \"$HEAD_IP\" ]; then "
        "echo '[worker] ERROR: Could not find Ray head'; "
        "exit 1; fi\n",
        # 与 head 保持一致：HCCL_IF_IP 复用 VLLM_HOST_IP，避免与 8.8.8.8 路由探测/
        # socket 出口探测落到不同网卡（业务网 vs 管理网），导致 HCCL 性能/稳定性问题。
        "export HCCL_IF_IP=$VLLM_HOST_IP",
        "export HCCL_SOCKET_IFNAME=" + _SH_IF_DETECT,
        "export TP_SOCKET_IFNAME=" + _SH_IF_DETECT,
        "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
        "export ASCEND_PROCESS_LOG_PATH=/tmp/ray_vllm010",
        # Ascend NPU 首次推理需 JIT 编译算子，耗时可能超过 Ray 编译DAG默认 300s 超时。
        "export RAY_CGRAPH_get_timeout=" + os.getenv('RAY_CGRAPH_get_timeout', '3600'),
    ]


def _build_ray_worker_commands(
    params: Dict[str, Any],
    ctx: DistScriptCtx,
) -> List[str]:
    """构建 Ray worker 节点 (rank > 0) 的脚本命令列表。"""
    parts: List[str] = []
    if ctx.is_ascend:
        parts.extend(_build_ascend_ray_worker_env(ctx.ray_port, ctx.node_ips, ctx.head_addr))
    else:
        nccl_if = os.getenv('NCCL_SOCKET_IFNAME', 'eth0')
        parts.extend([
            f"export NCCL_SOCKET_IFNAME={nccl_if}",
            f"export TP_SOCKET_IFNAME={nccl_if}",
            _SH_VLLM_HOST,
            "for i in $(seq 1 60); do",
            f"  python3 -c \"import socket;"
            f" s=socket.socket(); s.settimeout(2);"
            f" s.connect(('{ctx.head_addr}',{ctx.ray_port}));"
            f" s.close()\""
            f" 2>/dev/null && break",
            "  sleep 5",
            "done",
            f"HEAD_IP=\"{ctx.head_addr}\"",
        ])
    parts.append("export GLOO_SOCKET_IFNAME=" + _SH_IF_DETECT + "\n")
    ray_worker_resource = _get_ray_resource_flag(ctx.engine, params)
    ray_worker_cmd = (
        f"exec ray start"
        f" --address=$HEAD_IP:{ctx.ray_port}"
        f" --node-ip-address=$VLLM_HOST_IP"
        f" {ray_worker_resource} --block"
    )
    logger.info("[ray] worker start command: %s", ray_worker_cmd)
    parts.append(f'echo "[ray] worker start command: {ray_worker_cmd}"')
    parts.append(ray_worker_cmd)
    return parts


def _build_dp_env_commands(is_ascend: bool, params: Dict[str, Any]) -> List[str]:
    """返回 dp_deployment 模式的分布式通信环境变量命令。"""
    net_if = os.getenv("NETWORK_INTERFACE", os.getenv("GLOO_SOCKET_IFNAME", "eth0"))
    if is_ascend:
        hccl_connect_timeout = os.getenv('HCCL_CONNECT_TIMEOUT', '1800')
        hccl_exec_timeout = os.getenv('HCCL_EXEC_TIMEOUT', '7200')
        is_deepseek_v3_family = _is_deepseek_v3_family_ascend_dp_deployment(params)
        omp_threads = os.getenv('OMP_NUM_THREADS', '1' if is_deepseek_v3_family else '100')
        hccl_buffsize = os.getenv('HCCL_BUFFSIZE', '200' if is_deepseek_v3_family else '1024')
        env_commands = [
            # 与 Ray 路径保持一致：先建立 VLLM_HOST_IP（POD_IP > RANK_IP > 路由探测），
            # HCCL_IF_IP 直接复用，避免多网卡场景下与 vLLM 通信走错网卡。
            _SH_VLLM_HOST,
            "export HCCL_WHITELIST_DISABLE=1",
            "export HCCL_IF_IP=$VLLM_HOST_IP",
            f"export GLOO_SOCKET_IFNAME={net_if}",
            f"export TP_SOCKET_IFNAME={net_if}",
            f"export HCCL_SOCKET_IFNAME={net_if}",
            f"export HCCL_CONNECT_TIMEOUT={hccl_connect_timeout}",
            f"export HCCL_EXEC_TIMEOUT={hccl_exec_timeout}",
            "export OMP_PROC_BIND=false",
            f"export OMP_NUM_THREADS={omp_threads}",
            f"export HCCL_BUFFSIZE={hccl_buffsize}",
            'echo "[wings-env] final HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"',
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        ]
        if is_deepseek_v3_family:
            env_commands.extend([
                "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
                "export HCCL_INTRA_PCIE_ENABLE=1",
                "export HCCL_INTRA_ROCE_ENABLE=0",
            ])
        else:
            env_commands.extend([
                "export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/"
                "latest/opp/deepseek-v32/vendors/customize:${ASCEND_CUSTOM_OPP_PATH:-}",
                "export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/"
                "opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH:-}",
            ])
        if _is_deepseek_ascend_dp_deployment(params):
            engine_ready_timeout = os.getenv("VLLM_ENGINE_READY_TIMEOUT_S", "7200")
            env_commands.append(f"export VLLM_ENGINE_READY_TIMEOUT_S={engine_ready_timeout}")
        return env_commands
    return [
        f"export GLOO_SOCKET_IFNAME={net_if}",
        f"export TP_SOCKET_IFNAME={net_if}",
        f"export NCCL_SOCKET_IFNAME={net_if}",
        f"export VLLM_NIXL_SIDE_CHANNEL_PORT="
        f"{params.get('nixl_port', os.getenv('VLLM_NIXL_SIDE_CHANNEL_PORT', '12345'))}",
        "export NCCL_IB_DISABLE=0",
        "export NCCL_CUMEM_ENABLE=0",
        "export NCCL_NET_GDR_LEVEL=SYS",
    ]


def _transform_dp_cmd(cmd: str) -> str:
    """将 vllm api_server 命令转换为 vllm serve 格式（dp_deployment 入口）。"""
    _model_match = re.search(r"--model\s+('(?:[^']*)'|\S+)", cmd)
    if not _model_match:
        return cmd
    _model_val = _model_match.group(1)
    dp_cmd = re.sub(r"\s*--model\s+(?:'[^']*'|\S+)", "", cmd)
    return re.sub(
        r"^python3\s+-m\s+vllm\.entrypoints\.openai\.api_server",
        f"vllm serve {_model_val}",
        dp_cmd,
    )


def _strip_dp_cli_flags(cmd: str) -> str:
    """移除基础命令中已有的 data-parallel 参数，避免 dp_deployment 追加时重复传参。

    vLLM 0.18 会对重复 CLI key 打印 warning；更重要的是，dp_deployment
    的 rank / start-rank / local-size 组合由本模块按节点统一计算，不能被
    engine_config 中的同名字段污染。
    """
    flags_with_value = [
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--data-parallel-size",
        "--data-parallel-size-local",
        "--data-parallel-rank",
        "--data-parallel-start-rank",
    ]
    for flag in flags_with_value:
        cmd = _strip_cli_flag(cmd, flag)
    cmd = re.sub(r"\s+--data-parallel-external-lb\b", "", cmd)
    cmd = re.sub(r"\s+--headless\b", "", cmd)
    return cmd


def _build_dp_deployment_commands(
    params: Dict[str, Any],
    ctx: DistScriptCtx,
    sparse_args: str = "",
) -> List[str]:
    """构建 dp_deployment 模式的脚本命令列表。"""
    parts: List[str] = []
    dp_rpc_port = str(params.get("rpc_port", os.getenv('VLLM_DP_RPC_PORT', '13355')))

    model_info = ModelIdentifier(
        params.get("model_name"), params.get("model_path"), params.get("model_type"))
    if (model_info.model_architecture in ["DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"]
            and ctx.engine == "vllm_ascend"):
        dp_size, dp_size_local = "4", "2"
        dp_start_rank = "2" if ctx.node_rank != 0 else "0"
    else:
        dp_size, dp_size_local = str(ctx.nnodes), "1"
        dp_start_rank = str(ctx.node_rank)

    parts.extend(_build_dp_env_commands(ctx.is_ascend, params))
    dp_cmd = _strip_dp_cli_flags(_transform_dp_cmd(ctx.cmd))
    speculative_extra = _build_speculative_cmd(params, ctx.engine) if _should_append_auto_speculative_config(params) else ""
    dp_cmd = f"{dp_cmd}{speculative_extra}{sparse_args}"

    if ctx.node_rank == 0:
        parts.append(
            f"exec {dp_cmd}"
            f" --data-parallel-address {shlex.quote(ctx.head_addr)}"
            f" --data-parallel-rpc-port {dp_rpc_port}"
            f" --data-parallel-size {dp_size}"
            f" --data-parallel-size-local {dp_size_local}"
        )
    else:
        dp_cmd_headless = re.sub(r"\s*--host\s+(?:'[^']*'|\S+)", "", dp_cmd)
        dp_cmd_headless = re.sub(r"\s*--port\s+(?:'[^']*'|\S+)", "", dp_cmd_headless)
        parts.append(
            f"exec {dp_cmd_headless}"
            f" --data-parallel-address {shlex.quote(ctx.head_addr)}"
            f" --data-parallel-rpc-port {dp_rpc_port}"
            f" --data-parallel-size {dp_size}"
            f" --data-parallel-size-local {dp_size_local}"
            f" --headless"
            f" --data-parallel-start-rank {dp_start_rank}"
        )
    return parts


def _resolve_vllm_dist_params(params: Dict[str, Any]) -> tuple[str, str, str]:
    """从 params / 环境变量解析分布式拓扑基础参数，返回 (head_addr, node_ips, ray_port)。"""
    head_addr = (
        params.get("ray_head_ip")
        or params.get("master_ip")
        or params.get("head_node_addr", "infer-0.infer-hl")
    )
    # NODE_IPS: params["nodes"] 优先（由 config_loader / Master 注入），其次环境变量
    node_ips = params.get("node_ips") or params.get("nodes") or os.getenv("NODE_IPS", head_addr)
    # ray_head_port: params 优先，其次环境变量，最后回退到 28020（与 wings 对齐）
    ray_port = str(params.get("ray_head_port", os.getenv("RAY_PORT", "28020")))
    return head_addr, node_ips, ray_port


def _build_vllm_common_env_cmds(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 vLLM 公共环境变量命令链（对所有部署模式均适用）。"""
    # sidecar 容器无 GPU/NPU，使用环境变量代替 netifaces 探测网络接口
    current_ip = os.getenv("POD_IP", get_local_ip())
    net_if = os.getenv("NETWORK_INTERFACE", os.getenv("GLOO_SOCKET_IFNAME", "eth0"))
    cmds: List[str] = []
    cmds.extend(_build_base_env_commands(params, engine, root_dir))
    cmds.extend(_build_cache_env_commands(engine))

    cmds.extend(_build_qat_env_commands(engine))
    cmds.extend(_build_pd_role_env_commands(engine, current_ip, net_if))
    cmds.extend(_build_speculative_env_commands(params, engine))
    # 架构专用环境变量（GLM-4.7 / Qwen3 / Qwen3.5 / MiniMax-M2.5 / DeepSeek V3.2 / LLaMA 等）
    # 之前只在未被引用的 _build_env_commands 里调用，导致架构专用 env 一行都没进 start_command.sh
    cmds.extend(_build_model_env_commands(params, engine))
    # DeepSeek FP8 / Ascend910_9362 专用 env 也一并挂上，保持与 _build_env_commands 等价
    cmds.extend(_build_deepseek_fp8_env_commands(params, engine))
    cmds.extend(_build_ascend910_9362_env_commands(params, engine))
    return cmds


def _build_vllm_distributed_script(
    params: Dict[str, Any],
    cmd: str,
    common_env_cmds: List[str],
    engine: str,
    sparse_args: str,
) -> str:
    """组装分布式模式（nnodes > 1）的 bash 脚本体并返回。"""
    node_rank = params.get("node_rank", 0)
    nnodes = params.get("nnodes", 1)
    backend = params.get("distributed_executor_backend", "ray")
    head_addr, node_ips, ray_port = _resolve_vllm_dist_params(params)

    is_ascend = (engine == "vllm_ascend")
    ctx = DistScriptCtx(
        engine=engine, cmd=cmd, is_ascend=is_ascend,
        node_rank=node_rank, nnodes=nnodes,
        head_addr=head_addr, ray_port=ray_port, node_ips=node_ips,
    )
    script_parts = list(common_env_cmds)
    if backend == "ray":
        # 注意: Triton NPU 补丁已移到 preamble 层（build_triton_patch_preamble），
        # 不再硬编码在 build_start_script 中，避免 retry/fallback 命令
        # 缩进 heredoc 闭合标记导致 bash 语法错误。
        if node_rank == 0:
            script_parts.extend(_build_ray_head_commands(params, ctx, sparse_args))
        else:
            script_parts.extend(_build_ray_worker_commands(params, ctx))
    else:
        script_parts.extend(_build_dp_deployment_commands(params, ctx, sparse_args))
    return "\n".join(script_parts) + "\n"


def _build_vllm_single_script(
    params: Dict[str, Any],
    cmd: str,
    common_env_cmds: List[str],
    engine: str,
    sparse_args: str,
) -> str:
    """组装单机模式的 bash 脚本体并返回。"""
    env_prefix = "\n".join(common_env_cmds) + "\n" if common_env_cmds else ""
    speculative_extra = _build_speculative_cmd(params, engine) if _should_append_auto_speculative_config(params) else ""
    # A+X 环境下需要 --enforce-eager 绕过 triton 版本冲突（与 Ray 路径一致）
    eager_flag = " --enforce-eager" if _need_enforce_eager(engine) else ""

    return env_prefix + f"exec {cmd}{eager_flag}{speculative_extra}{sparse_args}\n"


def build_start_script(params: Dict[str, Any]) -> str:
    """生成完整的 bash 启动脚本体（start_command.sh 内容，不含 shebang）。

    这是 vLLM 适配器的主要入口，生成的脚本将写入共享卷，
    由 engine 容器读取并执行。

    支持的部署模式:

    1. 单机 vllm:
       exec python3 -m vllm.entrypoints.openai.api_server ...

    2. 单机 vllm_ascend:
       source /usr/local/Ascend/.../set_env.sh  # 加载 CANN 环境
       exec python3 -m vllm.entrypoints.openai.api_server ...

    3. Ray 分布式 (rank0 - head 节点) / Ray 分布式 (rank>0 - worker 节点)

    4. DP 分布式 (dp_deployment 后端):
       exec python3 -m vllm... --data-parallel-address ... --data-parallel-rank ...

    Args:
        params: 参数字典，包含 engine/distributed/nnodes/node_rank 等关键字段

    Returns:
        str: 完整的 bash 脚本体（不含 shebang）
    """
    engine = params.get("engine", "vllm")
    # KV 稀疏：必须在 _build_vllm_cmd_parts 之前调用，
    # FP8 路径会就地修改 engine_config，避免 --kv-cache-dtype 重复
    sparse_args = _build_kv_sparse_cmd(params, engine) if params.get("enable_sparse") else ""
    # GLM-4.7-W8A8 引擎参数注入（必须在 _build_vllm_cmd_parts 之前，且只动 W8A8 量化变体）
    _inject_glm47_w8a8_engine_config(params)
    cmd = _build_vllm_cmd_parts(params)
    is_distributed = params.get("distributed", False)
    nnodes = params.get("nnodes", 1)
    common_env_cmds = _build_vllm_common_env_cmds(params, engine)

    if is_distributed and nnodes > 1:
        script = _build_vllm_distributed_script(params, cmd, common_env_cmds, engine, sparse_args)
    else:
        script = _build_vllm_single_script(params, cmd, common_env_cmds, engine, sparse_args)

    script = _inject_env_echo(script)

    return script


def _inject_env_echo(script: str) -> str:
    """在脚本中每条 'export VAR=...' 语句前插入 echo 打印，方便排查环境变量注入情况。

    同时对关键命令行（python3 引擎启动 / ray start / source set_env）前置
    `echo "[wings-cmd] >>> ..."`，便于在 engine.log 里快速定位每条实际执行的命令。

    打印格式：`[wings-env] export VAR=<value>`，使用 `${VAR}` 在 bash 运行时
    展开实际值。注意：值会原样进日志，不再脱敏；如有 token / API key 等敏感
    变量，请避免通过 export 注入或在调用方自行脱敏后再传入。

    为便于排查最终执行环境，脚本内显式 export 的变量都会追加一行
    `[wings-env] export VAR=<value>` 日志。

    Args:
        script: 原始 bash 脚本字符串

    Returns:
        str: 插入 echo 打印后的脚本字符串
    """
    import re as _re
    lines = script.splitlines(keepends=True)
    result = []
    # 命令前缀白名单：匹配到则在前面 echo 一行（截断超长以免日志爆炸）
    # exec \./... 涵盖 MindIE 等用 exec ./bin/daemon 形式启动的可执行文件；
    # \./[A-Za-z0-9_./-]+ 去除 .sh 限制，同时覆盖 ./bin/mindieservice_daemon & 等无扩展名程序。
    cmd_prefix_re = _re.compile(
        r'^(exec\s+(?:python3?|vllm\s+serve|\./\S+)|vllm\s+serve|python3?\s+-m\s+vllm|python3?\s+-m\s+sglang|'
        r'ray\s+(start|stop|status)|source\s+/|nohup\s+|\./[A-Za-z0-9_./-]+)'
    )
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        m = _re.match(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)', stripped)
        if m:
            var_name = m.group(1)
            # 先输出 export 本身，再 echo（${VAR:-} 兜底，避免 `set -u` 触发
            # unbound variable，且能反映 export 后的实际值——含 LD_LIBRARY_PATH
            # 这种追加合并的最终结果）。
            result.append(line)
            indent = line[: len(line) - len(stripped)]
            next_line = lines[idx + 1].lstrip() if idx + 1 < len(lines) else ""
            already_echoed = (
                f"[wings-env] export {var_name}=" in next_line
                or f"[mindie-env] {var_name}=" in next_line
            )
            if not already_echoed:
                result.append(
                    f'{indent}echo "[wings-env] export {var_name}=${{{var_name}:-}}"\n'
                )
            continue
        if cmd_prefix_re.match(stripped):
            indent = line[: len(line) - len(stripped)]
            # 截断到 800 字符，避免单行超长污染日志；shell 单引号转义
            preview = stripped.rstrip("\n").rstrip("&").rstrip()
            if len(preview) > 800:
                preview = preview[:800] + "...<truncated>"
            preview_safe = preview.replace("'", "'\"'\"'")
            already_echoed = bool(result and "[wings-cmd] >>>" in result[-1])
            if not already_echoed:
                result.append(f"{indent}echo '[wings-cmd] >>> {preview_safe}'\n")
        result.append(line)
    return "".join(result)


def start_vllm_distributed(params: Dict):
    """分布式模式入口（sidecar MVP 中不支持）。

    Raises:
        RuntimeError: sidecar 架构不允许直接启动进程
    """
    raise RuntimeError("分布式模式在 sidecar launcher MVP 中已禁用。")


def start_engine(params: Dict[str, Any]):
    """旧版兼容接口（sidecar launcher 模式中已禁用）。

    在 sidecar 架构中，适配器不允许直接启动推理进程。
    应使用 build_start_script() 生成脚本，写入共享卷，
    由 engine 容器执行。

    Raises:
        RuntimeError: 始终抛出，阻止意外调用
    """
    raise RuntimeError(
        "start_engine 在 launcher 模式中已禁用。"
        "请使用 build_start_command() 并将结果写入共享卷。"
    )
