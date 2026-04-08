
"""将 launcher 参数转换成 engine 启动计划。

它是 launcher 控制链路里的中枢桥接层：
- 上游拿到的是 CLI/环境变量；
- 下游需要的是一段可执行的 shell 脚本；
- 中间还要结合硬件探测、默认配置、用户配置和端口规划。

最终产物 `LauncherPlan.command` 会被写入共享卷，供 engine 容器执行。
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings
from core.config_loader import load_and_merge_configs
from core.engine_manager import start_engine_service
from core.hardware_detect import detect_hardware
from core.port_plan import PortPlan
from core.start_args_compat import LaunchArgs
from core.version_util import normalize_engine_version
from engines.vllm_adapter import build_triton_patch_preamble
from utils.env_utils import get_master_ip

logger = logging.getLogger(__name__)

# ── Accel 加速包补丁选项 ────────────────────────────────────────────────────
# 当 ENABLE_ACCEL=true 时，sidecar 会向 start_command.sh 注入：
#   1. export WINGS_ENGINE_PATCH_OPTIONS='{...}'
#   2. python3 $WINGS_ACCEL_DIR/install.py --features "$WINGS_ENGINE_PATCH_OPTIONS"
# WINGS_ACCEL_DIR 由 settings.WINGS_ACCEL_DIR 决定（默认 /accel-volume，可通过环境变量覆盖）
#
# features 列表由以下高级特性环境变量决定（名称与 supported_features.json 对齐）：
#   ENABLE_SPECULATIVE_DECODE → adaptive_draft_model
#   ENABLE_SPARSE             → sparse_kv
#   LMCACHE_OFFLOAD           → lmcache_offload
#   ENABLE_SOFT_FP8           → soft_fp8
#   ENABLE_SOFT_FP4           → soft_fp4
#
# 可通过 WINGS_ENGINE_PATCH_OPTIONS 环境变量直接覆盖（JSON 字符串），
# 此时直接使用用户提供的值，不再按特性开关自动生成。
# ────────────────────────────────────────────────────────────────────────────

# 引擎名到 patch options key 的映射
# 仅包含 supported_features.json 中实际注册的引擎
# vllm_ascend 复用 vllm 的补丁体系（install.py _ENGINE_TO_EXTRAS 中统一为 "vllm"）
_ENGINE_PATCH_KEY_MAP = {
    "vllm": "vllm",
    "vllm_ascend": "vllm",
}

# 高级特性环境变量 → features 名称映射（与 supported_features.json 中的 feature key 对齐）
_FEATURE_SWITCH_MAP = {
    "ENABLE_SPECULATIVE_DECODE": "adaptive_draft_model",
    "ENABLE_SPARSE": "sparse_kv",
    "LMCACHE_OFFLOAD": "lmcache_offload",
    "ENABLE_SOFT_FP8": "soft_fp8",
    "ENABLE_SOFT_FP4": "soft_fp4",
}


def _shell_escape_single_quote(value: str) -> str:
    """对字符串中的单引号进行 shell 安全转义。"""
    return value.replace("'", "'\"'\"'")


def _inject_legacy_distributed_aliases(merged: dict, launch_args: LaunchArgs) -> None:
    """Preserve legacy distributed top-level fields across launcher and worker hops."""
    topology_csv = (
        getattr(launch_args, "node_ips", "")
        or getattr(launch_args, "nodes", "")
        or merged.get("node_ips", "")
        or merged.get("nodes", "")
    )
    if topology_csv:
        merged["node_ips"] = topology_csv
        merged["nodes"] = topology_csv

    master_ip = (
        getattr(launch_args, "master_ip", "")
        or merged.get("master_ip", "")
        or get_master_ip()
        or merged.get("head_node_addr", "")
    )
    if master_ip:
        if merged.get("distributed") and (
            not merged.get("head_node_addr") or merged.get("head_node_addr") == "127.0.0.1"
        ):
            merged["head_node_addr"] = master_ip
        merged["master_ip"] = master_ip
        merged["ray_head_ip"] = (
            getattr(launch_args, "ray_head_ip", "")
            or merged.get("ray_head_ip", "")
            or master_ip
        )
        if merged.get("engine") == "mindie":
            merged["mindie_master_addr"] = merged.get("mindie_master_addr") or master_ip


@dataclass(frozen=True)
class LauncherPlan:
    """launcher 生成的最终计划。

    Attributes:
        command:       完整的 bash 启动脚本内容（含 shebang + set -euo pipefail），
                       将被写入 /shared-volume/start_command.sh 供 engine 容器执行。
        merged_params: 多层合并后的完整参数字典，便于日志审计和调试。
        hardware_env:  硬件探测结果（device/count/details），便于下游判断。
    """

    command: str
    merged_params: dict
    hardware_env: dict


def _prepare_merged_params(launch_args: LaunchArgs, port_plan: PortPlan, hardware: dict) -> dict:
    """配置合并、分布式参数注入与 host/port 分配，返回可直接传入 adapter 的 merged 字典。"""
    known_args = launch_args.to_namespace()
    merged = load_and_merge_configs(hardware_env=hardware, known_args=known_args)
    merged["model_name"] = launch_args.model_name
    merged["model_path"] = launch_args.model_path
    is_distributed = getattr(launch_args, "distributed", False)
    node_rank = getattr(launch_args, "node_rank", 0)
    merged["distributed"] = is_distributed
    merged["nnodes"] = getattr(launch_args, "nnodes", 1)
    merged["node_rank"] = node_rank
    merged["head_node_addr"] = getattr(launch_args, "head_node_addr", "127.0.0.1")
    merged["distributed_executor_backend"] = getattr(
        launch_args, "distributed_executor_backend", "ray",
    )
    _inject_legacy_distributed_aliases(merged, launch_args)
    engine_cfg = dict(merged.get("engine_config", {}))
    # rank0 或单机场景需要显式注入 host/port，让 backend engine 真正提供服务。
    if not is_distributed or node_rank == 0:
        merged["host"] = "0.0.0.0"
        merged["port"] = port_plan.backend_port
        engine_cfg["host"] = "0.0.0.0"
        engine_cfg["port"] = port_plan.backend_port
    else:
        # 非 0 号节点一般只承担计算，不直接对外提供 engine 监听地址。
        merged.pop("host", None)
        merged.pop("port", None)
        engine_cfg.pop("host", None)
        engine_cfg.pop("port", None)
    merged["engine_config"] = engine_cfg
    return merged


def _validate_accel_user_override() -> str:
    """Return the cleaned WINGS_ENGINE_PATCH_OPTIONS override, or '' if invalid."""
    user_override = os.getenv("WINGS_ENGINE_PATCH_OPTIONS", "").strip()
    if not user_override:
        return ""
    try:
        parsed = json.loads(user_override)
        if not isinstance(parsed, dict):
            user_override = ""
    except json.JSONDecodeError:
        user_override = ""
    return user_override


def _build_accel_user_override_snippet(safe_value: str) -> str:
    """Build the fault-tolerant accel install shell snippet for user-override path."""
    accel_dir = settings.WINGS_ACCEL_DIR.rstrip("/")
    return (
        "# --- wings-accel: install patches (user override, fault-tolerant) ---\n"
        f"export WINGS_ENGINE_PATCH_OPTIONS='{safe_value}'\n"
        + f"if [ -f \"{accel_dir}/install.py\" ]; then\n"
        "    echo '[wings-accel] Installing patches (user override)...'\n"
        "    set +e\n"
        f"    python3 {accel_dir}/install.py --features \"$WINGS_ENGINE_PATCH_OPTIONS\"\n"
        "    ACCEL_RC=$?\n"
        "    set -e\n"
        "    if [ $ACCEL_RC -ne 0 ]; then\n"
        "        echo \"[wings-accel] WARNING: Patch install failed"
        " (exit=$ACCEL_RC), skipping. Service will continue without patches.\"\n"
        "    else\n"
        "        echo '[wings-accel] Patches installed successfully.'\n"
        "    fi\n"
        "else\n"
        f"    echo '[wings-accel] WARNING: {accel_dir}/install.py not found, skipping.'\n"
        "fi\n"
    )


def _collect_enabled_features() -> list[str]:
    """Return the list of accel feature names whose environment switches are enabled.

    ENABLE_SPECULATIVE_DECODE=true 时统一安装 adaptive_draft_model 补丁，
    不区分 MTP/Suffix/Draft Model 方案——补丁内部自行判断是否需要生效。
    """
    features = []
    for env_key, feat_name in _FEATURE_SWITCH_MAP.items():
        if os.getenv(env_key, "").strip().lower() != "true":
            continue
        features.append(feat_name)
    return features


def _build_per_feature_fallback_code(patch_key: str, engine_version: str, features: list[str]) -> str:
    """Build per-feature fallback shell code block for fault-tolerant accel install."""
    accel_dir = settings.WINGS_ACCEL_DIR.rstrip("/")
    blocks = []
    for feat in features:
        single_options = json.dumps({patch_key: {"version": engine_version, "features": [feat]}})
        blocks.append(
            f"        echo \"[wings-accel] Trying feature: {feat}\"\n"
            f"        set +e\n"
            f"        python3 {accel_dir}/install.py --features '{single_options}'\n"
            f"        FEAT_RC=$?\n"
            f"        set -e\n"
            f"        if [ $FEAT_RC -ne 0 ]; then\n"
            f"            echo \"[wings-accel] WARNING: Feature '{feat}' install failed (exit=$FEAT_RC), skipping.\"\n"
            f"        else\n"
            f"            echo \"[wings-accel] Feature '{feat}' installed successfully.\"\n"
            f"        fi\n"
        )
    return "".join(blocks)


def _build_accel_auto_snippet(patch_key: str, features: list[str]) -> str:
    """Build the fault-tolerant accel install shell snippet for auto-generated path.

    版本策略：从 ENGINE_VERSION 环境变量解析，由上层（K8s Deployment）传入。
    install.py 内部有 future_fallback 逻辑，即使传入的版本号不在
    supported_features.json 中也会自动回退到默认版本。
    """
    accel_dir = settings.WINGS_ACCEL_DIR.rstrip("/")
    engine_version = normalize_engine_version()
    all_options = json.dumps({patch_key: {"version": engine_version, "features": features}})
    per_feature_code = _build_per_feature_fallback_code(patch_key, engine_version, features)
    return (
        f"export WINGS_ENGINE_PATCH_OPTIONS='{all_options}'\n"
        f"echo \"[wings-accel] Patch version: {engine_version} (from ENGINE_VERSION)\"\n"
        + f"if [ -f \"{accel_dir}/install.py\" ]; then\n"
        "    echo '[wings-accel] Installing all patches...'\n"
        "    set +e\n"
        f"    python3 {accel_dir}/install.py --features \"$WINGS_ENGINE_PATCH_OPTIONS\"\n"
        "    ACCEL_RC=$?\n"
        "    set -e\n"
        "    if [ $ACCEL_RC -ne 0 ]; then\n"
        "        echo \"[wings-accel] WARNING: "
        "Batch install failed (exit=$ACCEL_RC), trying per-feature fallback...\"\n"
        + per_feature_code
        + "    else\n"
        "        echo '[wings-accel] All patches installed successfully.'\n"
        "    fi\n"
        "else\n"
        f"    echo '[wings-accel] WARNING: {accel_dir}/install.py not found, skipping patch install.'\n"
        "fi\n"
    )


def _build_accel_preamble(engine: str) -> str:
    """若 Accel 加速功能已开启，生成容错的 shell 安装片段；否则返回空字符串。

    安装策略（容错）：
      1. 先尝试批量安装所有特性
      2. 若批量安装失败（install.py exit 非零），回退到逐特性安装
      3. 单个特性安装失败时记录警告并跳过，继续安装其余特性
      4. 无论安装结果如何，始终继续拉起引擎服务
    """
    if not settings.ENABLE_ACCEL:
        logger.debug("Accel disabled: skipping WINGS_ENGINE_PATCH_OPTIONS injection")
        return ""

    # ── 路径 A：用户直接通过环境变量覆盖 ──
    user_override = _validate_accel_user_override()
    if user_override:
        logger.info("Accel: using user-provided WINGS_ENGINE_PATCH_OPTIONS (fault-tolerant)")
        return _build_accel_user_override_snippet(_shell_escape_single_quote(user_override))

    # ── 路径 B：根据特性开关自动构建 ──
    patch_key = _ENGINE_PATCH_KEY_MAP.get(engine)
    if not patch_key:
        logger.warning("Engine '%s' has no known accel patch mapping; skipping.", engine)
        return ""

    features = _collect_enabled_features()
    if not features:
        logger.info("Accel enabled but no advanced features active; skipping patch injection")
        return ""

    logger.info(
        "Accel enabled (fault-tolerant): injecting %d features for engine '%s'",
        len(features), engine,
    )
    return _build_accel_auto_snippet(patch_key, features)


def _is_env_override_file(path: Path) -> bool:
    """Return True if *path* is a valid env-override file (not hidden, not README)."""
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.name.upper() != "README.MD"
    )


def _parse_env_file(fpath: Path) -> list[str]:
    """Parse a KEY=VALUE .env file and return a list of 'export KEY=VALUE' shell lines."""
    export_lines: list[str] = []
    try:
        content = fpath.read_text(encoding="utf-8")
        for lineno, raw_line in enumerate(content.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                logger.warning(
                    "Skipping invalid line %d in %s: no '=' found",
                    lineno, fpath.name,
                )
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 去掉可选的引号包裹
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            export_lines.append(f"export {key}={shlex.quote(value)}")
            logger.debug("  env: %s=%s", key, value)
    except Exception as e:
        logger.error("Failed to parse env file %s: %s", fpath, e)
    return export_lines


def _build_env_overrides_preamble() -> str:
    """读取 env_overrides 目录下的 .env/.sh 文件，生成注入到 start_command.sh 的环境变量前置片段。

    - .env 文件: 逐行解析 KEY=VALUE，生成 export 语句
    - .sh 文件: 通过 source 命令执行

    文件按名称字母序排列，隐藏文件（.开头）和 README 被忽略。
    """
    env_dir = Path(settings.ENV_OVERRIDES_DIR)
    if not env_dir.is_absolute():
        # 相对路径基于工作目录（通常是 /opt/wings-control/wings_control）
        env_dir = Path(os.getcwd()) / env_dir

    if not env_dir.is_dir():
        logger.debug("env_overrides directory not found: %s, skipping", env_dir)
        return ""

    lines: list[str] = []
    files = sorted(f for f in env_dir.iterdir() if _is_env_override_file(f))

    if not files:
        logger.debug("No env override files in %s", env_dir)
        return ""

    lines.append("# --- wings: user env overrides ---")
    for fpath in files:
        suffix = fpath.suffix.lower()
        logger.info("Loading env override: %s", fpath.name)

        if suffix == ".env":
            export_lines = _parse_env_file(fpath)
            lines.extend(export_lines)

        elif suffix == ".sh":
            # shell 脚本直接 source
            lines.append(f"source {shlex.quote(str(fpath))}")

        else:
            logger.debug("Ignoring unsupported file type: %s", fpath.name)

    if len(lines) <= 1:  # 只有注释头
        return ""

    lines.append("# --- end env overrides ---\n")
    preamble = "\n".join(lines) + "\n"
    logger.info("Injecting %d env override entries into start_command.sh", len(lines) - 2)
    return preamble


def _build_faulthandler_patch_preamble(engine: str) -> str:
    """为 SGLang 引擎注入 faulthandler.enable() 安全补丁。

    SGLang ≤ 0.5.10 的 scheduler.py 无保护地调用 ``faulthandler.enable()``，
    在 K8s 容器中因 /dev/shm (tmpfs) 计入 cgroup 内存限制，可能触发
    ``OSError: [Errno 12] Cannot allocate memory``。

    本函数通过 ``sitecustomize.py`` 注入 monkey-patch，用 try/except
    包裹原始 ``faulthandler.enable``，使其在 OOM 时静默降级而非崩溃。
    仅对 SGLang 引擎生效。

    Args:
        engine: 引擎类型

    Returns:
        str: shell 脚本片段；非 sglang 引擎返回空字符串。
    """
    if engine != "sglang":
        return ""

    patch_dir = "/tmp/wings_sitecustomize"
    # sitecustomize.py 在 Python 解释器启动时自动加载（早于任何用户代码），
    # 因此能在 SGLang import 链之前完成 monkey-patch。
    return (
        f"# --- wings: SGLang faulthandler.enable() OOM workaround ---\n"
        f"mkdir -p {patch_dir}\n"
        f"cat > {patch_dir}/sitecustomize.py << 'WINGS_FAULTHANDLER_PATCH'\n"
        f"import faulthandler as _fh\n"
        f"_original_enable = _fh.enable\n"
        f"def _safe_enable(*args, **kwargs):\n"
        f"    try:\n"
        f"        return _original_enable(*args, **kwargs)\n"
        f"    except OSError:\n"
        f"        pass  # /dev/shm tmpfs counted against cgroup memory limit\n"
        f"_fh.enable = _safe_enable\n"
        f"WINGS_FAULTHANDLER_PATCH\n"
        f'export PYTHONPATH="{patch_dir}:${{PYTHONPATH:-}}"\n'
        f"echo \"[wings] Injected faulthandler.enable() OOM patch for SGLang\"\n"
        f"# --- end faulthandler patch ---\n"
    )


def _build_analyzer_preamble(engine: str, merged: dict, hardware: dict) -> str:
    """生成 log_analyzer 进度监控的 shell 片段（仅 master 节点且非 mindie 引擎）。
    
    Args:
        engine: 引擎类型（vllm/vllm_ascend/sglang/mindie 等）
        merged: 合并后的参数字典
        hardware: 硬件探测结果
        
    Returns:
        str: log_analyzer 启动脚本片段。以下情况返回空字符串：
             - worker 节点 (node_rank > 0)
             - mindie 引擎（不支持 log_analyzer 进度条）
    """
    # mindie 引擎不支持 log_analyzer 进度条功能
    if engine == "mindie":
        logger.info("Skipped log_analyzer for mindie engine (not supported)")
        return ""
    
    is_distributed = merged.get("distributed", False)
    node_rank = merged.get("node_rank", 0)
    
    if not is_distributed or node_rank == 0:
        # 构建分析器配置
        analyzer_config = {
            "engine": engine,
            "deployment_mode": "distributed" if is_distributed else "single",
            "hardware": hardware.get("device", "nvidia"),
            "nnodes": merged.get("nnodes", 1),
            "node_rank": node_rank,
            "distributed_backend": merged.get("distributed_executor_backend", "ray"),
            "tensor_parallel_size": merged.get("device_count", 1),
            "model_name": merged.get("model_name", ""),
            "model_path": merged.get("model_path", ""),
            "backend_port": merged.get("port", 17000)
        }

        analyzer_preamble = f"""
# --- log_analyzer: 启动部署进度监控（仅master节点） ---
# 清空旧的日志文件，确保 log_analyzer 只分析新的日志（避免残留内容触发误判）
rm -f /var/log/wings/engine.log
rm -f /var/log/wings/engine-full.log
rm -f {settings.PROGRESS_FILE}

# 记录脚本开始时间（用于计算耗时）
SCRIPT_START_EPOCH=$(date +%s)

ANALYZER_CONFIG='{_shell_escape_single_quote(json.dumps(analyzer_config))}'
echo "[log_analyzer] 配置信息: $ANALYZER_CONFIG"

# 启动日志分析器（后台）
# 使用模块方式运行，Python会自动使用pyc文件（生产环境）
cd {settings.SHARED_VOLUME_PATH} && python3 -m log_analyzer.log_analyzer \\
    --config "$ANALYZER_CONFIG" \\
    --log-file /var/log/wings/engine.log \\
    --progress-file {settings.PROGRESS_FILE} \\
    --accel-file {settings.ACCEL_FILE} &
LOG_ANALYZER_PID=$!
echo "[log_analyzer] 分析器PID: $LOG_ANALYZER_PID"

# 注册清理函数（等待分析器完全退出）
cleanup_analyzer() {{
    local exit_code=$?
    echo "[log_analyzer] 停止分析器..."
    if [ -n "$LOG_ANALYZER_PID" ]; then
        kill $LOG_ANALYZER_PID 2>/dev/null || true
        # 等待分析器进程完全退出，确保完成收尾工作
        wait $LOG_ANALYZER_PID 2>/dev/null || true
    fi

    if [ -n "${{ENGINE_PID:-}}" ]; then
        echo "[cleanup] 发送 SIGTERM 给引擎进程..."
        kill -TERM "$ENGINE_PID" 2>/dev/null || true
    else
        # ENGINE_PID 未设置说明引擎启动前脚本就失败了（如 ray: command not found）
        # 写入失败进度，让上层感知到部署失败
        if [ "$exit_code" -ne 0 ]; then
            echo "[cleanup] 引擎启动前脚本异常退出，退出码: $exit_code"
            local curr_time
            curr_time=$(date -Iseconds)
            local start_time
            start_time=$(date -Iseconds -d "@${{SCRIPT_START_EPOCH}}")
            local elapsed
            elapsed=$(( $(date +%s) - SCRIPT_START_EPOCH ))
            cat >> "{settings.PROGRESS_FILE}" <<EARLY_FAIL_EOF
{{"progress": 0, "phase_code": "script_error", "phase_name": "启动脚本执行失败", "status": "failed", "key_log": "引擎启动前脚本异常退出，退出码: $exit_code", "curr_time": "$curr_time", "start_time": "$start_time", "elapsed_time_s": $elapsed}}
EARLY_FAIL_EOF
        fi
    fi
}}
trap cleanup_analyzer EXIT  SIGTERM SIGINT

"""
        logger.info("Injected log_analyzer for master node (node_rank=%d)", node_rank)
        return analyzer_preamble
    else:
        # Worker节点：不运行分析器，但确保共享卷目录存在
        analyzer_preamble = """
# --- log_analyzer: Worker节点不运行分析器 ---
echo "[log_analyzer] Worker节点(node_rank > 0)，跳过分析器启动"
# 确保共享卷目录存在
mkdir -p /shared-volume

"""
        logger.info("Skipped log_analyzer for worker node (node_rank=%d)", node_rank)
        return analyzer_preamble


# ── 高级特性回退策略 ──
# 启用高级特性（投机解码/KV稀疏/KV卸载）时，若引擎崩溃则无条件禁用所有高级特性重试一次。
# 采用一刀切策略：不区分启动阶段或运行阶段，崩溃即回退。
# 后续打补丁机制会通过特性状态码实现更精细的回退控制。


def _has_advanced_features(merged: dict) -> bool:
    """判断是否启用了任何高级特性（投机解码、KV 稀疏、KV 卸载）。"""
    if merged.get("enable_speculative_decode"):
        return True
    if merged.get("enable_sparse"):
        return True
    if os.getenv("LMCACHE_OFFLOAD", "").strip().lower() == "true":
        return True
    return False


def _collect_active_feature_names(merged: dict) -> list[str]:
    """收集当前激活的高级特性名称列表（用于日志）。"""
    names: list[str] = []
    if merged.get("enable_speculative_decode"):
        names.append("speculative_decode")
    if merged.get("enable_sparse"):
        names.append("sparse_kv")
    if os.getenv("LMCACHE_OFFLOAD", "").strip().lower() == "true":
        names.append("lmcache_offload")
    return names


def _build_monitor_script(
    fallback_cmd: str = "",
    retry_cmd: str = "",
    active_features: str = "",
) -> str:
    """生成引擎进程等待和异常处理的 shell 片段。

    Args:
        fallback_cmd:    当高级特性导致引擎快速失败时的回退启动命令（含 & 和 ENGINE_PID 赋值）。
                         如果为空字符串，则不生成高级特性回退逻辑。
        retry_cmd:       当默认模式引擎崩溃时的重试启动命令（与原始命令相同）。
                         如果为空字符串，则不生成重试逻辑。优先级低于 fallback_cmd。
        active_features: 当前激活的高级特性名称（逗号分隔），用于日志。

    Returns:
        str: 进程监控脚本片段
    """
    progress_file = settings.PROGRESS_FILE

    # ── 公共片段：清理 analyzer + 写失败进度 ──
    cleanup_analyzer = (
        '  echo "[引擎] 停止日志解析进程..."\n'
        '  [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true\n'
        '  trap - EXIT'
    )
    write_progress = (
        '  CURR_TIME=$(date -Iseconds)\n'
        '  START_TIME=$(date -Iseconds -d "@${SCRIPT_START_EPOCH}")\n'
        '  ELAPSED_TIME=$(( $(date +%s) - SCRIPT_START_EPOCH ))\n'
        '\n'
        f'  cat >> "{progress_file}" <<EOF\n'
        '{"progress": 0, "phase_code": "engine_crash", "phase_name": "引擎进程异常退出", '
        '"status": "failed", "key_log": "引擎进程异常退出，退出码: $EXIT_CODE", '
        '"curr_time": "$CURR_TIME", "start_time": "$START_TIME", "elapsed_time_s": $ELAPSED_TIME}\n'
        'EOF'
    )

    if not fallback_cmd and not retry_cmd:
        # ── 基础版：无回退 / 无重试 ──
        return f"""
# --- 引擎进程等待和异常处理 ---
if wait "$ENGINE_PID"; then
  echo "[引擎] 引擎进程正常退出"
{cleanup_analyzer}
else
  EXIT_CODE=$?
  echo "[引擎] 引擎进程异常退出，退出码: $EXIT_CODE"

{write_progress}

{cleanup_analyzer}

  exit "$EXIT_CODE"
fi

"""

    if not fallback_cmd and retry_cmd:
        # ── 默认模式重试版：崩溃后用相同参数重试一次 ──
        indented_retry = "\n".join(
            "    " + line if line.strip() else line
            for line in retry_cmd.rstrip("\n").split("\n")
        )
        cleanup_4 = cleanup_analyzer.replace("  ", "      ")
        write_progress_4 = write_progress.replace("  ", "      ")

        return f"""
# --- Engine process wait and exception handling (with crash retry) ---
echo "[Engine] Engine process monitor started, PID=$ENGINE_PID"
if wait "$ENGINE_PID"; then
  echo "[Engine] Engine process exited normally"
{cleanup_analyzer}
else
  EXIT_CODE=$?
  ENGINE_DURATION=$(( $(date +%s) - ENGINE_START_EPOCH ))
  echo "[Engine] Engine process exited abnormally, exit_code=$EXIT_CODE, runtime=${{ENGINE_DURATION}}s"
  echo "[Engine] ┌── Engine Crash Retry ──"
  echo "[Engine] │ Reason: Engine crashed (exit_code=$EXIT_CODE, runtime=${{ENGINE_DURATION}}s)"
  echo "[Engine] │ Action: Retrying engine startup with same parameters (attempt 2/2)"
  echo "[Engine] └── Retry command about to execute..."
  echo "[Engine] Waiting 5s for port release before retry..."
  sleep 5
  ENGINE_START_EPOCH=$(date +%s)
{indented_retry}
  echo "[Engine] Retry engine started, waiting for process exit..."
  if wait "$ENGINE_PID"; then
    echo "[Engine] Engine process exited normally (retry mode)"
{cleanup_4}
  else
    EXIT_CODE=$?
    echo "[Engine] Retry also failed, exit_code=$EXIT_CODE — unrecoverable"

{write_progress_4}

{cleanup_4}

    exit "$EXIT_CODE"
  fi
fi

"""

    # ── 增强版：高级特性快速失败回退 ──
    # 缩进 fallback_cmd 以匹配 if 块内的层级
    indented_fallback = "\n".join(
        "    " + line if line.strip() else line
        for line in fallback_cmd.rstrip("\n").split("\n")
    )
    # 缩进 cleanup 和 progress 到回退 if/else 内部
    cleanup_4 = cleanup_analyzer.replace("  ", "    ")  # 4-space indent (回退逻辑减少了一层if嵌套)
    write_progress_4 = write_progress.replace("  ", "    ")
    feat_label = active_features or "advanced_features"

    return f"""
# --- Engine process wait and exception handling (with advanced feature fallback) ---
echo "[AdvFeature] Engine process monitor started, PID=$ENGINE_PID"
echo "[AdvFeature] Active advanced features: {feat_label}"
if wait "$ENGINE_PID"; then
  echo "[Engine] Engine process exited normally"
{cleanup_analyzer}
else
  EXIT_CODE=$?
  ENGINE_DURATION=$(( $(date +%s) - ENGINE_START_EPOCH ))
  echo "[Engine] Engine process exited abnormally, exit_code=$EXIT_CODE, runtime=${{ENGINE_DURATION}}s"

  # 一刀切策略：高级特性启用时崩溃 → 无条件禁用所有高级特性重试一次
  echo "[AdvFeature] ┌── Advanced Feature Fallback Triggered ──"
  echo "[AdvFeature] │ Reason: Engine crashed (exit_code=$EXIT_CODE, runtime=${{ENGINE_DURATION}}s)"
  echo "[AdvFeature] │ Features disabled: {feat_label}"
  echo "[AdvFeature] │ Action: Restarting engine without advanced features"
  echo "[AdvFeature] └── Fallback command about to execute..."
  echo "[Engine] Falling back to basic mode (disabled: {feat_label})..."
  echo "[Engine] Waiting 5s for port release before restart..."
  sleep 5
  ENGINE_START_EPOCH=$(date +%s)
{indented_fallback}
  echo "[AdvFeature] Fallback-mode engine started, waiting for process exit..."
  if wait "$ENGINE_PID"; then
    echo "[Engine] Engine process exited normally (fallback mode)"
    echo "[AdvFeature] Fallback-mode engine exited normally"
{cleanup_4}
  else
    EXIT_CODE=$?
    echo "[Engine] Fallback mode also exited abnormally, exit_code=$EXIT_CODE"
    echo "[AdvFeature] ✗ Fallback mode also failed, exit_code=$EXIT_CODE — unrecoverable"

{write_progress_4}

{cleanup_4}

    exit "$EXIT_CODE"
  fi
fi

"""


def _strip_exec_and_backgroundify(script_body: str) -> str:
    """将引擎脚本从 'exec cmd' 格式转换为 'cmd &' 后台运行格式。

    逐行从末尾扫描，找到最后一个非空行：
    - 若以 'exec ' 开头：剔除 exec 前缀并追加 ' &'
    - 否则：直接追加 ' &'
    """
    lines = script_body.rstrip("\n").split("\n")
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if stripped.startswith("exec "):
            indent = lines[i][: len(lines[i]) - len(stripped)]
            lines[i] = indent + stripped[5:] + " &"
            break
        if stripped:
            lines[i] = lines[i] + " &"
            break
    return "\n".join(lines) + "\n"


def _log_advanced_feature_config(
    engine: str, merged: dict, has_advanced_feature: bool,
) -> None:
    """记录所有高级特性（投机解码 / KV 稀疏 / KV 卸载）的配置日志。"""
    if not has_advanced_feature:
        logger.info("[AdvFeature] No advanced features enabled")
        return
    feature_names = _collect_active_feature_names(merged)
    logger.info("[AdvFeature] ┌── Advanced Feature Config ──")
    logger.info("[AdvFeature] │ engine = %s", engine)
    logger.info("[AdvFeature] │ active features = %s", ", ".join(feature_names))
    # 投机解码
    if merged.get("enable_speculative_decode"):
        spec_model_path = merged.get("speculative_decode_model_path", "")
        logger.info("[AdvFeature] │ [speculative_decode]")
        logger.info("[AdvFeature] │   model_path = %s",
                    spec_model_path or "(none, using auto strategy)")
        logger.info("[AdvFeature] │   token_range = %s",
                    merged.get("speculative_token_range", "") or "(not set)")
        logger.info("[AdvFeature] │   draft_confidence = %s",
                    merged.get("draft_confidence_threshold", 0.0))
    # KV 稀疏
    if merged.get("enable_sparse"):
        logger.info("[AdvFeature] │ [sparse_kv]")
        logger.info("[AdvFeature] │   total_budget = %s", merged.get("total_budget", 0))
        logger.info("[AdvFeature] │   lc_sparse_threshold = %s",
                    merged.get("lc_sparse_threshold", 0))
        logger.info("[AdvFeature] │   local_kvstore_capacity = %s",
                    merged.get("local_kvstore_capacity", 0))
    # KV 卸载
    if os.getenv("LMCACHE_OFFLOAD", "").strip().lower() == "true":
        logger.info("[AdvFeature] │ [lmcache_offload]")
        logger.info("[AdvFeature] │   kv_transfer_config = %s",
                    merged.get("kv_transfer_config", "(not set)"))
    logger.info("[AdvFeature] │ 回退策略 = 一刀切 (崩溃即回退)")
    logger.info("[AdvFeature] └── Advanced features injected into engine start command")


def _build_advanced_feature_fallback_cmd(merged: dict) -> str:
    """生成禁用所有高级特性的引擎回退启动命令。

    禁用的特性：
    - 投机解码（--speculative-config）
    - KV 稀疏（--sparse-config + sparse --kv-transfer-config）
    - KV 卸载（LMCache --kv-transfer-config）

    Returns:
        回退命令字符串（已后台化 + ENGINE_PID 记录）。
    """
    feature_names = _collect_active_feature_names(merged)
    feature_label = ", ".join(feature_names)
    merged_no_features = dict(merged)
    merged_no_features["enable_speculative_decode"] = False
    merged_no_features["enable_sparse"] = False
    # kv_transfer_config 由 config_loader._set_kv_cache_config() 注入到
    # engine_config 嵌套字典中，需要从正确的层级移除。
    # 使用浅拷贝 engine_config 避免污染原始 merged 数据。
    # （PD 分离的 kv_transfer_config 也会一并移除，但在崩溃回退场景下可接受）
    if os.getenv("LMCACHE_OFFLOAD", "").strip().lower() == "true":
        original_ec = merged_no_features.get("engine_config", {})
        if isinstance(original_ec, dict) and "kv_transfer_config" in original_ec:
            ec_copy = dict(original_ec)
            ec_copy.pop("kv_transfer_config", None)
            merged_no_features["engine_config"] = ec_copy
            logger.info(
                "[AdvFeature] Removed kv_transfer_config from engine_config "
                "for fallback (LMCache Offload was enabled)"
            )
    fallback_body = start_engine_service(merged_no_features)
    fallback_cmd = _strip_exec_and_backgroundify(fallback_body)
    fallback_cmd += "ENGINE_PID=$!\n"
    fallback_cmd += (
        f'echo "[Engine] Engine PID: $ENGINE_PID '
        f'(advanced features disabled: {feature_label}, fallback mode)"\n'
    )
    logger.info(
        "[AdvFeature] Generated fallback command (disabled: %s) for fast-fail recovery",
        feature_label,
    )
    return fallback_cmd


def _build_pid_tracked_script(script_body: str, has_advanced_feature: bool) -> str:
    """将引擎启动脚本转换为后台模式并注入 ENGINE_PID / ENGINE_START_EPOCH 跟踪。

    始终注入 ENGINE_START_EPOCH，用于高级特性快速失败检测和默认模式崩溃重试。

    Args:
        script_body:          原始引擎启动脚本体
        has_advanced_feature: 是否启用了高级特性（决定日志标签）

    Returns:
        修改后的脚本体字符串
    """
    body = _strip_exec_and_backgroundify(script_body)
    body += "ENGINE_PID=$!\n"
    if has_advanced_feature:
        body += 'echo "[Engine] Engine PID: $ENGINE_PID (advanced features enabled)"\n'
    else:
        body += 'echo "[Engine] Engine PID: $ENGINE_PID"\n'
    # 始终注入 ENGINE_START_EPOCH，用于崩溃重试的运行时长统计
    body = "ENGINE_START_EPOCH=$(date +%s)\n" + body
    return body


def _build_engine_retry_cmd(merged: dict) -> str:
    """生成引擎重试启动命令（与原始命令相同，用于默认参数崩溃后的一次重试）。

    Returns:
        重试命令字符串（已后台化 + ENGINE_PID 记录）。
    """
    retry_body = start_engine_service(merged)
    retry_cmd = _strip_exec_and_backgroundify(retry_body)
    retry_cmd += "ENGINE_PID=$!\n"
    retry_cmd += 'echo "[Engine] Engine PID: $ENGINE_PID (retry mode)"\n'
    logger.info("[Engine] Generated retry command for default-mode crash recovery")
    return retry_cmd


def build_launcher_plan(launch_args: LaunchArgs, port_plan: PortPlan) -> LauncherPlan:
    """根据启动参数、硬件信息和端口规划生成完整启动脚本。

    执行流程：
    1. 调用 detect_hardware() 获取硬件环境（设备类型、数量、型号）
    2. 调用 load_and_merge_configs() 多层配置合并
    3. 用显式参数覆盖合并结果（engine/model_name/model_path 等）
    4. 注入分布式信息（nnodes/node_rank/head_node_addr）
    5. 根据 node_rank 决定是否注入 host/port
    6. 调用 start_engine_service() 分发给具体 adapter 生成脚本
    7. 添加 shebang + set -euo pipefail 包装成安全脚本

    Args:
        launch_args: 标准化的启动参数（来自 parse_launch_args）
        port_plan:   三层端口分配方案（来自 derive_port_plan）

    Returns:
        LauncherPlan: 包含完整 shell 脚本、合并参数和硬件信息
    """
    hardware = detect_hardware()
    merged = _prepare_merged_params(launch_args, port_plan, hardware)
    # engine 已在 load_and_merge_configs 中经过 _auto_select_engine 的
    # 自动选择、校验和升级（如 vllm → vllm_ascend），不可用原始值覆盖。
    engine = merged.get("engine", launch_args.engine)
    has_advanced_feature = _has_advanced_features(merged)
    active_feature_names = _collect_active_feature_names(merged)
    active_features_label = ", ".join(active_feature_names)

    _log_advanced_feature_config(engine, merged, has_advanced_feature)
    fallback_cmd = _build_advanced_feature_fallback_cmd(merged) if has_advanced_feature else ""
    retry_cmd = _build_engine_retry_cmd(merged) if not has_advanced_feature else ""
    script_body = _build_pid_tracked_script(start_engine_service(merged), has_advanced_feature)

    analyzer_preamble = _build_analyzer_preamble(engine, merged, hardware)
    faulthandler_patch = _build_faulthandler_patch_preamble(engine)
    triton_patch = build_triton_patch_preamble(engine)
    accel_preamble = _build_accel_preamble(engine)
    env_overrides = _build_env_overrides_preamble()
    monitor_script = _build_monitor_script(
        fallback_cmd=fallback_cmd, retry_cmd=retry_cmd,
        active_features=active_features_label,
    )

    command = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "mkdir -p /var/log/wings\n"
        # Prometheus multi-process metrics directory: ensure a clean dir
        # on every engine restart so stale worker .db files don't pollute
        # /metrics output.  Located under the shared log volume so that
        # both the engine container and the sidecar proxy can reach it.
        "rm -rf /var/log/wings/prometheus_multiproc\n"
        "mkdir -p /var/log/wings/prometheus_multiproc\n"
        "export PROMETHEUS_MULTIPROC_DIR=/var/log/wings/prometheus_multiproc\n"
        + analyzer_preamble
        # Disable Python stdout full-buffering so that engine ready
        # messages (e.g. "Starting vLLM server on") reach engine.log
        # immediately rather than being stuck in an 8 KB buffer.
        + "export PYTHONUNBUFFERED=1\n"
        # Filter engine noise from console output and engine.log:
        #   1) /health and /metrics access logs (uvicorn)
        #   2) "Prefill batch" / "Decode batch" scheduler metrics (SGLang)
        # Complete unfiltered logs are saved to engine-full.log for debugging.
        + "exec > >(tee -a /var/log/wings/engine-full.log"
        " | grep --line-buffered -vE"
        " '\"GET\\s+/(health|metrics)\\s|\\b(Prefill|Decode) batch\\b'"
        " | tee -a /var/log/wings/engine.log) 2>&1\n"
        + faulthandler_patch
        + triton_patch
        + env_overrides
        + accel_preamble
        + script_body
        + monitor_script
    )
    logger.info("Generated start_command.sh (%d bytes)", len(command))
    logger.debug(
        "start_command.sh content:\n"
        "╔══════════════ start_command.sh ══════════════╗\n%s\n"
        "╚══════════════ end start_command.sh ══════════╝",
        command,
    )
    return LauncherPlan(command=command, merged_params=merged, hardware_env=hardware)

