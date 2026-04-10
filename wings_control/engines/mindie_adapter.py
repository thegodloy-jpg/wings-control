# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""
MindIE 引擎适配器。

在 sidecar launcher 模式下，本模块执行以下工作：
  1. 组装需要应用到 MindIE config.json 的配置覆盖；
  2. 生成 bash 脚本体 (start_command.sh)，包含：
       a. 加载 Ascend CANN / MindIE 环境脚本
       b. 注入分布式环境变量 (HCCL / MASTER_ADDR等) [多节点时]
       c. 通过内联 Python 片段合并更新 conf/config.json
       d. 启动 mindieservice_daemon
"""

import json
import logging
import os
import shlex
import shutil
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# 模块根目录：用于定位本地开发环境的环境脚本
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# MindIE 服务路径常量（可通过环境变量覆盖）
#
# MINDIE_WORK_DIR:    MindIE 服务工作目录，包含 bin/、conf/ 等子目录
# MINDIE_CONFIG_PATH: MindIE 配置文件路径，会在启动前被合并更新
#
# 环境变量覆盖：
#   - MINDIE_WORK_DIR:   自定义工作目录
#   - MINDIE_CONFIG_PATH: 自定义配置文件路径
# =============================================================================
MINDIE_WORK_DIR: str = os.getenv(
    "MINDIE_WORK_DIR",
    "/usr/local/Ascend/mindie/latest/mindie-service"
)
MINDIE_CONFIG_PATH: str = os.getenv(
    "MINDIE_CONFIG_PATH",
    os.path.join(MINDIE_WORK_DIR, "conf/config.json")
)

# 默认端口配置
DEFAULT_SERVER_PORT = 18000              # MindIE HTTP API 端口
DEFAULT_MINDIE_MASTER_PORT = int(os.getenv("MINDIE_MASTER_PORT", "27070"))  # 分布式主节点端口
DEFAULT_HCCL_IP_EXCHANGE_PORT = int(os.getenv("HCCL_IP_EXCHANGE_PORT", "27071"))  # hccnX IP 交换端口


# =============================================================================
# 内部函数：构建环境设置命令列表
# =============================================================================


def _split_node_ips(node_ips: str | None) -> List[str]:
    """Normalize a node IP CSV string into a clean list."""
    if not node_ips:
        return []
    return [ip.strip() for ip in node_ips.split(",") if ip.strip()]


def _resolve_distributed_topology(params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve and validate the MindIE multi-node topology."""
    nnodes = int(params.get("nnodes", 1) or 1)
    node_rank = int(params.get("node_rank", 0) or 0)
    device_count = int(params.get("device_count", 1) or 1)
    node_ips_list = _split_node_ips(params.get("node_ips"))
    external_rank_table = os.getenv("RANK_TABLE_PATH", "").strip()

    if nnodes > 1 and node_ips_list and len(node_ips_list) != nnodes:
        raise ValueError(
            f"MindIE distributed topology mismatch: nnodes={nnodes}, "
            f"but node_ips has {len(node_ips_list)} entries"
        )

    if nnodes > 1 and not node_ips_list and not external_rank_table:
        raise ValueError(
            "MindIE multi-node startup requires node_ips or a valid RANK_TABLE_PATH"
        )

    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError(
            f"Invalid MindIE node_rank={node_rank}; expected range [0, {nnodes - 1}]"
        )

    global_world_size = params.get("worldSize")
    if global_world_size is None:
        resolved_nodes = len(node_ips_list) if node_ips_list else nnodes
        global_world_size = device_count * resolved_nodes
    global_world_size = int(global_world_size)

    if nnodes > 1 and global_world_size % nnodes != 0:
        raise ValueError(
            f"MindIE worldSize={global_world_size} must be divisible by nnodes={nnodes}"
        )

    # 交叉校验: 显式指定的 worldSize 应等于 device_count × nnodes
    expected_world_size = device_count * (len(node_ips_list) if node_ips_list else nnodes)
    if params.get("worldSize") is not None and global_world_size != expected_world_size:
        logger.warning(
            "[mindie] worldSize=%d does not match device_count(%d) × nnodes(%d) = %d. "
            "Rank table will have %d ranks but HCCL expects %d. "
            "Ensure this is intentional (e.g. PP parallelism).",
            global_world_size, device_count, nnodes, expected_world_size,
            expected_world_size, global_world_size,
        )

    return {
        "nnodes": nnodes,
        "node_rank": node_rank,
        "device_count": device_count,
        "node_ips_list": node_ips_list,
        "external_rank_table": external_rank_table,
        "global_world_size": global_world_size,
    }


def _build_ascend_env_source_commands() -> List[str]:
    """生成 Ascend 容器标准路径环境脚本的 source 命令列表（含 set +u/-u 守卫）。

    MindIE 容器中的环境脚本可能引用未绑定变量（如 ZSH_VERSION），
    因此在 set +u / set -u 守卫块内加载。同时追加驱动库路径和 GRPC 策略。
    """
    cmds: List[str] = [
        "# set +u: Ascend env scripts may reference unbound vars (e.g. ZSH_VERSION)",
        "set +u",
    ]
    cmds.append(
        "[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] "
        "&& source /usr/local/Ascend/ascend-toolkit/set_env.sh "
        "|| echo 'WARN: ascend-toolkit/set_env.sh not found'"
    )
    cmds.append(
        "[ -f /usr/local/Ascend/mindie/set_env.sh ] "
        "&& source /usr/local/Ascend/mindie/set_env.sh "
        "|| echo 'WARN: mindie/set_env.sh not found'"
    )
    cmds.append(
        "[ -f /usr/local/Ascend/atb-models/set_env.sh ] "
        "&& source /usr/local/Ascend/atb-models/set_env.sh "
        "|| echo 'WARN: atb-models/set_env.sh not found'"
    )
    cmds.append(
        "[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] "
        "&& source /usr/local/Ascend/nnal/atb/set_env.sh "
        "|| echo 'WARN: nnal/atb/set_env.sh not found'"
    )
    cmds.append("set -u")
    cmds.append(
        "export LD_LIBRARY_PATH=\"/usr/local/Ascend/driver/lib64/driver"
        ":/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}\""
    )
    cmds.append("export GRPC_POLL_STRATEGY=poll")
    return cmds


def _build_env_commands(params: Dict[str, Any]) -> List[str]:
    """构建 MindIE 环境初始化命令列表（CANN 工具包 + MindIE set_env.sh）。

    优先加载本地开发环境脚本；容器环境下调用 _build_ascend_env_source_commands()。
    最终追加 NPU_MEMORY_FRACTION（如已配置）和 ASCEND 调试相关环境变量。

    Args:
        params: 参数字典，可包含 engine_config.npu_memory_fraction

    Returns:
        List[str]: shell 命令列表，每个元素是一条环境设置命令
    """
    cmds: List[str] = []

    # Dev environment: prefer local wings project env script if available
    env_script = os.path.join(root_dir, "wings", "config", "set_mindie_single_env.sh")
    if os.path.exists(env_script):
        cmds.append(f"source {env_script}")
    else:
        cmds.extend(_build_ascend_env_source_commands())

    engine_config = params.get("engine_config") or {}
    npu_memory_fraction = engine_config.get("npu_memory_fraction")
    if npu_memory_fraction is not None:
        # Validate and quote to prevent shell injection
        try:
            frac_val = float(npu_memory_fraction)
            if not (0.0 < frac_val <= 1.0):
                logger.warning("[mindie] NPU_MEMORY_FRACTION=%s out of range (0,1], ignoring", frac_val)
            else:
                cmds.append(f"export NPU_MEMORY_FRACTION={shlex.quote(str(frac_val))}")
        except (ValueError, TypeError):
            logger.warning("[mindie] Invalid NPU_MEMORY_FRACTION=%r, ignoring", npu_memory_fraction)

    # ── ASCEND debug / HCCL tuning env vars ──
    # ASCEND_GLOBAL_LOG_LEVEL: 0=debug, 1=info, 2=warn, 3=error (default)
    for _env_name, _default in (
        ("ASCEND_GLOBAL_LOG_LEVEL", "1"),
        ("ASCEND_SLOG_PRINT_TO_STDOUT", "0"),
    ):
        cmds.append(f"export {_env_name}={shlex.quote(os.getenv(_env_name, _default))}")

    return cmds


def _resolve_hccl_if_ip(
    node_rank: int,
    node_ips_list: List[str],
    master_addr: str,
) -> str:
    """返回设置 HCCL_IF_IP 的 shell 命令。"""
    if node_rank < len(node_ips_list):
        return f'export HCCL_IF_IP={shlex.quote(node_ips_list[node_rank])}'
    # hostname -i 可能返回多个 IP（多网卡），用 awk 取第一个
    return (
        f"export HCCL_IF_IP=$(hostname -i 2>/dev/null | awk '{{print $1}}' "
        f"|| python3 -c 'import socket; print(socket.gethostbyname(socket.gethostname()))' 2>/dev/null "
        f"|| echo {shlex.quote(master_addr)})"
    )


def _copy_external_rank_table(
    external_path: str,
    shared_path: str,
) -> tuple:
    """策略 1: sidecar 可见外部文件 → 复制到共享卷并返回共享卷路径。

    复制失败时 fallback 到原路径（需要 engine 容器也能访问）。
    """
    logger.info(
        "[mindie] External RANK_TABLE_PATH found in sidecar: %s, "
        "copying to shared volume: %s",
        external_path, shared_path,
    )
    try:
        os.makedirs(os.path.dirname(shared_path), exist_ok=True)
        shutil.copy2(external_path, shared_path)
        os.chmod(shared_path, 0o640)
    except OSError as e:
        logger.error(
            "[mindie] Failed to copy rank table to shared volume: %s", e
        )
        return [
            f"# ── HCCL rank table (external, copy failed: {e}) ──",
            f"chmod 640 {shlex.quote(external_path)} 2>/dev/null || true",
        ], external_path

    return [
        f"# ── HCCL rank table (copied from {external_path} to shared volume) ──",
    ], shared_path


def _build_runtime_rank_table_check(
    external_path: str,
    shared_path: str,
    node_ips_list: List[str],
    master_addr: str,
    device_count: int,
) -> tuple:
    """策略 2: 外部路径已设但 sidecar 不可见 → 生成运行时 if/else 检查脚本。

    engine 执行时尝试从原路径复制到共享卷，失败则使用动态生成的 fallback。
    """
    logger.warning(
        "[mindie] RANK_TABLE_PATH=%s set but file not found in sidecar container. "
        "Will generate runtime check in start_command.sh: "
        "engine will try to copy from original path at execution time.",
        external_path,
    )
    rank_table_nodes = node_ips_list if node_ips_list else [master_addr]
    fallback_cmds = _build_rank_table_commands(
        rank_table_nodes, device_count, "/tmp/hccl_ranktable_fallback.json",
        node_offset=0,
    )
    safe_ext = shlex.quote(external_path)
    safe_shared = shlex.quote(shared_path)

    cmds = [
        f"# ── HCCL rank table (runtime resolve: prefer {external_path}) ──",
        f"if [ -f {safe_ext} ]; then",
        f"    cp {safe_ext} {safe_shared}",
        f"    chmod 640 {safe_shared}",
        f"    echo '[mindie] Copied external rank table to shared volume'",
    ] + [
        "else",
        f"    echo '[mindie] WARNING: {external_path} not found in engine container, "
        f"using dynamically generated fallback'",
    ] + [f"    {line}" for line in fallback_cmds] + [
        f"    cp /tmp/hccl_ranktable_fallback.json {safe_shared}",
        f"    chmod 640 {safe_shared}",
        "fi",
    ]
    return cmds, shared_path


def _resolve_rank_table(
    topology: Dict[str, Any],
    master_addr: str,
    device_count: int,
) -> tuple:
    """确定 rank table 策略并返回 (rank_table_cmds, ranktable_path)。

    三段式策略:
      1. sidecar 能看到外部文件 → 复制到共享卷 (_copy_external_rank_table)
      2. RANK_TABLE_PATH 已设但 sidecar 看不到 → 运行时检查 (_build_runtime_rank_table_check)
      3. 未设置外部路径 → 动态生成 rank table 内联到脚本中

    关键设计: 最终 RANK_TABLE_FILE 始终指向 engine 容器可访问的路径
    （共享卷或 /tmp），避免跨容器路径不可见问题。
    """
    external_rank_table = topology["external_rank_table"]
    node_ips_list = topology["node_ips_list"]

    from config.settings import settings
    # 使用 posixpath 而非 os.path.join，因为生成的路径将在 Linux 容器中使用
    import posixpath
    shared_ranktable = posixpath.join(settings.SHARED_VOLUME_PATH, "hccl_ranktable.json")

    # ── 策略 1: sidecar 容器能直接看到外部文件 ──
    if external_rank_table and os.path.isfile(external_rank_table):
        return _copy_external_rank_table(external_rank_table, shared_ranktable)

    # ── 策略 2: RANK_TABLE_PATH 已设但 sidecar 无法看到文件 ──
    if external_rank_table:
        return _build_runtime_rank_table_check(
            external_rank_table, shared_ranktable,
            node_ips_list, master_addr, device_count,
        )

    # ── 策略 3: 未设置外部路径，动态生成 ──
    rank_table_nodes = node_ips_list if node_ips_list else [master_addr]
    return _build_rank_table_commands(
        rank_table_nodes, device_count, shared_ranktable, node_offset=0,
    ), shared_ranktable


def _build_distributed_env_commands(params: Dict[str, Any]) -> List[str]:
    """构建多节点 MindIE 分布式环境变量设置命令。

    昇腾多节点 HCCL 通信所需的环境变量：
      MASTER_ADDR           - rank-0 节点的 IP 地址 (head_node_addr)
      MASTER_PORT           - 集合通信初始化端口 (默认 27070)
      RANK                  - 当前节点的编号
      WORLD_SIZE            - 总节点数
      HCCL_WHITELIST_DISABLE - 禁用 HCCL 白名单检查 (容器环境必需)
      HCCL_IF_IP            - HCCL 网络接口 IP (自动检测或由 node_ips 提供)
      RANK_TABLE_FILE       - HCCL rank table JSON 文件路径 (MindIE 多节点必需)

    单节点或非分布式模式时返回空列表。

    Args:
        params: 参数字典，包含以下关键字段:
            - distributed:        是否分布式模式
            - nnodes:             总节点数
            - node_rank:          当前节点编号
            - mindie_master_addr: 主节点地址 (可缺省，回退到 head_node_addr)
            - mindie_master_port: 主节点端口
            - device_count:       每节点设备数
            - node_ips:           所有节点 IP 列表 (逗号分隔)

    Returns:
        List[str]: 分布式环境变量设置命令列表

    注意:
        - MindIE 的 worldSize 设为全局 TP 总卡数 (device_count × nnodes)
        - rank table 包含所有节点信息，HCCL 可发现全局 rank 拓扑
        - multiNodesInferEnabled 在多节点时设为 True
    """
    is_distributed = params.get("distributed", False)
    nnodes = params.get("nnodes", 1)
    if not is_distributed or nnodes <= 1:
        return []

    topology = _resolve_distributed_topology(params)
    node_rank = topology["node_rank"]
    master_addr = (
        params.get("mindie_master_addr")
        or params.get("master_ip")
        or params.get("head_node_addr", "127.0.0.1")
    )
    master_port = params.get("mindie_master_port", DEFAULT_MINDIE_MASTER_PORT)
    device_count = topology["device_count"]
    global_world_size = topology["global_world_size"]
    node_ips_list = topology["node_ips_list"]

    hccl_if_ip_cmd = _resolve_hccl_if_ip(node_rank, node_ips_list, master_addr)
    container_ip = node_ips_list[node_rank] if node_rank < len(node_ips_list) else master_addr
    rank_table_cmds, ranktable_path = _resolve_rank_table(topology, master_addr, device_count)

    return [
        "# ── Ascend HCCL distributed env vars (multi-node TP) ──",
        f"export MASTER_ADDR={shlex.quote(master_addr)}",
        f"export MASTER_PORT={shlex.quote(str(master_port))}",
        f"export RANK={node_rank}",
        f"export WORLD_SIZE={global_world_size}",
        f"export MINDIE_MODEL_WORLD_SIZE={global_world_size}",
        "export HCCL_WHITELIST_DISABLE=1",
        hccl_if_ip_cmd,
        f"export HCCL_SOCKET_IFNAME={os.getenv('HCCL_SOCKET_IFNAME', 'eth0')}",
        f"export GLOO_SOCKET_IFNAME={os.getenv('GLOO_SOCKET_IFNAME', 'eth0')}",
        f"export MIES_CONTAINER_IP={shlex.quote(container_ip)}",
        # HCCL timeouts (overridable via container env vars)
        f"export HCCL_CONNECT_TIMEOUT={os.getenv('HCCL_CONNECT_TIMEOUT', '1800')}",
        f"export HCCL_EXEC_TIMEOUT={os.getenv('HCCL_EXEC_TIMEOUT', '7200')}",
    ] + rank_table_cmds + [
        f"export RANK_TABLE_FILE={ranktable_path}",
    ]


def _parse_hccl_device_ips() -> List[List[str]]:
    """解析 HCCL_DEVICE_IPS 环境变量为二维 IP 列表。

    格式: ``"ip0,ip1;ip2,ip3"``（分号分隔节点，逗号分隔设备）

    Returns:
        List[List[str]]: 每节点的设备 IP 列表；未设置时返回空列表

    注意: 此函数仅用于 sidecar 容器端的预校验和日志记录。
    实际 rank table 生成通过 _build_rank_table_commands() 内的
    运行时 Python 脚本在 engine 容器中读取 HCCL_DEVICE_IPS。
    """
    raw = os.environ.get("HCCL_DEVICE_IPS", "")
    if not raw:
        return []
    result: List[List[str]] = []
    for node_part in raw.split(";"):
        ips = [ip.strip() for ip in node_part.split(",") if ip.strip()]
        if ips:
            result.append(ips)
    logger.info("[mindie] HCCL_DEVICE_IPS parsed (sidecar): %s", result)
    return result


def _resolve_device_ip(
    host_ip: str,
    hccl_node_idx: int,
    dev_id: int,
    node_device_ips: List[List[str]],
    is_multinode: bool,
) -> str:
    """解析单个设备的 HCCL IP，找不到时回退到主机 IP。

    Args:
        host_ip:         该节点的主机 IP
        hccl_node_idx:   节点在全局 HCCL 设备列表中的索引
        dev_id:          设备在该节点内的序号
        node_device_ips: 解析后的 HCCL 设备 IP 二维列表
        is_multinode:    是否为多节点部署

    Returns:
        str: 设备 IP 地址
    """
    if hccl_node_idx < len(node_device_ips) and dev_id < len(node_device_ips[hccl_node_idx]):
        return node_device_ips[hccl_node_idx][dev_id]

    if is_multinode:
        logger.error(
            "[mindie] CRITICAL: No HCCL device IP for node=%d dev=%d, "
            "falling back to host IP %s. Multi-node HCCL will likely fail! "
            "Set HCCL_DEVICE_IPS=<rdma_ip_node0>;<rdma_ip_node1>",
            hccl_node_idx, dev_id, host_ip,
        )
    else:
        logger.warning(
            "[mindie] No HCCL device IP for node=%d dev=%d, fallback to host IP %s",
            hccl_node_idx, dev_id, host_ip,
        )
    return host_ip


def _build_server_list(
    node_ips: List[str],
    device_count: int,
    node_device_ips: List[List[str]],
    node_offset: int,
) -> List[Dict[str, Any]]:
    """构建 HCCL rank table 的 server_list。

    为每个节点生成包含设备 rank 信息的 server entry。

    Args:
        node_ips:       参与节点的 IP 列表
        device_count:   每节点的 NPU 设备数
        node_device_ips: 解析后的 HCCL 设备 IP 二维列表
        node_offset:    本节点在全局 HCCL 列表中的起始偏移

    Returns:
        List[Dict]: 每节点对应一个 server entry 字典
    """
    server_list = []
    global_rank = 0
    is_multinode = len(node_ips) > 1
    for local_idx, ip in enumerate(node_ips):
        hccl_node_idx = node_offset + local_idx
        devices = []
        for dev_id in range(device_count):
            device_ip = _resolve_device_ip(ip, hccl_node_idx, dev_id, node_device_ips, is_multinode)
            devices.append({"device_id": str(dev_id), "device_ip": device_ip, "rank_id": str(global_rank)})
            global_rank += 1
        server_list.append({"server_id": ip, "device": devices, "container_ip": ip, "host_nic_ip": ip})
    return server_list


def _build_rank_table_commands(
    node_ips: List[str],
    device_count: int,
    output_path: str,
    node_offset: int = 0,
) -> List[str]:
    """生成写入 HCCL rank table JSON 文件的 shell 命令。

    MindIE 分布式模式需要 HCCL rank table 来确定集群拓扑。
    支持单节点（server_count=1）和多节点 TP（server_count=N）两种模式。

    实现策略:
      - 如果 sidecar 容器中已能读到 HCCL_DEVICE_IPS → 使用静态 heredoc（快速路径）
      - 否则 → 生成运行时 Python 脚本内联在 start_command.sh 中，
        在 engine 容器执行时读取 HCCL_DEVICE_IPS 环境变量。
        这解决了 HCCL_DEVICE_IPS 仅在 engine 容器中设置的跨容器问题。

    Args:
        node_ips:     节点 IP 列表（所有参与节点）
        device_count: 每节点的 NPU 设备数量
        output_path:  rank table JSON 文件输出路径
        node_offset:  本节点在全局 HCCL_DEVICE_IPS 列表中的索引
    """
    sidecar_device_ips = _parse_hccl_device_ips()

    if sidecar_device_ips:
        # ── 快速路径: sidecar 已能读到 HCCL_DEVICE_IPS，用静态 heredoc ──
        server_list = _build_server_list(node_ips, device_count, sidecar_device_ips, node_offset)
        rank_table_json = json.dumps({
            "version": "1.0",
            "server_count": str(len(node_ips)),
            "server_list": server_list,
            "status": "completed",
        }, indent=2, ensure_ascii=False)

        return [
            f"# ── HCCL rank table ({len(node_ips)} nodes, {device_count} devices/node, static) ──",
            f"cat > {output_path} << 'RANK_TABLE_EOF'",
            rank_table_json,
            "RANK_TABLE_EOF",
            f"chmod 640 {output_path}",
        ]

    # ── 运行时路径: 通过 Python 脚本在 engine 容器中动态生成 rank table ──
    # 三级回退: HCCL_DEVICE_IPS 环境变量 → hccnX 网卡自动探测 → Pod IP（仅单节点可靠）
    is_multinode = len(node_ips) > 1
    if is_multinode:
        logger.warning(
            "[mindie] HCCL_DEVICE_IPS not available in sidecar for multi-node (%d nodes). "
            "Runtime rank table will attempt hccnX interface auto-detection + TCP exchange "
            "in engine container. For reliable multi-node HCCL, "
            "set HCCL_DEVICE_IPS=<rdma_ips_node0>;<rdma_ips_node1>",
            len(node_ips),
        )
    else:
        logger.info(
            "[mindie] HCCL_DEVICE_IPS not available in sidecar, "
            "generating runtime rank table script for engine container"
        )
    node_ips_json = json.dumps(node_ips)
    exchange_port = DEFAULT_HCCL_IP_EXCHANGE_PORT

    runtime_script = f"""# ── HCCL rank table ({len(node_ips)} nodes, {device_count} devices/node, runtime) ──
python3 << 'RANK_TABLE_GEN_EOF'
import json, os, sys, subprocess, socket, time

node_ips = {node_ips_json}
device_count = {device_count}
node_offset = {node_offset}
output_path = {json.dumps(output_path)}
nnodes = len(node_ips)
exchange_port = {exchange_port}
node_rank = int(os.environ.get("RANK", "0"))

# ---------------------------------------------------------------------------
# hccnX 网卡自动探测：昇腾 NPU 设备插件通常会在容器中创建 hccn0/hccn1/...
# 这些网卡绑定 RoCE RDMA IP，是 HCCL 跨节点通信的正确 device_ip。
# ---------------------------------------------------------------------------
def detect_hccn_device_ips(dev_count):
    ips = []
    for dev_id in range(dev_count):
        iface = f"hccn{{dev_id}}"
        try:
            result = subprocess.run(
                ["ip", "-4", "-o", "addr", "show", iface],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\\n"):
                if "inet " in line:
                    parts = line.split()
                    idx = parts.index("inet")
                    addr = parts[idx + 1].split("/")[0]
                    ips.append(addr)
                    break
        except Exception as e:
            print(f"[mindie] hccn{{dev_id}} detection failed: {{e}}")
    if len(ips) == dev_count:
        return ips
    print(f"[mindie] hccnX auto-detection incomplete: found {{ips}} for {{dev_count}} devices")
    return []

# ---------------------------------------------------------------------------
# 多节点 TCP IP 交换：各节点探测到本地 hccnX IP 后，通过 Rank 0 汇聚再下发，
# 确保所有节点生成一致的 rank table。
# 协议: Worker → Master: JSON line  {{"rank": N, "ips": [...]}}
#        Master → Worker: JSON line  {{"0": [...], "1": [...]}}
# ---------------------------------------------------------------------------
def exchange_device_ips_tcp(rank, total_nodes, master_addr, port, local_ips, timeout=120):
    if total_nodes <= 1:
        return {{0: local_ips}}

    all_ips = {{}}
    if rank == 0:
        all_ips[0] = local_ips
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(timeout)
        try:
            srv.bind(("0.0.0.0", port))
            srv.listen(total_nodes)
            print(f"[mindie] Rank 0: waiting for {{total_nodes - 1}} worker(s) on port {{port}}...")
            conns = []
            for _ in range(total_nodes - 1):
                conn, addr = srv.accept()
                conn.settimeout(timeout)
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\\n" in data:
                        break
                msg = json.loads(data.decode().strip())
                all_ips[msg["rank"]] = msg["ips"]
                conns.append(conn)
                print(f"[mindie] Rank 0: received device IPs from rank {{msg['rank']}}: {{msg['ips']}}")
            response = json.dumps(all_ips) + "\\n"
            for conn in conns:
                try:
                    conn.sendall(response.encode())
                except Exception as e:
                    print(f"[mindie] Rank 0: send aggregated IPs failed: {{e}}")
                finally:
                    conn.close()
        except Exception as e:
            print(f"[mindie] Rank 0: IP exchange server error: {{e}}")
            return None
        finally:
            srv.close()
    else:
        deadline = time.time() + timeout
        sock = None
        while time.time() < deadline:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(min(10, max(1, deadline - time.time())))
                sock.connect((master_addr, port))
                break
            except (ConnectionRefusedError, OSError):
                if sock:
                    sock.close()
                    sock = None
                time.sleep(2)
        if not sock:
            print(f"[mindie] Rank {{rank}}: failed to connect to master {{master_addr}}:{{port}} for IP exchange")
            return None
        try:
            sock.sendall((json.dumps({{"rank": rank, "ips": local_ips}}) + "\\n").encode())
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            all_ips = json.loads(data.decode().strip())
            all_ips = {{int(k): v for k, v in all_ips.items()}}
        except Exception as e:
            print(f"[mindie] Rank {{rank}}: IP exchange failed: {{e}}")
            return None
        finally:
            sock.close()
    return all_ips

# ── Step 1: 读取 HCCL_DEVICE_IPS 环境变量（优先级最高）──
raw_device_ips = os.environ.get("HCCL_DEVICE_IPS", "")
node_device_ips = []
if raw_device_ips:
    for node_part in raw_device_ips.split(";"):
        ips = [ip.strip() for ip in node_part.split(",") if ip.strip()]
        if ips:
            node_device_ips.append(ips)
    print(f"[mindie] HCCL_DEVICE_IPS from engine env: {{node_device_ips}}")

# ── Step 2: hccnX 网卡自动探测 + 多节点 TCP 交换 ──
if not node_device_ips:
    local_detected = detect_hccn_device_ips(device_count)
    if local_detected:
        print(f"[mindie] Auto-detected local hccnX device IPs: {{local_detected}}")
        if nnodes > 1:
            master_addr = os.environ.get("MASTER_ADDR", node_ips[0])
            all_detected = exchange_device_ips_tcp(
                node_rank, nnodes, master_addr, exchange_port, local_detected,
            )
            if all_detected and len(all_detected) >= nnodes:
                node_device_ips = [all_detected.get(i, []) for i in range(nnodes)]
                print(f"[mindie] Aggregated device IPs from all nodes: {{node_device_ips}}")
            else:
                print("[mindie] ERROR: hccnX IP exchange between nodes failed. "
                      "Falling back to host IPs — HCCL will likely fail!")
                print("[mindie] FIX: Set HCCL_DEVICE_IPS=<rdma_ip_node0>;<rdma_ip_node1>")
        else:
            node_device_ips = [local_detected]
    else:
        if nnodes > 1:
            print("[mindie] ERROR: HCCL_DEVICE_IPS not set and hccnX auto-detection failed!")
            print("[mindie] Multi-node HCCL communication will almost certainly fail.")
            print("[mindie] FIX: Set HCCL_DEVICE_IPS=<rdma_ip_node0>;<rdma_ip_node1> "
                  "in the engine container environment.")
        else:
            print("[mindie] HCCL_DEVICE_IPS not set, using host IPs (single-node OK)")

# ── Step 3: 构建 server_list ──
server_list = []
global_rank = 0
for local_idx, ip in enumerate(node_ips):
    hccl_node_idx = node_offset + local_idx
    devices = []
    for dev_id in range(device_count):
        if hccl_node_idx < len(node_device_ips) and dev_id < len(node_device_ips[hccl_node_idx]):
            device_ip = node_device_ips[hccl_node_idx][dev_id]
        else:
            device_ip = ip
            if nnodes > 1:
                print(f"[mindie] ERROR: No HCCL device IP for node={{hccl_node_idx}} dev={{dev_id}}, "
                      f"fallback to host IP {{ip}} — multi-node HCCL will likely fail!")
            else:
                print(f"[mindie] WARN: No HCCL device IP for node={{hccl_node_idx}} dev={{dev_id}}, "
                      f"fallback to host IP {{ip}}")
        devices.append({{"device_id": str(dev_id), "device_ip": device_ip, "rank_id": str(global_rank)}})
        global_rank += 1
    server_list.append({{"server_id": ip, "device": devices, "container_ip": ip, "host_nic_ip": ip}})

rank_table = {{
    "version": "1.0",
    "server_count": str(len(node_ips)),
    "server_list": server_list,
    "status": "completed",
}}

os.makedirs(os.path.dirname(output_path) or "/tmp", exist_ok=True)
with open(output_path, "w") as f:
    json.dump(rank_table, f, indent=2, ensure_ascii=False)
os.chmod(output_path, 0o640)
print(f"[mindie] HCCL rank table written to {{output_path}}")
print(json.dumps(rank_table, indent=2, ensure_ascii=False))
RANK_TABLE_GEN_EOF"""

    return runtime_script.split("\n")


# =============================================================================
# 公开 API 函数
#
# 这些函数是 launcher 模块调用的入口点。
# =============================================================================


def _build_server_overrides(
    engine_config: Dict[str, Any],
    is_distributed: bool,
    node_rank: int,
    nnodes: int,
) -> Dict[str, Any]:
    """构建 MindIE ServerConfig 覆盖参数。"""
    ip_address = "0.0.0.0" if (not is_distributed or node_rank == 0) else "127.0.0.1"
    overrides: Dict[str, Any] = {
        "ipAddress": engine_config.get("ipAddress", ip_address),
        "port": engine_config.get("port", DEFAULT_SERVER_PORT),
        "httpsEnabled": engine_config.get("httpsEnabled", False),
        "inferMode": engine_config.get("inferMode", "standard"),
        "openAiSupport": engine_config.get("openAiSupport", "vllm"),
        "tokenTimeout": engine_config.get("tokenTimeout", 600),
        "e2eTimeout": engine_config.get("e2eTimeout", 600),
        "allowAllZeroIpListening": engine_config.get("allowAllZeroIpListening", True),
    }
    if is_distributed and nnodes > 1:
        overrides["interCommTLSEnabled"] = engine_config.get("interCommTLSEnabled", False)
    return overrides


def _build_backend_overrides(
    engine_config: Dict[str, Any],
    is_distributed: bool,
    nnodes: int,
    npu_device_ids: Any,
) -> Dict[str, Any]:
    """构建 MindIE BackendConfig 覆盖参数。"""
    overrides: Dict[str, Any] = {
        "npuDeviceIds": npu_device_ids,
        "multiNodesInferEnabled": (
            True if (is_distributed and nnodes > 1)
            else engine_config.get("multiNodesInferEnabled", False)
        ),
    }
    if is_distributed and nnodes > 1:
        overrides["interNodeTLSEnabled"] = engine_config.get("interNodeTLSEnabled", False)
    return overrides


def _build_model_deploy_overrides(engine_config: Dict[str, Any]) -> Dict[str, Any]:
    """构建 MindIE ModelDeployConfig 覆盖参数。"""
    return {
        "maxSeqLen": engine_config.get("maxSeqLen", 2560),
        "maxInputTokenLen": engine_config.get("maxInputTokenLen", 2048),
        "truncation": engine_config.get("truncation", False),
    }


def _inject_multinode_tp_dp(
    engine_config: Dict[str, Any],
    is_distributed: bool,
    nnodes: int,
    global_world_size: int,
    overrides: Dict[str, Any],
) -> None:
    """MindIE 2.3.0 多节点必须显式设置 tp/dp，否则内部 DP 计算返回 0 触发 C++ 崩溃。

    公式: global_world_size = TP × DP × PP
    """
    if not (is_distributed and nnodes > 1 and global_world_size > 0):
        return
    # PP（流水线并行）优先从 engine_config 读取，默认为 1
    pp = int(engine_config.get("pp", 1) or 1)
    effective_tp = global_world_size // pp if pp > 1 else global_world_size
    effective_dp = 1
    if engine_config.get("tp") is None:
        overrides["tp"] = effective_tp
        if pp > 1:
            logger.info(
                    "[mindie] Multi-node: auto-set tp=%d "
                    "(worldSize=%d / pp=%d)",
                    effective_tp, global_world_size, pp
                )
        else:
            logger.info("[mindie] Multi-node: auto-set tp=%d (global worldSize)", effective_tp)
    if engine_config.get("dp") is None:
        overrides["dp"] = effective_dp
        logger.info("[mindie] Multi-node: auto-set dp=%d", effective_dp)


def _inject_moe_config(
    engine_config: Dict[str, Any],
    world_size: int,
    overrides: Dict[str, Any],
) -> None:
    """注入 MoE 并行配置（isMOE=True 时生效）。"""
    if not engine_config.get("isMOE", False):
        return
    overrides.update({
        "tp": engine_config.get("tp", world_size),
        "dp": engine_config.get("dp", -1),
        "moe_tp": engine_config.get("moe_tp", world_size),
        "moe_ep": engine_config.get("moe_ep", -1),
    })


def _inject_parallel_passthrough(engine_config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """透传 sp/cp/dp/tp 并行配置（非 MoE 场景）。"""
    is_moe = engine_config.get("isMOE", False)
    if engine_config.get("sp") is not None:
        overrides["sp"] = engine_config["sp"]
    if engine_config.get("cp") is not None:
        overrides["cp"] = engine_config["cp"]
    if engine_config.get("dp") is not None and not is_moe:
        overrides["dp"] = engine_config["dp"]
    if engine_config.get("tp") is not None and not is_moe:
        overrides["tp"] = engine_config["tp"]


def _inject_mtp_plugin(engine_config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """注入 MTP（Multi-Token Prediction）投机解码插件参数。"""
    if not engine_config.get("isMTP", False):
        return
    overrides["plugin_params"] = {
        "plugin_type": "mtp",
        "num_speculative_tokens": 1,
    }


def _inject_function_call_config(engine_config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """注入 Function Call 工具调用配置（models.<model_type>.tool_call_options）。

    MindIE 通过 ModelConfig[0].models 嵌套结构配置 tool_call_parser。
    格式: {"models": {"<model_type>": {"tool_call_options": {"tool_call_parser": "xxx"}}}}
    """
    mindie_tool_call_parser = engine_config.get("mindie_tool_call_parser")
    mindie_model_type = engine_config.get("mindie_model_type")
    if not (mindie_tool_call_parser and mindie_model_type):
        return
    model_entry: Dict[str, Any] = {
        "tool_call_options": {"tool_call_parser": mindie_tool_call_parser},
    }
    chat_template = engine_config.get("mindie_chat_template")
    if chat_template:
        model_entry["chat_template"] = chat_template
    overrides["models"] = {mindie_model_type: model_entry}
    logger.info(
        "[mindie] Function Call enabled: model_type=%s, parser=%s",
        mindie_model_type, mindie_tool_call_parser,
    )


def _build_model_config_overrides(
    engine_config: Dict[str, Any],
    is_distributed: bool,
    world_size: int,
    *,
    global_world_size: int = 0,
    nnodes: int = 1,
) -> Dict[str, Any]:
    """构建 MindIE ModelConfig[0] 覆盖参数。

    MindIE 2.3.0 多节点要求显式指定 tp/dp 并行策略，否则内部 DP 计算
    返回 0，导致 std::out_of_range / vector::_M_range_check 崩溃。
    参考: https://www.hiascend.com/forum/thread-02144208667617662184-1-1.html
    """
    overrides: Dict[str, Any] = {
        "modelName": engine_config.get("modelName", "default_llm"),
        "modelWeightPath": engine_config.get("modelWeightPath", ""),
        "worldSize": world_size,
        "cpuMemSize": engine_config.get("cpuMemSize", 5),
        "npuMemSize": engine_config.get("npuMemSize", -1),
        "trustRemoteCode": engine_config.get("trustRemoteCode", True),
    }

    _inject_multinode_tp_dp(engine_config, is_distributed, nnodes, global_world_size, overrides)
    _inject_moe_config(engine_config, world_size, overrides)
    _inject_parallel_passthrough(engine_config, overrides)
    _inject_mtp_plugin(engine_config, overrides)
    _inject_function_call_config(engine_config, overrides)

    return overrides


def _build_schedule_overrides(engine_config: Dict[str, Any]) -> Dict[str, Any]:
    """构建 MindIE ScheduleConfig 覆盖参数。"""
    return {
        "cacheBlockSize": engine_config.get("cacheBlockSize", 128),
        "maxPrefillBatchSize": engine_config.get("maxPrefillBatchSize", 50),
        "maxPrefillTokens": engine_config.get("maxPrefillTokens", 8192),
        "prefillTimeMsPerReq": engine_config.get("prefillTimeMsPerReq", 150),
        "prefillPolicyType": engine_config.get("prefillPolicyType", 0),
        "decodeTimeMsPerReq": engine_config.get("decodeTimeMsPerReq", 50),
        "decodePolicyType": engine_config.get("decodePolicyType", 0),
        "maxBatchSize": engine_config.get("maxBatchSize", 200),
        "maxIterTimes": engine_config.get("maxIterTimes", 2048),
        "maxPreemptCount": engine_config.get("maxPreemptCount", 0),
        "supportSelectBatch": engine_config.get("supportSelectBatch", False),
        "maxQueueDelayMicroseconds": engine_config.get("maxQueueDelayMicroseconds", 5000),
        "bufferResponseEnabled": engine_config.get("bufferResponseEnabled", False),
        "decodeExpectedTime": engine_config.get("decodeExpectedTime", 50),
        "prefillExpectedTime": engine_config.get("prefillExpectedTime", 1500),
    }


def _build_config_merge_script(
    overrides_json: str,
    safe_config_path: str,
    safe_work_dir: str,
) -> str:
    """生成 config.json 合并更新 + 守护进程启动的脚本片段。

    支持两种配置来源:
      1. 本地模板: /opt/wings-control/config/defaults/mindie_service_config.json
         如果存在，直接作为基础配置（完全替代镜像内默认配置）
      2. 镜像默认: MINDIE_CONFIG_PATH 指向的原始 config.json
         本地模板不存在时使用镜像内配置

    幂等性保证:
      首次 merge 前备份原始配置为 config.json.orig，后续 merge 始终从
      .orig 重新开始，避免多次 merge 导致配置累积/污染。
    """
    return f"""# ── Merge-update MindIE config.json (preserve original, override changed) ──
export _MINDIE_CONFIG_PATH={safe_config_path}
_LOCAL_TEMPLATE=/opt/wings-control/config/defaults/mindie_service_config.json

cat > /tmp/_mindie_overrides.json << 'OVERRIDES_EOF'
{overrides_json}
OVERRIDES_EOF

python3 << 'MERGE_SCRIPT_EOF'
import json, os, sys

CONFIG_PATH = os.environ['_MINDIE_CONFIG_PATH']
LOCAL_TEMPLATE = os.environ.get('_LOCAL_TEMPLATE', '/opt/wings-control/config/defaults/mindie_service_config.json')
OVERRIDES_PATH = '/tmp/_mindie_overrides.json'
BACKUP_PATH = CONFIG_PATH + '.orig'

# 1. Load base config (idempotent: always merge from original/template, never from already-merged file)
if os.path.isfile(LOCAL_TEMPLATE):
    with open(LOCAL_TEMPLATE, 'r') as f:
        config = json.load(f)
    for meta_key in ('_comment', '_usage'):
        config.pop(meta_key, None)
    print(f'[mindie] Loaded LOCAL template config ({{len(json.dumps(config))}} chars)')
elif os.path.isfile(BACKUP_PATH):
    # Idempotent: re-merge from the original backup, not the already-merged file
    with open(BACKUP_PATH, 'r') as f:
        config = json.load(f)
    print(f'[mindie] Loaded original backup config ({{BACKUP_PATH}}, {{len(json.dumps(config))}} chars)')
else:
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        # First merge: save a backup for future idempotent re-merges
        import shutil
        shutil.copy2(CONFIG_PATH, BACKUP_PATH)
        print(f'[mindie] Loaded original config.json ({{len(json.dumps(config))}} chars), '
              f'backup saved to {{BACKUP_PATH}}')
    except Exception as e:
        print(f'[mindie] ERROR: Cannot read {{CONFIG_PATH}}: {{e}}', file=sys.stderr)
        sys.exit(1)

# 2. Load overrides
with open(OVERRIDES_PATH, 'r') as f:
    ov = json.load(f)

# 3. Merge (update only specified keys; keep all other original fields intact)
if 'ServerConfig' in config:
    config['ServerConfig'].update(ov['server'])

if 'BackendConfig' in config:
    bc = config['BackendConfig']
    bc.update(ov['backend'])

    if 'ModelDeployConfig' in bc:
        bc['ModelDeployConfig'].update(ov['model_deploy'])
        mc = bc['ModelDeployConfig'].get('ModelConfig')
        if isinstance(mc, list) and len(mc) > 0 and isinstance(mc[0], dict):
            mc[0].update(ov['model_config'])
        elif mc is not None:
            print(f'[mindie] WARNING: ModelConfig has unexpected type/value: '
                  f'{{type(mc).__name__}} = {{mc}}', file=sys.stderr)
            # Force-create a valid ModelConfig array with our overrides
            bc['ModelDeployConfig']['ModelConfig'] = [ov['model_config']]
            print('[mindie] Created ModelConfig[0] from overrides')
        else:
            bc['ModelDeployConfig']['ModelConfig'] = [ov['model_config']]
            print('[mindie] ModelConfig was missing/None, created from overrides')

    if 'ScheduleConfig' in bc:
        bc['ScheduleConfig'].update(ov['schedule'])

# 4. Apply extra pass-through keys to config root level
extra = ov.get('extra', {{}})
if extra:
    config.update(extra)
    print(f'[mindie] Applied {{len(extra)}} extra pass-through keys: {{list(extra.keys())}}')

# 5. Write back (atomic: write to tmp then rename)
tmp_out = CONFIG_PATH + '.tmp'
with open(tmp_out, 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
os.chmod(tmp_out, 0o640)
os.replace(tmp_out, CONFIG_PATH)

print('[mindie] config.json merge-updated successfully')
print(json.dumps(config, indent=2, ensure_ascii=False))
MERGE_SCRIPT_EOF

# ── Start MindIE daemon (background + wait, per official boot.sh) ────────────
cd {safe_work_dir}
./bin/mindieservice_daemon &
MINDIE_PID=$!
echo "[mindie] Daemon started as PID $MINDIE_PID"
wait $MINDIE_PID
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "[mindie] ERROR: daemon exited with code $exit_code"
fi
exit $exit_code
"""


def build_start_command(params: Dict[str, Any]) -> str:
    """返回 MindIE 守护进程的核心启动命令（不含配置写入步骤）。

    警告：
        此命令单独使用不足以启动 MindIE！
        - MindIE 需要预先配置好的 config.json
        - 分布式模式需要环境变量和 rank table

        请使用 build_start_script() 获取完整启动脚本。

    Args:
        params: 参数字典（当前未使用，为接口一致性保留）

    Returns:
        str: 切换到工作目录并启动守护进程的命令
    """
    return f"cd {shlex.quote(MINDIE_WORK_DIR)} && exec ./bin/mindieservice_daemon"


# ---------- 已知键分类 ----------
# _MINDIE_CONSUMED_KEYS: 已被 wings-control 显式处理的键，不需要再透传
_MINDIE_CONSUMED_KEYS: frozenset = frozenset({
    "port", "ipAddress", "httpsEnabled", "inferMode", "openAiSupport",
    "tokenTimeout", "e2eTimeout", "allowAllZeroIpListening", "interCommTLSEnabled",
    "npuDeviceIds", "multiNodesInferEnabled", "interNodeTLSEnabled",
    "maxSeqLen", "maxInputTokenLen", "truncation",
    "modelName", "modelWeightPath", "worldSize", "cpuMemSize", "npuMemSize",
    "trustRemoteCode", "isMOE", "isMTP", "tp", "dp", "moe_tp", "moe_ep", "sp", "cp",
    "cacheBlockSize", "maxPrefillBatchSize", "maxPrefillTokens",
    "prefillTimeMsPerReq", "prefillPolicyType", "decodeTimeMsPerReq",
    "decodePolicyType", "maxBatchSize", "maxIterTimes", "maxPreemptCount",
    "supportSelectBatch", "maxQueueDelayMicroseconds", "bufferResponseEnabled",
    "decodeExpectedTime", "prefillExpectedTime",
    "npu_memory_fraction",
    "node_ips", "device_count",
    "mindie_tool_call_parser", "mindie_model_type", "mindie_chat_template",
    # wings-control 内部参数（non-MindIE），防止透传到 config.json 引发
    # MindIE C++ std::out_of_range crash
    "host", "save_path", "accel", "enable_speculative_decode",
})

# _MINDIE_SAFE_PASSTHROUGH_KEYS: MindIE config.json 根级别接受的合法扩展键白名单。
# 只有在此白名单中且不在 _MINDIE_CONSUMED_KEYS 中的键才会被透传。
# 引用: MindIE 2.x config.json schema (官方文档)
_MINDIE_SAFE_PASSTHROUGH_KEYS: frozenset = frozenset({
    # 调度相关
    "schedulerType", "schedulerConfig", "inputPaddingTo",
    "enableRollingBatch", "rollingBatchTimeout",
    # 推理精度 / 量化
    "quantType", "kvCacheType", "weightQuantType",
    "enableFloatAtten", "enableKvQuant",
    # 内存 / cache
    "blockSize", "numBlocks", "gpuMemoryUtilization",
    # tokenizer / chat template
    "tokenizerPath", "chatTemplate", "toolCallParser",
    # plugin / 扩展
    "pluginParams", "customOp",
    # 其他 MindIE 已知合法根键
    "enableLogits", "enablePromptLogprobs",
    "enablePrefill", "enableDecode",
    "pp",
})


def _resolve_npu_device_ids(
    engine_config: Dict[str, Any],
    is_distributed: bool,
    device_count: int = 8,
) -> list:
    """解析 npuDeviceIds 配置：优先使用 engine_config，其次读取环境变量，最后使用默认值。

    Args:
        engine_config: 引擎配置字典
        is_distributed: 是否分布式模式
        device_count: 本节点 NPU 设备数量，用于生成默认 npuDeviceIds
    """
    npu_default = [list(range(device_count))]
    npu_device_ids = engine_config.get("npuDeviceIds", None)
    if npu_device_ids is not None:
        return npu_device_ids
    npu_ids_env = os.getenv("MINDIE_NPU_DEVICE_IDS", "")
    if npu_ids_env:
        try:
            return json.loads(npu_ids_env)
        except ValueError:
            logger.warning("MINDIE_NPU_DEVICE_IDS parse error: %s, fallback to %s", npu_ids_env, npu_default)
    return npu_default


def _collect_extra_overrides(engine_config: Dict[str, Any]) -> Dict[str, Any]:
    """收集 engine_config 中可安全透传给 MindIE config.json 根级别的额外配置项。

    使用白名单策略（_MINDIE_SAFE_PASSTHROUGH_KEYS）：
    - 只有在白名单中且未被 _MINDIE_CONSUMED_KEYS 消耗的键才会被透传
    - 未识别的键会记录警告并跳过，防止 MindIE C++ 崩溃
      (std::out_of_range from _Map_base::at)

    如需新增透传键，请将其加入 _MINDIE_SAFE_PASSTHROUGH_KEYS 白名单。
    """
    extra = {}
    unknown = []
    for k, v in engine_config.items():
        if k in _MINDIE_CONSUMED_KEYS:
            continue
        if k in _MINDIE_SAFE_PASSTHROUGH_KEYS:
            extra[k] = v
        else:
            unknown.append(k)
    if unknown:
        logger.warning(
            "[mindie] Ignoring %d unknown engine_config key(s) "
            "(not in allow-list, may crash MindIE if passed through): %s",
            len(unknown), unknown,
        )
    if extra:
        logger.info("[mindie] Extra config-file keys passed through (allow-list): %s", list(extra.keys()))
    return extra


def build_start_script(params: Dict[str, Any]) -> str:
    """返回完整的 bash 启动脚本内容（不含 shebang 行）。

    生成的脚本将写入共享卷，由 engine 容器执行。执行流程：
      1. 加载 Ascend CANN 工具包和 MindIE 环境脚本
      2. 导出分布式环境变量（仅当 nnodes > 1 时）
      3. 内联 Python 脚本合并更新 MindIE conf/config.json
      4. 启动 mindieservice_daemon

    Args:
        params: 包含 engine_config / distributed / nnodes / node_rank 等字段。

    Returns:
        str: 完整的 bash 脚本内容（不含 #!/bin/bash）
    """
    # 浅拷贝 params，避免原地修改调用方传入的字典
    # (上游 wings_entry.build_launcher_plan 会将同一个 dict 存入 LauncherPlan.merged_params)
    params = dict(params)
    engine_config = params.get("engine_config", {})
    is_distributed = params.get("distributed", False)
    nnodes = params.get("nnodes", 1)
    node_rank = params.get("node_rank", 0)

    # 将 engine_config 中的关键字段提升到 params 顶层（供下游 helpers 使用）
    # 使用 `key not in params or params[key] is None` 而非 `not params.get(key)`，
    # 避免将 falsy 但合法的值（如 0、False）误判为未设置并被覆盖。
    for key in ("node_ips", "device_count", "worldSize", "multiNodesInferEnabled"):
        if (key not in params or params[key] is None) and engine_config.get(key) is not None:
            params[key] = engine_config[key]

    topology = _resolve_distributed_topology(params) if (is_distributed and nnodes > 1) else None
    local_device_count = int(params.get("device_count", 1) or 1)
    npu_device_ids = _resolve_npu_device_ids(engine_config, is_distributed, local_device_count)

    # 构建各配置覆盖区块
    server_overrides = _build_server_overrides(engine_config, is_distributed, node_rank, nnodes)
    backend_overrides = _build_backend_overrides(engine_config, is_distributed, nnodes, npu_device_ids)
    model_deploy_overrides = _build_model_deploy_overrides(engine_config)

    # ── worldSize 语义修正 (多节点) ────────────────────────────────
    # engine_config["worldSize"] 来自 config_loader，是全局 TP 总卡数。
    # 但 MindIE config.json 中 ModelConfig.worldSize 必须等于本节点的
    # npuDeviceIds 数量，否则 ConfigManager 校验失败。
    # 全局 TP 总卡数已通过 MINDIE_MODEL_WORLD_SIZE 环境变量 + HCCL
    # rank table 传递，config.json 中只需填本节点卡数。
    global_world_size = engine_config.get(
        "worldSize",
        topology["global_world_size"] if topology else (8 if is_distributed else 1),
    )
    if topology and topology["nnodes"] > 1:
        config_world_size = topology["device_count"]
    else:
        config_world_size = global_world_size

    model_config_overrides = _build_model_config_overrides(
        engine_config, is_distributed, config_world_size,
        global_world_size=global_world_size,
        nnodes=nnodes,
    )
    schedule_overrides = _build_schedule_overrides(engine_config)
    extra_overrides = _collect_extra_overrides(engine_config)

    overrides_json = json.dumps({
        "server": server_overrides,
        "backend": backend_overrides,
        "model_deploy": model_deploy_overrides,
        "model_config": model_config_overrides,
        "schedule": schedule_overrides,
        "extra": extra_overrides,
    }, indent=2, ensure_ascii=False)

    # 组装环境变量命令块
    env_cmds = _build_env_commands(params)
    dist_cmds = _build_distributed_env_commands(params)
    all_cmds = env_cmds + ([""] + dist_cmds if dist_cmds else [])
    env_block = "\n".join(all_cmds) + "\n" if all_cmds else ""

    dist_label = f"distributed rank={node_rank}/{nnodes}" if (is_distributed and nnodes > 1) else "single-node"
    logger.info(
        "[mindie] Generating start_command.sh: %s, globalWorldSize=%d, configWorldSize=%d",
        dist_label, global_world_size, config_world_size,
    )

    merge_script = _build_config_merge_script(
        overrides_json, shlex.quote(MINDIE_CONFIG_PATH), shlex.quote(MINDIE_WORK_DIR)
    )
    return f"""{env_block}{merge_script}"""


def start_engine(params: Dict[str, Any]):
    """直接启动引擎存根函数 - sidecar launcher 模式下已禁用。

    此函数为 sidecar 架构契约的一部分：
      - launcher 容器永远不直接启动引擎进程
      - 启动逻辑通过 build_start_script() 生成脚本
      - 脚本写入共享卷由 engine 容器执行

    调用此函数会抛出 RuntimeError，阻止意外的直接启动。

    Args:
        params: 参数字典（未使用）

    Raises:
        RuntimeError: 总是抛出，说明应使用 build_start_script()
    """
    raise RuntimeError(
        "start_engine is disabled in launcher mode. "
        "Use build_start_script() and write to shared volume instead."
    )
