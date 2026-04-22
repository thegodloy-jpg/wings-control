#!/bin/bash
# ============================================================================
# Ascend NPU hccn IP 配置脚本 — 16 卡 (2 × 8 NPU 或 1 × 16 NPU)
#
# 用途: 为每张 NPU 配置 RDMA IP，使 PD 分离的 Mooncake KV Transfer 正常工作
# 前提: 在宿主机上以 root 权限执行，hccn_tool 已安装
# 说明: IP 使用 192.168.100.0/24 子网，按需修改 SUBNET 和 NETMASK 变量
# ============================================================================

set -euo pipefail

# ── 配置区（按实际环境修改） ──
SUBNET="192.168.100"      # 子网前三位
NETMASK="255.255.255.0"   # 子网掩码
NPU_COUNT=16              # NPU 总数
START_IP=1                # 起始 IP 后缀 (device_0 → .1, device_1 → .2, ...)

# ── 检查权限 ──
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: 请用 root 权限执行此脚本"
    exit 1
fi

# ── 检查 hccn_tool ──
if ! command -v hccn_tool &>/dev/null; then
    echo "ERROR: hccn_tool 未找到，请确认 CANN 驱动已安装"
    exit 1
fi

echo "========== 当前 hccn.conf =========="
cat /etc/hccn.conf 2>/dev/null || echo "(文件不存在)"
echo ""

echo "========== 开始配置 ${NPU_COUNT} 张 NPU IP =========="
for i in $(seq 0 $((NPU_COUNT - 1))); do
    ip_suffix=$((START_IP + i))
    ip_addr="${SUBNET}.${ip_suffix}"
    echo "[NPU ${i}] 配置 IP: ${ip_addr} / ${NETMASK}"
    hccn_tool -i "${i}" -ip -s address "${ip_addr}" netmask "${NETMASK}"
    if [ $? -ne 0 ]; then
        echo "WARNING: NPU ${i} 配置失败，可能设备不存在或驱动异常"
    fi
done

echo ""
echo "========== 验证配置 =========="
for i in $(seq 0 $((NPU_COUNT - 1))); do
    echo -n "[NPU ${i}] "
    hccn_tool -i "${i}" -ip -g 2>&1 || echo "获取失败"
done

echo ""
echo "========== 更新后 hccn.conf =========="
cat /etc/hccn.conf

echo ""
echo "========== 完成 =========="
echo "已配置 ${NPU_COUNT} 张 NPU 的 IP 地址 (${SUBNET}.${START_IP} ~ ${SUBNET}.$((START_IP + NPU_COUNT - 1)))"
echo ""
echo "后续步骤:"
echo "  1. 重启相关 Pod 使配置生效"
echo "  2. 进入 Pod 检查: cat /etc/hccn.conf"
echo "  3. 验证 NPU 网卡可见: ip addr show | grep hccn"
