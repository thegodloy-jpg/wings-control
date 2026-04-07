#!/bin/bash

# Wings 镜像构建主脚本
# 依次调用 wings-accel 和 wings-infer 的构建脚本完成整个镜像的构建流程

# 设置工作目录
SCRIPT_PATH=$(readlink -f "$0")
WORK_DIR=$(dirname "$SCRIPT_PATH")

# 获取系统架构信息
ARCH=$(uname -m)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取当前时间戳
function get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}


function log_info() {
    echo -e "${GREEN}[$(get_timestamp)][Wings][INFO]${NC} $1"
}


function log_warn() {
    echo -e "${YELLOW}[$(get_timestamp)][Wings][WARN]${NC} $1"
}


function log_error() {
    echo -e "${RED}[$(get_timestamp)][Wings][ERROR]${NC} $1"
}


function print_banner() {
    log_info ""
    log_info "=========================================="
    log_info "  Wings 镜像构建主脚本"
    log_info "  版本: ${WINGS_VERSION}"
    log_info "  架构: ${ARCH}"
    log_info "=========================================="
    log_info ""
}


function check_docker() {
    log_info "检查 Docker 环境..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    log_info "Docker 环境检查通过"
}


function build_wings_accel() {
    log_info "=========================================="
    log_info "开始构建 wings-accel 镜像..."
    log_info "=========================================="

    local accel_script="${WORK_DIR}/wings-accel/build_wings_accel.sh"

    if [ ! -f "$accel_script" ]; then
        log_error "wings-accel 构建脚本不存在: $accel_script"
        exit 1
    fi

    # 赋予执行权限
    chmod +x "$accel_script"

    # 执行构建脚本
    if bash "$accel_script"; then
        log_info "wings-accel 镜像构建成功"
    else
        log_error "wings-accel 镜像构建失败"
        exit 1
    fi
}


function build_wings_control() {
    log_info "=========================================="
    log_info "开始构建 wings-control 镜像..."
    log_info "=========================================="

    local control_script="${WORK_DIR}/wings-control/build_wings_control.sh"

    if [ ! -f "$control_script" ]; then
        log_error "wings-control 构建脚本不存在: $control_script"
        exit 1
    fi

    # 赋予执行权限
    chmod +x "$control_script"

    # 执行构建脚本
    if bash "$control_script"; then
        log_info "wings-control 镜像构建成功"
    else
        log_error "wings-control 镜像构建失败"
        exit 1
    fi
}


function print_summary() {
    log_info "=========================================="
    log_info "镜像构建完成！"
    log_info "=========================================="
    log_info "构建的镜像列表:"

    ls -al ./output

    log_info "=========================================="
}


function main() {
    # 检查 Docker 环境
    check_docker

    log_info "create tar file"

    # 版本控制
    if [ "$#" -eq 1 ]; then
        # 由CI构建控制最终的出包版本
        WINGS_VERSION=$1
    else
        WINGS_VERSION="26.0.0"
    fi
    export WINGS_VERSION=$WINGS_VERSION
    echo "[Wings][INFO] version: $WINGS_VERSION"

    print_banner
    build_wings_accel
    build_wings_control
    print_summary

}


main "$@"