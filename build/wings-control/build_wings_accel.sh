#!/bin/bash

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

# wings-control 实际以wings-infer镜像名称呈现
WINGS_VERSION="${WINGS_VERSION:-26.0.0}"
REPO="fusionregistry:5000"
WINGS_CONTROL_TAG="${REPO}/wings-infer:${WINGS_VERSION}"

# 根据不同的架构选择基础镜像文件
case $ARCH in
    x86_64|amd64)
        log_info "检测到x86架构"
        # 基础镜像
        BASE_IMG="docker.artifactrepo.wux-g.tools.xfusion.com/docker_universal2/linux/amd64/python:3.10.19-slim"
        # 最终输出文件
        WINGS_CONTROL_FILE="Wings-Infer_${WINGS_VERSION}_x86_64.tar"
        ;;
    arm|aarch64)
        log_info "检测到ARM架构"
        # 基础镜像
        BASE_IMG="docker.artifactrepo.wux-g.tools.xfusion.com/docker_universal2/linux/arm64/python:3.10.19-slim"
        # 最终输出文件
        WINGS_CONTROL_FILE="Wings-Infer_${WINGS_VERSION}_aarch64.tar"
        ;;
    *)
        log_error "不支持的架构: ${ARCH}"
        exit 1
        ;;
esac


function build_wings_control_image(){
    log_info "开始构建Wings Control 镜像，当前CPU架构： ${ARCH}"

    # 导入基础镜像
    log_info "拉取基础镜像: ${BASE_IMG}"
    docker pull ${BASE_IMG} || { log_error "拉取基础镜像失败: ${BASE_IMG}"; exit 1; }

    # 构建镜像
    log_info "构建Docker镜像: ${WINGS_CONTROL_TAG}"
    docker build --build-arg BASE_IMG="${BASE_IMG}" \
                 --build-arg WINGS_VERSION="${WINGS_VERSION}" \
                 --build-arg WINGS_CONTROL_BUILD_DATE=$(date +%Y%m%d_%H%M%S) \
                 -f Dockerfile \
                 -t "${WINGS_CONTROL_TAG}" . || { log_error "Docker镜像构建失败"; exit 1; }

    # 打包镜像
    log_info "打包镜像到: ../output/${WINGS_CONTROL_FILE}"
    docker save "${WINGS_CONTROL_TAG}" -o "${WINGS_CONTROL_FILE}" || { log_error "镜像打包失败"; exit 1; }
    mv "${WINGS_CONTROL_FILE}" ../output/
    log_info "${ARCH}镜像构建完成: ${WINGS_CONTROL_FILE}"
}


function main(){
    cd "$WORK_DIR" && ls -al

    log_info "=========================================="
    log_info "Wings-Control 镜像构建脚本"
    log_info "版本: ${WINGS_VERSION}"
    log_info "架构: ${ARCH}"
    log_info "=========================================="

    # 准备文件：加密脚本、wings-control代码
    cp ../compile_wings.sh .
    cp -r ../../wings_control .

    build_wings_control_image

    log_info "=========================================="
    log_info "Wings-Control 镜像构建完成"
    log_info "=========================================="
}


main