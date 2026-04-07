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

# wings accel 的文件打包成一个压缩包，不区分arm和x86；
WINGS_ACCEL_PACKAGE="wings-accel-package.tar.gz"

# 最终镜像名称
WINGS_VERSION="${WINGS_VERSION:-26.0.0}"
REPO="fusionregistry:5000"
WINGS_ACCEL_TAG="${REPO}/wings-accel:${WINGS_VERSION}"

# 根据不同的CPU架构选择基础镜像以及最终的归档文件名
case $ARCH in
    x86_64|amd64)
        log_info "检测到x86架构"

        # 基础镜像
        BASE_IMG="docker.artifactrepo.wux-g.tools.xfusion.com/docker_universal2/linux/amd64/docker.io/library/busybox:1.36"

        # 最终输出文件
        WINGS_ACCEL_FILE="Wings-Accel_${WINGS_VERSION}_x86_64.tar"

        ;;
    arm|aarch64)
        log_info "检测到ARM架构"

        # 基础镜像
        BASE_IMG="docker.artifactrepo.wux-g.tools.xfusion.com/docker_universal2/linux/arm64/busybox:1.36"

        # 最终输出文件
        WINGS_ACCEL_FILE="Wings-Accel_${WINGS_VERSION}_aarch64.tar"

        ;;
    *)
        log_error "不支持的架构: ${ARCH}"
        exit 1
        ;;
esac


function build_wings_accel_image(){
    log_info "开始构建wings accel镜像，CPU架构： ${ARCH}"

    # 准备文件
    mv "${WORK_DIR}"/../input/* "${WORK_DIR}"/

    # 检查文件
    if [ -e "${WORK_DIR}/${WINGS_ACCEL_PACKAGE}" ]; then
        log_info "File ${WINGS_ACCEL_PACKAGE} exist"
    else
        log_error "File ${WINGS_ACCEL_PACKAGE} does not exist."
        exit 1
    fi

    # 导入基础镜像
    log_info "拉取基础镜像: ${BASE_IMG}"
    docker pull ${BASE_IMG} || { log_error "拉取基础镜像失败: ${BASE_IMG}"; exit 1; }

    # 构建镜像
    log_info "构建Docker镜像: ${WINGS_ACCEL_TAG}"

    docker build --build-arg BASE_IMG="${BASE_IMG}" \
                 --build-arg WINGS_VERSION="${WINGS_VERSION}" \
                 --build-arg WINGS_ACCEL_BUILD_DATE=$(date +%Y%m%d_%H%M%S) \
                 --build-arg WINGS_ACCEL_PACKAGE=${WINGS_ACCEL_PACKAGE} \
                 -f Dockerfile \
                 -t "${WINGS_ACCEL_TAG}" . || { log_error "Docker镜像构建失败"; exit 1; }

    # 打包镜像
    log_info "打包镜像到: ../output/${WINGS_ACCEL_FILE}"
    docker save "${WINGS_ACCEL_TAG}" -o "${WINGS_ACCEL_FILE}" || { log_error "镜像打包失败"; exit 1; }
    mv "${WINGS_ACCEL_FILE}" ../output/
    log_info "${ARCH}镜像构建完成: ${WINGS_ACCEL_FILE}"
}


function main(){
    cd "$WORK_DIR" && ls -al

    log_info "=========================================="
    log_info "Wings-Accel_ 镜像构建脚本"
    log_info "版本: ${WINGS_VERSION}"
    log_info "架构: ${ARCH}"
    log_info "=========================================="

    build_wings_accel_image

    log_info "=========================================="
    log_info "Wings-AcceL 镜像构建完成"
    log_info "=========================================="
}

main