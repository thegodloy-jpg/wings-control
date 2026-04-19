# Mooncake Transfer Engine 安装指南（ARM + Ascend NPU）

> 面向 vLLM-Ascend PD 分离场景的 Mooncake 源码编译与部署指南。
> 基于 aarch64 + Ascend 910B 环境的真实实践总结。

## 1. 背景

### 1.1 为什么不能 pip install

在 ARM + Ascend NPU 环境下，**不能通过 `pip install` 安装 Mooncake**，原因有两个：

| 问题 | 说明 |
|------|------|
| **架构不匹配** | PyPI 上的 `mooncake-transfer-engine` 和 `mooncake-transfer-engine-non-cuda` 仅提供 x86_64 wheel，aarch64 无预编译包 |
| **缺少 Ascend 传输层** | 即使 pip 包能安装，`non-cuda` 版本也只包含 TCP/HTTP 传输层，**不包含 `ascend_direct_transport` 模块**。NPU 间 KV Cache 直传需要该模块通过 HIXL/HCCL 走 RDMA/SDMA 通道 |

```
non-cuda = "关掉 CUDA"   ← 不等于 "打开 Ascend"
Ascend   = "打开 Ascend"  ← 需要编译 cmake -DUSE_ASCEND_DIRECT=ON
```

| 安装方式 | 传输路径 | 性能 |
|---------|---------|------|
| pip install (non-cuda) | NPU → Host CPU → TCP → Host CPU → NPU | 差（额外 D2H + H2D 拷贝） |
| **源码编译 (Ascend Direct)** | **NPU → RDMA/HCCS → NPU（零拷贝）** | **接近硬件上限** |

### 1.2 版本要求

| 组件 | 最低版本 | 推荐 |
|------|---------|------|
| Mooncake | >= 0.3.9 | v0.3.10.post1（修复竞态条件） |
| CANN | >= 8.5.0 | 8.5.1+ |
| vLLM | main 分支 | - |
| vLLM-Ascend | main 分支 | - |
| Python | 3.10 / 3.11 / 3.12 / 3.13 | 3.11 |
| CMake | >= 3.20 | - |
| GCC | >= 9.4 | 11+ |
| OS | Ubuntu 22.04 LTS+ | - |

---

## 2. 前置条件检查

在 Pod/容器中执行：

```bash
# 架构确认（必须为 aarch64）
uname -m

# Python 版本
python3 --version

# CANN 安装检查
ls /usr/local/Ascend/ascend-toolkit/
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null

# NPU 设备可见性
npu-smi info 2>/dev/null | head -20

# 编译工具检查
cmake --version
g++ --version
git --version
```

---

## 3. 完整安装步骤（一次性脚本）

将以下内容保存为 `install_mooncake_ascend.sh` 并执行：

```bash
#!/bin/bash
set -e

echo "===== Mooncake Ascend Direct 安装脚本 ====="
echo "目标：aarch64 + Ascend NPU 源码编译安装"
echo ""

# ========== 1. 清理旧版本 ==========
echo "[1/8] 清理旧版本..."
pip uninstall mooncake-transfer-engine mooncake-transfer-engine-non-cuda mooncake -y 2>/dev/null || true
rm -rf /tmp/Mooncake /tmp/yalantinglibs-0.5.7 /tmp/0.5.7.zip

# ========== 2. 克隆源码 ==========
echo "[2/8] 克隆 Mooncake 源码..."
cd /tmp

# 如果需要代理访问 GitHub（K8s Pod 常见场景）：
# export http_proxy=http://<PROXY_HOST>:<PROXY_PORT>
# export https_proxy=$http_proxy
# export GIT_SSL_NO_VERIFY=1          # 代理导致 SSL 证书验证失败时使用
# git config --global http.sslVerify false

git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git submodule update --init --recursive

# ========== 3. 安装系统依赖 ==========
echo "[3/8] 安装系统依赖..."
apt-get update -qq
apt-get install -y -qq \
    build-essential cmake git pkg-config patchelf unzip wget \
    mpich libmpich-dev \
    libibverbs-dev libnuma-dev libcurl4-openssl-dev \
    libjsoncpp-dev libyaml-cpp-dev libgoogle-glog-dev \
    libhiredis-dev libjemalloc-dev libunwind-dev liburing-dev \
    libboost-all-dev libasio-dev libssl-dev \
    libprotobuf-dev protobuf-compiler-grpc \
    libgrpc++-dev libgrpc-dev libgtest-dev \
    libpython3-dev

# ========== 4. 运行官方依赖脚本 ==========
echo "[4/8] 运行 dependencies.sh..."
cd /tmp/Mooncake

# 可选：国内网络加速 Go 下载
# sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh

bash dependencies.sh -y || true
# yalantinglibs 下载可能因网络失败，下一步手动处理

# ========== 5. 手动安装 yalantinglibs（如果上一步失败） ==========
if ! find /usr/local/include -name "coro_rpc" -type d 2>/dev/null | grep -q .; then
    echo "[5/8] 手动安装 yalantinglibs..."
    cd /tmp
    wget -q https://github.com/alibaba/yalantinglibs/archive/refs/tags/0.5.7.zip || \
        curl -L -o 0.5.7.zip https://github.com/alibaba/yalantinglibs/archive/refs/tags/0.5.7.zip
    unzip -qo 0.5.7.zip
    cd yalantinglibs-0.5.7
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc) && make install
else
    echo "[5/8] yalantinglibs 已安装，跳过"
fi

# ========== 6. 编译 Mooncake（Ascend Direct） ==========
echo "[6/8] 编译 Mooncake（-DUSE_ASCEND_DIRECT=ON）..."
cd /tmp/Mooncake
rm -rf build && mkdir build && cd build
cmake .. \
    -DUSE_ASCEND_DIRECT=ON \
    -DBUILD_UNIT_TESTS=OFF
make -j$(nproc)

# ========== 7. 安装 ==========
echo "[7/8] 安装到系统路径..."
make install

# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-}

# ========== 8. 验证 ==========
echo ""
echo "===== [8/8] 安装验证 ====="

PASS=0
TOTAL=0

# 验证动态库
TOTAL=$((TOTAL+1))
if ls /usr/local/lib/libmooncake_common.so /usr/local/lib/libmooncake_store.so >/dev/null 2>&1; then
    echo "✓ libmooncake_common.so + libmooncake_store.so"
    PASS=$((PASS+1))
else
    echo "✗ libmooncake 动态库未找到"
fi

# 验证 ascend_transport.so
TOTAL=$((TOTAL+1))
if ls /usr/local/lib/ascend_transport.so >/dev/null 2>&1; then
    echo "✓ ascend_transport.so（Ascend Direct 传输层）"
    PASS=$((PASS+1))
else
    echo "✗ ascend_transport.so 未找到"
fi

# 验证 mooncake_master
TOTAL=$((TOTAL+1))
if command -v mooncake_master >/dev/null 2>&1; then
    echo "✓ mooncake_master 可执行"
    PASS=$((PASS+1))
else
    echo "✗ mooncake_master 未找到"
fi

# 验证 Python 模块
TOTAL=$((TOTAL+1))
if python3 -c "from mooncake.engine import TransferEngine; print('✓ TransferEngine 导入成功')" 2>/dev/null; then
    PASS=$((PASS+1))
else
    echo "✗ TransferEngine 导入失败"
fi

# 验证 CANN
TOTAL=$((TOTAL+1))
if [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
    echo "✓ CANN ascend-toolkit 已安装"
    PASS=$((PASS+1))
else
    echo "✗ CANN 未安装"
fi

echo ""
echo "===== 结果：${PASS}/${TOTAL} 通过 ====="

if [ "$PASS" -eq "$TOTAL" ]; then
    echo "✅ Mooncake Ascend Direct 安装完成，可用于 PD 分离部署"
else
    echo "⚠  存在未通过项，请检查上述输出"
fi
```

执行方式：

```bash
bash install_mooncake_ascend.sh
```

---

## 4. 环境变量配置

安装完成后，使用前需确保以下环境变量已设置：

```bash
# Mooncake 库路径（必须）
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-}

# Python 特定版本的库路径（根据实际 Python 版本调整）
export LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/mooncake:${LD_LIBRARY_PATH}
```

建议将这些加入容器启动脚本或 `.bashrc`。

---

## 5. pip list 中不显示的说明

`make install` 安装的是 C++ 库和 Python 模块文件到系统路径，**不注册 pip 元数据**，因此 `pip list` 不显示。

- `from mooncake.engine import TransferEngine` 能导入 = 安装成功
- vLLM-Ascend 运行时**不检查 pip 包注册**

如果确实需要在 `pip list` 中显示（例如依赖检查系统需要），**不推荐**使用 `mooncake-wheel/` 目录的包装器安装，因为它可能覆盖 `make install` 的真实模块导致导入失败。

---

## 6. mooncake.json 配置

PD 分离场景需要配置 `mooncake.json`，NPU 环境 `protocol` **必须**设为 `ascend`：

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "<MASTER_IP>:50088",
    "global_segment_size": "1GB"
}
```

| 字段 | 说明 |
|------|------|
| `metadata_server` | 配置为 `P2PHANDSHAKE` |
| `protocol` | NPU 上**必须**设为 `ascend`（走 ascend_direct_transport） |
| `device_name` | 留空 |
| `master_server_address` | Master 服务的 IP:Port |
| `global_segment_size` | 每张卡注册到 KV 池的内存大小，**需对齐到 1GB** |

通过环境变量 `MOONCAKE_CONFIG_PATH` 指向该文件：

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
```

---

## 7. 硬件适配环境变量

根据 Ascend 硬件型号设置对应环境变量：

| 硬件型号 | 条件 | 环境变量 | 说明 |
|---------|------|---------|------|
| 800 I/T **A3** 系列 | HDK >= 26.0.0, CANN >= 9.0.0 | `export ASCEND_ENABLE_USE_FABRIC_MEM=1` | 推荐。统一内存地址直传 |
| 800 I/T **A3** 系列 | 25.5.0 <= HDK < 26.0.0 | `export ASCEND_BUFFER_POOL=4:8` | 4 个 8MB 缓冲区 |
| 800 I/T **A2** 系列 | - | `export HCCL_INTRA_ROCE_ENABLE=1` | A2 系列直传必需 |

通用环境变量：

```bash
export PYTHONHASHSEED=0                    # 所有节点必须一致
export HCCL_RDMA_TIMEOUT=17                # RDMA 最小重传超时
export ASCEND_CONNECT_TIMEOUT=10000        # 单边通信连接超时（ms）
export ASCEND_TRANSFER_TIMEOUT=10000       # 单边通信传输超时（ms）
```

---

## 8. 启动 mooncake_master

```bash
mooncake_master \
    --port 50088 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000
```

| 参数 | 说明 |
|------|------|
| `--port` | Master 监听端口 |
| `--eviction_high_watermark_ratio` | 触发驱逐的水位线 |
| `--eviction_ratio` | 被驱逐的对象比例 |
| `--default_kv_lease_ttl` | KV 对象的默认租约 TTL（ms），须大于 `ASCEND_CONNECT_TIMEOUT` 和 `ASCEND_TRANSFER_TIMEOUT` |

---

## 9. 已知问题与社区坑点

### 9.1 编译期问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `pybind11/CMakeLists.txt does not exist` | `git clone --depth 1` 不拉取 submodule | `git submodule update --init --recursive` |
| `Failed to download yalantinglibs` | Pod 网络无法访问 GitHub releases | 手动下载 0.5.7.zip 并编译安装（见脚本步骤 5） |
| `_mm_pause` 编译错误 | Intel 专有指令，ARM 不支持 | Mooncake >= v0.3.8.post1 已修复（PR#1313） |
| `dependencies.sh` 需要 Go 1.20+ | etcd 编译需要 | 可跳过：`-DUSE_ETCD=OFF` |
| `Could not resolve host: github.com` | Pod 内 DNS 无法解析外网地址 | 配置 `http_proxy` / `https_proxy` 环境变量 |
| `server certificate verification failed` | 代理导致 SSL 中间人验证失败 | `export GIT_SSL_NO_VERIFY=1` + `git config --global http.sslVerify false` |

### 9.2 运行期问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `corrupted size vs. prev_size` | segment descriptor 竞态条件 | 升级到 v0.3.10.post1（含 PR#1599 修复） |
| `free(): invalid pointer` | 初始化失败时的内存管理 bug | 升级到最新版（#1601） |
| 跨架构 RPC 反序列化失败（ARM↔x86） | 内存对齐方式不同 | 确保 P/D 节点使用相同架构（#1573，仍未修复） |
| ctypes 加载 `ascend_transport.so` 报 undefined symbol | ctypes 默认 `RTLD_LOCAL` 模式不共享符号 | **预期行为**，不影响 vLLM-Ascend 运行时 |
| `pip list` 不显示 mooncake | `make install` 不注册 pip 元数据 | 正常现象，不影响功能 |

### 9.3 版本选择建议

| 场景 | 推荐版本 | 说明 |
|------|---------|------|
| 生产环境 | v0.3.10.post1 | 包含竞态条件修复 |
| 保守环境 | v0.3.9 | vLLM-Ascend 官方文档推荐的最低版本 |
| 开发测试 | main 分支 | 最新特性 |

---

## 10. HIXL 故障排查

Ascend Direct 后端基于 HIXL（华为互联库），遇到传输层问题可参考官方故障手册：

- [HIXL 常见问题定位手册](https://gitcode.com/cann/hixl/wiki/HIXL%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E6%89%8B%E5%86%8C.md)

常见排查命令：

```bash
# 检查 NPU HCCN 配置（Docker 中需挂载）
cat /etc/hccn.conf

# 检查 NPU 设备状态
npu-smi info

# 检查 RDMA 网卡
ibstat 2>/dev/null || echo "libibverbs-utils 未安装"
```

---

## 11. 容器持久化建议

容器中 `make install` 的产物在容器重启后会丢失。建议：

1. **Docker Commit**：安装成功后 `docker commit` 保存为新镜像
2. **Dockerfile 固化**：将安装脚本写入 Dockerfile 的构建阶段
3. **共享卷**：将编译产物（`/usr/local/lib/libmooncake*`、`/usr/local/lib/ascend_transport.so`、`/usr/local/bin/mooncake_master`）拷贝到持久化存储

---

## 12. 参考链接

- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- [Mooncake Build Guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html)
- [vLLM-Ascend KV Pool 部署指南](https://docs.vllm.ai/projects/ascend/zh-cn/main/user_guide/feature_guide/kv_pool.html)
- [PyPI: mooncake-transfer-engine-non-cuda](https://pypi.org/project/mooncake-transfer-engine-non-cuda/)（仅 x86_64）
- [PR #1599 - 修复 removeSegmentDesc 竞态条件](https://github.com/kvcache-ai/Mooncake/pull/1599)
- [Issue #1242 - ARM _mm_pause 修复](https://github.com/kvcache-ai/Mooncake/issues/1242)
- [Issue #1573 - ARM↔x86 跨架构 RPC 问题](https://github.com/kvcache-ai/Mooncake/issues/1573)
