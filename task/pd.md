# PD 分离 - Mooncake 升级方案

## 当前环境

| 组件 | 版本 |
|------|------|
| vllm-ascend | 0.18rc1 |
| mooncake-transfer-engine | 0.3.9 |
| 硬件 | Ascend 910B2C (aarch64) |

## 问题描述

PD 分离运行时出现 `corrupted size vs. prev_size` 堆内存损坏错误，根因是 mooncake 0.3.9 中残留的竞态条件。

## 升级目标

升级 mooncake-transfer-engine 从 **0.3.9 → 0.3.10.post1**

### 为什么需要升级

| PR | 修复内容 | 所在版本 | 当前状态 |
|----|---------|---------|---------|
| #1373 | `receivePeerMetadata` 和 `getSegmentDesc` 竞态条件，使用 `RWSpinlock::ReadGuard` 保护 `segment_id_to_desc_map_` | v0.3.9 | ✅ 已包含 |
| #1599 | `removeSegmentDesc` 和 `updateLocalSegmentDesc` 残留竞态条件，`WriteGuard` 修复 + `operator[]` 替换为 `find()` | v0.3.10 | ❌ **未包含** |

PR #1599 是 #1373 的后续补丁，修复了 segment descriptor 管理中剩余的竞态条件。在高并发 PD 场景下，`removeSegmentDesc` 未持锁导致的 use-after-free/double-free 是 `corrupted size vs. prev_size` 的典型触发源。

---

## 升级步骤

### 步骤 0：环境检查

```bash
# 进入容器后执行

# 1. 确认当前 mooncake 版本
pip show mooncake-transfer-engine 2>/dev/null || pip show mooncake-transfer-engine-non-cuda 2>/dev/null

# 2. 确认 Python 版本（需要 3.10/3.11/3.12/3.13）
python3 --version

# 3. 确认 glibc 版本（aarch64 wheel 要求 >= 2.39）
ldd --version 2>&1 | head -1

# 4. 确认架构
uname -m  # 应输出 aarch64
```

**关键判断点：glibc 版本**
- `glibc >= 2.39` → 可以直接 pip 安装预编译 wheel（方案 A）
- `glibc < 2.39` → 必须从源码编译（方案 B）

---

### 方案 A：pip 直接升级（glibc >= 2.39）

Ascend 环境无 CUDA，使用 `non-cuda` 变体：

```bash
# 卸载旧版
pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-non-cuda

# 安装新版（non-cuda 变体，适用于 Ascend）
pip install mooncake-transfer-engine-non-cuda==0.3.10.post1

# 验证版本
python3 -c "import mooncake; print(mooncake.__version__)"
```

> **注意**: 如果容器原来安装的是 `mooncake-transfer-engine`（CUDA 版），需要先卸载再安装 non-cuda 版。也可安装 CUDA 版 `mooncake-transfer-engine==0.3.10.post1`，但 Ascend 环境无需 CUDA 特性。

---

### 方案 B：从源码编译（glibc < 2.39）

```bash
# 1. 安装编译依赖
apt-get update && apt-get install -y \
    build-essential cmake git \
    libibverbs-dev libgoogle-glog-dev libgtest-dev \
    libjsoncpp-dev libnuma-dev libunwind-dev \
    libpython3-dev libboost-all-dev libssl-dev \
    pybind11-dev libcurl4-openssl-dev libhiredis-dev \
    pkg-config patchelf

# 2. 克隆指定版本源码
cd /tmp
git clone --branch v0.3.10.post1 --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake

# 3. 安装依赖（如网络可达）
bash dependencies.sh

# 4. 编译（Ascend 场景：关闭 CUDA）
mkdir build && cd build
cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_REDIS=ON \
    -DUSE_HTTP=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF
make -j$(nproc)

# 5. 安装
sudo make install

# 6. 验证
python3 -c "import mooncake; print(mooncake.__version__)"
```

> **说明**: Ascend Direct Transport 会在检测到 CANN SDK 时自动启用，无需额外 cmake 参数。

---

### 方案 C：在容器外构建 wheel 再拷入

如果容器内编译环境受限，可在同架构的构建机上编译 wheel：

```bash
# 在 aarch64 构建机上
cd /tmp/Mooncake
pip wheel . --no-deps -w /tmp/wheels/

# 拷贝到目标机器
scp /tmp/wheels/mooncake_transfer_engine-*.whl target-host:/tmp/

# 在容器内安装
pip install /tmp/mooncake_transfer_engine-*.whl
```

---

## 升级后验证

### 1. 版本确认

```bash
python3 -c "
try:
    import mooncake
    print(f'mooncake version: {mooncake.__version__}')
except Exception as e:
    print(f'import error: {e}')
"
```

### 2. Transfer Engine 基本功能验证

```bash
# 确认 transfer engine 模块可正常加载
python3 -c "
from mooncake.engine import TransferEngine
print('TransferEngine import OK')
"
```

### 3. PD 分离功能测试

按正常流程启动 prefill 和 decode 实例，观察是否仍出现 `corrupted size vs. prev_size`。

建议先用小模型 + 少量并发（如 2-4 并发）验证稳定性，再逐步提升负载。

---

## 注意事项

1. **包名选择**：Ascend 环境建议使用 `mooncake-transfer-engine-non-cuda`，避免引入不必要的 CUDA 依赖
2. **glibc 兼容性**：aarch64 预编译 wheel 依赖 `glibc >= 2.39`（manylinux_2_39），较老的 OS 基础镜像可能不满足
3. **vllm-ascend 兼容性**：mooncake 0.3.10.post1 对 vllm-ascend 0.18rc1 的 API 应保持兼容，Transfer Engine 接口在 0.3.x 系列内向后兼容
4. **回滚方案**：如升级后出现新问题，可回退到 0.3.9：
   ```bash
   pip install mooncake-transfer-engine-non-cuda==0.3.9
   ```
5. **jemalloc 缓解**：如果升级后问题仍存在，可尝试 preload jemalloc 作为额外缓解手段：
   ```bash
   apt-get install -y libjemalloc2
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
   ```
6. **镜像持久化**：容器内 pip 安装在重启后会丢失，建议升级成功后 `docker commit` 保存镜像或在 Dockerfile 中固化版本

---

## 参考链接

- [PR #1373 - Fix race condition in receivePeerMetadata and getSegmentDesc](https://github.com/kvcache-ai/Mooncake/pull/1373)
- [PR #1599 - Fix remaining race conditions in removeSegmentDesc and updateLocalSegmentDesc](https://github.com/kvcache-ai/Mooncake/pull/1599)
- [Mooncake Build Guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html)
- [PyPI: mooncake-transfer-engine-non-cuda](https://pypi.org/project/mooncake-transfer-engine-non-cuda/)
- [vLLM-Ascend Disaggregated Prefill 文档](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/feature_guide/disaggregated_prefill.html)