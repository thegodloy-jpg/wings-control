"""wings-control 统一推理 sidecar 后端应用的顶层包。

项目架构概述:
    wings-control 是一个运行在 Kubernetes 中的 sidecar 容器，负责协调推理引擎的
    启动配置、API 代理转发和健康状态监控。整体分为三大子系统：

    1. launcher (wings_control.py) —— 解析参数、生成引擎启动脚本、托管 proxy/health 子进程
    2. proxy   (proxy/)  —— 对外暴露 OpenAI 兼容 API，转发请求到后端引擎
    3. health  (proxy/)  —— 独立健康检查服务，供 Kubernetes 探针使用

支持的推理引擎: vllm, vllm_ascend, sglang, mindie
端口规划: backend=17000, proxy=18000, health=19000

设计原则:
    - 包初始化保持轻量，不产生运行时副作用
    - 避免在 __init__.py 中引入重型依赖，确保导入速度快
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

_ENTRY_MODULE_NAME = "wings_control._launcher_entry"


def _resolve_entry_module_path() -> Path:
    """Locate the launcher entry module across flat and nested layouts."""
    base_dir = Path(__file__).resolve().parent
    candidates = (
        base_dir / "wings_control.py",
        base_dir / "wings_control" / "wings_control.py",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise ImportError(
        "Cannot locate launcher entry module 'wings_control.py' "
        f"next to package root {base_dir}"
    )


def _load_entry_module() -> ModuleType:
    """Load and cache the launcher entry module by file path.

    This avoids ambiguity between the sibling module ``wings_control.py`` and
    any nested package named ``wings_control`` when the code is mounted in
    different layouts.
    """
    cached = sys.modules.get(_ENTRY_MODULE_NAME)
    if cached is not None:
        return cached

    module_path = _resolve_entry_module_path()
    spec = importlib.util.spec_from_file_location(_ENTRY_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_ENTRY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def run(argv: Sequence[str] | None = None) -> int:
    """Package-level launcher entry used by ``python -m wings_control``."""
    return _load_entry_module().run(argv)


__all__ = ["run"]

