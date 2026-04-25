# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""wings_control 包入口 —— 支持 python -m wings_control 方式启动。

当使用 ``python -m wings_control`` 时，Python 会执行本文件。
本文件将调用 wings_control.py 中定义的 run() 函数完成实际启动。
"""

import sys
from . import run

if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))  # 主进程入口，G.ERR.11 例外  # noqa: avoid-using-exit
