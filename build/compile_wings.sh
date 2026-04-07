#!/bin/bash

# 进入工程目录
cd /opt/wings-control || exit 1

# 1. 编译所有Python文件（生成.pyc）
echo "Compiling Python files to .pyc..."
python -m compileall -b -f .  # -b: 生成同目录.pyc -f: 强制重新编译

# 2. 安全删除所有.py文件（包括__init__.py）
echo "Cleaning up all .py files..."
find . -name "*.py" -exec rm -f {} +

# 3. 设置权限
chmod +x /opt/wings-control/wings_start.sh

echo "Operation completed successfully!"