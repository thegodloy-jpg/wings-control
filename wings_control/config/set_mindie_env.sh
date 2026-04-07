#!/bin/bash
# =============================================================================
# MindIE 单机引擎环境初始化脚本
# 用途: 被 _build_base_env_commands() 读取并内联到 start_command.sh
# 来源: 参考 wings/config/set_mindie_single_env.sh，适配 sidecar 架构
#
# 注意: 此脚本在 engine 容器内执行，不是在 wings-control 容器内。
# =============================================================================

# set +u: CANN 环境脚本引用未绑定变量
set +u
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    || echo 'WARN: ascend-toolkit/set_env.sh not found'
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] \
    && source /usr/local/Ascend/nnal/atb/set_env.sh \
    || echo 'WARN: nnal/atb/set_env.sh not found'
[ -f /usr/local/Ascend/mindie/set_env.sh ] \
    && source /usr/local/Ascend/mindie/set_env.sh \
    || echo 'WARN: mindie/set_env.sh not found'
[ -f /opt/atb-models/set_env.sh ] \
    && source /opt/atb-models/set_env.sh \
    || echo 'WARN: atb-models/set_env.sh not found'
set -u

export NPU_MEMORY_FRACTION=0.96
