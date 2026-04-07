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

