"""端口规划模块。

该项目把端口职责固定为三层：
- 17000: backend engine 监听端口（engine 容器内部服务地址）
- 18000: proxy 对外服务端口（K8s Service 暴露的端口）
- 19000: health 独立探针端口（K8s liveness/readiness 探针端口）

这里的职责是根据启动参数导出一个稳定的 `PortPlan`，让 launcher、proxy、
health 和 Kubernetes 清单都围绕同一套端口约定工作。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortPlan:
    """sidecar 四类端口的统一描述。

    Attributes:
        enable_proxy:  是否启用反向代理层。启用时外部流量经 proxy 转发到 backend；
                       禁用时外部直连 backend（仅限调试或特殊部署场景）。
        backend_port:  推理引擎实际监听端口（默认 17000），仅 sidecar 内部可访问。
        proxy_port:    反向代理对外暴露端口（默认 18000），K8s Service/Ingress 指向此端口。
        health_port:   健康检查服务端口（默认 19000），K8s liveness/readiness 探针对接此端口。
        monitor_proxy_port: 监控代理服务端口（默认 19100），用于透传 Engine 侧的监控接口。
    """
    enable_proxy: bool
    backend_port: int
    proxy_port: int
    health_port: int
    monitor_proxy_port: int


def derive_port_plan(*, port: int, enable_reason_proxy: bool, health_port: int = 19000,
                     monitor_proxy_port: int = 19100) -> PortPlan:
    """根据启动参数推导出完整的四层端口分配方案。

    Args:
        port:                 用户通过 --port 或 PORT 环境变量指定的对外端口。
        enable_reason_proxy:  是否启用反向代理（当前版本默认始终为 True）。
        health_port:          健康检查端口，默认 19000。
        monitor_proxy_port:   监控代理端口，默认 19100。

    Returns:
        PortPlan: 包含 backend/proxy/health/monitor_proxy 四层端口的完整方案。

    设计说明：
    - 启用 proxy 时：backend 固定 17000（内部），proxy 使用用户指定端口或 18000。
    - 禁用 proxy 时：backend 直接使用用户指定端口或 18000，proxy_port 置 0。
    - monitor_proxy 始终启用，用于透传 Engine 侧的监控接口（/metrics, /version 等）。
    """
    # 当前版本默认必须走 proxy，对外端口 `port` 只影响 proxy，不影响 backend。
    if enable_reason_proxy:
        return PortPlan(
            enable_proxy=True,
            backend_port=17000,
            proxy_port=port or 18000,
            health_port=health_port,
            monitor_proxy_port=monitor_proxy_port,
        )
    return PortPlan(
        enable_proxy=False,
        backend_port=port or 18000,
        proxy_port=0,
        health_port=health_port,
        monitor_proxy_port=monitor_proxy_port,
    )
