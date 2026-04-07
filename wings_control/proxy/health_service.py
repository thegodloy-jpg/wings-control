# -*- coding: utf-8 -*-
"""独立健康服务。

与 gateway 中的 `/health` 不同，这个模块单独跑在健康端口上，
便于 Kubernetes 探针在 proxy 高负载时仍然可靠读取健康状态。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from config.settings import settings

import httpx
import uvicorn
from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse

from core.version_util import normalize_engine_version
from proxy.health_router import (
    _jittered_sleep_base,
    build_health_body,
    build_v1_health_body,
    build_health_headers,
    init_health_state,
    map_http_code_from_state,
    teardown_health_monitor,
    tick_observe_and_advance,
)
from utils.log_config import setup_root_logging, LOGGER_HEALTH
from proxy.speaker_logging import configure_worker_logging
from utils.progress_utils import StartupProgressManager

setup_root_logging(stderr_level="ERROR")
_logger = logging.getLogger(LOGGER_HEALTH)

# 配置 worker 日志：归一化 uvicorn/httpx 子 logger 格式，
# 安装 /health 日志过滤器以抑制 httpx 高频探活噪声。
configure_worker_logging()

# health 服务的 httpx 活动仅有后端探活轮询，全部是低价值重复日志。
# 将 httpx 日志级别提升至 WARNING，彻底消除噪声。
# 注意：设置父 logger "httpx" 的级别会通过 effective level 影响所有子 logger
# （如 httpx._client），因此无需逐个设置。
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 单独的 FastAPI 应用，通常监听 `HEALTH_SERVICE_PORT`。
app = FastAPI()

# 独立健康服务对外监听端口，通常由 launcher 注入。
HEALTH_SERVICE_PORT = int(os.getenv("HEALTH_SERVICE_PORT", "19000"))


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化健康服务所需的资源。

    初始化内容包括：
      1. 创建异步 HTTP 客户端，用于轮询后端 /health 接口。
      2. 初始化健康状态字典（分数、连续状态计数等）。
      3. 初始化启动进度管理器。
      4. 启动后台健康轮询任务。
    """
    app.state.client = httpx.AsyncClient()
    app.state.health = init_health_state()
    app.state.progress_manager = StartupProgressManager(os.getenv("ENGINE", "vllm"))
    app.state.health_task = asyncio.create_task(health_monitor_loop(), name="health-monitor")


async def health_monitor_loop():
    """后台健康轮询循环，周期性探测后端引擎状态。

    不断调用 tick_observe_and_advance() 更新健康状态机，
    并根据当前状态动态调整轮询间隔（包含随机抱动以避免雷群效应）。
    发生异常时仅记录警告日志而不中断循环，确保健康探测始终运行。
    """
    while True:
        try:
            await tick_observe_and_advance(app.state.health, app.state.client)
        except Exception as e:
            _logger.warning("health_monitor_error: %s", e)
        await asyncio.sleep(_jittered_sleep_base(app.state.health))


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源。

    依次取消后台健康轮询任务，然后关闭异步 HTTP 客户端，
    确保连接池和文件句柄被正确释放。
    """
    await teardown_health_monitor(app)
    await app.state.client.aclose()


@app.get("/v1/health")
async def v1_health_check():
    """返回当前健康状态。

    根据健康状态机的当前状态映射为 HTTP 状态码（200/503），
    并在响应头中注入状态摘要信息。

    Returns:
        Response | JSONResponse: 健康检查响应，HTTP 200 表示健康，503 表示异常。
    """
    h = app.state.health
    code = map_http_code_from_state(h)
    headers = build_health_headers(h)

    body = build_v1_health_body(h, code)
    return JSONResponse(status_code=code, content=body, headers=headers)


@app.get("/health")
async def health_check(minimal: bool = False):
    """返回当前健康状态。

    根据健康状态机的当前状态映射为 HTTP 状态码（200/503），
    并在响应头中注入状态摘要信息。

    Args:
        minimal: 为 True 时返回空 body 的精简响应（仅状态码 + 头部），
            适用于 K8s livenessProbe。为 False 时返回包含详细分数、
            连续状态计数等信息的 JSON body。

    Returns:
        Response | JSONResponse: 健康检查响应，HTTP 200 表示健康，503 表示异常。
    """
    h = app.state.health
    code = map_http_code_from_state(h)
    headers = build_health_headers(h)

    if minimal:
        return Response(status_code=code, headers=headers)

    body = build_health_body(h, code)
    return JSONResponse(status_code=code, content=body, headers=headers)


@app.head("/health")
async def health_head():
    """轻量级 HEAD 健康接口，供 Kubernetes 探针使用。

    仅返回 HTTP 状态码和状态头部，不包含响应 body，
    最大限度减少健康探测的网络开销。

    Returns:
        Response: 空 body 响应，状态码 200（健康）或 503（异常）。
    """
    h = app.state.health
    code = map_http_code_from_state(h)
    headers = build_health_headers(h)
    return Response(status_code=code, headers=headers)


@app.get("/v1/startup/progress")
async def get_startup_progress():
    """获取部署进度信息

    Returns:
        JSONResponse: 包含进度信息的响应
    """
    from utils.progress_utils import read_progress_file, build_progress_response
    
    progress_file = settings.PROGRESS_FILE
    try:
        if os.path.exists(progress_file):
            file_progress = read_progress_file(progress_file)
            progress_data = app.state.progress_manager.update_from_file(file_progress)
        else:
            # 文件不存在，使用初始化信息
            progress_data = app.state.progress_manager.get_initial_progress_data()
        
        return JSONResponse(status_code=200, content=build_progress_response(progress_data))
    except Exception as e:
        _logger.error(f"Failed to get progress info: {e}")
        from utils.progress_utils import create_error_progress_data
        error_data = create_error_progress_data(e)
        response_body = build_progress_response(
            error_data, f"Failed to get progress info: {str(e)}"
        )
        return JSONResponse(status_code=200, content=response_body)


def _parse_accel_feature_line(line: str) -> dict | None:
    """解析加速特性文件的单行数据。

    Args:
        line: JSON 格式的行数据

    Returns:
        dict | None: 解析后的特性数据，解析失败返回 None
    """
    try:
        data = json.loads(line)
        return {
            "name": data.get("feature", ""),
            "enabled": data.get("status", "") == "success",
            "errMsg": data.get("error_msg", "")
        }
    except json.JSONDecodeError:
        return None


def _read_accel_features(file_path: str) -> list[dict]:
    """读取加速特性文件并解析所有特性。

    Args:
        file_path: 加速特性文件路径

    Returns:
        list[dict]: 特性列表
    """
    features = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feature = _parse_accel_feature_line(line)
            if not feature:
                continue
            features.append(feature)
    return features


def _build_accel_data(features: list[dict], version: str | None = None) -> dict:
    """构建加速特性响应数据。

    Args:
        features: 特性列表
        version: 引擎版本号。为 None 时从 ENGINE_VERSION 环境变量解析。

    Returns:
        dict: 加速特性数据
    """
    return {
        "engine": os.getenv("ENGINE", "vllm"),
        "version": version or normalize_engine_version(),
        "features": features
    }


def _build_accel_response(accel_data: dict, message: str = "") -> JSONResponse:
    """构建加速特性响应。

    Args:
        accel_data: 加速特性数据
        message: 响应消息

    Returns:
        JSONResponse: 加速特性响应
    """
    return JSONResponse(status_code=200, content={
        "code": 200,
        "msg": message,
        "data": accel_data
    })


async def _detect_engine_version(client: httpx.AsyncClient) -> str | None:
    """尝试从运行中的后端获取真实引擎版本号。

    向后端 /version 端点发起 GET 请求（vLLM/vLLM-Ascend 标准接口）。
    成功后将结果缓存在 app.state 中，后续调用直接返回缓存值。

    Returns:
        str | None: 引擎版本字符串（如 "0.17.0rc1"），失败时返回 None。
    """
    cached = getattr(app.state, "cached_engine_version", None)
    if cached:
        return cached
    try:
        url = f"http://{settings.ENGINE_HOST}:{settings.ENGINE_PORT}/version"
        resp = await client.get(url, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            version = data.get("version", "")
            if version:
                app.state.cached_engine_version = version
                return version
    except Exception as exc:  # noqa: BLE001
        _logger.debug("Failed to detect engine version: %s", exc)
    return None


@app.get("/v1/startup/accel")
async def get_startup_accel():
    """获取加速特性使能信息

    Returns:
        JSONResponse: 包含加速特性信息的响应
    """
    accel_file = settings.ACCEL_FILE
    features = []
    # 优先从运行中的后端获取真实版本号，回退到 ENGINE_VERSION 环境变量
    version = await _detect_engine_version(app.state.client)
    try:
        if os.path.exists(accel_file):
            features = _read_accel_features(accel_file)
        
        accel_data = _build_accel_data(features, version=version)
        return _build_accel_response(accel_data)
    except Exception as e:
        _logger.error(f"Failed to get acceleration feature info: {e}")
        accel_data = _build_accel_data([], version=version)
        return _build_accel_response(accel_data, f"Failed to get acceleration feature info: {str(e)}")


def run_standalone():
    """以独立进程方式启动健康服务，供本地开发调试使用。

    监听地址固定为 0.0.0.0，端口由环境变量 HEALTH_SERVICE_PORT 决定（默认 19000）。
    生产环境通常由 launcher 通过 uvicorn 启动，不使用此入口。
    """
    uvicorn.run(app, host="0.0.0.0", port=HEALTH_SERVICE_PORT)


if __name__ == "__main__":
    run_standalone()
