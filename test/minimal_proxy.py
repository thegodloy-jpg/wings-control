#!/usr/bin/env python3
"""Minimal proxy — replicates core wings proxy forwarding behavior.

Multi-worker uvicorn + HTTP/SSE relay, for benchmarking worker count impact on
TTFT, TPS, CPU, and memory WITHOUT full wings_control dependencies.

Key: uses aiter_raw() + small chunk reads to avoid SSE buffering.

Usage:
    PROXY_WORKERS=4 BACKEND_URL=http://127.0.0.1:17000 \
        python3 -m uvicorn minimal_proxy:app --host 0.0.0.0 --port 18000 --workers 4
"""
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:17000").rstrip("/")
WORKERS = int(os.getenv("PROXY_WORKERS", "4"))

# Connection pool mimicking wings proxy defaults
_pool = httpx.AsyncClient(
    base_url=BACKEND_URL,
    limits=httpx.Limits(
        max_connections=2048,
        max_keepalive_connections=256,
        keepalive_expiry=30,
    ),
    timeout=httpx.Timeout(connect=20, read=None, write=20, pool=30),
)

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    is_stream = body.get("stream", False)

    if is_stream:
        # Use stream context manager for proper SSE relay
        # aiter_raw() avoids content-encoding buffering
        async def relay():
            async with _pool.stream(
                "POST", "/v1/chat/completions",
                json=body,
                headers={"content-type": "application/json", "accept": "text/event-stream"},
            ) as backend_resp:
                async for chunk in backend_resp.aiter_raw():
                    yield chunk

        return StreamingResponse(
            relay(),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming: forward and relay JSON
        resp = await _pool.post(
            "/v1/chat/completions",
            json=body,
            headers={"content-type": "application/json"},
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/v1/models")
async def models():
    resp = await _pool.get("/v1/models")
    return JSONResponse(content=resp.json())


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.on_event("shutdown")
async def shutdown():
    await _pool.aclose()
