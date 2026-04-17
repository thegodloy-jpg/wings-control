#!/usr/bin/env python3
"""Mock vLLM backend — returns fixed chat completions (stream & non-stream).

Listens on port 17000 by default. Used to isolate proxy performance from engine.
"""
import json
import time
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI()

MODEL_NAME = "mock-model"

NON_STREAM_RESP = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": MODEL_NAME,
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! This is a mock response for benchmark testing."},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
}


def _stream_chunks():
    """Yield SSE chunks mimicking vLLM streamed output."""
    words = "Hello this is a mock streamed response for benchmark testing period end of sentence".split()
    for i, w in enumerate(words):
        delta = {"role": "assistant"} if i == 0 else {}
        delta["content"] = w + " "
        chunk = {
            "id": "chatcmpl-mock-stream",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    final = {
        "id": "chatcmpl-mock-stream",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("stream"):
        return StreamingResponse(_stream_chunks(), media_type="text/event-stream")
    return JSONResponse(NON_STREAM_RESP)


@app.get("/v1/models")
async def models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "mock"}],
    })


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=17000, log_level="warning")
