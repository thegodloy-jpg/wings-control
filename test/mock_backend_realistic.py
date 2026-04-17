#!/usr/bin/env python3
"""Mock vLLM backend with realistic inference delays.

Simulates real LLM behavior:
  - Prefill delay  → contributes to TTFT (Time To First Token)
  - Per-token delay → determines TPS (Tokens Per Second) from engine side
  - Configurable via environment variables or query params

Env vars:
  PREFILL_MS    : Engine prefill latency in ms (default: 100)
  TOKEN_DELAY_MS: Per-token generation delay in ms (default: 30 → ~33 tok/s)
  NUM_TOKENS    : Number of output tokens per response (default: 50)

Usage:
    python mock_backend_realistic.py [--port 17000] [--prefill-ms 100] [--token-delay-ms 30] [--num-tokens 50]
"""
import argparse
import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI()

MODEL_NAME = "mock-model"

# Configurable via env or CLI
PREFILL_MS = float(os.getenv("PREFILL_MS", "100"))
TOKEN_DELAY_MS = float(os.getenv("TOKEN_DELAY_MS", "30"))
NUM_TOKENS = int(os.getenv("NUM_TOKENS", "50"))


VOCAB = (
    "The quick brown fox jumps over the lazy dog and then runs across the field "
    "while the sun sets behind the mountains creating a beautiful golden horizon "
    "that stretches endlessly into the distance where clouds drift slowly past "
).split()


def _make_non_stream_response(n_tokens: int) -> dict:
    """Build a non-stream chat completion response."""
    content = " ".join(VOCAB[i % len(VOCAB)] for i in range(n_tokens))
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": n_tokens,
            "total_tokens": 20 + n_tokens,
        },
    }


async def _stream_generator(prefill_ms: float, token_delay_ms: float, n_tokens: int):
    """Yield SSE chunks with realistic timing."""
    # Prefill phase — simulates KV cache computation
    await asyncio.sleep(prefill_ms / 1000.0)

    # First token
    first_chunk = {
        "id": "chatcmpl-mock-stream",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": VOCAB[0]},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    # Subsequent tokens — each with per-token delay
    for i in range(1, n_tokens):
        await asyncio.sleep(token_delay_ms / 1000.0)
        chunk = {
            "id": "chatcmpl-mock-stream",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": {"content": " " + VOCAB[i % len(VOCAB)]},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final = {
        "id": "chatcmpl-mock-stream",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": n_tokens,
            "total_tokens": 20 + n_tokens,
        },
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    # Allow per-request override via body
    prefill = body.get("_prefill_ms", PREFILL_MS)
    token_delay = body.get("_token_delay_ms", TOKEN_DELAY_MS)
    n_tokens = body.get("max_tokens", NUM_TOKENS)

    if body.get("stream"):
        return StreamingResponse(
            _stream_generator(prefill, token_delay, n_tokens),
            media_type="text/event-stream",
        )
    else:
        # Non-stream: simulate total generation time
        total_delay = prefill + token_delay * n_tokens
        await asyncio.sleep(total_delay / 1000.0)
        return JSONResponse(_make_non_stream_response(n_tokens))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=17000)
    parser.add_argument("--prefill-ms", type=float, default=None)
    parser.add_argument("--token-delay-ms", type=float, default=None)
    parser.add_argument("--num-tokens", type=int, default=None)
    args = parser.parse_args()

    if args.prefill_ms is not None:
        PREFILL_MS = args.prefill_ms
    if args.token_delay_ms is not None:
        TOKEN_DELAY_MS = args.token_delay_ms
    if args.num_tokens is not None:
        NUM_TOKENS = args.num_tokens

    print(f"Mock backend starting on port {args.port}")
    print(f"  PREFILL_MS={PREFILL_MS}, TOKEN_DELAY_MS={TOKEN_DELAY_MS}, NUM_TOKENS={NUM_TOKENS}")
    print(f"  Expected engine TTFT ≈ {PREFILL_MS}ms")
    print(f"  Expected engine TPS ≈ {1000/TOKEN_DELAY_MS:.1f} tok/s (per request)")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
