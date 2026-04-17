#!/usr/bin/env python3
"""Benchmark client — 测量不同 PROXY_WORKERS 配置下 proxy 的延迟和吞吐量。

支持:
  - 串行延迟测试 (latency mode)
  - 并发吞吐测试 (throughput mode)
  - 流式 & 非流式场景
  - 内存采样

Usage:
    python bench_workers.py --url http://localhost:18000 --label "workers-4" \
        --output results_w4.json -n 300 --warmup 30 --concurrency 1,10,50,100
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from typing import List, Dict, Any

import httpx


# ── Payload generators ──────────────────────────────────────────────

def payload_normal():
    return {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "max_tokens": 50,
        "stream": False,
    }


def payload_stream():
    return {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a story."},
        ],
        "max_tokens": 100,
        "stream": True,
    }


def payload_large():
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(50):
        msgs.append({"role": "user", "content": f"Message {i}: " + "x" * 200})
        msgs.append({"role": "assistant", "content": f"Reply {i}: " + "y" * 200})
    msgs.append({"role": "user", "content": "Final question"})
    return {
        "model": "mock-model",
        "messages": msgs,
        "max_tokens": 50,
        "stream": False,
    }


def payload_tool_calls():
    return {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'}}
            ]},
            {"role": "tool", "content": '{"temp": 15}', "tool_call_id": "call_1"},
        ],
        "max_tokens": 50,
        "stream": False,
    }


SCENARIOS = {
    "normal": payload_normal,
    "stream": payload_stream,
    "large_messages": payload_large,
    "tool_calls": payload_tool_calls,
}


# ── Stats ───────────────────────────────────────────────────────────

def compute_stats(latencies: List[float]) -> Dict[str, Any]:
    if not latencies:
        return {"count": 0}
    s = sorted(latencies)
    return {
        "count": len(s),
        "mean_ms": round(statistics.mean(s), 3),
        "median_ms": round(statistics.median(s), 3),
        "p50_ms": round(s[len(s) // 2], 3),
        "p90_ms": round(s[int(len(s) * 0.90)], 3),
        "p95_ms": round(s[int(len(s) * 0.95)], 3),
        "p99_ms": round(s[min(int(len(s) * 0.99), len(s) - 1)], 3),
        "min_ms": round(min(s), 3),
        "max_ms": round(max(s), 3),
        "stdev_ms": round(statistics.stdev(s), 3) if len(s) > 1 else 0,
    }


# ── Serial benchmark ───────────────────────────────────────────────

def bench_serial_nonstream(client: httpx.Client, url: str, payload: dict, n: int):
    latencies = []
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            r = client.post(f"{url}/v1/chat/completions", json=payload, timeout=30)
            r.raise_for_status()
        except Exception:
            errors += 1
            continue
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies, errors


def bench_serial_stream(client: httpx.Client, url: str, payload: dict, n: int):
    ttft_list = []
    total_list = []
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            with client.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=30) as r:
                r.raise_for_status()
                first = True
                for line in r.iter_lines():
                    if first and line.startswith("data:"):
                        ttft_list.append((time.perf_counter() - t0) * 1000)
                        first = False
        except Exception:
            errors += 1
            continue
        total_list.append((time.perf_counter() - t0) * 1000)
    return ttft_list, total_list, errors


# ── Concurrent benchmark ───────────────────────────────────────────

async def _async_request(client: httpx.AsyncClient, url: str, payload: dict):
    t0 = time.perf_counter()
    try:
        if payload.get("stream"):
            async with client.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=30) as r:
                r.raise_for_status()
                async for _ in r.aiter_lines():
                    pass
        else:
            r = await client.post(f"{url}/v1/chat/completions", json=payload, timeout=30)
            r.raise_for_status()
        return (time.perf_counter() - t0) * 1000, None
    except Exception as e:
        return (time.perf_counter() - t0) * 1000, str(e)


async def bench_concurrent(url: str, payload: dict, total_requests: int, concurrency: int):
    """Run total_requests with up to `concurrency` in-flight at once."""
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async with httpx.AsyncClient() as client:
        async def worker():
            nonlocal errors
            async with sem:
                lat, err = await _async_request(client, url, payload)
                if err:
                    errors += 1
                else:
                    latencies.append(lat)

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
        wall_time = (time.perf_counter() - t0) * 1000

    qps = len(latencies) / (wall_time / 1000) if wall_time > 0 else 0
    return latencies, errors, wall_time, qps


# ── Main ────────────────────────────────────────────────────────────

def run_all(url: str, label: str, n: int, warmup: int, concurrencies: List[int]):
    results: Dict[str, Any] = {
        "label": label,
        "url": url,
        "iterations": n,
        "warmup": warmup,
        "concurrencies": concurrencies,
        "serial": {},
        "concurrent": {},
    }

    client = httpx.Client()

    # ── Serial tests ──
    print(f"\n{'='*60}")
    print(f"  Serial latency tests (label={label})")
    print(f"{'='*60}")

    for name, gen_fn in SCENARIOS.items():
        payload = gen_fn()
        is_stream = payload.get("stream", False)
        print(f"  [{label}] {name} ({'stream' if is_stream else 'non-stream'}) "
              f"warmup={warmup}, n={n}")

        # warmup
        for _ in range(warmup):
            try:
                if is_stream:
                    with client.stream("POST", f"{url}/v1/chat/completions",
                                       json=payload, timeout=30) as r:
                        for _ in r.iter_lines():
                            pass
                else:
                    client.post(f"{url}/v1/chat/completions", json=payload, timeout=30)
            except Exception:
                pass

        if is_stream:
            ttft, total, errs = bench_serial_stream(client, url, payload, n)
            results["serial"][name] = {
                "type": "stream",
                "ttft": compute_stats(ttft),
                "total": compute_stats(total),
                "errors": errs,
            }
            t = results["serial"][name]["ttft"]
            print(f"    TTFT: mean={t.get('mean_ms','N/A')}ms p95={t.get('p95_ms','N/A')}ms errors={errs}")
        else:
            lats, errs = bench_serial_nonstream(client, url, payload, n)
            results["serial"][name] = {
                "type": "non-stream",
                "latency": compute_stats(lats),
                "errors": errs,
            }
            s = results["serial"][name]["latency"]
            print(f"    mean={s.get('mean_ms','N/A')}ms p95={s.get('p95_ms','N/A')}ms errors={errs}")

    client.close()

    # ── Concurrent tests ──
    print(f"\n{'='*60}")
    print(f"  Concurrent throughput tests (label={label})")
    print(f"{'='*60}")

    for c in concurrencies:
        payload = payload_normal()
        total_reqs = max(n, c * 5)  # at least 5x concurrency
        print(f"  [{label}] concurrency={c}, total_requests={total_reqs} (non-stream)")
        lats, errs, wall_ms, qps = asyncio.run(
            bench_concurrent(url, payload, total_reqs, c)
        )
        results["concurrent"][str(c)] = {
            "concurrency": c,
            "total_requests": total_reqs,
            "latency": compute_stats(lats),
            "errors": errs,
            "wall_time_ms": round(wall_ms, 3),
            "qps": round(qps, 2),
        }
        print(f"    QPS={qps:.1f} mean={compute_stats(lats).get('mean_ms','N/A')}ms "
              f"p95={compute_stats(lats).get('p95_ms','N/A')}ms errors={errs} wall={wall_ms:.0f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Proxy worker performance benchmark")
    parser.add_argument("--url", default="http://localhost:18000")
    parser.add_argument("--label", default="test")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("-n", type=int, default=300, help="Iterations per serial scenario")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--concurrency", default="1,10,50,100",
                        help="Comma-separated concurrency levels for throughput test")
    args = parser.parse_args()

    concurrencies = [int(x.strip()) for x in args.concurrency.split(",")]
    results = run_all(args.url, args.label, args.n, args.warmup, concurrencies)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
