#!/usr/bin/env python3
"""Benchmark TTFT & TPS with resource monitoring for different PROXY_WORKERS configs.

Measures:
  - TTFT (Time To First Token): delay from request to first SSE chunk
  - TPS  (Tokens Per Second):   output token throughput per request
  - CPU  (% cores used by proxy processes during test)
  - MEM  (RSS in MB of proxy processes during test)

The benchmark uses streaming exclusively, since TTFT and TPS are streaming metrics.
Non-stream tests are measured as "total latency" for reference.

Usage:
    python bench_workers_v2.py \
        --proxy-url http://localhost:18000 \
        --direct-url http://localhost:17000 \
        --label "workers-4" \
        --proxy-pids 1234,1235,1236,1237 \
        --output results_w4.json \
        --concurrency 1,5,10,20,50 \
        -n 30
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

import httpx


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class StreamResult:
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    token_count: int = 0
    tps: float = 0.0  # tokens per second (decode phase only)
    error: Optional[str] = None


@dataclass
class ResourceSample:
    timestamp: float = 0.0
    cpu_percent: float = 0.0  # sum of per-process CPU % (multi-core aware)
    rss_mb: float = 0.0       # sum of per-process RSS in MB
    num_processes: int = 0


# ── Resource monitor ────────────────────────────────────────────────

class ResourceMonitor:
    """Sample CPU and memory of proxy processes at regular intervals."""

    def __init__(self, pids: List[int], interval_s: float = 0.5):
        self.pids = pids
        self.interval = interval_s
        self.samples: List[ResourceSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_proc_stat(self, pid: int) -> Optional[Tuple[float, float]]:
        """Read CPU time (user+system) and RSS from /proc/<pid>/stat."""
        try:
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().split()
            # utime=13, stime=14 (in clock ticks)
            utime = int(parts[13])
            stime = int(parts[14])
            cpu_ticks = utime + stime
            # rss=23 (in pages)
            rss_pages = int(parts[23])
            page_size = os.sysconf("SC_PAGE_SIZE")
            rss_mb = rss_pages * page_size / (1024 * 1024)
            return cpu_ticks, rss_mb
        except (FileNotFoundError, IndexError, ValueError):
            return None

    def _sample_once(self, prev_ticks: Dict[int, float], dt: float) -> ResourceSample:
        """Take one sample of all proxy processes."""
        clk_tck = os.sysconf("SC_CLK_TCK")
        total_cpu_pct = 0.0
        total_rss = 0.0
        alive = 0
        new_ticks = {}

        for pid in self.pids:
            result = self._read_proc_stat(pid)
            if result is None:
                continue
            cpu_ticks, rss_mb = result
            new_ticks[pid] = cpu_ticks
            total_rss += rss_mb
            alive += 1

            if pid in prev_ticks and dt > 0:
                dticks = cpu_ticks - prev_ticks[pid]
                # CPU% = (ticks_delta / clk_tck) / dt * 100
                cpu_pct = (dticks / clk_tck) / dt * 100
                total_cpu_pct += cpu_pct

        self._prev_ticks = new_ticks
        return ResourceSample(
            timestamp=time.time(),
            cpu_percent=round(total_cpu_pct, 2),
            rss_mb=round(total_rss, 2),
            num_processes=alive,
        )

    def _run(self):
        self._prev_ticks: Dict[int, float] = {}
        # Initial tick reading
        for pid in self.pids:
            result = self._read_proc_stat(pid)
            if result:
                self._prev_ticks[pid] = result[0]
        prev_time = time.time()

        while not self._stop.is_set():
            self._stop.wait(self.interval)
            now = time.time()
            dt = now - prev_time
            sample = self._sample_once(self._prev_ticks, dt)
            self.samples.append(sample)
            prev_time = now

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"cpu_avg_pct": 0, "cpu_max_pct": 0, "rss_avg_mb": 0, "rss_max_mb": 0, "samples": 0}
        cpus = [s.cpu_percent for s in self.samples]
        mems = [s.rss_mb for s in self.samples]
        return {
            "cpu_avg_pct": round(statistics.mean(cpus), 2),
            "cpu_max_pct": round(max(cpus), 2),
            "cpu_cores_avg": round(statistics.mean(cpus) / 100, 2),
            "cpu_cores_max": round(max(cpus) / 100, 2),
            "rss_avg_mb": round(statistics.mean(mems), 2),
            "rss_max_mb": round(max(mems), 2),
            "num_processes": self.samples[-1].num_processes if self.samples else 0,
            "samples": len(self.samples),
        }


# ── Prompt generation ──────────────────────────────────────────────

# English prose, avg ~1.3 tokens/word in common tokenizers
_LOREM = (
    "The quick brown fox jumps over the lazy dog while the sun slowly sets "
    "behind the towering mountains creating beautiful golden hues across the "
    "endless sky. Performance testing is a critical practice in software "
    "engineering that helps identify bottlenecks latency issues and resource "
    "consumption problems before they affect production users. In distributed "
    "systems understanding how different components scale under varying load "
    "conditions is essential for making informed architectural decisions. "
    "Proxy servers play a vital role in modern microservice architectures by "
    "providing load balancing request routing health checking and failover "
    "capabilities that improve both reliability and performance. When scaling "
    "worker processes the trade-off between memory consumption and request "
    "latency must be carefully evaluated to find the optimal configuration "
    "for each specific deployment scenario and workload pattern. "
    "Streaming responses present unique challenges for proxy architectures "
    "because each token must be forwarded with minimal buffering to preserve "
    "the real-time nature of the generation process. Server-sent events are "
    "the standard protocol for delivering incremental responses from large "
    "language model inference engines to client applications. "
)


def generate_prompt(target_tokens: int) -> str:
    """Generate a user message with approximately target_tokens tokens.
    English text ≈ 1.3 tokens/word, so we target ~target_tokens/1.3 words."""
    words = _LOREM.split()
    target_words = max(5, int(target_tokens / 1.3))
    # repeat until we have enough words
    result_words = []
    while len(result_words) < target_words:
        result_words.extend(words)
    return " ".join(result_words[:target_words])


# ── Streaming benchmark ────────────────────────────────────────────

_CACHED_PROMPT: Optional[str] = None


def _get_prompt(input_tokens: int) -> str:
    """Get (and cache) a prompt of approximately input_tokens length."""
    global _CACHED_PROMPT
    if _CACHED_PROMPT is None:
        _CACHED_PROMPT = generate_prompt(input_tokens)
    return _CACHED_PROMPT


async def bench_stream_single(client: httpx.AsyncClient, url: str,
                              max_tokens: int = 50,
                              input_tokens: int = 15) -> StreamResult:
    """Single streaming request — measure TTFT, total time, token count, TPS."""
    payload = {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": _get_prompt(input_tokens)},
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = 0.0
    token_count = 0

    try:
        async with client.stream("POST", f"{url}/v1/chat/completions",
                                 json=payload, timeout=120) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    token_count += 1
                    if token_count == 1:
                        ttft = (time.perf_counter() - t_start) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000
        decode_ms = total_ms - ttft
        tps = ((token_count - 1) / (decode_ms / 1000)) if decode_ms > 0 and token_count > 1 else 0

        return StreamResult(
            ttft_ms=round(ttft, 3),
            total_ms=round(total_ms, 3),
            token_count=token_count,
            tps=round(tps, 2),
        )
    except Exception as e:
        return StreamResult(error=str(e))


async def bench_concurrent_stream(url: str, concurrency: int, total_reqs: int,
                                  max_tokens: int = 50,
                                  input_tokens: int = 15) -> List[StreamResult]:
    """Run concurrent streaming requests."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient() as client:
        async def worker():
            async with sem:
                r = await bench_stream_single(client, url, max_tokens, input_tokens)
                results.append(r)

        tasks = [asyncio.create_task(worker()) for _ in range(total_reqs)]
        await asyncio.gather(*tasks)

    return results


# ── Stats helpers ───────────────────────────────────────────────────

def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    return {
        "count": len(s),
        "mean": round(statistics.mean(s), 3),
        "median": round(statistics.median(s), 3),
        "p50": round(s[len(s) // 2], 3),
        "p90": round(s[int(len(s) * 0.9)], 3),
        "p95": round(s[int(len(s) * 0.95)], 3),
        "p99": round(s[min(int(len(s) * 0.99), len(s) - 1)], 3),
        "min": round(min(s), 3),
        "max": round(max(s), 3),
        "stdev": round(statistics.stdev(s), 3) if len(s) > 1 else 0,
    }


def analyze_results(results: List[StreamResult]) -> Dict[str, Any]:
    """Compute stats from a list of StreamResult."""
    ok = [r for r in results if r.error is None]
    errors = len(results) - len(ok)

    ttft_vals = [r.ttft_ms for r in ok]
    tps_vals = [r.tps for r in ok if r.tps > 0]
    total_vals = [r.total_ms for r in ok]
    tokens = [r.token_count for r in ok]

    return {
        "requests": len(results),
        "success": len(ok),
        "errors": errors,
        "ttft": compute_stats(ttft_vals),
        "tps": compute_stats(tps_vals),
        "total_ms": compute_stats(total_vals),
        "tokens": compute_stats([float(t) for t in tokens]),
    }


# ── PID discovery ──────────────────────────────────────────────────

def find_proxy_pids(port: int = 18000) -> List[int]:
    """Find all uvicorn worker PIDs serving the proxy on the given port.
    Finds master PID then all child worker processes."""
    try:
        # Find master PID
        out = subprocess.check_output(
            ["bash", "-c", f"ps aux | grep 'uvicorn.*--port.*{port}' | grep -v grep"],
            text=True, timeout=5
        ).strip()
        master_pid = None
        for line in out.split("\n"):
            parts = line.split()
            if len(parts) > 1:
                try:
                    master_pid = int(parts[1])
                    break
                except ValueError:
                    pass
        if not master_pid:
            return []

        # Find all child processes
        try:
            desc_out = subprocess.check_output(
                ["bash", "-c", f"pgrep -P {master_pid}"],
                text=True, timeout=5
            ).strip()
            child_pids = [int(p.strip()) for p in desc_out.split("\n") if p.strip()]
        except Exception:
            child_pids = []

        return [master_pid] + child_pids
    except Exception:
        return []


# ── Main test loop ──────────────────────────────────────────────────

def run_test(proxy_url: str, direct_url: Optional[str], label: str,
             proxy_pids: List[int], concurrencies: List[int],
             n_serial: int, n_concurrent: int, warmup: int,
             max_tokens: int, input_tokens: int = 15) -> Dict[str, Any]:
    """Run full test suite for one PROXY_WORKERS configuration."""

    results: Dict[str, Any] = {
        "label": label,
        "proxy_url": proxy_url,
        "direct_url": direct_url,
        "input_tokens": input_tokens,
        "max_tokens": max_tokens,
        "n_serial": n_serial,
        "n_concurrent": n_concurrent,
        "warmup": warmup,
        "concurrencies": concurrencies,
        "proxy_pids": proxy_pids,
        "idle_resources": {},
        "serial": {},
        "concurrent": {},
    }

    # ── Idle resource snapshot ──
    print(f"\n{'='*60}")
    print(f"  [{label}] Idle resource snapshot (proxy PIDs: {proxy_pids})")
    print(f"{'='*60}")

    if proxy_pids:
        mon = ResourceMonitor(proxy_pids, interval_s=0.5)
        mon.start()
        time.sleep(3)  # sample idle for 3 seconds
        mon.stop()
        results["idle_resources"] = mon.summary()
        print(f"  Idle CPU: {mon.summary()['cpu_avg_pct']}% "
              f"({mon.summary()['cpu_cores_avg']} cores)")
        print(f"  Idle RSS: {mon.summary()['rss_avg_mb']} MB")

    # ── Serial streaming test (through proxy) ──
    print(f"\n{'='*60}")
    print(f"  [{label}] Serial streaming test (n={n_serial}, warmup={warmup})")
    print(f"{'='*60}")

    # Warmup
    print(f"  Warming up ({warmup} requests)...")
    asyncio.run(_warmup(proxy_url, warmup, max_tokens, input_tokens))

    # Serial: send one at a time
    serial_results = asyncio.run(_run_serial(proxy_url, n_serial, max_tokens, input_tokens))
    serial_analysis = analyze_results(serial_results)
    results["serial"]["proxy"] = serial_analysis
    print(f"  Proxy TTFT: mean={serial_analysis['ttft'].get('mean',0):.1f}ms "
          f"p95={serial_analysis['ttft'].get('p95',0):.1f}ms")
    print(f"  Proxy TPS:  mean={serial_analysis['tps'].get('mean',0):.1f} "
          f"p95={serial_analysis['tps'].get('p95',0):.1f}")

    # Serial: direct to backend (baseline)
    if direct_url:
        print(f"  Direct backend baseline...")
        direct_results = asyncio.run(_run_serial(direct_url, n_serial, max_tokens, input_tokens))
        direct_analysis = analyze_results(direct_results)
        results["serial"]["direct"] = direct_analysis
        print(f"  Direct TTFT: mean={direct_analysis['ttft'].get('mean',0):.1f}ms "
              f"p95={direct_analysis['ttft'].get('p95',0):.1f}ms")
        print(f"  Direct TPS:  mean={direct_analysis['tps'].get('mean',0):.1f} "
              f"p95={direct_analysis['tps'].get('p95',0):.1f}")

    # ── Concurrent streaming tests ──
    print(f"\n{'='*60}")
    print(f"  [{label}] Concurrent streaming tests")
    print(f"{'='*60}")

    for c in concurrencies:
        total_reqs = max(n_concurrent, c * 3)
        print(f"\n  [{label}] concurrency={c}, total_requests={total_reqs}")

        # Start resource monitor
        mon = None
        if proxy_pids:
            mon = ResourceMonitor(proxy_pids, interval_s=0.3)
            mon.start()

        t0 = time.perf_counter()
        conc_results = asyncio.run(
            bench_concurrent_stream(proxy_url, c, total_reqs, max_tokens, input_tokens)
        )
        wall_ms = (time.perf_counter() - t0) * 1000

        if mon:
            mon.stop()

        analysis = analyze_results(conc_results)
        analysis["wall_ms"] = round(wall_ms, 1)
        analysis["concurrency"] = c
        analysis["total_requests"] = total_reqs
        if mon:
            analysis["resources"] = mon.summary()

        results["concurrent"][str(c)] = analysis

        # Print summary
        res_str = ""
        if mon:
            s = mon.summary()
            res_str = f" | CPU: avg={s['cpu_avg_pct']}% max={s['cpu_max_pct']}% " \
                      f"({s['cpu_cores_avg']}/{s['cpu_cores_max']} cores) " \
                      f"| RSS: avg={s['rss_avg_mb']}MB max={s['rss_max_mb']}MB"

        print(f"    TTFT: mean={analysis['ttft'].get('mean',0):.1f}ms "
              f"p95={analysis['ttft'].get('p95',0):.1f}ms")
        print(f"    TPS:  mean={analysis['tps'].get('mean',0):.1f} "
              f"p95={analysis['tps'].get('p95',0):.1f}")
        print(f"    Errors: {analysis['errors']} | Wall: {wall_ms:.0f}ms{res_str}")

    return results


async def _warmup(url: str, n: int, max_tokens: int, input_tokens: int = 15):
    async with httpx.AsyncClient() as client:
        for _ in range(n):
            try:
                await bench_stream_single(client, url, max_tokens, input_tokens)
            except Exception:
                pass


async def _run_serial(url: str, n: int, max_tokens: int, input_tokens: int = 15) -> List[StreamResult]:
    results = []
    async with httpx.AsyncClient() as client:
        for _ in range(n):
            r = await bench_stream_single(client, url, max_tokens, input_tokens)
            results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(description="PROXY_WORKERS TTFT/TPS/Resource benchmark")
    parser.add_argument("--proxy-url", default="http://localhost:18000")
    parser.add_argument("--direct-url", default="http://localhost:17000",
                        help="Direct backend URL for baseline comparison")
    parser.add_argument("--label", default="test")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--proxy-pids", default="",
                        help="Comma-separated PIDs of proxy processes. Auto-detect if empty.")
    parser.add_argument("--proxy-port", type=int, default=18000,
                        help="Proxy port for auto-detecting PIDs")
    parser.add_argument("-n", type=int, default=30,
                        help="Serial test iterations")
    parser.add_argument("--n-concurrent", type=int, default=60,
                        help="Minimum total requests per concurrency test")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--concurrency", default="1,5,10,20,50",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max tokens per request (controls stream length)")
    parser.add_argument("--input-tokens", type=int, default=15,
                        help="Approximate input prompt length in tokens")
    args = parser.parse_args()

    # Pre-generate and cache the prompt
    _get_prompt(args.input_tokens)

    # Parse PIDs
    if args.proxy_pids:
        pids = [int(p.strip()) for p in args.proxy_pids.split(",") if p.strip()]
    else:
        print(f"Auto-detecting proxy PIDs on port {args.proxy_port}...")
        pids = find_proxy_pids(args.proxy_port)
        print(f"  Found PIDs: {pids}")

    concurrencies = [int(x.strip()) for x in args.concurrency.split(",")]

    results = run_test(
        proxy_url=args.proxy_url,
        direct_url=args.direct_url,
        label=args.label,
        proxy_pids=pids,
        concurrencies=concurrencies,
        n_serial=args.n,
        n_concurrent=args.n_concurrent,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        input_tokens=args.input_tokens,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
