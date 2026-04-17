#!/usr/bin/env python3
"""Supplementary test: Run concurrent Direct baseline tests and merge into existing results.

This runs concurrent tests directly against the mock backend (bypassing proxy)
to provide a Direct baseline comparison for concurrent TTFT/TPS data.

The Direct baseline is the same regardless of PROXY_WORKERS, so we only need
to run it once per scenario and copy to all worker configs.

Usage:
    python run_direct_concurrent_supplement.py \
        --work-dir /home/zhanghui/workers-perf-test-v2 \
        --scenarios "128:128,512:512"
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time


BACKEND_PORT = 17000


def run(cmd: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def is_port_in_use(port: int) -> bool:
    out = run(f"ss -tlnp | grep :{port}")
    return bool(out)


def kill_port(port: int):
    pids = run(f"lsof -ti :{port}")
    if pids:
        for pid in pids.split("\n"):
            pid = pid.strip()
            if pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
        time.sleep(1)


def start_mock_backend(work_dir: str, prefill_ms: float, token_delay_ms: float,
                       num_tokens: int) -> subprocess.Popen:
    cmd = [
        sys.executable, os.path.join(work_dir, "mock_backend_realistic.py"),
        "--port", str(BACKEND_PORT),
        "--prefill-ms", str(prefill_ms),
        "--token-delay-ms", str(token_delay_ms),
        "--num-tokens", str(num_tokens),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setpgrp if hasattr(os, 'setpgrp') else None,
    )
    print(f"Mock backend started (PID={proc.pid})")
    for i in range(30):
        time.sleep(1)
        if is_port_in_use(BACKEND_PORT):
            print(f"  Mock backend ready on port {BACKEND_PORT}")
            return proc
    raise RuntimeError("Mock backend failed to start")


def kill_process_group(proc):
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="Supplement: concurrent Direct baseline tests")
    parser.add_argument("--work-dir", default="/home/zhanghui/workers-perf-test-v2")
    parser.add_argument("--scenarios", default="128:128,512:512",
                        help="Comma-separated input:output token configs")
    parser.add_argument("--prefill-ms", type=float, default=100)
    parser.add_argument("--token-delay-ms", type=float, default=30)
    parser.add_argument("--concurrency", default="1,5,10,20,50")
    parser.add_argument("--workers", default="1,4,16,64,128",
                        help="Worker configs to merge results into")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    concurrencies = [int(x.strip()) for x in args.concurrency.split(",")]
    worker_configs = [int(x.strip()) for x in args.workers.split(",")]
    scenarios = []
    for s in args.scenarios.split(","):
        inp, out = s.strip().split(":")
        scenarios.append((int(inp), int(out)))

    # Import bench functions
    sys.path.insert(0, args.work_dir)
    from bench_workers_v2 import bench_concurrent_stream, analyze_results, _get_prompt

    for input_tokens, max_tokens in scenarios:
        print(f"\n{'#'*60}")
        print(f"  Scenario: input={input_tokens}, output={max_tokens}")
        print(f"{'#'*60}")

        # Determine result dir and n_concurrent from existing data
        result_dir = os.path.join(args.work_dir, f"results_{input_tokens}_{max_tokens}")
        if not os.path.isdir(result_dir):
            # Try flat naming
            result_dir = args.work_dir

        # Read existing config to match n_concurrent
        sample_file = None
        for w in worker_configs:
            candidate = os.path.join(result_dir, f"results_workers-{w}.json")
            if os.path.exists(candidate):
                sample_file = candidate
                break
        if not sample_file:
            candidate = os.path.join(args.work_dir, f"results_workers-{worker_configs[0]}.json")
            if os.path.exists(candidate):
                sample_file = candidate

        n_concurrent = 40  # default
        if sample_file:
            with open(sample_file) as f:
                sample = json.load(f)
            n_concurrent = sample.get("n_concurrent", 40)
            print(f"  Using n_concurrent={n_concurrent} from {os.path.basename(sample_file)}")

        # Pre-cache prompt
        _get_prompt(input_tokens)

        # Kill any existing backend
        if is_port_in_use(BACKEND_PORT):
            print(f"  Killing existing backend on port {BACKEND_PORT}...")
            kill_port(BACKEND_PORT)
            time.sleep(2)

        # Start mock backend
        backend_proc = start_mock_backend(
            args.work_dir, args.prefill_ms, args.token_delay_ms, max_tokens
        )

        try:
            # Warmup
            print(f"  Warming up ({args.warmup} requests)...")
            direct_url = f"http://localhost:{BACKEND_PORT}"

            async def do_warmup():
                from bench_workers_v2 import bench_stream_single
                import httpx
                async with httpx.AsyncClient() as client:
                    for _ in range(args.warmup):
                        await bench_stream_single(client, direct_url, max_tokens, input_tokens)

            asyncio.run(do_warmup())

            # Run concurrent direct tests
            direct_results = {}
            for c in concurrencies:
                total_reqs = max(n_concurrent, c * 3)
                print(f"\n  Direct concurrency={c}, total_requests={total_reqs}")

                t0 = time.perf_counter()
                conc_results = asyncio.run(
                    bench_concurrent_stream(direct_url, c, total_reqs, max_tokens, input_tokens)
                )
                wall_ms = (time.perf_counter() - t0) * 1000

                analysis = analyze_results(conc_results)
                analysis["wall_ms"] = round(wall_ms, 1)
                analysis["concurrency"] = c
                analysis["total_requests"] = total_reqs

                direct_results[str(c)] = analysis

                print(f"    Direct TTFT: mean={analysis['ttft'].get('mean',0):.1f}ms "
                      f"p95={analysis['ttft'].get('p95',0):.1f}ms")
                print(f"    Direct TPS:  mean={analysis['tps'].get('mean',0):.1f} "
                      f"p95={analysis['tps'].get('p95',0):.1f}")
                print(f"    Errors: {analysis['errors']} | Wall: {wall_ms:.0f}ms")

            # Merge into all worker config result files
            print(f"\n  Merging concurrent_direct data into result files...")
            for w in worker_configs:
                # Try result dir naming
                json_path = os.path.join(result_dir, f"results_workers-{w}.json")
                if not os.path.exists(json_path):
                    json_path = os.path.join(args.work_dir, f"results_workers-{w}.json")
                if not os.path.exists(json_path):
                    print(f"    SKIP: {json_path} not found")
                    continue

                with open(json_path) as f:
                    data = json.load(f)

                data["concurrent_direct"] = direct_results

                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"    Updated: {os.path.basename(json_path)}")

        finally:
            print("\n  Stopping mock backend...")
            kill_process_group(backend_proc)

    print("\nDone! All result files updated with concurrent_direct data.")


if __name__ == "__main__":
    main()
