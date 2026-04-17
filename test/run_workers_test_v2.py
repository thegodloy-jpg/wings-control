#!/usr/bin/env python3
"""Test runner — orchestrate PROXY_WORKERS performance test on 148 machine.

This script:
1. Starts the mock backend
2. For each PROXY_WORKERS value:
   a. Starts the proxy with that worker count
   b. Waits for readiness
   c. Discovers proxy PIDs
   d. Runs bench_workers_v2.py
   e. Kills proxy
3. Generates comparison report

Usage:
    python run_workers_test_v2.py [--workers 1,4,16,64,128] [--work-dir /home/zhanghui/workers-perf-test]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


BACKEND_PORT = 17000
PROXY_PORT = 18000


def run(cmd: str, timeout: int = 30) -> str:
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"ERROR: {e}"


def is_port_in_use(port: int) -> bool:
    out = run(f"ss -tlnp | grep :{port}")
    return bool(out)


def kill_port(port: int):
    """Kill any process listening on the given port."""
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
    """Start mock backend in background."""
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

    # Wait for readiness
    for i in range(30):
        time.sleep(1)
        if is_port_in_use(BACKEND_PORT):
            print(f"  Mock backend ready on port {BACKEND_PORT}")
            return proc
    raise RuntimeError("Mock backend failed to start within 30s")


def start_proxy(work_dir: str, wings_dir: str, proxy_workers: int) -> subprocess.Popen:
    """Start minimal proxy via uvicorn with specified worker count."""
    env = os.environ.copy()
    env["PROXY_WORKERS"] = str(proxy_workers)
    env["BACKEND_URL"] = f"http://127.0.0.1:{BACKEND_PORT}"

    # Use minimal_proxy.py in work_dir
    proxy_module = os.path.join(work_dir, "minimal_proxy.py")
    if not os.path.exists(proxy_module):
        raise RuntimeError(f"minimal_proxy.py not found in {work_dir}")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "minimal_proxy:app",
        "--host", "0.0.0.0",
        "--port", str(PROXY_PORT),
        "--log-level", "error",
    ]
    if proxy_workers > 1:
        cmd += ["--workers", str(proxy_workers)]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        cwd=work_dir,
        preexec_fn=os.setpgrp if hasattr(os, 'setpgrp') else None,
    )
    print(f"Proxy started with {proxy_workers} worker(s) (PID={proc.pid})")

    # Wait for readiness
    for i in range(60):
        time.sleep(1)
        try:
            import httpx
            r = httpx.get(f"http://localhost:{PROXY_PORT}/health", timeout=3)
            if r.status_code == 200:
                print(f"  Proxy ready on port {PROXY_PORT}")
                return proc
        except Exception:
            pass
    raise RuntimeError(f"Proxy failed to start within 60s (workers={proxy_workers})")


def find_proxy_pids() -> list:
    """Find all proxy worker PIDs (master + child workers)."""
    try:
        # Find master PID first (the one with --port in args)
        out = subprocess.check_output(
            ["bash", "-c", f"ps aux | grep 'uvicorn.*--port.*{PROXY_PORT}' | grep -v grep"],
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

        # Find all descendants of master PID (child workers)
        try:
            desc_out = subprocess.check_output(
                ["bash", "-c", f"pgrep -P {master_pid}"],
                text=True, timeout=5
            ).strip()
            child_pids = [int(p.strip()) for p in desc_out.split("\n") if p.strip()]
        except Exception:
            child_pids = []

        # Return master + all children
        return [master_pid] + child_pids
    except Exception:
        return []


def kill_process_group(proc: subprocess.Popen):
    """Kill process and all children."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    proc.wait(timeout=5)


def get_machine_info() -> dict:
    """Get machine CPU and memory info."""
    cpu_count = int(run("nproc"))
    cpu_model = run("lscpu | grep 'Model name' | head -1 | cut -d: -f2").strip()
    mem_total = run("free -g | awk '/Mem:/{print $2}'")
    return {
        "cpu_cores": cpu_count,
        "cpu_model": cpu_model,
        "mem_total_gb": int(mem_total) if mem_total.isdigit() else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default="1,4,16,64,128",
                        help="Comma-separated PROXY_WORKERS values to test")
    parser.add_argument("--work-dir", default="/home/zhanghui/workers-perf-test-v2",
                        help="Working directory on test machine")
    parser.add_argument("--wings-dir", default="",
                        help="Path to wings_control directory")
    parser.add_argument("--prefill-ms", type=float, default=100,
                        help="Mock engine prefill delay (ms)")
    parser.add_argument("--token-delay-ms", type=float, default=30,
                        help="Mock engine per-token delay (ms)")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Tokens per response")
    parser.add_argument("--n-serial", type=int, default=30,
                        help="Serial test iterations")
    parser.add_argument("--n-concurrent", type=int, default=60,
                        help="Min total requests per concurrency level")
    parser.add_argument("--concurrency", default="1,5,10,20,50",
                        help="Concurrency levels to test")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--input-tokens", type=int, default=15,
                        help="Approximate input prompt length in tokens")
    args = parser.parse_args()

    worker_configs = [int(x.strip()) for x in args.workers.split(",")]

    # Auto-detect wings_dir (optional — minimal proxy doesn't need it)
    wings_dir = args.wings_dir
    if not wings_dir:
        # Look for wings_control relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, "..", "wings_control")
        if os.path.isdir(candidate):
            wings_dir = os.path.abspath(candidate)
        else:
            candidate = os.path.join(args.work_dir, "wings_control")
            if os.path.isdir(candidate):
                wings_dir = candidate
            else:
                wings_dir = ""  # minimal proxy doesn't need wings_dir

    print(f"Wings dir: {wings_dir or '(not needed — using minimal proxy)'}")
    print(f"Work dir:  {args.work_dir}")
    os.makedirs(args.work_dir, exist_ok=True)

    machine = get_machine_info()
    print(f"Machine: {machine['cpu_cores']} cores, {machine['mem_total_gb']}GB RAM, {machine['cpu_model']}")

    # Ensure ports are free
    for port in [BACKEND_PORT, PROXY_PORT]:
        if is_port_in_use(port):
            print(f"Port {port} is in use, killing existing process...")
            kill_port(port)

    # Start mock backend
    print("\n" + "=" * 60)
    print("  Starting mock backend")
    print("=" * 60)
    backend_proc = start_mock_backend(
        args.work_dir, args.prefill_ms, args.token_delay_ms, args.max_tokens
    )

    result_files = []

    try:
        for w in worker_configs:
            label = f"workers-{w}"
            output_file = os.path.join(args.work_dir, f"results_{label}.json")

            print(f"\n{'#' * 60}")
            print(f"  Testing PROXY_WORKERS={w}")
            print(f"{'#' * 60}")

            # Kill any existing proxy
            if is_port_in_use(PROXY_PORT):
                kill_port(PROXY_PORT)
                time.sleep(2)

            # Start proxy
            proxy_proc = start_proxy(args.work_dir, wings_dir, w)
            time.sleep(3)  # let workers stabilize

            # Discover PIDs
            pids = find_proxy_pids()
            print(f"  Proxy PIDs: {pids} ({len(pids)} processes)")

            # Run benchmark
            bench_cmd = [
                sys.executable, os.path.join(args.work_dir, "bench_workers_v2.py"),
                "--proxy-url", f"http://localhost:{PROXY_PORT}",
                "--direct-url", f"http://localhost:{BACKEND_PORT}",
                "--label", label,
                "--output", output_file,
                "--proxy-pids", ",".join(str(p) for p in pids),
                "-n", str(args.n_serial),
                "--n-concurrent", str(args.n_concurrent),
                "--warmup", str(args.warmup),
                "--concurrency", args.concurrency,
                "--max-tokens", str(args.max_tokens),
                "--input-tokens", str(args.input_tokens),
            ]

            print(f"  Running benchmark...")
            # Longer timeout for large token counts
            bench_timeout = max(600, args.max_tokens * 2 * 10)
            bench_result = subprocess.run(bench_cmd, timeout=bench_timeout)
            if bench_result.returncode != 0:
                print(f"  WARNING: Benchmark exited with code {bench_result.returncode}")

            # Kill proxy
            print(f"  Stopping proxy...")
            kill_process_group(proxy_proc)
            time.sleep(2)

            if os.path.exists(output_file):
                result_files.append(output_file)
                print(f"  Results saved: {output_file}")
            else:
                print(f"  WARNING: No results file generated!")

    finally:
        # Clean up backend
        print("\nStopping mock backend...")
        kill_process_group(backend_proc)

    # Generate report
    if result_files:
        print(f"\n{'=' * 60}")
        print(f"  Generating report from {len(result_files)} result files")
        print(f"{'=' * 60}")

        report_path = os.path.join(args.work_dir, "report_workers_v2.md")
        gen_cmd = [
            sys.executable, os.path.join(args.work_dir, "gen_report_v2.py"),
            *result_files,
            "-o", report_path,
        ]
        subprocess.run(gen_cmd)
        print(f"\nReport: {report_path}")
    else:
        print("\nNo result files generated!")

    print("\nDone!")


if __name__ == "__main__":
    main()
