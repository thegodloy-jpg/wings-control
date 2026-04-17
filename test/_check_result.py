#!/usr/bin/env python3
"""Quick result validator."""
import json, sys, glob

files = sorted(glob.glob("/home/zhanghui/workers-perf-test-v2/results_*.json"))
if not files:
    print("No result files found yet")
    sys.exit(0)

for f in files:
    d = json.load(open(f))
    label = d["label"]
    pids = d.get("proxy_pids", [])
    ir = d.get("idle_resources", {})
    s = d.get("serial", {})
    sp = s.get("proxy", {})
    sd = s.get("direct", {})
    print(f"\n=== {label} ===")
    print(f"  PIDs: {pids[:5]}{'...' if len(pids) > 5 else ''} ({len(pids)} total)")
    print(f"  Idle RSS: {ir.get('rss_avg_mb', 'N/A')} MB, CPU: {ir.get('cpu_avg_pct', 'N/A')}%")
    print(f"  Serial Proxy  TTFT: {sp.get('ttft',{}).get('mean','N/A')} ms")
    print(f"  Serial Direct TTFT: {sd.get('ttft',{}).get('mean','N/A')} ms")
    print(f"  Serial Proxy  TPS:  {sp.get('tps',{}).get('mean','N/A')}")
    print(f"  Serial Direct TPS:  {sd.get('tps',{}).get('mean','N/A')}")
