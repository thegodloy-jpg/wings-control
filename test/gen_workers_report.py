#!/usr/bin/env python3
"""Generate comparison report for PROXY_WORKERS performance test.

Usage:
    python gen_workers_report.py results_w1.json results_w4.json results_w16.json ... -o report.md
"""
import json
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(v, unit="ms"):
    if v is None or v == "N/A":
        return "N/A"
    return f"{v:.3f}{unit}"


def pct_change(old, new):
    if old is None or new is None or old == 0:
        return "N/A"
    delta = ((new - old) / old) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}%"


def generate_report(datasets: List[Dict], output_path: str):
    """Generate markdown comparison report from multiple result files."""
    lines = []
    labels = [d["label"] for d in datasets]

    lines.append("# PROXY_WORKERS 性能影响验证报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试机器**: 7.6.52.148 (A100, 64 CPU core, 219GB RAM)")
    lines.append(f"**测试配置组**: {', '.join(labels)}")
    lines.append(f"**每场景串行迭代**: {datasets[0].get('iterations', 'N/A')} 次, "
                 f"预热 {datasets[0].get('warmup', 'N/A')} 次")
    lines.append(f"**并发级别**: {datasets[0].get('concurrencies', [])}")
    lines.append("")

    # ── Background ──
    lines.append("## 背景")
    lines.append("")
    lines.append("`proxy_config.py` 中 `_MAX_PROXY_WORKERS = 128` 定义了 proxy worker 进程数的上限。")
    lines.append("`WORKERS = min(int(os.getenv(\"PROXY_WORKERS\", \"128\")), _MAX_PROXY_WORKERS)` 决定实际 worker 数。")
    lines.append("")
    lines.append("worker 数量影响：")
    lines.append("1. **进程数**: uvicorn 启动的工作进程数量")
    lines.append("2. **内存占用**: 每 worker 约 65MB RAM")
    lines.append("3. **配额分配**: `_split_strict()` 将全局配额 (`GLOBAL_PASS_THROUGH_LIMIT=1024`) 均分给每个 worker")
    lines.append("4. **Gate 容量**: Gate-0 = WORKERS 个 slot, Gate-1 = LOCAL_PASS_THROUGH_LIMIT - Gate0_local")
    lines.append("")

    lines.append("### 配额分配计算")
    lines.append("")
    lines.append("| Workers | LOCAL_PASS_THROUGH | LOCAL_QUEUE | GATE0_LOCAL | GATE1_LOCAL | 内存估算 |")
    lines.append("|---------|-------------------|-------------|-------------|-------------|---------|")
    for w in [1, 4, 8, 16, 32, 64, 128]:
        local_pt = 1024 // w
        local_q = 1024 // w
        g0_total = w
        g0_local = g0_total // w  # = 1
        g1_local = max(0, local_pt - g0_local)
        mem = w * 65
        lines.append(f"| {w} | {local_pt} | {local_q} | {g0_local} | {g1_local} | ~{mem}MB |")
    lines.append("")

    # ── Serial latency table ──
    lines.append("## 串行延迟对比")
    lines.append("")
    lines.append("每个请求顺序发送，测量单请求 round-trip 延迟。")
    lines.append("")

    scenarios = list(datasets[0].get("serial", {}).keys())
    for scenario in scenarios:
        stype = datasets[0]["serial"][scenario].get("type", "non-stream")
        lines.append(f"### {scenario} ({stype})")
        lines.append("")

        if stype == "stream":
            header = "| 配置 | TTFT Mean | TTFT P95 | Total Mean | Total P95 | Errors |"
            sep = "|------|-----------|----------|------------|-----------|--------|"
            lines.append(header)
            lines.append(sep)
            for d in datasets:
                s = d["serial"].get(scenario, {})
                ttft = s.get("ttft", {})
                total = s.get("total", {})
                lines.append(f"| {d['label']} | {fmt(ttft.get('mean_ms'))} | "
                             f"{fmt(ttft.get('p95_ms'))} | {fmt(total.get('mean_ms'))} | "
                             f"{fmt(total.get('p95_ms'))} | {s.get('errors', 0)} |")
        else:
            header = "| 配置 | Mean | Median | P90 | P95 | P99 | Errors |"
            sep = "|------|------|--------|-----|-----|-----|--------|"
            lines.append(header)
            lines.append(sep)
            for d in datasets:
                s = d["serial"].get(scenario, {})
                lat = s.get("latency", {})
                lines.append(f"| {d['label']} | {fmt(lat.get('mean_ms'))} | "
                             f"{fmt(lat.get('median_ms'))} | {fmt(lat.get('p90_ms'))} | "
                             f"{fmt(lat.get('p95_ms'))} | {fmt(lat.get('p99_ms'))} | "
                             f"{s.get('errors', 0)} |")
        lines.append("")

    # ── Concurrent throughput table ──
    lines.append("## 并发吞吐对比")
    lines.append("")
    lines.append("固定请求数，以不同并发度发送，测量 QPS 和延迟。")
    lines.append("")

    conc_keys = sorted(datasets[0].get("concurrent", {}).keys(), key=lambda x: int(x))
    if conc_keys:
        header = "| 配置 | 并发 | QPS | Mean (ms) | P95 (ms) | P99 (ms) | Errors | Wall (ms) |"
        sep = "|------|------|-----|-----------|----------|----------|--------|-----------|"
        lines.append(header)
        lines.append(sep)
        for d in datasets:
            for ck in conc_keys:
                cd = d["concurrent"].get(ck, {})
                lat = cd.get("latency", {})
                lines.append(f"| {d['label']} | {cd.get('concurrency','?')} | "
                             f"{cd.get('qps','N/A')} | {fmt(lat.get('mean_ms'))} | "
                             f"{fmt(lat.get('p95_ms'))} | {fmt(lat.get('p99_ms'))} | "
                             f"{cd.get('errors', 0)} | {fmt(cd.get('wall_time_ms'))} |")
        lines.append("")

    # ── QPS comparison across configs ──
    if conc_keys:
        lines.append("### QPS 对比汇总")
        lines.append("")
        header_parts = ["| 并发 \\\\  配置"] + [f" {d['label']}" for d in datasets] + [" |"]
        lines.append(" |".join(header_parts))
        sep_parts = ["|---"] * (len(datasets) + 1) + ["|"]
        lines.append("".join(sep_parts))
        for ck in conc_keys:
            row = [f"| {ck}"]
            for d in datasets:
                cd = d["concurrent"].get(ck, {})
                row.append(f" {cd.get('qps', 'N/A')}")
            row.append(" |")
            lines.append(" |".join(row))
        lines.append("")

    # ── Conclusion ──
    lines.append("## 分析与结论")
    lines.append("")
    lines.append("### 1. Worker 数量对串行延迟的影响")
    lines.append("")
    lines.append("串行延迟（单请求 RTT）主要反映进程启动开销和闸门 acquire 开销。")
    lines.append("当 worker 数增加时：")
    lines.append("- 每个 worker 分配的本地配额减少（`_split_strict` 均分）")
    lines.append("- Gate-0 有 1 个 slot / worker，Gate-1 = `LOCAL_PASS_THROUGH_LIMIT - 1`")
    lines.append("- 串行场景下不会触发排队，延迟差异主要来自进程间资源竞争")
    lines.append("")
    lines.append("### 2. Worker 数量对并发吞吐的影响")
    lines.append("")
    lines.append("高并发场景是 worker 数量影响最大的维度：")
    lines.append("- 更多 worker = 更多并行处理能力 → QPS 提升")
    lines.append("- 但也 = 更多进程竞争 CPU / 内存 → 单请求延迟可能增加")
    lines.append("- 过多 worker（如 128）在低并发下是浪费")
    lines.append("")
    lines.append("### 3. `_MAX_PROXY_WORKERS = 128` 的合理性")
    lines.append("")
    lines.append("- 128 worker × 65MB ≈ 8GB 内存，在 219GB 的测试机器上完全可接受")
    lines.append("- 但 128 是**上限**，实际部署应根据并发需求设置 `PROXY_WORKERS`")
    lines.append("- **重要发现**: 当 `PROXY_WORKERS` 未设置时，`proxy_config.py` 默认 128，")
    lines.append("  但 `wings_control.py` launcher 默认只启动 4 个 worker。")
    lines.append("  这导致配额分配不匹配：4 个 worker 各自以为有 128 个 worker，每个只分到 1024/128=8 的配额")
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {output_path}")
    print(report[:2000] + "\n...(truncated)")


def main():
    parser = argparse.ArgumentParser(description="Generate workers performance comparison report")
    parser.add_argument("inputs", nargs="+", help="Result JSON files")
    parser.add_argument("-o", "--output", default="report_workers.md")
    args = parser.parse_args()

    datasets = [load_json(f) for f in args.inputs]
    generate_report(datasets, args.output)


if __name__ == "__main__":
    main()
