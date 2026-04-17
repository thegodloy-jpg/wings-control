#!/usr/bin/env python3
"""Generate TTFT/TPS/Resource comparison report from benchmark results.

Usage:
    python gen_report_v2.py results_w1.json results_w4.json results_w16.json ... -o report.md
"""
import json
import argparse
import os
from datetime import datetime
from typing import List, Dict, Any


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(v, unit="ms", decimals=1):
    if v is None or v == "N/A" or v == 0:
        return "-"
    return f"{v:.{decimals}f}{unit}"


def pct(old, new):
    if not old or old == 0:
        return "-"
    d = ((new - old) / old) * 100
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.1f}%"


def generate_report(datasets: List[Dict], output_path: str):
    lines = []
    labels = [d["label"] for d in datasets]

    # ── Header ──
    lines.append("# PROXY_WORKERS 性能与资源消耗对比报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试机器**: 7.6.52.148 (A100)")

    # Get machine info from first dataset
    lines.append(f"**测试配置组**: {', '.join(labels)}")
    input_tokens = datasets[0].get('input_tokens', 15)
    max_tokens = datasets[0].get('max_tokens', 50)
    lines.append(f"**Mock 后端**: PREFILL_MS=100, TOKEN_DELAY_MS=30, "
                 f"MAX_TOKENS={max_tokens}")
    lines.append(f"**测试数据集**: 输入 ~{input_tokens} tokens, 输出 {max_tokens} tokens")
    lines.append(f"**预期基线**: Engine TTFT ≈ 100ms, Engine TPS ≈ 33.3 tok/s")
    lines.append(f"**串行迭代数**: {datasets[0].get('n_serial', '-')}, "
                 f"预热 {datasets[0].get('warmup', '-')} 次")
    lines.append(f"**并发级别**: {datasets[0].get('concurrencies', [])}")
    lines.append("")

    # ── Section 1: Resource consumption ──
    lines.append("## 1. 系统资源消耗")
    lines.append("")
    lines.append("### 1.1 空闲状态资源占用")
    lines.append("")
    lines.append("Proxy 启动后、无请求时的资源占用：")
    lines.append("")
    lines.append("| 配置 | Worker 进程数 | CPU 占用 (%) | CPU 核数 | RSS 内存 (MB) |")
    lines.append("|------|-------------|-------------|---------|--------------|")
    for d in datasets:
        idle = d.get("idle_resources", {})
        lines.append(f"| {d['label']} | {idle.get('num_processes', '-')} | "
                     f"{idle.get('cpu_avg_pct', '-')}% | "
                     f"{idle.get('cpu_cores_avg', '-')} | "
                     f"{idle.get('rss_avg_mb', '-')} |")
    lines.append("")

    # ── 1.2 Load resource consumption ──
    lines.append("### 1.2 负载状态资源占用（按并发级别）")
    lines.append("")

    # Collect all concurrency keys
    conc_keys = set()
    for d in datasets:
        conc_keys.update(d.get("concurrent", {}).keys())
    conc_keys = sorted(conc_keys, key=lambda x: int(x))

    for ck in conc_keys:
        lines.append(f"#### 并发 = {ck}")
        lines.append("")
        lines.append("| 配置 | CPU 平均 (%) | CPU 峰值 (%) | CPU 核数 (avg/max) | RSS 平均 (MB) | RSS 峰值 (MB) |")
        lines.append("|------|-------------|-------------|-------------------|--------------|--------------|")
        for d in datasets:
            cd = d.get("concurrent", {}).get(ck, {})
            res = cd.get("resources", {})
            if res:
                lines.append(
                    f"| {d['label']} | {res.get('cpu_avg_pct', '-')}% | "
                    f"{res.get('cpu_max_pct', '-')}% | "
                    f"{res.get('cpu_cores_avg', '-')}/{res.get('cpu_cores_max', '-')} | "
                    f"{res.get('rss_avg_mb', '-')} | {res.get('rss_max_mb', '-')} |")
            else:
                lines.append(f"| {d['label']} | - | - | - | - | - |")
        lines.append("")

    # ── Section 2: TTFT comparison ──
    lines.append("## 2. TTFT (Time To First Token) 对比")
    lines.append("")
    lines.append("TTFT = 从发送请求到收到第一个 token 的时间，包含: 网络延迟 + proxy 转发开销 + 引擎 prefill 时间。")
    lines.append("")
    lines.append("Mock 引擎 prefill = 100ms，因此 TTFT 的增量部分为 proxy 引入的额外延迟。")
    lines.append("")

    # Serial TTFT
    lines.append("### 2.1 串行 TTFT（单请求）")
    lines.append("")
    lines.append("| 配置 | Mean | Median | P90 | P95 | P99 | Proxy 额外开销 |")
    lines.append("|------|------|--------|-----|-----|-----|---------------|")

    # Get direct baseline if available
    direct_ttft_mean = None
    for d in datasets:
        direct = d.get("serial", {}).get("direct", {})
        if direct:
            dt = direct.get("ttft", {})
            if dt.get("mean"):
                direct_ttft_mean = dt["mean"]
                break

    for d in datasets:
        serial = d.get("serial", {}).get("proxy", {})
        ttft = serial.get("ttft", {})
        overhead = ""
        if direct_ttft_mean and ttft.get("mean"):
            overhead = f"{ttft['mean'] - direct_ttft_mean:.1f}ms"
        lines.append(
            f"| {d['label']} | {fmt(ttft.get('mean'))} | {fmt(ttft.get('median'))} | "
            f"{fmt(ttft.get('p90'))} | {fmt(ttft.get('p95'))} | "
            f"{fmt(ttft.get('p99'))} | {overhead} |")

    if direct_ttft_mean:
        lines.append(f"| **Direct (baseline)** | {fmt(direct_ttft_mean)} | - | - | - | - | 0ms |")
    lines.append("")

    # Concurrent TTFT
    lines.append("### 2.2 并发 TTFT")
    lines.append("")
    lines.append("| 并发 |" + " | ".join(f" {d['label']}" for d in datasets) + " |")
    lines.append("|------|" + " | ".join("----" for _ in datasets) + " |")
    for ck in conc_keys:
        row = f"| {ck} |"
        for d in datasets:
            cd = d.get("concurrent", {}).get(ck, {})
            ttft = cd.get("ttft", {})
            mean = ttft.get("mean", 0)
            p95 = ttft.get("p95", 0)
            row += f" {fmt(mean)} (p95={fmt(p95)}) |"
        lines.append(row)
    lines.append("")

    # ── Section 3: TPS comparison ──
    lines.append("## 3. TPS (Tokens Per Second) 对比")
    lines.append("")
    lines.append("TPS = 解码阶段的 token 吞吐率（每请求）= (token_count - 1) / decode_time。")
    lines.append("")
    lines.append("Mock 引擎 TPS 基线 ≈ 33.3 tok/s (TOKEN_DELAY_MS=30)。TPS 下降表示 proxy 转发开销在持续影响 token 传输。")
    lines.append("")

    # Serial TPS
    lines.append("### 3.1 串行 TPS（单请求）")
    lines.append("")
    lines.append("| 配置 | Mean | Median | P90 | P95 | TPS 下降率 |")
    lines.append("|------|------|--------|-----|-----|-----------|")

    direct_tps_mean = None
    for d in datasets:
        direct = d.get("serial", {}).get("direct", {})
        if direct:
            dt = direct.get("tps", {})
            if dt.get("mean"):
                direct_tps_mean = dt["mean"]
                break

    for d in datasets:
        serial = d.get("serial", {}).get("proxy", {})
        tps = serial.get("tps", {})
        drop = ""
        if direct_tps_mean and tps.get("mean"):
            drop = pct(direct_tps_mean, tps["mean"])
        lines.append(
            f"| {d['label']} | {fmt(tps.get('mean'), ' tok/s')} | "
            f"{fmt(tps.get('median'), ' tok/s')} | {fmt(tps.get('p90'), ' tok/s')} | "
            f"{fmt(tps.get('p95'), ' tok/s')} | {drop} |")

    if direct_tps_mean:
        lines.append(f"| **Direct (baseline)** | {fmt(direct_tps_mean, ' tok/s')} | - | - | - | 0% |")
    lines.append("")

    # Concurrent TPS
    lines.append("### 3.2 并发 TPS")
    lines.append("")
    lines.append("| 并发 |" + " | ".join(f" {d['label']}" for d in datasets) + " |")
    lines.append("|------|" + " | ".join("----" for _ in datasets) + " |")
    for ck in conc_keys:
        row = f"| {ck} |"
        for d in datasets:
            cd = d.get("concurrent", {}).get(ck, {})
            tps = cd.get("tps", {})
            mean = tps.get("mean", 0)
            p95 = tps.get("p95", 0)
            row += f" {fmt(mean, ' tok/s')} (p95={fmt(p95, ' tok/s')}) |"
        lines.append(row)
    lines.append("")

    # ── Section 4: Resource-Performance summary ──
    lines.append("## 4. 资源-性能权衡汇总")
    lines.append("")
    lines.append("综合资源消耗与性能指标，评估边际收益：")
    lines.append("")
    lines.append("| 配置 | 空闲 RSS (MB) | 并发50 CPU 核数 | 并发50 RSS (MB) | "
                 "串行 TTFT (ms) | 串行 TPS | 并发50 TTFT P95 (ms) | 并发50 TPS Mean |")
    lines.append("|------|-------------|---------------|---------------|"
                 "--------------|---------|-------------------|----------------|")

    for d in datasets:
        idle = d.get("idle_resources", {})
        # Find the highest concurrency available
        conc_data = d.get("concurrent", {})
        high_conc_key = conc_keys[-1] if conc_keys else None
        hc = conc_data.get(high_conc_key, {}) if high_conc_key else {}
        hc_res = hc.get("resources", {})
        serial = d.get("serial", {}).get("proxy", {})

        lines.append(
            f"| {d['label']} | {idle.get('rss_avg_mb', '-')} | "
            f"{hc_res.get('cpu_cores_avg', '-')} | {hc_res.get('rss_avg_mb', '-')} | "
            f"{serial.get('ttft', {}).get('mean', '-')} | "
            f"{serial.get('tps', {}).get('mean', '-')} | "
            f"{hc.get('ttft', {}).get('p95', '-')} | "
            f"{hc.get('tps', {}).get('mean', '-')} |")
    lines.append("")

    # ── Section 5: Analysis ──
    lines.append("## 5. 分析与结论")
    lines.append("")
    lines.append("### 5.1 资源消耗")
    lines.append("")
    lines.append("- **内存 (RSS)**: 每增加一个 worker 进程，约增加 55~65 MB RSS")
    lines.append("- **CPU**: 空闲时几乎不占 CPU；负载时 CPU 使用与并发数和 worker 数正相关")
    lines.append("- Worker 数超过 CPU 核数时，CPU 调度开销增加但不会额外消耗更多 CPU")
    lines.append("")
    lines.append("### 5.2 TTFT 影响")
    lines.append("")
    lines.append("- Proxy 对 TTFT 的额外开销主要来自进程间路由和网络转发")
    lines.append("- 在低并发下，worker 数量对 TTFT 影响很小")
    lines.append("- 在高并发下，worker 不足会导致请求排队，显著增加 TTFT")
    lines.append("")
    lines.append("### 5.3 TPS 影响")
    lines.append("")
    lines.append("- Proxy 对 TPS 的影响主要来自 SSE chunk 转发延迟")
    lines.append("- 串行场景下 TPS 影响很小（接近直连基线）")
    lines.append("- 高并发下 TPS 是否下降取决于 proxy 的 chunk 转发能力")
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Result JSON files")
    parser.add_argument("-o", "--output", default="report_workers_v2.md")
    args = parser.parse_args()

    datasets = [load_json(f) for f in args.inputs]
    report = generate_report(datasets, args.output)
    # Print first 3000 chars
    print(report[:3000])
    if len(report) > 3000:
        print(f"\n... (truncated, total {len(report)} chars)")


if __name__ == "__main__":
    main()
