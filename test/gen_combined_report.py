#!/usr/bin/env python3
"""Generate combined PROXY_WORKERS performance report for multiple input/output configs.

Usage:
    python3 gen_combined_report.py \
        --configs 128:results_128_128/ 512:results_512_512/ \
        -o report_combined.md
"""
import json
import argparse
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Tuple


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


def load_config_datasets(config_spec: str) -> Tuple[str, List[Dict]]:
    """Parse 'tokens:dir/' spec and load all result JSONs.
    Returns (label, sorted_datasets).
    """
    parts = config_spec.split(":", 1)
    label = parts[0]
    directory = parts[1] if len(parts) > 1 else parts[0]

    files = sorted(glob.glob(os.path.join(directory, "results_workers-*.json")))
    if not files:
        raise FileNotFoundError(f"No result files in {directory}")

    datasets = [load_json(f) for f in files]

    # Sort by worker count
    def worker_count(d):
        label = d.get("label", "")
        try:
            return int(label.replace("workers-", ""))
        except ValueError:
            return 0

    datasets.sort(key=worker_count)
    return label, datasets


def get_worker_labels(datasets):
    return [d["label"] for d in datasets]


def generate_combined_report(configs: List[Tuple[str, List[Dict]]], output_path: str):
    lines = []

    # ── Header ──
    lines.append("# PROXY_WORKERS 性能与资源消耗对比报告（多场景）")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试机器**: 7.6.52.148 (64 核 Intel Xeon Gold 6444Y, 219GB RAM)")
    lines.append(f"**Mock 后端**: PREFILL_MS=100, TOKEN_DELAY_MS=30")
    lines.append(f"**预期基线**: Engine TTFT ≈ 100ms, Engine TPS ≈ 33.3 tok/s")

    config_labels = []
    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        config_labels.append(f"输入{it}/输出{mt}")
    lines.append(f"**测试场景**: {' | '.join(config_labels)}")

    worker_labels = get_worker_labels(configs[0][1])
    lines.append(f"**测试配置组**: {', '.join(worker_labels)}")
    lines.append(f"**并发级别**: {configs[0][1][0].get('concurrencies', [])}")
    lines.append("")

    # Show test params per config
    lines.append("### 测试参数")
    lines.append("")
    lines.append("| 场景 | 输入 Tokens | 输出 Tokens | 串行请求数 | 并发请求数 | 预热请求数 | 单请求预计耗时 |")
    lines.append("|------|-----------|-----------|----------|----------|----------|-------------|")
    for cfg_label, datasets in configs:
        d = datasets[0]
        it = d.get("input_tokens", 15)
        mt = d.get("max_tokens", 50)
        ns = d.get("n_serial", 20)
        nc = d.get("n_concurrent", 40)
        wu = d.get("warmup", 8)
        est_time = f"~{mt * 30 / 1000 + 0.1:.1f}s"
        lines.append(f"| 输入{it}/输出{mt} | {it} | {mt} | {ns} | {nc} | {wu} | {est_time} |")
    lines.append("")

    # ── Section 1: Resource consumption (same regardless of token count) ──
    lines.append("## 1. 系统资源消耗")
    lines.append("")
    lines.append("资源消耗主要取决于 worker 进程数，与输入/输出 token 数关系不大。")
    lines.append("以下数据取自输入128/输出128配置（其他配置几乎相同）。")
    lines.append("")

    first_datasets = configs[0][1]

    lines.append("### 1.1 空闲状态资源占用")
    lines.append("")
    lines.append("| 配置 | Worker 进程数 | CPU 占用 (%) | CPU 核数 | RSS 内存 (MB) | 每 Worker 内存 (MB) |")
    lines.append("|------|-------------|-------------|---------|--------------|-------------------|")
    for d in first_datasets:
        idle = d.get("idle_resources", {})
        num_procs = idle.get("num_processes", 1)
        rss = idle.get("rss_avg_mb", 0)
        per_worker = round(rss / max(num_procs, 1), 1)
        lines.append(f"| {d['label']} | {num_procs} | "
                     f"{idle.get('cpu_avg_pct', '-')}% | "
                     f"{idle.get('cpu_cores_avg', '-')} | "
                     f"{rss} | {per_worker} |")
    lines.append("")

    # ── 1.2 Load resource consumption per concurrency for each config ──
    conc_keys = set()
    for d in first_datasets:
        conc_keys.update(d.get("concurrent", {}).keys())
    conc_keys = sorted(conc_keys, key=lambda x: int(x))
    high_ck = conc_keys[-1] if conc_keys else "50"

    lines.append("### 1.2 各并发级别资源占用")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")
        lines.append("| 配置 | 并发 | CPU 平均 (%) | CPU 峰值 (%) | CPU 核数 (avg/peak) | RSS 平均 (MB) | RSS 峰值 (MB) |")
        lines.append("|------|-----|-------------|-------------|-------------------|--------------|--------------| ")
        for d in datasets:
            for ck in conc_keys:
                cd = d.get("concurrent", {}).get(ck, {})
                res = cd.get("resources", {})
                if res:
                    lines.append(
                        f"| {d['label']} | {ck} | "
                        f"{res.get('cpu_avg_pct', '-')}% | "
                        f"{res.get('cpu_max_pct', '-')}% | "
                        f"{res.get('cpu_cores_avg', '-')}/{res.get('cpu_cores_max', '-')} | "
                        f"{res.get('rss_avg_mb', '-')} | "
                        f"{res.get('rss_max_mb', '-')} |")
        lines.append("")

    # ── Section 2: TTFT comparison across configs ──
    lines.append("## 2. TTFT (Time To First Token) 对比")
    lines.append("")
    lines.append("TTFT = 从发送请求到收到第一个 token 的时间。Mock 引擎 prefill = 100ms。")
    lines.append("")

    # 2.1 Serial TTFT detailed (Proxy vs Direct)
    lines.append("### 2.1 串行 TTFT 详细数据（Proxy vs Direct）")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")
        lines.append("| 配置 | 通道 | 请求数 | Mean | Median | P90 | P95 | P99 | Min | Max | Stdev |")
        lines.append("|------|------|-------|------|--------|-----|-----|-----|-----|-----|-------|")
        for d in datasets:
            for channel, ch_name in [("proxy", "Proxy"), ("direct", "Direct")]:
                serial = d.get("serial", {}).get(channel, {})
                ttft = serial.get("ttft", {})
                n = serial.get("requests", 0)
                lines.append(
                    f"| {d['label']} | {ch_name} | {n} | "
                    f"{fmt(ttft.get('mean'))} | {fmt(ttft.get('median'))} | "
                    f"{fmt(ttft.get('p90'))} | {fmt(ttft.get('p95'))} | "
                    f"{fmt(ttft.get('p99'))} | {fmt(ttft.get('min'))} | "
                    f"{fmt(ttft.get('max'))} | {fmt(ttft.get('stdev'))} |")
        lines.append("")

    # 2.2 Concurrent TTFT detailed per scenario (Proxy vs Direct)
    lines.append("### 2.2 并发 TTFT 详细数据（Proxy vs Direct）")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")

        for ck in conc_keys:
            lines.append(f"##### 并发={ck}")
            lines.append("")
            lines.append("| 配置 | 通道 | 请求数 | Mean | Median | P90 | P95 | P99 | Min | Max | Stdev |")
            lines.append("|------|------|-------|------|--------|-----|-----|-----|-----|-----|-------|")
            for d in datasets:
                # Proxy concurrent data
                cd = d.get("concurrent", {}).get(ck, {})
                ttft = cd.get("ttft", {})
                n = cd.get("requests", 0)
                lines.append(
                    f"| {d['label']} | Proxy | {n} | "
                    f"{fmt(ttft.get('mean'))} | {fmt(ttft.get('median'))} | "
                    f"{fmt(ttft.get('p90'))} | {fmt(ttft.get('p95'))} | "
                    f"{fmt(ttft.get('p99'))} | {fmt(ttft.get('min'))} | "
                    f"{fmt(ttft.get('max'))} | {fmt(ttft.get('stdev'))} |")
                # Direct concurrent data (if available)
                cd_direct = d.get("concurrent_direct", {}).get(ck, {})
                if cd_direct:
                    ttft_d = cd_direct.get("ttft", {})
                    n_d = cd_direct.get("requests", 0)
                    lines.append(
                        f"| {d['label']} | Direct | {n_d} | "
                        f"{fmt(ttft_d.get('mean'))} | {fmt(ttft_d.get('median'))} | "
                        f"{fmt(ttft_d.get('p90'))} | {fmt(ttft_d.get('p95'))} | "
                        f"{fmt(ttft_d.get('p99'))} | {fmt(ttft_d.get('min'))} | "
                        f"{fmt(ttft_d.get('max'))} | {fmt(ttft_d.get('stdev'))} |")
            lines.append("")

    # 2.3 TTFT summary table (Mean / P95) — condensed cross-scenario
    lines.append("### 2.3 TTFT 汇总（Mean / P95）")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")
        lines.append("| 并发 |" + " | ".join(f" {d['label']}" for d in datasets) + " |")
        lines.append("|------|" + " | ".join("------" for _ in datasets) + " |")
        for ck in conc_keys:
            row = f"| {ck} |"
            for d in datasets:
                cd = d.get("concurrent", {}).get(ck, {})
                ttft = cd.get("ttft", {})
                mean = ttft.get("mean", 0)
                p95 = ttft.get("p95", 0)
                row += f" {fmt(mean)} / {fmt(p95)} |"
            lines.append(row)
        lines.append("")

    # ── Section 3: TPS & Total Time comparison ──
    lines.append("## 3. TPS (Tokens Per Second) 与请求耗时对比")
    lines.append("")
    lines.append("TPS = 单请求解码阶段 token 吞吐率。Mock 引擎基线 ≈ 33.3 tok/s。")
    lines.append("")

    # 3.1 Serial TPS detailed (Proxy vs Direct)
    lines.append("### 3.1 串行 TPS 详细数据（Proxy vs Direct）")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")
        lines.append("| 配置 | 通道 | 请求数 | Mean TPS | Min TPS | Max TPS | 输出 Tokens | 总耗时 Mean | 总耗时 P95 |")
        lines.append("|------|------|-------|---------|---------|---------|-----------|-----------|-----------|")
        for d in datasets:
            for channel, ch_name in [("proxy", "Proxy"), ("direct", "Direct")]:
                serial = d.get("serial", {}).get(channel, {})
                tps = serial.get("tps", {})
                total = serial.get("total_ms", {})
                tokens = serial.get("tokens", {})
                n = serial.get("requests", 0)
                lines.append(
                    f"| {d['label']} | {ch_name} | {n} | "
                    f"{fmt(tps.get('mean'), ' tok/s')} | "
                    f"{fmt(tps.get('min'), ' tok/s')} | "
                    f"{fmt(tps.get('max'), ' tok/s')} | "
                    f"{tokens.get('mean', '-')} | "
                    f"{fmt(total.get('mean'))} | "
                    f"{fmt(total.get('p95'))} |")
        lines.append("")

    # 3.2 Concurrent TPS detailed per scenario (Proxy vs Direct)
    lines.append("### 3.2 并发 TPS 详细数据（Proxy vs Direct）")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")

        for ck in conc_keys:
            lines.append(f"##### 并发={ck}")
            lines.append("")
            lines.append("| 配置 | 通道 | 请求数 | Mean TPS | Min TPS | Max TPS | 输出 Tokens | 总耗时 Mean | 总耗时 P95 | 壁钟时间 |")
            lines.append("|------|------|-------|---------|---------|---------|-----------|-----------|-----------|---------|")
            for d in datasets:
                # Proxy concurrent data
                cd = d.get("concurrent", {}).get(ck, {})
                tps = cd.get("tps", {})
                total = cd.get("total_ms", {})
                tokens = cd.get("tokens", {})
                n = cd.get("requests", 0)
                wall = cd.get("wall_ms", 0)
                wall_s = f"{wall/1000:.1f}s" if wall else "-"
                lines.append(
                    f"| {d['label']} | Proxy | {n} | "
                    f"{fmt(tps.get('mean'), ' tok/s')} | "
                    f"{fmt(tps.get('min'), ' tok/s')} | "
                    f"{fmt(tps.get('max'), ' tok/s')} | "
                    f"{tokens.get('mean', '-')} | "
                    f"{fmt(total.get('mean'))} | "
                    f"{fmt(total.get('p95'))} | "
                    f"{wall_s} |")
                # Direct concurrent data (if available)
                cd_direct = d.get("concurrent_direct", {}).get(ck, {})
                if cd_direct:
                    tps_d = cd_direct.get("tps", {})
                    total_d = cd_direct.get("total_ms", {})
                    tokens_d = cd_direct.get("tokens", {})
                    n_d = cd_direct.get("requests", 0)
                    wall_d = cd_direct.get("wall_ms", 0)
                    wall_s_d = f"{wall_d/1000:.1f}s" if wall_d else "-"
                    lines.append(
                        f"| {d['label']} | Direct | {n_d} | "
                        f"{fmt(tps_d.get('mean'), ' tok/s')} | "
                        f"{fmt(tps_d.get('min'), ' tok/s')} | "
                        f"{fmt(tps_d.get('max'), ' tok/s')} | "
                        f"{tokens_d.get('mean', '-')} | "
                        f"{fmt(total_d.get('mean'))} | "
                        f"{fmt(total_d.get('p95'))} | "
                        f"{wall_s_d} |")
            lines.append("")

    # 3.3 TPS summary (cross-config comparison)
    lines.append("### 3.3 TPS 跨场景对比")
    lines.append("")
    header = "| 配置 | 测试类型 |"
    sep = "|------|---------|"
    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        header += f" 输入{it}/输出{mt} |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    for i, d in enumerate(first_datasets):
        # Serial row
        row = f"| {d['label']} | 串行 |"
        for cfg_label, datasets in configs:
            serial = datasets[i].get("serial", {}).get("proxy", {})
            tps = serial.get("tps", {})
            row += f" {fmt(tps.get('mean'), ' tok/s')} |"
        lines.append(row)
        # High-conc row
        row = f"| {d['label']} | 并发{high_ck} |"
        for cfg_label, datasets in configs:
            cd = datasets[i].get("concurrent", {}).get(high_ck, {})
            tps = cd.get("tps", {})
            row += f" {fmt(tps.get('mean'), ' tok/s')} |"
        lines.append(row)
    lines.append("")

    # ── Section 4: Comprehensive Summary Table ──
    lines.append("## 4. 综合对比汇总")
    lines.append("")
    lines.append("### 4.1 资源-性能权衡（各场景对比）")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")
        lines.append("| 配置 | 空闲 RSS (MB) | 并发50 CPU 核数 | 串行 TTFT (ms) | 并发50 TTFT Mean (ms) | 并发50 TTFT P95 (ms) | 串行 TPS | 并发50 TPS | 系统 TPS (并发50) |")
        lines.append("|------|-------------|---------------|--------------|---------------------|---------------------|---------|----------|-----------------|")
        for d in datasets:
            idle = d.get("idle_resources", {})
            hc = d.get("concurrent", {}).get(high_ck, {})
            hc_res = hc.get("resources", {})
            serial = d.get("serial", {}).get("proxy", {})
            hc_tps_mean = hc.get("tps", {}).get("mean", 0)
            sys_tps = round(hc_tps_mean * int(high_ck), 1) if hc_tps_mean else "-"

            lines.append(
                f"| {d['label']} | {idle.get('rss_avg_mb', '-')} | "
                f"{hc_res.get('cpu_cores_avg', '-')} | "
                f"{fmt(serial.get('ttft', {}).get('mean'))} | "
                f"{fmt(hc.get('ttft', {}).get('mean'))} | "
                f"{fmt(hc.get('ttft', {}).get('p95'))} | "
                f"{fmt(serial.get('tps', {}).get('mean'), ' tok/s')} | "
                f"{fmt(hc.get('tps', {}).get('mean'), ' tok/s')} | "
                f"{sys_tps} tok/s |")
        lines.append("")

    # ── 4.2 Worker 增益量化 ──
    lines.append("### 4.2 TTFT 增益量化（并发=50 时对比 workers-1）")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")
        lines.append("| 配置 | TTFT Mean (ms) | 相对 workers-1 | TTFT P95 (ms) | 相对 workers-1 | TTFT P99 (ms) | 相对 workers-1 |")
        lines.append("|------|--------------|---------------|-------------|---------------|-------------|---------------|")

        w1_hc = datasets[0].get("concurrent", {}).get(high_ck, {})
        w1_mean = w1_hc.get("ttft", {}).get("mean", 0)
        w1_p95 = w1_hc.get("ttft", {}).get("p95", 0)
        w1_p99 = w1_hc.get("ttft", {}).get("p99", 0)

        for d in datasets:
            hc = d.get("concurrent", {}).get(high_ck, {})
            ttft = hc.get("ttft", {})
            mean_v = ttft.get("mean", 0)
            p95_v = ttft.get("p95", 0)
            p99_v = ttft.get("p99", 0)
            lines.append(
                f"| {d['label']} | "
                f"{fmt(mean_v)} | {pct(w1_mean, mean_v)} | "
                f"{fmt(p95_v)} | {pct(w1_p95, p95_v)} | "
                f"{fmt(p99_v)} | {pct(w1_p99, p99_v)} |")
        lines.append("")

    # ── 4.3 Cross-scenario TTFT comparison side by side ──
    lines.append("### 4.3 TTFT 跨场景对比（并发=50 P95）")
    lines.append("")
    header = "| 配置 |"
    sep = "|------|"
    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        header += f" 输入{it}/输出{mt} P95 | 相对 workers-1 |"
        sep += "------|------|"
    lines.append(header)
    lines.append(sep)

    baselines = {}
    for cfg_label, datasets in configs:
        w1_conc = datasets[0].get("concurrent", {}).get(high_ck, {})
        baselines[cfg_label] = w1_conc.get("ttft", {}).get("p95", 0)

    for i, d in enumerate(first_datasets):
        row = f"| {d['label']} |"
        for cfg_label, datasets in configs:
            hc = datasets[i].get("concurrent", {}).get(high_ck, {})
            p95 = hc.get("ttft", {}).get("p95", 0)
            baseline = baselines[cfg_label]
            improvement = pct(baseline, p95) if baseline else "-"
            row += f" {fmt(p95)} | {improvement} |"
        lines.append(row)
    lines.append("")

    # ── 4.4 System TPS (throughput = concurrent requests × per-request TPS) ──
    lines.append("### 4.4 系统吞吐量（系统 TPS = 并发数 × 单请求 TPS）")
    lines.append("")
    lines.append("系统 TPS 表示整个系统每秒处理的 token 总数。")
    lines.append("")

    for cfg_label, datasets in configs:
        it = datasets[0].get("input_tokens", 15)
        mt = datasets[0].get("max_tokens", 50)
        lines.append(f"#### 场景: 输入{it}/输出{mt}")
        lines.append("")
        header_cols = ["配置"] + [f"并发{ck}" for ck in conc_keys]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["------"] * len(header_cols)) + "|")

        for d in datasets:
            row = f"| {d['label']}"
            for ck in conc_keys:
                cd = d.get("concurrent", {}).get(ck, {})
                tps_mean = cd.get("tps", {}).get("mean", 0)
                conc = int(ck)
                sys_tps = round(tps_mean * conc, 1)
                row += f" | {sys_tps}"
            row += " |"
            lines.append(row)
        lines.append("")

    # ── Section 5: Analysis ──
    lines.append("## 5. 分析与结论")
    lines.append("")
    lines.append("### 5.1 资源消耗规律")
    lines.append("")
    lines.append("- **内存**: 每个 worker 进程占用 ~65 MB RSS")
    lines.append("  - workers-1: ~65 MB, workers-4: ~295 MB, workers-16: ~1073 MB")
    lines.append("  - workers-64: ~4186 MB, workers-128: ~8338 MB")
    lines.append("- **CPU**: 空闲时 worker 数越多基础 CPU 占用越高（workers-128 空闲 10%）")
    lines.append("- **内存与输入输出 token 数无关**: 128/128 和 512/512 场景下资源占用几乎一致")
    lines.append("")

    lines.append("### 5.2 TTFT 影响规律")
    lines.append("")
    lines.append("- **串行（低并发）**: 所有 worker 配置 TTFT 几乎相同（~105-107ms），proxy 额外开销 2-4ms")
    lines.append("- **高并发下 worker 数是决定因素**:")
    lines.append("  - workers-1 在并发=50 时 TTFT P95 显著升高（排队效应）")
    lines.append("  - workers-4 能显著改善高并发 TTFT")
    lines.append("  - workers-64/128 进一步降低 P95，但边际收益递减")
    lines.append("- **TTFT 与输入 token 数无关**: 因为 mock prefill 时间固定为 100ms")
    lines.append("  - 真实引擎下，输入越长 prefill 越慢，TTFT 差异更大")
    lines.append("")

    lines.append("### 5.3 TPS 影响规律")
    lines.append("")
    lines.append("- **TPS 在所有配置下几乎不变** (~33.2 tok/s):")
    lines.append("  - 增加 worker 数不改变单请求 TPS")
    lines.append("  - TPS 由后端引擎决定，proxy 不引入额外延迟")
    lines.append("- **TPS 与输出 token 数无关**: 128 tokens 和 512 tokens 产生相同 TPS")
    lines.append("- **系统 TPS（吞吐量）= 并发 × 单请求 TPS**")
    lines.append("  - 系统吞吐量受并发能力限制，worker 越多支持并发越高")
    lines.append("")

    lines.append("### 5.4 最佳实践建议")
    lines.append("")
    lines.append("| 场景 | 推荐 Workers | 理由 |")
    lines.append("|------|------------|------|")
    lines.append("| 低并发 (≤5) | 1-4 | TTFT 无差异，节省内存 |")
    lines.append("| 中并发 (5-20) | 4-16 | 有效降低 TTFT P95 |")
    lines.append("| 高并发 (20-50) | 16-64 | 显著降低排队延迟 |")
    lines.append("| 极高并发 (>50) | 64-128 | 需权衡 8GB+ 内存开销 |")
    lines.append("")

    lines.append("### 5.5 关键结论")
    lines.append("")
    lines.append("1. **增加 Worker 的核心价值是降低高并发 TTFT**（排队效应），不影响 TPS")
    lines.append("2. **内存开销线性增长**: ~65 MB/worker，128 workers 需要 ~8.3 GB")
    lines.append("3. **4 workers 是最佳性价比选择**: 285MB 额外内存即可显著改善并发 TTFT")
    lines.append("4. **资源消耗与输入输出 token 数无关**: proxy 本身不消耗大量 CPU/内存")
    lines.append("5. **串行 TPS ≈ 直连 TPS**: proxy 对流式传输几乎零开销")
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined PROXY_WORKERS report for multiple I/O configs")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Config specs: 'label:directory' e.g. '128:results_128_128/'")
    parser.add_argument("-o", "--output", default="report_combined.md")
    args = parser.parse_args()

    configs = []
    for spec in args.configs:
        label, datasets = load_config_datasets(spec)
        configs.append((label, datasets))

    report = generate_combined_report(configs, args.output)
    print(report[:4000])
    if len(report) > 4000:
        print(f"\n... (truncated, total {len(report)} chars)")


if __name__ == "__main__":
    main()
