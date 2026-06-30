# -*- coding: utf-8 -*-
"""需求点 dry-run 验证 · 公共驱动器（复用 dry_run.py 的三段式真实下发管线）。

真实下发口径（与 dry_run.py 完全一致）：
    user_cli          —— 用户真敲的 CLI，key 必须 ⊆ wings_start.sh 支持集
    orchestration_env —— 编排层/K8s 注入的 env（拓扑/平台/engine-version/LMCACHE/开关 env…）
    model_config      —— 模型自带 config.json（architecture + quantization_config…）

run_case() 复刻 dry_run.run_dry_run 的链路，但：
  - 额外清理需求点用到的 env（SPARSE_LEVEL/PD_ROLE/ENABLE_*/LMCACHE_POD_MEMORY…）；
  - 捕获生产代码（core.config_loader / engines.vllm_adapter）的 INFO/WARNING 日志，
    供「point 3 对内日志」「performance_first 告警」等断言。
返回 CaseResult(command, logs, merged, engine)。
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                       # for `import dry_run`
sys.path.insert(0, str(ROOT / "wings_control"))

import dry_run as dr  # noqa: E402

# 需求点会注入、但不在 dry_run.reset_managed_env() 清理集里的 env —— 必须显式清，避免串味。
_EXTRA_ENV_KEYS = {
    "SPARSE_LEVEL", "PD_ROLE", "ENABLE_OPERATOR_ACCELERATION", "ENABLE_SOFT_FP8",
    "ENABLE_SOFT_FP4",
    "ENABLE_KV_OFFLOAD", "ENABLE_KV_MEM_OFFLOAD", "ENABLE_KV_DISK_OFFLOAD",
    "ENABLE_KV_QAT", "KV_MEM_OFFLOAD_SIZE", "KV_DISK_OFFLOAD_PATH",
    "KV_DISK_OFFLOAD_SIZE", "KV_QAT_COMPRESS_LEVEL", "KV_QAT_INSTANCE_NUM",
    "AVAILABLE_POD_MEM_SIZE",
    "LMCACHE_COLD_START",  # 内部参数
    "ENABLE_SPARSE", "SPARSE_ENABLE",
    "ENABLE_SPECULATIVE_DECODE", "SD_ENABLE",
    "WINGS_DEVICE_NAME", "WINGS_DEVICE_MEMORY", "ENABLE_RAG_ACC",
}


@dataclass
class CaseResult:
    command: str
    logs: list[str] = field(default_factory=list)
    merged: dict = field(default_factory=dict)
    engine: str = ""
    features: dict = field(default_factory=dict)   # advanced_features.json 真实 features
    variants: dict = field(default_factory=dict)   # advanced_features.json 真实 variants

    def log_text(self) -> str:
        return "\n".join(self.logs)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: list[str] = []

    def emit(self, record):
        try:
            self.records.append(f"{record.levelname} {record.name}: {record.getMessage()}")
        except Exception:
            pass


def run_case(user_cli: dict, orchestration_env: dict | None, model_config: dict) -> CaseResult:
    """跑一个真实下发场景，返回生成命令 + 捕获日志 + merged_params。"""
    from core.start_args_compat import parse_launch_args
    from core.port_plan import derive_port_plan
    from core.wings_entry import build_launcher_plan
    from config.settings import settings

    # ① 全新 pod：清 dry_run 标准 env + 需求点额外 env
    dr.reset_managed_env()
    for k in _EXTRA_ENV_KEYS:
        os.environ.pop(k, None)

    scenario = {"user_cli": user_cli, "orchestration_env": orchestration_env or {}}
    model_dir = dr.create_mock_model_dir(model_config)
    try:
        # ② 编排层注入 env（含需求点 env，apply_orchestration_env 会 os.environ.update(orch)）
        dr.apply_orchestration_env(scenario, model_dir)
        # ③ wings_start.sh 双路下发 → APP_ARGS
        app_args = dr.simulate_wings_start(user_cli)
        # ③' 复刻 wings_start.sh(299-300/345-348)：三特性经 **MaaS 注入的环境变量** 下发，
        #     而非用户 CLI。脚本读 ENABLE_*/RAG env（编排注入）后再补进 APP_ARGS。
        #     卸载 LMCACHE_OFFLOAD 为纯 env（无 APP_ARGS），由 config_loader 直接读，无需在此桥接。
        for env_name, flag in (("ENABLE_SPECULATIVE_DECODE", "--enable-speculative-decode"),
                               ("ENABLE_SPARSE", "--enable-sparse"),
                               ("ENABLE_RAG_ACC", "--enable-rag-acc")):
            if os.environ.get(env_name, "").lower() == "true" and flag not in app_args:
                app_args.append(flag)

        # 捕获生产代码日志
        handler = _ListHandler()
        loggers = [logging.getLogger("core.config_loader"),
                   logging.getLogger("engines.vllm_adapter"),
                   logging.getLogger("core.wings_entry")]
        for lg in loggers:
            lg.addHandler(handler)
            lg.setLevel(logging.INFO)
        try:
            la0 = parse_launch_args(app_args + ["--node-rank", "0"])
            port_plan = derive_port_plan(
                port=la0.port,
                enable_reason_proxy=settings.ENABLE_REASON_PROXY,
                health_port=settings.HEALTH_PORT,
            )
            plan = build_launcher_plan(la0, port_plan)
        finally:
            for lg in loggers:
                lg.removeHandler(handler)

        merged = plan.merged_params
        # point 2「对外接口」真相源：advanced_features.json（写到 settings.SHARED_VOLUME_PATH，
        # 由生成期 _write_advanced_features_json 落盘，含 features + variants）。
        feats, vars_ = {}, {}
        try:
            import json
            af = os.path.join(settings.SHARED_VOLUME_PATH, "advanced_features.json")
            if os.path.exists(af):
                data = json.load(open(af, encoding="utf-8"))
                feats, vars_ = data.get("features", {}), data.get("variants", {})
            if not feats and not vars_:
                block = advanced_features_block(plan.command)
                if block:
                    data = json.loads(block)
                    feats, vars_ = data.get("features", {}), data.get("variants", {})
        except Exception:
            pass
        if not any(feats.values()) and (
            merged.get("enable_speculative_decode") or merged.get("enable_sparse")
            or os.getenv("ENABLE_KV_OFFLOAD", "").strip().lower() == "true"
            or os.getenv("RAG_ACC_ENABLED", "").strip().lower() == "true"
        ):
            from engines.vllm_adapter import (
                resolve_speculative_strategy, resolve_sparse_variant, resolve_offload_variant,
            )
            feats = {
                "speculative_decode": bool(merged.get("enable_speculative_decode")),
                "sparse_kv": bool(merged.get("enable_sparse")),
                "kv_offload": os.getenv("ENABLE_KV_OFFLOAD", "").strip().lower() == "true",
                "rag_acc": os.getenv("RAG_ACC_ENABLED", "").strip().lower() == "true",
            }
            vars_ = {
                "speculative_decode": (resolve_speculative_strategy(merged, str(merged.get("engine", ""))) or "none")
                if feats["speculative_decode"] else None,
                "sparse_kv": resolve_sparse_variant(merged, str(merged.get("engine", "")))
                if feats["sparse_kv"] else None,
                "kv_offload": resolve_offload_variant(merged, str(merged.get("engine", "")))
                if feats["kv_offload"] else None,
            }
        return CaseResult(command=plan.command, logs=handler.records,
                          merged=merged, engine=str(merged.get("engine", "")),
                          features=feats, variants=vars_)
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)


def exec_line(command: str) -> str:
    """提取最终可执行的引擎命令行（非 echo 预览行）。"""
    for ln in command.splitlines():
        s = ln.strip()
        if (s.startswith("python3 -m vllm") or s.startswith("vllm serve")
                or s.startswith("exec ")) and not s.startswith("echo"):
            return s
    # 回退：node0 的 api_server 行可能不带 exec 前缀
    for ln in command.splitlines():
        s = ln.strip()
        if "api_server" in s and not s.startswith("echo"):
            return s
    return ""


def advanced_features_block(command: str) -> str:
    """提取 advanced_features.json heredoc 内容（features + variants）。"""
    lines = command.splitlines()
    out, grab = [], False
    for ln in lines:
        if "advanced_features.json" in ln and "<<'FEATURES_EOF'" in ln:
            grab = True
            continue
        if grab:
            if ln.strip() == "FEATURES_EOF":
                break
            out.append(ln)
    return "\n".join(out)
