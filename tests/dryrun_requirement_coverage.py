# -*- coding: utf-8 -*-
"""需求一 · 三特性使能改造 —— 八个需求点的 dry-run 覆盖验证（模拟用户真实下发）。

真实下发口径（三段式，与 dry_run.py / wings_start.sh 一致）：
    user_cli          —— 用户真敲 CLI（key ⊆ wings_start.sh 支持集）
    orchestration_env —— 编排层注入 env（拓扑/平台/engine-version/LMCACHE/开关 env…）
    model_config      —— 模型 config.json（architecture + quantization_config…）

每个 case 端到端生成 start_command.sh + 捕获日志 + 读取 advanced_features.json，
对「需求点的可观测产物」做断言。覆盖：
    P1 删 fp4/fp8 + 引擎路由删除      P2 对外接口 advanced_features.json
    P3 对内特性日志                   P4 三特性依赖白名单（含 PD 一票否决）
    P5 内存自动计算 C4                P6 稀疏多模式 SPARSE_LEVEL
    P7 硬件信息/ENGINE-VERSION 卡型   P8 打补丁逻辑（NV 装 / ascend 不装）

运行：python tests/dryrun_requirement_coverage.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
from _dryrun_req_harness import run_case, CaseResult  # noqa: E402

OUT_PATH = TESTS_DIR / "dryrun_requirement_coverage_output.txt"
_OUT = None
_total = {"pass": 0, "fail": 0}


def emit(s: str = ""):
    if _OUT is not None:
        _OUT.write(s + "\n")
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


# ── 产物提取助手 ─────────────────────────────────────────────────────────────
def spec_method(r: CaseResult):
    m = re.search(r'--speculative-config \'\{[^\']*?"method"\s*:\s*"([a-z0-9_]+)"', r.command)
    return m.group(1) if m else None


def cpu_size(r: CaseResult):
    """auto/custom 写回的 LMCACHE_MAX_LOCAL_CPU_SIZE 数值（取 export 赋值，非 echo 回显）。"""
    vals = re.findall(r'export LMCACHE_MAX_LOCAL_CPU_SIZE=([0-9]+)', r.command)
    return int(vals[0]) if vals else None


def native_cpu_swap(r: CaseResult):
    m = re.search(r'cpu_swap_space_gb"?\s*:?\s*([0-9]+)', r.command)
    return int(m.group(1)) if m else None


def has(r: CaseResult, sub: str) -> bool:
    return sub in r.command


def log_has(r: CaseResult, sub: str) -> bool:
    return any(sub in l for l in r.logs)


def install_indexcache(r: CaseResult) -> bool:
    return any("install.py" in l and "indexcache" in l for l in r.command.splitlines())


# ── case 框架 ────────────────────────────────────────────────────────────────
def check(point: str, name: str, r: CaseResult, asserts: list[tuple[bool, str]]):
    ok = all(a for a, _ in asserts)
    _total["pass" if ok else "fail"] += 1
    emit(f"  [{ '✓ PASS' if ok else '✗ FAIL'}] {point} · {name}")
    for a, desc in asserts:
        if not a:
            emit(f"        ✗ {desc}")
        else:
            emit(f"        · {desc}")
    return ok


# ════════════════════════════════════════════════════════════════════════════
def run_all():
    emit("=" * 92)
    emit(" 需求一 · 三特性使能 —— 八点 dry-run 覆盖验证（真实下发）")
    emit("=" * 92)

    # ── P1 · 删 fp4/fp8 + 引擎路由删除 ──────────────────────────────────────
    emit("\n## P1 · 删 fp4/fp8 + 引擎路由删除（ENABLE_OPERATOR_ACCELERATION / ENABLE_SOFT_FP8 旁路）")
    uc = {"model-name": "Qwen3.5-397B-A17B", "engine": "vllm_ascend", "device-count": 16}
    mc = {"architecture": "Qwen3_5MoeForConditionalGeneration",
          "quantization_config": {"quant_method": "ascend"}}
    base_orch = {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_ASCEND_PLATFORM": "a3"}
    r_off = run_case(uc, base_orch, mc)
    r_on = run_case(uc, {**base_orch, "ENABLE_OPERATOR_ACCELERATION": "true",
                         "ENABLE_SOFT_FP8": "true"}, mc)

    def _norm(s):
        return re.sub(r'/build/(model|sv)_[A-Za-z0-9_]+', '<T>', s)
    check("P1", "ENABLE_OPERATOR_ACCELERATION/ENABLE_SOFT_FP8 对启动产物完全无效（旁路已删）", r_on, [
        (r_on.engine == "vllm_ascend" == r_off.engine, f"engine 不受影响 (on={r_on.engine}, off={r_off.engine})"),
        (_norm(r_on.command) == _norm(r_off.command), "设/不设这两个 env，start_command 归一化后字节一致"),
    ])
    check("P1", "use_kunlun_atb 死代码已删：命令不导出 USE_KUNLUN_ATB", r_on, [
        (not has(r_on, "USE_KUNLUN_ATB"), "命令中不出现 USE_KUNLUN_ATB（即便设了算子加速 env）"),
    ])

    # ── P2 · 对外接口 advanced_features.json（features + variants）──────────
    emit("\n## P2 · 对外特性状态接口 advanced_features.json（features + variants）")
    r = run_case({"model-name": "GLM-5.2-w8a8", "engine": "vllm_ascend", "device-count": 16,
                  "enable-speculative-decode": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3"},
                 {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    check("P2", "GLM-5.2·Ascend spec：features+variants 如实透出 deepseek_mtp", r, [
        (set(("speculative_decode", "sparse_kv", "kv_offload", "rag_acc")) <= set(r.features), "features 含四特性 bool"),
        (r.features.get("speculative_decode") is True, "features.speculative_decode == true"),
        (r.variants.get("speculative_decode") == "deepseek_mtp", "variants.speculative_decode == deepseek_mtp"),
        (r.features.get("sparse_kv") is False and r.features.get("kv_offload") is False, "sparse_kv / kv_offload == false（GLM-5.2 仅投机）"),
    ])
    r2 = run_case({"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8, "enable-sparse": True},
                  {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", "WINGS_ASCEND_PLATFORM": "a2"},
                  {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    check("P2", "GLM-5.1·Ascend sparse：variants 透出 indexcache_topk8", r2, [
        (r2.features.get("sparse_kv") is True, "features.sparse_kv == true"),
        (r2.variants.get("sparse_kv") == "indexcache_topk8", "variants.sparse_kv == indexcache_topk8"),
    ])

    # ── P3 · 对内特性日志（本轮新增）────────────────────────────────────────
    emit("\n## P3 · 对内特性日志（卡型 miss 告警 / 收口 req→eff 摘要 / sparse 抑制对称日志）")
    r = run_case({"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "ray"},  # 无 platform / engine-version / device-name
                 {"architecture": "GlmMoeDsaForCausalLM"})
    check("P3", "Ascend 卡型解析失败 → WARNING（白名单将全 miss）", r, [
        (log_has(r, "card_token unresolved on Ascend"), "命中 [SmartFeature] card_token unresolved 告警"),
        (log_has(r, "effective enablement"), "收口摘要行存在"),
    ])
    r = run_case({"model-name": "Llama-3-70B", "engine": "vllm", "device-count": 8,
                  "enable-sparse": True, "enable-speculative-decode": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"}, {"architecture": "LlamaForCausalLM"})
    check("P3", "非白名单 + enable-sparse → sparse 抑制对称日志 + req→eff 摘要", r, [
        (log_has(r, "sparse requested but not in whitelist"), "命中 sparse 抑制日志"),
        (log_has(r, "sparse True->False"), "摘要含 sparse True->False（请求→有效）"),
    ])
    r = run_case({"model-name": "glm-4.7", "engine": "vllm", "device-count": 8, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "LMCACHE_OFFLOAD": "true"},
                 {"architecture": "Glm4MoeForCausalLM"})  # glm-4.7 NV 含 offload 白名单 → 不抑制；换非白名单看抑制
    r = run_case({"model-name": "Llama-3-70B", "engine": "vllm", "device-count": 8},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "LMCACHE_OFFLOAD": "true"},
                 {"architecture": "LlamaForCausalLM"})
    check("P3", "非白名单 + LMCACHE_OFFLOAD → offload 抑制对称日志", r, [
        (log_has(r, "offload requested but not in whitelist"), "命中 offload 抑制日志"),
    ])

    # ── P4 · 三特性依赖白名单（含 PD 一票否决）──────────────────────────────
    emit("\n## P4 · 三特性依赖白名单门控")
    r = run_case({"model-name": "GLM-5.2-w8a8", "engine": "vllm_ascend", "device-count": 16,
                  "enable-speculative-decode": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3"},
                 {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    check("P4", "命中白名单 spec → deepseek_mtp（GLM-5.2·Ascend）", r, [
        (spec_method(r) == "deepseek_mtp", f"spec method == deepseek_mtp (实际 {spec_method(r)})"),
    ])
    r = run_case({"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8,
                  "enable-speculative-decode": True, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", "WINGS_ASCEND_PLATFORM": "a2"},
                 {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    check("P4", "白名单 sparse-only → spec 地板 suffix（GLM-5.1·Ascend，B.4 bug 修复）", r, [
        (spec_method(r) == "suffix", f"spec method == suffix (实际 {spec_method(r)})"),
        (has(r, '"index_topk_freq": 8'), "sparse 仍产 indexcache_topk8"),
    ])
    r = run_case({"model-name": "qwen3.5-397b", "engine": "vllm", "device-count": 8,
                  "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "LMCACHE_OFFLOAD": "true", "LMCACHE_POD_MEMORY": "512"},
                 {"architecture": "Qwen3_5MoeForConditionalGeneration"})
    check("P4", "offload 不在白名单 → 收口关（Qwen3.5-397B·NV：spec,sparse 无 offload）", r, [
        (r.features.get("kv_offload") is False, "features.kv_offload == false"),
        (not has(r, "export LMCACHE_OFFLOAD=true"), "不导出 LMCACHE_OFFLOAD=true"),
    ])
    r = run_case({"model-name": "glm-4.7", "engine": "vllm", "device-count": 8,
                  "enable-speculative-decode": True, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "LMCACHE_OFFLOAD": "true",
                  "LMCACHE_POD_MEMORY": "512", "PD_ROLE": "P"},
                 {"architecture": "Glm4MoeForCausalLM"})
    check("P4", "PD 一票否决 → 三特性全关", r, [
        (log_has(r, "PD role detected"), "命中 PD veto 日志"),
        (r.features.get("speculative_decode") is False and r.features.get("sparse_kv") is False
         and r.features.get("kv_offload") is False, "advanced_features 三特性全 false"),
        (spec_method(r) in (None, "suffix"), "无 mtp 投机（PD 下不产 mtp）"),
    ])

    # ── P5 · 内存自动计算 C4（auto per-card / native node-total / custom / 熔断）──
    emit("\n## P5 · KV 卸载 auto 容量反向预算（C4）")
    # M_offload = 512 - (7*TP*DP+3) - 512*0.1 = 512-59-51 = 401(向下取整/取整规则); per-card = 401//8 = 50
    r = run_case({"model-name": "glm-4.7", "engine": "vllm", "device-count": 8, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "LMCACHE_OFFLOAD": "true", "LMCACHE_POD_MEMORY": "512"},
                 {"architecture": "Glm4MoeForCausalLM"})
    check("P5", "auto LMCache：写回 per-card 容量 + 强制 swap_space=0（glm-4.7·NV, POD=512/8卡）", r, [
        (cpu_size(r) == 50, f"LMCACHE_MAX_LOCAL_CPU_SIZE == 50 = M_offload(401)//8 (实际 {cpu_size(r)})"),
        (has(r, "--swap-space 0"), "强制 --swap-space 0"),
    ])
    r = run_case({"model-name": "DeepSeek-V4-Flash", "engine": "vllm_ascend", "device-count": 8},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "WINGS_ASCEND_PLATFORM": "a2",
                  "LMCACHE_OFFLOAD": "true", "LMCACHE_POD_MEMORY": "512"},
                 {"architecture": "DeepseekV4ForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    check("P5", "auto native：整节点 M_offload 不除卡数（V4-Flash·Ascend cpu_swap_space_gb）", r, [
        (native_cpu_swap(r) == 401, f"cpu_swap_space_gb == 401 = 整节点 M_offload（不除卡数）(实际 {native_cpu_swap(r)})"),
    ])
    r = run_case({"model-name": "glm-4.7", "engine": "vllm", "device-count": 8, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "LMCACHE_OFFLOAD": "true",
                  "LMCACHE_MAX_LOCAL_CPU_SIZE": "200"},  # custom：给数值 → 透传不算
                 {"architecture": "Glm4MoeForCausalLM"})
    check("P5", "custom：给 LMCACHE_MAX_LOCAL_CPU_SIZE 数值 → 透传不计算", r, [
        (cpu_size(r) == 200, f"LMCACHE_MAX_LOCAL_CPU_SIZE == 200（透传）(实际 {cpu_size(r)})"),
    ])
    r = run_case({"model-name": "glm-4.7", "engine": "vllm", "device-count": 8, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "LMCACHE_OFFLOAD": "true", "LMCACHE_POD_MEMORY": "100"},
                 {"architecture": "Glm4MoeForCausalLM"})  # M_offload=100-59-10=31 < 100 → 熔断
    check("P5", "熔断：可用容量 < 下限(100G) → 不建 CPU 卸载池", r, [
        (cpu_size(r) is None, f"无 auto 写回的 CPU 容量（熔断）(实际 {cpu_size(r)})"),
        (log_has(r, "below floor") or log_has(r, "skip CPU offload"), "命中熔断告警日志"),
    ])

    # ── P6 · 稀疏多模式 SPARSE_LEVEL ────────────────────────────────────────
    emit("\n## P6 · 稀疏精度/性能档位 SPARSE_LEVEL")
    base = ({"model-name": "glm-5.1", "engine": "vllm", "device-count": 8, "enable-sparse": True},
            {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"}, {"architecture": "GlmMoeDsaForCausalLM"})
    r_acc = run_case(base[0], base[1], base[2])
    check("P6", "缺省 → accuracy_first（无告警）", r_acc, [
        (log_has(r_acc, "effective SPARSE_LEVEL=accuracy_first"), "日志 effective SPARSE_LEVEL=accuracy_first"),
        (not log_has(r_acc, "performance_first not implemented"), "无 performance_first 告警"),
    ])
    r_perf = run_case(base[0], {**base[1], "SPARSE_LEVEL": "performance_first"}, base[2])
    check("P6", "performance_first → 告警回落 accuracy_first，命令不变", r_perf, [
        (log_has(r_perf, "performance_first not implemented"), "命中 performance_first 告警"),
        (log_has(r_perf, "effective SPARSE_LEVEL=accuracy_first"), "有效档位回落 accuracy_first"),
        (re.sub(r'/build/(model|sv)_[A-Za-z0-9_]+', '<T>', r_perf.command)
         == re.sub(r'/build/(model|sv)_[A-Za-z0-9_]+', '<T>', r_acc.command), "命令与 accuracy_first 一致"),
    ])
    r_bad = run_case(base[0], {**base[1], "SPARSE_LEVEL": "turbo"}, base[2])
    check("P6", "非法值 turbo → 回落 accuracy_first（无 performance 告警）", r_bad, [
        (log_has(r_bad, "effective SPARSE_LEVEL=accuracy_first"), "回落 accuracy_first"),
        (not log_has(r_bad, "performance_first not implemented"), "非法值不触发 performance 告警"),
    ])

    # ── P7 · 硬件信息 / ENGINE-VERSION 卡型解析 ─────────────────────────────
    emit("\n## P7 · 卡型解析（保留老逻辑 + 依赖 ENGINE-VERSION）")
    # deepseek-v3.2 仅 910c 白名单：a3→910c 命中 spec；a2→910b 不命中 → suffix
    v32_uc = {"model-name": "DeepSeek-V3.2", "engine": "vllm_ascend", "device-count": 16,
              "enable-speculative-decode": True}
    v32_mc = {"architecture": "DeepseekV32ForCausalLM", "quantization_config": {"quant_method": "ascend"}}
    r_a3 = run_case(v32_uc, {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3"}, v32_mc)
    r_a2 = run_case(v32_uc, {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a2"}, v32_mc)
    sv_a3 = r_a3.variants.get("speculative_decode")
    sv_a2 = r_a2.variants.get("speculative_decode")
    check("P7", "ENGINE_VERSION 后缀定卡型：-a3→910c 命中 vs -a2→910b 不命中（DeepSeek-V3.2 仅910c）", r_a3, [
        (sv_a3 in ("mtp", "deepseek_mtp"), f"a3 → 910c 命中 → spec mtp 族（variant={sv_a3}）"),
        (sv_a2 == "suffix", f"a2 → 910b 不命中 → spec 地板 suffix（variant={sv_a2}）"),
        (sv_a3 != sv_a2, "a2/a3 卡型差异确实改变了 spec 决策"),
    ])
    r_name = run_case({"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8, "enable-sparse": True},
                      {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", "WINGS_DEVICE_NAME": "ascend910b3"},
                      {"architecture": "GlmMoeDsaForCausalLM"})
    check("P7", "WINGS_DEVICE_NAME=ascend910b3 → 卡型 910b 命中 GLM-5.1 sparse 白名单", r_name, [
        (r_name.features.get("sparse_kv") is True, "features.sparse_kv == true（卡型解析成功）"),
    ])

    # ── P8 · 打补丁逻辑（NV 装 indexcache / ascend day0 不装）────────────────
    emit("\n## P8 · 打补丁逻辑保持现状")
    r = run_case({"model-name": "glm-5.1", "engine": "vllm", "device-count": 8, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"}, {"architecture": "GlmMoeDsaForCausalLM"})
    check("P8", "NV vllm + IndexCache 架构 + enable-sparse → install.py 装 indexcache 补丁", r, [
        (install_indexcache(r), "命令含 install.py --features {...indexcache...}"),
    ])
    r = run_case({"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8, "enable-sparse": True},
                 {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", "WINGS_ASCEND_PLATFORM": "a2"},
                 {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    check("P8", "Ascend day0 GLM-5.1 + enable-sparse → 不装补丁，仅 --hf-overrides", r, [
        (not install_indexcache(r), "命令不含 install.py indexcache（ascend 门控）"),
        (has(r, '"index_topk_freq": 8'), "仍通过 --hf-overrides 启用 IndexCache"),
    ])

    emit("\n" + "=" * 92)
    emit(f" 总计：{_total['pass']} PASS / {_total['fail']} FAIL")
    emit(f" 输出：{OUT_PATH}")
    emit("=" * 92)
    return _total["fail"] == 0


def main():
    global _OUT
    _OUT = open(OUT_PATH, "w", encoding="utf-8")
    try:
        ok = run_all()
        return 0 if ok else 1
    finally:
        _OUT.close()


if __name__ == "__main__":
    sys.exit(main())
