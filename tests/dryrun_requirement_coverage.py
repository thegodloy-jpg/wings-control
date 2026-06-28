# -*- coding: utf-8 -*-
"""需求一 · 三特性使能改造 —— 八需求点 dry-run 覆盖验证（模拟用户真实下发）。

★ 关键口径：投机/稀疏/卸载三特性**仅经环境变量下发**（MaaS 页面开关 → 编排层注入
   ENABLE_SPECULATIVE_DECODE / ENABLE_SPARSE / LMCACHE_OFFLOAD env），**不走 wings_start.sh CLI**。
   因此本方案所有用例的三特性开关一律置于 `orchestration_env`（env），`user_cli` **不含**
   --enable-speculative-decode / --enable-sparse 等特性标志。驱动器复刻 wings_start.sh
   (299-300/345-348) 的 env→APP_ARGS 传播，等价真实链路。

真实下发口径（三段式）：
    user_cli          —— 用户真敲 CLI（model-name/engine/device-count… 不含特性开关）
    orchestration_env —— 编排层注入 env（特性开关 + 拓扑/平台/engine-version/LMCACHE/SPARSE_LEVEL…）
    model_config      —— 模型 config.json（architecture + quantization_config）

每个用例端到端生成 start_command.sh + 捕获日志 + 读取 advanced_features.json，
并**显式打印入参（三段）与出参（期望 vs 实际）**。运行：python tests/dryrun_requirement_coverage.py
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
_TOTAL = {"pass": 0, "fail": 0, "cases": 0, "case_fail": 0}

# 三特性的「环境变量下发」键（MaaS 注入；非 CLI）
SPEC = "ENABLE_SPECULATIVE_DECODE"
SPARSE = "ENABLE_SPARSE"
OFFLOAD = "LMCACHE_OFFLOAD"


def emit(s: str = ""):
    if _OUT is not None:
        _OUT.write(s + "\n")
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


# ── 出参提取助手（全部基于真实产物）──────────────────────────────────────────
def v_spec(r):    return r.variants.get("speculative_decode")
def v_sparse(r):  return r.variants.get("sparse_kv")
def v_offload(r): return r.variants.get("kv_offload")
def feat(r, k):   return r.features.get(k)


def cpu_size(r):
    vals = re.findall(r'export LMCACHE_MAX_LOCAL_CPU_SIZE=([0-9]+)', r.command)
    return int(vals[0]) if vals else None


def native_cpu_swap(r):
    m = re.search(r'cpu_swap_space_gb"?\s*:?\s*([0-9]+)', r.command)
    return int(m.group(1)) if m else None


def has(r, sub):  return sub in r.command
def logh(r, sub): return any(sub in l for l in r.logs)
def install_indexcache(r):
    return any("install.py" in l and "indexcache" in l for l in r.command.splitlines())
def norm(s):      return re.sub(r'/build/(model|sv)_[A-Za-z0-9_]+', '<T>', s)


# ── 用例框架：显式入参 + 期望/实际出参 ───────────────────────────────────────
class TC:
    def __init__(self, tid, point, title, user_cli, orch, model_config):
        self.tid, self.point, self.title = tid, point, title
        self.user_cli, self.orch, self.model_config = user_cli, orch, model_config
        self.r: CaseResult = run_case(user_cli, orch, model_config)
        self.rows = []

    def out(self, label, expected, actual, ok):
        self.rows.append((label, expected, actual, ok))
        return self

    def done(self):
        ok_all = all(ok for *_, ok in self.rows)
        _TOTAL["cases"] += 1
        if not ok_all:
            _TOTAL["case_fail"] += 1
        # 标注三特性下发通道（env，而非 CLI）
        feat_env = {k: self.orch.get(k) for k in (SPEC, SPARSE, OFFLOAD, "SPARSE_LEVEL", "PD_ROLE")
                    if k in self.orch}
        cli_has_feature = any(k in self.user_cli for k in
                              ("enable-speculative-decode", "enable-sparse", "enable-rag-acc"))
        emit("")
        emit(f"[{self.tid}] {self.point} · {self.title}  ==> {'PASS' if ok_all else 'FAIL'}")
        emit(f"  入参 user_cli          : {self.user_cli}")
        emit(f"  入参 orchestration_env : {self.orch}")
        emit(f"  入参 model_config      : {self.model_config}")
        emit(f"  下发通道 三特性=env下发({feat_env})  user_cli含特性CLI标志={cli_has_feature}")
        for label, expected, actual, ok in self.rows:
            _TOTAL["pass" if ok else "fail"] += 1
            emit(f"  出参 {label}")
            emit(f"        期望 = {expected}")
            emit(f"        实际 = {actual}    [{'PASS' if ok else 'FAIL'}]")
        return self.r


def run_all():
    emit("=" * 96)
    emit(" 需求一 · 三特性使能 —— 八需求点 dry-run 覆盖验证（三特性仅经 env 下发 · 入参/出参逐项）")
    emit("=" * 96)

    # ════════ P0 · 下发通道：三特性仅经 env 下发，不走 CLI ════════
    emit("\n############ P0 · 下发通道验证：三特性仅经环境变量下发（非 wings_start.sh CLI）############")
    t = TC("TC-P0-01", "P0", "ENABLE_SPARSE env（user_cli 无 --enable-sparse）→ 稀疏生效",
           {"model-name": "glm-5.1", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPARSE: "true"},
           {"architecture": "GlmMoeDsaForCausalLM"})
    r = t.r
    t.out("user_cli 含 --enable-sparse 标志", "否（特性不走 CLI）", "enable-sparse" in t.user_cli, "enable-sparse" not in t.user_cli)
    t.out("features.sparse_kv（仅由 ENABLE_SPARSE env 驱动）", "True", feat(r, "sparse_kv"), feat(r, "sparse_kv") is True)
    t.out("variants.sparse_kv", "indexcache_topk4", v_sparse(r), v_sparse(r) == "indexcache_topk4")
    t.done()
    t = TC("TC-P0-02", "P0", "ENABLE_SPECULATIVE_DECODE env（user_cli 无 CLI 标志）→ 投机生效",
           {"model-name": "GLM-5.2-w8a8", "engine": "vllm_ascend", "device-count": 16},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3", SPEC: "true"},
           {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("user_cli 含 --enable-speculative-decode", "否", "enable-speculative-decode" in t.user_cli, "enable-speculative-decode" not in t.user_cli)
    t.out("variants.speculative_decode（仅由 env 驱动）", "deepseek_mtp", v_spec(r), v_spec(r) == "deepseek_mtp")
    t.done()
    t = TC("TC-P0-03", "P0", "LMCACHE_OFFLOAD env → 卸载生效（卸载本就纯 env，无 CLI）",
           {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", OFFLOAD: "true", "LMCACHE_POD_MEMORY": "512"},
           {"architecture": "Glm4MoeForCausalLM"})
    r = t.r
    t.out("features.kv_offload（仅由 LMCACHE_OFFLOAD env 驱动）", "True", feat(r, "kv_offload"), feat(r, "kv_offload") is True)
    t.out("variants.kv_offload", "lmcache_cpu+auto", v_offload(r), v_offload(r) == "lmcache_cpu+auto")
    t.done()

    # ════════ P1 · 删 fp4/fp8 + 引擎路由删除 ════════
    emit("\n############ P1 · 删 fp4/fp8 + 引擎路由删除 ############")
    uc = {"model-name": "Qwen3.5-397B-A17B", "engine": "vllm_ascend", "device-count": 16}
    mc = {"architecture": "Qwen3_5MoeForConditionalGeneration", "quantization_config": {"quant_method": "ascend"}}
    base = {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_ASCEND_PLATFORM": "a3"}
    r_off = run_case(uc, base, mc)
    t = TC("TC-P1-01", "P1", "ENABLE_OPERATOR_ACCELERATION/ENABLE_SOFT_FP8 旁路已删→对产物无效",
           uc, {**base, "ENABLE_OPERATOR_ACCELERATION": "true", "ENABLE_SOFT_FP8": "true"}, mc)
    r_on = t.r
    t.out("engine（设两个开关 env 后）", "vllm_ascend（与不设时一致）", r_on.engine, r_on.engine == r_off.engine == "vllm_ascend")
    t.out("start_command 归一化", "与不设这两个 env 时字节一致", "一致" if norm(r_on.command) == norm(r_off.command) else "不一致",
          norm(r_on.command) == norm(r_off.command))
    t.out("命令导出 USE_KUNLUN_ATB（死代码已删）", "不出现", "出现" if has(r_on, "USE_KUNLUN_ATB") else "不出现", not has(r_on, "USE_KUNLUN_ATB"))
    t.done()

    # ════════ P2 · 对外接口 advanced_features.json ════════
    emit("\n############ P2 · 对外特性状态接口 advanced_features.json（features + variants）############")
    t = TC("TC-P2-01", "P2", "GLM-5.2·Ascend spec → features+variants 如实透出",
           {"model-name": "GLM-5.2-w8a8", "engine": "vllm_ascend", "device-count": 16},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3", SPEC: "true"},
           {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("features 键集合", "{speculative_decode,sparse_kv,kv_offload,rag_acc}",
          sorted(r.features), {"speculative_decode", "sparse_kv", "kv_offload", "rag_acc"} <= set(r.features))
    t.out("features.speculative_decode", "True", feat(r, "speculative_decode"), feat(r, "speculative_decode") is True)
    t.out("variants.speculative_decode", "deepseek_mtp", v_spec(r), v_spec(r) == "deepseek_mtp")
    t.out("features.sparse_kv / kv_offload", "False / False（仅投机）",
          f"{feat(r,'sparse_kv')} / {feat(r,'kv_offload')}", feat(r, "sparse_kv") is False and feat(r, "kv_offload") is False)
    t.done()
    t = TC("TC-P2-02", "P2", "GLM-5.1·Ascend sparse → variants 透出 indexcache_topk8",
           {"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", "WINGS_ASCEND_PLATFORM": "a2", SPARSE: "true"},
           {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("features.sparse_kv", "True", feat(r, "sparse_kv"), feat(r, "sparse_kv") is True)
    t.out("variants.sparse_kv", "indexcache_topk8", v_sparse(r), v_sparse(r) == "indexcache_topk8")
    t.done()

    # ════════ P3 · 对内特性日志 ════════
    emit("\n############ P3 · 对内特性日志（本轮新增）############")
    t = TC("TC-P3-01", "P3", "Ascend 卡型解析失败 → WARNING（白名单将全 miss）",
           {"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", SPARSE: "true"}, {"architecture": "GlmMoeDsaForCausalLM"})
    r = t.r
    t.out("日志 card_token unresolved on Ascend", "出现（WARNING）", "出现" if logh(r, "card_token unresolved on Ascend") else "缺失", logh(r, "card_token unresolved on Ascend"))
    t.out("日志 [SmartFeature] effective enablement 摘要", "出现", "出现" if logh(r, "effective enablement") else "缺失", logh(r, "effective enablement"))
    t.done()
    t = TC("TC-P3-02", "P3", "非白名单 + ENABLE_SPARSE → sparse 抑制对称日志 + req→eff 摘要",
           {"model-name": "Llama-3-70B", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPARSE: "true", SPEC: "true"}, {"architecture": "LlamaForCausalLM"})
    r = t.r
    t.out("日志 sparse requested but not in whitelist", "出现", "出现" if logh(r, "sparse requested but not in whitelist") else "缺失", logh(r, "sparse requested but not in whitelist"))
    t.out("摘要含 sparse True->False（请求→有效）", "出现", "出现" if logh(r, "sparse True->False") else "缺失", logh(r, "sparse True->False"))
    t.done()
    t = TC("TC-P3-03", "P3", "非白名单 + LMCACHE_OFFLOAD → offload 抑制对称日志",
           {"model-name": "Llama-3-70B", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", OFFLOAD: "true"}, {"architecture": "LlamaForCausalLM"})
    r = t.r
    t.out("日志 offload requested but not in whitelist", "出现", "出现" if logh(r, "offload requested but not in whitelist") else "缺失", logh(r, "offload requested but not in whitelist"))
    t.done()

    # ════════ P4 · 三特性依赖白名单 + 开关基线 ════════
    emit("\n############ P4 · 三特性依赖白名单门控（含 PD 一票否决 + 开关基线）############")
    t = TC("TC-P4-01", "P4", "命中白名单 spec → deepseek_mtp（GLM-5.2·Ascend）",
           {"model-name": "GLM-5.2-w8a8", "engine": "vllm_ascend", "device-count": 16},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3", SPEC: "true"},
           {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("variants.speculative_decode", "deepseek_mtp", v_spec(r), v_spec(r) == "deepseek_mtp")
    t.done()
    t = TC("TC-P4-02", "P4", "白名单 sparse-only → spec 地板 suffix（GLM-5.1·Ascend, B.4 bug 修复）",
           {"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", "WINGS_ASCEND_PLATFORM": "a2", SPEC: "true", SPARSE: "true"},
           {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("variants.speculative_decode", "suffix（清单 sparse-only → 地板）", v_spec(r), v_spec(r) == "suffix")
    t.out("variants.sparse_kv", "indexcache_topk8（sparse 仍产）", v_sparse(r), v_sparse(r) == "indexcache_topk8")
    t.done()
    t = TC("TC-P4-03", "P4", "offload 不在白名单 → 收口关（Qwen3.5-397B·NV：spec,sparse 无 offload）",
           {"model-name": "qwen3.5-397b", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPARSE: "true", OFFLOAD: "true", "LMCACHE_POD_MEMORY": "512"},
           {"architecture": "Qwen3_5MoeForConditionalGeneration"})
    r = t.r
    t.out("features.kv_offload", "False", feat(r, "kv_offload"), feat(r, "kv_offload") is False)
    t.out("命令 export LMCACHE_OFFLOAD=true", "不出现", "出现" if has(r, "export LMCACHE_OFFLOAD=true") else "不出现", not has(r, "export LMCACHE_OFFLOAD=true"))
    t.done()
    t = TC("TC-P4-04", "P4", "PD 一票否决 → 三特性全关",
           {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPEC: "true", SPARSE: "true", OFFLOAD: "true",
            "LMCACHE_POD_MEMORY": "512", "PD_ROLE": "P"},
           {"architecture": "Glm4MoeForCausalLM"})
    r = t.r
    t.out("日志 PD role detected -> veto", "出现", "出现" if logh(r, "PD role detected") else "缺失", logh(r, "PD role detected"))
    t.out("features（spec/sparse/offload）", "全 False",
          f"{feat(r,'speculative_decode')}/{feat(r,'sparse_kv')}/{feat(r,'kv_offload')}",
          not feat(r, "speculative_decode") and not feat(r, "sparse_kv") and not feat(r, "kv_offload"))
    t.done()

    emit("\n---- P4 开关基线（env 不设 → 不产；与白名单收口正交）----")
    t = TC("TC-P4-05", "P4", "spec 开关 OFF（不设 ENABLE_SPECULATIVE_DECODE）→ 不产投机",
           {"model-name": "GLM-5.2-w8a8", "engine": "vllm_ascend", "device-count": 16},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3"},
           {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("命令 --speculative-config", "不出现", "出现" if has(r, "--speculative-config") else "不出现", not has(r, "--speculative-config"))
    t.out("features.speculative_decode", "False", feat(r, "speculative_decode"), feat(r, "speculative_decode") is False)
    t.done()
    t = TC("TC-P4-06", "P4", "offload 开关 OFF（不设 LMCACHE_OFFLOAD）→ 不产卸载",
           {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"}, {"architecture": "Glm4MoeForCausalLM"})
    r = t.r
    t.out("命令 export LMCACHE_OFFLOAD=true", "不出现", "出现" if has(r, "export LMCACHE_OFFLOAD=true") else "不出现", not has(r, "export LMCACHE_OFFLOAD=true"))
    t.out("features.kv_offload", "False", feat(r, "kv_offload"), feat(r, "kv_offload") is False)
    t.done()

    # ════════ P5 · 内存自动计算 C4 ════════
    emit("\n############ P5 · KV 卸载 auto 容量反向预算（C4）"
         "  M_offload=POD-(7*TP*DP+3)-10% ############")
    t = TC("TC-P5-01", "P5", "auto LMCache：写回 per-card 容量 + 强制 swap_space=0（POD=512, 8卡, TP8/DP1）",
           {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPARSE: "true", OFFLOAD: "true", "LMCACHE_POD_MEMORY": "512"},
           {"architecture": "Glm4MoeForCausalLM"})
    r = t.r
    t.out("LMCACHE_MAX_LOCAL_CPU_SIZE", "50 = M_offload(512-59-51=401) ÷ 8卡", cpu_size(r), cpu_size(r) == 50)
    t.out("命令 --swap-space 0", "出现（auto 原子绑定）", "出现" if has(r, "--swap-space 0") else "不出现", has(r, "--swap-space 0"))
    t.done()
    t = TC("TC-P5-02", "P5", "auto native：整节点 M_offload 不除卡数（V4-Flash·Ascend cpu_swap_space_gb）",
           {"model-name": "DeepSeek-V4-Flash", "engine": "vllm_ascend", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "WINGS_ASCEND_PLATFORM": "a2", OFFLOAD: "true", "LMCACHE_POD_MEMORY": "512"},
           {"architecture": "DeepseekV4ForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("cpu_swap_space_gb", "401 = 整节点 M_offload（不除卡数）", native_cpu_swap(r), native_cpu_swap(r) == 401)
    t.done()
    t = TC("TC-P5-03", "P5", "custom：给 LMCACHE_MAX_LOCAL_CPU_SIZE 数值 → 透传不计算",
           {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPARSE: "true", OFFLOAD: "true", "LMCACHE_MAX_LOCAL_CPU_SIZE": "200"},
           {"architecture": "Glm4MoeForCausalLM"})
    r = t.r
    t.out("LMCACHE_MAX_LOCAL_CPU_SIZE", "200（原样透传，不计算）", cpu_size(r), cpu_size(r) == 200)
    t.done()
    t = TC("TC-P5-04", "P5", "熔断：可用容量 < 下限(100G) → 不建 CPU 卸载池（POD=100→31）",
           {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPARSE: "true", OFFLOAD: "true", "LMCACHE_POD_MEMORY": "100"},
           {"architecture": "Glm4MoeForCausalLM"})
    r = t.r
    t.out("LMCACHE_MAX_LOCAL_CPU_SIZE（auto 写回）", "无（熔断）", cpu_size(r), cpu_size(r) is None)
    t.out("日志 熔断告警（below floor / skip CPU offload）", "出现",
          "出现" if (logh(r, "below floor") or logh(r, "skip CPU offload")) else "缺失",
          logh(r, "below floor") or logh(r, "skip CPU offload"))
    t.done()

    # ════════ P6 · 稀疏：开关 × 白名单 × 档位 三层门控 ════════
    emit("\n############ P6 · 稀疏使能三层门控：开关(env) → 白名单(whitelist) → 档位(SPARSE_LEVEL) ############")
    NV = {"model-name": "glm-5.1", "engine": "vllm", "device-count": 8}   # GlmMoeDsa·NV → 命中 sparse 白名单
    MC = {"architecture": "GlmMoeDsaForCausalLM"}
    OB = {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"}                            # 不设 ENABLE_SPARSE = 开关 OFF
    ON = {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPARSE: "true"}           # 设 ENABLE_SPARSE = 开关 ON
    # 层1 开关 OFF
    t = TC("TC-P6-01", "P6", "层1 开关 OFF（不设 ENABLE_SPARSE）→ 不产稀疏、不评估档位", NV, OB, MC)
    r = t.r
    t.out("命令 --hf-overrides（稀疏）", "不出现", "出现" if has(r, "--hf-overrides") else "不出现", not has(r, "--hf-overrides"))
    t.out("features.sparse_kv / variants.sparse_kv", "False / None", f"{feat(r,'sparse_kv')} / {v_sparse(r)}", feat(r, "sparse_kv") is False and v_sparse(r) is None)
    t.out("日志 effective SPARSE_LEVEL", "不出现（产出口未运行）", "出现" if logh(r, "effective SPARSE_LEVEL") else "不出现", not logh(r, "effective SPARSE_LEVEL"))
    t.done()
    # 层1 优先于层3：开关 OFF 时 SPARSE_LEVEL 不生效
    t = TC("TC-P6-02", "P6", "层1 优先于层3：开关 OFF 时 SPARSE_LEVEL=performance_first 不触发告警",
           NV, {**OB, "SPARSE_LEVEL": "performance_first"}, MC)
    r = t.r
    t.out("features.sparse_kv", "False", feat(r, "sparse_kv"), feat(r, "sparse_kv") is False)
    t.out("日志 performance_first 告警", "不出现（开关 OFF 门控档位）", "出现" if logh(r, "performance_first not impl") else "不出现", not logh(r, "performance_first not impl"))
    t.done()
    # 层2 白名单：开关 ON 但不在 sparse 白名单 → 抑制
    t = TC("TC-P6-03", "P6", "层2 白名单：开关 ON 但模型不在 sparse 白名单 → 收口抑制（glm-4.7·Ascend）",
           {"model-name": "glm-4.7", "engine": "vllm_ascend", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_ASCEND_PLATFORM": "a3", SPARSE: "true"},
           {"architecture": "Glm4MoeForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("features.sparse_kv", "False（glm-4.7·Ascend 白名单为 spec,offload）", feat(r, "sparse_kv"), feat(r, "sparse_kv") is False)
    t.out("日志 sparse requested but not in whitelist", "出现", "出现" if logh(r, "sparse requested but not in whitelist") else "缺失", logh(r, "sparse requested but not in whitelist"))
    t.done()
    # 层3 档位：开关 ON + 白名单命中 + 缺省 → accuracy_first
    t = TC("TC-P6-04", "P6", "层3 档位：开关 ON + 白名单命中 + 缺省 SPARSE_LEVEL → accuracy_first", NV, ON, MC)
    r_acc = t.r
    t.out("variants.sparse_kv", "indexcache_topk4（NV GlmMoeDsa）", v_sparse(r_acc), v_sparse(r_acc) == "indexcache_topk4")
    t.out("日志 effective SPARSE_LEVEL", "accuracy_first", "accuracy_first" if logh(r_acc, "effective SPARSE_LEVEL=accuracy_first") else "其它", logh(r_acc, "effective SPARSE_LEVEL=accuracy_first"))
    t.out("日志 performance_first 告警", "不出现", "出现" if logh(r_acc, "performance_first not impl") else "不出现", not logh(r_acc, "performance_first not impl"))
    t.done()
    # 层3 performance_first：档位生效；若当前 sparse 行未声明 performance topk，则回退该行 accuracy topk
    t = TC("TC-P6-05", "P6", "层3 档位：SPARSE_LEVEL=performance_first → 档位生效，GLM-5.1·NV topk 回退本行 accuracy",
           NV, {**ON, "SPARSE_LEVEL": "performance_first"}, MC)
    r = t.r
    t.out("日志 performance_first not implemented", "不出现（已实现）", "出现" if logh(r, "performance_first not impl") else "不出现", not logh(r, "performance_first not impl"))
    t.out("有效档位日志", "performance_first", "performance_first" if logh(r, "effective SPARSE_LEVEL=performance_first") else "其它", logh(r, "effective SPARSE_LEVEL=performance_first"))
    t.out("start_command 与缺省(TC-P6-04)归一化", "一致（本行无 performance topk，回退 accuracy=4）", "一致" if norm(r.command) == norm(r_acc.command) else "不一致", norm(r.command) == norm(r_acc.command))
    t.done()
    # 层3 非法值 → 回落，无告警
    t = TC("TC-P6-06", "P6", "层3 档位：SPARSE_LEVEL=turbo（非法）→ 回落 accuracy_first，不触发 performance 告警",
           NV, {**ON, "SPARSE_LEVEL": "turbo"}, MC)
    r = t.r
    t.out("有效档位日志", "accuracy_first", "accuracy_first" if logh(r, "effective SPARSE_LEVEL=accuracy_first") else "其它", logh(r, "effective SPARSE_LEVEL=accuracy_first"))
    t.out("日志 performance_first 告警", "不出现（非法值非 performance_first）", "出现" if logh(r, "performance_first not impl") else "不出现", not logh(r, "performance_first not impl"))
    t.done()


    # ════════ P5 · sparse 表 per-row topk + performance_first 产出路径 ════════
    emit("\n############ P5 · sparse 表档位 topk：performance_first 产出路径 ############")
    V4 = {"model-name": "DeepSeek-V4-Flash", "engine": "vllm", "device-count": 8}
    V4_MC = {"architecture": "DeepseekV4ForCausalLM"}
    t = TC("TC-P5-01", "P5", "V4-Flash·NV sparse accuracy_first → topk4", V4, ON, V4_MC)
    r_v4_acc = t.r
    t.out("variants.sparse_kv", "indexcache_use_index_cache_topk4", v_sparse(r_v4_acc), v_sparse(r_v4_acc) == "indexcache_use_index_cache_topk4")
    t.out("命令 index_topk_freq:4", "出现", "出现" if has(r_v4_acc, '"index_topk_freq": 4') else "缺失", has(r_v4_acc, '"index_topk_freq": 4'))
    t.done()
    t = TC("TC-P5-02", "P5", "V4-Flash·NV sparse performance_first → topk8", V4, {**ON, "SPARSE_LEVEL": "performance_first"}, V4_MC)
    r_v4_perf = t.r
    t.out("variants.sparse_kv", "indexcache_use_index_cache_topk8", v_sparse(r_v4_perf), v_sparse(r_v4_perf) == "indexcache_use_index_cache_topk8")
    t.out("有效档位日志", "performance_first", "performance_first" if logh(r_v4_perf, "effective SPARSE_LEVEL=performance_first") else "其它", logh(r_v4_perf, "effective SPARSE_LEVEL=performance_first"))
    t.out("命令 index_topk_freq:8", "出现", "出现" if has(r_v4_perf, '"index_topk_freq": 8') else "缺失", has(r_v4_perf, '"index_topk_freq": 8'))
    t.out("与 accuracy 命令归一化", "不一致（topk 4→8）", "不一致" if norm(r_v4_perf.command) != norm(r_v4_acc.command) else "一致", norm(r_v4_perf.command) != norm(r_v4_acc.command))
    t.done()

    # ════════ P7 · 硬件信息 / ENGINE-VERSION 卡型解析 ════════
    emit("\n############ P7 · 卡型解析（保留老逻辑 + 依赖 ENGINE-VERSION）############")
    v32_uc = {"model-name": "DeepSeek-V3.2", "engine": "vllm_ascend", "device-count": 16}
    v32_mc = {"architecture": "DeepseekV32ForCausalLM", "quantization_config": {"quant_method": "ascend"}}
    r_a3 = run_case(v32_uc, {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3", SPEC: "true"}, v32_mc)
    t = TC("TC-P7-01", "P7", "ENGINE_VERSION 后缀定卡型：-a3→910c 命中 vs -a2→910b 不命中（DeepSeek-V3.2 仅910c）",
           v32_uc, {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a2", SPEC: "true"}, v32_mc)
    r_a2 = t.r
    t.out("ENGINE_VERSION=…-a3 → variants.speculative_decode", "mtp 族（910c 命中）", v_spec(r_a3), v_spec(r_a3) in ("mtp", "deepseek_mtp"))
    t.out("ENGINE_VERSION=…-a2 → variants.speculative_decode", "suffix（910b 不命中→地板）", v_spec(r_a2), v_spec(r_a2) == "suffix")
    t.out("a2 vs a3 决策差异", "不同（卡型确实改变决策）", f"a3={v_spec(r_a3)} / a2={v_spec(r_a2)}", v_spec(r_a3) != v_spec(r_a2))
    t.done()
    t = TC("TC-P7-02", "P7", "WINGS_DEVICE_NAME=ascend910b3 → 卡型 910b 命中 GLM-5.1 sparse 白名单",
           {"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", "WINGS_DEVICE_NAME": "ascend910b3", SPARSE: "true"},
           {"architecture": "GlmMoeDsaForCausalLM"})
    r = t.r
    t.out("features.sparse_kv", "True（卡型解析成功 → 白名单命中）", feat(r, "sparse_kv"), feat(r, "sparse_kv") is True)
    t.done()

    # ════════ P8 · 打补丁逻辑 ════════
    emit("\n############ P8 · 打补丁逻辑保持现状 ############")
    t = TC("TC-P8-01", "P8", "NV vllm + IndexCache 架构 + ENABLE_SPARSE → install.py 装 indexcache 补丁",
           {"model-name": "glm-5.1", "engine": "vllm", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", SPARSE: "true"}, {"architecture": "GlmMoeDsaForCausalLM"})
    r = t.r
    t.out("命令 install.py --features {...indexcache...}", "出现", "出现" if install_indexcache(r) else "不出现", install_indexcache(r))
    t.done()
    t = TC("TC-P8-02", "P8", "Ascend day0 GLM-5.1 + ENABLE_SPARSE → 不装补丁，仅 --hf-overrides",
           {"model-name": "glm-5.1", "engine": "vllm_ascend", "device-count": 8},
           {"DISTRIBUTED_EXECUTOR_BACKEND": "ray", "WINGS_ASCEND_PLATFORM": "a2", SPARSE: "true"},
           {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}})
    r = t.r
    t.out("命令 install.py indexcache", "不出现（ascend 门控）", "出现" if install_indexcache(r) else "不出现", not install_indexcache(r))
    t.out("命令 --hf-overrides index_topk_freq:8", "出现（引擎内置启用 IndexCache）", "出现" if has(r, '"index_topk_freq": 8') else "不出现", has(r, '"index_topk_freq": 8'))
    t.done()

    emit("\n" + "=" * 96)
    emit(f" 用例：{_TOTAL['cases']} 个（{_TOTAL['cases'] - _TOTAL['case_fail']} PASS / {_TOTAL['case_fail']} FAIL）"
         f"  ·  断言：{_TOTAL['pass']} PASS / {_TOTAL['fail']} FAIL")
    emit(f" 输出：{OUT_PATH}")
    emit("=" * 96)
    return _TOTAL["fail"] == 0


def main():
    global _OUT
    _OUT = open(OUT_PATH, "w", encoding="utf-8")
    try:
        return 0 if run_all() else 1
    finally:
        _OUT.close()


if __name__ == "__main__":
    sys.exit(main())
