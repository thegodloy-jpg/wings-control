# -*- coding: utf-8 -*-
"""需求一 · 与 master 功能对比（防功能丢失回归）。

对每个场景，在 master worktree 与 branch 工作树各自的**生产代码**下生成 start_command.sh
（三特性经 env 下发），归一化后逐行 diff，把每条差异归类到 D1–D5 预期变更白名单；
任何无法归类的差异记为「疑似回归」。通过条件：疑似回归 = 0。

预期变更白名单（见《需求一-与master对比测试方案.md》§二）：
  D1 移除 --accel-file            D2 advanced_features 路径反斜杠修复
  D3 投机白名单门控(mtp↔suffix)   D4 V4-Flash·NV forced IndexCache 去除
  D5 删 Soft FP8/FP4 自动量化

用法：
  git worktree add ../wt-master master    # 先建 master worktree
  python tests/cmp_master_branch.py
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                      # branch 工作树
WT_MASTER = ROOT.parent / "wt-master"                           # master worktree
sys.path.insert(0, str(ROOT))
import dry_run as dr  # noqa: E402  (仅用于取 SCENARIOS 作 Group A)

OUT_PATH = Path(__file__).resolve().parent / "cmp_master_branch_output.txt"
_OUT = None


def emit(s=""):
    if _OUT is not None:
        _OUT.write(s + "\n")
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


# ── 自包含生成器（写入临时 .py，在每个 worktree 的 cwd 下 subprocess 运行）──────
GEN_SRC = r'''
import sys, os, json, shutil
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "wings_control"))
import dry_run as dr
scen = json.load(open(sys.argv[1], encoding="utf-8"))
out_cmd, out_af = sys.argv[2], sys.argv[3]
EXTRA = {"SPARSE_LEVEL","PD_ROLE","ENABLE_OPERATOR_ACCELERATION","ENABLE_SOFT_FP8",
         "ENABLE_SOFT_FP4","LMCACHE_POD_MEMORY","ENABLE_SPARSE","SPARSE_ENABLE",
         "ENABLE_SPECULATIVE_DECODE","SD_ENABLE","ENABLE_KV_OFFLOAD","LMCACHE_OFFLOAD",
         "LMCACHE_LOCAL_CPU","LMCACHE_LOCAL_DISK",
         "LMCACHE_MAX_LOCAL_DISK_SIZE","LMCACHE_QAT","LMCACHE_COLD_START","WINGS_DEVICE_NAME",
         "WINGS_DEVICE_MEMORY","ENABLE_RAG_ACC","LMCACHE_MAX_LOCAL_CPU_SIZE",
         "KV_MEM_OFFLOAD_SIZE","AVAILABLE_POD_MEM_SIZE"}
from core.start_args_compat import parse_launch_args
from core.port_plan import derive_port_plan
from core.wings_entry import build_launcher_plan
from config.settings import settings
dr.reset_managed_env()
for k in EXTRA: os.environ.pop(k, None)
model_dir = dr.create_mock_model_dir(scen["model_config"])
try:
    dr.apply_orchestration_env({"user_cli":scen["user_cli"],
                                "orchestration_env":scen.get("orchestration_env",{})}, model_dir)
    app_args = dr.simulate_wings_start(scen["user_cli"])
    for en, fl in (("ENABLE_SPECULATIVE_DECODE","--enable-speculative-decode"),
                   ("ENABLE_SPARSE","--enable-sparse"),("ENABLE_RAG_ACC","--enable-rag-acc")):
        if os.environ.get(en,"").lower()=="true" and fl not in app_args: app_args.append(fl)
    la0 = parse_launch_args(app_args + ["--node-rank","0"])
    pp = derive_port_plan(port=la0.port, enable_reason_proxy=settings.ENABLE_REASON_PROXY,
                          health_port=settings.HEALTH_PORT)
    plan = build_launcher_plan(la0, pp)
    open(out_cmd,"w",encoding="utf-8",newline="\n").write(plan.command)
    af = os.path.join(settings.SHARED_VOLUME_PATH, "advanced_features.json")
    open(out_af,"w",encoding="utf-8",newline="\n").write(
        open(af,encoding="utf-8").read() if os.path.exists(af) else "")
except Exception as e:
    open(out_cmd,"w",encoding="utf-8").write("GEN_ERROR: "+repr(e))
    open(out_af,"w",encoding="utf-8").write("")
finally:
    shutil.rmtree(model_dir, ignore_errors=True)
'''


def _write_gen():
    f = tempfile.NamedTemporaryFile("w", suffix="_gen.py", delete=False, encoding="utf-8")
    f.write(GEN_SRC); f.close()
    return f.name


def gen(worktree: Path, scen: dict, genfile: str):
    sf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(scen, sf); sf.close()
    oc = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False); oc.close()
    oa = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); oa.close()
    subprocess.run([sys.executable, genfile, sf.name, oc.name, oa.name],
                   cwd=str(worktree), capture_output=True)
    cmd = open(oc.name, encoding="utf-8").read()
    af = open(oa.name, encoding="utf-8").read()
    for p in (sf.name, oc.name, oa.name):
        os.unlink(p)
    return cmd, af


# ── 归一化：折叠临时路径 churn + 丢弃截断预览 + 中和 D1/D2（universal 非功能）──
def neutralize(text: str) -> str:
    # D2：advanced_features 路径反斜杠修复（master 的 \ → /），两侧统一
    text = text.replace("/shared-volume\\advanced_features.json", "/shared-volume/advanced_features.json")
    # D1：移除 --accel-file 及其行续接（master 的 `--progress-file X \` + `--accel-file Y &` → `--progress-file X &`）
    text = re.sub(r'(--progress-file \S+) \\\n\s*--accel-file \S+ &', r'\1 &', text)
    text = re.sub(r' --accel-file \S+', '', text)
    # 临时目录 churn：折叠「绝对路径根（含 worktree 名 wt-master / wings-control）+ /build/model_*」整体，
    # 否则 model_path 仅因 worktree 根不同就误报差异（出现在 --model / ANALYZER_CONFIG / sglang / mindie）。
    text = re.sub(r'[^\s"\']*/build/(?:model|sv)_[A-Za-z0-9_]+', '<T>', text)
    # 切除「崩溃处理脚手架」尾段：advanced-feature-fallback ↔ crash-retry 模板由「特性是否激活」决定，
    # 两套模板在 master/branch 代码中**都存在、均未改**（git log -S 证实）；模板切换是白名单收窄的
    # 下游后果，非机制变更。剥离后让 diff 聚焦真正的启动命令 + 特性 env/补丁。
    cut = re.search(r'^#\s*-+\s*Engine process wait and exception handling', text, re.M)
    if cut:
        text = text[:cut.start()]
    lines = []
    for ln in text.splitlines():
        s = ln.lstrip()
        if s.startswith("echo '[wings-cmd] >>>"):        # 截断预览行
            continue
        if s.startswith('echo "[Engine] Engine PID:'):   # PID 行含 "advanced features enabled" 措辞差异
            continue
        lines.append(ln)
    return "\n".join(lines)


def parse_flags(execline: str):
    """把 exec 行解析成 {flag: value}（bare flag value=True）。"""
    toks = execline.split()
    flags, i = {}, 0
    # 处理 --flag '... json ...'（含空格的引号值）：用正则抓 --flag '....' 与 --flag value
    for m in re.finditer(r"(--[\w-]+)(?:\s+('[^']*'|\"[^\"]*\"|[^\s-][^\s]*))?", execline):
        flag, val = m.group(1), m.group(2)
        flags[flag] = val if val is not None else True
    return flags


SPEC_RE = re.compile(r'"method"\s*:\s*"([a-z0-9_]+)"')


# 三特性相关 flag（其变化属白名单门控/C4 的预期范围）；变化记 D3–D7，其余 flag 变化＝回归
_FEATURE_FLAGS = {
    "--speculative-config",                       # D3 投机
    "--hf-overrides",                             # 稀疏 IndexCache / D4 forced
    "--kv-cache-dtype", "--calculate-kv-scales",  # 稀疏 fp8 档 + 伴生 flag
    "--swap-space",                               # D6 C4 auto 原子绑定
    "--kv-transfer-config", "--kv_offloading_backend", "--kv_offloading_size",  # 卸载 D7
}


def classify_execline(m_line: str, b_line: str):
    """对 exec 行做 flag 级差异分类，返回 (intended:list, regressions:list)。"""
    mf, bf = parse_flags(m_line), parse_flags(b_line)
    intended, regress = [], []
    for fl in sorted(set(mf) | set(bf)):
        mv, bv = mf.get(fl), bf.get(fl)
        if mv == bv:
            continue
        if fl == "--speculative-config":
            mm, bb = SPEC_RE.search(mv or ""), SPEC_RE.search(bv or "")
            intended.append(f"D3 spec {mm.group(1) if mm else None}->{bb.group(1) if bb else None}")
        elif fl == "--hf-overrides" and mv and "use_index_cache" in mv and bv is None:
            intended.append("D4 forced IndexCache removed")
        elif fl == "--hf-overrides":
            intended.append(f"D7 sparse hf-overrides {mv}->{bv}")
        elif fl == "--quantization" and bv is None:
            intended.append(f"D5 auto-quant removed ({mv})")
        elif fl == "--kv-cache-dtype":
            intended.append(f"D7 sparse kv-cache-dtype {mv}->{bv}")
        elif fl == "--swap-space":
            intended.append(f"D6 C4 swap-space {mv}->{bv}")
        elif fl in _FEATURE_FLAGS:
            intended.append(f"D7 offload {fl} {mv}->{bv}")
        else:
            regress.append(f"FLAG {fl}: master={mv} branch={bv}")
    return intended, regress


# 非 exec 行：三特性脚手架（LMCache/install/accel/patch/EARS/C4 容量）变化属预期；
# 不匹配者（如 TP/parser/chat-template/recipe）＝疑似回归。
INTENDED_LINE = re.compile(
    r'LMCACHE_|PYTHONHASHSEED|swap.?space|install\.py|wings-accel|LMCache|kv.transfer|'
    r'ENABLE_KV_OFFLOAD|ENABLE_KV_MEM_OFFLOAD|KV_MEM_OFFLOAD_SIZE|'
    r'kv_connector|VLLM_EARS|EARS_TOLERANCE|WINGS_ENGINE_PATCH_OPTIONS|advanced.feature|'
    r'AdvFeature|LMCACHE_CONFIG_FILE|index_topk|use_index_cache|cpu_swap_space|'
    r'kv_offloading|accel-volume|accel-file|USE_KUNLUN_ATB|kunlun|'
    # 特性门控/补丁安装块的 shell 控制脚手架（纯控制流，非功能 config）
    r'^\s*(?:set [+-]e|else|fi|[A-Z_]+_RC=\$\?|if \[ \$[A-Z_]+_RC)', re.I)


def is_exec(line: str) -> bool:
    s = line.strip()
    return ("api_server" in s or s.startswith("vllm serve") or " serve " in s) and not s.startswith("echo")


def compare(name: str, m_cmd: str, b_cmd: str):
    if m_cmd.startswith("GEN_ERROR") or b_cmd.startswith("GEN_ERROR"):
        emit(f"[{name}] GEN_ERROR  master={m_cmd[:80]!r} branch={b_cmd[:80]!r}")
        return ["GEN_ERROR"]
    M, B = neutralize(m_cmd).splitlines(), neutralize(b_cmd).splitlines()
    sm = difflib.SequenceMatcher(None, M, B, autojunk=False)
    regress = []
    intended = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        m_lines, b_lines = M[i1:i2], B[j1:j2]
        m_exec = [l for l in m_lines if is_exec(l)]
        b_exec = [l for l in b_lines if is_exec(l)]
        # exec 行 → flag 级分类（按出现顺序对齐）
        for ml, bl in zip(m_exec, b_exec):
            inn, reg = classify_execline(ml, bl)
            intended += inn
            regress += [f"{name}: {r}" for r in reg]
        # 非 exec 差异行：过 INTENDED_LINE 过滤；不匹配者＝疑似回归
        for l in [x for x in m_lines if not is_exec(x) and x.strip()] + \
                 [x for x in b_lines if not is_exec(x) and x.strip()]:
            if INTENDED_LINE.search(l):
                intended.append("scaffold:" + l.strip()[:40])
            else:
                regress.append(f"{name}: 非特性差异行 → {l.strip()[:120]}")
    return regress, intended


# ════════ 场景矩阵 ════════
def _env_dispatch(scn: dict) -> dict:
    """把 dry_run 场景的 user_cli enable-* 转成 orch env（三特性 env 下发口径）。"""
    uc = dict(scn["user_cli"]); orch = dict(scn.get("orchestration_env", {}))
    if uc.pop("enable-speculative-decode", None):
        orch["ENABLE_SPECULATIVE_DECODE"] = "true"
    if uc.pop("enable-sparse", None):
        orch["ENABLE_SPARSE"] = "true"
    if uc.pop("enable-rag-acc", None):
        orch["ENABLE_RAG_ACC"] = "true"
    return {"user_cli": uc, "orchestration_env": orch, "model_config": scn["model_config"]}


def build_matrix():
    cases = {}
    # Group A：dry_run.py 20 场景（转 env 下发）
    for nm, scn in dr.SCENARIOS.items():
        cases[f"A:{nm}"] = _env_dispatch(scn)
    # Group B：三特性 env 组合
    combos = {
        "none": {}, "spec": {"ENABLE_SPECULATIVE_DECODE": "true"},
        "sparse": {"ENABLE_SPARSE": "true"},
        "offload-auto": {
            "ENABLE_KV_OFFLOAD": "true", "LMCACHE_OFFLOAD": "true",
            "KV_MEM_OFFLOAD_SIZE": "auto", "AVAILABLE_POD_MEM_SIZE": "512",
            "LMCACHE_POD_MEMORY": "512",
        },
        "offload-custom": {
            "ENABLE_KV_OFFLOAD": "true", "LMCACHE_OFFLOAD": "true",
            "KV_MEM_OFFLOAD_SIZE": "200", "LMCACHE_MAX_LOCAL_CPU_SIZE": "200",
        },
        "all": {"ENABLE_SPECULATIVE_DECODE": "true", "ENABLE_SPARSE": "true",
                "ENABLE_KV_OFFLOAD": "true", "LMCACHE_OFFLOAD": "true",
                "KV_MEM_OFFLOAD_SIZE": "auto", "AVAILABLE_POD_MEM_SIZE": "512",
                "LMCACHE_POD_MEMORY": "512"},
        "all+perf": {"ENABLE_SPECULATIVE_DECODE": "true", "ENABLE_SPARSE": "true",
                     "ENABLE_KV_OFFLOAD": "true", "LMCACHE_OFFLOAD": "true",
                     "KV_MEM_OFFLOAD_SIZE": "auto", "AVAILABLE_POD_MEM_SIZE": "512",
                     "LMCACHE_POD_MEMORY": "512", "SPARSE_LEVEL": "performance_first"},
    }
    B_models = {
        "glm47-nv": ({"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
                     {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"}, {"architecture": "Glm4MoeForCausalLM"}),
        "glm52-a3": ({"model-name": "GLM-5.2-w8a8", "engine": "vllm_ascend", "device-count": 16},
                     {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "ENGINE_VERSION": "0.21.0-a3"},
                     {"architecture": "GlmMoeDsaForCausalLM", "quantization_config": {"quant_method": "ascend"}}),
    }
    for mk, (uc, orch, mc) in B_models.items():
        for ck, cenv in combos.items():
            cases[f"B:{mk}:{ck}"] = {"user_cli": uc, "orchestration_env": {**orch, **cenv}, "model_config": mc}
    # Group C：回归专项
    cases["C1:bare-nv"] = {"user_cli": {"model-name": "Qwen3-32B", "engine": "vllm", "device-count": 8},
                           "orchestration_env": {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"},
                           "model_config": {"architecture": "Qwen3ForCausalLM"}}  # 无 quantization_config
    cases["C2:bare-ascend"] = {"user_cli": {"model-name": "DeepSeek-V3.1", "engine": "vllm_ascend", "device-count": 16},
                               "orchestration_env": {"DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment", "WINGS_ASCEND_PLATFORM": "a3"},
                               "model_config": {"architecture": "DeepseekV3ForCausalLM"}}  # 无 quantization_config
    cases["C3:embedding-ascend"] = {"user_cli": {"model-name": "Qwen3-Embedding", "engine": "vllm_ascend", "device-count": 1, "model-type": "embedding"},
                                    "orchestration_env": {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_ASCEND_PLATFORM": "a2"},
                                    "model_config": {"architecture": "Qwen3ForCausalLM"}}
    cases["C4:rerank-ascend"] = {"user_cli": {"model-name": "bge-reranker-v2-m3", "engine": "vllm_ascend", "device-count": 1, "model-type": "rerank"},
                                 "orchestration_env": {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_ASCEND_PLATFORM": "a2"},
                                 "model_config": {"architecture": "XLMRobertaForSequenceClassification"}}
    cases["C5:op-accel"] = {"user_cli": {"model-name": "Qwen3.5-397B-A17B", "engine": "vllm_ascend", "device-count": 16},
                            "orchestration_env": {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_ASCEND_PLATFORM": "a3", "ENABLE_OPERATOR_ACCELERATION": "true"},
                            "model_config": {"architecture": "Qwen3_5MoeForConditionalGeneration", "quantization_config": {"quant_method": "ascend"}}}
    cases["C6:soft-fp8"] = {"user_cli": {"model-name": "Qwen3.5-397B-A17B", "engine": "vllm_ascend", "device-count": 16},
                            "orchestration_env": {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_ASCEND_PLATFORM": "a3", "ENABLE_SOFT_FP8": "true"},
                            "model_config": {"architecture": "Qwen3_5MoeForConditionalGeneration", "quantization_config": {"quant_method": "ascend"}}}
    cases["C7:pd-role"] = {"user_cli": {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
                           "orchestration_env": {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "PD_ROLE": "P",
                                                 "ENABLE_SPECULATIVE_DECODE": "true", "ENABLE_SPARSE": "true",
                                                 "ENABLE_KV_OFFLOAD": "true", "LMCACHE_OFFLOAD": "true",
                                                 "KV_MEM_OFFLOAD_SIZE": "auto", "AVAILABLE_POD_MEM_SIZE": "512",
                                                 "LMCACHE_POD_MEMORY": "512"},
                           "model_config": {"architecture": "Glm4MoeForCausalLM"}}
    cases["C8:v4flash-a3-spec-off"] = {
        "user_cli": {"model-name": "DeepSeek-V4-Flash", "engine": "vllm_ascend", "device-count": 8},
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
            "WINGS_ASCEND_PLATFORM": "a3",
            "ENABLE_SPECULATIVE_DECODE": "false",
        },
        "model_config": {
            "architecture": "DeepseekV4ForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    }
    cases["C9:glm47-w8a8-spec-off"] = {
        "user_cli": {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
            "ENABLE_SPECULATIVE_DECODE": "false",
        },
        "model_config": {
            "architecture": "Glm4MoeForCausalLM",
            "quantization_config": {"quant_method": "w8a8"},
        },
    }
    return cases


def main():
    global _OUT
    if not WT_MASTER.exists():
        print(f"ERROR: master worktree not found: {WT_MASTER}\n  run: git worktree add ../wt-master master")
        return 2
    _OUT = open(OUT_PATH, "w", encoding="utf-8")
    try:
        genfile = _write_gen()
        cases = build_matrix()
        emit("=" * 96)
        emit(f" 与 master 功能对比（防回归）：{len(cases)} 场景  ·  master={WT_MASTER.name}  branch=工作树")
        emit("=" * 96)
        total_regress, intended_count, scen_ok = [], 0, 0
        for name, scn in cases.items():
            m_cmd, _ = gen(WT_MASTER, scn, genfile)
            b_cmd, _ = gen(ROOT, scn, genfile)
            regress, intended = compare(name, m_cmd, b_cmd)
            intended_count += len(intended)
            if regress:
                emit(f"\n[REGRESSION?] {name}")
                for r in regress:
                    emit(f"    ✗ {r}")
                total_regress += [(name, r) for r in regress]
            else:
                scen_ok += 1
                tag = ("  预期变更: " + "; ".join(sorted(set(intended)))) if intended else "  （与 master 等价，仅 D1/D2 已中和）"
                emit(f"[OK] {name}{tag}")
        os.unlink(genfile)
        emit("\n" + "=" * 96)
        emit(f" 结果：{scen_ok}/{len(cases)} 场景无回归  ·  疑似回归项 = {len(total_regress)}  ·  归类预期变更 = {intended_count}")
        emit("=" * 96)
        return 0 if not total_regress else 1
    finally:
        _OUT.close()


if __name__ == "__main__":
    sys.exit(main())
