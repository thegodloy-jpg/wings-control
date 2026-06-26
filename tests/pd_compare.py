#!/usr/bin/env python3
"""对比 dry_run 生成的 PD 脚本 与 官方 launch_online_dp.py + run_dp_template.sh 命令。

用法:
  python dry_run.py --pd glm5 && python dry_run.py --pd v4flash   # 先生成
  python tests/pd_compare.py                                       # 再对比

逐 flag 报告 match / MISMATCH / MISSING / EXTRA(仅对关注的语义 flag)。
官方基准取自 vllm-ascend tutorials（GLM5 / DeepSeek-V4-Flash），数值以官方为准。
"""
import json
import os
import re
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "output")

# ── 官方基准（关注的语义 flag；None=只校验存在）──
#   topo/kv 在代码里单独校验，这里放 engine flag + env。
OFFICIAL = {
    ("glm5", "P"): {
        "flags": {"tensor-parallel-size": "16", "data-parallel-size": "2",
                  "max-num-batched-tokens": "4096", "max-num-seqs": "64",
                  "gpu-memory-utilization": "0.95", "max-model-len": "200000",
                  "enable-expert-parallel": True, "quantization": "ascend",
                  "data-parallel-external-lb": True},
        "spec": {"num_speculative_tokens": 3, "method": "deepseek_mtp"},
        "kv": {"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_producer", "kv_port": "30000"},
        "kv_extra": {"use_ascend_direct": True, "prefill": {"dp_size": 2, "tp_size": 16},
                     "decode": {"dp_size": 16, "tp_size": 4}},
        "env": ["VLLM_ASCEND_ENABLE_FLASHCOMM1", "VLLM_ASCEND_ENABLE_FUSED_MC2"],
    },
    ("glm5", "D"): {
        "flags": {"tensor-parallel-size": "4", "data-parallel-size": "16",
                  "max-num-batched-tokens": "32", "max-num-seqs": "8",
                  "gpu-memory-utilization": "0.92", "enable-expert-parallel": True,
                  "data-parallel-external-lb": True},
        "compilation": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "kv": {"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_consumer", "kv_port": "30100"},
        "kv_extra": {"use_ascend_direct": True, "prefill": {"dp_size": 2, "tp_size": 16},
                     "decode": {"dp_size": 16, "tp_size": 4}},
        "env": ["VLLM_ASCEND_ENABLE_FUSED_MC2", "VLLM_ASCEND_ENABLE_MLAPO"],
    },
    # 官方 GLM-5.2 A2 4P4D（用户提供的手工 P/D 脚本，kv_p2p MooncakeConnector + role 级 engine_id 0/1）。
    # max-model-len 故意不校验：官方 P=115168/D=135168 vs wings 131072/200000 是用户可控长度（刻意 delta，见报告 §2）。
    ("glm52-a2", "P"): {
        "flags": {"tensor-parallel-size": "8", "data-parallel-size": "4",
                  "max-num-batched-tokens": "4096", "max-num-seqs": "64",
                  "gpu-memory-utilization": "0.95", "quantization": "ascend",
                  "enable-expert-parallel": True, "enable-chunked-prefill": True,
                  "enforce-eager": True, "data-parallel-external-lb": True},
        "spec": {"num_speculative_tokens": 3, "method": "deepseek_mtp"},
        "kv": {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer", "kv_port": "30000"},
        "kv_extra": {"use_ascend_direct": True, "prefill": {"dp_size": 4, "tp_size": 8},
                     "decode": {"dp_size": 8, "tp_size": 4}},
        "engine_id_role": "0",
        "env": ["VLLM_ASCEND_ENABLE_FLASHCOMM1", "VLLM_NIXL_ABORT_REQUEST_TIMEOUT"],
    },
    ("glm52-a2", "D"): {
        "flags": {"tensor-parallel-size": "4", "data-parallel-size": "8",
                  "max-num-batched-tokens": "164", "max-num-seqs": "48",
                  "gpu-memory-utilization": "0.92", "quantization": "ascend",
                  "enable-expert-parallel": True, "data-parallel-external-lb": True},
        "compilation": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "spec": {"num_speculative_tokens": 3, "method": "deepseek_mtp"},
        "kv": {"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer", "kv_port": "30100"},
        "kv_extra": {"use_ascend_direct": True, "prefill": {"dp_size": 4, "tp_size": 8},
                     "decode": {"dp_size": 8, "tp_size": 4}},
        "engine_id_role": "1",
        "env": ["VLLM_ASCEND_ENABLE_MLAPO", "DYNAMIC_EPLB", "TASK_QUEUE_ENABLE"],
    },
    ("v4flash", "P"): {
        "flags": {"tensor-parallel-size": "4", "data-parallel-size": "4",
                  "max-num-batched-tokens": "4096", "max-num-seqs": "64",
                  "gpu-memory-utilization": "0.95", "enable-expert-parallel": True,
                  "data-parallel-external-lb": True},
        "kv": {"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_producer", "kv_port": "30000"},
        "kv_extra": {"prefill": {"dp_size": 4, "tp_size": 4}, "decode": {"dp_size": 16, "tp_size": 1}},
        "engine_id_per_rank": True,
        "env": ["VLLM_ASCEND_ENABLE_FLASHCOMM1", "VLLM_ASCEND_ENABLE_FUSED_MC2"],
    },
    ("v4flash", "D"): {
        "flags": {"tensor-parallel-size": "1", "data-parallel-size": "16",
                  "max-num-batched-tokens": "32", "max-num-seqs": "8",
                  "gpu-memory-utilization": "0.92", "enable-expert-parallel": True,
                  "data-parallel-external-lb": True},
        "compilation": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "kv": {"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_consumer", "kv_port": "30100"},
        "kv_extra": {"prefill": {"dp_size": 4, "tp_size": 4}, "decode": {"dp_size": 16, "tp_size": 1}},
        "engine_id_per_rank": True,
        "env": ["VLLM_ASCEND_ENABLE_FUSED_MC2", "VLLM_ASCEND_ENABLE_MLAPO"],
    },
}

OK, NG = 0, 0


def _r(name, ok, detail=""):
    global OK, NG
    print(f"    {'OK  ' if ok else 'FAIL'} {name}" + ("" if ok else f"  <{detail}>"))
    OK += ok
    NG += (not ok)


def _engine_line(text):
    """取 fork 循环里的 vllm serve 命令行（含 --data-parallel-external-lb 的那行，去前缀/尾&）。"""
    for line in text.splitlines():
        if "--data-parallel-external-lb" in line and ("vllm" in line):
            s = line.strip()
            s = re.sub(r"^ASCEND_RT_VISIBLE_DEVICES=\S+\s+", "", s)
            s = re.sub(r"\s*&\s*$", "", s)
            return s
    return None


def _parse_flags(line):
    toks = shlex.split(line)
    flags, jsons = {}, {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                val = toks[i + 1]
                flags[key] = val
                if val.startswith("{"):
                    try:
                        jsons[key] = json.loads(val)
                    except json.JSONDecodeError:
                        pass
                i += 2
            else:
                flags[key] = True
                i += 1
        else:
            i += 1
    return flags, jsons


def compare(scenario, role):
    fn = os.path.join(OUT, f"start_command_pd-{scenario}-{role}_node0.sh")
    print(f"\n== {scenario} {role}  ({os.path.basename(fn)}) ==")
    if not os.path.exists(fn):
        _r("script exists", False, "未生成，先跑 dry_run.py --pd")
        return
    text = open(fn, encoding="utf-8").read()
    line = _engine_line(text)
    if not line:
        _r("engine line found", False)
        return
    # 归一化 bash 占位（kv_port 按 base+i 偏移、engine_id 按 rank），使 JSON 可解析
    norm = line.replace("'\"$KVPORT\"'", "OFFSET_KVPORT").replace("'\"$RANK\"'", "PERRANK")
    flags, jsons = _parse_flags(norm)
    spec = OFFICIAL[(scenario, role)]

    # 1) engine flags
    for k, v in spec["flags"].items():
        if v is True:
            _r(f"flag --{k}", k in flags, "缺失")
        else:
            _r(f"flag --{k}={v}", flags.get(k) == v, f"got={flags.get(k)}")

    # 2) speculative-config
    if "spec" in spec:
        sc = jsons.get("speculative-config", {})
        _r("speculative tokens/method",
           sc.get("num_speculative_tokens") == spec["spec"]["num_speculative_tokens"]
           and sc.get("method") == spec["spec"]["method"], f"got={sc}")

    # 3) compilation-config
    if "compilation" in spec:
        cc = jsons.get("compilation-config", {})
        _r("compilation cudagraph_mode",
           cc.get("cudagraph_mode") == spec["compilation"]["cudagraph_mode"], f"got={cc}")

    # 4) kv-transfer-config
    kv = jsons.get("kv-transfer-config", {})
    _r(f"kv kv_connector={spec['kv']['kv_connector']}", kv.get("kv_connector") == spec["kv"]["kv_connector"], f"got={kv.get('kv_connector')}")
    _r(f"kv kv_role={spec['kv']['kv_role']}", kv.get("kv_role") == spec["kv"]["kv_role"], f"got={kv.get('kv_role')}")
    # kv_port 按 base+i 每 service 偏移：JSON 为占位，循环里 KVPORT=$((base + i))
    _r(f"kv kv_port 偏移 base={spec['kv']['kv_port']}+i",
       kv.get("kv_port") == "OFFSET_KVPORT" and f"KVPORT=$(({spec['kv']['kv_port']} + i))" in text,
       f"got kv_port={kv.get('kv_port')}")
    extra = kv.get("kv_connector_extra_config", {})
    for k, v in spec["kv_extra"].items():
        _r(f"kv_extra {k}", extra.get(k) == v, f"got={extra.get(k)}")
    if spec.get("engine_id_per_rank"):
        _r("kv engine_id 按rank占位", kv.get("engine_id") == "PERRANK", f"got={kv.get('engine_id')}")
    if "engine_id_role" in spec:
        # 官方 kv_p2p MooncakeConnector：engine_id 是 role 级常量（P=0 / D=1），非 per-rank
        _r(f"kv engine_id role 级={spec['engine_id_role']}",
           kv.get("engine_id") == spec["engine_id_role"], f"got={kv.get('engine_id')}")

    # 5) 角色 env（全脚本搜 export）
    for e in spec["env"]:
        _r(f"env {e}", f"export {e}=" in text, "缺失")


def main():
    for scenario in ("glm5", "glm52-a2", "v4flash"):
        for role in ("P", "D"):
            compare(scenario, role)
    print(f"\n==== 对比结果: {OK} OK / {NG} FAIL ====")
    return 1 if NG else 0


if __name__ == "__main__":
    sys.exit(main())
