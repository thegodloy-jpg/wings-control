#!/usr/bin/env python3
"""PD external-lb 验证 harness（离线层 A/B/C/D/E）。

用法: python tests/pd_external_lb_verify.py
返回码 0 = 全过；非 0 = 有失败。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wings_control"))

PASS, FAIL = 0, 0


def _find_bash():
    """解析真正的 bash（避开 Windows System32\\bash.exe = WSL 启动器）。"""
    cands = [shutil.which("bash"),
             r"D:\app\Git\usr\bin\bash.exe",
             r"C:\Program Files\Git\bin\bash.exe",
             r"C:\Program Files\Git\usr\bin\bash.exe"]
    for c in cands:
        if c and os.path.exists(c) and "System32" not in c and "system32" not in c:
            return c
    return shutil.which("bash")


BASH = _find_bash()


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def _clear_pd_env():
    for k in list(os.environ):
        if k.startswith(("PD_", "DP_", "TP_")) or k in (
                "NODE_IPS", "HOST_IP", "Master_IP", "MASTER_IP",
                "VLLM_LLMDD_RPC_PORT", "RANK_IP"):
            os.environ.pop(k, None)


# ───────────────────────── 层 A：参数矩阵 ─────────────────────────
def layer_a():
    print("\n== 层 A：参数矩阵（契约解析 + rank 派生）==")
    from core import config_loader as C

    cases = [
        # (label, env, expect_or_None)
        ("no role", {}, None),
        ("role D, no DP_SIZE (1P1D)", {"PD_ROLE": "D"}, None),
        ("role D, DP_SIZE=1", {"PD_ROLE": "D", "DP_SIZE": "1"}, None),
        ("D-node0 (HOST=...10 idx0)",
         {"PD_ROLE": "D", "DP_SIZE": "4", "TP_SIZE": "1", "DP_SIZE_LOCAL": "2",
          "Master_IP": "70.0.0.10", "VLLM_LLMDD_RPC_PORT": "12321",
          "NODE_IPS": "70.0.0.10,70.0.0.11", "HOST_IP": "70.0.0.10"},
         {"dp_rank_start": 0, "dp_size": 4, "tp_size": 1, "dp_size_local": 2,
          "dp_address": "70.0.0.10", "rpc_port": "12321"}),
        ("D-node1 (HOST=...11 idx1 -> start=2)",
         {"PD_ROLE": "D", "DP_SIZE": "4", "TP_SIZE": "1", "DP_SIZE_LOCAL": "2",
          "Master_IP": "70.0.0.10", "VLLM_LLMDD_RPC_PORT": "12321",
          "NODE_IPS": "70.0.0.10,70.0.0.11", "HOST_IP": "70.0.0.11"},
         {"dp_rank_start": 2}),
        ("P single node",
         {"PD_ROLE": "P", "DP_SIZE": "2", "TP_SIZE": "2", "DP_SIZE_LOCAL": "2",
          "Master_IP": "94.0.0.1", "VLLM_LLMDD_RPC_PORT": "12321",
          "NODE_IPS": "94.0.0.1", "HOST_IP": "94.0.0.1"},
         {"dp_rank_start": 0, "dp_size": 2}),
        ("legacy PD_* fallback",
         {"PD_ROLE": "D", "PD_DP_SIZE": "8", "PD_TP_SIZE": "4",
          "PD_DP_SIZE_LOCAL": "4", "PD_DP_RANK_START": "4", "PD_DP_ADDRESS": "1.1.1.1"},
         {"dp_rank_start": 4, "dp_size": 8, "tp_size": 4}),
        # 真机回归（serving-...-8bz4m/7bnhc）：同宿主机两 pod 共享同一 HOST_IP（节点物理 IP），
        # 但各自 RANK_IP 不同。必须用 RANK_IP 派生 rank —— 用 HOST_IP 会让两 pod 都算出
        # rank_start=0 → rank 撞车 → rank0 去 bind 别人 IP 的 rpc 端口 → ZMQ "Cannot assign requested address"。
        ("D same-HOST node0 (RANK=.46 idx0)",
         {"PD_ROLE": "D", "DP_SIZE": "4", "TP_SIZE": "1", "DP_SIZE_LOCAL": "2",
          "Master_IP": "94.254.215.46", "VLLM_LLMDD_RPC_PORT": "12321",
          "NODE_IPS": "94.254.215.46,94.254.215.43",
          "HOST_IP": "94.254.215.46", "RANK_IP": "94.254.215.46"},
         {"dp_rank_start": 0}),
        ("D same-HOST node1 (RANK=.43 idx1 -> start=2; HOST=.46 与 node0 相同)",
         {"PD_ROLE": "D", "DP_SIZE": "4", "TP_SIZE": "1", "DP_SIZE_LOCAL": "2",
          "Master_IP": "94.254.215.46", "VLLM_LLMDD_RPC_PORT": "12321",
          "NODE_IPS": "94.254.215.46,94.254.215.43",
          "HOST_IP": "94.254.215.46", "RANK_IP": "94.254.215.43"},
         {"dp_rank_start": 2}),
    ]
    for label, env, expect in cases:
        _clear_pd_env()
        os.environ.update(env)
        got = C._get_pd_external_lb_params()
        if expect is None:
            check(label, got is None, f"got={got}")
        else:
            ok = got is not None and all(got.get(k) == v for k, v in expect.items())
            check(label, ok, f"got={got}")
    _clear_pd_env()


# ───────────────────────── 层 B/C/E：生成 + 静态 + 回归 ─────────────────────────
def _gen(arch, role, dp, tp, local, node_ips, host_ip, model_name="M", extra_env=None):
    """生成一个场景的 start_command.sh 文本。"""
    _clear_pd_env()
    from core.start_args_compat import parse_launch_args
    from core.port_plan import derive_port_plan
    from core.wings_entry import build_launcher_plan
    from config.settings import settings

    d = tempfile.mkdtemp(prefix="pdm_", dir=os.path.join(ROOT, "build")).replace("\\", "/")
    json.dump({"architectures": [arch], "model_type": "deepseek_v3", "torch_dtype": "bfloat16",
               "num_hidden_layers": 4, "hidden_size": 512, "num_attention_heads": 8, "head_dim": 64},
              open(os.path.join(d, "config.json"), "w"))
    sv = tempfile.mkdtemp(prefix="sv_", dir=os.path.join(ROOT, "build")).replace("\\", "/")
    env = {"ENGINE": "vllm_ascend", "MODEL_NAME": model_name, "MODEL_PATH": d, "MODEL_TYPE": "auto",
           "DEVICE_COUNT": str(local * tp), "DISTRIBUTED": "false", "NNODES": "1", "NODE_RANK": "0",
           "POD_IP": host_ip, "WINGS_DEVICE": "ascend", "WINGS_ASCEND_PLATFORM": "a3",
           "SHARED_VOLUME_PATH": sv, "ENGINE_PORT": "18000", "PORT": "18000"}
    if role:
        env.update({"PD_ROLE": role, "DP_SIZE": str(dp), "TP_SIZE": str(tp),
                    "DP_SIZE_LOCAL": str(local), "Master_IP": node_ips.split(",")[0],
                    "VLLM_LLMDD_RPC_PORT": "12321", "NODE_IPS": node_ips, "HOST_IP": host_ip,
                    "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "16",
                    "PD_DECODE_DP_SIZE": "8", "PD_DECODE_TP_SIZE": "4"})
    if extra_env:
        env.update(extra_env)
    os.environ.update(env)
    la = parse_launch_args(["--model-name", model_name, "--model-path", d, "--engine", "vllm_ascend",
                            "--device-count", str(local * tp), "--nnodes", "1", "--node-rank", "0"])
    pp = derive_port_plan(port=la.port, enable_reason_proxy=settings.ENABLE_REASON_PROXY,
                          health_port=settings.HEALTH_PORT)
    cmd = build_launcher_plan(la, pp).command
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(sv, ignore_errors=True)
    _clear_pd_env()
    return cmd


def _bash_n(script_text):
    """bash -n 语法检查；返回 (ok, stderr)。"""
    path = os.path.join(ROOT, "build", "_bashn.sh").replace("\\", "/")
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(script_text)
    try:
        r = subprocess.run([BASH, "-n", path], capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.returncode == 0, r.stderr
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def layer_bce():
    print("\n== 层 B：生成断言 ==")
    # V3.2 D (DP8 TP4 local4)
    d = _gen("DeepseekV32ForCausalLM", "D", 8, 4, 4, "7.0.0.1", "7.0.0.1", "V32D")
    check("D: fork 4 (seq 0 3)", "for i in $(seq 0 3)" in d)
    check("D: dp-size 8", "--data-parallel-size 8" in d)
    check("D: tp 4", "--tensor-parallel-size 4" in d)
    check("D: Layerwise consumer + kv_port 偏移(30100+i)",
          '"kv_connector":"MooncakeLayerwiseConnector","kv_role":"kv_consumer"' in d
          and "KVPORT=$((30100 + i))" in d and "BOOTSTRAP=$((23100 + i))" in d)
    check("D: registry batched 12", "--max-num-batched-tokens 12" in d)
    check("D: FULL_DECODE_ONLY", "FULL_DECODE_ONLY" in d)
    check("D: TASK_QUEUE env", "TASK_QUEUE_ENABLE" in d)
    check("D: external-lb flag", "--data-parallel-external-lb" in d)
    check("D: subshell backgrounded ') &'", ") &" in d)
    check("D: ENGINE_PID monitor", "ENGINE_PID=$!" in d)
    check("D: rpc 12321", "--data-parallel-rpc-port 12321" in d)
    check("D: no rank-start flag (derived)", "--data-parallel-start-rank" not in d.split("for i in")[1])

    # V3.2 P (DP2 TP16 local1) single service
    p = _gen("DeepseekV32ForCausalLM", "P", 2, 16, 1, "9.0.0.1", "9.0.0.1", "V32P")
    check("P: single service (seq 0 0)", "for i in $(seq 0 0)" in p)
    check("P: tp 16 / dp 2", "--tensor-parallel-size 16" in p and "--data-parallel-size 2" in p)
    check("P: producer + kv_port 偏移(30000+i)",
          '"kv_role":"kv_producer"' in p and "KVPORT=$((30000 + i))" in p)
    check("P: enforce-eager", "--enforce-eager" in p)
    check("P: layer_sharding", "layer_sharding" in p)
    check("P: FLASHCOMM1", "VLLM_ASCEND_ENABLE_FLASHCOMM1" in p)

    # 未注册 -> default (V1 + engine_id 占位)
    u = _gen("TotallyUnknownForCausalLM", "D", 4, 1, 4, "7.0.0.1", "7.0.0.1", "U")
    check("default: MooncakeConnectorV1", '"kv_connector":"MooncakeConnectorV1"' in u)
    check("default: engine_id 占位按rank", '"engine_id":"\'"$RANK"\'"' in u)
    check("default: kv_port 偏移(30400+i)", "KVPORT=$((30400 + i))" in u)
    check("default: disable-hybrid-kv", "--disable-hybrid-kv-cache-manager" in u)

    # D-node1 rank 派生
    d1 = _gen("DeepseekV32ForCausalLM", "D", 4, 1, 2, "7.0.0.1,7.0.0.2", "7.0.0.2", "V32D1")
    check("D-node1: rank_start=2 (RANK=$((2 + i)))", "RANK=$((2 + i))" in d1)

    print("\n== 层 C：bash -n 静态语法 ==")
    for label, txt in [("V32-D", d), ("V32-P", p), ("default", u), ("D-node1", d1)]:
        ok, err = _bash_n(txt)
        check(f"bash -n {label}", ok, err.strip()[:200])

    print("\n== 层 E：回归（无 PD -> 不含 external-lb）==")
    base = _gen("DeepseekV32ForCausalLM", "", 0, 0, 0, "7.0.0.1", "7.0.0.1", "NOPD")
    check("non-PD: 无 external-lb", "--data-parallel-external-lb" not in base)
    check("non-PD: 无 fork pids", "pids=()" not in base)


# ───────────────────────── 层 D：fork 运行时仿真（mock vllm）─────────────────────────
def layer_d():
    print("\n== 层 D：fork 运行时仿真（mock vllm，真 bash 跑 fork）==")
    from engines import vllm_adapter as V

    workdir = tempfile.mkdtemp(prefix="pdsim_", dir=os.path.join(ROOT, "build"))
    wd = workdir.replace("\\", "/")
    record = f"{wd}/record.txt"
    mock = f"{wd}/mock_vllm.sh"
    with open(mock, "w", newline="\n") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            "rank=''; port=''; kv=''\n"
            "while [ $# -gt 0 ]; do case \"$1\" in\n"
            "  --data-parallel-rank) rank=\"$2\"; shift 2;;\n"
            "  --port) port=\"$2\"; shift 2;;\n"
            "  --kv-transfer-config) kv=\"$2\"; shift 2;;\n"
            "  *) shift;; esac; done\n"
            "eid=$(printf %s \"$kv\" | grep -oE '\"engine_id\":\"[0-9]+\"' || true)\n"
            "kvp=$(printf %s \"$kv\" | grep -oE '\"kv_port\":\"[0-9]+\"' || true)\n"
            f"echo \"rank=$rank port=$port cards=$ASCEND_RT_VISIBLE_DEVICES eid=$eid kvp=$kvp bootstrap=$VLLM_MOONCAKE_BOOTSTRAP_PORT pid=$$\" >> {record}\n"
            # exec sleep：被 kill 时立即退出（不用 trap，避免 bash 在 sleep 期间延迟 trap）\n"
            "exec sleep 30\n"
        )
    os.chmod(mock, 0o755)

    # 构造一个 mock base cmd（含 --port 占位 + engine_id/kv_port 占位）
    cmd = (f"bash {mock} --host 0.0.0.0 --port 17000 --model /x "
           "--kv-transfer-config '{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_port\":\"__PD_KVPORT__\",\"engine_id\":\"__PD_RANK__\"}'")
    ext = {"role": "D", "dp_size": 8, "tp_size": 4, "dp_size_local": 4,
           "dp_rank_start": 0, "dp_address": "1.2.3.4", "rpc_port": "12777",
           "kv_port_base": 30100, "bootstrap_base": 23100}
    params = {"_pd_env": {"TASK_QUEUE_ENABLE": "1"}}
    block = V._build_vllm_pd_external_lb_script(params, cmd, [], ext)
    block_path = f"{wd}/fork.sh"
    with open(block_path, "w", newline="\n") as f:
        f.write(block)

    # 驱动脚本：后台跑 fork 子shell；待 4 个 mock 记录；杀其中一个；验证整组被拆卸
    driver = f"""#!/usr/bin/env bash
bash {block_path} &
BLOCK=$!
for _ in $(seq 1 50); do
  [ -f {record} ] && [ "$(wc -l < {record})" -ge 4 ] && break
  sleep 0.2
done
echo "PHASE=started count=$(wc -l < {record} 2>/dev/null || echo 0)"
ONEPID=$(head -1 {record} | sed -E 's/.*pid=([0-9]+).*/\\1/')
echo "PHASE=killing pid=$ONEPID"
kill "$ONEPID" 2>/dev/null
for _ in $(seq 1 50); do kill -0 "$BLOCK" 2>/dev/null || break; sleep 0.2; done
ALIVE=0
for p in $(sed -E 's/.*pid=([0-9]+).*/\\1/' {record}); do kill -0 "$p" 2>/dev/null && ALIVE=$((ALIVE+1)); done
echo "PHASE=teardown alive_mocks=$ALIVE block_alive=$(kill -0 $BLOCK 2>/dev/null && echo 1 || echo 0)"
"""
    drv = f"{wd}/drive.sh"
    with open(drv, "w", newline="\n") as f:
        f.write(driver)
    r = subprocess.run([BASH, drv], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    out = r.stdout + r.stderr

    lines = [l for l in open(record).read().splitlines() if l.strip()] if os.path.exists(record) else []
    parsed = {}
    for l in lines:
        kv = dict(p.split("=", 1) for p in l.split() if "=" in p)
        parsed[kv.get("rank")] = kv

    check("D: 4 service 启动", len(lines) == 4, f"got {len(lines)} lines: {lines}")
    check("D: ranks = {0,1,2,3}", set(parsed) == {"0", "1", "2", "3"}, f"ranks={set(parsed)}")
    ports = {parsed[r]["port"] for r in parsed} if parsed else set()
    check("D: ports = 17000..17003", ports == {"17000", "17001", "17002", "17003"}, f"ports={ports}")
    cards = {parsed[r]["cards"] for r in parsed} if parsed else set()
    check("D: 卡组互异 (i*4..)", cards == {"0,1,2,3", "4,5,6,7", "8,9,10,11", "12,13,14,15"}, f"cards={cards}")
    eids_ok = all(parsed[r].get("eid", "") == f'"engine_id":"{r}"' for r in parsed) if parsed else False
    check("D: engine_id 按 rank 互异", eids_ok, f"eids={{r: parsed[r].get('eid') for r in parsed}}")
    kvps = {parsed[r].get("kvp") for r in parsed} if parsed else set()
    check("D: kv_port 偏移互异 = 30100..30103",
          kvps == {'"kv_port":"30100"', '"kv_port":"30101"', '"kv_port":"30102"', '"kv_port":"30103"'}, f"kvps={kvps}")
    bts = {parsed[r].get("bootstrap") for r in parsed} if parsed else set()
    check("D: bootstrap 偏移互异 = 23100..23103",
          bts == {"23100", "23101", "23102", "23103"}, f"bootstrap={bts}")
    check("D: 杀一个 service -> 整组拆卸(无存活 mock)", "alive_mocks=0" in out, out)
    check("D: fork 子shell 已退出", "block_alive=0" in out, out)

    shutil.rmtree(workdir, ignore_errors=True)


def layer_f():
    """层 F：注册表表达力修复（L2 角色 extra / L3 common_env / L4 平台 overlay / L5 键存活）。"""
    print("\n== 层 F：注册表表达力修复（L2/L3/L4/L5）==")

    # ── L4：V4-Flash 平台 overlay（a3 基条目 vs a2 overlay）──
    v4_a3_p = _gen("DeepseekV4ForCausalLM", "P", 4, 4, 4, "8.0.0.1", "8.0.0.1", "V4P",
                   extra_env={"WINGS_ASCEND_PLATFORM": "a3"})
    check("L4 a3(基): P batched 8192", "--max-num-batched-tokens 8192" in v4_a3_p)
    v4_a2_p = _gen("DeepseekV4ForCausalLM", "P", 4, 4, 4, "8.0.0.1", "8.0.0.1", "V4P",
                   extra_env={"WINGS_ASCEND_PLATFORM": "a2"})
    check("L4 a2 overlay: P batched 4096（覆盖 8192）", "--max-num-batched-tokens 4096" in v4_a2_p)
    v4_a2_d = _gen("DeepseekV4ForCausalLM", "D", 16, 1, 16, "8.0.1.1", "8.0.1.1", "V4D",
                   extra_env={"WINGS_ASCEND_PLATFORM": "a2"})
    check("L4 a2 overlay: D batched 60 / seqs 30",
          "--max-num-batched-tokens 60" in v4_a2_d and "--max-num-seqs 30" in v4_a2_d)
    check("L4 a2 继承基条目: max-model-len 1048576", "--max-model-len 1048576" in v4_a2_d)

    # ── L3：GLM5 common_env 覆盖 base 且整段去重 ──
    glm_p = _gen("GlmMoeDsaForCausalLM", "P", 2, 16, 1, "7.0.0.1", "7.0.0.1", "GP")
    seg = glm_p.split("for i in")[0]  # 首个 exec path 的 env 段
    check("L3 common_env: HCCL_BUFFSIZE=256 生效", "export HCCL_BUFFSIZE=256" in seg)
    check("L3 dedupe: 无残留 HCCL_BUFFSIZE=1024", "export HCCL_BUFFSIZE=1024" not in seg)
    check("L3 common_env: Mooncake 共用 env 注入",
          "export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480" in seg
          and "export ASCEND_AGGREGATE_ENABLE=1" in seg)

    # ── L2：Qwen3.5 角色级 extra_config（D 有 kv_buffer_device，P 无）──
    q_d = _gen("Qwen3_5MoeForConditionalGeneration", "D", 16, 2, 8, "5.0.1.1", "5.0.1.1", "Q5D")
    q_p = _gen("Qwen3_5MoeForConditionalGeneration", "P", 8, 2, 4, "5.0.0.1", "5.0.0.1", "Q5P")
    check("L2 D: kv_buffer_device=npu 在消费端 KV extra", "kv_buffer_device" in q_d)
    check("L2 P: 不含 kv_buffer_device（不污染 P）", "kv_buffer_device" not in q_p)

    # ── L5：注册表「易被注入器回填」的键全部存活（锁防回退）──
    check("L5 GLM5-P: enforce-eager 存活", "--enforce-eager" in glm_p)
    check("L5 GLM5-P: prefix 关（无 --enable-prefix-caching）", "--enable-prefix-caching" not in glm_p)
    check("L5 GLM5-P: compilation 删（无 --compilation-config）", "--compilation-config" not in glm_p)
    check("L5 GLM5-P: max-model-len 131072 存活", "--max-model-len 131072" in glm_p)
    check("L5 V4-P(a3): async 关 + compilation 删",
          "--async-scheduling" not in v4_a3_p and "--compilation-config" not in v4_a3_p)
    check("L5 V4-P(a3): --no-enable-prefix-caching 存活", "--no-enable-prefix-caching" in v4_a3_p)
    check("L5 V4-P(a3): speculative method=mtp 存活", '"method":"mtp"' in v4_a3_p)


def main():
    layer_a()
    layer_bce()
    layer_f()
    if BASH:
        layer_d()
    else:
        print("\n[skip] 层 D：未找到 bash")
    print(f"\n==== 结果: {PASS} PASS / {FAIL} FAIL ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
