# 三特性使能 — 全量下发 + 白名单过滤 验证方案

> 关联：[需求一-三特性使能.md](../需求一-三特性使能.md)
> 运行：`python tests/dryrun_requirement_coverage.py`（本文件用例已纳入该驱动器，新增用例待补充入 Python）
>
> **核心口径**：MaaS 全量下发三开关=true → 白名单收口决定实际产出。
> 链路：`三开关全开(env) → resolve_feature_whitelist(engine,model,card) → apply_effective_feature_enablement → 产出口 → start_command.sh + advanced_features.json`
>
> **三段式入参**（真实下发口径）：
>
> | 入参段 | 代表谁 | 内容 |
> | --- | --- | --- |
> | `user_cli` | 用户 `wings_start.sh` CLI | `--model-name/--engine/--device-count/--distributed`…（**不含三特性开关**） |
> | `orchestration_env` | 编排层/MaaS 注入 env | 三特性开关 + 拓扑/平台/ENGINE_VERSION/SPARSE_LEVEL/KV_*… |
> | `model_config` | 模型权重 `config.json` | `architecture` + `quantization_config` |
>
> **出参断言对象**：
>
> | 出参 | 来源 | 断言示例 |
> | --- | --- | --- |
> | `command` | `plan.command`（start_command.sh） | 含/不含 `--speculative-config`、`--hf-overrides`、`KV_MEM_OFFLOAD_SIZE=N`… |
> | `features` | `advanced_features.json` | `speculative_decode`/`sparse_kv`/`kv_offload` bool |
> | `variants` | `advanced_features.json` | `speculative_decode="deepseek_mtp"`、`sparse_kv="fp8"`… |
> | `logs` | 生产代码 INFO/WARNING | 收口摘要 / 卡型 miss 告警 / 抑制日志 |

---

## 第一部分 · 白名单 12 行全开关开（核心验证）

> 每行：三开关全开（`ENABLE_SPECULATIVE_DECODE/ENABLE_SPARSE/ENABLE_KV_OFFLOAD=true`）
> 白名单收口后各模型产出不同——这是验证的核心。

---

### WL-01 · vllm / qwen3.5-397b / * → {spec, sparse}

```yaml
入参:
  user_cli:
    model-name: "qwen3.5-397b"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
  model_config:
    architecture: "Qwen3_5MoeForConditionalGeneration"

期望出参:
  command:
    - 含 "--speculative-config"              # spec 命中白名单
    - 含 "kv_cache_dtype=fp8"               # sparse=fp8（Qwen3_5 ∉ INDEXCACHE_ARCHS）
    - 不含 "ENABLE_KV_OFFLOAD=true"          # offload 不在白名单 → C14 收口关
    - 不含 "KV_MEM_OFFLOAD_SIZE"             # 同上
  features:
    speculative_decode: true
    sparse_kv: true
    kv_offload: false                        # 白名单无 offload → 收口关
  variants:
    speculative_decode: "qwen3_5_mtp"        # 无 offload 在场 → mtp 保留
    sparse_kv: "fp8"
    kv_offload: null
  logs:
    - 含 "offload requested but not in whitelist"   # 抑制日志
```

---

### WL-02 · vllm / glm-4.7 / * → {spec, sparse, offload}

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 含 "--speculative-config"              # spec 命中白名单，但被 offload 降级为 suffix
    - 含 "kv_cache_dtype=fp8"               # Glm4Moe ∉ INDEXCACHE_ARCHS
    - 含 "export KV_MEM_OFFLOAD_SIZE="       # offload 命中白名单，auto 自算容量
  features:
    speculative_decode: true
    sparse_kv: true
    kv_offload: true
  variants:
    speculative_decode: "suffix"             # ❗ offload 在场 → glm4_moe_mtp 降级为 suffix
    sparse_kv: "fp8"
    kv_offload: "lmcache_cpu+auto"           # LMCache CPU + C4 自算
```

---

### WL-03 · vllm / glm-5.1 / * → {sparse}

```yaml
入参:
  user_cli:
    model-name: "glm-5.1"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
  model_config:
    architecture: "GlmMoeDsaForCausalLM"

期望出参:
  command:
    - 含 "--speculative-config"              # suffix 地板也会合成 speculative-config
    - 含 "--hf-overrides"                    # sparse 命中 → IndexCache
    - 不含 "ENABLE_KV_OFFLOAD=true"          # offload 不在白名单
  features:
    speculative_decode: true                 # suffix 地板也是投机
    sparse_kv: true
    kv_offload: false
  variants:
    speculative_decode: "suffix"             # 白名单无 spec → 地板（修原误产 deepseek_mtp bug）
    sparse_kv: "indexcache_topk4"            # GlmMoeDsa ∈ INDEXCACHE_ARCHS, NV
    kv_offload: null
  logs:
    - 含 "spec not in whitelist → suffix floor"
    - 含 "offload requested but not in whitelist"
```

---

### WL-04 · vllm / minimax-m2.7 / * → {spec, sparse, offload}

```yaml
入参:
  user_cli:
    model-name: "minimax-m2.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
  model_config:
    architecture: "MinimaxM2_7ForCausalLM"

期望出参:
  command:
    - 含 "--speculative-config"              # suffix 地板
    - 含 "kv_cache_dtype=fp8"               # 非 IndexCache 架构
    - 含 "export KV_MEM_OFFLOAD_SIZE="
  features:
    speculative_decode: true
    sparse_kv: true
    kv_offload: true
  variants:
    speculative_decode: "suffix"             # arch 无 mtp_method → 恒 suffix
    sparse_kv: "fp8"
    kv_offload: "lmcache_cpu+auto"
```

---

### WL-05 · vllm / deepseek-v4-flash / * → {spec, sparse, offload}

```yaml
入参:
  user_cli:
    model-name: "deepseek-v4-flash"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
  model_config:
    architecture: "DeepseekV4ForCausalLM"

期望出参:
  command:
    - 含 "--speculative-config"              # mtp
    - 含 "method" "deepseek_mtp"             # V4 豁免 → mtp 保留
    - 含 "--hf-overrides"                    # V4-Flash NV IndexCache
    - 含 "use_index_cache"                   # sparse=indexcache_use_index_cache_topk4
    - 不含 "lmcache"                         # NV native 后端，非 LMCache
    - 含 "kv_offloading_backend"             # native_kv_offloading_backend
  features:
    speculative_decode: true
    sparse_kv: true
    kv_offload: true
  variants:
    speculative_decode: "deepseek_mtp"       # ✅ V4 豁免 → 不受 offload 影响
    sparse_kv: "indexcache_use_index_cache_topk4"
    kv_offload: "native_kv_offloading_backend"
```

---

### WL-06 · vllm_ascend / glm-4.7 / 910b(c) → {spec, offload}

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"                    # 全量下发 true
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
    WINGS_ASCEND_PLATFORM: "a2"              # 卡型=910b（命中白名单行#6）
  model_config:
    architecture: "Glm4MoeForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"              # spec 命中白名单
    - 不含 "--hf-overrides"                  # sparse 不在白名单 → C14 收口关
    - 含 "export KV_MEM_OFFLOAD_SIZE="       # offload 命中
  features:
    speculative_decode: true
    sparse_kv: false                         # 白名单无 sparse
    kv_offload: true
  variants:
    speculative_decode: "suffix"             # ❗ offload 在场 → mtp 降级（非 V4）
    sparse_kv: null
    kv_offload: "lmcache_cpu+auto"
  logs:
    - 含 "sparse requested but not in whitelist → suppressed"
```

---

### WL-07 · vllm_ascend / minimax-m2.5 / 910b(c) → {spec, offload}

```yaml
入参:
  user_cli:
    model-name: "minimax-m2.5"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
    WINGS_ASCEND_PLATFORM: "a2"
  model_config:
    architecture: "MinimaxM2_5ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"
    - 不含 "--hf-overrides"                  # sparse 不在白名单
    - 含 "export KV_MEM_OFFLOAD_SIZE="
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: true
  variants:
    speculative_decode: "suffix"             # arch 无 mtp_method → 恒 suffix
    sparse_kv: null
    kv_offload: "lmcache_cpu+auto"
```

---

### WL-08 · vllm_ascend / deepseek-v3.2 / 910c → {spec, offload}

```yaml
入参:
  user_cli:
    model-name: "deepseek-v3.2"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
    ENGINE_VERSION: "0.21.0-a3"              # 兜底卡型=910c（命中白名单行#8）
  model_config:
    architecture: "DeepseekV32ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"
    - 不含 "--hf-overrides"                  # sparse 不在白名单
    - 含 "export KV_MEM_OFFLOAD_SIZE="
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: true
  variants:
    speculative_decode: "suffix"             # ❗ offload 在场 → mtp 降级（非 V4，对照 WL-10）
    sparse_kv: null
    kv_offload: "lmcache_cpu+auto"
```

---

### WL-09 · vllm_ascend / qwen3.6 / 910b/910c → {spec}

```yaml
入参:
  user_cli:
    model-name: "qwen3.6"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENGINE_VERSION: "0.21.0-a3"
  model_config:
    architecture: "Qwen3_6ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"              # spec 命中白名单
    - 不含 "--hf-overrides"                  # sparse 不在白名单
    - 不含 "ENABLE_KV_OFFLOAD=true"          # offload 不在白名单
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: "qwen3_5_mtp"        # ✅ 无 offload 在场 → mtp 保留
    sparse_kv: null
    kv_offload: null
```

---

### WL-10 · vllm_ascend / deepseek-v4-flash / 910b(c) → {spec, sparse, offload}

```yaml
入参:
  user_cli:
    model-name: "deepseek-v4-flash"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
    WINGS_ASCEND_PLATFORM: "a2"              # 910b 或 a3=910c，均命中行#10
  model_config:
    architecture: "DeepseekV4ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"
    - 含 "method" "deepseek_mtp"             # ✅ V4 豁免 → mtp 共存
    - 含 "cpu_swap_space_gb=401"              # native 整节点 M_offload=401，不除卡数（对照 KV-01 LMCache=50均卡）
    - 不含 "lmcache"                         # Ascend V4 走 native connector，非 LMCache
  features:
    speculative_decode: true
    sparse_kv: true
    kv_offload: true
  variants:
    speculative_decode: "deepseek_mtp"       # ✅ V4 豁免共存
    sparse_kv: "noop"                        # Ascend 待 630 切 use_index_cache
    kv_offload: "native_cpu_connector"       # Ascend CPUOffloadingConnector
```

---

### WL-11 · vllm_ascend / glm-5.1 / 910b(c) → {sparse}

```yaml
入参:
  user_cli:
    model-name: "glm-5.1"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    WINGS_ASCEND_PLATFORM: "a2"
  model_config:
    architecture: "GlmMoeDsaForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"              # suffix 地板合成
    - 含 "--hf-overrides"                    # sparse 命中 → IndexCache
    - 含 "index_topk_freq:8"                 # Ascend GLM-5.1 → topk8
    - 不含 "ENABLE_KV_OFFLOAD=true"          # offload 不在白名单
  features:
    speculative_decode: true
    sparse_kv: true
    kv_offload: false
  variants:
    speculative_decode: "suffix"             # 白名单无 spec → 地板（修原误产 deepseek_mtp bug）
    sparse_kv: "indexcache_topk8"            # Ascend GLM-5.1 tmp scope
    kv_offload: null
  logs:
    - 含 "spec not in whitelist → suffix floor"
```

---

### WL-12 · vllm_ascend / glm-5.2 / 910b(c) → {spec}

```yaml
入参:
  user_cli:
    model-name: "glm-5.2"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENGINE_VERSION: "0.21.0-a3"
  model_config:
    architecture: "GlmMoeDsaForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"
    - 含 "method" "deepseek_mtp"             # GlmMoeDsa 架构
    - 不含 "--hf-overrides"                  # sparse 不在白名单
    - 不含 "ENABLE_KV_OFFLOAD=true"          # offload 不在白名单
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: "deepseek_mtp"       # ✅ 无 offload → mtp 保留
    sparse_kv: null
    kv_offload: null
```

---

## 第二部分 · 非白名单 + PD（默认路径 / 最高优先级）

### N-01 · 非白名单 NV（Qwen2.5-72B），三开关全开

```yaml
入参:
  user_cli:
    model-name: "Qwen2.5-72B"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
  model_config:
    architecture: "Qwen2ForCausalLM"

期望出参:
  command:
    - 含 "--speculative-config"              # suffix 地板合成
    - 不含 "--hf-overrides"                  # sparse 收口关
    - 不含 "ENABLE_KV_OFFLOAD=true"          # offload 收口关
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: "suffix"             # 白名单 miss → suffix 地板
    sparse_kv: null
    kv_offload: null
  logs:
    - 含 "sparse requested but not in whitelist → suppressed"
    - 含 "offload requested but not in whitelist → ENABLE_KV_OFFLOAD=false"
```

### N-02 · 非白名单 Ascend（Qwen3-30B-A3B），未设三开关

```yaml
入参:
  user_cli:
    model-name: "Qwen3-30B-A3B"
    engine: "vllm_ascend"
    model-path: "/usr/local/serving/models/"
    port: 18000
    distributed: true
    seed: 42
    trust-remote-code: true
    dtype: "auto"
    output-length: 2048
    enable-prefix-caching: true
    max-num-batched-tokens: 4096
    max-num-seqs: 256
    kv-cache-dtype: "auto"
    block-size: 16
    gpu-memory-utilization: 0.95
    input-length: 2048
    enable-chunked-prefill: true
    gpu-usage-mode: "full"
    device-count: 4
  orchestration_env: {}                      # 无三开关 env
  model_config:
    architecture: "Qwen3ForCausalLM"

期望出参:
  command:
    - 不含 "--speculative-config"            # 开关未置 true → 不合成
    - 不含 "--hf-overrides"
    - 不含 "ENABLE_KV_OFFLOAD"
    - 不含 "KV_MEM_OFFLOAD_SIZE"
  features:
    speculative_decode: false
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: null
    sparse_kv: null
    kv_offload: null
```

### N-03 · 非白名单 Ascend（Qwen3-30B-A3B），三开关全开

```yaml
入参:
  user_cli:
    model-name: "Qwen3-30B-A3B"
    engine: "vllm_ascend"
    model-path: "/usr/local/serving/models/"
    port: 18000
    distributed: true
    seed: 42
    trust-remote-code: true
    dtype: "auto"
    output-length: 2048
    enable-prefix-caching: true
    max-num-batched-tokens: 4096
    max-num-seqs: 256
    kv-cache-dtype: "auto"
    block-size: 16
    gpu-memory-utilization: 0.95
    input-length: 2048
    enable-chunked-prefill: true
    gpu-usage-mode: "full"
    device-count: 4
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
  model_config:
    architecture: "Qwen3ForCausalLM"

期望出参:
  command:
    - 含 "--speculative-config"              # suffix 地板合成
    - 不含 "--hf-overrides"                  # 白名单 miss → 收口关
    - 不含 "ENABLE_KV_OFFLOAD=true"
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: "suffix"
    sparse_kv: null
    kv_offload: null
```

### PD-01 · PD_ROLE=P 一票否决（glm-4.7 NV + 三开关全开）

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
    distributed: true
  orchestration_env:
    PD_ROLE: "P"                             # ★ PD 角色 → C14 第一判直接全关
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 不含 "--speculative-config"            # PD 全关
    - 不含 "--hf-overrides"
    - 不含 "ENABLE_KV_OFFLOAD=true"
    - 不含 "lmcache"                         # 无 LMCache MultiConnector
    - 含 "pd_role" 或 "rpc-port"            # 仅保留 PD connector
  features:
    speculative_decode: false
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: null
    sparse_kv: null
    kv_offload: null
  logs:
    - 含 "PD veto"                           # PD 一票否决日志
```

---

## 第三部分 · 反面验证：模型命中白名单，但用户关闭部分开关（§0 裁定1：无 forced）

> **核心口径**：`有效使能 = 开关 on AND 白名单命中`。开关 off → 白名单命中也不产。
> 白名单**只收窄、永不强开**（§0 裁定1）。删除 forced 列与所有 day0 强制开。
>
> 反面矩阵：对每种白名单行为模式，枚举「命中但用户关某开关」的场景，验证不强制复活。

### 反面覆盖矩阵（按行为模式 × 关开关组合）

```
图例: S=spec, P=sparse, O=offload  大写=开关开  小写=开关关
      ✅=特性产出(不因关开关受影响)  ✗=特性不产(用户关)  ❗=间接影响
      —=白名单无此特性,始终不产
```

#### 模式A · {spec,sparse,offload} 三全 + offload抢占spec（行2:glm-4.7 NV, 行4:minimax-m2.7 NV）

| 用例 | vs WL | 开关 | spec | sparse | offload | 说明 |
|------|-------|------|------|--------|---------|------|
| OFF-A1 | WL-02 | `s_P_O` | **✗关** | ✅ fp8 | ✅ lmcache_cpu+auto | spec关→不触发降级，offload/sparse独立产 |
| OFF-A2 | WL-02 | `S_p_O` | ✅ **suffix❗** | **✗关** | ✅ lmcache_cpu+auto | sparse关→spec仍被offload降级 |
| OFF-A3 | WL-02 | `S_P_o` | ✅ **glm4_moe_mtp✅救回** | ✅ fp8 | **✗关** | ★ offload关→spec不被降级！最反直觉 |
| OFF-A4 | WL-02 | `s_p_O` | **✗关** | **✗关** | ✅ lmcache_cpu+auto | 两关→仅offload |
| OFF-A5 | WL-02 | `s_P_o` | **✗关** | ✅ fp8 | **✗关** | spec关+offload关→仅sparse |
| OFF-A6 | WL-02 | `S_p_o` | ✅ **glm4_moe_mtp✅** | **✗关** | **✗关** | 仅spec开→无offload在场→mtp保留 |
| OFF-A7 | WL-02 | `s_p_o` | **✗关** | **✗关** | **✗关** | 三开关全关→全off |

#### 模式B · {spec,sparse,offload} 三全 + V4豁免共存（行5:V4-Flash NV, 行10:V4-Flash Ascend）

| 用例 | vs WL | 开关 | spec | sparse | offload | 说明 |
|------|-------|------|------|--------|---------|------|
| OFF-B1 | WL-10 | `s_P_O` | **✗关** | ✅ noop | ✅ native_cpu_connector | V4 Ascend关spec→sparse+offload仍产 |
| OFF-B2 | WL-05 | `S_p_O` | ✅ deepseek_mtp✅ | **✗关** | ✅ native_kv_offloading_backend | V4 NV关sparse→spec+offload仍共存 |
| OFF-B3 | WL-10 | `S_P_o` | ✅ deepseek_mtp✅ | ✅ noop | **✗关** | V4 Ascend关offload→spec+sparse仍产 |
| OFF-B4 | WL-05 | `S_p_o` | ✅ deepseek_mtp✅ | **✗关** | **✗关** | 仅spec开→mtp保留 |
| OFF-B5 | WL-05 | `s_p_o` | **✗关** | **✗关** | **✗关** | 三开关全关→全off |

#### 模式C · {spec,sparse} 双特性（行1:qwen3.5-397b NV）

| 用例 | vs WL | 开关 | spec | sparse | offload | 说明 |
|------|-------|------|------|--------|---------|------|
| OFF-C1 | WL-01 | `s_P_o` | **✗关** | ✅ fp8 | —(收口关) | 关spec→仅sparse产 |
| OFF-C2 | WL-01 | `S_p_o` | ✅ qwen3_5_mtp✅ | **✗关** | —(收口关) | 关sparse→仅spec产，无offload→mtp保留 |
| OFF-C3 | WL-01 | `s_p_o` | **✗关** | **✗关** | —(收口关) | 全关→全off |

#### 模式D · {sparse} 单特性（行3:glm-5.1 NV, 行11:glm-5.1 Ascend）

| 用例 | vs WL | 开关 | spec | sparse | offload | 说明 |
|------|-------|------|------|--------|---------|------|
| OFF-D1 | WL-11 | `s_p_o` | **✗关**(开关关→不合成) | **✗关** | —(收口关) | Ascend: day0 forced已删→关即关 |
| OFF-D2 | WL-03 | `S_p_o` | ✅ suffix(地板) | **✗关** | —(收口关) | NV: spec开关开但白名单无→地板,sparse关 |
| OFF-D3 | WL-11 | `s_P_o` | **✗关** | ✅ indexcache_topk8 | —(收口关) | Ascend: 关spec→sparse独立产 |

#### 模式E · {spec,offload} 双特性 + offload抢占spec（行6:glm-4.7 Asc, 行7:minimax-m2.5 Asc, 行8:v3.2 Asc）

| 用例 | vs WL | 开关 | spec | sparse | offload | 说明 |
|------|-------|------|------|--------|---------|------|
| OFF-E1 | WL-08 | `s_p_O` | **✗关** | —(收口关) | ✅ lmcache_cpu+auto | 关spec→仅offload产 |
| OFF-E2 | WL-08 | `S_p_o` | ✅ **mtp✅救回** | —(收口关) | **✗关** | ★ 关offload→spec救回mtp |
| OFF-E3 | WL-06 | `S_p_o` | ✅ **mtp✅救回** | —(收口关) | **✗关** | 同OFF-E2, glm-4.7 Ascend |
| OFF-E4 | WL-08 | `s_p_o` | **✗关** | —(收口关) | **✗关** | 全关→全off |

#### 模式F · {spec} 单特性（行9:qwen3.6 Ascend 910b/910c, 行12:glm-5.2 Ascend）

| 用例 | vs WL | 开关 | spec | sparse | offload | 说明 |
|------|-------|------|------|--------|---------|------|
| OFF-F1 | WL-12 | `s_p_o` | **✗关** | —(收口关) | —(收口关) | glm-5.2关spec→全off |
| OFF-F2 | WL-09 | `s_p_o` | **✗关** | —(收口关) | —(收口关) | qwen3.6关spec→全off |
| OFF-F3 | WL-09 | `S_p_o` | ✅ qwen3_5_mtp✅ | —(收口关) | —(收口关) | 仅spec开→mtp保留 |

---

### ★ 关键反面用例详情（入参三段 + 期望出参）

> 以下选取交叉最多的代表用例展开完整 YAML，其余用例参照上表。

#### OFF-A3 · 模式A 关 offload → spec 救回 mtp ★★

```yaml
# 对照 WL-02：三开关全开→spec=suffix(降级)
# 本用例：仅关 offload → spec 从 suffix 救回 glm4_moe_mtp
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "false"               # ★ 用户显式关 offload
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 含 "--speculative-config"
    - 含 "method" "glm4_moe_mtp"             # ★ offload关→mtp保留
    - 含 "kv_cache_dtype=fp8"                # sparse 仍产
    - 不含 "ENABLE_KV_OFFLOAD=true"
    - 不含 "KV_MEM_OFFLOAD_SIZE"
  features:
    speculative_decode: true
    sparse_kv: true
    kv_offload: false                         # 用户关
  variants:
    speculative_decode: "glm4_moe_mtp"        # ★ 救回！
    sparse_kv: "fp8"
    kv_offload: null
```

#### OFF-A1 · 模式A 关 spec → spec 不产，sparse+offload 仍产

```yaml
# 对照 WL-02：三开关全开→spec=suffix(降级), sparse=fp8, offload=lmcache_cpu+auto
# 本用例：关 spec → spec 不产（不触发降级），sparse+offload 独立产出
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "false"        # ★ 用户关 spec
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 不含 "--speculative-config"             # spec 关 → 不合成
    - 含 "kv_cache_dtype=fp8"                # sparse 仍产
    - 含 "export KV_MEM_OFFLOAD_SIZE="       # offload 仍产
  features:
    speculative_decode: false                 # 用户关
    sparse_kv: true
    kv_offload: true
  variants:
    speculative_decode: null
    sparse_kv: "fp8"
    kv_offload: "lmcache_cpu+auto"
```

#### OFF-A6 · 模式A 关 sparse+offload，仅 spec → mtp 保留

```yaml
# 仅 spec 开 → 无 offload 在场 → mtp 不降级
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "false"
    ENABLE_KV_OFFLOAD: "false"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 含 "--speculative-config"
    - 含 "method" "glm4_moe_mtp"
    - 不含 "--hf-overrides"
    - 不含 "ENABLE_KV_OFFLOAD"
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: "glm4_moe_mtp"
    sparse_kv: null
    kv_offload: null
```

#### OFF-A7 · 模式A 三开关全关 → 全 off

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "false"
    ENABLE_SPARSE: "false"
    ENABLE_KV_OFFLOAD: "false"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 不含 "--speculative-config"
    - 不含 "--hf-overrides"
    - 不含 "ENABLE_KV_OFFLOAD"
  features:
    speculative_decode: false
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: null
    sparse_kv: null
    kv_offload: null
```

---

#### OFF-B1 · 模式B V4 Ascend 关 spec → sparse+offload 仍产 ★

```yaml
# 对照 WL-10：三开关全开→spec=deepseek_mtp, sparse=noop, offload=native_cpu_connector
# 本用例：关 spec → spec=off, sparse+offload 仍独立产出
# ★ 验证 V4-Flash 适配层默认不得强制复活投机
入参:
  user_cli:
    model-name: "deepseek-v4-flash"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "false"        # ★ 用户关 spec
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
    WINGS_ASCEND_PLATFORM: "a3"
  model_config:
    architecture: "DeepseekV4ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 不含 "--speculative-config"             # spec 关 → 不合成
    - 含 "cpu_swap_space_gb"                  # offload 仍产
  features:
    speculative_decode: false
    sparse_kv: true
    kv_offload: true
  variants:
    speculative_decode: null
    sparse_kv: "noop"
    kv_offload: "native_cpu_connector"
```

#### OFF-B3 · 模式B V4 Ascend 关 offload → spec+sparse 仍产

```yaml
# V4 豁免下，关 offload 不会"救回"spec（因为 V4 下 spec 本就不被降级）
# 但需验证 offload 关后 spec+sparse 不受影响
入参:
  user_cli:
    model-name: "deepseek-v4-flash"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "false"               # ★ 用户关 offload
    WINGS_ASCEND_PLATFORM: "a3"
  model_config:
    architecture: "DeepseekV4ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"              # spec 仍产
    - 不含 "cpu_swap_space_gb"               # offload 关
  features:
    speculative_decode: true
    sparse_kv: true
    kv_offload: false
  variants:
    speculative_decode: "deepseek_mtp"        # ★ V4 豁免下 offload 关→mtp 仍在（本就共存）
    sparse_kv: "noop"
    kv_offload: null
```

---

#### OFF-C2 · 模式C qwen3.5-397b NV 关 sparse → 仅 spec（mtp 保留）

```yaml
# 对照 WL-01：三开关全开→spec=qwen3_5_mtp, sparse=fp8, offload=off(收口关)
# 关 sparse → spec 独立产，无 offload 在场 → mtp 保留
入参:
  user_cli:
    model-name: "qwen3.5-397b"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "false"                    # ★ 用户关 sparse
    ENABLE_KV_OFFLOAD: "true"                 # 全量下发true，但白名单无→收口关
  model_config:
    architecture: "Qwen3_5MoeForConditionalGeneration"

期望出参:
  command:
    - 含 "--speculative-config"
    - 不含 "--hf-overrides"                   # sparse 关
    - 不含 "ENABLE_KV_OFFLOAD=true"           # 白名单无 → 收口关
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: "qwen3_5_mtp"
    sparse_kv: null
    kv_offload: null
```

---

#### OFF-E2 · 模式E deepseek-v3.2 Ascend 关 offload → spec 救回 mtp ★

```yaml
# 对照 WL-08：三开关全开→spec=suffix(降级), offload=lmcache_cpu+auto
# 关 offload → spec=mtp(救回)，与 OFF-A3 同型交互
入参:
  user_cli:
    model-name: "deepseek-v3.2"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"                     # 全量下发true，白名单无→收口关
    ENABLE_KV_OFFLOAD: "false"               # ★ 用户关 offload
    ENGINE_VERSION: "0.21.0-a3"
  model_config:
    architecture: "DeepseekV32ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"
    - 不含 "--hf-overrides"                   # sparse 白名单无
    - 不含 "ENABLE_KV_OFFLOAD=true"
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: "mtp"                 # ★ offload 关 → mtp 救回！（对照 WL-08=suffix）
    sparse_kv: null
    kv_offload: null
```

---

#### OFF-D1 · 模式D GLM-5.1 Ascend 关 sparse → day0 forced 已删 ★

```yaml
# 对照 WL-11：三开关全开→spec=suffix(地板), sparse=indexcache_topk8
# 关 sparse → sparse=off（day0 _force_kv_sparse 已删，关即关）
入参:
  user_cli:
    model-name: "glm-5.1"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "false"                    # ★ 用户关 sparse
    ENABLE_KV_OFFLOAD: "true"
    WINGS_ASCEND_PLATFORM: "a2"
  model_config:
    architecture: "GlmMoeDsaForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"              # suffix 地板（spec 开关开但白名单无）
    - 不含 "--hf-overrides"                  # ★ sparse 关 → 不产（day0 forced 已删）
    - 不含 "ENABLE_KV_OFFLOAD=true"          # 白名单无 offload
  features:
    speculative_decode: true                  # suffix 地板也是投机
    sparse_kv: false                          # ★ 用户关 → false
    kv_offload: false
  variants:
    speculative_decode: "suffix"
    sparse_kv: null
    kv_offload: null
```

#### OFF-F1 · 模式F glm-5.2 Ascend 关 spec → 全 off

```yaml
# 对照 WL-12：三开关全开→spec=deepseek_mtp
# 关 spec → spec=off（白名单仅 spec，关后全 off）
入参:
  user_cli:
    model-name: "glm-5.2"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "false"        # ★ 用户关 spec
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    ENGINE_VERSION: "0.21.0-a3"
  model_config:
    architecture: "GlmMoeDsaForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 不含 "--speculative-config"             # spec 关
    - 不含 "--hf-overrides"                   # 白名单无 sparse
    - 不含 "ENABLE_KV_OFFLOAD=true"           # 白名单无 offload
  features:
    speculative_decode: false
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: null
    sparse_kv: null
    kv_offload: null
```

---

### 反面用例汇总

| 模式 | 白名单 | 代表模型 | 全开参照 | 反面用例 | 覆盖的关开关组合 |
|------|--------|---------|---------|---------|----------------|
| A | {S,P,O}+抢占 | glm-4.7 NV | WL-02 | OFF-A1~A7 | 关spec/关sparse/关offload/关spec+sparse/关spec+offload/关sparse+offload/全关 |
| B | {S,P,O}+豁免 | V4-Flash Ascend | WL-10 | OFF-B1,B3 | 关spec/关offload |
| B | {S,P,O}+豁免 | V4-Flash NV | WL-05 | OFF-B2,B4 | 关sparse/关sparse+offload |
| C | {S,P} | qwen3.5-397b NV | WL-01 | OFF-C2 | 关sparse |
| D | {P} | glm-5.1 Ascend | WL-11 | OFF-D1 | 关sparse ★day0 forced已删 |
| E | {S,O}+抢占 | v3.2 Ascend | WL-08 | OFF-E2 | 关offload→mtp救回 |
| F | {S} | glm-5.2 Ascend | WL-12 | OFF-F1 | 关spec→全off |

> 矩阵中其余组合（如 OFF-A2/A4/A5、OFF-C1/C3、OFF-D2/D3、OFF-E1/E3/E4、OFF-F2/F3）入参/出参可从上表推导，不再逐一展开完整 YAML。

---

## 第四部分 · offload 子变量 / SPARSE_LEVEL / C4 容量 / 卡型边界

### KV-01 · lmcache_cpu+auto（C4 自算容量，512G/8卡）

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 含 "KV_MEM_OFFLOAD_SIZE=50"            # M_offload=512-59-51=402, 均卡=402÷8≈50
  variants:
    kv_offload: "lmcache_cpu+auto"
```

### KV-02 · lmcache_cpu+custom（透传 200G）

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "200"               # GB 数值字符串 → custom 透传
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 含 "KV_MEM_OFFLOAD_SIZE=200"           # 透传，不计算
  variants:
    kv_offload: "lmcache_cpu+custom"
```

### KV-03 · lmcache_cpu_disk（CPU+Disk 分层）

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "200"
    ENABLE_KV_DISK_OFFLOAD: "true"
    KV_DISK_OFFLOAD_PATH: "/mnt/kv_cache"
    KV_DISK_OFFLOAD_SIZE: "500"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 含 "KV_MEM_OFFLOAD_SIZE=200"
    - 含 "KV_DISK_OFFLOAD_PATH=/mnt/kv_cache"
    - 含 "KV_DISK_OFFLOAD_SIZE=500"
  variants:
    kv_offload: "lmcache_cpu_disk+custom"
```

### KV-04 · lmcache_cpu_disk+qat（+QAT 硬件压缩）

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "200"
    ENABLE_KV_DISK_OFFLOAD: "true"
    KV_DISK_OFFLOAD_PATH: "/mnt/kv_cache"
    KV_DISK_OFFLOAD_SIZE: "500"
    ENABLE_KV_QAT: "true"
    KV_QAT_COMPRESS_LEVEL: "4"
    KV_QAT_INSTANCE_NUM: "2"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 含 "QAT_COMPRESS_LEVEL=4"
    - 含 "QAT_INSTANCE_NUM=2"
  variants:
    kv_offload: "lmcache_cpu_disk+custom+qat"
```

### KV-05 · L1=false 门控 L2（层级门控关键）

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_KV_OFFLOAD: "false"               # ★ L1=false
    ENABLE_KV_MEM_OFFLOAD: "true"            # L2=true 但 L1=false → 忽略
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 不含 "KV_MEM_OFFLOAD_SIZE"             # L1=false → 不读 L2 子变量
    - 不含 "ENABLE_KV_OFFLOAD=true"
  features:
    kv_offload: false
```

### KV-06 · C4 auto 熔断（100G 不足）

```yaml
入参:
  user_cli:
    model-name: "glm-4.7"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "100"            # M_offload=100-59-10=31 < 100(下限)
  model_config:
    architecture: "Glm4MoeForCausalLM"

期望出参:
  command:
    - 不含 "KV_MEM_OFFLOAD_SIZE"             # 熔断 → 不写回容量
  logs:
    - 含 "offload capacity"                   # 熔断告警
```

### KV-07 · native Ascend auto（CPUOffloadingConnector，整节点不除卡数）★

```yaml
# ★ LMCache 与 native 共用 M_offload 公式，但落地单位不同：
#   LMCache:    KV_MEM_OFFLOAD_SIZE = M_offload ÷ N_card  (均卡, KV-01 为 50G)
#   native:     cpu_swap_space_gb   = M_offload           (整节点, 不除卡数)
#   M_offload 具体值取决于实际 TP×DP（非 device-count），此处断言模式不硬编码数值
入参:
  user_cli:
    model-name: "deepseek-v4-flash"
    engine: "vllm_ascend"
    device-count: 8
    distributed: true
  orchestration_env:
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
    WINGS_ASCEND_PLATFORM: "a3"
  model_config:
    architecture: "DeepseekV4ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "cpu_swap_space_gb"                  # native CPUOffloadingConnector（整节点，不除卡数）
    - 不含 "KV_MEM_OFFLOAD_SIZE"              # native 不写 LMCache env
  variants:
    kv_offload: "native_cpu_connector"
```

### KV-08 · native NV auto（--kv_offloading_size，整节点不除卡数）★

```yaml
# NV V4-Flash native 卸载，同样 M_offload 整节点不除；M_offload 具体值取决于实际 TP
入参:
  user_cli:
    model-name: "deepseek-v4-flash"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_KV_OFFLOAD: "true"
    ENABLE_KV_MEM_OFFLOAD: "true"
    KV_MEM_OFFLOAD_SIZE: "auto"
    AVAILABLE_POD_MEM_SIZE: "512"
  model_config:
    architecture: "DeepseekV4ForCausalLM"

期望出参:
  command:
    - 含 "kv_offloading_backend native"       # native NV 后端 CLI 标志
    - 含 "kv_offloading_size"                 # 整节点 M_offload（不除卡数）
    - 不含 "KV_MEM_OFFLOAD_SIZE"              # native 不写 LMCache env
    - 不含 "export ENABLE_KV_MEM_OFFLOAD"     # L2 不走 LMCache env 导出
  variants:
    kv_offload: "native_kv_offloading_backend"
```

---

### SP-01 · V4-Flash NV SPARSE_LEVEL=performance_first → topk8

```yaml
入参:
  user_cli:
    model-name: "deepseek-v4-flash"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPARSE: "true"
    SPARSE_LEVEL: "performance_first"
  model_config:
    architecture: "DeepseekV4ForCausalLM"

期望出参:
  command:
    - 含 "index_topk_freq:8"                 # perf → topk8（sparse 表已声明）
  variants:
    sparse_kv: "indexcache_use_index_cache_topk8"
```

### SP-02 · glm-5.1 NV SPARSE_LEVEL=performance_first → 回退 accuracy

```yaml
入参:
  user_cli:
    model-name: "glm-5.1"
    engine: "vllm"
    device-count: 8
  orchestration_env:
    ENABLE_SPARSE: "true"
    SPARSE_LEVEL: "performance_first"
  model_config:
    architecture: "GlmMoeDsaForCausalLM"

期望出参:
  command:
    - 含 "index_topk_freq:4"                 # GLM-5.1 未声明 perf topk → 回退 accuracy topk4
  variants:
    sparse_kv: "indexcache_topk4"
  logs:
    - 含 "performance_first" "fallback"      # 回退告警
```

---

### CARD-01 · Ascend details:[] → card_token='' → 白名单全 miss ★

```yaml
# hardware_info.json = {"count":16,"details":[],"units":"GB","device":"ascend"}
入参:
  user_cli:
    model-name: "glm-5.2"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENABLE_SPARSE: "true"
    ENABLE_KV_OFFLOAD: "true"
    # 无 ENGINE_VERSION、无 WINGS_DEVICE_NAME、无 WINGS_ASCEND_PLATFORM
    # → resolve_card_token → '' → Ascend 白名单全行 910b/910c 不匹配 → miss
  model_config:
    architecture: "GlmMoeDsaForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  command:
    - 含 "--speculative-config"              # suffix 地板合成
    - 不含 "--hf-overrides"                  # sparse miss
    - 不含 "ENABLE_KV_OFFLOAD=true"          # offload miss
  features:
    speculative_decode: true
    sparse_kv: false
    kv_offload: false
  variants:
    speculative_decode: "suffix"             # glm-5.2 本应 deepseek_mtp，但卡型 miss → 地板
  logs:
    - 含 "card_token unresolved on Ascend"   # 卡型告警
```

### CARD-02 · deepseek-v3.2 Ascend ENGINE_VERSION=a2 → 910b → 白名单 miss

```yaml
入参:
  user_cli:
    model-name: "deepseek-v3.2"
    engine: "vllm_ascend"
    device-count: 16
    distributed: true
  orchestration_env:
    ENABLE_SPECULATIVE_DECODE: "true"
    ENGINE_VERSION: "0.21.0-a2"              # 兜底 910b，v3.2 白名单仅 910c → miss
  model_config:
    architecture: "DeepseekV32ForCausalLM"
    quantization_config: {quant_method: "ascend"}

期望出参:
  variants:
    speculative_decode: "suffix"             # 910b 不命中 → 地板（对照 WL-08 的 910c→suffix降级）
```

---

## 覆盖汇总

### 白名单 12 行全覆盖（三开关全开 → 白名单过滤）

| # | engine | 模型 | 卡 | 白名单 | spec variant | sparse variant | offload variant | 用例 | 关键交互 |
|---|--------|------|----|--------|-------------|---------------|-----------------|------|---------|
| 1 | vllm | qwen3.5-397b | * | spec,sparse | qwen3_5_mtp | fp8 | — | WL-01 | 无offload→mtp保留 |
| 2 | vllm | glm-4.7 | * | spec,sparse,offload | **suffix** ❗ | fp8 | lmcache_cpu+auto | WL-02 | offload抢占spec |
| 3 | vllm | glm-5.1 | * | sparse | **suffix** | indexcache_topk4 | — | WL-03 | spec地板(修bug) |
| 4 | vllm | minimax-m2.7 | * | spec,sparse,offload | suffix | fp8 | lmcache_cpu+auto | WL-04 | arch无mtp |
| 5 | vllm | deepseek-v4-flash | * | spec,sparse,offload | deepseek_mtp ✅ | indexcache_use_index_cache_topk4 | native_kv_offloading_backend | WL-05 | V4豁免共存 |
| 6 | vllm_ascend | glm-4.7 | 910b/c | spec,offload | **suffix** ❗ | — | lmcache_cpu+auto | WL-06 | offload抢占spec |
| 7 | vllm_ascend | minimax-m2.5 | 910b/c | spec,offload | suffix | — | lmcache_cpu+auto | WL-07 | arch无mtp |
| 8 | vllm_ascend | deepseek-v3.2 | 910c | spec,offload | **suffix** ❗ | — | lmcache_cpu+auto | WL-08 | offload抢占spec |
| 9 | vllm_ascend | qwen3.6 | 910b/910c | spec | qwen3_5_mtp ✅ | — | — | WL-09 | 无offload→mtp保留 |
| 10 | vllm_ascend | deepseek-v4-flash | 910b/c | spec,sparse,offload | deepseek_mtp ✅ | noop | native_cpu_connector | WL-10 | V4豁免共存 |
| 11 | vllm_ascend | glm-5.1 | 910b/c | sparse | **suffix** | indexcache_topk8 | — | WL-11 | spec地板(修bug) |
| 12 | vllm_ascend | glm-5.2 | 910b/c | spec | deepseek_mtp ✅ | — | — | WL-12 | 无offload→mtp保留 |

### 专项覆盖

| 类别 | 用例 | 验证点 |
|------|------|--------|
| 非白名单 NV（三开关全开） | N-01 | miss → 全关，仅 suffix 地板 |
| 非白名单 Ascend（无开关） | N-02 | 原始 test.md 用例保留 |
| 非白名单 Ascend（三开关全开） | N-03 | 全开仍 miss → 仅 suffix |
| PD 一票否决 | PD-01 | 最高优先级 force-OFF |
| **反面·模式A** 关 offload | OFF-A3 | offload 关→spec 救回 mtp ★ |
| **反面·模式A** 关 spec | OFF-A1 | spec 关→sparse+offload 仍产 |
| **反面·模式A** 仅 spec 开 | OFF-A6 | 关 sparse+offload→仅 spec mtp |
| **反面·模式A** 全关 | OFF-A7 | 三开关全关→全 off |
| **反面·模式B** V4 Ascend 关 spec | OFF-B1 | spec 关→适配层默认不得复活 ★ |
| **反面·模式B** V4 Ascend 关 offload | OFF-B3 | offload 关→spec+sparse 仍产 |
| **反面·模式C** 关 sparse | OFF-C2 | sparse 关→仅 spec mtp 保留 |
| **反面·模式D** Ascend 关 sparse | OFF-D1 | day0 forced 已删→关即关 ★ |
| **反面·模式E** 关 offload | OFF-E2 | offload 关→spec 救回 mtp ★ |
| **反面·模式F** 关 spec | OFF-F1 | 仅有的 spec 关→全 off |
| offload L2 auto/custom/disk/qat/分层 | KV-01~04 | 卸载子变量全覆盖 |
| L1 门控 L2 | KV-05 | L1=false → L2 不读 |
| C4 熔断 | KV-06 | 容量不足 → 告警不写回 |
| SPARSE_LEVEL perf topk8 | SP-01 | V4-Flash NV per-row perf |
| SPARSE_LEVEL 回退 | SP-02 | GLM-5.1 未声明 perf → accuracy |
| 卡型 details:[] miss | CARD-01 | Ascend 白名单全失效 |
| ENGINE_VERSION 兜底 a2 vs a3 | CARD-02 | 910b 不命中 v3.2 |

### 反面覆盖矩阵（6 行为模式 × 关开关组合）

| 模式 | 代表模型 | 白名单 | 反面用例数 | 覆盖的关开关组合 |
|------|---------|--------|-----------|----------------|
| A | glm-4.7 NV | {S,P,O}+抢占 | 7 (OFF-A1~A7) | 关S/关P/关O/关SP/关SO/关PO/全关 |
| B | V4-Flash Ascend | {S,P,O}+豁免 | 2 (OFF-B1,B3) | 关S/关O |
| B | V4-Flash NV | {S,P,O}+豁免 | 2 (OFF-B2,B4) | 关P/关PO |
| C | qwen3.5-397b NV | {S,P} | 1 (OFF-C2) | 关P |
| D | glm-5.1 Ascend | {P} | 1 (OFF-D1) | 关P ★day0 forced |
| E | v3.2 Ascend | {S,O}+抢占 | 1 (OFF-E2) | 关O→mtp救回 |
| F | glm-5.2 Ascend | {S} | 1 (OFF-F1) | 关S→全off |

### 总计

| 维度 | 数量 |
| --- | --- |
| 白名单 12 行全开关开 | 12 用例 |
| 非白名单 / PD | 4 用例 |
| **反面·命中白名单但用户关开关** | **15 用例**（7 展开完整 YAML + 8 矩阵推导） |
| offload 子变量 / 门控 | 6 用例 |
| SPARSE_LEVEL | 2 用例 |
| 卡型边界 | 2 用例 |
| **合计** | **41 用例** |
| 三类关键交互（❗抢占 / ✅豁免 / 地板） | 全覆盖 |
| 反面关键交互（关 offload→救回 mtp / 关开关→不强制复活） | 全覆盖 |
