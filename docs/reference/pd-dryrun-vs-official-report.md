# PD 分离 dry-run 下发字段 vs 官方 —— 核对报告（问题 · 原因 · 修复）

> **状态（2026-06-13 已修复并验证）**：R1–R4 全部修复，GLM-5 / V4-Flash 两个 dry-run 场景的下发字段已逐项对齐官方。修复 = `pd_config.json` 数据 + 一处 PD 专属代码改动（让注册表成为 PD external-lb 的唯一真相源）。详见 §0。`tests/pd_external_lb_verify.py` 45/45 PASS（含非 PD 回归）；非 PD 部署字节级不变。
>
> ## §0 修复实施记录
>
> **数据（`wings_control/config/defaults/pd_config.json`）**
> - `GlmMoeDsaForCausalLM`：max_model_len 拆角色级（P=131072/D=200000）；P 加 `enforce_eager`、关 `enable_prefix_caching`、`compilation_config=null`（删 base 图捕获）；D 关 prefix/chunked、补 `cudagraph_capture_sizes`；common 补 `seed/enable_auto_tool_choice/tool_call_parser=glm47/reasoning_parser=glm45`；P/D `additional_config` 补 `recompute_scheduler_enable`、P 补 `enable_dsa_cp`+`layer_sharding`。
> - `DeepseekV4ForCausalLM`：批量/显存/seed/max_model_len/speculative 对齐 A3（P 8192/16/0.9、D 120/60/0.9、seed1024、max_model_len 1048576、spec `{1,mtp,enforce_eager}`）；P 加 `enforce_eager`、关 `async_scheduling`/`compilation_config=null`；common 补 `no_enable_prefix_caching`、`enable_chunked_prefill=false`、`reasoning_parser`、`model_loader_extra_config`、`no_disable_hybrid_kv_cache_manager`；P/D `additional_config` 对齐官方。
> - `Qwen3_5MoeForConditionalGeneration`：静态补齐 `max_model_len/no_enable_prefix_caching/speculative_config(qwen3_5_mtp)/additional_config`（无 dry-run 场景，未经命令验证）。
>
> **代码（两处，均严格门控到 PD external-lb，非 PD 路径不变）**
> - `core/config_loader.py::_apply_pd_external_lb`：把注册表 engine 覆盖**深拷贝**后透传为 `cmd_known_params["_pd_engine_overrides"]`（深拷贝同时修掉了「模型默认注入器就地改动模块级缓存 `_PD_CONFIG_CACHE`」的潜在 bug）。
> - `engines/vllm_adapter.py::_prepare_engine_config`：在所有模型默认注入器之后**重申**注册表覆盖（`None` 值表示删除该 base 键），使 pd_config 成为 PD external-lb 引擎参数的唯一真相源，压住 `_force_set_*` / `_merge_dict_default_*` 对注册表值的回填。
>
> **根因订正**：原报告假设「`_apply_pd_external_lb` 直接覆盖即为最终值」。实际 `_prepare_engine_config`（命令构建阶段，晚于注册表）会用 `_force_set_*` 回填 `enable_prefix_caching`/`compilation_config`/`max_model_len` 等，且 `additional_config` 因共享引用被就地深合并——故必须「深拷贝 + 注入器后重申」才能真正生效。
>
> **遗留（仍需真机确认，未改）**：`engine_id`——V4-Flash(Hybrid) 官方示例固定 `0/1`，wings 仍按 dp_rank 注入（多 service 下更合理，待真机验证 Mooncake Hybrid 期望）。

| 项 | 内容 |
|----|------|
| 日期 | 2026-06-13 |
| 工具 | `dry_run.py --pd <scenario>` → `build/output/start_command_pd-*.sh` |
| 配置源 | `wings_control/config/defaults/pd_config.json` + 各模型 base 默认 config |
| 官方基准 | vLLM-Ascend tutorials（zh-cn/latest，经 WebFetch 抽取，落地前建议再逐字核对网页精确 JSON） |
| 约束 | **不动任何 Python 代码**；修复均落在数据文件 `pd_config.json`（设计文档 §7.6：新增/调整 = 改 JSON，零代码）。少数项需代码/真机确认，单列 §七。 |

> dry_run 内仅注册了 `glm5`、`v4flash` 两个 PD 场景，故只有这两个能「生成命令」逐字对比；`DeepSeek-V3.2` / `Qwen3.5-397B` / `Qwen3-30B` 无 PD 场景，只能对 `pd_config.json` 注册值做**静态**核对（§5）。

---

## 一、一句话结论

两个能跑出命令的 PD 模型（GLM-5、V4-Flash）**都与官方存在多处字段偏差**；**约 90% 的偏差是同一个根因**：生成命令是「**base 单实例默认 config** 被 `pd_config.json` 少量覆盖」得到的，凡 pd_config 没显式写的键，全部由 base 默认泄漏进来 —— 而 base 是「decode/单实例风味」，套到 PD 的 **Prefill 角色就错了**。

```
官方 PD 命令   =  为 P/D 各自手写的完整 recipe
wings 生成命令 =  base(单实例默认)  ⊕  pd_config 覆盖的少数键
                  └────────────┬───────────┘
                   没被覆盖的键 = base 泄漏 = 偏差来源
```

---

## 二、根因总览（5 类）

| # | 根因 | 触发的问题 | 为什么会这样 |
|---|------|-----------|-------------|
| **R1** | **Prefill 继承了 base 的 decode 风味开关** | P 出现 `--compilation-config FULL_DECODE_ONLY`、`--enable-prefix-caching`、`--async-scheduling`，且**缺 `--enforce-eager`** | pd_config 的 `prefill.engine` 只「setdefault 补值」、不会删 base 已有的键 |
| **R2** | **pd_config 数值是「摘要待核」占位值，未对齐官方** | `max-num-batched-tokens`/`max-num-seqs`/`gpu-memory-utilization` 与官方不同（V4-Flash 尤甚） | 注册表 `_comment` 已自标「部分经摘要，落地前需逐项核对」 |
| **R3** | **角色相关键被放进了 `common`** | `max-model-len` 一个值套 P/D，但官方 P/D 不同（GLM5 P=131072 / D=200000） | 建表时未区分该键 P/D 不同 |
| **R4** | **pd_config 漏写官方有的键** | 缺 `reasoning-parser`/`tool-call-parser`/`model-loader-extra-config`/`no-enable-prefix-caching`/`additional-config` 子键/`speculative` 的 method·enforce | 条目不完整、base 也没有 → 直接缺失 |
| **R5** | **wings 既有 PD 规则与官方该模型相反** | V4-Flash：wings 强制 `--disable-hybrid-kv-cache-manager`，官方要 `--no-disable-...`；`engine_id` 注入策略与官方示例不同 | 通用 PD 规则未对个别模型放行（设计文档 §10.2 已标为真机验证项） |

> R1–R4 改 `pd_config.json` 即可解决；**R5 改 JSON 解决不了**，见 §七。

---

## 三、逐字段对比明细（证据）

拓扑(P/D 的 dp×tp)、连接器、kv_port 两模型均与官方**一致 ✅**，下表只列引擎下发字段。

### 3.1 GLM-5（`GlmMoeDsaForCausalLM`，MooncakeConnectorV1，P=DP2×TP16 / D=DP16×TP4）

**Prefill（producer）**

| 字段 | dry-run 下发 | 官方 | 判定 |
|------|-------------|------|------|
| tensor/data-parallel-size | 16 / 2 | 16 / 2 | ✅ |
| max-num-batched-tokens / seqs | 4096 / 64 | 4096 / 64 | ✅ |
| gpu-memory-utilization | 0.95 | 0.95 | ✅ |
| **max-model-len** | **200000** | **131072** | ❌ |
| **enforce-eager** | **无** | **有** | ❌ |
| **enable-prefix-caching** | **开** | **不开** | ❌ |
| **compilation-config** | **`{FULL_DECODE_ONLY}`** | **无** | ❌ |
| **enable-auto-tool-choice / tool-call-parser / reasoning-parser** | 无 / 无 / 无 | 有 / `glm47` / `glm45` | ❌ |
| **additional-config** | `{fuse_muls_add, multistream_overlap_shared_expert, ascend_compilation_config{enable_npugraph_ex}}` | 上述 + `recompute_scheduler_enable` + `enable_dsa_cp` + `layer_sharding:[q_b_proj,o_proj]` | ❌ |
| speculative-config / quant / seed / EP / chunked-prefill | `{3,deepseek_mtp}` / ascend / 1024 / 开 / 开 | 同 | ✅ |

**Decode（consumer）**

| 字段 | dry-run 下发 | 官方 | 判定 |
|------|-------------|------|------|
| tensor/data-parallel-size | 4 / 16 | 4 / 16 | ✅ |
| max-num-batched-tokens / seqs | 32 / 8 | 32 / 8 | ✅ |
| gpu-memory-utilization / max-model-len | 0.92 / 200000 | 0.92 / 200000 | ✅ |
| **compilation-config** | `{FULL_DECODE_ONLY}` | `{FULL_DECODE_ONLY, cudagraph_capture_sizes:[4,8,12,16,20,24,28,32]}` | ❌ |
| **enable-prefix-caching / enable-chunked-prefill** | 开 / 开 | 不开 / 不开 | ❌ |
| **enable-auto-tool-choice / tool-call-parser / reasoning-parser** | 无 / 无 / 无 | 有 / glm47 / glm45 | ❌ |
| **additional-config** | 缺 `recompute_scheduler_enable` | 含 `recompute_scheduler_enable` | ❌ |
| speculative-config | `{3,deepseek_mtp}` | 同 | ✅ |

**环境变量**：角色专属一致 ✅（P=`FLASHCOMM1+FUSED_MC2`，D=`MLAPO+TASK_QUEUE+FUSED_MC2`）；偏差 = `HCCL_BUFFSIZE=1024`（官方 256），缺共用项 `ASCEND_AGGREGATE_ENABLE/ACL_OP_INIT_MODE/ASCEND_A3_ENABLE/VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`；env 段有重复/抖动 export（`OMP_NUM_THREADS` 1→100→1 等），最终值以最后一次为准。

### 3.2 DeepSeek-V4-Flash（`DeepseekV4ForCausalLM`，MooncakeHybridConnector，对齐 A3 1P1D，P=DP4×TP4 / D=DP16×TP1）

> 偏差最严重；pd_config `_comment` 本就标注「部分经摘要，落地前需逐项核对」。

**Prefill（producer）**

| 字段 | dry-run 下发 | 官方(A3) | 判定 |
|------|-------------|---------|------|
| tensor/data-parallel-size | 4 / 4 | 4 / 4 | ✅ |
| **max-num-batched-tokens / seqs** | **4096 / 64** | **8192 / 16** | ❌ |
| **gpu-memory-utilization** | **0.95** | **0.9** | ❌ |
| **max-model-len** | **1024000** | **1048576** | ❌ |
| **seed** | **0** | **1024** | ❌ |
| **enforce-eager** | **无** | **有** | ❌ |
| **enable-prefix-caching** | **开** | **`--no-enable-prefix-caching`** | ❌ |
| **async-scheduling** | **开** | **无**（仅 decode 用） | ❌ |
| **compilation-config** | `{FULL_DECODE_ONLY}` | **无** | ❌ |
| **speculative-config** | `{3, deepseek_mtp}` | `{1, mtp, enforce_eager:true}` | ❌ |
| **additional-config** | `{enable_cpu_binding, multistream_overlap_shared_expert:false, multistream_dsa_preprocess:false, ascend_compilation_config{...}}` | `{enable_cpu_binding, enable_shared_expert_dp, enable_dsa_cp}` | ❌ |
| **reasoning-parser / model-loader-extra-config** | 无 / 无 | `deepseek_v4` / `{enable_multithread_load,num_threads:128}` | ❌ |
| **disable-hybrid-kv-cache-manager** | wings 强制 disable | 官方 `--no-disable-...`（保留） | ❌（语义相反） |
| tokenizer-mode / tool-call-parser / block-size / quant / safetensors | deepseek_v4 / deepseek_v4 / 128 / ascend / prefetch | 同 | ✅ |
| engine_id | 按 rank `$RANK` | 示例固定 `"0"` | ⚠️ 待确认 |

**Decode（consumer）**

| 字段 | dry-run 下发 | 官方(A3) | 判定 |
|------|-------------|---------|------|
| tensor/data-parallel-size | 1 / 16 | 1 / 16 | ✅ |
| **max-num-batched-tokens / seqs** | **32 / 8** | **120 / 60** | ❌ |
| **gpu-memory-utilization / max-model-len / seed** | 0.92 / 1024000 / 0 | 0.9 / 1048576 / 1024 | ❌ |
| **enable-prefix-caching** | 开 | `--no-enable-prefix-caching` | ❌ |
| **speculative-config** | `{3, deepseek_mtp}` | `{1, mtp, enforce_eager:true}` | ❌ |
| **additional-config** | `multistream_overlap_shared_expert:false`、含 `multistream_dsa_preprocess:false`、缺 `recompute_scheduler_enable` | `multistream_overlap_shared_expert:true` + `recompute_scheduler_enable` + `ascend_compilation_config{...}` + `enable_cpu_binding` | ❌ |
| **reasoning-parser / model-loader-extra-config** | 无 / 无 | deepseek_v4 / 有 | ❌ |
| **disable-hybrid-kv-cache-manager** | wings 强制 disable | 官方保留 | ❌ |
| compilation-config / async-scheduling / tokenizer-mode / block-size / quant | `{FULL_DECODE_ONLY}` / 开 / deepseek_v4 / 128 / ascend | 同 | ✅ |

---

## 四、逐问题清单（问题 / 原因 / 修复）

### 4.1 GLM-5

| # | 问题（下发 → 官方） | 原因 | 修复（pd_config.json） |
|---|--------------------|------|----------------------|
| G1 | P 缺 `--enforce-eager` | R1 | `prefill.engine` 加 `enforce_eager:true` |
| G2 | P 误开 `--enable-prefix-caching` | R1 | `prefill.engine` 加 `enable_prefix_caching:false` |
| G3 | P 误带 `--compilation-config FULL_DECODE_ONLY` | R1 | `prefill.engine` 加 `compilation_config:null` 清掉泄漏 |
| G4 | P `max-model-len 200000 → 131072` | R3 | `max_model_len` 移出 `common`，P=131072 / D=200000 |
| G5 | P/D 缺 `additional-config` 子键（`recompute_scheduler_enable`；P 另缺 `enable_dsa_cp`+`layer_sharding`） | R4 | 按官方补全 P、D 的 `additional_config` |
| G6 | D `compilation-config` 缺 `cudagraph_capture_sizes` | R4 | 补 `cudagraph_capture_sizes:[4,8,12,16,20,24,28,32]` |
| G7 | D 误开 `--enable-prefix-caching`/`--enable-chunked-prefill` | R1 | `decode.engine` 加 `enable_prefix_caching:false`、`enable_chunked_prefill:false` |
| G8 | P/D 缺 `--enable-auto-tool-choice`/`--tool-call-parser glm47`/`--reasoning-parser glm45` | R4 | `common` 补三项 |
| G9 | env：`HCCL_BUFFSIZE 1024→256`、缺 Mooncake 共用 env、env 段抖动 | R4+base env | 角色 env 对齐官方；共用 env 注入层确认后再补 |

### 4.2 DeepSeek-V4-Flash

| # | 问题（下发 → 官方） | 原因 | 修复（pd_config.json） |
|---|--------------------|------|----------------------|
| V1 | P 批量 `4096/64 → 8192/16` | R2 | `prefill.engine` 改 `max_num_batched_tokens:8192, max_num_seqs:16` |
| V2 | D 批量 `32/8 → 120/60` | R2 | `decode.engine` 改 `120/60` |
| V3 | P/D `gpu-mem 0.95/0.92 → 0.9/0.9` | R2 | P、D 均改 `0.9` |
| V4 | `max-model-len 1024000 → 1048576` | R2 | `common.max_model_len:1048576` |
| V5 | `seed 0 → 1024` | R1 | `common` 加 `seed:1024` |
| V6 | `speculative {3,deepseek_mtp} → {1,mtp,enforce_eager:true}` | R2/R4 | `common.speculative_config` 改 `{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}` |
| V7 | P 缺 `--enforce-eager`；误带 `--async-scheduling`/`--enable-prefix-caching`/`--compilation-config FULL_DECODE_ONLY` | R1 | `prefill.engine` 加 `enforce_eager:true`、`async_scheduling:false`、`compilation_config:null` |
| V8 | P/D 缺 `--no-enable-prefix-caching` | R4 | `common` 加 `no_enable_prefix_caching:true` |
| V9 | `additional-config` 与官方完全不同 | R2/R4 | 按官方分别重写 P、D 的 `additional_config` |
| V10 | 缺 `--reasoning-parser deepseek_v4` | R4 | `common` 加 `reasoning_parser:"deepseek_v4"` |
| V11 | 缺 `--model-loader-extra-config {enable_multithread_load,num_threads:128}` | R4 | `common` 加 `model_loader_extra_config` |
| V12 | wings 强制 disable hybrid-kv，官方要 `--no-disable-hybrid-kv-cache-manager` | **R5** | 见 §七-1（改 JSON 可能不够） |
| V13 | `engine_id` 按 dp_rank 注入，官方 Hybrid 示例固定 `0/1` | **R5** | 见 §七-2（需真机确认） |

### 4.3 Qwen3.5-397B-A17B（静态，dry_run 无场景）

| # | 问题（注册表 → 官方） | 原因 | 修复 |
|---|----------------------|------|------|
| Q1 | `common` 缺 `max_model_len`（官方 16384） | R4 | 加 `max_model_len:16384` |
| Q2 | 缺 `no_enable_prefix_caching` | R4 | 加 `no_enable_prefix_caching:true` |
| Q3 | 缺 `speculative_config`（官方 `{qwen3_5_mtp,3,enforce_eager}`） | R4 | 加 `speculative_config` |
| Q4 | P/D 缺 `additional_config {recompute_scheduler_enable,enable_cpu_binding}`；P 缺 `compilation_config FULL_DECODE_ONLY` | R4 | 按官方补 |
| Q5 | D 消费端 extra 官方多 `kv_buffer_device:"npu"` | R4 | 评估是否补 |

---

## 五、静态核对（无 dry-run 场景的注册表项）

- **DeepSeek-V3.2（`DeepseekV32ForCausalLM`）—— ✅ 逐项匹配官方**：连接器 Layerwise、kv_port 30000/30100、P(32560/64/0.82/enforce_eager/layer_sharding+enable_dsa_cp)、D(12/4/0.95/FULL_DECODE_ONLY+capture[3,6,9,12]/recompute_scheduler_enable)、common(EP/ascend/68000/no_prefix_caching/spec{2,deepseek_mtp})。无需改。
- **Qwen3.5-397B（`Qwen3_5MoeForConditionalGeneration`）—— ❌ 注册表偏薄**：见 §4.3 Q1–Q5。
- **Qwen3-30B-A3B（`Qwen3MoeForCausalLM`）—— ⚠️ 非官方教程基线**：来自用户自定义 `run_dp_template.sh`，无官方逐字基线；`_confirm` 已自标 4 项待真机确认，按真机实测为准，不纳入官方对比。

---

## 六、修复落地：完整可替换 JSON 条目（只改数据）

### 6.1 GLM-5

```jsonc
"GlmMoeDsaForCausalLM": {
  "connector": "MooncakeConnectorV1",
  "kv_port": { "P": "30000", "D": "30100" },
  "extra_config": { "use_ascend_direct": true },
  "common": {
    "enable_expert_parallel": true, "quantization": "ascend", "seed": 1024,
    "enable_auto_tool_choice": true,            // G8
    "tool_call_parser": "glm47",                // G8
    "reasoning_parser": "glm45",                // G8
    "speculative_config": { "num_speculative_tokens": 3, "method": "deepseek_mtp" }
    // max_model_len 不放 common（P/D 不同）→ 见角色级
  },
  "prefill": {
    "engine": {
      "max_model_len": 131072,                  // G4
      "max_num_batched_tokens": 4096, "max_num_seqs": 64, "gpu_memory_utilization": 0.95,
      "enforce_eager": true,                    // G1
      "enable_prefix_caching": false,           // G2
      "enable_chunked_prefill": true,
      "compilation_config": null,               // G3 清掉 base 的 FULL_DECODE_ONLY
      "additional_config": {                    // G5
        "fuse_muls_add": true, "multistream_overlap_shared_expert": true,
        "recompute_scheduler_enable": true,
        "ascend_compilation_config": { "enable_npugraph_ex": true },
        "enable_dsa_cp": true, "layer_sharding": ["q_b_proj", "o_proj"]
      }
    },
    "env": { "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1", "VLLM_ASCEND_ENABLE_FUSED_MC2": "1" }
  },
  "decode": {
    "engine": {
      "max_model_len": 200000,
      "max_num_batched_tokens": 32, "max_num_seqs": 8, "gpu_memory_utilization": 0.92,
      "enable_prefix_caching": false,           // G7
      "enable_chunked_prefill": false,          // G7
      "compilation_config": {                   // G6
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [4,8,12,16,20,24,28,32]
      },
      "additional_config": {                    // G5
        "fuse_muls_add": true, "multistream_overlap_shared_expert": true,
        "recompute_scheduler_enable": true,
        "ascend_compilation_config": { "enable_npugraph_ex": true }
      }
    },
    "env": { "VLLM_ASCEND_ENABLE_MLAPO": "1", "TASK_QUEUE_ENABLE": "1", "VLLM_ASCEND_ENABLE_FUSED_MC2": "1" }
  }
}
```
> env 另：`HCCL_BUFFSIZE` 对齐 256，补 `ASCEND_AGGREGATE_ENABLE/ACL_OP_INIT_MODE/ASCEND_A3_ENABLE/VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` —— 确认由哪层注入后再调（部分在 base ascend env）。

### 6.2 DeepSeek-V4-Flash（对齐 A3 1P1D）

```jsonc
"DeepseekV4ForCausalLM": {
  "connector": "MooncakeHybridConnector",
  "kv_port": { "P": "30000", "D": "30100" },
  "extra_config": {},
  "common": {
    "enable_expert_parallel": true, "quantization": "ascend",
    "max_model_len": 1048576,                   // V4
    "seed": 1024,                               // V5
    "no_enable_prefix_caching": true,           // V8
    "block_size": 128,
    "tokenizer_mode": "deepseek_v4", "tool_call_parser": "deepseek_v4",
    "reasoning_parser": "deepseek_v4",          // V10
    "enable_auto_tool_choice": true,
    "model_loader_extra_config": { "enable_multithread_load": "true", "num_threads": 128 }, // V11
    "disable_hybrid_kv_cache_manager": false,   // V12（注意 §七-1：可能被 guard 覆盖）
    "speculative_config": { "num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true } // V6
  },
  "prefill": {
    "engine": {
      "max_num_batched_tokens": 8192,           // V1
      "max_num_seqs": 16,                       // V1
      "gpu_memory_utilization": 0.9,            // V3
      "enforce_eager": true,                    // V7
      "compilation_config": null,               // V7
      "async_scheduling": false,                // V7
      "additional_config": { "enable_cpu_binding": true, "enable_shared_expert_dp": true, "enable_dsa_cp": true } // V9
    },
    "env": { "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1", "VLLM_ASCEND_ENABLE_FUSED_MC2": "1" }
  },
  "decode": {
    "engine": {
      "max_num_batched_tokens": 120,            // V2
      "max_num_seqs": 60,                        // V2
      "gpu_memory_utilization": 0.9,            // V3
      "async_scheduling": true,
      "compilation_config": { "cudagraph_mode": "FULL_DECODE_ONLY" },
      "additional_config": {                    // V9
        "ascend_compilation_config": { "enable_npugraph_ex": true, "enable_static_kernel": false },
        "enable_cpu_binding": true, "multistream_overlap_shared_expert": true,
        "recompute_scheduler_enable": true
      }
    },
    "env": { "VLLM_ASCEND_ENABLE_FUSED_MC2": "1", "VLLM_ASCEND_ENABLE_MLAPO": "1" }
  }
}
```
> ⚠️ 官方有 A2(4P1D) 与 A3(1P1D) 两套；上表对齐 **A3**。若同条目要兼顾 A2，需按平台再分（A2 P=4096/16，D=60/30）。

### 6.3 Qwen3.5（静态补齐要点）

`common` 补 `max_model_len:16384`、`no_enable_prefix_caching:true`、`speculative_config:{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}`；P/D `additional_config` 补 `{recompute_scheduler_enable,enable_cpu_binding}`；P `compilation_config:{FULL_DECODE_ONLY}`；评估 D 消费端 `kv_buffer_device:"npu"`。

---

## 七、改 JSON 解决不了的（代码/真机待决，本次不改）

1. **V12 hybrid-kv 冲突**：V4-Flash 官方要求**保留** hybrid kv manager，但 wings `_guard_pd_hybrid_kv_cache`（config_loader）对所有 PD **无条件移除**。即便 JSON 写 `disable_hybrid_kv_cache_manager:false`，也可能被该 guard 覆盖 → 需代码侧对 V4-Flash 放行（设计文档 §10.2 真机验证项）。
2. **engine_id 注入策略（V13 / GLM5）**：wings 一律按 `dp_rank` 注 `engine_id`；官方 GLM5(V1) 命令**无** engine_id、V4-Flash(Hybrid) 示例**固定 `0/1`**。多 service 下按 rank 更合理（设计文档附录 B3 就 V1 给了「按 rank」结论），但 Hybrid 的期望需真机确认。
3. **R1 根治（机制层）**：与其每模型在 `prefill` 逐键对冲 base 泄漏，更稳的是在 loader 层约定「**Prefill 不继承 base 的 `compilation_config`/`enable_prefix_caching`/`async_scheduling`**」。属代码改动；未改前按 §六逐键覆盖即等效。

---

## 八、修复优先级

| 级别 | 条目 | 影响 |
|------|------|------|
| **P0 功能性** | V1–V9（V4-Flash 批量/显存/seed/max-len/spec/prefix/P 风味）、G1–G4/G7（GLM5 P 风味 + max-len） | 直接决定能否对齐官方吞吐、是否启动异常 |
| **P1 输出质量** | G8、V10（reasoning/tool 解析器）、G5/G6/V9（additional-config 子键） | 工具调用 & 推理标签解析失效、长序列/显存优化未生效 |
| **P2 一致性** | G9（env）、Q1–Q5（Qwen3.5 静态补齐） | 稳定性/可读性；未来启用即正确 |
| **待决** | V12、V13、R1 根治 | 需代码/真机，见 §七 |

---

## 九、修复后如何验证（复用 dry-run diff 工作流）

1. 改完 `pd_config.json` 后重跑：`python dry_run.py --pd glm5` / `--pd v4flash`。
2. 取 `build/output/start_command_pd-*-{P,D}_node0.sh` 里的 `vllm serve` 行，逐字段对照 §三「官方」列，确认每个 ❌ 已消除。
3. 重点回归：P 角色不再出现 `FULL_DECODE_ONLY/--enable-prefix-caching/--async-scheduling` 且含 `--enforce-eager`；批量/显存/`max-model-len`/`speculative` 与官方一致。
4. V12（hybrid-kv）须真机确认 wings guard 是否已对 V4-Flash 放行 —— dry-run 命令里若仍带 `--disable-hybrid-kv-cache-manager` 则说明需走代码侧。
