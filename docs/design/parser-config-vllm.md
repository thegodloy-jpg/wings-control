# vLLM 引擎 reasoning_parser / tool_call_parser 配置对照

> 数据来源：`wings_control/config/defaults/nvidia_default.json`（`vllm` 引擎）、`wings_control/config/defaults/ascend_default.json`（`vllm_ascend` 引擎）
> 用途：将两份默认配置中 **reasoning_parser** 与 **tool_call_parser** 两项功能单独抽出，按引擎区分对照，便于核对 NVIDIA vLLM 与昇腾 vLLM（vllm-ascend）的解析器差异。
> 说明：本表只覆盖 vLLM 系两个引擎；`sglang`（仅 `tool_call_parser`，命名风格不同）与 `mindie`（使用 `mindie_tool_call_parser` / `mindie_model_type`）的解析器对照见 [model-engine-function-call-analysis.md](model-engine-function-call-analysis.md)。

---

## 1. 两个引擎说明

| 引擎键 | 文件 | 硬件平台 | 配置字段 |
|--------|------|----------|----------|
| `vllm` / `vllm_distributed` | `nvidia_default.json` | NVIDIA GPU | `tool_call_parser`, `reasoning_parser` |
| `vllm_ascend` / `vllm_ascend_distributed` | `ascend_default.json` | 昇腾 910B/910C | `tool_call_parser`, `reasoning_parser` |

> 两个引擎的单机版（`vllm` / `vllm_ascend`）与分布式版（`*_distributed`）在 parser 字段上完全一致，下表合并展示，差异仅在并行/显存等非 parser 参数上。

---

## 2. 逐架构 parser 对照表

字段含义：`tcp` = `tool_call_parser`，`rp` = `reasoning_parser`，`—` 表示该项未配置，`N/A` 表示该引擎未配置此架构。

| 架构 / 模型变体 | vllm tcp | vllm rp | vllm_ascend tcp | vllm_ascend rp | 差异 |
|---|---|---|---|---|---|
| **DeepseekV3ForCausalLM** · default | `deepseek_v3` | `deepseek_r1` | `deepseek_v3` | **—** | ⚠️ ascend default 缺 rp |
| &nbsp;&nbsp;↳ DeepSeek-R1 | `deepseek_v3` | `deepseek_r1` | `deepseek_v3` | `deepseek_r1` | 一致 |
| &nbsp;&nbsp;↳ DeepSeek-R1-w8a8 | N/A | N/A | `deepseek_v3` | `deepseek_r1` | 仅 ascend |
| &nbsp;&nbsp;↳ DeepSeek-V3.1 | `deepseek_v31` | `deepseek_v3` | `deepseek_v31` | `deepseek_v3` | 一致 |
| &nbsp;&nbsp;↳ DeepSeek-V3.1-w8a8 | N/A | N/A | `deepseek_v31` | `deepseek_v3` | 仅 ascend |
| **DeepseekV32ForCausalLM** · default | `deepseek_v32` | `deepseek_r1` | `deepseek_v32` | **—** | ⚠️ ascend 缺 rp |
| **DeepseekV4ForCausalLM** · default | `deepseek_v4` | `deepseek_v4` | `deepseek_v4` | `deepseek_v4` | 一致（ascend 另含 `tokenizer_mode`） |
| &nbsp;&nbsp;↳ DeepSeek-V4-Flash | `deepseek_v4` | `deepseek_v4` | `deepseek_v4` | `deepseek_v4` | 一致 |
| &nbsp;&nbsp;↳ DeepSeek-V4-Pro | N/A | N/A | `deepseek_v4` | `deepseek_v4` | 仅 ascend（仅 distributed） |
| **Qwen3MoeForCausalLM** · default | `hermes` | `qwen3` | `hermes` | `qwen3` | 一致 |
| &nbsp;&nbsp;↳ Qwen3-235B-A22B | N/A | N/A | `hermes` | `qwen3` | 仅 ascend |
| &nbsp;&nbsp;↳ Qwen3-Coder-480B-A35B-Instruct | `qwen3_xml` | — | N/A | N/A | 仅 vllm |
| &nbsp;&nbsp;↳ Qwen3-Coder-30B-A3B-Instruct | `qwen3_xml` | — | N/A | N/A | 仅 vllm |
| **Qwen3ForCausalLM** · default | `hermes` | `qwen3` | `hermes` | `qwen3` | 一致 |
| **Qwen2ForCausalLM** · default | `hermes` | — | `hermes` | — | 一致（均无 rp） |
| **Qwen3NextForCausalLM** · default | `hermes` | `qwen3` | `hermes` | `qwen3` | 一致 |
| **Qwen3_5ForConditionalGeneration** · default | `hermes` | `qwen3` | `hermes` | `qwen3` | 一致 |
| **Qwen3_5MoeForConditionalGeneration** · default | `hermes` | `qwen3` | `hermes` | `qwen3` | 一致 |
| **Glm4MoeForCausalLM** · default | `glm47` | `glm45` | `glm47` | `glm45` | 一致 |
| **GlmMoeDsaForCausalLM** · default | `glm47` | `glm45` | `glm47` | `glm45` | 一致 |
| **Glm4ForCausalLM** · default | `hermes` | — | `hermes` | — | 一致（均无 rp） |
| **MiniMaxM2ForCausalLM** · default | `minimax_m2` | `minimax_m2_append_think` | `minimax_m2` | `minimax_m2_append_think` | 一致 |
| **LlamaForCausalLM** · default | `llama3_json` | — | `llama3_json` | — | 一致（均无 rp） |
| **KimiK25ForConditionalGeneration** · default | N/A | N/A | `kimi_k2` | `kimi_k2` | 仅 ascend |

---

## 3. 关键差异小结

1. **ascend default 缺省 reasoning_parser**：`DeepseekV3ForCausalLM` 与 `DeepseekV32ForCausalLM` 的 `vllm_ascend` default 段未配置 `reasoning_parser`，而 NVIDIA `vllm` 配置了 `deepseek_r1`。
   - DeepseekV3 的 R1 / V3.1 等具体变体在 ascend 侧已补齐 rp，仅 `default` 兜底段缺失。
   - DeepseekV32 在 ascend 侧无任何 rp（包括 default）。
2. **仅 NVIDIA 有的解析器配置**：`Qwen3-Coder-480B-A35B-Instruct`、`Qwen3-Coder-30B-A3B-Instruct`（均为 `qwen3_xml`，无 rp）。
3. **仅昇腾有的解析器配置**：`KimiK25ForConditionalGeneration`（`kimi_k2` / `kimi_k2`）、`DeepSeek-V4-Pro`、`DeepSeek-R1-w8a8`、`DeepSeek-V3.1-w8a8`、`Qwen3-235B-A22B` 等量化/具体变体。
4. **昇腾 V4 附加 tokenizer_mode**：`DeepseekV4ForCausalLM` 在 `vllm_ascend` 侧额外带 `tokenizer_mode: "deepseek_v4"`，NVIDIA 侧用 `kv_cache_dtype: fp8` + `block_size: 256`（与 parser 无关，仅供对照）。
5. **两引擎 parser 命名完全一致**：除上述缺漏/独有项外，凡两边都配置的架构，`tool_call_parser` 与 `reasoning_parser` 取值完全相同——即昇腾 vLLM 直接复用了上游 vLLM 的 parser 命名约定。

---

## 4. 待对齐项（建议）

| 项 | 现状 | 建议 |
|----|------|------|
| `DeepseekV3ForCausalLM` ascend default rp | 缺失 | 补 `reasoning_parser: "deepseek_r1"` 与 NVIDIA 对齐 |
| `DeepseekV32ForCausalLM` ascend rp | 缺失 | 确认昇腾 vllm-ascend 是否支持 V3.2 思维解析，支持则补 `deepseek_r1` |
| `Qwen3-Coder` 变体 ascend | 未配置 | 如昇腾需支持 Qwen3-Coder，补 `tool_call_parser: "qwen3_xml"` |
