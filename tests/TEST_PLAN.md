# Wings Control 完整测试方案

> 生成日期：2026-05-11；最后更新：2026-05-11（新增 GAP-7/8/9，Issue #5/#6/#7，修正"九/十"编号，补充 MindIE FC 测试需求）
> 覆盖范围：单元测试 / 集成测试 / E2E 快照测试
> 当前实现状态：快照层（SNAP-01~10）已完成，单元层 UT-TCP/EV/Accel/Speculative/bash-n 已完成，其余待实现

> **审阅批注（2026-05-11，按当前仓库状态核对）**：
> - `python -m pytest --collect-only -q tests` 当前可收集 **352 个测试**，所以本文档的“待实现/已完成”状态只按新增测试计划本身统计，不等同于项目整体测试覆盖。其中文档标为待实现的 `tests/test_unit_engine_select.py`、`tests/test_unit_kv_sparse.py`、`tests/test_unit_mindie_fc.py` 当前已存在，三者合跑为 **55 passed**。
> - 抽跑 `tests/test_unit_security.py tests/test_unit_tcp.py tests/test_unit_accel.py tests/test_unit_speculative.py tests/test_unit_bash_syntax.py tests/test_vllm_kv_sparse.py -q` 得到 **103 passed / 14 failed**。失败集中在 UT-EV 期望值与 `_parse_env_file` 实际 `shlex.quote` 输出不一致，以及 Windows 下 `bash` 指向未初始化 WSL 导致 bash-n 误失败。
> - 抽跑 `tests/test_glm_moe_dsa_ascend_defaults.py tests/test_minimax_m2_ascend.py -q` 得到 **4 passed / 2 failed**。配置已更新为 `max_num_seqs=256`、`reasoning_parser=minimax_m2`，但旧测试仍断言 `8` 和 `minimax_m2_reasoning`。
> - 本文档整体覆盖思路是合理的，但当前存在若干“按文档实现会测错对象”的条目，下面已逐处批注。

---

## 一、测试设计原则

项目本质是一个**多维度参数合并 + Shell 脚本生成器**，测试体系分三层：

```
单元层（Unit）     → 验证单个函数的参数映射逻辑（确定性最强）
集成层（Integration）→ 验证多层配置合并的优先级与覆盖（功能验证）
快照层（Snapshot） → 验证完整生成脚本的回归（防退化）
```

---

## 二、测试维度全集

```
引擎（6）× 硬件（3）× 模型架构（15）× 模型变体（N）× 任务（3）× 高级特性（5）× 部署（2）× 配置层（4）
```

> **审阅批注**：当前代码层 `SUPPORTED_ENGINES = {"vllm", "vllm_ascend", "sglang", "mindie"}`，`SUPPORTED_DEVICE_TYPES = {"nvidia", "ascend"}`。因此“引擎（6）× 硬件（3）”与当前项目不一致；如果这里包含 `wings`、部署别名或 Ascend310/910B/H20 这类细分硬件，需要在下方显式列出，否则统计口径不准确。

### 引擎列表

| 引擎名（用户传入） | 平台 | 分布式触发方式 |
|----------------|------|-------------|
| `vllm` | NVIDIA | `distributed=True` |
| `vllm_ascend` | Ascend NPU | `distributed=True` |
| `sglang` | NVIDIA | `distributed=True` |
| `mindie` | Ascend NPU | `distributed=True` |

> ⚠️ **备注 [GAP-7]**：`mindie` 引擎与 `vllm` 系列在 Function Call 机制上完全不同——MindIE 使用 `mindie_tool_call_parser` / `mindie_model_type` 作为 wings 内部控制字段，最终由 `_inject_function_call_config`（`mindie_adapter.py`）转化为 MindIE `config.json` 中的 `ModelConfig[0].models.<model_type>.tool_call_options.tool_call_parser`。当前引擎列表未区分两套 FC 机制，后续测试用例设计需专门覆盖 MindIE 路径（见 GAP-7 补充）。

### 模型架构全集

| # | 架构类名 | Ascend 专属配置 | NVIDIA 专属配置 | 模型变体 |
|---|---------|--------------|--------------|--------|
| 1 | `DeepseekV3ForCausalLM` | ✅ | ✅ | default / DeepSeek-R1-w8a8 / DeepSeek-V3.1 / DeepSeek-R1(H20-96G/141G) |
| 2 | `DeepseekV32ForCausalLM` | ✅ | ✅ | default |
| 3 | `Qwen3ForCausalLM` | ✅ | ✅ | default |
| 4 | `Qwen3MoeForCausalLM` | ✅ | ✅ | default / Qwen3-235B-A22B / Qwen3-Coder-480B / Qwen3-Coder-30B |
| 5 | `Qwen3NextForCausalLM` | ✅ | ✅ | default |
| 6 | `Qwen3_5ForConditionalGeneration` | ✅（Ascend 额外优化） | ✅ | default |
| 7 | `Qwen3_5MoeForConditionalGeneration` | ✅ | ✅ | default |
| 8 | `Qwen2ForCausalLM` | ✅ | ✅ | default |
| 9 | `Glm4ForCausalLM` | ✅ | ✅ | default |
| 10 | `Glm4MoeForCausalLM` | ✅ | ✅ | default |
| 11 | `GlmMoeDsaForCausalLM` | ✅（大量 NPU 优化） | ✅ | default |
| 12 | `LlamaForCausalLM` | ✅ | ✅ | default |
| 13 | `MiniMaxM2ForCausalLM` | ✅（存在 Issue #4） | ✅ | default |
| 14 | `KimiK25ForConditionalGeneration` | ✅ | ✅ | default |
| 15 | `(未知架构)` | — | — | — |

---

## 三、单元测试层（Unit Tests）

### 3.1 `_parse_env_file` — env 文件安全解析

**文件**：`tests/test_unit_security.py`  
**目标**：验证注入到脚本的变量名和值是否安全、正确。

| 测试 ID | 输入 `.env` 内容 | 期望输出 | 验证重点 |
|--------|--------------|---------|---------|
| UT-EV-01 | `MY_VAR=hello` | `export MY_VAR='hello'` | 普通 key=value 正确映射 |
| UT-EV-02 | `KEY="quoted value"` | `export KEY='quoted value'` | 双引号去除后正确 quote |
| UT-EV-03 | `KEY='single quoted'` | `export KEY='single quoted'` | 单引号去除后正确 quote |
| UT-EV-04 | `# comment` / 空行 | 无输出行 | 注释和空行跳过 |
| UT-EV-05 | `INVALID_LINE_NO_EQUALS` | 无输出行 + warning log | 缺少 `=` 时跳过 |
| UT-EV-06 | `A$(cmd)=val` | 无输出行 + warning log | **命令注入 key 被拦截** |
| UT-EV-07 | `A B=val` | 无输出行 + warning log | 含空格 key 被拦截 |
| UT-EV-08 | `A-B=val` | 无输出行 + warning log | 含连字符 key 被拦截 |
| UT-EV-09 | `LD_PRELOAD=/evil.so` | `export LD_PRELOAD='/evil.so'` | 合法 key、危险 value 被安全 quote |
| UT-EV-10 | `VAL=$(whoami)` | `export VAL='$(whoami)'` | value 含 `$()` 被安全 quote，不执行 |
| UT-EV-11 | `VAL=it's a test` | `export VAL='it'"'"'s a test'` | value 含单引号正确 shell 转义 |
| UT-EV-12 | 文件 UTF-8 带 BOM | 正常解析 | 编码兼容性 |

> **审阅批注**：此表与当前实现/测试不一致。`_parse_env_file()` 使用 `shlex.quote(value)`，对 `hello`、`ok`、`yes` 这类 shell-safe 字符串实际输出是 `export KEY=value`，不会强制输出 `export KEY='value'`，导致现有 UT-EV-01/04/05/12 失败。UT-EV-12 也不是“正常解析”：当前实现用 `encoding="utf-8"`，BOM 会挂到第一行 key 上并使第一行被跳过，现有测试名也是 `test_ev12_utf8_bom_first_key_skipped`。这里需要先决定是修改实现为强制单引号/`utf-8-sig`，还是修改测试和文档期望。

**验证方法**：
1. 直接调用 `_parse_env_file(path)` 断言返回的 `list[str]`
2. 将输出行 join 后通过 `subprocess.run(['bash', '-n'])` 验证语法合法

---

### 3.2 工具调用解析器映射（tool_call_parser / reasoning_parser）

**文件**：`tests/test_unit_tcp.py`  
**目标**：每个模型架构 + 引擎 + model_name 变体组合，验证 `tool_call_parser` / `reasoning_parser` 字段值。  
**方法**：调用 `load_and_merge_configs()`，提取 `merged['tool_call_parser']` 和 `merged['reasoning_parser']`。

> **审阅批注**：当前测试和实现中的 parser 字段位于 `merged["engine_config"]`，不是顶层 `merged`。`tests/test_unit_tcp.py` 实际通过 `merged.get("engine_config", {})` 断言。此处方法描述应改成提取 `merged["engine_config"]["tool_call_parser"]` / `reasoning_parser`，否则会误导后续集成测试实现。

#### vLLM（NVIDIA）

| 测试 ID | 模型架构 | model_name 变体 | `enable_auto_tool_choice` | 期望 tool_call_parser | 期望 reasoning_parser |
|--------|---------|--------------|--------------------------|---------------------|---------------------|
| UT-TCP-01 | Qwen3ForCausalLM | default | True | `hermes` | `qwen3` |
| UT-TCP-02 | Qwen3MoeForCausalLM | default | True | `hermes` | `qwen3` |
| UT-TCP-03 | Qwen3MoeForCausalLM | Qwen3-Coder-480B-A35B-Instruct | True | `qwen3_xml` | None |
| UT-TCP-04 | Qwen3MoeForCausalLM | Qwen3-Coder-30B-A3B-Instruct | True | `qwen3_xml` | None |
| UT-TCP-05 | DeepseekV3ForCausalLM | DeepSeek-V3 | True | `deepseek_v3` | `deepseek_v3` |
| UT-TCP-06 | DeepseekV3ForCausalLM | DeepSeek-V3.1 | True | `deepseek_v31` | `deepseek_v3` |
| UT-TCP-07 | DeepseekV32ForCausalLM | default | True | `deepseek_v32` | `deepseek_v3` |
| UT-TCP-08 | Glm4MoeForCausalLM | default | True | `glm47` | `glm45` |
| UT-TCP-09 | GlmMoeDsaForCausalLM | default | True | `glm47` | `glm45` |
| UT-TCP-10 | LlamaForCausalLM | default | True | `llama3_json` | None |
| UT-TCP-11 | Glm4ForCausalLM | default | True | `hermes` | None |
| UT-TCP-12 | Qwen3NextForCausalLM | default | True | `hermes` | `qwen3` |
| UT-TCP-13 | MiniMaxM2ForCausalLM | default | True | `minimax_m2` | `minimax_m2` |
| UT-TCP-14 | **任意架构** | default | **False** | None（无） | None（无） |
| UT-TCP-14K | KimiK25ForConditionalGeneration | Kimi-K2.5 | True | `kimi_k2` | `kimi_k2` |

#### vLLM-Ascend（Ascend NPU）

| 测试 ID | 模型架构 | model_name 变体 | 期望 tool_call_parser | 期望 reasoning_parser |
|--------|---------|--------------|---------------------|---------------------|
| UT-TCP-15 | KimiK25ForConditionalGeneration | default | `kimi_k2` | `kimi_k2` |
| UT-TCP-16 | DeepseekV3ForCausalLM | DeepSeek-R1-w8a8 | `deepseek_v3` | `deepseek_r1` |
| UT-TCP-17 | DeepseekV3ForCausalLM | DeepSeek-V3.1 | `deepseek_v31` | `deepseek_v3` |
| UT-TCP-18 | Qwen3MoeForCausalLM | default | `hermes` | `qwen3` |
| UT-TCP-19 | GlmMoeDsaForCausalLM | default | `glm47` | `glm45` |

#### SGLang（NVIDIA）

| 测试 ID | 模型架构 | 期望 tool_call_parser |
|--------|---------|---------------------|
| UT-TCP-20 | DeepseekV3ForCausalLM | `deepseekv3` |
| UT-TCP-21 | Qwen3ForCausalLM | `qwen25` |
| UT-TCP-22 | Qwen3MoeForCausalLM | `qwen25` |
| UT-TCP-23 | LlamaForCausalLM | `llama3` |
| UT-TCP-24 | Glm4MoeForCausalLM | None（sglang 不配置 GLM parser） |

> ⚠️ **备注 [GAP-7a]：MindIE 引擎的 parser 映射测试完全缺失。**
> 当前 UT-TCP 仅覆盖 `vllm`/`vllm_ascend`/`sglang` 三种引擎，**`mindie` 引擎的 `mindie_tool_call_parser` 字段没有任何测试用例**。
> 需补充以下最低覆盖：
>
> | 测试 ID（建议）| 引擎 | 架构 | model_name | `enable_auto_tool_choice` | 期望 `mindie_tool_call_parser` | 期望 MindIE `tool_call_options` 是否注入 |
> |--------------|------|-----|-----------|--------------------------|-------------------------------|----------------------------------------|
> | UT-TCP-25 | mindie | DeepseekV3ForCausalLM | default | True | `deepseekv3` | ❌ 不注入（_inject_function_call_config 跳过非 deepseek_v31） |
> | UT-TCP-26 | mindie | DeepseekV3ForCausalLM | DeepSeek-V3.1 | True | `deepseek_v31` | ✅ 注入 `tool_call_options.tool_call_parser="deepseek_v31"` |
> | UT-TCP-27 | mindie | Qwen3MoeForCausalLM | default | True | `qwen3` | ❌ 不注入（只记日志） |
> | UT-TCP-28 | mindie | DeepseekV3ForCausalLM | DeepSeek-V3.1 | False（未设置）| — | ❌ mindie_tool_call_parser 被剥离，不注入 |
> | UT-TCP-29 | mindie | DeepseekV32ForCausalLM | default | True | — | ❌ ascend_default.json 无 mindie 块，应回退 default |
>
> 相关实现：`config_loader.py:_set_mindie_function_call`（line 1145）、`mindie_adapter.py:_inject_function_call_config`（line 1444）

> **审阅批注**：这里“MindIE parser 映射未进 UT-TCP”是准确的，但“没有任何测试用例/零覆盖”不准确。`tests/test_unit_mindie_fc.py` 当前已存在并通过，覆盖 `_set_mindie_function_call()` 门控和 `_inject_function_call_config` 注入；`tests/test_mindie_distributed_env_defaults.py` 也覆盖了 `_build_model_config_overrides()` 直接注入/跳过逻辑。缺口主要剩在 `load_and_merge_configs()` 端到端剥离路径，以及快照层。

---

### 3.3 vLLM CLI Flag 转换

**文件**：`tests/test_unit_vllm_flags.py`  
**目标**：`merged` 字典参数 → CLI flag 的逐一验证。

| 测试 ID | merged 参数 | 期望 CLI flag |
|--------|-----------|-------------|
| UT-VF-01 | `max_model_len=8192` | `--max-model-len 8192` |
| UT-VF-02 | `tensor_parallel_size=8` | `--tensor-parallel-size 8` |
| UT-VF-03 | `gpu_memory_utilization=0.92` | `--gpu-memory-utilization 0.92` |
| UT-VF-04 | `enable_prefix_caching=True` | `--enable-prefix-caching` |
| UT-VF-05 | `no_enable_prefix_caching=True` | 不含 `--enable-prefix-caching` |
| UT-VF-06 | `enable_chunked_prefill=True` | `--enable-chunked-prefill` |
| UT-VF-07 | `enable_expert_parallel=True` | `--enable-expert-parallel` |
| UT-VF-08 | `tool_call_parser=hermes` | `--tool-call-parser hermes` |
| UT-VF-09 | `reasoning_parser=qwen3` | `--reasoning-parser qwen3` |
| UT-VF-10 | `speculative_config='{"num_speculative_tokens":3}'` | `--speculative-config '{"num_speculative_tokens":3}'` |
| UT-VF-11 | `compilation_config='{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | `--compilation-config '...'` |
| UT-VF-12 | `additional_config='{"fuse_muls_add":true}'` | `--additional-config '...'` |
| UT-VF-13 | `mm_encoder_tp_mode=data` | `--mm-encoder-tp-mode data` |
| UT-VF-14 | `task=generate` | `--task generate` |
| UT-VF-15 | `quantization=ascend` | `--quantization ascend` |
| UT-VF-16 | `enable_auto_tool_choice=True` | `--enable-auto-tool-choice` |
| UT-VF-17 | `async_scheduling=True` | `--async-scheduling`（或对应 flag） |
| UT-VF-18 | `seed=1024` | `--seed 1024` |
| UT-VF-19 | `max_num_seqs=16` | `--max-num-seqs 16` |
| UT-VF-20 | `max_num_batched_tokens=8192` | `--max-num-batched-tokens 8192` |

> **审阅批注**：`tests/test_unit_vllm_flags.py` 当前不存在；但 `_format_cli_arg()`、快照、`test_glm_moe_dsa_ascend_defaults.py`、`test_vllm_kv_sparse.py` 已间接覆盖部分 CLI 渲染。注意 UT-VF-05 只写“不含 `--enable-prefix-caching`”不够，当前 bool 渲染会把 `no_enable_prefix_caching=True` 输出为 `--no-enable-prefix-caching`，应明确断言该负向 flag 是否出现。dict 型字段当前会被紧凑 JSON 序列化，例如 `{"cudagraph_mode":"FULL_DECODE_ONLY"}`。

---

### 3.4 并行度计算

**文件**：`tests/test_unit_parallelism.py`

| 测试 ID | device_count | nnodes | 用户显式 TP | 期望 tensor_parallel_size | 验证点 |
|--------|-------------|--------|-----------|--------------------------|--------|
| UT-TP-01 | 8 | 1 | 无 | 8 | 单节点自动计算 |
| UT-TP-02 | 4 | 1 | 无 | 4 | 单节点 4 卡 |
| UT-TP-03 | 1 | 1 | 无 | 1 | 单卡 |
| UT-TP-04 | 8 | 2 | 无 | 8 | 分布式：TP=per-node device_count |
| UT-TP-05 | 8 | 1 | `--tp 4` | 4 | 用户显式值优先 |

> **审阅批注**：`tests/test_unit_parallelism.py` 当前不存在，且 UT-TP-04 与当前实现不符。`_adjust_tensor_parallelism()` 在非 PD 分布式路径中设置 `tensor_parallel_size = device_count * nnodes`；现有 `SNAP-02` 也生成 `--tensor-parallel-size 16`（2 节点 × 8 卡），不是 8。若目标是 Ray 全局 TP，UT-TP-04 期望应改为 16；若目标改为 per-node TP，则需要先改实现和快照。

---

## 四、集成测试层（Integration Tests）

### 4.1 模型变体专属参数覆盖

**文件**：`tests/test_integration_model_variants.py`  
**目标**：`model_name` 变体是否正确覆盖同架构默认配置。

| 测试 ID | 引擎 | 架构 | model_name | 被覆盖参数 | 期望值 | 对比默认值 |
|--------|------|-----|-----------|----------|-------|----------|
| IT-MV-01 | sglang | DeepseekV3ForCausalLM | DeepSeek-R1（H20-96G） | `mem_fraction_static` | 0.9 | 无默认 |
| IT-MV-02 | sglang | DeepseekV3ForCausalLM | DeepSeek-R1（H20-141G） | `dp` | 8 | 无默认 |
| IT-MV-03 | sglang | DeepseekV3ForCausalLM | DeepSeek-R1（H20-141G） | `enable_dp_attention` | True | False |
| IT-MV-04 | vllm | Qwen3MoeForCausalLM | Qwen3-Coder-480B-A35B-Instruct | `tool_call_parser` | `qwen3_xml` | `hermes` |
| IT-MV-05 | vllm | Qwen3MoeForCausalLM | Qwen3-Coder-30B-A3B-Instruct | `tool_call_parser` | `qwen3_xml` | `hermes` |
| IT-MV-06 | vllm_ascend | DeepseekV3ForCausalLM | DeepSeek-R1-w8a8 | 与 default 一致 | `deepseek_v3` | 同 default |
| IT-MV-07 | vllm_ascend | DeepseekV3ForCausalLM | DeepSeek-V3.1 | `tool_call_parser` | `deepseek_v31` | `deepseek_v3` |
| IT-MV-08 | mindie | DeepseekV3ForCausalLM | DeepSeek-R1-w8a8（distributed） | `maxSeqLen`, `maxBatchSize` | 16384, 130 | 4096, 默认 |
| IT-MV-09 | vllm_ascend | Qwen3MoeForCausalLM | Qwen3-235B-A22B（distributed） | `maxBatchSize` | 130 | 默认 |

> **审阅批注**：IT-MV-09 的引擎写错了。`maxBatchSize=130` 是 `ascend_default.json` 中 `Qwen3-235B-A22B.mindie_distributed` 的 MindIE 字段；`vllm_ascend`/`vllm_ascend_distributed` 下没有 `maxBatchSize`，只有 vLLM 风格字段（如 `max_model_len`、`task`、parser 等）。如果保留 `maxBatchSize` 断言，引擎应改为 `mindie`。

**如何判断准确**：断言 `merged['tool_call_parser']` / `merged['max_model_len']` 等字段值与期望一致，而非仅检查存在性。

> **审阅批注**：同 3.2，这里的断言路径应是 `merged["engine_config"]` 内字段。MindIE 场景还要区分 `engine_config` 扁平字段和最终 `config.json` 覆盖结构，不能统一用 `merged['max_model_len']` 表述。

---

### 4.2 配置优先级（四层合并顺序）

**文件**：`tests/test_integration_priority.py`

| 测试 ID | 冲突场景 | 期望结果 | 胜出层级 |
|--------|---------|---------|---------|
| IT-PRI-01 | 硬件默认 `max_model_len=4096` vs CLI `--max-model-len 8192` | 8192 | CLI |
| IT-PRI-02 | 模型默认 `tool_call_parser=hermes` vs CLI `--tool-call-parser llama3` | `llama3` | CLI |
| IT-PRI-03 | 模型默认 `gpu_memory_utilization=0.92` vs 硬件默认 0.9 | 0.92 | 模型默认 |
| IT-PRI-04 | CLI `gpu_memory_utilization=0.95` vs 模型默认 0.92 | 0.95 | CLI |
| IT-PRI-05 | 用户 `config_file` JSON `max_model_len=32768` vs 模型默认 4096 | 32768 | 用户 config |
| IT-PRI-06 | 用户 `config_file` vs CLI 同字段 | CLI 值 | CLI（最高） |

---

### 4.3 高级特性开关验证

**文件**：`tests/test_integration_features.py`

#### A. 投机推理（Speculative Decoding）

| 测试 ID | 引擎 | 输入 | 期望 merged 参数 | 期望 CLI |
|--------|------|-----|---------------|---------|
| IT-SD-01 | vllm | `enable_speculative_decode=True` + spec_model_path | `speculative_config` 含 `draft_model_path` | `--speculative-config '{...}'` |
| IT-SD-02 | vllm | `enable_speculative_decode=True`，无 spec_model | 使用 MTP 或默认 | 含 `speculative` 相关 flag |
| IT-SD-03 | vllm_ascend | `enable_speculative_decode=True` | Ascend 特化配置 | 含 ascend spec 参数 |
| IT-SD-04 | vllm | `enable_speculative_decode=False` | 无 speculative 字段 | 不含 `--speculative-config` |

#### B. KV Cache 策略

| 测试 ID | 引擎 | 输入 | 期望输出 |
|--------|------|-----|---------|
| IT-KV-01 | vllm | `enable_sparse=True` | IndexCache 或 `kv_cache_dtype=fp8` |
| IT-KV-02 | vllm | `kv_cache_dtype=fp8` | `--kv-cache-dtype fp8` |
| IT-KV-03 | vllm | `enable_prefix_caching=True` | `--enable-prefix-caching` |
| IT-KV-04 | vllm_ascend | `no_enable_prefix_caching=True`（Ascend 配置） | 不含 `--enable-prefix-caching` |

#### C. RAG 加速

| 测试 ID | 输入 | 期望 |
|--------|-----|-----|
| IT-RAG-01 | `enable_rag_acc=True` | RAG 进程 / 端口配置出现在脚本中 |
| IT-RAG-02 | `enable_rag_acc=False` | 无 RAG 相关内容 |

---

### 4.4 GlmMoeDsaForCausalLM — Ascend 全参数验证

**文件**：`tests/test_integration_model_variants.py`
**目标**：11 个 Ascend 专属优化参数全部正确应用。

| 参数 | 期望值 | 期望 CLI flag 或位置 |
|-----|--------|-------------------|
| `enable_expert_parallel` | True | `--enable-expert-parallel` |
| `quantization` | `ascend` | `--quantization ascend` |
| `gpu_memory_utilization` | 0.95 | `--gpu-memory-utilization 0.95` |
| `max_num_seqs` | 256 | `--max-num-seqs 256` |
| `max_num_batched_tokens` | 4096 | `--max-num-batched-tokens 4096` |
| `enable_chunked_prefill` | True | `--enable-chunked-prefill` |
| `enable_prefix_caching` | True | `--enable-prefix-caching` |
| `async_scheduling` | True | `--async-scheduling` |
| `compilation_config.cudagraph_mode` | `FULL_DECODE_ONLY` | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` |
| `speculative_config` | 含 `deepseek_mtp` | `--speculative-config '{"method":"deepseek_mtp",...}'` |
| `additional_config.fuse_muls_add` | True | `--additional-config '{"fuse_muls_add":true,...}'` |
| `tool_call_parser` | `glm47` | `--tool-call-parser glm47` |
| `reasoning_parser` | `glm45` | `--reasoning-parser glm45` |

> ⚠️ **备注 [GAP-8]**：以下两个参数在 2026-05-11 合并修复后已存在于配置，**但未列入本表**，实现测试时需补充：
> - `seed: 1024` → `--seed 1024`（来自原第一条 `GlmMoeDsaForCausalLM` 条目，合并保留）
> - `additional_config.ascend_compilation_config.enable_npugraph_ex: true`（嵌套字段，需验证序列化后出现在 `--additional-config` 中）
>
> 另：`max_num_seqs` 期望值已更新为 **256**（2026-05-11 合并修复，旧值 8 来自第一条重复条目）。若测试代码中写死了 `max_num_seqs=8` 需同步修正。

> **审阅批注**：本表对当前配置和 SNAP-07 是准确的：`ascend_default.json` 与 `snap07_vllm_ascend_glmmoe_dsa.sh` 均为 `max_num_seqs=256`，并包含 `seed=1024` 与 `additional_config.ascend_compilation_config.enable_npugraph_ex=true`。但 `tests/test_glm_moe_dsa_ascend_defaults.py` 仍断言 `max_num_seqs=8`，当前会失败，需把该测试列为“已存在但待同步”，不能归为待实现。

---

### 4.5 跨平台覆盖对比

**文件**：`tests/test_integration_cross_platform.py`

#### KimiK25ForConditionalGeneration

| 测试 ID | 引擎 | 期望行为 |
|--------|------|---------|
| IT-K2-01 | vllm_ascend | 完整专属参数：`kimi_k2` parsers + `mm_encoder_tp_mode=data` + `max_num_seqs=16` + `no_enable_prefix_caching=True` |
| IT-K2-02 | vllm（NVIDIA） | 官方 Kimi 配置：`kimi_k2` parsers + `mm_encoder_tp_mode=data` + `max_model_len=4096` |

#### MiniMaxM2ForCausalLM（已知 Issue #4：序列长度不对称）

| 测试 ID | 引擎 | 期望 max_model_len / maxSeqLen | 备注 |
|--------|------|-------------------------------|------|
| IT-MM-01 | vllm_ascend | 34816 | 已在 ascend 配置中指定 |
| IT-MM-02 | mindie | 4096 | **当前值**，记录 Issue #4 存在 |
| IT-MM-03 | vllm（NVIDIA） | 4096 | NVIDIA 无专属配置，用默认 |

> **审阅批注**：MiniMaxM2 的 `max_model_len=34816` 结论正确，但当前 `tests/test_minimax_m2_ascend.py` 与 Issue #7 修复状态冲突：配置中 `reasoning_parser` 已是 `minimax_m2`，测试仍断言 `minimax_m2_reasoning` 并失败。应补一条“既有测试需要同步 Issue #7”的跟踪项。

---

## 五、E2E 快照层

### 5.1 已完成快照（SNAP-01~10）

| 快照 ID | 描述 | 状态 |
|--------|------|------|
| SNAP-01 | vLLM + NVIDIA + Qwen3（单节点 + tool_call） | ✅ |
| SNAP-02 | vLLM + NVIDIA + DeepseekV3（分布式） | ✅ |
| SNAP-03 | vLLM-Ascend + KimiK25（Ascend 专属参数） | ✅ |
| SNAP-04 | MindIE + Ascend + Qwen3MoE（单节点） | ✅ |
| SNAP-05 | SGLang + NVIDIA（Embedding 任务） | ✅ |
| SNAP-06 | vLLM + NVIDIA + 投机推理 | ✅ |
| SNAP-07 | vLLM-Ascend + GlmMoeDsa（NPU 优化参数） | ✅ |
| SNAP-08 | vLLM + NVIDIA + env_overrides 注入 | ✅ |
| SNAP-09 | vLLM-Ascend + DeepseekV3 + RAG 加速 | ✅ |
| SNAP-10 | vLLM + NVIDIA + 未知架构（降级 default） | ✅ |

### 5.2 待补充快照（SNAP-11~20）

| 快照 ID | 描述 | 关键验证点 |
|--------|------|----------|
| SNAP-11 | vLLM + NVIDIA + DeepseekV3 + H20-96G | `mem_fraction_static=0.9` |
| SNAP-12 | vLLM + NVIDIA + DeepseekV3 + H20-141G | `dp=8, enable_dp_attention=True` |
| SNAP-13 | vLLM + NVIDIA + Qwen3MoE + Qwen3-Coder-480B | `tool_call_parser=qwen3_xml` |
| SNAP-14 | vLLM-Ascend + GlmMoeDsa 全参数验证 | 13 个参数全部验证 |
| SNAP-15 | vLLM + NVIDIA + KimiK25（验证降级） | 无 kimi 专属参数 |
| SNAP-16 | vLLM-Ascend + MiniMaxM2（验证 34816） | `--max-model-len 34816` |
| SNAP-17 | vLLM + CLI 覆盖 max_model_len | `--max-model-len 8192` 覆盖 4096 |
| SNAP-18 | vLLM-Ascend + Qwen3_5（async+compilation） | `no_enable_prefix_caching + async_scheduling + compilation_config` |
| SNAP-19 | vLLM + NVIDIA + Rerank 任务 | rerank 任务相关参数 |
| SNAP-20 | SGLang + NVIDIA + Glm4MoE | sglang 无 tool_call_parser（GLM 不支持） |

> **审阅批注**：SNAP-11/12 的引擎写错了。`mem_fraction_static`、`dp`、`enable_dp_attention` 是 `nvidia_default.json` 中 SGLang H20 变体参数，不是 vLLM 参数；若目标验证这些字段，应改成 “SGLang + NVIDIA + DeepseekV3 + H20-*”。SNAP-14 也应从“13 个参数”更新为表 4.4 的完整集合：原 13 项 + `seed` + `additional_config.ascend_compilation_config.enable_npugraph_ex`，否则与 GAP-8 批注不一致。

> ⚠️ **备注 [GAP-7b]**：待补充快照未覆盖 **MindIE + Function Call** 路径，建议新增：
>
> | 快照 ID | 描述 | 关键验证点 |
> |--------|------|----------|
> | SNAP-21 | MindIE + Ascend + DeepSeek-V3.1 + `enable_auto_tool_choice=True` | MindIE `config.json` 覆盖块中出现 `models.deepseekv2.tool_call_options.tool_call_parser="deepseek_v31"` |
> | SNAP-22 | MindIE + Ascend + DeepSeek-V3（V3.0）+ `enable_auto_tool_choice=True` | `tool_call_options` **不出现**（`_inject_function_call_config` 跳过非 `deepseek_v31`），仅日志记录 |
>
> SNAP-04 已有 MindIE+Qwen3MoE 基础场景，但未测 FC 注入分支。

---

## 六、验证判断方法

### 6.1 如何判断启动命令准确

**三重验证**：

```
1. 语法合法性（Shell Level）
   bash -n <script.sh>  → 零错误

2. 参数语义正确性（Value Level）
   assert "--tool-call-parser hermes" in command
   assert "--max-model-len 8192" in command
   assert "--tensor-parallel-size 8" in command
   assert "10.0.0.1" in command     # host IP

3. 参数完整性（Coverage Level）
   - 模型默认 JSON 中的所有非布尔参数都出现在 CLI 中
   - 启用的特性对应 flag 存在
   - 禁用的特性对应 flag 缺失
```

### 6.2 如何判断环境变量准确

```
1. 注入完整性
   所有 env_overrides/.env 中的 KEY=VALUE 都以 export KEY='VALUE' 形式出现

2. 安全性（恶意 key 检查）
   含 $()、空格、;、& 的 key 不出现在 export 语句中

3. 顺序正确性
   engine 相关 env 在 accel 之后、引擎启动之前设置
```

### 6.3 如何判断分布式配置准确

```
vLLM 分布式（Ray）：
   ray start 命令含 --head 或 --address
   --tensor-parallel-size == 单节点 device_count
   Ray address 指向 head_node_addr

MindIE 分布式：
   worldSize = device_count × nnodes
   server_id == node_rank
   rankTableFile 路径正确
```

> **审阅批注**：Ray 分布式的 TP 判定与当前实现/快照不符。当前 `_adjust_tensor_parallelism()` 对非 PD 分布式使用全局 TP（`device_count * nnodes`），`SNAP-02` 是 16。这里如果保留“单节点 device_count”，需要说明仅适用于 PD 或特定路径；否则应改为全局 TP。

---

## 七、补充测试覆盖（代码分析后识别的遗漏）

> 2026-05-11 分析补充：通过细粒度代码走查识别出原方案未覆盖的 6 个关键区域。

### 7.1 GAP-1：Accel 特性收集逻辑

**文件**：`tests/test_unit_accel.py`（已实现）

| 测试函数 | 覆盖点 | 关键断言 |
|--------|--------|---------|
| `_dedupe_features` | 去重保序 | 重复项只保留首次出现 |
| `_merge_patch_options` | JSON 合并 + 版本保留 | 已有 features 不被覆盖，新 features 追加 |
| `_validate_accel_user_override` | 无效 JSON/非 dict 返回 "" | 防止用户注入非法 patch 选项 |
| `_collect_ears_patch_features` | EARS 仅 vllm 触发，vllm_ascend 跳过 | strategy=mtp/suffix/\*_mtp → ["ears"] |
| `_collect_ears_patch_features` | draft_model_path 存在 → 不触发 EARS | strategy=draft_model 时返回 [] |
| `_collect_indexcache_patch_features` | 仅 GlmMoeDsa/DeepseekV32 触发 indexcache | 其他架构走 FP8 路径，无需补丁 |
| `_collect_indexcache_patch_features` | vllm_ascend 不触发 indexcache | Ascend 侧自有优化 |

### 7.2 GAP-2：投机推理策略解析

**文件**：`tests/test_unit_speculative.py`（已实现）

| 测试场景 | 期望策略 | 说明 |
|--------|---------|------|
| `_resolve_mtp_method` × 7 个架构 | 各自 MTP 方法名 | DeepseekV3→mtp, GlmMoeDsa→deepseek_mtp 等 |
| 非 vllm/vllm_ascend 引擎 | "" | sglang/mindie 不用投机推理 |
| `spec_model_path` + eagle3 架构 | "eagle3" | 草稿模型路径 + eagle3 关键词 |
| `spec_model_path` + 普通架构 | "draft_model" | 草稿模型路径，非 eagle3 |
| Qwen3Next + vllm_ascend（无草稿模型）| "suffix" | Ascend 特化：强制 suffix |
| MTP 架构 + LMCACHE_OFFLOAD=true | "suffix" | lmcache 与 MTP 不兼容 |
| MTP 架构 + 无 lmcache | MTP method | 正常 MTP 路径 |
| 未知架构（无 MTP 映射）| "suffix" | 兜底策略 |

### 7.3 GAP-3：KV 稀疏双路径（待实现）

**文件**：`tests/test_unit_kv_sparse.py`（待实现）  
**函数**：`_build_kv_sparse_cmd(params, engine)`

| 架构 | 期望输出 |
|-----|---------|
| `GlmMoeDsaForCausalLM` | `--hf-overrides '{"index_topk_freq": 4}'` |
| `DeepseekV32ForCausalLM` | `--hf-overrides '{"index_topk_freq": 4}'` |
| `LlamaForCausalLM` | `engine_config["kv_cache_dtype"] = "fp8"` + `calculate_kv_scales=True` |
| `Qwen3ForCausalLM` | 同上 FP8 路径 |

> **审阅批注**：GAP-3 当前不再是待实现。仓库已有 `tests/test_unit_kv_sparse.py`，并与 `tests/test_unit_engine_select.py`、`tests/test_unit_mindie_fc.py` 合跑通过（55 passed）。另有 `tests/test_vllm_kv_sparse.py` 覆盖 `_build_kv_sparse_cmd()` 的 IndexCache 路径、FP8 路径、非 vLLM 跳过、与 speculative/kv offload 组合渲染。这里应改为“已实现，需确认覆盖是否重复/是否需要整合命名”。

### 7.4 GAP-4：引擎自动选择（待实现）

**文件**：`tests/test_unit_engine_select.py`（待实现）  
**规则**：Ascend 硬件 + `engine=vllm` → 自动升级为 `vllm_ascend`

| 输入 | 期望输出 |
|-----|---------|
| Ascend + vllm | vllm_ascend |
| NVIDIA + vllm | vllm（不变） |
| Ascend + vllm_ascend | vllm_ascend（不变） |

> **审阅批注**：文件当前已存在且抽跑通过，但它验证的实际行为与本表规则相反：`_resolve_engine_choice()` 对非空 `engine` 只调用 `_validate_user_engine()` 并原样返回，不会把用户显式传入的 `vllm` 自动升级为 `vllm_ascend`。自动选择只发生在 `engine` 缺失/空值时，且 Ascend 上可能选 `mindie` 或 `vllm_ascend`，取决于模型类型、算子加速、router、soft fp8 和架构白名单。测试计划应改成“engine 缺省自动选择”或先改实现。

### 7.5 GAP-5：Worker 节点不注入 host/port（待实现）

**文件**：`tests/test_integration_priority.py`（扩展）  
当 `distributed=True` 且 `node_rank > 0` 时：
- `merged` 中不应有 `host` / `port` 键
- `engine_config` 中不应有 `host` / `port` / `ipAddress` 键

> **审阅批注**：脚本层已经有部分覆盖：`tests/test_vllm_dp_deployment_script.py::test_deepseek_dp_deployment_rank1_is_headless` 断言 worker 脚本不含 `--host 10.254.124.178` / `--port 17000`。但它没有覆盖 `merged`/`engine_config` 字典层，所以这里应标为“字典层待实现，脚本层已有覆盖”，而不是完全待实现。

### 7.6 GAP-6：Shell 脚本语法验证

**文件**：`tests/test_unit_bash_syntax.py`（已实现）
对所有 10 个快照文件执行 `bash -n`，验证语法零报错。
在 bash 不可用的环境（如 Windows 裸机）自动跳过。

> **审阅批注**：当前测试“已实现但不稳”。本机 `shutil.which("bash")` 能找到 Windows 的 bash/WSL 启动器，但未安装默认 WSL distro，`bash -n` 返回 1；pytest 中还出现 GBK 解码线程异常。因此现在不能把 GAP-6 视为“已完成且通过”。建议 `_bash_available()` 改成实际运行 `bash --version` 或识别 `WSL_E_DEFAULT_DISTRO_NOT_FOUND` 后 skip，并显式设置 `encoding="utf-8", errors="replace"`。

---

### 7.7 GAP-7：MindIE Function Call 注入路径（待实现）

**文件**：`tests/test_unit_tcp.py`（扩展）或新建 `tests/test_unit_mindie_fc.py`
**背景**：MindIE 的 Function Call 不经过 vLLM `--tool-call-parser` CLI flag，而是通过 `_inject_function_call_config`（`mindie_adapter.py:1444`）将 `mindie_tool_call_parser` 写入 MindIE `config.json` 的 `ModelConfig[0].models.<type>.tool_call_options`。当前快照/单元体系对此路径**零覆盖**。

> **审阅批注**：这里“零覆盖”需改成“端到端覆盖不足”。`tests/test_unit_mindie_fc.py` 当前已存在且抽跑通过，覆盖 `_inject_function_call_config` 和 `_set_mindie_function_call` 的关键路径；`tests/test_mindie_distributed_env_defaults.py` 也已直接覆盖 `_build_model_config_overrides()` 对 deepseek_v31 注入和非 deepseek_v31 跳过的行为。真正剩余的缺口是 `load_and_merge_configs()` 级联路径，以及 SNAP-21/22 快照。

**关键测试场景**：

| 场景 | 输入 engine_config 关键字段 | 期望 overrides["models"] |
|-----|--------------------------|------------------------|
| deepseek_v31 注入 | `mindie_tool_call_parser="deepseek_v31"`, `mindie_model_type="deepseekv2"` | `{"deepseekv2": {"tool_call_options": {"tool_call_parser": "deepseek_v31"}}}` |
| deepseekv3 跳过 | `mindie_tool_call_parser="deepseekv3"`, `mindie_model_type="deepseekv2"` | `"models"` key 不存在 |
| enable_auto_tool_choice=False 剥离 | config 有 parser/model_type，但 engine_cmd_parameter 无 enable_auto_tool_choice | merged params 中 `mindie_tool_call_parser` 已被删除 → 不注入 |
| DeepseekV32 无 mindie 块 | ascend_default.json 的 `DeepseekV32ForCausalLM` 无 mindie/mindie_distributed 配置 | 使用 `llm.default.mindie` 规则降级，无 `mindie_tool_call_parser` |

**参考**：`tests/test_mindie_distributed_env_defaults.py:255-273` 已有部分覆盖，但仅测试 `_build_model_config_overrides` 直接调用，未测试 `_set_mindie_function_call` 剥离路径。

---

### 7.8 GAP-8：JSON 配置文件重复 key 自动检测（待实现）

**文件**：`tests/test_config_lint.py`（待实现）
**背景**：Issue #5/#6 展示 `json.load()` last-wins 语义会导致重复 key 静默丢弃配置。当前没有任何测试检测此问题。

| 测试函数 | 覆盖点 | 验证方式 |
|--------|--------|---------|
| `test_no_duplicate_keys_ascend_default` | `ascend_default.json` | `object_pairs_hook` 统计重复 key，assertions |
| `test_no_duplicate_keys_nvidia_default` | `nvidia_default.json` | 同上 |
| `test_no_duplicate_keys_all_defaults` | `config/defaults/*.json` | 遍历所有配置文件 |

**优先级**：P0（防退化）

> **审阅批注**：本节编号与 Issue 区域不一致：这里叫 GAP-8，Issue 注释里又叫 GAP-9。另，示例实现会把整棵 JSON 中所有层级的 key 放进同一个 `keys` 列表，`trust_remote_code`、`vllm_ascend` 等在不同对象中重复时会误报。重复 key 检测应在 `object_pairs_hook` 的每个对象局部检查，或携带路径后按对象作用域判断，而不是全文件扁平统计。

---

## 八、测试工程结构

```
tests/
├── snapshot_framework.py              # 快照测试基础设施（已完成）
├── test_script_snapshots.py           # 快照测试 SNAP-01~10（已完成）
├── snapshots/                         # Golden files（已生成，共 10 个）
│
├── test_unit_security.py              # UT-EV-01~12（已完成）
├── test_unit_tcp.py                   # UT-TCP-01~24（已完成）；待扩展 UT-TCP-25~29（MindIE）
├── test_unit_accel.py                 # GAP-1 Accel 特性收集（已完成）
├── test_unit_speculative.py           # GAP-2 投机推理策略（已完成）
├── test_unit_bash_syntax.py           # GAP-6 bash -n 语法验证（已完成）
├── test_unit_vllm_flags.py            # UT-VF-01~20（待实现）
├── test_unit_parallelism.py           # UT-TP-01~05（待实现）
├── test_unit_kv_sparse.py             # GAP-3 KV 稀疏双路径（待实现）
├── test_unit_engine_select.py         # GAP-4 引擎自动选择（待实现）
├── test_unit_mindie_fc.py             # GAP-7 MindIE FC 注入路径（待实现）⚠️ 新增
├── test_config_lint.py                # GAP-8 JSON 重复 key 检测（待实现）⚠️ 新增 P0
│
├── test_integration_model_variants.py # IT-MV-01~09 + GlmMoeDsa 全参数（待实现）
├── test_integration_priority.py       # IT-PRI-01~06 + GAP-5 worker 节点（待实现）
├── test_integration_features.py       # IT-SD/KV/RAG（待实现）
├── test_integration_cross_platform.py # IT-K2/MM（待实现）
│
└── TEST_PLAN.md                       # 本文档
```

> **审阅批注**：实际 `tests/` 中仍不存在 `test_unit_vllm_flags.py`、`test_unit_parallelism.py`、`test_config_lint.py`、四个 `test_integration_*.py`。`test_unit_kv_sparse.py`、`test_unit_engine_select.py`、`test_unit_mindie_fc.py` 当前已存在且合跑通过。另存在未列入本结构的 `test_vllm_kv_sparse.py`、`test_vllm_dp_deployment_script.py`、`test_config_loader_engine_selection.py`、`test_glm_moe_dsa_ascend_defaults.py`、`test_minimax_m2_ascend.py` 等，它们已经覆盖了部分“待实现”项。建议把“计划文件名”与“当前已有覆盖文件”分开列。

---

## 九、实现优先级

| 优先级 | 测试集 | 状态 | 覆盖价值 |
|--------|--------|------|---------|
| P0 | UT-EV-01~12（安全专项） | ✅ 已完成 | 极高（防命令注入） |
| P0 | UT-TCP-01~24（解析器映射） | ✅ 已完成 | 极高（模型兼容性） |
| P0 | GAP-1 Accel 特性收集 | ✅ 已完成 | 极高（EARS/IndexCache 逻辑） |
| P0 | GAP-2 投机推理策略解析 | ✅ 已完成 | 高（5条独立分支） |
| P0 | GAP-6 bash -n 语法验证 | ✅ 已完成 | 高（脚本生成正确性底线） |
| P0 | **GAP-8 JSON 重复 key 检测** | **待实现** | **极高（防 Issue #5/#6 退化）** |
| P0 | IT-K2-01~02（KimiK25 跨平台） | 待实现 | 高（已知缺失配置） |
| P1 | **GAP-7 MindIE FC 注入路径** | **待实现** | **高（零覆盖路径）** |
| P1 | **UT-TCP-25~29（MindIE parser 映射）** | **待实现** | **高（引擎间 FC 机制差异）** |
| P1 | UT-VF-01~20（CLI flag 转换） | 待实现 | 高（CLI 正确性） |
| P1 | IT-MV-01~09（模型变体覆盖） | 待实现 | 高（变体专属配置） |
| P1 | IT-PRI-01~06（优先级验证） | 待实现 | 高（配置分层正确性） |
| P2 | GAP-3 KV 稀疏双路径 | 待实现 | 中（IndexCache vs FP8） |
| P2 | GAP-4 引擎自动选择 | 待实现 | 中（Ascend 自动升级） |
| P2 | GAP-5 Worker 节点 host/port | 待实现 | 中（分布式正确性） |
| P2 | IT-SD-01~04（投机推理） | 待实现 | 中（高级特性） |
| P2 | GlmMoeDsa 全参数验证（含 seed/enable_npugraph_ex） | 待实现 | 中（Ascend 复杂配置） |
| P2 | **SNAP-21~22（MindIE FC 快照）** | **待实现** | **中（FC 注入回归）** |
| P2 | SNAP-11~20（补充快照） | 待实现 | 中（回归保护扩展） |
| P3 | IT-KV-01~04（KV 缓存策略） | 待实现 | 中 |
| P3 | IT-RAG-01~02（RAG 加速） | 待实现 | 低 |

> **审阅批注**：优先级表的状态需要拆成“文件存在 / 当前通过 / 覆盖完整”。当前 UT-EV 文件存在但有断言失败；GAP-6 文件存在但在 Windows+未初始化 WSL 下失败；GAP-3/GAP-4/GAP-7 的计划文件已存在并合跑通过，但 GAP-4 的通过测试与本表“Ascend + vllm 自动升级”规则不一致；GAP-5 有脚本层覆盖但缺字典层覆盖。

---

## 十、已发现问题追踪

| Issue # | 位置 | 描述 | 相关测试 |
|---------|------|------|---------|
| #1（已修复） | `wings_entry.py:_parse_env_file` | env key 未验证，存在命令注入风险 | UT-EV-06~08 |
| #2（已修复） | `nvidia_default.json` | `KimiK25ForConditionalGeneration` 已补 NVIDIA vLLM 专属配置 | IT-K2-02, UT-TCP-14K, SNAP-15 |
| #3 | `ascend_default.json:379-489` | `MiniMaxM2ForCausalLM` vllm_ascend=34816 vs mindie=4096，序列长度不对称 | IT-MM-01~03, SNAP-16 |
| #4（已修复） | `nvidia_default.json:467` | 缩进错误（11 空格→12 空格） | — |
| #5（已修复） | `ascend_default.json` | `GlmMoeDsaForCausalLM` 存在重复 JSON key，`json.load()` last-wins 导致第一条目（含 `quantization:ascend`、`seed:1024`、NPU 优化）被静默覆盖 | 无（需新增 GAP-9：JSON 重复 key 自动检测测试） |
| #6（已修复） | `ascend_default.json` | `KimiK25ForConditionalGeneration` 存在重复 JSON key，第一条（仅 vllm_ascend）被第二条（含 mindie）覆盖 | 无（同 GAP-9） |
| #7（已修复） | `ascend_default.json` + `nvidia_default.json` | `tool_call_parser: "deepseekv32"` → 应为 `"deepseek_v32"`；`reasoning_parser: "minimax_m2_reasoning"` → 应为 `"minimax_m2"`（不在 vLLM 注册表中） | UT-TCP-07, UT-TCP-13 |

> **审阅批注**：Issue 状态与当前测试不同步。#5 的配置修复已反映到 `ascend_default.json` 和 SNAP-07，但 `tests/test_glm_moe_dsa_ascend_defaults.py` 仍断言旧值 `max_num_seqs=8`。#7 的配置修复已反映到 `MiniMaxM2ForCausalLM.reasoning_parser=minimax_m2`，但 `tests/test_minimax_m2_ascend.py` 仍断言旧值 `minimax_m2_reasoning`。另外 #3 的行号范围已不稳定，应避免在测试计划里写死配置文件行号。

> ⚠️ **备注 [GAP-9]**：Issue #5/#6 暴露了配置文件中存在重复 JSON Key 的静默风险。建议新增一个轻量级 lint 测试：
> ```python
> # tests/test_config_lint.py
> def test_no_duplicate_json_keys():
>     """所有 defaults/*.json 不得含重复 key（json.load 会静默 last-wins）。"""
>     for path in DEFAULT_CONFIG_PATHS:
>         pairs = []
>         json.loads(path.read_text(), object_pairs_hook=lambda p: pairs.extend(p) or dict(p))
>         keys = [k for k, _ in pairs]
>         assert len(keys) == len(set(keys)), f"Duplicate keys in {path}: {[k for k in keys if keys.count(k) > 1]}"
> ```
> 此测试应列为 **P0**，防止合并操作引入的重复 key 再次被静默忽略。
