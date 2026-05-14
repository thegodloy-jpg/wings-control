# vLLM / vLLM-Ascend 高级特性 start_command 验证

> 目的：明确 `engine=vllm` 与 `engine=vllm_ascend` 场景下，开启各高级特性后，最终写入共享卷的 `start_command.sh` 中会出现哪些关键片段，尤其是 `install.py --features` 的 engine key、补丁安装命令和 `python3 -m vllm.entrypoints.openai.api_server` 启动参数。

---

## 1. 验证结论速览

| 特性 | vLLM (`engine=vllm`) | vLLM-Ascend (`engine=vllm_ascend`) |
|------|----------------------|------------------------------------|
| 基线启动 | `python3 -m vllm.entrypoints.openai.api_server ...` | 复用 `vllm_adapter`，同样启动 `python3 -m vllm.entrypoints.openai.api_server ...`，但前面会先加载 Ascend/CANN 环境 |
| 投机推理（无草稿模型） | 先执行 `install.py --install-runtime-deps`；再生成 `WINGS_ENGINE_PATCH_OPTIONS={"vllm": ... "ears" ...}`；启动命令追加 `--speculative-config` | 先执行 `install.py --install-runtime-deps`；再生成 `WINGS_ENGINE_PATCH_OPTIONS={"vllm_ascend": ... "ears" ...}`；启动命令追加 `--speculative-config` |
| KV 稀疏：IndexCache 架构 | 生成 `WINGS_ENGINE_PATCH_OPTIONS={"vllm": ... "indexcache" ...}`；启动命令追加 `--hf-overrides '{"index_topk_freq": 4}'` | 不生成 sparse CLI 参数；不生成 `--features indexcache` |
| KV 稀疏：非 IndexCache 架构 | 不走 `install.py --features`；启动命令把 `--kv-cache-dtype` 改为 `fp8` 并追加 `--calculate-kv-scales` | 不生成 sparse CLI 参数；启动命令基本等同基线 |
| LMCache KV 卸载 | 执行 `install.py --lmcache-target nvidia-x86`；启动命令追加 LMCache `--kv-transfer-config` | 执行 `install.py --lmcache-target ascend-arm`；启动命令追加 Ascend 版 LMCache `--kv-transfer-config` |
| RAG 加速 | 不改变 engine 启动命令；主要影响 proxy/RAG 路由和 `advanced_features.json` | 同左 |
| Function Call | 不是 wings-accel 补丁特性；只有模型配置中存在 parser 字段时才会注入对应 vLLM CLI 参数 | 同左 |

关键修正点：`vllm_ascend` 在 `install.py --features` 的 JSON 顶层 key 中保持 `vllm_ascend`，不再改写成 `vllm`。

---

## 2. start_command.sh 的通用结构

最终脚本不是一条单独命令，而是由 `core/wings_entry.py` 组装出的完整 shell 脚本，顺序如下：

1. shebang、安全选项、日志目录、Prometheus multiproc 目录。
2. log analyzer 启动进度监控。
3. stdout/stderr tee 到 `engine-full.log` 和过滤后的 `engine.log`。
4. faulthandler / Triton NPU / ModelSlim Quarot patch preamble。
5. `config/env_overrides` 用户环境变量覆盖。
6. wings-accel 安装片段：`install.py --install-runtime-deps` / `--lmcache-target` / `--features`。
7. 引擎启动命令：`python3 -m vllm.entrypoints.openai.api_server ... &`。
8. 进程监控与 fallback/retry 逻辑。

注意：`vllm_adapter.build_start_script()` 原始脚本多以 `exec python3 ...` 结尾，但 `wings_entry._build_pid_tracked_script()` 会把最后一条 `exec ...` 转换为后台启动的 `python3 ... &`，并注入 `ENGINE_PID=$!`。因此 `start_command.sh` 中会同时看到：

- `[wings-cmd] >>> ...` 的打印行；
- 实际执行的 `python3 -m vllm.entrypoints.openai.api_server ... &` 行；
- 开启高级特性时，还会出现一份禁用特性的 fallback 命令。

---

## 3. 基线启动命令

### 3.1 vLLM 基线

验证得到的核心命令形态：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --host <pod-ip> \
  --port 17000 \
  --served-model-name <model-name> \
  --model <model-path> \
  --trust-remote-code \
  --dtype auto \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 4096 \
  --block-size 16 \
  --max-num-seqs 32 \
  --seed 0 \
  --max-model-len 5120 \
  --tensor-parallel-size <device-count> &
```

### 3.2 vLLM-Ascend 基线

核心命令形态与 vLLM 相同，仍由 `vllm_adapter` 生成：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --host <pod-ip> \
  --port 17000 \
  --served-model-name <model-name> \
  --model <model-path> \
  --trust-remote-code \
  --dtype auto \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 4096 \
  --block-size 16 \
  --max-num-seqs 32 \
  --seed 0 \
  --max-model-len 5120 \
  --tensor-parallel-size <device-count> &
```

差异在命令前置环境：`vllm_ascend` 会加载 `set_vllm_ascend_env.sh` / Ascend 环境，并可能根据模型架构注入 `VLLM_USE_V1`、`VLLM_ASCEND_*` 等 Ascend 专用变量。

---

## 4. 投机推理（ENABLE_SPECULATIVE_DECODE）

### 4.1 vLLM：无草稿模型，DeepSeek MTP 示例

输入条件：

- `engine=vllm`
- `--enable-speculative-decode`
- 模型架构：`DeepseekV3ForCausalLM`
- 未设置 `SPECULATIVE_DECODE_MODEL_PATH`

关键 preamble：

```bash
python3 /accel-volume/install.py --install-runtime-deps

export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "<ENGINE_VERSION>", "features": ["ears"]}}'
python3 /accel-volume/install.py --features "$WINGS_ENGINE_PATCH_OPTIONS"

export VLLM_EARS_TOLERANCE=0.5
```

关键 engine 命令：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 3}' &
```

fallback 命令会移除 `--speculative-config`，并在重启前执行：

```bash
unset WINGS_ENGINE_PATCH_OPTIONS
unset VLLM_EARS_TOLERANCE
```

### 4.2 vLLM-Ascend：无草稿模型，Qwen3Next suffix 示例

输入条件：

- `engine=vllm_ascend`
- `--enable-speculative-decode`
- 模型架构：`Qwen3NextForCausalLM`
- 未设置 `SPECULATIVE_DECODE_MODEL_PATH`

关键 preamble：

```bash
python3 /accel-volume/install.py --install-runtime-deps

export WINGS_ENGINE_PATCH_OPTIONS='{"vllm_ascend": {"version": "<ENGINE_VERSION>", "features": ["ears"]}}'
python3 /accel-volume/install.py --features "$WINGS_ENGINE_PATCH_OPTIONS"

export VLLM_EARS_TOLERANCE=0.5
```

关键 engine 命令：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --speculative-config '{"method" : "suffix", "num_speculative_tokens": 5, "suffix_decoding_max_cached_requests": 1000}' &
```

这里最关键的是：`WINGS_ENGINE_PATCH_OPTIONS` 的顶层 key 是 `vllm_ascend`，不是 `vllm`。

### 4.3 有草稿模型路径时

如果设置 `SPECULATIVE_DECODE_MODEL_PATH` / `--speculative-decode-model-path`，则策略改为 draft model / eagle3：

```bash
python3 /accel-volume/install.py --install-runtime-deps

python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --speculative-config '{"model": "<draft-model-path>", "draft_tensor_parallel_size": 1, "method" : "draft_model", "num_speculative_tokens": 4}' &
```

若草稿模型 config 中识别到 eagle3 架构，则 `method` 变为 `eagle3`。草稿模型路径模式不会因为投机推理本身注入 `ears` feature；只有无草稿模型的 MTP/suffix 策略会触发 `ears`。

---

## 5. KV 稀疏（ENABLE_SPARSE）

### 5.1 vLLM + IndexCache 架构

输入条件：

- `engine=vllm`
- `--enable-sparse`
- 模型架构属于 `INDEXCACHE_ARCHS`，例如 `DeepseekV32ForCausalLM` 或 `GlmMoeDsaForCausalLM`

关键 preamble：

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "<ENGINE_VERSION>", "features": ["indexcache"]}}'
python3 /accel-volume/install.py --features "$WINGS_ENGINE_PATCH_OPTIONS"
```

关键 engine 命令：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --hf-overrides '{"index_topk_freq": 4}' &
```

fallback 命令会移除 `--hf-overrides`，退回基线启动命令。

### 5.2 vLLM + 非 IndexCache 架构（FP8 KV Cache 路径）

输入条件：

- `engine=vllm`
- `--enable-sparse`
- 模型架构不属于 `INDEXCACHE_ARCHS`

验证得到的命令变化：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --kv-cache-dtype fp8 \
  --calculate-kv-scales &
```

这个路径不生成 `WINGS_ENGINE_PATCH_OPTIONS`，也不执行 `install.py --features`。fallback 命令会把 `--kv-cache-dtype` 恢复到基线值 `auto`，并移除 `--calculate-kv-scales`。

### 5.3 vLLM-Ascend + ENABLE_SPARSE

当前 `vllm_adapter._build_kv_sparse_cmd()` 只对 `engine == "vllm"` 生效。因此：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --kv-cache-dtype auto \
  ... &
```

不会生成：

```bash
--hf-overrides '{"index_topk_freq": 4}'
--kv-cache-dtype fp8
--calculate-kv-scales
python3 /accel-volume/install.py --features ... indexcache ...
```

但是 `enable_sparse=True` 仍会被 `_has_advanced_features()` 识别为高级特性，因此脚本中仍会出现高级特性 fallback 监控逻辑。也就是说，vLLM-Ascend 下该开关目前不会改变主 engine 启动参数，但会改变监控/fallback 结构。

---

## 6. LMCache KV 卸载（LMCACHE_OFFLOAD）

### 6.1 vLLM

输入条件：

- `engine=vllm`
- `LMCACHE_OFFLOAD=true`
- 示例中同时设置了 `LMCACHE_LOCAL_CPU=true`，因此会生成 `LMCACHE_CONFIG_FILE`

关键 preamble：

```bash
python3 /accel-volume/install.py --lmcache-target nvidia-x86

export LMCACHE_OFFLOAD=true
export LMCACHE_CONFIG_FILE=<shared-volume>/lmcache_config.yaml
```

关键 engine 命令：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' &
```

fallback 命令会移除 `--kv-transfer-config`，但仍可能保留部分 `LMCACHE_*` 环境打印；真正影响 engine CLI 的 offload 参数会被移除。

### 6.2 vLLM-Ascend

输入条件：

- `engine=vllm_ascend`
- `LMCACHE_OFFLOAD=true`
- 示例中同时设置了 `LMCACHE_LOCAL_CPU=true`，因此会生成 `LMCACHE_CONFIG_FILE`

关键 preamble：

```bash
python3 /accel-volume/install.py --lmcache-target ascend-arm

export LMCACHE_OFFLOAD=true
export LMCACHE_CONFIG_FILE=<shared-volume>/lmcache_config.yaml
```

关键 engine 命令：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","engine_id":"lmca1","kv_buffer_device":"npu"}' &
```

与 vLLM 的差异：

- patch target 从 `nvidia-x86` 变为 `ascend-arm`；
- `kv_transfer_config` 额外带 `engine_id=lmca1` 和 `kv_buffer_device=npu`。

---

## 7. RAG 加速（ENABLE_RAG_ACC）

RAG 加速是 proxy 层特性，不是 engine CLI 特性。验证结果显示：开启 `--enable-rag-acc` 后，engine 主命令仍保持基线形态：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --kv-cache-dtype auto \
  ... &
```

它的影响点在：

1. `config_loader._set_rag_acc_config()` 设置 `RAG_ACC_ENABLED=true`。
2. `proxy.gateway` 在 `/v1/chat/completions` 流式请求中判断 RAG/Dify 场景并走 `rag_acc_chat()`。
3. `advanced_features.json` 的 `rag_acc` 字段会记录启用状态。
4. 只有 RAG 时不触发 `_has_advanced_features()` 的 engine fallback；它不会注入 `install.py --features`。

---

## 8. Function Call（ENABLE_AUTO_TOOL_CHOICE）

`ENABLE_AUTO_TOOL_CHOICE` 不属于 `wings_entry._has_advanced_features()` 管理的三类 engine 高级特性，也不会触发 wings-accel patch 安装。

它的行为由 `config_loader` 决定：

- 用户开启 `enable_auto_tool_choice`；
- 模型配置中存在 `tool_call_parser` / `reasoning_parser` 等字段；
- 则这些字段会保留或注入到 `engine_config`，最终体现为 vLLM CLI 参数；
- 如果模型没有 parser 配置，则会移除 `enable_auto_tool_choice` 并打印 warning。

因此它不是固定输出项，不能像投机推理或 LMCache 那样只凭开关推导出固定 `start_command.sh` 片段。

---

## 9. 多特性组合规则

### 9.1 preamble 顺序

当多个特性同时开启时，`_build_accel_preamble()` 的顺序固定：

1. 投机推理 runtime deps：

```bash
python3 /accel-volume/install.py --install-runtime-deps
```

2. LMCache patch target：

```bash
python3 /accel-volume/install.py --lmcache-target <nvidia-x86|ascend-arm>
```

3. `install.py --features` 批量安装：

```bash
export WINGS_ENGINE_PATCH_OPTIONS='<merged-json>'
python3 /accel-volume/install.py --features "$WINGS_ENGINE_PATCH_OPTIONS"
```

4. 如果批量安装失败，逐 feature fallback：

```bash
python3 /accel-volume/install.py --features '<single-feature-json>'
```

### 9.2 vLLM 组合示例

如果同时开启：

- `ENABLE_SPECULATIVE_DECODE=true`，且无草稿模型 → `ears`
- `ENABLE_SPARSE=true`，且模型是 IndexCache 架构 → `indexcache`

则自动合并为：

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "<ENGINE_VERSION>", "features": ["ears", "indexcache"]}}'
```

### 9.3 vLLM-Ascend 组合示例

如果 `vllm_ascend` 开启无草稿模型的投机推理，则自动生成：

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm_ascend": {"version": "<ENGINE_VERSION>", "features": ["ears"]}}'
```

不会再改写为：

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "<ENGINE_VERSION>", "features": ["ears"]}}'
```

### 9.4 用户手动传 WINGS_ENGINE_PATCH_OPTIONS

如果用户或页面直接传入 `WINGS_ENGINE_PATCH_OPTIONS`，代码会优先使用该 JSON；同时会把运行时判定出的 required features 合并到当前 engine 对应 key 下。

当前映射为：

```python
{
    "vllm": "vllm",
    "vllm_ascend": "vllm_ascend",
}
```

因此页面如果手动构造 vLLM-Ascend 的 patch options，也应使用 `vllm_ascend` 顶层 key。

---

## 10. 验证方法

本文件基于本地直接调用 `build_launcher_plan()` 验证，使用临时模型目录写入最小 `config.json`，覆盖以下场景：

| 场景 | engine | 设备 | 模型架构 | 开关 |
|------|--------|------|----------|------|
| vLLM baseline | `vllm` | NVIDIA | `Qwen3ForCausalLM` | 无 |
| vLLM speculative | `vllm` | NVIDIA | `DeepseekV3ForCausalLM` | `--enable-speculative-decode` |
| vLLM sparse IndexCache | `vllm` | NVIDIA | `DeepseekV32ForCausalLM` | `--enable-sparse` |
| vLLM sparse FP8 | `vllm` | NVIDIA | `Qwen3ForCausalLM` | `--enable-sparse` |
| vLLM LMCache | `vllm` | NVIDIA | `Qwen3ForCausalLM` | `LMCACHE_OFFLOAD=true` |
| vLLM RAG | `vllm` | NVIDIA | `Qwen3ForCausalLM` | `--enable-rag-acc` |
| vLLM-Ascend baseline | `vllm_ascend` | Ascend910B | `Qwen3NextForCausalLM` | 无 |
| vLLM-Ascend speculative | `vllm_ascend` | Ascend910B | `Qwen3NextForCausalLM` | `--enable-speculative-decode` |
| vLLM-Ascend sparse | `vllm_ascend` | Ascend910B | `Qwen3NextForCausalLM` | `--enable-sparse` |
| vLLM-Ascend LMCache | `vllm_ascend` | Ascend910B | `Qwen3NextForCausalLM` | `LMCACHE_OFFLOAD=true` |
| vLLM-Ascend RAG | `vllm_ascend` | Ascend910B | `Qwen3NextForCausalLM` | `--enable-rag-acc` |

验证要点：

1. `vllm_ascend` 的 `WINGS_ENGINE_PATCH_OPTIONS` 顶层 key 保持 `vllm_ascend`。
2. `vllm` 的 IndexCache sparse 走 `--features indexcache` + `--hf-overrides`。
3. `vllm` 的非 IndexCache sparse 走 `--kv-cache-dtype fp8 --calculate-kv-scales`，不走 `install.py --features`。
4. `vllm_ascend` 的 sparse 开关不改变 engine CLI。
5. LMCache 通过 `--lmcache-target` 区分 `nvidia-x86` 与 `ascend-arm`。
6. RAG 不改变 engine CLI。
