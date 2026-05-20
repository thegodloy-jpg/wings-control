> 状态：历史专项记录。当前正式部署口径请以 [../README.md](../README.md)、[../docs/deployment/docker-compose.md](../docs/deployment/docker-compose.md) 和 [../docs/deployment/k8s.md](../docs/deployment/k8s.md) 为准；`wings_start.sh` 支持的启动项优先使用 CLI 字段。

使用 vLLM v0.19 (NVIDIA GPU) 拉起 GLM-5.1-FP8 模型的启动命令。

## 调研结论

### 1. 模型架构信息

| 项目 | 值 |
|------|-----|
| 模型名称 | GLM-5.1-FP8 |
| 模型架构 | `GlmMoeDsaForCausalLM` |
| 架构特点 | MoE + DSA (差异化稀疏注意力)，混合 KV Cache |
| vLLM 内置支持 | **否** — 需要 `--trust-remote-code` |
| 量化方式 | FP8 权重量化（模型自带，vLLM 自动识别 `quantization_config`） |

### 2. 启动命令（NVIDIA GPU，单机 8 卡 TP=8）

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/GLM-5.1-FP8 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --trust-remote-code \
    --block-size 64 \
    --hf-overrides '{"index_topk_freq": 4}' \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --port 8000
```

### 3. 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model` | `/path/to/GLM-5.1-FP8` | 本地模型路径，替换为实际路径 |
| `--tensor-parallel-size` | `8` | 8 卡张量并行（MoE 大模型必需）|
| `--max-model-len` | `4096` | 最大序列长度（可按需调大，DSA 架构理论支持更长） |
| `--trust-remote-code` | - | `GlmMoeDsaForCausalLM` 不在 vLLM 内置架构列表，**必须开启** |
| `--block-size` | `64` | IndexCache (FLASHMLA_SPARSE) 后端要求 block_size=64 |
| `--hf-overrides` | `'{"index_topk_freq": 4}'` | DSA 稀疏注意力 IndexCache 加速参数 |
| `--tool-call-parser` | `glm47` | GLM 系列工具调用解析器（不需要可去掉） |
| `--reasoning-parser` | `glm45` | GLM 系列推理标签解析器（不需要可去掉） |

### 4. 验证命令

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
    "max_tokens": 128
  }'
```

### 5. 重要注意事项

1. **`--trust-remote-code` 必须开启** — `GlmMoeDsaForCausalLM` 不在 vLLM v0.19 官方内置架构列表中（官方仅内置 `Glm4ForCausalLM`、`Glm4MoeForCausalLM`），模型代码通过 HuggingFace 远程加载。
2. **不要加 `--kv-cache-dtype fp8`** — GLM-5.1 的 DSA 架构属于混合 KV Cache 架构，使用 IndexCache 策略而非 FP8 KV Cache。误用会导致 `ValueError`（不同层的 KV Cache Spec 类型不一致）。
3. **FP8 权重量化是模型自身属性**，权重已是 FP8 格式，vLLM 会从模型目录下 `config.json` 的 `quantization_config` 字段自动识别，无需手动指定。
4. **`--tool-call-parser glm47` 和 `--reasoning-parser glm45`** 是可选的，仅在需要工具调用（function calling）或推理标签解析时才需要。基本推理场景可去掉。

### 6. 社区实践验证

> 已通过 vLLM v0.19.0 源代码交叉验证，以下参数均为**开源社区标准做法**，非私有定制。

| 参数 | 来源 | 社区验证状态 |
|------|------|-------------|
| `--trust-remote-code` | vLLM 标准机制 | ✅ `GlmMoeDsaForCausalLM` 不在 v0.19 内置表中，社区部署必须开启 |
| `--block-size 64` | DSA 架构要求 | ✅ IndexCache/FLASHMLA_SPARSE 后端硬性要求 |
| `--hf-overrides '{"index_topk_freq": 4}'` | DSA 架构参数 | ✅ 模型自身的稀疏注意力配置，从模型设计层面决定 |
| `--tool-call-parser glm47` | vLLM 原生注册 | ✅ 已在 `vllm/tool_parsers/__init__.py` 中注册为 `Glm47MoeModelToolParser` |
| `--reasoning-parser glm45` | vLLM 原生注册 | ✅ 与 tool_parsers 同期注册，命名约定一致 |
| `--tensor-parallel-size 8` | 通用做法 | ✅ 261B MoE 大模型标准 TP 配置 |

**结论**: 该启动命令完全由 vLLM v0.19 原生能力组成，不依赖任何自研组件。社区用户按此命令即可部署。

> **vLLM v0.19 推荐写法**（等价，使用 `vllm serve` 替代 `python -m`）：
> ```bash
> vllm serve /path/to/GLM-5.1-FP8 \
>     --tensor-parallel-size 8 \
>     --max-model-len 4096 \
>     --trust-remote-code \
>     --block-size 64 \
>     --hf-overrides '{"index_topk_freq": 4}' \
>     --tool-call-parser glm47 \
>     --reasoning-parser glm45 \
>     --port 8000
> ```

### 7. 社区 Open Issues（截至 2025-07）

> 以下是 vLLM GitHub 上与 GLM-5/GLM-5.1-FP8 相关的 **NVIDIA GPU** 侧 open issues，部署前建议关注：

| Issue | 标题 | 影响范围 |
|-------|------|---------|
| [#39757](https://github.com/vllm-project/vllm/issues/39757) | GLM-5 tool calls in stream mode get error tool name | 流式模式下 tool calling 返回错误工具名 |
| [#39614](https://github.com/vllm-project/vllm/issues/39614) | GLM-5.1-FP8: tool result content replaced with `<tools>` tag | `--chat-template-content-format auto` 时 tool result 内容被替换 |
| [#39574](https://github.com/vllm-project/vllm/issues/39574) | glm4_moe_tool_parser crash on `/v1/responses` streaming | tool parser 在流式 responses API 中崩溃（影响 GLM-4.5/4.7/5.1） |
| [#39211](https://github.com/vllm-project/vllm/issues/39211) | FP8 MoE ep_scatter Triton illegal-address on H200 | H200 GPU 上 FP8 prefill 路径地址越界 |
| [#38911](https://github.com/vllm-project/vllm/issues/38911) | tool_choice='required' + PD disaggregation internal error | PD 分离部署 + 强制 tool calling 时内部错误 |
| [#38652](https://github.com/vllm-project/vllm/issues/38652) | `--kv-cache-dtype fp8` produces garbage on MLA models | ⚠️ 印证本文档"不要加 `--kv-cache-dtype fp8`"的建议 |

**风险评估**：
- **基础推理（不使用 tool calling）**：低风险，可正常部署
- **Tool Calling 场景（流式）**：存在 3 个相关 bug（#39757, #39614, #39574），建议关注修复进展或暂用非流式
- **H200 GPU**：FP8 路径有已知问题（#39211），H100/A100 未见报告

