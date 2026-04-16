# Qwen3.5-27B 推理服务部署文档

## 机器环境

### 基础硬件

| 项目 | 详情 |
|------|------|
| 机器名 | ubuntu2204 |
| IP | 7.6.16.150 |
| OS | Ubuntu 22.04.5 LTS |
| 内核 | 5.15.0-171-generic (x86_64) |
| CPU | Intel Xeon Gold 6426Y × 2 Socket (16C/32T each, 共 64 逻辑核) |
| 内存 | 503 GiB |
| 系统盘 | /dev/sda2 1.8T (已用 468G, 可用 1.2T) |
| 数据盘 | /dev/nvme3n1p1 1.5T (已用 941G, 可用 452G) → 挂载 /data |

### GPU & 驱动

| 项目 | 详情 |
|------|------|
| NVIDIA Driver | **580.142** |
| CUDA Version | **13.0** (nvidia-smi 报告) |
| GPU 总数 | 6 张 (4× L20 + 2× RTX 4090) |

| GPU ID | 型号 | 显存 | 说明 |
|--------|------|------|------|
| 0 | NVIDIA L20 | 49140 MiB (48 GB) | ← 本次部署使用 |
| 1 | NVIDIA L20 | 49140 MiB (48 GB) | ← 本次部署使用 |
| 2 | NVIDIA GeForce RTX 4090 | 24564 MiB (24 GB) | |
| 3 | NVIDIA L20 | 49140 MiB (48 GB) | |
| 4 | NVIDIA L20 | 49140 MiB (48 GB) | |
| 5 | NVIDIA GeForce RTX 4090 | 23028 MiB (22.5 GB) | |

> ⚠️ 混合 GPU 环境，需 `NVIDIA_VISIBLE_DEVICES=0,1` 指定同型号 GPU，避免 TP 跨架构。

### 软件环境

| 项目 | 版本 |
|------|------|
| Docker | 29.2.1 (build a5c7197) |
| Python (宿主机) | 3.10.12 |
| fastapi (宿主机) | 0.135.3 |
| uvicorn (宿主机) | 0.44.0 |

### 可用 vLLM 镜像

| 镜像 | 大小 | 说明 |
|------|------|------|
| `vllm/vllm-openai:v0.17.0` | 30.2 GB | **本次部署使用** |
| `vllm/vllm-openai:v0.17.1` | 30.2 GB | 已验证可用 |
| `vllm/vllm-openai:v0.17.1-cu130` | 27.4 GB | CUDA 13.0 定制版 |
| `vllm/vllm-openai:v0.18.0` / latest | 32.2 GB | |
| `vllm/vllm-openai:v0.16.0` | 40.8 GB | |
| `vllm/vllm-openai:v0.15.0` / v0.15.1 | 40.6 / 29.5 GB | |
| `vllm/vllm-openai:v0.13.0` | 28.7 GB | |
| `vllm/vllm-openai:v0.12.0` | 28.6 GB | |
| `vllm/vllm-openai:v0.11.0` | 38.5 GB | |

### 模型信息

| 项目 | 详情 |
|------|------|
| 模型名 | Qwen3.5-27B |
| 路径 | `/data/models/Qwen3.5-27B` |
| 总大小 | 52 GB (11 safetensors shards) |
| 架构 | `Qwen3_5ForConditionalGeneration` |
| 精度 | bfloat16 |
| hidden_size | 5120 |
| intermediate_size | 17408 |
| num_hidden_layers | 64 |
| num_attention_heads | 24 |
| num_key_value_heads | 4 (GQA, 6:1 ratio) |
| head_dim | 256 |
| max_position_embeddings | 262144 (256K) |
| vocab_size | 248320 |
| 特殊层 | linear_attention + full_attention 交替（每 4 层 1 个 full_attention） |
| total_size (index) | 55,562,872,800 bytes |
| tensor总数 | 1,199 |

### 模型文件 SHA256 校验 (来自 7.6.16.150 已验证正常推理的副本)

**配置与Tokenizer文件：**

| 文件 | 大小 | SHA256 |
|------|------|--------|
| config.json | 4,134 | `f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c` |
| configuration.json | 51 | `2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0` |
| generation_config.json | 244 | `303aba891d66ab63908a7b3cc9163bcb835fdf8b9f6301c73216f3f1eb3992dd` |
| tokenizer_config.json | 16,710 | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| preprocessor_config.json | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| video_preprocessor_config.json | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| model.safetensors.index.json | 126,601 | `b3737e9d00bda0e37f0b873629d98bf9b407bef35735b9193c23d9844bcc96a6` |
| chat_template.jinja | 7,756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| merges.txt | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| vocab.json | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| tokenizer.json | 12,807,982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| README.md | 92,415 | `6ad083438390549c80214f3d4da70f33effac390cb9e8133ce85848e70912be0` |

**权重切片 (safetensors)：**

| 文件 | 大小 (bytes) | SHA256 |
|------|-------------|--------|
| model.safetensors-00001-of-00011 | 5,263,851,872 | `9019228d172c87d5603266c2d56672d119e838facffa164de803a1ebf0d716d2` |
| model.safetensors-00002-of-00011 | 5,347,741,440 | `890ef00c920b01c1c02755088c8d8ca5cdd6a1faa2a9a729eecd00857648b411` |
| model.safetensors-00003-of-00011 | 5,347,741,504 | `8aca03689ad0717fb91455809a6670eecca818bea8c0b6bd3a591a8807cc3223` |
| model.safetensors-00004-of-00011 | 5,347,741,504 | `20f539430c60fa611b3522b8408749a080b63f1fe034e551a6539ef864ca079a` |
| model.safetensors-00005-of-00011 | 5,347,741,504 | `57a0c074c654f05fc2d6b5112eeec387ac04986e928e42f502993e69aa03a49d` |
| model.safetensors-00006-of-00011 | 5,347,741,504 | `cfa4e6fbfc600854ef6c8e5465d0a1e43832a64b5e539f1e880333e3b1086703` |
| model.safetensors-00007-of-00011 | 5,347,741,496 | `d426963325b2319cb9e1442bc8b89b9a7748a8f7a907aa4041c11531cda6b014` |
| model.safetensors-00008-of-00011 | 5,368,714,128 | `fe60dbb9d25354c4eb2a9a59d9b1afd07741c8bebab5870f5eea6eadb4cf9a06` |
| model.safetensors-00009-of-00011 | 5,347,745,520 | `71a153d882242734a1fc7e000734727f00f6ec8ca70b044f5e44fcf97736ef8f` |
| model.safetensors-00010-of-00011 | 5,347,749,200 | `146745698b9f21940e2982beb1816a0eef3b80d77ab9cd884695b9a08c2697eb` |
| model.safetensors-00011-of-00011 | 2,148,512,760 | `d947ce7483c4109b55039f1359f4494d22390cf123568100abd89816802f097d` |

> 使用 `verify_qwen35_model.py` 脚本可一键验证：
> ```bash
> python3 verify_qwen35_model.py /path/to/Qwen3.5-27B --quick      # 快速模式（秒级）
> python3 verify_qwen35_model.py /path/to/Qwen3.5-27B --full       # 含权重SHA256（2-5分钟）
> python3 verify_qwen35_model.py /path/to/Qwen3.5-27B --fix-hint   # 带修复建议
> ```

---

## 部署信息

| 项目 | 详情 |
|------|------|
| 容器名 | `zhanghui-vllm-qwen35-27b` |
| 镜像 | `vllm/vllm-openai:v0.17.0` |
| GPU | GPU 0,1 (2× L20-48GB), TP=2 |
| 显存占用 | ~25.68 GiB |
| 服务端口 | `8000` |
| API 地址 | `http://7.6.16.150:8000/v1/chat/completions` |
| max_model_len | 4096 |
| 部署日期 | 2026-04-10 |

> 注：已测试 v0.17.0 和 v0.17.1 两个版本均可正常运行。当前使用 v0.17.0。

## 启动命令

```bash
docker run -d \
  --name zhanghui-vllm-qwen35-27b \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -v /data/models/Qwen3.5-27B:/model \
  --shm-size=16g \
  -p 8000:8000 \
  vllm/vllm-openai:v0.17.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3.5-27B \
  --max-model-len 4096 \
  --trust-remote-code
```

## 测试请求（curl）

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "你好，请简单介绍一下你自己"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## 测试请求（Python）

```python
#!/usr/bin/env python3
import urllib.request
import json

url = "http://7.6.16.150:8000/v1/chat/completions"
payload = {
    "model": "Qwen3.5-27B",
    "messages": [
        {"role": "user", "content": "你好，请简单介绍一下你自己"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=120)
result = json.loads(resp.read())
print(json.dumps(result, ensure_ascii=False, indent=2))
```

## 返回体示例（v0.17.0）

```json
{
  "id": "chatcmpl-91831ad9be5acfdf",
  "object": "chat.completion",
  "created": 1775835481,
  "model": "Qwen3.5-27B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "嗯，用户让我简单介绍一下自己。首先，我需要确定用户想知道什么...\n</think>\n\n你好！我是 Qwen3.5，是通义千问系列中最新推出的",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": null
      },
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 16,
    "total_tokens": 272,
    "completion_tokens": 256,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```

> 说明：模型输出包含思考链（`</think>` 标记），finish_reason 为 length 表示被 max_tokens=256 截断。

## 返回体示例（v0.17.1）

```json
{
  "id": "chatcmpl-9aee6a0e4896f73f",
  "object": "chat.completion",
  "created": 1775834953,
  "model": "Qwen3.5-27B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "嗯，用户让我简单介绍一下自己。首先，我需要明确自己的身份。我是 Qwen3.5，是通义千问系列的最新版本。用户可能想知道我的基本功能、能力和特点，但需要简洁，不要太冗长。\n\n接下来，我应该考虑用户的需求。他们可能是在初次接触，想了解我能做什么。所以需要突出关键能力，比如多语言支持、逻辑推理、代码生成、长上下文处理等。但因为是简单介绍，不能太详细，要分点但保持简短。\n\n还要注意语气友好，符合对话风格。避免使用技术术语过多，用通俗易懂的话。可能需要提到我是通义实验室研发的，增强可信度。同时，要说明应用场景，比如回答问题、创作、分析等，让用户知道实际用途。\n\n需要检查是否有遗漏的重要点，比如上下文窗口大小、知识截止时间，但可能用户不需要这么详细的数据，保持简洁。另外，要强调我的多语言能力和对复杂任务的支持，比如文档分析、代码生成。\n\n最后，确保回答结构清晰，分点列出，但用自然的方式表达。可能还要邀请用户提问，促进互动。需要避免错误信息，比如版本号是否正确，功能是否准确。确认 Qwen3.5 的特性，比如上下文 256K，支持",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": null
      },
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 16,
    "total_tokens": 272,
    "completion_tokens": 256,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```

> 说明：v0.17.1 的输出未包含 `</think>` 标记（思考链未显式截断），其余结构与 v0.17.0 一致。

## 两版本对比

| 对比项 | v0.17.0 | v0.17.1 |
|--------|---------|---------|
| 镜像 | `vllm/vllm-openai:v0.17.0` | `vllm/vllm-openai:v0.17.1` |
| 启动时间 | ~2 min | ~2 min |
| 推理正常 | ✅ | ✅ |
| 输出含 `</think>` | 是 | 否（在 256 token 内未出现） |
| 返回结构 | 相同 | 相同 |

## 常用管理命令

```bash
# 查看容器日志
docker logs -f zhanghui-vllm-qwen35-27b

# 查看容器状态
docker ps --filter name=zhanghui-vllm-qwen35-27b

# 停止容器
docker stop zhanghui-vllm-qwen35-27b

# 启动容器
docker start zhanghui-vllm-qwen35-27b

# 删除容器
docker rm -f zhanghui-vllm-qwen35-27b

# 查看 GPU 占用
nvidia-smi
```
---

## Wings-Control Sidecar 项目验证

### 验证架构

使用 `20260407/wings_control` 项目的 Sidecar 架构，在 ubuntu2204 上完成端到端验证：

```
┌─ wings_control (宿主机 Python 进程) ─────────────┐
│  wings_start.sh → python -m wings_control         │
│    ├── proxy    :18000 (uvicorn, 16 workers)       │
│    ├── health   :19000 (uvicorn)                   │
│    ├── monitor  :19100 (uvicorn)                   │
│    └── 生成 start_command.sh → /shared-volume/     │
├────────────────────────────────────────────────────┤
│  vLLM Engine (Docker: vllm/vllm-openai:v0.17.0)   │
│    └── 读取 start_command.sh 启动推理引擎 :17000   │
└────────────────────────────────────────────────────┘
```

### 启动命令

**1. 宿主机启动 wings_control：**

```bash
# 安装依赖
pip3 install fastapi uvicorn httpx orjson python-dotenv

# 部署代码
cp -r wings_control /opt/wings-control
ln -sfn /opt/wings-control /opt/wings_control
chmod +x /opt/wings-control/wings_start.sh
dos2unix /opt/wings-control/**/*   # 如果从 Windows 传输需转换换行符

# 启动
export WINGS_SKIP_PID_CHECK=true
export SHARED_VOLUME_PATH=/shared-volume
export ENABLE_REASON_PROXY=true
nohup bash /opt/wings-control/wings_start.sh \
  --model-name Qwen3.5-27B \
  --model-path /data/models/Qwen3.5-27B \
  --engine vllm \
  --device-count 2 \
  > /tmp/wings_control.log 2>&1 &
```

**2. Docker 启动 vLLM 引擎（执行 wings_control 生成的脚本）：**

```bash
docker run -d \
  --name zhanghui-wings-engine \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  --network host \
  -v /shared-volume:/shared-volume \
  -v /data/models/Qwen3.5-27B:/data/models/Qwen3.5-27B \
  -v /var/log/wings:/var/log/wings \
  --shm-size=16g \
  --entrypoint bash \
  vllm/vllm-openai:v0.17.0 \
  /shared-volume/start_command.sh
```

### wings_control 生成的 vLLM 参数

| 参数 | 值 | 来源 |
|------|------|------|
| `--trust-remote-code` | (flag) | nvidia_default.json 模型配置 |
| `--compilation-config` | `'{"cudagraph_mode":"PIECEWISE"}'` | nvidia_default.json 模型配置 |
| `--port` | 17000 | wings_control 默认 backend port |
| `--tensor-parallel-size` | 2 | `--device-count 2` |
| `--max-model-len` | 5120 | config_loader 合并 (input_length + output_length) |
| `--gpu-memory-utilization` | 0.9 | argparse 默认值 / ENV |
| `--max-num-batched-tokens` | 4096 | vllm_default.json |
| `--block-size` | 16 | vllm_default.json |
| `--max-num-seqs` | 32 | argparse 默认值 / ENV |
| `--served-model-name` | Qwen3.5-27B | CLI `--model-name` |
| `--dtype` | auto | argparse 默认值 |
| `--kv-cache-dtype` | auto | argparse 默认值 |
| `--seed` | 0 | argparse 默认值 / ENV |

#### nvidia_default.json 中未出现在 CLI 中的字段

| 字段 | nvidia_default.json 值 | 处理方式 |
|------|------------------------|----------|
| `task` | `"generate"` | 被 `_prepare_engine_config` 显式 pop（新版 vLLM 已弃用 `--task` 参数） |
| `reasoning_parser` | `"qwen3"` | 被 `_set_function_call` pop（因 `enable_auto_tool_choice` 未启用） |

### 配置优先级 Bug 修复 (config_loader.py `_set_common_params`)

**问题**：`_set_common_params()` 中 argparse 默认值（如 `trust_remote_code=False`、`compilation_config=""`）无条件覆盖了 nvidia_default.json 中的模型专属配置，导致 `--trust-remote-code` 和 `--compilation-config` 未出现在生成的 start_command.sh 中。

**根因**：`_set_common_params` 无法区分「用户显式传参」和「argparse 默认值」，两者在 `engine_cmd_parameter` 中均为非 None。

**修复**：引入 `_detect_explicit_cli_keys()` 区分显式设置与默认值：
- 用户显式设置的参数（CLI 或环境变量）→ 始终覆盖模型配置
- 模型配置中已有的值 → 保留（不被 argparse 默认值覆盖）
- 模型配置中不存在的参数 → 用 argparse 默认值补充

```python
# 修复后的 _set_common_params (config_loader.py)
def _set_common_params(params, engine_cmd_parameter, config_path):
    vllm_param_map_config = _load_mapping(config_path, 'default_to_vllm_parameter_mapping')
    explicit_keys = _detect_explicit_cli_keys()
    for key, value in vllm_param_map_config.items():
        if not value:
            continue
        cli_val = engine_cmd_parameter.get(key)
        if cli_val is None:
            continue
        if key in explicit_keys:          # 用户显式设置 → 始终覆盖
            params[value] = cli_val
        elif value not in params:          # 模型配置不存在 → 用默认值补充
            params[value] = cli_val
        # 否则：保留模型配置中的值
```

**修复效果**（start_command.sh 5400 → 5548 bytes）：

| 参数 | 修复前 | 修复后 |
|------|--------|--------|
| `--trust-remote-code` | ❌ 缺失 | ✅ 已添加 |
| `--compilation-config` | ❌ 缺失 | ✅ 已添加 |

### 验证结果 (2026-04-11)

| 测试项 | 端口 | 结果 | 说明 |
|--------|------|------|------|
| Engine Direct | 17000 | **PASS** | vLLM 引擎直连，推理正常 |
| Proxy | 18000 | **PASS** | wings_control 代理转发，推理正常 |
| Health | 19000 | **PASS** | 返回 `{"s":1,"p":"ready","engine_alive":true,"backend_ok":true}` |
| Progress File | /shared-volume/progress.jsonl | **PASS** | 46条进度记录，最终 100% completed |

**部署进度追踪（log_analyzer）：**
- 5% → wings_control_init
- 22-30% → engine初始化 (NCCL, TP rank分配)
- 32-60% → 模型加载 (11/11 shards, 7.3秒, 25.68 GiB)
- 62-73% → 模型编译 (torch.compile, 34.24秒)
- 100% → 启动终态 (总耗时 134秒)

**推理结果示例（通过 proxy 18000）：**

```json
{
  "model": "Qwen3.5-27B",
  "usage": {"prompt_tokens": 15, "completion_tokens": 128},
  "choices": [{"message": {"content": "..."}}]
}
```

### Sidecar 架构验证要点

1. **脚本生成**：wings_control 根据 `--engine vllm --device-count 2 --model-name Qwen3.5-27B` 自动生成 5400 字节的 `start_command.sh`
2. **配置合并**：4层配置 (CLI → 环境变量 → nvidia_default.json → vllm_default.json) 正确合并
3. **模型识别**：自动识别架构 `Qwen3_5ForConditionalGeneration`，应用对应默认配置
4. **进程管理**：proxy(16 workers), health, monitor_proxy 三个子进程自动启动
5. **进度监控**：log_analyzer 实时分析引擎日志，生成 progress.jsonl（46条记录）
6. **健康检查**：`/health` 端点返回引擎存活状态、后端连通性、延迟等信息

---

## 模型推理输出乱码排查指南（通用）

> 基于 vLLM GitHub Issues 全网搜索（Open 171+ / Closed 199+），涵盖 Qwen3.5、Gemma4、GLM、Nemotron、Kimi-K2.5 等多个模型系列的共性问题。
> 搜索日期：2026-04-10 | 涉及 vLLM v0.13 ~ v0.19+ / sglang / transformers

---

### 一、模型文件层面

#### 1.1 模型文件损坏 / 不完整

**影响范围**：所有模型，所有推理框架

**常见诱因**：
- 下载中断导致 safetensors 文件不完整（最常见）
- NFS / 网络存储的静默数据腐败（bit-rot）
- 多来源拼凑的模型文件版本不一致（config.json 与权重不匹配）
- 容器挂载路径错误导致读取到空文件或旧版本

**典型现象**：
- 模型**能加载但推理乱码**（最隐蔽的情况——header 正常但权重数据损坏）
- 加载时报 tensor shape mismatch 错误

**排查**：

```bash
# 通用方法：对比文件大小和 SHA256
sha256sum /path/to/model/*.safetensors
# 与 HuggingFace Hub 或已验证环境的校验值对比

# 本项目 Qwen3.5-27B 专用：
python verify_qwen35_model.py --model-dir /data/models/Qwen3.5-27B --quick   # 10秒
python verify_qwen35_model.py --model-dir /data/models/Qwen3.5-27B --full    # 3-5分钟
```

#### 1.2 量化检查点与推理框架不匹配

**代表 Issue**：
- [#39407](https://github.com/vllm-project/vllm/issues/39407) — Gemma4 FP8_BLOCK 双重缩放，输出 `" a a a a a..."`
- [#39049](https://github.com/vllm-project/vllm/issues/39049) — Gemma4 FP8 dynamic quantization = gibberish
- [#38197](https://github.com/vllm-project/vllm/issues/38197) — Qwen3.5-dense wfp8afp8 per-tensor 量化在 vLLM 乱码，sglang 正常
- [#36094](https://github.com/vllm-project/vllm/issues/36094) — Qwen3.5 NVFP4 Checkpoint 精度极差
- [#36337](https://github.com/vllm-project/vllm/issues/36337) — Kimi-K2.5 MXFP4 在 ROCm(gfx950) 上 gibberish

**根因**：
- FP8_BLOCK 量化的 activation scale 已"吸收"进权重，推理时又动态 per-token 量化 → 双重缩放
- 量化格式（NVFP4 / MXFP4 / wfp8afp8）在特定 vLLM 版本或硬件上存在 kernel 适配问题
- 不同推理框架（vLLM vs sglang）对同一量化格式的处理逻辑不一致

**典型现象**：
- 输出为单个 token 无限重复（`" a a a a"` 或 `"!!!!!!!!!!"`)
- 所有 prompt 产生相同乱码模式
- 换非量化版本（bf16/fp16）就正常

**排查**：
- 确认量化格式是否被你的 vLLM 版本支持（参考 vLLM release notes）
- 先用**原始 bf16 权重**验证推理正常，再切换量化版本
- 不同推理框架交叉验证（vLLM vs sglang）

---

### 二、KV Cache 层面

#### 2.1 FP8 KV Cache 在 MLA / 混合架构模型上乱码

**代表 Issue**：
- [#38652](https://github.com/vllm-project/vllm/issues/38652) — `--kv-cache-dtype fp8` 在 MLA 模型（GLM-4.7-Flash）多轮对话时 garbage
- [#37554](https://github.com/vllm-project/vllm/issues/37554) — `--calculate-kv-scales` 在 hybrid GDN+Attention 模型（Qwen3.5）上产生 corrupted FP8 KV cache

**根因**：
- FP8 KV cache 对 MLA（Multi-head Latent Attention）的 latent vectors 缺少正确的 per-tensor scaling
- 多轮对话时量化误差逐轮累积，conversation 越长越乱
- 混合架构（GDN + full attention）的两类层对 KV scale 需求不同

**典型现象**：
- **单轮正常，多轮乱码**（最典型的 FP8 KV cache 症状）
- 对话历史越长越乱，短问答看不出问题

**排查**：

```bash
# 去掉 fp8 kv cache
# 原来的：--kv-cache-dtype fp8
# 改为：  --kv-cache-dtype auto（或直接不加此参数）
```

#### 2.2 KV Block 调度器腐败（并发场景）

**代表 Issue**：
- [#39146](https://github.com/vllm-project/vllm/issues/39146) — base scheduler KV block corruption，temperature=0 输出不确定
- [#38606](https://github.com/vllm-project/vllm/issues/38606) — rapid LoRA adapter alternation 下 KV block corruption

**根因**：
- scheduler 的 KV block 分配/释放逻辑 bug → 不同请求共享同一 KV block
- prefix caching 在特定并发/LoRA 切换场景下产生脏数据

**典型现象**：
- `temperature=0` 输出不确定（多次请求结果不同）
- 高并发偶发乱码，单请求正常
- LoRA 切换后第一次推理乱码

**排查**：
- 单请求测试确认基线
- 禁用前缀缓存：`--enable-prefix-caching false`
- 如果用 LoRA：降低切换频率或重启服务

---

### 三、投机解码（Speculative Decoding）层面

#### 3.1 Ngram/Suffix 投机解码在混合架构（SSM/GDN）模型上 output corruption

**代表 Issue**：
- [#39273](https://github.com/vllm-project/vllm/issues/39273) — Qwen3.5 ngram spec decode → corrupted output | **修复 PR**: [#39463](https://github.com/vllm-project/vllm/pull/39463)
- [#36872](https://github.com/vllm-project/vllm/issues/36872) — Qwen3.5-35B-A3B-FP8 speculative decoding → gibberish + throughput collapse

**根因**：
- SSM/GDN 的循环状态（recurrent state）在 draft token 被拒绝后未正确回滚
- 非投机解码 kernel 从 position 0 读取变质的 stale state
- MTP 方式不受影响（状态管理路径不同），ngram/suffix 均受影响

**典型现象**：
- 输出开始正常，逐渐退化为重复/截断片段
- 吞吐量随时间下降（state leak 导致越来越多的错误累积）

**排查**：

```bash
grep -i "speculative" /shared-volume/start_command.sh
# 有 ngram 或 suffix → 高度嫌疑
# 临时禁用投机解码测试
```

#### 3.2 Thinking 模式 + 投机解码组合问题

**代表 Issue**：
- [#39104](https://github.com/vllm-project/vllm/issues/39104) — Qwen3.5 思考模式下生成随机词流

**现象**：`<think>` 模式先输出正常推理，然后突然变成无意义词流。

**高风险参数组合**：
- `--speculative-config` + `--kv-cache-dtype fp8` + `--reasoning-parser`
- MTP + fp8 KV cache 在混合架构上触发状态不一致

---

### 四、推理框架 / 编译 层面

#### 4.1 vLLM 版本 Regression

**代表 Issue**：
- [#39223](https://github.com/vllm-project/vllm/issues/39223) — Nemotron3 super 在 v0.19.0 corrupted，v0.18.1 正常
- [#39179](https://github.com/vllm-project/vllm/issues/39179) — GLM5 在 B300 上 garbage output（已 closed，版本 bug）

**排查**：

```bash
python -c "import vllm; print(vllm.__version__)"
# 对比已验证正常的版本。在 150 上验证通过的是 v0.17.0
```

**建议**：固定 vLLM 版本，不要在生产环境用 nightly。

#### 4.2 编译优化导致长 prompt 乱码（-O3 / torch.compile）

**代表 Issue**：
- [#37732](https://github.com/vllm-project/vllm/issues/37732) — `-O3` 编译级别 + FlashInfer workspace 在 profiling 后被 invalidated → 长 prompt garbage

**根因**：
- `-O3` 在 `__init__` 阶段预分配 FlashInfer GPU workspace
- 模型 profiling 阶段清理/重组 GPU 内存 → workspace 成为悬空指针
- 短 prompt 恰好不触及损坏区域，长 prompt（>~2048 tokens）必乱

**典型现象**：
- 短 prompt 正常，**长 prompt 乱码**（临界点约 2048-2500 tokens）
- `temperature=0` 仍然乱码（确认是 logit 而非 sampling 问题）

**排查**：
- 去掉 `-O3`，使用默认编译级别
- 或添加 `--enforce-eager` 禁用 torch.compile

#### 4.3 FLA（Flash Linear Attention）Tensor 格式不匹配

**代表 Issue**：
- [#38643](https://github.com/vllm-project/vllm/issues/38643) — Qwen3.5 FLA ops 收到 head-first 格式 `[B,H,T,...]`，但期望 `[B,T,H,...]` → gibberish

**根因**：
- Qwen3.5 的 GDN 层使用 FLA kernel
- 特定 vLLM 版本（nightly）中 tensor layout 传参错误
- 无论换 attention-backend、dtype、enforce-eager 都无效

**典型现象**：
- Qwen3.5 全面乱码（不是退化，而是从一开始就乱）
- 日志中有 `Input tensor shape suggests potential format mismatch` 警告

**排查**：查看推理日志是否有 FLA format mismatch 警告。

---

### 五、长时间运行劣化

**代表 Issue**：
- [#35718](https://github.com/vllm-project/vllm/issues/35718) — Kimi-K2.5 on 8×H200 运行一段时间后 garbled
- 同 issue 评论：Qwen3.5-397B-A17B-FP8 on 4×B200 "fine at first then degrades over time"

**根因**：
- reasoning parser 未正确处理隐式结束（如 `</think>` 被跳过）→ tool call markers 泄漏到 reasoning 字段
- 内存泄漏或 GPU 状态累积导致长期劣化
- 多请求之间的上下文污染（cross-request context contamination）

**典型现象**：
- 服务**刚启动正常**，运行若干小时/天后出现乱码
- 只发生在 reasoning/thinking 字段
- **重启 vLLM 后恢复**

**排查**：
- 检查 reasoning parser 是否匹配模型（如 Kimi-K2 需要专用 parser）
- 设置自动重启策略（cron / health check 触发restart）
- 添加 `--tokenizer-mode 'hf'` 尝试

---

### 六、硬件 / 平台 层面

#### 6.1 GPU 型号特定问题

**代表 Issue**：
- [#38994](https://github.com/vllm-project/vllm/issues/38994) — Qwen3.5 9B on **Intel Backend** 重复/乱码
- [#38718](https://github.com/vllm-project/vllm/issues/38718) — NVFP4 MoE 在 SM120（RTX 5080）+ CPU weight offloading → garbage
- [#36999](https://github.com/vllm-project/vllm/issues/36999) — CPU offloading + flashinfer autotuner → gibberish
- [#36337](https://github.com/vllm-project/vllm/issues/36337) — MXFP4 在 MI350X (gfx950 ROCm 7.2) → gibberish

**共性**：
- 新硬件（Blackwell SM12.0、Intel XPU、AMD MI350X）的 kernel 支持不完善
- CPU offloading 与特定 attention kernel 组合有 bug
- TP 模式下混合不同型号 GPU → 计算结果不一致

#### 6.2 驱动 / CUDA 版本不匹配

**排查**：

```bash
nvidia-smi                  # GPU 型号、驱动版本
nvcc --version              # CUDA toolkit 版本
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA 版本
# ↑ 三者需要兼容
```

#### 6.3 跨请求上下文污染（多节点 Pipeline Parallel）

**代表 Issue**：
- [#38903](https://github.com/vllm-project/vllm/issues/38903) — async scheduling + pipeline parallelism on multi-node → 请求间上下文交叉

---

### 七、Safetensors 权重加载顺序

**代表 Issue**：
- [#38991](https://github.com/vllm-project/vllm/issues/38991) — `runai_safetensors_weights_iterator` 以非确定性顺序 yield tensors → FP8 推理在某些平台上破坏

**根因**：权重加载顺序不确定 → FP8 scale tensor 和 weight tensor 对应关系错乱。

---

### 快速排查清单（通用版）

| 优先级 | 步骤 | 命令 / 方法 | 针对哪些模型 |
|--------|------|------------|-------------|
| **P0** | 模型文件 SHA256 校验 | `sha256sum` 对比源头 / `verify_qwen35_model.py` | 所有 |
| **P1** | 用**最简参数**裸跑引擎 | 去掉量化/投机解码/编译优化/KV-fp8 | 所有 |
| **P2** | 确认 vLLM 版本与已验证环境一致 | `python -c "import vllm; print(vllm.__version__)"` | 所有 |
| **P3** | 检查是否启用投机解码 | `grep -i speculative start_command.sh` | Qwen3.5 / MoE / SSM 混合架构 |
| **P4** | 去掉 `--kv-cache-dtype fp8` | 用默认 auto | MLA 模型（GLM/DeepSeek）/ 混合架构 |
| **P5** | 去掉 `-O3`，改用 `--enforce-eager` | 禁用 torch.compile | 长 prompt 乱码 |
| **P6** | 单请求 vs 并发对比 | curl 单请求确认基线 | 并发乱码场景 |
| **P7** | 禁用 prefix caching | `--enable-prefix-caching false` | 高并发 / LoRA 切换 |
| **P8** | 对比 GPU 型号、驱动版本 | `nvidia-smi` | TP/PP 多卡 |
| **P9** | 换框架交叉验证 | sglang serve / transformers generate | 判断是框架 bug 还是模型/数据 bug |

### 最小可复现测试命令（通用模板）

```bash
# 最简启动（去掉所有优化参数）
vllm serve /path/to/model \
  --dtype auto \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 4096 \
  --tensor-parallel-size <GPU_COUNT> \
  --port 8000

# 最简测试（单请求、低 max_tokens、temperature=0 确保可复现）
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"temperature":0}'
```

---

### 附录：受影响模型分布与危险场景统计

> 数据来源：vLLM GitHub Issues（截至 2026-04-10），Open 171+ / Closed 199+

#### 按模型系列统计

| 模型系列 | 受影响的具体模型 | Issue 数 | 架构特点 |
|---------|----------------|---------|---------|
| **Qwen3.5** | 27B, 4B, 9B, 35B-A3B, 397B-A17B, 122B-A10B | 7+ | 混合 GDN+Attention（最高发） |
| **Gemma 4** | 31B (FP8/BF16), 27B | 3+ | softcap + 异构 head dim |
| **GLM** | GLM-4.7-Flash, GLM5, GLM-5.1-FP8 | 3+ | MLA 架构 |
| **Nemotron** | Nemotron3 super, NemotronH 120B | 3+ | 混合 Mamba+Transformer |
| **Kimi** | K2.5 (INT4/MXFP4), K2 | 3+ | MoE + 专用 reasoning parser |
| **GPT-OSS** | 120B (MXFP4) | 2 | Blackwell SM12 特有 |
| **Nemotron-Cascade** | 30B-A3B (NVFP4) | 1 | MoE + CPU offloading |
| **通用 Llama/Qwen3** | 各种 | 散见 | 版本 regression |

#### 场景危险等级

| 触发条件 | 危险等级 | 涉及模型范围 |
|---------|---------|------------|
| 投机解码（ngram/suffix）+ 混合架构 | **极高** | Qwen3.5, NemotronH |
| FP8 KV cache + MLA 架构 | **极高** | GLM, DeepSeek |
| FP8/NVFP4 量化 + 新硬件（Blackwell SM12） | **高** | 所有量化模型 |
| -O3 编译 + 长 prompt（>2K tokens） | **高** | 所有模型 |
| 长时间不重启服务（数小时/天级别） | **中** | Kimi-K2.5, Qwen3.5-397B, 大 MoE 模型 |
| 高并发 + prefix caching | **中** | 所有模型 |
| LoRA 快速切换 | **中** | 所有 LoRA 微调模型 |
| 非 NVIDIA 平台（Intel XPU / AMD ROCm） | **中** | 所有模型 |

#### 稳定性梯度

```
最稳定 ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ → 最易乱码

纯 Transformer    加量化       加投机解码    加新硬件     长时间运行
+ BF16           (FP8/NVFP4)  (ngram/MTP)   (SM12/ROCm)  + 高并发
+ 单卡                                                    + LoRA 切换
────────────── → ─────────── → ─────────── → ─────────── → ──────────
  极少乱码         偶发问题      混合架构      kernel 适配    状态管理
                                容易出问题     不全           bug 暴露
```

#### 核心结论

1. **不是某个模型的问题**，而是推理框架在"新架构 × 新量化 × 新硬件"三重组合下的成熟度不均匀
2. **混合架构**（GDN+Attention / Mamba+Transformer / MoE）是最高危群体
3. **量化**（FP8/NVFP4/MXFP4）是第二大诱因，几乎每个 vLLM 版本都有新的量化 bug
4. **最安全的配置**：原始 bf16 权重 + 无投机解码 + 无 FP8 KV cache + enforce-eager + 固定 vLLM 版本