# 多模态支持（Qwen VL 和 Hunyuan Video）

## 混元 Video 和 Qwen2.5-VL 模型支持

### 需求背景

多模态理解场景虽有推理引擎但尚未接入统一服务化框架，需进行整合支持；同时多模态生成场景目前缺乏专用的推理引擎和服务化支持，需要自行开发以适配MaaS平台。

### 需求价值

支持Qwen2.5-VL多模态理解与混元Video多模态生成。

### 需求详情

**多模态理解**

- 支持 Qwen2.5-VL-7B 和 Qwen2.5-VL-72B 模型。
- 提供标准 OpenAI 接口，可直接通过 API 调用多模态理解能力。
- 兼容 x86 和 Arm 平台。

**多模态生成**

- 支持混元Video模型，实现文本到视频的生成能力。
- 基于 PyTorch + Transformer 方案，服务化启动，可通过 API 请求生成内容。
- 兼容 x86 和 Arm 平台。

---

## 实现设计

**多模态理解：** 面向 Qwen2.5-VL-7B 和 Qwen2.5-VL-72B 两个模型，提供基于 OpenAI 标准接口（`/v1/chat/completions` 和 `/v1/completions`）的多模态理解服务；支持用户通过 POST 请求提交文本与图片输入，其中图片可通过 URL（如 `"image": "http://path/to/your/image.jpg"`）或 Base64 编码（如 `"image": "data:image;base64,/9j/..."`）的方式上传；并可在 x86（GPU）和 Arm（Ascend NPU）平台灵活部署，满足多平台、多场景的应用需求。

**设计方案：** 针对 `/v1/chat/completions` 和 `/v1/completions`，沿用现有 vllm 和 mindie 的实现。

**多模态生成：** 针对混元video模型，构建基于 PyTorch + Transformer 方案的 FastAPI 服务端；提供快速响应的API接口（`/v1/videos/text2video`，`/v1/videos/text2video/{taskid}`）；并可在 x86（GPU）和 Arm（Ascend NPU）平台灵活部署，满足多平台、多场景的应用需求。

### 服务端设计

- **数据结构：** 任务表；`task_id → {状态, 进度, 参数, 结果列表, 错误等}`
- **POST接口**（`/v1/videos/text2video`）
  - 验证请求
  - 生成 task_id
  - 写入任务表
  - 启动后台 worker
  - 立即返回 task_id
- **后台 worker**（顺序执行）
  - 更新状态为"运行中"
  - 获取任务级互斥（手动设定并发阈值，以控制后端同时处理的任务数量）
  - 执行"单次多视频生成函数"（过程内持续更新状态/错误）
  - 释放任务级互斥
- **单次多视频生成函数**
  - 输入：提示词、分辨率、帧数、每个提示词生成视频数 K、seed等
  - 推理过程：一次前向推理，批量生成 K 段视频，并返回对应元数据
  - 任务状态：仅在推理过程中显示进行中，推理结束后显示完成或失败
  - 说明：采用单次推理实现多视频生成，提升一致性与效率，整个推理阶段对外仅暴露"进行中 / 完成 / 失败"等关键状态
- **GET接口**（`/v1/videos/text2video/{taskid}`）
  - 返回任务状态、错误信息和（若有）结果文件列表及 URL

### 启动脚本新增

- 对变量 `model_type` 进行判定，若为 `MultiModel Generate`，则将 17000 端口以及 Host 分配给多模态后端，启动多模态后端，原本 wings 不启动。
- 启动 `wings_proxy`，等待用户请求。

---

## 接口设计

### 多模态理解的接口

#### `/v1/chat/completions`

**请求示例 1：图片通过 HTTP URL 提交**

```bash
curl -X POST 'http://127.0.0.1:18000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen2.5-VL-7B",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "http://example.com/path/to/your/image.jpg"},
          {"type": "text",  "text":  "Describe this image and answer: What is the largest animal?"}
        ]
      }
    ]
  }'
```

**请求示例 2：图片以 Base64 编码内嵌提交**

```bash
curl -X POST 'http://127.0.0.1:18000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen2.5-VL-7B",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."},
          {"type": "text",  "text":  "Describe this image and answer: What is the largest animal?"}
        ]
      }
    ]
  }'
```

#### `/v1/completions`

**请求示例 1：图片通过 HTTP URL 提交**

```bash
curl -X POST 'http://127.0.0.1:18000/v1/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen2.5-VL-7B",
    "prompt": [
      {"type": "image", "image": "http://example.com/path/to/your/image.jpg"},
      {"type": "text",  "text":  "Describe this image and answer: What is the largest animal?"}
    ]
  }'
```

**请求示例 2：图片以 Base64 编码内嵌提交**

```bash
curl -X POST 'http://127.0.0.1:18000/v1/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen2.5-VL-7B",
    "prompt": [
      {"type": "image", "image": "data:image/jpeg;base64,/9j/4AAQABAAD..."},
      {"type": "text",  "text":  "Describe this image and answer: What is the largest animal?"}
    ]
  }'
```

**返回示例**

```json
{
  "id": "chatcmpl-1",
  "model": "Qwen2.5-VL-7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "This is a large blue whale, the largest animal."},
          {"type": "structured", "data": {"objects": [{"label": "whale", "confidence": 0.98}]}}
        ]
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

### 多模态生成的接口

#### `/v1/videos/text2video`

**请求示例**

```bash
curl -X POST "http://127.0.0.1:18000/v1/videos/text2video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "一只熊猫在竹林里吃竹子，阳光透过竹叶洒下斑驳的光影",
    "resolution": "720x720",
    "frames": 129,
    "seed": -1,
    "num_infer_steps": 50,
    "num_videos_per_prompt": 1
  }'
```

> `num_videos_per_prompt`：单个提示词生成视频的数目

**返回示例**

```json
{
  "task_id": "d7cf2b12098748e2a3f9a88b7a2e6c65",
  "task_status": "in_queued",
  "message": "The video generation task has been submitted. Please use the task_id to check the task status."
}
```

#### `/v1/videos/text2video/{taskid}`

**请求示例**

```bash
curl -X GET "http://127.0.0.1:18000/v1/videos/text2video/d7cf2b12098748e2a3f9a88b7a2e6c65"
```

**返回示例 — 任务状态说明**

| 状态码 | 含义说明 |
| --- | --- |
| `in_queue` | 任务已提交，等待执行 |
| `running` | 任务正在处理 |
| `done` | 处理完成，成功 |
| `failed` | 执行失败（附错误说明） |
| `notfound` | 任务已过期或不存在 |

**`in_queue`（排队中）**

```json
{
  "task_id": "d7cf2b12098748e2a3f9a88b7a2e6c65",
  "task_status": "in_queue",
  "task_info": {
    "video_url": null,
    "error": null
  },
  "message": "Task has been submitted and is queued for processing."
}
```

**`running`（正在处理）**

```json
{
  "task_id": "d7cf2b12098748e2a3f9a88b7a2e6c65",
  "task_status": "running",
  "task_info": {
    "video_url": null,
    "error": null
  },
  "message": "Task is currently being processed."
}
```

**`done`（处理完成）**

```json
{
  "task_id": "d7cf2b12098748e2a3f9a88b7a2e6c65",
  "task_status": "done",
  "task_info": {
    "video_url": "http://xxxx",
    "error": null
  },
  "message": "Task has been completed."
}
```

**`failed`（任务失败）**

```json
{
  "task_id": "d7cf2b12098748e2a3f9a88b7a2e6c65",
  "task_status": "failed",
  "task_info": {
    "video_url": null,
    "error": "Insufficient GPU memory during inference. (Exception caught)"
  },
  "message": "Task execution failed. Please check the error details for more information."
}
```

**`notfound`（任务过期/无效）**

```json
{
  "task_id": "d7cf2b12098748e2a3f9a88b7a2e6c65",
  "task_status": "notfound",
  "task_info": {
    "video_url": null,
    "error": null
  },
  "message": "Invalid task ID."
}
```
