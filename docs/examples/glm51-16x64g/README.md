# GLM-5.1 16×64GB 示例文件

本目录提供 GLM-5.1-FP8 在 16 张 64GB Ascend 卡上的 Compose / K8s 启动示例。

- [docker-compose.glm51-16x64g.yml](docker-compose.glm51-16x64g.yml)：Docker Compose 三容器模板。
- [glm51-16x64g.yaml](glm51-16x64g.yaml)：K8s Deployment + Service 模板。

使用前必须按实际环境替换镜像仓库、镜像版本、模型宿主机路径、Ascend 设备挂载、CANN 路径、K8s NPU 资源名和端口规划。

## 容量口径

| 项 | 值 |
|----|----|
| 总显存 | 16 × 64GB = 1024GB |
| `gpu_memory_utilization` | 0.95 |
| 引擎预算 | 约 972.8GB |
| 模型权重 | 约 764GB |
| 理论剩余 | 约 208.8GB |
| 建议稳态在途 token | 约 19 万 tokens |

按 `zai-org/GLM-5.1-FP8` 的 `config.json`，关键结构字段如下：

| 字段 | 值 |
|------|----|
| `architectures` | `GlmMoeDsaForCausalLM` |
| `num_hidden_layers` | `78` |
| `hidden_size` | `6144` |
| `num_attention_heads` / `num_key_value_heads` | `64` / `64` |
| `head_dim` | `64` |
| `kv_lora_rank` / `qk_rope_head_dim` | `512` / `64` |
| `index_n_heads` / `index_head_dim` | `32` / `128` |
| `max_position_embeddings` | `202752` |
| `quantization_config.quant_method` | `fp8` |

边界估算：

| 口径 | 每 token cache 估算 | 208.8GB 理论 token 上限 | 180GB 安全 token 上限 | 说明 |
|------|---------------------|--------------------------|-----------------------|------|
| 传统 KV | 约 1.22MiB | 约 16.3 万 | 约 14.1 万 | `78 × 2 × 64 × 64 × 2B`，保守下限 |
| DSA latent cache | 约 87.8KiB | 约 232 万 | 约 200 万 | `78 × (512 + 64) × 2B`，仅压缩 latent cache |
| DSA + 全量 index cache | 约 711.8KiB | 约 28.6 万 | 约 24.7 万 | `78 × ((512 + 64) + 32 × 128) × 2B`，偏保守 |

本示例按“DSA + 全量 index cache”的偏保守口径落档：`16384 × 12 = 196608` 在途 token，低于 180GB 安全 token 上限；`16384 × 16 = 262144` 接近理论上限，更适合作为压测上沿。

本示例使用稳态起步配置：

| 字段 | 值 | 说明 |
|------|----|------|
| `--input-length` | `12288` | 输入长度 |
| `--output-length` | `4096` | 输出长度 |
| `max_model_len` | `16384` | 由输入和输出长度相加得到 |
| `--max-num-seqs` | `12` | 稳态起步并发序列数 |
| `--max-num-batched-tokens` | `8192` | 控制 prefill 峰值 |
| `--gpu-memory-utilization` | `0.95` | 显存利用率上限 |

可按压测结果调整：

| 目标 | 建议字段 |
|------|----------|
| 吞吐优先 | `--input-length 6144 --output-length 2048 --max-num-seqs 24` |
| 稳态生产 | `--input-length 12288 --output-length 4096 --max-num-seqs 12` |
| 长上下文 | `--input-length 24576 --output-length 8192 --max-num-seqs 6` |
| 极限验证 | `--input-length 49152 --output-length 16384 --max-num-seqs 2` |
| 上沿压测 | `--input-length 12288 --output-length 4096 --max-num-seqs 16` |

## Docker Compose

```bash
docker compose -f docs/examples/glm51-16x64g/docker-compose.glm51-16x64g.yml up -d
docker compose -f docs/examples/glm51-16x64g/docker-compose.glm51-16x64g.yml logs -f wings-control engine
```

查看生成的引擎命令：

```bash
docker compose -f docs/examples/glm51-16x64g/docker-compose.glm51-16x64g.yml exec wings-control cat /shared-volume/start_command.sh
```

## K8s

```bash
kubectl apply -f docs/examples/glm51-16x64g/glm51-16x64g.yaml
kubectl rollout status deploy/glm51-16x64g
kubectl logs -f deploy/glm51-16x64g -c wings-control
```

## 验证

```bash
curl http://127.0.0.1:19000/health
curl http://127.0.0.1:18000/v1/models
```

## 注意事项

- GLM-5.1-FP8 属于 `GlmMoeDsaForCausalLM` / IndexCache 场景，不建议盲目添加 `--kv-cache-dtype fp8`。
- 16×64GB + 764GB 权重的空间主要受 KV Cache、图编译峰值和通信 buffer 影响，建议从本示例的 `16384 × 12` 起步压测。
- 如果启动阶段 OOM，优先降低 `--max-num-seqs`，其次降低 `--input-length` / `--output-length`。
- 如果吞吐不足，先在 `8192 × 24` 档位测试，再逐步提高并发。
