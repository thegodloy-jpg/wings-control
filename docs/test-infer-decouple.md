# wings-infer 控制服务层解耦 — 展开测试要求

> **精简原则**：多机多卡验证已隐含单机多卡；多卡 TP 验证已隐含单卡。
> 类似地，三容器部署已包含双容器。因此采用**递进式覆盖**，优先验证最复杂场景，单独列出仅需独立验证的差异点。

## 一、测试环境与资源

| 维度 | 内容 |
|------|------|
| **硬件** | L20 (NV)、H20 (NV)、910B A2/B3/B4 (Ascend)、910C A3 (Ascend) |
| **模型** | GLM-4.7、Qwen3.5-27B、Qwen3.5-397B-A17B (MoE)、MiniMax-M2.5、DeepSeek-V3.2、DeepSeek-V4 |
| **引擎** | vLLM (`vllm`)、vLLM-Ascend (`vllm_ascend`)、SGLang (`sglang`)、MindIE (`mindie`) |

---

## 二、部署架构与拓扑验证

> 三容器完整部署（control + engine + accel）在多机多卡拓扑下验证，隐含覆盖：双容器、单机多卡、单机单卡。

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-2.1 | 三容器多机 Ray 部署 | 共享卷写入（权限 644 + 原子写入）、accel 卷挂载、head/worker Ray 启动、HCCL 环境 | 服务正常，`/health` 返回 200，补丁安装成功 |
| T-2.2 | MoE 模型自动 PP+TP | Qwen3MoeForCausalLM + Ray 多机 → `--pipeline-parallel-size` = 节点数，`--tensor-parallel-size` = 卡数 | PP+TP 混合并行正常 |
| T-2.3 | PD 分离 | PD_ROLE=prefill/decode，kv_transfer_config 注入正确 connector | PD 分离模式正常 |
| T-2.4 | DP 模式 (独立验证) | dp_deployment 后端，`data-parallel-address/rank` 正确注入 | DP 多副本服务正常 |
| T-2.5 | Worker 节点端口 | node_rank>0 时 Worker 不暴露 proxy | 仅启动 health 进程 |
| T-2.6 | 容器启动顺序容错 | 随机打乱容器启动顺序 | 最终服务正常 |
| T-2.7 | 控制容器崩溃重启 | 模拟 OOMKill 后 K8s 重启 | 重新生成脚本，proxy/health 恢复 |

---

## 三、多引擎适配与命令校验

### 3.1 引擎选择与四引擎命令

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-3.1 | Ascend + vllm → 自动升级 | `_auto_select_engine` 升级 vllm → vllm_ascend，CANN 环境加载，算子加速注入 | engine 为 vllm_ascend |
| T-3.2 | NV + vllm 命令生成 | 参数名 snake_case→kebab-case、布尔 flag、JSON 单引号包裹、空值/None 跳过、`shlex.quote` 转义 | 命令格式完全正确 |
| T-3.3 | SGLang 参数翻译 | `gpu_memory_utilization → mem_fraction_static` 等通用名映射 | 翻译正确 |
| T-3.4 | MindIE 配置生成 | service_config.json 正确生成、set_mindie_env.sh 加载 | MindIE 正常启动 |

### 3.2 非法参数与边界

| 编号 | 测试用例 | 输入 | 预期结果 |
|------|---------|------|---------|
| T-3.5 | 不支持的引擎名 | engine=unknown | 错误提示 |
| T-3.6 | max_num_batched_tokens ≤ 0 | 值为 0 或负数 | 跳过 + WARNING 日志 |
| T-3.7 | Shell 注入防护 | model_path 含 `; rm -rf /` | 转义，无注入 |
| T-3.8 | 内部参数过滤 | use_kunlun_atb=True | 不传递给 CLI |

---

## 四、配置合并与优先级验证

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-4.1 | 四层优先级 | 硬件默认 < 模型默认 < config-file < CLI | CLI 最终覆盖所有层 |
| T-4.2 | CONFIG_FORCE=true | 跳过硬件默认 | 仅用用户配置 |
| T-4.3 | 引擎参数映射 | engine_parameter_mapping.json 翻译后进入 engine_config | `enable_sparse` 等无映射项保持顶层 |
| T-4.4 | kv_transfer_config 注入层级 | LMCache/PD 注入的值 | 位于 `engine_config` 嵌套字典中 |

---

## 五、历史特性继承与功能验证

### 5.1 KV 稀疏 (Sparse KV)

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-5.1 | 稀疏使能 + capacity 自动/显式 | `--sparse-config` + `--kv-transfer-config` + `--compilation-config` 三组参数正确生成；capacity 显式优先、自动推算兜底 | 命令完整，LD_LIBRARY_PATH 含 vsparse |
| T-5.2 | compilation_config 去重 | engine_config 已有 compilation_config 时不重复追加 | 命令中只有一个 `--compilation-config` |
| T-5.3 | kv_transfer_config 冲突 | Sparse + LMCache 同时启用 → Sparse 优先，LMCache 被移除 + WARNING | 不出现两个 `--kv-transfer-config` |
| T-5.4 | 仅 vllm 引擎支持 | engine=vllm_ascend + enable_sparse → 返回空 | 无 sparse 参数 |
| T-5.5 | total_budget 无效值 | total_budget=0 → WARNING | 日志记录，不阻塞启动 |

### 5.2 KV 卸载 (LMCache / PD)

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-5.6 | LMCache + PD 组合 | MultiConnector 包含两个子 connector | 配置正确 |
| T-5.7 | 仅 PD | 按设备选择 MooncakeConnector / NixlConnector | connector 类型正确 |

### 5.3 投机解码 (Speculative Decoding)

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-5.8 | 策略自动选择 | draft_model 路径存在 → eagle3/draft_model；Ascend+Qwen3Next → suffix；DeepSeek → MTP | 选到正确策略 |
| T-5.9 | SVIP 自适应草稿长度 | accel adaptive_draft_model 生效后注入自适应字段 | 长度策略生效 |
| T-5.10 | 无 Draft Model 路径 | ENABLE=true 但 MODEL_PATH 为空 → 跳过 adaptive_draft_model | 降级到 suffix/MTP |

### 5.4 Accel 补丁容错

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-5.11 | 批量安装 → 失败 → 逐特性回退 | batch exit≠0 → per-feature fallback | 成功的安装，失败的 WARNING 跳过 |
| T-5.12 | install.py 不存在 | accel 卷未挂载 | WARNING，引擎仍启动 |
| T-5.13 | 版本号传递 | ENGINE_VERSION → normalize → "major.minor.0" | install.py 收到正确版本 |

### 5.5 快速失败与回退

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-5.14 | 高级特性启用时崩溃 → 回退 | 一刀切策略：崩溃即禁用所有高级特性 | 回退命令不含 sparse/speculative/kv_transfer_config |

> **说明**：原 T-5.15（120s后崩溃不回退）测试用例已移除，改为一刀切策略：不区分启动阶段或运行阶段，崩溃即回退。

---

## 六、进程管理与端口

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-6.1 | 默认/自定义端口 | backend=17000, proxy=用户指定, health=19000, monitor=19100 | 四层端口正确 |
| T-6.2 | 启动重试与耗尽 | 首次失败 → 5s 后重试 → 两次都失败 → CRITICAL + 退出码 1 | 重试逻辑正确 |

---

## 七、硬件探测

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-7.1 | 探测优先级 | JSON 文件 > 环境变量 > 默认值 | 按优先级正确取值 |

---

## 八、性能测试

| 编号 | 测试用例 | 指标 | 预期 |
|------|---------|------|------|
| T-8.1 | 控制面启动耗时 | `build_launcher_plan()` + 文件写入 | < 5s |
| T-8.2 | 四引擎推理性能 | TTFT、TPOT、吞吐量 vs 社区原版直接启动 | 不劣化 |

---

## 九、兼容性 / 升级 / 稳定性 / 安全

| 编号 | 测试用例 | 验证要点 | 预期结果 |
|------|---------|---------|---------|
| T-9.1 | 版本升级兼容 | 解耦前 → 解耦后升级；ENGINE_VERSION 多版本 | 功能不受影响 |
| T-9.2 | 滚动升级 + 回滚 | K8s 滚动更新 / 回滚 | 服务不中断 |
| T-9.3 | 7×24h 稳定性 | 持续推理请求 | 无崩溃、无内存泄漏、日志不爆盘 |
| T-9.4 | 命令注入防护 | 特殊字符参数 + 符号链接拒绝 + 文件权限 644 | 安全 |

---

## 附录：代码模块与测试映射

| 代码模块 | 关键文件 | 测试编号 |
|----------|---------|---------|
| 主控制流 | `wings_control.py` | T-2.x, T-6.x |
| 启动计划 | `core/wings_entry.py` | T-5.11~T-5.15 |
| 配置合并 | `core/config_loader.py` | T-4.x, T-5.1~T-5.7 |
| 引擎适配 | `engines/vllm_adapter.py`、`sglang_adapter.py`、`mindie_adapter.py` | T-3.x |
| 硬件探测 | `core/hardware_detect.py` | T-7.1 |
| 端口 | `core/port_plan.py` | T-6.1 |
| 文件工具 | `utils/file_utils.py` | T-9.4 |
| 分布式 | `distributed/` | T-2.1~T-2.5 |
