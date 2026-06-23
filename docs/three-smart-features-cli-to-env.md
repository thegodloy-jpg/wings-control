# 三大 Smart 特性:CLI → 环境变量映射与触发验证

## 目的

把三大 Smart 特性的 CLI 形式对应到项目中实际的环境变量,并回答一个迁移关键问题:

> **页面不再拼 CLI、改为只下发环境变量,能否依旧触发对应特性?**

代码依据:CLI⇄ENV 映射层 [start_args_compat.py](../wings_control/core/start_args_compat.py);特性消费层 [vllm_adapter.py](../wings_control/engines/vllm_adapter.py)。

---

## 一、判定原则(为什么 ENV 能替代 CLI)

`build_parser()` 中每个参数的默认值都读自环境变量:

```python
p.add_argument("--xxx", default=_env("XXX", "default"))   # 字符串/数值
_add_bool(p, "--xxx", "XXX", False)                       # 布尔
```

因此 argparse 取值优先级为:

```
显式 CLI 参数  >  环境变量(default)  >  内置默认值
```

推论:**CLI 未传 → 用 default → 即环境变量值**。所以"页面不拼 CLI"前提下,凡走 `build_parser()` 的特性,下发环境变量即等价触发。SmartKVCache 例外——它绕过 parser,由 adapter 直接 `os.getenv`,本就只认环境变量。

---

## 二、逐特性映射

### SmartDecoding(智能投机)——全量可 ENV

| CLI 形式 | 环境变量 | 绑定点 |
| --- | --- | --- |
| `--enable-speculative-decode` | `ENABLE_SPECULATIVE_DECODE` | [:294](../wings_control/core/start_args_compat.py#L294) |
| `--speculative-decode-model-path` | `SPECULATIVE_DECODE_MODEL_PATH` | [:295](../wings_control/core/start_args_compat.py#L295) |
| `--enable-rag-acc` | `ENABLE_RAG_ACC` | [:296](../wings_control/core/start_args_compat.py#L296) |

> 派生:策略判定为 MTP/suffix 时自动注入 `VLLM_EARS_TOLERANCE=0.5`([:2547](../wings_control/engines/vllm_adapter.py#L2547))。

### SmartSparse(智能稀疏)——仅开关可 ENV

| CLI 形式 | 环境变量 | 说明 |
| --- | --- | --- |
| `--enable-sparse` | `ENABLE_SPARSE`([:304](../wings_control/core/start_args_compat.py#L304)) | 有 ENV |
| `--lc-sparse-threshold` | 无 | 仅作 `engine_config` JSON 键透传 |
| `--total-budget` | 无 | 同上 |
| `--local-kvstore-capacity` | 无 | 同上 |

> 后三项不在 `build_parser()`,旧 [wings_start.sh](../wings_control/wings_start.sh#L97) 也只在 help 文本出现、不解析。唯一下发通道是 `engine_config`(JSON)。

### SmartKVCache(智能 KV 卸载)——ENV 原生,无 CLI

adapter 直接 `os.getenv` 消费,不经 parser:

| 环境变量 | 作用 |
| --- | --- |
| `LMCACHE_OFFLOAD` | 卸载总开关([:685](../wings_control/engines/vllm_adapter.py#L685)) |
| `LMCACHE_LOCAL_CPU` / `LMCACHE_MAX_LOCAL_CPU_SIZE` | CPU 卸载开关 / 容量 |
| `LMCACHE_LOCAL_DISK` / `LMCACHE_MAX_LOCAL_DISK_SIZE` | 磁盘卸载路径 / 容量 |
| `LMCACHE_CHUNK_SIZE` | 缓存分块大小 |
| `LMCACHE_QAT` + `LMCACHE_QAT_{INSTANCE_NUM,LOSS_LEVEL,LOG_ENABLED,MODULE}` | QAT 压缩子特性 |

> 派生:设置任一容量/路径类 ENV 时自动生成 `lmcache_config.yaml` 并导出 `LMCACHE_CONFIG_FILE`([:700](../wings_control/engines/vllm_adapter.py#L700));`LMCACHE_QAT` 推导出 `LMCACHE_QAT_ENABLED`。

---

## 三、问题结论:只下发 ENV 能否触发

| 特性 | 子项 | 环境变量 | 纯 ENV 触发 |
| --- | --- | --- | --- |
| SmartDecoding | 开关 / 路径 / RAG | `ENABLE_SPECULATIVE_DECODE`、`SPECULATIVE_DECODE_MODEL_PATH`、`ENABLE_RAG_ACC` | ✅ |
| SmartSparse | 开关 | `ENABLE_SPARSE` | ✅ |
| SmartSparse | 阈值 / 预算 / 容量 | 无 | ❌ 需新增 ENV |
| SmartKVCache | 卸载 / 容量 / QAT | `LMCACHE_*` | ✅(原生唯一通道) |

**一句话**:SmartDecoding 与 SmartKVCache 已能纯环境变量全量触发;SmartSparse 仅开关可以,阈值/预算/容量三项是唯一缺口。

---

## 四、缺口补齐建议

让 SmartSparse 也支持纯 ENV,只需在 [build_parser()](../wings_control/core/start_args_compat.py#L304) 的 `--enable-sparse` 处补三个 `_env` 绑定:

| 建议 ENV | engine_config 键 | 类型 |
| --- | --- | --- |
| `LC_SPARSE_THRESHOLD` | `lc_sparse_threshold` | float |
| `TOTAL_BUDGET` | `total_budget` | float |
| `LOCAL_KVSTORE_CAPACITY` | `local_kvstore_capacity` | int/float |

补齐后,三大 Smart 特性即可统一做到"页面不拼 CLI、仅下发环境变量"全量等价触发。
