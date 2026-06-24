# 需求三 · JSON自定义透传 — wings-control 实施文档

> 母文档/索引见 [smart.md](smart.md)。本文件承载原 **§0.5-C / §4.4**。
> 触发极性：**字段「存在」触发**（deployParams/envParams 非空 → 透传）。与需求一（开关置真）、需求二（字段缺省）三向互补。
> 跨需求的「§6 未决事实」「§7 影响范围/排期」仍在索引 [smart.md](smart.md)。

---

## C) 需求三 · JSON自定义透传（字段「存在」触发）

| 页面字段                              | wings 入参                                   | 判定点                                  | 执行                                               |
| ------------------------------------- | -------------------------------------------- | --------------------------------------- | -------------------------------------------------- |
| `deployParams`（自定义启动字段 JSON） | `CONFIG_FILE` / `--config-file`（JSON 文本） | `_load_user_config` 1718（config 非空） | 归一 kebab→snake → 合并 `engine_config` → 渲染 CLI |
| `deployParams` + 强制覆盖             | `CONFIG_FORCE=true`                          | `get_config_force_env()` 2733           | 用户配置**独占**，跳过模板                         |
| `envParams`（自定义环境变量 JSON）    | **直接 K8s Pod env**（MaaS 映射 `EnvVar[]`） | **wings 无判定**（引擎容器继承 env）    | 直达 vLLM 进程，**不过 wings、不拼 CLI**           |

> Maas：找傲宇确认json内部参数设计的逻辑（vllm，sglang，mindie），json页面开启后，需要环境变量承载；引擎侧环境变量，需要统一放置在一起，不要放在全局中。
>
> wings：依旧需要使能加速特性。（0708先不做）

---

## 4. 自定义透传（精确 diff）

### 4.4 C11/C12/C13 透传

- **C11 deployParams**：复用现有 `--config-file`/`CONFIG_FILE`→`_load_user_config`(1718)，wings 无改。
- **C12 envParams**：MaaS 注入 Pod env；wings 加保留字黑名单（禁覆盖 `WINGS_*/LMCACHE_*/PD_*/SD_*/SPARSE_*`）。
- **C13 自定义 vs 白名单优先级**（需求一×三耦合点）：用户在 deployParams 显式写 `kv_transfer_config/kv_cache_dtype/speculative_config` 时是否覆盖白名单——**待定见 §6-③**；实现挂靠现有「已预置不合成」范式（`_should_append_auto_speculative_config` 2722）。

> ⚠ **C13 优先级未决**：deployParams 手写 `kv_transfer_config` 等，用户显式 > 白名单 还是反之？暂按「用户显式优先」。给值后 §4.4 即最终态——详见索引 [smart.md](smart.md) §6-③。
