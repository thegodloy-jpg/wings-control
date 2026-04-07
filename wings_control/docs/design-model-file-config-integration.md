# 模型文件化配置方案代码集成与迁移说明

> 关联文档:
> - [design-model-file-config.md](design-model-file-config.md)
> - [design-model-file-config-schema.md](design-model-file-config-schema.md)

---

## 1. 集成目标

文档方案需要满足三个前提：

1. 对现有 `config_loader.py` 的侵入尽量小。
2. 新旧配置来源边界清晰，不通过私有键做 side-channel 传递。
3. 迁移可以分阶段进行，且单个模型迁移后不需要继续依赖旧 `model_deploy_config` 补齐。

---

## 2. 与原设计相比的关键调整

| 主题 | 原设计口径 | 优化后口径 | 原因 |
|------|-----------|-----------|------|
| 文件命中策略 | 命中一个文件后直接返回 | 先收集四层候选，再按顺序合并 | 这样版本默认和架构默认才能真正复用 |
| 返回值 | 通过 `_model_env_vars` 等私有键挂到 `merged` | 返回结构化的 `ResolvedModelConfig` | 明确区分“最终引擎参数”和“运行时模型 env” |
| env 能力 | `env_vars + env_scripts` | 首版仅 `env_vars` | 避免引入脚本执行顺序和路径安全问题 |
| 大小写匹配 | 文档一处说不敏感，一处又按精确文件名查 | 新增模型文件名不敏感查找 helper | 保持与当前 `MODEL_NAME` 习惯一致 |

---

## 3. 推荐的数据结构

推荐把“解析出来的模型文件配置”作为独立结果返回：

```python
@dataclass
class ResolvedModelConfig:
    engine_params: dict[str, Any]
    env_vars: dict[str, str]
    source_files: list[str]
    matched_variant: str | None = None
```

`load_and_merge_configs()` 推荐返回：

```python
tuple[dict[str, Any], ResolvedModelConfig | None]
```

### 为什么不继续把 env 塞进 `merged`

1. `merged` 当前还会继续流向 `start_engine_service()` 和各 adapter。
2. 模型 env 不属于 `engine_config`，也不属于 CLI 参数，混在一起会让类型边界模糊。
3. 后续如果要记录 `source_files`、`matched_variant`、命中的层级，side-channel 会越来越难维护。

---

## 4. 核心接入点

### 4.1 `core/config_loader.py`

接入点仍然放在 `_get_model_specific_config()` 附近，但逻辑改为两段式：

```text
先尝试文件链:
    engine/_default
      → version/_default
      → arch/_default
      → model.json

若至少命中一层:
    只使用文件链结果

若四层全部未命中:
    回退到旧 model_deploy_config
```

### 4.2 `core/wings_entry.py`

`build_launcher_plan()` 需要拿到 `ResolvedModelConfig.env_vars`，生成单独的 `model_env_block`，再拼入最终 `start_command.sh`。

### 4.3 其他模块

以下模块不需要因为“文件化配置”本身而改变核心逻辑：

- `utils/model_utils.py`
- `engine_parameter_mapping.json`
- `sglang_adapter.py` / `mindie_adapter.py` / `vllm_adapter.py` 的参数消费逻辑

这些模块只消费最终合并后的 `engine_config`，不感知它来自旧大文件还是新文件链。

---

## 5. 建议新增的辅助函数

### 5.1 `_resolve_model_config_candidates()`

职责：根据 `engine / version / architecture / model_name` 生成低到高的四层候选路径。

```python
def _resolve_model_config_candidates(
    base_dir: Path,
    engine: str,
    version_dir: str | None,
    architecture: str | None,
    model_name: str | None,
) -> list[tuple[str, Path]]:
    candidates = [( "engine_default", base_dir / engine / "_default.json" )]
    if version_dir:
        candidates.append(("version_default", base_dir / engine / version_dir / "_default.json"))
    if version_dir and architecture:
        arch_dir = base_dir / engine / version_dir / architecture
        candidates.append(("arch_default", arch_dir / "_default.json"))
        if model_name:
            candidates.append(("model_exact", _find_case_insensitive_model_file(arch_dir, model_name)))
    return candidates
```

### 5.2 `_find_case_insensitive_model_file()`

职责：在架构目录下做大小写不敏感匹配，返回真实文件路径。

原因：Linux 文件系统大小写敏感，如果直接用 `Path(model_name + ".json")`，文档里承诺的大小写不敏感就落不了地。

### 5.3 `_validate_model_file_config()`

职责：做轻量 schema 校验。

重点校验项：

1. 顶层 key 是否都在白名单内。
2. `env_vars` 的 value 是否都是字符串。
3. 至少有一个有效贡献字段非空。
4. `meta.architecture` 若存在，是否与目录名一致。

### 5.4 `_load_model_file_config()`

职责：收集命中的文件、按顺序合并，再应用 `distributed_overrides` 和 `hardware_variants`。

```python
def _load_model_file_config(...) -> ResolvedModelConfig | None:
    merged_params = {}
    merged_env = {}
    source_files = []

    for layer_name, candidate in candidates:
        if not candidate or not candidate.is_file():
            continue
        raw = load_json_config(candidate)
        _validate_model_file_config(raw, candidate)
        merged_params = _merge_configs(merged_params, raw.get("engine_params", {}))
        merged_env = {**merged_env, **raw.get("env_vars", {})}
        source_files.append(str(candidate))

    if not source_files:
        return None

    if distributed:
        merged_params = _merge_configs(merged_params, raw_dist_engine_params)
        merged_env = {**merged_env, **raw_dist_env}

    if variant_hit:
        merged_params = _merge_configs(merged_params, raw_variant_engine_params)
        merged_env = {**merged_env, **raw_variant_env}

    return ResolvedModelConfig(
        engine_params=merged_params,
        env_vars=merged_env,
        source_files=source_files,
        matched_variant=variant_name,
    )
```

这里的核心不是伪代码细节，而是流程边界：

1. 先做文件链合并。
2. 再做运行时分支覆盖。
3. 最后把 `engine_params` 和 `env_vars` 作为结构化结果返回。

---

## 6. 推荐改造后的调用链

### 6.1 `load_and_merge_configs()`

推荐从：

```python
merged = load_and_merge_configs(...)
```

调整为：

```python
merged, resolved_model_cfg = load_and_merge_configs(...)
```

### 6.2 `_get_model_specific_config()`

推荐逻辑：

```python
resolved_model_cfg = _load_model_file_config(...)
if resolved_model_cfg is not None:
    engine_specific_defaults = _merge_cmd_params(
        hardware_env,
        resolved_model_cfg.engine_params,
        cmd_known_params,
        model_info,
    )
    return engine_specific_defaults, resolved_model_cfg

# 否则走旧逻辑
legacy_defaults = ...
legacy_defaults = _merge_cmd_params(...)
return legacy_defaults, None
```

### 6.3 `_prepare_merged_params()` 与 `build_launcher_plan()`

`_prepare_merged_params()` 不只返回 `merged`，也要把 `resolved_model_cfg` 继续带下去。这样 `build_launcher_plan()` 可以直接消费 `resolved_model_cfg.env_vars`，而不是再从 `merged` 中捞私有字段。

---

## 7. `wings_entry.py` 中的 env 注入

### 7.1 推荐新增函数

```python
def _build_model_env_block(env_vars: dict[str, str]) -> str:
    lines = []
    for key, value in env_vars.items():
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", key):
            logger.warning("Skipping invalid model env key: %s", key)
            continue
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines) + ("\n" if lines else "")
```

### 7.2 拼接位置

推荐顺序：

```text
analyzer_preamble
  → model_env_block
  → env_overrides
  → accel_preamble
  → script_body
```

### 7.3 为什么这样排

1. `model_env_block` 表达的是方案内建默认值。
2. `env_overrides` 是 operator 显式覆盖，优先级应更高。
3. `accel_preamble` 属于补丁级动作，应继续保持最高干预能力。

### 7.4 一个重要约束

adapter 内部的基础环境脚本仍然会在 `script_body` 里执行，因此模型文件里的 `env_vars` 必须限定为“模型/版本差异变量”，不能承担 `CANN`、`MindIE` 这类 bootstrap 初始化职责。否则会把“配置默认值”和“引擎基础环境”混成同一层。

这也是为什么首版不开放 `env_scripts`。

---

## 8. 迁移策略

### 8.1 文件落地顺序

推荐按下列顺序逐步落文件：

1. `engine/_default.json`
2. `engine/version/_default.json`
3. `engine/version/architecture/_default.json`
4. `engine/version/architecture/model.json`

这样迁移时可以把重复项持续向低层收敛，避免模型文件膨胀。

### 8.2 回退策略

```text
四层文件全未命中:
    使用旧 model_deploy_config

至少命中一层文件:
    仅使用文件链
```

这个边界必须明确写进实现文档，否则开发阶段很容易出现“先命中文件，再偷偷补旧逻辑”的混合状态。

---

## 9. 测试建议

### 9.1 单元测试

| 测试项 | 验证点 |
|--------|--------|
| `test_model_file_layer_merge` | 四层文件按顺序正确叠加 |
| `test_model_file_fallback_to_legacy` | 四层都未命中时回退旧逻辑 |
| `test_case_insensitive_model_match` | `DeepSeek-R1` 与 `deepseek-r1.json` 可正确匹配 |
| `test_distributed_overrides_after_layer_merge` | 先做文件层合并，再做 distributed 覆盖 |
| `test_hardware_variant_has_higher_priority_than_distributed` | 硬件变体覆盖优先级正确 |
| `test_env_vars_are_returned_out_of_band` | `env_vars` 不混入 `engine_config` |
| `test_invalid_top_level_key_rejected` | 错误 key 能被校验拦住 |

### 9.2 迁移验证

对首批迁移模型，建议保留“新旧结果对比”测试：

1. 旧路径解析出 `engine_specific_defaults`
2. 新文件链解析出 `engine_params`
3. 对比两者在迁移期是否等价

这样文档方案不仅能说明“怎么做”，还能给出“怎么证明迁移没有回归”的落地方法。
