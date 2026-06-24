# 芯片识别逻辑重构需求：以 `hardware_family` 为单一真相源

> 作者：Claude Code 分析　|　日期：2026-06-24　|　状态：分析 / 待评审
> 范围：当 `/shared-volume/hardware_info.json` 提供真实可靠且以 `hardware_family` 为权威字段时，
> 现有项目中所有需要改动的位置（完备清单 + 严重度 + 落地顺序）。
>
> **修订（2026-06-24，dry_run 重构后校准）**：`dry_run.py` 已重构为三段式
> （`user_cli` / `orchestration_env` / `model_config`），原 #27 引用的 `setup_env:454,476` 已不存在。
> 本次校准 #27 并新增 §5.1，说明 dry_run 与硬件探测链的真实关系；family 链路（#0–#16）仍未落地。

---

## 1. 背景与触发场景

### 1.1 新输入样例
页面（MaaS 编排层）写入共享卷的 `hardware_info.json` 改为如下结构：

```json
{ "hardware_family": "rtx_pro_5000_72G", "device": "ascend", "count": 16, "cann_version": "8.0.0" }
```

关键事实（用户确认）：

1. **`device` 字段不可信**——样例里是一张 NVIDIA RTX 卡，却被写成 `"ascend"`。
2. **`hardware_family` 是跨 ascend/nv 统一的规范芯片标识槽位**——ascend 场景它就是 `910b`/`910c` 这类
   token，nv 场景是 `rtx_pro_5000_72G`、`h20` 等，语义对齐。
3. 新结构**丢弃了 `details`/`units`**，**新增 `cann_version`**。

### 1.2 旧输入样例（文档记录的真实样本）
```json
{ "count": 1, "details": [], "units": "GB", "device": "ascend" }
```

### 1.3 新旧对比

| 维度 | 旧（分裂信号） | 新（统一到 hardware_family） |
|---|---|---|
| 厂商真相源 | `device` | **`hardware_family`**（`device` 作废，降级为兜底/校验位） |
| 芯片代际真相源 | `details[].name` 含 `910b/910c` | **`hardware_family`** |
| 单卡显存 | `details[].total_memory/free_memory` | 编码在 family 名内（如 `_72G`） |
| 设备型号名 | `details[0].name` | 由 `hardware_family` 派生 |

---

## 2. 设计原则

### 2.1 单一真相源（SSOT）
`hardware_family` 同时承载**厂商**与**芯片代际**两个维度，取代原来分裂的
`device`（厂商）+ `details[].name`（代际）。`device` 字段**不再被直接信任**，仅在
`hardware_family` 缺失时作兜底，并可用于一致性校验告警。

### 2.2 单一归口分类器（L0，地基）
新增一个函数作为唯一解析入口，所有下游共用，杜绝多份 token 表漂移：

```
_classify_family(family: str) -> (vendor, platform, device_name)
   "910b" / "*a2*"        -> ("ascend", "a2",  规范名如 "Ascend910B")
   "910c" / "*a3*"        -> ("ascend", "a3",  规范名如 "Ascend910C")
   "rtx_*" / "*nvidia*"   -> ("nvidia", None,  原样/规范名)
   "h20*"                 -> ("nvidia", None,  ...)
   未命中                  -> (None, None, "unknown") + WARNING（不静默兜底）
```

- **vendor**：喂第一层厂商识别（nvidia/ascend）。
- **platform**：喂第二层 a2/a3 代际识别（nv 卡为 `None`）。
- **device_name**：喂引擎自动选择（310→mindie）、`Ascend910_9362` 专属 env 等
  依赖设备型号串的判定。

### 2.3 复用红利
ascend 的 `hardware_family` 就是 `910b`/`910c`，与现有 `device_details[].name`
匹配的 token **完全一致**，因此 token 表对 ascend 天然前向兼容；**新增的只有 nv 家族**，
而 nv 不涉及 a2/a3，只需映射 vendor。

---

## 3. 数据流逐跳（问题定位）

```
hardware_info.json
  { hardware_family, device(不可信), count, cann_version }
        │
        ▼ ① hardware_detect._load_hardware_from_file()  [L120-140]
   data["device"] = _normalize_device("ascend") → "ascend"   ← 信了错字段（缺陷1）
   data.setdefault("details", []) → []                       ← family 没派生出 name/显存（缺陷2）
   # hardware_family / cann_version 留在 dict 但下游无人读
        │
        ▼ ② config_loader._build_common_context()  [L271-288]
   "device_details": hardware_env.get("details") → []        ← family 在此跳被彻底丢弃（缺陷3）
   # 无 "hardware_family" key
        │
        ▼ ③ 两条并行消费链
   ├─ config_loader._resolve_device_name() → 'unknown'       ← 断引擎选择/310/9362（缺陷4，最严重）
   └─ vllm_adapter._resolve_deepseek_v4_flash_platform() → 默认 "a2"（缺陷5）
```

**核心结论**：`hardware_family` 当前是死字段，全链路无人读取；厂商误判 + 代际静默兜底 a2 +
设备名退化 'unknown' 三处同时发生。

---

## 4. 完备改动清单（按层 + 严重度）

严重度：🔴 不改则功能错误/回归　🟡 不改则退化到兜底（影响性能/校验）　🟢 配套（测试/文档/一致性）

### L0　新建分类器（地基）
| # | 位置 | 动作 | 级 |
|---|---|---|---|
| 0 | 新增 `_classify_family()`（建议 [hardware_detect.py](../../wings_control/core/hardware_detect.py)） | vendor + platform + device_name 三出口；未命中告警 | 🔴 |

### L1　文件读取层（两个并行入口，必须同步改）
| # | 位置 | 现状 | 需改 | 级 |
|---|---|---|---|---|
| 1 | [hardware_detect.py:130](../../wings_control/core/hardware_detect.py#L130) | `data["device"]=_normalize_device(data["device"])` | vendor 由 `_classify_family(family)` 定；`device` 仅 family 缺失回退 | 🔴 |
| 2 | [hardware_detect.py:131-132](../../wings_control/core/hardware_detect.py#L131-L132) | `setdefault("details",[])` | 由 family 合成一条 `details[{name, total_memory}]`（从 `_72G` 解析显存），否则 L2 显存校验 + name 判定全失效 | 🔴 |
| 3 | [hardware_detect.py:56-78](../../wings_control/core/hardware_detect.py#L56-L78) `_normalize_device` | 未识别→nvidia 静默 | family 模式未命中应告警，不静默误判 | 🟡 |
| 4 | [hardware_detect.py:190-208](../../wings_control/core/hardware_detect.py#L190-L208) env 兜底 | 无 family 概念 | 增 `WINGS_HARDWARE_FAMILY` env 兜底，与文件路径对齐 | 🟡 |
| 5 | [device_utils.py:84-88](../../wings_control/utils/device_utils.py#L84-L88) `_get_hardware_info` | 缓存原始 `device` | **与 #1/#2 完全相同的派生**（第二个并行读取入口，不同步会分叉） | 🔴 |
| 6 | [device_utils.py:200-223](../../wings_control/utils/device_utils.py#L200-L223) `get_nvidia_gpu_info`/`get_device_info` | 读 `details` | 依赖 #2 合成 details，否则 N卡显存内省恒空 | 🟡 |

### L2　config_loader.py（消费主战场）
| # | 位置 | 现状 | 需改 | 级 |
|---|---|---|---|---|
| 7 | [_build_common_context:277](../../wings_control/core/config_loader.py#L277) | 只摘 `details` | 加 `"hardware_family"` 透传到 params | 🔴 |
| 8 | [_resolve_device_name:1943-1947](../../wings_control/core/config_loader.py#L1943-L1947) | `details[0]['name']`，缺失→`'unknown'` | **改由 `_classify_family` 派生 device_name**；引擎选择/310/9362 总开关，最关键 | 🔴 |
| 9 | [_select_ascend_engine:2125](../../wings_control/core/config_loader.py#L2125) | `"310" in device_name`→mindie | 依赖 #8；否则 310 卡误走 vllm_ascend（310 不支持→崩） | 🔴 |
| 10 | [_prepare_mindie_model_config:1981](../../wings_control/core/config_loader.py#L1981) | `"310" in device_name` dtype 检查 | 依赖 #8 | 🟡 |
| 11 | [_check_vram_requirements:238-248](../../wings_control/core/config_loader.py#L218-L268) | 读 `details[].free_memory`，双重 guard 跳过 | 新格式无 free_memory→静默跳过；需从 family 显存重建或显式声明跳过 | 🟡 |
| 12 | [memory ctx:547-550](../../wings_control/core/config_loader.py#L547-L550) | `device_details[0].total_memory`→12GB | 依赖 #2 合成显存，否则 CUDA Graph 尺寸/显存推断退化 | 🟡 |
| 13 | [_load_default_config:1639-1644](../../wings_control/core/config_loader.py#L1639-L1644) | `device` 选 nvidia/ascend 配置 | `device` 派生后即正确；若需按 family 细分默认（如 rtx_pro_5000 专属）则扩展 | 🟡 |
| 14 | [L933](../../wings_control/core/config_loader.py#L933) / [L2631](../../wings_control/core/config_loader.py#L2631) / [L2772](../../wings_control/core/config_loader.py#L2772) `device=="nvidia"` | 直接信 device | 派生后自动正确，**无需改但要回归验证** | 🟢 |
| 15 | [_resolve_device_name:1946](../../wings_control/core/config_loader.py#L1946) `['name']` 直接下标 | details[0] 无 name 会 KeyError | 健壮性：改 `.get('name','unknown')`（顺带 #8 一并处理） | 🟡 |

### L3　vllm_adapter.py（代际 + 设备专属 env）
| # | 位置 | 现状 | 需改 | 级 |
|---|---|---|---|---|
| 16 | [_resolve_deepseek_v4_flash_platform:1368-1388](../../wings_control/engines/vllm_adapter.py#L1368-L1388) | 遍历 `device_details[].name` | 优先 `_classify_family(hardware_family).platform`；device_details 降级为 legacy 兜底；token 判断抽出复用 | 🔴 |
| 17 | [_get_engine_config_platform:1231-1254](../../wings_control/engines/vllm_adapter.py#L1231-L1254) | env+ENGINE_VERSION | 决定 family 在优先级链插入位（见 L4） | 🟡 |
| 18 | [_build_ascend910_9362_env_commands:965-973](../../wings_control/engines/vllm_adapter.py#L965-L973) | `detect_hardware().details[0].name` / `WINGS_DEVICE_NAME` | 改读 hardware_family/派生 device_name，否则 9362 专属 env 永不下发 | 🔴 |
| 19 | 6 处 platform 消费：[1423](../../wings_control/engines/vllm_adapter.py#L1423)/[1551](../../wings_control/engines/vllm_adapter.py#L1551)/[1606](../../wings_control/engines/vllm_adapter.py#L1606)/[1846](../../wings_control/engines/vllm_adapter.py#L1846)/[1979+](../../wings_control/engines/vllm_adapter.py#L1979)/[2096](../../wings_control/engines/vllm_adapter.py#L2096) | 消费 #16 返回值 | #16 修好后**无需改**；但 nv 卡 `platform=None` 要确认这些点都被 `engine==vllm_ascend` 门控 | 🟢 |

### L4　version_util.py（独立的第二条 a3 信号链，易漏）
| # | 位置 | 说明 | 级 |
|---|---|---|---|
| 20 | [engine_version_platform:81-104](../../wings_control/core/version_util.py#L81-L104) | 读 `ENGINE_VERSION` 后缀 `-a3/-a2`，被 [_get_engine_config_platform:1254](../../wings_control/engines/vllm_adapter.py#L1254) 和 [model_utils.py:190](../../wings_control/utils/model_utils.py#L190)（GLM-5.2 单机门控）共用。**必须定 family 与它冲突时的优先级**，否则两条链给出矛盾代际 | 🟡 |

### L5　配置数据文件
| # | 位置 | 说明 | 级 |
|---|---|---|---|
| 21 | [DEFAULT_CONFIG_FILES:79-88](../../wings_control/core/config_loader.py#L79-L88) / [SUPPORTED_DEVICE_TYPES:90](../../wings_control/core/config_loader.py#L90) | 当前默认只按 `nvidia/ascend` × 模型架构两维，**无芯片 family 维度**。若 `rtx_pro_5000_72G` 需区别于通用 nvidia，则引入 family 维度；复用则不动 | 🟢 |

### L6　测试与夹具（schema 变更必须同步）
| # | 位置 | 动作 | 级 |
|---|---|---|---|
| 22 | [snapshot_framework.py:59-97](../../tests/snapshot_framework.py#L59-L97) `nvidia_hardware`/`ascend_hardware` | mock dict 增 `hardware_family`，对齐新 schema | 🟢 |
| 23 | [test_dp_topology_sync.py](../../tests/test_dp_topology_sync.py)（12 处 `device_details:[{name:910x}]`） | 改用 family 注入，或新增 family 用例 | 🟢 |
| 24 | [test_real_user_official_vllm_ascend_alignment.py:91,160,193,210](../../tests/test_real_user_official_vllm_ascend_alignment.py#L91) | `hardware={device,count,details:[{name:910c}]}` → 改 family | 🟢 |
| 25 | [test_glm51_ascend_kvsparse.py:812-817](../../tests/test_glm51_ascend_kvsparse.py#L812-L817) | device_details a3 用例 → 补 family | 🟢 |
| 26 | [_smoke_launcher.py:24-26](../../wings_control/_smoke_launcher.py#L24-L26)、[dryrun_real_user_launch.py:80](../../tests/dryrun_real_user_launch.py#L80) | `WINGS_DEVICE_NAME`/env 路径 → 对齐 | 🟢 |
| 27 | [dry_run.py](../../dry_run.py) `apply_orchestration_env` / `simulate_wings_start`（已重构为三段式；旧 `setup_env:454,476` 不复存在） | dry_run **不写 hardware_info.json、不设 `WINGS_DEVICE_NAME`**，靠 `orchestration_env` 段的 `WINGS_ASCEND_PLATFORM`/`ENGINE_VERSION` 显式声明 env **短路整条探测链**（详见 §5.1）。family 化后把 `WINGS_HARDWARE_FAMILY` 当「编排注入项」放进 `orchestration_env`（依赖 #4 env 兜底），与 `WINGS_ASCEND_PLATFORM` 同层二选一 | 🟢 |

### L7　文档（描述旧 schema，会误导）
| # | 位置 | 级 |
|---|---|---|
| 28 | [maas_interface.md:249-272](../../wings_control/docs/maas_interface.md#L249-L272)（hardware_info.json schema） | 🟢 |
| 29 | [DPV4_VERSION_COMPATIBILITY_REPORT.md:83-84](../../DPV4_VERSION_COMPATIBILITY_REPORT.md#L83-L84)（details 含 a2/910b 表） | 🟢 |
| 30 | [README.md:93,118,221](../../README.md#L93)、各模块 docstring（hardware_detect/device_utils 头注） | 🟢 |

---

## 5. 两条独立的代际信号链（必须都覆盖）

| 链 | 来源 | 经过 | 影响 |
|---|---|---|---|
| 硬件探测链 | `hardware_info.json` 的 `hardware_family` | L1→L2(#7/#8)→L3(#16) | 本需求主体 |
| 版本号链 | `ENGINE_VERSION` 后缀 `-a3/-a2` | [version_util](../../wings_control/core/version_util.py#L81-L104) → `_get_engine_config_platform`、`model_utils.is_glm52_single_node_even` | 独立，不经 JSON |

**风险**：两链可能给出矛盾代际。落地前须确定优先级（见第 7 节决策点）。

### 5.1 dry_run 的覆盖边界（落地 family 时必读）

`dry_run.py`（已重构为三段式 `user_cli` / `orchestration_env` / `model_config`）是本地预览工具，
但它**不模拟硬件探测链**，这一点决定了它能验证什么、不能验证什么：

- **不写 `/shared-volume/hardware_info.json`、不设 `WINGS_DEVICE_NAME`** —— 因此
  [detect_hardware()](../../wings_control/core/hardware_detect.py#L143) 的 JSON 主路径与
  `details[].name` 探测路径在 dry_run 下**永远走不到**（只会落到 env 兜底，且 `details` 恒空）。
- 芯片代际靠 `orchestration_env` 段注入的 `WINGS_ASCEND_PLATFORM`（A2/A3）或 `ENGINE_VERSION`
  后缀（`-a2/-a3`）**显式声明短路**，即第 3 节数据流图里「① 显式声明」那条最高优先级分支。

**含义**：#16（`details`→family 的 a2/a3 resolver）即使改好，**dry_run 默认也覆盖不到它**，
因为 dry_run 从不喂 `details`/`hardware_family`，只喂显式 platform。要让 dry_run 真正验证
family 探测链，需二选一：

1. **轻**：等 #4 落地后，把 `WINGS_HARDWARE_FAMILY` 作为「编排层注入项」加进相关场景的
   `orchestration_env` 段（与 `WINGS_ASCEND_PLATFORM` 同层，二选一），走 env 兜底分支即可；
2. **重**：让 dry_run 在 `apply_orchestration_env` 阶段额外写一个临时
   `hardware_info.json{ hardware_family, device, count }` 到 `SHARED_VOLUME_PATH`，
   真正驱动 #1/#2/#16 的探测链（依赖这些项先落地）。

⚠️ 在 family 链路（#0–#16）落地前，对 dry_run 做上述任何改动都是**空转**——下游无人读
`hardware_family`，生成的 `start_command.sh` 不会有任何变化。

---

## 6. 最关键的 3 个 🔴 风险点（优先级排序）

1. **#8 `_resolve_device_name` 退化为 'unknown'**：连锁断掉 #9 引擎选择（310→mindie）、
   #18 Ascend910_9362 env。**影响面大于 a2/a3，是真实落地的头号回归。**
2. **#1/#2 与 #5 必须同步**：`hardware_detect` 与 `device_utils` 两个并行读取入口，
   改一漏一会导致 sidecar 两条路径厂商判定分叉。
3. **#16 a2/a3 resolver**：最初诉求，但实际优先级低于 #8。

---

## 7. 落地前必须拍板的语义决策点

1. **代际链优先级**：`hardware_family`（910c） vs `ENGINE_VERSION -a3` 后缀冲突时谁赢？
   建议沿用「显式 env > 镜像版本号 > 硬件探测」，即把 family 放在 `device_details` 之上、
   `ENGINE_VERSION` 之下；需确认。
2. **family 命名约定**：ascend 是裸 `910b`/`910c` 还是 `ascend_910b_64g` 带前缀？
   决定分类器用**子串包含**还是**前缀**匹配。`rtx_pro_5000_72G` 带规格后缀，暗示子串/前缀
   而非精确等值。
3. **family 级默认配置（#21）**：`rtx_pro_5000_72G` 是否需要区别于通用 nvidia 的默认参数？
   决定要不要给默认配置加第三维。
4. **显存解析（#2/#12）**：是否从 family 名（`_72G`）解析单卡显存回填 details，以救活
   VRAM 校验与 CUDA Graph 尺寸推断？还是接受退化到兜底。
5. **`device` 字段处置**：完全忽略，还是保留为「与 family 推导厂商不一致时」的告警源？

---

## 8. 建议落地顺序

| 阶段 | 内容 | 对应项 |
|---|---|---|
| P0 | 建 `_classify_family` 分类器 | #0 |
| P1 | 厂商层 family 化（两入口同步）+ device_name 派生 | #1 #2 #5 #8 #15 |
| P2 | 透传 hardware_family + 代际 resolver 接入 | #7 #16 |
| P3 | 设备专属 env + 引擎选择验证 | #18 #9 #10 |
| P4 | 显存/VRAM 重建（如决定做） | #11 #12 #6 |
| P5 | 版本号链优先级对齐 | #17 #20 |
| P6 | 测试夹具 + 文档同步 | #22–#30 |

---

## 9. 附：受影响文件清单（去重）

| 文件 | 涉及项 |
|---|---|
| `wings_control/core/hardware_detect.py` | #0 #1 #2 #3 #4 |
| `wings_control/utils/device_utils.py` | #5 #6 |
| `wings_control/core/config_loader.py` | #7 #8 #9 #10 #11 #12 #13 #14 #15 |
| `wings_control/engines/vllm_adapter.py` | #16 #17 #18 #19 |
| `wings_control/core/version_util.py` | #20 |
| `wings_control/utils/model_utils.py` | #20（共用方） |
| 默认配置 JSON（如需 family 维度） | #21 |
| `tests/*`、`dry_run.py`、`_smoke_launcher.py` | #22–#27 |
| `*.md` 文档 | #28 #29 #30 |
