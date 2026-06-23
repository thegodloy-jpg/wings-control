# UserPromptSubmit hook: 适配场景 → 强制触发 model-day0-adapt skill
# 读取 stdin 的 hook JSON，匹配模型适配关键词，命中则注入 additionalContext。
# 注意：本文件须以 UTF-8 BOM 保存，否则 Windows PowerShell 5.1 会按 GBK 解析中文导致语法错误。
$ErrorActionPreference = 'SilentlyContinue'

# 以 UTF-8 显式读取 stdin，避免控制台代码页把中文 prompt 解码错乱
$reader = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), [System.Text.Encoding]::UTF8)
$raw = $reader.ReadToEnd()
$prompt = ''
try { $prompt = ($raw | ConvertFrom-Json).prompt } catch { $prompt = $raw }
if ([string]::IsNullOrWhiteSpace($prompt)) { exit 0 }

$keywords = @(
  '适配','接入','新模型','新版本','day0','启动命令','对齐','recipe','tutorial',
  '三方对比','模型架构','模型名识别','对齐字段','环境变量对齐','vllm_ascend','对齐官方'
)

$lower = $prompt.ToLower()
$hit = $false
foreach ($k in $keywords) { if ($lower.Contains($k.ToLower())) { $hit = $true; break } }
if (-not $hit) { exit 0 }

$ctx = @'
[强制] 检测到「模型适配」场景。在作答前，必须先通过 Skill 工具调用 model-day0-adapt skill
（.claude/skills/model-day0-adapt/SKILL.md），并按其 Phase 0→6 流程执行。
核心强约束：产出的「CLI flag 名 ∪ JSON 嵌套 key 路径 ∪ 环境变量名」三集合必须与契约精确相等
（数值可差异）；新逻辑只能挂在已有结构/函数上，不开并行路径。不得跳过此 skill 直接改代码。
'@

$out = @{ hookSpecificOutput = @{ hookEventName = 'UserPromptSubmit'; additionalContext = $ctx } }
$out | ConvertTo-Json -Compress
exit 0
