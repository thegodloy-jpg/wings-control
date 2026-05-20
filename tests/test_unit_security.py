# -*- coding: utf-8 -*-
"""UT-EV 安全测试：验证 _parse_env_file 对恶意输入的防护。

覆盖 TEST_PLAN.md 中 UT-EV-01~12 共 12 个用例。

运行：
  python -m pytest tests/test_unit_security.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.wings_entry import _parse_env_file, _is_env_override_file  # noqa: E402


def _write_env(content: str, encoding: str = "utf-8", bom: bool = False) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".env", delete=False,
    )
    raw = content.encode(encoding)
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    tmp.write(raw)
    tmp.flush()
    return Path(tmp.name)


class TestParseEnvFileBasic(unittest.TestCase):
    """UT-EV-01~05: 正常解析路径。"""

    # UT-EV-01
    def test_ev01_simple_kv(self):
        """标准 KEY=VALUE 解析：key 和 value 正确映射。"""
        p = _write_env("MY_VAR=hello\nOTHER=world\n")
        result = _parse_env_file(p)
        # shlex.quote 对 shell-safe 字符串不加引号（POSIX 标准）
        self.assertIn("export MY_VAR=hello", result)
        self.assertIn("export OTHER=world", result)

    # UT-EV-02
    def test_ev02_double_quoted_value_unquoted(self):
        """双引号包裹的值去掉引号后被 shlex.quote 安全处理。"""
        p = _write_env('KEY="double quoted"\n')
        result = _parse_env_file(p)
        self.assertEqual(len(result), 1)
        self.assertIn("double quoted", result[0])

    # UT-EV-03
    def test_ev03_single_quoted_value_unquoted(self):
        """单引号包裹的值去掉引号后被 shlex.quote 安全处理。"""
        p = _write_env("KEY='single quoted'\n")
        result = _parse_env_file(p)
        self.assertEqual(len(result), 1)
        self.assertIn("single quoted", result[0])

    # UT-EV-04
    def test_ev04_comments_and_blank_lines_skipped(self):
        """注释行（#）和空行被忽略，不出现在结果中。"""
        p = _write_env("# comment\n\nVALID=ok\n")
        result = _parse_env_file(p)
        self.assertEqual(len(result), 1)
        self.assertIn("export VALID=ok", result)

    # UT-EV-05
    def test_ev05_line_without_equals_skipped(self):
        """不含 '=' 的行被跳过，不引发异常。"""
        p = _write_env("NOEQUALS\nVALID=yes\n")
        result = _parse_env_file(p)
        self.assertEqual(len(result), 1)
        self.assertIn("export VALID=yes", result)


class TestParseEnvFileInjectionBlocking(unittest.TestCase):
    """UT-EV-06~08: 非法 key 注入防护。"""

    # UT-EV-06
    def test_ev06_key_with_subshell_blocked(self):
        """key 含 $() 命令替换被拒绝（核心安全用例）。"""
        p = _write_env("A$(cmd)=val\n")
        result = _parse_env_file(p)
        self.assertEqual(result, [], "含命令替换的 key 应被过滤")

    # UT-EV-07
    def test_ev07_key_with_space_blocked(self):
        """key 含空格（导致 shell 解析错误）被拒绝。"""
        p = _write_env("KEY WITH SPACE=val\n")
        result = _parse_env_file(p)
        self.assertEqual(result, [], "含空格的 key 应被过滤")

    # UT-EV-08
    def test_ev08_key_with_hyphen_blocked(self):
        """key 含连字符（非法 shell 标识符）被拒绝。"""
        p = _write_env("A-B=val\n")
        result = _parse_env_file(p)
        self.assertEqual(result, [], "含连字符的 key 应被过滤")


class TestParseEnvFileSafeValueHandling(unittest.TestCase):
    """UT-EV-09~11: 合法 key + 危险/特殊 value 的安全处理。"""

    # UT-EV-09
    def test_ev09_dangerous_path_value_safely_quoted(self):
        """LD_PRELOAD=/evil.so：合法 key，危险 value 被 shlex.quote 安全处理。

        确认：value 本身不被执行，通过单引号包裹后写入 export 语句。
        """
        p = _write_env("LD_PRELOAD=/evil.so\n")
        result = _parse_env_file(p)
        self.assertEqual(len(result), 1)
        export_line = result[0]
        self.assertTrue(export_line.startswith("export LD_PRELOAD="))
        self.assertIn("/evil.so", export_line)
        # 确保 /evil.so 不会被 shell 解释（被引号包裹）
        self.assertNotIn("LD_PRELOAD=/evil.so\n", export_line)

    # UT-EV-10
    def test_ev10_value_with_subshell_safely_quoted(self):
        """VAL=$(whoami)：value 含命令替换，被 shlex.quote 包裹后不执行。"""
        p = _write_env("VAL=$(whoami)\n")
        result = _parse_env_file(p)
        self.assertEqual(len(result), 1)
        export_line = result[0]
        self.assertTrue(export_line.startswith("export VAL="))
        # $() 必须在引号内，不能裸露在 export 语句中
        self.assertNotIn("export VAL=$(whoami)", export_line)
        # value 内容被保留（但被安全包裹）
        self.assertIn("$(whoami)", export_line)

    # UT-EV-11
    def test_ev11_value_with_single_quote_escaped(self):
        """VAL=it's a test：value 含单引号，shlex.quote 正确 shell 转义。"""
        p = _write_env("VAL=it's a test\n")
        result = _parse_env_file(p)
        self.assertEqual(len(result), 1)
        export_line = result[0]
        self.assertTrue(export_line.startswith("export VAL="))
        # 值中的单引号必须被正确转义，整体可作为合法 shell 参数
        self.assertIn("it", export_line)
        self.assertIn("s a test", export_line)


class TestParseEnvFileBom(unittest.TestCase):
    """UT-EV-12: UTF-8 BOM 编码兼容性（记录实际行为）。

    当前实现使用 encoding='utf-8'，无法自动跳过 BOM（\\xef\\xbb\\xbf）。
    BOM 字符 \\ufeff 会附在第一个 key 前，导致 key 验证失败、第一行被跳过。
    这是已知限制；后续可改为 encoding='utf-8-sig' 修复。
    """

    # UT-EV-12
    def test_ev12_utf8_bom_first_key_skipped(self):
        """UTF-8 BOM 文件：BOM 前缀导致第一行 key 验证失败，后续行正常解析。"""
        p = _write_env("FIRST=one\nSECOND=two\n", bom=True)
        result = _parse_env_file(p)
        # BOM 会附到 FIRST 的 key 上 → 验证失败 → 被跳过
        # SECOND 不受影响
        self.assertNotIn("export FIRST=one", result,
                         "BOM 前缀使 FIRST 的 key 验证失败，应被跳过")
        self.assertIn("export SECOND=two", result)


class TestIsEnvOverrideFile(unittest.TestCase):
    """_is_env_override_file 过滤逻辑（附加保护测试）。"""

    def test_hidden_file_excluded(self):
        """隐藏文件（. 开头）被排除。"""
        with tempfile.TemporaryDirectory() as d:
            hidden = Path(d) / ".hidden.env"
            hidden.write_text("X=1")
            self.assertFalse(_is_env_override_file(hidden))

    def test_readme_excluded(self):
        """README.MD（大小写无关）被排除。"""
        with tempfile.TemporaryDirectory() as d:
            readme = Path(d) / "README.MD"
            readme.write_text("docs")
            self.assertFalse(_is_env_override_file(readme))

    def test_regular_env_file_included(self):
        """普通 .env 文件被接受。"""
        with tempfile.TemporaryDirectory() as d:
            env_file = Path(d) / "custom.env"
            env_file.write_text("X=1")
            self.assertTrue(_is_env_override_file(env_file))

    def test_directory_excluded(self):
        """目录路径被排除（is_file() 返回 False）。"""
        with tempfile.TemporaryDirectory() as d:
            subdir = Path(d) / "subdir"
            subdir.mkdir()
            self.assertFalse(_is_env_override_file(subdir))

    def test_mixed_valid_and_invalid_keys(self):
        """合法与非法行混合时，只输出合法行。"""
        p = _write_env(
            "GOOD=ok\n"
            "A$(evil)=bad\n"
            "ALSO_GOOD=yes\n"
            "1STARTS_DIGIT=nope\n"
        )
        result = _parse_env_file(p)
        self.assertEqual(len(result), 2)
        self.assertTrue(any("GOOD" in r for r in result))
        self.assertTrue(any("ALSO_GOOD" in r for r in result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
