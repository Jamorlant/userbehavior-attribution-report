"""报告渲染：Markdown 表格/图片必须真的变成 HTML，而不是留成纯文本。"""

from __future__ import annotations

import pytest

from uba.report import build, render


def test_render_turns_table_into_html():
    html = render("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<table>" in html
    assert "<td>1</td>" in html
    assert 'lang="zh-CN"' in html


def test_render_keeps_images_and_fenced_code():
    html = render("![图](figures/fig1.png)\n\n```bash\nmake all\n```\n")
    assert '<img alt="图"' in html
    assert "<pre>" in html


def test_build_requires_markdown(tmp_settings):
    with pytest.raises(FileNotFoundError, match="报告正文不存在"):
        build(tmp_settings)


def test_build_writes_single_file_html(tmp_settings):
    tmp_settings.report_dir.mkdir(parents=True, exist_ok=True)
    tmp_settings.report_markdown.write_text("# 标题\n\n正文\n", encoding="utf-8")
    out = build(tmp_settings)

    assert out == tmp_settings.report_html
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
