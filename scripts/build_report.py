"""把 report/attribution-report.md 渲染成排版整洁的单文件 HTML。"""

from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "attribution-report.md"
DST = ROOT / "report" / "attribution-report.html"

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>淘宝用户行为数据 · 购买转化归因报告</title>
<style>
  :root {{
    --ink: #1b2733; --muted: #66727f; --line: #e3e7ec;
    --accent: #c0553f; --blue: #2f4b63; --bg: #ffffff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 56px 28px 80px; background: #f4f5f7; color: var(--ink);
    font-family: "PingFang SC", "Microsoft YaHei", "Hiragino Sans GB", system-ui, sans-serif;
    font-size: 15.5px; line-height: 1.85;
  }}
  main {{
    max-width: 860px; margin: 0 auto; background: var(--bg);
    padding: 64px 72px 72px; border-radius: 4px;
    box-shadow: 0 1px 3px rgba(27,39,51,.08), 0 12px 32px rgba(27,39,51,.06);
  }}
  h1 {{ font-size: 27px; line-height: 1.4; margin: 0 0 6px; letter-spacing: -.3px; }}
  h1::after {{ content: ""; display: block; width: 56px; height: 3px; background: var(--accent); margin: 18px 0 0; }}
  h2 {{ font-size: 19px; margin: 40px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }}
  h3 {{ font-size: 16px; margin: 26px 0 8px; color: var(--blue); }}
  p {{ margin: 12px 0; }}
  main > ol, main ol ol {{ margin: 12px 0; padding-left: 24px; }}
  li {{ margin: 6px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 13.5px; }}
  th, td {{ border: 1px solid var(--line); padding: 8px 10px; text-align: left; }}
  th {{ background: #f7f8fa; font-weight: 600; color: var(--blue); }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 18px auto; border: 1px solid var(--line); }}
  blockquote {{
    margin: 20px 0; padding: 14px 18px; background: #fbf7f4;
    border-left: 3px solid var(--accent); color: #4a4137; font-size: 14px;
  }}
  code {{ background: #f2f4f7; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
  pre {{ background: #f7f8fa; border: 1px solid var(--line); padding: 14px 16px; overflow-x: auto;
        font-size: 13px; line-height: 1.7; border-radius: 3px; }}
  pre code {{ background: none; padding: 0; }}
  a {{ color: var(--blue); }}
  strong {{ color: var(--ink); }}
  @media print {{
    body {{ padding: 0; background: #fff; }}
    main {{ box-shadow: none; padding: 0; max-width: none; }}
    h2 {{ break-after: avoid; }}
    img {{ break-inside: avoid; max-height: 88vh; }}
    table {{ break-inside: avoid; }}
  }}
  @page {{ size: A4; margin: 16mm; }}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def main():
    text = SRC.read_text(encoding="utf-8")
    html = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    DST.write_text(TEMPLATE.format(body=html), encoding="utf-8")
    print(f"written: {DST.relative_to(ROOT)}  ({DST.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
