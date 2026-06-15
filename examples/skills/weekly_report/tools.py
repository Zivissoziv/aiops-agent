"""周报生成技能 — 工具函数。"""

import json
from datetime import datetime
from pathlib import Path


def generate_report(sections: str, author: str = "Agent") -> str:
    """根据结构化内容生成 HTML 周报文件。

    Args:
        sections: JSON 字符串，格式见下方说明
        author: 报告作者

    Returns:
        HTML 文件路径

    sections 格式:
    {
        "summary": "本周工作概述",
        "categories": [
            {"name": "新功能", "items": ["实现了xxx", "上线了xxx"]},
            {"name": "问题修复", "items": ["修复了xxx", "处理了xxx告警"]},
        ],
        "todo": ["下周计划1", "下周计划2"]
    }
    """
    data = json.loads(sections) if isinstance(sections, str) else sections

    now = datetime.now()
    week_num = now.isocalendar()[1]

    items_html = ""
    for cat in data.get("categories", []):
        items_html += f"    <h3>{cat['name']}</h3>\n    <ul>\n"
        for item in cat.get("items", []):
            items_html += f"      <li>{item}</li>\n"
        items_html += "    </ul>\n"

    todo_html = ""
    for t in data.get("todo", []):
        todo_html += f"      <li>{t}</li>\n"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>第{week_num}周周报 - {author}</title>
<style>
body {{ font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }}
h1 {{ color: #333; border-bottom: 2px solid #4A90D9; padding-bottom: 10px; }}
h2 {{ color: #555; margin-top: 30px; }}
h3 {{ color: #4A90D9; }}
ul {{ padding-left: 20px; }}
li {{ margin: 6px 0; line-height: 1.6; }}
.summary {{ background: #f5f8ff; padding: 15px; border-radius: 6px; }}
.todo {{ background: #fff8f0; padding: 15px; border-radius: 6px; }}
.footer {{ margin-top: 40px; color: #999; font-size: 12px; text-align: center; }}
</style></head>
<body>
<h1>第{week_num}周周报</h1>
<p>作者: {author} | 日期: {now.strftime('%Y-%m-%d')}</p>
<div class="summary"><h2>本周概览</h2><p>{data.get('summary', '')}</p></div>
<h2>工作详情</h2>
{items_html}
<div class="todo"><h2>下周计划</h2><ul>{todo_html}</ul></div>
<div class="footer">由 AIOps Agent 自动生成</div>
</body>
</html>"""

    # 写入 examples/skills/weekly_report/ 目录下
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / f"weekly_report_{now.strftime('%Y%m%d')}.html"
    output_path.write_text(html, encoding="utf-8")

    return str(output_path)
