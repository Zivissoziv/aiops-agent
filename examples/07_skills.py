"""
examples/07_skills.py — 技能系统（Skill System）

学习目标:
  1. 理解 Skills 和 Tool Calling 的本质区别
  2. 理解"技能调度器"完整链路：用户输入 → LLM 判断意图 → 加载技能 → LLM 决策 → 执行工具
  3. 理解渐进式加载：第一阶段只发技能摘要，第二阶段才发完整内容
  4. 学会用 SKILL.md 声明式描述技能，配合可执行 tools.py

运行方式:
  python examples/07_skills.py

前置条件:
  已完成 02_tool_calling.py，理解 Tool Calling 机制
"""

import importlib
import json
import sys
from pathlib import Path

from openai import OpenAI

from _common import load_config
from _ui import console, title, subtitle, note, success, info, diagram, divider, wait_for_enter, make_table


def _escape(text: str) -> str:
    """转义 Rich markup 特殊字符，防止 LLM 输出中的 [] 引发 MarkupError。"""
    return text.replace("[", r"\[").replace("]", r"\]")


config = load_config()
model = config["model"]
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])


# ============================================================
# 技能加载器
# ============================================================

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def discover_skills() -> dict:
    skills = {}
    if not SKILLS_DIR.exists():
        return skills
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        text = skill_file.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        name = fm.get("name", skill_dir.name)
        skills[name] = {
            "name": name,
            "description": fm.get("description", ""),
            "keywords": fm.get("match_keywords", []),
            "tools": fm.get("tools", []),
            "content": body,
        }
    return skills


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.strip().split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        return {}, text
    fm = {}
    for line in lines[1:end]:
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                v = []
        else:
            if (v.startswith('"') and v.endswith('"')) or \
               (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
        fm[k] = v
    body = "\n".join(lines[end + 1:]).strip()
    return fm, body

RECORDS = """本周操作记录:
- 修复 Nginx 502 错误: 调整 worker_connections 从 512 到 1024
- 清理磁盘空间: 删除 30 天前的日志，释放 15GB
- 部署后端 v2.1.3: 包含 3 个 bugfix
- 配置数据库备份策略: 每天凌晨 3 点全量备份，保留 7 天"""


# ============================================================
# 技能调度器
# ============================================================

class SkillDispatcher:
    """技能调度器 — 连接 LLM 和技能的核心组件。

    完整调用链路:
    ┌─────────────────────────────────────────────────────────┐
    │ ① match(user_input)                                    │
    │    → 发 [技能名+短描述] 给 LLM                          │
    │    → LLM 返回技能名（或无）                              │
    │ ② activate(skill_name)                                 │
    │    → 加载完整的 SKILL.md + 导入 tools.py                │
    │ ③ 构造 System Prompt                                   │
    │    → 发 [SKILL.md 全文 + 工具定义] 给 LLM               │
    │    → LLM 返回工具调用参数                                │
    │ ④ call_tool(tool_name, **kwargs)                       │
    │    → 调度器执行工具 → 返回结果                           │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.registry = discover_skills()
        self.active_skill = None
        self.active_tools = {}

    def match(self, user_input: str) -> str | None:
        """① 意图识别：只发技能摘要给 LLM（轻量调用）。"""
        skills_desc = "\n".join(
            f"- {n}: {s['description']}" for n, s in self.registry.items()
        )
        prompt = f"""用户说: "{user_input}"

可用技能:
{skills_desc}

判断用户是否需要激活某个技能。
- 如果匹配，只返回技能名
- 如果不匹配，返回"无"
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = resp.choices[0].message.content.strip().strip('"').strip("'").strip("`")
        return answer if answer in self.registry else None

    def activate(self, skill_name: str):
        """② 加载技能：读完整 SKILL.md + 导入 tools.py。"""
        skill = self.registry.get(skill_name)
        if not skill:
            raise ValueError(f"未知技能: {skill_name}")
        self.active_skill = skill_name
        self.active_tools = {}
        if skill["tools"]:
            if str(SKILLS_DIR.parent) not in sys.path:
                sys.path.insert(0, str(SKILLS_DIR.parent))
            try:
                mod = importlib.import_module(f"skills.{skill_name}.tools")
                self.active_tools = {
                    n: f for n, f in vars(mod).items()
                    if callable(f) and not n.startswith("_")
                }
            except (ImportError, ModuleNotFoundError):
                pass

    def get_skill_prompt(self) -> str:
        """获取完整的 SKILL.md 正文。"""
        skill = self.registry.get(self.active_skill) if self.active_skill else None
        return skill["content"] if skill else ""

    def call_tool(self, tool_name: str, **kwargs):
        """④ 执行工具。"""
        if tool_name not in self.active_tools:
            raise ValueError(f"技能 {self.active_skill} 没有工具 {tool_name}")
        return self.active_tools[tool_name](**kwargs)


# ============================================================
# Part 1: 无技能
# ============================================================

def run_part1():
    title("Part 1: 没有技能的时候，Agent 回答什么样？")

    console.print("""
  问 LLM 一个周报需求:

    [bold yellow]"这周做了几个运维任务，帮我生成周报"[/bold yellow]

  没有技能调度器介入，LLM 只能凭通用知识回答。\n""")

    subtitle("调用 LLM（无技能注入）")

    records = RECORDS

    messages = [
        {"role": "system", "content": "你是一个运维助手。根据工作记录生成周报。"},
        {"role": "user", "content": f"{records}\n\n帮我生成周报"},
    ]

    info("System Prompt: [dim]你是一个运维助手。根据工作记录生成周报。[/dim]")
    info("调用 LLM...")

    response = client.chat.completions.create(
        model=model, messages=messages, extra_body={"thinking": {"type": "disabled"}},
    )
    answer = response.choices[0].message.content or ""
    console.print(f"  [dim]{_escape(answer)}[/dim]\n")

    note("LLM 输出不可控——格式不固定，也没有生成文件的能力。")

    console.print("""
  接下来看看[green]技能调度器[/green]介入后，流程会有什么不同。\n""")
    wait_for_enter("按 Enter 进入 Part 2...")


# ============================================================
# Part 2: 调度器完整链路演示
# ============================================================

def run_part2():
    title("Part 2: 调度器完整链路演示")

    console.print("""
  [bold yellow]技能调度器（Skill Dispatcher）[/bold yellow]加在 LLM 和工具之间，分四步走:\n""")

    diagram("""
    ① match()    用户输入 + 技能摘要 → LLM → 技能名
    ② activate() 加载 SKILL.md 全文 + 导入 tools.py
    ③ LLM + 完整体 SKILL.md + 工具定义 → LLM → 调用参数
    ④ call_tool() 执行 → 返回结果
    """)

    # ── Step 1 ──
    divider()
    subtitle("Step 1: 调度器扫描目录")

    dispatcher = SkillDispatcher()
    for name, s in dispatcher.registry.items():
        info(f"发现技能: [cyan]{name}[/cyan] — {s['description']}")

    # ── Step 2 ──
    divider()
    subtitle("Step 2: match() — 调度器问 LLM 该用哪个技能")

    records = RECORDS
    user_input = "帮我根据这些记录生成周报"
    info(f"用户: [bold]\"{user_input}\"[/bold]")

    console.print("""  调度器.match() 发给 LLM 的内容:
    [dim]用户说: "帮我根据这些记录生成周报"
    可用技能:
    - weekly_report: 周报生成 — 根据工作记录自动生成格式化 HTML 周报
    判断用户是否需要激活某个技能...
    [/dim]\n""")

    matched = dispatcher.match(user_input)
    console.print(f"  → LLM 返回: [bold green]{_escape(str(matched))}[/bold green]\n")
    note("第一阶段只发了技能名和短描述，没有发完整 SKILL.md。这就是渐进式加载。")

    # ── Step 3 ──
    divider()
    subtitle("Step 3: activate() — 加载完整技能")
    info(f"调度器.activate(\"{matched}\")")
    dispatcher.activate("weekly_report")

    skill_prompt = dispatcher.get_skill_prompt()
    info(f"SKILL.md 全文已加载: {len(skill_prompt)} 字符")
    info(f"tools.py 已导入: {list(dispatcher.active_tools.keys())}")

    console.print("""
  现在把[bold]完整的 SKILL.md 正文[/bold] + 工具定义发给 LLM。
  第二阶段才发完整内容，这就是[bold]渐进式[/bold]——不浪费 Token。\n""")

    # ── Step 4 ──
    divider()
    subtitle("Step 4: LLM + 完整技能 → 工具调用参数")

    system_with_skill = f"""你是一个运维助手。根据工作记录生成周报。

参考以下周报方法论:
{skill_prompt}

可用工具:
- generate_report(sections: str, author: str) → 生成 HTML 周报文件
  sections 是 JSON: {{"summary": str, "categories": [{{"name": str, "items": [str]}}], "todo": [str]}}
"""

    messages = [
        {"role": "system", "content": system_with_skill},
        {"role": "user", "content": f"{records}\n\n按方法论整理内容后，调用 generate_report()。"},
    ]

    info("发送完整 System Prompt（含 SKILL.md 全文 + 工具签名）给 LLM...")
    response = client.chat.completions.create(
        model=model, messages=messages, extra_body={"thinking": {"type": "disabled"}},
    )
    llm_output = response.choices[0].message.content or ""
    console.print(f"  [green]{_escape(llm_output[:300])}[/green]\n")
    success("LLM 基于完整技能内容输出了结构化的周报内容")

    # ── Step 5 ──
    divider()
    subtitle("Step 5: call_tool() — 调度器执行")

    info("调度器.call_tool(\"generate_report\", sections=...)")
    report_data = json.dumps({
        "summary": "本周完成 4 项运维任务。",
        "categories": [
            {"name": "问题修复", "items": ["修复 Nginx 502 错误，调整 worker_connections 到 1024"]},
            {"name": "优化", "items": ["清理磁盘空间，释放 15GB"]},
            {"name": "新功能", "items": ["部署后端 v2.1.3", "配置数据库备份策略"]},
        ],
        "todo": ["监控 Nginx 状态", "整理运维文档"],
    }, ensure_ascii=False)

    output_path = dispatcher.call_tool("generate_report", sections=report_data, author="运维小A")
    success(f"文件已生成: [cyan]{output_path}[/cyan]")

    html_preview = Path(output_path).read_text(encoding="utf-8")[:200]
    console.print(f"  HTML 内容预览: [dim]{html_preview}...[/dim]")

    # ── 完整链路 ──
    divider()
    subtitle("完整链路回顾")

    diagram("""
    ① match()
       用户: "帮我生成周报"
       调度器: [技能名+描述] → LLM → 返回"weekly_report"
       （轻量调用，只有几十个 token）

    ② activate("weekly_report")
       调度器: 读取 SKILL.md 全文 + import tools.py
       （第一阶段只拿了技能名，现在才加载全文）

    ③ System Prompt = 基础 Prompt + SKILL.md 全文 + 工具定义
       调度器 → LLM → 返回工具调用参数
       （第二阶段才发完整内容）

    ④ call_tool("generate_report", sections=...)
       调度器 → tools.py → 写出 HTML 文件
       （真正执行）
    """)

    console.print("""
  [bold]和 Tool Calling 的区别:[/bold]
    Tool Calling: LLM 直接看到所有函数签名 → 自己决定调哪个
    Skills:       [bold]调度器[/bold]先轻量判断场景 → 再加载完整方法论 → LLM 看到全部上下文后再决策
    """)

    wait_for_enter("按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 动态发现
# ============================================================

def run_part3():
    title("Part 3: 动态发现 + 调用方式")

    console.print("""
  技能不是写死在代码里的，而是[bold]运行时动态发现[/bold]的。\n""")

    subtitle("扫描目录")

    dispatcher = SkillDispatcher()
    skills = dispatcher.registry
    s_table = make_table(headers=["技能名", "描述", "关键词", "工具"])
    for name, s in skills.items():
        kw = ", ".join(s["keywords"][:3]) + ("..." if len(s["keywords"]) > 3 else "")
        s_table.add_row(f"[cyan]{name}[/cyan]", s["description"][:35] + "...", kw, str(s["tools"]))
    console.print(s_table)

    note("新增一个技能 = 建目录 + 写 SKILL.md + tools.py。不改调度器代码。")

    divider()
    subtitle("两种激活方式")

    diagram("""
    自动: 调度器.match(用户输入) → LLM 判断意图 → 自动激活
    手动: 用户输入 /skill weekly_report → 直接激活
    """)

    test_queries = [
        "帮我生成这周的周报",
        "总结一下这周的工作",
        "今天天气真好",
    ]

    subtitle("自动匹配演示")
    for q in test_queries:
        console.print(f"  [bold]用户:[/bold] \"{q}\"")
        matched = dispatcher.match(q)
        if matched:
            console.print(f"  [bold green]→ LLM 判定: {_escape(matched)}[/bold green]")
        else:
            console.print(f"  [dim]→ LLM 判定: 无匹配，维持基础模式[/dim]")
        console.print()

    wait_for_enter("按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 总结
# ============================================================

def run_part4():
    title("Part 4: 总结")

    diagram("""
    Tool Calling:  LLM ←→ 工具函数（直接暴露全部）
    Skills:        LLM → [调度器] → 按需加载 SKILL.md + tools.py → 执行

    调度器 = 意图识别 + 渐进加载 + 工具路由
    """)

    table = make_table(
        title="Tool Calling vs Skills",
        headers=["", "Tool Calling", "Skills"],
    )
    table.add_row("注册方式", "代码硬编码", "文件系统动态发现")
    table.add_row("给 Agent 什么", "函数签名", "方法论 + 工具")
    table.add_row("中间层", "无", "[bold]调度器[/bold]")
    table.add_row("加载方式", "全量注入", "[bold]渐进式[/bold]（先摘要后全文）")
    table.add_row("新增能力", "改代码", "建目录 + 写 SKILL.md")
    console.print(table)

    console.print("""
  [bold]四条链路回顾:[/bold]
    [green]v[/green] ① match() → 技能摘要给 LLM → 返回技能名
    [green]v[/green] ② activate() → 加载完整 SKILL.md + tools.py
    [green]v[/green] ③ LLM + 完整上下文 → 返回工具调用参数
    [green]v[/green] ④ call_tool() → 执行 → 产出文件
    """)

    console.print("[bold cyan]再见！[/bold cyan]")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    console.rule("[bold cyan]技能系统示例[/bold cyan]")
    console.print(f"  Model: {model}")
    console.rule()

    run_part1()
    run_part2()
    run_part3()
    run_part4()
