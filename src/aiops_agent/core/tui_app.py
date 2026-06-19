# d:\workspace\aiops-agent\src\aiops_agent\core\tui_app.py
"""Textual TUI — 三段式分屏。"""

import asyncio, traceback
from datetime import datetime

from langchain_core.messages import HumanMessage
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Header, Input, Label, RichLog, Static

from ..agents import ALL_AGENTS
from ..config import Config, _find_project_root
from ..graph import AppState, build_complex_graph
from ..graph.complex import TOOL_MAP
from ..llm import create_llm
from ..memory.tiered import TieredMemory
from ..tools.file_tools import configure_write_approval
from ..tools.file_tools import configure_workspace as cfg_fw
from ..tools.shell import configure_approval as cfg_sh_a
from ..tools.shell import configure_workspace as cfg_sh_w


class ConfirmScreen(ModalScreen[bool]):
    def __init__(self, cmd: str, reason: str, **kw):
        super().__init__(**kw)
        self._cmd, self._reason = cmd, reason
    def compose(self) -> ComposeResult:
        yield Label(f"\n⚠️ [bold yellow]高风险操作需要确认[/]\n\n{self._reason}\n\n[dim]{self._cmd}[/]\n")
        with Horizontal():
            yield Button("执行", variant="error", id="yes")
            yield Button("取消", variant="primary", id="no")
    def on_button_pressed(self, e: Button.Pressed) -> None:
        self.dismiss(e.button.id == "yes")


class AiOpsTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #main-area { height: 1fr; }
    #chat-panel { width: 5fr; border-right: thick $primary; height: 1fr; }
    #chat-log { height: 1fr; }
    #flow-panel { width: 2fr; border-right: thick $surface; height: 1fr; }
    #node-box, #tool-box { height: 1fr; }
    #node-box { border-bottom: solid $surface; }
    #node-text, #tool-text { max-height: 100%; }
    #info-panel { width: 3fr; height: 1fr; }
    #workspace-box, #knowledge-box, #memory-box { height: 1fr; }
    #workspace-box { border-bottom: solid $surface; }
    #input-bar { dock: bottom; height: 3; }
    #input-bar > Input { width: 1fr; }
    Header { background: $primary 10%; }
    """

    BINDINGS = [Binding("ctrl+c", "quit", "退出", priority=True)]

    def __init__(self, config: Config, **kw):
        super().__init__(**kw)
        self._config = config
        self._llm = create_llm(config)
        self._memory = self._graph = self._state = None
        self._workspace_id = ""
        self._node_lines: list[str] = []
        self._tool_lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-area"):
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat-log", markup=True, highlight=True, wrap=True, max_lines=10_000)
            with Vertical(id="flow-panel"):
                for id_, title in [("node-box", "🤖 [bold]节点流转[/]"), ("tool-box", "🔧 [bold]工具调用[/]")]:
                    with Vertical(id=id_):
                        yield Static(title)
                        yield RichLog(id={"node-box": "node-text", "tool-box": "tool-text"}[id_], markup=True, highlight=True, max_lines=50)
            with Vertical(id="info-panel"):
                for id_, title in [("workspace-box", "📦 [bold]WORKSPACE[/]"), ("knowledge-box", "📚 [bold]KNOWLEDGE[/]"), ("memory-box", "🧠 [bold]MEMORY[/]")]:
                    with Vertical(id=id_):
                        yield Static(title)
                        yield RichLog(id={"workspace-box": "workspace-text", "knowledge-box": "knowledge-text", "memory-box": "memory-text"}[id_], markup=True, highlight=True, max_lines=50)
        yield Input(id="input-bar", placeholder="输入 /help 查看命令, /exit 退出")

    def on_mount(self) -> None:
        self.title = "AIOps Agent"
        self.sub_title = f"模型: {self._config.model}"
        self._init()
        for fn in [self._show_workspace, self._show_knowledge, self._show_memory, self._show_banner]:
            fn()

    def _show_banner(self):
        c = self.query_one("#chat-log", RichLog)
        c.write("\n[bold cyan]╔════════════════════════════════════════╗\n║        AIOps Agent v0.4.0[/]       ║\n║  输入 /help 查看命令, /exit 退出   ║\n╚════════════════════════════════════════╝\n")

    def _init(self):
        d = _find_project_root() / ".aiops_data"
        self._workspace_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        ws = d / "workspaces" / self._workspace_id
        self._memory = TieredMemory(llm=self._llm, compaction_enabled=True, working_max_messages=2, working_max_tokens=500, core_persist_path=d / "core_memory.json", episodic_persist_path=ws / "episodic_memory.json")
        cfg_sh_a(handler=self._confirm, mode="inline")
        cfg_fw(ws)
        cfg_sh_w(ws)
        configure_write_approval(lambda p, _: self._confirm(f"访问文件({p})", "workspace 外路径"))
        self._graph = build_complex_graph(self._config, self._llm, self._memory)
        from langchain_core.messages import AIMessage, ToolMessage
        msgs = []
        for m in self._memory.working.get_messages():
            k = m["role"]
            if k == "user": msgs.append(HumanMessage(content=m["content"]))
            elif k == "assistant": msgs.append(AIMessage(content=m["content"]))
            elif k == "tool": msgs.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"], name=m.get("name")))
        self._state = AppState(messages=msgs, task="", need_worker=False, todos=[], worker_round=0, intent_route="", intent_reason="", intent_confidence=0.0, chat_response="", session_context="")

    def _show_workspace(self):
        w = self.query_one("#workspace-text", RichLog); w.clear()
        m = self._config.model
        w.write(f"模型: {m.rsplit('/', 1)[-1] if '/' in m else m}\n工作区: {self._workspace_id}")

    def _show_knowledge(self):
        w = self.query_one("#knowledge-text", RichLog); w.clear()
        kb = _find_project_root() / "knowledge_base"
        if kb.exists():
            fs = sorted(kb.glob("*.md"))
            [w.write(f"• {f.stem}") for f in fs] if fs else w.write("[dim]空[/]")
        else:
            w.write("[dim]无[/]")

    def _show_memory(self):
        w = self.query_one("#memory-text", RichLog); w.clear()
        s = self._memory.get_stats()
        w.write(f"核心: {s['core_facts']}  情景: {s['episodic_count']}  工作: {s['working_messages']}")
        for f in self._memory.get_core_facts()[:3]:
            w.write(f"• {f[:30]}")

    def _refresh(self):
        self.query_one("#node-text", RichLog).clear(); [self.query_one("#node-text", RichLog).write(l) for l in self._node_lines] or self.query_one("#node-text", RichLog).write("等待任务...")
        self.query_one("#tool-text", RichLog).clear(); [self.query_one("#tool-text", RichLog).write(l) for l in self._tool_lines] or self.query_one("#tool-text", RichLog).write("空闲")

    def on_input_submitted(self, e: Input.Submitted) -> None:
        r = e.value.strip()
        if not r: return
        self.query_one("#input-bar", Input).clear()
        (self._handle_cmd if r.startswith("/") else self._handle_input)(r.lower() if r.startswith("/") else r)

    def _handle_input(self, text: str):
        c = self.query_one("#chat-log", RichLog)
        c.write(f"\n[bold cyan]┌─ 🙋 你 ─────────────────────────────────[/]\n[bold cyan]│[/] {text}\n[bold cyan]└──────────────────────────────────────────[/]")
        self._state["messages"] = list(self._state["messages"]) + [HumanMessage(content=text)]
        self._state["task"] = text
        self._state["session_context"] = "\n".join(f"[{m.type}]: {str(m.content)[:200]}" for m in self._state["messages"] if hasattr(m, "type") and m.type in ("human", "ai") and m.content)[-6:]
        self.run_graph(text)

    def _handle_cmd(self, cmd: str):
        c = self.query_one("#chat-log", RichLog); p = cmd.split()
        if cmd in ("/exit", "/quit"): self.exit(); return
        if cmd == "/help": [c.write(f"  {l}") for l in ["可用命令:", "  [b]/help[/b] 帮助", "  [b]/exit[/b] 退出", "  [b]/tools[/b] 工具", "  [b]/memory[/b] 记忆", "  [b]/workspace[/b] 工作区", "  [b]/remember <事实>[/b] 记住", "  [b]/forget <事实>[/b] 忘记", "  [b]/core[/b] 核心记忆", "  [b]/clear[/b] 清空", "  [b]/config[/b] 配置"]]; return
        if cmd == "/tools": c.write(f"[yellow]工具:[/] {', '.join(TOOL_MAP.keys())}"); return
        if cmd == "/memory":
            s = self._memory.get_stats(); c.write(f"[dim]记忆:[/]  工作{s['working_messages']}/{s['working_max_messages']}  情景{s['episodic_count']}  核心{s['core_facts']}"); return
        if cmd == "/clear":
            self._memory.reset(); self._state["messages"] = []; self.query_one("#chat-log", RichLog).clear()
            self._node_lines.clear(); self._tool_lines.clear(); self._refresh(); self._show_memory(); c.write("[green]✅ 已清空[/]"); return
        if cmd == "/core":
            fs = self._memory.get_core_facts()
            [c.write(f"  {i}. {f}") for i, f in enumerate(fs, 1)] if fs else c.write("  [dim]空[/]"); return
        if cmd == "/config": c.write(f"  {self._config.llm_provider} | {self._config.model} | {'→'.join(a['name'] for a in ALL_AGENTS)}"); return
        if cmd == "/workspace": c.write(f"  {self._workspace_id}"); return
        if p[0] == "/remember" and len(p) >= 2:
            self._memory.remember(" ".join(p[1:])); self._show_memory(); c.write("[green]✅ 已记住[/]"); return
        if p[0] == "/forget" and len(p) >= 2:
            c.write("[green]✅ 已忘记[/]" if self._memory.forget(" ".join(p[1:])) else "[yellow]⚠️ 未找到[/]"); self._show_memory(); return
        c.write(f"[yellow]未知: {cmd}[/]")

    @work(thread=True)
    def run_graph(self, inp: str) -> None:
        try:
            for mode, ev in self._graph.stream(self._state, stream_mode=["updates", "custom"]):
                self.call_from_thread(self._on_custom_event if mode == "custom" else self._on_updates, ev)
            self._memory.add_message({"role": "user", "content": inp})
            self.call_from_thread(self._show_memory)
        except Exception as e:
            self.call_from_thread(lambda: self.query_one("#chat-log", RichLog).write(f"[red]❌ {e}[/]"))

    def _on_custom_event(self, ev: dict):
        t = ev["type"]
        if t == "agent_start":
            n = ev["agent"]
            if not (self._node_lines and n in self._node_lines[-1]):
                self._node_lines.append(f"▸ {n}"); self._refresh()
        elif t == "agent_end":
            n = ev["agent"]
            for i in range(len(self._node_lines)-1, -1, -1):
                if n in self._node_lines[i] and "▸" in self._node_lines[i]:
                    self._node_lines[i] = self._node_lines[i].replace("▸", "✓"); break
            self._refresh()
        elif t == "tool_start":
            self._tool_lines.append(f"▸ {ev['tool']}"); self._refresh()
        elif t == "tool_result":
            n = ev["tool"]
            for i in range(len(self._tool_lines)-1, -1, -1):
                if n in self._tool_lines[i] and "▸" in self._tool_lines[i]:
                    self._tool_lines[i] = self._tool_lines[i].replace("▸", "✓"); break
            self._refresh()
            if ev.get("error"): self.query_one("#chat-log", RichLog).write(f"  [red]✗ {ev['error']}[/]")
            if ev.get("output"): self.query_one("#chat-log", RichLog).write(f"  [dim]{ev['output'][:120].replace(chr(10),' ')}[/]")

    def _on_updates(self, ev: dict):
        c = self.query_one("#chat-log", RichLog)
        for d in ev.values():
            for msg in d.get("messages", []):
                if getattr(msg, "type", "") == "ai" and msg.content:
                    c.write(f"\n[bold green]┌─ 🤖 AI ───────────────────────────────────[/]\n[bold green]│[/] {msg.content}\n[bold green]└──────────────────────────────────────────[/]")

    def _confirm(self, cmd: str, reason: str) -> bool:
        f: asyncio.Future[bool] = asyncio.Future()
        def done(r):
            f.set_result(r or False)
        self.call_from_thread(lambda: self.push_screen(ConfirmScreen(cmd, reason), done))
        return asyncio.run(f)
