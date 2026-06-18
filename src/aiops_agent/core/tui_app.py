# d:\workspace\aiops-agent\src\aiops_agent\core\tui_app.py
"""Textual TUI — 三段式分屏交互。"""

import asyncio
import traceback
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
from ..tools.file_tools import configure_workspace as configure_file_workspace
from ..tools.shell import configure_approval as configure_shell_approval
from ..tools.shell import configure_workspace as configure_shell_workspace


HELP_LINES = [
    "可用命令:",
    "  [b]/help[/b]             显示此帮助",
    "  [b]/exit[/b]             退出程序",
    "  [b]/tools[/b]            查看可用工具",
    "  [b]/memory[/b]           查看三层记忆状态",
    "  [b]/workspace[/b]        查看当前 Workspace",
    "  [b]/remember <事实>[/b]  添加核心记忆",
    "  [b]/forget <事实>[/b]    删除核心记忆",
    "  [b]/core[/b]             查看核心记忆列表",
    "  [b]/clear[/b]            清空对话",
    "  [b]/config[/b]           查看当前配置",
]


class ConfirmScreen(ModalScreen[bool]):
    def __init__(self, command: str, reason: str, **kwargs):
        super().__init__(**kwargs)
        self._command = command
        self._reason = reason

    def compose(self) -> ComposeResult:
        yield Label(
            "\n⚠️ [bold yellow]高风险操作需要确认[/]\n\n"
            f"{self._reason}\n\n"
            f"[dim]{self._command}[/]\n"
        )
        with Horizontal():
            yield Button("执行", variant="error", id="yes")
            yield Button("取消", variant="primary", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class AiOpsTUI(App):
    """AIOps Agent TUI 主应用。"""

    CSS = """
    Screen { layout: vertical; }

    #main-area { height: 1fr; }

    #chat-panel {
        width: 5fr;
        border-right: thick $primary;
        height: 1fr;
    }
    #chat-log { height: 1fr; }

    #flow-panel {
        width: 2fr;
        border-right: thick $surface;
        height: 1fr;
    }
    #node-box, #tool-box { height: 1fr; }
    #node-box { border-bottom: solid $surface; }
    #node-text, #tool-text { max-height: 100%; }

    #info-panel {
        width: 3fr;
        height: 1fr;
    }
    #workspace-box, #knowledge-box, #memory-box { height: 1fr; }
    #workspace-box { border-bottom: solid $surface; }

    #input-bar {
        dock: bottom;
        height: 3;
    }
    #input-bar > Input { width: 1fr; }

    Header { background: $primary 10%; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "退出", priority=True),
    ]

    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._llm = create_llm(config)
        self._memory: TieredMemory | None = None
        self._graph = None
        self._state: AppState | None = None
        self._workspace_id = ""
        self._node_lines: list[str] = []
        self._tool_lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-area"):
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat-log", markup=True, highlight=True, wrap=True, max_lines=10_000)
            with Vertical(id="flow-panel"):
                with Vertical(id="node-box"):
                    yield Static("🤖 [bold]节点流转[/]")
                    yield RichLog(id="node-text", markup=True, highlight=True, max_lines=50)
                with Vertical(id="tool-box"):
                    yield Static("🔧 [bold]工具调用[/]")
                    yield RichLog(id="tool-text", markup=True, highlight=True, max_lines=50)
            with Vertical(id="info-panel"):
                with Vertical(id="workspace-box"):
                    yield Static("📦 [bold]WORKSPACE[/]")
                    yield RichLog(id="workspace-text", markup=True, highlight=True, max_lines=50)
                with Vertical(id="knowledge-box"):
                    yield Static("📚 [bold]KNOWLEDGE[/]")
                    yield RichLog(id="knowledge-text", markup=True, highlight=True, max_lines=50)
                with Vertical(id="memory-box"):
                    yield Static("🧠 [bold]MEMORY[/]")
                    yield RichLog(id="memory-text", markup=True, highlight=True, max_lines=50)
        yield Input(id="input-bar", placeholder="输入 /help 查看命令, /exit 退出")

    def on_mount(self) -> None:
        self.title = "AIOps Agent"
        self.sub_title = f"模型: {self._config.model}"
        self._init_services()
        self._refresh_workspace()
        self._refresh_knowledge()
        self._refresh_memory()
        self._show_banner()

    def _show_banner(self):
        chat = self.query_one("#chat-log", RichLog)
        chat.write("")
        chat.write("[bold cyan]╔════════════════════════════════════════╗")
        chat.write("[bold cyan]║        AIOps Agent v0.4.0[/]       ║")
        chat.write("[bold cyan]║  输入 /help 查看命令, /exit 退出   ║")
        chat.write("[bold cyan]╚════════════════════════════════════════╝")
        chat.write("")

    def _init_services(self):
        data_dir = _find_project_root() / ".aiops_data"
        self._workspace_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._memory = TieredMemory(
            llm=self._llm, compaction_enabled=True,
            working_max_messages=2, working_max_tokens=500,
            core_persist_path=data_dir / "core_memory.json",
            episodic_persist_path=data_dir / "workspaces" / self._workspace_id / "episodic_memory.json",
        )

        configure_shell_approval(handler=self._confirm, mode="inline")
        configure_file_workspace(data_dir / "workspaces" / self._workspace_id)
        configure_write_approval(
            lambda path, preview: self._confirm(f"访问文件({path})", f"workspace 外路径: {preview}")
        )
        configure_shell_workspace(data_dir / "workspaces" / self._workspace_id)

        self._graph = build_complex_graph(self._config, self._llm, self._memory)

        from langchain_core.messages import AIMessage, ToolMessage
        restored: list = []
        for m in self._memory.working.get_messages():
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                restored.append(HumanMessage(content=content))
            elif role == "assistant":
                restored.append(AIMessage(content=content))
            elif role == "tool":
                restored.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", ""), name=m.get("name") or None))

        self._state: AppState = {
            "messages": restored, "task": "", "need_worker": False,
            "todos": [], "worker_round": 0, "intent_route": "",
            "intent_reason": "", "intent_confidence": 0.0,
            "chat_response": "", "session_context": "",
        }

    def _refresh_workspace(self):
        w = self.query_one("#workspace-text", RichLog)
        w.clear()
        m = self._config.model
        short = m.rsplit("/", 1)[-1] if "/" in m else m
        w.write(f"模型: {short}")
        w.write(f"工作区: {self._workspace_id}")

    def _refresh_knowledge(self):
        w = self.query_one("#knowledge-text", RichLog)
        w.clear()
        kb_dir = _find_project_root() / "knowledge_base"
        if kb_dir.exists():
            files = sorted(kb_dir.glob("*.md"))
            if files:
                for f in files:
                    w.write(f"• {f.stem}")
            else:
                w.write("[dim]知识库为空[/]")
        else:
            w.write("[dim]无知识库目录[/]")

    def _refresh_memory(self):
        w = self.query_one("#memory-text", RichLog)
        w.clear()
        stats = self._memory.get_stats()
        w.write(f"核心记忆: {stats.get('core_facts', 0)} 条")
        w.write(f"情景记忆: {stats.get('episodic_count', 0)} 个")
        w.write(f"工作记忆: {stats.get('working_messages', 0)} 条")
        facts = self._memory.get_core_facts()
        if facts:
            w.write("")
            w.write("[dim]── 核心 ──[/]")
            for f in facts[:3]:
                w.write(f"• {f[:30]}")

    def _refresh_node_text(self):
        w = self.query_one("#node-text", RichLog)
        w.clear()
        if self._node_lines:
            for line in self._node_lines:
                w.write(line)
        else:
            w.write("等待任务...")

    def _refresh_tool_text(self):
        w = self.query_one("#tool-text", RichLog)
        w.clear()
        if self._tool_lines:
            for line in self._tool_lines:
                w.write(line)
        else:
            w.write("空闲")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if not raw:
            return
        self.query_one("#input-bar", Input).clear()
        if raw.startswith("/"):
            self._handle_slash_command(raw.lower())
        else:
            self._handle_user_input(raw)

    def _handle_user_input(self, text: str):
        chat = self.query_one("#chat-log", RichLog)
        chat.write("")
        chat.write(f"[bold cyan]┌─ 🙋 你 ─────────────────────────────────[/]")
        chat.write(f"[bold cyan]│[/] {text}")
        chat.write(f"[bold cyan]└──────────────────────────────────────────[/]")
        self._state["messages"] = list(self._state["messages"]) + [HumanMessage(content=text)]
        self._state["task"] = text
        self._state["session_context"] = self._build_session_context()
        self.run_graph(text)

    def _build_session_context(self) -> str:
        lines = []
        for m in self._state.get("messages", []):
            if hasattr(m, "type") and m.type in ("human", "ai") and m.content:
                lines.append(f"[{m.type}]: {str(m.content)[:200]}")
        return "\n".join(lines[-6:]) if lines else ""

    def _handle_slash_command(self, cmd: str):
        chat = self.query_one("#chat-log", RichLog)
        cmd_parts = cmd.split()
        if cmd in ("/exit", "/quit"):
            self.exit()
        elif cmd == "/help":
            for line in HELP_LINES:
                chat.write(f"  {line}")
        elif cmd == "/tools":
            chat.write(f"[yellow]工具:[/] {', '.join(TOOL_MAP.keys())}")
        elif cmd == "/memory":
            stats = self._memory.get_stats()
            chat.write(f"[dim]记忆:[/] 工作{stats.get('working_messages', 0)}/{stats.get('working_max_messages', 0)} 情景{stats.get('episodic_count', 0)} 核心{stats.get('core_facts', 0)}")
        elif cmd == "/clear":
            self._memory.reset(); self._state["messages"] = []
            chat.clear(); self._node_lines.clear(); self._tool_lines.clear()
            self._refresh_node_text(); self._refresh_tool_text(); self._refresh_memory()
            chat.write("[green]✅ 已清空[/]")
        elif cmd == "/core":
            facts = self._memory.get_core_facts()
            if facts:
                chat.write("[dim]核心:[/]")
                for i, f in enumerate(facts, 1):
                    chat.write(f"  {i}. {f}")
            else:
                chat.write("  [dim]空[/]")
        elif cmd == "/config":
            chat.write(f"  {self._config.llm_provider} | {self._config.model} | {'→'.join(a['name'] for a in ALL_AGENTS)}")
        elif cmd == "/workspace":
            chat.write(f"  {self._workspace_id}")
        else:
            if cmd_parts[0] == "/remember" and len(cmd_parts) >= 2:
                self._memory.remember(" ".join(cmd_parts[1:]))
                self._refresh_memory()
                chat.write("[green]✅ 已记住[/]")
            elif cmd_parts[0] == "/forget" and len(cmd_parts) >= 2:
                if self._memory.forget(" ".join(cmd_parts[1:])):
                    self._refresh_memory()
                    chat.write("[green]✅ 已忘记[/]")
                else:
                    chat.write("[yellow]⚠️ 未找到[/]")
            else:
                chat.write(f"[yellow]未知: {cmd}[/]")

    @work(thread=True)
    def run_graph(self, user_input: str) -> None:
        try:
            for mode, event in self._graph.stream(self._state, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    self.call_from_thread(self._on_custom_event, event)
                elif mode == "updates":
                    self.call_from_thread(self._on_updates_event, event)
            self._memory.add_message({"role": "user", "content": user_input})
            self.call_from_thread(self._refresh_memory)
        except Exception as e:
            self.call_from_thread(self._show_error, str(e), traceback.format_exc())

    def _on_custom_event(self, event: dict):
        t = event.get("type")
        if t == "agent_start":
            name = event["agent"]
            if self._node_lines and name in self._node_lines[-1]:
                return
            self._node_lines.append(f"▸ {name}")
            self._refresh_node_text()
        elif t == "agent_end":
            name = event.get("agent", "")
            for i in range(len(self._node_lines)-1, -1, -1):
                if name in self._node_lines[i] and "▸" in self._node_lines[i]:
                    self._node_lines[i] = self._node_lines[i].replace("▸", "✓")
                    break
            self._refresh_node_text()
        elif t == "tool_start":
            name = event["tool"]
            self._tool_lines.append(f"▸ {name}")
            self._refresh_tool_text()
        elif t == "tool_result":
            name = event.get("tool", "")
            for i in range(len(self._tool_lines)-1, -1, -1):
                if name in self._tool_lines[i] and "▸" in self._tool_lines[i]:
                    self._tool_lines[i] = self._tool_lines[i].replace("▸", "✓")
                    break
            self._refresh_tool_text()
            error = event.get("error", "")
            output = event.get("output", "")
            if error:
                self.query_one("#chat-log", RichLog).write(f"  [red]✗ {error}[/]")
            if output:
                d = output[:120].replace("\n", " ")
                self.query_one("#chat-log", RichLog).write(f"  [dim]{d}[/]")

    def _on_updates_event(self, event: dict):
        chat = self.query_one("#chat-log", RichLog)
        for data in event.values():
            resp = data.get("chat_response", "")
            if resp:
                chat.write(f"\n[bold green]┌─ 🤖 AI ───────────────────────────────────[/]")
                chat.write(f"[bold green]│[/] {resp}")
                chat.write(f"[bold green]└──────────────────────────────────────────[/]")
            for msg in data.get("messages", []):
                if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                    chat.write(f"\n[bold green]┌─ 🤖 AI ───────────────────────────────────[/]")
                    chat.write(f"[bold green]│[/] {msg.content}")
                    chat.write(f"[bold green]└──────────────────────────────────────────[/]")

    def _show_error(self, msg: str, tb: str):
        chat = self.query_one("#chat-log", RichLog)
        chat.write(f"[red]❌ {msg}[/]")

    def _confirm(self, command: str, reason: str) -> bool:
        future: asyncio.Future[bool] = asyncio.Future()
        self.call_from_thread(self._push_confirm_screen, command, reason, future)
        return asyncio.run(future)

    def _push_confirm_screen(self, command: str, reason: str, future: asyncio.Future):
        def on_result(result: bool | None):
            future.set_result(result or False)
        self.push_screen(ConfirmScreen(command, reason), on_result)
