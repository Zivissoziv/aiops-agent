"""
examples/02_tool_calling.py — 工具调用

学习目标:
  1. 理解 Function Calling / Tool Calling 的概念
  2. 学会定义工具的 JSON Schema
  3. 理解"LLM 决定调用哪个工具 → 我们执行 → 结果返回给 LLM"的循环
  4. 为后续实现 Agent 核心打下基础

运行方式:
  python examples/02_tool_calling.py

前置条件:
  已完成 01_simple_chat.py，理解基础对话流程

核心概念:
  - Tool Definition: 用 JSON Schema 描述工具的功能和参数
  - Tool Call: LLM 返回的调用请求（包含工具名和参数）
  - Tool Loop: LLM 决定→我们执行→结果返回→LLM 继续

注意:
  不是所有模型都支持 Tool Calling！请确保你的模型支持。
  DeepSeek V2/V3、GPT-4/GPT-3.5-turbo 均支持。
"""

import json
import subprocess
from openai import OpenAI

from _common import load_config


# ============================================================
# 第一步: 加载配置（同上一个示例）
# ============================================================
# load_config() 来自 _common.py，自动加载 .env 并验证 API Key
config = load_config()
model = config["model"]
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])


# ============================================================
# 第二步: 定义工具
# ============================================================
# 工具定义的核心是 JSON Schema，它告诉 LLM:
#   - 工具有什么功能（description）
#   - 需要什么参数（properties）
#   - 哪些参数是必须的（required）
# LLM 根据这些信息决定"什么时候调用哪个工具、传什么参数"

# 工具 1: 执行 Shell 命令
SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "在服务器上执行 Shell 命令，返回命令输出",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 Shell 命令，例如: ls -la, df -h, whoami",
                },
            },
            "required": ["command"],
        },
    },
}

# 工具 2: 读取文件内容
READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取指定文件的内容，返回前 N 行",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径，例如: /var/log/syslog",
                },
                "lines": {
                    "type": "integer",
                    "description": "读取的行数，默认 50",
                },
            },
            "required": ["path"],
        },
    },
}

TOOLS = [SHELL_TOOL, READ_FILE_TOOL]


# ============================================================
# 第三步: 实现工具执行函数
# ============================================================
# 当 LLM 决定调用工具时，会返回一个 tool_calls 列表
# 我们需要根据 name 找到对应的函数，传入参数，执行，返回结果

def execute_shell(command: str) -> str:
    """执行 Shell 命令并返回输出。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "(命令执行成功，无输出)"
        else:
            return f"退出码: {result.returncode}\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时（30秒）"
    except Exception as e:
        return f"错误: {e}"


def execute_read_file(path: str, lines: int = 50) -> str:
    """读取文件内容并返回。"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = []
            for i, line in enumerate(f):
                if i >= lines:
                    content.append(f"... (文件共 {i}+ 行，仅显示前 {lines} 行)")
                    break
                content.append(line.rstrip())
            return "\n".join(content) if content else "(文件为空)"
    except FileNotFoundError:
        return f"错误: 文件不存在: {path}"
    except Exception as e:
        return f"错误: {e}"


# 工具名称 → 执行函数的映射
TOOL_FUNCTIONS = {
    "shell": execute_shell,
    "read_file": execute_read_file,
}


# ============================================================
# 第四步: Agent 工具调用循环
# ============================================================
# 这是 Agent 的核心模式:
#   1. 调用 LLM（传入消息 + 工具定义）
#   2. 检查 LLM 的回复:
#      a. 如果是文本 → 输出并结束
#      b. 如果是工具调用 → 执行工具 → 结果返回给 LLM → 回到步骤 1
#   3. 设置最大循环次数防止无限循环
#   4. 如果 LLM 连续多次调用工具（比如先查磁盘，再分析结果），这是正常的

SYSTEM_PROMPT = "你是一个 AIOps 运维助手，擅长通过工具执行运维任务。"

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print(f"\n{'='*50}")
print(f"  工具调用示例 (Model: {model})")
print(f"  可用工具: shell, read_file")
print(f"  输入 exit 退出")
print(f"{'='*50}\n")

MAX_TOOL_ROUNDS = 5  # 防止无限循环的安全限制


while True:
    user_input = input("你: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("再见！\n")
        break

    messages.append({"role": "user", "content": user_input})

    # 工具循环: 允许 LLM 多次调用工具
    tool_round = 0
    while tool_round < MAX_TOOL_ROUNDS:
        tool_round += 1

        # 调用 LLM（传入工具定义）
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        message = choice.message

        # 如果 LLM 返回了文本内容，打印它
        if message.content:
            print(f"\n助手: {message.content}", flush=True)

        # 检查是否有工具调用请求
        if not message.tool_calls:
            break  # 没有工具调用，结束本轮

        # 有工具调用 — 逐个执行
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            func = TOOL_FUNCTIONS.get(func_name)

            print(f"\n🔧 调用工具: {func_name}({json.dumps(func_args, ensure_ascii=False)})")
            print("─── 输出 ──────────────────────────")

            if func:
                result = func(**func_args)
            else:
                result = f"错误: 未知工具 '{func_name}'"

            print(result[:500])  # 防止输出过长
            if len(result) > 500:
                print("...(输出过长已截断)")
            print("─── 结束 ──────────────────────────")

            # 将工具调用和结果追加到消息列表
            # LLM 需要看到工具调用的结果才能继续推理
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    else:
        print(f"\n⚠️ 已达到最大工具调用轮次（{MAX_TOOL_ROUNDS}），停止执行")
