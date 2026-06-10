"""
examples/01_simple_chat.py — 基础 LLM 对话

学习目标:
  1. 理解 LLM 的基本调用方式
  2. 理解 System Prompt 的作用
  3. 理解多轮对话的消息结构
  4. 理解流式输出的效果

运行方式:
  python examples/01_simple_chat.py

前置条件:
  在项目根目录配置 .env 文件

核心概念:
  - System Message: 设定 AI 的角色和行为规则
  - User Message: 用户输入
  - Assistant Message: AI 回复（也会用作后续对话的历史）
  - 流式输出: LLM 逐字生成回复，而不是等全部生成完再显示
"""

import os
from openai import OpenAI

from _common import load_config, estimate_tokens


# ============================================================
# 第一步: 加载配置
# ============================================================
# load_config() 自动从项目根目录加载 .env 文件
# 如果 API Key 未配置会直接提示退出
config = load_config()
model = config["model"]


# ============================================================
# 第二步: 创建 LLM 客户端
# ============================================================
# OpenAI SDK 兼容所有 OpenAI 格式的 API
# 只需修改 .env 中的 base_url 即可切换不同的 LLM 提供商
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])


# ============================================================
# 第三步: 定义系统提示词
# ============================================================
# System Prompt 是 AI 的行为指南，告诉它:
#   - 它是什么角色
#   - 应该怎么回答问题
#   - 有什么限制和规则
# 一个好的 System Prompt 能显著提升回复质量
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你是一个 AIOps 运维助手。你擅长回答运维相关问题，"
    "会用中文清晰、简洁地解释技术概念。",
)


# ============================================================
# 第四步: 进入对话循环
# ============================================================
# 消息列表的结构:
#   [
#       {"role": "system", "content": "..."},    # 系统设定（固定不变）
#       {"role": "user", "content": "..."},       # 用户问题
#       {"role": "assistant", "content": "..."},  # AI 回答
#       {"role": "user", "content": "..."},       # 下一个问题
#       ...
#   ]
# 每次请求都会携带全部历史，所以 LLM 能"记住"前面的对话

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print(f"\n{'='*50}")
print(f"  LangChain 对话机器人 (Model: {model})")
print(f"  输入 exit 或 quit 退出")
print(f"{'='*50}\n")

# 主循环: 一问一答
while True:
    user_input = input("你: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("再见！\n")
        break

    # 将用户输入添加到消息列表
    messages.append({"role": "user", "content": user_input})

    # 调用 LLM（流式输出）
    # stream=True 表示逐字返回回复，体验更流畅
    print("助手: ", end="", flush=True)
    full_response = ""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    print("\n")

    # 将 AI 回复也保存到消息列表
    # 这样下一轮对话时 AI 能看到上文
    messages.append({"role": "assistant", "content": full_response})

    # 简单的 token 消耗提示（近似估算）
    print(f"  [当前历史约 {estimate_tokens(messages)} tokens，共 {len(messages)} 条消息]")
