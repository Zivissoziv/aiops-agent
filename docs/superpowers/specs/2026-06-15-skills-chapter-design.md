# 07_skills — 技能系统教学示例设计

> 日期: 2026-06-15
> 状态: 初稿

## 1. 概述

### 1.1 学习目标

- 理解 Skills 和 Tool Calling 的本质区别：**注入方法论 vs 注入函数签名**
- 学会用 SKILL.md 声明式描述技能
- 实现技能加载器：运行时扫描 `skills/` 目录，动态注册
- 掌握两种技能调用方式：`/skill` 手动触发 + `match_keywords` 自动匹配
- 理解按需加载对 Token 利用率和 Agent 行为质量的提升

### 1.2 前置知识

- 已完成 02_tool_calling.py（理解 Tool Calling 机制）
- 已完成 05_langgraph.py（理解 ToolNode 消息流转）
- 已完成 06_rag.py（理解 Embedding 和 RAG 可为自动匹配提供语义支持）

### 1.3 新增依赖

无新增依赖。纯原理教学，不需要 chromadb/langgraph 等额外库。

---

## 2. 示例文件

### 2.1 主文件

`examples/07_skills.py` — 技能系统教学示例

### 2.2 预置技能目录

```
examples/skills/                     ← 与 07_skills.py 同级
├── troubleshooting/                  ← 故障排查方法论（主要教学示例）
│   ├── SKILL.md                     ← 技能本体
│   └── tools.py                     ← 辅助工具（模拟输出）
└── greeting/                        ← 问候技能（极简演示，用于对比）
    └── SKILL.md
```

### 2.3 修改文件

- `_common.py` — 无变更
- `_ui.py` — 无变更

---

## 3. Part 设计

### Part 1: 为什么需要技能系统？（~60 行）

**核心论点：** 有了 Tool Calling 的 Agent，就像一名有工具但没受过训练的实习生——能执行操作，但缺乏"在什么场景下、按什么顺序、怎么思考"的方法论。

**演示流程：**

1. **展示问题** — 让 LLM（无技能）回答"服务器变慢了排查一下"，观察输出：
   - Agent 只看 `top` 或 `free`，缺乏系统性的排查方法
   - LLM 凭通用知识猜测，而非按方法论推理

2. **引出 Skills 概念** — Tool Calling 给了 Agent 手（工具），Skills 给了 Agent 大脑（方法论）
   - Tool: `shell("top")` → 拿到 CPU 数据
   - Skill: 知道"第一阶段先做全景采集，不急于下结论"

3. **对比表**：

| 维度 | Tool Calling (02) | Skills (07) |
|------|------------------|-------------|
| 注入什么 | 函数签名 + 参数 Schema | 系统提示词 + 排查方法论 + 辅助工具 |
| 粒度 | 一次调一个函数 | 一套组合动作 + 判断逻辑 |
| Agent 参与度 | 执行指令 | 按方法论推理决策 |
| 注册方式 | 代码硬编码 | 文件系统 / 动态发现 |
| 可维护性 | 改代码 | 写文档 |

4. **类比：** Claude Code 的 `/skill` 系统，用户不用写代码就能扩展 Agent 能力

---

### Part 2: SKILL.md — 技能的声明式描述（~80 行）

**核心概念：** 每个技能是一个文件系统目录，核心载体是 SKILL.md。

**教学演示步骤：**

1. **展示目录结构**

```
examples/skills/
└── troubleshooting/
    ├── SKILL.md
    └── tools.py
```

2. **逐字段解析 SKILL.md frontmatter**

```yaml
---
name: troubleshooting
description: 服务器故障排查方法论 - 从症状到根因的系统排查
match_keywords: ["变慢", "故障", "排查", "诊断", "异常", "高负载"]
tools: ["gather_system_info", "check_recent_changes"]
---
```

各字段说明：
- `name`: 技能标识，用于 `/skill <name>` 调用
- `description`: 一句话描述，用于 `/skills` 列表展示
- `match_keywords`: 自动匹配关键词，Agent 检测到用户输入包含这些词时可自动激活
- `tools`: 技能需要的辅助工具

3. **展示 SKILL.md 正文（核心）**

正文是纯 Markdown，注入到 Agent 的 System Prompt 中。它不定义函数，而是定义**思维方式**：

```markdown
## 排查流程

### 第一阶段：全景采集
1. 调用 `gather_system_info` 采集：CPU、内存、磁盘 IO、网络、进程列表
2. 调用 `check_recent_changes` 查看：最近部署、配置变更
3. 先记录异常指标，不要急于下结论

### 第二阶段：假设驱动
基于异常指标形成 1-3 个合理假设，逐一验证...
```

**教学瞬间：** 学生看到 SKILL.md 会意识到——写技能不是在写代码，而是在**写一份方法论文档**，让 Agent 按文档思考。

4. **展示辅助工具（tools.py）**

```python
def gather_system_info() -> str:
    """模拟系统信息采集"""
    return json.dumps({
        "cpu": {"usage_percent": 85},
        "memory": {"used_percent": 60},
        "disk_io": {"await_ms": 350},
        "top_processes": ["nginx", "java", "mysql"],
    })

def check_recent_changes() -> str:
    """模拟最近变更查询"""
    return json.dumps({
        "changes": [
            {"time": "2 小时前", "type": "配置更新", "target": "Nginx"},
            {"time": "1 天前", "type": "部署", "target": "后端 v2.1.3"},
        ]
    })
```

工具函数返回模拟数据（教学场景不需要真的执行命令），但展示了 Skill 可以**携带自己的工具**，而不是全局注册。

5. **对比极简技能 greeting**

```
examples/skills/greeting/SKILL.md
---
name: greeting
description: 问候语风格指南
match_keywords: ["你好", "您好", "嗨"]
---

当用户问候时，用热情但不浮夸的语气回复。
不要用 "有什么可以帮助你的吗"，直接说 "来了啊，今天查什么？"
```

用来展示 Skill 可以**纯提示词，没有工具**——让概念更清晰。

6. **与 Tool Calling 的本质区别：**

- Tool Calling: `name` + `description` + `parameters` = **你可以这么用**
- Skill: `name` + `match_keywords` + markdown 方法论文本 = **面对这类问题，你应该这么想**

---

### Part 3: 技能加载器 — 运行时发现技能（~100 行）

**核心概念：** 技能不是代码里写死的，而是运行时从文件系统发现的。这是 07 章和前面所有章节最本质的分水岭。

**教学演示步骤（分三层，层层递进）：**

1. **第一层：扫描目录**

```python
def discover_skills(skills_dir: str = "skills") -> dict:
    """扫描 skills/ 目录，发现所有技能。"""
    skills_path = Path(__file__).resolve().parent / skills_dir
    skills = {}
    for skill_dir in sorted(skills_path.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        # 读取 frontmatter...
    return skills
```

展示输出：
```
发现 2 个技能:
  [troubleshooting] 服务器故障排查方法论
  [greeting]         问候语风格指南
```

2. **第二层：解析 frontmatter**

用最简方式解析 YAML frontmatter（`---` 分隔），不用引入 yaml 库。提取：
- `name` / `description` / `match_keywords` / `tools`
- `content` — frontmatter 后的 markdown 正文（即注入内容）

构建技能注册表：

```python
skills = {
    "troubleshooting": {
        "description": "服务器故障排查方法论...",
        "keywords": ["变慢", "故障", "排查", ...],
        "tools": ["gather_system_info", "check_recent_changes"],
        "content": "## 排查流程\n\n### 第一阶段...",  # ← 注入到 System Prompt
        "tools_module": "skills.troubleshooting.tools",  # ← 可导入的工具模块
    },
    ...
}
```

3. **第三层：动态加载工具**

解析 `tools` 字段后，动态 import 对应的 tools.py：

```python
def load_skill_tools(skill_name: str):
    """动态加载技能的工具模块。"""
    # 举例说明原理即可
    # 实际教学演示中用 importlib 动态导入 tools.py
    ...
```

**教学时刻：** 此时学生应该意识到——**新增一个技能 = 建目录 + 写 SKILL.md + 可选写 tools.py。不用改 agent.py，不用重启，不用注册。**

4. **可视化对比：**

| | 传统 Tool Calling | Skills |
|---|---|---|
| 加一个工具 | 写代码 → 注册到 ToolRegistry | **新建目录 + SKILL.md** |
| 修改行为 | 改 agent.py system prompt | **改 SKILL.md 文本** |
| 查看可用能力 | 读代码 | `/skills` 命令 |
| 权限控制 | 代码逻辑 | 文件系统目录权限 |

---

### Part 4: 技能调用 — 手动 + 自动（~80 行）

**核心概念：** 两种调用方式，覆盖不同场景。

1. **方式一：手动 `/skill` 命令**

教学展示：

```
用户: 排查一下这台服务器的性能问题
Agent: 要我启动故障排查方法论吗？
       输入 /skill troubleshooting 开始，或者直接描述你的问题

用户: /skill troubleshooting

Agent: ✅ 已加载「故障排查方法论」技能
─────────────────────────────────
第一阶段：全景采集
让我先全面采集系统信息...
...
```

实现原理：Agent 检测到用户输入的 `/skill <name>` 命令后：
- 从技能注册表找到对应技能
- 将 SKILL.md 正文拼接到 System Prompt
- 注册 tools.py 中的工具函数
- 继续对话（第一阶段自动开始）

2. **方式二：自动匹配 `match_keywords`**

教学展示：

```
用户: 最近那台数据库服务器老报警怎么回事

Agent: （检测到关键词：报警、服务器 → 匹配 troubleshooting 技能）
✅ 已自动激活「故障排查方法论」技能

让我先看看最近有什么变更...
> gather_system_info()
> check_recent_changes()
```

实现原理：
- 每次用户输入后，遍历技能注册表的 `keywords`
- 如果用户输入中包含任一关键词，提示用户激活技能
- 用户可以确认（自动激活）或拒绝（继续普通对话）

3. **技能状态管理展示**

```
/skills           — 列出所有可用技能（未激活状态）
/skill <name>     — 手动激活指定技能
/skill deactivate — 取消当前激活的技能（回到无技能状态）
```

4. **实验对比**

这是该章的"对比时刻"——让同一台 Agent 面对"排查服务器问题"：

| 维度 | 无 Skill | 有 Skill |
|------|---------|---------|
| 第一步做什么 | 随机选工具 | **全景采集** |
| 思维过程 | 无/猜测 | **先观察后假设** |
| 输出质量 | 零散的信息片段 | **结构化诊断报告** |
| Token 浪费 | 可能执行 3 次无关操作 | **按方法论走，不跑偏** |

---

### Part 5: 总结（~30 行）

**核心脉络图：**

```
问题：Agent 有工具没方法论 → 回答碎片化、缺乏深度
                  ↓
方案：Skills = markdown 方法论 + 辅助工具 + 动态加载
                  ↓
实现：skills/ 目录 → SKILL.md → 技能加载器 → 运行时注册
                  ↓
调用：/skill 手动 / match_keywords 自动
                  ↓
结论：Skills 让 Agent 从"执行指令"升级为"按方法论推理"

下一步：src/aiops_agent/ 中集成技能系统
```

**核心概念表：**

| 概念 | 说明 | 和 Tool Calling 的区别 |
|------|------|----------------------|
| SKILL.md | YAML frontmatter + markdown 方法论 | **注入思维方式**，不只是函数签名 |
| 技能加载器 | 运行时扫描 skills/ 目录 | **动态发现**，不是编译时固定 |
| match_keywords | 自动匹配技能触发条件 | Agent **自主判断**是否需要方法指导 |
| 技能状态 | 已发现 / 已激活 / 未激活 | 按需加载，**不浪费 Token** |
| tools.py | 技能专用的辅助工具 | 工具**跟随技能**，不是全局暴露 |

**与 02_tool_calling 的对比，从头到尾串起来：**

```
02: Agent 有了"手"（能调用工具执行操作）
03: Agent 有了"记忆"（能记住上下文）
04: Agent 有了"思考"（能计划多步行动）
05: Agent 有了"框架"（用状态机管理流程）
06: Agent 有了"知识库"（能查文档回答内部问题）
07: Agent 有了"技能"（能加载外部方法论指导行为）
```

---

## 4. 与实战项目的关联

07_skills 的教学设计直接映射到 `src/aiops_agent/` 中已有的技能系统：

| 教学概念 | 实战映射 |
|---------|---------|
| skills/ 目录 | `src/aiops_agent/agents/` — 多 Agent 角色 |
| SKILL.md frontmatter + 正文 | `agents/worker.py` 中 system_prompt + tools |
| 技能加载器 | `graph/complex.py` 中根据任务动态构建工具列表 |
| /skill 命令 | CLI 中扩展 `/skill` 命令 |
| match_keywords 自动匹配 | `core/intent_router.py` 意图路由 |

---

## 5. 边界说明

- **不涉及**多技能同时激活（Part 5 已移除）
- **不涉及**技能热重载（文件变更监听）
- **不涉及**跨网络技能市场
- **不涉及**技能版本管理和依赖
- **模拟数据**：tools.py 中的工具函数返回模拟 JSON 数据，不执行真实系统命令
- **零依赖**：不引入 yaml 库，用简单字符串分割解析 frontmatter

---

## 6. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `examples/07_skills.py` | 新增 | 主教学示例，5 个 Part |
| `examples/skills/troubleshooting/SKILL.md` | 新增 | 故障排查方法论技能本体 |
| `examples/skills/troubleshooting/tools.py` | 新增 | 模拟故障排查工具函数 |
| `examples/skills/greeting/SKILL.md` | 新增 | 问候风格技能（极简对比用） |
| 其他文件 | 无变更 | — |
