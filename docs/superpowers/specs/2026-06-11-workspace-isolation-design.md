# Workspace 隔离设计

## 需求

每次运行 `aiops-agent` 时，在 `.aiops_data` 下单独创建一个 workspace 目录，用于隔离不同 session 之间的记忆持久化文件（core_memory.json / episodic_memory.json），避免不同对话互相污染。

## 设计

### 目录结构

```
.aiops_data/
└── workspaces/
    └── YYYYMMDD_HHMMSS/          # 每次启动自动创建
        ├── core_memory.json      # 核心记忆（当前 session 专属）
        └── episodic_memory.json  # 情景记忆（当前 session 专属）
```

- workspace ID 格式: `YYYYMMDD_HHMMSS`（如 `20260611_143025`）
- 按时间排序，直观可读，无需额外序号管理

### 改动范围

仅修改 `src/aiops_agent/cli.py`：

| 改动 | 说明 |
|---|---|
| 新增 `WORKSPACE_ID` 模块常量 | `datetime.now().strftime("%Y%m%d_%H%M%S")` |
| 新增 `WORKSPACE_DIR` 模块常量 | `DATA_DIR / "workspaces" / WORKSPACE_ID` |
| Banner 增加 Workspace 行 | 显示当前 workspace ID |
| 记忆持久化路径 | `core_persist_path` / `episodic_persist_path` 指向 `WORKSPACE_DIR` |
| `/workspace` 命令 | 查看当前 workspace ID 和路径 |
| `/help` 更新 | 列出 `/workspace` 命令 |

其他模块无需修改 — `TieredMemory` 已经接受 `core_persist_path` / `episodic_persist_path` 参数。

### 清理策略

当前不做自动清理，用户可手动删除不需要的 workspace 目录。`.aiops_data/` 已在 `.gitignore` 中忽略。

### 未纳入

- 不提供 `/workspace switch` 动态切换（每次启动固定一个 workspace）
- 不提供自动清理过期 workspace 的逻辑
