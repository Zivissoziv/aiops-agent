# Workspace 沙箱与记忆隔离设计

## 1. 需求

每次启动 AIOps Agent 时，在 `.aiops_data` 下创建一个 workspace 目录用于隔离情景记忆和 Agent 操作，同时核心记忆保持全局共享。Agent 的工具操作默认在 workspace 内进行，越界需审批。

## 2. 目录结构

```
.aiops_data/
├── core_memory.json              # 核心记忆 — 全局共享
└── workspaces/
    └── 20260611_143025/          # 每次启动自动创建
        ├── episodic_memory.json  # 情景记忆 — 按 workspace 隔离
        ├── ...                   # Agent 操作生成的文件
```

## 3. 改动范围

### 3.1 记忆路径调整（cli.py）

| 路径 | 改变 |
|---|---|
| `core_persist_path` | → `DATA_DIR / "core_memory.json"`（全局共享） |
| `episodic_persist_path` | → `WORKSPACE_DIR / "episodic_memory.json"`（workspace 隔离） |

### 3.2 文件工具沙箱（file_tools.py）

新增功能：
- `configure_workspace(workspace_path)` — 设置沙箱路径
- 在 `read_file` 和 `write_file` 中通过 `Path(path).resolve()` 判断路径是否在沙箱内

沙箱规则：

| 操作 | 路径在 workspace 内 | 路径在 workspace 外 |
|---|---|---|
| `read_file` | 直接读取 | 弹审批「读取 workspace 外路径：xxx，是否批准？」 |
| `write_file` | 直接写入（不再弹审批） | 弹审批「写入 workspace 外路径：xxx，是否批准？」 |

> 注：write_file 现有的审批回调（无论路径都弹）被替换为沙箱感知的逻辑——在 workspace 内不再需要审批。

### 3.3 Shell 默认工作目录（shell.py）

- 新增 `configure_workspace(workspace_path)` — 设置沙箱路径
- `subprocess.run()` 增加 `cwd=_workspace_path` 参数（有配置时生效）
- agent 用相对路径时默认在当前 workspace 下操作
- 越界操作（cd .. / 绝对路径）由现有风险分级机制管控

### 3.4 CLI 注入（cli.py）

在 `main()` 中 memory 创建之后、graph 构建之前，注入 workspace 路径：

```python
configure_workspace(WORKSPACE_DIR)  # file_tools
configure_shell_workspace(WORKSPACE_DIR)  # shell
```

## 4. 未纳入

- shell 不做强制路径沙箱（仅设 cwd），超出 workspace 由现有风险分级处理
- 无自动清理旧 workspace 的逻辑
