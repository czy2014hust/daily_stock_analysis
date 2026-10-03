# 每日交易策略研究（Cursor Skill → 飞书）

每天 **北京时间 09:00** 自动调用 Cursor Cloud Agents API，执行仓库内 skill  
`.cursor/skills/trading-strategy-research/`（`/trading-strategy-research`），  
并把结果摘要推送到你已配置的飞书群。

## 架构

```text
GitHub Actions cron (01:00 UTC = 09:00 Asia/Shanghai)
        │
        ▼
scripts/trigger_trading_strategy_research.py
        │
        ├─ POST /v1/agents   → 启动 Cursor Cloud Agent（跑 trading-strategy-research）
        ├─ GET  /v1/agents/{id}/runs/{runId}  → 轮询直到完成
        └─ FeishuSender（复用 FEISHU_WEBHOOK_* / App Bot 配置）→ 飞书群
```

Workflow 文件：`.github/workflows/daily-trading-strategy-research.yml`

## 一次性配置

### 1. Cursor API Key

1. 打开 [Cursor Dashboard → API Keys](https://cursor.com/dashboard?tab=integrations)
2. 创建 User API Key 或 Service Account Key
3. 在 GitHub 仓库添加 Secret：`CURSOR_API_KEY`

> Cloud Agent 需要能访问本仓库（已连接 GitHub 的 Cursor 账号）。

### 2. 飞书（复用现有配置）

与「每日股票分析」相同，任选其一：

| 模式 | Secrets / Variables |
| --- | --- |
| Webhook | `FEISHU_WEBHOOK_URL`，可选 `FEISHU_WEBHOOK_SECRET` / `FEISHU_WEBHOOK_KEYWORD` |
| App Bot | `FEISHU_APP_ID` + `FEISHU_APP_SECRET` + `FEISHU_CHAT_ID` |

若「每日股票分析」已能推飞书，本任务无需额外飞书配置。

### 3. 可选 Variables

| Name | 说明 | 默认 |
| --- | --- | --- |
| `CURSOR_AGENT_REPO_URL` | Cloud Agent 仓库 URL | `https://github.com/<this-repo>` |
| `CURSOR_AGENT_STARTING_REF` | 起始分支 | `main` |
| `CURSOR_AGENT_MODEL` | Cloud Agent 模型 ID | Cursor 默认 |
| `CURSOR_AGENT_TIMEOUT_SECONDS` | 等待上限（秒） | `3300` |
| `TRADING_STRATEGY_RESEARCH_PROMPT_EXTRA` | 固定追加研究焦点 | 空 |
| `TRADING_STRATEGY_RESEARCH_TIMEOUT_MINUTES` | Actions job 超时 | `90` |

## 飞书触发（收 text → 研究 → 回飞书）

### 无服务器（推荐）：Cloudflare Worker → GitHub Actions

不需要自建 VPS。飞书事件 HTTP 打到 Worker，Worker 触发 `repository_dispatch`，Actions 跑 Cursor skill，结论走 `FEISHU_WEBHOOK_URL`。

- 代码：`deploy/feishu-cursor-bridge/`
- 说明：[桥接 README](../deploy/feishu-cursor-bridge/README.md)
- 配置摘要见 [飞书 Bot 配置 · 方案 A](bot/feishu-bot-config.md#方案-a无服务器cloudflare-worker-桥接)

### 有服务器：Stream Bot 进程

```text
/策略研究 算力板块量化策略
```

实现：`bot/commands/strategy_research.py`（需 `FEISHU_STREAM_ENABLED=true` 常驻进程）。

## 使用方式

### GitHub Actions（推荐）

- 定时：每天 09:00（北京时间）自动跑
- 手动：Actions → **每日交易策略研究 (Cursor Skill)** → Run workflow  
  - 可勾选「跳过飞书」  
  - 可填写额外研究焦点

### 本机 / 服务器（可选）

```bash
export CURSOR_API_KEY=...
# 飞书沿用 .env
python scripts/trigger_trading_strategy_research.py

# 仅预览 prompt
python scripts/trigger_trading_strategy_research.py --dry-run-prompt

# 触发 Cursor 但不发飞书
python scripts/trigger_trading_strategy_research.py --skip-feishu

# 仅测试飞书连通性（不调用 Cursor API）
python scripts/trigger_trading_strategy_research.py --feishu-test
```

GitHub Actions 手动运行时也可勾选 **仅测试飞书连通性**。

## Cursor Automations UI（备选）

若你更想用 Cursor 控制台而不是 GitHub Actions：

1. 打开 [cursor.com/automations](https://cursor.com/automations)
2. 新建 Automation，Trigger 选 **Scheduled**，cron：`0 1 * * *`（UTC）或按 UI 时区选每天 09:00
3. Repository 选本仓库 `main`
4. Prompt 可直接使用：

```text
Follow `.cursor/skills/trading-strategy-research/SKILL.md`.
Produce today's ranked strategy research report, save Markdown under reports/,
and keep the final reply as a concise Chinese summary for Feishu.
```

5. 飞书：Automations 目前没有原生飞书工具；请在 prompt 中让 Agent 调用 webhook，或继续用本仓库的 GitHub Actions 脚本负责推送（推荐后者，复用已有飞书 Secret）。

## 输出与排障

- Actions Artifact：`trading-strategy-research-<run>`（摘要 Markdown）
- 本地摘要：`reports/trading_strategy_research_YYYYMMDD_summary.md`
- Cursor Agent 链接会出现在飞书消息中，便于点开完整过程
- 常见失败：
  - `CURSOR_API_KEY` 未配置 / 无仓库权限
  - Cloud Agent 超时：调大 `CURSOR_AGENT_TIMEOUT_SECONDS` / job timeout
  - 飞书未配置：检查与每日分析相同的 FEISHU_* Secrets

## 安全说明

- `CURSOR_API_KEY` 与飞书 Secret 只放在 GitHub Secrets / 本机 `.env`，不要写入代码或 prompt 明文
- Cloud Agent 默认 `autoCreatePR=false`；完整报告文件由 Agent 写入仓库分支（如有推送），摘要始终经本脚本推飞书
