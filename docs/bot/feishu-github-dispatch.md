# 飞书自动化 → GitHub API → Cursor 策略研究（无服务器、不用 Cloudflare）

```text
飞书（多维表格 / 工作流「发送 HTTP 请求」）
        │  POST api.github.com/.../dispatches
        ▼
GitHub Actions（每日交易策略研究）
        │  Cursor Cloud Agents API
        ▼
skill /trading-strategy-research
        │  FEISHU_WEBHOOK_URL
        ▼
飞书群自定义机器人（结论卡片）
```

飞书国内访问 `api.github.com` 通常比访问 `*.workers.dev` 稳定，因此 **不再需要 Cloudflare Worker**。

群里的「自定义机器人 Webhook」只能**往群里发消息**，不能收你的 text。收口用 **飞书自动化 / 多维表格**；出口继续用现有 Webhook。

## 需要配置的

### 1. GitHub 仓库 Secrets（多半已有）

| Name | 必填 | 说明 |
| --- | --- | --- |
| `CURSOR_API_KEY` | ✅ | Cursor Dashboard → API Keys |
| `FEISHU_WEBHOOK_URL` | ✅ | 结论推回飞书群 |
| `FEISHU_WEBHOOK_SECRET` / `FEISHU_WEBHOOK_KEYWORD` | 按需 | 与群机器人安全设置一致 |

### 2. GitHub PAT（给飞书 HTTP 请求用）

1. [Fine-grained token](https://github.com/settings/personal-access-tokens) 或 [Classic token](https://github.com/settings/tokens)
2. 仓库限定本仓库；权限：**Contents: Read** + **Actions: Write**（Classic 勾选 `repo`）
3. 复制 `github_pat_...` / `ghp_...`，只填进飞书自动化请求头，不要提交到 git

### 3. 飞书：一张表 + 一条自动化

**3.1 建多维表格**

1. 飞书新建「多维表格」
2. 新建字段：`研究焦点`（多行文本）
3. 可选字段：`备注`

**3.2 添加自动化**

路径一般是：多维表格 → 自动化 → 添加自动化

- **触发**：当记录被创建时（或「当记录被修改且研究焦点不为空」）
- **操作**：发送 HTTP 请求

填下面各项（把仓库名、Token、焦点变量换成你的）：

| 项 | 值 |
| --- | --- |
| 方法 | `POST` |
| URL | `https://api.github.com/repos/czy2014hust/daily_stock_analysis/dispatches` |
| Header | `Accept: application/vnd.github+json` |
| Header | `Authorization: Bearer <GITHUB_PAT>` |
| Header | `X-GitHub-Api-Version: 2022-11-28` |
| Header | `Content-Type: application/json` |
| Body | 见下方 JSON |

Body（把「研究焦点」映射成表格字段；飞书里一般点「+」插入变量）：

```json
{
  "event_type": "feishu-strategy-research",
  "client_payload": {
    "text": "{{研究焦点}}"
  }
}
```

成功时 GitHub 返回 **204 无正文**。飞书自动化只要把 2xx 当作成功即可，不要要求 JSON 响应。

**3.3 使用**

在表格新增一行，`研究焦点` 填：

```text
算力板块量化策略
```

或：

```text
/策略研究 埋伏低位高景气方向
```

前缀 `/策略研究`、`/tsr` 等会被 Actions 自动去掉。

几秒后：GitHub Actions 出现运行（触发类型 `repository_dispatch`）。数分钟到数十分钟后：飞书群 Webhook 收到结论。

## 先用 curl 验证（推荐）

在本机确认 PAT 和仓库名无误，再配飞书：

```bash
export GITHUB_TOKEN=github_pat_xxx   # 或 ghp_xxx
export GITHUB_REPO=czy2014hust/daily_stock_analysis

curl -sS -o /dev/stderr -w "\nHTTP %{http_code}\n" \
  -X POST "https://api.github.com/repos/${GITHUB_REPO}/dispatches" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -H "Content-Type: application/json" \
  -d '{"event_type":"feishu-strategy-research","client_payload":{"text":"curl 连通性测试：只验证能否触发 Actions"}}'
```

期望：`HTTP 204`。然后打开仓库 Actions → **每日交易策略研究 (Cursor Skill)**。

仓库内同样示例：`deploy/feishu-github-dispatch/`。

## 没有多维表格时

飞书工作流 / 捷径里只要有「发送 HTTP 请求」，用同一 URL / Header / Body 即可。把 Body 里的 `text` 绑到流程的文本输入。

也可不经过飞书：GitHub Actions → Run workflow → 填 `extra_prompt`。

## 排障

| 现象 | 原因 |
| --- | --- |
| curl 不是 204，是 401/403 | PAT 错了或权限不够 / 仓库没授权 |
| curl 404 | `GITHUB_REPO` 写错，或 token 看不到该仓库 |
| 飞书自动化显示失败，但 Actions 已跑 | GitHub 204 空 body，把飞书成功条件改成「状态码 2xx」 |
| 飞书自动化失败，Actions 没有 | Header 没带 `Authorization`，或 Body 不是合法 JSON |
| Actions 有、群里无结论 | `FEISHU_WEBHOOK_URL` 未配或关键词/签名不匹配 |
| 飞书填不了自定义 Header | 换「请求头」高级模式，或先只用 curl / 快捷指令 |

## 安全

- PAT 权限尽量只给这一个仓库  
- 自动化里的 Token 相当于钥匙，不要截图发群  
- 可随时在 GitHub 吊销 token
