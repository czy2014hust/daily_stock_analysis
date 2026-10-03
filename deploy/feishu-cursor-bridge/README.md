# 飞书 → Cursor 策略研究桥接（无服务器）

不需要自建 VPS / `python main.py`。用 **Cloudflare Worker（免费）** 收飞书 text，触发 **GitHub Actions** 调用 Cursor skill，结论经已有 **飞书群 Webhook** 推回。

```text
飞书 App（事件订阅 HTTP）
        │
        ▼
Cloudflare Worker（本目录） ──立刻回复「已收到」
        │
        │ repository_dispatch
        ▼
GitHub Actions（daily-trading-strategy-research.yml）
        │
        │ Cursor Cloud Agents API
        ▼
trading-strategy-research skill
        │
        │ FEISHU_WEBHOOK_URL
        ▼
飞书群机器人（结论卡片）
```

研究可能跑数十分钟，所以必须 **异步**：Worker 只负责接单，长任务交给 Actions。

## 你需要准备的账号

| 项 | 用途 |
| --- | --- |
| 飞书企业自建应用 | 收消息 + 立刻回复 ACK |
| 飞书群自定义机器人 Webhook | 推送最终结论（你已配置过） |
| Cursor API Key | 跑 Cloud Agent / skill |
| GitHub PAT | Worker 触发 `repository_dispatch` |
| Cloudflare 账号 | 部署 Worker（免费） |

## 配置清单

### A. GitHub 仓库 Secrets / Variables

| Name | 必填 | 说明 |
| --- | --- | --- |
| `CURSOR_API_KEY` | ✅ | [Cursor Dashboard → API Keys](https://cursor.com/dashboard?tab=integrations) |
| `FEISHU_WEBHOOK_URL` | ✅ | 结论推群 |
| `FEISHU_WEBHOOK_SECRET` | 按需 | Webhook 签名 |
| `FEISHU_WEBHOOK_KEYWORD` | 按需 | Webhook 关键词 |
| `CURSOR_AGENT_REPO_URL` | 推荐 | Variable，默认本仓库 |
| `CURSOR_AGENT_STARTING_REF` | 可选 | 默认 `main` |

> 不需要在本机跑 `FEISHU_STREAM_ENABLED`，也不需要自建服务器。

### B. GitHub PAT（给 Worker 用）

1. GitHub → Settings → Developer settings → Personal access tokens  
2. 经典 token 勾选 `repo`（或 fine-grained：对该仓库允许 **Contents: Read** + **Actions: Write** / 能创建 `repository_dispatch`）  
3. 复制 token，稍后 `wrangler secret put GITHUB_TOKEN`

### C. 飞书开放平台应用

1. [创建企业自建应用](https://open.feishu.cn/document/develop-an-echo-bot/introduction)  
2. 权限：接收消息、回复消息（`im:message` 等，按控制台提示）  
3. **事件订阅**：
   - 选择 **将事件发送至开发者服务器**（HTTP，不要选长连接）
   - 请求地址填 Worker URL：`https://<name>.workers.dev/`
   - **Encrypt Key 先留空**（本 Worker 默认未实现加密解密；先跑通再加密）
   - Verification Token 可选；若填了，同步配置 Worker secret `FEISHU_VERIFICATION_TOKEN`
   - 添加事件：`im.message.receive_v1`
4. 发布应用，机器人拉进目标群  
5. 记下 `App ID` / `App Secret`

### D. 部署 Cloudflare Worker

```bash
npm i -g wrangler
cd deploy/feishu-cursor-bridge

# 编辑 wrangler.toml 里的 GITHUB_REPO（默认已是本 fork）
wrangler login

wrangler secret put FEISHU_APP_ID
wrangler secret put FEISHU_APP_SECRET
wrangler secret put GITHUB_TOKEN
# 可选：
# wrangler secret put FEISHU_VERIFICATION_TOKEN

wrangler deploy
```

把输出的 `https://xxx.workers.dev/` 填回飞书「事件订阅 → 请求地址」，保存并通过 URL 验证。

## 使用方式

在飞书群 @机器人 或私聊发送：

```text
/策略研究 算力板块量化策略
/tsr 埋伏低位高景气方向
策略研究 半导体供应链
```

预期：
1. **几秒内** 飞书回复「已收到研究请求」  
2. GitHub Actions 出现运行：`每日交易策略研究 (Cursor Skill)`，触发类型 `repository_dispatch`  
3. [cursor.com/agents](https://cursor.com/agents) 出现 Agent  
4. 完成后 **群 Webhook** 收到「交易策略研究」结论卡片  

## 无 Cloudflare 时的临时办法

Actions → **每日交易策略研究 (Cursor Skill)** → Run workflow → 填写 `extra_prompt` = 你的研究焦点。  
效果相同，只是手动点一下，不是飞书直接触发。

## 排障

| 现象 | 检查 |
| --- | --- |
| 飞书无 ACK | Worker 是否部署；事件订阅 URL / 权限 / 是否发布应用；群聊是否 @机器人 |
| 有 ACK 无 Actions | `GITHUB_TOKEN` 权限；`GITHUB_REPO`；Actions 是否 enable；workflow 是否在默认分支 |
| Actions 失败 | Secrets 是否有 `CURSOR_API_KEY`、`FEISHU_WEBHOOK_URL` |
| 有 Actions 无飞书结论 | Webhook / 关键词 / 签名 |

## 安全注意

- PAT 与飞书密钥只放在 Cloudflare Secrets / GitHub Secrets，不要写进代码  
- 建议为桥接单独建一个权限最小的 GitHub PAT，并可随时吊销  
- 生产环境可再加：IP 校验、限流、Encrypt Key 解密（需扩展 Worker）
