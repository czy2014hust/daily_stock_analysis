/**
 * Feishu → GitHub Actions bridge (no VPS required).
 *
 * Flow:
 * 1. Feishu App event subscription POSTs im.message.receive_v1 here
 * 2. Worker ACKs in Feishu (immediate reply)
 * 3. Worker fires GitHub repository_dispatch
 * 4. Actions runs trading-strategy-research via Cursor API and pushes result
 *    to FEISHU_WEBHOOK_URL (configured as a GitHub Secret)
 *
 * Secrets (wrangler secret put ...):
 *   FEISHU_APP_ID
 *   FEISHU_APP_SECRET
 *   GITHUB_TOKEN          # classic PAT with "repo" scope, or fine-grained: Actions write
 *
 * Optional secrets:
 *   FEISHU_VERIFICATION_TOKEN
 *
 * Vars (wrangler.toml [vars]):
 *   GITHUB_REPO           # owner/repo
 *   GITHUB_EVENT_TYPE     # default feishu-strategy-research
 *   FEISHU_DOMAIN         # feishu | lark
 *   COMMAND_PREFIXES      # comma-separated; empty = any text
 */

const FEISHU_HOST = {
  feishu: "https://open.feishu.cn",
  lark: "https://open.larksuite.com",
};

export default {
  async fetch(request, env, ctx) {
    if (request.method !== "POST") {
      return json({ ok: true, service: "feishu-cursor-bridge" });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "invalid json" }, 400);
    }

    // URL verification challenge from Feishu open platform
    if (body.type === "url_verification" || body.challenge) {
      if (env.FEISHU_VERIFICATION_TOKEN) {
        const token = body.token || body.header?.token;
        if (token && token !== env.FEISHU_VERIFICATION_TOKEN) {
          return json({ error: "bad verification token" }, 403);
        }
      }
      return json({ challenge: body.challenge });
    }

    // Acknowledge Feishu quickly; process async so Feishu does not retry
    ctx.waitUntil(handleEvent(body, env));
    return json({ code: 0 });
  },
};

async function handleEvent(body, env) {
  try {
    const header = body.header || {};
    const eventType = header.event_type || body.event?.type || "";
    if (eventType && eventType !== "im.message.receive_v1") {
      return;
    }

    const event = body.event || {};
    const message = event.message || {};
    const sender = event.sender || {};

    if ((message.message_type || "") !== "text") {
      return;
    }

    let text = "";
    try {
      text = JSON.parse(message.content || "{}").text || "";
    } catch {
      text = message.content || "";
    }
    text = stripMentions(text).trim();
    if (!text) {
      return;
    }

    const focus = extractFocus(text, env.COMMAND_PREFIXES || "");
    if (focus === null) {
      // Not a strategy-research command; ignore quietly
      return;
    }
    if (!focus) {
      await replyFeishu(
        env,
        message.message_id,
        "用法: `/策略研究 <研究焦点>`\n示例: `/策略研究 算力板块量化策略`",
      );
      return;
    }

    await replyFeishu(
      env,
      message.message_id,
      `📥 已收到研究请求\n焦点: ${truncate(focus, 80)}\n` +
        "已提交 GitHub Actions → Cursor `/trading-strategy-research`，" +
        "完成后会推送到飞书群（通常数分钟到数十分钟）。",
    );

    await dispatchGitHub(env, {
      text: focus,
      message_id: message.message_id || "",
      chat_id: message.chat_id || "",
      chat_type: message.chat_type || "",
      user_id: sender.sender_id?.open_id || sender.sender_id?.user_id || "",
    });
  } catch (err) {
    console.error("handleEvent failed", err && err.stack ? err.stack : err);
  }
}

function stripMentions(text) {
  return String(text || "")
    .replace(/@_user_\d+\s*/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Returns:
 *   null  — message is not a research command (ignore)
 *   ""    — command without focus (show usage)
 *   string — focus text
 */
function extractFocus(text, prefixesCsv) {
  const prefixes = String(prefixesCsv || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  // Empty prefixes: any non-empty text is treated as research focus
  if (!prefixes.length) {
    return text;
  }

  for (const prefix of prefixes) {
    if (text === prefix) {
      return "";
    }
    if (text.startsWith(prefix + " ") || text.startsWith(prefix + "\n")) {
      return text.slice(prefix.length).trim();
    }
    // Allow "/策略研究焦点" without space for short prefixes only when followed by non-ascii/word
    if (text.startsWith(prefix) && text.length > prefix.length) {
      const rest = text.slice(prefix.length).trim();
      if (rest) {
        return rest;
      }
    }
  }
  return null;
}

async function dispatchGitHub(env, payload) {
  const repo = (env.GITHUB_REPO || "").trim();
  const token = (env.GITHUB_TOKEN || "").trim();
  const eventType = (env.GITHUB_EVENT_TYPE || "feishu-strategy-research").trim();
  if (!repo || !token) {
    throw new Error("GITHUB_REPO / GITHUB_TOKEN not configured on Worker");
  }

  const resp = await fetch(`https://api.github.com/repos/${repo}/dispatches`, {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "feishu-cursor-bridge",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      event_type: eventType,
      client_payload: payload,
    }),
  });

  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`GitHub dispatch failed: HTTP ${resp.status} ${detail.slice(0, 300)}`);
  }
}

async function replyFeishu(env, messageId, text) {
  if (!messageId) {
    return;
  }
  const token = await getFeishuTenantToken(env);
  if (!token) {
    console.error("skip reply: no tenant access token");
    return;
  }
  const host = FEISHU_HOST[(env.FEISHU_DOMAIN || "feishu").toLowerCase()] || FEISHU_HOST.feishu;

  const card = {
    config: { wide_screen_mode: true },
    header: {
      title: { tag: "plain_text", content: "交易策略研究" },
    },
    elements: [
      {
        tag: "div",
        text: { tag: "lark_md", content: text },
      },
    ],
  };

  const resp = await fetch(`${host}/open-apis/im/v1/messages/${messageId}/reply`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json; charset=utf-8",
    },
    body: JSON.stringify({
      content: JSON.stringify(card),
      msg_type: "interactive",
    }),
  });

  if (!resp.ok) {
    // Fallback to plain text
    const textResp = await fetch(`${host}/open-apis/im/v1/messages/${messageId}/reply`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json; charset=utf-8",
      },
      body: JSON.stringify({
        content: JSON.stringify({ text }),
        msg_type: "text",
      }),
    });
    if (!textResp.ok) {
      console.error("feishu reply failed", await textResp.text());
    }
  }
}

async function getFeishuTenantToken(env) {
  const appId = (env.FEISHU_APP_ID || "").trim();
  const appSecret = (env.FEISHU_APP_SECRET || "").trim();
  if (!appId || !appSecret) {
    return "";
  }
  const host = FEISHU_HOST[(env.FEISHU_DOMAIN || "feishu").toLowerCase()] || FEISHU_HOST.feishu;
  const resp = await fetch(`${host}/open-apis/auth/v3/tenant_access_token/internal`, {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify({ app_id: appId, app_secret: appSecret }),
  });
  const data = await resp.json();
  if (!resp.ok || data.code !== 0) {
    console.error("tenant token failed", data);
    return "";
  }
  return data.tenant_access_token || "";
}

function truncate(s, n) {
  return s.length <= n ? s : s.slice(0, n) + "…";
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}
