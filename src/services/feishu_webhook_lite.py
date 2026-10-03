# -*- coding: utf-8 -*-
"""Lightweight Feishu webhook sender (stdlib only).

Safe for GitHub Actions / bot workers without pulling the full FeishuSender
import graph (pydantic, formatters, etc.).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def feishu_webhook_security_fields(secret: str) -> Dict[str, Any]:
    """Build Feishu webhook signature fields when a secret is configured."""
    secret = (secret or "").strip()
    if not secret:
        return {}

    timestamp = str(int(time.time()))
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return {"timestamp": timestamp, "sign": base64.b64encode(digest).decode("utf-8")}


def send_feishu_webhook(
    content: str,
    *,
    title: str = "交易策略研究日报",
) -> bool:
    """Post an interactive card (text fallback) to FEISHU_WEBHOOK_URL."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass

    webhook_url = (os.getenv("FEISHU_WEBHOOK_URL") or "").strip()
    if not webhook_url:
        return False

    keyword = (os.getenv("FEISHU_WEBHOOK_KEYWORD") or "").strip()
    body = content.strip()
    if keyword and keyword not in body:
        body = f"{keyword}\n\n{body}"

    security = feishu_webhook_security_fields(os.getenv("FEISHU_WEBHOOK_SECRET") or "")
    card_payload: Dict[str, Any] = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": body},
                }
            ],
        },
    }
    card_payload.update(security)
    text_payload: Dict[str, Any] = {
        "msg_type": "text",
        "content": {"text": body},
    }
    text_payload.update(security)

    def _post(payload: Dict[str, Any]) -> Tuple[bool, str]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "daily-stock-analysis-feishu-webhook-lite/1.0",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return False, f"HTTP {exc.code}: {detail[:300]}"
        except URLError as exc:
            return False, f"network error: {exc}"
        try:
            result = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return False, f"non-JSON response: {raw[:200]}"
        if not isinstance(result, dict):
            return False, f"unexpected response: {raw[:200]}"
        code = result.get("StatusCode", result.get("code", 0))
        if code in (0, "0"):
            return True, "ok"
        return False, str(result)[:300]

    ok, detail = _post(card_payload)
    if ok:
        logger.info("Feishu webhook card delivery succeeded")
        return True
    logger.warning("Feishu webhook card failed (%s); trying text fallback", detail)
    ok, detail = _post(text_payload)
    if ok:
        logger.info("Feishu webhook text delivery succeeded")
        return True
    logger.error("Feishu webhook delivery failed: %s", detail)
    return False
