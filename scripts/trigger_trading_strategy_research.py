#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger Cursor Cloud Agent for trading-strategy-research and notify Feishu.

Daily flow (also used by GitHub Actions):
1. POST https://api.cursor.com/v1/agents with a prompt that runs the
   ``trading-strategy-research`` skill against this repository.
2. Poll the run until it finishes (or times out).
3. Push the final assistant result to the already-configured Feishu channel
   via lightweight webhook or App Bot.

Required env:
  CURSOR_API_KEY

Optional env:
  CURSOR_AGENT_REPO_URL          default: inferred from git remote / GITHUB_REPOSITORY
  CURSOR_AGENT_STARTING_REF      default: main
  CURSOR_AGENT_MODEL             optional model id for Cloud Agents API
  CURSOR_AGENT_NAME              display name (default: Daily Trading Strategy Research)
  CURSOR_AGENT_POLL_SECONDS      default: 20
  CURSOR_AGENT_TIMEOUT_SECONDS   default: 3600
  CURSOR_API_BASE_URL            default: https://api.cursor.com
  TRADING_STRATEGY_RESEARCH_PROMPT_EXTRA  appended to the agent prompt
  SKIP_FEISHU                    set to 1/true to skip Feishu delivery

Feishu uses the project's existing config (FEISHU_WEBHOOK_URL or App Bot keys).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from urllib.request import urlopen  # noqa: F401 — re-exported for tests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.cursor_cloud_agent import (  # noqa: E402
    DEFAULT_API_BASE,
    SKILL_RELATIVE_PATH,
    TERMINAL_STATUSES,
    build_research_prompt as _build_research_prompt,
    create_agent_run,
    cursor_api_request,
    env_bool as _env_bool,
    env_int as _env_int,
    infer_repo_url,
    wait_for_run,
)
from src.services.feishu_webhook_lite import (  # noqa: E402
    feishu_webhook_security_fields as _feishu_webhook_security_fields,
    send_feishu_webhook as _send_feishu_webhook_impl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("trading-strategy-research-trigger")


def build_research_prompt(*, extra: str = "") -> str:
    """Compatibility wrapper used by CLI and unit tests."""
    return _build_research_prompt(extra=extra, source="scheduled")


def _send_feishu_webhook(content: str) -> bool:
    """Compatibility wrapper; implementation lives in feishu_webhook_lite."""
    return _send_feishu_webhook_impl(content)


def _feishu_config_from_env() -> Any:
    """Build a minimal config object for FeishuSender without full Config bootstrap."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)

    def _get(*names: str) -> str:
        for name in names:
            value = (os.getenv(name) or "").strip()
            if value:
                return value
        return ""

    max_bytes_raw = _get("FEISHU_MAX_BYTES")
    try:
        max_bytes = int(max_bytes_raw) if max_bytes_raw else 20000
    except ValueError:
        max_bytes = 20000

    return SimpleNamespace(
        feishu_webhook_url=_get("FEISHU_WEBHOOK_URL") or None,
        feishu_webhook_secret=_get("FEISHU_WEBHOOK_SECRET") or None,
        feishu_webhook_keyword=_get("FEISHU_WEBHOOK_KEYWORD") or None,
        feishu_app_id=_get("FEISHU_APP_ID") or None,
        feishu_app_secret=_get("FEISHU_APP_SECRET") or None,
        feishu_chat_id=_get("FEISHU_CHAT_ID") or None,
        feishu_receive_id_type=_get("FEISHU_RECEIVE_ID_TYPE") or "chat_id",
        feishu_domain=_get("FEISHU_DOMAIN") or "feishu",
        feishu_max_bytes=max_bytes,
        feishu_send_as_file=False,
    )


def send_feishu_report(content: str) -> bool:
    """Send report text through the project's configured Feishu channel."""
    webhook_url = (os.getenv("FEISHU_WEBHOOK_URL") or "").strip()
    if not webhook_url:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        webhook_url = (os.getenv("FEISHU_WEBHOOK_URL") or "").strip()

    if webhook_url:
        return _send_feishu_webhook(content)

    try:
        from src.notification_sender.feishu_sender import FeishuSender
    except Exception as exc:  # pragma: no cover - import graph varies by env
        logger.error(
            "Feishu App Bot path unavailable (%s). Set FEISHU_WEBHOOK_URL or "
            "install full project dependencies for App Bot.",
            exc,
        )
        return False

    config = _feishu_config_from_env()
    sender = FeishuSender(config)
    has_app_bot = bool(
        getattr(sender, "_feishu_app_id", None)
        and getattr(sender, "_feishu_app_secret", None)
        and getattr(sender, "_feishu_chat_id", None)
    )
    if not has_app_bot:
        logger.error(
            "Feishu is not configured. Set FEISHU_WEBHOOK_URL or "
            "FEISHU_APP_ID + FEISHU_APP_SECRET + FEISHU_CHAT_ID."
        )
        return False

    keyword = (getattr(sender, "_feishu_keyword", None) or "").strip()
    body = content.strip()
    if keyword and keyword not in body:
        body = f"{keyword}\n\n{body}"
    ok = sender.send_to_feishu(body)
    if ok:
        logger.info("Feishu App Bot delivery succeeded")
    else:
        logger.error("Feishu App Bot delivery failed")
    return ok


def format_feishu_message(
    *,
    result_text: str,
    agent_url: str,
    run_status: str,
    report_date: str,
) -> str:
    header = f"📈 交易策略研究日报 ({report_date})"
    status_line = f"Cursor Cloud Agent 状态: {run_status}"
    link_line = f"详情: {agent_url}" if agent_url else ""
    body = (result_text or "").strip() or "（无最终文本结果；请打开 Cursor Agent 链接查看完整过程）"
    parts = [header, status_line]
    if link_line:
        parts.append(link_line)
    parts.extend(["", body])
    return "\n".join(parts)


def run(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch Cursor trading-strategy-research and notify Feishu",
    )
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Print the agent prompt and exit without calling Cursor API",
    )
    parser.add_argument(
        "--feishu-test",
        action="store_true",
        help="Send a Feishu connectivity test message and exit (no Cursor API)",
    )
    parser.add_argument(
        "--skip-feishu",
        action="store_true",
        help="Do not send Feishu notification after the run",
    )
    parser.add_argument(
        "--extra-prompt",
        default="",
        help="Extra focus instructions appended to the research prompt",
    )
    args = parser.parse_args(argv)

    extra = (args.extra_prompt or os.getenv("TRADING_STRATEGY_RESEARCH_PROMPT_EXTRA") or "").strip()
    prompt = build_research_prompt(extra=extra)
    if args.dry_run_prompt:
        print(prompt)
        return 0

    if args.feishu_test or _env_bool("FEISHU_TEST_ONLY", False):
        report_date = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        message = format_feishu_message(
            result_text=(
                "这是一条连通性测试消息（未调用 Cursor API）。\n"
                "若你能看到本条，说明飞书推送配置可用；"
                "每日 09:00 的 trading-strategy-research 将使用同一通道推送摘要。"
            ),
            agent_url="",
            run_status="FEISHU_TEST",
            report_date=report_date,
        )
        logger.info("Sending Feishu connectivity test message")
        return 0 if send_feishu_report(message) else 1

    api_key = (os.getenv("CURSOR_API_KEY") or "").strip()
    if not api_key:
        logger.error("CURSOR_API_KEY is required")
        return 2

    api_base = (os.getenv("CURSOR_API_BASE_URL") or DEFAULT_API_BASE).strip() or DEFAULT_API_BASE
    repo_url = infer_repo_url()
    starting_ref = (os.getenv("CURSOR_AGENT_STARTING_REF") or "main").strip() or "main"
    model_id = (os.getenv("CURSOR_AGENT_MODEL") or "").strip()
    name = (os.getenv("CURSOR_AGENT_NAME") or "Daily Trading Strategy Research").strip()
    poll_seconds = _env_int("CURSOR_AGENT_POLL_SECONDS", 20)
    timeout_seconds = _env_int("CURSOR_AGENT_TIMEOUT_SECONDS", 3600)
    skip_feishu = args.skip_feishu or _env_bool("SKIP_FEISHU", False)

    logger.info("Creating Cursor agent for %s @ %s", repo_url, starting_ref)
    agent_id, run_id, agent_url = create_agent_run(
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        repo_url=repo_url,
        starting_ref=starting_ref,
        model_id=model_id,
        name=name,
    )
    logger.info("Agent created: id=%s run=%s url=%s", agent_id, run_id, agent_url or "(none)")

    run_payload = wait_for_run(
        api_key=api_key,
        api_base=api_base,
        agent_id=agent_id,
        run_id=run_id,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    status = str(run_payload.get("status") or "").upper()
    result_text = str(run_payload.get("result") or "").strip()
    logger.info("Run finished with status=%s result_chars=%d", status, len(result_text))

    report_date = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    message = format_feishu_message(
        result_text=result_text,
        agent_url=agent_url,
        run_status=status,
        report_date=report_date,
    )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    local_report = reports_dir / f"trading_strategy_research_{report_date.replace('-', '')}_summary.md"
    local_report.write_text(message, encoding="utf-8")
    logger.info("Saved local summary: %s", local_report)

    if status != "FINISHED":
        logger.error("Cursor run did not finish successfully: %s", status)
        if not skip_feishu:
            send_feishu_report(message)
        return 1

    if skip_feishu:
        logger.info("Skipping Feishu delivery (--skip-feishu / SKIP_FEISHU)")
        return 0

    return 0 if send_feishu_report(message) else 1


if __name__ == "__main__":
    sys.exit(run())
