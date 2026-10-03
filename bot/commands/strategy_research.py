# -*- coding: utf-8 -*-
"""Feishu/text command: run trading-strategy-research via Cursor Cloud Agents API.

Usage:
    /tsr 算力板块量化策略
    /策略研究 关注半导体供应链相对强弱
    策略研究 埋伏低位高景气方向

Flow:
1. Ack immediately in Feishu (command returns quickly).
2. Background thread launches Cursor Cloud Agent with the trading-strategy-research skill.
3. When finished, reply in the same Feishu thread and optionally broadcast via webhook.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import List, Optional

from bot.commands.base import BotCommand
from bot.models import BotMessage, BotResponse

logger = logging.getLogger(__name__)

_JOB_LOCK = threading.Lock()
_JOB_ACTIVE = False
_MAX_REPLY_CHARS = 3500


class StrategyResearchCommand(BotCommand):
    """Trigger Cursor skill trading-strategy-research from Feishu text."""

    @property
    def name(self) -> str:
        return "strategy_research"

    @property
    def aliases(self) -> List[str]:
        return [
            "tsr",
            "策略研究",
            "交易策略研究",
            "trading-strategy-research",
            "trading_strategy_research",
        ]

    @property
    def description(self) -> str:
        return "调用 Cursor trading-strategy-research skill，对文本做策略研究并回飞书"

    @property
    def usage(self) -> str:
        return "/策略研究 <研究焦点文本>"

    def execute(self, message: BotMessage, args: List[str]) -> BotResponse:
        focus = " ".join(args).strip()
        if not focus:
            return BotResponse.text_response(
                f"用法: `{self.usage}`\n"
                "示例: `/策略研究 算力板块量化策略，关注相对强弱与回撤`\n"
                "示例: `/tsr 埋伏低位高景气方向`"
            )

        api_key = (os.getenv("CURSOR_API_KEY") or "").strip()
        if not api_key:
            return BotResponse.text_response(
                "⚠️ 未配置 `CURSOR_API_KEY`，无法调用 Cursor Cloud Agents API。\n"
                "请到 Cursor Dashboard → API Keys 创建后写入 `.env` / 部署环境。"
            )

        global _JOB_ACTIVE
        with _JOB_LOCK:
            if _JOB_ACTIVE:
                return BotResponse.text_response(
                    "⏳ 已有交易策略研究任务进行中，请稍后再试。"
                )
            _JOB_ACTIVE = True

        worker = threading.Thread(
            target=self._run_job,
            kwargs={"message": message, "focus": focus},
            name="strategy-research-cursor",
            daemon=True,
        )
        worker.start()

        preview = focus if len(focus) <= 80 else focus[:80] + "…"
        return BotResponse.markdown_response(
            "📥 **已收到研究请求**\n"
            f"焦点: {preview}\n"
            "正在调用 Cursor skill `/trading-strategy-research`，完成后会回复本会话"
            "（可能需要数分钟到数十分钟）。"
        )

    def _run_job(self, *, message: BotMessage, focus: str) -> None:
        global _JOB_ACTIVE
        try:
            from src.services.cursor_cloud_agent import run_trading_strategy_research

            logger.info(
                "[StrategyResearch] start focus=%s user=%s chat=%s",
                focus[:120],
                message.user_id,
                message.chat_id,
            )
            outcome = run_trading_strategy_research(focus_text=focus, source="bot")
            status = outcome.get("status") or "UNKNOWN"
            result = (outcome.get("result") or "").strip()
            agent_url = (outcome.get("agent_url") or "").strip()
            report_date = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

            if not result:
                result = "（无最终文本结果；请打开 Cursor Agent 链接查看完整过程）"

            body = self._format_result(
                focus=focus,
                status=status,
                result=result,
                agent_url=agent_url,
                report_date=report_date,
            )
            self._deliver(message, body)
        except Exception as exc:
            logger.error("[StrategyResearch] failed: %s", exc, exc_info=True)
            self._deliver(
                message,
                f"❌ 交易策略研究失败: {exc}",
            )
        finally:
            with _JOB_LOCK:
                _JOB_ACTIVE = False

    @staticmethod
    def _format_result(
        *,
        focus: str,
        status: str,
        result: str,
        agent_url: str,
        report_date: str,
    ) -> str:
        focus_line = focus if len(focus) <= 120 else focus[:120] + "…"
        truncated = result
        if len(truncated) > _MAX_REPLY_CHARS:
            truncated = truncated[:_MAX_REPLY_CHARS] + "\n\n…（已截断，完整过程见 Cursor Agent 链接）"

        lines = [
            f"📈 **交易策略研究结论** ({report_date})",
            f"焦点: {focus_line}",
            f"Cursor 状态: {status}",
        ]
        if agent_url:
            lines.append(f"详情: {agent_url}")
        lines.extend(["", truncated])
        return "\n".join(lines)

    def _deliver(self, message: BotMessage, text: str) -> None:
        """Reply in Feishu thread when possible; also broadcast via webhook if configured."""
        replied = False
        if (message.platform or "").lower() == "feishu" and message.message_id:
            replied = self._reply_feishu_stream(message, text)

        webhook_ok = self._broadcast_feishu_webhook(text)
        if not replied and not webhook_ok:
            logger.error(
                "[StrategyResearch] unable to deliver result "
                "(stream reply failed and webhook unavailable/failed)"
            )

    def _reply_feishu_stream(self, message: BotMessage, text: str) -> bool:
        try:
            from bot.platforms.feishu_stream import FEISHU_SDK_AVAILABLE, FeishuReplyClient
            from src.config import get_config

            if not FEISHU_SDK_AVAILABLE:
                logger.warning("[StrategyResearch] lark-oapi unavailable for stream reply")
                return False

            config = get_config()
            app_id = (getattr(config, "feishu_app_id", None) or os.getenv("FEISHU_APP_ID") or "").strip()
            app_secret = (
                getattr(config, "feishu_app_secret", None) or os.getenv("FEISHU_APP_SECRET") or ""
            ).strip()
            if not app_id or not app_secret:
                logger.warning("[StrategyResearch] FEISHU_APP_ID/SECRET missing for stream reply")
                return False

            client = FeishuReplyClient(
                app_id,
                app_secret,
                domain=getattr(config, "feishu_domain", None),
            )
            ok = client.reply_text(
                message_id=message.message_id,
                text=text,
                at_user=True,
                user_id=message.user_id or None,
            )
            if ok:
                logger.info("[StrategyResearch] Feishu stream reply succeeded")
            return bool(ok)
        except Exception as exc:
            logger.error("[StrategyResearch] Feishu stream reply error: %s", exc, exc_info=True)
            return False

    @staticmethod
    def _broadcast_feishu_webhook(text: str) -> bool:
        """Optional group broadcast via FEISHU_WEBHOOK_URL (same path as daily job)."""
        webhook = (os.getenv("FEISHU_WEBHOOK_URL") or "").strip()
        if not webhook:
            return False
        try:
            from src.services.feishu_webhook_lite import send_feishu_webhook

            ok = send_feishu_webhook(text, title="交易策略研究结论")
            if ok:
                logger.info("[StrategyResearch] Feishu webhook broadcast succeeded")
            return bool(ok)
        except Exception as exc:
            logger.warning("[StrategyResearch] webhook broadcast skipped/failed: %s", exc)
            return False


def reset_strategy_research_job_state() -> None:
    """Test helper to clear the in-flight job flag."""
    global _JOB_ACTIVE
    with _JOB_LOCK:
        _JOB_ACTIVE = False
