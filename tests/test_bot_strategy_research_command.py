# -*- coding: utf-8 -*-
"""Tests for bot /strategy_research command."""

from __future__ import annotations

import threading
from datetime import datetime

import pytest

from bot.commands.strategy_research import (
    StrategyResearchCommand,
    reset_strategy_research_job_state,
)
from bot.models import BotMessage, ChatType


@pytest.fixture(autouse=True)
def _reset_job_flag():
    reset_strategy_research_job_state()
    yield
    reset_strategy_research_job_state()


def _message(content: str = "/策略研究 算力") -> BotMessage:
    return BotMessage(
        platform="feishu",
        message_id="om_test",
        user_id="ou_user",
        user_name="tester",
        chat_id="oc_chat",
        chat_type=ChatType.GROUP,
        content=content,
        mentioned=True,
        timestamp=datetime.now(),
    )


def test_strategy_research_requires_focus_text() -> None:
    cmd = StrategyResearchCommand()
    resp = cmd.execute(_message("/策略研究"), [])
    assert "用法" in resp.text


def test_strategy_research_requires_cursor_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    cmd = StrategyResearchCommand()
    resp = cmd.execute(_message(), ["算力板块"])
    assert "CURSOR_API_KEY" in resp.text


def test_strategy_research_acks_and_runs_background(monkeypatch: pytest.MonkeyPatch) -> None:
    done = threading.Event()
    delivered: dict = {}

    def fake_run(**kwargs):
        assert "算力" in kwargs["focus_text"]
        assert kwargs["source"] == "bot"
        return {
            "status": "FINISHED",
            "result": "Top strategy: relative strength sleeve",
            "agent_id": "bc-1",
            "run_id": "run-1",
            "agent_url": "https://cursor.com/agents/bc-1",
        }

    def fake_deliver(self, message, text):
        delivered["text"] = text
        delivered["message_id"] = message.message_id
        done.set()

    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.services.cursor_cloud_agent.run_trading_strategy_research",
        fake_run,
    )
    monkeypatch.setattr(StrategyResearchCommand, "_deliver", fake_deliver)

    cmd = StrategyResearchCommand()
    resp = cmd.execute(_message(), ["算力板块量化策略"])
    assert "已收到研究请求" in resp.text
    assert done.wait(timeout=3.0)
    assert "Top strategy" in delivered.get("text", "")
    assert "FINISHED" in delivered.get("text", "")


def test_chinese_command_routing_maps_to_strategy_research() -> None:
    msg = _message("策略研究 半导体供应链")
    cmd, args = msg.get_command_and_args("/")
    assert cmd == "strategy_research"
    assert args == ["半导体供应链"]
