# -*- coding: utf-8 -*-
"""Unit tests for scripts/trigger_trading_strategy_research.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "trigger_trading_strategy_research.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "trigger_trading_strategy_research",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_build_research_prompt_includes_skill_path_and_extra() -> None:
    prompt = mod.build_research_prompt(extra="Focus on semiconductor supply chain")
    assert ".cursor/skills/trading-strategy-research/SKILL.md" in prompt
    assert "Focus on semiconductor supply chain" in prompt
    assert "after costs" in prompt.lower() or "realistic costs" in prompt


def test_infer_repo_url_prefers_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURSOR_AGENT_REPO_URL", "https://github.com/acme/demo")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert mod.infer_repo_url() == "https://github.com/acme/demo"


def test_infer_repo_url_from_github_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURSOR_AGENT_REPO_URL", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "czy2014hust/daily_stock_analysis")
    assert mod.infer_repo_url() == "https://github.com/czy2014hust/daily_stock_analysis"


def test_format_feishu_message_includes_header_and_link() -> None:
    text = mod.format_feishu_message(
        result_text="Top strategy: mean reversion sleeve",
        agent_url="https://cursor.com/agents/bc-test",
        run_status="FINISHED",
        report_date="2026-10-03",
    )
    assert "2026-10-03" in text
    assert "FINISHED" in text
    assert "https://cursor.com/agents/bc-test" in text
    assert "mean reversion sleeve" in text


def test_create_agent_run_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_request(method, path, *, api_key, api_base, payload=None, timeout=60.0):
        captured.update(
            {
                "method": method,
                "path": path,
                "api_key": api_key,
                "api_base": api_base,
                "payload": payload,
            }
        )
        return {
            "agent": {
                "id": "bc-1",
                "latestRunId": "run-1",
                "url": "https://cursor.com/agents/bc-1",
            },
            "run": {"id": "run-1"},
        }

    monkeypatch.setattr(mod, "cursor_api_request", fake_request)
    agent_id, run_id, url = mod.create_agent_run(
        api_key="key",
        api_base="https://api.cursor.com",
        prompt="do research",
        repo_url="https://github.com/acme/demo",
        starting_ref="main",
        model_id="composer-2",
        name="Daily Trading Strategy Research",
    )
    assert (agent_id, run_id, url) == ("bc-1", "run-1", "https://cursor.com/agents/bc-1")
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/agents"
    assert captured["payload"]["prompt"]["text"] == "do research"
    assert captured["payload"]["repos"][0]["url"] == "https://github.com/acme/demo"
    assert captured["payload"]["model"]["id"] == "composer-2"
    assert captured["payload"]["autoCreatePR"] is False


def test_wait_for_run_returns_on_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_request(method, path, *, api_key, api_base, payload=None, timeout=60.0):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"id": "run-1", "status": "RUNNING"}
        return {"id": "run-1", "status": "FINISHED", "result": "done"}

    monkeypatch.setattr(mod, "cursor_api_request", fake_request)
    monkeypatch.setattr(mod.time, "sleep", lambda *_args, **_kwargs: None)
    run = mod.wait_for_run(
        api_key="key",
        api_base="https://api.cursor.com",
        agent_id="bc-1",
        run_id="run-1",
        poll_seconds=1,
        timeout_seconds=30,
    )
    assert run["status"] == "FINISHED"
    assert run["result"] == "done"
    assert calls["n"] == 2


def test_run_dry_run_prompt_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = mod.run(["--dry-run-prompt"])
    out = capsys.readouterr().out
    assert code == 0
    assert "trading-strategy-research" in out


def test_run_feishu_test_sends_message(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict = {}

    def fake_send(content: str) -> bool:
        sent["content"] = content
        return True

    monkeypatch.setattr(mod, "send_feishu_report", fake_send)
    code = mod.run(["--feishu-test"])
    assert code == 0
    assert "连通性测试" in sent["content"]
    assert "FEISHU_TEST" in sent["content"]


def test_send_feishu_webhook_posts_card(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"code":0,"msg":"success"}'

    def fake_urlopen(request, timeout=30):
        posts.append(json.loads(request.data.decode("utf-8")))
        return FakeResp()

    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://open.feishu.cn/open-apis/bot/v2/hook/test")
    monkeypatch.delenv("FEISHU_WEBHOOK_SECRET", raising=False)
    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    assert mod._send_feishu_webhook("hello from probe") is True
    assert posts
    assert posts[0]["msg_type"] == "interactive"
    assert "hello from probe" in posts[0]["card"]["elements"][0]["text"]["content"]


def test_run_end_to_end_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setenv("CURSOR_AGENT_REPO_URL", "https://github.com/acme/demo")
    monkeypatch.setenv("SKIP_FEISHU", "true")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()

    monkeypatch.setattr(
        mod,
        "create_agent_run",
        lambda **_kwargs: ("bc-1", "run-1", "https://cursor.com/agents/bc-1"),
    )
    monkeypatch.setattr(
        mod,
        "wait_for_run",
        lambda **_kwargs: {
            "status": "FINISHED",
            "result": "Executive Summary: strategy A leads.",
        },
    )

    code = mod.run([])
    assert code == 0
    summaries = list((tmp_path / "reports").glob("trading_strategy_research_*_summary.md"))
    assert len(summaries) == 1
    assert "strategy A leads" in summaries[0].read_text(encoding="utf-8")


def test_cursor_api_request_surfaces_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHTTPError(HTTPError):
        def __init__(self):
            super().__init__(
                url="https://api.cursor.com/v1/agents",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=None,
            )

        def read(self):
            return b'{"error":"unauthorized"}'

    def fake_urlopen(*_args, **_kwargs):
        raise FakeHTTPError()

    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        mod.cursor_api_request(
            "GET",
            "/v1/agents",
            api_key="bad",
            api_base="https://api.cursor.com",
        )
