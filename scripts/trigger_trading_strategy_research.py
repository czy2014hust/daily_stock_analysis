#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger Cursor Cloud Agent for trading-strategy-research and notify Feishu.

Daily flow (also used by GitHub Actions):
1. POST https://api.cursor.com/v1/agents with a prompt that runs the
   ``trading-strategy-research`` skill against this repository.
2. Poll the run until it finishes (or times out).
3. Push the final assistant result to the already-configured Feishu channel
   via the project's ``FeishuSender`` (webhook or App Bot).

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
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("trading-strategy-research-trigger")

DEFAULT_API_BASE = "https://api.cursor.com"
TERMINAL_STATUSES = frozenset({"FINISHED", "ERROR", "CANCELLED", "EXPIRED"})
SKILL_RELATIVE_PATH = ".cursor/skills/trading-strategy-research/SKILL.md"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def infer_repo_url() -> str:
    """Infer GitHub HTTPS repo URL for Cloud Agents API."""
    explicit = (os.getenv("CURSOR_AGENT_REPO_URL") or "").strip()
    if explicit:
        return explicit

    github_repo = (os.getenv("GITHUB_REPOSITORY") or "").strip()
    if github_repo:
        return f"https://github.com/{github_repo}"

    try:
        completed = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        remote = (completed.stdout or "").strip()
    except OSError:
        remote = ""

    if remote.endswith(".git"):
        remote = remote[:-4]
    if remote.startswith("git@github.com:"):
        return "https://github.com/" + remote[len("git@github.com:") :]
    if remote.startswith("ssh://git@github.com/"):
        return "https://github.com/" + remote[len("ssh://git@github.com/") :]
    if remote.startswith("https://github.com/"):
        return remote
    raise RuntimeError(
        "Unable to infer repository URL. Set CURSOR_AGENT_REPO_URL="
        "https://github.com/<owner>/<repo>"
    )


def build_research_prompt(*, extra: str = "") -> str:
    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    extra_block = (extra or "").strip()
    extra_section = f"\n\nAdditional focus from operator:\n{extra_block}\n" if extra_block else ""
    return f"""You are running a scheduled daily job for this repository.

Follow the Cursor skill at `{SKILL_RELATIVE_PATH}` strictly
(`/trading-strategy-research`).

Task for {today} (timezone: Asia/Shanghai market context preferred when relevant):
1. Refresh market/sector regime with the latest reliable data available in this repo's tools.
2. Ideate, implement, and backtest 3–10 fundamentally different strategy candidates under realistic costs.
3. Rank survivors against the quality bar in the skill (after costs, robustness, clear thesis).
4. Write the final report using the skill's output format.
5. Save the full Markdown report under `reports/trading_strategy_research_{today.replace('-', '')}.md`.
6. Keep the final assistant reply as a concise Feishu-ready summary (executive summary + top 3–5 recommendations + key risks). Prefer Chinese if the repo defaults to Chinese reports.

Hard constraints:
- Do not claim edge without backtest evidence.
- Model fees/slippage/liquidity.
- If data is missing, document the gap instead of inventing fills.
- Prefer committing the report file; opening a PR is optional (`autoCreatePR` may be false).
{extra_section}
"""


def cursor_api_request(
    method: str,
    path: str,
    *,
    api_key: str,
    api_base: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Call Cursor Cloud Agents API with Basic auth (api_key as username)."""
    url = f"{api_base.rstrip('/')}{path}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "daily-stock-analysis-trading-strategy-research/1.0",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    # Basic auth with empty password, matching Cursor docs (`-u YOUR_API_KEY:`).
    import base64

    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    headers["Authorization"] = f"Basic {token}"

    request = Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cursor API {method} {path} failed: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cursor API {method} {path} network error: {exc}") from exc

    if not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cursor API returned non-JSON body: {body[:500]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Cursor API returned unexpected JSON type: {type(parsed).__name__}")
    return parsed


def create_agent_run(
    *,
    api_key: str,
    api_base: str,
    prompt: str,
    repo_url: str,
    starting_ref: str,
    model_id: str = "",
    name: str = "",
) -> Tuple[str, str, str]:
    """Create a cloud agent + initial run. Returns (agent_id, run_id, agent_url)."""
    payload: Dict[str, Any] = {
        "prompt": {"text": prompt},
        "repos": [
            {
                "url": repo_url,
                "startingRef": starting_ref,
            }
        ],
        "autoCreatePR": False,
        "workOnCurrentBranch": False,
    }
    if name:
        payload["name"] = name[:100]
    if model_id:
        payload["model"] = {"id": model_id}

    response = cursor_api_request(
        "POST",
        "/v1/agents",
        api_key=api_key,
        api_base=api_base,
        payload=payload,
        timeout=120.0,
    )
    agent = response.get("agent") or {}
    run = response.get("run") or {}
    agent_id = str(agent.get("id") or "").strip()
    run_id = str(run.get("id") or agent.get("latestRunId") or "").strip()
    agent_url = str(agent.get("url") or "").strip()
    if not agent_id or not run_id:
        raise RuntimeError(f"Cursor API create response missing ids: {response}")
    return agent_id, run_id, agent_url


def wait_for_run(
    *,
    api_key: str,
    api_base: str,
    agent_id: str,
    run_id: str,
    poll_seconds: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Poll Get A Run until terminal status or timeout."""
    deadline = time.monotonic() + max(30, timeout_seconds)
    last_status = ""
    while time.monotonic() < deadline:
        run = cursor_api_request(
            "GET",
            f"/v1/agents/{agent_id}/runs/{run_id}",
            api_key=api_key,
            api_base=api_base,
            timeout=60.0,
        )
        status = str(run.get("status") or "").upper()
        if status != last_status:
            logger.info("Cursor run %s status=%s", run_id, status or "?")
            last_status = status
        if status in TERMINAL_STATUSES:
            return run
        time.sleep(max(5, poll_seconds))
    raise TimeoutError(
        f"Cursor run {run_id} did not finish within {timeout_seconds}s (last={last_status or 'unknown'})"
    )


def _feishu_config_from_env() -> Any:
    """Build a minimal config object for FeishuSender without full Config bootstrap."""
    from types import SimpleNamespace

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
    from src.notification_sender.feishu_sender import FeishuSender

    config = _feishu_config_from_env()
    sender = FeishuSender(config)
    has_webhook = bool(getattr(sender, "_feishu_url", None))
    has_app_bot = bool(
        getattr(sender, "_feishu_app_id", None)
        and getattr(sender, "_feishu_app_secret", None)
        and getattr(sender, "_feishu_chat_id", None)
    )
    if not has_webhook and not has_app_bot:
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
        logger.info("Feishu delivery succeeded")
    else:
        logger.error("Feishu delivery failed")
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
