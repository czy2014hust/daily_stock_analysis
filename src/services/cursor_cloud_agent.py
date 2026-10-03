# -*- coding: utf-8 -*-
"""Cursor Cloud Agents API helpers.

Used by:
- scripts/trigger_trading_strategy_research.py (scheduled job)
- bot/commands/strategy_research.py (Feishu / strategy research command)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_BASE = "https://api.cursor.com"
TERMINAL_STATUSES = frozenset({"FINISHED", "ERROR", "CANCELLED", "EXPIRED"})
SKILL_RELATIVE_PATH = ".cursor/skills/trading-strategy-research/SKILL.md"


def env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
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


def build_research_prompt(*, extra: str = "", source: str = "scheduled") -> str:
    """Build the Cloud Agent prompt for trading-strategy-research."""
    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    extra_block = (extra or "").strip()
    if source == "bot":
        focus = extra_block or "（用户未提供额外焦点，请做当日市场与策略扫描）"
        return f"""You are handling an interactive Feishu bot request for this repository.

Follow the Cursor skill at `{SKILL_RELATIVE_PATH}` strictly
(`/trading-strategy-research`).

User focus text (treat as the research topic / constraints):
{focus}

Task for {today} (timezone: Asia/Shanghai market context preferred when relevant):
1. Ground the research in the user focus above (sector, ticker, theme, or question).
2. Ideate, implement, and backtest 3–10 fundamentally different strategy candidates under realistic costs when data allows; if the focus is narrow, still compare multiple approaches.
3. Rank survivors against the quality bar in the skill (after costs, robustness, clear thesis).
4. Write the final report using the skill's output format.
5. Save the full Markdown report under `reports/trading_strategy_research_{today.replace('-', '')}.md` when practical.
6. Keep the final assistant reply as a concise Feishu-ready summary in Chinese:
   executive summary + top recommendations + key risks + invalidation conditions.

Hard constraints:
- Do not claim edge without backtest evidence.
- Model fees/slippage/liquidity.
- If data is missing, document the gap instead of inventing fills.
- Prefer not opening a PR (`autoCreatePR` may be false).
"""

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
        "User-Agent": "daily-stock-analysis-cursor-cloud-agent/1.0",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

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
        "repos": [{"url": repo_url, "startingRef": starting_ref}],
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


def run_trading_strategy_research(
    *,
    focus_text: str = "",
    source: str = "scheduled",
    name: str = "",
) -> Dict[str, Any]:
    """Launch trading-strategy-research and wait for completion.

    Returns dict with keys: status, result, agent_id, run_id, agent_url, prompt.
    """
    api_key = (os.getenv("CURSOR_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("CURSOR_API_KEY is required")

    api_base = (os.getenv("CURSOR_API_BASE_URL") or DEFAULT_API_BASE).strip() or DEFAULT_API_BASE
    repo_url = infer_repo_url()
    starting_ref = (os.getenv("CURSOR_AGENT_STARTING_REF") or "main").strip() or "main"
    model_id = (os.getenv("CURSOR_AGENT_MODEL") or "").strip()
    poll_seconds = env_int("CURSOR_AGENT_POLL_SECONDS", 20)
    timeout_seconds = env_int("CURSOR_AGENT_TIMEOUT_SECONDS", 3600)
    display_name = (
        name
        or (os.getenv("CURSOR_AGENT_NAME") or "").strip()
        or ("Feishu Trading Strategy Research" if source == "bot" else "Daily Trading Strategy Research")
    )
    prompt = build_research_prompt(extra=focus_text, source=source)

    agent_id, run_id, agent_url = create_agent_run(
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        repo_url=repo_url,
        starting_ref=starting_ref,
        model_id=model_id,
        name=display_name,
    )
    run_payload = wait_for_run(
        api_key=api_key,
        api_base=api_base,
        agent_id=agent_id,
        run_id=run_id,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    return {
        "status": str(run_payload.get("status") or "").upper(),
        "result": str(run_payload.get("result") or "").strip(),
        "agent_id": agent_id,
        "run_id": run_id,
        "agent_url": agent_url,
        "prompt": prompt,
    }
