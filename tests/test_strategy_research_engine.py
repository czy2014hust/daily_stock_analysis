# -*- coding: utf-8 -*-
"""Synthetic tests for the trading-strategy research engine. No network."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "trading_strategy_research"
sys.path.insert(0, str(ENGINE_DIR))

from research_engine import (  # noqa: E402
    is_new_month,
    performance_stats,
    simulate_open_to_open,
)


def _run(opens, weight_fn, rebalance_fn, one_way_cost):
    dates = pd.bdate_range("2020-01-01", periods=len(opens))
    return simulate_open_to_open(
        opens=np.asarray(opens, dtype=float),
        dates=dates,
        weight_fn=weight_fn,
        rebalance_fn=rebalance_fn,
        one_way_cost=one_way_cost,
        asset_names=["A"],
    )


def test_signal_uses_prior_close_and_earns_next_open_interval():
    opens = [[100], [100], [110], [110], [110], [110]]

    def always_long(idx, state):
        del idx, state
        return np.array([1.0])

    result = _run(opens, always_long, lambda k: True, 0.0)
    # Buy at the second open (100) and earn the move to 110. Later intervals are flat.
    assert result.returns.iloc[0] == pytest.approx(0.10)
    assert result.returns.iloc[1] == pytest.approx(0.0)
    assert result.exposure.iloc[0] == pytest.approx(1.0)


def test_one_way_cost_reduces_entry():
    opens = [[100], [100], [110], [110], [110]]

    def always_long(idx, state):
        del idx, state
        return np.array([1.0])

    result = _run(opens, always_long, lambda k: True, 0.01)
    # 1% of NAV on a 100% buy, then +10% on the remaining 0.99.
    assert result.returns.iloc[0] == pytest.approx(0.99 * 1.10 - 1.0)
    assert result.turnover.iloc[0] == pytest.approx(1.0)


def test_future_decision_cannot_capture_the_jump_it_has_not_seen():
    opens = [[100], [100], [100], [100], [150]]

    def late(idx, state):
        del state
        # idx >= 3 is the close of the bar whose open already started the jump interval.
        return np.array([1.0 if idx >= 3 else 0.0])

    result = _run(opens, late, lambda k: True, 0.0)
    # The only up move is the last interval, decided before idx 3, so it stays in cash.
    assert result.returns.iloc[-1] == pytest.approx(0.0)
    assert float((1.0 + result.returns).prod() - 1.0) == pytest.approx(0.0)


def test_rejects_shorts_and_leverage():
    opens = [[100], [100], [100], [100]]

    def short(idx, state):
        del idx, state
        return np.array([-0.2])

    with pytest.raises(ValueError, match="short"):
        _run(opens, short, lambda k: True, 0.0)

    def levered(idx, state):
        del idx, state
        return np.array([1.2])

    with pytest.raises(ValueError, match="leverage"):
        _run(opens, levered, lambda k: True, 0.0)


def test_month_boundary_helper():
    dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"])
    assert is_new_month(dates, 1) is False
    assert is_new_month(dates, 2) is True


def test_split_and_spike_repairs_do_not_invent_a_crash_or_a_rally():
    from research_engine import repair_price_artifacts

    idx = pd.bdate_range("2024-01-01", periods=12)
    # Permanent 2-for-1 on day 6, plus a one-day pre-split print on day 9.
    close = pd.Series(
        [20, 20, 20, 20, 20, 10, 10, 10, 20, 10, 10, 10],
        index=idx,
        dtype=float,
    )
    market = pd.Series(100.0, index=idx)
    panels = {
        "Close": pd.DataFrame({"510010.SS": close, "510300.SS": market}),
        "Open": pd.DataFrame({"510010.SS": close.copy(), "510300.SS": market.copy()}),
        "High": pd.DataFrame({"510010.SS": close.copy(), "510300.SS": market.copy()}),
        "Low": pd.DataFrame({"510010.SS": close.copy(), "510300.SS": market.copy()}),
    }
    log = repair_price_artifacts(panels)
    fixed = panels["Close"]["510010.SS"]
    # History before the ex-date is halved; the spike session is put back on the new scale.
    assert fixed.iloc[0] == pytest.approx(10.0)
    assert fixed.iloc[5] == pytest.approx(10.0)
    assert fixed.iloc[8] == pytest.approx(10.0)
    assert fixed.pct_change().abs().max() < 0.05
    kinds = {row["type"] for row in log}
    assert "split_2for1_history_halved" in kinds
    assert "one_day_scale_error" in kinds


def test_performance_stats_known_series():
    # 252 flat +1% daily is unrealistic but the math is closed-form enough to check signs.
    rets = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    stats = performance_stats(rets, rf_annual=0.0)
    assert stats["total_return"] > 0
    assert stats["max_drawdown"] == pytest.approx(0.0)
    assert stats["sharpe_rf0"] > 10
