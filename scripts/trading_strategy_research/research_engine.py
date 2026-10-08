# -*- coding: utf-8 -*-
"""Long-only open-to-open backtest engine for strategy research.

Execution contract (no same-bar fill):
- A signal may use prices through the close of dates[k - 1].
- The resulting weights are traded at the open of dates[k].
- Those weights earn the move from dates[k] open to dates[k + 1] open.
- Transaction cost is charged on traded notional at the rebalance open.

Costs are one-way: a round trip that sells 100% of NAV and buys 100% of NAV
has turnover = 2 and pays 2 * one_way_cost. Long-only, no leverage: weights
are non-negative and sum to at most 1. Residual weight is cash.

Cash yield defaults to 0. That understates a money-market residual and keeps
results conservative relative to crediting an assumed deposit rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


WeightFn = Callable[[int, dict], Optional[np.ndarray]]
RebalanceFn = Callable[[int], bool]


@dataclass
class SimResult:
    """Daily outcomes indexed by the open that ENDS each holding interval."""

    returns: pd.Series
    turnover: pd.Series
    exposure: pd.Series
    weights: pd.DataFrame


def simulate_open_to_open(
    opens: np.ndarray,
    dates: pd.DatetimeIndex,
    weight_fn: WeightFn,
    rebalance_fn: RebalanceFn,
    one_way_cost: float,
    asset_names: list,
    cash_daily_yield: float = 0.0,
) -> SimResult:
    """Simulate a long-only strategy on an open-price panel.

    Parameters
    ----------
    opens:
        Shape (n_dates, n_assets). Prices must be split/dividend adjusted
        consistently with the signal prices. Non-positive or non-finite
        prints are treated as untradable for that interval (position value
        is frozen for that asset, which is conservative only if gaps are rare).
    dates:
        Length n_dates, aligned to ``opens`` rows.
    weight_fn:
        ``(decision_idx, state) -> weights`` using information through
        ``dates[decision_idx]`` close. Return None when the signal is not
        ready; the engine then keeps the existing book.
    rebalance_fn:
        ``(k) -> bool`` where k is the open index at which a trade may occur.
    one_way_cost:
        Fraction of NAV charged per unit of one-way turnover. 0.0008 = 8 bps.
    """

    if opens.ndim != 2:
        raise ValueError("opens must be a 2-d array")
    n_dates, n_assets = opens.shape
    if len(dates) != n_dates:
        raise ValueError("dates and opens length mismatch")
    if len(asset_names) != n_assets:
        raise ValueError("asset_names length mismatch")
    if one_way_cost < 0:
        raise ValueError("one_way_cost must be non-negative")

    values = np.zeros(n_assets, dtype=float)
    cash = 1.0
    state: Dict[str, object] = {}
    rets = []
    turns = []
    exposures = []
    end_dates = []
    weight_rows = []

    for k in range(1, n_dates - 1):
        equity_before = float(cash + values.sum())
        if equity_before <= 0:
            raise RuntimeError("equity destroyed; check prices and costs")

        turn = 0.0
        if rebalance_fn(k):
            raw = weight_fn(k - 1, state)
            if raw is not None:
                weights = np.asarray(raw, dtype=float)
                if weights.shape != (n_assets,):
                    raise ValueError("weight vector has the wrong shape")
                if not np.isfinite(weights).all():
                    raise ValueError("non-finite weight")
                if np.any(weights < -1e-8):
                    raise ValueError("short sales are outside the research contract")
                weights = np.clip(weights, 0.0, None)
                if float(weights.sum()) > 1.0 + 1e-6:
                    raise ValueError("leverage is outside the research contract")
                target_values = weights * equity_before
                turn = float(np.abs(target_values - values).sum() / equity_before)
                equity_after = equity_before * (1.0 - turn * one_way_cost)
                values = weights * equity_after
                cash = equity_after - float(values.sum())

        o0 = opens[k]
        o1 = opens[k + 1]
        good = np.isfinite(o0) & np.isfinite(o1) & (o0 > 0) & (o1 > 0)
        growth = np.ones(n_assets, dtype=float)
        growth[good] = o1[good] / o0[good]
        values = values * growth
        if cash_daily_yield:
            cash *= 1.0 + cash_daily_yield
        equity_end = float(cash + values.sum())
        rets.append(equity_end / equity_before - 1.0)
        turns.append(turn)
        exposures.append(float(values.sum()) / equity_end if equity_end else 0.0)
        end_dates.append(dates[k + 1])
        weight_rows.append(values / equity_end if equity_end else values)

    index = pd.DatetimeIndex(end_dates)
    return SimResult(
        returns=pd.Series(rets, index=index, name="return"),
        turnover=pd.Series(turns, index=index, name="turnover"),
        exposure=pd.Series(exposures, index=index, name="exposure"),
        weights=pd.DataFrame(weight_rows, index=index, columns=list(asset_names)),
    )


def performance_stats(returns: pd.Series, rf_annual: float = 0.015) -> Optional[dict]:
    """Annualized stats. ``rf_annual`` is only used for Sharpe, not for PnL."""

    rets = returns.dropna()
    if len(rets) < 40:
        return None
    equity = (1.0 + rets).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    n = len(rets)
    years = n / 252.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(rets.std(ddof=1) * np.sqrt(252.0))
    rf_daily = rf_annual / 252.0
    excess = rets - rf_daily
    excess_std = float(excess.std(ddof=1))
    sharpe = float(excess.mean() / excess_std * np.sqrt(252.0)) if excess_std > 0 else np.nan
    sharpe_rf0 = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252.0)) if rets.std(ddof=1) > 0 else np.nan
    downside = rets.clip(upper=0.0)
    down_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252.0))
    sortino = float((rets.mean() * 252.0) / down_dev) if down_dev > 0 else np.nan
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    monthly = (1.0 + rets).resample("ME").prod() - 1.0
    monthly = monthly.dropna()
    if len(monthly):
        win_rate = float((monthly > 0).mean())
        gains = float(monthly[monthly > 0].sum())
        losses = float(monthly[monthly < 0].sum())
        profit_factor = float(gains / abs(losses)) if losses < 0 else np.nan
        worst_month = float(monthly.min())
        best_month = float(monthly.max())
    else:
        win_rate = profit_factor = worst_month = best_month = np.nan
    return {
        "observations": int(n),
        "years": years,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sharpe_rf0": sharpe_rf0,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "monthly_win_rate": win_rate,
        "profit_factor": profit_factor,
        "worst_month": worst_month,
        "best_month": best_month,
        "skew": float(rets.skew()),
    }


def yearly_returns(returns: pd.Series) -> pd.Series:
    rets = returns.dropna()
    if rets.empty:
        return pd.Series(dtype=float)
    return (1.0 + rets).groupby(rets.index.year).prod() - 1.0


def monthly_excess_ttest(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """t-stat of monthly return difference. Normal approximation, not Newey-West."""

    both = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if both.empty:
        return {"n_months": 0, "mean_monthly_excess": np.nan, "t_stat": np.nan, "p_value_two_sided": np.nan}
    monthly = (1.0 + both).resample("ME").prod() - 1.0
    diff = (monthly["s"] - monthly["b"]).dropna()
    n = len(diff)
    if n < 8 or float(diff.std(ddof=1)) == 0:
        return {
            "n_months": int(n),
            "mean_monthly_excess": float(diff.mean()) if n else np.nan,
            "t_stat": np.nan,
            "p_value_two_sided": np.nan,
        }
    t_stat = float(diff.mean() / diff.std(ddof=1) * np.sqrt(n))
    # Abramowitz-Stegun style erf approximation via math.erf
    from math import erf, sqrt

    p = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0)))))
    return {
        "n_months": int(n),
        "mean_monthly_excess": float(diff.mean()),
        "t_stat": t_stat,
        "p_value_two_sided": p,
    }


def repair_price_artifacts(
    panels: Dict[str, pd.DataFrame],
    market_col: str = "510300.SS",
) -> list:
    """Correct Yahoo split glitches on A-share ETFs.

    Two observed failure modes, both idiosyncratic versus the CSI300 ETF on the
    same day (so they are not market crashes):

    1. A permanent ~50% level shift (2-for-1 share split) with no Yahoo split
       factor. History strictly before the ex-date is divided by 2. The ex-date
       itself keeps its post-split print, so any true move that day remains.
    2. A single session printed on the pre-split scale (about +100% to +300%)
       that fully reverses the next session. That session's OHLC is rescaled
       onto the surrounding level. Two-day total return is unchanged.

    Prices are not forward-filled across missing sessions.
    """

    close = panels["Close"]
    log = []
    symbols = []
    for col in close.columns:
        code = str(col).split(".")[0]
        if len(code) == 6 and code.startswith(("15", "51", "56", "58")):
            symbols.append(col)
    market = close[market_col].dropna() if market_col in close.columns else None

    for symbol in symbols:
        values = close[symbol].dropna()
        if len(values) < 10:
            continue
        split_dates = []
        for i in range(1, len(values) - 5):
            prev = float(values.iloc[i - 1])
            cur = float(values.iloc[i])
            if prev <= 0 or cur <= 0:
                continue
            ratio = cur / prev
            prev_jump = float(values.iloc[i - 1] / values.iloc[i - 2]) if i >= 2 else 1.0
            if not (0.45 <= ratio <= 0.55) or prev_jump > 1.4:
                continue
            if float(values.iloc[i:i + 5].median()) / cur < 0.85:
                continue
            if float(values.iloc[i:i + 5].median()) / cur > 1.15:
                continue
            dt = values.index[i]
            if market is not None and symbol != market_col and dt in market.index:
                m_i = market.index.get_loc(dt)
                if isinstance(m_i, slice) or m_i == 0:
                    continue
                mkt_ret = float(market.iloc[m_i] / market.iloc[m_i - 1] - 1.0)
                if mkt_ret <= -0.15:
                    continue
            split_dates.append(dt)
        for dt in split_dates:
            for panel in panels.values():
                if symbol not in panel.columns:
                    continue
                panel.loc[panel.index < dt, symbol] = panel.loc[panel.index < dt, symbol] / 2.0
            log.append({
                "symbol": symbol,
                "date": str(pd.Timestamp(dt).date()),
                "type": "split_2for1_history_halved",
            })

        values = close[symbol].dropna()
        spikes = []
        for i in range(1, len(values) - 1):
            prev = float(values.iloc[i - 1])
            cur = float(values.iloc[i])
            nxt = float(values.iloc[i + 1])
            if prev <= 0 or cur <= 0 or nxt <= 0:
                continue
            r1 = cur / prev - 1.0
            r2 = nxt / cur - 1.0
            two = nxt / prev - 1.0
            if r1 > 0.70 and r2 < -0.40 and abs(two) < 0.20:
                spikes.append((values.index[i], prev / cur))
        for dt, scale in spikes:
            for panel in panels.values():
                if symbol not in panel.columns or dt not in panel.index:
                    continue
                panel.at[dt, symbol] = panel.at[dt, symbol] * scale
            log.append({
                "symbol": symbol,
                "date": str(pd.Timestamp(dt).date()),
                "type": "one_day_scale_error",
                "scale": float(scale),
            })
    return log


def is_new_month(dates: pd.DatetimeIndex, k: int) -> bool:
    prev, cur = dates[k - 1], dates[k]
    return (prev.year, prev.month) != (cur.year, cur.month)


def is_new_week(dates: pd.DatetimeIndex, k: int) -> bool:
    prev, cur = dates[k - 1].isocalendar(), dates[k].isocalendar()
    return (int(prev[0]), int(prev[1])) != (int(cur[0]), int(cur[1]))
