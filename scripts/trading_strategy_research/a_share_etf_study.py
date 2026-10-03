#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-share ETF strategy study for the 2026-10-03 research run.

Pre-registered design (do not retune after seeing results):
- Universe: liquid onshore ETFs with Tencent qfq history.
- Execution: signal at close t, position earns the return of t+1.
- Costs: 10 bps one-way (commission ~2.5 bps + slippage ~7.5 bps). No ETF stamp tax.
- No short, no leverage. Residual weight is cash at 0% yield (conservative).
- Primary sample uses ETFs already listed by 2017-01-01 so 2018+ regimes are comparable.
- Parameters below are literature defaults, not an in-sample search.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports"
CACHE_DIR = OUT_DIR / "trading_strategy_research_20261003_cache"
START = "20160101"
END = "20261003"
COST_BPS = 10.0
STRESS_BPS = 20.0
TRADING_DAYS = 252
MIN_AMOUNT_CNY = 3.0e7  # 20-day median turnover floor at rebalance
CORE_LISTED_BY = pd.Timestamp("2017-01-01")

# code, sleeve. Display names are overwritten from the Sina ETF catalog.
CANDIDATES: List[Tuple[str, str]] = [
    ("510300", "broad"),
    ("510500", "broad"),
    ("159915", "broad"),
    ("510050", "broad"),
    ("512100", "broad"),
    ("512880", "sector"),
    ("512010", "sector"),
    ("159928", "sector"),
    ("512400", "sector"),
    ("512660", "sector"),
    ("512800", "sector"),
    ("512690", "sector"),
    ("512480", "sector"),
    ("512170", "sector"),
    ("515030", "sector"),
    ("159995", "sector"),
    ("515790", "sector"),
    ("512200", "sector"),
    ("515220", "sector"),
    ("518880", "defensive"),
    ("510880", "defensive"),
    ("512890", "defensive"),
]

INDEXES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sh000852": "中证1000",
}


def _tx_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def _retry(fn: Callable, attempts: int = 3, pause: float = 1.2):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — network sources fail transiently
            last = exc
            time.sleep(pause * (i + 1))
    raise last


def load_name_map() -> Dict[str, str]:
    frame = _retry(lambda: ak.fund_etf_category_sina(symbol="ETF基金"))
    out = {}
    for _, row in frame.iterrows():
        raw = str(row["代码"])
        code = raw[-6:]
        out[code] = str(row["名称"])
    return out


def fetch_etf(code: str) -> pd.DataFrame:
    symbol = _tx_symbol(code)

    def _pull():
        df = ak.stock_zh_a_hist_tx(
            symbol=symbol, start_date=START, end_date=END, adjust="qfq"
        )
        if df is None or df.empty:
            raise RuntimeError(f"empty history for {code}")
        return df

    df = _retry(_pull)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date")
    df = df.drop_duplicates("date").set_index("date")
    return df[["open", "high", "low", "close", "amount"]]


def fetch_index(symbol: str) -> pd.Series:
    df = _retry(lambda: ak.stock_zh_index_daily(symbol=symbol))
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").set_index("date")
    return df["close"]


def backtest(
    price: pd.DataFrame, weights: pd.DataFrame, cost_bps: float
) -> Tuple[pd.Series, pd.Series]:
    """Close-t signal earns next-session return; cost charged on the position change.

    ``weights`` are target weights known at the close of each date.
    """
    rets = price.pct_change()
    w = weights.reindex(index=price.index, columns=price.columns).fillna(0.0)
    # Renormalize if a bug pushes gross exposure above 1.
    gross = w.sum(axis=1)
    overflow = gross > 1.0 + 1e-9
    if overflow.any():
        w.loc[overflow] = w.loc[overflow].div(gross[overflow], axis=0)
    held = w.shift(1).fillna(0.0)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    asset = rets.reindex(columns=w.columns).fillna(0.0)
    # A 0 fill is only for the first return bar / leading gap. Missing sessions
    # after history starts are zeroed only when the weight on that name is 0;
    # callers must not assign weight onto an asset with a gap.
    port = (held * asset).sum(axis=1) - turnover * (cost_bps / 10000.0)
    port.name = "net"
    return port, turnover


def perf_stats(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    if len(r) < 5:
        return {}
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = len(r) / TRADING_DAYS
    cagr = float((equity.iloc[-1]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((r.mean() * TRADING_DAYS) / vol) if vol > 1e-12 else np.nan
    downside = r[r < 0]
    down_vol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 2 else np.nan
    sortino = float((r.mean() * TRADING_DAYS) / down_vol) if down_vol and down_vol > 1e-12 else np.nan
    dd = equity / equity.cummax() - 1.0
    maxdd = float(dd.min())
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    win = float((monthly > 0).mean()) if len(monthly) else np.nan
    gains = float(monthly[monthly > 0].sum())
    losses = float(-monthly[monthly < 0].sum())
    profit_factor = float(gains / losses) if losses > 1e-12 else np.nan
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else np.nan
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": maxdd,
        "win_rate_monthly": win,
        "profit_factor_monthly": profit_factor,
        "calmar": calmar,
        "n_days": int(len(r)),
        "worst_month": float(monthly.min()) if len(monthly) else np.nan,
        "best_month": float(monthly.max()) if len(monthly) else np.nan,
    }


def yearly_returns(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {}
    annual = (1.0 + r).groupby(r.index.year).prod() - 1.0
    return {str(int(k)): float(v) for k, v in annual.items()}


def slice_stats(returns: pd.Series, start: str, end: str) -> Dict[str, float]:
    part = returns.loc[start:end]
    stats = perf_stats(part)
    return stats


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    stamps = pd.Series(index, index=index)
    return pd.DatetimeIndex(stamps.groupby(index.to_period("M")).tail(1).values)


def _loc(index: pd.DatetimeIndex, dt: pd.Timestamp) -> int:
    pos = index.get_loc(dt)
    if isinstance(pos, slice):
        raise KeyError(dt)
    return int(pos)


def trailing_return(prices: pd.DataFrame, dt: pd.Timestamp, lookback: int, skip: int = 0) -> pd.Series:
    i = _loc(prices.index, dt)
    if i < lookback:
        return pd.Series(np.nan, index=prices.columns)
    end = i - skip
    start = i - lookback
    if end <= start:
        return pd.Series(np.nan, index=prices.columns)
    p0 = prices.iloc[start]
    p1 = prices.iloc[end]
    return p1 / p0 - 1.0


def rolling_vol(prices: pd.DataFrame, dt: pd.Timestamp, window: int) -> pd.Series:
    i = _loc(prices.index, dt)
    if i < window:
        return pd.Series(np.nan, index=prices.columns)
    window_px = prices.iloc[i - window + 1 : i + 1]
    rets = window_px.pct_change()
    return rets.std(ddof=1) * np.sqrt(TRADING_DAYS)


def liquid_names(amount: pd.DataFrame, dt: pd.Timestamp, names: List[str]) -> List[str]:
    i = _loc(amount.index, dt)
    if i < 20:
        return []
    med = amount.iloc[i - 19 : i + 1][names].median()
    return [n for n in names if np.isfinite(med.get(n, np.nan)) and med[n] >= MIN_AMOUNT_CNY]


def empty_weights(index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=columns)


def ffill_rebalance(weights: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep weights constant between explicit rebalance rows. Other rows are unset."""
    mask = weights.index.isin(rebalance_dates)
    held = weights.copy()
    held.loc[~mask] = np.nan
    return held.ffill().fillna(0.0)


def strategy_trend(prices: pd.DataFrame, anchor: str = "510300", window: int = 200) -> pd.DataFrame:
    w = empty_weights(prices.index, list(prices.columns))
    sma = prices[anchor].rolling(window).mean()
    w[anchor] = (prices[anchor] > sma).astype(float)
    w.loc[sma.isna(), anchor] = 0.0
    return w


def strategy_dual_momentum(
    prices: pd.DataFrame,
    amount: pd.DataFrame,
    universe: List[str],
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 3,
) -> pd.DataFrame:
    cols = list(prices.columns)
    w = empty_weights(prices.index, cols)
    dates = month_ends(prices.index)
    for dt in dates:
        i = _loc(prices.index, dt)
        if i < lookback:
            w.loc[dt] = 0.0
            continue
        names = [n for n in liquid_names(amount, dt, universe) if np.isfinite(prices.at[dt, n])]
        score = trailing_return(prices[names], dt, lookback, skip) if names else pd.Series(dtype=float)
        score = score.replace([np.inf, -np.inf], np.nan).dropna()
        score = score[score > 0]
        picks = list(score.sort_values(ascending=False).head(top_n).index)
        w.loc[dt] = 0.0
        if picks:
            for name in picks:
                w.at[dt, name] = 1.0 / len(picks)
    return ffill_rebalance(w, dates)


def strategy_reversal(
    prices: pd.DataFrame,
    amount: pd.DataFrame,
    universe: List[str],
    anchor: str = "510300",
    hold_days: int = 5,
    lookback: int = 5,
    top_n: int = 3,
    trend_window: int = 200,
) -> pd.DataFrame:
    cols = list(prices.columns)
    w = empty_weights(prices.index, cols)
    sma = prices[anchor].rolling(trend_window).mean()
    dates = list(prices.index[trend_window::hold_days])
    date_index = pd.DatetimeIndex(dates)
    for dt in date_index:
        w.loc[dt] = 0.0
        if not np.isfinite(sma.at[dt]) or prices.at[dt, anchor] <= sma.at[dt]:
            continue
        names = [n for n in liquid_names(amount, dt, universe) if np.isfinite(prices.at[dt, n])]
        score = trailing_return(prices[names], dt, lookback, 0) if names else pd.Series(dtype=float)
        score = score.replace([np.inf, -np.inf], np.nan).dropna()
        picks = list(score.sort_values(ascending=True).head(top_n).index)
        if picks:
            for name in picks:
                w.at[dt, name] = 1.0 / len(picks)
    return ffill_rebalance(w, date_index)


def strategy_low_vol(
    prices: pd.DataFrame,
    amount: pd.DataFrame,
    universe: List[str],
    window: int = 63,
    top_n: int = 3,
) -> pd.DataFrame:
    cols = list(prices.columns)
    w = empty_weights(prices.index, cols)
    dates = month_ends(prices.index)
    for dt in dates:
        names = [n for n in liquid_names(amount, dt, universe) if np.isfinite(prices.at[dt, n])]
        vol = rolling_vol(prices[names], dt, window) if names else pd.Series(dtype=float)
        vol = vol.replace([np.inf, -np.inf], np.nan).dropna()
        vol = vol[vol > 0]
        picks = list(vol.sort_values(ascending=True).head(top_n).index)
        w.loc[dt] = 0.0
        if picks:
            for name in picks:
                w.at[dt, name] = 1.0 / len(picks)
    return ffill_rebalance(w, dates)


def strategy_relative_value(
    prices: pd.DataFrame,
    left: str = "510300",
    right: str = "510500",
    window: int = 60,
    z_window: int = 252,
    z_entry: float = 1.0,
    tilt: float = 0.70,
) -> pd.DataFrame:
    cols = list(prices.columns)
    w = empty_weights(prices.index, cols)
    rel = prices[left].pct_change(window) - prices[right].pct_change(window)
    mu = rel.rolling(z_window).mean()
    sd = rel.rolling(z_window).std(ddof=1)
    z = (rel - mu) / sd
    # Weekly decision, held for 5 sessions.
    dates = list(prices.index[z_window + window :: 5])
    date_index = pd.DatetimeIndex(dates)
    for dt in date_index:
        zv = z.at[dt]
        w.loc[dt] = 0.0
        if not np.isfinite(zv):
            w.at[dt, left] = 0.5
            w.at[dt, right] = 0.5
            continue
        if zv > z_entry:
            # Left has outperformed; overweight the laggard (right).
            w.at[dt, right] = tilt
            w.at[dt, left] = 1.0 - tilt
        elif zv < -z_entry:
            w.at[dt, left] = tilt
            w.at[dt, right] = 1.0 - tilt
        else:
            w.at[dt, left] = 0.5
            w.at[dt, right] = 0.5
    return ffill_rebalance(w, date_index)


def strategy_donchian(
    prices: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    universe: List[str],
    entry: int = 20,
    exit_: int = 10,
) -> pd.DataFrame:
    cols = list(prices.columns)
    w = empty_weights(prices.index, cols)
    n = len(universe)
    slot = 1.0 / n if n else 0.0
    state = {name: 0.0 for name in universe}
    for i, dt in enumerate(prices.index):
        if i < entry:
            continue
        for name in universe:
            px = prices.at[dt, name]
            if not np.isfinite(px):
                state[name] = 0.0
                continue
            prior_high = high[name].iloc[i - entry : i].max()
            prior_low = low[name].iloc[i - exit_ : i].min()
            if state[name] == 0.0 and np.isfinite(prior_high) and px > prior_high:
                state[name] = 1.0
            elif state[name] == 1.0 and np.isfinite(prior_low) and px < prior_low:
                state[name] = 0.0
        for name in universe:
            w.at[dt, name] = state[name] * slot
    return w


def strategy_turn_of_month(prices: pd.DataFrame, anchor: str = "510300") -> pd.DataFrame:
    w = empty_weights(prices.index, list(prices.columns))
    idx = prices.index
    period = idx.to_period("M")
    grouped = pd.Series(np.arange(len(idx)), index=idx).groupby(period)
    active = pd.Series(False, index=idx)
    for _, pos in grouped:
        positions = list(pos.values)
        # First 3 and last 2 trading sessions of the month.
        keep = set(positions[:3] + positions[-2:])
        for loc in positions:
            if loc in keep:
                active.iloc[loc] = True
    w[anchor] = active.astype(float)
    return w


def strategy_vol_target(
    prices: pd.DataFrame,
    anchor: str = "510300",
    window: int = 20,
    target: float = 0.15,
) -> pd.DataFrame:
    w = empty_weights(prices.index, list(prices.columns))
    rets = prices[anchor].pct_change()
    rv = rets.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    dates = month_ends(prices.index)
    raw = pd.Series(np.nan, index=prices.index)
    for dt in dates:
        vol = rv.at[dt]
        raw.at[dt] = 0.0 if not np.isfinite(vol) or vol <= 0 else float(min(1.0, target / vol))
    w[anchor] = raw.ffill().fillna(0.0)
    return w


def strategy_barbell(
    prices: pd.DataFrame,
    anchor: str = "510300",
    gold: str = "518880",
    bank: str = "512800",
    window: int = 20,
    vol_cut: float = 0.25,
) -> pd.DataFrame:
    w = empty_weights(prices.index, list(prices.columns))
    rv = prices[anchor].pct_change().rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    dates = month_ends(prices.index)
    for dt in dates:
        vol = rv.at[dt]
        w.loc[dt] = 0.0
        if not np.isfinite(vol):
            continue
        gold_ok = gold in prices.columns and np.isfinite(prices.at[dt, gold])
        bank_ok = bank in prices.columns and np.isfinite(prices.at[dt, bank])
        if vol > vol_cut and gold_ok and bank_ok:
            w.at[dt, gold] = 0.5
            w.at[dt, bank] = 0.5
        else:
            w.at[dt, anchor] = 1.0
    return ffill_rebalance(w, dates)


def strategy_equal_weight(
    prices: pd.DataFrame, amount: pd.DataFrame, universe: List[str]
) -> pd.DataFrame:
    cols = list(prices.columns)
    w = empty_weights(prices.index, cols)
    dates = month_ends(prices.index)
    for dt in dates:
        names = [n for n in liquid_names(amount, dt, universe) if np.isfinite(prices.at[dt, n])]
        w.loc[dt] = 0.0
        if names:
            for name in names:
                w.at[dt, name] = 1.0 / len(names)
    return ffill_rebalance(w, dates)


def buy_hold(prices: pd.DataFrame, anchor: str) -> pd.DataFrame:
    w = empty_weights(prices.index, list(prices.columns))
    w[anchor] = 1.0
    return w


def exposure(weights: pd.DataFrame) -> float:
    held = weights.shift(1).fillna(0.0)
    return float(held.sum(axis=1).mean())


def block_map(returns: pd.Series) -> Dict[str, Dict[str, float]]:
    blocks = {
        "2018_2019": ("2018-01-01", "2019-12-31"),
        "2020_2021": ("2020-01-01", "2021-12-31"),
        "2022_2023": ("2022-01-01", "2023-12-31"),
        "2024_2026": ("2024-01-01", "2026-12-31"),
        "IS_2018_2022": ("2018-01-01", "2022-12-31"),
        "OOS_2023_2026": ("2023-01-01", "2026-12-31"),
    }
    return {name: slice_stats(returns, a, b) for name, (a, b) in blocks.items()}


def regime_table(close: pd.Series) -> Dict[str, float]:
    close = close.dropna().sort_index()
    last = close.iloc[-1]
    def _ret(days: int) -> float:
        if len(close) <= days:
            return np.nan
        return float(close.iloc[-1] / close.iloc[-1 - days] - 1.0)

    sma200 = close.rolling(200).mean().iloc[-1]
    sma60 = close.rolling(60).mean().iloc[-1]
    peak = close.cummax()
    dd = float(close.iloc[-1] / peak.iloc[-1] - 1.0)
    rv20 = float(close.pct_change().tail(20).std(ddof=1) * np.sqrt(TRADING_DAYS))
    rv60 = float(close.pct_change().tail(60).std(ddof=1) * np.sqrt(TRADING_DAYS))
    ytd_base = close[close.index < pd.Timestamp(f"{close.index[-1].year}-01-01")]
    ytd = float(last / ytd_base.iloc[-1] - 1.0) if len(ytd_base) else np.nan
    return {
        "last_date": str(close.index[-1].date()),
        "last": float(last),
        "ret_20d": _ret(20),
        "ret_60d": _ret(60),
        "ret_120d": _ret(120),
        "ret_252d": _ret(252),
        "ytd": ytd,
        "dd_from_high": dd,
        "dist_sma60": float(last / sma60 - 1.0) if np.isfinite(sma60) else np.nan,
        "dist_sma200": float(last / sma200 - 1.0) if np.isfinite(sma200) else np.nan,
        "rv20": rv20,
        "rv60": rv60,
    }


def _self_test() -> None:
    idx = pd.bdate_range("2020-01-01", periods=40)
    px = pd.DataFrame({"A": np.linspace(100, 139, 40)}, index=idx)
    w = pd.DataFrame(1.0, index=idx, columns=["A"])
    net, _ = backtest(px, w, cost_bps=10)
    asset = px["A"].pct_change().fillna(0.0)
    expected = asset.copy()
    expected.iloc[1] -= 0.001
    if not np.allclose(net.values, expected.values, atol=1e-12):
        raise AssertionError("backtest cost/lag self-test failed")
    # A signal that depends on the same-day close must not earn that day's return.
    spike = px.copy()
    spike.iloc[-1, 0] = spike.iloc[-2, 0] * 1.5
    w2 = pd.DataFrame(0.0, index=idx, columns=["A"])
    w2.iloc[-1, 0] = 1.0
    net2, _ = backtest(spike, w2, cost_bps=0)
    if abs(net2.iloc[-1]) > 1e-12:
        raise AssertionError("same-bar lookahead detected")


def summarize_run(
    name: str,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    bench: pd.Series,
    cost_bps: float = COST_BPS,
) -> Dict:
    net, turnover = backtest(prices, weights, cost_bps)
    # Align comparison to dates where the strategy has finished warmup:
    # first date with a non-zero lagged weight, then keep the whole path including cash.
    held = weights.shift(1).fillna(0.0).sum(axis=1)
    active = held[held > 0]
    if active.empty:
        start = net.index.min()
    else:
        start = active.index[0]
    # Common economic window starts at the later of warmup and 2018 so early listing noise drops.
    start = max(start, pd.Timestamp("2018-01-01"))
    net = net.loc[start:]
    bench_aligned = bench.reindex(net.index).fillna(0.0)
    stats = perf_stats(net)
    bstats = perf_stats(bench_aligned)
    excess = net - bench_aligned
    estats = perf_stats(excess)
    blocks = block_map(net)
    regime_blocks = ["2018_2019", "2020_2021", "2022_2023", "2024_2026"]
    positive_blocks = sum(1 for k in regime_blocks if blocks.get(k, {}).get("sharpe", -9) > 0)
    oos = blocks.get("OOS_2023_2026", {})
    stress, _ = backtest(prices, weights, STRESS_BPS)
    stress = stress.loc[net.index]
    sstats = perf_stats(stress)
    stats.update(
        {
            "name": name,
            "start": str(net.index[0].date()),
            "end": str(net.index[-1].date()),
            "avg_exposure": float(weights.shift(1).fillna(0.0).sum(axis=1).loc[net.index].mean()),
            "ann_turnover": float(turnover.loc[net.index].mean() * TRADING_DAYS),
            "benchmark_sharpe": bstats.get("sharpe"),
            "benchmark_cagr": bstats.get("cagr"),
            "benchmark_max_drawdown": bstats.get("max_drawdown"),
            "excess_cagr": estats.get("cagr"),
            "excess_sharpe": estats.get("sharpe"),
            "positive_regime_blocks": int(positive_blocks),
            "oos_sharpe": oos.get("sharpe"),
            "oos_cagr": oos.get("cagr"),
            "oos_max_drawdown": oos.get("max_drawdown"),
            "stress20_sharpe": sstats.get("sharpe"),
            "stress20_cagr": sstats.get("cagr"),
            "yearly": yearly_returns(net),
            "blocks": blocks,
        }
    )
    return stats


def quality_flag(row: Dict) -> Tuple[bool, List[str]]:
    reasons = []
    if not row:
        return False, ["no stats"]
    if row.get("sharpe", -9) <= 1.0:
        reasons.append("sharpe<=1")
    if row.get("max_drawdown", -1) <= -0.30:
        reasons.append("maxdd<=-30%")
    if row.get("sharpe", -9) <= (row.get("benchmark_sharpe") or 99):
        reasons.append("sharpe<=benchmark")
    if (row.get("cagr") or -1) <= 0:
        reasons.append("cagr<=0")
    if (row.get("oos_sharpe") or -9) <= 0.5:
        reasons.append("oos_sharpe<=0.5")
    if row.get("positive_regime_blocks", 0) < 3:
        reasons.append("regime_blocks<3")
    if (row.get("stress20_sharpe") or -9) <= 0.7:
        reasons.append("cost20_sharpe<=0.7")
    return (len(reasons) == 0), reasons


def main() -> None:
    _self_test()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    names = load_name_map()
    frames = {}
    errors = {}
    meta = []
    for code, sleeve in CANDIDATES:
        try:
            frames[code] = fetch_etf(code)
            meta.append(
                {
                    "code": code,
                    "name": names.get(code, code),
                    "sleeve": sleeve,
                    "start": str(frames[code].index[0].date()),
                    "end": str(frames[code].index[-1].date()),
                    "rows": int(len(frames[code])),
                    "last_close": float(frames[code]["close"].iloc[-1]),
                    "last_amount": float(frames[code]["amount"].iloc[-1]),
                }
            )
            print(f"OK {code} {names.get(code, '')} {meta[-1]['start']}->{meta[-1]['end']} n={meta[-1]['rows']}")
        except Exception as exc:  # noqa: BLE001
            errors[code] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {code} {errors[code]}")

    if "510300" not in frames:
        raise SystemExit("anchor 510300 unavailable; aborting without invented prices")

    calendar = frames["510300"].index
    close = pd.DataFrame({c: frames[c]["close"].reindex(calendar) for c in frames})
    high = pd.DataFrame({c: frames[c]["high"].reindex(calendar) for c in frames})
    low = pd.DataFrame({c: frames[c]["low"].reindex(calendar) for c in frames})
    amount = pd.DataFrame({c: frames[c]["amount"].reindex(calendar) for c in frames})
    # Bridge only isolated missing prints. Longer gaps stay missing and are not traded.
    gap_filled = close.isna() & close.ffill(limit=3).notna()
    close = close.ffill(limit=3)
    high = high.ffill(limit=3)
    low = low.ffill(limit=3)

    core = []
    for code, sleeve in CANDIDATES:
        if code not in close.columns or sleeve == "broad":
            continue
        first = close[code].first_valid_index()
        if first is not None and first <= CORE_LISTED_BY:
            core.append(code)

    index_stats = {}
    for symbol, label in INDEXES.items():
        try:
            series = fetch_index(symbol)
            series = series[series.index <= calendar.max()]
            index_stats[symbol] = {"label": label, **regime_table(series)}
            print(f"IDX {label} last={index_stats[symbol]['last_date']} {index_stats[symbol]['last']:.2f}")
        except Exception as exc:  # noqa: BLE001
            index_stats[symbol] = {"label": label, "error": f"{type(exc).__name__}: {exc}"}
            print(f"IDX FAIL {symbol} {exc}")

    # Sector regime snapshot on the last session, core + later listings that are liquid.
    snap_names = [c for c, s in CANDIDATES if c in close.columns and s != "broad"]
    last = close.index[-1]
    snapshot = []
    for code in snap_names:
        series = close[code].dropna()
        if series.empty:
            continue
        info = regime_table(series)
        info["code"] = code
        info["name"] = names.get(code, code)
        info["sleeve"] = dict(CANDIDATES)[code]
        med_amt = float(amount[code].tail(20).median())
        info["median_amount_20d"] = med_amt
        info["above_sma200"] = bool(info.get("dist_sma200", -1) > 0)
        snapshot.append(info)

    above = [r for r in snapshot if r.get("above_sma200")]
    breadth = float(len(above) / len(snapshot)) if snapshot else np.nan

    anchor_w = buy_hold(close, "510300")
    bench_net, _ = backtest(close, anchor_w, COST_BPS)
    # Strategies
    builders = {
        "S1_trend_ma200": lambda: strategy_trend(close),
        "S2_dual_momentum": lambda: strategy_dual_momentum(close, amount, core),
        "S3_short_reversal": lambda: strategy_reversal(close, amount, core),
        "S4_low_vol": lambda: strategy_low_vol(close, amount, core),
        "S5_relative_300_500": lambda: strategy_relative_value(close),
        "S6_donchian": lambda: strategy_donchian(close, high, low, core),
        "S7_turn_of_month": lambda: strategy_turn_of_month(close),
        "S8_vol_target": lambda: strategy_vol_target(close),
        "S9_defensive_barbell": lambda: strategy_barbell(close),
        "B_equal_weight_sectors": lambda: strategy_equal_weight(close, amount, core),
    }
    results = {}
    weight_store = {}
    for name, builder in builders.items():
        w = builder()
        weight_store[name] = w
        results[name] = summarize_run(name, close, w, bench_net)
        row = results[name]
        print(
            f"{name} sharpe={row.get('sharpe'):.3f} cagr={row.get('cagr'):.3f} "
            f"maxdd={row.get('max_drawdown'):.3f} oos={row.get('oos_sharpe'):.3f}"
        )

    # Benchmark itself on the 2018+ window.
    results["B_csi300"] = summarize_run("B_csi300", close, anchor_w, bench_net)

    # Sensitivity: pre-specified neighbors. Not used to pick a new champion.
    sensitivity = {
        "S1_ma100": summarize_run("S1_ma100", close, strategy_trend(close, window=100), bench_net),
        "S1_ma250": summarize_run("S1_ma250", close, strategy_trend(close, window=250), bench_net),
        "S2_top2_lb126": summarize_run(
            "S2_top2_lb126",
            close,
            strategy_dual_momentum(close, amount, core, lookback=126, skip=21, top_n=2),
            bench_net,
        ),
        "S2_top4_lb252": summarize_run(
            "S2_top4_lb252",
            close,
            strategy_dual_momentum(close, amount, core, lookback=252, skip=21, top_n=4),
            bench_net,
        ),
        "S4_vol126_top3": summarize_run(
            "S4_vol126_top3", close, strategy_low_vol(close, amount, core, window=126, top_n=3), bench_net
        ),
        "S8_target12": summarize_run(
            "S8_target12", close, strategy_vol_target(close, target=0.12), bench_net
        ),
    }

    # Expanded universe is a robustness check, reported separately from the primary rank.
    expanded = []
    for code, sleeve in CANDIDATES:
        if sleeve == "broad" or code not in close.columns:
            continue
        first = close[code].first_valid_index()
        if first is not None and first <= pd.Timestamp("2020-01-01"):
            expanded.append(code)
    expanded_w = strategy_dual_momentum(close, amount, expanded)
    # Evaluate only once every name in the expanded set is eligible.
    sensitivity["S2_expanded_universe"] = summarize_run(
        "S2_expanded_universe", close, expanded_w, bench_net
    )

    ranked = []
    for name, row in results.items():
        if name.startswith("B_"):
            passed, reasons = False, ["benchmark"]
        else:
            passed, reasons = quality_flag(row)
        ranked.append({"name": name, "pass": passed, "fail_reasons": reasons, **{
            k: row.get(k) for k in (
                "start", "end", "total_return", "cagr", "vol", "sharpe", "sortino",
                "max_drawdown", "win_rate_monthly", "profit_factor_monthly", "calmar",
                "avg_exposure", "ann_turnover", "benchmark_sharpe", "benchmark_cagr",
                "benchmark_max_drawdown", "excess_sharpe", "oos_sharpe", "oos_cagr",
                "oos_max_drawdown", "stress20_sharpe", "positive_regime_blocks",
            )
        }})

    payload = {
        "as_of_rule": "latest session in fetched bars; study date is 2026-10-03 (holiday/weekend)",
        "cost_bps_one_way": COST_BPS,
        "stress_bps_one_way": STRESS_BPS,
        "cash_yield": 0.0,
        "execution": "signal close t earns return t+1",
        "min_amount_cny": MIN_AMOUNT_CNY,
        "core_universe": [{"code": c, "name": names.get(c, c)} for c in core],
        "expanded_universe": [{"code": c, "name": names.get(c, c)} for c in expanded],
        "catalog": meta,
        "fetch_errors": errors,
        "ffill_cells": int(gap_filled.sum().sum()),
        "index_regime": index_stats,
        "sector_snapshot": snapshot,
        "sector_breadth_above_sma200": breadth,
        "results": results,
        "sensitivity": {
            k: {kk: vv for kk, vv in row.items() if kk not in {"yearly", "blocks"}}
            for k, row in sensitivity.items()
        },
        "ranking_input": ranked,
        "last_session": str(last.date()),
        "anchor_last_close": float(close["510300"].iloc[-1]),
    }
    # yearly and blocks stay inside results
    out = OUT_DIR / "trading_strategy_research_20261003_metrics.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"WROTE {out}")
    print("CORE", core)
    print("BREADTH", round(breadth, 3) if np.isfinite(breadth) else None)


if __name__ == "__main__":
    main()
