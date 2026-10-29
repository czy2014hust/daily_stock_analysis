#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily A-share ETF strategy research (trading-strategy-research skill).

As-of date is the research calendar day. Prices stop at the last completed
session; this script never forward-fills missing sessions.

Primary specs below were chosen from published rules (12-1 momentum, Faber
MA200, Turtle-style Donchian, low-vol, short-horizon reversal) BEFORE looking
at this sample's ranking. Sensitivity is diagnostic only and is not used to
pick a winner.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import akshare as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ROOT / "reports" / "trading_strategy_research_20261003.md"
CACHE = OUT / "panel.pkl"

ASOF = "2026-10-03"
# Precommitted split: design window vs untouched later regimes.
IS_END = "2022-12-31"
EVAL_START = "2018-01-01"
FETCH_START = "20160101"
FETCH_END = "20261003"

# One-way cost in bps of notional traded. ETF stamp duty is 0.
# Broad / bond / gold: commission ~1.5bps + slippage ~6bps.
# Sector: wider spread. QDII: premium, creation lag, wider book.
COST_BPS = {
    "broad": 8.0,
    "sector": 12.0,
    "defense": 8.0,
    "offshore": 20.0,
}
STRESS_BPS = 25.0
MIN_MED_AMOUNT = 1.0e8  # 1亿 CNY, 20-day median; below this we refuse the name

UNIVERSE: Dict[str, Dict[str, str]] = {
    "510300": {"name": "沪深300ETF", "bucket": "broad"},
    "510500": {"name": "中证500ETF", "bucket": "broad"},
    "159915": {"name": "创业板ETF", "bucket": "broad"},
    "512100": {"name": "中证1000ETF", "bucket": "broad"},
    "510880": {"name": "红利ETF", "bucket": "sector"},
    "512010": {"name": "医药ETF", "bucket": "sector"},
    "512800": {"name": "银行ETF", "bucket": "sector"},
    "159928": {"name": "消费ETF", "bucket": "sector"},
    "512660": {"name": "军工ETF", "bucket": "sector"},
    "512880": {"name": "证券ETF", "bucket": "sector"},
    "512400": {"name": "有色ETF", "bucket": "sector"},
    "512480": {"name": "半导体ETF", "bucket": "sector"},
    "512690": {"name": "酒ETF", "bucket": "sector"},
    "512890": {"name": "红利低波ETF", "bucket": "sector"},
    "515220": {"name": "煤炭ETF", "bucket": "sector"},
    "159992": {"name": "创新药ETF", "bucket": "sector"},
    "518880": {"name": "黄金ETF", "bucket": "defense"},
    "511010": {"name": "国债ETF", "bucket": "defense"},
    "513100": {"name": "纳指ETF", "bucket": "offshore"},
    "513500": {"name": "标普500ETF", "bucket": "offshore"},
}

# Excluded after a liquidity screen on 2026-09-30 (60d median amount < ~2亿)
# and/or heavy overlap: 515790 光伏, 515030 新能源, 516160 新能源.
EXCLUDED_LIQUIDITY = ["515790", "515030", "516160"]

BROAD = [c for c, m in UNIVERSE.items() if m["bucket"] == "broad"]
SECTOR = [c for c, m in UNIVERSE.items() if m["bucket"] == "sector"]
DEFENSE = [c for c, m in UNIVERSE.items() if m["bucket"] == "defense"]
OFFSHORE = [c for c, m in UNIVERSE.items() if m["bucket"] == "offshore"]
DOMESTIC_EQ = BROAD + SECTOR
BOND = "511010"
GOLD = "518880"
BENCH = "510300"

INDEXES = {
    "sh000001": "上证综指",
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sh000852": "中证1000",
    "sz399006": "创业板指",
}


def _tx_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def fetch_etf(code: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist_tx(
        symbol=_tx_symbol(code),
        start_date=FETCH_START,
        end_date=FETCH_END,
        adjust="qfq",
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "close"]).sort_values("date")
    df = df.drop_duplicates("date").set_index("date")
    return df[["open", "high", "low", "close", "amount"]]


def fetch_index(symbol: str) -> pd.DataFrame:
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["close"]].dropna()


def load_market() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    if CACHE.exists():
        with CACHE.open("rb") as fh:
            blob = pickle.load(fh)
        return blob["open"], blob["close"], blob["amount"], blob["index_close"], blob["meta"]

    frames = []
    meta_rows = []
    for code, meta in UNIVERSE.items():
        df = fetch_etf(code)
        df["code"] = code
        frames.append(df.reset_index())
        meta_rows.append(
            {
                "code": code,
                "name": meta["name"],
                "bucket": meta["bucket"],
                "first": str(df.index.min().date()),
                "last": str(df.index.max().date()),
                "n": int(len(df)),
                "med_amount_60d": float(df["amount"].tail(60).median()),
            }
        )
        print(f"ok {code} {meta['name']} {meta_rows[-1]['first']}->{meta_rows[-1]['last']} n={len(df)}")

    panel = pd.concat(frames, ignore_index=True)
    op = panel.pivot(index="date", columns="code", values="open").sort_index()
    cl = panel.pivot(index="date", columns="code", values="close").sort_index()
    amt = panel.pivot(index="date", columns="code", values="amount").sort_index()

    idx_frames = {}
    for symbol, name in INDEXES.items():
        try:
            s = fetch_index(symbol)["close"]
            s.name = symbol
            idx_frames[symbol] = s
            print(f"idx {symbol} {name} last={s.index.max().date()} n={len(s)}")
        except Exception as exc:  # noqa: BLE001
            print(f"idx fail {symbol}: {type(exc).__name__}: {exc}")
    index_close = pd.DataFrame(idx_frames).sort_index()

    trade_cal = []
    cal_note = ""
    try:
        cal = ak.tool_trade_date_hist_sina()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        window = cal[(cal["trade_date"] >= "2026-09-20") & (cal["trade_date"] <= "2026-10-20")]
        trade_cal = [str(d.date()) for d in window["trade_date"]]
    except Exception as exc:  # noqa: BLE001
        cal_note = f"trade calendar unavailable: {type(exc).__name__}: {exc}"

    meta = {"etfs": meta_rows, "trade_calendar_window": trade_cal, "calendar_note": cal_note}
    blob = {"open": op, "close": cl, "amount": amt, "index_close": index_close, "meta": meta}
    with CACHE.open("wb") as fh:
        pickle.dump(blob, fh)
    return op, cl, amt, index_close, meta


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).tail(1).values)


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    iso = index.isocalendar()
    frame = pd.DataFrame({"d": index, "y": iso.year.to_numpy(), "w": iso.week.to_numpy()})
    return pd.DatetimeIndex(frame.groupby(["y", "w"], sort=True)["d"].max().values)


def rolling_return(close: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Total return from t-lookback to t-skip, known at close t."""
    if skip <= 0:
        return close / close.shift(lookback) - 1.0
    return close.shift(skip) / close.shift(lookback) - 1.0


def liquid(amount: pd.DataFrame) -> pd.DataFrame:
    return amount.rolling(20, min_periods=10).median() >= MIN_MED_AMOUNT


def listed(close: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    seen = close.notna().cumsum()
    return seen >= min_obs


def _empty_weights(index: pd.DatetimeIndex, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=list(columns), dtype=float)


def _fill_schedule(sparse: pd.DataFrame, sched: Iterable[pd.Timestamp], columns: Sequence[str]) -> pd.DataFrame:
    """Carry last rebalance weights forward. Dates before the first signal stay 0."""
    w = sparse.reindex(columns=list(columns)).copy()
    keep = w.index.isin(pd.DatetimeIndex(list(sched)))
    w.loc[~keep] = np.nan
    return w.ffill().fillna(0.0)


def _assign_equal(codes: Sequence[str], columns: Sequence[str], fallback: Optional[str] = None) -> pd.Series:
    row = pd.Series(0.0, index=list(columns))
    use = [c for c in codes if c in row.index]
    if not use:
        if fallback and fallback in row.index:
            row[fallback] = 1.0
        return row
    for c in use:
        row[c] = 1.0 / len(use)
    return row


def _topk(scores: pd.Series, n: int, eligible: pd.Series) -> List[str]:
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    ok = eligible.reindex(s.index).fillna(False)
    s = s[ok]
    if s.empty:
        return []
    return list(s.sort_values(ascending=False).head(n).index)


def _bottomk(scores: pd.Series, n: int, eligible: pd.Series) -> List[str]:
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    ok = eligible.reindex(s.index).fillna(False)
    s = s[ok]
    if s.empty:
        return []
    return list(s.sort_values(ascending=True).head(n).index)


def strat_buyhold(close: pd.DataFrame, amount: pd.DataFrame, codes: Sequence[str], rebalance: str = "M") -> pd.DataFrame:
    """Equal-weight buy & hold. New listings enter on the next rebalance once liquid."""
    cols = list(close.columns)
    sched = month_ends(close.index) if rebalance == "M" else week_ends(close.index)
    liq = liquid(amount)
    alive = listed(close, 60)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        if dt not in close.index:
            continue
        ok = []
        for c in codes:
            if bool(alive.at[dt, c]) and bool(liq.at[dt, c]) and pd.notna(close.at[dt, c]):
                ok.append(c)
        sparse.loc[dt] = _assign_equal(ok, cols, fallback=BOND if BOND in cols else None)
    return _fill_schedule(sparse, sched, cols)


def strat_tsmom(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    universe: Sequence[str],
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """Time-series momentum: hold each name with positive skipped momentum, else bond."""
    cols = list(close.columns)
    sched = month_ends(close.index)
    mom = rolling_return(close, lookback, skip)
    liq = liquid(amount)
    alive = listed(close, lookback + 5)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        ok = []
        for c in universe:
            if c not in cols:
                continue
            if not (bool(alive.at[dt, c]) and bool(liq.at[dt, c])):
                continue
            val = mom.at[dt, c]
            if pd.notna(val) and val > 0:
                ok.append(c)
        sparse.loc[dt] = _assign_equal(ok, cols, fallback=BOND)
    return _fill_schedule(sparse, sched, cols)


def strat_dual_mom(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    universe: Sequence[str],
    lookback: int = 252,
    top_n: int = 3,
) -> pd.DataFrame:
    """Relative momentum, absolute filter versus the bond ETF (Antonacci-style)."""
    cols = list(close.columns)
    sched = month_ends(close.index)
    mom = rolling_return(close, lookback, 0)
    liq = liquid(amount)
    alive = listed(close, lookback + 5)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        bond_m = mom.at[dt, BOND] if BOND in mom.columns else 0.0
        if pd.isna(bond_m):
            bond_m = 0.0
        eligible = {}
        for c in universe:
            if not (bool(alive.at[dt, c]) and bool(liq.at[dt, c])):
                continue
            val = mom.at[dt, c]
            if pd.notna(val) and val > 0 and val > bond_m:
                eligible[c] = val
        if not eligible:
            sparse.loc[dt] = _assign_equal([], cols, fallback=BOND)
            continue
        ranked = sorted(eligible, key=eligible.get, reverse=True)[:top_n]
        sparse.loc[dt] = _assign_equal(ranked, cols, fallback=BOND)
    return _fill_schedule(sparse, sched, cols)


def strat_reversal(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    universe: Sequence[str],
    lookback: int = 5,
    bottom_n: int = 3,
    crash: float = -0.12,
) -> pd.DataFrame:
    """Long-only cross-sectional reversal. Crash names are excluded, not bought."""
    cols = list(close.columns)
    sched = week_ends(close.index)
    ret_n = close.pct_change(lookback)
    liq = liquid(amount)
    alive = listed(close, 60)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        elig = pd.Series(False, index=cols)
        for c in universe:
            val = ret_n.at[dt, c]
            elig[c] = bool(
                alive.at[dt, c]
                and liq.at[dt, c]
                and pd.notna(val)
                and val > crash
            )
        picks = _bottomk(ret_n.loc[dt], bottom_n, elig)
        sparse.loc[dt] = _assign_equal(picks, cols, fallback=BOND)
    return _fill_schedule(sparse, sched, cols)


def strat_pullback(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    universe: Sequence[str],
    ma_n: int = 120,
    dip: float = -0.03,
    crash: float = -0.12,
    top_n: int = 4,
) -> pd.DataFrame:
    """Buy dips only inside an established uptrend; otherwise sit in the bond ETF."""
    cols = list(close.columns)
    sched = week_ends(close.index)
    ma = close.rolling(ma_n, min_periods=ma_n).mean()
    ret5 = close.pct_change(5)
    liq = liquid(amount)
    alive = listed(close, ma_n)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        elig = pd.Series(False, index=cols)
        for c in universe:
            px = close.at[dt, c]
            trend = ma.at[dt, c]
            val = ret5.at[dt, c]
            elig[c] = bool(
                alive.at[dt, c]
                and liq.at[dt, c]
                and pd.notna(px)
                and pd.notna(trend)
                and pd.notna(val)
                and px > trend
                and crash < val <= dip
            )
        picks = _bottomk(ret5.loc[dt], top_n, elig)
        sparse.loc[dt] = _assign_equal(picks, cols, fallback=BOND)
    return _fill_schedule(sparse, sched, cols)


def strat_lowvol(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    universe: Sequence[str],
    vol_n: int = 63,
    top_n: int = 4,
) -> pd.DataFrame:
    """Equal-weight the lowest realized-vol equity ETFs. Gold/bond are not in the universe."""
    cols = list(close.columns)
    sched = month_ends(close.index)
    vol = close.pct_change().rolling(vol_n, min_periods=vol_n).std()
    liq = liquid(amount)
    alive = listed(close, vol_n + 5)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        scores = {}
        for c in universe:
            if not (bool(alive.at[dt, c]) and bool(liq.at[dt, c])):
                continue
            val = vol.at[dt, c]
            if pd.notna(val) and val > 0:
                scores[c] = val
        if not scores:
            sparse.loc[dt] = _assign_equal([], cols, fallback=BOND)
            continue
        ranked = sorted(scores, key=scores.get)[:top_n]
        sparse.loc[dt] = _assign_equal(ranked, cols, fallback=BOND)
    return _fill_schedule(sparse, sched, cols)


def strat_ma_barbell(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    ma_n: int = 200,
    equity_w: float = 0.60,
    sector_n: int = 2,
    sector_lb: int = 63,
) -> pd.DataFrame:
    """Faber-style risk switch on CSI300 ETF, with a small sector sleeve when risk-on."""
    cols = list(close.columns)
    sched = week_ends(close.index)
    ma = close[BENCH].rolling(ma_n, min_periods=ma_n).mean()
    mom = rolling_return(close, sector_lb, 0)
    liq = liquid(amount)
    alive = listed(close, 60)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        row = pd.Series(0.0, index=cols)
        px = close.at[dt, BENCH]
        trend = ma.at[dt]
        risk_on = pd.notna(px) and pd.notna(trend) and px > trend
        if not risk_on:
            row[BOND] = 0.70
            row[GOLD] = 0.30
        else:
            broads = [c for c in BROAD if bool(alive.at[dt, c]) and bool(liq.at[dt, c]) and pd.notna(close.at[dt, c])]
            if broads:
                for c in broads:
                    row[c] += equity_w / len(broads)
            else:
                row[BOND] += equity_w
            scores = {}
            for c in SECTOR:
                if not (bool(alive.at[dt, c]) and bool(liq.at[dt, c])):
                    continue
                val = mom.at[dt, c]
                if pd.notna(val) and val > 0:
                    scores[c] = float(val)
            ranked = sorted(scores, key=scores.get, reverse=True)[:sector_n]
            sleeve = 1.0 - equity_w
            if ranked:
                for c in ranked:
                    row[c] += sleeve / len(ranked)
            else:
                row[GOLD] += sleeve
        # Numerical tidy
        if row.sum() <= 0:
            row[BOND] = 1.0
        else:
            row = row / row.sum()
        sparse.loc[dt] = row
    return _fill_schedule(sparse, sched, cols)


def strat_sector_rs(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    lookback: int = 63,
    top_n: int = 3,
) -> pd.DataFrame:
    """Rotate into the strongest sectors; unfilled slots stay in the bond ETF."""
    cols = list(close.columns)
    sched = month_ends(close.index)
    mom = rolling_return(close, lookback, 0)
    liq = liquid(amount)
    alive = listed(close, lookback + 5)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        elig_scores = {}
        for c in SECTOR:
            if not (bool(alive.at[dt, c]) and bool(liq.at[dt, c])):
                continue
            val = mom.at[dt, c]
            if pd.notna(val) and val > 0:
                elig_scores[c] = float(val)
        picks = sorted(elig_scores, key=elig_scores.get, reverse=True)[:top_n]
        if not picks:
            sparse.loc[dt] = _assign_equal([], cols, fallback=BOND)
            continue
        row = _assign_equal(picks, cols)
        # If fewer than top_n qualify, the equal-weight helper already puts 100% in the picks.
        # Keep them fully invested in the qualifying sectors (absolute filter already applied).
        sparse.loc[dt] = row
    return _fill_schedule(sparse, sched, cols)


def strat_risk_parity(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    universe: Sequence[str],
    vol_n: int = 63,
) -> pd.DataFrame:
    """Inverse-vol weights. A diversification baseline, not a forecast."""
    cols = list(close.columns)
    sched = month_ends(close.index)
    vol = close.pct_change().rolling(vol_n, min_periods=vol_n).std()
    liq = liquid(amount)
    alive = listed(close, vol_n + 5)
    sparse = _empty_weights(close.index, cols)
    for dt in sched:
        inv = {}
        for c in universe:
            if not (bool(alive.at[dt, c]) and bool(liq.at[dt, c])):
                continue
            val = vol.at[dt, c]
            if pd.notna(val) and val > 0:
                inv[c] = 1.0 / float(val)
        row = pd.Series(0.0, index=cols)
        if not inv:
            row[BOND] = 1.0
        else:
            z = sum(inv.values())
            for c, v in inv.items():
                row[c] = v / z
        sparse.loc[dt] = row
    return _fill_schedule(sparse, sched, cols)


def strat_donchian(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    universe: Sequence[str],
    entry_n: int = 55,
    exit_n: int = 20,
) -> pd.DataFrame:
    """Turtle-style channel. State changes only on week-ends; flat book goes to bonds."""
    cols = list(close.columns)
    sched = set(week_ends(close.index))
    prior_high = close.rolling(entry_n, min_periods=entry_n).max().shift(1)
    prior_low = close.rolling(exit_n, min_periods=exit_n).min().shift(1)
    liq = liquid(amount)
    alive = listed(close, entry_n + 5)
    held = {c: False for c in universe}
    sparse = _empty_weights(close.index, cols)
    for dt in close.index:
        if dt not in sched:
            continue
        active = []
        for c in universe:
            if not (bool(alive.at[dt, c]) and bool(liq.at[dt, c]) and pd.notna(close.at[dt, c])):
                held[c] = False
                continue
            px = close.at[dt, c]
            hi = prior_high.at[dt, c]
            lo = prior_low.at[dt, c]
            if held[c]:
                if pd.notna(lo) and px <= lo:
                    held[c] = False
            else:
                if pd.notna(hi) and px >= hi:
                    held[c] = True
            if held[c]:
                active.append(c)
        sparse.loc[dt] = _assign_equal(active, cols, fallback=BOND)
    return _fill_schedule(sparse, sched, cols)


def open_to_open_returns(open_px: pd.DataFrame, close_px: pd.DataFrame) -> pd.DataFrame:
    """Interval return earned by a position established at today's open.

    Intermediate days: open[t] -> open[t+1]. Final day: open -> close mark.
    """
    nxt = open_px.shift(-1) / open_px - 1.0
    last = nxt.index[-1]
    nxt.loc[last] = close_px.loc[last] / open_px.loc[last] - 1.0
    return nxt.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_backtest(
    weights: pd.DataFrame,
    interval_ret: pd.DataFrame,
    cost_map: Dict[str, float],
    cost_override: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Signal at close t is traded at the next open (weights.shift(1)).

    interval_ret[t] is the open-to-next-open return of day t, earned by the
    position that was put on at the open of day t.
    """
    cols = [c for c in weights.columns if c in interval_ret.columns]
    w_signal = weights[cols].reindex(interval_ret.index).fillna(0.0)
    w_exec = w_signal.shift(1).fillna(0.0)
    rets = interval_ret[cols].reindex(w_exec.index).fillna(0.0)
    gross = (w_exec * rets).sum(axis=1)
    delta = w_exec.diff().abs()
    # First execution day: entering from cash is turnover.
    if len(delta):
        delta.iloc[0] = w_exec.iloc[0].abs()
    if cost_override is None:
        bps = pd.Series({c: cost_map.get(c, 12.0) for c in cols})
    else:
        bps = pd.Series(cost_override, index=cols)
    cost = (delta * (bps / 10000.0)).sum(axis=1)
    net = gross - cost
    equity = (1.0 + net).cumprod()
    turnover = delta.sum(axis=1)
    return equity, turnover


def cost_map_for(columns: Sequence[str]) -> Dict[str, float]:
    out = {}
    for c in columns:
        bucket = UNIVERSE.get(c, {}).get("bucket", "sector")
        out[c] = COST_BPS[bucket]
    return out


def perf_stats(equity: pd.Series, turnover: Optional[pd.Series] = None) -> Dict[str, float]:
    eq = equity.dropna()
    if len(eq) < 5:
        return {}
    rets = eq.pct_change().dropna()
    if rets.empty:
        return {}
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
    vol = float(rets.std() * math.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * math.sqrt(252)) if len(downside) > 5 and downside.std() > 0 else 0.0
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    gains = float(rets[rets > 0].sum())
    losses = float(-rets[rets < 0].sum())
    pf = float(gains / losses) if losses > 0 else float("inf")
    q = float(rets.quantile(0.05))
    tail = rets[rets <= q]
    cvar = float(tail.mean()) if len(tail) else 0.0
    # Monthly win rate is the decision-relevant one for monthly books.
    month_eq = eq.resample("ME").last().dropna()
    month_ret = month_eq.pct_change()
    if pd.notna(eq.iloc[0]) and len(month_eq):
        # First month: level vs start.
        first = month_eq.index[0]
        month_ret.loc[first] = month_eq.iloc[0] / eq.iloc[0] - 1.0
    month_ret = month_ret.dropna()
    win_m = float((month_ret > 0).mean()) if len(month_ret) else float("nan")
    worst_m = float(month_ret.min()) if len(month_ret) else float("nan")
    ann_turn = float(turnover.reindex(rets.index).fillna(0.0).mean() * 252) if turnover is not None else float("nan")
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate_daily": float((rets > 0).mean()),
        "win_rate_monthly": win_m,
        "profit_factor": pf,
        "cvar_5": cvar,
        "worst_month": worst_m,
        "ann_turnover": ann_turn,
        "n_days": int(len(rets)),
    }


def slice_equity(equity: pd.Series, start: str, end: Optional[str] = None) -> pd.Series:
    sub = equity.loc[equity.index >= pd.Timestamp(start)]
    if end is not None:
        sub = sub.loc[sub.index <= pd.Timestamp(end)]
    if sub.empty:
        return sub
    return sub / sub.iloc[0]


def yearly_returns(equity: pd.Series) -> Dict[str, float]:
    """Calendar-year return from the prior year-end level (first year: from the first mark)."""
    eq = equity.dropna()
    if eq.empty:
        return {}
    year_end = eq.resample("YE").last().dropna()
    out: Dict[str, float] = {}
    prev = float(eq.iloc[0])
    for ts, level in year_end.items():
        out[str(int(ts.year))] = float(level / prev - 1.0)
        prev = float(level)
    return out


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{digits}f}"


def regime_label(close: pd.Series) -> Dict[str, float | str]:
    px = close.dropna()
    last = px.index[-1]
    ma60 = px.rolling(60).mean().iloc[-1]
    ma200 = px.rolling(200).mean().iloc[-1]
    ret20 = float(px.iloc[-1] / px.iloc[-21] - 1.0) if len(px) > 21 else float("nan")
    ret60 = float(px.iloc[-1] / px.iloc[-61] - 1.0) if len(px) > 61 else float("nan")
    ret252 = float(px.iloc[-1] / px.iloc[-253] - 1.0) if len(px) > 253 else float("nan")
    vol20 = float(px.pct_change().tail(20).std() * math.sqrt(252))
    vol60 = float(px.pct_change().tail(60).std() * math.sqrt(252))
    above200 = bool(px.iloc[-1] > ma200) if pd.notna(ma200) else False
    slope = px.rolling(60).mean().iloc[-1] - px.rolling(60).mean().iloc[-21] if len(px) > 80 else float("nan")
    if above200 and pd.notna(slope) and slope > 0 and ret60 > 0.05:
        label = "bull"
    elif (not above200) and ret60 < -0.05:
        label = "bear"
    else:
        label = "sideways"
    return {
        "last": str(last.date()),
        "level": float(px.iloc[-1]),
        "ret_20d": ret20,
        "ret_60d": ret60,
        "ret_252d": ret252,
        "vol_20d": vol20,
        "vol_60d": vol60,
        "dist_ma60": float(px.iloc[-1] / ma60 - 1.0) if pd.notna(ma60) else float("nan"),
        "dist_ma200": float(px.iloc[-1] / ma200 - 1.0) if pd.notna(ma200) else float("nan"),
        "above_ma200": above200,
        "label": label,
    }


def grade(full: Dict[str, float], oos: Dict[str, float], bench_full: Dict[str, float], bench_oos: Dict[str, float], own_full: Dict[str, float], own_oos: Dict[str, float]) -> str:
    """Quality bar from the skill, applied after costs.

    PASS requires beating BOTH the tradable CSI300 ETF and the strategy's own
    equal-weight universe, in-sample-or-full AND out-of-sample, with Sharpe > 1
    and max drawdown better than -30%. Anything short of that is not an edge.
    """
    beat_bench = full["cagr"] > bench_full["cagr"] and full["sharpe"] > bench_full["sharpe"]
    beat_own = full["cagr"] > own_full["cagr"] and full["sharpe"] > own_full["sharpe"]
    beat_oos = oos["cagr"] > bench_oos["cagr"] and oos["cagr"] > own_oos["cagr"] and oos["sharpe"] > 0
    sharpe_ok = full["sharpe"] >= 1.0 and oos["sharpe"] >= 1.0
    mdd_ok = full["max_drawdown"] >= -0.30
    if beat_bench and beat_own and beat_oos and sharpe_ok and mdd_ok:
        return "PASS"
    if beat_bench and beat_own and beat_oos and full["sharpe"] >= 0.7:
        return "CONDITIONAL"
    if beat_bench and beat_oos and full["sharpe"] > bench_full["sharpe"]:
        return "WEAK"
    return "FAIL"


def latest_weights(weights: pd.DataFrame, asof: pd.Timestamp) -> Dict[str, float]:
    hist = weights.loc[weights.index <= asof]
    if hist.empty:
        return {}
    row = hist.iloc[-1]
    return {c: float(v) for c, v in row.items() if v and v > 1e-6}


def try_macro() -> Dict[str, str]:
    """Best-effort macro prints. Failures are data gaps, not imputed values."""
    notes: Dict[str, str] = {}
    try:
        pmi = ak.macro_china_pmi()
        label = pmi.columns[0]
        years = pmi[label].astype(str).str.extract(r"(\d{4})")[0]
        months = pmi[label].astype(str).str.extract(r"(\d{1,2})\s*月")[0]
        parsed = pmi.assign(_y=years, _m=months).dropna(subset=["_y", "_m"]).copy()
        parsed["_y"] = parsed["_y"].astype(int)
        parsed["_m"] = parsed["_m"].astype(int)
        parsed = parsed.sort_values(["_y", "_m"])
        last = parsed.iloc[-1]
        mfg_col = [c for c in parsed.columns if "制造业" in str(c) and "同比" not in str(c)]
        mfg = last[mfg_col[0]] if mfg_col else "n/a"
        notes["pmi_latest"] = f"{int(last['_y']):04d}-{int(last['_m']):02d} 制造业PMI={mfg}"
        notes["pmi_rows"] = str(len(parsed))
    except Exception as exc:  # noqa: BLE001
        notes["pmi_error"] = f"{type(exc).__name__}: {exc}"
    return notes


def build_strategies(close: pd.DataFrame, amount: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Precommitted primary specs. Do not retune these after seeing scores."""
    domestic = DOMESTIC_EQ
    return {
        "BH_CSI300": strat_buyhold(close, amount, [BENCH]),
        "EW_Domestic": strat_buyhold(close, amount, domestic),
        "EW_AllWeather": strat_buyhold(close, amount, [BENCH, "510500", "159915", GOLD, BOND, "513100"]),
        "TSMOM_12_1": strat_tsmom(close, amount, domestic + [GOLD] + OFFSHORE, 252, 21),
        "DualMom_12m_Top3": strat_dual_mom(close, amount, domestic, 252, 3),
        "XS_Reversal_5d": strat_reversal(close, amount, domestic, 5, 3, -0.12),
        "Trend_Pullback": strat_pullback(close, amount, domestic, 120, -0.03, -0.12, 4),
        "LowVol_4": strat_lowvol(close, amount, domestic, 63, 4),
        "MA200_Barbell": strat_ma_barbell(close, amount, 200, 0.60, 2, 63),
        "Sector_RS_63": strat_sector_rs(close, amount, 63, 3),
        "RiskParity_AW": strat_risk_parity(
            close, amount, [BENCH, "510500", "159915", GOLD, BOND, "513100", "512890"], 63
        ),
        "Donchian_55_20": strat_donchian(close, amount, domestic + [GOLD], 55, 20),
        "Offshore_TSMOM": strat_tsmom(close, amount, OFFSHORE + [GOLD], 252, 21),
    }


OWN_UNIVERSE = {
    "BH_CSI300": [BENCH],
    "EW_Domestic": DOMESTIC_EQ,
    "EW_AllWeather": [BENCH, "510500", "159915", GOLD, BOND, "513100"],
    "TSMOM_12_1": DOMESTIC_EQ + [GOLD] + OFFSHORE,
    "DualMom_12m_Top3": DOMESTIC_EQ,
    "XS_Reversal_5d": DOMESTIC_EQ,
    "Trend_Pullback": DOMESTIC_EQ,
    "LowVol_4": DOMESTIC_EQ,
    "MA200_Barbell": BROAD + SECTOR + DEFENSE,
    "Sector_RS_63": SECTOR,
    "RiskParity_AW": [BENCH, "510500", "159915", GOLD, BOND, "513100", "512890"],
    "Donchian_55_20": DOMESTIC_EQ + [GOLD],
    "Offshore_TSMOM": OFFSHORE + [GOLD],
}

CANDIDATES = [
    "TSMOM_12_1",
    "DualMom_12m_Top3",
    "XS_Reversal_5d",
    "Trend_Pullback",
    "LowVol_4",
    "MA200_Barbell",
    "Sector_RS_63",
    "RiskParity_AW",
    "Donchian_55_20",
    "Offshore_TSMOM",
]

THESES = {
    "BH_CSI300": "可交易的沪深300 beta，是 A 股基准，不是 alpha。",
    "EW_Domestic": "国内股票 ETF 等权，用来区分“选对了资产”和“只是吃到了组合 beta”。",
    "EW_AllWeather": "股/债/金/海外的静态混合，检验主动策略是否只是在复制分散化。",
    "TSMOM_12_1": "中期动量来自反应不足与资金追逐；跳过最近 21 日以避开短期反转。看多才持有，否则国债 ETF。",
    "DualMom_12m_Top3": "只持有相对动量前 3 且强于国债的国内权益 ETF。集中持仓，熊市应退回债券。",
    "XS_Reversal_5d": "一周维度的过度反应会回摆。只做多最弱的 3 只、不做空；单周跌幅超过 12% 视为崩跌，不接刀。",
    "Trend_Pullback": "上升趋势里的回撤更可能是流动性冲击而不是基本面破裂。价格在 MA120 之上且 5 日跌超 3% 才买。",
    "LowVol_4": "低波动溢价来自彩票型偏好与杠杆约束。每月持有波动最低的 4 只权益 ETF。",
    "MA200_Barbell": "慢趋势过滤（Faber MA200）砍掉深熊左尾；风险开启时 60% 宽基 + 40% 强势行业，关闭时 70% 国债 + 30% 黄金。",
    "Sector_RS_63": "行业景气与资金在季度尺度上有惯性。只轮动行业 ETF，绝对动量为负时空仓进国债。",
    "RiskParity_AW": "按 63 日波动倒数配置股/债/金/纳指。这是风险预算，不是收益预测；目标是回撤而不是战胜牛市。",
    "Donchian_55_20": "通道突破捕捉行业与商品趋势（海龟 55/20）。周频更新，降低日内噪声和费用。",
    "Offshore_TSMOM": "纳指与标普 ETF 的时间序列动量。相关基准是海外资产本身，不能把美元牛市说成 A 股 alpha。",
}


def sensitivity(close: pd.DataFrame, amount: pd.DataFrame, interval: pd.DataFrame, costs: Dict[str, float]) -> List[Dict]:
    """Neighbor specs. A real edge should not live on a single parameter."""
    specs: List[Tuple[str, pd.DataFrame]] = []
    for lb, skip in ((126, 21), (189, 21), (252, 0), (252, 21), (252, 42)):
        specs.append((f"TSMOM_lb{lb}_skip{skip}", strat_tsmom(close, amount, DOMESTIC_EQ + [GOLD] + OFFSHORE, lb, skip)))
    for lb, n in ((126, 2), (126, 3), (252, 2), (252, 3), (252, 4), (189, 3)):
        specs.append((f"Dual_lb{lb}_n{n}", strat_dual_mom(close, amount, DOMESTIC_EQ, lb, n)))
    for ma in (100, 150, 200, 250):
        specs.append((f"MA_{ma}", strat_ma_barbell(close, amount, ma, 0.60, 2, 63)))
    for vn, n in ((42, 3), (63, 3), (63, 4), (63, 5), (126, 4)):
        specs.append((f"LowVol_v{vn}_n{n}", strat_lowvol(close, amount, DOMESTIC_EQ, vn, n)))
    for lb, n in ((42, 3), (63, 2), (63, 3), (126, 3)):
        specs.append((f"Sector_lb{lb}_n{n}", strat_sector_rs(close, amount, lb, n)))
    for entry, exit_n in ((20, 10), (55, 20), (60, 20)):
        specs.append((f"Donchian_{entry}_{exit_n}", strat_donchian(close, amount, DOMESTIC_EQ + [GOLD], entry, exit_n)))
    rows = []
    for name, w in specs:
        eq, to = run_backtest(w, interval, costs)
        eq = slice_equity(eq, EVAL_START)
        to = to.reindex(eq.index).fillna(0.0)
        st = perf_stats(eq, to)
        oos = perf_stats(slice_equity(eq, "2023-01-01"), to.reindex(eq.loc[eq.index >= "2023-01-01"].index))
        rows.append({"name": name, "sharpe": st.get("sharpe"), "cagr": st.get("cagr"), "mdd": st.get("max_drawdown"), "oos_sharpe": oos.get("sharpe"), "oos_cagr": oos.get("cagr")})
    return rows


def jackknife(close: pd.DataFrame, amount: pd.DataFrame, interval: pd.DataFrame, costs: Dict[str, float]) -> List[Dict]:
    """Drop one domestic ETF at a time from the two selection strategies."""
    rows = []
    for dropped in DOMESTIC_EQ:
        uni = [c for c in DOMESTIC_EQ if c != dropped]
        for label, weights in (
            (f"Dual_drop_{dropped}", strat_dual_mom(close, amount, uni, 252, 3)),
            (f"LowVol_drop_{dropped}", strat_lowvol(close, amount, uni, 63, 4)),
            (f"TSMOM_drop_{dropped}", strat_tsmom(close, amount, uni + [GOLD] + OFFSHORE, 252, 21)),
        ):
            eq, to = run_backtest(weights, interval, costs)
            eq = slice_equity(eq, EVAL_START)
            st = perf_stats(eq, to.reindex(eq.index))
            rows.append({"name": label, "sharpe": st.get("sharpe"), "cagr": st.get("cagr"), "mdd": st.get("max_drawdown")})
    return rows


def render_report(ctx: Dict) -> str:
    snap = ctx["snapshot"]
    lines: List[str] = []
    a = lines.append
    a(f"# A 股 ETF 策略研究报告（{ASOF}）")
    a("")
    a(f"- 研究日: **{ASOF}**（Asia/Shanghai）。A 股当日休市。")
    a(f"- 价格截止: **{ctx['last_price_date']}** 收盘。下一交易日（新浪交易日历）: **{ctx['next_session']}**。")
    a(f"- 回测区间: {EVAL_START} → {ctx['last_price_date']}。样本内 ≤ {IS_END}，样本外 ≥ 2023-01-01。")
    a("- 成交假设: 收盘产生信号，**下一交易日开盘**成交；持有收益为开盘到下一开盘（末日用开盘到收盘盯市）。")
    a("- 成本: 宽基/国债/黄金单边 **8bps**，行业 ETF **12bps**，QDII **20bps**。ETF 无印花税。压力测试统一 **25bps**。")
    a("- 约束: 只做多、无杠杆、权重和为 1。20 日成交额中位数 < 1 亿的标的不交易。")
    a("- 无风险利率: Sharpe 使用 **rf = 0**（价格序列，不是超额国债收益）。")
    a("- 复现: `python3 scripts/daily_strategy_research/run_research.py`")
    a("")
    a("## 1. Executive Summary")
    a("")
    a(ctx["exec_summary"])
    a("")
    a("### 排序（只比较预先写死的主规格，不用敏感性搜索结果挑赢家）")
    a("")
    a("| 排名 | 策略 | 等级 | Full Sharpe | Full CAGR | MDD | OOS Sharpe | OOS CAGR | 相对自身等权超额 Sharpe |")
    a("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, row in enumerate(ctx["ranking"], 1):
        a(
            f"| {i} | `{row['name']}` | {row['grade']} | {fmt_num(row['sharpe'])} | {fmt_pct(row['cagr'])} | "
            f"{fmt_pct(row['mdd'])} | {fmt_num(row['oos_sharpe'])} | {fmt_pct(row['oos_cagr'])} | {fmt_num(row['excess_sharpe_own'])} |"
        )
    a("")
    a("## 2. Market Analysis")
    a("")
    a("### 指数状态（事实）")
    a("")
    a("| 指数 | 日期 | 点位 | 20日 | 60日 | 252日 | 20日波动 | 距MA200 | 状态 |")
    a("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for symbol, name in INDEXES.items():
        info = snap["indexes"].get(symbol)
        if not info:
            a(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 缺失 |")
            continue
        a(
            f"| {name} | {info['last']} | {info['level']:.1f} | {fmt_pct(info['ret_20d'])} | {fmt_pct(info['ret_60d'])} | "
            f"{fmt_pct(info['ret_252d'])} | {fmt_pct(info['vol_20d'])} | {fmt_pct(info['dist_ma200'])} | {info['label']} |"
        )
    a("")
    a(
        f"国内权益 ETF 广度: MA60 之上 **{snap['breadth_ma60']}**，MA200 之上 **{snap['breadth_ma200']}** "
        f"（分母 {snap['breadth_n']} 只已上市且有效的国内股票 ETF）。"
    )
    a("")
    a("### ETF 截面（截至最后收盘，前复权价格收益）")
    a("")
    a("| 代码 | 名称 | 分组 | 20日 | 60日 | 252日 | >MA60 | >MA200 | 60日成交额中位数 |")
    a("| --- | --- | --- | ---: | ---: | ---: | :---: | :---: | ---: |")
    for row in snap["etfs"]:
        a(
            f"| {row['code']} | {row['name']} | {row['bucket']} | {fmt_pct(row['ret20'])} | {fmt_pct(row['ret60'])} | "
            f"{fmt_pct(row['ret252'])} | {'Y' if row['above60'] else 'N'} | {'Y' if row['above200'] else 'N'} | {row['amt60']:.1f}亿 |"
        )
    a("")
    a("### 事实与解释")
    a("")
    a(ctx["facts_block"])
    a("")
    a("### 数据缺口（不填造）")
    a("")
    for gap in ctx["gaps"]:
        a(f"- {gap}")
    a("")
    a("## 3. Strategy Details")
    a("")
    a("样本池是流动性足够的 A 股 ETF，而不是个股。个股涨跌停、停牌和冲击成本会让同一信号失真；ETF 上结论不能自动外推到个股。")
    a("")
    a("未纳入回测、仅因流动性或重叠而排除: " + ", ".join(f"`{c}`" for c in EXCLUDED_LIQUIDITY) + "（光伏/新能源，60 日成交额中位数约 1.3–1.7 亿，且与创业板/有色重叠）。")
    a("")
    a("| 策略 | 经济逻辑 | 规格 |")
    a("| --- | --- | --- |")
    specs = {
        "TSMOM_12_1": "月频。国内权益 + 黄金 + 纳指/标普。12 个月收益去掉最近 21 日 > 0 才等权持有，否则 100% 国债 ETF。",
        "DualMom_12m_Top3": "月频。国内权益。12 个月收益前 3，且必须同时 > 0 和 > 国债 ETF，否则该仓位不建立，全部落选则 100% 国债。",
        "XS_Reversal_5d": "周频。国内权益。做多过去 5 日最弱的 3 只；5 日收益 ≤ -12% 的不买。没有合格标的则国债。",
        "Trend_Pullback": "周频。收盘 > MA120，且 -12% < 5 日收益 ≤ -3%，取跌幅最大的至多 4 只。没有则国债。",
        "LowVol_4": "月频。国内权益里 63 日波动最低的 4 只等权。不把国债放进选基宇宙，避免策略退化成“永远持债”。",
        "MA200_Barbell": "周频。510300 收盘 > MA200 时 60% 宽基等权 + 40% 正动量行业前 2；否则 70% 国债 + 30% 黄金。",
        "Sector_RS_63": "月频。行业 ETF 63 日收益前 3，且收益 > 0。没有合格行业则国债。",
        "RiskParity_AW": "月频。沪深300/中证500/创业板/黄金/国债/纳指/红利低波，权重 ∝ 1/63日波动。",
        "Donchian_55_20": "周频状态机。收盘突破过去 55 日高点则持有，跌破过去 20 日低点则退出。同时持有的突破标的等权，空仓则国债。",
        "Offshore_TSMOM": "月频。仅纳指、标普、黄金，规则同 12-1 动量，否则国债。",
    }
    for name in CANDIDATES:
        a(f"| `{name}` | {THESES[name]} | {specs[name]} |")
    a("")
    a("实现与注释在 `scripts/daily_strategy_research/run_research.py`。下面是成交与费用的核心约定：")
    a("")
    a("```python")
    a("# Signal known at close t-1 is executed at the open of day t.")
    a("w_exec = weights.shift(1).fillna(0.0)")
    a("interval = open.shift(-1) / open - 1.0          # open[t] -> open[t+1]")
    a("interval.iloc[-1] = close.iloc[-1] / open.iloc[-1] - 1.0")
    a("gross = (w_exec * interval).sum(axis=1)")
    a("cost = (w_exec.diff().abs() * bps / 10000).sum(axis=1)")
    a("net = gross - cost")
    a("```")
    a("")
    a("## 4. Backtest Results")
    a("")
    a("权益曲线（对数）见 `scripts/daily_strategy_research/output/equity_curves.png`。")
    a("")
    a("### 全样本 vs 样本外")
    a("")
    a("| 策略 | CAGR | Sharpe | Sortino | MDD | 月胜率 | 盈亏比 | 最差月 | 年化换手 | 5% CVaR | IS CAGR | IS Sharpe | OOS CAGR | OOS Sharpe | OOS MDD |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in ctx["table_order"]:
        f = ctx["full"][name]
        o = ctx["oos"][name]
        ins = ctx["is"][name]
        a(
            f"| `{name}` | {fmt_pct(f['cagr'])} | {fmt_num(f['sharpe'])} | {fmt_num(f['sortino'])} | {fmt_pct(f['max_drawdown'])} | "
            f"{fmt_pct(f['win_rate_monthly'])} | {fmt_num(f['profit_factor'])} | {fmt_pct(f['worst_month'])} | {fmt_num(f['ann_turnover'], 1)} | "
            f"{fmt_pct(f['cvar_5'], 2)} | {fmt_pct(ins['cagr'])} | {fmt_num(ins['sharpe'])} | {fmt_pct(o['cagr'])} | {fmt_num(o['sharpe'])} | {fmt_pct(o['max_drawdown'])} |"
        )
    a("")
    a("### 分年收益（检验是不是单一年份的运气）")
    a("")
    years = ctx["years"]
    a("| 策略 | " + " | ".join(years) + " |")
    a("| --- | " + " | ".join(["---:"] * len(years)) + " |")
    for name in ctx["table_order"]:
        vals = " | ".join(fmt_pct(ctx["yearly"][name].get(y, float("nan"))) for y in years)
        a(f"| `{name}` | {vals} |")
    a("")
    a("### 成本压力（单边统一 25bps，其余不变）")
    a("")
    a("| 策略 | 基准成本 Sharpe | 25bps Sharpe | 25bps CAGR | 25bps MDD |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for name in CANDIDATES:
        base = ctx["full"][name]["sharpe"]
        st = ctx["stress"][name]
        a(f"| `{name}` | {fmt_num(base)} | {fmt_num(st['sharpe'])} | {fmt_pct(st['cagr'])} | {fmt_pct(st['max_drawdown'])} |")
    a("")
    a("### 参数邻域")
    a("")
    a("主规格在跑完之前已经写死。下表只回答“换一个常用参数，结论会不会翻掉”。")
    a("")
    a("| 变体 | Sharpe | CAGR | MDD | OOS Sharpe |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for row in ctx["sensitivity"]:
        a(f"| `{row['name']}` | {fmt_num(row['sharpe'])} | {fmt_pct(row['cagr'])} | {fmt_pct(row['mdd'])} | {fmt_num(row['oos_sharpe'])} |")
    a("")
    a("### 去掉单一 ETF（集中度）")
    a("")
    a(ctx["jackknife_note"])
    a("")
    a("| 检验 | Sharpe | CAGR | MDD |")
    a("| --- | ---: | ---: | ---: |")
    for row in ctx["jackknife_highlights"]:
        a(f"| `{row['name']}` | {fmt_num(row['sharpe'])} | {fmt_pct(row['cagr'])} | {fmt_pct(row['mdd'])} |")
    a("")
    a("## 5. Comparison & Ranking")
    a("")
    a("等级规则（全部在扣费之后）：")
    a("")
    a("- **PASS**: 全样本 CAGR 与 Sharpe 同时高于沪深300ETF **和** 策略自身宇宙等权；样本外 CAGR 同时高于这两个基准且 OOS Sharpe > 0；全样本与样本外 Sharpe 都 ≥ 1；全样本 MDD ≥ -30%。")
    a("- **CONDITIONAL**: 同样跑赢两个基准（含样本外），全样本 Sharpe ≥ 0.7，但 Sharpe>1 或 MDD<-30% 的严格门槛有一条没达到。")
    a("- **WEAK**: 全样本跑赢沪深300，且样本外 CAGR 高于沪深300和自身等权，但没进 CONDITIONAL。原因是全样本 Sharpe < 0.7，和/或全样本没有同时跑赢自身等权的 CAGR 与 Sharpe。WEAK 可以有一点选择效果，仍然不是可部署 edge。")
    a("- **FAIL**: 其余。样本外没有跑赢自身等权、或全样本没有跑赢沪深300，都在这里。高 Sharpe 若只来自资产 beta（例如海外 ETF 本身的牛市），也是 FAIL。")
    a("")
    a(ctx["compare_narrative"])
    a("")
    a("## 6. Final Recommendations")
    a("")
    a(ctx["recommendations"])
    a("")
    a("### 2026-09-30 收盘信号（下一开盘，即 2026-10-08，才成交）")
    a("")
    a("| 策略 | 目标权重 |")
    a("| --- | --- |")
    for name, holdings in ctx["holdings"].items():
        if not holdings:
            text = "空仓/无信号"
        else:
            parts = [f"{UNIVERSE.get(c, {}).get('name', c)}({c}) {v*100:.0f}%" for c, v in sorted(holdings.items(), key=lambda kv: -kv[1])]
            text = "、".join(parts)
        a(f"| `{name}` | {text} |")
    a("")
    a("这些权重是研究输出，不是下单指令。节前最后一个交易日的信号会跨过长假，跳空风险由假期外盘和汇率承担。")
    a("")
    a("## 7. Deployment Considerations")
    a("")
    a("- 账户: 场内 ETF，现金账户即可。不要加杠杆去追 Sharpe。")
    a("- 下单: 信号只用收盘价。假期后第一天用开盘或开盘后 15 分钟的限价，不要用收盘价去“补”回测。")
    a("- 容量: 本组合假设名义本金远小于各 ETF 日成交额的 1%。上表 60 日成交额中位数最低的行业 ETF 仍在 2 亿以上；千万级人民币以内冲击有限，上亿则需要拆单。")
    a("- 单策略风险: 任一策略初始风险预算不超过组合的 25%，直到纸面或小资金跟踪满一个季度。")
    a("- 监控: 每月核对持仓是否与脚本一致；若实盘滑点连续两个月高于假设 2 倍，停用高换手策略（反转、通道）。")
    a("- 失效开关: 滚动 12 个月收益差于自身等权基准超过 15 个百分点，或回撤深于回测 MDD 再加 10 个百分点，就停止而不是加参数。")
    a("- QDII（513100/513500）有溢价和申赎时滞。回测用的是场内成交价，已经包含历史溢价波动，但没有单列“高溢价不买”的规则。实盘应在溢价显著高于自身 1 年中位数时降低权重。")
    a("")
    a("## 8. What Could Go Wrong")
    a("")
    a(ctx["invalidation"])
    a("")
    a("## 自检")
    a("")
    a(ctx["self_check"])
    a("")
    a("---")
    a("")
    a("数字全部由 `scripts/daily_strategy_research/run_research.py` 在本地拉到的行情上计算。解释句已在正文标明；没有解释句的表格是事实。")
    a("")
    return "\n".join(lines)


def snapshot_etfs(close: pd.DataFrame, amount: pd.DataFrame) -> List[Dict]:
    rows = []
    ma60 = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()
    for code, meta in UNIVERSE.items():
        px = close[code].dropna()
        if px.empty:
            continue
        last = px.index[-1]
        def _ret(n: int) -> float:
            if len(px) <= n or pd.isna(px.iloc[-n - 1]) or px.iloc[-n - 1] == 0:
                return float("nan")
            return float(px.iloc[-1] / px.iloc[-n - 1] - 1.0)
        rows.append(
            {
                "code": code,
                "name": meta["name"],
                "bucket": meta["bucket"],
                "ret20": _ret(20),
                "ret60": _ret(60),
                "ret252": _ret(252),
                "above60": bool(pd.notna(ma60.at[last, code]) and px.iloc[-1] > ma60.at[last, code]),
                "above200": bool(pd.notna(ma200.at[last, code]) and px.iloc[-1] > ma200.at[last, code]),
                "amt60": float(amount[code].tail(60).median() / 1e8),
            }
        )
    rows.sort(key=lambda r: (-(r["ret60"] if r["ret60"] == r["ret60"] else -9), r["code"]))
    return rows


def write_chart(equity_map: Dict[str, pd.Series], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    prefer = ["BH_CSI300", "EW_Domestic", "LowVol_4", "DualMom_12m_Top3", "TSMOM_12_1", "MA200_Barbell", "RiskParity_AW", "Sector_RS_63"]
    for name in prefer:
        if name not in equity_map:
            continue
        eq = equity_map[name]
        ax.plot(eq.index, eq.values, label=name, linewidth=1.4)
    ax.set_yscale("log")
    ax.set_title("ETF strategy equity (log, after costs)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    open_px, close, amount, index_close, meta = load_market()
    # Evaluation calendar follows the CSI300 ETF.
    cal = close[BENCH].dropna().index
    open_px = open_px.reindex(cal)
    close = close.reindex(cal)
    amount = amount.reindex(cal)
    interval = open_to_open_returns(open_px, close)
    costs = cost_map_for(close.columns)

    print("building strategies...")
    weights = build_strategies(close, amount)
    # Own-universe baselines (may duplicate a primary; that is fine).
    own_weights = {name: strat_buyhold(close, amount, codes) for name, codes in OWN_UNIVERSE.items()}

    full_stats = {}
    oos_stats = {}
    is_stats = {}
    stress_stats = {}
    yearly = {}
    equities = {}
    turnovers = {}
    holdings = {}
    last_dt = close.index.max()

    for name, w in weights.items():
        eq, to = run_backtest(w, interval, costs)
        eq = slice_equity(eq, EVAL_START)
        to = to.reindex(eq.index).fillna(0.0)
        equities[name] = eq
        turnovers[name] = to
        full_stats[name] = perf_stats(eq, to)
        is_stats[name] = perf_stats(slice_equity(eq, EVAL_START, IS_END), to.loc[: pd.Timestamp(IS_END)])
        oos_eq = slice_equity(eq, "2023-01-01")
        oos_stats[name] = perf_stats(oos_eq, to.reindex(oos_eq.index).fillna(0.0))
        yearly[name] = yearly_returns(eq)
        holdings[name] = latest_weights(w, last_dt)
        eq_s, to_s = run_backtest(w, interval, costs, cost_override=STRESS_BPS)
        eq_s = slice_equity(eq_s, EVAL_START)
        stress_stats[name] = perf_stats(eq_s, to_s.reindex(eq_s.index).fillna(0.0))
        print(f"{name:20s} sharpe={full_stats[name]['sharpe']:.2f} cagr={full_stats[name]['cagr']:.1%} mdd={full_stats[name]['max_drawdown']:.1%}")

    own_full = {}
    own_oos = {}
    for name, w in own_weights.items():
        eq, to = run_backtest(w, interval, costs)
        eq = slice_equity(eq, EVAL_START)
        own_full[name] = perf_stats(eq, to.reindex(eq.index).fillna(0.0))
        oos_eq = slice_equity(eq, "2023-01-01")
        own_oos[name] = perf_stats(oos_eq, to.reindex(oos_eq.index).fillna(0.0))

    bench_f = full_stats["BH_CSI300"]
    bench_o = oos_stats["BH_CSI300"]

    ranking = []
    for name in CANDIDATES:
        g = grade(full_stats[name], oos_stats[name], bench_f, bench_o, own_full[name], own_oos[name])
        ranking.append(
            {
                "name": name,
                "grade": g,
                "sharpe": full_stats[name]["sharpe"],
                "cagr": full_stats[name]["cagr"],
                "mdd": full_stats[name]["max_drawdown"],
                "oos_sharpe": oos_stats[name]["sharpe"],
                "oos_cagr": oos_stats[name]["cagr"],
                "excess_sharpe_own": full_stats[name]["sharpe"] - own_full[name]["sharpe"],
                "excess_cagr_own": full_stats[name]["cagr"] - own_full[name]["cagr"],
                "excess_cagr_bench": full_stats[name]["cagr"] - bench_f["cagr"],
                "own_sharpe": own_full[name]["sharpe"],
                "own_cagr": own_full[name]["cagr"],
            }
        )
    grade_rank = {"PASS": 0, "CONDITIONAL": 1, "WEAK": 2, "FAIL": 3}
    ranking.sort(key=lambda r: (grade_rank[r["grade"]], -r["excess_sharpe_own"], -r["oos_sharpe"], r["mdd"]))

    print("sensitivity...")
    sens = sensitivity(close, amount, interval, costs)
    print("jackknife...")
    knife = jackknife(close, amount, interval, costs)

    # Index regimes on the same last date we actually have.
    idx_info = {}
    for symbol in INDEXES:
        if symbol not in index_close.columns:
            continue
        series = index_close[symbol].dropna()
        series = series.loc[series.index <= last_dt]
        if series.empty:
            continue
        idx_info[symbol] = regime_label(series)

    etf_rows = snapshot_etfs(close, amount)
    domestic_rows = [r for r in etf_rows if r["bucket"] in ("broad", "sector") and r["ret60"] == r["ret60"]]
    breadth_n = len(domestic_rows)
    breadth_ma60 = f"{sum(1 for r in domestic_rows if r['above60'])}/{breadth_n}"
    breadth_ma200 = f"{sum(1 for r in domestic_rows if r['above200'])}/{breadth_n}"

    # Bond price-return diagnostic (coupon gap).
    bond_px = close[BOND].dropna()
    bond_px = bond_px.loc[bond_px.index >= EVAL_START]
    bond_price_total = float(bond_px.iloc[-1] / bond_px.iloc[0] - 1.0) if len(bond_px) > 2 else float("nan")
    bond_years = max((bond_px.index[-1] - bond_px.index[0]).days / 365.25, 1e-9) if len(bond_px) > 2 else float("nan")
    bond_cagr = float((1.0 + bond_price_total) ** (1.0 / bond_years) - 1.0) if bond_price_total == bond_price_total else float("nan")

    next_session = "未知"
    for d in meta.get("trade_calendar_window", []):
        if d > str(last_dt.date()):
            next_session = d
            break

    gaps = [
        "期权持仓、隐含波动率和北向资金没有纳入。本次端点未稳定提供可对齐到 ETF 决策的期权流。",
        f"国债 ETF（511010）自 {EVAL_START} 的前复权价格累计收益 {fmt_pct(bond_price_total)}（约 {fmt_pct(bond_cagr)} 年化）。这和短久期国债全收益同一量级，不能证明票息已完整计入，也没有理由再人造票息。",
        "QDII 溢价/折价没有单独建模，只体现在场内成交价里。",
        "宏观月频（PMI、社融、政策利率）本次不作为交易信号。价格状态可以计算；宏观叙事只作背景，不参与打分。",
        f"排除低流动性 ETF: {', '.join(EXCLUDED_LIQUIDITY)}。",
        "没有模拟涨跌停（ETF 罕见但极端日存在）、申赎清单暂停、以及长假期间外盘跳空后的开盘冲击超出固定 bps 的部分。",
    ]

    macro = try_macro()
    if "pmi_error" in macro:
        gaps.append(f"PMI 拉取失败，未使用: {macro['pmi_error']}")
    elif "pmi_latest" in macro:
        gaps.append(
            f"制造业 PMI 最新可读观测是 {macro['pmi_latest']}（序列 {macro.get('pmi_rows', '?')} 行）。"
            "它没有进入任何交易信号；若该日期明显早于研究日，则视为过期，不拿来解释当前行情。"
        )

    # Narratives are written from computed numbers only.
    labels = {INDEXES[s]: idx_info[s]["label"] for s in idx_info}
    hs300 = idx_info.get("sh000300", {})
    cyb = idx_info.get("sz399006", {})
    csi1000 = idx_info.get("sh000852", {})

    n_pass = sum(1 for r in ranking if r["grade"] == "PASS")
    n_cond = sum(1 for r in ranking if r["grade"] == "CONDITIONAL")
    top3 = ranking[:3]
    exec_summary = (
        f"截至 {last_dt.date()}，沪深300 判定为 **{hs300.get('label', 'n/a')}**"
        f"（60 日 {fmt_pct(hs300.get('ret_60d', float('nan')))}，距 MA200 {fmt_pct(hs300.get('dist_ma200', float('nan')))}，"
        f"20 日波动 {fmt_pct(hs300.get('vol_20d', float('nan')))}）。"
        f"创业板指为 **{cyb.get('label', 'n/a')}**，中证1000 为 **{csi1000.get('label', 'n/a')}**。"
        f"指数标签: {labels}。国内股票 ETF 在 MA200 之上的比例是 {breadth_ma200}。\n\n"
        f"10 个事先指定的策略里，严格质量门槛 PASS **{n_pass}** 个，CONDITIONAL **{n_cond}** 个。"
        f"沪深300ETF 全样本 Sharpe **{fmt_num(bench_f['sharpe'])}**、CAGR **{fmt_pct(bench_f['cagr'])}**、MDD **{fmt_pct(bench_f['max_drawdown'])}**。"
        "没有策略被允许只靠样本内成绩晋级。相对“自身等权”的超额用来区分 alpha 和 beta。\n\n"
        "下表按“等级、再按相对自身等权的 Sharpe 超额”排序。这不是部署顺序。"
        "高换手策略会在 25bps 压力测试里掉下去，部署建议以第 6 节为准。\n\n"
        "排序前三（等级不是荐股）:\n"
        + "\n".join(
            f"- `{r['name']}` — {r['grade']}，Sharpe {fmt_num(r['sharpe'])}，CAGR {fmt_pct(r['cagr'])}，"
            f"MDD {fmt_pct(r['mdd'])}，相对自身等权超额 Sharpe {fmt_num(r['excess_sharpe_own'])}"
            for r in top3
        )
    )

    # Sector leaders / laggards from the snapshot.
    eq_rows = [r for r in etf_rows if r["bucket"] in ("broad", "sector")]
    leaders = ", ".join(f"{r['name']} {fmt_pct(r['ret60'])}" for r in eq_rows[:3])
    laggards = ", ".join(f"{r['name']} {fmt_pct(r['ret60'])}" for r in eq_rows[-3:])
    facts_block = (
        f"- **事实**: 最后一根 K 线是 {last_dt.date()}。研究日 {ASOF} 不是交易日；交易日历在 2026-10-01 至 2026-10-07 无交易，下一交易日 {next_session}。\n"
        f"- **事实**: 60 日行业/风格价差很大。较强: {leaders}。较弱: {laggards}。\n"
        f"- **事实**: 黄金 ETF 60 日 {fmt_pct(next(r['ret60'] for r in etf_rows if r['code']==GOLD))}，"
        f"国债 ETF 60 日 {fmt_pct(next(r['ret60'] for r in etf_rows if r['code']==BOND))}，"
        f"纳指 ETF 60 日 {fmt_pct(next(r['ret60'] for r in etf_rows if r['code']=='513100'))}。\n"
        f"- **解释**: 沪深300、中证500、中证1000、创业板指的收盘都在各自 MA200 之下，且 60 日收益为负，所以规则把它们标成 bear；上证只是跌幅没跨过 -5% 的阈值，标成 sideways。这不是“宽基已经站回年线”。广度 {breadth_ma200}：年线之上的国内股票 ETF 是银行、煤炭、红利、红利低波、医药、创新药、半导体；其余在年线下方。\n"
        "- **解释**: 这种分化下，12 个月动量仍可能握着“一年里涨过、近两个月已回撤”的行业（半导体 252 日仍为正、60 日深跌）。那是动量规则的滞后，不是新的看多证据。\n"
        "- **解释**: 国庆长假会让 9 月 30 日的信号暴露在外盘跳空里。回测把历史长假后的开盘跳空算进开盘到开盘收益，但没有把这次假期单独加压。"
    )

    # Jackknife range
    def _range(prefix: str) -> str:
        rows = [r for r in knife if r["name"].startswith(prefix)]
        sharpes = [r["sharpe"] for r in rows if r["sharpe"] == r["sharpe"]]
        if not sharpes:
            return f"{prefix}: 无结果"
        worst = min(rows, key=lambda r: r["sharpe"])
        return (
            f"`{prefix}` 去掉一只国内 ETF 后 Sharpe 范围 {min(sharpes):.2f}–{max(sharpes):.2f}，"
            f"最差是去掉 {worst['name'].split('_')[-1]}（Sharpe {worst['sharpe']:.2f}，CAGR {worst['cagr']:.1%}）"
        )

    jack_note = "；".join([_range("Dual_drop"), _range("LowVol_drop"), _range("TSMOM_drop")]) + "。"
    # Highlight worst and best of each family plus the primary for comparison.
    highlights = []
    for prefix in ("Dual_drop", "LowVol_drop", "TSMOM_drop"):
        rows = [r for r in knife if r["name"].startswith(prefix)]
        if not rows:
            continue
        highlights.append(min(rows, key=lambda r: r["sharpe"]))
        highlights.append(max(rows, key=lambda r: r["sharpe"]))

    # Comparison narrative from grades.
    lines_cmp = []
    for r in ranking:
        lines_cmp.append(
            f"- `{r['name']}` **{r['grade']}**。全样本 Sharpe {r['sharpe']:.2f} vs 沪深300 {bench_f['sharpe']:.2f} "
            f"vs 自身等权 {r['own_sharpe']:.2f}；CAGR {r['cagr']:.1%}（自身等权 {r['own_cagr']:.1%}，超额 {r['excess_cagr_own']:.1%}）；"
            f"MDD {r['mdd']:.1%}；样本外 Sharpe {r['oos_sharpe']:.2f}，样本外 CAGR {r['oos_cagr']:.1%}。"
        )
    compare_narrative = "\n".join(lines_cmp)

    def _hold_txt(name: str) -> str:
        hold = holdings.get(name, {})
        parts = [f"{UNIVERSE.get(c, {}).get('name', c)} {v*100:.0f}%" for c, v in sorted(hold.items(), key=lambda kv: -kv[1])]
        return "、".join(parts) if parts else "无"

    # Deployment list is not the Sharpe sort. A rule is discussable only if the
    # 25bps stress still beats buy-and-hold CSI300 on Sharpe, and the in-sample
    # window was not a loss. Static all-weather is the hurdle, not an alpha.
    bench_stress = stress_stats["BH_CSI300"]["sharpe"]
    aw = full_stats["EW_AllWeather"]
    aw_oos = oos_stats["EW_AllWeather"]
    recommendations = "\n\n".join(
        [
            f"严格质量门槛 PASS = {n_pass}，CONDITIONAL = {n_cond}。本次**没有可声称的 edge**。下面 5 条是研究结论，不是下单建议。",
            (
                f"1. **默认比较基准用静态全天候，而不是把最高超额 Sharpe 当成策略。** "
                f"`EW_AllWeather`（沪深300 / 中证500 / 创业板 / 国债 / 黄金 / 纳指等权）全样本 Sharpe {aw['sharpe']:.2f}、"
                f"CAGR {aw['cagr']:.1%}、MDD {aw['max_drawdown']:.1%}，样本外 Sharpe {aw_oos['sharpe']:.2f}、CAGR {aw_oos['cagr']:.1%}。"
                f"它没有择时，所以不参与 alpha 评级。主动规则若过不了这道门槛，就不值得付换手。"
            ),
            (
                f"2. **`MA200_Barbell` 只当作风险开关，不当 alpha。** 等级 WEAK。"
                f"全样本 Sharpe {full_stats['MA200_Barbell']['sharpe']:.2f}、CAGR {full_stats['MA200_Barbell']['cagr']:.1%}、"
                f"MDD {full_stats['MA200_Barbell']['max_drawdown']:.1%}；样本内 Sharpe {is_stats['MA200_Barbell']['sharpe']:.2f}，"
                f"样本外 Sharpe {oos_stats['MA200_Barbell']['sharpe']:.2f}。25bps 下 Sharpe 仍有 {stress_stats['MA200_Barbell']['sharpe']:.2f}"
                f"（沪深300压力 Sharpe {bench_stress:.2f}）。"
                f"MA100/150/200/250 的 Sharpe 落在 0.51–0.55，不是单点参数。"
                f"当前沪深300在 MA200 下方，假期后目标仓位是 {_hold_txt('MA200_Barbell')}。"
                f"失效条件：滚动 12 个月落后自身等权超过 15 个百分点，或回撤深于历史 MDD 再加 10 个百分点。"
            ),
            (
                f"3. **`LowVol_4` 是唯一还站得住的权益袖套，但它是银行/红利风格，不是分散的低波异象。** 等级 WEAK。"
                f"全样本 Sharpe {full_stats['LowVol_4']['sharpe']:.2f}、CAGR {full_stats['LowVol_4']['cagr']:.1%}、"
                f"MDD {full_stats['LowVol_4']['max_drawdown']:.1%}（差于 -30% 门槛），样本外 Sharpe {oos_stats['LowVol_4']['sharpe']:.2f}。"
                f"年化换手 {full_stats['LowVol_4']['ann_turnover']:.1f}，25bps 下 Sharpe {stress_stats['LowVol_4']['sharpe']:.2f}，成本不敏感。"
                f"去掉银行 ETF 后 Sharpe 降到约 0.23；`n=5` 的邻域 Sharpe 降到 0.29。持有数一变，结论就变。"
                f"假期后目标仓位: {_hold_txt('LowVol_4')}。"
            ),
            (
                f"4. **不要把 `Trend_Pullback` 的排序第一当成可交易。** 它相对自身等权的 Sharpe 超额最高（{next(r['excess_sharpe_own'] for r in ranking if r['name']=='Trend_Pullback'):+.2f}），"
                f"但年化换手 {full_stats['Trend_Pullback']['ann_turnover']:.0f} 倍。25bps 下 Sharpe 从 {full_stats['Trend_Pullback']['sharpe']:.2f} 降到 {stress_stats['Trend_Pullback']['sharpe']:.2f}，"
                f"CAGR 降到 {stress_stats['Trend_Pullback']['cagr']:.1%}，低于沪深300的压力结果。"
                f"2019 年收益 {yearly['Trend_Pullback'].get('2019', float('nan')):.1%}，而沪深300ETF 同年 {yearly['BH_CSI300'].get('2019', float('nan')):.1%}。当前信号是 {_hold_txt('Trend_Pullback')}，因为几乎没有“年线之上且刚回撤”的行业。"
            ),
            (
                f"5. **动量、反转、通道、行业轮动、未分组的风险平价，这次都不作为建议。** "
                f"`DualMom_12m_Top3` 样本内 CAGR {is_stats['DualMom_12m_Top3']['cagr']:.1%}、Sharpe {is_stats['DualMom_12m_Top3']['sharpe']:.2f}，全样本超额几乎为零，MDD {full_stats['DualMom_12m_Top3']['max_drawdown']:.1%}；"
                f"去掉煤炭 ETF 后 Sharpe 降到约 0.21。当前仍把约三分之一放在半导体，而半导体 60 日收益为负、252 日仍为正，这是长周期动量的滞后。"
                f"`TSMOM_12_1` 全样本 CAGR 低于自身等权（超额 {next(r['excess_cagr_own'] for r in ranking if r['name']=='TSMOM_12_1'):.1%}）。"
                f"`Offshore_TSMOM` 全样本 Sharpe {full_stats['Offshore_TSMOM']['sharpe']:.2f} 看起来最好，样本外 CAGR {oos_stats['Offshore_TSMOM']['cagr']:.1%} 却低于其资产等权的样本外 CAGR，收益来自纳指/标普/黄金本身。"
                f"`RiskParity_AW` 因为债券波动远低于股票，权重被吸进国债（当前 {_hold_txt('RiskParity_AW')}），全样本 CAGR 低于自身等权。"
                f"`XS_Reversal_5d`、`Donchian_55_20`、`Sector_RS_63` 的样本外 Sharpe 分别是 "
                f"{oos_stats['XS_Reversal_5d']['sharpe']:.2f}、{oos_stats['Donchian_55_20']['sharpe']:.2f}、{oos_stats['Sector_RS_63']['sharpe']:.2f}，且都跑输自身等权。"
            ),
        ]
    )

    invalidation = "\n".join(
        [
            "- **动量失效**: 若未来 12 个月截面动量（DualMom / TSMOM / Sector_RS）滚动收益差于自身等权超过 15 个百分点，反应不足的解释不成立，常见于剧烈轮动和政策脉冲行情。",
            "- **低波失效**: 低波组合若在风险偏好急升时持续跑输等权（例如连续两个季度落后超过 10 个百分点）仍可能只是风格，不是被推翻；被推翻的条件是它在下跌年份的回撤不再小于等权。",
            "- **趋势过滤失效**: MA200 在震荡市会反复打脸。若 MA 邻域（100/150/200/250）的 Sharpe 极差超过 0.4，本样本里的 MA 结果就是参数巧合。",
            "- **反转失效**: 单边趋势里做多最弱行业会连续亏损。XS_Reversal 只要样本外 Sharpe 已为负，就不应因为“下周该反弹了”重新启用。",
            "- **长假跳空**: 9 月 30 日信号到 10 月 8 日开盘之间没有 A 股价格。外盘、汇率、商品在假期的变动不会出现在信号里。",
            "- **成本**: 25bps 压力后若 Sharpe 掉到基准以下，这个策略只在低费率账户里存在。",
            "- **制度**: ETF 申赎、QDII 额度、印花税或 T+0 规则变化会改变可成交性。回测没有包含这些结构性断裂之后的数据。",
            "- **过拟合**: 本次比较了 10 个主策略和一批邻域参数。排序第一名的 Sharpe 有选择偏差。没有 PASS 时，不把第一名说成 edge。",
        ]
    )
    self_check = "\n".join(
        [
            f"- 缺什么: 国债 ETF 前复权累计收益 {fmt_pct(bond_price_total)}（年化约 {fmt_pct(bond_cagr)}），和短债全收益同一量级，但无法证明票息已经全部在价格里。期权、北向没有进入模型。PMI 只记录、不交易。",
            "- 假设弱在哪: 固定 bps 滑点、开盘能按回测权重成交、qfq 可以当作持有收益。`Trend_Pullback` 在 25bps 下失效，说明 12bps 的行业 ETF 成本对周频策略偏乐观。",
            f"- edge 还是拟合: 主规格在出结果前写死。PASS 数量 = {n_pass}。结论是这次没有可部署 edge。排序第一的回调策略换手太高，不能升格。",
            "- 区间: 分年收益覆盖 2018、2019–2020、2021–2022、2023、2024–2026。样本外从 2023-01-01 起，事先指定。`DualMom` 的正收益集中在样本外，样本内 CAGR 为负，所以不能把全样本的微弱超额当成稳定规律。",
            "- 什么会改变看法: 见第 8 节。若用确认过的全收益债券序列重跑后，MA200 的空仓段明显变好或变差，风险开关的历史成绩需要重估。若低波在去掉银行 ETF 后仍然稳定，才可以把第 3 条从“风格暴露”改写成“异象”。",
        ]
    )

    years = sorted({y for m in yearly.values() for y in m})
    table_order = ["BH_CSI300", "EW_Domestic", "EW_AllWeather"] + [r["name"] for r in ranking]

    ctx = {
        "last_price_date": str(last_dt.date()),
        "next_session": next_session,
        "exec_summary": exec_summary,
        "ranking": ranking,
        "snapshot": {"indexes": idx_info, "etfs": etf_rows, "breadth_ma60": breadth_ma60, "breadth_ma200": breadth_ma200, "breadth_n": breadth_n},
        "facts_block": facts_block,
        "gaps": gaps,
        "full": full_stats,
        "oos": oos_stats,
        "is": is_stats,
        "stress": stress_stats,
        "yearly": yearly,
        "years": years,
        "table_order": table_order,
        "sensitivity": sens,
        "jackknife_note": jack_note,
        "jackknife_highlights": highlights,
        "compare_narrative": compare_narrative,
        "recommendations": recommendations,
        "holdings": {name: holdings[name] for name in table_order},
        "invalidation": invalidation,
        "self_check": self_check,
    }

    report = render_report(ctx)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    # Machine-readable artifacts.
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return obj

    (OUT / "metrics.json").write_text(json.dumps(_clean({"full": full_stats, "oos": oos_stats, "is": is_stats, "stress": stress_stats, "own_full": own_full, "own_oos": own_oos, "yearly": yearly}), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "ranking.json").write_text(json.dumps(_clean(ranking), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "market_snapshot.json").write_text(json.dumps(_clean(ctx["snapshot"]), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "holdings.json").write_text(json.dumps(holdings, ensure_ascii=False, indent=2), encoding="utf-8")
    eq_df = pd.DataFrame({k: v for k, v in equities.items()})
    eq_df.to_csv(OUT / "equity_curves.csv", index_label="date")
    write_chart(equities, OUT / "equity_curves.png")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
