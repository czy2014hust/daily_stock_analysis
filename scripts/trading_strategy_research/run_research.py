#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-registered A-share ETF strategy research for 2026-10-05.

Eight economically distinct rules are specified before results are read.
Execution is next-open, open-to-open, after per-side costs. Parameters below
are literature defaults, not fitted on this sample. The sensitivity grid is a
stability check; the IS-best parameter is reported only as a diagnostic.

Data: Yahoo Finance adjusted OHLC via yfinance. A-share sessions missing from
the vendor are left missing (no invented fills).
"""

from __future__ import annotations

import json
import math
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

CACHE = Path("/tmp/tsr_20261005_yf.pkl")
EVAL_START = pd.Timestamp("2018-01-02")
IS_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")
SAMPLE_END = pd.Timestamp("2026-09-30")
DOWNLOAD_START = "2015-01-01"
DOWNLOAD_END = "2026-10-06"  # yfinance end is exclusive
RF_ANNUAL = 0.015  # assumption: China cash proxy, labeled in the report
RF_DAILY = (1.0 + RF_ANNUAL) ** (1.0 / 252.0) - 1.0
TRADING_DAYS = 252

# Per-side cost = commission + slippage. A-share ETFs have no stamp duty.
# Liquid ADV tens of亿: 2.5 bps commission + 5 bps slippage.
# Bond ETF: tighter spread. Thinner sector ETFs: extra 5 bps slippage.
COST_LIQUID = 0.00075
COST_THIN = 0.00125
COST_BOND = 0.00045

BOND = "511010.SS"
CSI300 = "510300.SS"
CSI500 = "510500.SS"

NAMES = {
    "510050.SS": "上证50",
    "510300.SS": "沪深300",
    "510500.SS": "中证500",
    "159915.SZ": "创业板",
    "512880.SS": "证券",
    "512010.SS": "医药",
    "512660.SS": "军工",
    "159928.SZ": "消费",
    "510880.SS": "红利",
    "512800.SS": "银行",
    "512480.SS": "半导体",
    "512690.SS": "酒",
    "518880.SS": "黄金",
    "513100.SS": "纳指ETF",
    "511010.SS": "国债ETF",
    "588000.SS": "科创50",
}

# Thinner names (60d ADV around or below ~5亿 in the 2026 probe).
THIN = {"512660.SS", "159928.SZ", "512690.SS", "513100.SS"}

EQUITY_CORE = [
    "510050.SS",
    "510300.SS",
    "510500.SS",
    "159915.SZ",
    "512880.SS",
    "512010.SS",
    "512660.SS",
    "159928.SZ",
    "510880.SS",
    "512800.SS",
    "512480.SS",
    "512690.SS",
]
MULTI = ["510300.SS", "510500.SS", "159915.SZ", "518880.SS", "513100.SS", BOND]
DONCHIAN_ASSETS = ["510300.SS", "510500.SS", "159915.SZ", "518880.SS", "513100.SS"]
RP_RISKY = ["510300.SS", "518880.SS", "513100.SS"]
STYLE = ["510050.SS", "510300.SS", "510500.SS", "159915.SZ"]

UNIVERSE = list(dict.fromkeys(EQUITY_CORE + MULTI + ["588000.SS"]))


def cost_of(symbol: str) -> float:
    if symbol == BOND:
        return COST_BOND
    if symbol in THIN:
        return COST_THIN
    return COST_LIQUID


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def download_prices(force: bool = False) -> Dict[str, pd.DataFrame]:
    if CACHE.exists() and not force:
        with CACHE.open("rb") as fh:
            return pickle.load(fh)
    import yfinance as yf

    frames: Dict[str, pd.DataFrame] = {}
    for symbol in UNIVERSE:
        raw = yf.download(
            symbol,
            start=DOWNLOAD_START,
            end=DOWNLOAD_END,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        raw = _flatten(raw)
        if raw.empty:
            raise RuntimeError(f"no prices for {symbol}")
        keep = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        keep.index = pd.to_datetime(keep.index).tz_localize(None)
        keep = keep[~keep.index.duplicated(keep="last")].sort_index()
        frames[symbol] = keep
    with CACHE.open("wb") as fh:
        pickle.dump(frames, fh)
    return frames


def master_calendar(frames: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """A-share sessions defined by the liquid CSI300 ETF. No synthetic dates."""
    idx = frames[CSI300].dropna(subset=["Open", "Close"]).index
    idx = idx[(idx >= pd.Timestamp("2016-01-01")) & (idx <= SAMPLE_END)]
    return pd.DatetimeIndex(idx)


def build_panels(
    frames: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opens, closes, volumes = [], [], []
    for symbol in UNIVERSE:
        df = frames[symbol].reindex(calendar)
        # Carry a halted/missing print at most 3 master sessions, else stay NaN.
        df_limited = df.ffill(limit=3)
        opens.append(df_limited["Open"].rename(symbol))
        closes.append(df_limited["Close"].rename(symbol))
        volumes.append(df["Volume"].rename(symbol))
    return pd.concat(opens, axis=1), pd.concat(closes, axis=1), pd.concat(volumes, axis=1)


def eligible_mask(close: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    seen = close.notna() & (close > 0)
    return seen.cumsum() >= min_obs


def month_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    periods = index.to_period("M")
    last = pd.Series(index, index=index).groupby(periods).transform("max")
    return pd.Series(index == last.values, index=index)


def empty_weights(index: pd.DatetimeIndex, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=list(columns))


def _ffill_rebalance(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Keep month-end (or signal-day) rows; forward-fill targets."""
    out = snapshot.copy()
    has = out.abs().sum(axis=1) > 0
    # A pure-cash signal is a row of zeros and would be lost. Callers must put
    # 100% in the bond sleeve so the row is non-zero.
    out = out.where(has)
    return out.ffill().fillna(0.0)


def strategy_trend(close: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """S1: long CSI300 when close > SMA, otherwise the government-bond ETF."""
    w = empty_weights(close.index, [CSI300, BOND])
    sma = close[CSI300].rolling(window).mean()
    ready = sma.notna()
    long = ready & (close[CSI300] > sma)
    w.loc[long, CSI300] = 1.0
    w.loc[ready & ~long, BOND] = 1.0
    return w


def strategy_xs_momentum(
    close: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 3,
) -> pd.DataFrame:
    """S2: monthly 12-1 cross-sectional momentum, top N, else bond sleeve."""
    cols = EQUITY_CORE + [BOND]
    w = empty_weights(close.index, cols)
    mom = close[EQUITY_CORE].shift(skip) / close[EQUITY_CORE].shift(lookback) - 1.0
    elig = eligible_mask(close[EQUITY_CORE], lookback)
    mom = mom.where(elig)
    me = month_end_mask(close.index)
    for dt in close.index[me]:
        row = mom.loc[dt].dropna()
        picks = row[row > 0].sort_values(ascending=False).head(top_n)
        if picks.empty:
            w.loc[dt, BOND] = 1.0
            continue
        share = 1.0 / len(picks)
        for symbol in picks.index:
            w.loc[dt, symbol] = share
        if len(picks) < top_n:
            # Unfilled slots stay in the bond sleeve (do not lever the winners).
            w.loc[dt, BOND] = 1.0 - share * len(picks)
    return _ffill_rebalance(w)


def strategy_reversal(
    close: pd.DataFrame,
    hold: int = 5,
    sma_window: int = 200,
    threshold: float = -0.03,
    pick_n: int = 2,
) -> pd.DataFrame:
    """S3: every `hold` sessions, buy the worst short-horizon names still in an uptrend."""
    cols = EQUITY_CORE + [BOND]
    w = empty_weights(close.index, cols)
    ret_h = close[EQUITY_CORE] / close[EQUITY_CORE].shift(hold) - 1.0
    sma = close[EQUITY_CORE].rolling(sma_window).mean()
    uptrend = close[EQUITY_CORE] > sma
    elig = eligible_mask(close[EQUITY_CORE], sma_window) & uptrend
    signal_rows = np.zeros(len(close.index), dtype=bool)
    signal_rows[sma_window::hold] = True
    for i, dt in enumerate(close.index):
        if not signal_rows[i]:
            continue
        losers = ret_h.loc[dt].where(elig.loc[dt]).dropna()
        losers = losers[losers <= threshold].sort_values().head(pick_n)
        if losers.empty:
            w.loc[dt, BOND] = 1.0
            continue
        share = 1.0 / len(losers)
        for symbol in losers.index:
            w.loc[dt, symbol] = share
        if len(losers) < pick_n:
            w.loc[dt, BOND] = 1.0 - share * len(losers)
    return _ffill_rebalance(w)


def strategy_dual_momentum(close: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """S4: monthly absolute + relative momentum across equity, gold, Nasdaq ETF, bonds."""
    risky = [c for c in MULTI if c != BOND]
    w = empty_weights(close.index, MULTI)
    mom = close[risky + [BOND]] / close[risky + [BOND]].shift(lookback) - 1.0
    elig = eligible_mask(close, lookback)
    me = month_end_mask(close.index)
    for dt in close.index[me]:
        row = mom.loc[dt, risky].where(elig.loc[dt, risky]).dropna()
        bond_mom = mom.loc[dt, BOND] if bool(elig.loc[dt, BOND]) else np.nan
        if row.empty or pd.isna(bond_mom):
            w.loc[dt, BOND] = 1.0
            continue
        best = row.idxmax()
        if float(row[best]) > 0 and float(row[best]) > float(bond_mom):
            w.loc[dt, best] = 1.0
        else:
            w.loc[dt, BOND] = 1.0
    return _ffill_rebalance(w)


def strategy_risk_parity(close: pd.DataFrame, target_vol: float = 0.08, window: int = 60) -> pd.DataFrame:
    """S5: monthly inverse-vol risky sleeve, scaled to target vol, residual in bonds. No leverage."""
    cols = RP_RISKY + [BOND]
    w = empty_weights(close.index, cols)
    rets = close[RP_RISKY].pct_change()
    me = month_end_mask(close.index)
    for dt in close.index[me]:
        hist = rets.loc[:dt].tail(window).dropna(how="any")
        if len(hist) < window:
            w.loc[dt, BOND] = 1.0
            continue
        vol = hist.std(ddof=1) * math.sqrt(TRADING_DAYS)
        if (vol <= 1e-8).any() or vol.isna().any():
            w.loc[dt, BOND] = 1.0
            continue
        inv = 1.0 / vol
        raw = inv / inv.sum()
        raw = raw.clip(upper=0.60)
        raw = raw / raw.sum()
        cov = hist.cov() * TRADING_DAYS
        port_var = float(np.asarray(raw) @ cov.values @ np.asarray(raw))
        port_vol = math.sqrt(max(port_var, 1e-12))
        scale = min(1.0, target_vol / port_vol)
        for symbol in RP_RISKY:
            w.loc[dt, symbol] = float(raw[symbol]) * scale
        w.loc[dt, BOND] = 1.0 - scale
    return _ffill_rebalance(w)


def strategy_low_vol(
    close: pd.DataFrame,
    sma_window: int = 120,
    vol_window: int = 60,
    top_n: int = 3,
) -> pd.DataFrame:
    """S6: monthly, among uptrend ETFs pick the lowest realized vol; else bonds."""
    cols = EQUITY_CORE + [BOND]
    w = empty_weights(close.index, cols)
    rets = close[EQUITY_CORE].pct_change()
    vol = rets.rolling(vol_window).std(ddof=1) * math.sqrt(TRADING_DAYS)
    sma = close[EQUITY_CORE].rolling(sma_window).mean()
    elig = eligible_mask(close[EQUITY_CORE], sma_window) & (close[EQUITY_CORE] > sma)
    me = month_end_mask(close.index)
    for dt in close.index[me]:
        row = vol.loc[dt].where(elig.loc[dt]).dropna()
        row = row[row > 0].sort_values().head(top_n)
        if row.empty:
            w.loc[dt, BOND] = 1.0
            continue
        share = 1.0 / len(row)
        for symbol in row.index:
            w.loc[dt, symbol] = share
        if len(row) < top_n:
            w.loc[dt, BOND] = 1.0 - share * len(row)
    return _ffill_rebalance(w)


def strategy_donchian(
    close: pd.DataFrame,
    entry: int = 120,
    exit_: int = 60,
) -> pd.DataFrame:
    """S7: equal-weight assets at a new N-day high; exit on M-day low; else bonds."""
    cols = DONCHIAN_ASSETS + [BOND]
    w = empty_weights(close.index, cols)
    prior_high = close[DONCHIAN_ASSETS].shift(1).rolling(entry).max()
    prior_low = close[DONCHIAN_ASSETS].shift(1).rolling(exit_).min()
    state = {symbol: False for symbol in DONCHIAN_ASSETS}
    started = close[DONCHIAN_ASSETS].notna().cumsum() >= entry
    for dt in close.index:
        active: List[str] = []
        for symbol in DONCHIAN_ASSETS:
            if not bool(started.loc[dt, symbol]) or pd.isna(prior_high.loc[dt, symbol]):
                state[symbol] = False
                continue
            px = float(close.loc[dt, symbol])
            if state[symbol]:
                if px < float(prior_low.loc[dt, symbol]):
                    state[symbol] = False
            elif px > float(prior_high.loc[dt, symbol]):
                state[symbol] = True
            if state[symbol]:
                active.append(symbol)
        if not active:
            w.loc[dt, BOND] = 1.0
        else:
            share = 1.0 / len(active)
            for symbol in active:
                w.loc[dt, symbol] = share
    return w


def strategy_pair_rv(
    close: pd.DataFrame,
    window: int = 60,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """S8: long-only CSI300 vs CSI500 relative value with hysteresis. No shorting."""
    w = empty_weights(close.index, [CSI300, CSI500])
    ratio = close[CSI300] / close[CSI500]
    mu = ratio.rolling(window).mean()
    sd = ratio.rolling(window).std(ddof=1)
    z = (ratio - mu) / sd
    mode = "neutral"  # neutral 50/50; rich_large -> 100% 500; cheap_large -> 100% 300
    for dt in close.index:
        zt = z.loc[dt]
        if pd.isna(zt):
            continue
        zt = float(zt)
        if mode == "neutral":
            if zt > entry_z:
                mode = "rich_large"
            elif zt < -entry_z:
                mode = "cheap_large"
        elif mode == "rich_large" and zt < exit_z:
            mode = "neutral"
        elif mode == "cheap_large" and zt > -exit_z:
            mode = "neutral"
        if mode == "rich_large":
            w.loc[dt, CSI500] = 1.0
        elif mode == "cheap_large":
            w.loc[dt, CSI300] = 1.0
        else:
            w.loc[dt, CSI300] = 0.5
            w.loc[dt, CSI500] = 0.5
    # Before z is defined the loop left zeros. Hold 50/50 once both prices exist,
    # matching the neutral state a trader would start from.
    both = close[CSI300].notna() & close[CSI500].notna()
    cold = both & (w.sum(axis=1) == 0)
    w.loc[cold, CSI300] = 0.5
    w.loc[cold, CSI500] = 0.5
    return w


def benchmark_buyhold(index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    w = empty_weights(index, [symbol])
    w[symbol] = 1.0
    return w


def benchmark_balanced(index: pd.DatetimeIndex) -> pd.DataFrame:
    w = empty_weights(index, [CSI300, BOND])
    me = month_end_mask(index)
    w.loc[me, CSI300] = 0.60
    w.loc[me, BOND] = 0.40
    return _ffill_rebalance(w)


def benchmark_equal(index: pd.DatetimeIndex, symbols: Sequence[str], close: pd.DataFrame) -> pd.DataFrame:
    w = empty_weights(index, list(symbols))
    me = month_end_mask(index)
    elig = eligible_mask(close[list(symbols)], 60)
    for dt in index[me]:
        names = [s for s in symbols if bool(elig.loc[dt, s])]
        if not names:
            continue
        share = 1.0 / len(names)
        for symbol in names:
            w.loc[dt, symbol] = share
    return _ffill_rebalance(w)


def close_to_close(close: pd.DataFrame) -> pd.DataFrame:
    """Total-return proxy from adjusted closes. Missing sessions stay missing."""
    return close / close.shift(1) - 1.0


def executed_weights(weights: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map a close-t signal onto the close-to-close return that ends on t+2.

    The signal close is not a fill. The next close is the execution price, so the
    first fully earned bar is the following close-to-close move. Closes are used
    instead of opens because several vendor opens (shared by Yahoo and Tencent)
    sit far from the rest of the bar and are not realistic fills; Sina/Tencent
    closes for 510300 match to the cent.
    """
    cols = list(weights.columns)
    w = weights.reindex(index).reindex(columns=cols).fillna(0.0)
    return w.shift(2).fillna(0.0)


def trade_cost(executed: pd.DataFrame) -> pd.Series:
    delta = executed.diff().abs()
    if len(delta):
        # Calendar starts from cash, so the first non-zero book pays entry.
        delta.iloc[0] = executed.iloc[0].abs()
    cost_vec = pd.Series({c: cost_of(c) for c in executed.columns})
    return (delta.fillna(0.0) * cost_vec).sum(axis=1)


def run_backtest(
    weights: pd.DataFrame,
    asset_ret: pd.DataFrame,
) -> pd.Series:
    """Net close-to-close return after per-side costs. Pre-window signals are kept.

    Positions already on at EVAL_START are not charged a second entry; only
    trades whose execution falls on a return date are costed.
    """
    cols = [c for c in weights.columns if c in asset_ret.columns]
    executed = executed_weights(weights[cols], asset_ret.index)
    gross = (executed * asset_ret[cols].fillna(0.0)).sum(axis=1)
    net = gross - trade_cost(executed)
    net = net.loc[net.index >= EVAL_START]
    return net.rename("ret")


def _equity(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod()


def max_drawdown(ret: pd.Series) -> float:
    eq = _equity(ret)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def cagr_of(ret: pd.Series) -> float:
    if ret.empty:
        return float("nan")
    eq = float(_equity(ret).iloc[-1])
    days = max((ret.index[-1] - ret.index[0]).days, 1)
    years = days / 365.25
    if eq <= 0 or years <= 0:
        return float("nan")
    return eq ** (1.0 / years) - 1.0


def sharpe_of(ret: pd.Series) -> float:
    excess = ret - RF_DAILY
    vol = float(excess.std(ddof=1))
    if not np.isfinite(vol) or vol < 1e-12:
        return float("nan")
    return float(excess.mean() / vol * math.sqrt(TRADING_DAYS))


def sortino_of(ret: pd.Series) -> float:
    excess = ret - RF_DAILY
    downside = np.minimum(excess.to_numpy(), 0.0)
    dd = math.sqrt(float(np.mean(downside ** 2)))
    if dd < 1e-12:
        return float("nan")
    return float(excess.mean() / dd * math.sqrt(TRADING_DAYS))


def profit_factor(ret: pd.Series) -> float:
    gains = float(ret[ret > 0].sum())
    losses = float(-ret[ret < 0].sum())
    if losses < 1e-12:
        return float("nan")
    return gains / losses


def annual_turnover(weights: pd.DataFrame, index: pd.DatetimeIndex) -> float:
    """One-way turnover: 0.5 * sum(|Δw|) annualized. 1.0 ≈ one full book replacement."""
    executed = executed_weights(weights, index)
    delta = executed.diff().abs().sum(axis=1)
    delta = delta.loc[delta.index >= EVAL_START].fillna(0.0)
    return 0.5 * float(delta.mean()) * TRADING_DAYS


def beta_to(ret: pd.Series, bench: pd.Series) -> float:
    both = pd.concat([ret, bench], axis=1, join="inner").dropna()
    if len(both) < 30:
        return float("nan")
    var = float(both.iloc[:, 1].var(ddof=1))
    if var < 1e-18:
        return float("nan")
    return float(both.iloc[:, 0].cov(both.iloc[:, 1]) / var)


def metrics(ret: pd.Series, bench: Optional[pd.Series] = None) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {}
    monthly = (1.0 + ret).resample("ME").prod() - 1.0
    out = {
        "start": str(ret.index[0].date()),
        "end": str(ret.index[-1].date()),
        "days": int(len(ret)),
        "total_return": float(_equity(ret).iloc[-1] - 1.0),
        "cagr": cagr_of(ret),
        "vol": float(ret.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "sharpe": sharpe_of(ret),
        "sortino": sortino_of(ret),
        "max_dd": max_drawdown(ret),
        "calmar": float("nan"),
        "win_rate_daily": float((ret > 0).mean()),
        "win_rate_monthly": float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "profit_factor": profit_factor(ret),
        "worst_day": float(ret.min()),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
        "best_month": float(monthly.max()) if len(monthly) else float("nan"),
    }
    mdd = out["max_dd"]
    out["calmar"] = float(out["cagr"] / abs(mdd)) if mdd and np.isfinite(mdd) and abs(mdd) > 1e-12 else float("nan")
    if bench is not None:
        aligned = pd.concat([ret, bench], axis=1, join="inner").dropna()
        if not aligned.empty:
            excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            out["excess_cagr"] = cagr_of(aligned.iloc[:, 0]) - cagr_of(aligned.iloc[:, 1])
            out["excess_sharpe_t"] = float(
                excess.mean() / excess.std(ddof=1) * math.sqrt(len(excess))
            ) if float(excess.std(ddof=1)) > 1e-12 else float("nan")
            out["beta"] = beta_to(aligned.iloc[:, 0], aligned.iloc[:, 1])
            out["corr_bench"] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    return out


def slice_period(ret: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return ret.loc[(ret.index >= start) & (ret.index <= end)]


def block_bootstrap_sharpe(ret: pd.Series, block: int = 20, reps: int = 400, seed: int = 7) -> dict:
    """Moving-block bootstrap of annualized Sharpe. Dependence-aware, still a model."""
    x = (ret - RF_DAILY).dropna().to_numpy()
    n = len(x)
    if n < block * 4:
        return {"p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    sharpes = np.empty(reps)
    for i in range(reps):
        pieces = [x[s : s + block] for s in starts[i]]
        sample = np.concatenate(pieces)[:n]
        vol = sample.std(ddof=1)
        sharpes[i] = sample.mean() / vol * math.sqrt(TRADING_DAYS) if vol > 1e-12 else np.nan
    return {
        "p05": float(np.nanpercentile(sharpes, 5)),
        "p50": float(np.nanpercentile(sharpes, 50)),
        "p95": float(np.nanpercentile(sharpes, 95)),
    }


def pnl_attribution(weights: pd.DataFrame, oo_ret: pd.DataFrame, ret_index: pd.DatetimeIndex) -> dict:
    cols = [c for c in weights.columns if c in oo_ret.columns]
    executed = executed_weights(weights[cols], oo_ret.index)
    executed = executed.loc[ret_index]
    contrib = executed * oo_ret[cols].reindex(ret_index).fillna(0.0)
    total = contrib.sum().sum()
    shares = {}
    if abs(total) > 1e-9:
        for symbol in cols:
            shares[NAMES.get(symbol, symbol)] = float(contrib[symbol].sum() / total)
    held = (executed.abs() > 0.05).mean().sort_values(ascending=False)
    top_asset = None
    top_share = 0.0
    if shares:
        top_asset = max(shares, key=lambda k: abs(shares[k]))
        top_share = shares[top_asset]
    return {
        "pnl_share": shares,
        "time_in_market_gt5pct": {NAMES.get(s, s): float(v) for s, v in held.items()},
        "top_asset": top_asset,
        "top_abs_pnl_share": float(top_share),
    }


def regime_table(ret: pd.Series, close: pd.DataFrame) -> dict:
    trend = close[CSI300] / close[CSI300].shift(120) - 1.0
    label = pd.Series("sideways", index=close.index)
    label = label.where(~(trend > 0.08), "bull")
    label = label.where(~(trend < -0.08), "bear")
    label = label.reindex(ret.index)
    out = {}
    for name, mask in (
        ("bull", label == "bull"),
        ("bear", label == "bear"),
        ("sideways", label == "sideways"),
    ):
        sub = ret.loc[mask.fillna(False)]
        out[name] = {
            "days": int(mask.fillna(False).sum()),
            "cagr": cagr_of(sub) if len(sub) > 30 else float("nan"),
            "sharpe": sharpe_of(sub) if len(sub) > 30 else float("nan"),
            "total_return": float(_equity(sub).iloc[-1] - 1.0) if len(sub) else float("nan"),
        }
    return out


def yearly_returns(ret: pd.Series) -> dict:
    years = {}
    for year, sub in ret.groupby(ret.index.year):
        years[str(int(year))] = float(_equity(sub).iloc[-1] - 1.0)
    return years


def double_cost_returns(weights: pd.DataFrame, oo_ret: pd.DataFrame) -> pd.Series:
    """Reprice with 2x per-side costs. Base result already subtracts 1x."""
    net = run_backtest(weights, oo_ret)
    cols = [c for c in weights.columns if c in oo_ret.columns]
    executed = executed_weights(weights[cols], oo_ret.index)
    cost = trade_cost(executed).reindex(net.index).fillna(0.0)
    return net - cost


def quality_flags(full: dict, is_m: dict, oos_m: dict, bench_oos: dict, attr: dict, stress: dict) -> dict:
    """Numerical bar from the skill, plus research-grade caveats. Not a claim of edge."""
    oos_sharpe = oos_m.get("sharpe", float("nan"))
    oos_mdd = oos_m.get("max_dd", float("nan"))
    oos_cagr = oos_m.get("cagr", float("nan"))
    bench_sharpe = bench_oos.get("sharpe", float("nan"))
    bench_cagr = bench_oos.get("cagr", float("nan"))
    clears_numeric = bool(
        np.isfinite(oos_sharpe)
        and oos_sharpe > 1.0
        and np.isfinite(oos_mdd)
        and oos_mdd > -0.30
        and np.isfinite(oos_cagr)
        and np.isfinite(bench_cagr)
        and oos_cagr > bench_cagr
        and np.isfinite(bench_sharpe)
        and oos_sharpe > bench_sharpe
        and full.get("sharpe", -1) > 1.0
        and full.get("max_dd", -1) > -0.30
        and is_m.get("sharpe", -1) > 0.5
    )
    is_excess = is_m.get("excess_cagr", float("nan"))
    oos_excess = oos_m.get("excess_cagr", float("nan"))
    same_sign = bool(
        np.isfinite(is_excess) and np.isfinite(oos_excess) and is_excess > 0.01 and oos_excess > 0.01
    )
    oos_t = oos_m.get("excess_sharpe_t", float("nan"))
    detectable = bool(np.isfinite(oos_t) and oos_t > 2.0)
    single_asset = bool(attr.get("top_abs_pnl_share", 0) > 0.65)
    survives_2x = bool(stress.get("oos_sharpe", -9) > bench_sharpe and stress.get("oos_cagr", -9) > bench_cagr)
    research_grade = bool(
        clears_numeric and same_sign and detectable and (not single_asset) and survives_2x
    )
    return {
        "clears_numeric_bar": clears_numeric,
        "is_and_oos_excess_cagr_gt_1pp": same_sign,
        "oos_excess_t_gt_2": detectable,
        "single_asset_pnl": single_asset,
        "survives_2x_cost": survives_2x,
        "research_grade_edge": research_grade,
    }


def rank_score(oos_m: dict, is_m: dict, bench_oos: dict) -> float:
    """Ranking aid only. Higher is better on risk-adjusted OOS evidence, not a p-value."""
    def _f(d, k):
        v = d.get(k, float("nan"))
        return float(v) if v is not None and np.isfinite(v) else -1.0

    excess = _f(oos_m, "cagr") - _f(bench_oos, "cagr")
    mdd_pen = max(-0.5, _f(oos_m, "max_dd"))  # negative
    return (
        0.40 * _f(oos_m, "sharpe")
        + 0.20 * _f(is_m, "sharpe")
        + 0.20 * max(-0.5, min(0.5, excess)) / 0.10
        + 0.20 * (mdd_pen / 0.30)
    )


def market_snapshot(close: pd.DataFrame, opens: pd.DataFrame) -> dict:
    asof = close.index.max()
    rows = {}
    windows = {"20d": 20, "60d": 60, "252d": 252, "756d": 756}
    for symbol, name in NAMES.items():
        if symbol not in close.columns:
            continue
        px = close[symbol].dropna()
        if px.empty:
            continue
        last = float(px.iloc[-1])
        item = {"last": last, "last_date": str(px.index[-1].date())}
        for label, n in windows.items():
            if len(px) > n:
                item[label] = float(px.iloc[-1] / px.iloc[-1 - n] - 1.0)
        sma200 = px.rolling(200).mean()
        if sma200.notna().iloc[-1]:
            item["dist_sma200"] = float(px.iloc[-1] / sma200.iloc[-1] - 1.0)
        item["vol60"] = float(px.pct_change().tail(60).std(ddof=1) * math.sqrt(TRADING_DAYS))
        peak = px.tail(252).max() if len(px) >= 20 else px.max()
        item["dd_252"] = float(px.iloc[-1] / peak - 1.0)
        rows[name] = item
    # YTD using first session of 2026
    ytd = {}
    y0 = close.index[close.index >= pd.Timestamp("2026-01-01")]
    if len(y0):
        base_dt = close.index[close.index < y0[0]][-1]
        for symbol, name in NAMES.items():
            if pd.notna(close.loc[base_dt, symbol]) and pd.notna(close.loc[asof, symbol]):
                ytd[name] = float(close.loc[asof, symbol] / close.loc[base_dt, symbol] - 1.0)
    return {"asof": str(asof.date()), "trailing": rows, "ytd_2026": ytd}


def eastmoney_last_close(secid: str) -> dict:
    """Eastmoney daily bar. Connection failures are data gaps, not fills."""
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        f"secid={secid}&klt=101&fqt=1&lmt=5&end=20261006"
        "&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — data gap is reported, not fatal
        return {"error": str(exc)}
    klines = (payload.get("data") or {}).get("klines") or []
    if not klines:
        return {"error": "empty klines"}
    # date, open, close, high, low, volume, amount
    last = klines[-1].split(",")
    return {"date": last[0], "close": float(last[2])}


def tencent_recent_bars(symbol: str = "sh510300", count: int = 5) -> dict:
    """Tencent qfq daily bars. Used only as a close-price cross-check."""
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
        f"{symbol},day,,,{count},qfq"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    node = (payload.get("data") or {}).get(symbol) or {}
    bars = node.get("qfqday") or node.get("day") or []
    parsed = []
    for bar in bars:
        # date, open, close, high, low, volume
        parsed.append({"date": bar[0], "open": float(bar[1]), "close": float(bar[2])})
    return {"bars": parsed}


def cross_check(frames: Dict[str, pd.DataFrame]) -> dict:
    checks = {}
    mapping = {"510300.SS": "1.510300", "510500.SS": "1.510500", "518880.SS": "1.518880"}
    for symbol, secid in mapping.items():
        vendor = eastmoney_last_close(secid)
        yf_px = frames[symbol]["Close"].dropna()
        checks[symbol] = {
            "yahoo_last_date": str(yf_px.index[-1].date()),
            "yahoo_last_close": float(yf_px.iloc[-1]),
            "eastmoney": vendor,
        }
        if isinstance(vendor, dict) and "close" in vendor:
            # Compare on the same date if Yahoo has it; adjustment methods can differ.
            dt = pd.Timestamp(vendor["date"])
            if dt in frames[symbol].index and pd.notna(frames[symbol].loc[dt, "Close"]):
                y = float(frames[symbol].loc[dt, "Close"])
                e = float(vendor["close"])
                checks[symbol]["same_day_yahoo"] = y
                checks[symbol]["rel_diff"] = (y - e) / e if e else None
    tencent = tencent_recent_bars("sh510300", 5)
    checks["tencent_510300"] = tencent
    bars = tencent.get("bars") or []
    if bars:
        last = bars[-1]
        yf_px = frames["510300.SS"]["Close"].dropna()
        dt = pd.Timestamp(last["date"])
        if dt in frames["510300.SS"].index and pd.notna(frames["510300.SS"].loc[dt, "Close"]):
            y = float(frames["510300.SS"].loc[dt, "Close"])
            checks["tencent_510300"]["same_day_yahoo_close"] = y
            checks["tencent_510300"]["rel_diff"] = (y - last["close"]) / last["close"] if last["close"] else None
            checks["tencent_510300"]["yahoo_last_date"] = str(yf_px.index[-1].date())
    return checks


def macro_notes() -> dict:
    """Best-effort Yahoo prints for context series. Failures are recorded, not filled."""
    import yfinance as yf

    out = {}
    for symbol in ["000001.SS", "^HSI", "^GSPC", "USDCNY=X", "^TNX", "GC=F"]:
        try:
            raw = yf.download(symbol, start="2025-01-01", end=DOWNLOAD_END, auto_adjust=True, progress=False, threads=False)
            raw = _flatten(raw)
            if raw.empty:
                out[symbol] = {"error": "empty"}
                continue
            px = raw["Close"].dropna()
            px.index = pd.to_datetime(px.index).tz_localize(None)
            out[symbol] = {
                "last_date": str(px.index[-1].date()),
                "last": float(px.iloc[-1]),
                "ret_60d": float(px.iloc[-1] / px.iloc[-61] - 1.0) if len(px) > 61 else None,
                "ytd": None,
            }
            pre = px[px.index < pd.Timestamp("2026-01-01")]
            if len(pre):
                out[symbol]["ytd"] = float(px.iloc[-1] / float(pre.iloc[-1]) - 1.0)
        except Exception as exc:  # noqa: BLE001
            out[symbol] = {"error": str(exc)}
    return out


def spike_report(oo: pd.DataFrame) -> dict:
    flags = {}
    for symbol in [CSI300, CSI500, "159915.SZ", "518880.SS", BOND]:
        r = oo[symbol].dropna()
        big = r[r.abs() > 0.11]
        flags[symbol] = {
            "n_abs_gt_11pct": int(len(big)),
            "max_abs": float(r.abs().max()) if len(r) else None,
            "dates": [str(i.date()) for i in big.index[:8]],
        }
    return flags


def adv_table(close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    dollar = (close * volume).tail(60)
    out = {}
    for symbol in UNIVERSE:
        series = dollar[symbol].dropna()
        if series.empty:
            continue
        out[NAMES.get(symbol, symbol)] = float(series.mean())
    return out


def assert_engine_sane(rets: pd.DataFrame, close: pd.DataFrame) -> None:
    """Buy-and-hold inside the window matches close-to-close; entry was pre-window.

    A flip dated on close t must show up on the close-to-close return ending t+2.
    Rewriting the future must not change past signals.
    """
    w = benchmark_buyhold(close.index, CSI300)
    net = run_backtest(w, rets)
    expected = rets[CSI300].reindex(net.index).fillna(0.0)
    if not np.allclose(net.to_numpy(), expected.to_numpy(), atol=1e-12):
        gap = (net - expected).abs().max()
        raise AssertionError(f"buyhold backtest diverges from close-to-close, max abs {gap}")

    idx = rets.index
    impulse = pd.DataFrame(0.0, index=idx, columns=[CSI300])
    loc = idx.get_loc(EVAL_START) + 50
    dt = idx[loc]
    impulse.loc[dt:, CSI300] = 1.0
    lagged = run_backtest(impulse, rets)
    prev_day = idx[loc + 1]
    earn_day = idx[loc + 2]
    if abs(float(lagged.loc[prev_day])) > 1e-12:
        raise AssertionError("execution lag is shorter than two closes")
    expected_earn = float(rets.loc[earn_day, CSI300]) - cost_of(CSI300)
    if abs(float(lagged.loc[earn_day]) - expected_earn) > 1e-12:
        raise AssertionError(
            f"lagged fill mismatch: got {float(lagged.loc[earn_day])} expected {expected_earn}"
        )

    perturbed = close.copy()
    perturbed.iloc[-30:] = perturbed.iloc[-30:] * 3
    a = strategy_trend(close)
    b = strategy_trend(perturbed)
    if not a.iloc[:-40].equals(b.iloc[:-40]):
        raise AssertionError("trend signal looks ahead")
    a2 = strategy_xs_momentum(close)
    b2 = strategy_xs_momentum(perturbed)
    if not np.allclose(a2.iloc[:-40].fillna(0), b2.iloc[:-40].fillna(0), atol=1e-12):
        raise AssertionError("momentum signal looks ahead")


def evaluate_one(
    name: str,
    thesis_id: str,
    weights: pd.DataFrame,
    oo: pd.DataFrame,
    close: pd.DataFrame,
    bench_ret: pd.Series,
) -> dict:
    ret = run_backtest(weights, oo)
    full = metrics(ret, bench_ret)
    is_ret = slice_period(ret, EVAL_START, IS_END)
    oos_ret = slice_period(ret, OOS_START, SAMPLE_END)
    is_b = slice_period(bench_ret, EVAL_START, IS_END)
    oos_b = slice_period(bench_ret, OOS_START, SAMPLE_END)
    is_m = metrics(is_ret, is_b)
    oos_m = metrics(oos_ret, oos_b)
    bench_oos = metrics(oos_b)
    bench_full = metrics(bench_ret.reindex(ret.index).dropna(), None)
    attr = pnl_attribution(weights, oo, ret.index)
    stress_ret = double_cost_returns(weights, oo)
    stress = {
        "full_sharpe": sharpe_of(stress_ret),
        "full_cagr": cagr_of(stress_ret),
        "oos_sharpe": sharpe_of(slice_period(stress_ret, OOS_START, SAMPLE_END)),
        "oos_cagr": cagr_of(slice_period(stress_ret, OOS_START, SAMPLE_END)),
        "oos_max_dd": max_drawdown(slice_period(stress_ret, OOS_START, SAMPLE_END)),
    }
    flags = quality_flags(full, is_m, oos_m, bench_oos, attr, stress)
    boot = block_bootstrap_sharpe(oos_ret)
    # Sit out the Sep-Oct 2024 policy-gap window; tests whether OOS excess is that episode.
    stim_lo, stim_hi = pd.Timestamp("2024-09-24"), pd.Timestamp("2024-10-18")
    ret_ex = ret.copy()
    bench_ex = bench_ret.reindex(ret.index).copy()
    stim_mask = (ret_ex.index >= stim_lo) & (ret_ex.index <= stim_hi)
    ret_ex.loc[stim_mask] = 0.0
    bench_ex.loc[stim_mask] = 0.0
    ex_stim = metrics(slice_period(ret_ex, OOS_START, SAMPLE_END), slice_period(bench_ex, OOS_START, SAMPLE_END))
    return {
        "id": thesis_id,
        "name": name,
        "full": full,
        "is": is_m,
        "oos": oos_m,
        "bench_oos": bench_oos,
        "bench_full_on_strategy_dates": bench_full,
        "turnover_one_way": annual_turnover(weights, oo.index),
        "attribution": attr,
        "regimes": regime_table(ret, close),
        "yearly": yearly_returns(ret),
        "stress_2x": stress,
        "flags": flags,
        "oos_sharpe_bootstrap": boot,
        "oos_ex_stimulus_20240924_1018": {
            "sharpe": ex_stim.get("sharpe"),
            "cagr": ex_stim.get("cagr"),
            "max_dd": ex_stim.get("max_dd"),
            "excess_cagr": ex_stim.get("excess_cagr"),
            "days_zeroed": int(stim_mask.sum()),
        },
        "rank_score": rank_score(oos_m, is_m, bench_oos),
        "returns": ret,
    }


def sensitivity(close: pd.DataFrame, oo: pd.DataFrame, benches: dict) -> dict:
    """Small pre-listed grid. Primary spec is NOT chosen from this grid."""
    specs = {
        "S1": (
            benches["bh300"],
            [
                ("sma_150", lambda: strategy_trend(close, 150)),
                ("sma_200_primary", lambda: strategy_trend(close, 200)),
                ("sma_250", lambda: strategy_trend(close, 250)),
            ],
        ),
        "S2": (
            benches["ew_equity"],
            [
                ("lb189_top3", lambda: strategy_xs_momentum(close, 189, 21, 3)),
                ("lb252_top2", lambda: strategy_xs_momentum(close, 252, 21, 2)),
                ("lb252_top3_primary", lambda: strategy_xs_momentum(close, 252, 21, 3)),
                ("lb252_top4", lambda: strategy_xs_momentum(close, 252, 21, 4)),
                ("lb126_top3", lambda: strategy_xs_momentum(close, 126, 21, 3)),
            ],
        ),
        "S3": (
            benches["bh300"],
            [
                ("hold3_thr2", lambda: strategy_reversal(close, 3, 200, -0.02)),
                ("hold5_thr3_primary", lambda: strategy_reversal(close, 5, 200, -0.03)),
                ("hold10_thr5", lambda: strategy_reversal(close, 10, 200, -0.05)),
            ],
        ),
        "S4": (
            benches["ew_multi"],
            [
                ("lb126", lambda: strategy_dual_momentum(close, 126)),
                ("lb252_primary", lambda: strategy_dual_momentum(close, 252)),
            ],
        ),
        "S5": (
            benches["ew_rp"],
            [
                ("vol6", lambda: strategy_risk_parity(close, 0.06)),
                ("vol8_primary", lambda: strategy_risk_parity(close, 0.08)),
                ("vol10", lambda: strategy_risk_parity(close, 0.10)),
            ],
        ),
        "S6": (
            benches["ew_equity"],
            [
                ("sma100_top3", lambda: strategy_low_vol(close, 100, 60, 3)),
                ("sma120_top3_primary", lambda: strategy_low_vol(close, 120, 60, 3)),
                ("sma150_top2", lambda: strategy_low_vol(close, 150, 60, 2)),
            ],
        ),
        "S7": (
            benches["ew_donchian"],
            [
                ("e100_x40", lambda: strategy_donchian(close, 100, 40)),
                ("e120_x60_primary", lambda: strategy_donchian(close, 120, 60)),
                ("e150_x80", lambda: strategy_donchian(close, 150, 80)),
            ],
        ),
        "S8": (
            benches["ew_style"],
            [
                ("z1.0", lambda: strategy_pair_rv(close, 60, 1.0, 0.25)),
                ("z1.5_primary", lambda: strategy_pair_rv(close, 60, 1.5, 0.5)),
                ("z2.0", lambda: strategy_pair_rv(close, 60, 2.0, 0.75)),
            ],
        ),
    }
    out = {}
    for sid, (bench, grid) in specs.items():
        rows = []
        is_best = None
        is_best_sharpe = -1e9
        for label, factory in grid:
            ret = run_backtest(factory(), oo)
            is_m = metrics(slice_period(ret, EVAL_START, IS_END), slice_period(bench, EVAL_START, IS_END))
            oos_m = metrics(slice_period(ret, OOS_START, SAMPLE_END), slice_period(bench, OOS_START, SAMPLE_END))
            row = {
                "param": label,
                "primary": label.endswith("primary"),
                "is_sharpe": is_m.get("sharpe"),
                "oos_sharpe": oos_m.get("sharpe"),
                "oos_cagr": oos_m.get("cagr"),
                "oos_max_dd": oos_m.get("max_dd"),
                "oos_excess_cagr": oos_m.get("excess_cagr"),
            }
            rows.append(row)
            if is_m.get("sharpe", -1e9) > is_best_sharpe:
                is_best_sharpe = is_m["sharpe"]
                is_best = row
        oos_sharpes = [r["oos_sharpe"] for r in rows if r["oos_sharpe"] is not None and np.isfinite(r["oos_sharpe"])]
        out[sid] = {
            "grid": rows,
            "is_selected": is_best,
            "oos_sharpe_min": float(min(oos_sharpes)) if oos_sharpes else None,
            "oos_sharpe_max": float(max(oos_sharpes)) if oos_sharpes else None,
        }
    return out


def save_chart(curves: Dict[str, pd.Series], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for name, ret in curves.items():
        eq = _equity(ret)
        ax.plot(eq.index, eq.values, label=name, linewidth=1.3)
    ax.set_yscale("log")
    ax.set_title("Net equity (log), costs included, 2018-01 to 2026-09")
    ax.set_ylabel("Growth of 1 CNY")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


def main() -> None:
    frames = download_prices()
    calendar = master_calendar(frames)
    opens, close, volume = build_panels(frames, calendar)
    rets = close_to_close(close)
    assert_engine_sane(rets, close)

    strategies: List[Tuple[str, str, pd.DataFrame, str]] = [
        ("S1", "趋势-债券切换", strategy_trend(close, 200), "bh300"),
        ("S2", "截面动量12-1", strategy_xs_momentum(close), "ew_equity"),
        ("S3", "趋势内短期反转", strategy_reversal(close), "bh300"),
        ("S4", "双动量", strategy_dual_momentum(close), "ew_multi"),
        ("S5", "逆波动风险平价", strategy_risk_parity(close), "ew_rp"),
        ("S6", "低波动", strategy_low_vol(close), "ew_equity"),
        ("S7", "唐奇安突破", strategy_donchian(close), "ew_donchian"),
        ("S8", "300/500相对价值", strategy_pair_rv(close), "ew_style"),
    ]

    benches_w = {
        "bh300": benchmark_buyhold(close.index, CSI300),
        "bh_bond": benchmark_buyhold(close.index, BOND),
        "bal6040": benchmark_balanced(close.index),
        "ew_equity": benchmark_equal(close.index, EQUITY_CORE, close),
        "ew_multi": benchmark_equal(close.index, MULTI, close),
        "ew_rp": benchmark_equal(close.index, RP_RISKY + [BOND], close),
        "ew_donchian": benchmark_equal(close.index, DONCHIAN_ASSETS, close),
        "ew_style": benchmark_equal(close.index, [CSI300, CSI500], close),
    }
    bench_rets = {k: run_backtest(v, rets) for k, v in benches_w.items()}

    results = []
    curves = {"BH_CSI300": bench_rets["bh300"], "BAL_60_40": bench_rets["bal6040"]}
    for sid, name, weights, bench_key in strategies:
        ev = evaluate_one(name, sid, weights, rets, close, bench_rets[bench_key])
        ev["benchmark_id"] = bench_key
        curves[sid] = ev["returns"]
        ev.pop("returns")
        results.append(ev)

    results.sort(key=lambda r: r["rank_score"], reverse=True)
    sens = sensitivity(close, rets, bench_rets)

    chart_path = REPORTS / "trading_strategy_research_20261005_equity.png"
    save_chart(curves, chart_path)

    # Benchmark yearly for the report
    bench_yearly = {k: yearly_returns(v) for k, v in bench_rets.items()}
    bench_metrics = {}
    for k, v in bench_rets.items():
        bench_metrics[k] = {
            "full": metrics(v),
            "is": metrics(slice_period(v, EVAL_START, IS_END)),
            "oos": metrics(slice_period(v, OOS_START, SAMPLE_END)),
            "yearly": bench_yearly[k],
        }

    payload = {
        "eval_start": str(EVAL_START.date()),
        "is_end": str(IS_END.date()),
        "oos_start": str(OOS_START.date()),
        "sample_end": str(min(SAMPLE_END, calendar.max()).date()),
        "calendar_last": str(calendar.max().date()),
        "n_sessions": int(len(calendar)),
        "rf_annual": RF_ANNUAL,
        "costs": {"liquid": COST_LIQUID, "thin": COST_THIN, "bond": COST_BOND, "thin_names": sorted(THIN)},
        "market": market_snapshot(close, opens),
        "cross_check": cross_check(frames),
        "macro": macro_notes(),
        "spikes": spike_report(rets),
        "adv60_cny": adv_table(close, volume),
        "benchmarks": bench_metrics,
        "strategies": results,
        "sensitivity": sens,
    }
    out_path = OUT / "metrics_20261005.json"
    out_path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"chart {chart_path}")
    for row in results:
        f = row["flags"]
        print(
            f"{row['id']} {row['name']} score={row['rank_score']:.2f} "
            f"OOS_sh={row['oos']['sharpe']:.2f} OOS_cagr={row['oos']['cagr']:.2%} "
            f"OOS_mdd={row['oos']['max_dd']:.2%} excess={row['oos'].get('excess_cagr')} "
            f"numeric={f['clears_numeric_bar']} grade={f['research_grade_edge']}"
        )


if __name__ == "__main__":
    main()
