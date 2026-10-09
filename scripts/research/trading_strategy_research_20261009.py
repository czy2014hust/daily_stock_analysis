#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A-share ETF strategy research for the 2026-10-09 daily job.

Sample end: last completed A-share session, 2026-10-08.
2026-10-09 is left incomplete (research clock is still in the cash session)
and is used only as the terminal open that closes the 10-08 -> 10-09-open
holding interval.

Tradable ETF prices: Tencent qfq daily bars (split- and dividend-adjusted).
Yahoo Finance was audited and rejected for this universe: 512480.SS shows
impossible one-day moves near -50% in June/July 2026 that are not in the
Tencent qfq series (a broken split scale). Macro series still come from Yahoo.
2026-10-09 is an incomplete session. Its open is kept only as the terminal
mark of the last holding interval; its close is not used in any signal.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Pre-registered design. Parameter grids are sensitivity checks, not a search
# used to crown a winner. PRIMARY specs below are fixed before looking at PnL.
# ---------------------------------------------------------------------------

ASOF_COMPLETE = pd.Timestamp("2026-10-08")
TERMINAL_OPEN = pd.Timestamp("2026-10-09")
EVAL_START = pd.Timestamp("2018-01-01")
IS_END = pd.Timestamp("2022-12-31")
OOS_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")

# ETF one-way cost: 1.5 bps commission + 5 bps slippage.
# Treasury ETF is charged wider (8 bps) because its share turnover is thinner.
COST_EQUITY = 0.00065
COST_BOND = 0.00080
AUM_CNY = 10_000_000.0
ADV_MIN_CNY = 30_000_000.0
MIN_NAMES = 4

HS300 = "510300.SS"
CSI500 = "510500.SS"
SSE50 = "510050.SS"
CHINEXT = "159915.SZ"
CSI1000 = "512100.SS"
DIVIDEND = "510880.SS"
GOLD = "518880.SS"
BOND = "511010.SS"
NASDAQ = "513100.SS"
SPX = "513500.SS"
HANGSENG = "159920.SZ"
STAR = "588000.SS"

SECTORS: Dict[str, str] = {
    "512880.SS": "证券",
    "159928.SZ": "消费",
    "512010.SS": "医药",
    "512660.SS": "军工",
    "512800.SS": "银行",
    "512400.SS": "有色",
    "512200.SS": "房地产",
    "512980.SS": "传媒",
    "512690.SS": "白酒",
    "512480.SS": "半导体",
    "515220.SS": "煤炭",
    "515030.SS": "新能源车",
    "513050.SS": "中概互联",
}

NAMES: Dict[str, str] = {
    HS300: "沪深300ETF",
    CSI500: "中证500ETF",
    SSE50: "上证50ETF",
    CHINEXT: "创业板ETF",
    CSI1000: "中证1000ETF",
    DIVIDEND: "红利ETF",
    GOLD: "黄金ETF",
    BOND: "国债ETF",
    NASDAQ: "纳指ETF",
    SPX: "标普500ETF",
    HANGSENG: "恒生ETF",
    STAR: "科创50ETF",
    **SECTORS,
}

MACRO_SYMBOLS = ["000001.SS", "399001.SZ", "^HSI", "^GSPC", "^VIX", "^TNX", "USDCNY=X", "GC=F", "CL=F"]


TENCENT_CACHE = "/tmp/tsr_tencent_20261009.pkl"


def _tencent_symbol(symbol: str) -> str:
    code, exch = symbol.split(".")
    if exch == "SS":
        return "sh" + code
    if exch == "SZ":
        return "sz" + code
    raise ValueError(symbol)


def _fetch_tencent_chunk(symbol: str, start: str, end: str) -> pd.DataFrame:
    """One year of qfq bars. Tencent truncates long ranges, so callers stay inside a year."""
    code = _tencent_symbol(symbol)
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
        f"{code},day,{start},{end},900,qfq"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode())
            node = payload.get("data", {}).get(code) if isinstance(payload.get("data"), dict) else None
            if not node:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            rows = node.get("qfqday") or node.get("day") or []
            parsed = []
            for row in rows:
                parsed.append(
                    {
                        "Date": pd.Timestamp(row[0]),
                        "Open": float(row[1]),
                        "Close": float(row[2]),
                        "High": float(row[3]),
                        "Low": float(row[4]),
                        # Tencent volume is in lots (手). Research dollar volume uses shares.
                        "Volume": float(row[5]) * 100.0,
                    }
                )
            frame = pd.DataFrame(parsed)
            if frame.empty:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            return frame.set_index("Date").sort_index()
        except Exception as exc:  # noqa: BLE001 - network retry is explicit
            last_err = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"tencent fetch failed for {symbol} {start}: {last_err}")


def _load_tencent_symbol(symbol: str) -> pd.DataFrame:
    frames = []
    for year in range(2016, 2027):
        end = "2026-10-09" if year == 2026 else f"{year}-12-31"
        chunk = _fetch_tencent_chunk(symbol, f"{year}-01-01", end)
        if not chunk.empty:
            frames.append(chunk)
        time.sleep(0.05)
    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _download_yahoo(symbols: Sequence[str], start: str = "2016-01-01") -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        list(symbols),
        start=start,
        end="2026-10-10",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = raw[symbol].dropna(how="all").copy()
        if frame.empty:
            continue
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        out[symbol] = frame
    return out


def load_ohlc(notes: List[str]) -> Dict[str, pd.DataFrame]:
    """Return qfq OHLC panels keyed by field: Open/High/Low/Close/Volume."""
    import pickle

    symbols = list(NAMES)
    cached: Dict[str, pd.DataFrame] = {}
    try:
        with open(TENCENT_CACHE, "rb") as handle:
            cached = pickle.load(handle)
    except FileNotFoundError:
        cached = {}
    frames: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = cached.get(symbol)
        if frame is None or frame.empty or ASOF_COMPLETE not in frame.index:
            frame = _load_tencent_symbol(symbol)
            cached[symbol] = frame
            with open(TENCENT_CACHE, "wb") as handle:
                pickle.dump(cached, handle)
        frames[symbol] = frame

    cleaned: Dict[str, pd.DataFrame] = {}
    jump_notes = []
    missing_terminal = []
    missing_complete = []
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            missing_complete.append(symbol)
            continue
        work = frame.sort_index().copy()
        work = work.loc[(work.index <= ASOF_COMPLETE) | (work.index == TERMINAL_OPEN)]
        if ASOF_COMPLETE not in work.index:
            missing_complete.append(symbol)
        if TERMINAL_OPEN in work.index:
            # 10-09 close is still printing. Keep the realized open only.
            work.loc[TERMINAL_OPEN, ["High", "Low", "Close", "Volume"]] = np.nan
        else:
            missing_terminal.append(symbol)
        completed = work.loc[work.index <= ASOF_COMPLETE, "Close"].dropna()
        jumps = completed.pct_change().abs()
        worst = float(jumps.max()) if len(jumps) else np.nan
        # 40% is above the real 2020-02-03 and 2024-09/10 gap days (~20-25%).
        # It still catches a broken split scale such as the Yahoo 512480 print.
        if pd.notna(worst) and worst > 0.40:
            when = str(jumps.idxmax().date())
            jump_notes.append(f"{symbol} {NAMES.get(symbol, symbol)} 最大单日 {worst:.1%} @ {when}，剔除")
            continue
        if pd.notna(worst) and worst > 0.12:
            when = str(jumps.idxmax().date())
            jump_notes.append(f"{symbol} 最大单日 {worst:.1%} @ {when}（保留，ETF 在极端日可以超过 10%）")
        cleaned[symbol] = work

    notes.append(
        "交易价格使用腾讯日线前复权（qfq），按自然年分段拉取以避免接口截断。"
        "Yahoo 复权在 512480.SS 2026-06/07 出现约 -50% 的假跳空，已弃用。"
    )
    notes.append(
        "信号使用截至 2026-10-08 的完整 K 线。2026-10-09 仅保留开盘价，作为最后一段持仓的终止成交价；当日收盘不进信号。"
    )
    if missing_complete:
        notes.append("缺少 2026-10-08 完整 K 线：" + ", ".join(missing_complete))
    if missing_terminal:
        notes.append("缺少 2026-10-09 开盘价：" + ", ".join(missing_terminal))
    if jump_notes:
        notes.append("单日涨跌审计：" + "；".join(jump_notes))
    else:
        notes.append("单日涨跌审计：全部标的最大收盘涨跌绝对值不超过 12%。")

    panels = {
        field: pd.DataFrame({sym: frame[field] for sym, frame in cleaned.items()}).sort_index()
        for field in ("Open", "High", "Low", "Close", "Volume")
    }
    return panels


def _ann_factor(index: pd.DatetimeIndex) -> float:
    counts = pd.Series(1, index=index).groupby(index.year).sum()
    counts = counts[counts.index < 2026]  # 2026 is partial
    if counts.empty:
        return 244.0
    return float(counts.median())


def performance(daily: pd.Series, ann_factor: float) -> dict:
    rets = daily.dropna()
    if len(rets) < 20:
        return {"n": int(len(rets))}
    wealth = (1.0 + rets).cumprod()
    total = float(wealth.iloc[-1] - 1.0)
    years = len(rets) / ann_factor
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(rets.std(ddof=1) * np.sqrt(ann_factor))
    sharpe = float(rets.mean() * ann_factor / vol) if vol > 0 else np.nan
    downside = rets[rets < 0]
    down_dev = float(downside.std(ddof=1) * np.sqrt(ann_factor)) if len(downside) > 1 else np.nan
    sortino = float(rets.mean() * ann_factor / down_dev) if down_dev and down_dev > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    max_dd = float(drawdown.min())
    monthly = (1.0 + rets).groupby(rets.index.to_period("M")).prod() - 1.0
    win_rate = float((monthly > 0).mean()) if len(monthly) else np.nan
    gains = float(monthly[monthly > 0].sum())
    losses = float(-monthly[monthly < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else np.nan
    var5 = float(rets.quantile(0.05))
    cvar5 = float(rets[rets <= rets.quantile(0.05)].mean())
    return {
        "n": int(len(rets)),
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "monthly_win_rate": win_rate,
        "profit_factor": profit_factor,
        "var5_daily": var5,
        "cvar5_daily": cvar5,
        "best_month": float(monthly.max()) if len(monthly) else np.nan,
        "worst_month": float(monthly.min()) if len(monthly) else np.nan,
    }


def slice_period(daily: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return daily[(daily.index >= start) & (daily.index <= end)]


def simulate(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    ann_factor: float,
    cost_scale: float = 1.0,
    extra_delay: int = 0,
) -> dict:
    """Trade at the open. ``weights`` are targets held from that open onward.

    Return on date t is open[t+1] / open[t] - 1, net of one-way costs on |Δw|.
    The last index row may be a terminal open with no close; its oo return is NaN
    and is dropped.
    """
    w = weights.sort_index().copy()
    if extra_delay:
        w = w.shift(extra_delay)
    cols = [c for c in w.columns if c in opens.columns]
    w = w[cols].astype(float)
    # Charge the evaluation-window entry against a flat book so pre-2018
    # positions do not skip the initial ticket.
    w.loc[w.index < EVAL_START, :] = 0.0
    oo = opens[cols].shift(-1) / opens[cols] - 1.0
    # A zero row before the first live weight so the entry pays cost.
    prev = w.shift(1).fillna(0.0)
    delta = w - prev
    cost_rate = pd.Series({c: (COST_BOND if c == BOND else COST_EQUITY) * cost_scale for c in cols})
    cost = delta.abs().mul(cost_rate, axis=1).sum(axis=1)
    gross = (w * oo).sum(axis=1, min_count=1)
    # If a held name has no open-to-open return, do not invent one.
    missing = (w.abs() > 1e-8) & oo.isna()
    gross = gross.where(~missing.any(axis=1))
    net = (gross - cost).dropna()
    # Evaluation starts at the first non-zero book inside EVAL_START.
    live = w.abs().sum(axis=1)
    first_live = live[live > 0.1].index.min()
    if pd.isna(first_live):
        return {"daily": net, "metrics": {"n": 0}}
    net = net[net.index >= max(first_live, EVAL_START)]
    gross_turnover = delta.abs().sum(axis=1)
    gross_turnover = gross_turnover.reindex(net.index).fillna(0.0)
    years = max(len(net) / ann_factor, 1e-9)
    equity_cols = [c for c in cols if c != BOND]
    equity_w = w[equity_cols].sum(axis=1).reindex(net.index)
    stats = performance(net, ann_factor)
    stats["gross_turnover_annual"] = float(gross_turnover.sum() / years)
    stats["avg_equity_weight"] = float(equity_w.mean())
    stats["time_in_market"] = float((equity_w > 0.2).mean())
    stats["cost_drag_annual"] = float((cost.reindex(net.index).fillna(0.0)).sum() / years)
    # Capacity at the stated AUM: trade notional vs 20-day median dollar volume.
    amount = (opens[cols] * weights.attrs.get("volume", pd.DataFrame())).reindex(index=w.index, columns=cols)
    return {"daily": net, "weights": w, "metrics": stats, "turnover": gross_turnover, "cost": cost}


def _attach_volume(weights: pd.DataFrame, volume: pd.DataFrame, close: pd.DataFrame) -> None:
    dollar = (volume * close).reindex(columns=weights.columns)
    weights.attrs["dollar_volume"] = dollar


def capacity(weights: pd.DataFrame, dollar_volume: pd.DataFrame, aum: float = AUM_CNY) -> dict:
    w = weights.dropna(how="all")
    dv = dollar_volume.reindex(index=w.index, columns=w.columns)
    adv = dv.rolling(20, min_periods=10).median()
    delta = w.fillna(0.0).diff().abs()
    # First row diff is NaN; treat initial build as the weight itself.
    if len(delta):
        delta.iloc[0] = w.iloc[0].abs()
    util = (delta * aum) / adv.replace(0, np.nan)
    traded = delta.sum(axis=1) > 0.05
    stacked = util.loc[traded].replace([np.inf, -np.inf], np.nan).stack().dropna()
    stacked = stacked[stacked > 0]
    if stacked.empty:
        return {}
    # AUM that puts the 95th percentile trade at 5% of ADV.
    p95 = float(stacked.quantile(0.95))
    scale = (0.05 / p95) if p95 > 0 else np.nan
    return {
        "aum_cny": aum,
        "p95_trade_over_adv": p95,
        "max_trade_over_adv": float(stacked.max()),
        "aum_at_5pct_adv_p95": float(aum * scale) if pd.notna(scale) else np.nan,
    }


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.groupby(index.to_period("M")).tail(1).values)


def _history_ok(close: pd.DataFrame, bars: int) -> pd.DataFrame:
    seen = close.notna().cumsum()
    return (seen >= bars) & close.notna()


def _adv_ok(close: pd.DataFrame, volume: pd.DataFrame, minimum: float = ADV_MIN_CNY) -> pd.DataFrame:
    dollar = (close * volume).rolling(20, min_periods=10).median()
    return dollar >= minimum


def allocate_rank(
    score: pd.DataFrame,
    eligible: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    columns: Sequence[str],
    topn: int,
    ascending: bool,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """Equal-weight the selected names. Residual stays in the treasury ETF."""
    weights = pd.DataFrame(np.nan, index=score.index, columns=list(columns) + [BOND])
    for dt in signal_dates:
        if dt not in score.index:
            continue
        mask = eligible.loc[dt].reindex(score.columns).fillna(False)
        vals = score.loc[dt].where(mask).dropna()
        if len(vals) < min_names:
            weights.loc[dt, BOND] = 1.0
            weights.loc[dt, [c for c in columns if c != BOND]] = 0.0
            continue
        picked = list(vals.sort_values(ascending=ascending).index[:topn])
        row = {c: 0.0 for c in list(columns) + [BOND]}
        slot = 1.0 / len(picked)
        for name in picked:
            row[name] = slot
        weights.loc[dt] = row
    return weights


def _to_execution(signal_weights: pd.DataFrame) -> pd.DataFrame:
    """Shift a close-of-day target to the next session's open and hold it."""
    exec_w = signal_weights.shift(1)
    exec_w = exec_w.ffill()
    live = exec_w.dropna(how="all")
    if live.empty:
        return exec_w.fillna(0.0)
    first = live.index[0]
    exec_w = exec_w.loc[exec_w.index >= first].ffill().fillna(0.0)
    return exec_w


def strat_tsmom(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 time-series momentum on HS300, else treasury (Moskowitz/Ooi/Pedersen)."""
    signal = close[HS300].shift(skip) / close[HS300].shift(lookback) - 1.0
    ok = _history_ok(close[[HS300]], lookback).iloc[:, 0]
    longs = (signal > 0) & ok
    weights = pd.DataFrame(np.nan, index=close.index, columns=[HS300, BOND])
    dates = month_ends(close.index[close.index <= ASOF_COMPLETE])
    for dt in dates:
        if dt not in weights.index or not bool(ok.loc[dt]):
            continue
        if bool(longs.loc[dt]):
            weights.loc[dt, HS300] = 1.0
            weights.loc[dt, BOND] = 0.0
        else:
            weights.loc[dt, HS300] = 0.0
            weights.loc[dt, BOND] = 1.0
    return _to_execution(weights)


def strat_xs_momentum(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    topn: int = 3,
) -> pd.DataFrame:
    """Cross-sectional 12-1 momentum across sector/theme ETFs. Always invested in winners."""
    cols = list(SECTORS)
    score = close[cols].shift(skip) / close[cols].shift(lookback) - 1.0
    eligible = _history_ok(close[cols], lookback) & _adv_ok(close[cols], volume[cols]) & score.notna()
    dates = month_ends(close.index[close.index <= ASOF_COMPLETE])
    signal = allocate_rank(score, eligible, dates, cols, topn=topn, ascending=False)
    return _to_execution(signal)


def strat_reversal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 5,
    hold: int = 5,
    topn: int = 3,
) -> pd.DataFrame:
    """5-day sector reversal: buy the recent losers, hold for 5 sessions."""
    cols = list(SECTORS)
    score = close[cols].pct_change(lookback)
    eligible = _history_ok(close[cols], 60) & _adv_ok(close[cols], volume[cols]) & score.notna()
    dates = close.index[(close.index <= ASOF_COMPLETE)]
    # Fixed 5-session grid anchored at the first eval date, not a searched weekday.
    anchor = dates[dates >= EVAL_START]
    if len(anchor) == 0:
        return pd.DataFrame(0.0, index=close.index, columns=cols + [BOND])
    grid = anchor[::hold]
    signal = allocate_rank(score, eligible, grid, cols, topn=topn, ascending=True)
    return _to_execution(signal)


def strat_low_vol(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    vol_window: int = 60,
    topn: int = 3,
) -> pd.DataFrame:
    """Hold the lowest realized-vol sector ETFs (Baker/Bradley/Wurgler low-vol anomaly)."""
    cols = list(SECTORS)
    vol = close[cols].pct_change().rolling(vol_window, min_periods=vol_window).std()
    eligible = _history_ok(close[cols], max(120, vol_window)) & _adv_ok(close[cols], volume[cols]) & vol.notna()
    dates = month_ends(close.index[close.index <= ASOF_COMPLETE])
    signal = allocate_rank(vol, eligible, dates, cols, topn=topn, ascending=True)
    return _to_execution(signal)


def strat_ma_trend(close: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Faber-style trend: HS300 above its SMA, else 50/50 gold and treasury."""
    sma = close[HS300].rolling(window, min_periods=window).mean()
    risk_on = close[HS300] > sma
    weights = pd.DataFrame(np.nan, index=close.index, columns=[HS300, GOLD, BOND])
    known = sma.notna() & (close.index <= ASOF_COMPLETE)
    weights.loc[known & risk_on, HS300] = 1.0
    weights.loc[known & risk_on, [GOLD, BOND]] = 0.0
    weights.loc[known & ~risk_on, HS300] = 0.0
    weights.loc[known & ~risk_on, GOLD] = 0.5
    weights.loc[known & ~risk_on, BOND] = 0.5
    return _to_execution(weights)


def strat_turn_of_month(close: pd.DataFrame, pre: int = 2, post: int = 3) -> pd.DataFrame:
    """Long HS300 on the last `pre` and first `post` sessions of each month, else treasury.

    The exchange calendar is knowable before the open, so this does not use today's close.
    """
    weights = pd.DataFrame(0.0, index=close.index, columns=[HS300, BOND])
    tradable = close.index[(close.index <= ASOF_COMPLETE) | (close.index == TERMINAL_OPEN)]
    grouped: Dict[pd.Period, List[pd.Timestamp]] = {}
    for dt in tradable:
        grouped.setdefault(dt.to_period("M"), []).append(dt)
    hot = set()
    for days in grouped.values():
        hot.update(days[:post])
        hot.update(days[-pre:])
    for dt in tradable:
        if dt < EVAL_START:
            continue
        if dt in hot:
            weights.loc[dt, HS300] = 1.0
            weights.loc[dt, BOND] = 0.0
        else:
            weights.loc[dt, HS300] = 0.0
            weights.loc[dt, BOND] = 1.0
    # Drop dates before the eval window so the first in-window day pays entry cost.
    weights = weights.loc[weights.index >= EVAL_START]
    return weights


def strat_donchian(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    entry: int = 55,
    exit_: int = 20,
) -> pd.DataFrame:
    """Long-only Donchian breakout on four sleeves. Idle sleeve capital sits in treasury."""
    sleeves = [HS300, CSI500, CHINEXT, GOLD]
    signal = pd.DataFrame(0.0, index=close.index, columns=sleeves + [BOND])
    for symbol in sleeves:
        hh = high[symbol].rolling(entry, min_periods=entry).max().shift(1)
        ll = low[symbol].rolling(exit_, min_periods=exit_).min().shift(1)
        state = False
        states = []
        for dt in close.index:
            px = close.at[dt, symbol]
            hi = hh.at[dt]
            lo = ll.at[dt]
            if dt > ASOF_COMPLETE or pd.isna(px) or pd.isna(hi) or pd.isna(lo):
                states.append(False if dt > ASOF_COMPLETE else state)
                continue
            if state:
                if px < lo:
                    state = False
            elif px > hi:
                state = True
            states.append(state)
        signal[symbol] = pd.Series(states, index=close.index).astype(float) * (1.0 / len(sleeves))
    signal[BOND] = 1.0 - signal[sleeves].sum(axis=1)
    signal.loc[close.index > ASOF_COMPLETE, :] = np.nan
    return _to_execution(signal)


def strat_relative_value(close: pd.DataFrame, z_entry: float = 1.5, window: int = 60) -> pd.DataFrame:
    """Long-only relative value between HS300 and CSI500. No short leg.

    z > +entry means HS300 is rich vs its 60-day ratio history, so hold CSI500.
    z < -entry means HS300 is cheap, so hold HS300. Otherwise hold 50/50.
    """
    ratio = np.log(close[HS300]) - np.log(close[CSI500])
    z = (ratio - ratio.rolling(window, min_periods=window).mean()) / ratio.rolling(window, min_periods=window).std()
    weights = pd.DataFrame(np.nan, index=close.index, columns=[HS300, CSI500])
    known = z.notna() & (close.index <= ASOF_COMPLETE)
    rich = known & (z > z_entry)
    cheap = known & (z < -z_entry)
    mid = known & ~rich & ~cheap
    weights.loc[rich, HS300] = 0.0
    weights.loc[rich, CSI500] = 1.0
    weights.loc[cheap, HS300] = 1.0
    weights.loc[cheap, CSI500] = 0.0
    weights.loc[mid, HS300] = 0.5
    weights.loc[mid, CSI500] = 0.5
    return _to_execution(weights)


def strat_vol_target(close: pd.DataFrame, target: float = 0.12, window: int = 20, hold: int = 5, ann: float = 244.0) -> pd.DataFrame:
    """Weekly volatility target on HS300. Residual is treasury. No leverage."""
    rv = close[HS300].pct_change().rolling(window, min_periods=window).std() * np.sqrt(ann)
    raw = (target / rv).clip(lower=0.0, upper=1.0)
    weights = pd.DataFrame(np.nan, index=close.index, columns=[HS300, BOND])
    dates = close.index[close.index <= ASOF_COMPLETE]
    anchor = dates[dates >= (EVAL_START - pd.Timedelta(days=30))]
    grid = set(anchor[::hold])
    for dt in dates:
        if dt not in grid or pd.isna(raw.loc[dt]):
            continue
        w = float(raw.loc[dt])
        weights.loc[dt, HS300] = w
        weights.loc[dt, BOND] = 1.0 - w
    return _to_execution(weights)


def benchmark_static(close: pd.DataFrame, holdings: Dict[str, float], monthly: bool = False) -> pd.DataFrame:
    weights = pd.DataFrame(np.nan, index=close.index, columns=list(holdings))
    if monthly:
        for dt in month_ends(close.index[close.index <= ASOF_COMPLETE]):
            if dt < EVAL_START - pd.Timedelta(days=10):
                continue
            for symbol, weight in holdings.items():
                weights.loc[dt, symbol] = weight
        return _to_execution(weights)
    # Constant book on every session open in the eval window. No signal shift:
    # buy-and-hold is entered at the first eval open, not the next close.
    mask = (close.index >= EVAL_START) & (close.index <= ASOF_COMPLETE)
    for symbol, weight in holdings.items():
        weights.loc[mask, symbol] = weight
    weights = weights.loc[weights.index >= EVAL_START].fillna(0.0)
    return weights


def benchmark_equal_sector(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    cols = list(SECTORS)
    score = close[cols].notna().astype(float)
    eligible = _history_ok(close[cols], 60) & _adv_ok(close[cols], volume[cols])
    dates = month_ends(close.index[close.index <= ASOF_COMPLETE])
    weights = pd.DataFrame(np.nan, index=close.index, columns=cols + [BOND])
    for dt in dates:
        if dt < EVAL_START - pd.Timedelta(days=40):
            continue
        names = [c for c in cols if bool(eligible.loc[dt, c])] if dt in eligible.index else []
        row = {c: 0.0 for c in cols + [BOND]}
        if len(names) < MIN_NAMES:
            row[BOND] = 1.0
        else:
            slot = 1.0 / len(names)
            for name in names:
                row[name] = slot
        weights.loc[dt] = row
    return _to_execution(weights)


def newey_west_tstat(values: pd.Series, lags: int = 5) -> float:
    """t-stat of the mean with Bartlett weights. Lag 5 covers a weekly hold."""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < lags + 5:
        return float("nan")
    u = x - x.mean()
    gamma0 = float(np.dot(u, u) / n)
    var = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(np.dot(u[lag:], u[:-lag]) / n)
        var += 2.0 * weight * gamma
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(x.mean() / se)


def excess_stats(strategy: pd.Series, benchmark: pd.Series, ann_factor: float) -> dict:
    both = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1, sort=False).dropna()
    if len(both) < 20:
        return {}
    excess = both["s"] - both["b"]
    vol = float(excess.std(ddof=1) * np.sqrt(ann_factor))
    ir = float(excess.mean() * ann_factor / vol) if vol > 0 else np.nan
    tstat = float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))) if excess.std(ddof=1) > 0 else np.nan
    tstat_nw = newey_west_tstat(excess, lags=5)
    s = performance(both["s"], ann_factor)
    b = performance(both["b"], ann_factor)
    return {
        "excess_cagr": float(s["cagr"] - b["cagr"]),
        "excess_sharpe": float(s["sharpe"] - b["sharpe"]),
        "information_ratio": ir,
        "tstat_daily_excess": tstat,
        "tstat_newey_west": tstat_nw,
        "corr": float(both["s"].corr(both["b"])),
    }


def yearly_returns(daily: pd.Series) -> Dict[str, float]:
    if daily.empty:
        return {}
    grouped = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
    return {str(int(year)): float(val) for year, val in grouped.items()}


def regime_table(daily: pd.Series, hs_close: pd.Series, ann_factor: float) -> dict:
    """Attribute returns to lagged 120-day HS300 regimes so the label is known before the session."""
    trend = hs_close.pct_change(120).shift(1)
    label = pd.Series("sideways", index=hs_close.index)
    label = label.where(trend.notna(), other=np.nan)
    label = label.mask(trend > 0.10, "bull")
    label = label.mask(trend < -0.10, "bear")
    out = {}
    for name in ("bull", "bear", "sideways"):
        mask = label.reindex(daily.index) == name
        subset = daily[mask.fillna(False)]
        stats = performance(subset, ann_factor)
        # Regime days are not a contiguous calendar, so report mean annualized
        # return instead of chaining them into a fake CAGR.
        ann_return = float(subset.mean() * ann_factor) if len(subset) else np.nan
        out[name] = {
            "n": stats.get("n", 0),
            "ann_return": ann_return,
            "sharpe": stats.get("sharpe"),
            "max_drawdown": stats.get("max_drawdown"),
        }
    return out


def run_book(name: str, weights: pd.DataFrame, opens: pd.DataFrame, ann_factor: float, dollar: pd.DataFrame, cost_scale: float = 1.0, extra_delay: int = 0) -> dict:
    result = simulate(weights, opens, ann_factor, cost_scale=cost_scale, extra_delay=extra_delay)
    metrics = dict(result["metrics"])
    metrics["capacity"] = capacity(result["weights"], dollar)
    metrics["yearly"] = yearly_returns(result["daily"])
    return {"name": name, "daily": result["daily"], "weights": result["weights"], "metrics": metrics}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        obj = int(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def market_snapshot(macro: Dict[str, pd.DataFrame], close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    def last_complete(series: pd.Series) -> pd.Series:
        series = series.dropna()
        series = series[series.index <= ASOF_COMPLETE]
        return series

    snap = {"asof": str(ASOF_COMPLETE.date()), "clock_note": "2026-10-09 A股未收盘，快照用 2026-10-08 收盘"}
    hs = last_complete(close[HS300])
    snap["hs300"] = {
        "close": float(hs.iloc[-1]),
        "ret_21d": float(hs.pct_change(21).iloc[-1]),
        "ret_63d": float(hs.pct_change(63).iloc[-1]),
        "ret_252d": float(hs.pct_change(252).iloc[-1]),
        "dd_from_peak": float(hs.iloc[-1] / hs.cummax().iloc[-1] - 1.0),
        "peak_date": str(hs.cummax().idxmax().date()) if False else str(hs.loc[: hs.idxmax()].index[-1].date()) if hs.idxmax() == hs.index[-1] else str(hs.idxmax().date()),
        "above_ma50": bool(hs.iloc[-1] > hs.rolling(50).mean().iloc[-1]),
        "above_ma200": bool(hs.iloc[-1] > hs.rolling(200).mean().iloc[-1]),
        "ma200": float(hs.rolling(200).mean().iloc[-1]),
        "vol_20": float(hs.pct_change().rolling(20).std().iloc[-1] * np.sqrt(244)),
        "vol_60": float(hs.pct_change().rolling(60).std().iloc[-1] * np.sqrt(244)),
        "vol_252": float(hs.pct_change().tail(252).std() * np.sqrt(244)),
    }
    # Fix peak date properly.
    snap["hs300"]["peak_date"] = str(hs.idxmax().date())

    sectors = {}
    for symbol, name in SECTORS.items():
        series = last_complete(close[symbol])
        if len(series) < 30:
            continue
        sectors[name] = {
            "symbol": symbol,
            "ret_21d": float(series.pct_change(21).iloc[-1]) if len(series) > 21 else None,
            "ret_63d": float(series.pct_change(63).iloc[-1]) if len(series) > 63 else None,
            "ret_252d": float(series.pct_change(252).iloc[-1]) if len(series) > 252 else None,
            "above_ma200": bool(series.iloc[-1] > series.rolling(200).mean().iloc[-1]) if len(series) > 200 else None,
            "adv20_cny": float((close[symbol] * volume[symbol]).rolling(20).median().reindex(series.index).iloc[-1]),
        }
    snap["sectors"] = sectors
    breadth = [v["above_ma200"] for v in sectors.values() if v["above_ma200"] is not None]
    snap["sector_breadth_above_ma200"] = float(np.mean(breadth)) if breadth else None

    macro_out = {}
    for symbol, frame in macro.items():
        series = frame["Close"].dropna() if "Close" in frame else frame.iloc[:, 0].dropna()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series[series.index <= ASOF_COMPLETE]
        if series.empty:
            macro_out[symbol] = None
            continue
        macro_out[symbol] = {
            "date": str(series.index[-1].date()),
            "close": float(series.iloc[-1]),
            "ret_63d": float(series.pct_change(63).iloc[-1]) if len(series) > 63 else None,
            "ret_252d": float(series.pct_change(252).iloc[-1]) if len(series) > 252 else None,
        }
    snap["macro"] = macro_out
    return snap


def main() -> None:
    notes: List[str] = []
    panels = load_ohlc(notes)
    close = panels["Close"]
    opens = panels["Open"]
    high = panels["High"]
    low = panels["Low"]
    volume = panels["Volume"]
    # Trading calendar is HS300 sessions that have a completed close.
    calendar = close.index[(close[HS300].notna()) & (close.index <= ASOF_COMPLETE)]
    # Keep the terminal open row on the price panels for the last oo return.
    keep_index = calendar.append(pd.DatetimeIndex([TERMINAL_OPEN])).unique().sort_values()
    close = close.reindex(keep_index)
    opens = opens.reindex(keep_index)
    high = high.reindex(keep_index)
    low = low.reindex(keep_index)
    volume = volume.reindex(keep_index)
    # Do not forward-fill the terminal close.
    ann = _ann_factor(pd.DatetimeIndex(calendar))
    notes.append(f"年化因子取 2016-2025 沪深300ETF 每年交易日数的中位数：{ann:.0f}")
    dropped = [symbol for symbol in list(SECTORS) if symbol not in close.columns]
    for symbol in dropped:
        SECTORS.pop(symbol, None)
        notes.append(f"行业池剔除 {symbol}（{NAMES.get(symbol, symbol)}），价格未通过完整性审计")
    required = [HS300, CSI500, CHINEXT, GOLD, BOND]
    absent = [symbol for symbol in required if symbol not in close.columns]
    if absent:
        raise RuntimeError("缺少核心价格: " + ", ".join(absent))

    # Gap diagnostic on completed sessions.
    gaps = {}
    for symbol in close.columns:
        series = close.loc[calendar, symbol]
        gaps[symbol] = int(series.isna().sum())
    worst = sorted(gaps.items(), key=lambda kv: kv[1], reverse=True)[:8]
    notes.append("对齐沪深300交易日后，各 ETF 收盘缺失天数（前 8）：" + ", ".join(f"{k}:{v}" for k, v in worst))

    dollar = close * volume
    builders = {
        "tsmom_12_1": lambda: strat_tsmom(close),
        "xs_sector_mom_12_1": lambda: strat_xs_momentum(close, volume),
        "sector_reversal_5d": lambda: strat_reversal(close, volume),
        "low_vol_sectors": lambda: strat_low_vol(close, volume),
        "ma200_trend": lambda: strat_ma_trend(close),
        "turn_of_month": lambda: strat_turn_of_month(close),
        "donchian_55_20": lambda: strat_donchian(close, high, low),
        "rv_hs300_csi500": lambda: strat_relative_value(close),
        "vol_target_12": lambda: strat_vol_target(close, ann=ann),
    }
    books = {}
    for name, builder in builders.items():
        weights = builder()
        books[name] = run_book(name, weights, opens, ann, dollar)

    benchmarks = {
        "bh_hs300": benchmark_static(close, {HS300: 1.0}),
        "bh_60_40": benchmark_static(close, {HS300: 0.6, BOND: 0.4}, monthly=True),
        "bh_50_50_300_500": benchmark_static(close, {HS300: 0.5, CSI500: 0.5}, monthly=True),
        "ew_sectors": benchmark_equal_sector(close, volume),
    }
    bench_books = {name: run_book(name, weights, opens, ann, dollar) for name, weights in benchmarks.items()}

    # Synthetic sanity: a 10% open-to-open ramp must survive the simulator.
    toy_idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    toy_open = pd.DataFrame({HS300: [100.0, 110.0, 121.0], BOND: [100.0, 100.0, 100.0]}, index=toy_idx)
    toy_w = pd.DataFrame({HS300: [1.0, 1.0, 1.0], BOND: [0.0, 0.0, 0.0]}, index=toy_idx)
    toy = simulate(toy_w, toy_open, ann_factor=244)
    toy_rets = toy["daily"].tolist()
    if len(toy_rets) != 2 or abs(toy_rets[0] - (0.10 - COST_EQUITY)) > 1e-9 or abs(toy_rets[1] - 0.10) > 1e-9:
        raise RuntimeError(f"simulator sanity failed: {toy_rets}")
    notes.append("模拟器单元检查通过：开盘到开盘 10% 涨幅，首日扣除权益 ETF 单边成本，次日不加成本。")

    def pack(book: dict, benchmark_name: str = "bh_hs300") -> dict:
        daily = book["daily"]
        bench = bench_books[benchmark_name]["daily"]
        metrics = dict(book["metrics"])
        windows = {
            "full": (EVAL_START, ASOF_COMPLETE),
            "in_sample": (EVAL_START, IS_END),
            "out_of_sample": (pd.Timestamp("2023-01-01"), OOS_END),
            "holdout": (HOLDOUT_START, ASOF_COMPLETE),
        }
        block = {"full_metrics": metrics}
        for label, (start, end) in windows.items():
            stats = performance(slice_period(daily, start, end), ann)
            vs = excess_stats(slice_period(daily, start, end), slice_period(bench, start, end), ann)
            block[label] = {"performance": stats, "vs_hs300": vs}
        block["vs_60_40"] = excess_stats(daily[daily.index >= EVAL_START], bench_books["bh_60_40"]["daily"], ann)
        block["vs_ew_sectors"] = excess_stats(
            daily[daily.index >= EVAL_START], bench_books["ew_sectors"]["daily"], ann
        )
        block["regimes"] = regime_table(daily[daily.index >= EVAL_START], close.loc[calendar, HS300], ann)
        return block

    packed = {name: pack(book) for name, book in books.items()}
    # Relative-value relevant benchmark is 50/50, stored alongside HS300.
    packed["rv_hs300_csi500"]["vs_50_50"] = excess_stats(
        books["rv_hs300_csi500"]["daily"], bench_books["bh_50_50_300_500"]["daily"], ann
    )
    packed_bench = {name: pack(book) for name, book in bench_books.items()}

    # Sensitivity. Primary key is flagged; we do not re-rank on this grid.
    sensitivity_specs = {
        "tsmom_12_1": [
            ("lb126_skip21", lambda: strat_tsmom(close, 126, 21)),
            ("lb252_skip21_PRIMARY", lambda: strat_tsmom(close, 252, 21)),
            ("lb252_skip0", lambda: strat_tsmom(close, 252, 0)),
        ],
        "xs_sector_mom_12_1": [
            ("lb126_top3", lambda: strat_xs_momentum(close, volume, 126, 21, 3)),
            ("lb252_top3_PRIMARY", lambda: strat_xs_momentum(close, volume, 252, 21, 3)),
            ("lb252_top2", lambda: strat_xs_momentum(close, volume, 252, 21, 2)),
            ("lb252_top4", lambda: strat_xs_momentum(close, volume, 252, 21, 4)),
        ],
        "sector_reversal_5d": [
            ("lb3", lambda: strat_reversal(close, volume, 3, 5, 3)),
            ("lb5_PRIMARY", lambda: strat_reversal(close, volume, 5, 5, 3)),
            ("lb10", lambda: strat_reversal(close, volume, 10, 5, 3)),
        ],
        "low_vol_sectors": [
            ("vol40", lambda: strat_low_vol(close, volume, 40, 3)),
            ("vol60_PRIMARY", lambda: strat_low_vol(close, volume, 60, 3)),
            ("vol120", lambda: strat_low_vol(close, volume, 120, 3)),
        ],
        "ma200_trend": [
            ("ma100", lambda: strat_ma_trend(close, 100)),
            ("ma150", lambda: strat_ma_trend(close, 150)),
            ("ma200_PRIMARY", lambda: strat_ma_trend(close, 200)),
        ],
        "turn_of_month": [
            ("pre1_post3", lambda: strat_turn_of_month(close, 1, 3)),
            ("pre2_post3_PRIMARY", lambda: strat_turn_of_month(close, 2, 3)),
            ("pre3_post1", lambda: strat_turn_of_month(close, 3, 1)),
        ],
        "donchian_55_20": [
            ("e40_x20", lambda: strat_donchian(close, high, low, 40, 20)),
            ("e55_x20_PRIMARY", lambda: strat_donchian(close, high, low, 55, 20)),
            ("e20_x10", lambda: strat_donchian(close, high, low, 20, 10)),
        ],
        "rv_hs300_csi500": [
            ("z1.0", lambda: strat_relative_value(close, 1.0)),
            ("z1.5_PRIMARY", lambda: strat_relative_value(close, 1.5)),
            ("z2.0", lambda: strat_relative_value(close, 2.0)),
        ],
        "vol_target_12": [
            ("t10", lambda: strat_vol_target(close, 0.10, ann=ann)),
            ("t12_PRIMARY", lambda: strat_vol_target(close, 0.12, ann=ann)),
            ("t15", lambda: strat_vol_target(close, 0.15, ann=ann)),
        ],
    }
    sensitivity = {}
    for family, specs in sensitivity_specs.items():
        sensitivity[family] = {}
        for label, builder in specs:
            result = simulate(builder(), opens, ann)
            stats = result["metrics"]
            vs = excess_stats(result["daily"], bench_books["bh_hs300"]["daily"], ann)
            oos = excess_stats(
                slice_period(result["daily"], pd.Timestamp("2023-01-01"), OOS_END),
                slice_period(bench_books["bh_hs300"]["daily"], pd.Timestamp("2023-01-01"), OOS_END),
                ann,
            )
            sensitivity[family][label] = {
                "cagr": stats.get("cagr"),
                "sharpe": stats.get("sharpe"),
                "max_drawdown": stats.get("max_drawdown"),
                "excess_cagr_vs_hs300": vs.get("excess_cagr"),
                "oos_excess_cagr_vs_hs300": oos.get("excess_cagr"),
            }

    # Stress on the primary books: 2x costs, and one extra session of delay.
    stress = {}
    for name in books:
        base_w = books[name]["weights"]
        # Rebuild from primary builders so delay is applied inside simulate via shift.
        # Using the already-executed weights and shifting again double-counts the
        # economic delay only when we pass extra_delay. That is what we want.
        hi_cost = simulate(base_w, opens, ann, cost_scale=2.0)
        delayed = simulate(base_w, opens, ann, extra_delay=1)
        stress[name] = {
            "base_sharpe": books[name]["metrics"].get("sharpe"),
            "base_cagr": books[name]["metrics"].get("cagr"),
            "cost_x2_sharpe": hi_cost["metrics"].get("sharpe"),
            "cost_x2_cagr": hi_cost["metrics"].get("cagr"),
            "extra_delay_sharpe": delayed["metrics"].get("sharpe"),
            "extra_delay_cagr": delayed["metrics"].get("cagr"),
        }

    # Strategy correlation on overlapping daily net returns.
    ret_df = pd.DataFrame({name: book["daily"] for name, book in books.items()})
    corr = ret_df.corr()

    macro_raw = _download_yahoo(MACRO_SYMBOLS, start="2016-01-01")
    # Macro downloader uses adjusted construction only when Adj Close exists.
    macro_frames = {}
    for symbol, frame in macro_raw.items():
        if "Close" not in frame:
            continue
        macro_frames[symbol] = frame
    snapshot = market_snapshot(macro_frames, close, volume)

    # Latest primary signals, computed through 2026-10-08 close, for the pending 10-09 open.
    # These are informational and are NOT included in backtest PnL.
    latest_weights = {}
    for name, book in books.items():
        w = book["weights"]
        held = w[w.index <= ASOF_COMPLETE]
        pending = w[w.index == TERMINAL_OPEN]
        if held.empty:
            continue
        last = held.iloc[-1]
        pending_row = pending.iloc[-1] if len(pending) else None
        latest_weights[name] = {
            "last_completed_execution_date": str(held.index[-1].date()),
            "holdings_last_completed": {
                NAMES.get(k, k): float(v) for k, v in last.items() if pd.notna(v) and abs(float(v)) > 1e-6
            },
            "pending_for_2026_10_09_open": (
                {
                    NAMES.get(k, k): float(v)
                    for k, v in pending_row.items()
                    if pd.notna(v) and abs(float(v)) > 1e-6
                }
                if pending_row is not None
                else None
            ),
        }

    output = {
        "ann_factor": ann,
        "eval_start": str(EVAL_START.date()),
        "sample_end_signal": str(ASOF_COMPLETE.date()),
        "notes": notes,
        "snapshot": snapshot,
        "strategies": packed,
        "benchmarks": packed_bench,
        "sensitivity": sensitivity,
        "stress": stress,
        "correlation": corr.round(3).to_dict(),
        "latest_weights": latest_weights,
        "cost_model": {
            "equity_etf_one_way": COST_EQUITY,
            "bond_etf_one_way": COST_BOND,
            "components": "equity = 1.5bps commission + 5bps slippage; bond = 8bps all-in; ETF stamp tax = 0",
            "aum_cny": AUM_CNY,
        },
    }
    path = "/tmp/tsr_20261009_results.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(output), handle, ensure_ascii=False, indent=2)
    print(path)
    # Compact console summary for the analyst.
    print(f"ANN {ann:.1f}")
    for group_name, group in (("STRAT", packed), ("BENCH", packed_bench)):
        print("\n==", group_name)
        for name, block in group.items():
            full = block["full"]["performance"]
            vs = block["full"]["vs_hs300"]
            print(
                f"{name:24} CAGR {full.get('cagr', float('nan')):7.2%} "
                f"Sh {full.get('sharpe', float('nan')):5.2f} "
                f"DD {full.get('max_drawdown', float('nan')):7.2%} "
                f"So {full.get('sortino', float('nan')):5.2f} "
                f"WR {full.get('monthly_win_rate', float('nan')):5.1%} "
                f"PF {full.get('profit_factor', float('nan')):4.2f} "
                f"exCAGR {vs.get('excess_cagr', float('nan')):7.2%} "
                f"t {vs.get('tstat_daily_excess', float('nan')):5.2f} "
                f"TO {block['full_metrics'].get('gross_turnover_annual', float('nan')):5.2f}"
            )


if __name__ == "__main__":
    main()
