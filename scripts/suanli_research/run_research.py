#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-share Compute (算力) sector strategy research & backtest.

Universe covers optical modules, AI chips, servers, PCB, liquid cooling, and
memory-interface names commonly associated with AI compute infrastructure.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Curated 算力 universe (liquid, representative across sub-chains)
UNIVERSE: Dict[str, Dict[str, str]] = {
    "300308": {"name": "中际旭创", "sub": "光模块", "yf": "300308.SZ"},
    "300502": {"name": "新易盛", "sub": "光模块", "yf": "300502.SZ"},
    "300394": {"name": "天孚通信", "sub": "光器件", "yf": "300394.SZ"},
    "688256": {"name": "寒武纪", "sub": "AI芯片", "yf": "688256.SS"},
    "688041": {"name": "海光信息", "sub": "AI芯片", "yf": "688041.SS"},
    "601138": {"name": "工业富联", "sub": "服务器", "yf": "601138.SS"},
    "002463": {"name": "沪电股份", "sub": "PCB", "yf": "002463.SZ"},
    "300476": {"name": "胜宏科技", "sub": "PCB", "yf": "300476.SZ"},
    "002837": {"name": "英维克", "sub": "液冷", "yf": "002837.SZ"},
    "688008": {"name": "澜起科技", "sub": "接口芯片", "yf": "688008.SS"},
    "000063": {"name": "中兴通讯", "sub": "通信设备", "yf": "000063.SZ"},
    "603019": {"name": "中科曙光", "sub": "服务器", "yf": "603019.SS"},
}

# One-way cost assumption for ChiNext/STAR-heavy names (bps of notional)
COST_BPS = 15.0  # 0.15% one-way ≈ commission + stamp (sell) amortized + slippage
START = "20220101"
END = "20260928"
IS_END = "20241231"  # in-sample / out-of-sample split


def _symbol_ak(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def fetch_stock(code: str) -> pd.DataFrame:
    """Fetch adjusted daily bars via Tencent endpoint (more reliable than EM)."""
    df = ak.stock_zh_a_hist_tx(
        symbol=_symbol_ak(code),
        start_date=START,
        end_date=END,
        adjust="qfq",
    )
    df = df.rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    df["code"] = code
    return df


def fetch_index(symbol: str = "sz399006") -> pd.DataFrame:
    """ChiNext index as broad growth/tech proxy benchmark."""
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp("2022-01-01")) & (df["date"] <= pd.Timestamp("2026-09-28"))]
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def load_panel() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    frames = []
    ok_names = {}
    for code, meta in UNIVERSE.items():
        try:
            df = fetch_stock(code)
            if len(df) < 200:
                print(f"skip {code}: too short ({len(df)})")
                continue
            frames.append(df)
            ok_names[code] = meta["name"]
            print(f"ok {code} {meta['name']}: {len(df)} bars, last={df['date'].iloc[-1].date()}")
        except Exception as exc:  # noqa: BLE001
            print(f"fail {code}: {type(exc).__name__}: {exc}")
    if not frames:
        raise RuntimeError("No stock data downloaded")
    panel = pd.concat(frames, ignore_index=True)
    idx = fetch_index()
    print(f"index bars: {len(idx)}, last={idx['date'].iloc[-1].date()}")
    return panel, idx, ok_names


def pivot_close(panel: pd.DataFrame) -> pd.DataFrame:
    px = panel.pivot(index="date", columns="code", values="close").sort_index()
    return px.ffill(limit=3)


def pivot_volume(panel: pd.DataFrame) -> pd.DataFrame:
    vol = panel.pivot(index="date", columns="code", values="volume").sort_index()
    return vol.ffill(limit=3)


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    metrics: Dict[str, float]
    metrics_is: Dict[str, float]
    metrics_oos: Dict[str, float]
    notes: str


def apply_costs(weights: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    """Turn target weights (end-of-day signal, next-day open approx via close) into net returns.

    Convention: weights at t are applied to returns from t -> t+1.
    Turnover cost charged on |Δw| * cost_bps.
    """
    rets = weights.columns  # placeholder to satisfy type checkers; overwritten below
    del rets
    # Align with price returns computed outside — caller should pass returns separately
    raise NotImplementedError


def portfolio_backtest(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = COST_BPS,
) -> pd.Series:
    """target_weights indexed by date, columns=codes; applied to next-day returns."""
    w = target_weights.reindex(returns.index).fillna(0.0)
    # Shift: signal at close t trades for return t->t+1
    w_exec = w.shift(1).fillna(0.0)
    gross = (w_exec * returns).sum(axis=1)
    turnover = w_exec.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 10000.0)
    net = gross - cost
    equity = (1.0 + net).cumprod()
    equity.name = "equity"
    return equity


def metrics_from_equity(equity: pd.Series, freq: int = 252) -> Dict[str, float]:
    if equity is None or len(equity) < 5:
        return {}
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {}
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if equity.iloc[0] > 0 else float("nan")
    vol = float(rets.std() * math.sqrt(freq))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(freq)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * math.sqrt(freq)) if len(downside) and downside.std() > 0 else 0.0
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    win_rate = float((rets > 0).mean())
    # Approximate trade-level profit factor via daily
    gains = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    pf = float(gains / losses) if losses > 0 else float("inf")
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": pf,
        "n_days": int(len(rets)),
    }


def split_metrics(equity: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    full = metrics_from_equity(equity)
    is_mask = equity.index <= pd.Timestamp(IS_END)
    oos_mask = equity.index > pd.Timestamp(IS_END)
    is_eq = equity.loc[is_mask]
    oos_eq = equity.loc[oos_mask]
    # Rebase segments
    if len(is_eq):
        is_eq = is_eq / is_eq.iloc[0]
    if len(oos_eq):
        oos_eq = oos_eq / oos_eq.iloc[0]
    return full, metrics_from_equity(is_eq), metrics_from_equity(oos_eq)


# -------------------- Strategies --------------------


def strat_equal_weight_buyhold(close: pd.DataFrame) -> pd.DataFrame:
    """Benchmark sleeve: always equal-weight the universe."""
    w = close.notna().astype(float)
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
    return w


def strat_cross_sectional_momentum(close: pd.DataFrame, lookback: int = 60, top_n: int = 4) -> pd.DataFrame:
    """Long top-N 60d relative strength, monthly rebalance."""
    mom = close.pct_change(lookback)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    # Rebalance on month-end / first available day of month
    months = close.index.to_period("M")
    rebalance_days = close.groupby(months).apply(lambda x: x.index[-1])
    for dt in rebalance_days:
        if dt not in mom.index:
            continue
        row = mom.loc[dt].dropna()
        if len(row) < top_n:
            continue
        picks = row.nlargest(top_n).index
        w.loc[dt, picks] = 1.0 / top_n
    # Hold between rebalances
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    # Zero weight when all nan prices
    avail = close.notna()
    w = w.where(avail, 0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return w


def strat_dual_momentum(close: pd.DataFrame, lookback: int = 90, top_n: int = 3) -> pd.DataFrame:
    """Absolute + relative momentum: long top-N with positive lookback return, else cash."""
    mom = close.pct_change(lookback)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    months = close.index.to_period("M")
    rebalance_days = close.groupby(months).apply(lambda x: x.index[-1])
    for dt in rebalance_days:
        row = mom.loc[dt].dropna()
        pos = row[row > 0]
        if len(pos) == 0:
            continue
        picks = pos.nlargest(min(top_n, len(pos))).index
        w.loc[dt, picks] = 1.0 / len(picks)
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    avail = close.notna()
    w = w.where(avail, 0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return w


def strat_ma_trend_filter(close: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.DataFrame:
    """Equal-weight names with MA_fast > MA_slow; cash if none."""
    ma_f = close.rolling(fast).mean()
    ma_s = close.rolling(slow).mean()
    signal = (close > ma_f) & (ma_f > ma_s)
    w = signal.astype(float)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return w


def strat_mean_reversion_pullback(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """In uptrend (close > MA60), buy oversold RSI<30 or >5% below MA20; exit when RSI>55."""
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    # RSI 14
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    uptrend = close > ma60
    entry = uptrend & ((rsi < 30) | (close < ma20 * 0.95))
    # Sticky state machine approximated via forward-fill of entry until exit
    hold = pd.DataFrame(False, index=close.index, columns=close.columns)
    for col in close.columns:
        state = False
        for i, dt in enumerate(close.index):
            if entry.loc[dt, col]:
                state = True
            if state and (rsi.loc[dt, col] > 55 or not uptrend.loc[dt, col]):
                # keep one day then exit next — mark False today after exit signal
                if rsi.loc[dt, col] > 55 or close.loc[dt, col] < ma60.loc[dt, col]:
                    state = False
            hold.loc[dt, col] = state
    w = hold.astype(float)
    # mild volume confirmation optional: prefer higher relative volume names
    rel_vol = vol / vol.rolling(20).mean()
    w = w * rel_vol.clip(0.5, 2.0).fillna(1.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return w


def strat_volume_breakout(close: pd.DataFrame, vol: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Breakout: close > N-day high and volume > 1.5x 20d avg; hold 10 days or trail."""
    hh = close.rolling(lookback).max().shift(1)
    avg_vol = vol.rolling(20).mean()
    breakout = (close > hh) & (vol > 1.5 * avg_vol)
    # Hold for 10 trading days after signal
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for col in close.columns:
        hold_left = 0
        for dt in close.index:
            if bool(breakout.loc[dt, col]):
                hold_left = 10
            if hold_left > 0:
                w.loc[dt, col] = 1.0
                hold_left -= 1
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return w


def strat_sector_breadth_timing(close: pd.DataFrame) -> pd.DataFrame:
    """Risk-on equal-weight when breadth (% above MA20) > 55%; else cash."""
    above = (close > close.rolling(20).mean()).astype(float)
    breadth = above.mean(axis=1)
    risk_on = breadth > 0.55
    ew = strat_equal_weight_buyhold(close)
    w = ew.mul(risk_on.astype(float), axis=0)
    return w


def strat_low_volatility(close: pd.DataFrame, lookback: int = 60, top_n: int = 4) -> pd.DataFrame:
    """Monthly long the lowest realized-vol names (quality/defensive sleeve within theme)."""
    rets = close.pct_change()
    vol = rets.rolling(lookback).std()
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    months = close.index.to_period("M")
    rebalance_days = close.groupby(months).apply(lambda x: x.index[-1])
    for dt in rebalance_days:
        row = vol.loc[dt].dropna()
        if len(row) < top_n:
            continue
        picks = row.nsmallest(top_n).index
        w.loc[dt, picks] = 1.0 / top_n
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    avail = close.notna()
    w = w.where(avail, 0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return w


def strat_subchain_rotation(close: pd.DataFrame, panel_meta: Dict[str, str]) -> pd.DataFrame:
    """Rotate into strongest sub-chain (60d), equal-weight names within that sub-chain."""
    # Build sub-chain map from UNIVERSE
    sub_map = {c: UNIVERSE[c]["sub"] for c in close.columns if c in UNIVERSE}
    subs = sorted(set(sub_map.values()))
    # Sub-chain equal-weight index returns
    mom = {}
    for sub in subs:
        cols = [c for c, s in sub_map.items() if s == sub]
        if not cols:
            continue
        sub_px = close[cols].mean(axis=1)
        mom[sub] = sub_px.pct_change(60)
    mom_df = pd.DataFrame(mom)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    months = close.index.to_period("M")
    rebalance_days = close.groupby(months).apply(lambda x: x.index[-1])
    for dt in rebalance_days:
        if dt not in mom_df.index:
            continue
        row = mom_df.loc[dt].dropna()
        if row.empty:
            continue
        best = row.idxmax()
        cols = [c for c, s in sub_map.items() if s == best]
        w.loc[dt, cols] = 1.0 / len(cols)
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    avail = close.notna()
    w = w.where(avail, 0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return w


def run_all(close: pd.DataFrame, vol: pd.DataFrame, idx: pd.DataFrame) -> List[BacktestResult]:
    rets = close.pct_change().fillna(0.0)
    strategies: List[Tuple[str, Callable[[], pd.DataFrame], str]] = [
        (
            "EW_BuyHold",
            lambda: strat_equal_weight_buyhold(close),
            "Equal-weight buy & hold of 算力 universe (internal benchmark sleeve)",
        ),
        (
            "CS_Momentum_60d",
            lambda: strat_cross_sectional_momentum(close, 60, 4),
            "Cross-sectional momentum: monthly long top-4 by 60d return",
        ),
        (
            "Dual_Momentum",
            lambda: strat_dual_momentum(close, 90, 3),
            "Dual momentum: long top-3 with positive 90d absolute momentum else cash",
        ),
        (
            "MA_Trend_Filter",
            lambda: strat_ma_trend_filter(close, 20, 60),
            "Trend filter: EW names with close>MA20>MA60",
        ),
        (
            "MR_Pullback",
            lambda: strat_mean_reversion_pullback(close, vol),
            "Mean-reversion pullback within MA60 uptrend (RSI/MA20 oversold)",
        ),
        (
            "Volume_Breakout",
            lambda: strat_volume_breakout(close, vol, 20),
            "Volume-confirmed 20d breakout, 10-day hold",
        ),
        (
            "Breadth_Timing",
            lambda: strat_sector_breadth_timing(close),
            "Sector breadth timing: EW only when >55% names above MA20",
        ),
        (
            "Low_Vol",
            lambda: strat_low_volatility(close, 60, 4),
            "Low-volatility sleeve: monthly long 4 lowest 60d vol names",
        ),
        (
            "Subchain_Rotation",
            lambda: strat_subchain_rotation(close, {}),
            "Rotate into strongest sub-chain by 60d momentum",
        ),
    ]

    results: List[BacktestResult] = []
    for name, factory, notes in strategies:
        w = factory()
        eq = portfolio_backtest(w, rets)
        full, mis, moos = split_metrics(eq)
        results.append(BacktestResult(name, eq, full, mis, moos, notes))
        print(f"{name}: Sharpe={full.get('sharpe', float('nan')):.2f} CAGR={full.get('cagr', float('nan')):.1%} MDD={full.get('max_drawdown', float('nan')):.1%}")

    # Index buy & hold benchmark
    idx = idx.set_index("date").sort_index()
    idx = idx.reindex(close.index).ffill()
    idx_ret = idx["close"].pct_change().fillna(0.0)
    idx_eq = (1 + idx_ret).cumprod()
    full, mis, moos = split_metrics(idx_eq)
    results.append(BacktestResult("ChiNext_Index", idx_eq, full, mis, moos, "创业板指 buy&hold benchmark"))
    print(f"ChiNext_Index: Sharpe={full.get('sharpe', float('nan')):.2f} CAGR={full.get('cagr', float('nan')):.1%}")
    return results


def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.1%}"


def fmt_num(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.2f}"


def market_snapshot(close: pd.DataFrame, names: Dict[str, str]) -> Dict:
    last = close.iloc[-1]
    rets_1m = close.pct_change(21).iloc[-1]
    rets_3m = close.pct_change(63).iloc[-1]
    rets_1y = close.pct_change(252).iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    above20 = float((last > ma20).mean())
    above60 = float((last > ma60).mean())
    rows = []
    for code in close.columns:
        rows.append(
            {
                "code": code,
                "name": names.get(code, code),
                "sub": UNIVERSE.get(code, {}).get("sub", ""),
                "last": float(last[code]) if pd.notna(last[code]) else None,
                "ret_1m": float(rets_1m[code]) if pd.notna(rets_1m[code]) else None,
                "ret_3m": float(rets_3m[code]) if pd.notna(rets_3m[code]) else None,
                "ret_1y": float(rets_1y[code]) if pd.notna(rets_1y[code]) else None,
                "above_ma20": bool(last[code] > ma20[code]) if pd.notna(last[code]) else False,
                "above_ma60": bool(last[code] > ma60[code]) if pd.notna(last[code]) else False,
            }
        )
    # Regime heuristics on EW
    ew = close.mean(axis=1)
    ew_ma60 = ew.rolling(60).mean()
    regime = "bull" if ew.iloc[-1] > ew_ma60.iloc[-1] else "bear/sideways"
    vol20 = ew.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252)
    return {
        "asof": str(close.index[-1].date()),
        "breadth_above_ma20": above20,
        "breadth_above_ma60": above60,
        "regime": regime,
        "ew_ann_vol_20d": float(vol20) if pd.notna(vol20) else None,
        "stocks": rows,
    }


def write_report(results: List[BacktestResult], snap: Dict, names: Dict[str, str]) -> Path:
    # Ranking: prefer OOS Sharpe, then full Sharpe, require not terrible MDD
    ranked = sorted(
        [r for r in results if r.name != "ChiNext_Index"],
        key=lambda r: (
            r.metrics_oos.get("sharpe", -99) if r.metrics_oos else -99,
            r.metrics.get("sharpe", -99),
        ),
        reverse=True,
    )
    bench = next(r for r in results if r.name == "ChiNext_Index")
    ew = next(r for r in results if r.name == "EW_BuyHold")

    lines: List[str] = []
    lines.append("# 算力板块交易策略研究报告")
    lines.append("")
    lines.append(f"- 数据截止: **{snap['asof']}**")
    lines.append(f"- 回测区间: 2022-01 ~ {snap['asof']}（IS≤{IS_END[:4]}-12-31 / OOS 之后）")
    lines.append(f"- 样本池: {len(names)} 只代表性算力链标的（光模块/AI芯片/服务器/PCB/液冷/接口芯片/通信设备）")
    lines.append(f"- 成本假设: 单边 **{COST_BPS:.0f}bps**（佣金+印花税摊销+滑点）")
    lines.append(f"- 基准: 创业板指 buy&hold；内部对照: 算力等权 buy&hold")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        f"当前算力样本池广度：MA20 上方占比 **{snap['breadth_above_ma20']:.0%}**，"
        f"MA60 上方 **{snap['breadth_above_ma60']:.0%}**；等权指数相对 MA60 判定为 **{snap['regime']}**，"
        f"20 日年化波动约 **{fmt_pct(snap['ew_ann_vol_20d'])}**。"
    )
    lines.append("")
    lines.append("样本外（OOS）夏普排名靠前的策略：")
    for i, r in enumerate(ranked[:5], 1):
        lines.append(
            f"{i}. **{r.name}** — OOS Sharpe {fmt_num(r.metrics_oos.get('sharpe'))}, "
            f"OOS CAGR {fmt_pct(r.metrics_oos.get('cagr'))}, "
            f"Full MDD {fmt_pct(r.metrics.get('max_drawdown'))}"
        )
    lines.append("")
    lines.append("## 2. Market Analysis")
    lines.append("")
    lines.append("| 代码 | 名称 | 细分 | 1M | 3M | 1Y | >MA20 | >MA60 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | :---: | :---: |")
    for s in sorted(snap["stocks"], key=lambda x: (x["ret_3m"] is None, -(x["ret_3m"] or -999))):
        lines.append(
            f"| {s['code']} | {s['name']} | {s['sub']} | {fmt_pct(s['ret_1m'])} | "
            f"{fmt_pct(s['ret_3m'])} | {fmt_pct(s['ret_1y'])} | "
            f"{'Y' if s['above_ma20'] else 'N'} | {'Y' if s['above_ma60'] else 'N'} |"
        )
    lines.append("")
    lines.append("### Opportunity notes (facts vs interpretation)")
    lines.append("")
    lines.append(
        "- **事实**: AI 训练/推理资本开支仍是全球主线；A 股映射集中在光模块（易中天）、国产算力芯片、AI 服务器与高速 PCB/液冷。"
    )
    lines.append(
        "- **解释**: 板块波动与估值拥挤显著高于宽基；动量类策略在趋势段有效，但回撤与拥挤踩踏风险并存，需要择时或绝对动量过滤。"
    )
    lines.append("")
    lines.append("## 3. Strategy Details")
    lines.append("")
    for r in results:
        if r.name == "ChiNext_Index":
            continue
        lines.append(f"### {r.name}")
        lines.append("")
        lines.append(f"- Thesis/Notes: {r.notes}")
        lines.append(
            f"- Full: CAGR {fmt_pct(r.metrics.get('cagr'))}, Sharpe {fmt_num(r.metrics.get('sharpe'))}, "
            f"Sortino {fmt_num(r.metrics.get('sortino'))}, MDD {fmt_pct(r.metrics.get('max_drawdown'))}, "
            f"Win {fmt_pct(r.metrics.get('win_rate'))}, PF {fmt_num(r.metrics.get('profit_factor'))}"
        )
        lines.append(
            f"- IS: Sharpe {fmt_num(r.metrics_is.get('sharpe'))}, CAGR {fmt_pct(r.metrics_is.get('cagr'))}, "
            f"MDD {fmt_pct(r.metrics_is.get('max_drawdown'))}"
        )
        lines.append(
            f"- OOS: Sharpe {fmt_num(r.metrics_oos.get('sharpe'))}, CAGR {fmt_pct(r.metrics_oos.get('cagr'))}, "
            f"MDD {fmt_pct(r.metrics_oos.get('max_drawdown'))}"
        )
        lines.append("")

    lines.append("## 4. Backtest Results (comparison table)")
    lines.append("")
    lines.append(
        "| Strategy | CAGR | Sharpe | Sortino | MDD | Win | PF | IS Sharpe | OOS Sharpe |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in [bench, ew] + ranked:
        m, a, b = r.metrics, r.metrics_is, r.metrics_oos
        lines.append(
            f"| {r.name} | {fmt_pct(m.get('cagr'))} | {fmt_num(m.get('sharpe'))} | "
            f"{fmt_num(m.get('sortino'))} | {fmt_pct(m.get('max_drawdown'))} | "
            f"{fmt_pct(m.get('win_rate'))} | {fmt_num(m.get('profit_factor'))} | "
            f"{fmt_num(a.get('sharpe'))} | {fmt_num(b.get('sharpe'))} |"
        )
    lines.append("")
    lines.append("## 5. Comparison & Ranking")
    lines.append("")
    lines.append("Ranking key: **OOS Sharpe first**, then full-sample Sharpe. Quality bar prefers Sharpe>1, MDD>-30%.")
    lines.append("")
    for i, r in enumerate(ranked, 1):
        passes = (
            (r.metrics.get("sharpe") or 0) > 1.0
            and (r.metrics.get("max_drawdown") or -1) > -0.30
            and (r.metrics.get("cagr") or 0) > (ew.metrics.get("cagr") or 0)
        )
        flag = "PASS-ish" if passes else "FAIL quality bar / conditional"
        lines.append(f"{i}. `{r.name}` — {flag}")
    lines.append("")
    lines.append("## 6. Final Recommendations")
    lines.append("")
    top = ranked[:3]
    for i, r in enumerate(top, 1):
        lines.append(f"{i}. **{r.name}** — {r.notes}")
    lines.append("")
    lines.append(
        f"对照：创业板指 Full Sharpe {fmt_num(bench.metrics.get('sharpe'))} / "
        f"算力等权 Full Sharpe {fmt_num(ew.metrics.get('sharpe'))}。"
    )
    lines.append("")
    lines.append("## 7. Deployment Considerations")
    lines.append("")
    lines.append("- 仓位：单票建议 ≤ 组合 25%；策略总暴露可用波动目标（如年化 20–25%）缩放。")
    lines.append("- 交易：月度再平衡策略优先；突破/回撤类注意涨跌停与流动性冲击。")
    lines.append("- 监控：样本池相对创业板指的 60 日相关与拥挤度；广度跌破 40% 降低暴露。")
    lines.append("- 合规：本报告为研究回测，不构成投资建议。")
    lines.append("")
    lines.append("## 8. What Could Go Wrong")
    lines.append("")
    lines.append("- AI 资本开支不及预期 / 出口管制冲击订单 → 动量与突破策略同步失效。")
    lines.append("- 估值拥挤后的风格切换（成长→价值）→ 等权与动量回撤扩大。")
    lines.append("- 过拟合：参数（lookback/topN/breadth 阈值）在小样本主题股上易不稳健。")
    lines.append("- 数据局限：复权行情、停牌、涨跌停未完全建模；实盘滑点可能高于 15bps。")
    lines.append("- 失效条件：OOS 滚动 6 个月夏普 < 0 且相对等权超额转负 → 停用或降杠杆。")
    lines.append("")
    lines.append("## Final self-check")
    lines.append("")
    lines.append("- Edge vs overfitting: 以 OOS 表现排序，不单看全样本最优。")
    lines.append("- Assumptions labeled: costs 15bps one-way; next-day execution; curated universe bias.")
    lines.append("- Missing: fundamentals/options flow, earnings event study, limit-up fills.")
    lines.append("")

    path = OUT_DIR / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    # Save equity curves + metrics json
    eq_df = pd.DataFrame({r.name: r.equity for r in results})
    eq_df.to_csv(OUT_DIR / "equity_curves.csv")
    metrics_payload = {
        r.name: {
            "full": r.metrics,
            "is": r.metrics_is,
            "oos": r.metrics_oos,
            "notes": r.notes,
        }
        for r in results
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "market_snapshot.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    panel, idx, names = load_panel()
    close = pivot_close(panel)
    vol = pivot_volume(panel)
    # Align
    common = close.dropna(how="all").index
    close = close.loc[common]
    vol = vol.reindex(close.index).fillna(0.0)
    snap = market_snapshot(close, names)
    results = run_all(close, vol, idx)
    report = write_report(results, snap, names)
    print(f"\nReport written: {report}")


if __name__ == "__main__":
    main()
