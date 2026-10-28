#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New candidates for the 2026-10-28 research run (pre-registered before any result was seen).

Usage (after run_research.py has filled the cache):
    RESEARCH_DATE=2026-10-28 LAST_BAR=2026-10-27 python3 new_candidates.py

All three rules switch between the CSI300 ETF (510300, dividends added back) and the 10Y treasury
ETF (511260) in the main window 2020-01-02 .. LAST_BAR, and are compared with EW_SB (50/50 month-end).
Each also gets a holdout test on data no earlier report examined: CSI300 price index (sh000300,
no dividends) vs cash earning lagged 3M Shibor. 511260 did not exist before 2017-08, hence cash.

1. Margin_Diverge. The 2026-10-27 report found, after looking at its data, that months where the
   exchange financing balance rose over 20 sessions while CSI300 fell over the same 20 sessions were
   followed by stock-minus-bond returns of -35.3% p.a. (t=-2.31). That cell is data-snooped on
   2020-2026, so the main-window backtest here is in-sample by construction and cannot qualify the
   rule. The decisive test is the 2012-01 .. 2019-12 holdout.
   Rule: month-end t, m = balance(t-1)/balance(t-21)-1, p = CSI300(t-1)/CSI300(t-21)-1.
   If m > 0 and p < 0 hold the defensive asset, otherwise hold equity.
   Holdout pass (pre-registered): diverge-state equity-minus-cash < 0 with t <= -1.0, AND the
   rule's holdout Sharpe-rf beats buy-and-hold index Sharpe-rf. Otherwise eliminated.

2. FX_Timing. Thesis: RMB appreciation vs USD (falling USD/CNY central parity) eases domestic
   liquidity and attracts foreign allocation into large caps; depreciation does the opposite.
   Rule: month-end t, f = USDCNY_fix(t-1)/USDCNY_fix(t-21)-1 (SAFE central parity, AkShare
   `currency_boc_safe`). f < 0 -> 510300, else 511260. One-session lag is conservative (the fixing
   of day t is public at 09:15 of day t).
   Holdout: 2012-01 .. 2019-12 and the post-reform sub-window 2015-09 .. 2019-12.

3. TOM (turn of month). Thesis: month-start fund inflows (salary/pension/insurance allocation,
   new-month fund subscriptions) and month-end liquidity squeezes concentrate equity returns in the
   last session of a month and the first three sessions of the next one.
   Rule: hold 510300 on return days {last session of month, sessions 1..3 of the next month},
   511260 on all other days. The position for day t+1 is set at the close of t from the public
   exchange calendar. Placebo: the same 4-session block on sessions 9..12 of each month.
   Holdout: 2007-01 .. 2019-12 on the index vs cash.
   Mechanism pass: TOM-day minus other-day mean CSI300 return t >= 2.0 over the full 2007-2026 span.

Common: costs 8 bps one-way on turnover (ETF); in the holdout the cash leg is not charged, only the
index leg (|delta equity weight| * 8 bps). Sensitivities are diagnostics only, never selection.
Quality bar is the same as run_research.py (vs CSI300 ETF and vs EW_SB, Sharpe-rf > 1, MDD > -30%,
OOS Sharpe > 0.5).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_research as R  # noqa: E402

OUT = R.OUT
CACHE = R.CACHE
LAST_BAR = R.LAST_BAR
EVAL_START, IS_END, OOS_START = R.EVAL_START, R.IS_END, R.OOS_START
COST = R.ETF_COST_BPS
TD = R.TRADING_DAYS

HOLDOUT = {
    "Margin_Diverge": [("2012-01-04", "2019-12-31")],
    "FX_Timing": [("2012-01-04", "2019-12-31"), ("2015-09-01", "2019-12-31")],
    "TOM": [("2007-01-04", "2019-12-31")],
}


def fetch_index_long(symbol: str) -> pd.Series:
    path = CACHE / f"{symbol}_long.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "close"]]
        df.to_csv(path, index=False)
    s = df.set_index("date")["close"].astype(float).sort_index()
    return s.loc[s.index <= LAST_BAR]


def fetch_usdcny_fix() -> pd.Series:
    path = CACHE / "usdcny_fix.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        import akshare as ak

        raw = ak.currency_boc_safe()
        df = pd.DataFrame({"date": pd.to_datetime(raw["日期"]), "usd": pd.to_numeric(raw["美元"], errors="coerce")})
        df.to_csv(path, index=False)
    s = df.dropna().set_index("date")["usd"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.loc[s.index <= LAST_BAR]


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))


def switch_weights(on: pd.Series, dates: pd.DatetimeIndex, cols: Tuple[str, str]) -> pd.DataFrame:
    """At each date in `dates`, hold cols[0] if on[d] else cols[1]; forward-filled between dates."""
    idx = on.index
    raw = pd.DataFrame(np.nan, index=idx, columns=list(cols))
    for d in dates:
        flag = bool(on.get(d, False))
        raw.loc[d] = [1.0, 0.0] if flag else [0.0, 1.0]
    return raw.ffill().fillna(0.0)


def holdout_net(eq_w: pd.Series, eq_ret: pd.Series, rf: pd.Series) -> pd.Series:
    """Equity weight set at close t earns index return t+1; the rest earns rf. Cost on equity leg only."""
    w = eq_w.reindex(eq_ret.index).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    cost = w.diff().abs().fillna(w.abs()).shift(1).fillna(0.0) * COST / 1e4
    return held * eq_ret.fillna(0.0) + (1 - held) * rf.reindex(eq_ret.index).fillna(0.0) - cost


def state_table(state: pd.Series, spread: pd.Series, labels: Dict[bool, str]) -> List[List[str]]:
    """Annualized mean daily spread by held state (state known at t applies to t+1)."""
    held = state.shift(1)
    rows = []
    for val, lab in labels.items():
        m = held == val
        x = spread[m].dropna()
        ins = x.loc[:IS_END]
        oos = x.loc[OOS_START:]
        rows.append([
            lab, R.fmt_pct(float(m.mean()), 0), R.fmt_pct(x.mean() * TD), R.fmt_num(t_stat(x)),
            R.fmt_pct(ins.mean() * TD) if len(ins) else "n/a", R.fmt_pct(oos.mean() * TD) if len(oos) else "n/a",
        ])
    return rows


def stats_row(name: str, net: pd.Series, turn: Optional[pd.Series], rf: pd.Series, bench: Dict[str, pd.Series]) -> Dict:
    full = R.perf_stats(net, turn)
    oos = R.perf_stats(net.loc[OOS_START:], turn.loc[OOS_START:] if turn is not None else None)
    ins = R.perf_stats(net.loc[:IS_END], turn.loc[:IS_END] if turn is not None else None)
    out = {"full": full, "is": ins, "oos": oos, "sharpe_cash": R.cash_excess_sharpe(net, rf),
           "years": {}}
    for bn, b in bench.items():
        al = pd.concat([net.rename("s"), b.rename("b")], axis=1).dropna()
        out[f"excess_{bn}"] = R.perf_stats(al["s"] - al["b"]).get("cagr", float("nan"))
        yr = {}
        for y in sorted(set(al.index.year)):
            a = al.loc[str(y)]
            yr[str(y)] = float((1 + a["s"]).prod() - (1 + a["b"]).prod())
        out["years"][bn] = yr
    return out


def run() -> None:
    etf_frames = {c: R.fetch_etf_sina(c) for c in ("510300", "511260")}
    for k in etf_frames:
        etf_frames[k] = etf_frames[k].loc[etf_frames[k]["date"] <= LAST_BAR].reset_index(drop=True)
    close, rets, _ = R.panel_from_frames(etf_frames, {c: 0.25 for c in etf_frames})
    divs = {c: R.fetch_etf_dividends(c) for c in etf_frames}
    dy, _ = R.dividend_yield_panel(etf_frames, divs, rets.index)
    rets = rets + dy.reindex_like(rets).fillna(0.0).where(rets.notna())
    rets = rets.mask(rets.abs() > 0.25)
    idx = close.index
    cols = ("510300", "511260")
    spread_sb = rets["510300"] - rets["511260"]

    shibor = R.fetch_shibor_3m()
    idx300 = fetch_index_long("sh000300")
    hidx = idx300.index
    rf_h = (shibor.reindex(hidx.union(shibor.index)).ffill().reindex(hidx).shift(1) / 100.0 / TD).fillna(0.0)
    rf_main = rf_h.reindex(idx).ffill().fillna(0.0)
    eq_h = idx300.pct_change()
    spread_hc = eq_h - rf_h

    margin = R.fetch_market_margin()["total"]
    fx = fetch_usdcny_fix()

    res: Dict[str, Dict] = {}
    tables: Dict[str, str] = {}
    weights: Dict[str, pd.DataFrame] = {}
    signals_now: Dict[str, str] = {}

    def run_main(name: str, w: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        net, turn = R.apply_costs(w, rets[list(cols)], COST)
        return net.loc[EVAL_START:], turn.loc[EVAL_START:]

    ew_sb = R.static_mix(idx, {"510300": 0.5, "511260": 0.5})
    net_ew, turn_ew = run_main("EW_SB", ew_sb)
    bh = pd.DataFrame({"510300": 1.0, "511260": 0.0}, index=idx)
    net_bh, _ = run_main("BH", bh)
    bench = {"EW_SB": net_ew, "BH_CSI300": net_bh}
    res["EW_SB"] = stats_row("EW_SB", net_ew, turn_ew, rf_main, {"BH_CSI300": net_bh})
    res["BH_CSI300"] = stats_row("BH_CSI300", net_bh, None, rf_main, {})

    sens: Dict[str, Dict] = {}

    def add_sens(label: str, w: pd.DataFrame) -> None:
        n, t = run_main(label, w)
        sens[label] = {"full": R.perf_stats(n, t), "oos": R.perf_stats(n.loc[OOS_START:]),
                       "sharpe_cash": R.cash_excess_sharpe(n, rf_main)}

    def holdout_block(name: str, eq_on: pd.Series, dates: pd.DatetimeIndex, state_labels: Dict[bool, str],
                      state_for_table: pd.Series) -> List[List[str]]:
        w = pd.Series(np.nan, index=hidx)
        for d in dates:
            w.loc[d] = 1.0 if bool(eq_on.get(d, False)) else 0.0
        w = w.ffill().fillna(0.0)
        rows = []
        for a, b in HOLDOUT[name]:
            sl = slice(a, b)
            n = holdout_net(w, eq_h, rf_h).loc[sl]
            bhn = holdout_net(pd.Series(1.0, index=hidx), eq_h, rf_h).loc[sl]
            mix = holdout_net(pd.Series(0.5, index=hidx), eq_h, rf_h).loc[sl]
            s, sb, sm = R.perf_stats(n), R.perf_stats(bhn), R.perf_stats(mix)
            st = state_for_table.shift(1).loc[sl]
            sp = spread_hc.loc[sl]
            cells = []
            for val, lab in state_labels.items():
                x = sp[st == val].dropna()
                cells.append(f"{lab} {R.fmt_pct(float((st == val).mean()), 0)} 天 / 年化 {R.fmt_pct(x.mean() * TD)} / t={R.fmt_num(t_stat(x))}")
            rows.append([
                f"{a[:7]}~{b[:7]}", R.fmt_pct(s.get("cagr")), R.fmt_num(R.cash_excess_sharpe(n, rf_h)), R.fmt_pct(s.get("max_dd")),
                R.fmt_pct(sb.get("cagr")), R.fmt_num(R.cash_excess_sharpe(bhn, rf_h)), R.fmt_pct(sb.get("max_dd")),
                R.fmt_pct(sm.get("cagr")), R.fmt_num(R.cash_excess_sharpe(mix, rf_h)),
                R.fmt_pct(float(w.loc[sl].mean()), 0), "；".join(cells),
            ])
            res.setdefault(name, {}).setdefault("holdout", {})[f"{a}~{b}"] = {
                "cagr": s.get("cagr"), "sharpe_cash": R.cash_excess_sharpe(n, rf_h), "mdd": s.get("max_dd"),
                "bh_cagr": sb.get("cagr"), "bh_sharpe_cash": R.cash_excess_sharpe(bhn, rf_h), "bh_mdd": sb.get("max_dd"),
                "mix_sharpe_cash": R.cash_excess_sharpe(mix, rf_h), "eq_share": float(w.loc[sl].mean()),
                "cells": {lab: {"ann": float(sp[st == val].mean() * TD), "t": t_stat(sp[st == val])} for val, lab in state_labels.items()},
            }
        return rows

    hold_headers = ["区间", "规则 CAGR", "规则 Sharpe-rf", "规则 MDD", "持有指数 CAGR", "持有 Sharpe-rf", "持有 MDD",
                    "50/50 CAGR", "50/50 Sharpe-rf", "平均股票仓位", "分状态 股票−现金（未扣费）"]

    # ---------- 1. Margin_Diverge ----------
    def margin_state(series_px: pd.Series, index: pd.DatetimeIndex, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        fin = margin.reindex(index.union(margin.index)).ffill(limit=5).reindex(index).shift(1)
        px = series_px.reindex(index).shift(1)
        m = fin / fin.shift(window) - 1.0
        p = px / px.shift(window) - 1.0
        return m, p

    m_main, p_main = margin_state(close["510300"], idx)
    div_main = (m_main > 0) & (p_main < 0)
    me_main = R.month_ends(idx)
    w = switch_weights(~div_main, me_main, cols)
    weights["Margin_Diverge"] = w
    n, t = run_main("Margin_Diverge", w)
    res["Margin_Diverge"] = {**stats_row("Margin_Diverge", n, t, rf_main, bench), **res.get("Margin_Diverge", {})}
    res["Margin_Diverge"]["net"] = n
    for win in (10, 20, 60):
        mm, pp = margin_state(close["510300"], idx, win)
        add_sens(f"MDiv_w{win}", switch_weights(~((mm > 0) & (pp < 0)), me_main, cols))
    add_sens("MDiv_weekly", switch_weights(~div_main, R.week_ends(idx), cols))
    add_sens("MDiv_price_only(p<0→债)", switch_weights(~(p_main < 0), me_main, cols))
    add_sens("MDiv_placebo(m<0&p<0→债)", switch_weights(~((m_main < 0) & (p_main < 0)), me_main, cols))
    state_me = div_main.where(idx.isin(me_main)).ffill().fillna(False).astype(bool)
    tables["Margin_Diverge_state"] = R.markdown_table(
        ["月末状态", "占评估期交易日", "股票−国债 年化", "t 值", "样本内年化", "样本外年化"],
        state_table(state_me, spread_sb, {True: "融资增、价格跌（持债）", False: "其他（持股）"}))
    m_h, p_h = margin_state(idx300, hidx)
    div_h = (m_h > 0) & (p_h < 0)
    me_h = R.month_ends(hidx)
    st_h = div_h.where(hidx.isin(me_h)).ffill().fillna(False).astype(bool)
    tables["Margin_Diverge_holdout"] = R.markdown_table(
        hold_headers, holdout_block("Margin_Diverge", ~div_h, me_h, {True: "背离（持现金）", False: "其他（持指数）"}, st_h))
    mh = m_h.reindex(me_h).loc["2012":"2019"]
    res["Margin_Diverge"]["holdout_margin_up_share"] = float((mh > 0).mean())
    signals_now["Margin_Diverge"] = (
        f"t−1 口径融资 20 日变化 {m_main.iloc[-1]:.2%}、沪深300ETF 20 日变化 {p_main.iloc[-1]:.2%} → "
        + ("背离，持十年国债ETF" if bool(div_main.iloc[-1]) else "非背离，持沪深300ETF")
        + "（月末才调仓）")

    # ---------- 2. FX_Timing ----------
    def fx_score(index: pd.DatetimeIndex, window: int = 20, lag: int = 1) -> pd.Series:
        f = fx.reindex(index.union(fx.index)).ffill(limit=5).reindex(index).shift(lag)
        return f / f.shift(window) - 1.0

    f_main = fx_score(idx)
    on_fx = f_main < 0
    w = switch_weights(on_fx, me_main, cols)
    weights["FX_Timing"] = w
    n, t = run_main("FX_Timing", w)
    res["FX_Timing"] = {**stats_row("FX_Timing", n, t, rf_main, bench), **res.get("FX_Timing", {})}
    res["FX_Timing"]["net"] = n
    for win in (10, 20, 60):
        add_sens(f"FX_w{win}", switch_weights(fx_score(idx, win) < 0, me_main, cols))
    add_sens("FX_weekly", switch_weights(on_fx, R.week_ends(idx), cols))
    add_sens("FX_lag0", switch_weights(fx_score(idx, 20, 0) < 0, me_main, cols))
    add_sens("FX_inverse", switch_weights(~on_fx, me_main, cols))
    add_sens("FX_price_control(p>0→股)", switch_weights(p_main > 0, me_main, cols))
    st_fx = on_fx.where(idx.isin(me_main)).ffill().fillna(False).astype(bool)
    tables["FX_state"] = R.markdown_table(
        ["月末状态", "占评估期交易日", "股票−国债 年化", "t 值", "样本内年化", "样本外年化"],
        state_table(st_fx, spread_sb, {True: "人民币 20 日升值（持股）", False: "人民币 20 日贬值或持平（持债）"}))
    fme = f_main.reindex(me_main).loc[EVAL_START:]
    pme = p_main.reindex(me_main).loc[EVAL_START:]
    res["FX_Timing"]["corr_fx_price"] = float((-fme).corr(pme))
    res["FX_Timing"]["agree_price"] = float(((fme < 0) == (pme > 0)).mean())
    f_h = fx_score(hidx)
    st_fh = (f_h < 0).where(hidx.isin(me_h)).ffill().fillna(False).astype(bool)
    tables["FX_holdout"] = R.markdown_table(
        hold_headers, holdout_block("FX_Timing", f_h < 0, me_h, {True: "升值（持指数）", False: "贬值（持现金）"}, st_fh))
    signals_now["FX_Timing"] = (
        f"t−1 口径美元兑人民币中间价 20 日变化 {f_main.iloc[-1]:.2%}（{fx.iloc[-1]:.2f}，{fx.index[-1].date()}）→ "
        + ("人民币升值，持沪深300ETF" if bool(on_fx.iloc[-1]) else "人民币未升值，持十年国债ETF") + "（月末才调仓）")

    # ---------- 3. TOM ----------
    def tom_days(index: pd.DatetimeIndex, first: Tuple[int, ...] = (1, 2, 3), last: bool = True) -> pd.Series:
        s = pd.Series(index, index=index)
        g = s.groupby([index.year, index.month])
        k = g.cumcount() + 1
        j = g.cumcount(ascending=False) + 1
        if R._last_period_open(index, "M"):
            # The panel's final month is still open: its last row is not the month's last session.
            cur = (index.year == index[-1].year) & (index.month == index[-1].month)
            j = j.where(~cur, 10**6)
        flag = k.isin(first)
        if last:
            flag = flag | (j == 1)
        return pd.Series(flag.values, index=index)

    def tom_weights(index: pd.DatetimeIndex, ret_day: pd.Series) -> pd.DataFrame:
        # Position at close t is for return day t+1.
        on = ret_day.shift(-1).fillna(False).astype(bool)
        return pd.DataFrame({"510300": on.astype(float), "511260": (~on).astype(float)}, index=index)

    tom_main = tom_days(idx)
    w = tom_weights(idx, tom_main)
    weights["TOM"] = w
    n, t = run_main("TOM", w)
    res["TOM"] = {**stats_row("TOM", n, t, rf_main, bench), **res.get("TOM", {})}
    res["TOM"]["net"] = n
    eq_share = float(w["510300"].loc[EVAL_START:].mean())
    static = R.static_mix(idx, {"510300": round(eq_share, 2), "511260": 1 - round(eq_share, 2)})
    n_st, t_st = run_main("Static", static)
    res["TOM"]["static_share"] = round(eq_share, 2)
    res["TOM"]["static_sharpe_cash"] = R.cash_excess_sharpe(n_st, rf_main)
    res["TOM"]["static_cagr"] = R.perf_stats(n_st).get("cagr")
    res["TOM"]["excess_static"] = R.perf_stats((n - n_st).dropna()).get("cagr")
    add_sens("TOM_-1..+3", w)
    add_sens("TOM_-1..+2", tom_weights(idx, tom_days(idx, (1, 2))))
    add_sens("TOM_+1..+3(no last)", tom_weights(idx, tom_days(idx, (1, 2, 3), False)))
    add_sens("TOM_-1..+5", tom_weights(idx, tom_days(idx, (1, 2, 3, 4, 5))))
    add_sens("TOM_placebo_+9..+12", tom_weights(idx, tom_days(idx, (9, 10, 11, 12), False)))
    add_sens("TOM_0bps", w)
    n0, t0 = R.apply_costs(w, rets[list(cols)], 0.0)
    sens["TOM_0bps"] = {"full": R.perf_stats(n0.loc[EVAL_START:], t0.loc[EVAL_START:]), "oos": R.perf_stats(n0.loc[OOS_START:]),
                        "sharpe_cash": R.cash_excess_sharpe(n0.loc[EVAL_START:], rf_main)}
    tom_h = tom_days(hidx)
    # Mechanism: day-level CSI300 index return on TOM days vs others, by era.
    mech_rows = []
    for a, b in (("2007-01-04", "2019-12-31"), ("2020-01-02", "2023-12-31"), ("2024-01-01", str(LAST_BAR.date())),
                 ("2007-01-04", str(LAST_BAR.date()))):
        r = eq_h.loc[a:b]
        f = tom_h.loc[a:b]
        on_r, off_r = r[f], r[~f]
        diff = on_r.mean() - off_r.mean()
        se = math.sqrt(on_r.var(ddof=1) / len(on_r) + off_r.var(ddof=1) / len(off_r))
        mech_rows.append([f"{a[:7]}~{b[:7]}", str(len(on_r)), f"{on_r.mean() * 1e4:.1f}", f"{off_r.mean() * 1e4:.1f}",
                          f"{diff * 1e4:.1f}", R.fmt_num(diff / se)])
        res["TOM"].setdefault("mech", {})[f"{a}~{b}"] = {"on_bp": on_r.mean() * 1e4, "off_bp": off_r.mean() * 1e4, "t": diff / se}
    tables["TOM_mech"] = R.markdown_table(["区间", "TOM 交易日数", "TOM 日均（bp）", "其他日均（bp）", "差（bp）", "Welch t"], mech_rows)
    eq_on_h = tom_h.shift(-1).fillna(False).astype(bool)
    tables["TOM_holdout"] = R.markdown_table(
        hold_headers, holdout_block("TOM", eq_on_h, hidx, {True: "TOM 日（持指数）", False: "其他日（持现金）"}, eq_on_h))
    nxt = "持有沪深300ETF" if bool(w["510300"].iloc[-1]) else "持有十年国债ETF"
    signals_now["TOM"] = f"{idx[-1].date()} 收盘目标：{nxt}；2026-10-29 收盘切入沪深300ETF（10-30 为 10 月最后交易日），11-04 收盘切回国债。"

    # ---------- summary tables ----------
    names = ["BH_CSI300", "EW_SB", "Margin_Diverge", "FX_Timing", "TOM"]
    main_rows, split_rows, year_rows = [], [], []
    for nm in names:
        s = res[nm]
        f = s["full"]
        main_rows.append([nm, R.fmt_pct(f["total_return"]), R.fmt_pct(f["cagr"]), R.fmt_num(f["sharpe"]), R.fmt_num(s["sharpe_cash"]),
                          R.fmt_num(f["sortino"]), R.fmt_pct(f["max_dd"]), R.fmt_pct(f["win_rate_daily"], 0), R.fmt_pct(f["win_rate_monthly"], 0),
                          R.fmt_num(f["profit_factor"]), R.fmt_num(f.get("ann_turnover"), 1),
                          R.fmt_pct(s.get("excess_BH_CSI300", 0.0)), R.fmt_pct(s.get("excess_EW_SB", float("nan")))])
        split_rows.append([nm, R.fmt_pct(s["is"]["cagr"]), R.fmt_num(s["is"]["sharpe"]), R.fmt_pct(s["is"]["max_dd"]),
                           R.fmt_pct(s["oos"]["cagr"]), R.fmt_num(s["oos"]["sharpe"]), R.fmt_pct(s["oos"]["max_dd"]),
                           R.fmt_pct(s["oos"]["total_return"])])
        if nm in ("Margin_Diverge", "FX_Timing", "TOM"):
            yr = s["years"]["EW_SB"]
            year_rows.append([nm, "EW_SB"] + [R.fmt_pct(yr.get(str(y), float("nan"))) for y in range(2020, 2027)]
                             + [f"{sum(1 for v in yr.values() if v > 0)}/{len(yr)}"])
    tables["main"] = R.markdown_table(
        ["Strategy", "Total", "CAGR", "Sharpe", "Sharpe-rf", "Sortino", "MaxDD", "WinD", "WinM", "PF", "Turn/yr", "超额 vs 沪深300", "超额 vs EW_SB"], main_rows)
    tables["split"] = R.markdown_table(["Strategy", "IS CAGR", "IS Sharpe", "IS MDD", "OOS CAGR", "OOS Sharpe", "OOS MDD", "OOS Total"], split_rows)
    tables["years"] = R.markdown_table(["Strategy", "对照"] + [str(y) for y in range(2020, 2026)] + ["2026（至今）", "正超额年数"], year_rows)
    tables["sens"] = R.markdown_table(
        ["Spec", "Full Sharpe", "Full Sharpe-rf", "Full MDD", "OOS Sharpe", "Turn/yr"],
        [[k, R.fmt_num(v["full"].get("sharpe")), R.fmt_num(v["sharpe_cash"]), R.fmt_pct(v["full"].get("max_dd")),
          R.fmt_num(v["oos"].get("sharpe")), R.fmt_num(v["full"].get("ann_turnover"), 1)] for k, v in sens.items()])

    # Correlation with the watchlist (daily net returns from the main run, if present).
    eq_path = OUT / "equity_curves.csv"
    if eq_path.exists():
        eqc = pd.read_csv(eq_path, index_col=0, parse_dates=True)
        rr = eqc.pct_change()
        new = pd.DataFrame({k: res[k]["net"] for k in ("Margin_Diverge", "FX_Timing", "TOM")})
        joined = pd.concat([new, rr], axis=1, join="inner").dropna()
        peers = [c for c in ("BH_CSI300", "EW_SB", "RiskParity_SBG", "Stock_MomLowVol", "Stock_MarginCrowd") if c in joined.columns]
        tables["corr"] = R.markdown_table(
            ["新候选"] + peers + ["Margin_Diverge", "FX_Timing", "TOM"],
            [[k] + [R.fmt_num(float(joined[k].corr(joined[p]))) for p in peers + ["Margin_Diverge", "FX_Timing", "TOM"]]
             for k in ("Margin_Diverge", "FX_Timing", "TOM")])

    # Quality gate (same definition as run_research.py) + pre-registered extra conditions.
    gate = {}
    for nm in ("Margin_Diverge", "FX_Timing", "TOM"):
        s = res[nm]
        checks = {
            "excess_cagr>0": s["excess_BH_CSI300"] > 0,
            "oos_total>=bh": s["oos"]["total_return"] >= res["BH_CSI300"]["oos"]["total_return"],
            "cash_sharpe>1": s["sharpe_cash"] > 1.0,
            "mdd>-30%": s["full"]["max_dd"] > -0.30,
            "oos_sharpe>0.5": s["oos"]["sharpe"] > 0.5,
            "beats_universe": s["excess_EW_SB"] > 0,
        }
        if nm == "Margin_Diverge":
            h = s["holdout"]["2012-01-04~2019-12-31"]
            cell = h["cells"]["背离（持现金）"]
            checks["holdout_cell_t<=-1"] = cell["t"] <= -1.0
            checks["holdout_beats_bh"] = h["sharpe_cash"] > h["bh_sharpe_cash"]
            checks["main_window_not_snooped"] = False
        if nm == "FX_Timing":
            h = s["holdout"]["2012-01-04~2019-12-31"]
            checks["holdout_beats_bh"] = h["sharpe_cash"] > h["bh_sharpe_cash"]
        if nm == "TOM":
            checks["mech_t>=2"] = s["mech"][f"2007-01-04~{LAST_BAR.date()}"]["t"] >= 2.0
            checks["beats_placebo"] = sens["TOM_-1..+3"]["full"]["sharpe"] > sens["TOM_placebo_+9..+12"]["full"]["sharpe"]
        gate[nm] = checks
    tables["gate"] = "\n".join(
        f"- **{nm}** {'PASS' if all(c.values()) else 'FAIL'} — " + ", ".join(f"{k}:{'Y' if v else 'N'}" for k, v in c.items())
        for nm, c in gate.items())

    for nm in ("Margin_Diverge", "FX_Timing", "TOM"):
        res[nm].pop("net", None)
    out = {"results": res, "sens": sens, "gate": {k: {kk: bool(vv) for kk, vv in v.items()} for k, v in gate.items()},
           "signals_now": signals_now, "last_bar": str(idx[-1].date()),
           "fx_last": [str(fx.index[-1].date()), float(fx.iloc[-1])], "margin_last": [str(margin.index[-1].date()), float(margin.iloc[-1])]}
    (OUT / "new_candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    md = []
    for k, v in tables.items():
        md.append(f"### {k}\n\n{v}\n")
    md.append("### signals_now\n\n" + "\n".join(f"- {k}: {v}" for k, v in signals_now.items()))
    (OUT / "new_candidates.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    run()
