#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-share PCB strategy research for the 2026-10-08 daily run.

Data: Tencent qfq daily bars (ifzq newfqkline). Amount field is万元.
Sample for signals and backtest ends on the last completed session
(2026-09-30). The 2026-10-08 session is intraday and is not used as a bar.

Costs: 15 bps one-way on traded notional (commission + stamp-tax share + slippage).
Execution: signal at close t, weight earns the next session's close-to-close
return. Limit-up buys and limit-down sells are rejected; cash is not reused
to buy names the account cannot fund.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "bars.pkl"

# Pre-registered window. Warmup history is fetched earlier so 2022 is evaluable.
EVAL_START = pd.Timestamp("2022-01-04")
EVAL_END = pd.Timestamp("2026-09-30")  # last completed A-share session before this run
IS_END = pd.Timestamp("2024-12-31")
FETCH_START = "2020-06-01"
COST_BPS = 15.0
COST_BPS_STRESS = 30.0
LIQ_WAN = 5000.0  # 20-day median turnover floor, 万元 = 5e7 CNY

UNIVERSE: Dict[str, Dict[str, str]] = {
    "002463": {"name": "沪电股份", "sub": "AI_PCB"},
    "300476": {"name": "胜宏科技", "sub": "AI_PCB"},
    "002916": {"name": "深南电路", "sub": "AI_PCB"},
    "688183": {"name": "生益电子", "sub": "AI_PCB"},
    "001389": {"name": "广合科技", "sub": "AI_PCB"},
    "603228": {"name": "景旺电子", "sub": "AI_PCB"},
    "301366": {"name": "一博科技", "sub": "AI_PCB"},
    "002938": {"name": "鹏鼎控股", "sub": "FPC"},
    "002384": {"name": "东山精密", "sub": "FPC"},
    "300657": {"name": "弘信电子", "sub": "FPC"},
    "600183": {"name": "生益科技", "sub": "CCL"},
    "688519": {"name": "南亚新材", "sub": "CCL"},
    "603186": {"name": "华正新材", "sub": "CCL"},
    "002636": {"name": "金安国纪", "sub": "CCL"},
    "002815": {"name": "崇达技术", "sub": "MID_PCB"},
    "002913": {"name": "奥士康", "sub": "MID_PCB"},
    "002436": {"name": "兴森科技", "sub": "MID_PCB"},
    "603920": {"name": "世运电路", "sub": "MID_PCB"},
    "000823": {"name": "超声电子", "sub": "MID_PCB"},
    "603328": {"name": "依顿电子", "sub": "MID_PCB"},
    "603936": {"name": "博敏电子", "sub": "MID_PCB"},
    "300814": {"name": "中富电路", "sub": "MID_PCB"},
    "301251": {"name": "威尔高", "sub": "MID_PCB"},
    "600601": {"name": "方正科技", "sub": "MID_PCB"},
    "002579": {"name": "中京电子", "sub": "MID_PCB"},
    "300903": {"name": "科翔股份", "sub": "MID_PCB"},
    "300964": {"name": "本川智能", "sub": "MID_PCB"},
    "301282": {"name": "金禄电子", "sub": "MID_PCB"},
    "603386": {"name": "骏亚科技", "sub": "MID_PCB"},
}

INDEXES = {
    "HS300": "sh000300",
    "CHINEXT": "sz399006",
    "SHCOMP": "sh000001",
}

WINDOWS = [
    ("2020-06-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
    ("2025-01-01", "2026-09-30"),
]


def _tx_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def limit_pct(code: str) -> float:
    """Daily price-limit band. ChiNext/STAR are 20%; main board is 10%."""
    if code.startswith(("300", "301", "688")):
        return 0.195
    return 0.095


def _get_json(url: str, retries: int = 3) -> dict:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    last_exc: Optional[Exception] = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"GET failed: {url} ({last_exc})")


def fetch_qfq_window(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
        f"?param={symbol},day,{start},{end},900,qfq"
    )
    payload = _get_json(url)
    node = (payload.get("data") or {}).get(symbol) or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        return pd.DataFrame()
    recs = []
    for row in rows:
        # date, open, close, high, low, volume(手), _, amplitude, amount(万元), _
        if len(row) < 6:
            continue
        amount = float(row[8]) if len(row) > 8 and row[8] not in ("", None) else np.nan
        recs.append(
            {
                "date": pd.Timestamp(row[0]),
                "open": float(row[1]),
                "close": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "volume": float(row[5]),
                "amount": amount,
                "symbol": symbol,
            }
        )
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.drop_duplicates("date").sort_values("date")


def fetch_symbol(symbol: str) -> pd.DataFrame:
    frames = []
    for a, b in WINDOWS:
        frames.append(fetch_qfq_window(symbol, a, b))
        time.sleep(0.12)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_prices(refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if CACHE.exists() and not refresh:
        panel = pd.read_pickle(CACHE)
    else:
        frames = []
        failures = []
        for code in UNIVERSE:
            symbol = _tx_symbol(code)
            try:
                df = fetch_symbol(symbol)
                if df.empty:
                    failures.append((code, "empty"))
                    continue
                df["code"] = code
                frames.append(df)
                print(f"ok {code} {UNIVERSE[code]['name']} {df['date'].iloc[0].date()}->{df['date'].iloc[-1].date()} n={len(df)}")
            except Exception as exc:  # noqa: BLE001
                failures.append((code, str(exc)))
                print(f"fail {code}: {exc}")
        for name, symbol in INDEXES.items():
            try:
                df = fetch_symbol(symbol)
                if df.empty:
                    failures.append((name, "empty"))
                    continue
                df["code"] = name
                frames.append(df)
                print(f"ok {name} {df['date'].iloc[0].date()}->{df['date'].iloc[-1].date()} n={len(df)}")
            except Exception as exc:  # noqa: BLE001
                failures.append((name, str(exc)))
                print(f"fail {name}: {exc}")
        if not frames:
            raise RuntimeError(f"no data: {failures}")
        panel = pd.concat(frames, ignore_index=True)
        panel.to_pickle(CACHE)
        (OUT / "fetch_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    names = {c: m["name"] for c, m in UNIVERSE.items()}
    return panel, names


def pivot_field(panel: pd.DataFrame, codes: Sequence[str], field: str) -> pd.DataFrame:
    sub = panel[panel["code"].isin(codes)]
    wide = sub.pivot(index="date", columns="code", values=field).sort_index()
    wide = wide[(wide.index >= pd.Timestamp(FETCH_START)) & (wide.index <= EVAL_END)]
    return wide


def month_ends(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period("M")).max())


def week_ends(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period("W-FRI")).max())


def _ffill_rebalance(raw: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill only rows that were explicitly set on a rebalance date."""
    marked = raw.copy()
    # Sentinel: a rebalance row is one we wrote. We store NaN for 'not a rebalance day'
    # and real weights (including zeros) on rebalance days.
    w = marked.ffill().fillna(0.0)
    return w


def liquid_mask(amount: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    med = amount.rolling(20, min_periods=20).median()
    return (med >= LIQ_WAN) & close.notna() & (amount.fillna(0) > 0)


def equal_weight_on(dates_set: Iterable[pd.Timestamp], eligible: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(np.nan, index=eligible.index, columns=eligible.columns)
    for dt in dates_set:
        if dt not in eligible.index:
            continue
        row = eligible.loc[dt].fillna(False)
        n = int(row.sum())
        weights = pd.Series(0.0, index=eligible.columns)
        if n > 0:
            weights.loc[row] = 1.0 / n
        w.loc[dt] = weights
    return _ffill_rebalance(w)


def top_n_weights(
    score: pd.DataFrame,
    eligible: pd.DataFrame,
    rebalance: Sequence[pd.Timestamp],
    top_n: int,
    min_names: int,
) -> pd.DataFrame:
    w = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for dt in rebalance:
        if dt not in score.index:
            continue
        mask = eligible.loc[dt].fillna(False) & score.loc[dt].notna()
        pool = score.loc[dt].where(mask).dropna()
        weights = pd.Series(0.0, index=score.columns)
        if len(pool) >= min_names:
            picks = pool.nlargest(min(top_n, len(pool))).index
            weights.loc[picks] = 1.0 / len(picks)
        w.loc[dt] = weights
    return _ffill_rebalance(w)


def strat_xs_mom(close: pd.DataFrame, eligible: pd.DataFrame, lookback: int = 120, top_n: int = 4) -> pd.DataFrame:
    """Monthly long top-N by lookback return. No absolute-momentum filter."""
    score = close.pct_change(lookback)
    return top_n_weights(score, eligible, month_ends(close.index), top_n, min_names=top_n)


def strat_dual_mom(close: pd.DataFrame, eligible: pd.DataFrame, lookback: int = 120, top_n: int = 3) -> pd.DataFrame:
    """Monthly top-N only if return > 0 and price above its own MA. Else cash."""
    mom = close.pct_change(lookback)
    ma = close.rolling(lookback, min_periods=lookback).mean()
    ok = eligible & (mom > 0) & (close > ma)
    return top_n_weights(mom, ok, month_ends(close.index), top_n, min_names=2)


def strat_lowvol(close: pd.DataFrame, eligible: pd.DataFrame, vol_win: int = 60, top_n: int = 4) -> pd.DataFrame:
    """Monthly lowest-vol names that still have positive 12-month return."""
    vol = close.pct_change().rolling(vol_win, min_periods=vol_win).std()
    trend = close.pct_change(250)
    ok = eligible & (trend > 0)
    # Lower vol is better: rank by negative vol.
    return top_n_weights(-vol, ok, month_ends(close.index), top_n, min_names=3)


def strat_reversal(close: pd.DataFrame, eligible: pd.DataFrame, lookback: int = 5, top_n: int = 4) -> pd.DataFrame:
    """Weekly buy the worst short-horizon losers that remain above MA200."""
    reb = close.pct_change(lookback)
    ma200 = close.rolling(200, min_periods=200).mean()
    ok = eligible & (close > ma200)
    # Most negative return ranks first.
    return top_n_weights(-reb, ok, week_ends(close.index), top_n, min_names=3)


def strat_breakout(close: pd.DataFrame, eligible: pd.DataFrame, entry_n: int = 20, exit_n: int = 10, max_pos: int = 4) -> pd.DataFrame:
    """Donchian breakout, long-only, equal weight, max_pos slots. Signal at close."""
    prior_high = close.rolling(entry_n, min_periods=entry_n).max().shift(1)
    prior_low = close.rolling(exit_n, min_periods=exit_n).min().shift(1)
    mom20 = close.pct_change(20)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    held: List[str] = []
    for dt in close.index:
        still = []
        for code in held:
            px = close.at[dt, code]
            floor = prior_low.at[dt, code]
            if pd.notna(px) and pd.notna(floor) and px < floor:
                continue
            if pd.isna(px):
                continue
            still.append(code)
        held = still
        if len(held) < max_pos:
            cands = []
            for code in close.columns:
                if code in held:
                    continue
                if not bool(eligible.at[dt, code]) if code in eligible.columns else False:
                    continue
                px = close.at[dt, code]
                hi = prior_high.at[dt, code]
                if pd.notna(px) and pd.notna(hi) and px > hi:
                    cands.append((float(mom20.at[dt, code]) if pd.notna(mom20.at[dt, code]) else -np.inf, code))
            cands.sort(reverse=True)
            for _, code in cands[: max_pos - len(held)]:
                held.append(code)
        if held:
            w.loc[dt, held] = 1.0 / len(held)
    return w


def strat_rotate(close: pd.DataFrame, eligible: pd.DataFrame, lookback: int = 60, max_names: int = 5) -> pd.DataFrame:
    """Monthly rotate into the strongest PCB sub-chain; cash if that chain is down."""
    sub_of = {c: UNIVERSE[c]["sub"] for c in close.columns if c in UNIVERSE}
    mom = close.pct_change(lookback)
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for dt in month_ends(close.index):
        if dt not in close.index:
            continue
        weights = pd.Series(0.0, index=close.columns)
        chain_scores = {}
        members: Dict[str, List[str]] = {}
        for code, sub in sub_of.items():
            if code not in close.columns:
                continue
            if not bool(eligible.at[dt, code]):
                continue
            val = mom.at[dt, code]
            if pd.isna(val):
                continue
            members.setdefault(sub, []).append(code)
        for sub, codes in members.items():
            if len(codes) < 2:
                continue
            chain_scores[sub] = float(np.nanmean([mom.at[dt, c] for c in codes]))
        if chain_scores:
            best = max(chain_scores, key=chain_scores.get)
            if chain_scores[best] > 0:
                ranked = sorted(members[best], key=lambda c: mom.at[dt, c], reverse=True)[:max_names]
                if ranked:
                    weights.loc[ranked] = 1.0 / len(ranked)
        w.loc[dt] = weights
    return _ffill_rebalance(w)


def strat_resid(close: pd.DataFrame, eligible: pd.DataFrame, lookback: int = 10, top_n: int = 4) -> pd.DataFrame:
    """Weekly long the names that lagged the liquid equal-weight basket the most."""
    basket = close.pct_change(lookback)
    # Equal-weight residual uses only names with a print that day.
    cross = basket.where(eligible).mean(axis=1)
    resid = basket.sub(cross, axis=0)
    ma60 = close.rolling(60, min_periods=60).mean()
    ok = eligible & (close > ma60)
    return top_n_weights(-resid, ok, week_ends(close.index), top_n, min_names=3)


def strat_sector_timing(close: pd.DataFrame, eligible: pd.DataFrame, ma_win: int = 200) -> pd.DataFrame:
    """Hold the liquid equal-weight basket only while the basket is above its MA."""
    daily_w = eligible.astype(float)
    daily_w = daily_w.div(daily_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # Basket index from close-to-close of the liquid book, rebuilt monthly for trading.
    rets = close.pct_change().fillna(0.0)
    basket_ret = (daily_w.shift(1).fillna(0.0) * rets).sum(axis=1)
    basket = (1.0 + basket_ret).cumprod()
    ma = basket.rolling(ma_win, min_periods=ma_win).mean()
    risk_on = basket > ma
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for dt in month_ends(close.index):
        if dt not in close.index:
            continue
        weights = pd.Series(0.0, index=close.columns)
        if bool(risk_on.loc[dt]):
            row = eligible.loc[dt].fillna(False)
            n = int(row.sum())
            if n > 0:
                weights.loc[row] = 1.0 / n
        w.loc[dt] = weights
    return _ffill_rebalance(w)


def cash_aware_trade(
    held: pd.Series,
    desired: pd.Series,
    day_ret: pd.Series,
    day_vol: pd.Series,
    limits: Dict[str, float],
) -> pd.Series:
    """Trade toward desired weights. Reject limit-up buys and limit-down sells."""
    final = held.copy().astype(float)
    cash = max(0.0, 1.0 - float(held.sum()))
    can_sell = {}
    can_buy = {}
    for code in held.index:
        ret = day_ret.get(code, np.nan)
        vol = day_vol.get(code, np.nan)
        tradable = pd.notna(vol) and vol > 0 and pd.notna(ret)
        lim = limits[code]
        can_sell[code] = bool(tradable and ret > -lim)
        can_buy[code] = bool(tradable and ret < lim)
    for code in held.index:
        target = float(desired.get(code, 0.0))
        prev = float(held.get(code, 0.0))
        if target < prev - 1e-12 and can_sell[code]:
            cash += prev - target
            final[code] = target
        else:
            final[code] = prev
    gaps = {}
    for code in held.index:
        target = float(desired.get(code, 0.0))
        if target > float(final[code]) + 1e-12 and can_buy[code]:
            gaps[code] = target - float(final[code])
    gap_sum = float(sum(gaps.values()))
    if gap_sum > 1e-12 and cash > 1e-12:
        scale = min(1.0, cash / gap_sum)
        for code, gap in gaps.items():
            final[code] = float(final[code]) + gap * scale
            cash -= gap * scale
    # Buys are cash-constrained, so gross should stay <= 1. A tiny numeric
    # overrun is clipped; unsellable names are not force-liquidated.
    total = float(final.sum())
    if total > 1.0 + 1e-6:
        final = final * (1.0 / total)
    return final.clip(lower=0.0)


def run_backtest(
    target: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    limits: Dict[str, float],
    cost_bps: float,
) -> Dict[str, pd.Series]:
    dates = close.index
    rets = close.pct_change()
    target = target.reindex(index=dates, columns=close.columns).fillna(0.0)
    volume = volume.reindex(index=dates, columns=close.columns)
    held = pd.Series(0.0, index=close.columns)
    net_list = []
    gross_list = []
    turn_list = []
    exposure_list = []
    weight_rows = []
    for i, dt in enumerate(dates):
        r = rets.iloc[i].fillna(0.0)
        # Weight held overnight earns today's close-to-close move.
        gross = float((held * r).sum())
        day_ret = rets.iloc[i]
        day_vol = volume.iloc[i]
        desired = target.iloc[i]
        new_held = cash_aware_trade(held, desired, day_ret, day_vol, limits)
        turnover = float((new_held - held).abs().sum())
        cost = turnover * (cost_bps / 10000.0)
        net = gross - cost
        net_list.append(net)
        gross_list.append(gross)
        turn_list.append(turnover)
        exposure_list.append(float(held.sum()))
        weight_rows.append(held.values.copy())
        held = new_held
    idx = dates
    net_s = pd.Series(net_list, index=idx, name="net")
    equity = (1.0 + net_s).cumprod()
    weights = pd.DataFrame(weight_rows, index=idx, columns=close.columns)
    return {
        "net": net_s,
        "gross": pd.Series(gross_list, index=idx),
        "turnover": pd.Series(turn_list, index=idx),
        "exposure": pd.Series(exposure_list, index=idx),
        "equity": equity,
        "weights": weights,
    }


def _slice_metrics(net: pd.Series, equity: pd.Series) -> Dict[str, float]:
    del equity  # equity is rebuilt from the slice so IS/OOS are self-contained
    if net.empty:
        return {}
    # (1+r).cumprod() is terminal wealth if capital is 1 before the first return.
    eq = (1.0 + net.fillna(0.0)).cumprod()
    years = max((net.index[-1] - net.index[0]).days / 365.25, 1e-9)
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1.0) if eq.iloc[-1] > 0 else float("nan")
    vol = float(net.std() * math.sqrt(252)) if len(net) > 2 else float("nan")
    sharpe = float(net.mean() / net.std() * math.sqrt(252)) if net.std() and net.std() > 0 else 0.0
    down = net[net < 0]
    sortino = float(net.mean() / down.std() * math.sqrt(252)) if len(down) > 2 and down.std() > 0 else 0.0
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")
    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    pf = float(gains / losses) if losses > 0 else float("inf")
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
        "profit_factor_daily": pf,
        "n_days": int(len(net)),
        "calmar": float(cagr / abs(max_dd)) if max_dd and max_dd < 0 else float("nan"),
    }


def pack_metrics(result: Dict[str, pd.Series], bench_net: Optional[pd.Series] = None) -> Dict[str, object]:
    net = result["net"]
    net = net[(net.index >= EVAL_START) & (net.index <= EVAL_END)]
    equity = (1.0 + net.fillna(0.0)).cumprod()
    full = _slice_metrics(net, equity)
    is_net = net[net.index <= IS_END]
    oos_net = net[net.index > IS_END]
    full["is"] = _slice_metrics(is_net, equity)
    full["oos"] = _slice_metrics(oos_net, equity)
    yearly = {}
    for year, part in net.groupby(net.index.year):
        yearly[str(int(year))] = _slice_metrics(part, equity)
    full["yearly"] = yearly
    expo = result["exposure"][(result["exposure"].index >= EVAL_START) & (result["exposure"].index <= EVAL_END)]
    turn = result["turnover"][(result["turnover"].index >= EVAL_START) & (result["turnover"].index <= EVAL_END)]
    w = result["weights"]
    w = w[(w.index >= EVAL_START) & (w.index <= EVAL_END)]
    avg_w = w.mean()
    full["avg_exposure"] = float(expo.mean()) if len(expo) else float("nan")
    full["pct_days_invested"] = float((expo > 0.05).mean()) if len(expo) else float("nan")
    full["ann_turnover"] = float(turn.mean() * 252) if len(turn) else float("nan")
    full["avg_weight_top3"] = [
        {"code": c, "avg_weight": float(avg_w[c])} for c in avg_w.sort_values(ascending=False).head(3).index
    ]
    full["max_avg_weight"] = float(avg_w.max()) if len(avg_w) else float("nan")
    if bench_net is not None:
        b = bench_net.reindex(net.index).fillna(0.0)
        active = net - b
        te = float(active.std() * math.sqrt(252)) if active.std() > 0 else float("nan")
        ir = float(active.mean() / active.std() * math.sqrt(252)) if active.std() > 0 else 0.0
        full["tracking_error"] = te
        full["info_ratio_vs_bench"] = ir
        full["corr_vs_bench"] = float(net.corr(b)) if len(net) > 5 else float("nan")
    return full


def buy_hold_index(index_close: pd.Series, eval_index: pd.DatetimeIndex, cost_bps: float) -> Dict[str, pd.Series]:
    px = index_close.reindex(eval_index).ffill()
    rets = px.pct_change().fillna(0.0)
    # Pay the entry cost once on the first session.
    net = rets.copy()
    if len(net):
        net.iloc[0] = net.iloc[0] - cost_bps / 10000.0
    equity = (1.0 + net).cumprod()
    zeros = pd.Series(0.0, index=eval_index)
    return {
        "net": net,
        "gross": rets,
        "turnover": zeros,
        "exposure": pd.Series(1.0, index=eval_index),
        "equity": equity,
        "weights": pd.DataFrame({"IDX": 1.0}, index=eval_index),
    }


def regime_snapshot(close: pd.DataFrame, amount: pd.DataFrame, eligible: pd.DataFrame, indexes: pd.DataFrame) -> dict:
    last = close.index[close.index <= EVAL_END].max()
    ew = eligible.astype(float)
    ew = ew.div(ew.sum(axis=1).replace(0, np.nan), axis=0)
    rets = close.pct_change()
    ew_ret = (ew.shift(1) * rets).sum(axis=1, min_count=1)
    ew_eq = (1.0 + ew_ret.fillna(0.0)).cumprod()

    def trail(series_or_eq_ret: pd.Series, n: int, kind: str = "ret") -> Optional[float]:
        if last not in close.index:
            return None
        if kind == "px":
            px = series_or_eq_ret.dropna()
            if last not in px.index or len(px.loc[:last]) <= n:
                return None
            return float(px.loc[last] / px.loc[:last].iloc[-(n + 1)] - 1.0)
        return None

    def px_ret(px: pd.Series, n: int) -> Optional[float]:
        s = px.dropna()
        s = s[s.index <= last]
        if len(s) <= n:
            return None
        return float(s.iloc[-1] / s.iloc[-(n + 1)] - 1.0)

    def realized_vol(ret: pd.Series, n: int) -> Optional[float]:
        s = ret.dropna()
        s = s[s.index <= last].iloc[-n:]
        if len(s) < n / 2:
            return None
        return float(s.std() * math.sqrt(252))

    sub_stats = {}
    for sub in sorted({m["sub"] for m in UNIVERSE.values()}):
        cols = [c for c, m in UNIVERSE.items() if m["sub"] == sub and c in close.columns]
        if not cols:
            continue
        sub_ew = close[cols].pct_change().mean(axis=1)
        sub_eq = (1.0 + sub_ew.fillna(0.0)).cumprod()
        sub_stats[sub] = {
            "n": len(cols),
            "ret_20": px_ret(sub_eq, 20),
            "ret_60": px_ret(sub_eq, 60),
            "ret_252": px_ret(sub_eq, 252),
            "vol_60": realized_vol(sub_ew, 60),
        }

    ma60 = close.rolling(60).mean().loc[last]
    ma200 = close.rolling(200).mean().loc[last]
    px_last = close.loc[last]
    valid60 = ma60.notna() & px_last.notna()
    valid200 = ma200.notna() & px_last.notna()
    breadth_ma60 = float((px_last[valid60] > ma60[valid60]).mean()) if valid60.any() else None
    breadth_ma200 = float((px_last[valid200] > ma200[valid200]).mean()) if valid200.any() else None
    dd = ew_eq / ew_eq.cummax() - 1.0
    out = {
        "last_completed_session": str(last.date()),
        "n_names": int(close.shape[1]),
        "basket_ret_20": px_ret(ew_eq, 20),
        "basket_ret_60": px_ret(ew_eq, 60),
        "basket_ret_120": px_ret(ew_eq, 120),
        "basket_ret_252": px_ret(ew_eq, 252),
        "basket_vol_20": realized_vol(ew_ret, 20),
        "basket_vol_60": realized_vol(ew_ret, 60),
        "basket_dd_from_peak": float(dd.loc[:last].iloc[-1]),
        "breadth_above_ma60": breadth_ma60,
        "breadth_above_ma200": breadth_ma200,
        "cross_section_std_20d": float(close.pct_change(20).loc[last].std(skipna=True)),
        "median_amount_wan_20d": float(amount.rolling(20).median().loc[last].median()),
        "subchains": sub_stats,
        "indexes": {},
    }
    for name in indexes.columns:
        px = indexes[name].dropna()
        px = px[px.index <= last]
        r = px.pct_change()
        eq = (1.0 + r.fillna(0.0)).cumprod()
        peak_dd = float((eq / eq.cummax() - 1.0).iloc[-1]) if len(eq) else None
        out["indexes"][name] = {
            "last": str(px.index[-1].date()) if len(px) else None,
            "ret_20": px_ret(px, 20),
            "ret_60": px_ret(px, 60),
            "ret_252": px_ret(px, 252),
            "vol_60": realized_vol(r, 60),
            "dd_from_peak": peak_dd,
        }
    return out


def fetch_intraday_snapshot() -> dict:
    """Live quotes for context only. Not used in the backtest."""
    symbols = [_tx_symbol(c) for c in UNIVERSE]
    symbols += list(INDEXES.values())
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("gbk", errors="replace")
    rows = []
    ts = None
    for line in raw.split(";"):
        parts = line.split("~")
        if len(parts) < 33:
            continue
        ts = parts[30]
        try:
            rows.append(
                {
                    "code": parts[2],
                    "name": parts[1],
                    "last": float(parts[3]) if parts[3] else None,
                    "prev_close": float(parts[4]) if parts[4] else None,
                    "pct": float(parts[32]) if parts[32] else None,
                }
            )
        except ValueError:
            continue
    return {"timestamp": ts, "quotes": rows}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def quality_flags(m: dict, bench_total: float) -> dict:
    oos = m.get("oos") or {}
    is_ = m.get("is") or {}
    flags = {
        "beat_bench_total": (m.get("total_return") or -1) > bench_total,
        "sharpe_gt_1": (m.get("sharpe") or 0) > 1.0,
        "oos_sharpe_gt_0_5": (oos.get("sharpe") or 0) > 0.5,
        "oos_positive": (oos.get("total_return") or 0) > 0,
        "is_positive": (is_.get("total_return") or 0) > 0,
        "mdd_within_30": (m.get("max_drawdown") or -1) >= -0.30,
        "not_one_name": (m.get("max_avg_weight") or 1) <= 0.40,
        "invested_enough": (m.get("pct_days_invested") or 0) >= 0.40,
    }
    flags["passes_quality_bar"] = all(flags.values())
    return flags


def main() -> None:
    panel, names = load_prices(refresh=not CACHE.exists())
    codes = [c for c in UNIVERSE if c in set(panel["code"])]
    close = pivot_field(panel, codes, "close")
    volume = pivot_field(panel, codes, "volume").reindex(close.index)
    amount = pivot_field(panel, codes, "amount").reindex(close.index)
    # Drop sessions after the last completed close. Also drop a bar that is
    # exactly EVAL_END+ (already filtered). Keep warmup before EVAL_START.
    eligible = liquid_mask(amount, close).reindex(close.index).fillna(False)
    limits = {c: limit_pct(c) for c in close.columns}
    idx_close = pivot_field(panel, list(INDEXES.keys()), "close")

    builders = {
        "XSMOM": lambda c, e: strat_xs_mom(c, e, 120, 4),
        "DUAL": lambda c, e: strat_dual_mom(c, e, 120, 3),
        "BREAK": lambda c, e: strat_breakout(c, e, 20, 10, 4),
        "REV": lambda c, e: strat_reversal(c, e, 5, 4),
        "LOWVOL": lambda c, e: strat_lowvol(c, e, 60, 4),
        "ROTATE": lambda c, e: strat_rotate(c, e, 60, 5),
        "RESID": lambda c, e: strat_resid(c, e, 10, 4),
        "SECTOR_MA": lambda c, e: strat_sector_timing(c, e, 200),
    }

    # Fair benchmark: monthly rebalanced liquid equal-weight basket.
    bench_w = equal_weight_on(month_ends(close.index), eligible)
    bench = run_backtest(bench_w, close, volume, limits, COST_BPS)
    bench_net = bench["net"]

    results = {"BENCH_EW_LIQ": bench}
    metrics = {}
    for name, fn in builders.items():
        print("run", name)
        w = fn(close, eligible)
        results[name] = run_backtest(w, close, volume, limits, COST_BPS)
        metrics[name] = pack_metrics(results[name], bench_net)

    metrics["BENCH_EW_LIQ"] = pack_metrics(bench, bench_net)
    eval_index = close.index[(close.index >= EVAL_START) & (close.index <= EVAL_END)]
    for iname in idx_close.columns:
        bh = buy_hold_index(idx_close[iname], eval_index, COST_BPS)
        # buy_hold already on eval index; pack_metrics filters again.
        # Rebuild a full-index compatible series.
        net_full = pd.Series(0.0, index=close.index)
        net_full.loc[bh["net"].index] = bh["net"]
        wrapped = {
            "net": net_full,
            "exposure": pd.Series(1.0, index=close.index),
            "turnover": pd.Series(0.0, index=close.index),
            "weights": pd.DataFrame(0.0, index=close.index, columns=close.columns),
            "equity": (1.0 + net_full).cumprod(),
        }
        results[iname] = wrapped
        metrics[iname] = pack_metrics(wrapped, bench_net)

    # Cost stress and a priori parameter neighbors. These do not change the
    # pre-registered specification used for ranking.
    sensitivity = {}
    specs = {
        "XSMOM": [
            ("lb60_n4", lambda: strat_xs_mom(close, eligible, 60, 4)),
            ("lb120_n4", lambda: strat_xs_mom(close, eligible, 120, 4)),
            ("lb180_n4", lambda: strat_xs_mom(close, eligible, 180, 4)),
            ("lb120_n3", lambda: strat_xs_mom(close, eligible, 120, 3)),
            ("lb120_n5", lambda: strat_xs_mom(close, eligible, 120, 5)),
        ],
        "DUAL": [
            ("lb60", lambda: strat_dual_mom(close, eligible, 60, 3)),
            ("lb120", lambda: strat_dual_mom(close, eligible, 120, 3)),
            ("lb180", lambda: strat_dual_mom(close, eligible, 180, 3)),
        ],
        "BREAK": [
            ("e10", lambda: strat_breakout(close, eligible, 10, 5, 4)),
            ("e20", lambda: strat_breakout(close, eligible, 20, 10, 4)),
            ("e55", lambda: strat_breakout(close, eligible, 55, 20, 4)),
        ],
        "REV": [
            ("lb3", lambda: strat_reversal(close, eligible, 3, 4)),
            ("lb5", lambda: strat_reversal(close, eligible, 5, 4)),
            ("lb10", lambda: strat_reversal(close, eligible, 10, 4)),
        ],
        "LOWVOL": [
            ("v20", lambda: strat_lowvol(close, eligible, 20, 4)),
            ("v60", lambda: strat_lowvol(close, eligible, 60, 4)),
            ("v120", lambda: strat_lowvol(close, eligible, 120, 4)),
        ],
        "ROTATE": [
            ("lb20", lambda: strat_rotate(close, eligible, 20, 5)),
            ("lb60", lambda: strat_rotate(close, eligible, 60, 5)),
            ("lb120", lambda: strat_rotate(close, eligible, 120, 5)),
        ],
        "SECTOR_MA": [
            ("ma120", lambda: strat_sector_timing(close, eligible, 120)),
            ("ma200", lambda: strat_sector_timing(close, eligible, 200)),
        ],
        "RESID": [
            ("lb5", lambda: strat_resid(close, eligible, 5, 4)),
            ("lb10", lambda: strat_resid(close, eligible, 10, 4)),
            ("lb20", lambda: strat_resid(close, eligible, 20, 4)),
        ],
    }
    for sname, variants in specs.items():
        sensitivity[sname] = {}
        for label, fn in variants:
            bt = run_backtest(fn(), close, volume, limits, COST_BPS)
            m = pack_metrics(bt, bench_net)
            sensitivity[sname][label] = {
                "sharpe": m["sharpe"],
                "oos_sharpe": (m.get("oos") or {}).get("sharpe"),
                "max_drawdown": m["max_drawdown"],
                "total_return": m["total_return"],
                "cagr": m["cagr"],
            }
        stressed = run_backtest(builders[sname](close, eligible), close, volume, limits, COST_BPS_STRESS)
        sm = pack_metrics(stressed, bench_net)
        sensitivity[sname]["cost30bps"] = {
            "sharpe": sm["sharpe"],
            "oos_sharpe": (sm.get("oos") or {}).get("sharpe"),
            "max_drawdown": sm["max_drawdown"],
            "total_return": sm["total_return"],
            "cagr": sm["cagr"],
        }

    # Concentration: drop the two AI PCB leaders and rerun the pre-registered specs.
    leaders = [c for c in ("002463", "300476") if c in close.columns]
    close_x = close.drop(columns=leaders)
    elig_x = eligible.drop(columns=leaders)
    vol_x = volume.drop(columns=leaders)
    lim_x = {c: limits[c] for c in close_x.columns}
    bench_x = run_backtest(equal_weight_on(month_ends(close_x.index), elig_x), close_x, vol_x, lim_x, COST_BPS)
    drop_leaders = {}
    for sname, fn in builders.items():
        bt = run_backtest(fn(close_x, elig_x), close_x, vol_x, lim_x, COST_BPS)
        m = pack_metrics(bt, bench_x["net"])
        drop_leaders[sname] = {
            "sharpe": m["sharpe"],
            "oos_sharpe": (m.get("oos") or {}).get("sharpe"),
            "max_drawdown": m["max_drawdown"],
            "total_return": m["total_return"],
            "cagr": m["cagr"],
        }

    bench_total = metrics["BENCH_EW_LIQ"]["total_return"]
    flags = {name: quality_flags(metrics[name], bench_total) for name in builders}

    # Correlation of strategy net returns on the eval window.
    corr_df = pd.DataFrame({name: results[name]["net"].reindex(eval_index) for name in list(builders) + ["BENCH_EW_LIQ"]})
    corr = corr_df.corr().round(3)

    regime = regime_snapshot(close, amount, eligible, idx_close)
    try:
        snap = fetch_intraday_snapshot()
    except Exception as exc:  # noqa: BLE001
        snap = {"error": str(exc)}

    # Per-name trailing returns for the report, last completed session.
    last = close.index.max()
    name_rows = []
    for code in close.columns:
        px = close[code].dropna()
        if px.empty:
            continue
        def r(n, series=px):
            if len(series) <= n:
                return None
            return float(series.iloc[-1] / series.iloc[-(n + 1)] - 1.0)
        amt = amount[code].rolling(20).median()
        name_rows.append(
            {
                "code": code,
                "name": names.get(code, code),
                "sub": UNIVERSE[code]["sub"],
                "last_date": str(px.index[-1].date()),
                "ret_20": r(20),
                "ret_60": r(60),
                "ret_252": r(252),
                "dd_252": float((px.iloc[-252:] / px.iloc[-252:].cummax() - 1.0).iloc[-1]) if len(px) >= 20 else None,
                "med_amount_wan_20": float(amt.loc[last]) if last in amt.index and pd.notna(amt.loc[last]) else None,
                "above_ma200": bool(px.iloc[-1] > px.rolling(200).mean().iloc[-1]) if len(px) >= 200 else None,
            }
        )

    coverage = {}
    for code in codes:
        px = close[code].dropna()
        coverage[code] = {
            "name": names[code],
            "sub": UNIVERSE[code]["sub"],
            "start": str(px.index[0].date()) if len(px) else None,
            "end": str(px.index[-1].date()) if len(px) else None,
            "n": int(len(px)),
        }

    payload = {
        "meta": {
            "eval_start": str(EVAL_START.date()),
            "eval_end": str(EVAL_END.date()),
            "is_end": str(IS_END.date()),
            "cost_bps_one_way": COST_BPS,
            "liquidity_floor_wan": LIQ_WAN,
            "n_names_loaded": len(codes),
            "notes": [
                "qfq close from Tencent newfqkline",
                "amount in 万元",
                "2026-10-08 intraday session excluded from backtest",
                "long-only, no leverage, T+1 approximated by next-session return",
            ],
        },
        "coverage": coverage,
        "regime": regime,
        "intraday_snapshot": snap,
        "metrics": metrics,
        "flags": flags,
        "sensitivity": sensitivity,
        "drop_leaders_002463_300476": drop_leaders,
        "corr": corr.to_dict(),
        "names_last": name_rows,
    }
    text = json.dumps(_clean(payload), ensure_ascii=False, indent=2)
    (OUT / "metrics.json").write_text(text, encoding="utf-8")
    print("wrote", OUT / "metrics.json")
    print("--- ranking snapshot ---")
    rows = []
    for name in builders:
        m = metrics[name]
        rows.append(
            (
                m.get("sharpe") or -99,
                name,
                m.get("cagr"),
                m.get("sharpe"),
                m.get("max_drawdown"),
                (m.get("oos") or {}).get("sharpe"),
                flags[name]["passes_quality_bar"],
            )
        )
    for row in sorted(rows, reverse=True):
        print(row)


if __name__ == "__main__":
    main()
