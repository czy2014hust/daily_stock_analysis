#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-share ETF strategy research for the 2026-10-04 daily run.

Research date is a Sunday inside the National Day market closure
(SSE: closed 2026-10-01 through 2026-10-07). All prices stop at the
last session, 2026-09-30. Signals are pre-registered literature defaults,
not searched to maximize the sample.

Execution model
- Signal at close t, earn asset return on t+1 (next close).
- One-way cost is charged on the sum of absolute weight changes
  (sell leg + buy leg each pay the one-way rate).
- Long only, weights sum to 1. Residual sleeve is the government-bond ETF.
- No leverage, no short sale, no borrowed cash.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
CACHE = OUT / "prices"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# Pre-registered cost. Liquid A-share ETFs pay no stamp tax.
# 8 bps one-way ≈ commission + half spread + small impact on names with
# tens of millions RMB of daily turnover. 15 bps is the stress case.
COST_ONE_WAY = 0.0008
COST_STRESS = 0.0015
EVAL_START = pd.Timestamp("2019-03-01")
IS_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
ASOF = pd.Timestamp("2026-09-30")
TRADING_DAYS = 252

# code -> Chinese name. Bond is the defensive residual, not an alpha sleeve.
UNIVERSE = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "512100": "中证1000ETF",
    "510050": "上证50ETF",
    "510880": "红利ETF",
    "518880": "黄金ETF",
    "511010": "国债ETF",
    "512880": "证券ETF",
    "512800": "银行ETF",
    "512010": "医药ETF",
    "159928": "消费ETF",
    "512660": "军工ETF",
    "512400": "有色ETF",
    "512690": "酒ETF",
    "512480": "半导体ETF",
    "512170": "医疗ETF",
}

EQUITY_BASKET = ["510300", "510500", "159915", "512100", "510050", "510880"]
DEFENSIVE_ASSETS = ["518880"]  # selectable risk assets; bond is the residual
BOND = "511010"
BENCHMARK = "510300"
SECTORS = ["512880", "512800", "512010", "159928", "512660", "512400", "512690", "512480", "512170"]
INDICES = {
    "sh000001": "上证指数",
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sh000852": "中证1000",
    "sh000016": "上证50",
    "sh000688": "科创50",
    "sz399006": "创业板指",
}

SLICES = {
    "FULL": ("2019-03-01", "2026-09-30"),
    "IS": ("2019-03-01", "2023-12-31"),
    "OOS": ("2024-01-01", "2026-09-30"),
    "2019-2020": ("2019-03-01", "2020-12-31"),
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026YTD": ("2026-01-01", "2026-09-30"),
}


def _prefix(code: str) -> str:
    return ("sh" if code[0] in "569" else "sz") + code


def fetch_etf(code: str) -> pd.DataFrame:
    """Tencent/AkShare qfq daily bars. Cached locally after the first pull."""
    import akshare as ak

    cache = CACHE / f"{code}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        return df.sort_values("date")
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            raw = ak.stock_zh_a_hist_tx(
                symbol=_prefix(code),
                start_date="20160101",
                end_date="20261004",
                adjust="qfq",
            )
            df = raw.rename(columns={"turnover": "turnover_rate"}).copy()
            df["date"] = pd.to_datetime(df["date"])
            for col in ("open", "high", "low", "close", "volume", "amount"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            df.to_csv(cache, index=False)
            return df
        except Exception as exc:  # network flake; retry then surface
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {code}: {last_err}")


def fetch_index(symbol: str) -> pd.DataFrame:
    import akshare as ak

    cache = CACHE / f"idx_{symbol}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).sort_values("date")
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df.to_csv(cache, index=False)
    return df


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    frames = []
    amounts = []
    skipped = {}
    for code in UNIVERSE:
        try:
            df = fetch_etf(code)
        except Exception as exc:
            skipped[code] = str(exc)
            continue
        px = df.set_index("date")["close"].rename(code)
        amt = df.set_index("date")["amount"].rename(code) if "amount" in df.columns else None
        frames.append(px)
        if amt is not None:
            amounts.append(amt)
        print(f"loaded {code} rows={px.dropna().shape[0]} last={px.dropna().index.max().date()}", flush=True)
        time.sleep(0.2)
    close = pd.concat(frames, axis=1).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    amount = pd.concat(amounts, axis=1).sort_index() if amounts else pd.DataFrame(index=close.index)
    amount = amount.reindex(close.index)
    return close, amount, skipped


def month_ends(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period("M")).max())


def week_ends(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period("W-FRI")).max())


def _empty_weights(index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)


def _assign(row: Dict[str, float], columns: List[str]) -> Optional[pd.Series]:
    """Map a target dict onto the full book. Empty dict means 'not ready'."""
    if not row:
        return None
    s = pd.Series(0.0, index=columns)
    for k, v in row.items():
        if k not in s.index:
            continue
        s[k] = v
    total = float(s.sum())
    if total <= 0:
        return None
    if abs(total - 1.0) > 1e-8:
        s = s / total
    return s


def build_scheduled(
    index: pd.DatetimeIndex,
    columns: List[str],
    rebalance_dates: List[pd.Timestamp],
    target_fn: Callable[[pd.Timestamp], Dict[str, float]],
) -> pd.DataFrame:
    w = _empty_weights(index, columns)
    reb = set(pd.Timestamp(d) for d in rebalance_dates)
    last: Optional[pd.Series] = None
    for dt in index:
        if dt in reb:
            assigned = _assign(target_fn(dt), columns)
            if assigned is not None:
                last = assigned
        if last is not None:
            w.loc[dt] = last
    return w


def backtest(weights: pd.DataFrame, rets: pd.DataFrame, cost: float) -> pd.Series:
    """Next-close execution. Cost = sum(|Δw|) * one-way rate.

    Weight dated t is chosen at close t and earns ret[t+1].
    The trade that established today's weight is deducted from today's P&L,
    so the entry cost is not dropped on the first live day.
    """
    cols = [c for c in weights.columns if c in rets.columns]
    w = weights[cols].copy()
    r = rets[cols].reindex(w.index).fillna(0.0)
    entered = w.notna().any(axis=1)
    w = w.where(entered, 0.0)
    w_prev = w.shift(1).fillna(0.0)
    w_prev2 = w.shift(2).fillna(0.0)
    gross = (w_prev * r).sum(axis=1)
    # Trade at close t-1 moves w[t-2] -> w[t-1]; charge it on day t.
    turnover_notional = (w_prev - w_prev2).abs().sum(axis=1)
    live = entered.shift(1, fill_value=False)
    net = gross - turnover_notional * cost
    net = net.where(live, np.nan)
    return net


def perf(returns: pd.Series, rf: Optional[pd.Series] = None) -> dict:
    r = returns.dropna()
    if len(r) < 20:
        return {"n": int(len(r)), "error": "too_few_observations"}
    years = len(r) / TRADING_DAYS
    total = float((1.0 + r).prod() - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = float((r.mean() * TRADING_DAYS) / vol) if vol > 0 else np.nan
    # Sortino downside deviation versus 0. Positive days contribute 0.
    down_dev = float(np.sqrt(np.mean(np.minimum(r.to_numpy(dtype=float), 0.0) ** 2)) * math.sqrt(TRADING_DAYS))
    sortino = float((r.mean() * TRADING_DAYS) / down_dev) if down_dev > 0 else np.nan
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    maxdd = float(dd.min())
    # Compound daily simple returns inside each month. Do not add 1 twice.
    monthly = r.groupby(r.index.to_period("M")).apply(lambda x: float((1.0 + x).prod() - 1.0))
    gains = float(monthly[monthly > 0].sum())
    losses = float(monthly[monthly < 0].sum())
    profit_factor = float(gains / abs(losses)) if losses < 0 else np.nan
    win_rate = float((monthly > 0).mean()) if len(monthly) else np.nan
    excess_sharpe = None
    if rf is not None:
        ex = (r - rf.reindex(r.index).fillna(0.0)).dropna()
        ex_vol = float(ex.std(ddof=1) * math.sqrt(TRADING_DAYS))
        excess_sharpe = float((ex.mean() * TRADING_DAYS) / ex_vol) if ex_vol > 0 else np.nan
    return {
        "n": int(len(r)),
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe_rf0": sharpe,
        "excess_sharpe_vs_bond": excess_sharpe,
        "sortino_rf0": sortino,
        "max_drawdown": maxdd,
        "calmar": float(cagr / abs(maxdd)) if maxdd < 0 else np.nan,
        "monthly_win_rate": win_rate,
        "monthly_profit_factor": profit_factor,
    }


def slice_returns(r: pd.Series, start: str, end: str) -> pd.Series:
    return r.loc[pd.Timestamp(start):pd.Timestamp(end)]


def latest_holdings(weights: pd.DataFrame, names: Dict[str, str]) -> List[dict]:
    w = weights.dropna(how="all")
    if w.empty:
        return []
    row = w.iloc[-1].fillna(0.0)
    out = []
    for code, weight in row[row > 0.001].sort_values(ascending=False).items():
        out.append({"code": code, "name": names.get(code, code), "weight": float(weight)})
    return out


def index_regime(index_close: pd.DataFrame) -> dict:
    asof = index_close.index.max()
    stats = {"asof": str(asof.date()), "indices": {}}
    for col in index_close.columns:
        s = index_close[col].dropna()
        if s.empty:
            continue
        px = float(s.iloc[-1])

        def _ret(days: int) -> Optional[float]:
            if len(s) <= days:
                return None
            return float(s.iloc[-1] / s.iloc[-1 - days] - 1.0)

        ytd_base = s[s.index < pd.Timestamp("2026-01-01")]
        q3_base = s[s.index <= pd.Timestamp("2026-06-30")]
        ma200 = s.rolling(200).mean().iloc[-1]
        peak252 = s.tail(252).max()
        vol20 = float(s.pct_change().tail(20).std() * math.sqrt(TRADING_DAYS))
        stats["indices"][col] = {
            "name": INDICES.get(col, col),
            "last": px,
            "ret_5d": _ret(5),
            "ret_20d": _ret(20),
            "ret_60d": _ret(60),
            "ret_252d": _ret(252),
            "ytd": float(px / ytd_base.iloc[-1] - 1.0) if len(ytd_base) else None,
            "q3_2026": float(px / q3_base.iloc[-1] - 1.0) if len(q3_base) else None,
            "dist_ma200": float(px / ma200 - 1.0) if pd.notna(ma200) and ma200 else None,
            "dd_252d": float(px / peak252 - 1.0) if peak252 else None,
            "vol_20d": vol20,
        }
    return stats


def try_fund_flow() -> dict:
    """Best-effort industry flow. Holiday sessions often fail; that is a gap, not a fill."""
    import akshare as ak
    import signal

    def _timeout(_signum, _frame):
        raise TimeoutError("fund flow timed out")

    old = signal.signal(signal.SIGALRM, _timeout)
    try:
        for symbol in ("即时", "5日排行", "10日排行"):
            signal.alarm(25)
            try:
                df = ak.stock_fund_flow_industry(symbol=symbol)
            except Exception:
                continue
            finally:
                signal.alarm(0)
            if df is None or df.empty:
                continue
            keep = [c for c in df.columns if c in ("行业", "行业-涨跌幅", "净额", "流入资金", "领涨股")]
            top = df.head(8)
            bot = df.tail(5)
            return {
                "symbol": symbol,
                "columns": [str(c) for c in df.columns],
                "top": top[keep].to_dict(orient="records") if keep else top.head(8).to_dict(orient="records"),
                "bottom": bot[keep].to_dict(orient="records") if keep else bot.to_dict(orient="records"),
                "note": "Snapshot from the vendor's latest published industry-flow table. On a holiday this can be the 2026-09-30 print or a stale page; it is context, not a backtest input.",
            }
    finally:
        signal.signal(signal.SIGALRM, old)
        signal.alarm(0)
    return {"error": "industry fund-flow endpoint unavailable on this run"}


# --- strategy targets (causal; only prices at or before dt) ---

def _valid(px: pd.DataFrame, dt: pd.Timestamp, codes: List[str], lookback: int) -> List[str]:
    ok = []
    for code in codes:
        hist = px[code].loc[:dt].dropna()
        if len(hist) > lookback and pd.notna(hist.iloc[-1]):
            ok.append(code)
    return ok


def target_trend(px: pd.DataFrame, dt: pd.Timestamp, window: int = 200) -> Dict[str, float]:
    hist = px[BENCHMARK].loc[:dt].dropna()
    if len(hist) <= window:
        return {}
    ma = hist.iloc[-window:].mean()
    if hist.iloc[-1] > ma:
        return {BENCHMARK: 1.0}
    return {BOND: 1.0}


def target_ts_basket(px: pd.DataFrame, dt: pd.Timestamp, window: int = 200) -> Dict[str, float]:
    picked = []
    for code in _valid(px, dt, EQUITY_BASKET, window):
        hist = px[code].loc[:dt].dropna()
        if hist.iloc[-1] > hist.iloc[-window:].mean():
            picked.append(code)
    if not picked:
        return {BOND: 1.0}
    w = 1.0 / len(picked)
    out = {code: w for code in picked}
    return out


def target_dual_mom(
    px: pd.DataFrame,
    dt: pd.Timestamp,
    lookback: int = 252,
    skip: int = 21,
) -> Dict[str, float]:
    cands = _valid(px, dt, EQUITY_BASKET + DEFENSIVE_ASSETS, lookback)
    scores = {}
    for code in cands:
        hist = px[code].loc[:dt].dropna()
        if len(hist) <= lookback:
            continue
        recent = hist.iloc[-skip] if skip else hist.iloc[-1]
        past = hist.iloc[-lookback]
        if past <= 0 or pd.isna(recent):
            continue
        scores[code] = recent / past - 1.0
    if not scores:
        return {}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return {best: 1.0}
    return {BOND: 1.0}


def target_sector_mom(
    px: pd.DataFrame,
    dt: pd.Timestamp,
    sectors: List[str],
    lookback: int = 126,
    top_n: int = 3,
) -> Dict[str, float]:
    scores = {}
    for code in _valid(px, dt, sectors, lookback):
        hist = px[code].loc[:dt].dropna()
        past = hist.iloc[-lookback]
        if past <= 0:
            continue
        scores[code] = hist.iloc[-1] / past - 1.0
    positive = {k: v for k, v in scores.items() if v > 0}
    if not positive:
        return {BOND: 1.0}
    ranked = sorted(positive, key=positive.get, reverse=True)[:top_n]
    w = 1.0 / len(ranked)
    return {code: w for code in ranked}


def target_reversal(
    px: pd.DataFrame,
    dt: pd.Timestamp,
    sectors: List[str],
    window: int = 5,
    bottom_n: int = 3,
) -> Dict[str, float]:
    scores = {}
    for code in _valid(px, dt, sectors, window):
        hist = px[code].loc[:dt].dropna()
        past = hist.iloc[-window]
        if past <= 0:
            continue
        scores[code] = hist.iloc[-1] / past - 1.0
    if len(scores) < bottom_n:
        return {}
    ranked = sorted(scores, key=scores.get)[:bottom_n]
    w = 1.0 / len(ranked)
    return {code: w for code in ranked}


def target_low_vol(
    px: pd.DataFrame,
    dt: pd.Timestamp,
    window: int = 60,
    bottom_n: int = 3,
) -> Dict[str, float]:
    scores = {}
    for code in _valid(px, dt, EQUITY_BASKET, window):
        hist = px[code].loc[:dt].dropna().pct_change().dropna()
        if len(hist) < window:
            continue
        scores[code] = float(hist.iloc[-window:].std())
    if len(scores) < bottom_n:
        return {}
    ranked = sorted(scores, key=scores.get)[:bottom_n]
    w = 1.0 / len(ranked)
    return {code: w for code in ranked}


def target_pair(
    px: pd.DataFrame,
    dt: pd.Timestamp,
    state: dict,
    window: int = 60,
    entry: float = 2.0,
    exit_: float = 0.5,
) -> Dict[str, float]:
    a = px[BENCHMARK].loc[:dt].dropna()
    b = px["510500"].loc[:dt].dropna()
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(joined) <= window:
        return {}
    spread = np.log(joined["a"]) - np.log(joined["b"])
    hist = spread.iloc[-window:]
    mu = float(hist.mean())
    sd = float(hist.std(ddof=1))
    if sd <= 0:
        return {}
    z = float((spread.iloc[-1] - mu) / sd)
    cur = state.get("pos", 0)  # +1: 300 cheap -> hold 300; -1: 300 rich -> hold 500; 0: 50/50
    if z >= entry:
        cur = -1
    elif z <= -entry:
        cur = 1
    elif abs(z) <= exit_:
        cur = 0
    state["pos"] = cur
    state["z"] = z
    if cur > 0:
        return {BENCHMARK: 1.0}
    if cur < 0:
        return {"510500": 1.0}
    return {BENCHMARK: 0.5, "510500": 0.5}


def target_vol(
    px: pd.DataFrame,
    dt: pd.Timestamp,
    window: int = 20,
    target_vol: float = 0.15,
) -> Dict[str, float]:
    hist = px[BENCHMARK].loc[:dt].dropna().pct_change().dropna()
    if len(hist) < window:
        return {}
    rv = float(hist.iloc[-window:].std(ddof=1) * math.sqrt(TRADING_DAYS))
    if rv <= 0:
        return {}
    w = min(1.0, target_vol / rv)
    return {BENCHMARK: w, BOND: 1.0 - w}


def target_donchian(
    px: pd.DataFrame,
    dt: pd.Timestamp,
    state: dict,
    entry_n: int = 20,
    exit_n: int = 10,
) -> Dict[str, float]:
    hist = px[BENCHMARK].loc[:dt].dropna()
    if len(hist) <= entry_n:
        return {}
    px_t = float(hist.iloc[-1])
    prior_high = float(hist.iloc[-entry_n - 1:-1].max()) if len(hist) > entry_n else np.nan
    prior_low = float(hist.iloc[-exit_n - 1:-1].min()) if len(hist) > exit_n else np.nan
    cur = state.get("pos", 0)
    if px_t > prior_high:
        cur = 1
    elif px_t < prior_low:
        cur = 0
    state["pos"] = cur
    if cur == 1:
        return {BENCHMARK: 1.0}
    return {BOND: 1.0}


def build_all(close: pd.DataFrame, sectors: List[str]) -> Dict[str, pd.DataFrame]:
    columns = list(close.columns)
    index = close.index
    m_ends = month_ends(index)
    w_ends = week_ends(index)

    books = {
        "S1_trend_300": build_scheduled(index, columns, m_ends, lambda dt: target_trend(close, dt, 200)),
        "S2_ts_basket": build_scheduled(index, columns, m_ends, lambda dt: target_ts_basket(close, dt, 200)),
        "S3_dual_mom": build_scheduled(index, columns, m_ends, lambda dt: target_dual_mom(close, dt, 252, 21)),
        "S4_sector_mom": build_scheduled(
            index, columns, m_ends, lambda dt: target_sector_mom(close, dt, sectors, 126, 3)
        ),
        "S5_reversal": build_scheduled(
            index, columns, w_ends, lambda dt: target_reversal(close, dt, sectors, 5, 3)
        ),
        "S6_low_vol": build_scheduled(index, columns, m_ends, lambda dt: target_low_vol(close, dt, 60, 3)),
        "S8_vol_target": build_scheduled(index, columns, w_ends, lambda dt: target_vol(close, dt, 20, 0.15)),
    }

    pair_state: dict = {"pos": 0}
    books["S7_pair_300_500"] = build_scheduled(
        index, columns, list(index), lambda dt: target_pair(close, dt, pair_state, 60, 2.0, 0.5)
    )
    don_state: dict = {"pos": 0}
    books["S9_donchian"] = build_scheduled(
        index, columns, list(index), lambda dt: target_donchian(close, dt, don_state, 20, 10)
    )

    # Benchmarks use the same engine so costs are comparable.
    books["BH_300"] = build_scheduled(index, columns, m_ends[:1], lambda dt: {BENCHMARK: 1.0})
    # Buy and hold should stay invested after the first month-end on/after eval.
    # Using only the first month-end in the whole history would enter in 2016.
    # That is fine: eval window slices later, and the entry cost is before IS.
    books["BH_60_40"] = build_scheduled(
        index, columns, m_ends, lambda dt: {BENCHMARK: 0.6, BOND: 0.4}
    )
    n_eq = len([c for c in EQUITY_BASKET if c in close.columns])
    books["BH_EW_equity"] = build_scheduled(
        index,
        columns,
        m_ends,
        lambda dt: {c: 1.0 / n_eq for c in EQUITY_BASKET if c in close.columns},
    )
    return books


def sensitivity(close: pd.DataFrame, rets: pd.DataFrame, rf: pd.Series, sectors: List[str]) -> dict:
    """Neighbor parameters. Selection itself stays on the pre-registered set."""
    columns = list(close.columns)
    index = close.index
    m_ends = month_ends(index)
    w_ends = week_ends(index)
    specs = []

    def add(name: str, weights: pd.DataFrame) -> None:
        net = backtest(weights, rets, COST_ONE_WAY)
        oos = perf(slice_returns(net, "2024-01-01", "2026-09-30"), rf)
        specs.append({
            "name": name,
            "oos_sharpe_rf0": oos.get("sharpe_rf0"),
            "oos_cagr": oos.get("cagr"),
            "oos_max_drawdown": oos.get("max_drawdown"),
        })

    for window in (150, 200, 250):
        add(f"S1_ma_{window}", build_scheduled(index, columns, m_ends, lambda dt, window=window: target_trend(close, dt, window)))
    for lookback in (189, 252, 315):
        add(
            f"S3_lb_{lookback}",
            build_scheduled(index, columns, m_ends, lambda dt, lookback=lookback: target_dual_mom(close, dt, lookback, 21)),
        )
    for lookback, top_n in ((63, 3), (126, 3), (189, 3), (126, 2)):
        add(
            f"S4_lb_{lookback}_top{top_n}",
            build_scheduled(
                index, columns, m_ends,
                lambda dt, lookback=lookback, top_n=top_n: target_sector_mom(close, dt, sectors, lookback, top_n),
            ),
        )
    for window in (3, 5, 10):
        add(
            f"S5_win_{window}",
            build_scheduled(index, columns, w_ends, lambda dt, window=window: target_reversal(close, dt, sectors, window, 3)),
        )
    for window, target in ((20, 0.10), (20, 0.15), (20, 0.20), (10, 0.15), (40, 0.15)):
        add(
            f"S8_w{window}_tv{int(target*100)}",
            build_scheduled(index, columns, w_ends, lambda dt, window=window, target=target: target_vol(close, dt, window, target)),
        )
    return {"oos_neighbors": specs}


def quality_flags(row: dict) -> dict:
    """Skill bar, applied to after-cost numbers. All must be true to 'pass'."""
    full = row["slices"]["FULL"]
    oos = row["slices"]["OOS"]
    ins = row["slices"]["IS"]
    y2022 = row["slices"].get("2022", {})
    flags = {
        "beats_bh_oos_sharpe": _gt(oos.get("sharpe_rf0"), row["bh_oos_sharpe"]),
        "beats_bh_full_sharpe": _gt(full.get("sharpe_rf0"), row["bh_full_sharpe"]),
        "sharpe_full_gt_1": _gt(full.get("sharpe_rf0"), 1.0),
        "maxdd_full_better_than_30pct": _gt(full.get("max_drawdown"), -0.30),
        "is_sharpe_positive": _gt(ins.get("sharpe_rf0"), 0.0),
        "oos_sharpe_positive": _gt(oos.get("sharpe_rf0"), 0.0),
        "bear_2022_not_catastrophic": _gt(y2022.get("max_drawdown"), -0.25) if y2022 else False,
        "not_single_name_always_on": row["avg_equity_weight"] < 0.98 or row["n_assets_used"] > 1,
    }
    flags["passes_quality_bar"] = all(flags.values())
    return flags


def _gt(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, float) and math.isnan(a):
        return False
    if isinstance(b, float) and math.isnan(b):
        return False
    return a > b


def main() -> None:
    import akshare as ak  # noqa: F401  (import error surfaces here)

    close, amount, skipped = load_panels()
    close = close.loc[:ASOF]
    # Fill only short vendor gaps. Do not paint pre-listing history.
    close = close.ffill(limit=5)
    amount = amount.reindex(close.index)
    # Require a real price path. Drop assets that are too short for the IS window.
    keep = [c for c in close.columns if close[c].dropna().shape[0] > 400]
    close = close[keep]
    amount = amount[[c for c in keep if c in amount.columns]]

    index_frames = []
    index_errors = {}
    for symbol in INDICES:
        try:
            df = fetch_index(symbol)
            index_frames.append(df.set_index("date")["close"].rename(symbol))
        except Exception as exc:
            index_errors[symbol] = str(exc)
    index_close = pd.concat(index_frames, axis=1).sort_index() if index_frames else pd.DataFrame()
    index_close = index_close.loc[:ASOF]

    sectors = [c for c in SECTORS if c in close.columns and close[c].dropna().index.min() <= pd.Timestamp("2018-06-01")]
    # Semiconductor / liquor often list in 2019. Keep them in a separate long-history check.
    sectors_extended = [c for c in SECTORS if c in close.columns]

    rets = close.pct_change()
    # Bond return is the excess-Sharpe funding leg. If missing, rf = 0 and we say so.
    rf = rets[BOND] if BOND in rets.columns else pd.Series(0.0, index=rets.index)

    books = build_all(close, sectors if len(sectors) >= 5 else sectors_extended)
    curves = {}
    summaries = {}
    for name, weights in books.items():
        net = backtest(weights, rets, COST_ONE_WAY)
        net_stress = backtest(weights, rets, COST_STRESS)
        net_lag = backtest(weights.shift(1), rets, COST_ONE_WAY)  # extra session of delay
        curves[name] = net
        sliced = {}
        for label, (a, b) in SLICES.items():
            sliced[label] = perf(slice_returns(net, a, b), rf)
        w_live = weights.loc[EVAL_START:ASOF]
        equity_cols = [c for c in EQUITY_BASKET + sectors_extended + DEFENSIVE_ASSETS if c in w_live.columns]
        occupancy = {}
        ann_turnover = None
        if len(w_live.dropna(how="all")) == 0:
            avg_eq = None
            n_assets = 0
        else:
            filled = w_live.fillna(0.0)
            avg_eq = float(filled[equity_cols].sum(axis=1).mean()) if equity_cols else 0.0
            used = (filled > 0.001).any()
            n_assets = int(used.sum())
            means = filled.mean().sort_values(ascending=False)
            occupancy = {
                code: float(wt)
                for code, wt in means.items()
                if wt > 0.005
            }
            deltas = (filled - filled.shift(1).fillna(0.0)).abs().sum(axis=1)
            years = max(len(filled) / TRADING_DAYS, 1e-6)
            # sum(|Δw|) / 2 is the one-way portfolio turnover (sell leg).
            ann_turnover = float(deltas.sum() / 2.0 / years)
        summaries[name] = {
            "slices": sliced,
            "stress_15bps_full_sharpe": perf(slice_returns(net_stress, "2019-03-01", "2026-09-30"), rf).get("sharpe_rf0"),
            "stress_15bps_oos_sharpe": perf(slice_returns(net_stress, "2024-01-01", "2026-09-30"), rf).get("sharpe_rf0"),
            "lag1_oos_sharpe": perf(slice_returns(net_lag, "2024-01-01", "2026-09-30"), rf).get("sharpe_rf0"),
            "avg_equity_or_gold_weight": avg_eq,
            "n_assets_used": n_assets,
            "avg_weight_by_asset": occupancy,
            "annual_one_way_turnover": ann_turnover,
            "latest_holdings": latest_holdings(weights.loc[:ASOF], UNIVERSE),
            "latest_holding_date": str(weights.dropna(how="all").index.max().date()) if weights.dropna(how="all").shape[0] else None,
        }

    bh_full = summaries["BH_300"]["slices"]["FULL"].get("sharpe_rf0")
    bh_oos = summaries["BH_300"]["slices"]["OOS"].get("sharpe_rf0")
    for name, row in summaries.items():
        row["bh_full_sharpe"] = bh_full
        row["bh_oos_sharpe"] = bh_oos
        row["avg_equity_weight"] = row["avg_equity_or_gold_weight"]
        if name.startswith("BH_"):
            row["quality"] = {"passes_quality_bar": False, "note": "benchmark"}
        else:
            row["quality"] = quality_flags(row)

    # Liquidity snapshot over the last 60 sessions.
    last_amt = amount.tail(60)
    liquidity = {}
    for code in amount.columns:
        med = float(last_amt[code].median()) if last_amt[code].notna().any() else None
        liquidity[code] = {
            "name": UNIVERSE.get(code, code),
            "median_amount_60d": med,
            "first_date": str(close[code].dropna().index.min().date()),
            "last_date": str(close[code].dropna().index.max().date()),
            "n": int(close[code].dropna().shape[0]),
        }

    # ETF vs price-index tracking, 2019-03-01 to as-of, price index has no dividend.
    tracking = {}
    if "sh000300" in index_close.columns and BENCHMARK in close.columns:
        a = close[BENCHMARK].dropna()
        b = index_close["sh000300"].dropna()
        joined = pd.concat([a.rename("etf"), b.rename("idx")], axis=1).dropna()
        joined = joined.loc[EVAL_START:ASOF]
        if len(joined) > 10:
            tracking = {
                "etf_total_return_qfq": float(joined["etf"].iloc[-1] / joined["etf"].iloc[0] - 1.0),
                "index_price_return": float(joined["idx"].iloc[-1] / joined["idx"].iloc[0] - 1.0),
                "note": "ETF uses qfq (dividend-adjusted). Index is a price index. Gap is mostly dividends plus tracking error, not a second alpha source.",
            }

    regime = index_regime(index_close) if not index_close.empty else {"error": "no index data"}
    regime["index_errors"] = index_errors
    flow = try_fund_flow()

    etf_regime = {}
    for code in close.columns:
        s = close[code].dropna()
        if len(s) < 80:
            continue
        etf_regime[code] = {
            "name": UNIVERSE.get(code, code),
            "ret_20d": float(s.iloc[-1] / s.iloc[-21] - 1.0) if len(s) > 21 else None,
            "ret_60d": float(s.iloc[-1] / s.iloc[-61] - 1.0) if len(s) > 61 else None,
            "ret_252d": float(s.iloc[-1] / s.iloc[-253] - 1.0) if len(s) > 253 else None,
            "dist_ma200": float(s.iloc[-1] / s.iloc[-200:].mean() - 1.0) if len(s) >= 200 else None,
        }

    sens = sensitivity(close, rets, rf, sectors if len(sectors) >= 5 else sectors_extended)

    # Daily curves for the report window only.
    curve_df = pd.DataFrame({k: v for k, v in curves.items()})
    curve_df = curve_df.loc[EVAL_START:ASOF]
    wealth = (1.0 + curve_df.fillna(0.0)).cumprod()
    # Restore NaN before each strategy's first live day so warmup is not a flat 1.
    for col in wealth.columns:
        live = curve_df[col].notna()
        if live.any():
            first = live.idxmax()
            wealth.loc[:first, col] = np.nan
            # idxmax on all-false is the first index; guard empty
            if not bool(live.loc[first]):
                wealth[col] = np.nan
            else:
                wealth.loc[first, col] = 1.0
                wealth[col] = wealth[col].ffill()
    wealth.to_csv(OUT / "equity_curves_20261004.csv")

    corr = curve_df.loc[OOS_START:ASOF].corr().round(3)

    payload = {
        "research_date": "2026-10-04",
        "timezone": "Asia/Shanghai",
        "price_asof": str(close.index.max().date()),
        "market_status": "SSE/SZSE closed 2026-10-01 to 2026-10-07; last session 2026-09-30; resumes 2026-10-08",
        "focus": "unsubstituted template {{研究焦点}}; default A-share liquid ETF research",
        "cost_one_way": COST_ONE_WAY,
        "cost_stress": COST_STRESS,
        "execution": "signal at close t earns return t+1; long-only; weights sum to 1; residual in 511010",
        "eval_start": "2019-03-01",
        "is_end": "2023-12-31",
        "oos": "2024-01-01 to 2026-09-30",
        "skipped_symbols": skipped,
        "sectors_used_long_history": sectors,
        "sectors_available": sectors_extended,
        "liquidity": liquidity,
        "tracking_300": tracking,
        "regime_indices": regime,
        "regime_etf": etf_regime,
        "fund_flow": flow,
        "strategies": summaries,
        "sensitivity": sens,
        "oos_correlation": corr.to_dict(),
        "data_source": "akshare stock_zh_a_hist_tx adjust=qfq (Tencent) for ETFs; akshare stock_zh_index_daily for indices",
    }
    def _conv(obj):
        if isinstance(obj, (np.floating,)):
            obj = float(obj)
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj.date())
        return obj

    (OUT / "metrics_20261004.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_conv),
        encoding="utf-8",
    )
    print(json.dumps({
        "asof": payload["price_asof"],
        "skipped": skipped,
        "sectors": sectors,
        "n_strategies": len(summaries),
        "passes": {k: v["quality"].get("passes_quality_bar") for k, v in summaries.items()},
    }, ensure_ascii=False, indent=2))
    # Compact leaderboard for the log.
    rows = []
    for name, row in summaries.items():
        full = row["slices"]["FULL"]
        oos = row["slices"]["OOS"]
        rows.append((
            name,
            None if full.get("sharpe_rf0") is None else round(full["sharpe_rf0"], 3),
            None if oos.get("sharpe_rf0") is None else round(oos["sharpe_rf0"], 3),
            None if full.get("cagr") is None else round(full["cagr"], 3),
            None if full.get("max_drawdown") is None else round(full["max_drawdown"], 3),
            row["quality"].get("passes_quality_bar"),
        ))
    print("LEADERBOARD name full_sharpe oos_sharpe full_cagr full_mdd pass")
    for row in rows:
        print("\t".join("" if v is None else str(v) for v in row))


if __name__ == "__main__":
    main()
