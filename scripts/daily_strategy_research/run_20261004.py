#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily A-share strategy research for 2026-10-04.

Pre-registered specs (do not retune after seeing results):
  long-only, gross exposure <= 1, single-name cap 25% (excess stays cash),
  signal at close t earns the close-to-close return ending t+1.

Costs (not a flat bps shortcut):
  commission 2.5 bps per side, slippage 5 bps per side,
  stamp duty 10 bps on sells before 2023-08-28 and 5 bps thereafter.
  Cash yield is 0 (no invented money-market series).

Universe is a curated liquid cross-section as of the research date.
That introduces survivorship / point-in-time bias versus a true historical
index. The fair selection benchmark is equal-weight of the SAME universe.
CSI 300 total return (H00300) is the external benchmark.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_START = pd.Timestamp("2019-01-01")
IS_END = pd.Timestamp("2023-12-31")
RECENT_START = pd.Timestamp("2025-01-01")
STAMP_CUTOFF = pd.Timestamp("2023-08-28")
FETCH_START = "20180101"
FETCH_END = "20261004"
NAME_CAP = 0.25
COMMISSION = 0.00025
SLIPPAGE = 0.0005
STAMP_OLD = 0.0010
STAMP_NEW = 0.0005
MIN_AMOUNT = 5e7  # 20-day median turnover, CNY

# Curated liquid names across economically different sleeves.
# Not a historical index reconstitution.
UNIVERSE: Dict[str, Dict[str, str]] = {
    "601398": {"name": "工商银行", "sector": "银行"},
    "600036": {"name": "招商银行", "sector": "银行"},
    "601166": {"name": "兴业银行", "sector": "银行"},
    "601318": {"name": "中国平安", "sector": "保险"},
    "600030": {"name": "中信证券", "sector": "证券"},
    "600900": {"name": "长江电力", "sector": "电力"},
    "600795": {"name": "国电电力", "sector": "电力"},
    "601088": {"name": "中国神华", "sector": "煤炭"},
    "600938": {"name": "中国海油", "sector": "石油"},
    "601899": {"name": "紫金矿业", "sector": "有色"},
    "600519": {"name": "贵州茅台", "sector": "白酒"},
    "000858": {"name": "五粮液", "sector": "白酒"},
    "000333": {"name": "美的集团", "sector": "家电"},
    "000651": {"name": "格力电器", "sector": "家电"},
    "600887": {"name": "伊利股份", "sector": "食品"},
    "600276": {"name": "恒瑞医药", "sector": "医药"},
    "603259": {"name": "药明康德", "sector": "医药"},
    "300750": {"name": "宁德时代", "sector": "新能源"},
    "002594": {"name": "比亚迪", "sector": "新能源"},
    "300274": {"name": "阳光电源", "sector": "新能源"},
    "300308": {"name": "中际旭创", "sector": "算力"},
    "601138": {"name": "工业富联", "sector": "算力"},
    "002475": {"name": "立讯精密", "sector": "电子"},
    "002371": {"name": "北方华创", "sector": "半导体"},
    "688981": {"name": "中芯国际", "sector": "半导体"},
    "002714": {"name": "牧原股份", "sector": "养殖"},
    "600048": {"name": "保利发展", "sector": "地产"},
    "601668": {"name": "中国建筑", "sector": "建筑"},
    "600760": {"name": "中航沈飞", "sector": "军工"},
    "000063": {"name": "中兴通讯", "sector": "通信"},
}


def ak_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def fetch_stock(code: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_zh_a_hist_tx(
        symbol=ak_symbol(code),
        start_date=FETCH_START,
        end_date=FETCH_END,
        adjust="qfq",
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date")
    df = df.drop_duplicates("date").reset_index(drop=True)
    df["code"] = code
    return df


def fetch_index_daily(symbol: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2018-01-01") & (df["date"] <= "2026-10-04")]
    return df.sort_values("date").reset_index(drop=True)


def fetch_csindex(symbol: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_zh_index_hist_csindex(
        symbol=symbol, start_date=FETCH_START, end_date=FETCH_END
    )
    df = df.rename(columns={"日期": "date", "收盘": "close", "滚动市盈率": "pe"})
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "pe" in df.columns:
        df["pe"] = pd.to_numeric(df["pe"], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def load_prices() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, str]]]:
    frames = []
    meta: Dict[str, Dict[str, str]] = {}
    for code, info in UNIVERSE.items():
        last_err = None
        for attempt in range(2):
            try:
                df = fetch_stock(code)
                if len(df) < 200:
                    print(f"skip {code} {info['name']}: short {len(df)}")
                    break
                frames.append(df)
                meta[code] = info
                print(
                    f"ok {code} {info['name']}: {len(df)} last={df['date'].iloc[-1].date()}"
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.8)
        else:
            print(f"fail {code} {info['name']}: {last_err}")
        time.sleep(0.25)
    if len(frames) < 15:
        raise RuntimeError(f"too few names downloaded: {len(frames)}")
    panel = pd.concat(frames, ignore_index=True)
    close = (
        panel.pivot(index="date", columns="code", values="close")
        .sort_index()
        .ffill(limit=3)
    )
    amount = (
        panel.pivot(index="date", columns="code", values="amount")
        .sort_index()
        .ffill(limit=3)
    )
    volume = (
        panel.pivot(index="date", columns="code", values="volume")
        .sort_index()
        .ffill(limit=3)
    )
    # Do not fabricate prices across long listing gaps.
    raw_close = panel.pivot(index="date", columns="code", values="close").sort_index()
    listed = raw_close.notna()
    # Allow only a 3-day ffill after a real print.
    close = close.where(listed.ffill(limit=3).fillna(False))
    amount = amount.where(close.notna())
    volume = volume.where(close.notna())
    return close, amount, volume, meta


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    period = index.to_period("M" if freq == "M" else "W")
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(period).tail(1).values)


def apply_cap(weights: pd.DataFrame, cap: float = NAME_CAP) -> pd.DataFrame:
    """Cap single-name weight. Excess stays in cash (not redistributed)."""
    out = weights.clip(lower=0.0).copy()
    excess = (out - cap).clip(lower=0.0)
    out = out.where(out <= cap, cap)
    # If a row was all zeros, keep zeros.
    out = out.where(weights.sum(axis=1) > 0, 0.0)
    del excess
    return out.fillna(0.0)


def allocate(index: pd.DatetimeIndex, columns: Sequence[str], picks_by_date: Dict[pd.Timestamp, List[str]]) -> pd.DataFrame:
    w = pd.DataFrame(np.nan, index=index, columns=list(columns))
    for dt, picks in picks_by_date.items():
        w.loc[dt] = 0.0
        if picks:
            w.loc[dt, picks] = 1.0 / len(picks)
    return apply_cap(w.ffill().fillna(0.0))


def eligible_mask(close: pd.DataFrame, amount: pd.DataFrame, min_history: int) -> pd.DataFrame:
    hist = close.notna().rolling(min_history).sum() >= min_history
    liquid = amount.rolling(20).median() >= MIN_AMOUNT
    return hist & liquid.fillna(False) & close.notna()


def monthly_rank_weights(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    score: pd.DataFrame,
    *,
    top_n: int,
    largest: bool,
    min_history: int,
    min_score: Optional[float] = None,
    min_names: int = 4,
) -> pd.DataFrame:
    eligible = eligible_mask(close, amount, min_history)
    picks: Dict[pd.Timestamp, List[str]] = {}
    for dt in rebalance_dates(close.index, "M"):
        row = score.loc[dt]
        ok = eligible.loc[dt] & row.notna()
        if min_score is not None:
            ok = ok & (row > min_score)
        pool = row[ok]
        if len(pool) < min_names:
            picks[dt] = []
            continue
        chosen = pool.nlargest(top_n) if largest else pool.nsmallest(min(top_n, len(pool)))
        picks[dt] = list(chosen.index[:top_n])
    return allocate(close.index, close.columns, picks)


def portfolio_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    slip_mult: float = 1.0,
) -> pd.DataFrame:
    """Position that earns r_t was chosen at t-1. Trade at close t from w_{t-1} to w_t.

    Entry cost before the sample is already in the 2018 warmup, so the
    evaluation window does not get a second artificial entry charge.
    """
    w = weights.reindex(asset_returns.index).fillna(0.0)
    w = w.where(asset_returns.notna(), 0.0)
    # Renormalize only if a price hole knocked out a held name and gross was ~1.
    # Keep intentional cash: scale only when previous gross was about 1 and
    # missing prices reduced it. Simpler rule: leave residual as cash.
    w_prev = w.shift(1).fillna(0.0)
    aligned_ret = asset_returns.reindex(columns=w.columns).fillna(0.0)
    gross = (w_prev * aligned_ret).sum(axis=1)
    dw = w - w_prev
    buys = dw.clip(lower=0.0).sum(axis=1)
    sells = (-dw.clip(upper=0.0)).sum(axis=1)
    slip = SLIPPAGE * slip_mult
    sell_rate = pd.Series(COMMISSION + slip + STAMP_NEW, index=w.index)
    sell_rate.loc[sell_rate.index < STAMP_CUTOFF] = COMMISSION + slip + STAMP_OLD
    buy_rate = COMMISSION + slip
    cost = buys * buy_rate + sells * sell_rate
    net = gross - cost
    out = pd.DataFrame(
        {
            "gross": gross,
            "net": net,
            "cost": cost,
            "turnover": buys + sells,
            "gross_exposure": w_prev.sum(axis=1),
        }
    )
    return out


def equity_from_net(net: pd.Series) -> pd.Series:
    net = net.dropna()
    if net.empty:
        return net
    eq = (1.0 + net).cumprod()
    eq.name = "equity"
    return eq


def slice_net(path: pd.DataFrame, start: pd.Timestamp, end: Optional[pd.Timestamp] = None) -> pd.Series:
    net = path["net"]
    net = net.loc[net.index >= start]
    if end is not None:
        net = net.loc[net.index <= end]
    return net


def metrics_from_net(net: pd.Series, freq: int = 252) -> Dict[str, float]:
    net = net.dropna()
    if len(net) < 5:
        return {}
    equity = (1.0 + net).cumprod()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0)
    vol = float(net.std() * math.sqrt(freq))
    sharpe = float(net.mean() / net.std() * math.sqrt(freq)) if net.std() > 0 else 0.0
    downside = net[net < 0]
    sortino = (
        float(net.mean() / downside.std() * math.sqrt(freq))
        if len(downside) and downside.std() > 0
        else 0.0
    )
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    pf = float(gains / losses) if losses > 0 else float("inf")
    # Monthly win rate is closer to a holding-period statistic than daily win rate.
    monthly = equity.resample("ME").last().pct_change().dropna()
    m_gains = float(monthly[monthly > 0].sum()) if len(monthly) else 0.0
    m_losses = float(-monthly[monthly < 0].sum()) if len(monthly) else 0.0
    sharpe_se = float(math.sqrt((1 + 0.5 * sharpe**2) / len(net))) if len(net) else float("nan")
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sharpe_se": sharpe_se,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else float("nan"),
        "win_rate_daily": float((net > 0).mean()),
        "profit_factor_daily": pf,
        "win_rate_monthly": float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "profit_factor_monthly": float(m_gains / m_losses) if m_losses > 0 else float("nan"),
        "n_days": int(len(net)),
        "worst_day": float(net.min()),
        "best_day": float(net.max()),
    }


def window_return(net: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    part = net.loc[(net.index >= start) & (net.index <= end)]
    if part.empty:
        return float("nan")
    return float((1.0 + part).prod() - 1.0)


def max_dd_window(equity: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp, float]:
    dd = equity / equity.cummax() - 1.0
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return start, end, float(dd.min())


def yearly_returns(net: pd.Series) -> Dict[str, float]:
    equity = (1.0 + net).cumprod()
    out = {}
    for year, part in equity.groupby(equity.index.year):
        if len(part) < 2:
            continue
        out[str(year)] = float(part.iloc[-1] / part.iloc[0] - 1.0)
    return out


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
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
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


# -------------------- strategies (pre-registered) --------------------


def strat_equal_weight(close: pd.DataFrame, amount: pd.DataFrame) -> pd.DataFrame:
    eligible = eligible_mask(close, amount, min_history=60)
    w = eligible.astype(float)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return apply_cap(w)


def strat_cs_mom_12_1(close: pd.DataFrame, amount: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """12-1 momentum: close[t-21] / close[t-252] - 1. Long winners."""
    score = close.shift(21) / close.shift(252) - 1.0
    return monthly_rank_weights(
        close, amount, score, top_n=top_n, largest=True, min_history=252, min_names=4
    )


def strat_reversal_21(close: pd.DataFrame, amount: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Monthly long the worst 21-day losers (short-horizon reversal)."""
    score = close.pct_change(21)
    return monthly_rank_weights(
        close, amount, score, top_n=top_n, largest=False, min_history=40, min_names=4
    )


def strat_reversal_5d(close: pd.DataFrame, amount: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Weekly long the worst 5-day losers."""
    score = close.pct_change(5)
    eligible = eligible_mask(close, amount, 20)
    picks: Dict[pd.Timestamp, List[str]] = {}
    for dt in rebalance_dates(close.index, "W"):
        row = score.loc[dt]
        ok = eligible.loc[dt] & row.notna()
        pool = row[ok]
        if len(pool) < 4:
            picks[dt] = []
            continue
        picks[dt] = list(pool.nsmallest(top_n).index)
    return allocate(close.index, close.columns, picks)


def strat_low_vol(close: pd.DataFrame, amount: pd.DataFrame, lookback: int = 60, top_n: int = 6) -> pd.DataFrame:
    score = close.pct_change().rolling(lookback).std()
    return monthly_rank_weights(
        close, amount, score, top_n=top_n, largest=False, min_history=lookback, min_names=4
    )


def strat_dual_momentum(close: pd.DataFrame, amount: pd.DataFrame, lookback: int = 126, top_n: int = 5) -> pd.DataFrame:
    """Top-N by lookback return among names with positive absolute momentum; else cash."""
    score = close.pct_change(lookback)
    return monthly_rank_weights(
        close,
        amount,
        score,
        top_n=top_n,
        largest=True,
        min_history=lookback,
        min_score=0.0,
        min_names=1,
    )


def strat_index_tsmom(close: pd.DataFrame, amount: pd.DataFrame, index_close: pd.Series, lookback: int = 120) -> pd.DataFrame:
    """Hold the equal-weight universe only when the benchmark's lookback return is positive."""
    ew = strat_equal_weight(close, amount)
    idx = index_close.reindex(close.index).ffill()
    mom = idx.pct_change(lookback)
    state = pd.Series(np.nan, index=close.index)
    for dt in rebalance_dates(close.index, "M"):
        state.loc[dt] = 1.0 if bool(pd.notna(mom.loc[dt]) and mom.loc[dt] > 0) else 0.0
    state = state.ffill().fillna(0.0)
    return apply_cap(ew.mul(state, axis=0))


def strat_sector_rotation(close: pd.DataFrame, amount: pd.DataFrame, meta: Dict[str, Dict[str, str]], lookback: int = 63, n_sectors: int = 2) -> pd.DataFrame:
    sectors = {}
    for code, info in meta.items():
        if code in close.columns:
            sectors.setdefault(info["sector"], []).append(code)
    sec_ret = {}
    for name, cols in sectors.items():
        px = close[cols].mean(axis=1)
        sec_ret[name] = px.pct_change(lookback)
    sec_df = pd.DataFrame(sec_ret)
    eligible = eligible_mask(close, amount, lookback)
    picks: Dict[pd.Timestamp, List[str]] = {}
    for dt in rebalance_dates(close.index, "M"):
        row = sec_df.loc[dt].dropna()
        if row.empty:
            picks[dt] = []
            continue
        best = list(row.nlargest(n_sectors).index)
        chosen = []
        for sec in best:
            for code in sectors.get(sec, []):
                if bool(eligible.loc[dt, code]):
                    chosen.append(code)
        picks[dt] = chosen
    return allocate(close.index, close.columns, picks)


def strat_trend_ma(close: pd.DataFrame, amount: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    eligible = eligible_mask(close, amount, window)
    ma = close.rolling(window).mean()
    signal = eligible & (close > ma)
    w = signal.astype(float)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return apply_cap(w)


def strat_volume_breakout(close: pd.DataFrame, amount: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    eligible = eligible_mask(close, amount, 40)
    prior_high = close.rolling(20).max().shift(1)
    vol_ma = volume.rolling(20).mean()
    trigger = eligible & (close > prior_high) & (volume > 1.5 * vol_ma)
    trigger = trigger.fillna(False)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for col in close.columns:
        hold = 0
        flags = trigger[col].to_numpy()
        for i, flag in enumerate(flags):
            if flag:
                hold = 10
            if hold > 0:
                w.iat[i, w.columns.get_loc(col)] = 1.0
                hold -= 1
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    return apply_cap(w)


def strat_quality_blend(close: pd.DataFrame, amount: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Positive 126d momentum, then lowest 60d vol. Cash if fewer than 3 qualify."""
    mom = close.pct_change(126)
    vol = close.pct_change().rolling(60).std()
    eligible = eligible_mask(close, amount, 126) & (mom > 0) & vol.notna()
    picks: Dict[pd.Timestamp, List[str]] = {}
    for dt in rebalance_dates(close.index, "M"):
        pool = vol.loc[dt][eligible.loc[dt]]
        if len(pool) < 3:
            picks[dt] = []
            continue
        picks[dt] = list(pool.nsmallest(top_n).index)
    return allocate(close.index, close.columns, picks)


def build_strategies(close, amount, volume, meta, bench_close) -> List[Tuple[str, pd.DataFrame, str, str]]:
    return [
        (
            "CS_Mom_12_1",
            strat_cs_mom_12_1(close, amount),
            "截面动量",
            "A股机构与主题资金存在反应不足与抱团，12-1个月赢家组合在下一月继续占优。",
        ),
        (
            "REV_21D",
            strat_reversal_21(close, amount),
            "月频反转",
            "散户与短线交易者过度反应，近一月输家在下一月均值回归。",
        ),
        (
            "REV_5D",
            strat_reversal_5d(close, amount),
            "周频反转",
            "更短窗口的过度反应；换手高，用来检验成本后边缘是否还在。",
        ),
        (
            "Low_Vol",
            strat_low_vol(close, amount),
            "低波",
            "杠杆约束与彩票偏好使高波资产被超买，低波组合风险调整后更好。",
        ),
        (
            "Dual_Mom",
            strat_dual_momentum(close, amount),
            "双动量",
            "只做绝对动量为正的相对强势股，熊市允许空仓，避免单边下跌里的相对强弱。",
        ),
        (
            "Index_TSMOM",
            strat_index_tsmom(close, amount, bench_close),
            "指数时序动量",
            "市场本身的趋势：沪深300过去120日收益为正才持有股票贝塔，否则现金。",
        ),
        (
            "Sector_Rotation",
            strat_sector_rotation(close, amount, meta),
            "行业轮动",
            "行业景气与资金在数月尺度上延续，持有近63日最强的两个行业。",
        ),
        (
            "Trend_MA60",
            strat_trend_ma(close, amount),
            "个股趋势",
            "价格高于60日均线的个股趋势延续；跌破则退出。",
        ),
        (
            "Vol_Breakout",
            strat_volume_breakout(close, amount, volume),
            "放量突破",
            "突破前高且放量代表新信息与资金进入，短持有10个交易日。",
        ),
        (
            "Quality_Blend",
            strat_quality_blend(close, amount),
            "质量混合",
            "在中期动量为正的股票里选低波，避开纯追涨与纯低波的两端。",
        ),
    ]


def regime_snapshot(close: pd.DataFrame, amount: pd.DataFrame, meta, indices: Dict[str, pd.DataFrame]) -> dict:
    last = close.index[-1]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    breadth20 = float((close.iloc[-1] > ma20.iloc[-1]).mean())
    breadth60 = float((close.iloc[-1] > ma60.iloc[-1]).mean())
    ret21 = close.pct_change(21).iloc[-1]
    ret63 = close.pct_change(63).iloc[-1]
    ret252 = close.pct_change(252).iloc[-1]
    rows = []
    for code in close.columns:
        rows.append(
            {
                "code": code,
                "name": meta[code]["name"],
                "sector": meta[code]["sector"],
                "close": float(close[code].iloc[-1]) if pd.notna(close[code].iloc[-1]) else None,
                "ret_21d": float(ret21[code]) if pd.notna(ret21[code]) else None,
                "ret_63d": float(ret63[code]) if pd.notna(ret63[code]) else None,
                "ret_252d": float(ret252[code]) if pd.notna(ret252[code]) else None,
                "above_ma20": bool(close[code].iloc[-1] > ma20[code].iloc[-1])
                if pd.notna(ma20[code].iloc[-1])
                else None,
                "above_ma60": bool(close[code].iloc[-1] > ma60[code].iloc[-1])
                if pd.notna(ma60[code].iloc[-1])
                else None,
                "amount_20d_med": float(amount[code].rolling(20).median().iloc[-1])
                if pd.notna(amount[code].rolling(20).median().iloc[-1])
                else None,
            }
        )
    idx_rows = {}
    for name, df in indices.items():
        px = df.set_index("date")["close"].sort_index()
        px = px[px.index <= last]
        if px.empty:
            continue
        vol20 = float(px.pct_change().tail(20).std() * math.sqrt(252))
        ma200 = px.rolling(200).mean().iloc[-1]
        idx_rows[name] = {
            "last": float(px.iloc[-1]),
            "date": px.index[-1].strftime("%Y-%m-%d"),
            "ret_5d": float(px.pct_change(5).iloc[-1]),
            "ret_21d": float(px.pct_change(21).iloc[-1]),
            "ret_63d": float(px.pct_change(63).iloc[-1]),
            "ret_252d": float(px.pct_change(252).iloc[-1]) if len(px) > 252 else None,
            "above_ma200": bool(px.iloc[-1] > ma200) if pd.notna(ma200) else None,
            "vol_20d_ann": vol20,
            "pe": float(df["pe"].iloc[-1]) if "pe" in df.columns and pd.notna(df["pe"].iloc[-1]) else None,
        }
    # Sector 63d equal-weight return inside the research universe.
    sec = {}
    for code, info in meta.items():
        if code not in close.columns or pd.isna(ret63[code]):
            continue
        sec.setdefault(info["sector"], []).append(float(ret63[code]))
    sector_63 = {k: float(np.mean(v)) for k, v in sec.items()}
    return {
        "asof": last.strftime("%Y-%m-%d"),
        "n_names": int(close.shape[1]),
        "breadth_above_ma20": breadth20,
        "breadth_above_ma60": breadth60,
        "median_ret_21d": float(ret21.median()),
        "median_ret_63d": float(ret63.median()),
        "median_ret_252d": float(ret252.median()),
        "indices": idx_rows,
        "sector_ret_63d": sector_63,
        "constituents": rows,
    }


def latest_book(weights: pd.DataFrame, meta) -> List[dict]:
    last = weights.iloc[-1]
    held = last[last > 0.01].sort_values(ascending=False)
    return [
        {
            "code": code,
            "name": meta[code]["name"],
            "sector": meta[code]["sector"],
            "weight": float(wt),
        }
        for code, wt in held.items()
    ]


def sensitivity(close, amount, volume, meta, bench_close, asset_returns) -> dict:
    """Neighbor grid around pre-registered parameters. Not used to pick a winner."""
    grid = {
        "CS_Mom_12_1": [],
        "REV_21D": [],
        "Low_Vol": [],
        "Dual_Mom": [],
        "Index_TSMOM": [],
        "Sector_Rotation": [],
    }
    for top_n in (4, 6, 8):
        w = strat_cs_mom_12_1(close, amount, top_n=top_n)
        net = slice_net(portfolio_returns(w, asset_returns), EVAL_START)
        m = metrics_from_net(net)
        grid["CS_Mom_12_1"].append({"top_n": top_n, "sharpe": m.get("sharpe"), "mdd": m.get("max_drawdown"), "cagr": m.get("cagr")})
        w = strat_reversal_21(close, amount, top_n=top_n)
        net = slice_net(portfolio_returns(w, asset_returns), EVAL_START)
        m = metrics_from_net(net)
        grid["REV_21D"].append({"top_n": top_n, "sharpe": m.get("sharpe"), "mdd": m.get("max_drawdown"), "cagr": m.get("cagr")})
    for lookback, top_n in ((40, 6), (60, 6), (120, 6), (60, 4), (60, 8)):
        w = strat_low_vol(close, amount, lookback=lookback, top_n=top_n)
        net = slice_net(portfolio_returns(w, asset_returns), EVAL_START)
        m = metrics_from_net(net)
        grid["Low_Vol"].append({"lookback": lookback, "top_n": top_n, "sharpe": m.get("sharpe"), "mdd": m.get("max_drawdown"), "cagr": m.get("cagr")})
    for lookback, top_n in ((63, 5), (126, 5), (252, 5), (126, 3), (126, 8)):
        w = strat_dual_momentum(close, amount, lookback=lookback, top_n=top_n)
        net = slice_net(portfolio_returns(w, asset_returns), EVAL_START)
        m = metrics_from_net(net)
        grid["Dual_Mom"].append({"lookback": lookback, "top_n": top_n, "sharpe": m.get("sharpe"), "mdd": m.get("max_drawdown"), "cagr": m.get("cagr")})
    for lookback in (60, 120, 200):
        w = strat_index_tsmom(close, amount, bench_close, lookback=lookback)
        net = slice_net(portfolio_returns(w, asset_returns), EVAL_START)
        m = metrics_from_net(net)
        grid["Index_TSMOM"].append({"lookback": lookback, "sharpe": m.get("sharpe"), "mdd": m.get("max_drawdown"), "cagr": m.get("cagr")})
    for lookback, n_sec in ((42, 2), (63, 2), (126, 2), (63, 1), (63, 3)):
        w = strat_sector_rotation(close, amount, meta, lookback=lookback, n_sectors=n_sec)
        net = slice_net(portfolio_returns(w, asset_returns), EVAL_START)
        m = metrics_from_net(net)
        grid["Sector_Rotation"].append({"lookback": lookback, "n_sectors": n_sec, "sharpe": m.get("sharpe"), "mdd": m.get("max_drawdown"), "cagr": m.get("cagr")})
    return grid


def try_sector_flow() -> dict:
    import akshare as ak
    from concurrent.futures import ThreadPoolExecutor

    out = {"ok": False, "error": None, "top": [], "bottom": []}
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(ak.stock_fund_flow_industry, symbol="即时")
            flow = fut.result(timeout=30)
        # Keep a stable subset of columns if present.
        cols = [c for c in ("行业", "行业-涨跌幅", "净额", "领涨股") if c in flow.columns]
        if not cols and "行业" not in flow.columns:
            # Fall back to first columns.
            out["columns"] = list(flow.columns)
            out["error"] = "unexpected columns"
            return out
        flow = flow.copy()
        if "净额" in flow.columns:
            flow["净额"] = pd.to_numeric(flow["净额"], errors="coerce")
            ordered = flow.sort_values("净额", ascending=False)
            out["top"] = ordered.head(8)[cols].to_dict(orient="records")
            out["bottom"] = ordered.tail(5)[cols].to_dict(orient="records")
        out["ok"] = True
        out["asof_note"] = "Eastmoney industry flow snapshot at fetch time; market is shut for National Day so this is the last available board tape, not a 2026-10-04 session."
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> None:
    print("loading prices...")
    close, amount, volume, meta = load_prices()
    print(f"panel {close.shape} {close.index[0].date()} -> {close.index[-1].date()}")

    print("loading indices...")
    indices = {}
    for label, symbol, kind in (
        ("上证指数", "sh000001", "sina"),
        ("沪深300", "000300", "cs"),
        ("沪深300全收益", "H00300", "cs"),
        ("中证500", "000905", "cs"),
        ("中证500全收益", "H00905", "cs"),
        ("创业板指", "sz399006", "sina"),
        ("科创50", "sh000688", "sina"),
    ):
        try:
            if kind == "cs":
                df = fetch_csindex(symbol)
            else:
                df = fetch_index_daily(symbol)
            indices[label] = df
            print(f"index {label} last={df['date'].iloc[-1].date()} close={df['close'].iloc[-1]:.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"index fail {label}: {exc}")

    bench = indices["沪深300全收益"].set_index("date")["close"].sort_index()
    bench = bench.reindex(close.index).ffill()
    asset_returns = close.pct_change()
    bench_ret = bench.pct_change()

    strategies = build_strategies(close, amount, volume, meta, bench)
    # Benchmarks: same cost model, no ongoing turnover except index entry outside window.
    ew = strat_equal_weight(close, amount)
    bench_w = pd.DataFrame(0.0, index=close.index, columns=["BENCH"])
    # Represent benchmark as a single asset so the cost engine can run.
    # Easier: synthetic net = bench return, zero turnover inside eval window.
    results = []
    curves = {}

    def pack(name, family, thesis, path, weights):
        net_full = slice_net(path, EVAL_START)
        net_is = slice_net(path, EVAL_START, IS_END)
        net_oos = slice_net(path, IS_END + pd.Timedelta(days=1))
        net_recent = slice_net(path, RECENT_START)
        m_full = metrics_from_net(net_full)
        m_is = metrics_from_net(net_is)
        m_oos = metrics_from_net(net_oos)
        m_recent = metrics_from_net(net_recent)
        eq = equity_from_net(net_full)
        curves[name] = eq
        dd_start, dd_end, dd = max_dd_window(eq) if len(eq) else (None, None, None)
        # Benchmark drawdown window applied later.
        ann_turn = float(path.loc[path.index >= EVAL_START, "turnover"].mean() * 252)
        avg_exp = float(path.loc[path.index >= EVAL_START, "gross_exposure"].mean())
        # Cost stress
        stress = {}
        if weights is not None:
            for mult, label in ((2.0, "slip_2x"), (3.0, "slip_3x")):
                sp = portfolio_returns(weights, asset_returns, slip_mult=mult)
                sm = metrics_from_net(slice_net(sp, EVAL_START))
                stress[label] = {"sharpe": sm.get("sharpe"), "cagr": sm.get("cagr"), "max_drawdown": sm.get("max_drawdown")}
        book = latest_book(weights, meta) if weights is not None else []
        cash = 1.0 - float(weights.iloc[-1].sum()) if weights is not None else 0.0
        return {
            "name": name,
            "family": family,
            "thesis": thesis,
            "full": m_full,
            "is": m_is,
            "oos": m_oos,
            "recent_2025_2026": m_recent,
            "yearly": yearly_returns(net_full),
            "annual_turnover": ann_turn,
            "avg_exposure": avg_exp,
            "own_mdd_window": {"start": dd_start, "end": dd_end, "max_drawdown": dd},
            "cost_stress": stress,
            "latest_book": book,
            "latest_cash": cash,
            "latest_gross": float(weights.iloc[-1].sum()) if weights is not None else 1.0,
        }

    print("benchmarks...")
    ew_path = portfolio_returns(ew, asset_returns)
    results.append(pack("EW_Universe", "基准", "研究样本等权买入持有，选股策略的公平基准。", ew_path, ew))
    # Index: price aligned, costs only if we model a one-time buy already outside window.
    idx_path = pd.DataFrame(
        {
            "gross": bench_ret,
            "net": bench_ret.fillna(0.0),
            "cost": 0.0,
            "turnover": 0.0,
            "gross_exposure": 1.0,
        },
        index=close.index,
    )
    results.append(pack("CSI300_TR", "基准", "沪深300全收益指数买入持有（H00300），外部基准。", idx_path, None))

    for name, weights, family, thesis in strategies:
        print("run", name)
        path = portfolio_returns(weights, asset_returns)
        results.append(pack(name, family, thesis, path, weights))

    # Correlations and benchmark-window stress.
    ret_df = pd.DataFrame({k: v.pct_change() for k, v in curves.items()}).dropna(how="all")
    corr = ret_df.corr().round(3)
    bench_eq = curves["CSI300_TR"]
    b0, b1, bdd = max_dd_window(bench_eq)
    for row in results:
        name = row["name"]
        net = ret_df[name].dropna() if name in ret_df.columns else pd.Series(dtype=float)
        # Use equity pct which equals net only approximately after rebase; recompute from curve.
        # ret_df is equity pct_change, first day missing. Good enough for window return if we use equity ratio.
        eq = curves[name]
        if b0 in eq.index or True:
            part = eq.loc[(eq.index >= b0) & (eq.index <= b1)]
            row["csi300_mdd_window_return"] = float(part.iloc[-1] / part.iloc[0] - 1.0) if len(part) > 2 else None
        y2022 = eq[eq.index.year == 2022]
        row["return_2022"] = float(y2022.iloc[-1] / y2022.iloc[0] - 1.0) if len(y2022) > 2 else None

    # Beta / residual vs CSI300 over full, IS, OOS.
    bench_r = ret_df["CSI300_TR"]
    for row in results:
        name = row["name"]
        if name not in ret_df.columns:
            continue
        betas = {}
        for label, mask in (
            ("full", ret_df.index >= EVAL_START),
            ("is", (ret_df.index >= EVAL_START) & (ret_df.index <= IS_END)),
            ("oos", ret_df.index > IS_END),
        ):
            df = pd.concat([ret_df.loc[mask, name], bench_r.loc[mask]], axis=1).dropna()
            if len(df) < 30 or df.iloc[:, 1].var() == 0:
                continue
            beta = float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())
            alpha_d = float(df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean())
            betas[label] = {"beta": beta, "alpha_ann": alpha_d * 252}
        row["vs_csi300"] = betas

    print("sensitivity...")
    sens = sensitivity(close, amount, volume, meta, bench, asset_returns)
    print("sector flow...")
    flow = try_sector_flow()
    snap = regime_snapshot(close, amount, meta, indices)

    # Quality bar flags. Selection alpha must beat EW of the same universe
    # on full-sample Sharpe and CAGR, and OOS Sharpe must stay positive.
    ew_row = next(r for r in results if r["name"] == "EW_Universe")
    idx_row = next(r for r in results if r["name"] == "CSI300_TR")
    for row in results:
        if row["name"] in ("EW_Universe", "CSI300_TR"):
            row["quality"] = "benchmark"
            continue
        full, oos = row["full"], row["oos"]
        beat_ew = (
            full.get("sharpe", -9) > ew_row["full"].get("sharpe", 0) + 0.05
            and full.get("cagr", -9) > ew_row["full"].get("cagr", 0)
        )
        beat_idx = full.get("sharpe", -9) > idx_row["full"].get("sharpe", 0)
        sharpe_ok = full.get("sharpe", 0) > 1.0
        mdd_ok = full.get("max_drawdown", -1) > -0.30
        oos_ok = oos.get("sharpe", -9) > 0.5 and oos.get("cagr", -9) > 0
        recent_ok = row["recent_2025_2026"].get("sharpe", -9) > 0
        if beat_ew and beat_idx and sharpe_ok and mdd_ok and oos_ok and recent_ok:
            flag = "PASS"
        elif beat_ew and oos_ok and full.get("sharpe", 0) > 0.7:
            flag = "CONDITIONAL"
        else:
            flag = "FAIL"
        row["quality"] = flag
        row["beat_ew_sharpe"] = full.get("sharpe", 0) - ew_row["full"].get("sharpe", 0)
        row["beat_ew_cagr"] = full.get("cagr", 0) - ew_row["full"].get("cagr", 0)
        row["beat_idx_sharpe"] = full.get("sharpe", 0) - idx_row["full"].get("sharpe", 0)

    payload = {
        "research_date": "2026-10-04",
        "data_asof": snap["asof"],
        "eval_start": str(EVAL_START.date()),
        "is_end": str(IS_END.date()),
        "recent_start": str(RECENT_START.date()),
        "cost": {
            "commission_bps": 2.5,
            "slippage_bps_per_side": 5.0,
            "stamp_bps_sell_before_20230828": 10.0,
            "stamp_bps_sell_from_20230828": 5.0,
            "cash_yield": 0.0,
            "name_cap": NAME_CAP,
            "min_amount_cny": MIN_AMOUNT,
        },
        "universe": {k: meta[k] for k in close.columns},
        "snapshot": snap,
        "sector_flow": flow,
        "csi300_mdd_window": {"start": b0, "end": b1, "depth": bdd},
        "results": results,
        "correlation": corr.to_dict(),
        "sensitivity": sens,
        "gaps": [
            "样本是2026年视角的流动性股票池，不是历史时点的指数成分，存在幸存者/后视偏差。",
            "未建模涨跌停无法成交、T+1在日内的细粒度约束、集合竞价跳空。信号用收盘，成交近似为下一交易日收盘到收盘收益。",
            "现金收益按0，空仓策略略偏保守。",
            "未使用北向、融资融券、一致预期盈利修订；缺少这些序列时不强行填数。",
            "行业资金流若抓取失败，只作盘面背景，不作为回测输入。",
        ],
    }
    text = json.dumps(clean(payload), ensure_ascii=False, indent=2)
    (OUT / "results.json").write_text(text, encoding="utf-8")
    # Small equity table for audit.
    eq_df = pd.DataFrame(curves)
    eq_df.to_csv(OUT / "equity_curves.csv")
    print("wrote", OUT / "results.json")
    for row in results:
        f = row["full"]
        print(
            f"{row['name']:16} {row['quality']:12} "
            f"Sharpe={f.get('sharpe'):.2f} CAGR={f.get('cagr'):.1%} "
            f"MDD={f.get('max_drawdown'):.1%} OOS={row['oos'].get('sharpe'):.2f}"
        )


if __name__ == "__main__":
    main()
