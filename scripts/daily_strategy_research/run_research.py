#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily A-share strategy research for 2026-10-14.

Pre-committed rules (chosen before looking at this run's results):

- Evaluation window: 2020-01-02 .. latest completed daily bar.
- History fetch starts 2018-01-01 so 120-day signals exist on the first eval day.
- Long-only, no leverage. Uninvested weight is cash at 0% (money-market yield ignored).
- Signal uses the close of day t. The position earns the close-to-close return of t+1.
- Costs are charged on turnover at the rebalance close and deducted from the next day's return.
- ETF one-way cost: 8 bps (commission + slippage; onshore ETFs have no stamp duty).
- Stock one-way cost: 15 bps (commission + sell-side stamp amortization + slippage).
- Sector momentum / low-vol / regime: month-end, top 3, 120-day lookback where applicable.
- Reversal: last session of each ISO week, 5-day losers, top 3 ETFs or top 5 stocks.
- No parameter is selected by maximizing in-sample Sharpe. Sensitivity is diagnostic only.

This is research code, not an order-routing system and not investment advice.
"""

from __future__ import annotations

import json
import math
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

ASOF_REQUESTED = "2026-10-14"
FETCH_START = "20180101"
FETCH_END = "20261014"
EVAL_START = pd.Timestamp("2020-01-02")
IS_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")

ETF_COST_BPS = 8.0
STOCK_COST_BPS = 15.0
TRADING_DAYS = 252
ETF_MIN_AMOUNT = 2.0e7  # 20 million CNY, 20-day median
STOCK_MIN_AMOUNT = 5.0e7

# Domestic equity sector ETFs. QDII names are excluded: premiums and
# subscription halts make close-to-close fills unrealistic.
SECTOR_ETFS: Dict[str, str] = {
    "510500": "中证500ETF",
    "512100": "中证1000ETF",
    "159915": "创业板ETF",
    "512880": "证券ETF",
    "512010": "医药ETF",
    "512690": "酒ETF",
    "159928": "消费ETF",
    "512480": "半导体ETF",
    "512660": "军工ETF",
    "512800": "银行ETF",
    "515220": "煤炭ETF",
    "512400": "有色ETF",
    "515030": "新能源车ETF",
    "159611": "电力ETF",
    "510880": "红利ETF",
}

DEFENSIVE_ETFS = ["512800", "510880", "515220", "159611", "518880", "511260"]

EXTRA_ETFS: Dict[str, str] = {
    "510300": "沪深300ETF",
    "518880": "黄金ETF",
    "511260": "十年国债ETF",
}

STOCKS: Dict[str, str] = {
    "601398": "工商银行",
    "600036": "招商银行",
    "601318": "中国平安",
    "600519": "贵州茅台",
    "000858": "五粮液",
    "000333": "美的集团",
    "000651": "格力电器",
    "600887": "伊利股份",
    "600276": "恒瑞医药",
    "603259": "药明康德",
    "601088": "中国神华",
    "600938": "中国海油",
    "601857": "中国石油",
    "600900": "长江电力",
    "601899": "紫金矿业",
    "002594": "比亚迪",
    "300750": "宁德时代",
    "601138": "工业富联",
    "002475": "立讯精密",
    "600030": "中信证券",
    "300059": "东方财富",
    "600760": "中航沈飞",
    "600941": "中国移动",
    "601012": "隆基绿能",
}

INDEXES: Dict[str, str] = {
    "sh000001": "上证指数",
    "sh000300": "沪深300",
    "sz399006": "创业板指",
    "sh000905": "中证500",
    "sh000852": "中证1000",
}


def _prefix(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def _cache_path(key: str) -> Path:
    return CACHE / f"{key}.csv"


def _load_cache(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        return None
    return df


def _save_cache(key: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(key), index=False)


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount") if c in out.columns]
    out = out[keep].dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    return out.reset_index(drop=True)


def _jump_limit(code: str, kind: str) -> float:
    """Hard ceiling for a single-session move. Prints beyond this are treated as bad ticks.

    Main-board stocks are 10% limit-up/down. ChiNext/STAR are 20% after the 2020 reform;
    23% leaves a small buffer. ETFs can gap more than the cash limit when the basket is
    locked (the 2024-09/10 ChiNext sessions printed +20%), so the ETF ceiling is 25%.
    """
    if kind == "etf":
        return 0.25
    if code.startswith(("300", "301", "688", "689")):
        return 0.23
    return 0.115


def _strip_bad_ticks(df: pd.DataFrame, code: str, kind: str) -> pd.DataFrame:
    """Drop non-positive prices and impossible jumps. Do not replace them with a guess."""
    out = df.copy()
    bad = (out["close"] <= 0) | (out["low"] <= 0) | (out["high"] < out["low"])
    out = out.loc[~bad].copy()
    limit = _jump_limit(code, kind)
    for _ in range(8):
        jump = out["close"].pct_change().abs() > limit
        if not bool(jump.fillna(False).any()):
            break
        out = out.loc[~jump.fillna(False)].copy()
    return out.reset_index(drop=True)


def _cache_ok(key: str) -> Optional[pd.DataFrame]:
    cached = _load_cache(key)
    if cached is not None and not cached.empty and cached["date"].max() >= pd.Timestamp("2026-10-13"):
        return cached
    return None


def fetch_stock_sina(code: str, retries: int = 3) -> pd.DataFrame:
    """Sina qfq daily bars. Tencent qfq was rejected after it printed impossible moves."""
    key = f"{code}_sina"
    cached = _cache_ok(key)
    if cached is not None:
        return cached

    import akshare as ak

    def _once(start: str, end: str) -> pd.DataFrame:
        df = ak.stock_zh_a_daily(symbol=_prefix(code), start_date=start, end_date=end, adjust="qfq")
        return _strip_bad_ticks(_normalize_ohlc(df), code, "stock")

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            try:
                df = _once(FETCH_START, FETCH_END)
            except Exception:
                left = _once(FETCH_START, "20211231")
                right = _once("20220101", FETCH_END)
                df = _strip_bad_ticks(pd.concat([left, right], ignore_index=True), code, "stock")
            if df["date"].max() < pd.Timestamp("2026-10-13") or len(df) < 100:
                raise RuntimeError(f"{code} sina history incomplete: n={len(df)} last={df['date'].max()}")
            _save_cache(key, df)
            time.sleep(0.2)
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"fetch stock {code} failed: {last_err}")


def fetch_etf_sina(code: str, retries: int = 3) -> pd.DataFrame:
    """Sina exchange prints for ETFs. Unadjusted: cash dividends are a documented gap.

    Tencent qfq was kept only when its daily returns matched Sina. Several sector ETFs
    did not (return correlation well below 0.98, with impossible jumps), so the whole
    ETF panel uses one unadjusted source instead of a mixed adjusted/unadjusted set.
    """
    key = f"{code}_sina_etf"
    cached = _cache_ok(key)
    if cached is not None:
        return cached

    import akshare as ak

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            raw = ak.fund_etf_hist_sina(symbol=_prefix(code))
            df = _strip_bad_ticks(_normalize_ohlc(raw), code, "etf")
            df = df[(df["date"] >= "2018-01-01") & (df["date"] <= FETCH_END)].reset_index(drop=True)
            if df["date"].max() < pd.Timestamp("2026-10-13") or len(df) < 60:
                raise RuntimeError(f"{code} etf history incomplete: n={len(df)}")
            _save_cache(key, df)
            time.sleep(0.15)
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"fetch etf {code} failed: {last_err}")


def fetch_index(symbol: str) -> pd.DataFrame:
    cached = _load_cache(symbol)
    if cached is not None and cached["date"].max() >= pd.Timestamp("2026-10-13"):
        return cached
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol=symbol)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    df = df[(df["date"] >= "2018-01-01") & (df["date"] <= FETCH_END)].reset_index(drop=True)
    _save_cache(symbol, df)
    return df


def fetch_optional_macro() -> Dict[str, object]:
    """Best-effort macro prints. Missing fields stay missing."""
    out: Dict[str, object] = {"gaps": []}
    try:
        import akshare as ak

        bond = ak.bond_zh_us_rate(start_date="20240101")
        bond["日期"] = pd.to_datetime(bond["日期"])
        last = bond.dropna(subset=["中国国债收益率10年"]).sort_values("日期").iloc[-1]
        out["bond_date"] = str(pd.Timestamp(last["日期"]).date())
        for src, dst in (
            ("中国国债收益率2年", "cn_2y"),
            ("中国国债收益率10年", "cn_10y"),
            ("中国国债收益率30年", "cn_30y"),
            ("美国国债收益率10年", "us_10y"),
        ):
            val = last.get(src)
            out[dst] = None if pd.isna(val) else float(val)
    except Exception as exc:  # noqa: BLE001
        out["gaps"].append(f"bond_zh_us_rate unavailable: {exc}")

    try:
        import akshare as ak

        hsgt = ak.stock_hsgt_hist_em(symbol="北向资金")
        hsgt["日期"] = pd.to_datetime(hsgt["日期"])
        tail = hsgt.sort_values("日期").tail(5)
        net_col = "当日成交净买额"
        valid = tail[net_col].notna().sum() if net_col in tail.columns else 0
        out["hsgt_last_date"] = str(tail["日期"].iloc[-1].date())
        out["hsgt_recent_net_buy_non_null"] = int(valid)
        if valid == 0:
            out["gaps"].append(
                "北向资金当日成交净买额在最近样本中为空（交易所自2024年起不再披露实时/日频净买入明细），本报告不使用北向作为证据。"
            )
    except Exception as exc:  # noqa: BLE001
        out["gaps"].append(f"northbound history unavailable: {exc}")

    try:
        import akshare as ak

        spot = ak.stock_zh_index_spot_sina()
        # Columns vary by version; keep a compact snapshot of major indexes if present.
        out["intraday_rows"] = int(len(spot))
        out["intraday_columns"] = [str(c) for c in list(spot.columns)[:8]]
        name_col = "名称" if "名称" in spot.columns else None
        if name_col:
            wanted = spot[spot[name_col].astype(str).str.contains("上证指数|沪深300|创业板")]
            records = wanted.head(6).to_dict(orient="records")
            # Cast timestamps / numpy types
            clean = []
            for rec in records:
                item = {}
                for k, v in rec.items():
                    if isinstance(v, (np.floating, float)):
                        item[str(k)] = None if math.isnan(float(v)) else float(v)
                    elif isinstance(v, (np.integer, int)):
                        item[str(k)] = int(v)
                    else:
                        item[str(k)] = str(v)
                clean.append(item)
            out["intraday_sample"] = clean
    except Exception as exc:  # noqa: BLE001
        out["gaps"].append(f"intraday index spot unavailable: {exc}")

    try:
        import akshare as ak

        flow = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        out["sector_flow_rows"] = int(len(flow))
    except Exception as exc:  # noqa: BLE001
        out["gaps"].append(f"行业资金流接口失败，板块资金不做叙事证据: {type(exc).__name__}")
    return out


def repair_scale_breaks(close: pd.DataFrame, limits: Dict[str, float]) -> pd.DataFrame:
    """Remove permanent price-scale breaks without inventing a return.

    If a print jumps by more than the session limit and the new level persists,
    later percentage changes are kept and the jump itself is scaled out. A one-day
    spike that reverts is flattened by applying the same rule twice. This does not
    fill a missing session with a forecast.
    """
    out = close.copy()
    for col in out.columns:
        limit = limits.get(str(col), 0.25)
        series = out[col].copy()
        for _ in range(8):
            prev = series.ffill().shift(1)
            ratio = series / prev - 1.0
            jump = series.notna() & prev.notna() & ratio.abs().gt(limit)
            if not bool(jump.any()):
                break
            dt = jump[jump].index[0]
            factor = float(prev.loc[dt] / series.loc[dt])
            if not math.isfinite(factor) or factor <= 0:
                series.loc[dt] = np.nan
                continue
            series.loc[dt:] = series.loc[dt:] * factor
        out[col] = series
    return out


def panel_from_frames(
    frames: Dict[str, pd.DataFrame],
    limits: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes, amounts = [], []
    for code, df in frames.items():
        closes.append(df.set_index("date")["close"].rename(code))
        if "amount" in df.columns:
            amounts.append(df.set_index("date")["amount"].rename(code))
    close = pd.concat(closes, axis=1).sort_index()
    if limits:
        close = repair_scale_breaks(close, limits)
    # Suspension / single-session holes: hold last price up to 5 sessions.
    close = close.ffill(limit=5)
    amount = pd.concat(amounts, axis=1).sort_index() if amounts else pd.DataFrame(index=close.index)
    amount = amount.reindex(close.index)
    rets = close.pct_change()
    return close, rets, amount


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).tail(1)
    return pd.DatetimeIndex(last.values)


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    iso = index.isocalendar()
    s = pd.Series(index, index=index)
    key = iso.year.astype(str) + "-" + iso.week.astype(str)
    last = s.groupby(key).tail(1)
    return pd.DatetimeIndex(last.values)


def eligible_mask(close: pd.DataFrame, amount: pd.DataFrame, min_amount: float, lookback: int = 120) -> pd.DataFrame:
    hist = close.notna().rolling(lookback).sum() >= lookback
    if amount is None or amount.empty:
        return hist
    med = amount.rolling(20).median()
    liquid = med >= min_amount
    # If a name has no amount at all (index-like), do not block it.
    has_amount = amount.notna().rolling(20).sum() > 0
    liquid = liquid | ~has_amount
    return hist & liquid.reindex(index=close.index, columns=close.columns).fillna(False)


def _empty_weights(index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=columns)


def _materialize(raw: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill targets set on rebalance days; pre-first-signal stays cash."""
    w = raw.ffill().fillna(0.0)
    w = w.clip(lower=0.0)
    row_sum = w.sum(axis=1)
    over = row_sum > 1.0 + 1e-9
    if over.any():
        w.loc[over] = w.loc[over].div(row_sum[over], axis=0)
    return w


def strategy_tsmom(index_close: pd.Series, lookback: int = 120) -> pd.DataFrame:
    """Time-series momentum: long the CSI300 ETF when its lookback return is positive."""
    ret = index_close.pct_change(lookback)
    w = pd.DataFrame(0.0, index=index_close.index, columns=["510300"])
    long = ret > 0
    w.loc[long, "510300"] = 1.0
    # Need a full lookback; before that stay in cash.
    w.iloc[:lookback] = 0.0
    return w


def strategy_sector_dual_mom(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    lookback: int = 120,
    n_hold: int = 3,
) -> pd.DataFrame:
    """Absolute + relative momentum. Hold the top n_hold sectors with positive lookback return."""
    mom = close.pct_change(lookback)
    ok = eligible_mask(close, amount, ETF_MIN_AMOUNT, lookback)
    raw = _empty_weights(close.index, list(close.columns))
    for dt in month_ends(close.index):
        if dt not in mom.index:
            continue
        snap = mom.loc[dt]
        elig = ok.loc[dt].reindex(snap.index).fillna(False)
        pool = snap[elig & (snap > 0)].dropna()
        row = pd.Series(0.0, index=close.columns)
        if len(pool) > 0:
            picks = pool.nlargest(min(n_hold, len(pool))).index
            row.loc[picks] = 1.0 / len(picks)
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_sector_reversal(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    lookback: int = 5,
    n_hold: int = 3,
) -> pd.DataFrame:
    """Short-term reversal: weekly, equal-weight the worst lookback losers."""
    mom = close.pct_change(lookback)
    ok = eligible_mask(close, amount, ETF_MIN_AMOUNT, 60)
    raw = _empty_weights(close.index, list(close.columns))
    for dt in week_ends(close.index):
        snap = mom.loc[dt]
        elig = ok.loc[dt].reindex(snap.index).fillna(False)
        pool = snap[elig].dropna()
        row = pd.Series(0.0, index=close.columns)
        if len(pool) >= n_hold:
            picks = pool.nsmallest(n_hold).index
            row.loc[picks] = 1.0 / n_hold
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_low_vol(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    vol_window: int = 60,
    n_hold: int = 3,
) -> pd.DataFrame:
    rets = close.pct_change()
    vol = rets.rolling(vol_window).std()
    ok = eligible_mask(close, amount, ETF_MIN_AMOUNT, vol_window)
    raw = _empty_weights(close.index, list(close.columns))
    for dt in month_ends(close.index):
        snap = vol.loc[dt]
        elig = ok.loc[dt].reindex(snap.index).fillna(False)
        pool = snap[elig].dropna()
        row = pd.Series(0.0, index=close.columns)
        if len(pool) >= n_hold:
            picks = pool.nsmallest(n_hold).index
            row.loc[picks] = 1.0 / n_hold
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_regime_switch(
    csi: pd.Series,
    sector_w: pd.DataFrame,
    defensive_close: pd.DataFrame,
    defensive_amount: pd.DataFrame,
) -> pd.DataFrame:
    """Risk-on: sector dual momentum. Risk-off (CSI300 120d <= 0): defensive ETFs above their MA60."""
    signal = csi.pct_change(120)
    ma = defensive_close.rolling(60).mean()
    ok = eligible_mask(defensive_close, defensive_amount, ETF_MIN_AMOUNT, 60)
    cols = sorted(set(sector_w.columns) | set(defensive_close.columns))
    raw = _empty_weights(sector_w.index, cols)
    sector_w = sector_w.reindex(index=raw.index).fillna(0.0)
    for dt in month_ends(raw.index):
        row = pd.Series(0.0, index=cols)
        if dt not in signal.index or pd.isna(signal.loc[dt]):
            raw.loc[dt] = row
            continue
        if float(signal.loc[dt]) > 0:
            taken = sector_w.loc[dt]
            for code, weight in taken.items():
                if weight > 0 and code in row.index:
                    row.loc[code] = float(weight)
        else:
            above = (defensive_close.loc[dt] > ma.loc[dt]) & ok.loc[dt].reindex(defensive_close.columns).fillna(False)
            picks = [c for c in defensive_close.columns if bool(above.get(c, False))]
            if not picks:
                # Last-resort hedge if every defensive name is below its own trend.
                for code in ("518880", "511260"):
                    if code in row.index and pd.notna(defensive_close.loc[dt].get(code)):
                        picks.append(code)
            if picks:
                for code in picks:
                    row.loc[code] = 1.0 / len(picks)
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_equal_weight(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    min_amount: float,
) -> pd.DataFrame:
    """Monthly equal weight of every name that already has history and liquidity.

    This is the universe benchmark for stock strategies, not an alpha claim.
    """
    ok = eligible_mask(close, amount, min_amount, 60)
    raw = _empty_weights(close.index, list(close.columns))
    for dt in month_ends(close.index):
        elig = [c for c in close.columns if bool(ok.at[dt, c])]
        row = pd.Series(0.0, index=close.columns)
        if elig:
            row.loc[elig] = 1.0 / len(elig)
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_stock_momentum(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    lookback: int = 120,
    n_hold: int = 5,
) -> pd.DataFrame:
    mom = close.pct_change(lookback)
    ok = eligible_mask(close, amount, STOCK_MIN_AMOUNT, lookback)
    raw = _empty_weights(close.index, list(close.columns))
    for dt in month_ends(close.index):
        snap = mom.loc[dt]
        elig = ok.loc[dt].reindex(snap.index).fillna(False)
        pool = snap[elig & (snap > 0)].dropna()
        row = pd.Series(0.0, index=close.columns)
        if len(pool) > 0:
            picks = pool.nlargest(min(n_hold, len(pool))).index
            row.loc[picks] = 1.0 / len(picks)
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_stock_reversal(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    lookback: int = 5,
    n_hold: int = 5,
) -> pd.DataFrame:
    mom = close.pct_change(lookback)
    ok = eligible_mask(close, amount, STOCK_MIN_AMOUNT, 60)
    raw = _empty_weights(close.index, list(close.columns))
    for dt in week_ends(close.index):
        snap = mom.loc[dt]
        elig = ok.loc[dt].reindex(snap.index).fillna(False)
        pool = snap[elig].dropna()
        row = pd.Series(0.0, index=close.columns)
        if len(pool) >= n_hold:
            picks = pool.nsmallest(n_hold).index
            row.loc[picks] = 1.0 / n_hold
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_channel_breakout(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    entry: int = 60,
    exit_: int = 20,
    max_hold: int = 4,
) -> pd.DataFrame:
    """Daily channel breakout on sector ETFs. Equal-weight up to max_hold names, else cash."""
    prior_high = close.rolling(entry).max().shift(1)
    prior_low = close.rolling(exit_).min().shift(1)
    ok = eligible_mask(close, amount, ETF_MIN_AMOUNT, entry)
    mom = close.pct_change(entry)
    held: Dict[str, bool] = {c: False for c in close.columns}
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i, dt in enumerate(close.index):
        if i < entry:
            continue
        for code in close.columns:
            px = close.at[dt, code]
            if pd.isna(px) or not bool(ok.at[dt, code]):
                held[code] = False
                continue
            hi = prior_high.at[dt, code]
            lo = prior_low.at[dt, code]
            if held[code]:
                if pd.notna(lo) and px < lo:
                    held[code] = False
            elif pd.notna(hi) and px > hi:
                held[code] = True
        active = [c for c, flag in held.items() if flag]
        if len(active) > max_hold:
            score = mom.loc[dt, active].dropna().sort_values(ascending=False)
            active = list(score.index[:max_hold])
        if active:
            weights.loc[dt, active] = 1.0 / len(active)
    return weights


def strategy_liquor_rv(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Long-only relative value between Kweichow Moutai and Wuliangye.

    z-score of the log price ratio. When |z| > 1, hold only the cheaper name.
    Otherwise hold 50/50. This is not market-neutral and keeps liquor beta.
    """
    a, b = "600519", "000858"
    log_spread = np.log(close[a]) - np.log(close[b])
    mu = log_spread.rolling(window).mean()
    sd = log_spread.rolling(window).std()
    z = (log_spread - mu) / sd
    w = pd.DataFrame(0.0, index=close.index, columns=[a, b])
    both = close[a].notna() & close[b].notna() & z.notna()
    rich_a = both & (z > 1)  # Moutai rich vs history => hold Wuliangye
    rich_b = both & (z < -1)
    neutral = both & ~rich_a & ~rich_b
    w.loc[rich_a, b] = 1.0
    w.loc[rich_b, a] = 1.0
    w.loc[neutral, a] = 0.5
    w.loc[neutral, b] = 0.5
    return w


def apply_costs(weights: pd.DataFrame, rets: pd.DataFrame, cost_bps: float) -> Tuple[pd.Series, pd.Series]:
    """Lag weights by one session and charge one-way bps on turnover.

    weights.loc[t] is the target set at the close of t.
    It earns rets.loc[t+1]. Turnover traded at t is charged on t+1.
    """
    aligned = weights.reindex(index=rets.index, columns=rets.columns).fillna(0.0)
    asset_rets = rets.reindex(index=aligned.index, columns=aligned.columns).fillna(0.0)
    held = aligned.shift(1).fillna(0.0)
    gross = (held * asset_rets).sum(axis=1)
    turnover = aligned.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = float(aligned.iloc[0].abs().sum())
    cost = turnover.shift(1).fillna(0.0) * (cost_bps / 10000.0)
    net = gross - cost
    return net, turnover


def perf_stats(net: pd.Series, turnover: Optional[pd.Series] = None) -> Dict[str, float]:
    r = net.dropna()
    if len(r) < 5:
        return {}
    equity = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else float("nan")
    downside = r[r < 0]
    down_dev = float(downside.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(downside) > 2 else float("nan")
    sortino = float(r.mean() * TRADING_DAYS / down_dev) if down_dev and down_dev > 0 else float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("nan")
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    win_day = float((r > 0).mean())
    win_month = float((monthly > 0).mean()) if len(monthly) else float("nan")
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    ann_turn = float("nan")
    if turnover is not None:
        t = turnover.reindex(r.index).fillna(0.0)
        ann_turn = float(t.sum() / years) if years > 0 else float("nan")
    return {
        "days": int(len(r)),
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "win_rate_daily": win_day,
        "win_rate_monthly": win_month,
        "profit_factor": profit_factor,
        "calmar": calmar,
        "ann_turnover": ann_turn,
    }


def slice_stats(net: pd.Series, turnover: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
    mask = (net.index >= start) & (net.index <= end)
    return perf_stats(net.loc[mask], turnover.loc[mask])


def window_return(net: pd.Series, start: str, end: str) -> float:
    r = net.loc[start:end].dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def max_dd_window(net: pd.Series, start: str, end: str) -> float:
    r = net.loc[start:end].dropna()
    if r.empty:
        return float("nan")
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def vol_target_overlay(net: pd.Series, target: float = 0.10, window: int = 60) -> pd.Series:
    """De-risk only. Scale = min(1, target / lagged realized vol). Cash fills the rest.

    Extra turnover from scaling is approximated at ETF cost (8 bps) on |Δscale|.
    """
    rv = net.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)
    scale = (target / rv).clip(upper=1.0).shift(1)
    scale = scale.fillna(1.0).clip(lower=0.0, upper=1.0)
    gross = scale * net
    extra_turn = scale.diff().abs().fillna(0.0)
    return gross - extra_turn * (ETF_COST_BPS / 10000.0)


def latest_holdings(weights: pd.DataFrame, names: Dict[str, str], n: int = 6) -> List[str]:
    if weights.empty:
        return []
    last = weights.iloc[-1]
    held = last[last > 1e-6].sort_values(ascending=False)
    lines = []
    for code, weight in held.head(n).items():
        lines.append(f"{names.get(code, code)}({code}) {weight:.0%}")
    return lines


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{digits}f}"


def self_check_engine() -> None:
    """Synthetic path: +1% every day, fully invested after day 0. Cost hits once."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    rets = pd.DataFrame({"A": np.full(len(idx), 0.01)}, index=idx)
    w = pd.DataFrame({"A": np.ones(len(idx))}, index=idx)
    net, turnover = apply_costs(w, rets, cost_bps=100.0)  # 1% one-way, easy arithmetic
    # Day 0 earns 0 (no prior weight). Day 1 earns 1% minus the 1% entry cost => ~0.
    assert abs(net.iloc[0]) < 1e-12
    assert abs(net.iloc[1] - 0.0) < 1e-12
    assert abs(net.iloc[2] - 0.01) < 1e-12
    assert turnover.iloc[0] == 1.0
    assert turnover.iloc[2] == 0.0


def trailing_return(close: pd.Series, days: int) -> float:
    if len(close.dropna()) <= days:
        return float("nan")
    s = close.dropna()
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1.0)


def regime_label(above_ma60: float, ret_120: float) -> str:
    if above_ma60 >= 0.6 and ret_120 > 0:
        return "bull"
    if above_ma60 <= 0.4 and ret_120 < 0:
        return "bear"
    return "sideways"


def build_report(ctx: dict) -> str:
    last_date = ctx["last_date"]
    lines: List[str] = []
    a = lines.append
    a("# 交易策略研究日报 2026-10-14")
    a("")
    a("研究日期：2026-10-14（Asia/Shanghai）。")
    a(f"行情截止：**{last_date} 收盘**。脚本运行时 A 股当日交易未结束（午间），回测不使用未完成的 10-14 K 线。")
    a("市场：A 股可交易工具（场内 ETF 与高流动性个股）。多头、无杠杆。")
    a("复现：`python3 scripts/daily_strategy_research/run_research.py`")
    a("声明：以下为历史回测研究，不是投资建议，也不构成实盘委托。")
    a("")
    a("## 1. Executive Summary")
    a("")
    a(ctx["exec_summary"])
    a("")
    a("## 2. Market Analysis")
    a("")
    a(ctx["market_section"])
    a("")
    a("## 3. Strategy Details")
    a("")
    a("参数在看到本次结果之前已固定。敏感度表只做稳健性诊断，不用于改选参数。")
    a("")
    a("| ID | 经济逻辑 | 规格 | 成本 |")
    a("| --- | --- | --- | --- |")
    a("| BH_CSI300 | 宽基 beta，作为基准而不是 alpha | 全程持有 510300 沪深300ETF（新浪未复权） | 单边 8bps，仅建仓 |")
    a("| EW_Stocks | 股票池本身的 beta，用来区分「池子选得偏」和「规则有超额」 | 24 只股票里已有足够历史的名字月末等权 | 15bps |")
    a("| TSMOM_120 | 风险资产的时间序列动量：正趋势时风险溢价更易兑现，负趋势时空仓避开左尾 | 沪深300ETF，120 日收益>0 则满仓，否则现金；日频检查 | 8bps |")
    a("| Sector_DualMom | 部门资金与景气的相对强弱会延续数月；绝对动量过滤用来避免熊市硬做多 | 国内行业/风格 ETF，月末选 120 日收益最高且为正的 3 只，等权 | 8bps |")
    a("| Sector_Reversal | 短期过度反应在板块层面会部分回归；与动量是相反的持有期 | 同一 ETF 池，每周买入过去 5 日最弱的 3 只 | 8bps |")
    a("| Sector_LowVol | 低波动/红利类资产在 A 股有持有期溢价，且回撤通常浅于高 beta | 月末选 60 日波动最低的 3 只行业 ETF，等权 | 8bps |")
    a("| Regime_Switch | 同一套动量在熊市会失效；用指数自身趋势把仓位切到银行/红利/煤炭/电力/黄金/国债 | 沪深300 120 日收益>0 时用 Sector_DualMom，否则等权持有仍在自身 MA60 之上的防御 ETF | 8bps |")
    a("| Channel_Breakout | 趋势以波动区间突破的形式扩散，假突破由 20 日低点离场约束 | 行业 ETF 收盘突破前 60 日高点则持有，跌破前 20 日低点离场；最多 4 只 | 8bps |")
    a("| Stock_CS_Mom | 个股截面动量来自基本面与资金抱团的缓慢扩散，绝对动量避免逆势抄底 | 24 只跨行业大盘股，月末选 120 日收益为正的前 5，等权 | 15bps |")
    a("| Stock_Reversal | 大盘股一周内超卖有流动性提供溢价，但崩盘期会接飞刀 | 同一股票池，每周买入过去 5 日最弱的 5 只 | 15bps |")
    a("| Liquor_RV | 茅台与五粮液共享白酒基本面，比价偏离后相对价格会回归；策略仍暴露白酒 beta | 60 日对数比价 z 分数，\\|z\\|>1 时只持有相对便宜的一只，否则 50/50 | 15bps |")
    a("")
    a("成交与流动性约束：")
    a("")
    a("- 信号用 t 日收盘，收益从 t+1 日收盘到再下一根收盘才计入，避免用到信号当日涨跌。")
    a("- ETF 使用新浪未复权成交价，现金分红没有加回，高股息品种的收益被低估；这是数据缺口，不用估计分红去填。")
    a("- 股票使用新浪前复权。腾讯前复权在中国神华等标的上出现过不可能的单日涨跌，已整段弃用。")
    a("- ETF 过去 20 日成交额中位数低于 2000 万人民币则当天不可入选；股票阈值为 5000 万。")
    a("- 停牌或单日缺口最多向前填充 5 个交易日；更长缺口不编造价格。")
    a("- 现金收益按 0 计。2026-10-13 中国 10 年国债收益率约 1.7%，现金拖累被略微高估。")
    a("- 个股单边 15bps、ETF 单边 8bps 是保守零售成本（佣金、卖出印花税摊销、滑点），不是交易所最低佣金。")
    a("")
    a("实现见 `scripts/daily_strategy_research/run_research.py`。成本与滞后是下面这段：")
    a("")
    a("```python")
    a("def apply_costs(weights, rets, cost_bps):")
    a("    # weights at close t earn rets at t+1; turnover at t is charged on t+1")
    a("    held = weights.shift(1).fillna(0.0)")
    a("    gross = (held * rets.fillna(0.0)).sum(axis=1)")
    a("    turnover = weights.diff().abs().sum(axis=1)")
    a("    turnover.iloc[0] = weights.iloc[0].abs().sum()")
    a("    cost = turnover.shift(1).fillna(0.0) * (cost_bps / 10000.0)")
    a("    return gross - cost, turnover")
    a("```")
    a("")
    a("## 4. Backtest Results")
    a("")
    a(f"统一比较窗口：{ctx['eval_start']} 至 {last_date}。样本内 ≤ 2023-12-31，样本外 ≥ 2024-01-01。")
    a("夏普 / 索提诺使用日收益、无风险利率 0、按 252 天年化。胜率同时给日频与月频；日频胜率接近 50% 并不否定正期望。")
    a("")
    a(ctx["full_table"])
    a("")
    a("### 样本内 / 样本外")
    a("")
    a(ctx["split_table"])
    a("")
    a("### 压力窗口收益（含成本）")
    a("")
    a(ctx["stress_table"])
    a("")
    a("### 相对沪深300ETF 的超额与相关")
    a("")
    a(ctx["excess_table"])
    a("")
    a("### 预注册敏感度（不改选参数）")
    a("")
    a(ctx["sensitivity_table"])
    a("")
    a("### 10% 波动目标覆盖（只减仓、不加杠杆）")
    a("")
    a("规则预先固定：杠杆 = min(1, 10% / 过去 60 日已实现波动)，滞后 1 日。这是风控，不是另一套 alpha。")
    a("")
    a(ctx["overlay_table"])
    a("")
    a("权益曲线数据：`scripts/daily_strategy_research/output/equity_curves.csv`。")
    if ctx.get("chart_note"):
        a(ctx["chart_note"])
    a("")
    a("## 5. Comparison & Ranking")
    a("")
    a("质量门槛（技能要求，须同时满足才算合格策略）：")
    a("")
    a("1. 扣费后跑赢沪深300ETF（全样本超额 CAGR > 0，且样本外收益不低于基准）。")
    a("2. 全样本夏普 > 1.0。")
    a("3. 全样本最大回撤浅于 30%。")
    a("4. 样本外夏普 > 0.5，且不是只在单一窗口有效。")
    a("5. 经济逻辑清楚，流动性与成本下可执行，不过度依赖单一参数或单一标的。")
    a("")
    a(ctx["ranking_section"])
    a("")
    a("## 6. Final Recommendations")
    a("")
    a(ctx["recommend_section"])
    a("")
    a("## 7. Deployment Considerations")
    a("")
    a(ctx["deploy_section"])
    a("")
    a("## 8. What Could Go Wrong")
    a("")
    a(ctx["risk_section"])
    a("")
    a("## 自检")
    a("")
    a(ctx["self_check"])
    a("")
    a("## 数据与假设")
    a("")
    a(ctx["data_section"])
    a("")
    return "\n".join(lines)


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def run() -> None:
    self_check_engine()
    print("engine self-check ok")

    index_frames: Dict[str, pd.DataFrame] = {}
    for symbol, name in INDEXES.items():
        try:
            index_frames[symbol] = fetch_index(symbol)
            print(f"index {name} {index_frames[symbol]['date'].iloc[-1].date()} n={len(index_frames[symbol])}")
        except Exception as exc:  # noqa: BLE001
            print(f"index fail {symbol}: {exc}")

    etf_meta = {**SECTOR_ETFS, **EXTRA_ETFS}
    etf_frames: Dict[str, pd.DataFrame] = {}
    failed = []
    for code, name in etf_meta.items():
        try:
            etf_frames[code] = fetch_etf_sina(code)
            print(f"etf {code} {name} {etf_frames[code]['date'].iloc[-1].date()} n={len(etf_frames[code])}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code} {name}: {exc}")
            print(f"etf fail {code}: {exc}")

    stock_frames: Dict[str, pd.DataFrame] = {}
    for code, name in STOCKS.items():
        try:
            stock_frames[code] = fetch_stock_sina(code)
            print(f"stk {code} {name} {stock_frames[code]['date'].iloc[-1].date()} n={len(stock_frames[code])}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code} {name}: {exc}")
            print(f"stk fail {code}: {exc}")

    macro_path = OUT / "macro.json"
    if macro_path.exists():
        macro = json.loads(macro_path.read_text(encoding="utf-8"))
    else:
        macro = fetch_optional_macro()
        macro_path.write_text(json.dumps(macro, ensure_ascii=False, indent=2), encoding="utf-8")

    if "510300" not in etf_frames:
        raise SystemExit("benchmark ETF 510300 missing; abort rather than invent a fill")

    sector_codes = [c for c in SECTOR_ETFS if c in etf_frames]
    defensive_codes = [c for c in DEFENSIVE_ETFS if c in etf_frames]
    etf_limits = {code: 0.25 for code in etf_frames}
    stock_limits = {
        code: (0.23 if code.startswith(("300", "301", "688", "689")) else 0.115)
        for code in stock_frames
    }
    etf_close, etf_rets, etf_amount = panel_from_frames(etf_frames, etf_limits)
    stock_close, stock_rets, stock_amount = panel_from_frames(stock_frames, stock_limits)

    # Impossible session moves are not filled. A hole longer than the 5-day
    # price carry already becomes a missing return; this mask also drops any
    # short-hole jump that survived the carry.
    etf_rets = etf_rets.mask(etf_rets.abs() > 0.25)
    for col in stock_rets.columns:
        limit = 0.23 if str(col).startswith(("300", "301", "688", "689")) else 0.115
        stock_rets[col] = stock_rets[col].mask(stock_rets[col].abs() > limit)
    etf_extreme = int((etf_rets.abs() > 0.18).sum().sum())
    stock_extreme = int((stock_rets.abs() > 0.11).sum().sum())

    last_date = etf_close["510300"].dropna().index.max()
    print("last_date", last_date.date(), "large_etf", etf_extreme, "large_stock", stock_extreme)

    csi = etf_close["510300"]
    sector_close = etf_close[sector_codes]
    sector_amount = etf_amount.reindex(columns=sector_codes)
    def_close = etf_close[defensive_codes]
    def_amount = etf_amount.reindex(columns=defensive_codes)

    weights: Dict[str, pd.DataFrame] = {}
    weights["BH_CSI300"] = pd.DataFrame(1.0, index=etf_close.index, columns=["510300"])
    weights["EW_Stocks"] = strategy_equal_weight(stock_close, stock_amount, STOCK_MIN_AMOUNT)
    weights["TSMOM_120"] = strategy_tsmom(csi, 120)
    weights["Sector_DualMom"] = strategy_sector_dual_mom(sector_close, sector_amount, 120, 3)
    weights["Sector_Reversal"] = strategy_sector_reversal(sector_close, sector_amount, 5, 3)
    weights["Sector_LowVol"] = strategy_low_vol(sector_close, sector_amount, 60, 3)
    weights["Regime_Switch"] = strategy_regime_switch(
        csi, weights["Sector_DualMom"], def_close, def_amount
    )
    weights["Channel_Breakout"] = strategy_channel_breakout(sector_close, sector_amount, 60, 20, 4)
    weights["Stock_CS_Mom"] = strategy_stock_momentum(stock_close, stock_amount, 120, 5)
    weights["Stock_Reversal"] = strategy_stock_reversal(stock_close, stock_amount, 5, 5)
    weights["Liquor_RV"] = strategy_liquor_rv(stock_close, 60)

    # Sensitivity diagnostics. Not used for selection.
    sensitivity = {
        "TSMOM_60": strategy_tsmom(csi, 60),
        "TSMOM_120": weights["TSMOM_120"],
        "TSMOM_250": strategy_tsmom(csi, 250),
        "DualMom_60_top3": strategy_sector_dual_mom(sector_close, sector_amount, 60, 3),
        "DualMom_120_top3": weights["Sector_DualMom"],
        "DualMom_180_top3": strategy_sector_dual_mom(sector_close, sector_amount, 180, 3),
        "DualMom_120_top2": strategy_sector_dual_mom(sector_close, sector_amount, 120, 2),
        "DualMom_120_top4": strategy_sector_dual_mom(sector_close, sector_amount, 120, 4),
    }

    cost_map = {
        "BH_CSI300": ETF_COST_BPS,
        "EW_Stocks": STOCK_COST_BPS,
        "TSMOM_120": ETF_COST_BPS,
        "Sector_DualMom": ETF_COST_BPS,
        "Sector_Reversal": ETF_COST_BPS,
        "Sector_LowVol": ETF_COST_BPS,
        "Regime_Switch": ETF_COST_BPS,
        "Channel_Breakout": ETF_COST_BPS,
        "Stock_CS_Mom": STOCK_COST_BPS,
        "Stock_Reversal": STOCK_COST_BPS,
        "Liquor_RV": STOCK_COST_BPS,
    }
    ret_map = {
        "BH_CSI300": etf_rets,
        "EW_Stocks": stock_rets,
        "TSMOM_120": etf_rets,
        "Sector_DualMom": etf_rets,
        "Sector_Reversal": etf_rets,
        "Sector_LowVol": etf_rets,
        "Regime_Switch": etf_rets,
        "Channel_Breakout": etf_rets,
        "Stock_CS_Mom": stock_rets,
        "Stock_Reversal": stock_rets,
        "Liquor_RV": stock_rets,
    }

    nets: Dict[str, pd.Series] = {}
    turns: Dict[str, pd.Series] = {}
    stats: Dict[str, Dict] = {}
    for name, w in weights.items():
        net, turnover = apply_costs(w, ret_map[name], cost_map[name])
        net = net.loc[EVAL_START:last_date]
        turnover = turnover.loc[EVAL_START:last_date]
        nets[name] = net
        turns[name] = turnover
        full = perf_stats(net, turnover)
        ins = slice_stats(net, turnover, EVAL_START, IS_END)
        oos = slice_stats(net, turnover, OOS_START, last_date)
        stats[name] = {"full": full, "is": ins, "oos": oos}

    bh = nets["BH_CSI300"]
    for name, net in nets.items():
        aligned = pd.concat([net.rename("s"), bh.rename("b")], axis=1).dropna()
        excess = aligned["s"] - aligned["b"]
        ex_stats = perf_stats(excess)
        stats[name]["excess_cagr"] = ex_stats.get("cagr", float("nan"))
        stats[name]["corr"] = float(aligned["s"].corr(aligned["b"])) if len(aligned) > 10 else float("nan")
        stats[name]["stress"] = {
            "covid_2020q1": window_return(net, "2020-01-15", "2020-03-31"),
            "year_2022": window_return(net, "2022-01-01", "2022-12-31"),
            "rally_202409_202503": window_return(net, "2024-09-01", "2025-03-31"),
            "ytd_2026": window_return(net, "2026-01-01", "2026-10-13"),
            "mdd_2022": max_dd_window(net, "2022-01-01", "2022-12-31"),
        }
        ov = vol_target_overlay(net)
        stats[name]["overlay"] = perf_stats(ov)
        stats[name]["excess_vs_ew_cagr"] = float("nan")

    ew = nets["EW_Stocks"]
    for name, net in nets.items():
        aligned = pd.concat([net.rename("s"), ew.rename("b")], axis=1).dropna()
        excess = aligned["s"] - aligned["b"]
        stats[name]["excess_vs_ew_cagr"] = perf_stats(excess).get("cagr", float("nan"))

    sens_rows = {}
    for name, w in sensitivity.items():
        net, turnover = apply_costs(w, etf_rets, ETF_COST_BPS)
        net = net.loc[EVAL_START:last_date]
        turnover = turnover.loc[EVAL_START:last_date]
        sens_rows[name] = {
            "full": perf_stats(net, turnover),
            "oos": slice_stats(net, turnover, OOS_START, last_date),
        }

    # Market snapshot
    index_rows = []
    for symbol, name in INDEXES.items():
        if symbol not in index_frames:
            continue
        s = index_frames[symbol].set_index("date")["close"].sort_index()
        s = s.loc[:last_date]
        ma60 = s.rolling(60).mean()
        ret20 = trailing_return(s, 20)
        vol20 = float(s.pct_change().tail(20).std(ddof=1) * math.sqrt(TRADING_DAYS))
        dd252 = float(s.iloc[-1] / s.tail(252).max() - 1.0)
        index_rows.append(
            {
                "name": name,
                "symbol": symbol,
                "last": float(s.iloc[-1]),
                "d1": trailing_return(s, 1),
                "m1": ret20,
                "m3": trailing_return(s, 60),
                "m6": trailing_return(s, 120),
                "y1": trailing_return(s, 240),
                "vol20": vol20,
                "dd252": dd252,
                "above_ma60": bool(s.iloc[-1] > ma60.iloc[-1]),
            }
        )

    sec_rows = []
    for code in sector_codes + [c for c in defensive_codes if c not in sector_codes]:
        s = etf_close[code].dropna().loc[:last_date]
        if s.empty:
            continue
        ma60 = s.rolling(60).mean()
        ma120 = s.rolling(120).mean()
        sec_rows.append(
            {
                "code": code,
                "name": etf_meta.get(code, code),
                "m1": trailing_return(s, 20),
                "m3": trailing_return(s, 60),
                "m6": trailing_return(s, 120),
                "y1": trailing_return(s, 240),
                "vol60": float(s.pct_change().tail(60).std(ddof=1) * math.sqrt(TRADING_DAYS)),
                "above_ma60": bool(s.iloc[-1] > ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else False,
                "above_ma120": bool(s.iloc[-1] > ma120.iloc[-1]) if pd.notna(ma120.iloc[-1]) else False,
            }
        )

    sector_only = [r for r in sec_rows if r["code"] in sector_codes]
    above60 = sum(1 for r in sector_only if r["above_ma60"]) / max(len(sector_only), 1)
    csi_ret120 = trailing_return(csi.loc[:last_date], 120)
    regime = regime_label(above60, csi_ret120)

    names = {**etf_meta, **STOCKS}
    holdings = {k: latest_holdings(weights[k].loc[:last_date], names) for k in weights}

    equity = pd.DataFrame({k: (1.0 + v.fillna(0.0)).cumprod() for k, v in nets.items()})
    equity.to_csv(OUT / "equity_curves.csv")
    payload = {
        "last_date": str(last_date.date()),
        "regime": regime,
        "above_ma60": above60,
        "csi_ret120": csi_ret120,
        "failed_symbols": failed,
        "extreme_etf_moves_gt_22pct": etf_extreme,
        "extreme_stock_moves_gt_22pct": stock_extreme,
        "stats": stats,
        "sensitivity": {
            k: {"full_sharpe": v["full"].get("sharpe"), "full_mdd": v["full"].get("max_dd"), "oos_sharpe": v["oos"].get("sharpe")}
            for k, v in sens_rows.items()
        },
        "index_rows": index_rows,
        "sector_rows": sec_rows,
        "holdings": holdings,
        "macro_gaps": macro.get("gaps", []),
        "bond": {k: macro.get(k) for k in ("bond_date", "cn_2y", "cn_10y", "cn_30y", "us_10y")},
        "sector_codes": sector_codes,
        "defensive_codes": defensive_codes,
        "stock_codes": list(stock_frames),
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    chart_note = ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 6))
        for col in equity.columns:
            ax.plot(equity.index, equity[col], label=col, linewidth=1.2)
        ax.set_title(f"Net equity (costs included), from {EVAL_START.date()} to {last_date.date()}")
        ax.set_ylabel("Growth of 1")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = OUT / "equity_curves.png"
        fig.savefig(fig_path, dpi=120)
        plt.close()
        chart_note = "图：`scripts/daily_strategy_research/output/equity_curves.png`（英文图例，避免环境缺中文字体）。"
    except Exception as exc:  # noqa: BLE001
        chart_note = f"权益曲线图未生成：{exc}"

    # ----- report text -----
    order = list(weights.keys())

    def row_full(name: str) -> List[str]:
        s = stats[name]["full"]
        return [
            name,
            fmt_pct(s["total_return"]),
            fmt_pct(s["cagr"]),
            fmt_num(s["sharpe"]),
            fmt_num(s["sortino"]),
            fmt_pct(s["max_dd"]),
            fmt_pct(s["win_rate_daily"], 0),
            fmt_pct(s["win_rate_monthly"], 0),
            fmt_num(s["profit_factor"]),
            fmt_num(s["ann_turnover"], 1),
        ]

    full_table = markdown_table(
        ["Strategy", "Total", "CAGR", "Sharpe", "Sortino", "MaxDD", "WinD", "WinM", "PF", "Turn/yr"],
        [row_full(n) for n in order],
    )

    def row_split(name: str) -> List[str]:
        a, b = stats[name]["is"], stats[name]["oos"]
        return [
            name,
            fmt_pct(a.get("cagr", float("nan"))),
            fmt_num(a.get("sharpe", float("nan"))),
            fmt_pct(a.get("max_dd", float("nan"))),
            fmt_pct(b.get("cagr", float("nan"))),
            fmt_num(b.get("sharpe", float("nan"))),
            fmt_pct(b.get("max_dd", float("nan"))),
            fmt_pct(b.get("total_return", float("nan"))),
        ]

    split_table = markdown_table(
        ["Strategy", "IS CAGR", "IS Sharpe", "IS MDD", "OOS CAGR", "OOS Sharpe", "OOS MDD", "OOS Total"],
        [row_split(n) for n in order],
    )

    stress_table = markdown_table(
        ["Strategy", "2020Q1", "2022", "2022 MDD", "2024-09~2025-03", "2026 YTD"],
        [
            [
                n,
                fmt_pct(stats[n]["stress"]["covid_2020q1"]),
                fmt_pct(stats[n]["stress"]["year_2022"]),
                fmt_pct(stats[n]["stress"]["mdd_2022"]),
                fmt_pct(stats[n]["stress"]["rally_202409_202503"]),
                fmt_pct(stats[n]["stress"]["ytd_2026"]),
            ]
            for n in order
        ],
    )

    excess_table = markdown_table(
        ["Strategy", "Excess CAGR vs BH", "Corr vs BH", "Full Sharpe", "OOS Sharpe"],
        [
            [
                n,
                fmt_pct(stats[n]["excess_cagr"]),
                fmt_num(stats[n]["corr"]),
                fmt_num(stats[n]["full"]["sharpe"]),
                fmt_num(stats[n]["oos"]["sharpe"]),
            ]
            for n in order
        ],
    )

    sensitivity_table = markdown_table(
        ["Spec", "Full Sharpe", "Full MDD", "OOS Sharpe"],
        [
            [
                n,
                fmt_num(sens_rows[n]["full"]["sharpe"]),
                fmt_pct(sens_rows[n]["full"]["max_dd"]),
                fmt_num(sens_rows[n]["oos"]["sharpe"]),
            ]
            for n in sensitivity
        ],
    )

    overlay_table = markdown_table(
        ["Strategy", "Raw Sharpe", "Raw MDD", "Overlay Sharpe", "Overlay MDD", "Overlay CAGR"],
        [
            [
                n,
                fmt_num(stats[n]["full"]["sharpe"]),
                fmt_pct(stats[n]["full"]["max_dd"]),
                fmt_num(stats[n]["overlay"].get("sharpe", float("nan"))),
                fmt_pct(stats[n]["overlay"].get("max_dd", float("nan"))),
                fmt_pct(stats[n]["overlay"].get("cagr", float("nan"))),
            ]
            for n in order
        ],
    )

    # Quality bar
    qual_lines = []
    survivors = []
    for n in order:
        if n in {"BH_CSI300", "EW_Stocks"}:
            continue
        f, o = stats[n]["full"], stats[n]["oos"]
        checks = {
            "excess_cagr>0": stats[n]["excess_cagr"] > 0,
            "oos_total>=bh": o.get("total_return", -1) >= stats["BH_CSI300"]["oos"].get("total_return", 0),
            "sharpe>1": f.get("sharpe", 0) > 1.0,
            "mdd>-30%": f.get("max_dd", -1) > -0.30,
            "oos_sharpe>0.5": o.get("sharpe", 0) > 0.5,
        }
        passed = all(checks.values())
        if passed:
            survivors.append(n)
        flag = "PASS" if passed else "FAIL"
        detail = ", ".join(f"{k}:{'Y' if v else 'N'}" for k, v in checks.items())
        qual_lines.append(f"- **{n}** {flag} — {detail}")

    # Rank by a fixed score among all non-benchmark, but label pass/fail separately.
    def score_of(n: str) -> float:
        f, o = stats[n]["full"], stats[n]["oos"]
        sharpe = f.get("sharpe", float("nan"))
        oos_s = o.get("sharpe", float("nan"))
        ex = stats[n]["excess_cagr"]
        mdd = f.get("max_dd", float("nan"))
        parts = [sharpe, oos_s, ex, mdd]
        if any(isinstance(p, float) and math.isnan(p) for p in parts):
            return -999.0
        # Reward risk-adjusted return and penalize drawdown. Not optimized.
        return 0.35 * sharpe + 0.30 * oos_s + 0.20 * (ex / 0.05) + 0.15 * (mdd / 0.30)

    ranked = sorted([n for n in order if n not in {"BH_CSI300", "EW_Stocks"}], key=score_of, reverse=True)
    rank_lines = ["按固定加权分排序（夏普、样本外夏普、超额、回撤），只用于并列展示，不能把 FAIL 说成合格：", ""]
    for i, n in enumerate(ranked, 1):
        f = stats[n]["full"]
        mark = "合格" if n in survivors else "未过门槛"
        rank_lines.append(
            f"{i}. **{n}**（{mark}，score {score_of(n):.2f}）— 全样本夏普 {fmt_num(f['sharpe'])}，"
            f"回撤 {fmt_pct(f['max_dd'])}，相对沪深300超额 CAGR {fmt_pct(stats[n]['excess_cagr'])}，"
            f"相对股票等权超额 CAGR {fmt_pct(stats[n].get('excess_vs_ew_cagr', float('nan')))}，"
            f"样本外夏普 {fmt_num(stats[n]['oos']['sharpe'])}。"
        )
    rank_lines.append("")
    rank_lines.append(
        "股票策略如果只跑赢沪深300、但跑不赢同一股票池的等权（EW_Stocks），"
        "超额来自池子本身（含幸存者偏差），不能记成交易规则的 edge。"
    )

    # Market section
    idx_table = markdown_table(
        ["指数", "收盘", "1日", "20日", "60日", "120日", "240日", "20日波动", "距252日高点", ">MA60"],
        [
            [
                r["name"],
                f"{r['last']:.2f}",
                fmt_pct(r["d1"]),
                fmt_pct(r["m1"]),
                fmt_pct(r["m3"]),
                fmt_pct(r["m6"]),
                fmt_pct(r["y1"]),
                fmt_pct(r["vol20"]),
                fmt_pct(r["dd252"]),
                "Y" if r["above_ma60"] else "N",
            ]
            for r in index_rows
        ],
    )
    sec_table = markdown_table(
        ["代码", "名称", "20日", "60日", "120日", "240日", "60日波动", ">MA60", ">MA120"],
        [
            [
                r["code"],
                r["name"],
                fmt_pct(r["m1"]),
                fmt_pct(r["m3"]),
                fmt_pct(r["m6"]),
                fmt_pct(r["y1"]),
                fmt_pct(r["vol60"]),
                "Y" if r["above_ma60"] else "N",
                "Y" if r["above_ma120"] else "N",
            ]
            for r in sorted(sec_rows, key=lambda x: -999 if math.isnan(x["m3"]) else -x["m3"])
        ],
    )

    bond = payload["bond"]
    bond_txt = (
        f"国债收益率（AkShare `bond_zh_us_rate`，日期 {bond.get('bond_date')}）："
        f"中国 2Y {bond.get('cn_2y')}%，10Y {bond.get('cn_10y')}%，30Y {bond.get('cn_30y')}%；"
        f"美国 10Y {bond.get('us_10y')}%。该美债点位未用第二数据源交叉验证，只作为宏观背景打印。"
        if bond.get("cn_10y") is not None
        else "国债收益率本次没有取到。"
    )

    leaders = sorted(sector_only, key=lambda r: r["m3"] if not math.isnan(r["m3"]) else -999, reverse=True)
    laggards = list(reversed(leaders))
    lead_txt = "、".join(f"{r['name']} {fmt_pct(r['m3'])}" for r in leaders[:3])
    lag_txt = "、".join(f"{r['name']} {fmt_pct(r['m3'])}" for r in laggards[:3])

    market_section = "\n".join(
        [
            f"规则化状态（预先定义）：行业 ETF 中收盘高于 MA60 的比例为 **{above60:.0%}**，沪深300ETF 120 日收益 **{fmt_pct(csi_ret120)}**，因此状态标为 **{regime}**。",
            "",
            "指数（AkShare `stock_zh_index_daily`，至最后完整交易日）：",
            "",
            idx_table,
            "",
            "行业与防御 ETF（新浪未复权收盘价，不含现金分红）：",
            "",
            sec_table,
            "",
            f"近 60 个交易日相对强的篮子：{lead_txt}。",
            f"近 60 个交易日相对弱的篮子：{lag_txt}。",
            "",
            bond_txt,
            "",
            "事实与解释分开：",
            "",
            f"- 事实：宽基 120 日收益与 MA60 广度给出的标签是 {regime}。上表的涨跌、波动和回撤是价格计算，不是叙事。",
            "- 事实：行业资金流接口本次失败，北向日频净买入在最近样本中为空。这两项不进入策略排序。",
            "- 解释：若广度低于 40% 且 120 日收益为负，动量满仓的赔率下降，防御与绝对动量过滤的权重应高于追涨。这是解释，要由第 4 节的扣费回测决定，而不是由解释本身决定。",
        ]
    )

    hold_lines = ["截至最后完整交易日，各策略目标仓位（下一交易日才按规则成交；此处只展示信号，不是下单）：", ""]
    for n in order:
        held = holdings.get(n) or ["现金"]
        hold_lines.append(f"- {n}: {('、'.join(held)) if held else '现金'}")

    if survivors:
        rec_intro = (
            f"过门槛的策略有 {len(survivors)} 个：{', '.join(survivors)}。"
            "下面按排序给出可研究的部署优先级，仓位必须再受波动目标约束。"
        )
    else:
        rec_intro = (
            "没有策略同时通过全部质量门槛。下面的名单是「研究观察顺序」，不是已验证可上实盘的 edge。"
            "任何一条都不能表述成稳定超额。"
        )

    rec_bits = [rec_intro, ""]
    for n in ranked[:5]:
        f, o = stats[n]["full"], stats[n]["oos"]
        rec_bits.append(
            f"- **{n}**：全样本 CAGR {fmt_pct(f['cagr'])}，夏普 {fmt_num(f['sharpe'])}，"
            f"最大回撤 {fmt_pct(f['max_dd'])}，样本外夏普 {fmt_num(o['sharpe'])}，"
            f"相对沪深300超额 CAGR {fmt_pct(stats[n]['excess_cagr'])}，"
            f"相对股票等权超额 CAGR {fmt_pct(stats[n].get('excess_vs_ew_cagr', float('nan')))}。"
            f"当前信号：{('、'.join(holdings.get(n) or [])) or '现金'}。"
        )
    rec_bits.append("")
    rec_bits.append("不作为推荐的做法：用本次敏感度表里夏普最高的参数替换预注册参数；把单一行业主题的历史 beta 当成策略 alpha。")

    deploy = "\n".join(
        [
            "- 交易时点：收盘信号，下一交易日成交。回测用下一根收盘到收盘近似，实盘用开盘或 VWAP 会再吃一点滑点。",
            "- 仓位：单策略名义敞口先按 10% 年化波动目标缩放，且杠杆不超过 1。回撤超过策略自身历史最大回撤的 1.25 倍时降到一半仓位并人工复核。",
            "- 容量：行业 ETF 只交易 20 日成交额中位数 ≥ 2000 万的品种；单日成交不超过该 ETF 20 日均成交额的 2%。股票池限定本次 24 只大盘股，不外推到小盘。",
            "- 成本监控：若实盘单边冲击超过回测假设（ETF 8bps / 股票 15bps），停止加仓。",
            "- 组合：动量与反转相关性不同，若两者都只是弱超额，不要因为「看起来不同」就叠加杠杆。",
            "- 现金：回测现金收益为 0。实盘空仓可放货币 ETF，但这部分收益不属于策略 alpha。",
            "",
            *hold_lines,
        ]
    )

    risk = "\n".join(
        [
            "- 动量类（TSMOM、Sector_DualMom、Stock_CS_Mom、Channel_Breakout）在快速反转和震荡市会连续止损。失效条件：样本外滚动 12 个月超额持续为负，或 120 日与 60/180 日敏感度符号相反。",
            "- 反转类（Sector_Reversal、Stock_Reversal）在单边下跌中买入最弱资产。失效条件：2022 这类年份的回撤深于基准 10 个百分点以上，且周频换手的成本上升。",
            "- Regime_Switch 的开关就是沪深300 120 日收益。失效条件：指数在零轴附近来回穿越，导致防御与进攻来回抽费；或防御 ETF 自身趋势同时破裂（利率急升打击红利，商品下跌打击黄金和煤炭）。",
            "- Liquor_RV 不是市场中性。白酒需求或估值系统性下移时，两只股票会一起跌。失效条件：比价 z 分数突破历史 3 倍标准差后不回归，或其中一只基本面结构性偏离（例如渠道库存或公司治理事件）。",
            "- 幸存者偏差：股票池和 ETF 池都是 2026 年仍在交易的主流工具，没有加入已清盘 ETF 和退市股。结果对「当时可投资宇宙」偏乐观。",
            "- 制度变化：印花税、涨跌停、T+1、行业 ETF 扩容都会改变成本和收益结构。2023-08 印花税下调前，股票策略的真实成本高于本报告使用的 15bps 单边假设的一部分年份，另一部分年份则更接近。15bps 是常数近似。",
            "- 数据：前复权在大比例分红除权日可能残留跳变。极端日涨跌超过 22% 的次数已记在数据节。",
            "- 宏观打印若与国债期货或财政部曲线不一致，以官方曲线为准，本报告不据此交易利率。",
        ]
    )

    self_check = "\n".join(
        [
            "- 缺了什么：行业资金流、北向净买入明细、期权持仓与隐含波动率、一致预期修正。这些缺失被记为缺口，没有用估计值填上。",
            "- 假设里较弱的一环：现金收益为 0、成本为常数 bps、ETF 收盘可成交、股票池有幸存者偏差。",
            "- Edge 还是拟合：参数在跑数前固定；敏感度若与 120 日 / top3 差异很大，对应策略只记为参数脆弱，不改冠军。合格与否看门槛，不看本次最高夏普。",
            "- 制度覆盖：2020Q1、2022 熊市、2024 年 9 月以来的上涨、2026 年初至最近收盘都单独列了收益。样本内止于 2023，样本外从 2024 起。",
            "- 会改变结论的证据：样本外继续一年后超额转负；实盘滑点显著高于 8/15bps；同一规则在包含退市股的宇宙里消失。",
        ]
    )

    gap_lines = [f"- {g}" for g in macro.get("gaps", [])] or ["- 宏观接口没有额外报错。"]
    if failed:
        gap_lines.append("- 未能下载的代码：" + "; ".join(failed))
    else:
        gap_lines.append("- 预注册的 ETF 与股票代码全部下载成功。")
    gap_lines.append(
        "- 回测收益使用对齐后的交易日历。停牌或缺口最多沿用 5 个交易日的价格；更长的空洞不连接前后两段，因此文件里相邻行的跳变不会被记成一天的收益。"
        f"实际进入回测、且绝对值较大的日收益次数：ETF 超过 18% 的 {payload['extreme_etf_moves_gt_22pct']} 次（含 2024 年 9–10 月创业板涨停缺口），"
        f"股票超过 11% 的 {payload['extreme_stock_moves_gt_22pct']} 次（均为创业板 20% 涨跌停，已保留）。超过涨跌停上限的打印被剔除，不用估计值填上。"
    )
    gap_lines.append(
        f"- 实际入选行业 ETF {len(sector_codes)} 只，防御 ETF {len(defensive_codes)} 只，股票 {len(stock_frames)} 只。"
    )
    gap_lines.append("- QDII（例如纳指 ETF）未纳入，因为溢价和申购限额会让收盘价成交假设失效。")
    gap_lines.append(
        "- 价格源：ETF 为新浪未复权（`fund_etf_hist_sina`），股票为新浪前复权（`stock_zh_a_daily`）。"
        "腾讯前复权与新浪日收益相关性不足，且出现过单日数倍的不可能跳变，本次不使用。"
    )
    gap_lines.append(
        "- ETF 现金分红未计入。银行、红利、煤炭等高股息品种的真实持有收益高于表中价格收益；"
        "这会让防御类策略看起来更差，而不是更好。没有用估计分红填补。"
    )
    gap_lines.append(
        f"- 股票池等权 EW_Stocks 全样本 CAGR {fmt_pct(stats['EW_Stocks']['full']['cagr'])}，"
        f"夏普 {fmt_num(stats['EW_Stocks']['full']['sharpe'])}，"
        f"最大回撤 {fmt_pct(stats['EW_Stocks']['full']['max_dd'])}。"
        "个股策略必须对照这一行，而不是只对照沪深300。"
    )

    # Executive summary written from numbers
    bh_f = stats["BH_CSI300"]["full"]
    top3 = ranked[:3]
    exec_bits = [
        f"最后完整交易日 {last_date.date()}，规则化市场状态为 **{regime}**（行业 ETF 在 MA60 之上的比例 {above60:.0%}，沪深300ETF 近 120 日 {fmt_pct(csi_ret120)}）。",
        f"基准 BH_CSI300 在 2020-01-02 至 {last_date.date()} 扣费后 CAGR {fmt_pct(bh_f['cagr'])}，夏普 {fmt_num(bh_f['sharpe'])}，最大回撤 {fmt_pct(bh_f['max_dd'])}。",
        f"严格质量门槛下合格策略数量：**{len(survivors)}**。"
        + ("名单：" + "、".join(survivors) + "。" if survivors else "没有任何候选同时满足扣费超额、夏普>1、回撤<30%、样本外夏普>0.5。"),
        "观察顺序前三（含未过门槛者，只表示相对排序）："
        + "；".join(
            f"{n} 夏普 {fmt_num(stats[n]['full']['sharpe'])} / 回撤 {fmt_pct(stats[n]['full']['max_dd'])}"
            for n in top3
        )
        + "。",
        "本次不把历史高 beta 行业描述成可重复 alpha。",
        (
            f"股票池等权 EW_Stocks 是对照基准，不是策略：CAGR {fmt_pct(stats['EW_Stocks']['full']['cagr'])}，"
            f"夏普 {fmt_num(stats['EW_Stocks']['full']['sharpe'])}，回撤 {fmt_pct(stats['EW_Stocks']['full']['max_dd'])}。"
            f"Stock_CS_Mom 相对这组等权的超额 CAGR 只有 {fmt_pct(stats['Stock_CS_Mom']['excess_vs_ew_cagr'])}，"
            "它相对沪深300的超额来自股票池构成（含幸存者偏差），不来自动量规则。"
        ),
    ]

    ctx = {
        "last_date": str(last_date.date()),
        "eval_start": str(EVAL_START.date()),
        "exec_summary": "\n\n".join(exec_bits),
        "market_section": market_section,
        "full_table": full_table,
        "split_table": split_table,
        "stress_table": stress_table,
        "excess_table": excess_table,
        "sensitivity_table": sensitivity_table,
        "overlay_table": overlay_table,
        "ranking_section": "\n".join(qual_lines + [""] + rank_lines),
        "recommend_section": "\n".join(rec_bits),
        "deploy_section": deploy,
        "risk_section": risk,
        "self_check": self_check,
        "data_section": "\n".join(gap_lines),
        "chart_note": chart_note,
    }
    report = build_report(ctx)
    report_path = ROOT / "reports" / "trading_strategy_research_20261014.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("wrote", report_path)
    print("survivors", survivors)
    print("ranked", ranked)
    for n in order:
        f = stats[n]["full"]
        print(
            f"{n:20} sharpe={f['sharpe']:.2f} cagr={f['cagr']:.1%} mdd={f['max_dd']:.1%} "
            f"oos={stats[n]['oos']['sharpe']:.2f} ex={stats[n]['excess_cagr']:.1%}"
        )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
