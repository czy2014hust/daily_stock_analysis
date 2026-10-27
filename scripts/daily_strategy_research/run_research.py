#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily A-share strategy research.

Usage: RESEARCH_DATE=2026-10-27 LAST_BAR=2026-10-26 python3 run_research.py

Pre-committed rules (chosen before looking at this run's results):

- Evaluation window: 2020-01-02 .. latest completed daily bar (LAST_BAR).
- History fetch starts 2018-01-01 so 120-day signals exist on the first eval day.
- Long-only, no leverage. Uninvested weight is cash at 0% (money-market yield ignored).
- Signal uses the close of day t. The position earns the close-to-close return of t+1.
- Costs are charged on turnover at the rebalance close and deducted from the next day's return.
- ETF one-way cost: 8 bps (commission + slippage; onshore ETFs have no stamp duty).
- Stock one-way cost: 15 bps (commission + sell-side stamp amortization + slippage).
- ETF returns add back Sina cash dividends on the ex-date (yield = dividend / prior raw close).
  ETFs without dividend records stay price-only; that is a documented gap, not a fill.
- Sector low-vol / regime: month-end, top 3, 60-day vol / 120-day lookback where applicable.
- Risk parity: month-end inverse 60-day vol across CSI300 / 10Y treasury / gold ETFs.
- Stock momentum + low vol: month-end, average rank of 120d momentum and 60d vol, top 5 stocks.
- Dividend yield (carry): month-end, the 5 stocks with the highest trailing 365-day cash
  dividend per share (Sina, counted from the ex-date) / Eastmoney raw close; a payment
  330+ days older than the newest one in the window is last year's slot and is dropped. Dividends paid
  before a later bonus/transfer issue inside the window are scaled to post-split shares.
- Residual momentum: month-end, regress each stock's daily return on the CSI300 ETF over
  sessions t-240..t-21, rank by sum(residual) / std(residual); top 5.
- Margin flow timing: total financing balance (融资余额) = SSE + SZSE exchange totals. The print for
  session d is published after the close of d, so the signal at the close of t uses the balance of
  t-1. Month-end: if the balance grew over the 20 sessions ending t-1, hold 510300; otherwise hold
  511260. Benchmark: EW_SB (510300 / 511260 50/50, month-end rebalance).
- Term-spread timing: ChinaBond 10Y minus 2Y yield (Eastmoney `bond_zh_us_rate`), lagged one
  session. Month-end: if the spread is above the median of its previous 250 sessions (at least
  120), hold 510300; otherwise hold 511260. Benchmark: EW_SB.
- Stock margin crowding (SSE-listed names only; the SZSE per-stock endpoint is unreachable from the
  research host): month-end, score = per-stock financing balance on the session before the
  month-end divided by the same field three month-ends earlier, minus 1. Equal-weight the 5 SSE
  stocks with the smallest increase. Benchmark: EW_SSE (equal weight of the same SSE names).
- Holder concentration: month-end, for each stock take the latest shareholder-count record whose
  announcement date is <= t (cut-off within the last 200 days), score = its % change in holder
  count vs the previous record; equal-weight the 5 stocks with the largest decrease. Benchmark:
  EW_Stocks.
- No parameter is selected by maximizing in-sample Sharpe. Sensitivity is diagnostic only.

This is research code, not an order-routing system and not investment advice.
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# AkShare calls have no timeout of their own; a stalled socket would hang the run instead of retrying.
socket.setdefaulttimeout(60)
# requests passes timeout=None explicitly, which overrides the socket default (seen hanging in a TLS handshake).
try:
    import requests as _requests

    _orig_request = _requests.Session.request

    def _request_with_timeout(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 60
        return _orig_request(self, method, url, **kwargs)

    _requests.Session.request = _request_with_timeout  # type: ignore[assignment]
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

ASOF_REQUESTED = os.environ.get("RESEARCH_DATE", "2026-10-27")
LAST_BAR = pd.Timestamp(os.environ.get("LAST_BAR", "2026-10-26"))
FETCH_START = "20180101"
FETCH_END = LAST_BAR.strftime("%Y%m%d")
REPORT_NAME = f"trading_strategy_research_{ASOF_REQUESTED.replace('-', '')}.md"
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
RISK_PARITY_ETFS = ["510300", "511260", "518880"]

EXTRA_ETFS: Dict[str, str] = {
    "510300": "沪深300ETF",
    "518880": "黄金ETF",
    "511260": "十年国债ETF",
    "510050": "上证50ETF",
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

# Margin-crowding rule universe: SSE-listed names (per-stock SZSE margin data unavailable here).
SSE_STOCKS: List[str] = [c for c in STOCKS if c.startswith("6")]

# Interpretation written after reading this run's tables. It explains results; it never
# feeds back into rules, parameters or the pass/fail checks.
ELIMINATED: List[str] = []
INTERP_NOTES: Dict[str, str] = {}
EXEC_INTERP: List[str] = []
RISK_INTERP: List[str] = []
SELF_CHECK_INTERP: List[str] = []
OVERLAP_NOTE = ""
FUNDAMENTAL_NOTES: List[str] = [
    "- 股息率实现沿用 10-21 期的同档去重修正（窗口内比最新一笔早 330 天以上的分红剔除）。",
    "- 东方财富 PB 按报告期回填（10-23 期发现，净资产在季末日跳变）；本期没有任何规则使用 PB 或 PE。",
    "- 股东户数时点抽查（2019-06 以来 953 条记录）：公告日减截止日的中位数 29 天，10%/90% 分位 6 / 88 天；公告日早于或等于截止日的 4 条全部是中国海油、中国移动上市初期的占位记录（户数 1 或上期 0），"
    "其中中国移动 2022-01-05 记录的增减比例为 4275 万%，只会让它排在最后，且当时不满足 120 日历史门槛，没有进入任何持仓；没有删改记录。",
    "- 股东户数缺口：东方财富接口对紫金矿业(601899) 返回空数据（重试后仍为空），Stock_HolderConc 实际在 23 只里选股，紫金矿业永远不会入选；"
    "EW_Stocks 基准仍包含紫金矿业，紫金矿业 2023、2025 年明显强于池子（新浪前复权 +21%、+132%，池子均值 +6%、+28%），2024 年弱于池子（+23% vs +32%），因此对该规则 2023–2025 年的相对比较略偏不利；没有用其他来源补数。",
]
REPRO_NOTE = (
    "- 先用上一版（10-26）原始脚本、LAST_BAR=2026-10-23 在独立目录重跑（独立缓存，重新下载全部行情与基本面）：16 个序列的 Sharpe、CAGR、最大回撤、样本外 Sharpe 与上一版报告逐一相同。"
)
REPRO_SELF_CHECK = (
    "- 复现性：上一版（10-26）脚本在 LAST_BAR=2026-10-23 上重跑，16/16 个序列与上一版报告一致；本期数据只多了 2026-10-26 一根日线。"
)

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
    if cached is not None and not cached.empty and cached["date"].max() >= LAST_BAR:
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
            if df["date"].max() < LAST_BAR or len(df) < 100:
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
            if df["date"].max() < LAST_BAR or len(df) < 60:
                raise RuntimeError(f"{code} etf history incomplete: n={len(df)}")
            _save_cache(key, df)
            time.sleep(0.15)
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"fetch etf {code} failed: {last_err}")


def fetch_etf_dividends(code: str) -> pd.Series:
    """Per-share cash dividend indexed by ex-date, from Sina cumulative payouts.

    Empty series means Sina has no record; the ETF then stays price-only.
    """
    key = f"{code}_div"
    path = _cache_path(key)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        import akshare as ak

        raw = ak.fund_etf_dividend_sina(symbol=_prefix(code))
        df = pd.DataFrame({"date": pd.to_datetime(raw["日期"]) if len(raw) else [], "cum": raw["累计分红"] if len(raw) else []})
        df.to_csv(path, index=False)
        time.sleep(0.15)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    per = pd.to_numeric(df["cum"], errors="coerce").diff()
    per.iloc[0] = pd.to_numeric(df["cum"], errors="coerce").iloc[0]
    out = pd.Series(per.values, index=pd.DatetimeIndex(df["date"]))
    return out[out > 0]


def dividend_yield_panel(
    frames: Dict[str, pd.DataFrame],
    divs: Dict[str, pd.Series],
    index: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, List[str]]:
    """Dividend / prior raw close on the first trading day on or after each ex-date.

    Uses the unrepaired raw close so the yield is unit-consistent even when a later
    share split was scaled out of the price panel. Yields above 15% are rejected and logged.
    """
    out = pd.DataFrame(0.0, index=index, columns=list(frames))
    notes: List[str] = []
    for code, ser in divs.items():
        if ser.empty or code not in frames:
            continue
        raw = frames[code].set_index("date")["close"].sort_index()
        for ex, amt in ser.items():
            pos = raw.index.searchsorted(ex)
            if pos <= 0 or pos >= len(raw):
                continue
            dt = raw.index[pos]
            y = float(amt) / float(raw.iloc[pos - 1])
            if not (0 < y < 0.15):
                notes.append(f"{code} {ex.date()} 分红收益率 {y:.1%} 异常，未计入")
                continue
            if dt in out.index:
                out.at[dt, code] += y
    return out, notes


def fetch_shibor_3m() -> pd.Series:
    """3M Shibor (% p.a.) by report date. Used only as the risk-free rate in cash-excess Sharpe."""
    path = _cache_path("shibor_3m")
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        import akshare as ak

        raw = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="3月")
        df = pd.DataFrame({"date": pd.to_datetime(raw["报告日"]), "rate": pd.to_numeric(raw["利率"], errors="coerce")})
        df.to_csv(path, index=False)
    df = df.dropna().sort_values("date").drop_duplicates("date")
    return df.set_index("date")["rate"]


def fetch_index(symbol: str) -> pd.DataFrame:
    cached = _load_cache(symbol)
    if cached is not None and cached["date"].max() >= LAST_BAR:
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


def fetch_stock_valuation(code: str, retries: int = 3) -> pd.DataFrame:
    """Eastmoney daily valuation: raw close, total market cap, PE(TTM), PB. Missing values stay NaN."""
    key = f"{code}_value_em_v2"
    cached = _cache_ok(key)
    if cached is not None:
        return cached
    import akshare as ak

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            raw = ak.stock_value_em(symbol=code)
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw["数据日期"]),
                    "raw_close": pd.to_numeric(raw["当日收盘价"], errors="coerce"),
                    "total_mv": pd.to_numeric(raw["总市值"], errors="coerce"),
                    "pe_ttm": pd.to_numeric(raw["PE(TTM)"], errors="coerce"),
                    "pb": pd.to_numeric(raw["市净率"], errors="coerce"),
                }
            ).sort_values("date").drop_duplicates("date")
            if df["date"].max() < LAST_BAR:
                raise RuntimeError(f"{code} valuation stops at {df['date'].max().date()}")
            _save_cache(key, df)
            time.sleep(0.2)
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"fetch valuation {code} failed: {last_err}")


def fetch_stock_dividends(code: str) -> pd.DataFrame:
    """Sina dividend history: ex-date, cash per share, bonus+transfer shares per share. Implemented rows only."""
    path = _cache_path(f"{code}_stkdiv")
    if path.exists():
        df = pd.read_csv(path, parse_dates=["ex_date", "announce"])
    else:
        import akshare as ak

        raw = None
        for attempt in range(3):
            try:
                raw = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raw = raw[raw["进度"].astype(str).str.contains("实施")]
        df = pd.DataFrame(
            {
                "announce": pd.to_datetime(raw["公告日期"], errors="coerce"),
                "ex_date": pd.to_datetime(raw["除权除息日"], errors="coerce"),
                "cash": pd.to_numeric(raw["派息"], errors="coerce").fillna(0.0) / 10.0,
                "split": (pd.to_numeric(raw["送股"], errors="coerce").fillna(0.0)
                          + pd.to_numeric(raw["转增"], errors="coerce").fillna(0.0)) / 10.0,
            }
        )
        df.to_csv(path, index=False)
        time.sleep(0.2)
    return df.dropna(subset=["ex_date"]).sort_values("ex_date").reset_index(drop=True)


def fetch_market_margin() -> pd.DataFrame:
    """Exchange-level financing balance (融资余额, CNY) by session: columns sh, sz, total.

    total is NaN on a session where either exchange print is missing; nothing is interpolated.
    """
    key = "margin_market"
    cached = _load_cache(key)
    if cached is None or cached["date"].max() < LAST_BAR:
        import akshare as ak

        parts = {}
        for name, fn in (("sh", ak.macro_china_market_margin_sh), ("sz", ak.macro_china_market_margin_sz)):
            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    raw = fn()
                    parts[name] = pd.Series(
                        pd.to_numeric(raw["融资余额"], errors="coerce").values,
                        index=pd.to_datetime(raw["日期"]),
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(1.5 * (attempt + 1))
            if name not in parts:
                raise RuntimeError(f"market margin {name} failed: {last_err}")
        df = pd.DataFrame(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["total"] = df["sh"] + df["sz"]
        cached = df.reset_index().rename(columns={"index": "date"})
        _save_cache(key, cached)
    df = cached.set_index("date").sort_index()
    return df.loc[df.index <= LAST_BAR]


def fetch_cn_curve() -> pd.DataFrame:
    """ChinaBond 2Y / 10Y yields (%) by date from Eastmoney `bond_zh_us_rate`."""
    key = "cn_curve"
    cached = _load_cache(key)
    if cached is None or cached["date"].max() < LAST_BAR - pd.Timedelta(days=4):
        import akshare as ak

        raw = ak.bond_zh_us_rate(start_date="20170101")
        cached = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["日期"]),
                "y2": pd.to_numeric(raw["中国国债收益率2年"], errors="coerce"),
                "y10": pd.to_numeric(raw["中国国债收益率10年"], errors="coerce"),
            }
        ).sort_values("date").drop_duplicates("date")
        _save_cache(key, cached)
    df = cached.set_index("date").sort_index()
    return df.loc[df.index <= LAST_BAR]


def fetch_stock_margin_sse(dates: List[pd.Timestamp], codes: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Per-stock SSE financing balance on the requested sessions (rows = dates, columns = codes).

    A date that cannot be downloaded stays an all-NaN row and is reported as a gap; a code absent
    from that day's file (not yet margin-eligible) stays NaN.
    """
    import akshare as ak

    rows: Dict[pd.Timestamp, pd.Series] = {}
    gaps: List[str] = []
    for dt in dates:
        path = _cache_path(f"margin_sse_{dt.strftime('%Y%m%d')}")
        if path.exists():
            df = pd.read_csv(path, dtype={"code": str})
        else:
            df = None
            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    raw = ak.stock_margin_detail_sse(date=dt.strftime("%Y%m%d"))
                    df = pd.DataFrame(
                        {
                            "code": raw["标的证券代码"].astype(str).str.zfill(6),
                            "fin_balance": pd.to_numeric(raw["融资余额"], errors="coerce"),
                        }
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(1.5 * (attempt + 1))
            if df is None:
                gaps.append(f"{dt.date()}: {type(last_err).__name__}")
                rows[dt] = pd.Series(np.nan, index=codes)
                continue
            df = df[df["code"].isin(codes)]
            df.to_csv(path, index=False)
            time.sleep(0.3)
        rows[dt] = df.set_index("code")["fin_balance"].reindex(codes)
    out = pd.DataFrame(rows).T.sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out, gaps


def fetch_holder_counts(code: str, retries: int = 3) -> pd.DataFrame:
    """Eastmoney shareholder-count history: cut-off date, announcement date, holders, % change,
    interval price change. Rows without an announcement date are unusable for timing and dropped."""
    path = _cache_path(f"{code}_gdhs")
    if path.exists():
        df = pd.read_csv(path, parse_dates=["cutoff", "announce"])
    else:
        import akshare as ak

        last_err: Optional[Exception] = None
        df = None
        for attempt in range(retries):
            try:
                raw = ak.stock_zh_a_gdhs_detail_em(symbol=code)
                df = pd.DataFrame(
                    {
                        "cutoff": pd.to_datetime(raw["股东户数统计截止日"], errors="coerce"),
                        "announce": pd.to_datetime(raw["股东户数公告日期"], errors="coerce"),
                        "holders": pd.to_numeric(raw["股东户数-本次"], errors="coerce"),
                        "holders_prev": pd.to_numeric(raw["股东户数-上次"], errors="coerce"),
                        "chg_pct": pd.to_numeric(raw["股东户数-增减比例"], errors="coerce"),
                        "interval_ret": pd.to_numeric(raw["区间涨跌幅"], errors="coerce") / 100.0,
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.2 * (attempt + 1))
        if df is None:
            raise RuntimeError(f"fetch holder counts {code} failed: {last_err}")
        df.to_csv(path, index=False)
        time.sleep(0.2)
    return df.dropna(subset=["cutoff", "announce"]).sort_values("cutoff").reset_index(drop=True)


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


def _last_period_open(index: pd.DatetimeIndex, freq: str) -> bool:
    """True when the panel stops before the final weekday of its last month / week.

    The panel's last row is then just "today", not a period end; treating it as one would
    rebalance (or time a turn-of-month entry) on information the schedule does not have.
    A holiday on the final weekday makes a complete period look open; that only delays a signal.
    """
    if len(index) == 0:
        return False
    last = index[-1]
    if freq == "M":
        period_end = last + pd.offsets.BMonthEnd(0)
    else:
        period_end = last + pd.offsets.Week(weekday=4) if last.weekday() != 4 else last
    return last < period_end


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    last = pd.DatetimeIndex(s.groupby([index.year, index.month]).tail(1).values)
    return last[:-1] if _last_period_open(index, "M") else last


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    iso = index.isocalendar()
    s = pd.Series(index, index=index)
    key = iso.year.astype(str) + "-" + iso.week.astype(str)
    last = pd.DatetimeIndex(s.groupby(key).tail(1).values)
    return last[:-1] if _last_period_open(index, "W") else last


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


def strategy_stock_mom_lowvol(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    mom_window: int = 120,
    vol_window: int = 60,
    n_hold: int = 5,
) -> pd.DataFrame:
    """Month-end composite rank: average of 120d momentum rank (high is good) and
    60d volatility rank (low is good); equal-weight the best n_hold names."""
    rets = close.pct_change()
    mom = close.pct_change(mom_window)
    vol = rets.rolling(vol_window).std()
    ok = eligible_mask(close, amount, STOCK_MIN_AMOUNT, max(mom_window, vol_window))
    raw = _empty_weights(close.index, list(close.columns))
    for dt in month_ends(close.index):
        elig = ok.loc[dt].reindex(close.columns).fillna(False)
        m = mom.loc[dt][elig].dropna()
        v = vol.loc[dt][elig].dropna()
        names = m.index.intersection(v.index)
        row = pd.Series(0.0, index=close.columns)
        if len(names) >= n_hold:
            score = m[names].rank(ascending=False) + v[names].rank(ascending=True)
            picks = score.nsmallest(n_hold).index
            row.loc[picks] = 1.0 / n_hold
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_risk_parity(close: pd.DataFrame, vol_window: int = 60) -> pd.DataFrame:
    """Month-end inverse-volatility weights across equity / duration / gold ETFs.

    Thesis: the three sleeves carry different risk premia with low long-run correlation,
    so equalizing risk contribution lowers drawdown more than it lowers return.
    Names without vol_window sessions of history are left out until they have it.
    """
    vol = close.pct_change().rolling(vol_window).std()
    raw = _empty_weights(close.index, list(close.columns))
    for dt in month_ends(close.index):
        v = vol.loc[dt].dropna()
        v = v[v > 0]
        row = pd.Series(0.0, index=close.columns)
        if len(v) >= 2:
            inv = 1.0 / v
            row.loc[inv.index] = inv / inv.sum()
        raw.loc[dt] = row
    return _materialize(raw)


def strategy_stock_low_vol(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    vol_window: int = 60,
    n_hold: int = 5,
) -> pd.DataFrame:
    """Month-end, equal-weight the n_hold lowest-volatility stocks of the pool."""
    vol = close.pct_change().rolling(vol_window).std()
    ok = eligible_mask(close, amount, STOCK_MIN_AMOUNT, vol_window)
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


def strategy_stock_rank_low(
    score: pd.DataFrame,
    close: pd.DataFrame,
    amount: pd.DataFrame,
    history: int,
    n_hold: int = 5,
) -> pd.DataFrame:
    """Month-end: equal-weight the n_hold eligible stocks with the lowest `score`.

    `score` must only use information up to and including the row's close.
    """
    ok = eligible_mask(close, amount, STOCK_MIN_AMOUNT, history)
    raw = _empty_weights(close.index, list(close.columns))
    for dt in month_ends(close.index):
        snap = score.loc[dt] if dt in score.index else pd.Series(dtype=float)
        elig = ok.loc[dt].reindex(snap.index).fillna(False)
        pool = snap[elig].dropna()
        row = pd.Series(0.0, index=close.columns)
        if len(pool) >= n_hold:
            row.loc[pool.nsmallest(n_hold).index] = 1.0 / n_hold
        raw.loc[dt] = row
    return _materialize(raw)


def trailing_dividend_yield(
    divs: Dict[str, pd.DataFrame],
    raw_close: pd.DataFrame,
    window_days: int = 365,
    same_slot_days: Optional[int] = 330,
) -> pd.DataFrame:
    """Month-end trailing cash dividend per current share / raw close.

    An annual payer whose ex-date moves a few days earlier year over year (ICBC 2019-07-03
    then 2020-06-30) has two annual payments inside 365 days. A payment is therefore dropped
    when a later payment in the window falls at least `same_slot_days` after it (last year's
    version of the same slot). Semi-annual payments ~6 months apart are both kept.
    same_slot_days=None reproduces the plain 365-day sum.

    A dividend counts from its ex-date (the announcement is earlier, so this is conservative).
    Cash paid on or before a bonus/transfer ex-date inside the window is divided by
    (1 + shares per share) so numerator and price refer to the same share count.
    Names whose dividend history failed to download stay NaN; a name with no payout scores 0.
    """
    me = month_ends(raw_close.index)
    out = pd.DataFrame(np.nan, index=raw_close.index, columns=raw_close.columns)
    for code in raw_close.columns:
        if code not in divs:
            continue
        ev = divs[code]
        for dt in me:
            px = raw_close.at[dt, code]
            if pd.isna(px) or px <= 0:
                continue
            win = ev[(ev["ex_date"] > dt - pd.Timedelta(days=window_days)) & (ev["ex_date"] <= dt)]
            paid = win[win["cash"] > 0]
            if same_slot_days is not None and len(paid) > 1:
                newest = paid["ex_date"].max()
                paid = paid[(newest - paid["ex_date"]).dt.days < same_slot_days]
            total = 0.0
            for _, e in paid.iterrows():
                later = ev[(ev["ex_date"] >= e["ex_date"]) & (ev["ex_date"] <= dt) & (ev["split"] > 0)]
                total += float(e["cash"]) / float(np.prod(1.0 + later["split"].values)) if len(later) else float(e["cash"])
            out.at[dt, code] = total / float(px)
    return out


def residual_momentum_score(
    stock_rets: pd.DataFrame,
    mkt_rets: pd.Series,
    window: int = 240,
    skip: int = 20,
) -> pd.DataFrame:
    """Month-end residual momentum (Blitz-Huij-Martens style).

    OLS of each stock's daily return on the market over sessions [t-window+1, t-skip], then
    sum(residual) / std(residual). Skipping the last month avoids the short-term reversal.
    Requires 80% valid sessions in the regression window.
    """
    idx = stock_rets.index
    mkt = mkt_rets.reindex(idx)
    out = pd.DataFrame(np.nan, index=idx, columns=stock_rets.columns)
    need = int(0.8 * (window - skip))
    for dt in month_ends(idx):
        pos = idx.get_loc(dt)
        if pos < window:
            continue
        sl = slice(pos - window + 1, pos - skip + 1)
        x_all = mkt.iloc[sl]
        for code in stock_rets.columns:
            y = stock_rets[code].iloc[sl]
            ok = y.notna() & x_all.notna()
            if int(ok.sum()) < need:
                continue
            yy, xx = y[ok].values, x_all[ok].values
            var = float(np.var(xx, ddof=1))
            if var <= 0:
                continue
            beta = float(np.cov(yy, xx, ddof=1)[0, 1]) / var
            alpha = float(yy.mean() - beta * xx.mean())
            resid = yy - alpha - beta * xx
            sd = float(resid.std(ddof=1))
            if sd > 0:
                out.at[dt, code] = float(resid.sum()) / sd
    return out


def strategy_sign_switch(score: pd.Series, risk_on: str, risk_off: str, columns: List[str],
                         threshold: float = 0.0, freq: str = "M") -> pd.DataFrame:
    """Period-end: hold `risk_on` when score > threshold, else `risk_off` (also when score is NaN)."""
    idx = score.index
    dates = month_ends(idx) if freq == "M" else week_ends(idx)
    raw = _empty_weights(idx, columns)
    for dt in dates:
        row = pd.Series(0.0, index=columns)
        sc = score.get(dt, np.nan)
        row[risk_on if (pd.notna(sc) and sc > threshold) else risk_off] = 1.0
        raw.loc[dt] = row
    return _materialize(raw)


def margin_change_panel(bal_prev: pd.DataFrame, index: pd.DatetimeIndex, lag_months: int = 3) -> pd.DataFrame:
    """Month-end score = financing balance (session before the month-end) / same field lag_months
    month-ends earlier - 1. Non-positive balances (not yet eligible) give NaN."""
    b = bal_prev.where(bal_prev > 0)
    chg = b / b.shift(lag_months) - 1.0
    return chg.reindex(index)


def strategy_state_switch(score: pd.Series, risk_on: str, risk_off: str, columns: List[str],
                          med_window: int = 250, min_obs: int = 120, freq: str = "M") -> pd.DataFrame:
    """Period-end: hold `risk_on` when score > median of its previous med_window sessions, else `risk_off`.

    The median excludes the current session, so the threshold is known before the comparison.
    Without enough history the strategy holds risk_off.
    """
    idx = score.index
    med = score.shift(1).rolling(med_window, min_periods=min_obs).median()
    dates = month_ends(idx) if freq == "M" else week_ends(idx)
    raw = _empty_weights(idx, columns)
    for dt in dates:
        row = pd.Series(0.0, index=columns)
        sc, md = score.get(dt, np.nan), med.get(dt, np.nan)
        on = bool(pd.notna(sc) and pd.notna(md) and sc > md)
        row[risk_on if on else risk_off] = 1.0
        raw.loc[dt] = row
    return _materialize(raw)


def holder_change_panel(holders: Dict[str, pd.DataFrame], index: pd.DatetimeIndex,
                        max_age_days: int = 200, periods: int = 1, field: str = "chg") -> pd.DataFrame:
    """Month-end score from shareholder-count disclosures known at t.

    Only records with announce <= t are visible. Among them the latest cut-off date is used, if it is
    within max_age_days of t. periods=1: the record's own % change vs the previous record.
    periods=2: holder count now vs two visible records earlier. field="ret": the record's interval
    price change (used to residualize the holder change against price moves).
    """
    out = pd.DataFrame(np.nan, index=index, columns=list(holders))
    me = month_ends(index)
    for code, df in holders.items():
        d = df.dropna(subset=["announce", "cutoff"]).sort_values(["cutoff", "announce"])
        for dt in me:
            vis = d[d["announce"] <= dt].drop_duplicates("cutoff", keep="last")
            if vis.empty:
                continue
            last = vis.iloc[-1]
            if (dt - last["cutoff"]).days > max_age_days:
                continue
            if field == "ret":
                out.at[dt, code] = float(last["interval_ret"]) if pd.notna(last["interval_ret"]) else np.nan
            elif periods == 1:
                out.at[dt, code] = float(last["chg_pct"]) / 100.0 if pd.notna(last["chg_pct"]) else np.nan
            else:
                if len(vis) <= periods - 1:
                    continue
                prev = vis.iloc[-periods]
                base = float(prev["holders_prev"])
                if base > 0:
                    out.at[dt, code] = float(last["holders"]) / base - 1.0
    return out


def residualize(score: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional OLS residual of score on control, row by row (rows with < 6 pairs stay NaN)."""
    out = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for dt in score.index:
        y, x = score.loc[dt], control.loc[dt].reindex(score.columns)
        ok = y.notna() & x.notna()
        if int(ok.sum()) < 6:
            continue
        xv, yv = x[ok].values, y[ok].values
        var = float(np.var(xv, ddof=1))
        if var <= 0:
            continue
        b = float(np.cov(yv, xv, ddof=1)[0, 1]) / var
        a = float(yv.mean() - b * xv.mean())
        out.loc[dt, ok[ok].index] = yv - a - b * xv
    return out


def static_mix(index: pd.DatetimeIndex, mix: Dict[str, float]) -> pd.DataFrame:
    """Month-end rebalance to fixed weights."""
    me = index.isin(month_ends(index))
    return _materialize(pd.DataFrame({c: np.where(me, w, np.nan) for c, w in mix.items()}, index=index))


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


def cash_excess_sharpe(net: pd.Series, rf_daily: Optional[pd.Series]) -> float:
    """Annualized Sharpe of (strategy - risk-free). Uninvested cash still earns 0 in the backtest,
    so this penalizes strategies that sit in cash; that is the conservative direction."""
    if rf_daily is None:
        return float("nan")
    r = net.dropna()
    ex = r - rf_daily.reindex(r.index).fillna(0.0)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


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

    # Month-end rank timing: B's score drops mid-March. The switch may only appear at the
    # March month-end close and only earn from the first April session.
    cal = pd.bdate_range("2019-01-01", "2020-04-10")
    px = pd.DataFrame({"A": 100.0, "B": 100.0}, index=cal)
    score = pd.DataFrame({"A": 1.0, "B": 2.0}, index=cal)
    score.loc["2020-03-16":, "B"] = 0.0
    sel = strategy_stock_rank_low(score, px, None, 60, 1)
    assert sel.loc["2020-03-30", "B"] == 0.0 and sel.loc["2020-03-31", "B"] == 1.0
    assert sel["B"].shift(1).loc["2020-03-31"] == 0.0 and sel["B"].shift(1).loc["2020-04-01"] == 1.0

    # Dividend yield: 1.0 cash, then a 10-for-10 transfer => 0.5 per current share; counts from ex-date only.
    dcal = pd.bdate_range("2020-01-01", "2020-06-30")
    rc = pd.DataFrame({"A": 10.0}, index=dcal)
    ev = {"A": pd.DataFrame({"ex_date": pd.to_datetime(["2020-03-16", "2020-04-15"]), "cash": [1.0, 0.0], "split": [0.0, 1.0]})}
    dy = trailing_dividend_yield(ev, rc)["A"]
    assert dy.loc["2020-02-28"] == 0.0 and abs(dy.loc["2020-03-31"] - 0.10) < 1e-12 and abs(dy.loc["2020-04-30"] - 0.05) < 1e-12
    # Annual ex-date drifting earlier (ICBC 2019-07-03 -> 2020-06-30) must not count two years.
    rc2 = pd.DataFrame({"A": 5.0}, index=pd.bdate_range("2019-06-01", "2020-07-31"))
    ev2 = {"A": pd.DataFrame({"ex_date": pd.to_datetime(["2019-07-03", "2020-06-30"]), "cash": [0.25, 0.26], "split": [0.0, 0.0]})}
    assert abs(trailing_dividend_yield(ev2, rc2)["A"].loc["2020-06-30"] - 0.052) < 1e-12
    assert abs(trailing_dividend_yield(ev2, rc2, 365, None)["A"].loc["2020-06-30"] - 0.102) < 1e-12

    # Sign switch: month-end only; NaN or a score at the threshold holds the risk-off asset.
    scal = pd.bdate_range("2020-01-01", "2020-03-31")
    ssc = pd.Series(np.nan, index=scal)
    ssc.loc["2020-02-28"] = 0.02
    ssc.loc["2020-03-31"] = 0.0
    ssw = strategy_sign_switch(ssc, "E", "B", ["E", "B"])
    assert ssw.loc["2020-01-31", "B"] == 1.0 and ssw.loc["2020-02-28", "E"] == 1.0 and ssw.loc["2020-03-02", "E"] == 1.0
    assert ssw.loc["2020-03-31", "B"] == 1.0
    # Margin change: ratio to the value lag_months month-ends earlier; non-positive balance is NaN.
    mes = pd.DatetimeIndex(pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]))
    bal = pd.DataFrame({"A": [100.0, 110.0, 120.0, 150.0], "B": [0.0, 50.0, 60.0, 75.0]}, index=mes)
    mc = margin_change_panel(bal, mes, 3)
    assert abs(mc.loc["2020-04-30", "A"] - 0.5) < 1e-12 and pd.isna(mc.loc["2020-04-30", "B"]) and pd.isna(mc.loc["2020-03-31", "A"])
    # State switch: the median excludes the current session; month-end only.
    scal2 = pd.bdate_range("2020-01-01", "2020-03-31")
    sc2 = pd.Series(1.0, index=scal2)
    sc2.loc["2020-02-28"] = 5.0
    sw = strategy_state_switch(sc2, "E", "B", ["E", "B"], 250, 20)
    assert sw.loc["2020-01-31", "B"] == 1.0 and sw.loc["2020-02-28", "E"] == 1.0 and sw.loc["2020-03-31", "B"] == 1.0
    # Holder counts: a record is invisible before its announcement date and stale after 200 days.
    hcal = pd.bdate_range("2020-01-01", "2020-12-31")
    hd = {"A": pd.DataFrame({"cutoff": pd.to_datetime(["2019-12-31", "2020-03-31"]),
                             "announce": pd.to_datetime(["2020-03-20", "2020-04-28"]),
                             "holders": [100.0, 90.0], "holders_prev": [80.0, 100.0],
                             "chg_pct": [25.0, -10.0], "interval_ret": [0.05, -0.02]})}
    hp = holder_change_panel(hd, hcal)["A"]
    assert pd.isna(hp.loc["2020-02-28"]) and abs(hp.loc["2020-03-31"] - 0.25) < 1e-12
    assert abs(hp.loc["2020-04-30"] + 0.10) < 1e-12 and pd.isna(hp.loc["2020-10-30"])
    hp2 = holder_change_panel(hd, hcal, periods=2)["A"]
    assert abs(hp2.loc["2020-04-30"] - (90.0 / 80.0 - 1.0)) < 1e-12

    assert list(month_ends(pd.bdate_range("2020-01-01", "2020-02-12")).strftime("%m-%d")) == ["01-31"]
    assert list(month_ends(pd.bdate_range("2020-01-01", "2020-02-28")).strftime("%m-%d")) == ["01-31", "02-28"]


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
    a(f"# 交易策略研究日报 {ASOF_REQUESTED}")
    a("")
    a(f"研究日期：{ASOF_REQUESTED}（Asia/Shanghai）。")
    if last_date == ASOF_REQUESTED:
        a(f"行情截止：**{last_date} 收盘**（脚本在 A 股收盘后运行，当日 K 线已完整）。")
    else:
        a(f"行情截止：**{last_date} 收盘**。回测不使用 {ASOF_REQUESTED} 未完成的 K 线。")
    a("市场：A 股可交易工具（场内 ETF 与高流动性个股）。多头、无杠杆。")
    a(f"复现：`RESEARCH_DATE={ASOF_REQUESTED} LAST_BAR={last_date} python3 scripts/daily_strategy_research/run_research.py`")
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
    a("相对 2026-10-26 版本的迭代（依据上一版证据，在跑本次数据之前确定）：")
    a("")
    if ctx["repro_note"]:
        a(ctx["repro_note"])
    a("- 丢弃 10-26 期淘汰的 2 个候选：US_LeadLag（美股信息已在开盘跳空定价，扣费后相对同口径持有 -8.6%/年、7 年全负）、"
      "VRP_Timing（机制 t=1.0，换已实现波动窗口或周度调仓后 Sharpe 掉到 0.00 / 0.16）。同时去掉标普500、50ETF QVIX 数据与开盘→开盘收益口径，以及 BH_CSI300_O2O 基准。")
    a("- 保留 7 个观察名单候选（规则与参数不变）：RiskParity_SBG、Stock_HolderConc、Stock_MomLowVol、Stock_ResMom、Stock_DivYield、Stock_LowVol、Sector_LowVol。")
    a("- 新增 3 个收益来源不同、此前各期都没测过的候选（10-03 至 10-26 期测过动量、反转、低波、股息、质量、价值、ERP、VRP、季节性、分红月、配对、趋势、跨市场领先-滞后、股东户数等，"
      "没有测过杠杆资金与收益率曲线）：融资余额择时 Margin_Timing（两市融资余额 20 日净增时持股、净减时持债，对照股债 50/50）、"
      "期限利差择时 TermSpread_Timing（10Y−2Y 国债利差高于 250 日中位数时持股，否则持债，对照股债 50/50）、"
      "个股融资拥挤 Stock_MarginCrowd（上交所 17 只里 3 个月融资余额增幅最小的 5 只，对照同 17 只等权 EW_SSE）。")
    a("")
    a("| ID | 经济逻辑 | 规格 | 成本 |")
    a("| --- | --- | --- | --- |")
    a("| BH_CSI300 | 宽基 beta，作为基准而不是 alpha | 全程持有 510300 沪深300ETF（含分红） | 单边 8bps，仅建仓 |")
    a("| EW_Stocks | 股票池本身的 beta，用来区分「池子选得偏」和「规则有超额」 | 24 只股票里已有足够历史的名字月末等权 | 15bps |")
    a("| EW_Sectors | 行业 ETF 池本身的 beta，行业策略的宇宙基准 | 15 只行业/风格 ETF 月末等权 | 8bps |")
    a("| Sector_LowVol | 低波动/红利类资产在 A 股有持有期溢价，且回撤通常浅于高 beta | 月末选 60 日波动最低的 3 只行业 ETF，等权 | 8bps |")
    a("| RiskParity_SBG | 股票、久期、黄金三类风险溢价长期低相关；按风险而非按金额分配可压低回撤 | 510300 / 511260 / 518880 月末按 60 日波动倒数配权 | 8bps |")
    a("| EW_SBG | 股票/国债/黄金三等分，检验「按波动倒数配权」是否比简单分散更好 | 510300 / 511260 / 518880 月末各 1/3 | 8bps |")
    a("| BH_Bond10Y | 久期 beta；作为 RiskParity_SBG 久期暴露的参照 | 全程持有 511260 十年国债ETF（含分红） | 8bps |")
    a("| Stock_LowVol | 低波动异象：机构有杠杆约束与彩票偏好，高 beta 被高估、低 beta 被低估 | 同一股票池，月末选 60 日波动最低的 5 只，等权 | 15bps |")
    a("| Stock_MomLowVol | 动量与低波动在 A 股相关性低、回撤期不同；复合排序可保留动量收益并压低动量崩盘 | 同一股票池，月末按「120 日收益排名 + 60 日波动排名（低者优）」之和选前 5，等权 | 15bps |")
    a("| Stock_DivYield | 股息 carry：现金分红是可兑现的回报；高股息大盘股受险资/红利指数配置需求支撑 | 同一股票池，月末按「过去 365 天每股现金分红（除息日计入、送转折算、同档去重）/ 原始收盘价」选最高的 5 只，等权 | 15bps |")
    a("| Stock_ResMom | 残差动量（Blitz-Huij-Martens）：剥离 beta 后的特质收益延续来自公司层面信息的缓慢扩散 | 同一股票池，月末对过去 240 日中的前 220 日做个股对沪深300ETF 的回归，按「残差之和 / 残差标准差」选最高的 5 只，等权 | 15bps |")
    a("| EW_SB | Margin_Timing / TermSpread_Timing 的同口径基准：同样只在股票与十年国债之间分配 | 510300 / 511260 月末各 50% | 8bps |")
    a("| EW_SSE | Stock_MarginCrowd 的宇宙基准：同一批上交所股票的等权，区分「上交所子池的 beta」与「规则超额」 | 24 只里 17 只上交所股票月末等权 | 15bps |")
    a("| Margin_Timing | 杠杆资金顺周期（A 股融资盘以散户为主，融资余额净增代表增量杠杆资金入场，资金流有持续性；2015 年与 2019、2024 年的上涨都伴随融资余额扩张）：融资余额持续净增时股票相对国债的预期收益更高 | 月末读取 t−1 日沪深两市融资余额合计（交易所在收盘后发布），若较 20 个交易日前增加则持 510300，否则持 511260 | 8bps |")
    a("| TermSpread_Timing | 期限利差的增长预期含义（Estrella-Mishkin；陡峭曲线 = 宽松货币 + 远期增长预期上修，平坦曲线 = 增长预期下修）：利差高时股票相对国债更有利 | 月末读取 t−1 日 10Y−2Y 中债国债收益率差，高于此前 250 个交易日中位数（至少 120 个）时持 510300，否则持 511260 | 8bps |")
    a("| Stock_MarginCrowd | 杠杆拥挤（融资盘以散户为主，融资余额快速上升的个股被追涨杠杆资金过度定价，杠杆约束收紧时被迫卖出）：融资增幅最小的个股后续相对更好 | 上交所 17 只股票，月末取月末前一交易日的个股融资余额，相对 3 个月末前同口径的增幅最小的 5 只，等权 | 15bps |")
    a("| Stock_HolderConc | 筹码集中（A 股文献的股东户数效应）：户数下降说明流通股从分散的散户流向少数持有人（机构、产业资本），后者信息更多；户数信息随定期报告与互动平台公开，扩散慢 | 同一股票池，月末取公告日 ≤ t 的最新一期股东户数（截止日在 200 天内），按「本期较上期户数变化比例」最小（降得最多）选 5 只，等权 | 15bps |")
    a("")
    a("成交与流动性约束：")
    a("")
    a("- 信号用 t 日收盘，收益从 t+1 日收盘到再下一根收盘才计入，避免用到信号当日涨跌。融资余额与国债收益率都再滞后 1 个交易日（t 日的融资余额在收盘后才发布）。")
    a("- ETF 使用新浪未复权成交价，并在除息日加回新浪披露的现金分红（分红 / 前一日原始收盘价）。没有分红记录的 ETF 不估计分红。")
    a("- 股票使用新浪前复权。腾讯前复权在中国神华等标的上出现过不可能的单日涨跌，已整段弃用。")
    a("- ETF 过去 20 日成交额中位数低于 2000 万人民币则当天不可入选；股票阈值为 5000 万。")
    a("- 停牌或单日缺口最多向前填充 5 个交易日；更长缺口不编造价格。")
    a("- 现金收益按 0 计（本期没有规则主动空仓，择时规则在股票与十年国债之间切换）。Sharpe-rf 扣除 3M Shibor。")
    a("- 基本面：个股原始收盘价来自东方财富 `stock_value_em`；现金分红与送转来自新浪 `stock_history_dividend_detail`，只用「实施」记录，股息率从除权除息日起计入；"
      "股东户数来自东方财富，按公告日生效；个股融资余额来自上交所 `stock_margin_detail_sse`，两市融资余额来自 AkShare `macro_china_market_margin_sh/sz`。")
    a("- 个股单边 15bps、ETF 单边 8bps 是保守零售成本（佣金、卖出印花税摊销、滑点），不是交易所最低佣金。另给出成本翻倍结果。")
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
    a("Sharpe / Sortino 使用日收益、无风险利率 0、按 252 天年化（与前几期报告口径一致）。Sharpe-rf 扣除 3M Shibor 后计算，质量门槛用 Sharpe-rf。")
    a(ctx["rf_note"])
    a("胜率同时给日频与月频；日频胜率接近 50% 并不否定正期望。")
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
    a("### 个股规则的因子重叠")
    a("")
    a("左侧为各规则「日收益 − EW_Stocks 日收益」之间的相关系数（全样本，扣费后）；最后一列是评估期内每个月末持仓与 Stock_LowVol 重合的平均比例。"
      "相关接近 1 或重合度高，说明新规则只是低波动换了个名字，不能算独立的收益来源。")
    a("")
    a(ctx["overlap_table"])
    a("")
    a(ctx["overlap_note"])
    a("")
    a("### 个股策略逐只剔除（leave-one-out）")
    a("")
    a("每次从 24 只股票池里去掉 1 只，规则、参数和宇宙基准都在同一个缩小后的池子上重跑。Stock_MarginCrowd 只逐只去掉 17 只上交所股票，对照同样缩小后的 EW_SSE。")
    a("")
    a(ctx["loo_table"])
    a("")
    a(ctx["loo_note"])
    a("")
    a("### 逐年超额")
    a("")
    a("每个日历年的策略收益减去其宇宙基准收益（扣费后）。用来区分「每年都有一点」和「只在一两个年份有一大段」。")
    a("")
    a(ctx["yearly_table"])
    a("")
    a("### 新候选机制拆分（预注册诊断）")
    a("")
    a("每个新规则都有一个可证伪的机制：如果超额存在、但机制表里看不到对应的条件差异，超额就不应归因于该机制。")
    a("")
    a(ctx["mech_table"])
    a("")
    a("### 10% 波动目标覆盖（只减仓、不加杠杆）")
    a("")
    a("规则预先固定：杠杆 = min(1, 10% / 过去 60 日已实现波动)，滞后 1 日。这是风控，不是另一套 alpha。")
    a("")
    a(ctx["overlay_table"])
    a("")
    a("### 成本翻倍压力（ETF 16bps / 股票 30bps 单边）")
    a("")
    a(ctx["cost_table"])
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
    a("2. 全样本 Sharpe-rf（扣 3M Shibor）> 1.0。")
    a("3. 全样本最大回撤浅于 30%。")
    a("4. 样本外夏普 > 0.5，且不是只在单一窗口有效。")
    a("5. 跑赢自身宇宙基准（行业 ETF 策略对 EW_Sectors，个股策略对 EW_Stocks，风险平价对 EW_SBG，Margin_Timing / TermSpread_Timing 对股债 50/50 EW_SB，Stock_MarginCrowd 对上交所等权 EW_SSE）。")
    a("6. 经济逻辑清楚，流动性与成本下可执行，不过度依赖单一参数或单一标的（第 6 条为定性判断，见敏感度与成本翻倍表）。")
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

    # Cached history may run past LAST_BAR; trimming keeps a dated rerun reproducible.
    for frames in (index_frames, etf_frames, stock_frames):
        for key in list(frames):
            frames[key] = frames[key].loc[frames[key]["date"] <= LAST_BAR].reset_index(drop=True)

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

    divs: Dict[str, pd.Series] = {}
    div_fail: List[str] = []
    for code in etf_frames:
        try:
            divs[code] = fetch_etf_dividends(code)
        except Exception as exc:  # noqa: BLE001
            div_fail.append(f"{code}: {type(exc).__name__}")
    div_yield, div_notes = dividend_yield_panel(etf_frames, divs, etf_rets.index)
    div_events = {
        code: int((div_yield.loc[EVAL_START:LAST_BAR, code] > 0).sum()) for code in div_yield.columns
    }
    div_total = {
        code: float(div_yield.loc[EVAL_START:LAST_BAR, code].sum()) for code in div_yield.columns
    }
    # A dividend on a session whose price return is missing (hole) is dropped with it.
    etf_rets = etf_rets + div_yield.reindex_like(etf_rets).fillna(0.0).where(etf_rets.notna())

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

    rf_daily: Optional[pd.Series] = None
    rf_full = pd.Series(0.0, index=etf_rets.index)
    rf_note = ""
    try:
        shibor = fetch_shibor_3m()
        rf_full = (shibor.reindex(etf_rets.index, method="ffill").shift(1) / 100.0 / TRADING_DAYS).fillna(0.0)
        rf_daily = rf_full.loc[EVAL_START:last_date]
        rf_note = (
            f"无风险利率：3M Shibor（AkShare `rate_interbank`），滞后 1 日、按 252 天折算；评估期均值 "
            f"{rf_daily.mean() * TRADING_DAYS:.2%}，最近 {shibor.loc[:last_date].iloc[-1]:.2f}%（{shibor.loc[:last_date].index[-1].date()}）。"
        )
    except Exception as exc:  # noqa: BLE001
        rf_note = (
            f"3M Shibor 获取失败（{type(exc).__name__}），超额现金夏普记为 n/a，门槛回退到 rf=0 夏普；"
            "现金收益仍按 0 计。"
        )

    weights: Dict[str, pd.DataFrame] = {}
    weights["BH_CSI300"] = pd.DataFrame(1.0, index=etf_close.index, columns=["510300"])
    weights["EW_Stocks"] = strategy_equal_weight(stock_close, stock_amount, STOCK_MIN_AMOUNT)
    weights["EW_Sectors"] = strategy_equal_weight(sector_close, sector_amount, ETF_MIN_AMOUNT)
    weights["Sector_LowVol"] = strategy_low_vol(sector_close, sector_amount, 60, 3)
    rp_close = etf_close[[c for c in RISK_PARITY_ETFS if c in etf_close.columns]]
    weights["RiskParity_SBG"] = strategy_risk_parity(rp_close, 60)
    weights["EW_SBG"] = _materialize(
        pd.DataFrame({c: np.where(etf_close.index.isin(month_ends(etf_close.index)), 1.0 / len(rp_close.columns), np.nan) for c in rp_close.columns}, index=etf_close.index)
    )
    weights["BH_Bond10Y"] = pd.DataFrame(1.0, index=etf_close.index, columns=["511260"])

    # Fundamentals. A failed download leaves the name out of that signal; nothing is estimated.
    val_frames: Dict[str, pd.DataFrame] = {}
    stock_divs: Dict[str, pd.DataFrame] = {}
    for code, name in STOCKS.items():
        if code not in stock_frames:
            continue
        try:
            val_frames[code] = fetch_stock_valuation(code)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code} {name} 估值: {exc}")
        try:
            stock_divs[code] = fetch_stock_dividends(code)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code} {name} 分红: {type(exc).__name__}")
    print(f"valuation {len(val_frames)}/{len(stock_frames)} dividends {len(stock_divs)}/{len(stock_frames)}")

    def val_panel(col: str) -> pd.DataFrame:
        cols = {c: f.loc[f["date"] <= LAST_BAR].set_index("date")[col] for c, f in val_frames.items()}
        return pd.DataFrame(cols).reindex(index=stock_close.index, columns=stock_close.columns)

    raw_close = val_panel("raw_close")
    divy_panel = trailing_dividend_yield(stock_divs, raw_close)
    mkt_rets = etf_rets["510300"]
    resmom_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

    def resmom(window: int = 240, skip: int = 20) -> pd.DataFrame:
        if (window, skip) not in resmom_cache:
            resmom_cache[(window, skip)] = residual_momentum_score(stock_rets, mkt_rets, window, skip)
        return resmom_cache[(window, skip)]

    def build_divy(c: pd.DataFrame, a: pd.DataFrame, n: int = 5, panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return strategy_stock_rank_low(-(divy_panel if panel is None else panel)[c.columns], c, a, 240, n)

    def build_resmom(c: pd.DataFrame, a: pd.DataFrame, window: int = 240, n: int = 5) -> pd.DataFrame:
        return strategy_stock_rank_low(-resmom(window)[c.columns], c, a, window, n)

    def build_rawmom_12_1(c: pd.DataFrame, a: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        raw_mom = c.shift(20) / c.shift(240) - 1.0
        return strategy_stock_rank_low(-raw_mom, c, a, 240, n)

    sb_cols = ["510300", "511260"]
    idx_all = etf_close.index

    # --- New candidate 1: market margin-flow timing (510300 vs 511260). ---
    try:
        mkt_margin = fetch_market_margin()
        print(f"market margin {mkt_margin.index.max().date()} n={len(mkt_margin)}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"market margin missing ({exc}); abort rather than invent a signal")
    # Balance of session d is published after the close of d: at the close of t only t-1 is known.
    fin_total = mkt_margin["total"].reindex(idx_all).ffill(limit=5).shift(1)

    def margin_flow(window: int = 20) -> pd.Series:
        return fin_total / fin_total.shift(window) - 1.0

    csi_ret_lag = etf_close["510300"].shift(1)

    def price_flow(window: int = 20) -> pd.Series:
        return csi_ret_lag / csi_ret_lag.shift(window) - 1.0

    def build_margin(window: int = 20, freq: str = "M", score: Optional[pd.Series] = None) -> pd.DataFrame:
        sc = margin_flow(window) if score is None else score
        return strategy_sign_switch(sc, "510300", "511260", sb_cols, 0.0, freq)

    # --- New candidate 2: term-spread timing (510300 vs 511260). ---
    try:
        curve = fetch_cn_curve()
        print(f"cn curve {curve.index.max().date()} n={len(curve)}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"CN curve missing ({exc}); abort rather than invent a signal")
    curve_a = curve.reindex(idx_all).ffill(limit=5).shift(1)
    slope = curve_a["y10"] - curve_a["y2"]

    def build_slope(med_window: int = 250, freq: str = "M", score: Optional[pd.Series] = None) -> pd.DataFrame:
        sc = slope if score is None else score
        return strategy_state_switch(sc, "510300", "511260", sb_cols, med_window, 120, freq)

    # --- New candidate 3: per-stock margin crowding (SSE names). ---
    sse_codes = [c for c in SSE_STOCKS if c in stock_close.columns]
    me_stock = month_ends(stock_close.index)
    pos = stock_close.index.searchsorted(me_stock)
    prev_sessions = [stock_close.index[p - 1] for p in pos if p >= 1]
    me_with_prev = [me for me, p in zip(me_stock, pos) if p >= 1]
    fetch_dates = [d for d, me in zip(prev_sessions, me_with_prev) if me >= pd.Timestamp("2019-06-01")]
    stk_margin, stk_margin_gaps = fetch_stock_margin_sse(fetch_dates, sse_codes)
    print(f"sse stock margin dates={len(stk_margin)} gaps={len(stk_margin_gaps)}")
    prev_to_me = dict(zip(prev_sessions, me_with_prev))
    bal_me = stk_margin.copy()
    bal_me.index = pd.DatetimeIndex([prev_to_me[d] for d in bal_me.index])
    crowd_cache: Dict[int, pd.DataFrame] = {}

    def crowd(lag_months: int = 3) -> pd.DataFrame:
        if lag_months not in crowd_cache:
            crowd_cache[lag_months] = margin_change_panel(bal_me, stock_close.index, lag_months).reindex(columns=stock_close.columns)
        return crowd_cache[lag_months]

    def build_crowd(c: pd.DataFrame, a: pd.DataFrame, n: int = 5, lag_months: int = 3, inverse: bool = False,
                    resid: bool = False) -> pd.DataFrame:
        cols = [x for x in c.columns if x in sse_codes]
        sc = crowd(lag_months)
        if resid:
            me_idx = month_ends(stock_close.index)
            mc = stock_close.reindex(me_idx)
            sc = residualize(sc.loc[me_idx], (mc / mc.shift(3) - 1.0)).reindex(stock_close.index)
        sc = -sc if inverse else sc
        return strategy_stock_rank_low(sc[cols], c[cols], a.reindex(columns=cols), 120, n).reindex(columns=c.columns, fill_value=0.0)

    # --- Watchlist: shareholder-count concentration. ---
    holder_frames: Dict[str, pd.DataFrame] = {}
    for code, name in STOCKS.items():
        if code not in stock_frames:
            continue
        try:
            holder_frames[code] = fetch_holder_counts(code)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code} {name} 股东户数: {type(exc).__name__}")
    print(f"holder counts {len(holder_frames)}/{len(stock_frames)}")
    hold_cache: Dict[Tuple[int, str], pd.DataFrame] = {}

    def holder_panel(periods: int = 1, field: str = "chg") -> pd.DataFrame:
        key = (periods, field)
        if key not in hold_cache:
            hold_cache[key] = holder_change_panel(holder_frames, stock_close.index, 200, periods, field).reindex(columns=stock_close.columns)
        return hold_cache[key]

    def build_holder(c: pd.DataFrame, a: pd.DataFrame, n: int = 5, periods: int = 1, resid: bool = False) -> pd.DataFrame:
        sc = holder_panel(periods)
        if resid:
            me_idx = month_ends(stock_close.index)
            sc = residualize(sc.loc[me_idx], holder_panel(1, "ret").loc[me_idx]).reindex(stock_close.index)
        return strategy_stock_rank_low(sc[c.columns], c, a, 120, n)

    weights["EW_SB"] = static_mix(etf_close.index, {"510300": 0.5, "511260": 0.5})
    weights["Margin_Timing"] = build_margin()
    weights["TermSpread_Timing"] = build_slope()
    weights["EW_SSE"] = strategy_equal_weight(stock_close[sse_codes], stock_amount.reindex(columns=sse_codes), STOCK_MIN_AMOUNT)
    weights["Stock_LowVol"] = strategy_stock_low_vol(stock_close, stock_amount, 60, 5)
    weights["Stock_MomLowVol"] = strategy_stock_mom_lowvol(stock_close, stock_amount, 120, 60, 5)
    weights["Stock_DivYield"] = build_divy(stock_close, stock_amount)
    weights["Stock_ResMom"] = build_resmom(stock_close, stock_amount)
    weights["Stock_HolderConc"] = build_holder(stock_close, stock_amount)
    weights["Stock_MarginCrowd"] = build_crowd(stock_close, stock_amount)

    stock_strats = {
        "EW_Stocks", "EW_SSE", "Stock_LowVol", "Stock_MomLowVol", "Stock_DivYield", "Stock_ResMom", "Stock_HolderConc",
        "Stock_MarginCrowd",
    }
    cost_map = {n: (STOCK_COST_BPS if n in stock_strats else ETF_COST_BPS) for n in weights}
    ret_map = {n: (stock_rets if n in stock_strats else etf_rets) for n in weights}
    universe_bench = {
        "Sector_LowVol": "EW_Sectors",
        "Stock_LowVol": "EW_Stocks",
        "Stock_MomLowVol": "EW_Stocks",
        "Stock_DivYield": "EW_Stocks",
        "Stock_ResMom": "EW_Stocks",
        "Stock_HolderConc": "EW_Stocks",
        "Stock_MarginCrowd": "EW_SSE",
        "RiskParity_SBG": "EW_SBG",
        "Margin_Timing": "EW_SB",
        "TermSpread_Timing": "EW_SB",
    }

    # Sensitivity diagnostics. Not used for selection. (spec -> weights, kind)
    sensitivity: Dict[str, Tuple[pd.DataFrame, str]] = {
        "Margin_w20": (weights["Margin_Timing"], "etf"),
        "Margin_w10": (build_margin(10), "etf"),
        "Margin_w60": (build_margin(60), "etf"),
        "Margin_weekly": (build_margin(20, "W"), "etf"),
        "Margin_inverse": (build_margin(20, "M", -margin_flow(20)), "etf"),
        "Price_w20_control": (build_margin(20, "M", price_flow(20)), "etf"),
        "Margin_minus_price_w20": (build_margin(20, "M", margin_flow(20) - price_flow(20)), "etf"),
        "Slope_med250": (weights["TermSpread_Timing"], "etf"),
        "Slope_med500": (build_slope(500), "etf"),
        "Slope_weekly": (build_slope(250, "W"), "etf"),
        "Slope_chg60>0": (build_margin(60, "M", slope - slope.shift(60)), "etf"),
        "Y10_level_med250": (build_slope(250, "M", curve_a["y10"]), "etf"),
        "Slope_inverse": (build_slope(250, "M", -slope), "etf"),
        "Crowd_n5": (weights["Stock_MarginCrowd"], "stock"),
        "Crowd_n3": (build_crowd(stock_close, stock_amount, 3), "stock"),
        "Crowd_n8": (build_crowd(stock_close, stock_amount, 8), "stock"),
        "Crowd_1m": (build_crowd(stock_close, stock_amount, 5, 1), "stock"),
        "Crowd_resid_vs_ret3m": (build_crowd(stock_close, stock_amount, 5, 3, False, True), "stock"),
        "Crowd_inverse": (build_crowd(stock_close, stock_amount, 5, 3, True), "stock"),
        "Holder_n5": (weights["Stock_HolderConc"], "stock"),
        "Holder_n3": (build_holder(stock_close, stock_amount, 3), "stock"),
        "Holder_n8": (build_holder(stock_close, stock_amount, 8), "stock"),
        "Holder_2period": (build_holder(stock_close, stock_amount, 5, 2), "stock"),
        "Holder_resid_vs_ret": (build_holder(stock_close, stock_amount, 5, 1, True), "stock"),
        "DivY_n5": (weights["Stock_DivYield"], "stock"),
        "DivY_n3": (build_divy(stock_close, stock_amount, 3), "stock"),
        "DivY_n8": (build_divy(stock_close, stock_amount, 8), "stock"),
        "ResMom_w240_n5": (weights["Stock_ResMom"], "stock"),
        "ResMom_w120_n5": (build_resmom(stock_close, stock_amount, 120, 5), "stock"),
        "ResMom_w240_n3": (build_resmom(stock_close, stock_amount, 240, 3), "stock"),
        "ResMom_w240_n8": (build_resmom(stock_close, stock_amount, 240, 8), "stock"),
        "RawMom_12_1_n5": (build_rawmom_12_1(stock_close, stock_amount, 5), "stock"),
        "RP_vol20": (strategy_risk_parity(rp_close, 20), "etf"),
        "RP_vol60": (weights["RiskParity_SBG"], "etf"),
        "RP_vol120": (strategy_risk_parity(rp_close, 120), "etf"),
        "MomLowVol_n3": (strategy_stock_mom_lowvol(stock_close, stock_amount, 120, 60, 3), "stock"),
        "MomLowVol_n5": (weights["Stock_MomLowVol"], "stock"),
        "MomLowVol_n8": (strategy_stock_mom_lowvol(stock_close, stock_amount, 120, 60, 8), "stock"),
        "MomLowVol_m60": (strategy_stock_mom_lowvol(stock_close, stock_amount, 60, 60, 5), "stock"),
        "StkLowVol_n3": (strategy_stock_low_vol(stock_close, stock_amount, 60, 3), "stock"),
        "StkLowVol_n5": (weights["Stock_LowVol"], "stock"),
        "StkLowVol_n8": (strategy_stock_low_vol(stock_close, stock_amount, 60, 8), "stock"),
        "StkLowVol_vol120": (strategy_stock_low_vol(stock_close, stock_amount, 120, 5), "stock"),
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
        net2, _ = apply_costs(w, ret_map[name], 2.0 * cost_map[name])
        net2 = net2.loc[EVAL_START:last_date]
        stats[name] = {
            "full": full,
            "is": ins,
            "oos": oos,
            "cost2x": perf_stats(net2),
            "cost2x_oos": slice_stats(net2, turnover, OOS_START, last_date),
            "sharpe_cash": cash_excess_sharpe(net, rf_daily),
            "sharpe_cash_oos": cash_excess_sharpe(net.loc[OOS_START:], rf_daily),
            "sharpe_cash_2x": cash_excess_sharpe(net2, rf_daily),
        }

    def excess_cagr(a: pd.Series, b: pd.Series) -> float:
        aligned = pd.concat([a.rename("s"), b.rename("b")], axis=1).dropna()
        return perf_stats(aligned["s"] - aligned["b"]).get("cagr", float("nan"))

    bh = nets["BH_CSI300"]
    for name, net in nets.items():
        aligned = pd.concat([net.rename("s"), bh.rename("b")], axis=1).dropna()
        stats[name]["excess_cagr"] = excess_cagr(net, bh)
        stats[name]["corr"] = float(aligned["s"].corr(aligned["b"])) if len(aligned) > 10 else float("nan")
        stats[name]["stress"] = {
            "covid_2020q1": window_return(net, "2020-01-15", "2020-03-31"),
            "year_2022": window_return(net, "2022-01-01", "2022-12-31"),
            "rally_202409_202503": window_return(net, "2024-09-01", "2025-03-31"),
            "ytd_2026": window_return(net, "2026-01-01", str(LAST_BAR.date())),
            "mdd_2022": max_dd_window(net, "2022-01-01", "2022-12-31"),
        }
        ov = vol_target_overlay(net)
        stats[name]["overlay"] = perf_stats(ov)
        ub = universe_bench.get(name)
        stats[name]["universe_bench"] = ub
        stats[name]["excess_vs_universe_cagr"] = excess_cagr(net, nets[ub]) if ub else float("nan")
        stats[name]["excess_vs_ew_cagr"] = excess_cagr(net, nets["EW_Stocks"])

    sens_rows = {}
    for name, (w, kind) in sensitivity.items():
        rr, cb = (stock_rets, STOCK_COST_BPS) if kind == "stock" else (etf_rets, ETF_COST_BPS)
        net, turnover = apply_costs(w, rr, cb)
        net = net.loc[EVAL_START:last_date]
        turnover = turnover.loc[EVAL_START:last_date]
        sens_rows[name] = {
            "full": perf_stats(net, turnover),
            "oos": slice_stats(net, turnover, OOS_START, last_date),
        }

    # Factor overlap: are the new stock rules a different bet, or Stock_LowVol relabelled?
    # Correlation of daily returns in excess of EW_Stocks, and the average share of month-end
    # holdings shared with Stock_LowVol.
    stock_rules = ["Stock_LowVol", "Stock_MomLowVol", "Stock_DivYield", "Stock_ResMom", "Stock_HolderConc", "Stock_MarginCrowd"]
    ew_net_full = nets["EW_Stocks"]
    excess_df = pd.DataFrame({n: nets[n] - ew_net_full for n in stock_rules}).dropna()
    excess_corr = excess_df.corr()
    me_eval = [d for d in month_ends(stock_close.index) if EVAL_START <= d <= last_date]

    def overlap_with_lowvol(n: str) -> float:
        a = weights[n].reindex(me_eval) > 1e-9
        b = weights["Stock_LowVol"].reindex(me_eval) > 1e-9
        shared = (a & b).sum(axis=1)
        size = a.sum(axis=1).replace(0, np.nan)
        return float((shared / size).mean())

    overlap_rows = []
    for n in stock_rules:
        overlap_rows.append(
            [n]
            + [fmt_num(float(excess_corr.at[n, m])) for m in stock_rules]
            + [fmt_pct(overlap_with_lowvol(n), 0)]
        )
    overlap_table = markdown_table(
        ["相对 EW 超额收益相关"] + [m.replace("Stock_", "") for m in stock_rules] + ["与 LowVol 持仓重合"],
        overlap_rows,
    )

    # Leave-one-stock-out: rerun each stock rule (and the pool benchmark) with one name removed.
    # A rule whose excess over EW_Stocks flips sign when a single name is dropped depends on that name.
    loo_builders = {
        "Stock_LowVol": lambda c, a: strategy_stock_low_vol(c, a, 60, 5),
        "Stock_MomLowVol": lambda c, a: strategy_stock_mom_lowvol(c, a, 120, 60, 5),
        "Stock_DivYield": lambda c, a: build_divy(c, a),
        "Stock_ResMom": lambda c, a: build_resmom(c, a),
        "Stock_HolderConc": lambda c, a: build_holder(c, a),
        "Stock_MarginCrowd": lambda c, a: build_crowd(c, a),
    }
    # Stock_MarginCrowd only trades SSE names: drop only SSE names and compare with the SSE equal weight.
    loo_sse_only = {"Stock_MarginCrowd"}
    loo: Dict[str, Dict[str, object]] = {n: {"cagr": [], "excess": []} for n in loo_builders}
    for drop in stock_close.columns:
        keep = [c for c in stock_close.columns if c != drop]
        c_sub, a_sub, r_sub = stock_close[keep], stock_amount.reindex(columns=keep), stock_rets[keep]
        ew_net, _ = apply_costs(strategy_equal_weight(c_sub, a_sub, STOCK_MIN_AMOUNT), r_sub, STOCK_COST_BPS)
        ew_net = ew_net.loc[EVAL_START:last_date]
        keep_sse = [c for c in keep if c in sse_codes]
        ew_sse_net = None
        if drop in sse_codes:
            ew_sse_net, _ = apply_costs(strategy_equal_weight(c_sub[keep_sse], a_sub.reindex(columns=keep_sse), STOCK_MIN_AMOUNT), r_sub, STOCK_COST_BPS)
            ew_sse_net = ew_sse_net.loc[EVAL_START:last_date]
        for n, build in loo_builders.items():
            if n in loo_sse_only and drop not in sse_codes:
                continue
            net, _ = apply_costs(build(c_sub, a_sub), r_sub, STOCK_COST_BPS)
            net = net.loc[EVAL_START:last_date]
            loo[n]["cagr"].append((drop, perf_stats(net)["cagr"]))
            loo[n]["excess"].append((drop, excess_cagr(net, ew_sse_net if n in loo_sse_only else ew_net)))
    loo_rows = []
    for n, d in loo.items():
        cagrs = sorted(d["cagr"], key=lambda x: x[1])
        exc = sorted(d["excess"], key=lambda x: x[1])
        positive = sum(1 for _, v in exc if v > 0)
        loo_rows.append(
            [
                n,
                fmt_pct(stats[n]["full"]["cagr"]),
                f"{fmt_pct(cagrs[0][1])}（去掉 {STOCKS.get(cagrs[0][0], cagrs[0][0])}）",
                f"{fmt_pct(cagrs[-1][1])}（去掉 {STOCKS.get(cagrs[-1][0], cagrs[-1][0])}）",
                fmt_pct(stats[n]["excess_vs_universe_cagr"]),
                f"{fmt_pct(exc[0][1])} ~ {fmt_pct(exc[-1][1])}",
                f"{positive}/{len(exc)}",
            ]
        )
        stats[n]["loo"] = {"cagr_min": cagrs[0], "cagr_max": cagrs[-1], "excess_min": exc[0], "excess_max": exc[-1], "excess_positive": positive, "n": len(exc)}
    loo_table = markdown_table(
        ["Strategy", "Full CAGR", "LOO min CAGR", "LOO max CAGR", "Excess vs universe", "LOO excess vs universe range", "LOO excess>0"],
        loo_rows,
    )

    # Calendar-year return minus the universe benchmark: does an edge hold across regimes or sit in one window?
    year_list = list(range(EVAL_START.year, last_date.year + 1))
    yearly_rows = []
    for n in ["Margin_Timing", "TermSpread_Timing", "Stock_MarginCrowd", "Stock_HolderConc", "RiskParity_SBG", "Stock_ResMom", "Stock_MomLowVol", "Stock_DivYield", "Stock_LowVol", "Sector_LowVol"]:
        ub = universe_bench[n]
        diffs = [
            window_return(nets[n], f"{y}-01-01", f"{y}-12-31") - window_return(nets[ub], f"{y}-01-01", f"{y}-12-31")
            for y in year_list
        ]
        stats[n]["yearly_excess"] = dict(zip(year_list, diffs))
        yearly_rows.append([n, ub] + [fmt_pct(d) for d in diffs] + [f"{sum(1 for d in diffs if d > 0)}/{len(diffs)}"])
    yearly_table = markdown_table(
        ["Strategy", "对照"] + [str(y) + ("（至今）" if y == last_date.year else "") for y in year_list] + ["正超额年数"],
        yearly_rows,
    )

    def tstat_of(x: pd.Series) -> float:
        x = x.dropna()
        sd = x.std(ddof=1) if len(x) > 2 else float("nan")
        return float(x.mean() / sd * math.sqrt(len(x))) if sd and sd > 0 else float("nan")

    mech: Dict[str, object] = {}

    eq_ret = etf_rets["510300"].loc[EVAL_START:last_date]
    bd_ret = etf_rets["511260"].loc[EVAL_START:last_date]
    eq_minus_bond = eq_ret - bd_ret

    def held_equity(w: pd.DataFrame) -> pd.Series:
        return w["510300"].shift(1).fillna(0.0).loc[EVAL_START:last_date] > 0.5

    def state_rows(name: str, labels: Tuple[str, str]) -> Tuple[List[List[str]], Dict[str, Dict[str, float]]]:
        on = held_equity(weights[name])
        rows, out = [], {}
        for label, m in ((labels[0], on), (labels[1], ~on)):
            part = eq_minus_bond[m]
            row = {"share": float(m.mean()), "eq": float(eq_ret[m].mean() * TRADING_DAYS), "bond": float(bd_ret[m].mean() * TRADING_DAYS),
                   "ann": float(part.mean() * TRADING_DAYS), "t": tstat_of(part),
                   "ann_is": float(part.loc[:IS_END].mean() * TRADING_DAYS), "ann_oos": float(part.loc[OOS_START:].mean() * TRADING_DAYS)}
            out[label] = row
            rows.append([label, fmt_pct(row["share"], 0), fmt_pct(row["eq"]), fmt_pct(row["bond"]), fmt_pct(row["ann"]), fmt_num(row["t"]),
                         fmt_pct(row["ann_is"]), fmt_pct(row["ann_oos"])])
        return rows, out

    state_headers = ["状态", "占评估期交易日", "沪深300ETF 年化", "十年国债ETF 年化", "股票−国债 年化", "t 值", "样本内年化", "样本外年化"]

    def switches_per_year(name: str) -> float:
        w = weights[name]["510300"].loc[EVAL_START:last_date]
        return float(w.diff().abs().gt(0.5).sum() / (len(w) / TRADING_DAYS))

    # Margin flow: state split, and whether the leverage signal says anything beyond 20-day price momentum.
    m_rows, mech["margin_states"] = state_rows("Margin_Timing", ("融资余额 20 日增加（持股）", "融资余额 20 日减少（持债）"))
    me_eval_etf = [d for d in month_ends(idx_all) if EVAL_START <= d <= last_date]
    mf, pf = margin_flow(20).reindex(me_eval_etf), price_flow(20).reindex(me_eval_etf)
    ok_mp = mf.notna() & pf.notna()
    on_m = held_equity(weights["Margin_Timing"])
    on_p = held_equity(sensitivity["Price_w20_control"][0])
    dis_rows = []
    mech["margin_disagree"] = {}
    for label, m in (("两者同为持股", on_m & on_p), ("两者同为持债", ~on_m & ~on_p),
                     ("融资增加、价格下跌（仅融资信号持股）", on_m & ~on_p), ("融资减少、价格上涨（仅价格信号持股）", ~on_m & on_p)):
        part = eq_minus_bond[m]
        row = {"share": float(m.mean()), "ann": float(part.mean() * TRADING_DAYS) if len(part) else float("nan"), "t": tstat_of(part)}
        mech["margin_disagree"][label] = row
        dis_rows.append([label, fmt_pct(row["share"], 0), fmt_pct(row["ann"]), fmt_num(row["t"])])
    mech["margin_corr_price"] = float(mf[ok_mp].corr(pf[ok_mp])) if int(ok_mp.sum()) > 10 else float("nan")
    mech["margin_sign_agree"] = float(((mf[ok_mp] > 0) == (pf[ok_mp] > 0)).mean()) if int(ok_mp.sum()) else float("nan")
    mech["margin_on_share_months"] = float((mf[ok_mp] > 0).mean()) if int(ok_mp.sum()) else float("nan")
    mech["margin_switches_per_year"] = switches_per_year("Margin_Timing")
    mf_now = margin_flow(20).loc[:last_date].dropna()
    mech["margin_last"] = {"date": str(mf_now.index[-1].date()), "flow20": float(mf_now.iloc[-1]),
                           "balance_date": str(mkt_margin["total"].dropna().index[-1].date()),
                           "balance": float(mkt_margin["total"].dropna().iloc[-1])}
    margin_table = "\n\n".join([
        markdown_table(state_headers, m_rows),
        f"月末融资 20 日变化与沪深300ETF 20 日收益的相关 {fmt_num(mech['margin_corr_price'])}，符号一致的月份占 {fmt_pct(mech['margin_sign_agree'], 0)}，"
        f"融资信号持股的月份占 {fmt_pct(mech['margin_on_share_months'], 0)}。两个信号分歧时的持有期「股票−国债」（年化，未扣费）：",
        markdown_table(["月末状态", "占评估期交易日", "股票−国债 年化", "t 值"], dis_rows),
    ])

    # Term spread: state split; the bond leg shows whether a steep curve paid through carry rather than growth.
    s_rows, mech["slope_states"] = state_rows("TermSpread_Timing", ("期限利差高于中位数（持股）", "期限利差低于中位数（持债）"))
    slope_now = slope.loc[:last_date].dropna()
    slope_med = slope.shift(1).rolling(250, min_periods=120).median().loc[:last_date].dropna()
    mech["slope_last"] = {"date": str(slope_now.index[-1].date()), "slope": float(slope_now.iloc[-1]), "median250": float(slope_med.iloc[-1]),
                          "y10": float(curve_a["y10"].loc[:last_date].dropna().iloc[-1]), "y2": float(curve_a["y2"].loc[:last_date].dropna().iloc[-1])}
    mech["slope_switches_per_year"] = switches_per_year("TermSpread_Timing")
    on_s = held_equity(weights["TermSpread_Timing"])
    mech["slope_margin_agree"] = float((on_s == on_m).mean())
    slope_table = markdown_table(state_headers, s_rows)

    # Holder concentration: rank IC with next-month return, overlap with price moves, coverage.
    me_all = [d for d in month_ends(stock_close.index) if EVAL_START <= d <= last_date]
    mclose = stock_close.reindex(month_ends(stock_close.index))
    fwd = (mclose.shift(-1) / mclose - 1.0).reindex(me_all)
    past1m = (mclose / mclose.shift(1) - 1.0).reindex(me_all)
    hchg = holder_panel(1).reindex(me_all)
    hret = holder_panel(1, "ret").reindex(me_all)
    ics, ic_dates, c_ret, c_past, cover = [], [], [], [], []
    for dt in me_all:
        h = hchg.loc[dt]
        ok = h.notna() & fwd.loc[dt].notna()
        cover.append(int(h.notna().sum()))
        if int(ok.sum()) >= 8:
            ics.append(float(-h[ok].rank().corr(fwd.loc[dt][ok].rank())))
            ic_dates.append(dt)
        ok2 = h.notna() & hret.loc[dt].notna()
        if int(ok2.sum()) >= 8:
            c_ret.append(float(h[ok2].rank().corr(hret.loc[dt][ok2].rank())))
        ok3 = h.notna() & past1m.loc[dt].notna()
        if int(ok3.sum()) >= 8:
            c_past.append(float(h[ok3].rank().corr(past1m.loc[dt][ok3].rank())))
    ic_s = pd.Series(ics, index=pd.DatetimeIndex(ic_dates))
    mech["holder"] = {
        "ic_mean": float(ic_s.mean()), "ic_t": tstat_of(ic_s), "ic_pos_share": float((ic_s > 0).mean()),
        "ic_is": float(ic_s.loc[:IS_END].mean()), "ic_oos": float(ic_s.loc[OOS_START:].mean()), "n_months": int(len(ic_s)),
        "corr_interval_ret": float(np.nanmean(c_ret)) if c_ret else float("nan"),
        "corr_past1m": float(np.nanmean(c_past)) if c_past else float("nan"),
        "coverage_mean": float(np.mean(cover)) if cover else float("nan"),
        "coverage_min": int(min(cover)) if cover else 0,
    }
    hm = mech["holder"]
    holder_table = markdown_table(
        ["指标", "数值"],
        [
            ["秩 IC（户数降幅 vs 下月收益，月均）", fmt_num(hm["ic_mean"], 3)],
            ["IC t 值 / IC>0 月份占比", f"{fmt_num(hm['ic_t'])} / {fmt_pct(hm['ic_pos_share'], 0)}"],
            ["样本内 / 样本外 IC 均值", f"{fmt_num(hm['ic_is'], 3)} / {fmt_num(hm['ic_oos'], 3)}"],
            ["户数变化与同期区间涨跌幅的秩相关（月均）", fmt_num(hm["corr_interval_ret"])],
            ["户数变化与过去 1 个月收益的秩相关（月均）", fmt_num(hm["corr_past1m"])],
            ["每个月末有有效得分的股票数（均值 / 最少）", f"{hm['coverage_mean']:.1f} / {hm['coverage_min']}"],
        ],
    )
    # Margin crowding: rank IC of the (negated) leverage build-up with next-month return, overlap with price moves.
    cchg = crowd(3).reindex(me_all)
    past3m = (mclose / mclose.shift(3) - 1.0).reindex(me_all)
    c_ics, c_dates, c_p3, c_cover = [], [], [], []
    for dt in me_all:
        h = cchg.loc[dt].reindex(sse_codes)
        f_ = fwd.loc[dt].reindex(sse_codes)
        ok = h.notna() & f_.notna()
        c_cover.append(int(h.notna().sum()))
        if int(ok.sum()) >= 8:
            c_ics.append(float(-h[ok].rank().corr(f_[ok].rank())))
            c_dates.append(dt)
        p3 = past3m.loc[dt].reindex(sse_codes)
        ok3 = h.notna() & p3.notna()
        if int(ok3.sum()) >= 8:
            c_p3.append(float(h[ok3].rank().corr(p3[ok3].rank())))
    cic = pd.Series(c_ics, index=pd.DatetimeIndex(c_dates))
    mech["crowd"] = {
        "ic_mean": float(cic.mean()), "ic_t": tstat_of(cic), "ic_pos_share": float((cic > 0).mean()),
        "ic_is": float(cic.loc[:IS_END].mean()), "ic_oos": float(cic.loc[OOS_START:].mean()), "n_months": int(len(cic)),
        "corr_past3m": float(np.nanmean(c_p3)) if c_p3 else float("nan"),
        "coverage_mean": float(np.mean(c_cover)) if c_cover else float("nan"),
        "coverage_min": int(min(c_cover)) if c_cover else 0,
        "n_sse": len(sse_codes),
    }
    cm = mech["crowd"]
    crowd_table = markdown_table(
        ["指标", "数值"],
        [
            ["秩 IC（3 个月融资余额增幅取负 vs 下月收益，月均）", fmt_num(cm["ic_mean"], 3)],
            ["IC t 值 / IC>0 月份占比", f"{fmt_num(cm['ic_t'])} / {fmt_pct(cm['ic_pos_share'], 0)}"],
            ["样本内 / 样本外 IC 均值", f"{fmt_num(cm['ic_is'], 3)} / {fmt_num(cm['ic_oos'], 3)}"],
            ["融资余额增幅与过去 3 个月收益的秩相关（月均）", fmt_num(cm["corr_past3m"])],
            [f"每个月末有有效得分的上交所股票数（均值 / 最少，共 {cm['n_sse']} 只）", f"{cm['coverage_mean']:.1f} / {cm['coverage_min']}"],
        ],
    )
    mech_table = "\n\n".join([
        "**Margin_Timing**：按月末状态分组，持有期内沪深300ETF、十年国债ETF 与二者之差的日收益（年化，未扣费）；再与只用 20 日价格动量的对照信号比较。",
        margin_table,
        "**TermSpread_Timing**：按月末状态分组（年化，未扣费）。若期限利差高时国债也更好，说明陡峭曲线的收益走的是久期 carry，而不是增长预期。",
        slope_table,
        "**Stock_MarginCrowd**：个股融资余额增幅（上交所 17 只）的截面预测力与它和价格变动的重叠。",
        crowd_table,
        "**Stock_HolderConc**（观察名单，沿用）：股东户数降幅的截面预测力与它和价格变动的重叠。",
        holder_table,
    ])

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
        "dividend_events_in_eval": div_events,
        "dividend_yield_sum_in_eval": div_total,
        "dividend_notes": div_notes,
        "dividend_fetch_failures": div_fail,
        "mechanism": mech,
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
            fmt_num(stats[name]["sharpe_cash"]),
            fmt_num(s["sortino"]),
            fmt_pct(s["max_dd"]),
            fmt_pct(s["win_rate_daily"], 0),
            fmt_pct(s["win_rate_monthly"], 0),
            fmt_num(s["profit_factor"]),
            fmt_num(s["ann_turnover"], 1),
        ]

    full_table = markdown_table(
        ["Strategy", "Total", "CAGR", "Sharpe", "Sharpe-rf", "Sortino", "MaxDD", "WinD", "WinM", "PF", "Turn/yr"],
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
        ["Strategy", "Excess CAGR vs BH", "Corr vs BH", "Universe bench", "Excess CAGR vs universe", "Full Sharpe", "OOS Sharpe"],
        [
            [
                n,
                fmt_pct(stats[n]["excess_cagr"]),
                fmt_num(stats[n]["corr"]),
                stats[n]["universe_bench"] or "n/a",
                fmt_pct(stats[n]["excess_vs_universe_cagr"]),
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

    cost_table = markdown_table(
        ["Strategy", "Base Sharpe-rf", "2x Sharpe-rf", "Base CAGR", "2x CAGR", "2x OOS Sharpe", "Turn/yr"],
        [
            [
                n,
                fmt_num(stats[n]["sharpe_cash"]),
                fmt_num(stats[n]["sharpe_cash_2x"]),
                fmt_pct(stats[n]["full"]["cagr"]),
                fmt_pct(stats[n]["cost2x"].get("cagr", float("nan"))),
                fmt_num(stats[n]["cost2x_oos"].get("sharpe", float("nan"))),
                fmt_num(stats[n]["full"]["ann_turnover"], 1),
            ]
            for n in order
        ],
    )

    benchmarks = {"BH_CSI300", "EW_Stocks", "EW_Sectors", "EW_SBG", "BH_Bond10Y", "EW_SB", "EW_SSE"}
    qual_lines = []
    survivors = []
    for n in order:
        if n in benchmarks:
            continue
        f, o = stats[n]["full"], stats[n]["oos"]
        checks = {
            "excess_cagr>0": stats[n]["excess_cagr"] > 0,
            "oos_total>=bh": o.get("total_return", -1) >= stats["BH_CSI300"]["oos"].get("total_return", 0),
            "cash_sharpe>1": (stats[n]["sharpe_cash"] if rf_daily is not None else f.get("sharpe", 0)) > 1.0,
            "mdd>-30%": f.get("max_dd", -1) > -0.30,
            "oos_sharpe>0.5": o.get("sharpe", 0) > 0.5,
        }
        if stats[n]["universe_bench"]:
            checks["beats_universe"] = stats[n]["excess_vs_universe_cagr"] > 0
        passed = all(checks.values())
        if passed:
            survivors.append(n)
        flag = "PASS" if passed else "FAIL"
        detail = ", ".join(f"{k}:{'Y' if v else 'N'}" for k, v in checks.items())
        if not stats[n]["universe_bench"]:
            detail += ", beats_universe:n/a"
        qual_lines.append(f"- **{n}** {flag} — {detail}")

    def score_of(n: str) -> float:
        """Fixed display score: Sharpe, OOS Sharpe, excess vs CSI300, drawdown. Not optimized."""
        f, o = stats[n]["full"], stats[n]["oos"]
        sharpe = f.get("sharpe", float("nan"))
        oos_s = o.get("sharpe", float("nan"))
        ex = stats[n]["excess_cagr"]
        mdd = f.get("max_dd", float("nan"))
        parts = [sharpe, oos_s, ex, mdd]
        if any(isinstance(p, float) and math.isnan(p) for p in parts):
            return -999.0
        return 0.35 * sharpe + 0.30 * oos_s + 0.20 * (ex / 0.05) + 0.15 * (mdd / 0.30)

    ranked = sorted([n for n in order if n not in benchmarks], key=score_of, reverse=True)
    rank_lines = ["按固定加权分排序（夏普、样本外夏普、超额、回撤），只用于并列展示，不能把 FAIL 说成合格：", ""]
    for i, n in enumerate(ranked, 1):
        f = stats[n]["full"]
        mark = "合格" if n in survivors else "未过门槛"
        ub = stats[n]["universe_bench"]
        ub_txt = f"相对 {ub} 超额 CAGR {fmt_pct(stats[n]['excess_vs_universe_cagr'])}，" if ub else ""
        rank_lines.append(
            f"{i}. **{n}**（{mark}，score {score_of(n):.2f}）— 全样本夏普 {fmt_num(f['sharpe'])}，"
            f"回撤 {fmt_pct(f['max_dd'])}，相对沪深300超额 CAGR {fmt_pct(stats[n]['excess_cagr'])}，"
            f"{ub_txt}样本外夏普 {fmt_num(stats[n]['oos']['sharpe'])}，成本翻倍夏普 {fmt_num(stats[n]['cost2x'].get('sharpe', float('nan')))}。"
        )
    rank_lines.append("")
    rank_lines.append(
        "行业或个股策略如果只跑赢沪深300、但跑不赢同一宇宙的等权（EW_Sectors / EW_Stocks），"
        "超额来自宇宙构成（含幸存者偏差），不能记成交易规则的 edge。"
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
            "- 事实：行业资金流与北向日频净买入的可用性见数据节；缺失项不进入策略排序。",
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

    def sc(n: str) -> str:
        return fmt_num(stats[n]["sharpe_cash"])

    def ub_txt(n: str) -> str:
        ub = stats[n]["universe_bench"]
        return f"相对 {ub} 超额 CAGR {fmt_pct(stats[n]['excess_vs_universe_cagr'])}" if ub else "无宇宙基准"

    rp_w = weights["RiskParity_SBG"].loc[EVAL_START:last_date].mean()
    rp_bond = float(rp_w.get("511260", float("nan")))
    rp_last_bond = float(weights["RiskParity_SBG"].loc[:last_date].iloc[-1].get("511260", float("nan")))

    def time_in(name: str, code: str) -> float:
        w = weights[name].loc[EVAL_START:last_date]
        return float((w.get(code, pd.Series(0.0, index=w.index)) > 0).mean())

    def loo_txt(n: str) -> str:
        d = stats[n].get("loo")
        if not d:
            return ""
        return (
            f"逐只剔除后相对 {stats[n]['universe_bench']} 超额 CAGR 在 {fmt_pct(d['excess_min'][1])} ~ {fmt_pct(d['excess_max'][1])}，"
            f"{d['excess_positive']}/{d['n']} 次为正。"
        )

    def sens_sharpe(spec: str) -> str:
        return fmt_num(sens_rows[spec]["full"].get("sharpe", float("nan")))

    def sens_oos(spec: str) -> str:
        return fmt_num(sens_rows[spec]["oos"].get("sharpe", float("nan")))

    def sens_mdd(spec: str) -> str:
        return fmt_pct(sens_rows[spec]["full"].get("max_dd", float("nan")))

    def sens_cagr(spec: str) -> str:
        return fmt_pct(sens_rows[spec]["full"].get("cagr", float("nan")))

    def new_stock_note(n: str, specs: List[Tuple[str, str]], ub: str = "EW_Stocks") -> str:
        s = stats[n]
        sens = "、".join(f"{lbl} {sens_sharpe(k)}/{sens_oos(k)}" for lbl, k in specs)
        return (
            f"样本内 Sharpe {fmt_num(s['is']['sharpe'])}、样本外 {fmt_num(s['oos']['sharpe'])}；"
            f"2022 年 {fmt_pct(s['stress']['year_2022'])}（{ub} {fmt_pct(stats[ub]['stress']['year_2022'])}），"
            f"2024-09~2025-03 {fmt_pct(s['stress']['rally_202409_202503'])}（{ub} {fmt_pct(stats[ub]['stress']['rally_202409_202503'])}），"
            f"2026 年至今 {fmt_pct(s['stress']['ytd_2026'])}（{ub} {fmt_pct(stats[ub]['stress']['ytd_2026'])}）；年换手 {fmt_num(s['full']['ann_turnover'], 1)} 倍。"
            f"敏感度（全样本/样本外 Sharpe）：{sens}。{loo_txt(n)}"
            + (
                f"与 Stock_LowVol 的超额收益相关 {fmt_num(float(excess_corr.at[n, 'Stock_LowVol']))}，持仓重合 {fmt_pct(overlap_with_lowvol(n), 0)}。"
                if n in excess_corr.index
                else ""
            )
        )

    notes = {
        "RiskParity_SBG": (
            f"评估期平均 {rp_bond:.0%} 仓位在十年国债ETF（当前 {rp_last_bond:.0%}），收益主要来自 2020–2025 年的利率下行；"
            f"单持国债 BH_Bond10Y 的 Sharpe-rf 为 {sc('BH_Bond10Y')}，静态三等分 EW_SBG 的全样本 CAGR {fmt_pct(stats['EW_SBG']['full']['cagr'])} 高于本策略。"
            "这是低波动组合，不是股票替代品；样本外总收益也低于沪深300。"
        ),
        "Stock_LowVol": (
            f"样本外 Sharpe {fmt_num(stats['Stock_LowVol']['oos']['sharpe'])} 但样本内只有 {fmt_num(stats['Stock_LowVol']['is']['sharpe'])}，"
            f"且全样本跑输股票池等权 {fmt_pct(-stats['Stock_LowVol']['excess_vs_universe_cagr'])}/年；它是更低回撤的池子 beta，不是选股 alpha。{loo_txt('Stock_LowVol')}"
        ),
        "Stock_MomLowVol": (
            f"样本内 Sharpe {fmt_num(stats['Stock_MomLowVol']['is']['sharpe'])}、样本外 {fmt_num(stats['Stock_MomLowVol']['oos']['sharpe'])}，"
            f"2022 年 {fmt_pct(stats['Stock_MomLowVol']['stress']['year_2022'])}（股票池等权 {fmt_pct(stats['EW_Stocks']['stress']['year_2022'])}），"
            f"2026 年至今 {fmt_pct(stats['Stock_MomLowVol']['stress']['ytd_2026'])}（股票池等权 {fmt_pct(stats['EW_Stocks']['stress']['ytd_2026'])}）。"
            f"持股数 3/5/8 的 Sharpe 为 {sens_sharpe('MomLowVol_n3')} / {sens_sharpe('MomLowVol_n5')} / {sens_sharpe('MomLowVol_n8')}，"
            f"动量窗口改为 60 日时 {sens_sharpe('MomLowVol_m60')}。仍跑输股票池等权，Sharpe-rf 未到 1。{loo_txt('Stock_MomLowVol')}"
        ),
        "Sector_LowVol": (
            f"样本外 Sharpe {fmt_num(stats['Sector_LowVol']['oos']['sharpe'])}，样本内 {fmt_num(stats['Sector_LowVol']['is']['sharpe'])}，"
            f"2026 年至今 {fmt_pct(stats['Sector_LowVol']['stress']['ytd_2026'])}，全样本相对行业 ETF 等权 {fmt_pct(stats['Sector_LowVol']['excess_vs_universe_cagr'])}/年；当前持仓与防御板块强势高度重合。"
        ),
        "Stock_DivYield": new_stock_note("Stock_DivYield", [("持 3 只", "DivY_n3"), ("持 8 只", "DivY_n8")]),
        "Stock_ResMom": new_stock_note(
            "Stock_ResMom",
            [("回归窗口 120 日", "ResMom_w120_n5"), ("持 3 只", "ResMom_w240_n3"), ("持 8 只", "ResMom_w240_n8"),
             ("对照：原始 12-1 动量", "RawMom_12_1_n5")],
        ),
        "Margin_Timing": new_stock_note(
            "Margin_Timing",
            [("窗口 10 日", "Margin_w10"), ("窗口 60 日", "Margin_w60"), ("周度调仓", "Margin_weekly"), ("反向（安慰剂）", "Margin_inverse"),
             ("对照：只用 20 日价格动量", "Price_w20_control"), ("融资变化减价格变化", "Margin_minus_price_w20")],
            "EW_SB",
        )
        + f"平均每年切换 {mech['margin_switches_per_year']:.1f} 次；融资 20 日变化与价格 20 日变化的月末相关 {fmt_num(mech['margin_corr_price'])}、符号一致 {fmt_pct(mech['margin_sign_agree'], 0)}。"
        f"最新两市融资余额 {mech['margin_last']['balance'] / 1e12:.3f} 万亿元（{mech['margin_last']['balance_date']}），信号口径（t−1）20 日变化 {fmt_pct(mech['margin_last']['flow20'], 2)}。",
        "TermSpread_Timing": new_stock_note(
            "TermSpread_Timing",
            [("中位数窗口 500 日", "Slope_med500"), ("周度调仓", "Slope_weekly"), ("60 日利差变化 > 0", "Slope_chg60>0"),
             ("只用 10Y 水平", "Y10_level_med250"), ("反向（安慰剂）", "Slope_inverse")],
            "EW_SB",
        )
        + f"平均每年切换 {mech['slope_switches_per_year']:.1f} 次；与 Margin_Timing 持仓状态一致的交易日 {fmt_pct(mech['slope_margin_agree'], 0)}。"
        f"最新（{mech['slope_last']['date']}，t−1 口径）10Y {mech['slope_last']['y10']:.4f}%、2Y {mech['slope_last']['y2']:.4f}%，利差 {mech['slope_last']['slope'] * 100:.1f}bp，"
        f"此前 250 日中位数 {mech['slope_last']['median250'] * 100:.1f}bp。",
        "Stock_MarginCrowd": new_stock_note(
            "Stock_MarginCrowd",
            [("持 3 只", "Crowd_n3"), ("持 8 只", "Crowd_n8"), ("1 个月变化", "Crowd_1m"), ("剔除 3 个月涨跌幅后的残差", "Crowd_resid_vs_ret3m"),
             ("反向（安慰剂：增幅最大的 5 只）", "Crowd_inverse")],
            "EW_SSE",
        )
        + f"秩 IC 月均 {fmt_num(mech['crowd']['ic_mean'], 3)}（t={fmt_num(mech['crowd']['ic_t'])}），融资增幅与过去 3 个月收益的秩相关 {fmt_num(mech['crowd']['corr_past3m'])}。",
        "Stock_HolderConc": new_stock_note(
            "Stock_HolderConc",
            [("持 3 只", "Holder_n3"), ("持 8 只", "Holder_n8"), ("两期累计变化", "Holder_2period"), ("剔除区间涨跌幅后的残差", "Holder_resid_vs_ret")],
        )
        + f"秩 IC 月均 {fmt_num(mech['holder']['ic_mean'], 3)}（t={fmt_num(mech['holder']['ic_t'])}），户数变化与同期区间涨跌幅的秩相关 {fmt_num(mech['holder']['corr_interval_ret'])}。",
    }
    for k, v in INTERP_NOTES.items():
        notes[k] = notes.get(k, "") + v

    rec_bits = [rec_intro, ""]
    if ELIMINATED:
        rec_bits.append(
            "本期淘汰的候选（" + "、".join(ELIMINATED) + "）不进入下面的观察名单，即使它们的展示分排在前面；逐项结论见本节末尾。"
        )
        rec_bits.append("")
    for n in [x for x in ranked if x not in ELIMINATED][:5]:
        f, o = stats[n]["full"], stats[n]["oos"]
        rec_bits.append(
            f"- **{n}**：全样本 CAGR {fmt_pct(f['cagr'])}，Sharpe {fmt_num(f['sharpe'])}（扣无风险 {sc(n)}），"
            f"最大回撤 {fmt_pct(f['max_dd'])}，样本外 Sharpe {fmt_num(o['sharpe'])}，"
            f"相对沪深300超额 CAGR {fmt_pct(stats[n]['excess_cagr'])}，{ub_txt(n)}。"
            f"{notes.get(n, '')}当前信号：{('、'.join(holdings.get(n) or [])) or '现金'}。"
        )
    rec_bits.append("")
    rec_bits.append(
        "不作为推荐的做法：用本次敏感度表里 Sharpe 最高的参数替换预注册参数（例如把 Stock_MomLowVol 改成持 8 只、把融资窗口或利差中位数窗口换成表里最好看的一档）；"
        "把单一行业主题或利率下行的历史 beta 当成策略 alpha。"
    )
    rec_bits.append("")
    new_cands = ["Margin_Timing", "TermSpread_Timing", "Stock_MarginCrowd"]
    rec_bits.append("本期新增 3 个候选的逐项结论（证据见第 4、5 节）：")
    rec_bits.append("")
    for n in new_cands:
        f = stats[n]["full"]
        verdict = "通过门槛" if n in survivors else "未过门槛"
        rec_bits.append(
            f"- **{n}**（{verdict}）：扣费 CAGR {fmt_pct(f['cagr'])}，Sharpe-rf {sc(n)}，最大回撤 {fmt_pct(f['max_dd'])}，{ub_txt(n)}。{notes[n]}"
        )

    deploy = "\n".join(
        [
            "- 交易时点：收盘信号，下一交易日成交。回测用下一根收盘到收盘近似，实盘用开盘或 VWAP 会再吃一点滑点。"
            "融资余额与国债收益率在回测里已滞后 1 个交易日；实盘在月末当天收盘前就能拿到 t−1 的数据，不依赖盘后信息。",
            "- 仓位：单策略名义敞口先按 10% 年化波动目标缩放，且杠杆不超过 1。回撤超过策略自身历史最大回撤的 1.25 倍时降到一半仓位并人工复核。",
            "- 容量：行业 ETF 只交易 20 日成交额中位数 ≥ 2000 万的品种；单日成交不超过该 ETF 20 日均成交额的 2%。股票池限定本次 24 只大盘股，不外推到小盘。"
            "十年国债ETF(511260) 与黄金ETF(518880) 近 20 日成交额中位数均在 30 亿元以上，个人与中小机构规模不构成约束。",
            "- 成本监控：若实盘单边冲击超过回测假设（ETF 8bps / 股票 15bps），停止加仓。新增规则的年换手："
            + "、".join(f"{n} {stats[n]['full']['ann_turnover']:.1f} 倍" for n in new_cands)
            + "。成本翻倍的影响见第 4 节。",
            "- 回测按目标权重差计费，没有计入两次调仓之间权重随价格漂移的再平衡成本（全部策略同一口径）。",
            "- 融资数据时点：交易所在 t 日收盘后（通常当晚或次日早间）发布 t 日融资余额，规则只用 t−1；若某月末前一日的个股融资文件缺失，该月该股不打分，不会用别的日期顶替。",
            "- 股东户数时点：只用东方财富记录的公告日之后的信息；公告日若晚于真实披露（例如互动平台答复）只会让信号更晚、更保守，若早于真实披露则有前视，上线前需抽查。",
            "- 个股策略的持仓只有 5 只：逐只剔除表显示单一名字能明显改变结果，实盘至少要把单票上限、行业上限写进风控，并接受与回测不同的持仓。",
            "- 低波动组合（RiskParity_SBG）的回撤阈值不能沿用股票口径：历史最大回撤约 4%，按 1.25 倍规则在回撤 5% 时就应减半复核。",
            "- 现金：回测现金收益为 0。实盘空仓可放货币 ETF，但这部分收益不属于策略 alpha。",
            "",
            *hold_lines,
        ]
    )

    cn10 = bond.get("cn_10y")
    risk = "\n".join(
        [
            "- Stock_MomLowVol 依赖「动量与低波动不同时失效」。当前持仓集中在银行、电力、家电、电信等防御名字，与 Stock_LowVol 高度重叠。"
            "失效条件：高 beta 急涨（2024-09 那种）时同时跑输两个因子，或样本外滚动 12 个月相对 EW_Stocks 超额持续为负。",
            (
                f"- RiskParity_SBG 是久期押注。中国 10Y 国债收益率当前约 {cn10:.2f}%，"
                if cn10 is not None
                else "- RiskParity_SBG 是久期押注。"
            )
            + "继续下行空间有限；若收益率上行 1 个百分点，十年国债ETF 价格大约下跌 8–9%（按约 8.5 年久期估算，属解释而非回测）。"
            "股债同时下跌时，按波动倒数配权无法分散。失效条件：国债ETF 滚动 12 个月收益转负且股票与国债相关性转正。",
            "- 低波动（Stock_LowVol、Sector_LowVol）当前持仓集中在银行、电力、红利，组合层面并不分散。失效条件：防御板块估值修复结束后的风格反转（如 2024-09 那种高 beta 急涨）。",
            *RISK_INTERP,
            "- 已淘汰的候选（10-19 至 10-26 期：WeeklyRev、NearHigh、VolMgd、Holiday_Post5、AbnTurnover、OvernightRev、LowMAX、Gold_TSMOM、Stock_Quality、ERP_Timing、"
            "Sector_DualMom、Stock_CS_Mom、Stock_EarnMom、Stock_PairMR、Bond_TSMOM、Stock_Value、Stock_DivMonth、Stock_Seasonal、US_LeadLag、VRP_Timing）不进入观察名单；不要换参数复活。",
            "- 幸存者偏差：股票池和 ETF 池都是 2026 年仍在交易的主流工具，没有加入已清盘 ETF 和退市股。结果对「当时可投资宇宙」偏乐观。",
            "- 制度变化：印花税、涨跌停、T+1、行业 ETF 扩容都会改变成本和收益结构。15bps 单边是常数近似；2023-08 印花税减半前的真实成本更高。",
            "- 数据：前复权在大比例分红除权日可能残留跳变，且新浪会回溯修订前复权历史；ETF 分红依赖新浪披露，部分 ETF 没有记录（见数据节）。",
            "- 宏观打印若与国债期货或财政部曲线不一致，以官方曲线为准，本报告不据此交易利率。",
        ]
    )

    self_check = "\n".join(
        [
            "- 缺了什么：深交所个股融资明细（接口在本环境 TLS 握手被重置，7 只深市股票没有个股融资数据）、行业资金流、北向净买入明细、一致预期修正、时点正确的净资产序列、货币基金 2020 年起的完整收益序列、紫金矿业的股东户数。这些缺失被记为缺口，没有用估计值填上。",
            "- 假设里较弱的一环：现金收益为 0、成本为常数 bps、ETF 收盘可成交、股票池有幸存者偏差、部分 ETF 分红缺失、调仓之间的权重漂移未计费、股东户数依赖东方财富记录的公告日、融资余额的发布时点按 T+1 处理、Stock_MarginCrowd 只在 17 只上交所股票里选。",
            *([REPRO_SELF_CHECK] if REPRO_SELF_CHECK else []),
            "- 制度覆盖：2020Q1、2022 熊市、2024 年 9 月以来的上涨、2026 年初至最近收盘都单独列了收益。样本内止于 2023，样本外从 2024 起。",
            *SELF_CHECK_INTERP,
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
        f"- 实际入选行业 ETF {len(sector_codes)} 只，防御 ETF {len(defensive_codes)} 只（只用于市场状态表），股票 {len(stock_frames)} 只。"
    )
    gap_lines.extend(FUNDAMENTAL_NOTES)
    gap_lines.append(
        f"- 基本面覆盖：估值（原始收盘价，用于股息率）{len(val_frames)}/{len(stock_frames)} 只、分红 {len(stock_divs)}/{len(stock_frames)} 只、股东户数 {len(holder_frames)}/{len(stock_frames)} 只、"
        f"个股融资余额 {len(sse_codes)}/{len(stock_frames)} 只（仅上交所）；"
        f"评估期月末有股息率的股票-月 {int(divy_panel.reindex(me_eval).notna().sum().sum())} 个、"
        f"有股东户数得分的 {int(holder_panel(1).reindex(me_eval).notna().sum().sum())} 个、"
        f"有残差动量得分的 {int(resmom().reindex(me_eval).notna().sum().sum())} 个（上限 {len(me_eval) * len(stock_close.columns)}）、"
        f"有融资拥挤得分的 {int(crowd(3).reindex(me_eval).notna().sum().sum())} 个（上限 {len(me_eval) * len(sse_codes)}）。"
    )
    gap_lines.append(
        f"- 两市融资余额：AkShare `macro_china_market_margin_sh` / `macro_china_market_margin_sz`（交易所口径，单位元），最后一个交易日 {mech['margin_last']['balance_date']}；"
        "任一交易所缺当日数据时合计记为缺失，最多沿用 5 个交易日，不插值。"
    )
    gap_lines.append(
        f"- 个股融资余额：上交所 `stock_margin_detail_sse`，只下载每个月末前一交易日的文件（{len(stk_margin)} 个日期）；"
        + (f"下载失败的日期：{'; '.join(stk_margin_gaps)}。" if stk_margin_gaps else "全部日期下载成功。")
        + "深交所 `stock_margin_detail_szse` 在本环境 TLS 握手被对端重置（requests 与 curl_cffi 浏览器指纹均失败），"
        "因此五粮液、美的集团、格力电器、比亚迪、宁德时代、立讯精密、东方财富 7 只深市股票不进入 Stock_MarginCrowd，规则对照同 17 只的 EW_SSE；没有用其他来源补数。"
        "中国海油(600938)、中国移动(600941) 成为融资标的之前没有记录，记为缺失。"
    )
    gap_lines.append(
        f"- 国债收益率曲线：AkShare `bond_zh_us_rate`（东方财富转载中债收益率），2Y 与 10Y，最后有效值 {curve.dropna().index.max().date()}；缺失日最多沿用 5 个交易日。"
        "未与中债官网逐日核对。"
    )
    gap_lines.append(
        "- 股东户数：东方财富 `stock_zh_a_gdhs_detail_em`，用「股东户数公告日期」做可见时点；没有公告日期的记录丢弃。"
        "截止日与公告日之间平均有 2–4 周延迟，部分记录来自互动平台而非定期报告。"
    )
    gap_lines.append("- QDII（例如纳指 ETF）未纳入，因为溢价和申购限额会让收盘价成交假设失效。")
    gap_lines.append(
        "- 价格源：ETF 为新浪未复权（`fund_etf_hist_sina`）+ 新浪分红（`fund_etf_dividend_sina`），股票为新浪前复权（`stock_zh_a_daily`），"
        "个股原始收盘价为东方财富 `stock_value_em`，个股分红为新浪 `stock_history_dividend_detail`。"
    )
    with_div = [c for c, k in div_events.items() if k > 0]
    no_div = [c for c, k in div_events.items() if k == 0]
    top_div = sorted(with_div, key=lambda c: -div_total[c])[:5]
    gap_lines.append(
        f"- ETF 分红：评估期内有除息记录的 {len(with_div)} 只，累计加回收益率最高的是 "
        + "、".join(f"{etf_meta.get(c, c)} {div_total[c]:.1%}" for c in top_div)
        + f"。没有记录的 {len(no_div)} 只（"
        + "、".join(etf_meta.get(c, c) for c in no_div)
        + "）保持价格收益；其中银行ETF 等若实际有分红则被低估，没有估计填补。"
    )
    if div_notes or div_fail:
        gap_lines.append("- 分红异常/失败：" + "; ".join(div_notes + div_fail))
    gap_lines.append(
        "- 货币ETF(511880) 的新浪分红记录从 2021 年才开始，不能作为 2020 年起的现金收益序列，因此改用 3M Shibor 作为无风险利率；回测里的现金仓位仍按 0 计。"
    )
    gap_lines.append(
        f"- 股票池等权 EW_Stocks 全样本 CAGR {fmt_pct(stats['EW_Stocks']['full']['cagr'])}，"
        f"Sharpe {fmt_num(stats['EW_Stocks']['full']['sharpe'])}，"
        f"最大回撤 {fmt_pct(stats['EW_Stocks']['full']['max_dd'])}。个股策略必须对照这一行，而不是只对照沪深300。"
    )

    bh_f = stats["BH_CSI300"]["full"]
    csi_row = next((r for r in index_rows if r["symbol"] == "sh000300"), None)
    cyb_row = next((r for r in index_rows if r["symbol"] == "sz399006"), None)
    day_txt = (
        f"最后一个交易日沪深300 {fmt_pct(csi_row['d1'])}、创业板指 {fmt_pct(cyb_row['d1'])}，"
        f"沪深300 距 252 日高点 {fmt_pct(csi_row['dd252'])}。"
        if csi_row and cyb_row
        else ""
    )
    n_cand = len([n for n in order if n not in benchmarks])
    exec_bits = [
        f"最后完整交易日 {last_date.date()}（{ASOF_REQUESTED} 当日盘中数据不使用），规则化市场状态为 **{regime}**（行业 ETF 在 MA60 之上的比例 {above60:.0%}，沪深300ETF 近 120 日 {fmt_pct(csi_ret120)}）。{day_txt}"
        f"近 60 日行业 ETF 领涨：{lead_txt}；领跌：{lag_txt}。",
        f"基准 BH_CSI300（含分红）在 2020-01-02 至 {last_date.date()} 扣费后 CAGR {fmt_pct(bh_f['cagr'])}，Sharpe {fmt_num(bh_f['sharpe'])}，最大回撤 {fmt_pct(bh_f['max_dd'])}。",
        f"{n_cand} 个候选（{n_cand - len(new_cands)} 个保留 + {len(new_cands)} 个新增），严格质量门槛下合格数量：**{len(survivors)}**。"
        + ("名单：" + "、".join(survivors) + "。" if survivors else "没有任何候选同时满足扣费跑赢沪深300（全样本与样本外）、Sharpe-rf>1、回撤<30%、样本外 Sharpe>0.5、跑赢自身宇宙基准。"),
        f"{len(new_cands)} 个新增候选（扣费后）："
        + "；".join(
            f"{n} CAGR {fmt_pct(stats[n]['full']['cagr'])}、Sharpe-rf {sc(n)}、回撤 {fmt_pct(stats[n]['full']['max_dd'])}、样本外 Sharpe {fmt_num(stats[n]['oos']['sharpe'])}、"
            f"相对 {stats[n]['universe_bench']} {fmt_pct(stats[n]['excess_vs_universe_cagr'])}/年"
            for n in new_cands
        )
        + "。",
        f"保留的观察名单：Stock_MomLowVol 扣费 CAGR {fmt_pct(stats['Stock_MomLowVol']['full']['cagr'])}，Sharpe-rf {sc('Stock_MomLowVol')}，"
        f"样本外 Sharpe {fmt_num(stats['Stock_MomLowVol']['oos']['sharpe'])}，相对股票池等权 {fmt_pct(stats['Stock_MomLowVol']['excess_vs_universe_cagr'])}/年；"
        f"Stock_ResMom 相对股票池等权 {fmt_pct(stats['Stock_ResMom']['excess_vs_universe_cagr'])}/年，2026 年至今 {fmt_pct(stats['Stock_ResMom']['stress']['ytd_2026'])}；"
        f"RiskParity_SBG Sharpe-rf {sc('RiskParity_SBG')}，回撤 {fmt_pct(stats['RiskParity_SBG']['full']['max_dd'])}，但平均 {rp_bond:.0%} 仓位是十年国债，本质是久期 beta。",
        *EXEC_INTERP,
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
        "loo_table": loo_table,
        "loo_note": (
            "解读：如果「LOO excess>0」远少于总次数，说明相对股票池等权的超额不是规则带来的，去掉一只股票就会改变结论；"
            "LOO CAGR 的最小/最大值标出了对结果影响最大的名字。"
        ),
        "overlay_table": overlay_table,
        "cost_table": cost_table,
        "rf_note": rf_note,
        "ranking_section": "\n".join(qual_lines + [""] + rank_lines),
        "recommend_section": "\n".join(rec_bits),
        "deploy_section": deploy,
        "risk_section": risk,
        "self_check": self_check,
        "data_section": "\n".join(gap_lines),
        "chart_note": chart_note,
        "overlap_table": overlap_table,
        "overlap_note": OVERLAP_NOTE,
        "repro_note": REPRO_NOTE,
        "yearly_table": yearly_table,
        "mech_table": mech_table,
    }
    report = build_report(ctx)
    report_path = ROOT / "reports" / REPORT_NAME
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
