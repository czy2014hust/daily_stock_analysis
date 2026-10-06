#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026-10-06 A-share ETF strategy research backtest.

Rules are fixed before looking at performance. Prices come from Eastmoney
forward-adjusted daily K-lines (the same public endpoint the repo's
EfinanceFetcher uses). Macro context comes from Yahoo Finance chart API.

Execution model
---------------
Price-based signals use the close of day t-1 and trade at the next open.
Overnight (prior close -> open) is earned by the position already held.
Intraday (open -> close) is earned by the position after the open trade.
One-way costs scale with the sum of absolute weight changes.

This file is research code for the daily strategy report. It is not a
production trading system.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

AS_OF = "2026-10-06"
BEG = "20140101"
END = "20261006"
EVAL_START = pd.Timestamp("2018-01-01")
OOS_START = pd.Timestamp("2022-01-01")
TRADING_DAYS = 252
RF_ANNUAL = 0.015  # assumption: constant 1.5% cash rate, used only for excess Sharpe

# One-way cost in bps. Commission ~1.5 bp; the rest is half-spread + slippage.
# ETFs are exempt from A-share stamp duty. Costs are charged on traded notional.
LIQUID_COST_BPS = 5.0
SECTOR_COST_BPS = 10.0

INDEX_ETFS = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "510050": "上证50ETF",
    "159915": "创业板ETF",
    "512100": "中证1000ETF",
}
DEFENSIVE_ETFS = {
    "511010": "国债ETF",
    "518880": "黄金ETF",
    "513100": "纳指ETF",
}
SECTOR_ETFS = {
    "512010": "医药ETF",
    "159928": "消费ETF",
    "512880": "证券ETF",
    "512800": "银行ETF",
    "512660": "军工ETF",
    "512400": "有色ETF",
    "512200": "房地产ETF",
    "512690": "酒ETF",
    "515220": "煤炭ETF",
    "512480": "半导体ETF",
    "512170": "医疗ETF",
    "510880": "红利ETF",
}
INDEX_SECIDS = {
    "000001": ("1.000001", "上证指数"),
    "000300": ("1.000300", "沪深300"),
    "000905": ("1.000905", "中证500"),
    "000852": ("1.000852", "中证1000"),
    "399006": ("0.399006", "创业板指"),
    "000016": ("1.000016", "上证50"),
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; DSA-research/1.0)"}


CACHE_DIR = "/tmp/strategy_research/cache"
QQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}


def qq_symbol(code: str) -> str:
    """Tencent symbol. Shenzhen ETFs and indices use the sz prefix."""
    if code.startswith(("15", "16", "18", "399")):
        return f"sz{code}"
    return f"sh{code}"


def http_json(url: str, timeout: int = 30, attempts: int = 3, headers: Optional[dict] = None) -> dict:
    last_error: Optional[Exception] = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=headers or UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001 - research fetch, retry then record
            last_error = exc
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"GET failed: {url} ({last_error})")


def _parse_qq_bar(bar: list) -> dict:
    """Parse a Tencent K-line row.

    Layout is date, open, close, high, low, volume (lots), optional extras.
    When present, field index 8 is turnover in 万元.
    """
    amount = float("nan")
    if len(bar) > 8:
        try:
            amount = float(bar[8]) * 10000.0
        except (TypeError, ValueError):
            amount = float("nan")
    close = float(bar[2])
    volume_lots = float(bar[5])
    if math.isnan(amount):
        amount = volume_lots * 100.0 * close
    return {
        "date": bar[0],
        "open": float(bar[1]),
        "close": close,
        "high": float(bar[3]),
        "low": float(bar[4]),
        "volume": volume_lots * 100.0,
        "amount": amount,
    }


def fetch_qq_kline(code: str, adjust: str = "qfq") -> Tuple[pd.DataFrame, str]:
    """Forward-adjusted daily bars from Tencent, paged backward in 640-row windows.

    Overlapping pages were checked to share the same adjusted prices, so the
    stitch does not rebase each window. Eastmoney was used as a cross-check
    for the latest 510300 print, then stopped answering.
    """
    symbol = qq_symbol(code)
    collected: Dict[str, dict] = {}
    end = "2026-10-06"
    field_used = "missing"
    for _ in range(12):
        url = (
            "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?"
            f"param={symbol},day,,{end},640,{adjust}"
        )
        payload = http_json(url, headers=QQ_HEADERS)
        data = (payload or {}).get("data") or {}
        inner = data.get(symbol) or {}
        rows = inner.get("qfqday") or inner.get("day") or []
        if inner.get("qfqday"):
            field_used = "qfqday"
        elif rows and field_used == "missing":
            field_used = "day"
        if not rows:
            break
        new_dates = 0
        for bar in rows:
            if not bar or bar[0] in collected:
                continue
            collected[bar[0]] = _parse_qq_bar(bar)
            new_dates += 1
        oldest = str(rows[0][0])
        if new_dates == 0 or oldest <= "2014-01-01":
            break
        end = (pd.Timestamp(oldest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.12)
    frame = pd.DataFrame(collected.values())
    if frame.empty:
        return frame, field_used
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates("date").sort_values("date").set_index("date")
    return frame, field_used


def load_cached_frame(code: str) -> Optional[pd.DataFrame]:
    path = f"{CACHE_DIR}/{code}.csv"
    try:
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    except FileNotFoundError:
        return None
    return frame.sort_index()


def fetch_yahoo(symbol: str, range_: str = "10y") -> pd.DataFrame:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?interval=1d&range={range_}"
    )
    payload = http_json(url)
    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(stamps, unit="s").tz_localize("UTC").tz_convert("Asia/Shanghai").tz_localize(None).normalize(),
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna(subset=["close"])


def load_price_frame(code: str, name: str, kind: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    import os

    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = load_cached_frame(code)
    if cached is not None and not cached.empty:
        print(
            f"{kind} {code} {name}: cache {cached.index.min().date()} -> {cached.index.max().date()} n={len(cached)}",
            flush=True,
        )
        return cached, None
    try:
        frame, field = fetch_qq_kline(code)
    except Exception as exc:  # noqa: BLE001
        return None, f"{code} {name}: 拉取失败 {exc}"
    if frame.empty or len(frame) < 60:
        return None, f"{code} {name}: 历史不足 ({0 if frame.empty else len(frame)} 行), field={field}"
    frame.to_csv(f"{CACHE_DIR}/{code}.csv", index_label="date")
    with open(f"{CACHE_DIR}/{code}.meta", "w", encoding="utf-8") as handle:
        handle.write(field)
    print(
        f"{kind} {code} {name}: {frame.index.min().date()} -> {frame.index.max().date()} "
        f"n={len(frame)} last={frame['close'].iloc[-1]:.4f} field={field} "
        f"med_amt_60={frame['amount'].tail(60).median():.0f}",
        flush=True,
    )
    return frame, None


def load_universe() -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], List[str]]:
    frames: Dict[str, pd.DataFrame] = {}
    names: Dict[str, str] = {}
    gaps: List[str] = []
    catalog = {}
    catalog.update(INDEX_ETFS)
    catalog.update(DEFENSIVE_ETFS)
    catalog.update(SECTOR_ETFS)
    for code, name in catalog.items():
        frame, gap = load_price_frame(code, name, "ETF")
        if gap:
            gaps.append(gap)
            print("GAP", gap, flush=True)
            continue
        frames[code] = frame
        names[code] = name
    return frames, names, gaps


def load_indices() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    frames: Dict[str, pd.DataFrame] = {}
    gaps: List[str] = []
    for code, (_secid, name) in INDEX_SECIDS.items():
        frame, gap = load_price_frame(code, name, "IDX")
        if gap or frame is None:
            gaps.append(gap or f"指数 {code} {name}: 空数据")
            continue
        frame.attrs["name"] = name
        frames[code] = frame
    return frames, gaps


def load_macro() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    symbols = {
        "^GSPC": "标普500",
        "^HSI": "恒生指数",
        "^TNX": "美国10年国债收益率",
        "^VIX": "VIX",
        "GC=F": "COMEX黄金",
        "CNY=X": "美元兑人民币",
    }
    frames: Dict[str, pd.DataFrame] = {}
    gaps: List[str] = []
    for symbol, name in symbols.items():
        try:
            frame = fetch_yahoo(symbol, "5y")
        except Exception as exc:  # noqa: BLE001
            gaps.append(f"宏观 {symbol} {name}: {exc}")
            continue
        frame.attrs["name"] = name
        frames[symbol] = frame
        print(
            f"MACRO {symbol} {name}: {frame.index.min().date()} -> {frame.index.max().date()} "
            f"last={frame['close'].iloc[-1]:.4f}",
            flush=True,
        )
    return frames, gaps


def fetch_spot_context() -> Dict[str, object]:
    """Best-effort valuation / northbound snapshot. Missing keys stay absent."""
    context: Dict[str, object] = {}
    gaps: List[str] = []
    try:
        url = (
            "https://push2.eastmoney.com/api/qt/stock/get?invt=2&fltt=2"
            "&secid=1.000300&fields=f43,f57,f58,f162,f163,f167,f168,f169,f170,f171"
        )
        payload = http_json(url)
        context["csi300_spot"] = (payload or {}).get("data")
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"沪深300现货估值接口失败: {exc}")
    try:
        url = (
            "https://push2.eastmoney.com/api/qt/kamt.rtmin/get?fields1=f1,f2,f3,f4"
            "&fields2=f51,f52,f53,f54,f55,f56"
        )
        payload = http_json(url)
        data = (payload or {}).get("data")
        context["northbound_keys"] = list(data.keys()) if isinstance(data, dict) else str(type(data))
        context["northbound"] = data
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"北向资金接口失败: {exc}")
    try:
        url = (
            "https://datacenter-web.eastmoney.com/api/data/v1/get?"
            "reportName=RPT_INDEX_TS_VALUE&columns=ALL&pageNumber=1&pageSize=5"
            "&sortColumns=TRADE_DATE&sortTypes=-1"
            "&filter=(SECURITY_CODE%3D%22000300%22)"
        )
        payload = http_json(url)
        context["valuation_report"] = payload.get("result") if isinstance(payload, dict) else None
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"历史估值接口失败: {exc}")
    context["gaps"] = gaps
    return context


def build_panels(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align to the 沪深300ETF calendar and forward-fill only short gaps."""
    master = frames["510300"].index
    panels = {}
    for field in ("open", "close", "amount"):
        panel = pd.DataFrame({code: df[field] for code, df in frames.items()})
        panel = panel.reindex(master)
        # Short gaps (holiday mismatch / missing print) only. Do not fill a listing backward.
        panel = panel.ffill(limit=3)
        panels[field] = panel
    return panels


@dataclass
class RunResult:
    name: str
    title: str
    thesis: str
    returns: pd.Series
    open_target: pd.DataFrame
    cost_paid: pd.Series


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    stamps = pd.Series(index, index=index)
    grouped = stamps.groupby([index.year, index.month]).max()
    return pd.DatetimeIndex(grouped.to_list())


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    iso = index.isocalendar()
    stamps = pd.Series(index, index=index)
    grouped = stamps.groupby([iso.year.to_numpy(), iso.week.to_numpy()]).max()
    return pd.DatetimeIndex(grouped.to_list())


def trading_day_every(index: pd.DatetimeIndex, step: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index[::step])


def _empty_signal(index: pd.DatetimeIndex, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=list(columns))


def _ffill_signal(signal: pd.DataFrame) -> pd.DataFrame:
    return signal.ffill().fillna(0.0)


def strategy_dual_momentum(close: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Antonacci-style dual momentum. 100% in the best 12-month index ETF if positive, else bonds."""
    universe = [c for c in INDEX_ETFS if c in close.columns]
    defensive = "511010" if "511010" in close.columns else None
    score = close[universe].pct_change(lookback)
    eligible = close[universe].notna() & close[universe].shift(lookback).notna()
    signal_dates = month_ends(close.index)
    signal = _empty_signal(close.index, list(universe) + ([defensive] if defensive else []))
    for dt in signal_dates:
        if dt not in score.index:
            continue
        row = score.loc[dt].where(eligible.loc[dt])
        signal.loc[dt] = 0.0
        positive = row[row > 0].dropna().sort_values(ascending=False)
        if not positive.empty:
            signal.loc[dt, positive.index[0]] = 1.0
        elif defensive is not None and pd.notna(close.at[dt, defensive]):
            signal.loc[dt, defensive] = 1.0
    return _ffill_signal(signal)


def strategy_sector_momentum(
    close: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
    top_n: int = 3,
    amount: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Cross-sectional industry momentum. Skip the most recent month to reduce short-term reversal."""
    universe = [c for c in SECTOR_ETFS if c in close.columns and c != "510880"]
    window = lookback + skip
    # close[t-skip] / close[t-skip-lookback] - 1, so the last month is excluded.
    score = close[universe].shift(skip) / close[universe].shift(window) - 1
    eligible = close[universe].shift(window).notna() & close[universe].shift(skip).notna()
    if amount is not None:
        median_amt = amount[universe].rolling(60, min_periods=20).median()
        eligible = eligible & (median_amt >= 3.0e7)
    signal_dates = month_ends(close.index)
    signal = _empty_signal(close.index, universe)
    for dt in signal_dates:
        row = score.loc[dt].where(eligible.loc[dt]).dropna().sort_values(ascending=False)
        signal.loc[dt] = 0.0
        picked = list(row.index[:top_n])
        if picked:
            for code in picked:
                signal.loc[dt, code] = 1.0 / len(picked)
    return _ffill_signal(signal)


def strategy_short_reversal(close: pd.DataFrame, lookback: int = 5, hold: int = 5) -> pd.DataFrame:
    """Buy the liquid index ETF with the worst trailing return and hold for `hold` sessions."""
    universe = [c for c in ("510300", "510500", "510050", "159915", "512100") if c in close.columns]
    score = close[universe].pct_change(lookback)
    eligible = close[universe].shift(lookback).notna()
    signal_dates = trading_day_every(close.index, hold)
    signal = _empty_signal(close.index, universe)
    for dt in signal_dates:
        row = score.loc[dt].where(eligible.loc[dt]).dropna().sort_values(ascending=True)
        signal.loc[dt] = 0.0
        if not row.empty:
            signal.loc[dt, row.index[0]] = 1.0
    return _ffill_signal(signal)


def strategy_trend_vol_target(
    close: pd.DataFrame,
    sma_window: int = 200,
    vol_window: int = 20,
    target_vol: float = 0.15,
    band: float = 0.10,
) -> pd.DataFrame:
    """Long CSI300 ETF only above its 200-day average. Scale exposure to target volatility. Residual in treasury ETF."""
    asset = "510300"
    defensive = "511010" if "511010" in close.columns else None
    sma = close[asset].rolling(sma_window, min_periods=sma_window).mean()
    daily = close[asset].pct_change()
    realized = daily.rolling(vol_window, min_periods=vol_window).std() * math.sqrt(TRADING_DAYS)
    risk_on = close[asset] > sma
    raw_weight = (target_vol / realized.replace(0, np.nan)).clip(upper=1.0)
    equity_w = raw_weight.where(risk_on, 0.0).fillna(0.0)
    columns = [asset] + ([defensive] if defensive else [])
    signal = pd.DataFrame(0.0, index=close.index, columns=columns)
    signal[asset] = equity_w
    if defensive is not None:
        bond_ok = close[defensive].notna()
        signal[defensive] = (1.0 - equity_w).where(bond_ok, 0.0)
    # No position until the moving average exists.
    signal.loc[sma.isna(), :] = 0.0
    # 10% weight band plus immediate flips. Daily vol-targeting without a band
    # trades noise; the band is part of the pre-committed spec, not a fitted knob.
    if band <= 0:
        return signal
    return band_rebalance(signal, band=band)


def strategy_low_vol(
    close: pd.DataFrame,
    vol_window: int = 60,
    top_n: int = 3,
    amount: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Equal-weight the three lowest 60-day realized-vol sector ETFs."""
    universe = [c for c in SECTOR_ETFS if c in close.columns]
    realized = close[universe].pct_change().rolling(vol_window, min_periods=vol_window).std()
    eligible = realized.notna()
    if amount is not None:
        median_amt = amount[universe].rolling(60, min_periods=20).median()
        eligible = eligible & (median_amt >= 3.0e7)
    signal_dates = month_ends(close.index)
    signal = _empty_signal(close.index, universe)
    for dt in signal_dates:
        row = realized.loc[dt].where(eligible.loc[dt]).dropna().sort_values(ascending=True)
        signal.loc[dt] = 0.0
        picked = list(row.index[:top_n])
        if picked:
            for code in picked:
                signal.loc[dt, code] = 1.0 / len(picked)
    return _ffill_signal(signal)


def strategy_size_reversion(
    close: pd.DataFrame,
    window: int = 60,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Long-only rotation between CSI1000 and CSI300 when their log-price ratio is stretched."""
    small, large = "512100", "510300"
    signal = _empty_signal(close.index, [small, large])
    if small not in close.columns or large not in close.columns:
        return signal.fillna(0.0)
    ratio = np.log(close[small] / close[large])
    z = (ratio - ratio.rolling(window, min_periods=window).mean()) / ratio.rolling(window, min_periods=window).std()
    signal_dates = week_ends(close.index)
    for dt in signal_dates:
        signal.loc[dt] = 0.0
        value = z.at[dt] if dt in z.index else np.nan
        if pd.isna(value):
            continue
        if value > threshold:
            signal.loc[dt, large] = 1.0
        elif value < -threshold:
            signal.loc[dt, small] = 1.0
        else:
            signal.loc[dt, large] = 0.5
            signal.loc[dt, small] = 0.5
    return _ffill_signal(signal)


def strategy_equity_gold(close: pd.DataFrame, sma_window: int = 200) -> pd.DataFrame:
    """Monthly binary switch: CSI300 ETF above its 200-day average, otherwise gold ETF."""
    asset, gold = "510300", "518880"
    sma = close[asset].rolling(sma_window, min_periods=sma_window).mean()
    signal_dates = month_ends(close.index)
    signal = _empty_signal(close.index, [asset, gold])
    for dt in signal_dates:
        signal.loc[dt] = 0.0
        if pd.isna(sma.at[dt]) or pd.isna(close.at[dt, gold]):
            continue
        if close.at[dt, asset] > sma.at[dt]:
            signal.loc[dt, asset] = 1.0
        else:
            signal.loc[dt, gold] = 1.0
    return _ffill_signal(signal)


def strategy_turn_of_month(index: pd.DatetimeIndex, asset: str = "510300") -> pd.DataFrame:
    """Long the last 2 and first 3 sessions of each month. Calendar is known before the open."""
    groups: Dict[Tuple[int, int], List[pd.Timestamp]] = {}
    for dt in index:
        groups.setdefault((dt.year, dt.month), []).append(dt)
    keys = sorted(groups)
    tom = set()
    for key in keys:
        days = groups[key]
        tom.update(days[-2:])
        tom.update(days[:3])
    signal = pd.DataFrame(0.0, index=index, columns=[asset])
    for dt in tom:
        signal.at[dt, asset] = 1.0
    signal.attrs["calendar_known"] = True
    return signal


def strategy_cross_asset(close: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Monthly best 12-month asset among A-share beta, Nasdaq ETF, gold, and treasuries."""
    universe = [c for c in ("510300", "513100", "518880", "511010") if c in close.columns]
    score = close[universe].pct_change(lookback)
    eligible = close[universe].shift(lookback).notna()
    signal_dates = month_ends(close.index)
    signal = _empty_signal(close.index, universe)
    for dt in signal_dates:
        row = score.loc[dt].where(eligible.loc[dt]).dropna().sort_values(ascending=False)
        signal.loc[dt] = 0.0
        if row.empty:
            continue
        # Absolute momentum: if every risky asset is negative, allow the treasury ETF to win
        # only when its own 12-month return is the least-bad or positive. Always hold the top score.
        signal.loc[dt, row.index[0]] = 1.0
    return _ffill_signal(signal)


def band_rebalance(signal: pd.DataFrame, band: float = 0.10) -> pd.DataFrame:
    """Keep the last traded weight until month-end, a 0/1 flip, or a large weight change."""
    values = signal.fillna(0.0).to_numpy(dtype=float)
    kept = np.zeros_like(values)
    if len(signal) == 0:
        return signal.copy()
    month_end = set(month_ends(signal.index))
    current = values[0].copy()
    for i, dt in enumerate(signal.index):
        desired = values[i]
        flip = bool(((current > 1e-8) != (desired > 1e-8)).any())
        drift = float(np.abs(desired - current).max()) >= band if i else True
        if i == 0 or dt in month_end or flip or drift:
            current = desired.copy()
        kept[i] = current
    return pd.DataFrame(kept, index=signal.index, columns=signal.columns)


def strategy_buy_hold(index: pd.DatetimeIndex, asset: str = "510300") -> pd.DataFrame:
    signal = pd.DataFrame(0.0, index=index, columns=[asset])
    signal[asset] = 1.0
    return signal


def strategy_equal_weight_monthly(close: pd.DataFrame, universe: Sequence[str], amount: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    cols = [c for c in universe if c in close.columns]
    signal = _empty_signal(close.index, cols)
    dates = month_ends(close.index)
    for dt in dates:
        alive = [c for c in cols if pd.notna(close.at[dt, c])]
        if amount is not None:
            alive = [
                c
                for c in alive
                if pd.notna(amount.at[dt, c]) and amount[c].loc[:dt].tail(60).median() >= 3.0e7
            ]
        signal.loc[dt] = 0.0
        if alive:
            for code in alive:
                signal.loc[dt, code] = 1.0 / len(alive)
    return _ffill_signal(signal)


def run_backtest(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps: Dict[str, float],
    *,
    lag: int = 1,
    calendar_known: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Convert close-time signals into next-open positions and net returns.

    lag=1 is the base case for signals that use today's close.
    calendar_known=True means the desired position for today's session was known
    before the open (turn-of-month), so no extra lag is applied.
    """
    columns = [c for c in signal.columns if c in close.columns]
    sig = signal[columns].reindex(close.index).fillna(0.0)
    # Price signals are known at the prior close. Calendar signals are known
    # before the open, so lag=1 means "trade this open".
    open_target = sig if calendar_known else sig.shift(1)
    if lag > 1:
        open_target = open_target.shift(lag - 1)
    open_target = open_target.fillna(0.0)

    prev_close = close[columns].shift(1)
    overnight = (open_[columns] / prev_close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    intraday = (close[columns] / open_[columns] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ov = overnight.to_numpy(dtype=float)
    intra = intraday.to_numpy(dtype=float)
    tgt = open_target.to_numpy(dtype=float)
    bps_arr = np.array([cost_bps.get(c, SECTOR_COST_BPS) for c in columns], dtype=float) / 10000.0
    n_days, n_assets = tgt.shape
    weights = np.zeros(n_assets, dtype=float)
    rets = np.zeros(n_days, dtype=float)
    costs = np.zeros(n_days, dtype=float)
    for i in range(n_days):
        r_ov = ov[i]
        port_ov = float(np.sum(weights * r_ov))
        growth_ov = 1.0 + port_ov
        if growth_ov <= 0.0:
            rets[i] = -1.0
            weights[:] = 0.0
            continue
        # Weights drift with the overnight gap. Cash earns zero and is the residual.
        w_drift = weights * (1.0 + r_ov) / growth_ov
        w_tgt = tgt[i]
        cost_rate = float(np.sum(np.abs(w_tgt - w_drift) * bps_arr))
        r_in = intra[i]
        port_in = float(np.sum(w_tgt * r_in))
        growth_in = 1.0 + port_in
        rets[i] = growth_ov * (1.0 - cost_rate) * growth_in - 1.0
        costs[i] = growth_ov * cost_rate
        if growth_in <= 0.0:
            weights[:] = 0.0
        else:
            weights = w_tgt * (1.0 + r_in) / growth_in
    net = pd.Series(rets, index=close.index, name="net")
    cost = pd.Series(costs, index=close.index, name="cost")
    return net, cost, open_target


def performance(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {}
    equity = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else float("nan")
    excess = r - RF_ANNUAL / TRADING_DAYS
    sharpe_rf = float(excess.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else float("nan")
    downside = r.clip(upper=0.0)
    down_dev = float(np.sqrt((downside ** 2).mean()))
    sortino = float(r.mean() / down_dev * math.sqrt(TRADING_DAYS)) if down_dev > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")
    monthly = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    win_rate = float((monthly > 0).mean()) if len(monthly) else float("nan")
    return {
        "days": int(len(r)),
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sharpe_rf": sharpe_rf,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "monthly_win_rate": win_rate,
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }


def slice_period(returns: pd.Series, start: pd.Timestamp, end: Optional[pd.Timestamp] = None) -> pd.Series:
    data = returns.loc[returns.index >= start]
    if end is not None:
        data = data.loc[data.index <= end]
    return data


def period_return(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((1.0 + returns).prod() - 1.0)


def information_ratio(strategy: pd.Series, benchmark: pd.Series) -> float:
    both = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    if both.empty:
        return float("nan")
    diff = both.iloc[:, 0] - both.iloc[:, 1]
    if diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1) * math.sqrt(TRADING_DAYS))


def exposure_stats(open_target: pd.DataFrame, start: pd.Timestamp) -> Dict[str, float]:
    held = open_target.loc[open_target.index >= start]
    if held.empty:
        return {}
    gross = held.sum(axis=1)
    out = {
        "avg_gross": float(gross.mean()),
        "pct_invested": float((gross > 0.05).mean()),
    }
    for col in held.columns:
        out[f"avg_w_{col}"] = float(held[col].mean())
    return out


def trade_win_rate(returns: pd.Series, open_target: pd.DataFrame) -> float:
    """Win rate of contiguous holding blocks, measured on net daily returns."""
    key = open_target.round(6).astype(str).agg("|".join, axis=1)
    changed = key.ne(key.shift(1))
    block = changed.cumsum()
    grouped = returns.groupby(block).apply(lambda s: float((1.0 + s).prod() - 1.0))
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna()
    if grouped.empty:
        return float("nan")
    return float((grouped > 0).mean())


def rolling_sharpe_summary(returns: pd.Series, window: int = 252) -> Dict[str, float]:
    r = returns.dropna()
    if len(r) < window:
        return {}
    roll = r.rolling(window).apply(
        lambda x: x.mean() / x.std(ddof=1) * math.sqrt(TRADING_DAYS) if x.std(ddof=1) > 0 else np.nan,
        raw=False,
    ).dropna()
    if roll.empty:
        return {}
    return {
        "roll_sharpe_median": float(roll.median()),
        "roll_sharpe_p05": float(roll.quantile(0.05)),
        "roll_sharpe_min": float(roll.min()),
        "pct_roll_sharpe_positive": float((roll > 0).mean()),
    }


def yearly_returns(returns: pd.Series) -> Dict[str, float]:
    if returns.empty:
        return {}
    out = {}
    for year, chunk in returns.groupby(returns.index.year):
        out[str(int(year))] = period_return(chunk)
    return out


def cost_map(columns: Iterable[str], multiplier: float = 1.0) -> Dict[str, float]:
    liquid = set(INDEX_ETFS) | set(DEFENSIVE_ETFS) | {"510880"}
    return {
        code: (LIQUID_COST_BPS if code in liquid else SECTOR_COST_BPS) * multiplier
        for code in columns
    }


def assert_engine_sanity(close: pd.DataFrame, open_: pd.DataFrame) -> None:
    """Buy-and-hold from the second session should track close-to-close within costs and the open fill."""
    signal = strategy_buy_hold(close.index, "510300")
    net, cost, _ = run_backtest(close, open_, signal, {"510300": 0.0}, lag=1)
    # First return is zero because the first open has no prior close position.
    bh = close["510300"].pct_change().fillna(0.0)
    # After the position is on, net return equals close-to-close (zero cost).
    aligned = pd.concat([net.rename("net"), bh.rename("bh")], axis=1).dropna().iloc[2:]
    gap = (aligned["net"] - aligned["bh"]).abs().max()
    if gap > 1e-8:
        raise AssertionError(f"engine sanity failed, max gap {gap}")
    print(f"engine sanity ok, max zero-cost buyhold gap={gap:.2e}, sample cost sum={cost.sum():.6f}")


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "na"
    return f"{value * 100:.2f}%"


def main() -> None:
    frames, names, etf_gaps = load_universe()
    if "510300" not in frames:
        raise SystemExit("510300 missing; cannot run the study")
    indices, index_gaps = load_indices()
    macro, macro_gaps = load_macro()
    spot = fetch_spot_context()
    panels = build_panels(frames)
    close, open_, amount = panels["close"], panels["open"], panels["amount"]
    assert_engine_sanity(close, open_)

    jump = close.pct_change().abs()
    jump_flags = []
    for code in close.columns:
        hits = jump[code][jump[code] > 0.12]
        for dt, value in hits.items():
            jump_flags.append({"code": code, "date": str(dt.date()), "abs_ret": float(value)})

    bps = cost_map(close.columns, 1.0)
    bps_2x = cost_map(close.columns, 2.0)

    specs = []

    def add(key, title, thesis, signal, calendar=False):
        specs.append((key, title, thesis, signal, calendar))

    add("DM", "双动量（指数ETF）", "12个月相对动量叠加绝对动量，弱势时持有国债ETF。", strategy_dual_momentum(close))
    add("SM", "行业横截面动量", "行业ETF 6个月动量、跳过最近1个月，持有前3名。", strategy_sector_momentum(close, amount=amount))
    add("REV", "短期反转", "每5个交易日买入近5日最弱的宽基ETF。", strategy_short_reversal(close))
    add("TVT", "趋势+波动率目标", "沪深300站上200日均线才持有，仓位按15%目标波动率缩放，其余配置国债ETF。", strategy_trend_vol_target(close))
    add("LV", "低波行业", "每月持有60日实现波动率最低的3只行业ETF。", strategy_low_vol(close, amount=amount))
    add("PAIR", "大小盘比值均值回归", "中证1000/沪深300对数价格比的60日z值极端时换到相对便宜的一边。", strategy_size_reversion(close))
    add("EG", "股金切换", "沪深300月度检查200日均线，跌破则持有黄金ETF。", strategy_equity_gold(close))
    tom = strategy_turn_of_month(close.index)
    add("TOM", "月初月末效应", "持有沪深300ETF在每月最后2个和最初3个交易日，其余时间空仓。", tom, True)
    add("CAM", "跨资产动量", "沪深300、纳指ETF、黄金ETF、国债ETF中持有12个月收益最高者。", strategy_cross_asset(close))

    benchmarks = {
        "BH300": ("沪深300ETF买入持有", strategy_buy_hold(close.index)),
        "EWIDX": (
            "宽基等权月度再平衡",
            strategy_equal_weight_monthly(close, list(INDEX_ETFS)),
        ),
        "EWSEC": (
            "行业等权月度再平衡",
            strategy_equal_weight_monthly(close, [c for c in SECTOR_ETFS if c != "510880"], amount),
        ),
        "EW4": (
            "沪深300/纳指/黄金/国债等权",
            strategy_equal_weight_monthly(close, ["510300", "513100", "518880", "511010"]),
        ),
    }

    def evaluate(signal: pd.DataFrame, calendar: bool, costs: Dict[str, float], lag: int = 1) -> RunResult:
        net, cost, target = run_backtest(
            close, open_, signal, costs, lag=lag, calendar_known=calendar
        )
        return RunResult("", "", "", net, target, cost)

    results = {}
    for key, title, thesis, signal, calendar in specs:
        run = evaluate(signal, calendar, bps, lag=1)
        run.name = key
        run.title = title
        run.thesis = thesis
        results[key] = run
        print(f"ran {key}", flush=True)

    bench_runs = {}
    for key, (title, signal) in benchmarks.items():
        run = evaluate(signal, False, bps, lag=1)
        run.name = key
        run.title = title
        bench_runs[key] = run

    def pack(run: RunResult, costs_label: str = "1x") -> Dict[str, object]:
        window = slice_period(run.returns, EVAL_START)
        ins = slice_period(run.returns, EVAL_START, OOS_START - pd.Timedelta(days=1))
        oos = slice_period(run.returns, OOS_START)
        bh = slice_period(bench_runs["BH300"].returns, EVAL_START)
        stats = performance(window)
        stats_is = performance(ins)
        stats_oos = performance(oos)
        stats.update(
            {
                "is": stats_is,
                "oos": stats_oos,
                "yearly": yearly_returns(window),
                "ir_vs_bh300": information_ratio(window, bh.reindex(window.index)),
                "ir_oos_vs_bh300": information_ratio(oos, slice_period(bench_runs["BH300"].returns, OOS_START)),
                "oos_excess_total": period_return(oos) - period_return(slice_period(bench_runs["BH300"].returns, OOS_START)),
                "full_excess_total": period_return(window) - period_return(bh),
                "exposure": exposure_stats(run.open_target, EVAL_START),
                "trade_win_rate": trade_win_rate(window, run.open_target.loc[window.index]),
                "rolling": rolling_sharpe_summary(window),
                "avg_daily_cost": float(slice_period(run.cost_paid, EVAL_START).mean()),
                "total_cost_drag": float(slice_period(run.cost_paid, EVAL_START).sum()),
                "last_weights": {
                    code: float(weight)
                    for code, weight in run.open_target.iloc[-1].items()
                    if abs(float(weight)) > 1e-6
                },
                "last_date": str(run.returns.index.max().date()),
                "costs": costs_label,
            }
        )
        return stats

    summary = {key: pack(run) for key, run in results.items()}
    bench_summary = {key: pack(run) for key, run in bench_runs.items()}

    # Robustness: 2x cost, extra execution lag, and pre-committed parameter neighbors.
    robustness: Dict[str, Dict[str, Dict[str, float]]] = {}

    neighbors = {
        "DM": [
            ("lb189", strategy_dual_momentum(close, 189)),
            ("lb315", strategy_dual_momentum(close, 315)),
        ],
        "SM": [
            ("lb84_top3", strategy_sector_momentum(close, 84, 21, 3, amount)),
            ("lb168_top3", strategy_sector_momentum(close, 168, 21, 3, amount)),
            ("lb126_top2", strategy_sector_momentum(close, 126, 21, 2, amount)),
            ("lb126_top4", strategy_sector_momentum(close, 126, 21, 4, amount)),
        ],
        "REV": [
            ("lb3_h3", strategy_short_reversal(close, 3, 3)),
            ("lb10_h10", strategy_short_reversal(close, 10, 10)),
        ],
        "TVT": [
            ("sma150_vol10", strategy_trend_vol_target(close, 150, 20, 0.10)),
            ("sma250_vol20", strategy_trend_vol_target(close, 250, 20, 0.20)),
            ("band0_daily", strategy_trend_vol_target(close, 200, 20, 0.15, 0.0)),
            ("band20", strategy_trend_vol_target(close, 200, 20, 0.15, 0.20)),
        ],
        "LV": [
            ("vol40", strategy_low_vol(close, 40, 3, amount)),
            ("vol120", strategy_low_vol(close, 120, 3, amount)),
        ],
        "PAIR": [
            ("z05_w40", strategy_size_reversion(close, 40, 0.5)),
            ("z15_w120", strategy_size_reversion(close, 120, 1.5)),
        ],
        "EG": [
            ("sma150", strategy_equity_gold(close, 150)),
            ("sma250", strategy_equity_gold(close, 250)),
        ],
        "TOM": [],
        "CAM": [
            ("lb189", strategy_cross_asset(close, 189)),
            ("lb315", strategy_cross_asset(close, 315)),
        ],
    }

    for key, title, thesis, signal, calendar in specs:
        bucket: Dict[str, Dict[str, float]] = {}
        stress = evaluate(signal, calendar, bps_2x, lag=1)
        oos = performance(slice_period(stress.returns, OOS_START))
        full = performance(slice_period(stress.returns, EVAL_START))
        bucket["cost2x"] = {"oos_sharpe": oos.get("sharpe", float("nan")), "oos_cagr": oos.get("cagr", float("nan")), "full_sharpe": full.get("sharpe", float("nan")), "full_maxdd": full.get("max_drawdown", float("nan"))}
        lagged = evaluate(signal, calendar, bps, lag=2)
        oos2 = performance(slice_period(lagged.returns, OOS_START))
        full2 = performance(slice_period(lagged.returns, EVAL_START))
        bucket["lag2"] = {"oos_sharpe": oos2.get("sharpe", float("nan")), "oos_cagr": oos2.get("cagr", float("nan")), "full_sharpe": full2.get("sharpe", float("nan"))}
        for label, alt in neighbors[key]:
            alt_run = evaluate(alt, calendar, bps, lag=1)
            full_alt = performance(slice_period(alt_run.returns, EVAL_START))
            oos_alt = performance(slice_period(alt_run.returns, OOS_START))
            bucket[label] = {
                "full_sharpe": full_alt.get("sharpe", float("nan")),
                "oos_sharpe": oos_alt.get("sharpe", float("nan")),
                "full_cagr": full_alt.get("cagr", float("nan")),
                "full_maxdd": full_alt.get("max_drawdown", float("nan")),
            }
        robustness[key] = bucket
        print(f"robust {key}", flush=True)

    # Regime facts from tradable ETF and indices.
    def trail(series: pd.Series, days: int) -> float:
        if len(series) <= days or pd.isna(series.iloc[-1]) or pd.isna(series.iloc[-days - 1] if False else series.shift(days).iloc[-1]):
            return float("nan")
        base = series.shift(days).iloc[-1]
        if pd.isna(base) or base == 0:
            return float("nan")
        return float(series.iloc[-1] / base - 1.0)

    def describe_price(series: pd.Series) -> Dict[str, float]:
        last = series.dropna()
        if last.empty:
            return {}
        sma200 = last.rolling(200).mean().iloc[-1]
        ret60 = last.pct_change().tail(60)
        vol20 = float(last.pct_change().tail(20).std(ddof=1) * math.sqrt(TRADING_DAYS))
        vol60 = float(ret60.std(ddof=1) * math.sqrt(TRADING_DAYS))
        peak = last.cummax()
        dd = float(last.iloc[-1] / peak.iloc[-1] - 1.0)
        ytd_base = last[last.index >= pd.Timestamp("2026-01-01")]
        ytd = float(last.iloc[-1] / ytd_base.iloc[0] - 1.0) if len(ytd_base) > 1 else float("nan")
        # YTD should be from prior year end.
        prev = last[last.index < pd.Timestamp("2026-01-01")]
        if not prev.empty:
            ytd = float(last.iloc[-1] / prev.iloc[-1] - 1.0)
        return {
            "last": float(last.iloc[-1]),
            "last_date": str(last.index.max().date()),
            "ret_21": trail(last, 21),
            "ret_63": trail(last, 63),
            "ret_126": trail(last, 126),
            "ret_252": trail(last, 252),
            "ytd": ytd,
            "vol_20": vol20,
            "vol_60": vol60,
            "dd_from_peak": dd,
            "dist_sma200": float(last.iloc[-1] / sma200 - 1.0) if pd.notna(sma200) and sma200 else float("nan"),
        }

    regime = {
        "etf": {code: describe_price(close[code].dropna()) for code in close.columns},
        "index": {code: describe_price(indices[code]["close"]) for code in indices},
        "macro": {},
    }
    for symbol, frame in macro.items():
        regime["macro"][symbol] = describe_price(frame["close"])
        regime["macro"][symbol]["name"] = frame.attrs.get("name", symbol)

    # Liquidity snapshot
    liquidity = {}
    for code in close.columns:
        tail = amount[code].dropna().tail(60)
        liquidity[code] = {
            "name": names.get(code, code),
            "median_amount_60d": float(tail.median()) if len(tail) else float("nan"),
            "last_amount": float(amount[code].dropna().iloc[-1]) if amount[code].dropna().size else float("nan"),
            "start": str(frames[code].index.min().date()),
        }

    # Correlation of strategy daily returns on the common window.
    ret_df = pd.DataFrame({key: slice_period(run.returns, EVAL_START) for key, run in results.items()})
    ret_df["BH300"] = slice_period(bench_runs["BH300"].returns, EVAL_START)
    corr = ret_df.corr().round(3)

    # Latest holdings translated to names.
    latest = {}
    for key, run in results.items():
        weights = summary[key]["last_weights"]
        latest[key] = {f"{code} {names.get(code, code)}": weight for code, weight in weights.items()}

    output = {
        "as_of_requested": AS_OF,
        "eval_start": str(EVAL_START.date()),
        "oos_start": str(OOS_START.date()),
        "rf_annual": RF_ANNUAL,
        "cost_bps_liquid": LIQUID_COST_BPS,
        "cost_bps_sector": SECTOR_COST_BPS,
        "names": names,
        "gaps": etf_gaps + index_gaps + macro_gaps + list(spot.get("gaps") or []),
        "spot": {
            "csi300_spot": spot.get("csi300_spot"),
            "northbound_keys": spot.get("northbound_keys"),
            "valuation_sample": None,
        },
        "jump_flags": jump_flags[:30],
        "jump_flag_count": len(jump_flags),
        "liquidity": liquidity,
        "regime": regime,
        "strategies": summary,
        "benchmarks": bench_summary,
        "robustness": robustness,
        "correlation": corr.to_dict(),
        "latest_holdings": latest,
        "calendar_last": str(close.index.max().date()),
        "calendar_first": str(close.index.min().date()),
    }
    val = spot.get("valuation_report")
    if isinstance(val, dict):
        data = val.get("data") or []
        output["spot"]["valuation_sample"] = data[:2]
        output["spot"]["valuation_columns"] = list(data[0].keys()) if data else []

    path = "/tmp/strategy_research/results_20261006.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, default=str)
    print("WROTE", path)

    def line(label: str, stats: Dict[str, object]) -> str:
        return (
            f"{label:8} total={fmt_pct(stats['total_return'])} cagr={fmt_pct(stats['cagr'])} "
            f"sharpe={stats['sharpe']:.2f} sortino={stats['sortino']:.2f} "
            f"maxdd={fmt_pct(stats['max_drawdown'])} win={fmt_pct(stats['monthly_win_rate'])} "
            f"pf={stats['profit_factor']:.2f} oos_sharpe={stats['oos']['sharpe']:.2f} "
            f"oos_cagr={fmt_pct(stats['oos']['cagr'])} oos_excess={fmt_pct(stats['oos_excess_total'])}"
        )

    print("\n=== BENCHMARKS ===")
    for key, stats in bench_summary.items():
        print(line(key, stats))
    print("\n=== STRATEGIES ===")
    for key, stats in summary.items():
        print(line(key, stats))
        print("   yearly", {k: fmt_pct(v) for k, v in stats["yearly"].items()})
        print("   last", latest[key])


if __name__ == "__main__":
    main()
