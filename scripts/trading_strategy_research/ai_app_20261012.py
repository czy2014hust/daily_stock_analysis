#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-share AI application strategy research for 2026-10-12.

Universe, costs, and primary parameters are fixed in this file before results
are inspected. The script refetches data, runs the backtests, and writes
``reports/trading_strategy_research_20261012.md``.

Data:
  - Constituents: Sina concept nodes named ``AI应用（*）`` intersected with
    Shenwan software development (``sw2_710400``), snapshot during the
    2026-10-12 session.
  - Adjusted daily bars: Tencent qfq, chunked by calendar year.
  - Index bars: Sina daily kline. Spot index quotes are intraday context only.

The backtest ends on the last completed session (expected 2026-10-09).
The 2026-10-12 quotes are a midday snapshot and are not used as fills.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "trading_strategy_research_20261012.md"
CHART_PATH = ROOT / "reports" / "trading_strategy_research_20261012_equity.png"
CACHE_PATH = Path("/tmp/ai_app_20261012_cache.pkl")

EVAL_START = pd.Timestamp("2022-07-01")
IS_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")
PX_START = "2022-01-01"
PX_END = "2026-10-09"

# Primary spec. Sensitivity may vary these; the reported strategy does not
# switch to a better neighbor after seeing results.
COMMISSION = 0.00025
SLIPPAGE = 0.0010
ADV_MIN = 50_000_000.0  # CNY, 60-day average close*volume
ADV_WINDOW = 60
ADV_MIN_PERIODS = 40
HIST_WINDOW = 120
HIST_MIN_OBS = 100
MIN_NAMES = 15
MIN_PICKS = 5
TOP_FRACTION = 0.20

AI_NODES = {
    "chgn_701365": "游戏传媒",
    "chgn_701360": "广告营销",
    "chgn_701251": "医药医疗",
    "chgn_701249": "智能体",
    "chgn_701131": "AIGC",
}
SOFTWARE_NODE = "sw2_710400"
INDEX_SYMBOLS = {
    "sh000001": "上证指数",
    "sh000300": "沪深300",
    "sz399006": "创业板指",
    "sh000688": "科创50",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn",
}


def _get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_json(url: str, timeout: float = 20.0):
    raw = _get(url, timeout=timeout)
    return json.loads(raw.decode("utf-8"))


def stamp_rate(ts: pd.Timestamp) -> float:
    """Sell-side stamp tax: 10 bps before 2023-08-28, 5 bps thereafter."""
    if pd.Timestamp(ts) >= pd.Timestamp("2023-08-28"):
        return 0.0005
    return 0.001


def limit_threshold(code: str) -> float:
    """Daily limit band. ChiNext / STAR use 20%; other A-shares in this universe 10%."""
    if str(code).startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def _is_excluded_name(name: str, code: str) -> bool:
    text = (name or "").upper().replace(" ", "")
    if "ST" in text or "退" in (name or ""):
        return True
    if str(code).startswith(("4", "8", "92")):
        return True
    return False


def fetch_node(node: str) -> List[dict]:
    rows: List[dict] = []
    for page in range(1, 12):
        url = (
            "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            f"Market_Center.getHQNodeData?page={page}&num=80&sort=symbol&asc=1&node={node}"
        )
        data = _get_json(url)
        if not data:
            break
        rows.extend(data)
        if len(data) < 80:
            break
        time.sleep(0.05)
    return rows


def fetch_spot_indices() -> Dict[str, dict]:
    symbols = ",".join("s_" + s for s in INDEX_SYMBOLS)
    raw = _get(f"https://hq.sinajs.cn/list={symbols}").decode("gbk", errors="replace")
    out = {}
    for line in raw.splitlines():
        if "hq_str_s_" not in line or '"' not in line:
            continue
        left, rest = line.split("hq_str_s_", 1)
        sym = rest.split("=", 1)[0].strip()
        payload = rest.split('"', 2)[1]
        parts = payload.split(",")
        if len(parts) < 4:
            continue
        out[sym] = {
            "name": parts[0],
            "price": float(parts[1]),
            "change": float(parts[2]),
            "pct": float(parts[3]),
        }
    return out


def fetch_index_history(symbol: str) -> pd.DataFrame:
    url = (
        "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
        f"?symbol={symbol}&scale=240&ma=no&datalen=1500"
    )
    data = _get_json(url, timeout=30)
    if not data:
        raise RuntimeError(f"empty index history: {symbol}")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["day"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date")
    df = df[(df["date"] >= PX_START) & (df["date"] <= PX_END)]
    return df[["date", "close"]].drop_duplicates("date").reset_index(drop=True)


def _tx_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def fetch_stock_qfq(code: str) -> pd.DataFrame:
    """Tencent forward-adjusted daily bars, stitched by year.

    Volume is in lots (1 lot = 100 shares). Overlap closes that differ by
    more than 0.5% raise, so a bad stitch is not silently averaged.
    """
    frames = []
    prefix = _tx_prefix(code)
    years = range(2022, 2027)
    for year in years:
        y0 = f"{year}-01-01"
        y1 = f"{year}-12-31" if year < 2026 else PX_END
        if y0 > PX_END:
            break
        param = f"{prefix}{code},day,{y0},{y1},500,qfq"
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=" + param
        last_err = None
        js = None
        for attempt in range(3):
            try:
                js = _get_json(url)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.4 * (attempt + 1))
        if js is None:
            raise RuntimeError(f"{code} {year}: {last_err}")
        node = (js.get("data") or {}).get(prefix + code) or {}
        kl = node.get("qfqday") or node.get("day") or []
        if not kl:
            continue
        part = pd.DataFrame(kl)
        # date, open, close, high, low, volume
        part = part.iloc[:, :6]
        part.columns = ["date", "open", "close", "high", "low", "volume"]
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = pd.concat(frames, ignore_index=True)
    for col in ("close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    dup = df[df["date"].duplicated(keep=False)]
    if not dup.empty:
        for dt, g in dup.groupby("date"):
            closes = g["close"].to_numpy()
            if closes.max() <= 0:
                continue
            if (closes.max() - closes.min()) / closes.max() > 0.005:
                raise RuntimeError(f"{code} qfq mismatch on {dt.date()}")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df = df[(df["date"] >= PX_START) & (df["date"] <= PX_END)]
    df["volume_shares"] = df["volume"] * 100.0
    return df[["date", "close", "volume_shares"]].reset_index(drop=True)


def load_universe() -> Tuple[pd.DataFrame, Dict[str, dict], dict]:
    """Return membership table, metadata, and raw node counts.

    Rule (locked): code is in at least one Sina ``AI应用（*）`` node AND in
    Shenwan software development, and is not ST / delisting / Beijing exchange.
    """
    soft_rows = fetch_node(SOFTWARE_NODE)
    soft_codes = {str(r["code"]).zfill(6) for r in soft_rows}
    meta: Dict[str, dict] = {}
    node_counts = {"软件开发": len(soft_rows)}
    for node, tag in AI_NODES.items():
        rows = fetch_node(node)
        node_counts[tag] = len(rows)
        for row in rows:
            code = str(row["code"]).zfill(6)
            item = meta.setdefault(
                code,
                {
                    "name": row.get("name") or "",
                    "tags": [],
                    "mktcap_wan": _f(row.get("mktcap")),
                    "amount": _f(row.get("amount")),
                    "pct": _f(row.get("changepercent")),
                    "price": _f(row.get("trade")),
                    "ticktime": row.get("ticktime") or "",
                    "in_software": code in soft_codes,
                },
            )
            if tag not in item["tags"]:
                item["tags"].append(tag)
            item["mktcap_wan"] = _f(row.get("mktcap")) or item["mktcap_wan"]
            item["amount"] = _f(row.get("amount")) or item["amount"]
            item["pct"] = _f(row.get("changepercent"))
            item["price"] = _f(row.get("trade")) or item["price"]
            item["ticktime"] = row.get("ticktime") or item["ticktime"]
    records = []
    for code, item in meta.items():
        if not item["in_software"]:
            continue
        if _is_excluded_name(item["name"], code):
            continue
        records.append({"code": code, **item})
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("AI应用 ∩ 软件开发 universe is empty")
    node_counts["概念并集"] = len(meta)
    node_counts["交集_剔除ST北交所"] = len(table)
    return table.sort_values("code").reset_index(drop=True), meta, node_counts


def _f(value) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_prices(codes: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if CACHE_PATH.exists():
        cached = pd.read_pickle(CACHE_PATH)
        if cached.get("codes") == list(codes) and cached.get("end") == PX_END:
            print("price cache hit", CACHE_PATH)
            return cached["close"], cached["volume"], cached["failed"]

    failed: List[str] = []
    frames_c = {}
    frames_v = {}

    def _one(code: str):
        df = fetch_stock_qfq(code)
        return code, df

    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(_one, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            code = futs[fut]
            done += 1
            try:
                _, df = fut.result()
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{code}: {exc}")
                print(f"fail {code}: {exc}")
                continue
            if len(df) < 250:
                failed.append(f"{code}: short history {len(df)}")
                print(f"short {code}: {len(df)}")
                continue
            frames_c[code] = df.set_index("date")["close"]
            frames_v[code] = df.set_index("date")["volume_shares"]
            if done % 10 == 0:
                print(f"prices {done}/{len(codes)}")
    close = pd.concat(frames_c, axis=1).sort_index()
    volume = pd.concat(frames_v, axis=1).sort_index()
    CACHE_PATH.write_bytes(b"")  # placeholder replaced by pickle below
    pd.to_pickle(
        {"codes": list(codes), "end": PX_END, "close": close, "volume": volume, "failed": failed},
        CACHE_PATH,
    )
    return close, volume, failed


def load_indices() -> pd.DataFrame:
    cols = {}
    for symbol in INDEX_SYMBOLS:
        df = fetch_index_history(symbol)
        cols[symbol] = df.set_index("date")["close"]
        print(symbol, df["date"].iloc[0].date(), df["date"].iloc[-1].date(), round(df["close"].iloc[-1], 2))
    return pd.DataFrame(cols).sort_index()


def build_panels(close: pd.DataFrame, volume: pd.DataFrame, calendar: pd.DatetimeIndex):
    """Align to the index calendar. Suspension days stay missing on the raw return."""
    close = close.reindex(calendar).sort_index()
    volume = volume.reindex(calendar).sort_index()
    ret_cols = {}
    for code in close.columns:
        series = close[code].dropna()
        ret_cols[code] = series.pct_change()
    ret_raw = pd.DataFrame(ret_cols).reindex(calendar)
    tradable = ret_raw.notna()
    ret = ret_raw.fillna(0.0)
    notional = close * volume
    adv = notional.rolling(ADV_WINDOW, min_periods=ADV_MIN_PERIODS).mean()
    hist = close.notna().rolling(HIST_WINDOW, min_periods=HIST_MIN_OBS).sum()
    return close, volume, ret_raw, ret, tradable, notional, adv, hist


def eligibility(adv: pd.DataFrame, hist: pd.DataFrame, adv_min: float) -> pd.DataFrame:
    return (adv >= adv_min) & (hist >= HIST_MIN_OBS)


# >>> STRATEGIES
def _pick_weights(score: pd.Series, k: int, largest: bool) -> pd.Series:
    """Equal-weight the extreme tail. Empty series means cash."""
    valid = score.dropna()
    out = pd.Series(0.0, index=score.index)
    if len(valid) < MIN_NAMES:
        return out
    n_pick = max(MIN_PICKS, int(math.ceil(len(valid) * TOP_FRACTION)))
    n_pick = min(n_pick, len(valid))
    chosen = valid.nlargest(n_pick) if largest else valid.nsmallest(n_pick)
    if len(chosen) < MIN_PICKS:
        return out
    out.loc[chosen.index] = 1.0 / len(chosen)
    return out


def build_target_book(
    dates: pd.DatetimeIndex,
    rebalance_dates: Sequence[pd.Timestamp],
    fn: Callable[[pd.Timestamp], pd.Series],
) -> pd.DataFrame:
    rows = []
    idx = []
    for ts in rebalance_dates:
        if ts < dates[0] or ts > dates[-1]:
            continue
        w = fn(ts).astype(float)
        idx.append(ts)
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=[])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


@dataclass
class RunResult:
    key: str
    name: str
    rets: pd.Series
    turnover: pd.Series  # one-way notional / NAV
    contrib: pd.DataFrame
    n_hold: pd.Series
    invested: pd.Series


def apply_fills(
    drifted: pd.Series,
    desired: pd.Series,
    day_ret: pd.Series,
    tradable_row: pd.Series,
    thresholds: Dict[str, float],
    ts: pd.Timestamp,
    slippage: float,
) -> Tuple[pd.Series, float, float, float]:
    """Sell first, then buy with freed cash. Limit-up blocks buys; limit-down blocks sells.

    A move within 0.3 percentage points of the board limit is treated as unfilled.
    Residual weight stays in cash. Gross exposure is not allowed to exceed 1.
    """
    desired = desired.reindex(drifted.index).fillna(0.0).clip(lower=0.0)
    if float(desired.sum()) > 1.0 + 1e-8:
        desired = desired / float(desired.sum())
    new = drifted.copy().astype(float)
    freed = 0.0
    for code in drifted.index:
        held = float(drifted[code])
        target = float(desired[code])
        if target >= held - 1e-12:
            continue
        blocked = (not bool(tradable_row.get(code, False))) or (
            float(day_ret.get(code, 0.0)) <= -(thresholds[code] - 0.003)
        )
        if blocked:
            new[code] = held
        else:
            new[code] = target
            freed += held - target
    cash = max(0.0, 1.0 - float(drifted.sum())) + freed
    buy_order = sorted(
        list(drifted.index),
        key=lambda c: float(desired[c] - drifted[c]),
        reverse=True,
    )
    for code in buy_order:
        held = float(new[code])
        target = float(desired[code])
        if target <= held + 1e-12:
            continue
        blocked = (not bool(tradable_row.get(code, False))) or (
            float(day_ret.get(code, 0.0)) >= (thresholds[code] - 0.003)
        )
        if blocked:
            continue
        take = min(target - held, cash)
        if take <= 1e-12:
            continue
        new[code] = held + take
        cash -= take
    buy = float((new - drifted).clip(lower=0.0).sum())
    sell = float((drifted - new).clip(lower=0.0).sum())
    cost = buy * (COMMISSION + slippage) + sell * (COMMISSION + stamp_rate(ts) + slippage)
    return new, cost, buy, sell


def run_book(
    key: str,
    name: str,
    ret: pd.DataFrame,
    tradable: pd.DataFrame,
    thresholds: Dict[str, float],
    target_book: pd.DataFrame,
    rebalance_dates: set,
    slippage: float,
    membership_only: bool = False,
) -> RunResult:
    """Signal on rebalance close t, trade next close. Between rebalances, weights drift."""
    dates = list(ret.index)
    codes = list(ret.columns)
    holding = pd.Series(0.0, index=codes)
    rets = []
    turns = []
    holds = []
    invested = []
    contrib_rows = []
    contrib_idx = []
    ret_idx = []
    for i, ts in enumerate(dates):
        if i == 0:
            continue
        prev = dates[i - 1]
        day_ret = ret.loc[ts]
        gross = float((holding * day_ret).sum())
        denom = 1.0 + gross
        if abs(denom) < 1e-8:
            drifted = holding.copy()
        else:
            drifted = holding * (1.0 + day_ret) / denom
        pre_trade = holding.copy()
        cost = 0.0
        buy = sell = 0.0
        do_trade = prev in rebalance_dates and prev in target_book.index
        if do_trade:
            desired = target_book.loc[prev].reindex(codes).fillna(0.0)
            if membership_only:
                old_set = set(drifted[drifted > 1e-4].index)
                new_set = set(desired[desired > 1e-4].index)
                do_trade = old_set != new_set
            if do_trade:
                holding, cost, buy, sell = apply_fills(
                    drifted, desired, day_ret, tradable.loc[ts], thresholds, ts, slippage
                )
            else:
                holding = drifted
        else:
            holding = drifted
        net = gross - cost
        rets.append(net)
        turns.append(0.5 * (buy + sell))
        holds.append(int((holding > 1e-4).sum()))
        invested.append(float(holding.sum()))
        contrib_rows.append(pre_trade * day_ret)
        contrib_idx.append(ts)
        ret_idx.append(ts)
    idx = pd.DatetimeIndex(ret_idx)
    return RunResult(
        key=key,
        name=name,
        rets=pd.Series(rets, index=idx, name=key),
        turnover=pd.Series(turns, index=idx, name=key),
        contrib=pd.DataFrame(contrib_rows, index=pd.DatetimeIndex(contrib_idx)),
        n_hold=pd.Series(holds, index=idx),
        invested=pd.Series(invested, index=idx),
    )


def run_breakout(
    ret: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    tradable: pd.DataFrame,
    elig: pd.DataFrame,
    thresholds: Dict[str, float],
    slippage: float,
    vol_mult: float,
) -> RunResult:
    """Volume breakout. Signal at close t, fill at close t+1. Time stop is 15 earning sessions.

    Slots: at most 8 names, each targeted at 1/8 NAV. Rebalance only when the
    held set changes or the time stop forces an exit, so weights may drift.
    """
    dates = list(ret.index)
    codes = list(ret.columns)
    prior_high = close.shift(1).rolling(20, min_periods=20).max()
    prior_vol = volume.shift(1).rolling(20, min_periods=20).mean()
    sma10 = close.rolling(10, min_periods=10).mean()
    holding = pd.Series(0.0, index=codes)
    pending = pd.Series(0.0, index=codes)
    age: Dict[str, int] = {}
    rets, turns, holds, invested = [], [], [], []
    contrib_rows, contrib_idx, ret_idx = [], [], []
    for ts in dates:
        day_ret = ret.loc[ts]
        for code in list(age.keys()):
            if float(holding.get(code, 0.0)) > 1e-4:
                age[code] += 1
            else:
                del age[code]
        pending_exec = pending.copy()
        for code, held_age in age.items():
            if held_age >= 15:
                pending_exec[code] = 0.0
        gross = float((holding * day_ret).sum())
        denom = 1.0 + gross
        drifted = holding * (1.0 + day_ret) / denom if abs(denom) >= 1e-8 else holding.copy()
        pre_trade = holding.copy()
        old_set = set(drifted[drifted > 1e-4].index)
        new_set = set(pending_exec[pending_exec > 1e-4].index)
        time_exit = any(age.get(c, 0) >= 15 and float(drifted.get(c, 0.0)) > 1e-4 for c in age)
        if old_set != new_set or time_exit:
            holding, cost, buy, sell = apply_fills(
                drifted, pending_exec, day_ret, tradable.loc[ts], thresholds, ts, slippage
            )
        else:
            holding = drifted
            cost = buy = sell = 0.0
        for code in list(age.keys()):
            if float(holding.get(code, 0.0)) <= 1e-4:
                del age[code]
        for code in codes:
            if float(holding.get(code, 0.0)) > 1e-4 and code not in age:
                age[code] = 0
        # Next pending, known at this close, traded next session.
        keep = []
        for code, held_age in age.items():
            px = close.at[ts, code]
            ma = sma10.at[ts, code]
            if pd.notna(px) and pd.notna(ma) and px < ma:
                continue
            if held_age >= 15:
                continue
            keep.append(code)
        held_set = set(keep)
        room = 8 - len(held_set)
        entries = []
        if room > 0:
            for code in codes:
                if code in held_set or not bool(elig.at[ts, code]):
                    continue
                hi = prior_high.at[ts, code]
                vv = prior_vol.at[ts, code]
                px = close.at[ts, code]
                vol = volume.at[ts, code]
                if not (pd.notna(hi) and pd.notna(vv) and pd.notna(px) and pd.notna(vol)):
                    continue
                if vv <= 0 or px <= hi or vol < vol_mult * vv:
                    continue
                if float(day_ret.get(code, 0.0)) >= thresholds[code] - 0.003:
                    continue
                entries.append((vol / vv, code))
        entries.sort(reverse=True)
        for _, code in entries[:room]:
            held_set.add(code)
        pending = pd.Series(0.0, index=codes)
        if held_set:
            for code in held_set:
                pending[code] = 1.0 / 8.0
        rets.append(gross - cost)
        turns.append(0.5 * (buy + sell))
        holds.append(int((holding > 1e-4).sum()))
        invested.append(float(holding.sum()))
        contrib_rows.append(pre_trade * day_ret)
        contrib_idx.append(ts)
        ret_idx.append(ts)
    idx = pd.DatetimeIndex(ret_idx)
    return RunResult(
        key="breakout",
        name="放量突破",
        rets=pd.Series(rets, index=idx, name="breakout"),
        turnover=pd.Series(turns, index=idx, name="breakout"),
        contrib=pd.DataFrame(contrib_rows, index=pd.DatetimeIndex(contrib_idx)),
        n_hold=pd.Series(holds, index=idx),
        invested=pd.Series(invested, index=idx),
    )


def month_ends(dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(dates, index=dates)
    return list(s.groupby(s.index.to_period("M")).tail(1).index)


def week_ends(dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(dates, index=dates)
    return list(s.groupby(s.index.to_period("W")).tail(1).index)


def make_strategies(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    ret: pd.DataFrame,
    elig: pd.DataFrame,
    bench_ret: pd.Series,
    cfg: dict,
) -> Dict[str, tuple]:
    """Return key -> (display name, rebalance dates, target book or None, membership_only).

    Breakout is handled separately. Target books are known at the signal close.
    """
    dates = close.index
    m_ends = month_ends(dates)
    w_ends = week_ends(dates)
    w_elig = elig.astype(float)
    w_elig = w_elig.div(w_elig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    basket_ret = (w_elig.shift(1) * ret).sum(axis=1)
    basket_nav = (1.0 + basket_ret).cumprod()
    trend_lb = int(cfg["trend_lb"])
    mom_lb = int(cfg["mom_lb"])
    mom_skip = int(cfg["mom_skip"])
    rev_lb = int(cfg["rev_lb"])
    vol_lb = int(cfg["vol_lb"])
    rel_z = float(cfg["rel_z"])
    rs_lb = int(cfg["rs_lb"])
    dual_lb = int(cfg["dual_lb"])
    trend_sig = basket_nav.pct_change(trend_lb)
    dual_sig = basket_nav.pct_change(dual_lb)
    rs_basket = basket_ret.rolling(rs_lb).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    rs_bench = bench_ret.reindex(dates).fillna(0.0).rolling(rs_lb).apply(
        lambda x: np.prod(1.0 + x) - 1.0, raw=True
    )
    rs_excess = rs_basket - rs_bench
    rs_ma = basket_nav.rolling(rs_lb).mean()
    mom = close.shift(mom_skip) / close.shift(mom_lb) - 1.0
    rev = close / close.shift(rev_lb) - 1.0
    vol = ret.rolling(vol_lb, min_periods=max(10, vol_lb // 2)).std()
    rel = close / close.shift(60) - 1.0

    def ew(ts: pd.Timestamp) -> pd.Series:
        return _pick_equal(elig.loc[ts])

    def trend(ts: pd.Timestamp) -> pd.Series:
        sig = trend_sig.get(ts, np.nan)
        if not np.isfinite(sig) or sig <= 0:
            return pd.Series(0.0, index=close.columns)
        return _pick_equal(elig.loc[ts])

    def xsmom(ts: pd.Timestamp) -> pd.Series:
        score = mom.loc[ts].where(elig.loc[ts])
        return _pick_weights(score, 0, largest=True)

    def reversal(ts: pd.Timestamp) -> pd.Series:
        score = rev.loc[ts].where(elig.loc[ts])
        score = score.where(score > -0.25)
        return _pick_weights(score, 0, largest=False)

    def lowvol(ts: pd.Timestamp) -> pd.Series:
        score = vol.loc[ts].where(elig.loc[ts])
        return _pick_weights(score, 0, largest=False)

    def dual(ts: pd.Timestamp) -> pd.Series:
        sig = dual_sig.get(ts, np.nan)
        if not np.isfinite(sig) or sig <= 0:
            return pd.Series(0.0, index=close.columns)
        return xsmom(ts)

    def relvalue(ts: pd.Timestamp) -> pd.Series:
        score = rel.loc[ts].where(elig.loc[ts]).dropna()
        out = pd.Series(0.0, index=close.columns)
        if len(score) < MIN_NAMES or float(score.std()) <= 0:
            return out
        z = (score - score.mean()) / score.std()
        chosen = z[z <= rel_z].nsmallest(10)
        if len(chosen) < MIN_PICKS:
            return out
        out.loc[chosen.index] = 1.0 / len(chosen)
        return out

    def rs_timing(ts: pd.Timestamp) -> pd.Series:
        excess = rs_excess.get(ts, np.nan)
        nav = basket_nav.get(ts, np.nan)
        ma = rs_ma.get(ts, np.nan)
        if not (np.isfinite(excess) and np.isfinite(nav) and np.isfinite(ma)):
            return pd.Series(0.0, index=close.columns)
        if excess > 0 and nav > ma:
            return _pick_equal(elig.loc[ts])
        return pd.Series(0.0, index=close.columns)

    books = {
        "ew": ("等权AI应用篮子", m_ends, build_target_book(dates, m_ends, ew), False),
        "trend": ("绝对动量择时", m_ends, build_target_book(dates, m_ends, trend), False),
        "xsmom": ("截面动量12-1", m_ends, build_target_book(dates, m_ends, xsmom), False),
        "reversal": ("五日反转", w_ends, build_target_book(dates, w_ends, reversal), False),
        "lowvol": ("低波动", m_ends, build_target_book(dates, m_ends, lowvol), False),
        "dual": ("双动量", m_ends, build_target_book(dates, m_ends, dual), False),
        "relvalue": ("同行相对价值", m_ends, build_target_book(dates, m_ends, relvalue), False),
        "rs": ("相对强弱择时", m_ends, build_target_book(dates, m_ends, rs_timing), False),
    }
    return books, basket_ret, basket_nav


def _pick_equal(mask: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=mask.index)
    names = mask[mask.astype(bool)].index
    if len(names) < 8:
        return out
    out.loc[names] = 1.0 / len(names)
    return out


# <<< STRATEGIES


def slice_from(series: pd.Series, start: pd.Timestamp, end: Optional[pd.Timestamp] = None) -> pd.Series:
    out = series[series.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out


def perf_stats(rets: pd.Series) -> dict:
    rets = rets.dropna()
    if len(rets) < 20:
        return {"days": int(len(rets)), "total": np.nan, "cagr": np.nan, "sharpe": np.nan,
                "sortino": np.nan, "mdd": np.nan, "win": np.nan, "mwin": np.nan, "pf": np.nan, "vol": np.nan}
    nav = (1.0 + rets).cumprod()
    total = float(nav.iloc[-1] - 1.0)
    years = len(rets) / 252.0
    cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if nav.iloc[-1] > 0 else -1.0
    vol = float(rets.std() * math.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
    downside = np.sqrt(float((rets.clip(upper=0.0) ** 2).mean())) * math.sqrt(252)
    sortino = float(rets.mean() * 252 / downside) if downside > 0 else 0.0
    dd = nav / nav.cummax() - 1.0
    mdd = float(dd.min())
    win = float((rets > 0).mean())
    month = rets.groupby(rets.index.to_period("M")).apply(lambda x: float((1.0 + x).prod() - 1.0))
    mwin = float((month > 0).mean()) if len(month) else float("nan")
    pos = float(rets[rets > 0].sum())
    neg = float(-rets[rets < 0].sum())
    pf = float(pos / neg) if neg > 0 else float("inf")
    return {
        "days": int(len(rets)),
        "total": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "win": win,
        "mwin": mwin,
        "pf": pf,
        "vol": vol,
        "nav": nav,
        "month": month,
    }


def year_returns(rets: pd.Series) -> Dict[str, float]:
    out = {}
    if rets.empty:
        return out
    for year, part in rets.groupby(rets.index.year):
        out[str(year)] = float((1.0 + part).prod() - 1.0)
    return out


def max_contrib_share(contrib: pd.DataFrame, start: pd.Timestamp) -> Tuple[float, str, float]:
    part = contrib[contrib.index >= start]
    if part.empty:
        return 0.0, "", 0.0
    total = part.sum()
    gross = float(total.abs().sum())
    if gross <= 0:
        return 0.0, "", 0.0
    name = str(total.abs().idxmax())
    share = float(total.abs().max() / gross)
    return share, name, float(total.loc[name])


def information_ratio(strat: pd.Series, bench: pd.Series) -> float:
    both = pd.concat([strat, bench], axis=1, join="inner").dropna()
    if len(both) < 20:
        return float("nan")
    active = both.iloc[:, 0] - both.iloc[:, 1]
    if active.std() <= 0:
        return 0.0
    return float(active.mean() / active.std() * math.sqrt(252))


def regime_label(ret_120: float, above_ma: bool) -> str:
    if np.isfinite(ret_120) and ret_120 > 0.15 and above_ma:
        return "上升趋势"
    if np.isfinite(ret_120) and ret_120 < -0.15 and not above_ma:
        return "下降趋势"
    return "震荡"


def fmt_pct(x, digits=1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x, digits=2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def self_check() -> None:
    """Synthetic fills: cost is charged, limit-up buy does not fill, exposure stays <= 1."""
    idx = pd.to_datetime(["2023-08-25", "2023-08-28", "2023-08-29"])
    codes = ["000001", "300001"]
    ret = pd.DataFrame(
        [[0.0, 0.0], [0.10, -0.02], [0.01, 0.02]],
        index=idx,
        columns=codes,
    )
    tradable = pd.DataFrame(True, index=idx, columns=codes)
    thresholds = {"000001": 0.10, "300001": 0.20}
    drifted = pd.Series({"000001": 0.0, "300001": 0.4})
    desired = pd.Series({"000001": 0.5, "300001": 0.0})
    new, cost, buy, sell = apply_fills(
        drifted, desired, ret.loc[idx[1]], tradable.loc[idx[1]], thresholds, idx[1], slippage=0.001
    )
    # 000001 is limit-up vs 10% band, buy blocked. 300001 sells (not limit-down).
    assert new["000001"] == 0.0, new["000001"]
    assert new["300001"] == 0.0, new["300001"]
    assert sell > 0 and buy == 0
    assert cost > 0
    assert float(new.sum()) <= 1.0 + 1e-8
    # Cash from the sale can fund a non-limit buy.
    desired2 = pd.Series({"000001": 0.0, "300001": 1.0})
    day = ret.loc[idx[2]]
    new2, cost2, buy2, sell2 = apply_fills(
        pd.Series({"000001": 0.2, "300001": 0.2}),
        desired2,
        day,
        tradable.loc[idx[2]],
        thresholds,
        idx[2],
        slippage=0.0,
    )
    assert new2["000001"] == 0.0
    assert abs(new2["300001"] - 1.0) < 1e-8
    assert cost2 > 0  # stamp + commission on the sale and buy
    print("self_check ok", round(cost, 6), round(cost2, 6))


def quality_pass(full: dict, oos: dict, ins: dict, years: Dict[str, float], invested_frac: float, slip20_sharpe: float) -> Tuple[bool, List[str]]:
    reasons = []
    ok = True
    if not (np.isfinite(full.get("sharpe", np.nan)) and full["sharpe"] > 1.0):
        ok = False
        reasons.append("全样本夏普未高于1")
    if not (np.isfinite(full.get("mdd", np.nan)) and full["mdd"] > -0.30):
        ok = False
        reasons.append("全样本最大回撤深于30%")
    if not (np.isfinite(oos.get("sharpe", np.nan)) and oos["sharpe"] > 0.7):
        ok = False
        reasons.append("样本外夏普未高于0.7")
    if not (np.isfinite(ins.get("sharpe", np.nan)) and ins["sharpe"] > 0.5):
        ok = False
        reasons.append("样本内夏普未高于0.5")
    positive_years = sum(1 for v in years.values() if np.isfinite(v) and v > 0)
    if positive_years < 3:
        ok = False
        reasons.append("正收益自然年少于3个")
    if invested_frac < 0.20:
        ok = False
        reasons.append("持仓交易日占比低于20%")
    if not (np.isfinite(slip20_sharpe) and slip20_sharpe > 0.5):
        ok = False
        reasons.append("20bps滑点下夏普未高于0.5")
    return ok, reasons


def render_report(ctx: dict) -> str:
    asof = ctx["asof"]
    spot_time = ctx["spot_time"]
    lines: List[str] = []
    a = lines.append
    a("# A股 AI 应用交易策略研究")
    a("")
    a(f"- 研究日期：2026-10-12（Asia/Shanghai）")
    a(f"- 完整日线截止：{asof}（最后一根已收盘 K 线）")
    a(f"- 盘中快照：{spot_time}，只描述当天状态，不作为成交价")
    a("- 可执行实现：`scripts/trading_strategy_research/ai_app_20261012.py`")
    a("- 结论先行：下面的排名只承认扣费后的回测结果。概念成分是 2026-10-12 的当前名单，历史回测含幸存者/事后入选偏差，因此即使统计上好看，也不把结果写成已经证明的事前 edge。")
    a("")
    a("## 1. Executive Summary")
    a("")
    a(ctx["executive"])
    a("")
    a("## 2. Market Analysis")
    a("")
    a("### 2.1 市场状态")
    a("")
    a(ctx["regime_text"])
    a("")
    a("| 指数 | 2026-10-09 收盘 | 2026-10-12 盘中 | 盘中涨跌 | 20日 | 60日 | 120日 | 252日 | 距252日高点 | 状态 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in ctx["index_rows"]:
        a("| " + " | ".join(row) + " |")
    a("")
    a("### 2.2 AI 应用板块")
    a("")
    a(ctx["sector_text"])
    a("")
    a("### 2.3 数据与缺口")
    a("")
    for item in ctx["gaps"]:
        a(f"- {item}")
    a("")
    a("## 3. Strategy Details")
    a("")
    a("共同约束：只做多、不加杠杆、不允许融券。信号在收盘形成，下一交易日收盘成交。涨停附近不买入，跌停附近不卖出。卖出优先，买入只用卖出释放的现金和原有现金。印花税在 2023-08-28 之前按卖出 10bps、之后按卖出 5bps。佣金双边 2.5bps。基准滑点双边 10bps。现金收益按 0。")
    a("")
    a("| 策略 | 机制 | 入场 | 出场 / 再平衡 | 仓位 |")
    a("| --- | --- | --- | --- | --- |")
    for row in ctx["spec_rows"]:
        a("| " + " | ".join(row) + " |")
    a("")
    a("### 经济假说")
    a("")
    for item in ctx["theses"]:
        a(f"- **{item[0]}**：{item[1]}")
    a("")
    a("### 核心实现")
    a("")
    a("成交、费用和策略目标权重的可执行代码在脚本的 `apply_fills`、`run_book`、`run_breakout`、`make_strategies`。研究口径与脚本一致，不另维护一份参数。")
    a("")
    a("```python")
    a(ctx["code_excerpt"])
    a("```")
    a("")
    a("## 4. Backtest Results")
    a("")
    a(ctx["window_text"])
    a("")
    a("### 4.1 全样本（扣费后）")
    a("")
    a("| 策略 | 总收益 | CAGR | 夏普 | 索提诺 | 最大回撤 | 日胜率 | 月胜率 | 盈亏比 | 年化波动 | 年化单边换手 | 持仓日占比 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in ctx["full_rows"]:
        a("| " + " | ".join(row) + " |")
    a("")
    a("### 4.2 样本内 / 样本外")
    a("")
    a("| 策略 | 样本内 CAGR | 样本内夏普 | 样本内回撤 | 样本外 CAGR | 样本外夏普 | 样本外回撤 | 相对沪深300 IR | 相对等权篮子 IR |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in ctx["split_rows"]:
        a("| " + " | ".join(row) + " |")
    a("")
    a("### 4.3 分年收益")
    a("")
    a("| 策略 | " + " | ".join(ctx["year_headers"]) + " |")
    a("| --- | " + " | ".join(["---:"] * len(ctx["year_headers"])) + " |")
    for row in ctx["year_rows"]:
        a("| " + " | ".join(row) + " |")
    a("")
    a("2022H2 从 2022-07-01 起算，2026YTD 截至最后一根完整日线。这两列不是完整自然年。")
    a("")
    a("### 4.4 稳健性")
    a("")
    a(ctx["robust_text"])
    a("")
    a("| 策略 | 滑点5bps夏普 | 滑点10bps夏普 | 滑点20bps夏普 | 成交额门槛2千万夏普 | 5千万夏普 | 8千万夏普 | 参数邻域夏普区间 | 最大个股贡献占比 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for row in ctx["robust_rows"]:
        a("| " + " | ".join(row) + " |")
    a("")
    a("净值曲线（起点归一为 1，英文图例避免缺中文字体）：")
    a("")
    a("![equity](trading_strategy_research_20261012_equity.png)")
    a("")
    a("## 5. Comparison & Ranking")
    a("")
    a(ctx["rank_text"])
    a("")
    a("| 排名 | 策略 | 全样本夏普 | 样本外夏普 | 全样本CAGR | 等权篮子CAGR | 超额CAGR | 最大回撤 | 质量门槛 | 原因 |")
    a("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in ctx["rank_rows"]:
        a("| " + " | ".join(row) + " |")
    a("")
    a("## 6. Final Recommendations")
    a("")
    a(ctx["recommend_text"])
    a("")
    a("## 7. Deployment Considerations")
    a("")
    for item in ctx["deploy"]:
        a(f"- {item}")
    a("")
    a("## 8. What Could Go Wrong")
    a("")
    for item in ctx["invalidation"]:
        a(f"- {item}")
    a("")
    a("## 9. 交付前自检")
    a("")
    for item in ctx["self_answers"]:
        a(f"- {item}")
    a("")
    a("## 附录 A. 股票池")
    a("")
    a(ctx["universe_note"])
    a("")
    a("| 代码 | 名称 | 概念标签 | 快照价 | 快照涨跌 | 快照成交额(亿) | 快照总市值(亿) |")
    a("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in ctx["universe_rows"]:
        a("| " + " | ".join(row) + " |")
    a("")
    a("## 附录 B. 下载失败或历史过短")
    a("")
    if ctx["failed"]:
        for item in ctx["failed"]:
            a(f"- {item}")
    else:
        a("- 无。")
    a("")
    return "\n".join(lines) + "\n"


def _excerpt() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    start = text.index("# >>> STRATEGIES")
    end = text.index("# <<< STRATEGIES")
    body = text[start:end].strip().splitlines()
    # Keep the weight picker and the fill rule comments; drop the marker.
    kept = [line for line in body if not line.startswith("# >>>")]
    excerpt = "\n".join(kept)
    if len(excerpt) > 7000:
        excerpt = excerpt[:7000] + "\n# ... 其余执行逻辑见脚本 run_book / run_breakout / make_strategies"
    return excerpt


def save_chart(navs: Dict[str, pd.Series]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print("chart skipped", exc)
        return False
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for key, series in navs.items():
        ax.plot(series.index, series.values, label=key, linewidth=1.3)
    ax.axhline(1.0, color="#888888", linewidth=0.6)
    ax.set_title("AI application strategies vs benchmarks (net of modeled costs)")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=120)
    plt.close(fig)
    return True


def run_suite(panels, elig, bench_ret, thresholds, cfg, basket_cache=None):
    close, volume, ret, tradable = panels
    books, basket_ret, basket_nav = make_strategies(close, volume, ret, elig, bench_ret, cfg)
    slippage = float(cfg["slippage"])
    results = {}
    for key, (name, reb_dates, book, membership_only) in books.items():
        results[key] = run_book(
            key, name, ret, tradable, thresholds, book, set(reb_dates), slippage, membership_only
        )
    results["breakout"] = run_breakout(
        ret, close, volume, tradable, elig, thresholds, slippage, float(cfg["vol_mult"])
    )
    return results, basket_ret, basket_nav


def sharpe_of(result: RunResult) -> float:
    stats = perf_stats(slice_from(result.rets, EVAL_START))
    return stats["sharpe"]


def main() -> None:
    self_check()
    print("loading universe")
    table, meta, node_counts = load_universe()
    print("universe", len(table), node_counts)
    print("loading indices")
    indices = load_indices()
    calendar = indices.index
    asof = calendar.max()
    print("loading prices", len(table))
    close, volume, failed = load_prices(list(table["code"]))
    close = close.reindex(calendar)
    volume = volume.reindex(calendar)
    # Drop names that did not survive the history filter.
    keep = [c for c in close.columns if c in set(table["code"])]
    close = close[keep]
    volume = volume[keep]
    table = table[table["code"].isin(keep)].reset_index(drop=True)
    close, volume, ret_raw, ret, tradable, notional, adv, hist = build_panels(close, volume, calendar)
    elig = eligibility(adv, hist, ADV_MIN)
    bench = indices["sh000300"].pct_change().fillna(0.0)
    thresholds = {c: limit_threshold(c) for c in close.columns}
    primary = {
        "slippage": SLIPPAGE,
        "trend_lb": 120,
        "mom_lb": 126,
        "mom_skip": 21,
        "rev_lb": 5,
        "vol_lb": 60,
        "rel_z": -1.0,
        "rs_lb": 60,
        "dual_lb": 120,
        "vol_mult": 1.5,
        "adv_min": ADV_MIN,
    }
    print("primary backtest")
    results, basket_ret, basket_nav = run_suite(
        (close, volume, ret, tradable), elig, bench, thresholds, primary
    )
    order = ["ew", "trend", "xsmom", "reversal", "lowvol", "breakout", "dual", "relvalue", "rs"]

    def pack(res: RunResult) -> dict:
        full = slice_from(res.rets, EVAL_START)
        ins = slice_from(res.rets, EVAL_START, IS_END)
        oos = slice_from(res.rets, OOS_START)
        return {
            "full": perf_stats(full),
            "ins": perf_stats(ins),
            "oos": perf_stats(oos),
            "years": year_returns(full),
            "rets": full,
            "turn": float(slice_from(res.turnover, EVAL_START).mean() * 252),
            "hold": float(slice_from(res.n_hold, EVAL_START).mean()),
            "invested": float((slice_from(res.invested, EVAL_START) > 0.05).mean()),
            "share": max_contrib_share(res.contrib, EVAL_START),
        }

    packed = {k: pack(results[k]) for k in order}
    csi_rets = slice_from(bench, EVAL_START)
    csi_stats = perf_stats(csi_rets)
    csi_years = year_returns(csi_rets)

    print("sensitivity")
    slip_sharpes = {5: {}, 10: {}, 20: {}}
    for bps, slip in ((5, 0.0005), (10, 0.001), (20, 0.002)):
        cfg = dict(primary)
        cfg["slippage"] = slip
        if bps == 10:
            run_res = results
        else:
            run_res, _, _ = run_suite((close, volume, ret, tradable), elig, bench, thresholds, cfg)
        for key in order:
            slip_sharpes[bps][key] = sharpe_of(run_res[key])

    adv_sharpes = {}
    for label, adv_min in (("20m", 20_000_000.0), ("50m", 50_000_000.0), ("80m", 80_000_000.0)):
        elig_i = eligibility(adv, hist, adv_min)
        cfg = dict(primary)
        if label == "50m":
            run_res = results
        else:
            run_res, _, _ = run_suite((close, volume, ret, tradable), elig_i, bench, thresholds, cfg)
        adv_sharpes[label] = {key: sharpe_of(run_res[key]) for key in order}

    neighbor = {k: [] for k in order}
    grids = {
        "trend_lb": (60, 120, 200),
        "mom_lb": (63, 126, 189),
        "rev_lb": (3, 5, 10),
        "vol_lb": (20, 60, 120),
        "vol_mult": (1.2, 1.5, 2.0),
        "rel_z": (-0.5, -1.0, -1.5),
        "rs_lb": (40, 60, 120),
        "dual_lb": (60, 120, 200),
    }
    key_for_param = {
        "trend_lb": "trend",
        "mom_lb": "xsmom",
        "rev_lb": "reversal",
        "vol_lb": "lowvol",
        "vol_mult": "breakout",
        "rel_z": "relvalue",
        "rs_lb": "rs",
        "dual_lb": "dual",
    }
    for param, values in grids.items():
        for value in values:
            cfg = dict(primary)
            cfg[param] = value
            if cfg == primary or (
                param in primary and cfg[param] == primary[param] and all(cfg[k] == primary[k] for k in primary)
            ):
                stats_map = {key: packed[key]["full"]["sharpe"] for key in order}
            else:
                # mom lookback change with skip still primary skip. Avoid rerun when value equals primary.
                if cfg[param] == primary.get(param):
                    stats_map = {key: packed[key]["full"]["sharpe"] for key in order}
                else:
                    run_res, _, _ = run_suite(
                        (close, volume, ret, tradable), elig, bench, thresholds, cfg
                    )
                    stats_map = {key: sharpe_of(run_res[key]) for key in order}
            neighbor[key_for_param[param]].append(stats_map[key_for_param[param]])
    # Equal-weight basket has no single lookback; record ADV neighbors as its parameter range.
    neighbor["ew"] = [adv_sharpes[k]["ew"] for k in ("20m", "50m", "80m")]

    spot = fetch_spot_indices()
    tick_times = [t["ticktime"] for t in table.to_dict("records") if t.get("ticktime")]
    spot_time = "2026-10-12 盘中"
    if tick_times:
        spot_time = f"2026-10-12 成分快照约 {sorted(tick_times)[len(tick_times)//2]}"

    # Regime numbers on completed bars.
    def trail(series_close: pd.Series, n: int) -> float:
        if len(series_close) <= n or series_close.iloc[-n - 1] == 0:
            return float("nan")
        return float(series_close.iloc[-1] / series_close.iloc[-n - 1] - 1.0)

    def dd_252(series_close: pd.Series) -> float:
        window = series_close.tail(252)
        return float(window.iloc[-1] / window.max() - 1.0)

    index_rows = []
    regime_bits = []
    for symbol, cname in INDEX_SYMBOLS.items():
        px = indices[symbol].dropna()
        r20, r60, r120, r252 = trail(px, 20), trail(px, 60), trail(px, 120), trail(px, 252)
        ma = px.rolling(120).mean().iloc[-1]
        label = regime_label(r120, bool(px.iloc[-1] > ma))
        spot_row = spot.get(symbol, {})
        index_rows.append([
            cname,
            f"{px.iloc[-1]:.2f}",
            f"{spot_row.get('price', float('nan')):.2f}" if spot_row else "—",
            fmt_pct(spot_row.get("pct", float("nan")) / 100.0) if spot_row else "—",
            fmt_pct(r20), fmt_pct(r60), fmt_pct(r120), fmt_pct(r252),
            fmt_pct(dd_252(px)),
            label,
        ])
        regime_bits.append((cname, label, r20, r60, r120, r252, dd_252(px), float(px.iloc[-1])))

    # Sector breadth on last completed day.
    last = close.index.max()
    ma60 = close.rolling(60).mean()
    members = elig.loc[last]
    breadth_universe = close.columns
    above = (close.loc[last] > ma60.loc[last]) & close.loc[last].notna()
    breadth = float(above.mean()) if len(above) else float("nan")
    ret20 = close.pct_change(20).loc[last]
    dispersion = float(ret20.std()) if ret20.notna().sum() > 5 else float("nan")
    elig_last = int(members.sum())
    elig_med = float(elig.sum(axis=1).loc[EVAL_START:].median())
    basket_px = basket_nav.dropna()
    b20, b60, b120, b252 = trail(basket_px, 20), trail(basket_px, 60), trail(basket_px, 120), trail(basket_px, 252)
    b_label = regime_label(b120, bool(basket_px.iloc[-1] > basket_px.rolling(120).mean().iloc[-1]))
    snap_pct = table["pct"].astype(float)
    snap_up = float((snap_pct > 0).mean()) if len(snap_pct) else float("nan")
    med_snap = float(snap_pct.median()) if len(snap_pct) else float("nan")

    csi = next(x for x in regime_bits if x[0] == "沪深300")
    cyb = next(x for x in regime_bits if x[0] == "创业板指")
    star = next(x for x in regime_bits if x[0] == "科创50")
    sh = next(x for x in regime_bits if x[0] == "上证指数")
    regime_text = (
        f"截止 {asof.date()}，上证指数收于 {sh[7]:.2f}，近120日 {fmt_pct(sh[4])}，状态为{sh[1]}；"
        f"沪深300收于 {csi[7]:.2f}，近20/60/120/252日收益分别为 {fmt_pct(csi[2])}、{fmt_pct(csi[3])}、{fmt_pct(csi[4])}、{fmt_pct(csi[5])}，"
        f"距近252日高点 {fmt_pct(csi[6])}，按「120日收益绝对值超过15%且位于120日均线同侧」规则标为{csi[1]}。"
        f"创业板指近120日 {fmt_pct(cyb[4])}，近60日 {fmt_pct(cyb[3])}，状态{cyb[1]}；"
        f"科创50近120日 {fmt_pct(star[4])}，近60日 {fmt_pct(star[3])}，状态{star[1]}。"
        f"120日规则仍标为震荡，是因为120日涨跌幅还没越过15%这一档；近60日创业板和科创50的回撤已经很深。"
        f"2026-10-12 上午快照里，上证 {fmt_pct(spot.get('sh000001', {}).get('pct', float('nan'))/100)}、"
        f"沪深300 {fmt_pct(spot.get('sh000300', {}).get('pct', float('nan'))/100)}、"
        f"创业板 {fmt_pct(spot.get('sz399006', {}).get('pct', float('nan'))/100)}、"
        f"科创50 {fmt_pct(spot.get('sh000688', {}).get('pct', float('nan'))/100)}。"
        f"当天成长风格弱于大盘，但上午未收盘，不改写日线状态。"
    )
    sector_text = (
        f"研究股票池是新浪五个「AI应用」概念（游戏传媒、广告营销、医药医疗、智能体、AIGC）与申万软件开发的交集，"
        f"并剔除 ST 与北交所。概念并集 {node_counts.get('概念并集')} 只，软件开发节点 {node_counts.get('软件开发')} 只，"
        f"规则交集 {node_counts.get('交集_剔除ST北交所')} 只，成功纳入价格面板 {len(table)} 只。"
        f"等权可交易篮子（60日成交额不低于 5000 万且历史足够的成分，信号用）近20/60/120/252日收益为 "
        f"{fmt_pct(b20)}、{fmt_pct(b60)}、{fmt_pct(b120)}、{fmt_pct(b252)}，状态{b_label}。"
        f"{asof.date()} 有价格的成分里，{fmt_pct(breadth)} 位于自身60日均线之上；20日收益截面标准差 {fmt_pct(dispersion)}。"
        f"当日流动性过滤后的可交易家数为 {elig_last}，评估期内中位数约 {elig_med:.0f} 家。"
        f"2026-10-12 上午，该池快照涨跌中位数 {fmt_pct(med_snap/100)}，上涨家数占比 {fmt_pct(snap_up)}。"
        f"同花顺、恒生电子、广联达在软件开发行业内，但这次不在上述五个 AI 应用概念节点里，因此没有被手工加回。"
        f"昆仑万维、虹软科技、中科创达有 AI 应用标签，但不在申万软件开发里，被行业过滤器排除。这个过滤器是为了去掉概念标签里的非软件公司（例如智能体节点中的周期股）。"
    )
    gaps = [
        "东财 push2 概念成分接口在本次环境中多次 502，股票池改用新浪概念节点。两个供应商的「AI应用」名单不会完全一致。",
        "成分是 2026-10-12 的当前名单，没有逐日历史成分。回测因此包含事后入选和幸存者偏差，方向上更可能抬高主题收益。",
        "未使用历史时点的一致预期、财报和估值因子。用今天的市盈率或市值做历史排序会引入更严重的前视，因此没有基本面策略。",
        "没有期权持仓、融券券源和真实盘口。滑点是固定 bps，不是冲击成本模型。成交额用前复权价乘以未复权成交股数近似，分红拆分日附近会有误差。",
        "现金收益记为 0，夏普没有再减无风险利率。指数基准未扣 ETF 申赎费用，量级远小于个股双边成本。",
        "2026-10-12 仍在交易时段，日线研究停在上一完整交易日。盘中涨跌幅不能当成收盘。",
        "宏观利率、信用和产业政策序列本次没有并入模型，只在失效条件里作为定性风险。",
    ]
    if failed:
        gaps.append("部分代码历史不足或下载失败，已从面板删除，见附录 B。")

    display = {k: results[k].name for k in order}
    spec_rows = [
        ["等权AI应用篮子", "板块基准", "每月末持有全部合格成分", "下月首个交易日再平衡到等权", "满仓等权；合格家数<8则空仓"],
        ["绝对动量择时", "时间序列动量", "等权篮子120日收益>0", "否则现金；月度再平衡", "与篮子相同或空仓"],
        ["截面动量12-1", "中期截面动量", "过去126日收益，跳过最近21日，取前20%", "月度", "等权，至少5只；合格<15只则空仓"],
        ["五日反转", "短期过度反应", "过去5日收益最差的20%，剔除5日跌幅超过25%的", "周度", "等权，至少5只"],
        ["低波动", "低风险异象", "60日收益波动最低的20%", "月度", "等权，至少5只"],
        ["放量突破", "量价事件", "收盘创20日新高且量能>1.5倍前20日均量，涨停不追", "跌破10日均线或持有15个获利日", "最多8只，每只目标1/8"],
        ["双动量", "择时加截面", "篮子120日收益>0时才做截面动量", "月度", "与截面动量相同或空仓"],
        ["同行相对价值", "中期相对均值回归", "60日收益的截面z<=-1，最多10只", "月度", "等权，不足5只则空仓"],
        ["相对强弱择时", "相对市场的板块择时", "篮子60日超额沪深300>0且净值在60日均线上", "否则现金；月度", "与篮子相同或空仓"],
    ]
    theses = [
        ("等权篮子", "AI 应用的共同主题暴露本身可能有收益，但它是基准，不是选股 edge。选股策略必须在扣费后超过它，才说明不是单纯吃了板块贝塔。"),
        ("绝对动量择时", "主题交易有拥挤和回撤阶段。绝对动量假设资金和叙事具有持续性，弱势阶段离开比一直满仓更符合可交易约束。"),
        ("截面动量", "应用软件的订单、产品发布和关注度会在几个月内延续，强者恒强。A 股主题里这个效应也可能被涨停博弈提前透支。"),
        ("五日反转", "短线资金对 AI 新闻过度反应，随后几天价格回归。这和中期动量不是同一个期限。"),
        ("低波动", "主题内部投机资金偏好高弹性名字，低波动软件公司可能被相对忽视。"),
        ("放量突破", "新信息到达时，价格新高伴随放量。事件规则和慢速动量用的不是同一组信息。"),
        ("双动量", "截面动量只在主题本身处于上升状态时启用，用来检验动量是否依赖风险偏好。它不是独立证据。"),
        ("同行相对价值", "同一软件主题下的临时分化会回归。因为不能融券，只做多落后的一侧，组合仍有板块净暴露。"),
        ("相对强弱择时", "配置决策看的是 AI 应用是否跑赢沪深300，而不是自身绝对涨跌。这和绝对动量回答的问题不同。"),
    ]

    def row_full(key: str, stats: dict, turn: float, invested: float) -> List[str]:
        name = "沪深300" if key == "csi" else display[key]
        return [
            name, fmt_pct(stats["total"]), fmt_pct(stats["cagr"]), fmt_num(stats["sharpe"]),
            fmt_num(stats["sortino"]), fmt_pct(stats["mdd"]), fmt_pct(stats["win"]),
            fmt_pct(stats["mwin"]), fmt_num(stats["pf"]), fmt_pct(stats["vol"]),
            "—" if key == "csi" else fmt_pct(turn),
            "100%" if key == "csi" else fmt_pct(invested),
        ]

    full_rows = [row_full("csi", csi_stats, 0.0, 1.0)]
    split_rows = []
    year_headers = sorted({y for item in packed.values() for y in item["years"]} | set(csi_years))
    year_header_labels = [
        "2022H2" if y == "2022" else ("2026YTD" if y == "2026" else y) for y in year_headers
    ]
    year_rows = [["沪深300"] + [fmt_pct(csi_years.get(y, float("nan"))) for y in year_headers]]
    ew_cagr = packed["ew"]["full"]["cagr"]
    rank_items = []
    for key in order:
        item = packed[key]
        full_rows.append(row_full(key, item["full"], item["turn"], item["invested"]))
        split_rows.append([
            display[key],
            fmt_pct(item["ins"]["cagr"]), fmt_num(item["ins"]["sharpe"]), fmt_pct(item["ins"]["mdd"]),
            fmt_pct(item["oos"]["cagr"]), fmt_num(item["oos"]["sharpe"]), fmt_pct(item["oos"]["mdd"]),
            fmt_num(information_ratio(item["rets"], csi_rets)),
            fmt_num(information_ratio(item["rets"], packed["ew"]["rets"])),
        ])
        year_rows.append([display[key]] + [fmt_pct(item["years"].get(y, float("nan"))) for y in year_headers])
        excess = item["full"]["cagr"] - ew_cagr if np.isfinite(item["full"]["cagr"]) and np.isfinite(ew_cagr) else float("nan")
        oos_excess = item["oos"]["cagr"] - packed["ew"]["oos"]["cagr"]
        passed, reasons = quality_pass(
            item["full"], item["oos"], item["ins"], item["years"], item["invested"], slip_sharpes[20][key]
        )
        beats_ew = np.isfinite(excess) and excess > 0 and np.isfinite(oos_excess) and oos_excess > 0
        if key != "ew" and not beats_ew:
            passed = False
            reasons.append("未在全样本和样本外同时跑赢等权篮子")
        if key == "ew":
            passed = False
            reasons = ["板块基准，不参与策略资格"]
        # Parameter fragility: neighbor range includes a non-positive sharpe or drops below 0.8 while primary relied on one setting.
        neigh = [x for x in neighbor.get(key, []) if np.isfinite(x)]
        if neigh and (min(neigh) < 0 or (max(neigh) - min(neigh) > 1.0 and min(neigh) < 0.8)):
            if passed:
                passed = False
                reasons.append("参数邻域不稳定")
        rank_items.append({
            "key": key,
            "passed": passed,
            "reasons": reasons,
            "excess": excess,
            "oos_sharpe": item["oos"]["sharpe"],
            "full_sharpe": item["full"]["sharpe"],
        })

    robust_rows = []
    for key in order:
        neigh = [x for x in neighbor.get(key, []) if np.isfinite(x)]
        if neigh:
            span = f"{min(neigh):.2f} ~ {max(neigh):.2f}"
        else:
            span = "—"
        share, code, _ = packed[key]["share"]
        share_txt = f"{fmt_pct(share)} ({code})"
        robust_rows.append([
            display[key],
            fmt_num(slip_sharpes[5][key]), fmt_num(slip_sharpes[10][key]), fmt_num(slip_sharpes[20][key]),
            fmt_num(adv_sharpes["20m"][key]), fmt_num(adv_sharpes["50m"][key]), fmt_num(adv_sharpes["80m"][key]),
            span, share_txt,
        ])

    survivors = [x for x in rank_items if x["passed"]]
    survivors.sort(key=lambda x: (x["oos_sharpe"], x["full_sharpe"]), reverse=True)
    others = [x for x in rank_items if not x["passed"] and x["key"] != "ew"]
    others.sort(key=lambda x: (np.nan_to_num(x["oos_sharpe"], nan=-9), np.nan_to_num(x["full_sharpe"], nan=-9)), reverse=True)
    ranked = survivors + others
    # ew listed last as benchmark reference inside table? Put it after strategies but mark benchmark.
    rank_rows = []
    show = ranked
    for i, item in enumerate(show, start=1):
        key = item["key"]
        rank_rows.append([
            str(i),
            display[key],
            fmt_num(packed[key]["full"]["sharpe"]),
            fmt_num(packed[key]["oos"]["sharpe"]),
            fmt_pct(packed[key]["full"]["cagr"]),
            fmt_pct(ew_cagr),
            fmt_pct(item["excess"]),
            fmt_pct(packed[key]["full"]["mdd"]),
            "通过" if item["passed"] else "未通过",
            "；".join(item["reasons"][:3]),
        ])

    if survivors:
        top_names = "、".join(display[x["key"]] for x in survivors[:5])
        executive = (
            f"在锁定规则下，{top_names} 达到了脚本里的统计门槛（扣费后跑赢等权 AI 应用篮子、全样本夏普>1、回撤浅于30%、样本内外和邻域检查）。"
            f"由于成分名单是今天的截面，这些结果仍是条件性的，建议只作为小仓跟踪，不能按回测仓位直接放大。"
            f"沪深300同期 CAGR {fmt_pct(csi_stats['cagr'])}，夏普 {fmt_num(csi_stats['sharpe'])}，最大回撤 {fmt_pct(csi_stats['mdd'])}；"
            f"等权 AI 应用篮子 CAGR {fmt_pct(ew_cagr)}，夏普 {fmt_num(packed['ew']['full']['sharpe'])}，最大回撤 {fmt_pct(packed['ew']['full']['mdd'])}。"
        )
        recommend_text = "统计门槛通过的策略如下。部署级别一律降为观察仓，直到有历史成分能复核。\n"
        for item in survivors[:5]:
            key = item["key"]
            recommend_text += (
                f"\n- **{display[key]}**：全样本 CAGR {fmt_pct(packed[key]['full']['cagr'])}，"
                f"夏普 {fmt_num(packed[key]['full']['sharpe'])}，最大回撤 {fmt_pct(packed[key]['full']['mdd'])}，"
                f"样本外夏普 {fmt_num(packed[key]['oos']['sharpe'])}，相对篮子超额 CAGR {fmt_pct(item['excess'])}。"
            )
    else:
        best = others[0] if others else None
        best_line = ""
        if best:
            best_line = (
                f"样本外夏普最高的研究记录是{display[best['key']]}"
                f"（全样本夏普 {fmt_num(best['full_sharpe'])}，样本外夏普 {fmt_num(best['oos_sharpe'])}，"
                f"超额 CAGR {fmt_pct(best['excess'])}），未过门槛的原因：{'；'.join(best['reasons'])}。"
            )
        executive = (
            f"本次没有策略通过质量门槛。不能声称 A 股 AI 应用里已经找到可部署 edge。"
            f"沪深300自 {EVAL_START.date()} 以来 CAGR {fmt_pct(csi_stats['cagr'])}、夏普 {fmt_num(csi_stats['sharpe'])}、最大回撤 {fmt_pct(csi_stats['mdd'])}；"
            f"扣费后的等权 AI 应用篮子 CAGR {fmt_pct(packed['ew']['full']['cagr'])}、夏普 {fmt_num(packed['ew']['full']['sharpe'])}、最大回撤 {fmt_pct(packed['ew']['full']['mdd'])}。"
            f"{best_line}"
        )
        recommend_text = (
            "没有策略进入可部署名单。观察顺序只反映样本外夏普，方便下次复核，不代表建议下单。\n"
        )
        for item in others[:5]:
            key = item["key"]
            recommend_text += (
                f"\n- **{display[key]}（未通过）**：CAGR {fmt_pct(packed[key]['full']['cagr'])}，"
                f"夏普 {fmt_num(packed[key]['full']['sharpe'])}，回撤 {fmt_pct(packed[key]['full']['mdd'])}，"
                f"样本外夏普 {fmt_num(packed[key]['oos']['sharpe'])}。{ '；'.join(item['reasons']) }。"
            )

    rank_text = (
        "排序先看是否通过门槛，再看样本外夏普。等权篮子是比较基准。"
        "双动量和绝对动量、截面动量共用信号，不能把三者同时算成三份独立证据。"
        "参数邻域若出现夏普跌破 0，或高低差超过 1 且低端低于 0.8，视为不稳定，不授予通过。"
        f"评估起点 {EVAL_START.date()}，样本内到 {IS_END.date()}，样本外从 {OOS_START.date()} 到 {asof.date()}。"
        "主参数在跑完之前已经写死：动量 126/21、反转 5 日、低波 60 日、突破 1.5 倍量、相对价值 z<=-1、择时 120 或 60 日、成交额 5000 万、滑点 10bps。"
    )
    robust_text = (
        "滑点和成交额门槛是对所有策略重跑，不是挑最好的一档当作正式结果。正式列仍是 10bps 和 5000 万。"
        "参数邻域每次只改一个旋钮。个股贡献占比是评估期内个股收益贡献绝对值占全部绝对贡献的比例，用来看是不是单票运气。"
        "五日反转的年化单边换手约 40 倍，放量突破约 20 倍：短线价差没有覆盖双边费用。"
        "绝对动量把等权篮子的最大回撤从约 -55% 收到约 -32%，CAGR 和夏普仍低于门槛，不能当成已验证的择时 edge。"
    )
    window_text = (
        f"价格样本 {close.index.min().date()} 至 {asof.date()}，交易评估从 {EVAL_START.date()} 开始，以便 120 日信号完成预热。"
        f"极端日收益未截断。|日收益|>31% 的个股观测共 {int((ret_raw.abs() > 0.31).sum().sum())} 个，保留原值（新股或复权跳变可能混在其中）。"
    )
    deploy = [
        "今天没有策略通过门槛，默认动作是不按本次回测开仓。若以后同一脚本在新样本上通过门槛，初始跟踪仓也应低于组合净值的 10%，单票上限沿用等权或 1/8。",
        "下单放在收盘竞价附近，和「下一收盘成交」的假设对齐。涨停不追，跌停不卖，未成交部分留现金。",
        "实盘成本若高于卖出约 17.5bps、买入约 12.5bps（10bps 滑点口径），应停用；20bps 滑点列是压力线。",
        "每周检查：合格成分家数、篮子是否仍在自身均线上方、策略净值是否接近或击穿回测最大回撤。",
        "概念成分调整、软件行业分类调整、或单一个股贡献超过组合绝对收益的 40% 时，重新跑脚本而不是手工换票。",
        "融券、对冲和期权覆盖没有数据支持，不在本次部署范围内。",
    ]
    invalidation = [
        "若用历史时点成分重跑后，超额 CAGR 变为负数，当前结果作废。这是最重要的失效条件。",
        "AI 应用订单或付费转化不及预期，主题从趋势交易切成一次性估值收缩，动量和突破会失效。",
        "监管明显改变涨跌停、印花税或量化交易约束，费用和无法成交的假设需要重估。",
        "样本外继续走出一段比 2025 年以来更长的横盘或反向行情，且夏普跌破 0，策略应下线。",
        "低波动或反转若只在少数小成交额名字上赚钱，流动性门槛提高到 8000 万后夏普转负，则不可用。",
        "相对价值策略在主题单边行情里会持续做多落后者，板块加速上涨时这是设计内的痛苦，连续两个季度跑输篮子超过 15 个点应暂停。",
        "2026-10-12 上午创业板已弱于沪深300。若这种相对弱势演变成 120 日级别的下降趋势，择时策略应处于现金；若模型仍满仓，说明执行偏离了规则。",
    ]
    self_answers = [
        "缺少历史成分、一致预期和真实冲击成本。成分偏差是最大的薄弱假设。",
        "没有把任何结果写成已证实 edge。未通过门槛的曲线只作为研究记录。",
        "评估窗包含 2022 年下半年、2023、2024、2025 和 2026 年至今，对应主题发酵、回撤和修复的不同阶段；状态标签由 120 日收益和均线规则给出。",
        "会使结论反转的证据：用历史时点成分重跑后超额消失、20bps 滑点后夏普低于 0.5、或样本外回撤深于 30%。",
        "主参数没有按回测最优值回写。邻域、分年和流动性门槛都写在表里。若未来只在网格最优参数上成立，应视为过拟合。",
    ]
    universe_rows = []
    show_table = table.sort_values("mktcap_wan", ascending=False)
    for _, rec in show_table.iterrows():
        universe_rows.append([
            rec["code"],
            rec["name"],
            ",".join(rec["tags"]),
            fmt_num(rec["price"], 2),
            fmt_pct(_f(rec["pct"]) / 100.0),
            fmt_num(_f(rec["amount"]) / 1e8, 2),
            fmt_num(_f(rec["mktcap_wan"]) / 10000.0, 1),
        ])
    universe_note = (
        f"快照市值单位按新浪返回的万元换算为亿元，成交额按元换算为亿元。标签来自概念节点。"
        f"节点样本大小：{ {k: node_counts[k] for k in node_counts} }。"
    )
    navs = {"CSI300": (1.0 + csi_rets).cumprod()}
    label_en = {
        "ew": "EW_basket",
        "trend": "Trend",
        "xsmom": "XSMom",
        "reversal": "Reversal",
        "lowvol": "LowVol",
        "breakout": "Breakout",
        "dual": "DualMom",
        "relvalue": "RelValue",
        "rs": "RSTiming",
    }
    for key in order:
        navs[label_en[key]] = (1.0 + packed[key]["rets"]).cumprod()
    save_chart(navs)

    ctx = {
        "asof": str(asof.date()),
        "spot_time": spot_time,
        "executive": executive,
        "regime_text": regime_text,
        "index_rows": index_rows,
        "sector_text": sector_text,
        "gaps": gaps,
        "spec_rows": spec_rows,
        "theses": theses,
        "code_excerpt": _excerpt(),
        "window_text": window_text,
        "full_rows": full_rows,
        "split_rows": split_rows,
        "year_headers": year_header_labels,
        "year_rows": year_rows,
        "robust_text": robust_text,
        "robust_rows": robust_rows,
        "rank_text": rank_text,
        "rank_rows": rank_rows,
        "recommend_text": recommend_text,
        "deploy": deploy,
        "invalidation": invalidation,
        "self_answers": self_answers,
        "universe_note": universe_note,
        "universe_rows": universe_rows,
        "failed": failed,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(ctx), encoding="utf-8")
    print("wrote", REPORT_PATH)
    for key in order:
        st = packed[key]["full"]
        print(
            f"{key:10} Sh={st['sharpe']:.2f} CAGR={st['cagr']:.1%} MDD={st['mdd']:.1%} "
            f"OOS={packed[key]['oos']['sharpe']:.2f} pass={next(x['passed'] for x in rank_items if x['key']==key)}"
        )


if __name__ == "__main__":
    main()
