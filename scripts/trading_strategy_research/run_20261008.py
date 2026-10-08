# -*- coding: utf-8 -*-
"""Pre-specified A-share ETF strategy research for 2026-10-08.

Parameters below were fixed from published factor definitions (dual momentum,
12-1 time-series momentum, 6-1 cross-sectional momentum, 5-day reversal,
200-day moving average, 12% vol targeting, 60-day low volatility, 60-day
relative-value z-score, 55/20 Donchian) BEFORE any result in this file was
inspected. Do not replace them with in-sample winners.

Universe choice is liquidity and listing date (all primary names listed by
2017-08), not recent relative strength.

Data rule:
- Completed A-share bars run through 2026-09-30 (last session before National
  Day). 2026-10-08 is the holiday reopen; at research time that bar was only
  a morning print, so its high/low/close are ignored.
- The 2026-10-08 open is kept solely as the terminal mark of the last
  open-to-open interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_engine import (  # noqa: E402
    is_new_month,
    is_new_week,
    monthly_excess_ttest,
    performance_stats,
    repair_price_artifacts,
    simulate_open_to_open,
    yearly_returns,
)

OUT = Path(__file__).resolve().parent / "output"
PARTIAL_DAY = pd.Timestamp("2026-10-08")
RF_ANNUAL = 0.015
BASE_COST = 0.0008  # 8 bps one-way; ETF stamp duty is 0, commission + slippage
COST_STRESS = (0.0, 0.0008, 0.0015, 0.0025)
COMMON_START = "2019-01-01"
IS_END = "2023-12-31"
OOS_START = "2024-01-01"

# Tradable multi-asset sleeve. All have Yahoo adjusted history from 2016-01-04.
MULTI = [
    "510300.SS",  # 沪深300
    "510500.SS",  # 中证500
    "159915.SZ",  # 创业板
    "510880.SS",  # 红利
    "518880.SS",  # 黄金
    "511010.SS",  # 国债
    "513100.SS",  # 纳指
    "159920.SZ",  # 恒生
]

# Sector / style sleeve listed by 2017-08-04. Broad CSI300 is excluded so
# selection is not just "hold the market".
SECTOR = [
    "512800.SS",  # 银行
    "512880.SS",  # 证券
    "512010.SS",  # 医药
    "159928.SZ",  # 消费
    "512660.SS",  # 军工
    "512400.SS",  # 有色
    "510880.SS",  # 红利
]

SNAPSHOT = [
    "510300.SS", "510500.SS", "159915.SZ", "510050.SS", "510880.SS",
    "512800.SS", "512880.SS", "512010.SS", "159928.SZ", "512690.SS",
    "512480.SS", "512660.SS", "512400.SS", "515220.SS", "515030.SS",
    "588000.SS", "518880.SS", "511010.SS", "513100.SS", "159920.SZ",
]

# CN10Y=RR is not listed on Yahoo (404 as of this run). China 10Y is a data gap.
MACRO = ["000001.SS", "^HSI", "^GSPC", "^VIX", "^TNX", "CNY=X", "GC=F", "CL=F", "DX-Y.NYB"]
DATA_GAPS = [
    "Yahoo has no usable history for CSI300/CSI500/CSI1000 index tickers (only a 2026-10-08 print); tradable ETF proxies are used.",
    "Yahoo symbol CN10Y=RR (China 10-year yield) returned 404. No China rates series is invented.",
    "Money-market ETF 511880.SS has no usable history on Yahoo, so residual cash earns 0 in the PnL.",
    "No point-in-time fundamentals or options-flow history were available in this run; those signals are not backtested.",
    "2026-10-08 is an incomplete post-holiday session (morning print only at data pull) and is excluded from signals.",
]

NAMES = {
    "510300.SS": "沪深300ETF",
    "510500.SS": "中证500ETF",
    "159915.SZ": "创业板ETF",
    "510050.SS": "上证50ETF",
    "510880.SS": "红利ETF",
    "512800.SS": "银行ETF",
    "512880.SS": "证券ETF",
    "512010.SS": "医药ETF",
    "159928.SZ": "消费ETF",
    "512690.SS": "酒ETF",
    "512480.SS": "半导体ETF",
    "512660.SS": "军工ETF",
    "512400.SS": "有色ETF",
    "515220.SS": "煤炭ETF",
    "515030.SS": "新能源车ETF",
    "588000.SS": "科创50ETF",
    "518880.SS": "黄金ETF",
    "511010.SS": "国债ETF",
    "513100.SS": "纳指ETF",
    "159920.SZ": "恒生ETF",
    "000001.SS": "上证指数",
    "^HSI": "恒生指数",
    "^GSPC": "标普500",
    "^VIX": "VIX",
    "^TNX": "美债10Y",
    "CNY=X": "USD/CNY",
    "GC=F": "COMEX黄金",
    "CL=F": "WTI原油",
    "DX-Y.NYB": "美元指数",
    "CN10Y=RR": "中国10Y",
}


def _download(symbols, start, end) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if frame is None or frame.empty:
        raise RuntimeError("price download returned no rows")
    return frame


def _field(frame: pd.DataFrame, field: str, symbols) -> pd.DataFrame:
    cols = {}
    tickers = set(frame.columns.get_level_values(0))
    for symbol in symbols:
        if symbol not in tickers:
            continue
        block = frame[symbol]
        if field not in block.columns:
            continue
        cols[symbol] = pd.to_numeric(block[field], errors="coerce")
    out = pd.DataFrame(cols)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_panels(refresh: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "prices.pkl"
    symbols = sorted(set(MULTI + SECTOR + SNAPSHOT + MACRO))
    if cache.exists() and not refresh:
        payload = pd.read_pickle(cache)
    else:
        frame = _download(symbols, "2016-01-01", "2026-10-09")
        payload = {
            field: _field(frame, field, symbols)
            for field in ("Open", "High", "Low", "Close", "Volume")
        }
        payload["symbols_missing"] = [s for s in symbols if s not in payload["Close"].columns]
        pd.to_pickle(payload, cache)

    close = payload["Close"].copy()
    open_ = payload["Open"].copy()
    high = payload["High"].copy()
    low = payload["Low"].copy()
    volume = payload["Volume"].copy()
    repair_log = repair_price_artifacts({"Open": open_, "High": high, "Low": low, "Close": close})

    # Incomplete reopen session: keep the open only, as a terminal mark.
    for panel in (close, high, low, volume):
        if PARTIAL_DAY in panel.index:
            panel.loc[PARTIAL_DAY] = np.nan

    master = close["510300.SS"].dropna().index.union(
        open_["510300.SS"].dropna().index
    ).sort_values()
    # Require a real open on the master calendar.
    master = master[open_["510300.SS"].reindex(master).notna()]
    return {
        "close": close.reindex(master),
        "open": open_.reindex(master),
        "high": high.reindex(master),
        "low": low.reindex(master),
        "volume": volume.reindex(master),
        "missing": payload.get("symbols_missing", []),
        "macro_close": payload["Close"],
        "repair_log": repair_log,
    }


def _window_ret(series: pd.Series, idx: int, lookback: int, skip: int = 0):
    end = idx - skip
    start = end - lookback
    if start < 0 or end < 0:
        return None
    a = series.iloc[start]
    b = series.iloc[end]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return None
    return float(b / a - 1.0)


def _ready_assets(close: pd.DataFrame, assets, idx: int, lookback: int) -> list:
    ready = []
    for asset in assets:
        value = _window_ret(close[asset], idx, lookback, 0)
        if value is not None:
            ready.append(asset)
    return ready


def make_weight_fns(close, high, low, dates):
    """Return name -> (assets, weight_fn, rebalance_fn, single_asset, primary_benchmark)."""

    loc = {name: i for i, name in enumerate(MULTI)}
    sec_loc = {name: i for i, name in enumerate(SECTOR)}

    def monthly(k):
        return is_new_month(dates, k)

    def weekly(k):
        return is_new_week(dates, k)

    def daily(k):
        return True

    def every_5(k):
        return k % 5 == 0

    def dual(idx, state, lookback=252):
        del state
        scores = {}
        for asset in MULTI:
            value = _window_ret(close[asset], idx, lookback, 0)
            if value is not None:
                scores[asset] = value
        if len(scores) < len(MULTI):
            return None
        best = max(scores, key=scores.get)
        weights = np.zeros(len(MULTI))
        if scores[best] > 0:
            weights[loc[best]] = 1.0
        return weights

    def tsmom(idx, state, lookback=252, skip=21):
        del state
        scores = {}
        for asset in MULTI:
            value = _window_ret(close[asset], idx, lookback, skip)
            if value is None:
                return None
            scores[asset] = value
        weights = np.zeros(len(MULTI))
        slot = 1.0 / len(MULTI)
        for asset, value in scores.items():
            if value > 0:
                weights[loc[asset]] = slot
        return weights

    def cs_mom(idx, state, lookback=126, skip=21, top_n=3):
        del state
        scores = {}
        for asset in SECTOR:
            value = _window_ret(close[asset], idx, lookback, skip)
            if value is None:
                return None
            scores[asset] = value
        ranked = sorted(scores, key=scores.get, reverse=True)
        picked = [asset for asset in ranked if scores[asset] > 0][:top_n]
        weights = np.zeros(len(SECTOR))
        for asset in picked:
            weights[sec_loc[asset]] = 1.0 / top_n
        return weights

    def reversal(idx, state, lookback=5, bottom_n=3):
        del state
        scores = {}
        for asset in SECTOR:
            value = _window_ret(close[asset], idx, lookback, 0)
            if value is None:
                return None
            scores[asset] = value
        ranked = sorted(scores, key=scores.get)  # worst first
        weights = np.zeros(len(SECTOR))
        for asset in ranked[:bottom_n]:
            weights[sec_loc[asset]] = 1.0 / bottom_n
        return weights

    def ma_trend(idx, state, window=200, asset="510300.SS"):
        del state
        if idx < window - 1:
            return None
        px = close[asset].iloc[idx - window + 1:idx + 1]
        if px.isna().any() or (px <= 0).any():
            return None
        weights = np.zeros(1)
        if float(px.iloc[-1]) > float(px.mean()):
            weights[0] = 1.0
        return weights

    def vol_target(idx, state, window=20, target=0.12, asset="510300.SS"):
        del state
        if idx < window:
            return None
        px = close[asset].iloc[idx - window:idx + 1]
        if px.isna().any() or (px <= 0).any():
            return None
        vol = float(px.pct_change().iloc[1:].std(ddof=1) * np.sqrt(252.0))
        weights = np.zeros(1)
        if vol > 0:
            weights[0] = min(1.0, target / vol)
        return weights

    def low_vol(idx, state, window=60, pick_n=4):
        del state
        vols = {}
        for asset in SECTOR:
            if idx < window:
                return None
            px = close[asset].iloc[idx - window:idx + 1]
            if px.isna().any() or (px <= 0).any():
                return None
            vols[asset] = float(px.pct_change().iloc[1:].std(ddof=1))
        ranked = sorted(vols, key=vols.get)
        weights = np.zeros(len(SECTOR))
        for asset in ranked[:pick_n]:
            weights[sec_loc[asset]] = 1.0 / pick_n
        return weights

    def size_rv(idx, state, window=60, threshold=1.0):
        del state
        if idx < window - 1:
            return None
        p300 = close["510300.SS"].iloc[idx - window + 1:idx + 1]
        p500 = close["510500.SS"].iloc[idx - window + 1:idx + 1]
        if p300.isna().any() or p500.isna().any() or (p300 <= 0).any() or (p500 <= 0).any():
            return None
        spread = np.log(p500.to_numpy()) - np.log(p300.to_numpy())
        std = float(np.std(spread, ddof=1))
        z_score = 0.0 if std == 0 else float((spread[-1] - spread.mean()) / std)
        # Order: 510300, 510500
        if z_score < -threshold:
            return np.array([0.30, 0.70])
        if z_score > threshold:
            return np.array([0.70, 0.30])
        return np.array([0.50, 0.50])

    def donchian(idx, state, entry=55, exit_n=20, asset="510300.SS"):
        if idx < entry:
            return None
        close_px = close[asset].iloc[idx]
        prior_high = high[asset].iloc[idx - entry:idx]
        prior_low = low[asset].iloc[idx - exit_n:idx]
        if not np.isfinite(close_px) or prior_high.isna().any() or prior_low.isna().any():
            return None
        pos = int(state.get("pos", 0))
        if pos == 1:
            if float(close_px) < float(prior_low.min()):
                pos = 0
        elif float(close_px) > float(prior_high.max()):
            pos = 1
        state["pos"] = pos
        return np.array([float(pos)])

    specs = {
        "S1_dual_momentum": {
            "assets": MULTI,
            "fn": dual,
            "reb": monthly,
            "single_asset": False,
            "benchmark": "B1_ew_multi",
            "also_vs": "B0_csi300",
            "thesis_id": "absolute_plus_relative_momentum",
        },
        "S2_cs_sector_momentum": {
            "assets": SECTOR,
            "fn": cs_mom,
            "reb": monthly,
            "single_asset": False,
            "benchmark": "B2_ew_sector",
            "also_vs": "B0_csi300",
            "thesis_id": "sector_underreaction",
        },
        "S3_short_reversal": {
            "assets": SECTOR,
            "fn": reversal,
            "reb": every_5,
            "single_asset": False,
            "benchmark": "B2_ew_sector",
            "also_vs": "B0_csi300",
            "thesis_id": "weekly_overreaction",
        },
        "S4_ma200": {
            "assets": ["510300.SS"],
            "fn": ma_trend,
            "reb": daily,
            "single_asset": True,
            "benchmark": "B0_csi300",
            "also_vs": None,
            "thesis_id": "slow_trend",
        },
        "S5_vol_target": {
            "assets": ["510300.SS"],
            "fn": vol_target,
            "reb": weekly,
            "single_asset": True,
            "benchmark": "B0_csi300",
            "also_vs": None,
            "thesis_id": "vol_clustering",
        },
        "S6_low_vol": {
            "assets": SECTOR,
            "fn": low_vol,
            "reb": monthly,
            "single_asset": False,
            "benchmark": "B2_ew_sector",
            "also_vs": "B0_csi300",
            "thesis_id": "low_vol_anomaly",
        },
        "S7_tsmom": {
            "assets": MULTI,
            "fn": tsmom,
            "reb": monthly,
            "single_asset": False,
            "benchmark": "B1_ew_multi",
            "also_vs": "B0_csi300",
            "thesis_id": "diversified_tsmom",
        },
        "S8_size_rv": {
            "assets": ["510300.SS", "510500.SS"],
            "fn": size_rv,
            "reb": daily,
            "single_asset": False,
            "benchmark": "B3_size_barbell",
            "also_vs": "B0_csi300",
            "thesis_id": "size_spread_mean_reversion",
        },
        "S9_donchian": {
            "assets": ["510300.SS"],
            "fn": donchian,
            "reb": daily,
            "single_asset": True,
            "benchmark": "B0_csi300",
            "also_vs": None,
            "thesis_id": "channel_breakout",
        },
    }
    return specs


def _static_weights(assets_all, target):
    def fn(idx, state):
        del idx, state
        return np.array([target[asset] for asset in assets_all], dtype=float)

    return fn


def run_book(panels, one_way_cost: float) -> dict:
    close = panels["close"]
    dates = close.index
    specs = make_weight_fns(close, panels["high"], panels["low"], dates)
    books = {}

    def run_one(name, assets, fn, reb):
        opens = panels["open"][assets].to_numpy(dtype=float)
        result = simulate_open_to_open(
            opens=opens,
            dates=dates,
            weight_fn=fn,
            rebalance_fn=reb,
            one_way_cost=one_way_cost,
            asset_names=assets,
        )
        books[name] = result

    for name, spec in specs.items():
        run_one(name, spec["assets"], spec["fn"], spec["reb"])

    run_one(
        "B0_csi300",
        ["510300.SS"],
        _static_weights(["510300.SS"], {"510300.SS": 1.0}),
        lambda k: k == 1,
    )
    run_one(
        "B1_ew_multi",
        MULTI,
        _static_weights(MULTI, {asset: 1.0 / len(MULTI) for asset in MULTI}),
        lambda k: is_new_month(dates, k),
    )
    def sector_ew(idx, state, close_panel=close):
        # Stay in cash until every sleeve has a real close. Prevents allocating
        # NAV to ETFs that were not listed yet (no invented backfill).
        del state
        for asset in SECTOR:
            px = close_panel[asset].iloc[idx]
            if not np.isfinite(px) or px <= 0:
                return None
        return np.array([1.0 / len(SECTOR)] * len(SECTOR))

    run_one(
        "B2_ew_sector",
        SECTOR,
        sector_ew,
        lambda k: is_new_month(dates, k),
    )
    run_one(
        "B3_size_barbell",
        ["510300.SS", "510500.SS"],
        _static_weights(["510300.SS", "510500.SS"], {"510300.SS": 0.5, "510500.SS": 0.5}),
        lambda k: is_new_month(dates, k),
    )
    return books


def slice_returns(series: pd.Series, start: str, end: str | None = None) -> pd.Series:
    out = series.loc[series.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def pack_stats(series: pd.Series, turnover: pd.Series, exposure: pd.Series) -> dict:
    stats = performance_stats(series, rf_annual=RF_ANNUAL)
    if stats is None:
        return {}
    aligned_turn = turnover.reindex(series.index).fillna(0.0)
    years = stats["years"]
    stats["annual_turnover"] = float(aligned_turn.sum() / years) if years else np.nan
    stats["avg_exposure"] = float(exposure.reindex(series.index).mean())
    return stats


def window_pack(books, name, start, end=None) -> dict:
    result = books[name]
    rets = slice_returns(result.returns, start, end)
    return pack_stats(rets, result.turnover, result.exposure)


def holding_share(books, name, start, end=None) -> dict:
    weights = slice_returns(books[name].weights, start, end)
    if weights.empty:
        return {}
    mean_w = weights.mean()
    # Days as the largest weight (cash ignored).
    risky = weights.copy()
    leaders = risky.idxmax(axis=1)
    leaders = leaders.where(risky.max(axis=1) > 0, other="CASH")
    share = leaders.value_counts(normalize=True)
    return {
        "mean_weight": {k: float(v) for k, v in mean_w.items()},
        "modal_share": {str(k): float(v) for k, v in share.items()},
    }


def neighbor_grid(panels):
    """Local parameter neighbors. The center is the pre-specified strategy and is excluded."""

    close = panels["close"]
    high = panels["high"]
    low = panels["low"]
    dates = close.index
    base = make_weight_fns(close, high, low, dates)
    # Rebuild variants by calling the same closures' defaults through fresh factories.
    # Implemented explicitly so the grid cannot silently drift from the prose spec.
    variants = []

    def add(strategy, label, assets, fn, reb):
        variants.append((strategy, label, assets, fn, reb))

    loc = {name: i for i, name in enumerate(MULTI)}
    sec_loc = {name: i for i, name in enumerate(SECTOR)}

    def monthly(k):
        return is_new_month(dates, k)

    def weekly(k):
        return is_new_week(dates, k)

    def daily(k):
        return True

    for lookback in (126, 378):
        def fn(idx, state, lookback=lookback):
            del state
            scores = {}
            for asset in MULTI:
                value = _window_ret(close[asset], idx, lookback, 0)
                if value is None:
                    return None
                scores[asset] = value
            best = max(scores, key=scores.get)
            weights = np.zeros(len(MULTI))
            if scores[best] > 0:
                weights[loc[best]] = 1.0
            return weights

        add("S1_dual_momentum", "lookback_%d" % lookback, MULTI, fn, monthly)

    for lookback, top_n in ((63, 3), (189, 3), (126, 2), (126, 4)):
        def fn(idx, state, lookback=lookback, top_n=top_n):
            del state
            scores = {}
            for asset in SECTOR:
                value = _window_ret(close[asset], idx, lookback, 21)
                if value is None:
                    return None
                scores[asset] = value
            ranked = [asset for asset in sorted(scores, key=scores.get, reverse=True) if scores[asset] > 0][:top_n]
            weights = np.zeros(len(SECTOR))
            for asset in ranked:
                weights[sec_loc[asset]] = 1.0 / top_n
            return weights

        add("S2_cs_sector_momentum", "lb%d_top%d" % (lookback, top_n), SECTOR, fn, monthly)

    for lookback, step in ((3, 3), (10, 10), (5, 3), (5, 10)):
        def fn(idx, state, lookback=lookback):
            del state
            scores = {}
            for asset in SECTOR:
                value = _window_ret(close[asset], idx, lookback, 0)
                if value is None:
                    return None
                scores[asset] = value
            ranked = sorted(scores, key=scores.get)[:3]
            weights = np.zeros(len(SECTOR))
            for asset in ranked:
                weights[sec_loc[asset]] = 1.0 / 3.0
            return weights

        def reb(k, step=step):
            return k % step == 0

        add("S3_short_reversal", "lb%d_hold%d" % (lookback, step), SECTOR, fn, reb)

    for window in (100, 150, 250):
        def fn(idx, state, window=window):
            del state
            if idx < window - 1:
                return None
            px = close["510300.SS"].iloc[idx - window + 1:idx + 1]
            if px.isna().any():
                return None
            return np.array([1.0 if float(px.iloc[-1]) > float(px.mean()) else 0.0])

        add("S4_ma200", "ma_%d" % window, ["510300.SS"], fn, daily)

    for target in (0.08, 0.16):
        def fn(idx, state, target=target):
            del state
            if idx < 20:
                return None
            px = close["510300.SS"].iloc[idx - 20:idx + 1]
            vol = float(px.pct_change().iloc[1:].std(ddof=1) * np.sqrt(252.0))
            return np.array([0.0 if vol <= 0 else min(1.0, target / vol)])

        add("S5_vol_target", "target_%.2f" % target, ["510300.SS"], fn, weekly)

    for window, pick_n in ((40, 4), (120, 4), (60, 3), (60, 5)):
        def fn(idx, state, window=window, pick_n=pick_n):
            del state
            vols = {}
            for asset in SECTOR:
                if idx < window:
                    return None
                px = close[asset].iloc[idx - window:idx + 1]
                if px.isna().any() or (px <= 0).any():
                    return None
                vols[asset] = float(px.pct_change().iloc[1:].std(ddof=1))
            weights = np.zeros(len(SECTOR))
            for asset in sorted(vols, key=vols.get)[:pick_n]:
                weights[sec_loc[asset]] = 1.0 / pick_n
            return weights

        add("S6_low_vol", "lb%d_n%d" % (window, pick_n), SECTOR, fn, monthly)

    for lookback in (126, 378):
        def fn(idx, state, lookback=lookback):
            del state
            scores = {}
            for asset in MULTI:
                value = _window_ret(close[asset], idx, lookback, 21)
                if value is None:
                    return None
                scores[asset] = value
            weights = np.zeros(len(MULTI))
            slot = 1.0 / len(MULTI)
            for asset, value in scores.items():
                if value > 0:
                    weights[loc[asset]] = slot
            return weights

        add("S7_tsmom", "lookback_%d" % lookback, MULTI, fn, monthly)

    for window, threshold in ((40, 1.0), (120, 1.0), (60, 0.5), (60, 1.5)):
        def fn(idx, state, window=window, threshold=threshold):
            del state
            if idx < window - 1:
                return None
            p300 = close["510300.SS"].iloc[idx - window + 1:idx + 1]
            p500 = close["510500.SS"].iloc[idx - window + 1:idx + 1]
            spread = np.log(p500.to_numpy()) - np.log(p300.to_numpy())
            std = float(np.std(spread, ddof=1))
            z_score = 0.0 if std == 0 else float((spread[-1] - spread.mean()) / std)
            if z_score < -threshold:
                return np.array([0.30, 0.70])
            if z_score > threshold:
                return np.array([0.70, 0.30])
            return np.array([0.50, 0.50])

        add("S8_size_rv", "lb%d_z%.1f" % (window, threshold), ["510300.SS", "510500.SS"], fn, daily)

    for entry, exit_n in ((20, 10), (100, 40), (55, 10), (55, 40)):
        def fn(idx, state, entry=entry, exit_n=exit_n):
            if idx < entry:
                return None
            close_px = close["510300.SS"].iloc[idx]
            prior_high = high["510300.SS"].iloc[idx - entry:idx]
            prior_low = low["510300.SS"].iloc[idx - exit_n:idx]
            if not np.isfinite(close_px) or prior_high.isna().any() or prior_low.isna().any():
                return None
            pos = int(state.get("pos", 0))
            if pos == 1:
                if float(close_px) < float(prior_low.min()):
                    pos = 0
            elif float(close_px) > float(prior_high.max()):
                pos = 1
            state["pos"] = pos
            return np.array([float(pos)])

        add("S9_donchian", "entry%d_exit%d" % (entry, exit_n), ["510300.SS"], fn, daily)

    # Silence unused warning for base; the center specs are the production run.
    del base
    return variants


def evaluate_neighbors(panels, books) -> dict:
    dates = panels["close"].index
    bh = slice_returns(books["B0_csi300"].returns, COMMON_START)
    bh_stats = performance_stats(bh, rf_annual=RF_ANNUAL)
    specs = make_weight_fns(panels["close"], panels["high"], panels["low"], dates)
    primary_stats = {}
    for name, spec in specs.items():
        primary_stats[name] = performance_stats(
            slice_returns(books[spec["benchmark"]].returns, COMMON_START),
            rf_annual=RF_ANNUAL,
        )
    grouped = {}
    for strategy, label, assets, fn, reb in neighbor_grid(panels):
        opens = panels["open"][assets].to_numpy(dtype=float)
        result = simulate_open_to_open(
            opens=opens,
            dates=dates,
            weight_fn=fn,
            rebalance_fn=reb,
            one_way_cost=BASE_COST,
            asset_names=assets,
        )
        stats = performance_stats(slice_returns(result.returns, COMMON_START), rf_annual=RF_ANNUAL)
        pref = primary_stats.get(strategy)
        if stats is None or bh_stats is None or pref is None:
            continue
        grouped.setdefault(strategy, []).append({
            "label": label,
            "cagr": stats["cagr"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "beats_csi300_cagr": bool(stats["cagr"] > bh_stats["cagr"]),
            "beats_csi300_sharpe": bool(stats["sharpe"] > bh_stats["sharpe"]),
            "beats_primary_cagr": bool(stats["cagr"] > pref["cagr"]),
            "beats_primary_sharpe": bool(stats["sharpe"] > pref["sharpe"]),
        })
    summary = {}
    for strategy, rows in grouped.items():
        summary[strategy] = {
            "n_neighbors": len(rows),
            "frac_beat_csi300_cagr": float(np.mean([row["beats_csi300_cagr"] for row in rows])),
            "frac_beat_csi300_sharpe": float(np.mean([row["beats_csi300_sharpe"] for row in rows])),
            "frac_beat_primary_cagr": float(np.mean([row["beats_primary_cagr"] for row in rows])),
            "neighbors": rows,
        }
    return summary


def jackknife(panels, books) -> dict:
    """Drop one sleeve at a time. Single-asset strategies are flagged separately."""

    close = panels["close"]
    dates = close.index
    out = {}

    def eval_fn(name, assets, fn, reb):
        result = simulate_open_to_open(
            opens=panels["open"][assets].to_numpy(dtype=float),
            dates=dates,
            weight_fn=fn,
            rebalance_fn=reb,
            one_way_cost=BASE_COST,
            asset_names=assets,
        )
        stats = performance_stats(slice_returns(result.returns, COMMON_START), rf_annual=RF_ANNUAL)
        return None if stats is None else stats["sharpe"]

    # Dual momentum / TSMOM: remove one multi asset.
    for dropped in MULTI:
        kept = [asset for asset in MULTI if asset != dropped]
        loc = {asset: i for i, asset in enumerate(kept)}

        def dual(idx, state, kept=kept, loc=loc):
            del state
            scores = {}
            for asset in kept:
                value = _window_ret(close[asset], idx, 252, 0)
                if value is None:
                    return None
                scores[asset] = value
            best = max(scores, key=scores.get)
            weights = np.zeros(len(kept))
            if scores[best] > 0:
                weights[loc[best]] = 1.0
            return weights

        def tsmom(idx, state, kept=kept, loc=loc):
            del state
            scores = {}
            for asset in kept:
                value = _window_ret(close[asset], idx, 252, 21)
                if value is None:
                    return None
                scores[asset] = value
            weights = np.zeros(len(kept))
            slot = 1.0 / len(kept)
            for asset, value in scores.items():
                if value > 0:
                    weights[loc[asset]] = slot
            return weights

        out.setdefault("S1_dual_momentum", {})[dropped] = eval_fn(
            "dual", kept, dual, lambda k: is_new_month(dates, k)
        )
        out.setdefault("S7_tsmom", {})[dropped] = eval_fn(
            "ts", kept, tsmom, lambda k: is_new_month(dates, k)
        )

    for dropped in SECTOR:
        kept = [asset for asset in SECTOR if asset != dropped]
        loc = {asset: i for i, asset in enumerate(kept)}

        def cs_mom(idx, state, kept=kept, loc=loc):
            del state
            scores = {}
            for asset in kept:
                value = _window_ret(close[asset], idx, 126, 21)
                if value is None:
                    return None
                scores[asset] = value
            ranked = [a for a in sorted(scores, key=scores.get, reverse=True) if scores[a] > 0][:3]
            weights = np.zeros(len(kept))
            for asset in ranked:
                weights[loc[asset]] = 1.0 / 3.0
            return weights

        def low_vol(idx, state, kept=kept, loc=loc):
            del state
            vols = {}
            for asset in kept:
                if idx < 60:
                    return None
                px = close[asset].iloc[idx - 60:idx + 1]
                if px.isna().any() or (px <= 0).any():
                    return None
                vols[asset] = float(px.pct_change().iloc[1:].std(ddof=1))
            weights = np.zeros(len(kept))
            for asset in sorted(vols, key=vols.get)[: min(4, len(kept))]:
                weights[loc[asset]] = 1.0 / min(4, len(kept))
            return weights

        out.setdefault("S2_cs_sector_momentum", {})[dropped] = eval_fn(
            "cs", kept, cs_mom, lambda k: is_new_month(dates, k)
        )
        out.setdefault("S6_low_vol", {})[dropped] = eval_fn(
            "lv", kept, low_vol, lambda k: is_new_month(dates, k)
        )
        out.setdefault("S3_short_reversal", {})[dropped] = eval_fn(
            "rv",
            kept,
            lambda idx, state, kept=kept, loc=loc: _reversal_kept(close, idx, kept, loc),
            lambda k: k % 5 == 0,
        )
    return out


def _reversal_kept(close, idx, kept, loc):
    scores = {}
    for asset in kept:
        value = _window_ret(close[asset], idx, 5, 0)
        if value is None:
            return None
        scores[asset] = value
    weights = np.zeros(len(kept))
    for asset in sorted(scores, key=scores.get)[:3]:
        weights[loc[asset]] = 1.0 / 3.0
    return weights


def asset_diagnostics(panels) -> dict:
    close = panels["close"]
    volume = panels["volume"]
    rows = {}
    completed = close["510300.SS"].dropna().index
    last = completed.max()
    for symbol in SNAPSHOT:
        if symbol not in close.columns:
            rows[symbol] = {"status": "missing"}
            continue
        px = close[symbol].dropna()
        px = px.loc[px.index <= last]
        if px.empty:
            rows[symbol] = {"status": "empty"}
            continue
        daily = px.pct_change().dropna()
        vol = volume[symbol].reindex(px.index) if symbol in volume.columns else None
        notional = None
        if vol is not None:
            notional_series = (vol * px).replace(0, np.nan).dropna()
            if len(notional_series):
                notional = float(notional_series.tail(60).median())
        rows[symbol] = {
            "name": NAMES.get(symbol, symbol),
            "first": str(px.index.min().date()),
            "last_completed": str(px.index.max().date()),
            "n": int(len(px)),
            "total_return": float(px.iloc[-1] / px.iloc[0] - 1.0),
            "max_abs_daily": float(daily.abs().max()) if len(daily) else None,
            "median_notional_60d": notional,
            "last_close": float(px.iloc[-1]),
        }
    return rows


def trailing_snapshot(panels) -> dict:
    close = panels["close"]
    completed = close.loc[close["510300.SS"].notna()]
    last_dt = completed.index.max()
    last_i = completed.index.get_loc(last_dt)
    rows = {}
    for symbol in SNAPSHOT:
        if symbol not in completed.columns:
            continue
        series = completed[symbol]
        point = {"last_close": None if pd.isna(series.iloc[last_i]) else float(series.iloc[last_i])}
        for label, lookback in (("d20", 20), ("d60", 60), ("d120", 120), ("d252", 252)):
            value = _window_ret(series, last_i, lookback, 0)
            point[label] = value
        px = series.iloc[:last_i + 1].dropna()
        if len(px) >= 200:
            ma50 = float(px.tail(50).mean())
            ma200 = float(px.tail(200).mean())
            last_px = float(px.iloc[-1])
            point["above_ma50"] = bool(last_px > ma50)
            point["above_ma200"] = bool(last_px > ma200)
            point["dd_252"] = float(last_px / px.tail(252).max() - 1.0)
        rows[symbol] = point

    # Realized vol percentile for CSI300.
    csi = completed["510300.SS"]
    daily = csi.pct_change()
    rv20 = daily.rolling(20).std() * np.sqrt(252.0)
    latest_rv = float(rv20.iloc[-1])
    hist = rv20.dropna().iloc[-252 * 5:]
    pct = float((hist <= latest_rv).mean()) if len(hist) else None
    d120 = rows["510300.SS"].get("d120")
    above = rows["510300.SS"].get("above_ma200")
    if above and d120 is not None and d120 > 0.05:
        regime = "bull"
    elif (not above) and d120 is not None and d120 < -0.05:
        regime = "bear"
    else:
        regime = "sideways"

    partial = {}
    raw_open = panels["open"]
    if PARTIAL_DAY in raw_open.index:
        partial = {
            symbol: None if pd.isna(raw_open.at[PARTIAL_DAY, symbol]) else float(raw_open.at[PARTIAL_DAY, symbol])
            for symbol in ("510300.SS", "159915.SZ", "518880.SS", "510500.SS")
            if symbol in raw_open.columns
        }

    return {
        "last_completed_session": str(last_dt.date()),
        "regime_rule": "bull if close>MA200 and 120d return>5%; bear if close<MA200 and 120d return<-5%; else sideways",
        "csi300_regime": regime,
        "csi300_rv20": latest_rv,
        "csi300_rv20_percentile_5y": pct,
        "snapshot": rows,
        "partial_open_20261008": partial,
    }


def macro_snapshot(macro_close: pd.DataFrame) -> dict:
    rows = {}
    for symbol in MACRO:
        if symbol not in macro_close.columns:
            rows[symbol] = {"status": "missing"}
            continue
        series = macro_close[symbol].dropna()
        if series.empty:
            rows[symbol] = {"status": "empty"}
            continue
        last_i = len(series) - 1

        def chg(lookback, last_i=last_i, series=series):
            value = _window_ret(series, last_i, lookback, 0)
            return value

        rows[symbol] = {
            "name": NAMES.get(symbol, symbol),
            "last_date": str(series.index.max().date()),
            "last": float(series.iloc[-1]),
            "d20": chg(20),
            "d60": chg(60),
            "d252": chg(252),
        }
    return rows


def stress_windows(books) -> dict:
    bh = books["B0_csi300"].returns.dropna()
    roll = (1.0 + bh).rolling(60).apply(lambda x: np.prod(x) - 1.0, raw=True)
    picks = []
    used = pd.Series(False, index=roll.index)
    ranked = roll.dropna().sort_values()
    for end_dt, value in ranked.items():
        if bool(used.loc[:end_dt].tail(60).any()):
            continue
        start_loc = bh.index.get_loc(end_dt) - 59
        if start_loc < 0:
            continue
        start_dt = bh.index[start_loc]
        used.loc[start_dt:end_dt] = True
        window = {}
        for name, result in books.items():
            sl = result.returns.loc[start_dt:end_dt]
            window[name] = float((1.0 + sl).prod() - 1.0) if len(sl) else None
        picks.append({"start": str(start_dt.date()), "end": str(end_dt.date()), "csi300": float(value), "strategies": window})
        if len(picks) >= 3:
            break
    return {"worst_60d_csi300": picks}


def quality_gate(row, neighbors, jackknife_map, single_asset: bool) -> dict:
    """Numeric screen. A pass still needs the economic and statistical write-up."""

    failures = []
    if not row["beat_primary_common"] or not row["beat_primary_oos"]:
        failures.append("does_not_beat_primary_benchmark_common_and_oos")
    if row["common"]["sharpe"] <= 1.0:
        failures.append("sharpe_not_above_1")
    if row["common"]["max_drawdown"] < -0.30:
        failures.append("max_drawdown_worse_than_30pct")
    if row["is"]["cagr"] <= row["primary_is_cagr"] or row["oos"]["cagr"] <= row["primary_oos_cagr"]:
        failures.append("edge_not_in_both_is_and_oos")
    if single_asset:
        failures.append("single_asset_timing_overlay")
    neigh = neighbors.get(row["id"], {})
    frac = neigh.get("frac_beat_primary_cagr")
    if frac is not None and frac < 0.5:
        failures.append("parameter_neighbors_mostly_fail_vs_primary")
    knives = jackknife_map.get(row["id"])
    min_knife = None
    if knives:
        vals = [v for v in knives.values() if v is not None]
        if vals:
            min_knife = float(min(vals))
            if min_knife < 0.5:
                failures.append("jackknife_sharpe_below_0.5")
    t_stat = row["vs_primary_common"]["t_stat"]
    if t_stat is None or (isinstance(t_stat, float) and (np.isnan(t_stat) or abs(t_stat) < 1.5)):
        failures.append("monthly_excess_t_stat_below_1.5")
    return {
        "passes_numeric_bar": len(failures) == 0,
        "failures": failures,
        "jackknife_min_sharpe": min_knife,
        "neighbor_frac_beat_primary_cagr": frac,
    }


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    panels = load_panels(refresh=args.refresh)
    diagnostics = asset_diagnostics(panels)
    bad = {
        symbol: row.get("max_abs_daily")
        for symbol, row in diagnostics.items()
        if isinstance(row.get("max_abs_daily"), float) and row["max_abs_daily"] > 0.35
    }
    if bad:
        raise RuntimeError("abnormal daily moves, refusing to backtest: %s" % bad)

    books = run_book(panels, BASE_COST)
    cost_books = {cost: run_book(panels, cost) for cost in COST_STRESS if cost != BASE_COST}

    strategy_ids = [name for name in books if name.startswith("S")]
    bench_ids = [name for name in books if name.startswith("B")]
    specs = make_weight_fns(panels["close"], panels["high"], panels["low"], panels["close"].index)

    results = {}
    for name in strategy_ids + bench_ids:
        results[name] = {
            "common": window_pack(books, name, COMMON_START),
            "is": window_pack(books, name, COMMON_START, IS_END),
            "oos": window_pack(books, name, OOS_START),
            "y2018": window_pack(books, name, "2018-01-01", "2018-12-31"),
            "yearly": {str(k): float(v) for k, v in yearly_returns(slice_returns(books[name].returns, "2017-01-01")).items()},
        }

    neighbors = evaluate_neighbors(panels, books)
    knives = jackknife(panels, books)
    stress = stress_windows(books)
    regime = trailing_snapshot(panels)
    macro = macro_snapshot(panels["macro_close"])

    ranked_rows = []
    for name in strategy_ids:
        spec = specs[name]
        primary = spec["benchmark"]
        common = results[name]["common"]
        primary_common = results[primary]["common"]
        primary_oos = results[primary]["oos"]
        primary_is = results[primary]["is"]
        also = spec["also_vs"]
        row = {
            "id": name,
            "benchmark": primary,
            "also_vs": also,
            "single_asset": spec["single_asset"],
            "common": common,
            "is": results[name]["is"],
            "oos": results[name]["oos"],
            "y2018": results[name]["y2018"],
            "yearly": results[name]["yearly"],
            "holdings": holding_share(books, name, COMMON_START),
            "beat_primary_common": bool(common["cagr"] > primary_common["cagr"] and common["sharpe"] > primary_common["sharpe"]),
            "beat_primary_oos": bool(results[name]["oos"]["cagr"] > primary_oos["cagr"] and results[name]["oos"]["sharpe"] > primary_oos["sharpe"]),
            "primary_is_cagr": primary_is["cagr"],
            "primary_oos_cagr": primary_oos["cagr"],
            "vs_primary_common": monthly_excess_ttest(
                slice_returns(books[name].returns, COMMON_START),
                slice_returns(books[primary].returns, COMMON_START),
            ),
            "vs_csi300_common": monthly_excess_ttest(
                slice_returns(books[name].returns, COMMON_START),
                slice_returns(books["B0_csi300"].returns, COMMON_START),
            ),
            "cost_stress_cagr": {
                str(cost): window_pack(cost_books[cost] if cost != BASE_COST else books, name, COMMON_START)["cagr"]
                for cost in COST_STRESS
            },
        }
        if also:
            row["beat_csi300_common"] = bool(
                common["cagr"] > results["B0_csi300"]["common"]["cagr"]
                and common["sharpe"] > results["B0_csi300"]["common"]["sharpe"]
            )
            row["beat_csi300_oos"] = bool(
                results[name]["oos"]["cagr"] > results["B0_csi300"]["oos"]["cagr"]
                and results[name]["oos"]["sharpe"] > results["B0_csi300"]["oos"]["sharpe"]
            )
        row["gate"] = quality_gate(row, neighbors, knives, spec["single_asset"])
        # Transparent score: not used to retune. Higher is better among failures too.
        row["rank_score"] = float(
            0.35 * common["sharpe"]
            + 0.25 * results[name]["oos"]["sharpe"]
            + 0.20 * (common["calmar"] if common["calmar"] is not None else 0.0)
            + 0.20 * (1.0 if row["beat_primary_common"] and row["beat_primary_oos"] else 0.0)
        )
        ranked_rows.append(row)

    ranked_rows.sort(key=lambda item: item["rank_score"], reverse=True)

    # Daily return panel for audit, common window only.
    daily = pd.DataFrame({name: books[name].returns for name in strategy_ids + bench_ids})
    daily = daily.loc[daily.index >= pd.Timestamp(COMMON_START)]
    daily.to_csv(OUT / "daily_returns.csv")

    corr = daily.corr().round(3)
    corr.to_csv(OUT / "return_corr.csv")

    payload = {
        "asof_rule": "Completed A-share bars through 2026-09-30; 2026-10-08 open is only the terminal mark.",
        "cost_one_way_base": BASE_COST,
        "rf_annual_for_sharpe_only": RF_ANNUAL,
        "cash_yield_in_pnl": 0.0,
        "common_window": [COMMON_START, "2026-10-08 open"],
        "in_sample": [COMMON_START, IS_END],
        "out_of_sample": [OOS_START, "2026-10-08 open"],
        "missing_symbols": panels["missing"],
        "data_gaps": DATA_GAPS,
        "price_repairs": panels["repair_log"],
        "asset_diagnostics": diagnostics,
        "regime": regime,
        "macro": macro,
        "benchmarks": {name: results[name] for name in bench_ids},
        "ranking": ranked_rows,
        "neighbors": neighbors,
        "jackknife_sharpe": knives,
        "stress": stress,
        "correlation": corr.round(3).to_dict(),
    }
    text = json.dumps(_clean(payload), ensure_ascii=False, indent=2)
    (OUT / "metrics.json").write_text(text, encoding="utf-8")
    print(text[:4000])
    print("WROTE", OUT / "metrics.json")
    print("RANK")
    for row in ranked_rows:
        print(
            row["id"],
            "score", round(row["rank_score"], 3),
            "cagr", round(row["common"]["cagr"], 3),
            "sharpe", round(row["common"]["sharpe"], 3),
            "dd", round(row["common"]["max_drawdown"], 3),
            "oos_cagr", round(row["oos"]["cagr"], 3),
            "gate", row["gate"]["passes_numeric_bar"],
            row["gate"]["failures"],
        )


if __name__ == "__main__":
    main()
