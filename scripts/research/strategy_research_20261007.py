#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026-10-07 交易策略研究：A 股 ETF 横截面与多资产回测。

可复现入口（需要外网拉取 Yahoo Finance 行情）::

    python3 scripts/research/strategy_research_20261007.py

设计约束（写死在本文件，跑完后不按结果改主参数）:

- 信号使用当日收盘，仓位从下一交易日起生效（与 A 股 / ETF T+1 相容，不做日内回转）。
- 调仓之间权重随价格漂移，只在目标权重发生变化时计费。
- 单边成本主设定 8bp（佣金 + 滑点），覆盖场内 ETF 免印花税后的冲击；压力测试 3bp / 20bp。
- 防御资产为国债 ETF 511010.SS，不是虚构的无风险现金收益。
- 不使用杠杆。单名上限见各策略说明。
- 流动性：过去 20 日成交额中位数低于 1 亿元人民币的标的当日不可开仓。

样本事实依赖数据源，不在代码里填补缺失交易日。

A 股 ETF 价格使用腾讯前复权日线，不用 Yahoo。交叉核对显示 Yahoo 的
auto_adjust 在 512800、512480、510500、512010、512100、512690 等标的上
留下了份额折算断裂和单日尖刺（例如 2025-06-30 银行 ETF 价格被腰斩、
次日又出现不可交易的翻倍）。腾讯行情在 2026-10-07 仍标明沪深「国庆节休市」，
因此 A 股样本止于 2026-09-30，不外推 10 月休市日。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path("/opt/cursor/artifacts")
JSON_PATH = Path("/tmp/tsr_20261007_results.json")
CACHE_PATH = Path("/tmp/tsr_20261007_raw_v2.pkl")

# --- 预先登记的主参数（禁止按回测结果回写） ---
COST_PRIMARY = 0.0008  # 8bp per side
COST_STRESS = (0.0003, 0.0008, 0.0020)
ADV_MIN_CNY = 1.0e8
COMMON_START = "2019-01-02"
COMMON_END = "2026-09-30"
OOS_START = "2023-01-03"
EXT_START = "2018-01-02"
TRADING_DAYS = 252

BOND = "511010.SS"
CSI300 = "510300.SS"
CSI500 = "510500.SS"
GOLD = "518880.SS"

# 2016 起有连续行情、可用于主样本的场内 ETF
CORE_ASSETS = [
    CSI300,       # 沪深300
    CSI500,       # 中证500
    "510050.SS",  # 上证50
    "159915.SZ",  # 创业板
    "512100.SS",  # 中证1000
    "512880.SS",  # 证券
    "512800.SS",  # 银行（2017-07 上市）
    "512010.SS",  # 医药
    "159928.SZ",  # 消费
    "512660.SS",  # 军工
    "510880.SS",  # 红利
    GOLD,         # 黄金
    BOND,         # 国债
]

# 上市较晚，只做稳健性切片，不进入 2019 主比较宇宙
LATE_ASSETS = [
    "512690.SS",  # 酒
    "512480.SS",  # 半导体
]

SECTOR_UNIVERSE = [
    "512880.SS",
    "512800.SS",
    "512010.SS",
    "159928.SZ",
    "512660.SS",
    "510880.SS",
    "159915.SZ",
    "512100.SS",
]

TSMOM_UNIVERSE = [
    CSI300,
    CSI500,
    "510050.SS",
    "159915.SZ",
    "512100.SS",
    "512880.SS",
    "512800.SS",
    "512010.SS",
    "159928.SZ",
    "512660.SS",
    "510880.SS",
    GOLD,
]

MACRO_TICKERS = [
    "^HSI",
    "^GSPC",
    "^VIX",
    "^TNX",
    "DX-Y.NYB",
    "CNY=X",
    "GC=F",
    "CL=F",
    "000001.SS",
]

NAMES = {
    "510300.SS": "沪深300ETF",
    "510500.SS": "中证500ETF",
    "510050.SS": "上证50ETF",
    "159915.SZ": "创业板ETF",
    "512100.SS": "中证1000ETF",
    "512880.SS": "证券ETF",
    "512800.SS": "银行ETF",
    "512010.SS": "医药ETF",
    "159928.SZ": "消费ETF",
    "512660.SS": "军工ETF",
    "510880.SS": "红利ETF",
    "518880.SS": "黄金ETF",
    "511010.SS": "国债ETF",
    "512690.SS": "酒ETF",
    "512480.SS": "半导体ETF",
    "^HSI": "恒生指数",
    "^GSPC": "标普500",
    "^VIX": "VIX",
    "^TNX": "美债10Y收益率",
    "DX-Y.NYB": "美元指数",
    "CNY=X": "USD/CNY",
    "GC=F": "COMEX黄金",
    "CL=F": "WTI原油",
    "000001.SS": "上证综指",
}


def self_check_engine() -> None:
    """用解析解校验：债券换到权益时，单边 8bp × 双边名义 = 16bp。"""
    idx = pd.bdate_range("2020-01-01", periods=6)
    rets = pd.DataFrame(
        {
            "EQ": [0.0, 0.01, 0.01, 0.01, 0.01, 0.01],
            "BD": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    target = pd.DataFrame(np.nan, index=idx, columns=["EQ", "BD"])
    target.iloc[1] = [1.0, 0.0]  # 第 2 天收盘由债券换成权益
    initial = np.array([0.0, 1.0])
    out = simulate(target, rets, cost=0.0008, initial=initial)
    # 第 2 天仍持有债券，收益 0，收盘换仓扣 16bp
    assert abs(out["net"].iloc[1] - (-0.0016)) < 1e-12, out["net"].iloc[1]
    # 第 3 天持有权益，赚 1%，无换手
    assert abs(out["net"].iloc[2] - 0.01) < 1e-12, out["net"].iloc[2]
    assert abs(out["turnover"].iloc[1] - 2.0) < 1e-12


def simulate(
    target: pd.DataFrame,
    rets: pd.DataFrame,
    cost: float,
    initial: np.ndarray,
) -> pd.DataFrame:
    """逐日模拟。target 在收盘时生效；NaN 行表示不交易、权重随收益漂移。

    费用 = cost × Σ|Δw|，只对 target 中的列计费（这些列都是真实交易的 ETF，
    包含国债 ETF）。从 100% 债券换到 100% 权益时 Σ|Δw| = 2。
    """
    cols = list(rets.columns)
    if list(target.columns) != cols:
        target = target.reindex(columns=cols)
    w = np.array(initial, dtype=float)
    w = w / w.sum()
    net_list = []
    gross_list = []
    turn_list = []
    expo_list = []
    for dt in rets.index:
        r = np.nan_to_num(rets.loc[dt].to_numpy(dtype=float), nan=0.0)
        gross = float(w @ r)
        grown = w * (1.0 + r)
        nav = float(grown.sum())
        w_drift = grown / nav if nav > 1e-12 else w.copy()
        tgt = target.loc[dt].to_numpy(dtype=float)
        if np.isnan(tgt).all():
            traded = 0.0
            w_new = w_drift
        else:
            tgt = np.nan_to_num(tgt, nan=0.0)
            total = float(tgt.sum())
            if total <= 1e-12:
                tgt = w_drift
                traded = 0.0
                w_new = w_drift
            else:
                tgt = tgt / total
                traded = float(np.abs(tgt - w_drift).sum())
                if traded < 1e-8:
                    traded = 0.0
                    w_new = w_drift
                else:
                    w_new = tgt
        net_list.append(gross - traded * cost)
        gross_list.append(gross)
        turn_list.append(traded)
        # 当日收益对应的是开盘前（上一日收盘后）权重里的非债券暴露
        bond_idx = cols.index(BOND) if BOND in cols else None
        if bond_idx is None:
            expo_list.append(float(w.sum()))
        else:
            expo_list.append(float(w.sum() - w[bond_idx]))
        w = w_new
    return pd.DataFrame(
        {"net": net_list, "gross": gross_list, "turnover": turn_list, "exposure": expo_list},
        index=rets.index,
    )


def max_drawdown(rets: pd.Series) -> float:
    eq = (1.0 + rets).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def perf_stats(rets: pd.Series, rf_annual: float = 0.0) -> dict:
    r = rets.dropna()
    if len(r) < 10:
        return {"n_days": int(len(r)), "error": "too_few_observations"}
    eq = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    excess = r - rf_annual / TRADING_DAYS
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)) if excess.std(ddof=1) > 0 else float("nan")
    downside = np.minimum(r.to_numpy() - rf_annual / TRADING_DAYS, 0.0)
    down_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS))
    sortino = float((r.mean() - rf_annual / TRADING_DAYS) * TRADING_DAYS / down_dev) if down_dev > 0 else float("nan")
    # 上式 sortino 用的是 mean * 252 / (sqrt(mean(min^2))*sqrt(252)) = mean/sqrt(mean(min^2))*sqrt(252)
    # 重新算一遍，避免笔误
    sortino = float((excess.mean() / np.sqrt(np.mean(downside ** 2))) * np.sqrt(TRADING_DAYS)) if down_dev > 0 else float("nan")
    mdd = max_drawdown(r)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float("nan")
    monthly = []
    for _, chunk in r.groupby(r.index.to_period("M")):
        monthly.append(float((1.0 + chunk).prod() - 1.0))
    monthly_arr = np.array(monthly) if monthly else np.array([0.0])
    wins = monthly_arr[monthly_arr > 0]
    losses = monthly_arr[monthly_arr < 0]
    win_rate = float((monthly_arr > 0).mean())
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")
    yearly = {}
    for year, chunk in r.groupby(r.index.year):
        yearly[str(int(year))] = float((1.0 + chunk).prod() - 1.0)
    worst_month = float(monthly_arr.min())
    return {
        "n_days": int(len(r)),
        "years": round(years, 3),
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe_rf0": sharpe,
        "sortino_rf0": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate_monthly": win_rate,
        "profit_factor_monthly": profit_factor,
        "worst_month": worst_month,
        "yearly": yearly,
    }


def slice_stats(sim: pd.DataFrame, start: str, end: str) -> dict:
    sub = sim.loc[start:end]
    stats = perf_stats(sub["net"])
    stats["avg_exposure"] = float(sub["exposure"].mean()) if len(sub) else float("nan")
    stats["ann_turnover"] = float(sub["turnover"].mean() * TRADING_DAYS) if len(sub) else float("nan")
    stats["ann_cost_drag"] = float((sub["gross"] - sub["net"]).mean() * TRADING_DAYS) if len(sub) else float("nan")
    return stats


def excess_stats(strategy: pd.Series, benchmark: pd.Series) -> dict:
    both = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if len(both) < 10:
        return {}
    diff = both["s"] - both["b"]
    ir = float(diff.mean() / diff.std(ddof=1) * np.sqrt(TRADING_DAYS)) if diff.std(ddof=1) > 0 else float("nan")
    years = len(diff) / TRADING_DAYS
    t_stat = float(ir * np.sqrt(years)) if years > 0 else float("nan")
    return {
        "excess_cagr_approx": float((1 + diff).prod() ** (TRADING_DAYS / len(diff)) - 1),
        "information_ratio": ir,
        "t_stat": t_stat,
        "hit_rate_daily": float((diff > 0).mean()),
    }


TENCENT_CODE = {
    "510300.SS": "sh510300",
    "510500.SS": "sh510500",
    "510050.SS": "sh510050",
    "159915.SZ": "sz159915",
    "512100.SS": "sh512100",
    "512880.SS": "sh512880",
    "512800.SS": "sh512800",
    "512010.SS": "sh512010",
    "159928.SZ": "sz159928",
    "512660.SS": "sh512660",
    "510880.SS": "sh510880",
    "518880.SS": "sh518880",
    "511010.SS": "sh511010",
    "512690.SS": "sh512690",
    "512480.SS": "sh512480",
}


def _tencent_year(code: str, year: int) -> list:
    import json
    import time
    import urllib.request

    start = f"{year}-01-01"
    end = "2026-10-07" if year == 2026 else f"{year}-12-31"
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={code},day,{start},{end},900,qfq"
    )
    last_error = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
            node = payload.get("data", {}).get(code, {})
            return node.get("qfqday") or node.get("day") or []
        except Exception as exc:  # noqa: BLE001 - 网络重试
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"tencent fetch failed {code} {year}: {last_error}")


def fetch_tencent_ohlcv(ticker: str) -> pd.DataFrame:
    """腾讯前复权日线。字段顺序为 open, close, high, low, volume（手）。"""
    code = TENCENT_CODE[ticker]
    frames = []
    for year in range(2016, 2027):
        rows = _tencent_year(code, year)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # 接口有时附带换手等额外列，只取前 6 列
        df = df.iloc[:, :6]
        df.columns = ["date", "open", "close", "high", "low", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        for col in ("open", "close", "high", "low", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # 手 -> 股，便于按人民币成交额做流动性过滤
        df["volume"] = df["volume"] * 100.0
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp("2026-10-07" if year == 2026 else f"{year}-12-31")
        df = df[(df["date"] >= year_start) & (df["date"] <= year_end)]
        frames.append(df)
        if year < 2026:
            import time
            time.sleep(0.15)
    if not frames:
        return pd.DataFrame(columns=["open", "close", "high", "low", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("date").set_index("date").sort_index()
    return out.loc["2016-01-01":"2026-10-07"]


def load_market_data() -> dict:
    if CACHE_PATH.exists():
        return pd.read_pickle(CACHE_PATH)
    frames_c, frames_h, frames_l, frames_v = {}, {}, {}, {}
    for ticker in CORE_ASSETS + LATE_ASSETS:
        print(f"tencent {ticker}", file=sys.stderr)
        ohlc = fetch_tencent_ohlcv(ticker)
        frames_c[ticker] = ohlc["close"]
        frames_h[ticker] = ohlc["high"]
        frames_l[ticker] = ohlc["low"]
        frames_v[ticker] = ohlc["volume"]
    macro = yf.download(
        MACRO_TICKERS,
        start="2016-01-01",
        end="2026-10-07",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    for t in MACRO_TICKERS:
        if t not in macro.columns.get_level_values(0):
            continue
        frames_c[t] = macro[t]["Close"]
        frames_h[t] = macro[t]["High"]
        frames_l[t] = macro[t]["Low"]
        frames_v[t] = macro[t]["Volume"]
    close = pd.DataFrame(frames_c).sort_index()
    high = pd.DataFrame(frames_h).sort_index()
    low = pd.DataFrame(frames_l).sort_index()
    volume = pd.DataFrame(frames_v).sort_index()
    payload = {"close": close, "high": high, "low": low, "volume": volume, "ashare_source": "tencent_qfq"}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, CACHE_PATH)
    return payload


def prepare_panels(bundle: dict) -> dict:
    close = bundle["close"]
    master = close[CSI300].dropna().index
    # 主日历以沪深300ETF 有成交的交易日为准
    vol300 = bundle["volume"][CSI300].reindex(master)
    master = master[vol300.fillna(0) > 0]
    assets = CORE_ASSETS + LATE_ASSETS
    px = close[assets].reindex(master).ffill(limit=2)
    high = bundle["high"][assets].reindex(master).ffill(limit=2)
    low = bundle["low"][assets].reindex(master).ffill(limit=2)
    vol = bundle["volume"][assets].reindex(master)
    yuan_turnover = px * vol
    adv20 = yuan_turnover.rolling(20).median()
    rets = px.pct_change()
    quality = {}
    for col in assets:
        r = rets[col].dropna()
        quality[col] = {
            "name": NAMES.get(col, col),
            "start": str(px[col].first_valid_index().date()) if px[col].first_valid_index() is not None else None,
            "end": str(px[col].last_valid_index().date()) if px[col].last_valid_index() is not None else None,
            "n": int(px[col].dropna().shape[0]),
            "max_abs_1d": float(r.abs().max()) if len(r) else None,
            "adv20_last_cny": float(adv20[col].dropna().iloc[-1]) if adv20[col].dropna().shape[0] else None,
        }
    macro = {}
    for t in MACRO_TICKERS:
        s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        macro[t] = s
    return {
        "px": px,
        "high": high,
        "low": low,
        "rets": rets,
        "adv20": adv20,
        "master": master,
        "quality": quality,
        "macro": macro,
        "close_all": close,
    }


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ser = pd.Series(index, index=index)
    return pd.DatetimeIndex(ser.groupby(index.to_period("M")).max().values)


def empty_target(index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=columns)


def row_from_weights(columns: list[str], weights: dict) -> np.ndarray:
    arr = np.zeros(len(columns))
    for key, val in weights.items():
        arr[columns.index(key)] = val
    leftover = 1.0 - arr.sum()
    if leftover > 1e-10:
        arr[columns.index(BOND)] += leftover
    elif leftover < -1e-6:
        arr = arr / arr.sum()
    return arr


def liquid(adv: pd.DataFrame, asset: str, dt: pd.Timestamp) -> bool:
    if asset not in adv.columns:
        return False
    val = adv.at[dt, asset] if dt in adv.index else np.nan
    if pd.isna(val):
        return False
    return bool(val >= ADV_MIN_CNY)


def history_ready(px: pd.DataFrame, asset: str, dt: pd.Timestamp, bars: int) -> bool:
    series = px[asset].loc[:dt].dropna()
    return len(series) > bars


def signal_return(px: pd.Series, dt: pd.Timestamp, lookback: int, skip: int = 0) -> float:
    hist = px.loc[:dt].dropna()
    if len(hist) <= lookback:
        return float("nan")
    end = hist.iloc[-1 - skip] if skip else hist.iloc[-1]
    start = hist.iloc[-1 - lookback]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return float("nan")
    return float(end / start - 1.0)


def build_s1(px: pd.DataFrame, adv: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """双重动量：权益与黄金谁强持谁，但必须跑赢国债 ETF，否则持国债。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    risky = [CSI300, GOLD]
    for dt in month_ends(px.index):
        bond_r = signal_return(px[BOND], dt, lookback)
        if pd.isna(bond_r) or not history_ready(px, BOND, dt, lookback):
            continue
        scores = {}
        for asset in risky:
            if not history_ready(px, asset, dt, lookback) or not liquid(adv, asset, dt):
                continue
            val = signal_return(px[asset], dt, lookback)
            if pd.notna(val) and val > bond_r:
                scores[asset] = val
        weights = {BOND: 1.0}
        if scores:
            best = max(scores, key=scores.get)
            weights = {best: 1.0}
        target.loc[dt] = row_from_weights(cols, weights)
    return target


def build_s2(
    px: pd.DataFrame,
    adv: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    cap: float = 0.25,
) -> pd.DataFrame:
    """时间序列动量：12-1 月收益为正才持有，等权且单名不超过 25%，其余国债。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    for dt in month_ends(px.index):
        selected = []
        for asset in TSMOM_UNIVERSE:
            if asset not in px.columns:
                continue
            if not history_ready(px, asset, dt, lookback) or not liquid(adv, asset, dt):
                continue
            val = signal_return(px[asset], dt, lookback, skip=skip)
            if pd.notna(val) and val > 0:
                selected.append(asset)
        weights = {}
        n = len(selected)
        if n == 0:
            weights[BOND] = 1.0
        else:
            raw = 1.0 / n
            w = min(cap, raw)
            for asset in selected:
                weights[asset] = w
            weights[BOND] = max(0.0, 1.0 - w * n)
        target.loc[dt] = row_from_weights(cols, weights)
    return target


def build_s3(
    px: pd.DataFrame,
    adv: pd.DataFrame,
    lookback: int = 126,
    abs_lookback: int = 252,
    top_n: int = 3,
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """截面行业动量：6 个月收益排序，只做 12 个月收益仍为正的前 3 名，每名 1/3。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    universe = universe or SECTOR_UNIVERSE
    for dt in month_ends(px.index):
        ranked = []
        for asset in universe:
            if asset not in px.columns:
                continue
            if not history_ready(px, asset, dt, abs_lookback) or not liquid(adv, asset, dt):
                continue
            rel = signal_return(px[asset], dt, lookback)
            absolute = signal_return(px[asset], dt, abs_lookback)
            if pd.notna(rel) and pd.notna(absolute) and absolute > 0:
                ranked.append((rel, asset))
        ranked.sort(reverse=True)
        chosen = [asset for _, asset in ranked[:top_n]]
        weights = {}
        slot = 1.0 / top_n
        for asset in chosen:
            weights[asset] = slot
        weights[BOND] = max(0.0, 1.0 - slot * len(chosen))
        if history_ready(px, BOND, dt, 20):
            target.loc[dt] = row_from_weights(cols, weights)
    return target


def build_s4(
    px: pd.DataFrame,
    adv: pd.DataFrame,
    hold_days: int = 5,
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """短期反转：每 5 个交易日买入过去 5 日跌幅最大的 3 个行业 ETF。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    universe = universe or SECTOR_UNIVERSE
    # 每 5 个交易日评估一次；信号需要 5 日收益
    dates = list(px.index)
    for i, dt in enumerate(dates):
        if i < hold_days or (i % hold_days) != 0:
            continue
        ranked = []
        for asset in universe:
            if not history_ready(px, asset, dt, hold_days) or not liquid(adv, asset, dt):
                continue
            val = signal_return(px[asset], dt, hold_days)
            if pd.notna(val):
                ranked.append((val, asset))
        ranked.sort()  # 最差在前
        chosen = [asset for _, asset in ranked[:3]]
        weights = {}
        slot = 1.0 / 3
        for asset in chosen:
            weights[asset] = slot
        weights[BOND] = max(0.0, 1.0 - slot * len(chosen))
        target.loc[dt] = row_from_weights(cols, weights)
    return target


def build_s5(
    px: pd.DataFrame,
    rets: pd.DataFrame,
    adv: pd.DataFrame,
    vol_window: int = 60,
    ma_window: int = 200,
    top_n: int = 3,
) -> pd.DataFrame:
    """低波 + 趋势：月度在站上 200 日均线的标的中选 60 日波动最低的 3 个。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    vol = rets.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    ma = px.rolling(ma_window).mean()
    universe = [a for a in TSMOM_UNIVERSE if a != GOLD]
    for dt in month_ends(px.index):
        ranked = []
        for asset in universe:
            if not history_ready(px, asset, dt, ma_window) or not liquid(adv, asset, dt):
                continue
            price = px.at[dt, asset]
            average = ma.at[dt, asset]
            sigma = vol.at[dt, asset]
            if pd.isna(price) or pd.isna(average) or pd.isna(sigma):
                continue
            if price > average and sigma > 0:
                ranked.append((float(sigma), asset))
        ranked.sort()
        chosen = [asset for _, asset in ranked[:top_n]]
        weights = {}
        slot = 1.0 / top_n
        for asset in chosen:
            weights[asset] = slot
        weights[BOND] = max(0.0, 1.0 - slot * len(chosen))
        if history_ready(px, BOND, dt, 20):
            target.loc[dt] = row_from_weights(cols, weights)
    return target


def build_s6(
    px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    entry_n: int = 55,
    exit_n: int = 20,
) -> pd.DataFrame:
    """Donchian 趋势：沪深300 突破前 55 日高点做多，跌破前 20 日低点回到国债。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    prior_high = high[CSI300].rolling(entry_n).max().shift(1)
    prior_low = low[CSI300].rolling(exit_n).min().shift(1)
    state = 0
    last_state = None
    for dt in px.index:
        price = px.at[dt, CSI300]
        hi = prior_high.at[dt]
        lo = prior_low.at[dt]
        if pd.isna(price) or pd.isna(hi) or pd.isna(lo):
            continue
        if state == 1 and price < lo:
            state = 0
        elif state == 0 and price > hi:
            state = 1
        if state != last_state:
            weights = {CSI300: 1.0} if state == 1 else {BOND: 1.0}
            target.loc[dt] = row_from_weights(cols, weights)
            last_state = state
    return target


def build_s7(
    px: pd.DataFrame,
    window: int = 60,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """大小盘相对价值（只做多）：300/500 对数比 z 分数极端时持有落后的一边。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    ratio = np.log(px[CSI300] / px[CSI500])
    z = (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std(ddof=1)
    last_state = None
    for dt in px.index:
        val = z.at[dt]
        if pd.isna(val) or not history_ready(px, CSI300, dt, window) or not history_ready(px, CSI500, dt, window):
            continue
        if val > threshold:
            state = "small"  # 300 相对贵，持有 500
        elif val < -threshold:
            state = "large"
        else:
            state = "half"
        if state != last_state:
            if state == "small":
                weights = {CSI500: 1.0}
            elif state == "large":
                weights = {CSI300: 1.0}
            else:
                weights = {CSI300: 0.5, CSI500: 0.5}
            target.loc[dt] = row_from_weights(cols, weights)
            last_state = state
    return target


def build_s8(
    px: pd.DataFrame,
    rets: pd.DataFrame,
    target_vol: float = 0.15,
    vol_window: int = 20,
    step: int = 5,
) -> pd.DataFrame:
    """波动率管理的沪深300：权重 = min(1, 目标波动 / 实现波动)，其余国债。无杠杆。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    rv = rets[CSI300].rolling(vol_window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    dates = list(px.index)
    for i, dt in enumerate(dates):
        if i < vol_window or i % step != 0:
            continue
        sigma = rv.at[dt]
        if pd.isna(sigma) or sigma <= 1e-6:
            continue
        weight = float(min(1.0, target_vol / sigma))
        target.loc[dt] = row_from_weights(cols, {CSI300: weight, BOND: 1.0 - weight})
    return target


def build_s9(px: pd.DataFrame, last_n: int = 2, first_n: int = 3) -> pd.DataFrame:
    """月初月末效应：月末最后 2 个交易日与月初前 3 个交易日持有沪深300，其余时间国债。

    交易所日历视为事前已知（假日安排会提前公布）。
    """
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    index = px.index
    period = index.to_period("M")
    rank = pd.Series(index, index=index).groupby(period).cumcount()
    count = pd.Series(1, index=index).groupby(period).transform("sum")
    is_tom = (rank < first_n) | (rank >= count - last_n)
    # target[t] 在收盘成交，赚的是 t+1 的收益，所以用下一交易日是否落在窗口内
    hold_next = is_tom.shift(-1)
    for dt in index[:-1]:
        flag = bool(hold_next.loc[dt])
        weights = {CSI300: 1.0} if flag else {BOND: 1.0}
        target.loc[dt] = row_from_weights(cols, weights)
    return target


def build_buy_hold(px: pd.DataFrame, weights: dict) -> pd.DataFrame:
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    # 首个债券与权益都有效的交易日建仓，之后漂移
    ready = px[[CSI300, BOND]].dropna().index
    if len(ready) == 0:
        return target
    target.loc[ready[0]] = row_from_weights(cols, weights)
    return target


def build_static_mix(px: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """月度再平衡到固定权重。用于解释性基准，不是事前登记的交易策略。"""
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    for dt in month_ends(px.index):
        if not history_ready(px, CSI300, dt, 2):
            continue
        target.loc[dt] = row_from_weights(cols, weights)
    return target


def build_balanced(px: pd.DataFrame, equity_weight: float = 0.60) -> pd.DataFrame:
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    for dt in month_ends(px.index):
        if not history_ready(px, CSI300, dt, 2):
            continue
        target.loc[dt] = row_from_weights(
            cols, {CSI300: equity_weight, BOND: 1.0 - equity_weight}
        )
    return target


def build_equal_weight(px: pd.DataFrame, adv: pd.DataFrame, universe: list[str], lookback: int = 60) -> pd.DataFrame:
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    for dt in month_ends(px.index):
        names = []
        for asset in universe:
            if history_ready(px, asset, dt, lookback) and liquid(adv, asset, dt):
                names.append(asset)
        if not names:
            continue
        w = 1.0 / len(names)
        weights = {asset: w for asset in names}
        target.loc[dt] = row_from_weights(cols, weights)
    return target


def initial_bond(columns: list[str]) -> np.ndarray:
    arr = np.zeros(len(columns))
    arr[columns.index(BOND)] = 1.0
    return arr


def run_one(name: str, target: pd.DataFrame, rets: pd.DataFrame, cost: float) -> pd.DataFrame:
    return simulate(target, rets, cost=cost, initial=initial_bond(list(rets.columns)))


def pack_strategy(
    sim: pd.DataFrame,
    bench_map: dict[str, pd.Series],
) -> dict:
    full = slice_stats(sim, COMMON_START, COMMON_END)
    insider = slice_stats(sim, COMMON_START, "2022-12-31")
    oos = slice_stats(sim, OOS_START, COMMON_END)
    extended = slice_stats(sim, EXT_START, COMMON_END)
    excess = {
        key: excess_stats(sim.loc[COMMON_START:COMMON_END, "net"], series.loc[COMMON_START:COMMON_END])
        for key, series in bench_map.items()
    }
    excess_oos = {
        key: excess_stats(sim.loc[OOS_START:COMMON_END, "net"], series.loc[OOS_START:COMMON_END])
        for key, series in bench_map.items()
    }
    return {
        "common": full,
        "in_sample": insider,
        "out_of_sample": oos,
        "extended_2018": extended,
        "excess_vs": excess,
        "excess_oos_vs": excess_oos,
    }


def yearly_frame(sims: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for name, sim in sims.items():
        stats = perf_stats(sim.loc[COMMON_START:COMMON_END, "net"])
        out[name] = stats.get("yearly", {})
    return out


def snapshot_asset(px: pd.Series, rets: pd.Series) -> dict:
    s = px.dropna()
    if s.empty:
        return {}
    last = s.iloc[-1]
    def ret(n):
        if len(s) <= n:
            return None
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0)
    ma200 = s.rolling(200).mean().iloc[-1]
    peak = s.rolling(252).max().iloc[-1]
    r = rets.dropna()
    vol20 = float(r.tail(20).std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) >= 20 else None
    vol60 = float(r.tail(60).std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) >= 60 else None
    ytd_start = s[s.index >= "2026-01-01"]
    ytd = float(s.iloc[-1] / ytd_start.iloc[0] - 1.0) if len(ytd_start) > 1 else None
    return {
        "last_date": str(s.index[-1].date()),
        "last": float(last),
        "ret_21d": ret(21),
        "ret_63d": ret(63),
        "ret_126d": ret(126),
        "ret_252d": ret(252),
        "ytd_2026": ytd,
        "vol_20d": vol20,
        "vol_60d": vol60,
        "above_ma200": bool(last > ma200) if pd.notna(ma200) else None,
        "dd_from_252d_high": float(last / peak - 1.0) if pd.notna(peak) and peak else None,
    }


def correlation_block(sims: dict[str, pd.DataFrame], names: list[str]) -> dict:
    df = pd.DataFrame({k: sims[k].loc[COMMON_START:COMMON_END, "net"] for k in names})
    corr = df.corr()
    return {
        "index": names,
        "matrix": [[None if pd.isna(corr.loc[a, b]) else round(float(corr.loc[a, b]), 3) for b in names] for a in names],
    }


def make_charts(sims: dict[str, pd.DataFrame], order: list[str]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in order:
        r = sims[name].loc[COMMON_START:COMMON_END, "net"].dropna()
        eq = (1 + r).cumprod()
        ax.plot(eq.index, eq.values, label=name, linewidth=1.2)
    ax.set_title("Net equity curves (common window, after 8bp/side)")
    ax.set_ylabel("Growth of 1")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = ARTIFACT_DIR / "tsr_20261007_equity.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    paths.append(str(p1))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in order:
        r = sims[name].loc[COMMON_START:COMMON_END, "net"].dropna()
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, label=name, linewidth=1.0)
    ax.set_title("Drawdown (common window, net)")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = ARTIFACT_DIR / "tsr_20261007_drawdown.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    paths.append(str(p2))
    return paths


def fmt_pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), 6)


def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


def main() -> None:
    self_check_engine()
    bundle = load_market_data()
    panel = prepare_panels(bundle)
    px = panel["px"]
    rets = panel["rets"].fillna(0.0)
    # 保留真实 NaN 用于信号，收益里首日 0 只影响建仓前
    # 重新取未填充的收益：停牌填充价会产生 0 收益，可接受；但 fillna(0) 会把 IPO 前抹成 0。
    raw_rets = panel["px"].pct_change()
    rets = raw_rets.copy()
    adv = panel["adv20"]
    cols = list(px.columns)

    builders = {
        "S1_dual_momentum": lambda: build_s1(px, adv, 252),
        "S2_tsmom": lambda: build_s2(px, adv, 252, 21, 0.25),
        "S3_sector_momentum": lambda: build_s3(px, adv, 126, 252, 3),
        "S4_reversal": lambda: build_s4(px, adv, 5),
        "S5_lowvol_trend": lambda: build_s5(px, raw_rets, adv, 60, 200, 3),
        "S6_donchian": lambda: build_s6(px, panel["high"], panel["low"], 55, 20),
        "S7_size_rotation": lambda: build_s7(px, 60, 1.0),
        "S8_vol_manage": lambda: build_s8(px, raw_rets, 0.15, 20, 5),
        "S9_turn_of_month": lambda: build_s9(px, 2, 3),
    }
    bench_builders = {
        "B0_csi300": lambda: build_buy_hold(px, {CSI300: 1.0}),
        "B1_60_40": lambda: build_balanced(px, 0.60),
        "B2_ew_tsmom_universe": lambda: build_equal_weight(px, adv, TSMOM_UNIVERSE, 60),
        "B3_ew_sector": lambda: build_equal_weight(px, adv, SECTOR_UNIVERSE, 60),
        "B4_50_50_size": lambda: build_balanced_size(px),
        # 下面两个是看到 S1 大量持有黄金之后补的解释性基准，不能当作事前策略。
        "B5_static_ew_3asset_POSTHOC": lambda: build_static_mix(
            px, {CSI300: 1.0 / 3.0, GOLD: 1.0 / 3.0, BOND: 1.0 / 3.0}
        ),
        "B6_permanent_25_25_50_POSTHOC": lambda: build_static_mix(
            px, {CSI300: 0.25, GOLD: 0.25, BOND: 0.50}
        ),
    }

    sims = {}
    for name, fn in {**bench_builders, **builders}.items():
        target = fn()
        sims[name] = run_one(name, target, rets.fillna(0.0), COST_PRIMARY)
        print(f"ran {name}", file=sys.stderr)

    bench_series = {
        "B0_csi300": sims["B0_csi300"]["net"],
        "B1_60_40": sims["B1_60_40"]["net"],
        "B2_ew_tsmom_universe": sims["B2_ew_tsmom_universe"]["net"],
        "B3_ew_sector": sims["B3_ew_sector"]["net"],
        "B4_50_50_size": sims["B4_50_50_size"]["net"],
        "B5_static_ew_3asset_POSTHOC": sims["B5_static_ew_3asset_POSTHOC"]["net"],
        "B6_permanent_25_25_50_POSTHOC": sims["B6_permanent_25_25_50_POSTHOC"]["net"],
    }
    relevant = {
        "S1_dual_momentum": ["B0_csi300", "B1_60_40", "B5_static_ew_3asset_POSTHOC"],
        "S2_tsmom": ["B0_csi300", "B2_ew_tsmom_universe"],
        "S3_sector_momentum": ["B0_csi300", "B3_ew_sector"],
        "S4_reversal": ["B0_csi300", "B3_ew_sector"],
        "S5_lowvol_trend": ["B0_csi300", "B3_ew_sector"],
        "S6_donchian": ["B0_csi300", "B1_60_40"],
        "S7_size_rotation": ["B0_csi300", "B4_50_50_size"],
        "S8_vol_manage": ["B0_csi300", "B1_60_40"],
        "S9_turn_of_month": ["B0_csi300", "B1_60_40"],
    }

    strategies = {}
    for name in builders:
        packed = pack_strategy(sims[name], {k: bench_series[k] for k in relevant[name]})
        strategies[name] = packed

    benchmarks = {}
    for name in bench_builders:
        benchmarks[name] = pack_strategy(sims[name], {"B0_csi300": bench_series["B0_csi300"]})

    # 成本压力与参数邻域：只记录，不用于改写主策略
    sensitivity = {}
    sensitivity["cost"] = {}
    for name, fn in builders.items():
        target = fn()
        sensitivity["cost"][name] = {}
        for cost in COST_STRESS:
            sim = run_one(name, target, rets.fillna(0.0), cost)
            sensitivity["cost"][name][f"{cost:.4f}"] = {
                "common": slice_stats(sim, COMMON_START, COMMON_END),
                "out_of_sample": slice_stats(sim, OOS_START, COMMON_END),
            }

    grids = {
        "S1_dual_momentum": [("lookback", lb, build_s1(px, adv, lb)) for lb in (126, 252, 378)],
        "S3_sector_momentum": [("lookback", lb, build_s3(px, adv, lb, 252, 3)) for lb in (63, 126, 252)],
        "S4_reversal": [("hold_days", h, build_s4(px, adv, h)) for h in (3, 5, 10)],
        "S6_donchian": [
            ("entry_exit", f"{e}/{x}", build_s6(px, panel["high"], panel["low"], e, x))
            for e, x in ((20, 10), (55, 20), (100, 40))
        ],
        "S7_size_rotation": [
            ("window_threshold", f"{w}/{t}", build_s7(px, w, t))
            for w, t in ((40, 1.0), (60, 1.0), (60, 1.5), (120, 1.0))
        ],
        "S8_vol_manage": [
            ("target_vol", tv, build_s8(px, raw_rets, tv, 20, 5))
            for tv in (0.10, 0.15, 0.20)
        ],
        "S2_tsmom": [
            ("lookback_skip", f"{lb}-{sk}", build_s2(px, adv, lb, sk, 0.25))
            for lb, sk in ((126, 10), (252, 21), (378, 21))
        ],
        "S5_lowvol_trend": [
            ("vol_ma", f"{v}/{m}", build_s5(px, raw_rets, adv, v, m, 3))
            for v, m in ((40, 200), (60, 200), (60, 100), (120, 200))
        ],
        "S9_turn_of_month": [
            ("last_first", f"{a}/{b}", build_s9(px, a, b))
            for a, b in ((1, 1), (2, 3), (3, 5))
        ],
    }
    sensitivity["params"] = {}
    for name, runs in grids.items():
        sensitivity["params"][name] = []
        for label, value, target in runs:
            sim = run_one(name, target, rets.fillna(0.0), COST_PRIMARY)
            common = slice_stats(sim, COMMON_START, COMMON_END)
            oos = slice_stats(sim, OOS_START, COMMON_END)
            sensitivity["params"][name].append(
                {
                    "param": label,
                    "value": value,
                    "primary": (name == "S1_dual_momentum" and value == 252)
                    or (name == "S2_tsmom" and value == "252-21")
                    or (name == "S3_sector_momentum" and value == 126)
                    or (name == "S4_reversal" and value == 5)
                    or (name == "S5_lowvol_trend" and value == "60/200")
                    or (name == "S6_donchian" and value == "55/20")
                    or (name == "S7_size_rotation" and value == "60/1.0")
                    or (name == "S8_vol_manage" and value == 0.15)
                    or (name == "S9_turn_of_month" and value == "2/3"),
                    "cagr": common["cagr"],
                    "sharpe_rf0": common["sharpe_rf0"],
                    "max_drawdown": common["max_drawdown"],
                    "oos_cagr": oos["cagr"],
                    "oos_sharpe_rf0": oos["sharpe_rf0"],
                    "oos_max_drawdown": oos["max_drawdown"],
                }
            )

    # 2019 之后才上市的半导体/酒，仅作为 S3 的样本外宇宙扩展，不替换主结果
    late_universe = SECTOR_UNIVERSE + LATE_ASSETS
    late_target = build_s3(px, adv, 126, 252, 3, universe=late_universe)
    late_sim = run_one("S3_late", late_target, rets.fillna(0.0), COST_PRIMARY)
    late_stats = {
        "window": "2020-01-02 to 2026-09-30",
        "note": "半导体ETF与酒ETF加入行业宇宙后的对照，不是主策略。",
        "stats": slice_stats(late_sim, "2020-01-02", COMMON_END),
    }

    regime_assets = CORE_ASSETS + LATE_ASSETS
    regime = {t: snapshot_asset(px[t], raw_rets[t]) for t in regime_assets}
    macro_snap = {}
    for t, series in panel["macro"].items():
        macro_snap[t] = snapshot_asset(series, series.pct_change())

    # 数据质量：极端日收益
    flags = []
    for col, meta in panel["quality"].items():
        if meta["max_abs_1d"] and meta["max_abs_1d"] > 0.12:
            flags.append({"ticker": col, "max_abs_1d": meta["max_abs_1d"]})

    chart_names = [
        "B0_csi300",
        "B1_60_40",
        "B5_static_ew_3asset_POSTHOC",
        "S1_dual_momentum",
        "S2_tsmom",
        "S9_turn_of_month",
        "S6_donchian",
        "S8_vol_manage",
    ]
    charts = make_charts(sims, chart_names)
    corr = correlation_block(
        sims,
        [
            "B0_csi300",
            "S1_dual_momentum",
            "S2_tsmom",
            "S3_sector_momentum",
            "S4_reversal",
            "S5_lowvol_trend",
            "S6_donchian",
            "S7_size_rotation",
            "S8_vol_manage",
            "S9_turn_of_month",
        ],
    )
    yearly = yearly_frame(
        {
            k: sims[k]
            for k in [
                "B0_csi300",
                "B1_60_40",
                "B5_static_ew_3asset_POSTHOC",
                "B6_permanent_25_25_50_POSTHOC",
                *builders.keys(),
            ]
        }
    )

    # 主比较窗口的暴露与换手已经在 stats 里
    result = {
        "generated_for": "2026-10-07",
        "timezone_context": "Asia/Shanghai",
        "price_source": "A-share ETFs: Tencent qfq daily bars; macro indices: Yahoo Finance",
        "common_window": [COMMON_START, COMMON_END],
        "oos_window": [OOS_START, COMMON_END],
        "cost_per_side_primary": COST_PRIMARY,
        "adv_min_cny": ADV_MIN_CNY,
        "ashare_last_date": str(panel["master"].max().date()),
        "ashare_first_date": str(panel["master"].min().date()),
        "n_ashare_days": int(len(panel["master"])),
        "data_quality": panel["quality"],
        "extreme_move_flags": flags,
        "regime": regime,
        "macro": macro_snap,
        "strategies": strategies,
        "benchmarks": benchmarks,
        "sensitivity": sensitivity,
        "s3_late_universe": late_stats,
        "yearly": yearly,
        "correlation": corr,
        "charts": charts,
        "assumptions": {
            "execution": "signal at close t, trade at close t, earn return on t+1",
            "costs": "8bp per side on sum of absolute weight changes; ETF stamp duty assumed 0",
            "no_leverage": True,
            "defensive_asset": BOND,
            "missing_oct_2026_ashare_bars": "not forward-filled",
            "rf_in_sharpe": 0.0,
        },
    }
    sanitized = sanitize(result)
    JSON_PATH.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(JSON_PATH)
    # 简表打印到 stdout，便于人工复核
    print("\nCOMMON WINDOW")
    header = f"{'name':24} {'CAGR':7} {'Sharpe':7} {'MaxDD':8} {'Sortino':8} {'Exp':6} {'Turn':7}"
    print(header)
    for name in ["B0_csi300", "B1_60_40", "B2_ew_tsmom_universe", "B3_ew_sector", "B4_50_50_size", *builders]:
        block = benchmarks[name]["common"] if name in benchmarks else strategies[name]["common"]
        print(
            f"{name:24} {block['cagr']*100:6.2f}% {block['sharpe_rf0']:7.2f} {block['max_drawdown']*100:7.1f}% "
            f"{block['sortino_rf0']:8.2f} {block['avg_exposure']*100:5.1f}% {block['ann_turnover']:7.2f}"
        )


def build_balanced_size(px: pd.DataFrame) -> pd.DataFrame:
    cols = list(px.columns)
    target = empty_target(px.index, cols)
    for dt in month_ends(px.index):
        if not history_ready(px, CSI300, dt, 2) or not history_ready(px, CSI500, dt, 2):
            continue
        target.loc[dt] = row_from_weights(cols, {CSI300: 0.5, CSI500: 0.5})
    return target


if __name__ == "__main__":
    main()
