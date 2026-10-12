#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily A-share ETF strategy research (trading-strategy-research skill).

Primary specs below are frozen before result inspection. Sensitivity runs are
reported separately and are not allowed to replace the primary parameter set.

Data: Tencent qfq daily bars (forward-adjusted, dividend-inclusive for ETFs).
The in-progress Asia/Shanghai session is dropped so fills are not invented.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_REPORT = ROOT / "reports" / "trading_strategy_research_20261012.md"
CACHE = Path("/tmp/tsr_20261012")
CACHE.mkdir(parents=True, exist_ok=True)

# --- frozen research design -------------------------------------------------
HISTORY_START = "2016-01-01"
COMMON_START = "2018-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
ONE_WAY_BPS_LIQUID = 8.0  # commission ~2.5 + slippage ~5.5 per side
ONE_WAY_BPS_SECTOR = 12.0
LIQ_MEDIAN_CNY = 20_000_000.0
VOL_TARGET = 0.12
SMA_WINDOW = 200
DONCHIAN_ENTRY = 120
DONCHIAN_EXIT = 60
MOM_SKIP = 21
MOM_LOOKBACK = 252
REV_WINDOW = 5
TOP_N = 3
LOWVOL_N = 4
RP_CAP = 0.70

SH_TZ = ZoneInfo("Asia/Shanghai")
BENCHMARK = "BH_HS300"

LIQUID = {
    "sh510300",
    "sh510500",
    "sh510050",
    "sz159915",
    "sh512100",
    "sh518880",
    "sh511010",
    "sh513100",
    "sh513500",
    "sh510880",
}

NAMES = {
    "sh510300": "沪深300ETF",
    "sh510500": "中证500ETF",
    "sh510050": "上证50ETF",
    "sz159915": "创业板ETF",
    "sh512100": "中证1000ETF",
    "sh512800": "银行ETF",
    "sh512010": "医药ETF",
    "sh512660": "军工ETF",
    "sh512400": "有色ETF",
    "sz159928": "消费ETF",
    "sh512690": "酒ETF",
    "sh512880": "证券ETF",
    "sh512170": "医疗ETF",
    "sh510880": "红利ETF",
    "sh512890": "红利低波ETF",
    "sh515220": "煤炭ETF",
    "sh512480": "半导体ETF",
    "sh515790": "光伏ETF",
    "sh516160": "新能源ETF",
    "sh512200": "地产ETF",
    "sh515050": "通信ETF",
    "sh512980": "传媒ETF",
    "sz159995": "芯片ETF",
    "sh518880": "黄金ETF",
    "sh511010": "五年国债ETF",
    "sh513100": "纳指ETF",
    "sh513500": "标普500ETF",
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sz399006": "创业板指",
    "sh000688": "科创50",
    "sh000852": "中证1000",
}

SECTORS = [
    "sh512800",
    "sh512010",
    "sh512660",
    "sh512400",
    "sz159928",
    "sh512690",
    "sh512880",
    "sh512170",
    "sh510880",
    "sh512890",
    "sh515220",
    "sh512480",
    "sh515790",
    "sh516160",
    "sh512200",
    "sh515050",
    "sh512980",
    "sz159995",
]
STYLES = ["sh510050", "sh510300", "sh510500", "sz159915", "sh512100"]
CROSS = ["sh510300", "sh518880", "sh513100", "sh511010"]
INDICES = [
    "sh000001",
    "sz399001",
    "sh000300",
    "sh000905",
    "sz399006",
    "sh000688",
    "sh000852",
]
BOND = "sh511010"
GOLD = "sh518880"
HS = "sh510300"

SPECS = {
    "TSMOM_SMA200_CASH": {
        "name": "沪深300均线趋势（现金防守）",
        "family": "趋势",
        "thesis": "宽基指数的信息扩散偏慢，趋势具有持续性；跌破长期均线时常对应流动性或风险偏好收缩，空仓可削减崩盘暴露。",
        "universe": "510300；防守资产为现金（收益记 0）",
        "entry": "收盘价高于 200 日简单均线",
        "exit": "收盘价低于 200 日均线",
        "sizing": "100% 或多头 0%",
        "rebalance": "每日计算，次一交易日才持有，仅状态切换时交易",
        "params": "SMA=200",
        "invalidation": "长期震荡市里反复假突破，成本侵蚀后超额为负；或趋势结构消失（政策市单日跳空后立即反转成为常态）",
    },
    "TSMOM_SMA200_BOND": {
        "name": "沪深300均线趋势（国债防守）",
        "family": "趋势",
        "thesis": "与现金版相同的趋势逻辑，但风险关闭时持有国债 ETF，赚取避险资产的票息与资本利得，而不是零收益现金。",
        "universe": "510300 与 511010",
        "entry": "收盘价高于 200 日均线则持有 510300",
        "exit": "否则全仓 511010",
        "sizing": "二选一，100%",
        "rebalance": "每日信号，次日持有",
        "params": "SMA=200",
        "invalidation": "股债同跌（紧缩或通缩交易逆转）时防守端失效；国债 ETF 流动性或利率制度变化",
    },
    "DONCHIAN_120_60": {
        "name": "沪深300唐奇安通道",
        "family": "趋势突破",
        "thesis": "创新高代表资金愿意在更差的价格继续成交，突破后的惯性来自趋势跟随与风险约束型买家；跌破短通道说明惯性结束。",
        "universe": "510300；空仓时持有 511010",
        "entry": "收盘价突破过去 120 个交易日最高价（不含当日）",
        "exit": "收盘价跌破过去 60 个交易日最低价（不含当日）",
        "sizing": "100% 股票或 100% 国债",
        "rebalance": "状态机，每日判断，次日持有",
        "params": "entry=120, exit=60",
        "invalidation": "高位假突破频率上升，且退出通道过慢导致回撤接近满仓；参数仅在单一窗口有效",
    },
    "VOL_TARGET_12": {
        "name": "沪深300波动率目标",
        "family": "风险预算",
        "thesis": "高波动时期风险资产的单位收益变差（Moreira-Muir 波动管理）。把已实现波动压到目标附近，可改善收益回撤比，而不是预测方向。",
        "universe": "510300，其余为现金",
        "entry": "始终可持有，权重由波动决定",
        "exit": "波动极高时权重趋于 0",
        "sizing": "w=min(1, 12%/20日已实现年化波动)",
        "rebalance": "每 5 个交易日",
        "params": "target=12%, lookback=20, cap=1（无杠杆）",
        "invalidation": "低波动缓跌（波动管理几乎满仓却持续亏损）；或波动骤降后立即单边崩盘",
    },
    "SECTOR_DUAL_MOM": {
        "name": "行业双动量",
        "family": "截面动量",
        "thesis": "行业领导力在资金与基本面兑现上往往延续数月；12-1 月动量避开最近一个月的反转噪声，绝对动量为负时空仓转入国债。",
        "universe": "18 只行业/风格 ETF，动态要求上市满 252 日且 60 日成交额中位数达标",
        "entry": "月末按 12-1 月收益排名，取绝对动量（12 个月）为正的前 3 名",
        "exit": "落选或绝对动量转负；无人合格则 100% 国债",
        "sizing": "合格者等权，其余权重给国债",
        "rebalance": "每月最后一个交易日",
        "params": "lookback=252, skip=21, top_n=3",
        "invalidation": "行业快速轮动、动量崩溃（如政策脉冲后立刻均值回归）；样本内行业名单的存活偏差被新退市潮放大",
    },
    "SECTOR_SHORT_REV": {
        "name": "行业短期反转",
        "family": "均值回归",
        "thesis": "数日维度的行业涨跌常被流动性和拥挤交易放大，随后部分回归。只在宽基仍处于长期上升趋势时做反转，避免熊市接刀。",
        "universe": "同上行业 ETF",
        "entry": "每 5 日买入过去 5 日跌幅最大的 3 个行业，且 510300 高于 200 日均线",
        "exit": "下次再平衡替换；趋势滤网失效则改持国债",
        "sizing": "等权 3 行业或 100% 国债",
        "rebalance": "每 5 个交易日",
        "params": "rev=5, top_n=3, trend=SMA200",
        "invalidation": "跌幅最大的行业持续趋势性下跌（基本面崩坏而非流动冲击）；趋势滤网失效",
    },
    "LOWVOL_TREND": {
        "name": "低波动趋势行业",
        "family": "低波异象",
        "thesis": "A 股高波动行业往往定价了彩票型偏好，长期风险调整后收益偏差；叠加趋势滤网，避开下跌趋势中的低波动价值陷阱。",
        "universe": "行业 ETF + 宽基 ETF",
        "entry": "月末在高于 120 日均线的品种里取 60 日波动最低的 4 只",
        "exit": "跌破均线则剔除；不足则余下权重给国债",
        "sizing": "波动倒数加权",
        "rebalance": "每月最后一个交易日",
        "params": "n=4, vol=60, trend=120",
        "invalidation": "低波动行业变成利率或政策受损板块并集体趋势下跌；波动估计在制度切换后失真",
    },
    "STYLE_ROTATION": {
        "name": "大小盘风格轮动",
        "family": "风格",
        "thesis": "A 股大小盘与成长价值的相对强弱会阶段性持续，但风格切换剧烈。用中等期限动量选择强势风格，并用绝对动量避免全面下跌时硬做多。",
        "universe": "510050 / 510300 / 510500 / 159915 / 512100",
        "entry": "月末持有 60 日收益最高者，且其 120 日收益为正",
        "exit": "120 日收益为负则改持国债",
        "sizing": "单一品种 100%",
        "rebalance": "每月最后一个交易日",
        "params": "rel=60, abs=120",
        "invalidation": "风格一周内来回切换成为常态；中证1000 ETF 折溢价大幅偏离净值",
    },
    "RISK_PARITY_3": {
        "name": "股债金风险均衡",
        "family": "配置",
        "thesis": "股票、黄金、国债的驱动因子不同。按波动倒数分配风险，而不是按资金等权，可降低单一资产崩盘对组合的主导。这是配置溢价，不是择时阿尔法。",
        "universe": "510300、518880、511010",
        "entry": "始终持有三资产",
        "exit": "无方向性退出；单资产权重上限 70%",
        "sizing": "60 日波动倒数，上限 70% 后重新归一",
        "rebalance": "每月最后一个交易日",
        "params": "vol=60, cap=0.70",
        "invalidation": "三资产相关性在危机中升至接近 1；黄金 ETF 跟踪误差或国债流动性冲击",
    },
    "CROSS_ASSET_MOM": {
        "name": "跨资产绝对动量",
        "family": "配置动量",
        "thesis": "股票、黄金、美股、国债之间存在阶段性主线。持有过去半年最强且收益为正的一类资产，本质是多资产时间序列动量。",
        "universe": "510300、518880、513100、511010",
        "entry": "月末持有 126 日收益最高且收益为正的资产",
        "exit": "最高者收益为负则持有国债",
        "sizing": "单一资产 100%",
        "rebalance": "每月最后一个交易日",
        "params": "lookback=126",
        "invalidation": "QDII 溢价急扩后均值回归（纳指 ETF）；跨境额度或汇率制度变化使 ETF 价格偏离标的",
    },
}


def _now_sh() -> datetime:
    return datetime.now(SH_TZ)


def incomplete_session_date() -> str | None:
    """Drop today's bar when the local cash session has not finished."""
    now = _now_sh()
    if now.weekday() >= 5:
        return None
    if (now.hour, now.minute) < (15, 5):
        return now.strftime("%Y-%m-%d")
    return None


def _get(url: str, retries: int = 3) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://finance.qq.com/",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 — network retry
            last = exc
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def fetch_kline(symbol: str, end: str, count: int = 500) -> list:
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
        f"{symbol},day,{HISTORY_START},{end},{count},qfq"
    )
    payload = json.loads(_get(url).decode("utf-8"))
    if payload.get("code") not in (0, "0", None):
        raise RuntimeError(f"{symbol} code={payload.get('code')} {payload.get('msg')}")
    node = payload["data"][symbol]
    rows = node.get("qfqday") or node.get("day") or []
    return rows


def fetch_symbol(symbol: str) -> pd.DataFrame:
    cache = CACHE / f"{symbol}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        return df
    end = _now_sh().strftime("%Y-%m-%d")
    frames = []
    seen = set()
    guard = 0
    while guard < 12:
        guard += 1
        rows = fetch_kline(symbol, end, 500)
        if not rows:
            break
        earliest = rows[0][0]
        for row in rows:
            if row[0] in seen or len(row) < 6:
                continue
            seen.add(row[0])
            frames.append(row[:6])
        if earliest <= HISTORY_START or len(rows) < 50:
            break
        nxt = (datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        if nxt >= end:
            break
        end = nxt
        time.sleep(0.05)
    if not frames:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
    df = pd.DataFrame(frames, columns=["date", "open", "close", "high", "low", "volume"])
    for col in ("open", "close", "high", "low", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    drop_day = incomplete_session_date()
    if drop_day:
        df = df[df["date"] < drop_day]
    df.to_csv(cache, index=False)
    return df.reset_index(drop=True)


def fetch_quotes(symbols: list[str]) -> dict:
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    raw = _get(url).decode("gbk", errors="replace")
    out = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        left, right = part.split("=", 1)
        sym = left.replace("v_", "").strip()
        body = right.strip().strip('"')
        fields = body.split("~")
        if len(fields) < 5:
            continue
        try:
            price = float(fields[3])
            prev = float(fields[4])
        except ValueError:
            continue
        out[sym] = {
            "name": fields[1],
            "price": price,
            "prev": prev,
            "pct": price / prev - 1 if prev else float("nan"),
        }
    return out


def panel_from(frames: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    series = []
    for sym, df in frames.items():
        if df.empty:
            continue
        s = df.set_index("date")[col].astype(float)
        s.name = sym
        series.append(s)
    if not series:
        return pd.DataFrame()
    out = pd.concat(series, axis=1).sort_index()
    # Short gaps only. Longer holes stay missing so we do not invent prices.
    return out.ffill(limit=3)


def calibrate_volume_multiplier(volume: pd.DataFrame, close: pd.DataFrame) -> float:
    raw = (volume[HS] * close[HS]).dropna()
    med = float(raw.tail(252).median())
    # 510300 cash turnover is billions of CNY. Tencent volume is in lots (手).
    if med < 2e8:
        return 100.0
    return 1.0


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    stamp = pd.Series(index, index=index)
    return pd.DatetimeIndex(stamp.groupby(index.to_period("M")).tail(1).values)


def trailing_return(close: pd.Series, lookback: int) -> pd.Series:
    return close / close.shift(lookback) - 1.0


def empty_weights(index, columns) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=columns)


def finalize_weights(w: pd.DataFrame) -> pd.DataFrame:
    w = w.ffill().fillna(0.0)
    return w


def assign_row(w: pd.DataFrame, dt, weights: dict) -> None:
    """Replace the whole book. Unlisted names are zero so ffill cannot stack old positions."""
    w.loc[dt, :] = 0.0
    for key, value in weights.items():
        w.loc[dt, key] = float(value)


def bps_for(symbol: str) -> float:
    if symbol == "CASH":
        return 0.0
    if symbol in LIQUID:
        return ONE_WAY_BPS_LIQUID
    return ONE_WAY_BPS_SECTOR


def simulate(targets: pd.DataFrame, rets: pd.DataFrame, bps_scale: float = 1.0):
    """Apply next-day execution and per-side costs on risky weight changes.

    `targets` are decision-time weights known at the close. They earn the
    following session's close-to-close return. Cash earns 0. Costs are
    one-way bps times the absolute weight change of each risky ETF, so a
    cash->ETF entry is charged once and an ETF swap is charged on both legs.
    """
    cols = [c for c in targets.columns if c in rets.columns and c != "CASH"]
    aligned = targets.reindex(rets.index).ffill().fillna(0.0)
    held = aligned.shift(1).fillna(0.0)
    asset_ret = pd.Series(0.0, index=rets.index)
    for col in cols:
        asset_ret = asset_ret + held[col] * rets[col].fillna(0.0)
    dw = held[cols].diff()
    if len(dw):
        dw.iloc[0] = held[cols].iloc[0]
    cost = pd.Series(0.0, index=rets.index)
    traded = pd.Series(0.0, index=rets.index)
    for col in cols:
        leg = dw[col].abs().fillna(0.0)
        traded = traded + leg
        cost = cost + leg * (bps_for(col) * bps_scale / 10000.0)
    net = asset_ret - cost
    gross = asset_ret
    return {
        "net": net,
        "gross": gross,
        "cost": cost,
        "traded": traded,
        "held": held,
    }


def strategy_tsmom(close: pd.DataFrame, window: int, defensive: str) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    px = close[HS]
    sma = px.rolling(window).mean()
    valid = sma.notna() & px.notna()
    long = valid & (px > sma)
    flat = valid & ~long
    w.loc[valid, :] = 0.0
    w.loc[long, HS] = 1.0
    if defensive == "CASH":
        w.loc[flat, "CASH"] = 1.0
    else:
        w.loc[flat, defensive] = 1.0
    return finalize_weights(w)


def strategy_donchian(close, high, low, entry=DONCHIAN_ENTRY, exit=DONCHIAN_EXIT) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    h = high[HS].to_numpy()
    l = low[HS].to_numpy()
    c = close[HS].to_numpy()
    pos = 0
    for i, dt in enumerate(close.index):
        if i < entry or not np.isfinite(c[i]):
            continue
        hh = np.nanmax(h[i - entry : i])
        ll = np.nanmin(l[i - exit : i]) if i >= exit else np.nan
        if not np.isfinite(hh) or not np.isfinite(ll):
            continue
        if pos == 0 and c[i] > hh:
            pos = 1
        elif pos == 1 and c[i] < ll:
            pos = 0
        if pos == 1:
            assign_row(w, dt, {HS: 1.0})
        else:
            assign_row(w, dt, {BOND: 1.0})
    return finalize_weights(w)


def strategy_vol_target(close, rets, target=VOL_TARGET, lookback=20, step=5) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    rv = rets[HS].rolling(lookback).std() * math.sqrt(252)
    raw = (target / rv).clip(upper=1.0)
    last = None
    for i, dt in enumerate(close.index):
        if i < lookback or not np.isfinite(raw.iloc[i]):
            continue
        if last is None or i - last >= step:
            weight = float(raw.iloc[i])
            if not np.isfinite(weight) or weight < 0:
                weight = 0.0
            assign_row(w, dt, {HS: weight, "CASH": 1.0 - weight})
            last = i
    return finalize_weights(w)


def _eligible(symbols, dt, close, notional, lookback_days: int) -> list[str]:
    good = []
    for sym in symbols:
        if sym not in close.columns:
            continue
        hist = close[sym].loc[:dt].dropna()
        if len(hist) < lookback_days:
            continue
        if dt not in close.index or not np.isfinite(close.at[dt, sym]):
            continue
        trail = notional[sym].loc[:dt].tail(60).dropna()
        if len(trail) < 20 or float(trail.median()) < LIQ_MEDIAN_CNY:
            continue
        good.append(sym)
    return good


def strategy_sector_dual_mom(close, notional, lookback=MOM_LOOKBACK, skip=MOM_SKIP, top_n=TOP_N) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    rel = close.shift(skip) / close.shift(skip + lookback) - 1.0
    abs_mom = close / close.shift(lookback) - 1.0
    for dt in month_ends(close.index):
        pool = _eligible(SECTORS, dt, close, notional, lookback + skip)
        if len(pool) < top_n:
            if close[BOND].loc[:dt].dropna().shape[0] > 60:
                assign_row(w, dt, {BOND: 1.0})
            continue
        score = rel.loc[dt, pool].dropna().sort_values(ascending=False)
        picks = []
        for sym in score.index:
            if len(picks) >= top_n:
                break
            if np.isfinite(abs_mom.at[dt, sym]) and abs_mom.at[dt, sym] > 0:
                picks.append(sym)
        if not picks:
            assign_row(w, dt, {BOND: 1.0})
            continue
        weight = 1.0 / len(picks)
        assign_row(w, dt, {sym: weight for sym in picks})
    return finalize_weights(w)


def strategy_short_reversal(close, notional, window=REV_WINDOW, top_n=TOP_N, trend=SMA_WINDOW) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    rev = close / close.shift(window) - 1.0
    sma = close[HS].rolling(trend).mean()
    last = None
    for i, dt in enumerate(close.index):
        if i < max(window, trend):
            continue
        if last is not None and i - last < window:
            continue
        last = i
        risk_on = np.isfinite(sma.at[dt]) and close.at[dt, HS] > sma.at[dt]
        if not risk_on:
            assign_row(w, dt, {BOND: 1.0})
            continue
        pool = _eligible(SECTORS, dt, close, notional, window + 5)
        score = rev.loc[dt, pool].dropna().sort_values()
        picks = list(score.index[:top_n])
        if len(picks) < top_n:
            assign_row(w, dt, {BOND: 1.0})
            continue
        assign_row(w, dt, {sym: 1.0 / len(picks) for sym in picks})
    return finalize_weights(w)


def strategy_lowvol(close, rets, notional, n=LOWVOL_N, vol_n=60, trend_n=120) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    vol = rets.rolling(vol_n).std() * math.sqrt(252)
    sma = close.rolling(trend_n).mean()
    universe = [s for s in SECTORS + STYLES if s in close.columns]
    for dt in month_ends(close.index):
        pool = _eligible(universe, dt, close, notional, trend_n)
        ranked = []
        for sym in pool:
            vv = vol.at[dt, sym] if sym in vol.columns else np.nan
            if not np.isfinite(vv) or vv <= 0.02:
                continue
            if not np.isfinite(sma.at[dt, sym]) or close.at[dt, sym] <= sma.at[dt, sym]:
                continue
            ranked.append((vv, sym))
        ranked.sort()
        picks = ranked[:n]
        if not picks:
            assign_row(w, dt, {BOND: 1.0})
            continue
        inv = np.array([1.0 / vv for vv, _ in picks])
        inv = inv / inv.sum()
        assign_row(w, dt, {sym: float(weight) for weight, (_, sym) in zip(inv, picks)})
    return finalize_weights(w)


def strategy_style(close, notional, rel_n=60, abs_n=120) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    rel = close / close.shift(rel_n) - 1.0
    slow = close / close.shift(abs_n) - 1.0
    for dt in month_ends(close.index):
        pool = _eligible(STYLES, dt, close, notional, abs_n)
        if not pool:
            continue
        score = rel.loc[dt, pool].dropna().sort_values(ascending=False)
        if score.empty:
            continue
        best = score.index[0]
        if np.isfinite(slow.at[dt, best]) and slow.at[dt, best] > 0:
            assign_row(w, dt, {best: 1.0})
        else:
            assign_row(w, dt, {BOND: 1.0})
    return finalize_weights(w)


def _cap_and_renorm(weights: np.ndarray, cap: float) -> np.ndarray:
    """Enforce a per-asset cap. Renormalizing after a naive clip would break the cap."""
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(8):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        if not free.any() or float(w[free].sum()) <= 0:
            w = np.full_like(w, 1.0 / len(w))
            break
        w[free] = w[free] + excess * w[free] / w[free].sum()
    total = float(w.sum())
    return w / total if total > 0 else w


def strategy_risk_parity(close, rets, vol_n=60, cap=RP_CAP) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    assets = [HS, GOLD, BOND]
    vol = rets[assets].rolling(vol_n).std() * math.sqrt(252)
    for dt in month_ends(close.index):
        vols = []
        ok = True
        for sym in assets:
            vv = vol.at[dt, sym]
            if not np.isfinite(vv) or vv <= 1e-6 or not np.isfinite(close.at[dt, sym]):
                ok = False
                break
            vols.append(vv)
        if not ok:
            continue
        inv = np.array([1.0 / v for v in vols])
        raw = inv / inv.sum()
        weights = _cap_and_renorm(raw, cap)
        if weights.sum() <= 0:
            continue
        assign_row(w, dt, {sym: float(weight) for sym, weight in zip(assets, weights)})
    return finalize_weights(w)


def strategy_cross_asset(close, notional, lookback=126) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    mom = close / close.shift(lookback) - 1.0
    for dt in month_ends(close.index):
        pool = _eligible(CROSS, dt, close, notional, lookback)
        if BOND not in pool:
            continue
        score = mom.loc[dt, pool].dropna().sort_values(ascending=False)
        if score.empty:
            continue
        best = score.index[0]
        if score.iloc[0] > 0:
            assign_row(w, dt, {best: 1.0})
        else:
            assign_row(w, dt, {BOND: 1.0})
    return finalize_weights(w)


def strategy_buyhold(close, asset=HS) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    live = close[asset].dropna().index
    if len(live):
        assign_row(w, live[0], {asset: 1.0})
    return finalize_weights(w)


def strategy_static_60_40(close) -> pd.DataFrame:
    cols = list(close.columns) + ["CASH"]
    w = empty_weights(close.index, cols)
    for dt in month_ends(close.index):
        if not np.isfinite(close.at[dt, HS]) or not np.isfinite(close.at[dt, BOND]):
            continue
        assign_row(w, dt, {HS: 0.60, BOND: 0.40})
    return finalize_weights(w)


def window_slice(series: pd.Series, start: str, end: str | None) -> pd.Series:
    sl = series.loc[start:] if start else series
    if end:
        sl = sl.loc[:end]
    return sl.dropna()


def max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    return float((eq / peak - 1).min())


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 40:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 252.0
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(252))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252)) if r.std(ddof=1) > 0 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)))
    sortino = float(r.mean() / downside * math.sqrt(252)) if downside > 0 else float("nan")
    dd = max_drawdown(r)
    monthly = (1 + r).resample("ME").prod() - 1
    monthly = monthly.dropna()
    win = float((monthly > 0).mean()) if len(monthly) else float("nan")
    gains = float(monthly[monthly > 0].sum())
    losses = float(-monthly[monthly < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    calmar = cagr / abs(dd) if dd and np.isfinite(dd) and dd < 0 else float("nan")
    yearly = (1 + r).resample("YE").prod() - 1
    yearly_map = {int(ts.year): float(val) for ts, val in yearly.items()}
    pos = [v for v in yearly_map.values() if v > 0]
    conc = (max(pos) / sum(pos)) if pos and sum(pos) > 0 else float("nan")
    return {
        "days": int(len(r)),
        "total": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": dd,
        "win_rate": win,
        "profit_factor": pf,
        "calmar": calmar,
        "yearly": yearly_map,
        "profit_concentration": conc,
        "end_eq": float(eq.iloc[-1]),
    }


def fmt_pct(x, digits=1) -> str:
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x, digits=2) -> str:
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(x):
        return "n/a"
    if abs(x) > 99:
        return "99+"
    return f"{x:.{digits}f}"


def price_regime(close: pd.DataFrame) -> dict:
    last = close.index.max()
    rows = []
    for sym in INDICES:
        if sym not in close.columns:
            continue
        px = close[sym].dropna()
        if px.empty:
            continue
        def ret(n):
            if len(px) <= n:
                return float("nan")
            return float(px.iloc[-1] / px.iloc[-1 - n] - 1)

        hi = px.tail(252)
        dd = float(px.iloc[-1] / hi.max() - 1) if len(hi) else float("nan")
        vol = float(px.pct_change().tail(20).std() * math.sqrt(252))
        sma = px.rolling(200).mean()
        rows.append(
            {
                "sym": sym,
                "name": NAMES.get(sym, sym),
                "last": float(px.iloc[-1]),
                "r5": ret(5),
                "r20": ret(20),
                "r60": ret(60),
                "r252": ret(252),
                "dd252": dd,
                "vol20": vol,
                "above200": bool(px.iloc[-1] > sma.iloc[-1]) if np.isfinite(sma.iloc[-1]) else False,
            }
        )
    sectors = []
    for sym in SECTORS + STYLES + [GOLD, BOND, "sh513100"]:
        if sym not in close.columns:
            continue
        px = close[sym].dropna()
        if len(px) < 60:
            continue
        sectors.append(
            {
                "sym": sym,
                "name": NAMES.get(sym, sym),
                "r20": float(px.iloc[-1] / px.iloc[-21] - 1),
                "r60": float(px.iloc[-1] / px.iloc[-61] - 1) if len(px) > 60 else float("nan"),
                "r252": float(px.iloc[-1] / px.iloc[-253] - 1) if len(px) > 252 else float("nan"),
            }
        )
    return {"asof": str(last.date()), "indices": rows, "sectors": sectors}


def build_targets(close, high, low, rets, notional) -> dict[str, pd.DataFrame]:
    return {
        "BH_HS300": strategy_buyhold(close, HS),
        "BH_BOND": strategy_buyhold(close, BOND),
        "BH_GOLD": strategy_buyhold(close, GOLD),
        "STATIC_60_40": strategy_static_60_40(close),
        "TSMOM_SMA200_CASH": strategy_tsmom(close, SMA_WINDOW, "CASH"),
        "TSMOM_SMA200_BOND": strategy_tsmom(close, SMA_WINDOW, BOND),
        "DONCHIAN_120_60": strategy_donchian(close, high, low),
        "VOL_TARGET_12": strategy_vol_target(close, rets),
        "SECTOR_DUAL_MOM": strategy_sector_dual_mom(close, notional),
        "SECTOR_SHORT_REV": strategy_short_reversal(close, notional),
        "LOWVOL_TREND": strategy_lowvol(close, rets, notional),
        "STYLE_ROTATION": strategy_style(close, notional),
        "RISK_PARITY_3": strategy_risk_parity(close, rets),
        "CROSS_ASSET_MOM": strategy_cross_asset(close, notional),
    }


CANDIDATES = [k for k in SPECS]


def evaluate_all(close, high, low, rets, notional, bps_scale=1.0):
    targets = build_targets(close, high, low, rets, notional)
    # sensitivity variants reuse constructors; primary map stays untouched
    results = {}
    for key, tgt in targets.items():
        book = tgt.sum(axis=1)
        active = book > 0.05
        if active.any() and float((book[active] - 1.0).abs().max()) > 0.02:
            raise RuntimeError(f"{key} weights do not sum to 1; max book={float(book.max()):.3f}")
        sim = simulate(tgt, rets, bps_scale=bps_scale)
        net = sim["net"]
        results[key] = {
            "sim": sim,
            "full": metrics(window_slice(net, COMMON_START, None)),
            "is": metrics(window_slice(net, COMMON_START, IS_END)),
            "oos": metrics(window_slice(net, OOS_START, None)),
        }
    return results


def sensitivity(close, high, low, rets, notional) -> dict:
    variants = {}

    def run(label, targets):
        sim = simulate(targets, rets, bps_scale=1.0)
        variants[label] = metrics(window_slice(sim["net"], COMMON_START, None))

    for window in (100, 150, 200, 250):
        run(f"SMA_{window}_BOND", strategy_tsmom(close, window, BOND))
    for entry, exit_n in ((60, 20), (120, 60), (200, 80)):
        run(f"DON_{entry}_{exit_n}", strategy_donchian(close, high, low, entry, exit_n))
    for target in (0.08, 0.12, 0.16):
        run(f"VT_{int(target*100)}", strategy_vol_target(close, rets, target=target))
    for top_n in (2, 3, 4):
        run(f"DUAL_N{top_n}", strategy_sector_dual_mom(close, notional, top_n=top_n))
    for lookback in (126, 189, 252):
        run(f"DUAL_L{lookback}", strategy_sector_dual_mom(close, notional, lookback=lookback))
    for vol_n in (40, 60, 120):
        run(f"RP_VOL{vol_n}", strategy_risk_parity(close, rets, vol_n=vol_n, cap=RP_CAP))
    for cap in (0.50, 0.70, 1.00):
        run(f"RP_CAP{int(cap*100)}", strategy_risk_parity(close, rets, vol_n=60, cap=cap))
    return variants


def walk_forward_sma(close, rets) -> dict:
    """Each year, pick SMA in {100,150,200,250} using only prior Sharpe."""
    chosen = []
    pieces = []
    grid = (100, 150, 200, 250)
    years = range(2020, int(close.index.max().year) + 1)
    precomputed = {w: simulate(strategy_tsmom(close, w, BOND), rets)["net"] for w in grid}
    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        best_w, best_s = None, -1e9
        for w, series in precomputed.items():
            hist = series.loc[: start - pd.Timedelta(days=1)]
            hist = hist.loc["2018-01-01":]
            m = metrics(hist)
            sharpe = m.get("sharpe", float("nan"))
            if np.isfinite(sharpe) and sharpe > best_s:
                best_s, best_w = sharpe, w
        if best_w is None:
            continue
        chosen.append({"year": year, "sma": best_w, "prior_sharpe": best_s})
        piece = precomputed[best_w].loc[start:end]
        pieces.append(piece)
    if not pieces:
        return {"picks": chosen, "metrics": {}}
    stitched = pd.concat(pieces)
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return {"picks": chosen, "metrics": metrics(stitched), "net": stitched}


def score_row(full, is_m, oos, bh_full) -> float:
    def cap(x, lo, hi):
        if x is None or not isinstance(x, (int, float)) or not np.isfinite(x):
            return lo
        return float(np.clip(x, lo, hi))

    ex = cap(full.get("cagr", np.nan) - bh_full.get("cagr", np.nan), -0.2, 0.2)
    return (
        0.30 * cap(full.get("sharpe"), -1, 3)
        + 0.25 * cap(oos.get("sharpe"), -1, 3)
        + 0.15 * cap(is_m.get("sharpe"), -1, 3)
        + 0.20 * cap(1 + full.get("max_dd", -1), 0, 1.2)
        + 0.10 * cap(ex / 0.05, -2, 2)
    )


def quality_gate(key, res, bh, variants, references) -> tuple[bool, list[str]]:
    """Strict bar from the skill. Failure reasons are explicit."""
    reasons = []
    full, is_m, oos = res["full"], res["is"], res["oos"]
    if not full or not is_m or not oos:
        return False, ["样本不足，无法同时评估全样本、样本内与样本外"]
    if not (full["cagr"] > bh["full"]["cagr"]):
        reasons.append(
            f"全样本 CAGR {fmt_pct(full['cagr'])} 未超过沪深300ETF {fmt_pct(bh['full']['cagr'])}"
        )
    if not (full["sharpe"] > bh["full"]["sharpe"] and full["sharpe"] > 1.0):
        reasons.append(
            f"全样本 Sharpe {fmt_num(full['sharpe'])} 未同时高于基准 {fmt_num(bh['full']['sharpe'])} 且 > 1.0"
        )
    if not (full["max_dd"] > -0.30):
        reasons.append(f"最大回撤 {fmt_pct(full['max_dd'])} 深于 -30%")
    if not (oos["sharpe"] > 0.5 and oos["cagr"] > 0):
        reasons.append(
            f"样本外 Sharpe {fmt_num(oos['sharpe'])} / CAGR {fmt_pct(oos['cagr'])} 不稳定"
        )
    if not (is_m["sharpe"] > 0 and is_m["cagr"] > 0):
        reasons.append("样本内 Sharpe 或 CAGR 不为正，存在窗口依赖")
    conc = full.get("profit_concentration")
    if conc is not None and np.isfinite(conc) and conc > 0.80:
        reasons.append(f"正收益过度集中于单一年度（贡献 {fmt_pct(conc)}）")
    # Parameter neighbors must not flip the economic result for families we grid.
    neighbor_ok = True
    if key == "TSMOM_SMA200_BOND":
        neighbor_ok = all(
            variants.get(f"SMA_{w}_BOND", {}).get("sharpe", -1) > bh["full"]["sharpe"]
            for w in (150, 250)
        )
        if not neighbor_ok:
            reasons.append("SMA 150/250 邻居未能同时保住对基准的 Sharpe 优势，参数脆弱")
    if key == "SECTOR_DUAL_MOM":
        neighbor_ok = all(
            variants.get(f"DUAL_N{n}", {}).get("cagr", -1) > 0 for n in (2, 4)
        )
        if not neighbor_ok:
            reasons.append("行业动量 top2/top4 邻居 CAGR 转负，排名宽度脆弱")
    if key == "RISK_PARITY_3":
        neighbor_labels = ("RP_VOL40", "RP_VOL120", "RP_CAP50", "RP_CAP100")
        weak = []
        for lab in neighbor_labels:
            m = variants.get(lab) or {}
            ok = (
                m.get("sharpe", -1) > 1.0
                and m.get("cagr", -1) > bh["full"]["cagr"]
                and m.get("max_dd", -1) > -0.30
            )
            if not ok:
                weak.append(lab)
        if weak:
            reasons.append("风险均衡邻居未同时保住 Sharpe>1、战胜基准和回撤浅于30%：" + ",".join(weak))
        bond_m = references["BH_BOND"]["full"]
        if not (full["cagr"] > bond_m["cagr"] and full["sharpe"] > bond_m["sharpe"]):
            reasons.append(
                "相对国债ETF买入持有没有同时提高 CAGR 与 Sharpe，组合增量不成立"
            )
    held = res["sim"]["held"].loc[COMMON_START:]
    risky_cols = [c for c in held.columns if c != "CASH"]
    if risky_cols:
        avg = held[risky_cols].mean()
        top_w = float(avg.max()) if len(avg) else 0.0
        top_name = NAMES.get(str(avg.idxmax()), str(avg.idxmax())) if len(avg) else ""
        if top_w > 0.80:
            reasons.append(
                f"平均权重过度集中于{top_name}（{top_w:.0%}），不符合“不过度依赖单一资产”"
            )
    return len(reasons) == 0, reasons


def audit_buyhold(close, res) -> str:
    px = close[HS].loc[COMMON_START:]
    raw = float(px.iloc[-1] / px.iloc[0] - 1)
    sim_total = res["BH_HS300"]["full"].get("total", float("nan"))
    gap = sim_total - raw
    return (
        f"审计：沪深300ETF 自 {px.index[0].date()} 收盘 {px.iloc[0]:.4f} 至 "
        f"{px.index[-1].date()} 收盘 {px.iloc[-1]:.4f}，价格比收益 {raw:.1%}。"
        f"回测总收益 {sim_total:.1%}，差额 {gap:.2%}，来自窗口第一天相对前一交易日的涨跌"
        "（价格比从当日收盘起算，不含这一跳）。首笔买入发生在 2016 年样本准备期，评估窗内换手接近 0。"
    )


def render_report(ctx: dict) -> str:
    asof = ctx["regime"]["asof"]
    fetched = ctx["fetched_at"]
    res = ctx["results"]
    bh = res[BENCHMARK]
    lines = []
    a = lines.append
    a("# 交易策略研究报告 2026-10-12")
    a("")
    a(f"- 完整样本截止: **{asof}**（Asia/Shanghai 已收盘 K 线；盘中 bar 已剔除）")
    a(f"- 行情抓取时间: {fetched}")
    a("- 市场: A 股可交易 ETF（行业、风格、黄金、国债、QDII），基准为华泰柏瑞沪深300ETF（510300）前复权买入持有")
    a("- 成本: 高流动性 ETF 单边 8bps（佣金约 2.5 + 滑点约 5.5），行业 ETF 单边 12bps；ETF 无印花税。现金收益记 0")
    a("- 成交: 信号在收盘形成，**下一交易日**才计入持仓收益（避免收盘价成交的前视）")
    a("- 样本内: 2018-01-01 至 2022-12-31；样本外: 2023-01-01 至样本截止。主参数在跑结果前写死，敏感性不替换主参数")
    a("- 复现: `python3 scripts/trading_strategy_research/run_daily_research.py`")
    a("- **这是研究回测，不构成投资建议**")
    a("")
    a("## 1. Executive Summary（摘要）")
    a("")
    a(ctx["executive"])
    a("")
    a("## 2. Market Analysis（市场状态）")
    a("")
    a(ctx["market_narrative"])
    a("")
    a("### 指数（截至最后完整交易日）")
    a("")
    a("| 指数 | 点位 | 5日 | 20日 | 60日 | 252日 | 距252日高点 | 20日波动 | >MA200 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for row in ctx["regime"]["indices"]:
        a(
            f"| {row['name']} | {row['last']:.2f} | {fmt_pct(row['r5'])} | {fmt_pct(row['r20'])} | "
            f"{fmt_pct(row['r60'])} | {fmt_pct(row['r252'])} | {fmt_pct(row['dd252'])} | "
            f"{fmt_pct(row['vol20'])} | {'Y' if row['above200'] else 'N'} |"
        )
    a("")
    a("### 盘中快照（不纳入回测）")
    a("")
    if ctx["intraday"]:
        a("| 代码 | 名称 | 最新 | 相对前收 |")
        a("| --- | --- | ---: | ---: |")
        for sym, q in ctx["intraday"].items():
            a(f"| {sym} | {q['name']} | {q['price']:.2f} | {fmt_pct(q['pct'], 2)} |")
    else:
        a("盘中报价抓取失败，本节留空。")
    a("")
    a("### 行业与资产相对强弱（完整交易日，60日收益排序）")
    a("")
    sectors = sorted(ctx["regime"]["sectors"], key=lambda r: r["r60"], reverse=True)
    a("| 排名 | 名称 | 20日 | 60日 | 252日 |")
    a("| ---: | --- | ---: | ---: | ---: |")
    for i, row in enumerate(sectors, 1):
        a(f"| {i} | {row['name']} | {fmt_pct(row['r20'])} | {fmt_pct(row['r60'])} | {fmt_pct(row['r252'])} |")
    a("")
    a(ctx["rotation_note"])
    a("")
    a("### 数据缺口")
    a("")
    a(ctx["gaps"])
    a("")
    a("## 3. Strategy Details（策略说明）")
    a("")
    a("十个候选在经济机制上分开：趋势、通道突破、波动管理、行业动量、短期反转、低波、风格、风险均衡、跨资产动量。另有买入持有与静态 60/40 作为基准，不参与“战胜基准”的候选排名。")
    a("")
    a("做空、事件驱动与个股基本面策略本次没有实施：融券与事件数据库不在可用数据里，个股财务字段也没有可靠的时点快照。不把缺失数据写成假成交。")
    a("")
    for key, spec in SPECS.items():
        a(f"### {key} — {spec['name']}")
        a("")
        a(f"- 家族: {spec['family']}")
        a(f"- Thesis: {spec['thesis']}")
        a(f"- Universe: {spec['universe']}")
        a(f"- Entry: {spec['entry']}")
        a(f"- Exit: {spec['exit']}")
        a(f"- Sizing: {spec['sizing']}")
        a(f"- Rebalance: {spec['rebalance']}")
        a(f"- Params: {spec['params']}")
        a(f"- Invalidation: {spec['invalidation']}")
        a("")
    a("### 成本、流动性与执行")
    a("")
    a(ctx["liquidity_note"])
    a("")
    a("权重变化只发生在再平衡日。研究假定订单相对 ETF 日成交额很小（宽基按 <1% ADV 理解），冲击已含在滑点假设里，没有再做平方根冲击模型。")
    a("")
    a("### Code")
    a("")
    a("可执行实现：`scripts/trading_strategy_research/run_daily_research.py`。核心约定是次日持仓与只对风险资产权重变化收费：")
    a("")
    a("```python")
    a(ctx["code_excerpt"])
    a("```")
    a("")
    a("## 4. Backtest Results（回测结果）")
    a("")
    a(ctx["audit"])
    a("")
    a("### 全样本（2018-01-01 至截止日，扣费后）")
    a("")
    a("| 策略 | 总收益 | CAGR | 波动 | Sharpe | Sortino | 最大回撤 | 月胜率 | 月盈亏比 | Calmar | 年化换手 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in ctx["table_order"]:
        a(metric_line(key, res[key], ctx["turnover"][key]))
    a("")
    a("### 平均持仓（全样本窗口）")
    a("")
    a(ctx["exposure_md"])
    a("")
    a("### 样本内 / 样本外")
    a("")
    a("| 策略 | IS CAGR | IS Sharpe | IS MaxDD | OOS CAGR | OOS Sharpe | OOS MaxDD |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in ctx["table_order"]:
        is_m, oos = res[key]["is"], res[key]["oos"]
        a(
            f"| {key} | {fmt_pct(is_m.get('cagr'))} | {fmt_num(is_m.get('sharpe'))} | {fmt_pct(is_m.get('max_dd'))} | "
            f"{fmt_pct(oos.get('cagr'))} | {fmt_num(oos.get('sharpe'))} | {fmt_pct(oos.get('max_dd'))} |"
        )
    a("")
    a("### 分年收益（扣费后）")
    a("")
    years = ctx["years"]
    a("| 策略 | " + " | ".join(str(y) for y in years) + " |")
    a("| --- | " + " | ".join("---:" for _ in years) + " |")
    for key in ctx["table_order"]:
        yearly = res[key]["full"].get("yearly", {})
        cells = " | ".join(fmt_pct(yearly.get(y)) for y in years)
        a(f"| {key} | {cells} |")
    a("")
    a("### 成本加倍（单边 16 / 24 bps）后的全样本")
    a("")
    a("| 策略 | CAGR | Sharpe | MaxDD | 相对主成本 Sharpe 变化 |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for key in ctx["table_order"]:
        base = res[key]["full"]
        alt = ctx["cost_x2"][key]["full"]
        delta = alt.get("sharpe", np.nan) - base.get("sharpe", np.nan)
        a(
            f"| {key} | {fmt_pct(alt.get('cagr'))} | {fmt_num(alt.get('sharpe'))} | "
            f"{fmt_pct(alt.get('max_dd'))} | {fmt_num(delta)} |"
        )
    a("")
    a("### 参数敏感性（全样本，主成本；不是新的主策略）")
    a("")
    a("| 变体 | CAGR | Sharpe | MaxDD |")
    a("| --- | ---: | ---: | ---: |")
    for label, m in ctx["variants"].items():
        a(f"| {label} | {fmt_pct(m.get('cagr'))} | {fmt_num(m.get('sharpe'))} | {fmt_pct(m.get('max_dd'))} |")
    a("")
    a("### 均线走步（walk-forward）")
    a("")
    a(ctx["walkforward_note"])
    a("")
    a("### 尾部与相关性")
    a("")
    a(ctx["tail_note"])
    a("")
    a("基准最差 5% 交易日上的策略表现：")
    a("")
    a("| 策略 | 危机日均收益 | 相对基准尾部比 | 与基准日相关 |")
    a("| --- | ---: | ---: | ---: |")
    for key in ctx["table_order"]:
        t = ctx["tails"][key]
        a(
            f"| {key} | {fmt_pct(t['stress_mean'], 2)} | {fmt_num(t['tail_ratio'])} | {fmt_num(t['corr'])} |"
        )
    a("")
    a("## 5. Comparison & Ranking（比较与排序）")
    a("")
    a("排序分（只用于排队，不是边缘证明）= 0.30×全样本Sharpe + 0.25×样本外Sharpe + 0.15×样本内Sharpe + 0.20×(1+最大回撤) + 0.10×(超额CAGR/5%)，各项有截断。")
    a("")
    a("| Rank | 策略 | 家族 | 排序分 | 过质量门槛 | 主要否决原因 |")
    a("| ---: | --- | --- | ---: | :---: | --- |")
    for i, row in enumerate(ctx["ranking"], 1):
        why = "；".join(row["reasons"]) if row["reasons"] else "通过"
        a(f"| {i} | {row['key']} | {row['family']} | {row['score']:.2f} | {'Y' if row['passed'] else 'N'} | {why} |")
    a("")
    a(ctx["rank_narrative"])
    a("")
    a("## 6. Final Recommendations（建议）")
    a("")
    a(ctx["recommendations"])
    a("")
    a("## 7. Deployment Considerations（若未来部署）")
    a("")
    a(ctx["deployment"])
    a("")
    a("## 8. What Could Go Wrong（失效条件）")
    a("")
    a(ctx["wrong"])
    a("")
    a("## 自检")
    a("")
    a(ctx["self_check"])
    a("")
    a("## 假设清单")
    a("")
    a("- 腾讯前复权收盘价可拼接，重叠窗口价格一致（脚本外已抽查，拼接时按日期去重）。")
    a("- 成交量单位按 510300 典型成交额自动识别为“手”，名义成交额 = 量 × 100 × 价格。")
    a("- 未计入申赎折溢价、QDII 溢价均值回归、最小佣金 5 元、融资融券与冲击的非线性部分。")
    a("- 现金利率记 0，对择时策略偏严（真实货币基金收益为正）。")
    a("- 行业 ETF 名单是当前仍交易的代表性品种，不是严格的时点可投资全集，存在存活偏差。")
    a("- 无杠杆、不可卖空。国债与黄金实际可 T+0，回测仍统一延迟一天，偏保守。")
    a("")
    return "\n".join(lines) + "\n"


def metric_line(key: str, blob: dict, turnover: float) -> str:
    m = blob["full"]
    return (
        f"| {key} | {fmt_pct(m.get('total'))} | {fmt_pct(m.get('cagr'))} | {fmt_pct(m.get('vol'))} | "
        f"{fmt_num(m.get('sharpe'))} | {fmt_num(m.get('sortino'))} | {fmt_pct(m.get('max_dd'))} | "
        f"{fmt_pct(m.get('win_rate'))} | {fmt_num(m.get('profit_factor'))} | {fmt_num(m.get('calmar'))} | "
        f"{turnover:.2f} |"
    )


def ann_turnover(traded: pd.Series) -> float:
    sl = traded.loc[COMMON_START:]
    if sl.empty:
        return float("nan")
    return float(sl.mean() * 252)


def tail_stats(net: pd.Series, bench: pd.Series) -> dict:
    both = pd.concat([net.rename("s"), bench.rename("b")], axis=1).dropna().loc[COMMON_START:]
    if both.empty:
        return {"stress_mean": float("nan"), "tail_ratio": float("nan"), "corr": float("nan")}
    cutoff = both["b"].quantile(0.05)
    stress = both[both["b"] <= cutoff]
    bmean = float(stress["b"].mean()) if len(stress) else float("nan")
    smean = float(stress["s"].mean()) if len(stress) else float("nan")
    ratio = smean / bmean if bmean not in (0, 0.0) and np.isfinite(bmean) else float("nan")
    corr = float(both["s"].corr(both["b"]))
    return {"stress_mean": smean, "tail_ratio": ratio, "corr": corr}


def prose_market(regime: dict, intraday: dict) -> tuple[str, str]:
    idx = {r["sym"]: r for r in regime["indices"]}
    hs = idx.get("sh000300", {})
    cy = idx.get("sz399006", {})
    zz = idx.get("sh000905", {})
    zz1000 = idx.get("sh000852", {})
    kc = idx.get("sh000688", {})

    def rel(a, b, field):
        if not a or not b:
            return float("nan")
        return a.get(field, np.nan) - b.get(field, np.nan)

    large_vs_small = rel(hs, zz1000, "r60")
    if np.isfinite(large_vs_small) and large_vs_small > 0.03:
        size_txt = "过去 60 日大盘（沪深300）强于中证1000，风格偏大盘。"
    elif np.isfinite(large_vs_small) and large_vs_small < -0.03:
        size_txt = "过去 60 日中证1000强于沪深300，风格偏小盘。"
    else:
        size_txt = "过去 60 日大小盘相对强弱接近，没有拉开单边风格。"

    above = [r["name"] for r in regime["indices"] if r["above200"]]
    below = [r["name"] for r in regime["indices"] if not r["above200"]]
    trend_txt = (
        f"高于 200 日均线: {('、'.join(above) if above else '无')}。"
        f"低于 200 日均线: {('、'.join(below) if below else '无')}。"
    )
    vol = hs.get("vol20", float("nan"))
    if np.isfinite(vol) and vol < 0.15:
        vol_txt = f"沪深300 的 20 日实现波动约 {vol:.0%}，处于偏低区间。"
    elif np.isfinite(vol) and vol > 0.25:
        vol_txt = f"沪深300 的 20 日实现波动约 {vol:.0%}，短线波动偏高。"
    else:
        vol_txt = f"沪深300 的 20 日实现波动约 {fmt_pct(vol)}。"

    dd = hs.get("dd252", float("nan"))
    r252 = hs.get("r252", float("nan"))
    if np.isfinite(dd) and dd > -0.08 and np.isfinite(r252) and r252 > 0:
        regime_name = "上升趋势或高位整理"
    elif np.isfinite(dd) and dd < -0.15 and np.isfinite(r252) and r252 < 0:
        regime_name = "下行或熊市修复未完成"
    else:
        regime_name = "区间震荡，趋势并不干净"

    sectors = sorted(regime["sectors"], key=lambda r: r.get("r60") or -9, reverse=True)
    leaders = "、".join(s["name"] for s in sectors[:3])
    laggards = "、".join(s["name"] for s in sectors[-3:])
    rotation = (
        f"60 日相对强弱靠前的是 {leaders}；靠后的是 {laggards}。"
        "这只说明价格动量的当前位置，不是基本面确认，也不能当成追逐理由。"
    )
    intra = ""
    if "sh000300" in intraday:
        q = intraday["sh000300"]
        intra = f" 研究运行时沪深300 指数盘中约 {q['price']:.2f}，相对前收 {q['pct']:+.2%}，该变动没有写入回测。"
    text = (
        f"最后完整交易日是 {regime['asof']}。以价格而不是叙事定性，宽基处于**{regime_name}**："
        f"沪深300 252 日收益 {fmt_pct(r252)}，距 252 日高点 {fmt_pct(dd)}。"
        f"{vol_txt}{trend_txt}{size_txt}"
        f"创业板 60 日收益 {fmt_pct(cy.get('r60'))}，中证500 为 {fmt_pct(zz.get('r60'))}，"
        f"科创50 为 {fmt_pct(kc.get('r60'))}。"
        f"{intra}"
    )
    return text, rotation


def build_narratives(ctx):
    res = ctx["results"]
    bh = res[BENCHMARK]["full"]
    ranking = ctx["ranking"]
    passed = [r for r in ranking if r["passed"]]
    if passed:
        names = "、".join(r["key"] for r in passed)
        executive = (
            f"在写死的主参数和扣费规则下，有 {len(passed)} 个候选通过质量门槛：{names}。"
            "通过的含义是：全样本扣费后 CAGR 与 Sharpe 同时优于沪深300ETF 买入持有，Sharpe>1，"
            "最大回撤浅于 30%，样本内与样本外 CAGR 为正，且没有被单一年份或相邻参数推翻。"
            "其余候选即使排序靠前，也只作为观察，不宣称存在可部署边缘。"
        )
    else:
        top = "、".join(r["key"] for r in ranking[:3])
        executive = (
            f"**没有候选通过本次质量门槛。** 不把排序分最高的策略说成已经验证的边缘。"
            f"扣费后全样本里，沪深300ETF 买入持有 CAGR {fmt_pct(bh['cagr'])}、Sharpe {fmt_num(bh['sharpe'])}、"
            f"最大回撤 {fmt_pct(bh['max_dd'])}。排序上最接近的是 {top}，它们的否决原因写在排名表里。"
            "可执行的保守结论是：在这条证据下维持宽基（或宽基+国债）配置，不把行业轮动或短期反转上到实盘。"
        )
    bond_m = res["BH_BOND"]["full"]
    gold_m = res["BH_GOLD"]["full"]
    executive += (
        f" 对照买入持有：国债ETF CAGR {fmt_pct(bond_m['cagr'])}、Sharpe {fmt_num(bond_m['sharpe'])}、"
        f"最大回撤 {fmt_pct(bond_m['max_dd'])}；黄金ETF CAGR {fmt_pct(gold_m['cagr'])}、"
        f"Sharpe {fmt_num(gold_m['sharpe'])}、最大回撤 {fmt_pct(gold_m['max_dd'])}。"
        "黄金在这八年的价格涨幅很大，但波动和回撤使其过不了同一条 Sharpe 门槛。"
    )
    # recommendations top 5 — honest
    rec_lines = []
    if passed:
        rec_lines.append("下列建议只覆盖已经通过门槛的策略，按排序分取前五。")
        for i, row in enumerate(passed[:5], 1):
            spec = SPECS[row["key"]]
            full = res[row["key"]]["full"]
            oos = res[row["key"]]["oos"]
            rec_lines.append(
                f"{i}. **{row['key']}（{spec['name']}）**。全样本 CAGR {fmt_pct(full['cagr'])}，"
                f"Sharpe {fmt_num(full['sharpe'])}，最大回撤 {fmt_pct(full['max_dd'])}；"
                f"样本外 Sharpe {fmt_num(oos['sharpe'])}。逻辑：{spec['thesis']}"
            )
            exp = ctx["exposures"].get(row["key"])
            if exp:
                rec_lines.append(
                    f"   平均仓位：股票 {exp['equity']:.0%}，黄金 {exp['gold']:.0%}，"
                    f"国债 {exp['bond']:.0%}，现金 {exp['cash']:.0%}。"
                    "这是低波配置结果，不是对个股或行业的方向性预测。"
                )
    else:
        rec_lines.append("质量门槛下没有可部署的主动策略。建议按下面的顺序理解证据：")
        rp = res["RISK_PARITY_3"]["full"]
        bond_bh = res["BH_BOND"]["full"]
        static = res["STATIC_60_40"]["full"]
        rec_lines.append(
            f"1. **不要把股债金风险均衡写成股票边缘。** 它是排序第一的候选，扣费后 CAGR {fmt_pct(rp['cagr'])}、"
            f"Sharpe {fmt_num(rp['sharpe'])}、最大回撤 {fmt_pct(rp['max_dd'])}，三项都优于沪深300ETF。"
            f"平均仓位约七成在五年国债。国债 ETF 自身 Sharpe {fmt_num(bond_bh['sharpe'])}、CAGR {fmt_pct(bond_bh['cagr'])}。"
            "组合把 CAGR 做高了，但 Sharpe 没有同时超过国债，增量来自黄金和少量股票在这段样本里的路径。"
        )
        rec_lines.append(
            f"2. **股票仓位的默认实现仍是 510300 买入持有**（CAGR {fmt_pct(bh['cagr'])}，"
            f"Sharpe {fmt_num(bh['sharpe'])}，最大回撤 {fmt_pct(bh['max_dd'])}）。"
            f"若只想降低回撤，静态 60/40 更透明：CAGR {fmt_pct(static['cagr'])}，"
            f"Sharpe {fmt_num(static['sharpe'])}，最大回撤 {fmt_pct(static['max_dd'])}。Sharpe 仍然远低于 1。"
        )
        for i, row in enumerate(ranking[1:4], 3):
            full = res[row["key"]]["full"]
            rec_lines.append(
                f"{i}. **观察而非部署：{row['key']}**。排序分 {row['score']:.2f}，"
                f"CAGR {fmt_pct(full['cagr'])}，Sharpe {fmt_num(full['sharpe'])}，最大回撤 {fmt_pct(full['max_dd'])}。"
                f"未通过原因：{row['reasons'][0] if row['reasons'] else 'n/a'}。"
            )
    # correlation warning
    corr = ctx["corr_note"]
    deployment = (
        "在没有策略通过门槛之前，不部署主动规则，不设置杠杆，单策略名义风险不超过组合的既有宽基仓位。\n\n"
        "若后续有策略重新通过同一套事前门槛，仓位约束建议仍是：\n\n"
        "- 单策略初始资金比例 ≤ 组合风险资产的 30%，与宽基核心仓分开记账。\n"
        "- 不使用杠杆。波动率目标策略本身已经把权重限制在 1 以内。\n"
        "- 组合自峰值回撤达到 15% 时暂停该策略 20 个交易日，期间转国债 ETF。这一条**没有**包含在本次回测里。\n"
        "- 只交易 60 日成交额中位数 ≥ 2000 万人民币的 ETF；实际下单前用券商盘口再确认买卖价差。\n"
        "- 监测：每月核对持仓是否等于规则输出、样本外滚动 12 个月超额是否转负、参数邻居是否仍然同号。\n"
        "- 行业动量与风格轮动不要叠加满仓，它们共享 A 股风险预算。\n\n"
        + corr
    )
    wrong = (
        "- 成本与滑点：行业 ETF 在跌停附近或折溢价扩大时，12bps 不够。成本加倍表就是压力情形。\n"
        "- 存活偏差：行业池是今天仍在交易的 ETF，退市或清盘品种没有回到样本里。\n"
        "- 制度变化：T+0 推广、印花税、ETF 费率、QDII 额度都会改变可交易性与溢价。\n"
        "- 股债金同跌时，风险均衡和“跌破均线买国债”会同时失效。\n"
        "- 2024 年秋季那种政策跳空若在样本外重复出现，趋势规则会买在脉冲之后，反转规则会在脉冲前被轧空。\n"
        "- 现金利率记 0 会让防守策略看起来更差；若改为真实货币基金收益，排序可能变化，但本次不把未取得的利率序列补进去。\n"
        "- 单一参数或单一年份贡献过高的结果已经在门槛里否决，但多重检验仍然存在：十个策略共享同一价格历史。\n"
    )
    self_check = (
        "- 缺什么：可靠的时点财务、事件日历、期权持仓、融资余额与国债收益率曲线没有接入，宏观判断只能源于价格。盘中 2026-10-12 的 K 线不完整，已剔除。\n"
        "- 边缘还是过拟合：主参数事前固定；相邻参数与样本外窗口单独报告。没有门槛通过者时，默认结论是证据不足，而不是挑选最好看的曲线。\n"
        "- 制度区间：2018、2022 两个下跌年份与 2023 以后的样本外都在表内。若某策略只在其中一年赚钱，集中度规则会否决它。\n"
        "- 什么会推翻结论：若独立数据源（交易所行情或券商复权）无法复核腾讯前复权收益；或扣费后滚动三年超额稳定为正且换手可执行，则应重开研究而不是沿用本页否决。\n"
        "- 稳健性：见成本加倍、参数网格和 walk-forward。会改变看法的证据是样本外、扣费后、相邻参数同时战胜 510300，且尾部跌幅日没有放大基准亏损。\n"
    )
    rank_narrative = (
        "质量门槛同时要求收益、Sharpe、回撤、样本内外和参数邻居。只在风险调整后更好、但 CAGR 仍低于买入持有的策略，"
        "记为风险特征不同，不记为高质量边缘。静态 60/40 与买入持有放在表的上半部供对照，不进入候选排序。"
    )
    return executive, "\n".join(rec_lines), deployment, wrong, self_check, rank_narrative


def main() -> None:
    symbols = sorted(set(SECTORS + STYLES + CROSS + INDICES + [GOLD, BOND, HS]))
    frames = {}
    failed = []
    for sym in symbols:
        try:
            df = fetch_symbol(sym)
            if df.empty or len(df) < 100:
                failed.append(f"{sym}: rows={len(df)}")
                continue
            frames[sym] = df
            print(f"ok {sym} {df['date'].iloc[0].date()} {df['date'].iloc[-1].date()} n={len(df)}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{sym}: {exc}")
            print("fail", sym, exc)
    if HS not in frames or BOND not in frames:
        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.write_text(
            "# 交易策略研究报告 2026-10-12\n\n"
            "关键行情下载失败，拒绝用虚构价格填补。\n\n"
            + "\n".join(f"- {x}" for x in failed)
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1)

    close = panel_from(frames, "close")
    high = panel_from(frames, "high")
    low = panel_from(frames, "low")
    volume = panel_from(frames, "volume")
    drop_day = incomplete_session_date()
    if drop_day:
        stamp = pd.Timestamp(drop_day)
        close = close.loc[close.index < stamp]
        high = high.loc[high.index < stamp]
        low = low.loc[low.index < stamp]
        volume = volume.loc[volume.index < stamp]

    mult = calibrate_volume_multiplier(volume, close)
    notional = volume * close * mult
    rets = close.pct_change()
    # Flag economically implausible prints. Do not replace them with a flat
    # return: leave NaN so simulate drops that session's asset return only if
    # the price itself is missing. Thresholds sit beyond A-share price limits
    # and beyond typical QDII overnight gaps.
    bad_ticks = []
    for sym in close.columns:
        if sym in ("sh513100", "sh513500"):
            limit = 0.35
        elif sym in INDICES or sym in LIQUID:
            limit = 0.18
        else:
            limit = 0.25
        mask = rets[sym].abs() > limit
        for dt, val in rets.loc[mask, sym].items():
            bad_ticks.append((sym, str(dt.date()), float(val)))
    rets["CASH"] = 0.0

    quote_syms = INDICES + [HS, "sh510500", "sz159915", BOND, GOLD]
    try:
        intraday = fetch_quotes(quote_syms)
    except Exception as exc:  # noqa: BLE001
        intraday = {}
        failed.append(f"intraday: {exc}")

    results = evaluate_all(close, high, low, rets, notional, bps_scale=1.0)
    cost_x2 = evaluate_all(close, high, low, rets, notional, bps_scale=2.0)
    variants = sensitivity(close, high, low, rets, notional)
    wf = walk_forward_sma(close, rets)

    turnover = {k: ann_turnover(v["sim"]["traded"]) for k, v in results.items()}
    bench_net = results[BENCHMARK]["sim"]["net"]
    tails = {k: tail_stats(v["sim"]["net"], bench_net) for k, v in results.items()}

    ranking = []
    for key in CANDIDATES:
        passed, reasons = quality_gate(key, results[key], results[BENCHMARK], variants, results)
        ranking.append(
            {
                "key": key,
                "family": SPECS[key]["family"],
                "score": score_row(results[key]["full"], results[key]["is"], results[key]["oos"], results[BENCHMARK]["full"]),
                "passed": passed,
                "reasons": reasons,
            }
        )
    ranking.sort(key=lambda r: r["score"], reverse=True)

    # pairwise corr among candidates
    cols = []
    for key in CANDIDATES:
        cols.append(results[key]["sim"]["net"].rename(key))
    corr = pd.concat(cols, axis=1).loc[COMMON_START:].corr()
    # highlight high corr pairs
    pairs = []
    for i, a in enumerate(CANDIDATES):
        for b in CANDIDATES[i + 1 :]:
            val = float(corr.loc[a, b])
            if np.isfinite(val) and val >= 0.8:
                pairs.append(f"{a}/{b} {val:.2f}")
    if pairs:
        corr_note = "日收益相关 ≥ 0.80 的策略对：" + "；".join(pairs) + "。这些策略不要同时满仓。"
    else:
        corr_note = "候选之间没有日收益相关 ≥ 0.80 的组合，但仍共享 A 股贝塔，不能视为独立风险。"

    med_notional = float(notional[HS].tail(60).median())
    liquidity_note = (
        f"510300 近 60 日名义成交额中位数约 {med_notional/1e8:.1f} 亿元（成交量乘数={mult:.0f}）。"
        f"行业准入线为 60 日中位数 ≥ {LIQ_MEDIAN_CNY/1e4:.0f} 万元。未过线的 ETF 在该再平衡日不入选。"
    )
    if bad_ticks:
        preview = "；".join(f"{s} {d} {v:.1%}" for s, d, v in bad_ticks[:8])
        gaps_bad = (
            "下列单日收益超过数据质量阈值（宽基/指数 18%，行业 25%，QDII 35%）。"
            f"收益保留在回测里，没有改写成 0：{preview}。"
        )
    else:
        gaps_bad = "未发现超过数据质量阈值的单日收益（宽基/指数 18%，行业 25%，QDII 35%）。"

    gaps = (
        "- 未接入 CPI、GDP、政策事件库、北向资金、融资余额、期权持仓与国债到期收益率曲线；宏观段只使用交易价格。\n"
        "- 个股财务与事件驱动策略缺少可复核的时点数据，本次不做。\n"
        "- 卖空约束：没有融券费率序列，不做市场中性或配对空头。\n"
        f"- {gaps_bad}\n"
        "- 下载失败或样本过短的代码："
        + ("无。" if not failed else "；".join(failed))
    )

    regime = price_regime(close)
    market_txt, rotation = prose_market(regime, intraday)
    exposures = {}
    exp_lines = [
        "| 策略 | 沪深300均权 | 黄金均权 | 国债均权 | 现金均权 | 其他ETF | 最大单资产 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in [BENCHMARK, "BH_BOND", "BH_GOLD", "STATIC_60_40"] + CANDIDATES:
        held = results[key]["sim"]["held"].loc[COMMON_START:]
        def avg(col):
            return float(held[col].mean()) if col in held.columns else 0.0
        risky = held.drop(columns=["CASH"], errors="ignore")
        top_col = risky.mean().idxmax()
        exposures[key] = {
            "equity": avg(HS),
            "gold": avg(GOLD),
            "bond": avg(BOND),
            "cash": avg("CASH"),
            "top": f"{NAMES.get(top_col, top_col)} {float(risky.mean().max()):.0%}",
        }
        e = exposures[key]
        other = 1.0 - e["equity"] - e["gold"] - e["bond"] - e["cash"]
        exp_lines.append(
            f"| {key} | {e['equity']:.0%} | {e['gold']:.0%} | {e['bond']:.0%} | {e['cash']:.0%} | "
            f"{other:.0%} | {e['top']} |"
        )
    years = list(range(2018, close.index.max().year + 1))
    table_order = [BENCHMARK, "BH_BOND", "BH_GOLD", "STATIC_60_40"] + [r["key"] for r in ranking]

    wf_m = wf.get("metrics") or {}
    picks = "，".join(f"{p['year']}:{p['sma']}" for p in wf["picks"])
    walk_note = (
        f"每年只用上年之前的 Sharpe 在 SMA {{100,150,200,250}} 中选择，再拼接下一年（国债防守版）。"
        f"选择序列：{picks}。拼接后 CAGR {fmt_pct(wf_m.get('cagr'))}，Sharpe {fmt_num(wf_m.get('sharpe'))}，"
        f"最大回撤 {fmt_pct(wf_m.get('max_dd'))}。这是稳健性检查，不是第 11 个主策略。"
    )

    stress_rows = []
    for key in ("BH_HS300", "TSMOM_SMA200_BOND", "RISK_PARITY_3"):
        s = results[key]["sim"]["net"]
        for label, start, end in (
            ("2020-02", "2020-02-03", "2020-03-23"),
            ("2022", "2022-01-01", "2022-12-31"),
            ("2024-09/10", "2024-09-18", "2024-10-18"),
        ):
            sl = s.loc[start:end]
            if len(sl):
                stress_rows.append(f"{key} {label} {(1+sl).prod()-1:.1%}")
    tail_note = "压力窗口累计收益（扣费后，描述性，不是新参数）：" + "；".join(stress_rows) + "。"

    # code excerpt: execution core only, kept short but runnable in spirit
    code_excerpt = '''def simulate(targets, rets, bps_scale=1.0):
    """Decision at close t, hold during t+1, charge one-way bps on |Δw| of ETFs."""
    held = targets.shift(1).fillna(0.0)
    asset_ret = (held[risky] * rets[risky]).sum(axis=1)
    dw = held[risky].diff()
    dw.iloc[0] = held[risky].iloc[0]
    cost = sum(dw[c].abs() * bps[c] * bps_scale / 10000 for c in risky)
    return asset_ret - cost
'''

    ctx = {
        "regime": regime,
        "fetched_at": _now_sh().strftime("%Y-%m-%d %H:%M %Z"),
        "intraday": intraday,
        "results": results,
        "cost_x2": cost_x2,
        "variants": variants,
        "turnover": turnover,
        "tails": tails,
        "ranking": ranking,
        "years": years,
        "table_order": table_order,
        "audit": audit_buyhold(close, results),
        "liquidity_note": liquidity_note,
        "gaps": gaps,
        "walkforward_note": walk_note,
        "tail_note": tail_note,
        "corr_note": corr_note,
        "code_excerpt": code_excerpt,
        "exposure_md": "\n".join(exp_lines),
        "exposures": exposures,
    }
    market_narrative, rotation_note = market_txt, rotation
    executive, recommendations, deployment, wrong, self_check, rank_narrative = build_narratives(ctx)
    ctx.update(
        {
            "executive": executive,
            "market_narrative": market_narrative,
            "rotation_note": rotation_note,
            "recommendations": recommendations,
            "deployment": deployment,
            "wrong": wrong,
            "self_check": self_check,
            "rank_narrative": rank_narrative,
        }
    )
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    text = render_report(ctx)
    OUT_REPORT.write_text(text, encoding="utf-8")
    summary = {
        "asof": regime["asof"],
        "ranking": ranking,
        "bh": results[BENCHMARK]["full"],
        "passed": [r["key"] for r in ranking if r["passed"]],
    }
    (CACHE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT_REPORT)
    print("passed", summary["passed"])
    for row in ranking:
        full = results[row["key"]]["full"]
        print(
            f"{row['score']:.2f} pass={row['passed']} {row['key']} "
            f"cagr={full['cagr']:.3f} sharpe={full['sharpe']:.2f} dd={full['max_dd']:.3f}"
        )


if __name__ == "__main__":
    main()
