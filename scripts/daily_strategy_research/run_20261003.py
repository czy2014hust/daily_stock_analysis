#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily trading-strategy research for 2026-10-03.

Data policy
-----------
Fetch with curl only. Keep a source if the HTTP body parses and the last
bar matches a second quote source. Do not fill missing sessions.

Primary prices are Tencent forward-adjusted (qfq) daily bars, stitched in
two-year chunks. Yahoo adjusted closes are a cross-check only: several
A-share ETFs there have close == adjclose and therefore drop dividends.

Execution
---------
Signal uses the close of day t. Orders fill at the next session open.
Shares are held constant between scheduled rebalances (weights may drift).
One-way cost is charged on traded notional. Residual cash earns 0.
No shorting and no leverage: weights are in [0, 1] and sum to at most 1.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ASOF = "2026-10-03"
LAST_SESSION = "2026-09-30"  # expected; overwritten by the newest bar actually returned
UA = "Mozilla/5.0 (compatible; dsa-strategy-research/1.0)"
COST_BPS = 12.0  # commission + slippage, one way, liquid ETF assumption
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scripts" / "daily_strategy_research" / "output" / "20261003"
CACHE = Path("/tmp/mkt/tx_20261003")
SAMPLE_START = pd.Timestamp("2018-01-01")
IS_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")

# code -> (exchange prefix, display name, role)
UNIVERSE = {
    "510300": ("sh", "沪深300ETF", "equity"),
    "510500": ("sh", "中证500ETF", "equity"),
    "510050": ("sh", "上证50ETF", "equity"),
    "159915": ("sz", "创业板ETF", "equity"),
    "512100": ("sh", "中证1000ETF", "equity"),
    "510880": ("sh", "红利ETF", "equity"),
    "512800": ("sh", "银行ETF", "equity"),
    "512010": ("sh", "医药ETF", "equity"),
    "159928": ("sz", "消费ETF", "equity"),
    "512880": ("sh", "证券ETF", "equity"),
    "518880": ("sh", "黄金ETF", "defensive"),
    "511260": ("sh", "十年国债ETF", "defensive"),
    "513100": ("sh", "纳指ETF", "overseas"),
    "513500": ("sh", "标普ETF", "overseas"),
    # Snapshot only. Listed too late for the 2018 common sample.
    "512480": ("sh", "半导体ETF", "snapshot"),
    "512690": ("sh", "酒ETF", "snapshot"),
    "588000": ("sh", "科创50ETF", "snapshot"),
    "515790": ("sh", "光伏ETF", "snapshot"),
    "515220": ("sh", "煤炭ETF", "snapshot"),
}

CHUNKS = [
    ("2016-01-01", "2017-12-31"),
    ("2018-01-01", "2019-12-31"),
    ("2020-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"),
    ("2024-01-01", "2026-09-30"),
]

INDEX_SYMBOLS = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sh000852": "中证1000",
    "sh000688": "科创50",
    "sh000016": "上证50",
}


def curl(url: str, timeout: int = 25) -> tuple[int, bytes, str]:
    """Return (exit_code, stdout, stderr). Non-zero exit is a failed probe."""
    proc = subprocess.run(
        ["curl", "-sS", "-m", str(timeout), "-A", UA, "--fail-with-body", url],
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", "replace")


def curl_json(url: str, timeout: int = 25, retries: int = 2) -> dict:
    last_err = "unknown"
    for attempt in range(retries + 1):
        code, body, err = curl(url, timeout=timeout)
        if code == 0 and body:
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                last_err = f"json:{exc}"
        else:
            last_err = err.strip() or f"exit:{code} bytes:{len(body)}"
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"curl failed {url} :: {last_err}")


def probe_sources() -> list[dict]:
    """Operator focus: record which public HTTP sources answer to curl."""
    tests = [
        ("tencent_kline", "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000300,day,,,5,qfq"),
        ("tencent_quote", "https://qt.gtimg.cn/q=sh000300"),
        ("sina_kline", "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol=sh000300&scale=240&ma=no&datalen=3"),
        ("yahoo_chart", "https://query1.finance.yahoo.com/v8/finance/chart/510300.SS?interval=1d&range=5d"),
        ("eastmoney_kline", "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&klt=101&fqt=1&beg=20240901&end=20261003&fields1=f1&fields2=f51,f53"),
        ("eastmoney_quote", "https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&secids=1.000300&fields=f2,f12"),
        ("newsnow_cls", "https://newsnow.busiyi.world/api/s?id=cls-hot"),
        ("newsnow_wscn", "https://newsnow.busiyi.world/api/s?id=wallstreetcn-quick"),
        ("newsnow_jin10", "https://newsnow.busiyi.world/api/s?id=jin10"),
    ]
    rows = []
    for name, url in tests:
        started = time.time()
        code, body, err = curl(url, timeout=20)
        elapsed = round(time.time() - started, 3)
        ok = False
        detail = ""
        if code == 0 and body:
            text = body.decode("utf-8", "replace")
            if name.startswith("tencent_quote"):
                ok = "000300" in body.decode("gbk", "replace")
                detail = "gbk quote" if ok else "quote parse failed"
            elif name.startswith("newsnow"):
                try:
                    payload = json.loads(text)
                    items = payload.get("items") if isinstance(payload, dict) else None
                    ok = isinstance(items, list) and len(items) > 0
                    detail = f"items={0 if not items else len(items)}"
                except json.JSONDecodeError:
                    detail = text[:80].replace("\n", " ")
            elif name.startswith("yahoo") or name.startswith("sina") or name.startswith("tencent_kline"):
                try:
                    payload = json.loads(text)
                    ok = isinstance(payload, (dict, list))
                    detail = "json"
                except json.JSONDecodeError:
                    detail = text[:80].replace("\n", " ")
            else:
                detail = text[:80].replace("\n", " ")
                ok = "klines" in text or "data" in text
        else:
            detail = (err or f"exit {code}").strip().splitlines()[-1][:180]
        rows.append(
            {
                "name": name,
                "ok": ok,
                "exit_code": code,
                "bytes": len(body),
                "seconds": elapsed,
                "detail": detail,
            }
        )
        print(f"probe {name}: ok={ok} exit={code} bytes={len(body)} {detail}")
    return rows


def load_tencent_bars(code: str, prefix: str) -> pd.DataFrame:
    """Stitch qfq daily bars. Overlap mismatches are raised, not averaged."""
    CACHE.mkdir(parents=True, exist_ok=True)
    symbol = f"{prefix}{code}"
    frames = []
    for start, end in CHUNKS:
        cache_path = CACHE / f"{symbol}_{start}_{end}.json"
        if cache_path.exists() and cache_path.stat().st_size > 50:
            payload = json.loads(cache_path.read_text())
        else:
            url = (
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                f"?param={symbol},day,{start},{end},800,qfq"
            )
            payload = curl_json(url)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False))
        node = payload.get("data", {}).get(symbol) or {}
        rows = node.get("qfqday") or node.get("day") or []
        if not rows:
            # Names listed after the chunk start have no bars. Keep later chunks.
            continue
        part = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        frames.append(part)
    if not frames:
        raise RuntimeError(f"no bars for {symbol}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "close", "high", "low", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df.sort_values("date")
    # Same qfq factor across chunks: identical dates must agree.
    dup = df.duplicated("date", keep=False)
    if dup.any():
        check = df.loc[dup].groupby("date")["close"].nunique()
        bad = check[check > 1]
        if len(bad):
            raise RuntimeError(f"{code} qfq overlap conflict on {list(bad.index[:3])}")
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    df["yuan_volume"] = df["volume"] * 100.0 * df["close"]  # Tencent volume is in lots
    return df


def fetch_spot_quotes(codes: list[str]) -> dict[str, dict]:
    symbols = []
    prefix = {c: UNIVERSE[c][0] for c in codes}
    for code in codes:
        symbols.append(f"{prefix[code]}{code}")
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    _code, body, err = curl(url, timeout=20)
    if _code != 0:
        raise RuntimeError(err)
    text = body.decode("gbk", "replace")
    out = {}
    for line in text.splitlines():
        if "~" not in line:
            continue
        parts = line.split("~")
        if len(parts) < 35:
            continue
        code = parts[2]
        out[code] = {
            "name": parts[1],
            "last": float(parts[3]),
            "prev_close": float(parts[4]),
            "open": float(parts[5]),
            "change": float(parts[31] or 0),
            "pct": float(parts[32] or 0),
            "high": float(parts[33] or 0),
            "low": float(parts[34] or 0),
            "timestamp": parts[30],
        }
    return out


def fetch_index_snapshot() -> dict[str, dict]:
    url = "https://qt.gtimg.cn/q=" + ",".join(INDEX_SYMBOLS)
    _code, body, err = curl(url, timeout=20)
    if _code != 0:
        raise RuntimeError(err)
    text = body.decode("gbk", "replace")
    out = {}
    for line in text.splitlines():
        parts = line.split("~")
        if len(parts) < 35:
            continue
        out[parts[2]] = {
            "name": parts[1],
            "last": float(parts[3]),
            "prev_close": float(parts[4]),
            "pct": float(parts[32] or 0),
            "timestamp": parts[30],
        }
    return out


def fetch_news() -> list[dict]:
    rows = []
    for source in ("cls-hot", "wallstreetcn-quick", "jin10", "gelonghui"):
        url = f"https://newsnow.busiyi.world/api/s?id={source}"
        try:
            payload = curl_json(url, timeout=20, retries=1)
        except RuntimeError as exc:
            rows.append({"source": source, "error": str(exc)})
            continue
        items = payload.get("items") or []
        for item in items[:8]:
            rows.append(
                {
                    "source": source,
                    "title": item.get("title") or "",
                    "url": item.get("url") or item.get("mobileUrl") or "",
                    "updated_time_ms": payload.get("updatedTime"),
                }
            )
    return rows


def fetch_yahoo_macro() -> dict[str, dict]:
    """US equity, rates, vol, and USDCNY. Not used as A-share fills."""
    symbols = {
        "^GSPC": "spx",
        "^TNX": "us10y",
        "^VIX": "vix",
        "CNY=X": "usdcny",
    }
    out = {}
    for symbol, key in symbols.items():
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=6mo"
        payload = curl_json(url)
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            out[key] = {"error": "empty", "symbol": symbol}
            continue
        node = result[0]
        ts = node.get("timestamp") or []
        quote = node["indicators"]["quote"][0]
        closes = quote.get("close") or []
        series = []
        for t, c in zip(ts, closes):
            if c is None:
                continue
            day = pd.to_datetime(t, unit="s", utc=True).tz_convert("Asia/Shanghai").date()
            series.append((str(day), float(c)))
        if not series:
            out[key] = {"error": "no closes", "symbol": symbol}
            continue
        last_day, last_px = series[-1]
        def _ret(days: int) -> float | None:
            if len(series) <= days:
                return None
            prev = series[-1 - days][1]
            if prev == 0:
                return None
            return last_px / prev - 1.0

        out[key] = {
            "symbol": symbol,
            "last_date": last_day,
            "last": last_px,
            "ret_5d": _ret(5),
            "ret_21d": _ret(21),
            "ret_63d": _ret(63),
        }
    return out


def build_panels(bars: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opens = pd.DataFrame({c: df["open"] for c, df in bars.items()}).sort_index()
    closes = pd.DataFrame({c: df["close"] for c, df in bars.items()}).sort_index()
    yuan = pd.DataFrame({c: df["yuan_volume"] for c, df in bars.items()}).sort_index()
    # Do not invent long gaps. One or two missing prints can be held flat.
    opens = opens.ffill(limit=2)
    closes = closes.ffill(limit=2)
    return opens, closes, yuan


def window_return(close: pd.Series, days: int) -> float | None:
    s = close.dropna()
    if len(s) <= days:
        return None
    prev = float(s.iloc[-1 - days])
    if prev == 0:
        return None
    return float(s.iloc[-1] / prev - 1.0)


def ytd_return(close: pd.Series, year: int = 2026) -> float | None:
    s = close.dropna()
    hist = s[s.index.year == year]
    if hist.empty:
        return None
    prev = s[s.index.year < year]
    if prev.empty:
        base = float(hist.iloc[0])
        last = float(hist.iloc[-1])
        return last / base - 1.0
    return float(hist.iloc[-1] / float(prev.iloc[-1]) - 1.0)


def max_drawdown(close: pd.Series) -> float:
    s = close.dropna()
    if s.empty:
        return float("nan")
    return float((s / s.cummax() - 1.0).min())


def snapshot_table(closes: pd.DataFrame) -> list[dict]:
    rows = []
    for code in closes.columns:
        px = closes[code].dropna()
        if px.empty:
            continue
        rows.append(
            {
                "code": code,
                "name": UNIVERSE[code][1],
                "role": UNIVERSE[code][2],
                "last_date": str(px.index[-1].date()),
                "last": float(px.iloc[-1]),
                "ret_5d": window_return(px, 5),
                "ret_21d": window_return(px, 21),
                "ret_63d": window_return(px, 63),
                "ret_126d": window_return(px, 126),
                "ret_252d": window_return(px, 252),
                "ytd_2026": ytd_return(px, 2026),
                "dd_from_high": max_drawdown(px[px.index >= "2024-01-01"]),
                "vol_20d": float(px.pct_change().tail(20).std() * math.sqrt(252)),
                "adv20_mn_cny": None,
            }
        )
    return rows


def attach_adv(rows: list[dict], yuan: pd.DataFrame) -> None:
    for row in rows:
        series = yuan[row["code"]].dropna().tail(20)
        row["adv20_mn_cny"] = float(series.median() / 1e6) if len(series) else None


def month_end_flag(index: pd.DatetimeIndex) -> pd.Series:
    stamp = pd.Series(index, index=index)
    last = stamp.groupby(index.to_period("M")).transform("max")
    return pd.Series(index == last.to_numpy(), index=index)


def every_n_flag(index: pd.DatetimeIndex, n: int) -> pd.Series:
    flag = pd.Series(False, index=index)
    flag.iloc[::n] = True
    return flag


def allocate_scores(
    score: pd.DataFrame,
    flags: pd.Series,
    k: int,
    *,
    smallest: bool = False,
    allowed: list[str] | None = None,
    liquid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cols = list(allowed or score.columns)
    weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for dt in score.index[flags.reindex(score.index).fillna(False).to_numpy()]:
        row = score.loc[dt, cols]
        if liquid is not None:
            ok = liquid.loc[dt, cols].fillna(False).astype(bool)
            row = row.where(ok)
        row = row.dropna()
        if row.empty:
            continue
        pick = row.nsmallest(min(k, len(row))) if smallest else row.nlargest(min(k, len(row)))
        for code in pick.index:
            weights.loc[dt, code] = 1.0 / len(pick)
    return weights


def strategy_weights(
    closes: pd.DataFrame,
    yuan: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Pre-committed rules. Parameters are not chosen from the result table."""
    idx = closes.index
    equity = [c for c, meta in UNIVERSE.items() if meta[2] == "equity" and c in closes.columns]
    cross = equity + [c for c in ("518880", "511260", "513100", "513500") if c in closes.columns]
    ret = closes.pct_change()
    adv = yuan.rolling(20).median()
    liquid = adv >= 50_000_000  # 50 million CNY median lot-notional
    daily = pd.Series(True, index=idx)
    monthly = month_end_flag(idx)
    weekly = every_n_flag(idx, 5)

    # 1. Buy and hold CSI 300 ETF. This is the benchmark.
    bh = pd.DataFrame(0.0, index=idx, columns=closes.columns)
    bh["510300"] = 1.0

    # 2. Time-series momentum on CSI 300. Cash when the trailing return is negative.
    mom120 = closes["510300"] / closes["510300"].shift(120) - 1.0
    tsmom = pd.DataFrame(0.0, index=idx, columns=closes.columns)
    tsmom["510300"] = (mom120 > 0).astype(float)

    # 3. Dual moving average on CSI 300.
    ma_fast = closes["510300"].rolling(20).mean()
    ma_slow = closes["510300"].rolling(60).mean()
    ma = pd.DataFrame(0.0, index=idx, columns=closes.columns)
    ma["510300"] = (ma_fast > ma_slow).astype(float)

    # 4. Cross-asset momentum, 6-month return skipping the last month.
    xs_score = closes.shift(21) / closes.shift(126) - 1.0
    xs = allocate_scores(xs_score, monthly, 3, allowed=cross, liquid=liquid)

    # 5. Five-day reversal inside the equity ETF universe.
    rev_score = closes / closes.shift(5) - 1.0
    rev = allocate_scores(rev_score, weekly, 3, smallest=True, allowed=equity, liquid=liquid)

    # 6. Low volatility: lowest 60-day realized vol.
    vol60 = ret.rolling(60).std()
    lowvol = allocate_scores(vol60, monthly, 3, smallest=True, allowed=equity, liquid=liquid)

    # 7. Regime barbell. Risk-on uses equity momentum; risk-off holds gold and the 10Y ETF.
    mkt_mom = closes["510300"] / closes["510300"].shift(60) - 1.0
    mkt_vol = ret["510300"].rolling(20).std()
    vol_med = mkt_vol.rolling(252).median()
    risk_on = (mkt_mom > 0) & (mkt_vol <= vol_med)
    mom60 = closes / closes.shift(60) - 1.0
    regime = pd.DataFrame(0.0, index=idx, columns=closes.columns)
    for dt in idx[weekly.to_numpy()]:
        if bool(risk_on.loc[dt]) if dt in risk_on.index and pd.notna(risk_on.loc[dt]) else False:
            row = mom60.loc[dt, equity]
            ok = liquid.loc[dt, equity].fillna(False).astype(bool)
            row = row.where(ok).dropna()
            if row.empty:
                continue
            pick = row.nlargest(min(2, len(row)))
            for code in pick.index:
                regime.loc[dt, code] = 1.0 / len(pick)
        else:
            regime.loc[dt, "518880"] = 0.5
            regime.loc[dt, "511260"] = 0.5

    # 8. Style rotation across growth, dividend, and mega-cap.
    styles = [c for c in ("159915", "510880", "510050") if c in closes.columns]
    style = allocate_scores(mom60, monthly, 1, allowed=styles, liquid=liquid)

    # 9. Donchian breakout. Prior channel excludes today. Empty book falls back to gold.
    prior_high = closes[equity].rolling(20).max().shift(1)
    breakout = closes[equity] > prior_high
    mom20 = closes / closes.shift(20) - 1.0
    don = pd.DataFrame(0.0, index=idx, columns=closes.columns)
    for dt in idx:
        names = [
            c
            for c in equity
            if pd.notna(breakout.loc[dt, c])
            and bool(breakout.loc[dt, c])
            and pd.notna(liquid.loc[dt, c])
            and bool(liquid.loc[dt, c])
        ]
        if not names:
            don.loc[dt, "518880"] = 1.0
            continue
        rank = mom20.loc[dt, names].dropna().sort_values(ascending=False)
        pick = list(rank.head(4).index)
        if not pick:
            don.loc[dt, "518880"] = 1.0
            continue
        for code in pick:
            don.loc[dt, code] = 1.0 / len(pick)

    # 10. Global overlay: weak CSI 300 moves the book toward Nasdaq ETF plus gold.
    mom120_all = closes["510300"] / closes["510300"].shift(120) - 1.0
    overlay = pd.DataFrame(0.0, index=idx, columns=closes.columns)
    for dt in idx[monthly.to_numpy()]:
        weak = bool(mom120_all.loc[dt] < 0) if pd.notna(mom120_all.loc[dt]) else True
        if weak:
            overlay.loc[dt, "513100"] = 0.5
            overlay.loc[dt, "518880"] = 0.5
        else:
            overlay.loc[dt, "510300"] = 0.7
            overlay.loc[dt, "518880"] = 0.3

    specs = {
        "BH_HS300": (bh, daily, "benchmark"),
        "TSMOM_HS300": (tsmom, daily, "time-series momentum"),
        "MA_20_60": (ma, daily, "moving-average trend"),
        "XSMOM_XASSET": (xs, monthly, "cross-asset momentum"),
        "REV_5D": (rev, weekly, "short-term reversal"),
        "LOWVOL_60": (lowvol, monthly, "low volatility"),
        "REGIME_BARBELL": (regime, weekly, "regime rotation"),
        "STYLE_ROTATE": (style, monthly, "style rotation"),
        "DONCHIAN_20": (don, daily, "channel breakout"),
        "GLOBAL_OVERLAY": (overlay, monthly, "global overlay"),
    }
    return {name: (frame, flag) for name, (frame, flag, _kind) in specs.items()}


def run_backtest(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    target: pd.DataFrame,
    flags: pd.Series,
    *,
    cost_bps: float,
    sample_start: pd.Timestamp = SAMPLE_START,
) -> dict[str, pd.Series]:
    """Fill yesterday's scheduled target at today's open. Mark to today's close."""
    assets = list(closes.columns)
    all_dates = list(closes.index)
    sim_dates = [d for d in all_dates if d >= sample_start]
    if not sim_dates:
        raise RuntimeError("empty simulation window")
    prev_map = {all_dates[i]: all_dates[i - 1] for i in range(1, len(all_dates))}
    cash = 1.0
    shares = {a: 0.0 for a in assets}
    equity_vals = []
    turnover_vals = []
    exposure_vals = []
    for dt in sim_dates:
        prev = prev_map.get(dt)
        o = opens.loc[dt]
        c = closes.loc[dt]
        traded = 0.0
        if prev is not None and bool(flags.loc[prev]):
            equity_open = cash
            for asset in assets:
                px = o[asset]
                if shares[asset] and pd.notna(px):
                    equity_open += shares[asset] * float(px)
            desired = {}
            tgt = target.loc[prev]
            gross = float(tgt.clip(lower=0).sum())
            scale = 1.0 if gross <= 1 else 1.0 / gross
            for asset in assets:
                px = o[asset]
                weight = max(float(tgt[asset]) if pd.notna(tgt[asset]) else 0.0, 0.0) * scale
                if weight <= 0 or pd.isna(px) or float(px) <= 0:
                    desired[asset] = 0.0
                else:
                    desired[asset] = equity_open * weight / float(px)
            for asset in assets:
                px = o[asset]
                if pd.isna(px) or float(px) <= 0:
                    continue
                traded += abs(desired[asset] - shares[asset]) * float(px)
                shares[asset] = desired[asset]
            invested_open = 0.0
            for asset in assets:
                px = o[asset]
                if pd.notna(px):
                    invested_open += shares[asset] * float(px)
            cost = traded * cost_bps / 10000.0
            cash = equity_open - invested_open - cost
            if cash < 0 and invested_open > 0:
                # Pay costs from positions when the book was fully invested.
                shrink = max(equity_open - cost, 0.0) / invested_open
                for asset in assets:
                    shares[asset] *= shrink
                invested_open = sum(
                    shares[asset] * float(o[asset])
                    for asset in assets
                    if pd.notna(o[asset])
                )
                cash = equity_open - cost - invested_open
        equity_close = cash
        invested_close = 0.0
        for asset in assets:
            px = c[asset]
            if shares[asset] and pd.notna(px):
                invested_close += shares[asset] * float(px)
        equity_close += invested_close
        base = equity_close if equity_close else np.nan
        equity_vals.append(equity_close)
        turnover_vals.append(traded / base if base else 0.0)
        exposure_vals.append(invested_close / base if base else 0.0)
    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(sim_dates), name="equity")
    turnover = pd.Series(turnover_vals, index=equity.index, name="turnover")
    exposure = pd.Series(exposure_vals, index=equity.index, name="exposure")
    # Daily close-to-close result. The first session is measured from the cash start of 1.0.
    net = equity.pct_change()
    net.iloc[0] = equity.iloc[0] - 1.0
    return {"equity": equity, "net": net, "turnover": turnover, "exposure": exposure}


def performance(net: pd.Series) -> dict[str, float]:
    r = net.dropna()
    if len(r) < 5:
        return {}
    wealth = float((1.0 + r).prod())
    total = wealth - 1.0
    span_days = max((r.index[-1] - r.index[0]).days, 1)
    years = span_days / 365.25
    cagr = wealth ** (1.0 / years) - 1.0 if wealth > 0 else float("nan")
    vol = float(r.std() * math.sqrt(252))
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std() * math.sqrt(252)) if len(downside) and downside.std() > 0 else 0.0
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    monthly = (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0
    sharpe_se = math.sqrt((1.0 + 0.5 * sharpe ** 2) / max(years, 1e-9))
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sharpe_se": sharpe_se,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "daily_win_rate": float((r > 0).mean()),
        "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "profit_factor": pf,
        "n_days": int(len(r)),
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else float("nan"),
    }


def slice_net(net: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    out = net
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out


def yearly_returns(net: pd.Series) -> dict[str, float]:
    if net.empty:
        return {}
    grouped = (1.0 + net).groupby(net.index.year).prod() - 1.0
    return {str(int(year)): float(val) for year, val in grouped.items()}


def buy_hold_reference(opens: pd.DataFrame, closes: pd.DataFrame, code: str, cost_bps: float) -> float:
    """Buy at the first in-sample open and hold to the last close, one entry cost."""
    dates = closes.index[closes.index >= SAMPLE_START]
    first, last = dates[0], dates[-1]
    entry = float(opens.loc[first, code])
    exit_px = float(closes.loc[last, code])
    return (exit_px / entry) * (1.0 - cost_bps / 10000.0) - 1.0


def variant_weights(closes: pd.DataFrame, yuan: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series, str]]:
    """Neighbor parameters. Used only to see whether a result is a single-point fluke."""
    idx = closes.index
    equity = [c for c, meta in UNIVERSE.items() if meta[2] == "equity" and c in closes.columns]
    cross = equity + [c for c in ("518880", "511260", "513100", "513500") if c in closes.columns]
    ret = closes.pct_change()
    adv = yuan.rolling(20).median()
    liquid = adv >= 50_000_000
    daily = pd.Series(True, index=idx)
    monthly = month_end_flag(idx)
    out = {}

    def add(name: str, frame: pd.DataFrame, flag: pd.Series, family: str) -> None:
        out[name] = (frame, flag, family)

    for lookback in (60, 120, 200):
        mom = closes["510300"] / closes["510300"].shift(lookback) - 1.0
        frame = pd.DataFrame(0.0, index=idx, columns=closes.columns)
        frame["510300"] = (mom > 0).astype(float)
        add(f"TSMOM_{lookback}", frame, daily, "TSMOM_HS300")

    for fast, slow in ((10, 40), (20, 60), (50, 200)):
        frame = pd.DataFrame(0.0, index=idx, columns=closes.columns)
        frame["510300"] = (
            closes["510300"].rolling(fast).mean() > closes["510300"].rolling(slow).mean()
        ).astype(float)
        add(f"MA_{fast}_{slow}", frame, monthly if False else daily, "MA_20_60")

    for lookback, skip, k in ((63, 10, 3), (126, 21, 3), (252, 21, 3), (126, 21, 2), (126, 21, 4)):
        score = closes.shift(skip) / closes.shift(lookback) - 1.0
        add(
            f"XS_{lookback}_{skip}_k{k}",
            allocate_scores(score, monthly, k, allowed=cross, liquid=liquid),
            monthly,
            "XSMOM_XASSET",
        )

    for n, k in ((3, 3), (5, 3), (10, 3)):
        score = closes / closes.shift(n) - 1.0
        add(
            f"REV_{n}",
            allocate_scores(score, every_n_flag(idx, n), k, smallest=True, allowed=equity, liquid=liquid),
            every_n_flag(idx, n),
            "REV_5D",
        )

    for window in (20, 60, 120):
        vol = ret.rolling(window).std()
        add(
            f"LOWVOL_{window}",
            allocate_scores(vol, monthly, 3, smallest=True, allowed=equity, liquid=liquid),
            monthly,
            "LOWVOL_60",
        )

    for window in (10, 20, 55):
        prior_high = closes[equity].rolling(window).max().shift(1)
        breakout = closes[equity] > prior_high
        mom = closes / closes.shift(window) - 1.0
        frame = pd.DataFrame(0.0, index=idx, columns=closes.columns)
        for dt in idx:
            names = [
                c
                for c in equity
                if pd.notna(breakout.loc[dt, c])
                and bool(breakout.loc[dt, c])
                and pd.notna(liquid.loc[dt, c])
                and bool(liquid.loc[dt, c])
            ]
            if not names:
                frame.loc[dt, "518880"] = 1.0
                continue
            pick = list(mom.loc[dt, names].dropna().sort_values(ascending=False).head(4).index)
            if not pick:
                frame.loc[dt, "518880"] = 1.0
                continue
            for code in pick:
                frame.loc[dt, code] = 1.0 / len(pick)
        add(f"DONCHIAN_{window}", frame, daily, "DONCHIAN_20")

    # Regime and overlay neighbors. These were not used to pick the primary rules.
    ret = closes.pct_change()
    adv = yuan.rolling(20).median()
    liquid = adv >= 50_000_000
    equity = [c for c, meta in UNIVERSE.items() if meta[2] == "equity" and c in closes.columns]
    for mom_n, vol_n, step, k in (
        (40, 20, 5, 2),
        (60, 20, 5, 2),
        (120, 20, 5, 2),
        (60, 10, 5, 2),
        (60, 40, 5, 2),
        (60, 20, 21, 2),
        (60, 20, 5, 1),
        (60, 20, 5, 3),
    ):
        mkt_mom = closes["510300"] / closes["510300"].shift(mom_n) - 1.0
        mkt_vol = ret["510300"].rolling(vol_n).std()
        vol_med = mkt_vol.rolling(252).median()
        risk_on = (mkt_mom > 0) & (mkt_vol <= vol_med)
        mom = closes / closes.shift(mom_n) - 1.0
        flag = every_n_flag(idx, step) if step != 21 else monthly
        frame = pd.DataFrame(0.0, index=idx, columns=closes.columns)
        for dt in idx[flag.to_numpy()]:
            on = bool(risk_on.loc[dt]) if pd.notna(risk_on.loc[dt]) else False
            if on:
                row = mom.loc[dt, equity]
                ok = liquid.loc[dt, equity].fillna(False).astype(bool)
                row = row.where(ok).dropna()
                if row.empty:
                    continue
                pick = row.nlargest(min(k, len(row)))
                for code in pick.index:
                    frame.loc[dt, code] = 1.0 / len(pick)
            else:
                frame.loc[dt, "518880"] = 0.5
                frame.loc[dt, "511260"] = 0.5
        add(f"REGIME_m{mom_n}_v{vol_n}_s{step}_k{k}", frame, flag, "REGIME_BARBELL")

    for lookback, risk_eq, risk_gold, def_nas, def_gold in (
        (120, 0.70, 0.30, 0.50, 0.50),
        (60, 0.70, 0.30, 0.50, 0.50),
        (200, 0.70, 0.30, 0.50, 0.50),
        (120, 1.00, 0.00, 0.50, 0.50),
        (120, 0.70, 0.30, 0.00, 1.00),
        (120, 0.50, 0.50, 0.50, 0.50),
    ):
        mom = closes["510300"] / closes["510300"].shift(lookback) - 1.0
        frame = pd.DataFrame(0.0, index=idx, columns=closes.columns)
        for dt in idx[monthly.to_numpy()]:
            weak = bool(mom.loc[dt] < 0) if pd.notna(mom.loc[dt]) else True
            if weak:
                if def_nas:
                    frame.loc[dt, "513100"] = def_nas
                frame.loc[dt, "518880"] = def_gold
            else:
                frame.loc[dt, "510300"] = risk_eq
                if risk_gold:
                    frame.loc[dt, "518880"] = risk_gold
        add(
            f"OVERLAY_{lookback}_{int(risk_eq*100)}_{int(def_nas*100)}",
            frame,
            monthly,
            "GLOBAL_OVERLAY",
        )
    return out


def fmt_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"
    return f"{value * 100:.2f}%"


def write_svg(curves: dict[str, pd.Series], path: Path) -> None:
    """Compact equity chart. Wealth is rebased to 1 at the sample start."""
    names = list(curves)
    if not names:
        return
    width, height = 960, 420
    left, right, top, bottom = 56, 16, 16, 36
    series = {}
    for name, equity in curves.items():
        wealth = equity / equity.iloc[0]
        series[name] = wealth
    stacked = pd.concat(series, axis=1).dropna(how="all")
    ymin = float(stacked.min().min())
    ymax = float(stacked.max().max())
    if ymin == ymax:
        ymax = ymin + 1
    ymin = min(ymin, 1.0)
    n = len(stacked)

    def xy(i: int, val: float) -> tuple[float, float]:
        x = left + (width - left - right) * (i / max(n - 1, 1))
        y = top + (height - top - bottom) * (1 - (val - ymin) / (ymax - ymin))
        return x, y

    colors = [
        "#9aa0a6",
        "#0b6e4f",
        "#1d4e89",
        "#b45309",
        "#9f1239",
        "#6d28d9",
        "#0f766e",
        "#b45309",
        "#1f2937",
        "#0369a1",
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="14" font-size="12" font-family="sans-serif">Equity, rebased (2018 start = 1), after costs</text>',
    ]
    # zero-growth line at 1.0 if inside range
    if ymin <= 1 <= ymax:
        y = xy(0, 1.0)[1]
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
    for i, name in enumerate(names):
        vals = stacked[name].to_numpy()
        pts = []
        for j, val in enumerate(vals):
            if np.isnan(val):
                continue
            x, y = xy(j, float(val))
            pts.append(f"{x:.1f},{y:.1f}")
        color = colors[i % len(colors)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{" ".join(pts)}"/>')
        parts.append(
            f'<text x="{left + 140 * (i % 6)}" y="{height - 8 if i < 6 else height - 22}" font-size="11" '
            f'font-family="sans-serif" fill="{color}">{name}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probes = probe_sources()
    print("downloading tencent qfq bars...")
    bars: dict[str, pd.DataFrame] = {}
    errors = {}

    def _load(code: str):
        prefix = UNIVERSE[code][0]
        return code, load_tencent_bars(code, prefix)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_load, code) for code in UNIVERSE]
        for fut in as_completed(futures):
            try:
                code, frame = fut.result()
                bars[code] = frame
                print(f"  {code} {UNIVERSE[code][1]} {frame.index[0].date()} -> {frame.index[-1].date()} n={len(frame)}")
            except Exception as exc:  # noqa: BLE001 — record the gap, continue
                # future holds the code only after success; recover from the exception message
                errors[str(exc)] = str(exc)
                print("  FAIL", exc)
    if errors:
        (OUT_DIR / "download_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2))
    trade_codes = [c for c, meta in UNIVERSE.items() if meta[2] != "snapshot" and c in bars]
    missing_trade = [c for c, meta in UNIVERSE.items() if meta[2] != "snapshot" and c not in bars]
    if missing_trade:
        raise SystemExit(f"missing tradeable history: {missing_trade}")

    spot = fetch_spot_quotes(list(bars))
    crosses = []
    for code, frame in bars.items():
        last_bar = float(frame["close"].iloc[-1])
        quote = spot.get(code)
        if not quote:
            crosses.append({"code": code, "bar": last_bar, "spot": None, "match": False})
            continue
        match = abs(last_bar - quote["last"]) < 0.02 or abs(last_bar / quote["last"] - 1) < 0.003
        crosses.append(
            {
                "code": code,
                "name": UNIVERSE[code][1],
                "bar_date": str(frame.index[-1].date()),
                "bar": last_bar,
                "spot": quote["last"],
                "spot_ts": quote["timestamp"],
                "match": bool(match),
            }
        )
        if not match:
            print(f"  PRICE MISMATCH {code} bar={last_bar} spot={quote['last']}")
    if not all(row["match"] for row in crosses if row.get("spot") is not None):
        raise SystemExit("refusing to backtest: qfq last bar does not match spot quote")

    opens, closes, yuan = build_panels({c: bars[c] for c in bars})
    # Tradeable panel drops snapshot names from the engine columns by keeping them
    # available for the snapshot, while strategies only allocate to trade codes.
    snap_rows = snapshot_table(closes)
    attach_adv(snap_rows, yuan)
    index_snap = fetch_index_snapshot()
    news = fetch_news()
    macro = fetch_yahoo_macro()

    trade_opens = opens[trade_codes]
    trade_closes = closes[trade_codes]
    trade_yuan = yuan[trade_codes]
    specs = strategy_weights(trade_closes, trade_yuan)
    results = {}
    curves = {}
    nets = {}
    for name, (target, flags) in specs.items():
        sim = run_backtest(trade_opens, trade_closes, target, flags, cost_bps=COST_BPS)
        nets[name] = sim["net"]
        curves[name] = sim["equity"]
        full = performance(sim["net"])
        ins = performance(slice_net(sim["net"], None, IS_END))
        oos = performance(slice_net(sim["net"], OOS_START, None))
        decisions = target.loc[flags.reindex(target.index).fillna(False).to_numpy()]
        avg_w = decisions.mean()
        last_dt = target.index[-1]
        # Last decision that would be sent to the next open.
        if bool(flags.reindex(target.index).fillna(False).loc[last_dt]):
            last_target_dt = last_dt
        else:
            decided = flags.reindex(target.index).fillna(False)
            last_target_dt = decided[decided].index[-1]
        last_w = target.loc[last_target_dt]
        last_w = {c: round(float(v), 4) for c, v in last_w.items() if float(v) > 0}
        avg_w = {c: round(float(v), 4) for c, v in avg_w.items() if float(v) > 0.01}
        results[name] = {
            "full": full,
            "in_sample": ins,
            "out_of_sample": oos,
            "yearly": yearly_returns(sim["net"]),
            "avg_exposure": float(sim["exposure"].mean()),
            "annual_turnover": float(sim["turnover"].sum() / max((sim["net"].index[-1] - sim["net"].index[0]).days / 365.25, 1e-9)),
            "cost_bps": COST_BPS,
            "avg_decision_weights": avg_w,
            "last_decision_date": str(pd.Timestamp(last_target_dt).date()),
            "last_decision_weights": last_w,
        }
        print(
            f"{name:16} CAGR {fmt_pct(full['cagr'])} Sharpe {full['sharpe']:.2f} "
            f"DD {fmt_pct(full['max_drawdown'])} OOS Sharpe {oos['sharpe']:.2f} "
            f"OOS CAGR {fmt_pct(oos['cagr'])} TO {results[name]['annual_turnover']:.1f}x"
        )

    ref = buy_hold_reference(trade_opens, trade_closes, "510300", COST_BPS)
    bh_total = results["BH_HS300"]["full"]["total_return"]
    print(f"BH engine total {fmt_pct(bh_total)} vs open-to-last-close reference {fmt_pct(ref)}")

    # Cost and delay stresses on the primary rules.
    stresses = {}
    for bps in (25.0, 40.0):
        stresses[f"cost_{int(bps)}bps"] = {}
        for name, (target, flags) in specs.items():
            sim = run_backtest(trade_opens, trade_closes, target, flags, cost_bps=bps)
            stresses[f"cost_{int(bps)}bps"][name] = {
                "full": performance(sim["net"]),
                "out_of_sample": performance(slice_net(sim["net"], OOS_START, None)),
            }

    # One extra session of delay: move the decision and the target together.
    # Trading the shifted flag against an unshifted (often zero) target would
    # flatten every non-daily rule, which is not a delay test.
    stresses["extra_delay"] = {}
    for name, (target, flags) in specs.items():
        delayed_target = target.shift(1)
        delayed_flags = flags.shift(1).fillna(False).astype(bool)
        sim = run_backtest(trade_opens, trade_closes, delayed_target, delayed_flags, cost_bps=COST_BPS)
        stresses["extra_delay"][name] = {
            "full": performance(sim["net"]),
            "out_of_sample": performance(slice_net(sim["net"], OOS_START, None)),
        }

    print("running parameter neighbors...")
    neighbors = variant_weights(trade_closes, trade_yuan)
    neighbor_stats = {}
    for name, (target, flags, family) in neighbors.items():
        sim = run_backtest(trade_opens, trade_closes, target, flags, cost_bps=COST_BPS)
        oos = performance(slice_net(sim["net"], OOS_START, None))
        full = performance(sim["net"])
        neighbor_stats[name] = {
            "family": family,
            "full_sharpe": full.get("sharpe"),
            "full_cagr": full.get("cagr"),
            "full_max_drawdown": full.get("max_drawdown"),
            "oos_sharpe": oos.get("sharpe"),
            "oos_cagr": oos.get("cagr"),
            "oos_max_drawdown": oos.get("max_drawdown"),
        }
        print(f"  {name:20} full Sharpe {full['sharpe']:.2f} OOS Sharpe {oos['sharpe']:.2f} OOS CAGR {fmt_pct(oos['cagr'])}")

    # Correlation of daily net returns.
    corr = pd.DataFrame(nets).corr().round(3)

    # Subperiod for the market itself.
    bh_net = nets["BH_HS300"]
    market_years = yearly_returns(bh_net)

    svg_path = OUT_DIR / "equity.svg"
    write_svg(curves, svg_path)
    # Also drop a copy next to the report so the markdown can reference a stable path.
    report_svg = ROOT / "reports" / "trading_strategy_research_20261003_equity.svg"
    report_svg.write_text(svg_path.read_text())

    adv_check = {
        code: float(trade_yuan[code].tail(60).median() / 1e6)
        for code in trade_codes
    }

    payload = {
        "asof": ASOF,
        "sample_start": str(SAMPLE_START.date()),
        "sample_end": str(trade_closes.index[-1].date()),
        "is_end": str(IS_END.date()),
        "oos_start": str(OOS_START.date()),
        "cost_bps_one_way": COST_BPS,
        "cash_return": 0.0,
        "probes": probes,
        "price_crosscheck": crosses,
        "index_snapshot": index_snap,
        "macro": macro,
        "news": news,
        "snapshot": snap_rows,
        "adv60_median_mn_cny": adv_check,
        "results": results,
        "stresses": stresses,
        "neighbors": neighbor_stats,
        "correlation": corr.to_dict(),
        "bh_reference_total_return": ref,
        "market_years_bh": market_years,
        "assumptions": {
            "prices": "Tencent qfq via curl, two-year chunks, overlap checked",
            "execution": "signal at close t, fill next open, hold shares between rebalances",
            "costs": "12 bps one-way primary; 25 and 40 bps stress; cash earns 0",
            "shorts": "not used",
            "leverage": "not used",
            "benchmark": "510300 buy and hold, same cost model",
            "split": "in-sample through 2022-12-31, out-of-sample from 2023-01-01",
        },
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
