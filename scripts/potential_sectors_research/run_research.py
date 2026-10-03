#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Potential-sector scan + stock ranking for A-shares (trading-strategy-research).

Identifies latent / rotating sectors using fund-flow + relative-strength evidence,
then ranks liquid constituents with a multi-factor score and simple backtests.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

COST_BPS = 15.0
START, END = "20220101", "20260928"
IS_END = "20241231"

# Potential sectors: latent opportunity theses (not "what's hottest today" alone)
SECTORS: Dict[str, Dict] = {
    "电力公用": {
        "thesis": "科技拥挤兑现后的避险承接；公用事业现金流稳定，资金净流入居前",
        "horizon": "short-medium",
        "risk": "若科技快速V反，资金回流后弹性不足",
        "stocks": {
            "600795": "国电电力",
            "600011": "华能国际",
            "600900": "长江电力",
            "601985": "中国核电",
            "600023": "浙能电力",
            "600886": "国投电力",
        },
    },
    "汽车整车": {
        "thesis": "央国企整合/换电政策催化 + 行业承压后的底部弹性；当日资金流入居前",
        "horizon": "short-medium",
        "risk": "重组节奏不确定；内销仍弱，主题炒作成分高",
        "stocks": {
            "600418": "江淮汽车",
            "002594": "比亚迪",
            "000625": "长安汽车",
            "600104": "上汽集团",
            "601238": "广汽集团",
            "600733": "北汽蓝谷",
        },
    },
    "生猪养殖": {
        "thesis": "深度亏损+产能去化的中期周期底部；资金开始回流，但期货仍弱",
        "horizon": "medium",
        "risk": "反转或延至2027；短期供应压力，左侧波动大",
        "stocks": {
            "002714": "牧原股份",
            "300498": "温氏股份",
            "000876": "新希望",
            "002567": "唐人神",
            "002124": "天邦食品",
        },
    },
    "能源资源": {
        "thesis": "红利+通胀/地缘预期；煤炭石化资金流入，中金高胜率框架靠前",
        "horizon": "medium",
        "risk": "商品价格回落则盈利下修；弹性弱于成长主题",
        "stocks": {
            "601088": "中国神华",
            "601225": "陕西煤业",
            "601857": "中国石油",
            "600938": "中国海油",
            "002493": "荣盛石化",
            "600028": "中国石化",
        },
    },
    "创新药": {
        "thesis": "盈利兑现/出海改善；相对科技杀估值更具基本面支撑（赔率修复）",
        "horizon": "medium",
        "risk": "政策与临床失败；成长风格退潮时同步承压",
        "stocks": {
            "600276": "恒瑞医药",
            "603259": "药明康德",
            "300122": "智飞生物",
            "000661": "长春高新",
            "688180": "君实生物",
            "300760": "迈瑞医疗",
        },
    },
}

# Contrast sleeve: overcrowded tech (document, do NOT recommend chase)
TECH_CONTRAST = {
    "300308": "中际旭创",
    "300502": "新易盛",
    "688256": "寒武纪",
    "601138": "工业富联",
}


def _ak_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def fetch_stock(code: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist_tx(symbol=_ak_symbol(code), start_date=START, end_date=END, adjust="qfq")
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    df["code"] = code
    return df


def fetch_index(symbol: str = "sh000300") -> pd.DataFrame:
    # stock_zh_index_daily uses sz/sh prefixes; HS300
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
    except Exception:
        df = ak.stock_zh_index_daily(symbol="sh000001")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2022-01-01") & (df["date"] <= "2026-09-28")].sort_values("date")
    return df.reset_index(drop=True)


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    names: Dict[str, str] = {}
    frames = []
    for sec, meta in SECTORS.items():
        for code, name in meta["stocks"].items():
            names[code] = name
    for code, name in TECH_CONTRAST.items():
        names[code] = name

    for code, name in names.items():
        try:
            df = fetch_stock(code)
            if len(df) < 180:
                print(f"skip {code} {name}: short")
                continue
            frames.append(df)
            print(f"ok {code} {name}: {len(df)}")
        except Exception as exc:  # noqa: BLE001
            print(f"fail {code} {name}: {exc}")
        time.sleep(0.35)
    panel = pd.concat(frames, ignore_index=True)
    idx = fetch_index("sh000300")
    return panel, idx, names


def pivot(panel: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    return panel.pivot(index="date", columns="code", values=col).sort_index().ffill(limit=3)


def metrics_from_equity(equity: pd.Series, freq: int = 252) -> Dict[str, float]:
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {}
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    vol = float(rets.std() * math.sqrt(freq))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(freq)) if rets.std() > 0 else 0.0
    down = rets[rets < 0]
    sortino = float(rets.mean() / down.std() * math.sqrt(freq)) if len(down) and down.std() > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1).min())
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "win_rate": float((rets > 0).mean()),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
    }


def portfolio_from_weights(w: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    w_exec = w.reindex(rets.index).fillna(0.0).shift(1).fillna(0.0)
    gross = (w_exec * rets).sum(axis=1)
    turnover = w_exec.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (COST_BPS / 10000.0)
    return (1 + net).cumprod()


def score_stocks(close: pd.DataFrame, vol: pd.DataFrame, idx_close: pd.Series, names: Dict[str, str]) -> pd.DataFrame:
    """Multi-factor score for ranking (higher = more attractive as latent long)."""
    last = close.index[-1]
    rows = []
    idx_ret_3m = idx_close.pct_change(63).loc[last]
    idx_ret_6m = idx_close.pct_change(126).loc[last]
    for code in close.columns:
        px = close[code].dropna()
        if len(px) < 120:
            continue
        r1m = px.pct_change(21).iloc[-1]
        r3m = px.pct_change(63).iloc[-1]
        r6m = px.pct_change(126).iloc[-1]
        r1y = px.pct_change(252).iloc[-1] if len(px) > 252 else np.nan
        ma20, ma60 = px.rolling(20).mean().iloc[-1], px.rolling(60).mean().iloc[-1]
        # Distance from 120d high (mean-reversion capacity for beaten-down names)
        hh = px.rolling(120).max().iloc[-1]
        dd_from_high = px.iloc[-1] / hh - 1.0
        # Realized vol
        rv = px.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252)
        # Relative strength vs HS300
        rs_3m = r3m - idx_ret_3m
        rs_6m = r6m - idx_ret_6m
        # Volume trend
        v = vol[code].dropna() if code in vol.columns else pd.Series(dtype=float)
        vol_ratio = (v.iloc[-5:].mean() / v.iloc[-60:].mean()) if len(v) > 60 and v.iloc[-60:].mean() > 0 else 1.0

        sector = next((s for s, m in SECTORS.items() if code in m["stocks"]), "科技对照")
        # Score components (z-like clipped): prefer mild relative strength, not over-extended;
        # reward not too far below MA60 for trend quality OR deep discount for cycle names.
        score = 0.0
        # RS
        score += float(np.clip(rs_3m, -0.4, 0.4)) * 2.0
        score += float(np.clip(rs_6m, -0.5, 0.5)) * 1.0
        # Trend quality
        if px.iloc[-1] > ma60:
            score += 0.15
        if ma20 > ma60:
            score += 0.10
        # For cycle/defensive: some drawdown can be opportunity (not free-fall)
        if sector in ("生猪养殖", "能源资源", "电力公用"):
            if -0.35 <= dd_from_high <= -0.08:
                score += 0.20
            if dd_from_high < -0.45:
                score -= 0.15  # too broken
        else:
            if dd_from_high < -0.35:
                score -= 0.10
        # Volatility penalty for lottery tickets
        if rv > 0.55:
            score -= 0.15
        # Volume confirmation
        if vol_ratio > 1.2:
            score += 0.05
        if vol_ratio > 2.5:
            score -= 0.05  # climax risk

        rows.append(
            {
                "code": code,
                "name": names.get(code, code),
                "sector": sector,
                "close": float(px.iloc[-1]),
                "ret_1m": float(r1m) if pd.notna(r1m) else None,
                "ret_3m": float(r3m) if pd.notna(r3m) else None,
                "ret_6m": float(r6m) if pd.notna(r6m) else None,
                "ret_1y": float(r1y) if pd.notna(r1y) else None,
                "rs_3m": float(rs_3m) if pd.notna(rs_3m) else None,
                "rs_6m": float(rs_6m) if pd.notna(rs_6m) else None,
                "dd_from_120h": float(dd_from_high),
                "ann_vol_20d": float(rv) if pd.notna(rv) else None,
                "above_ma60": bool(px.iloc[-1] > ma60),
                "vol_ratio_5_60": float(vol_ratio),
                "score": float(score),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def sector_ew_series(close: pd.DataFrame, codes: List[str]) -> pd.Series:
    cols = [c for c in codes if c in close.columns]
    return close[cols].mean(axis=1)


def backtest_strategies(close: pd.DataFrame, idx: pd.DataFrame) -> Dict[str, Dict]:
    """Backtest sector sleeves and a monthly sector-rotation strategy."""
    rets = close.pct_change().fillna(0.0)
    idx = idx.set_index("date").reindex(close.index).ffill()
    idx_eq = (1 + idx["close"].pct_change().fillna(0)).cumprod()

    results = {}
    sector_px = {}
    for sec, meta in SECTORS.items():
        codes = [c for c in meta["stocks"] if c in close.columns]
        if len(codes) < 2:
            continue
        w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        w[codes] = 1.0 / len(codes)
        eq = portfolio_from_weights(w, rets)
        results[f"EW_{sec}"] = {
            "full": metrics_from_equity(eq),
            "oos": metrics_from_equity(eq[eq.index > IS_END] / eq[eq.index > IS_END].iloc[0]),
            "equity": eq,
            "notes": f"Equal-weight {sec}: {meta['thesis'][:40]}",
        }
        sector_px[sec] = sector_ew_series(close, codes)

    # Tech contrast EW
    tcodes = [c for c in TECH_CONTRAST if c in close.columns]
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[tcodes] = 1.0 / len(tcodes)
    eq = portfolio_from_weights(w, rets)
    results["EW_科技对照"] = {
        "full": metrics_from_equity(eq),
        "oos": metrics_from_equity(eq[eq.index > IS_END] / eq[eq.index > IS_END].iloc[0]),
        "equity": eq,
        "notes": "Crowded AI/compute contrast sleeve",
    }

    # Monthly rotate into best 60d sector among potential set; cash if all 60d < 0
    sec_df = pd.DataFrame(sector_px).dropna(how="all")
    mom = sec_df.pct_change(60)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    months = close.index.to_period("M")
    reb = close.groupby(months).apply(lambda x: x.index[-1])
    for dt in reb:
        if dt not in mom.index:
            continue
        row = mom.loc[dt].dropna()
        pos = row[row > 0]
        if pos.empty:
            continue
        best = pos.idxmax()
        codes = [c for c in SECTORS[best]["stocks"] if c in close.columns]
        if not codes:
            continue
        w.loc[dt, codes] = 1.0 / len(codes)
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    eq = portfolio_from_weights(w, rets)
    results["Sector_Rotation_DualMom"] = {
        "full": metrics_from_equity(eq),
        "oos": metrics_from_equity(eq[eq.index > IS_END] / eq[eq.index > IS_END].iloc[0]),
        "equity": eq,
        "notes": "Monthly long strongest potential sector with positive 60d momentum else cash",
    }

    # Top-score static (use last score — for live ranking only; backtest via RS proxy)
    # Momentum top-N across potential universe monthly
    pot_codes = [c for sec in SECTORS.values() for c in sec["stocks"] if c in close.columns]
    mom_s = close[pot_codes].pct_change(60)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for dt in reb:
        row = mom_s.loc[dt].dropna()
        pos = row[row > 0].nlargest(5)
        if pos.empty:
            continue
        w.loc[dt, pos.index] = 1.0 / len(pos)
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    eq = portfolio_from_weights(w, rets)
    results["Stock_DualMom_Top5"] = {
        "full": metrics_from_equity(eq),
        "oos": metrics_from_equity(eq[eq.index > IS_END] / eq[eq.index > IS_END].iloc[0]),
        "equity": eq,
        "notes": "Monthly top-5 dual-momentum names within potential sectors",
    }

    results["HS300"] = {
        "full": metrics_from_equity(idx_eq),
        "oos": metrics_from_equity(idx_eq[idx_eq.index > IS_END] / idx_eq[idx_eq.index > IS_END].iloc[0]),
        "equity": idx_eq,
        "notes": "沪深300 buy&hold",
    }
    return results


def live_fund_flow() -> Dict:
    flow = ak.stock_fund_flow_industry(symbol="即时")
    ths = ak.stock_board_industry_summary_ths()
    idx = ak.stock_zh_index_spot_sina()
    mkt = {}
    for code in ["sh000001", "sz399006", "sh000300"]:
        r = idx[idx["代码"] == code]
        if len(r):
            mkt[code] = {
                "name": r.iloc[0]["名称"],
                "pct": float(r.iloc[0]["涨跌幅"]),
                "last": float(r.iloc[0]["最新价"]),
            }
    return {
        "asof": str(pd.Timestamp.today().date()),
        "market": mkt,
        "flow_top": flow.sort_values("净额", ascending=False).head(10).to_dict(orient="records"),
        "flow_bottom": flow.sort_values("净额", ascending=True).head(8).to_dict(orient="records"),
        "ths_top": ths.sort_values("净流入", ascending=False).head(10).to_dict(orient="records"),
    }


def pct(x) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.1%}"


def num(x) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.2f}"


def write_report(scores: pd.DataFrame, results: Dict, flow: Dict, names: Dict[str, str]) -> Path:
    # Top picks: exclude tech contrast; take top 2 per sector then global top
    pot = scores[scores["sector"] != "科技对照"].copy()
    top_global = pot.head(12)
    top_per = pot.groupby("sector", group_keys=False).head(2).sort_values("score", ascending=False)

    lines = []
    lines += [
        "# 潜在板块与个股推荐研究报告",
        "",
        f"- 数据截止: **{scores.attrs.get('asof', flow.get('asof'))}**",
        "- 方法: 资金流扫描 + 多因子打分 + 板块/个股动量回测（单边成本 15bps）",
        "- 基准: 沪深300；对照袖: 高拥挤算力科技",
        "- 复现: `python3 scripts/potential_sectors_research/run_research.py`",
        "- **本报告为研究回测，不构成投资建议**",
        "",
        "## 1. Executive Summary",
        "",
    ]
    mkt = flow.get("market", {})
    if mkt:
        bits = ", ".join(f"{v['name']} {v['pct']:.2f}%" for v in mkt.values())
        lines.append(f"当日市场: {bits}。资金从高位科技（通信设备/半导体等大额净流出）切向防御与底部板块。")
    lines += [
        "",
        "**潜在板块（按研究优先级）**:",
        "1. **电力公用** — 避险承接，资金净流入靠前，适合降波配置",
        "2. **汽车整车/换电** — 政策与重组催化，短线弹性最大但主题风险高",
        "3. **能源资源** — 红利+通胀预期，胜率框架靠前，弹性中等",
        "4. **生猪养殖** — 中期周期底部，短期仍承压，宜小仓左侧/等右侧",
        "5. **创新药** — 盈利兑现逻辑，相对科技杀估值更具基本面锚",
        "",
        "**明确不追**: 光模块/半导体等拥挤科技的下跌中继（对照袖用于风险提示）。",
        "",
        "### 个股推荐（综合分，潜在池内）",
        "",
        "| Rank | 代码 | 名称 | 板块 | Score | 3M | RS3M | 距120高 | >MA60 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for i, r in enumerate(top_global.to_dict("records"), 1):
        lines.append(
            f"| {i} | {r['code']} | {r['name']} | {r['sector']} | {r['score']:.2f} | {pct(r['ret_3m'])} | "
            f"{pct(r['rs_3m'])} | {pct(r['dd_from_120h'])} | {'Y' if r['above_ma60'] else 'N'} |"
        )

    lines += ["", "### 分板块优选（每板块 Top2）", ""]
    for r in top_per.to_dict("records"):
        lines.append(
            f"- **{r['sector']} / {r['name']}({r['code']})** — score {r['score']:.2f}, 3M {pct(r['ret_3m'])}, "
            f"RS3M {pct(r['rs_3m'])}, DD120 {pct(r['dd_from_120h'])}"
        )

    lines += ["", "## 2. Market Analysis", "", "### 当日行业资金净流入 Top", ""]
    for row in flow.get("flow_top", [])[:8]:
        lines.append(
            f"- {row.get('行业')}: 涨跌 {row.get('行业-涨跌幅')}%, 净额 {row.get('净额')}亿, "
            f"领涨 {row.get('领涨股')} ({row.get('领涨股-涨跌幅')}%)"
        )
    lines += ["", "### 当日行业资金净流出（拥挤兑现）", ""]
    for row in flow.get("flow_bottom", [])[:6]:
        lines.append(f"- {row.get('行业')}: 涨跌 {row.get('行业-涨跌幅')}%, 净额 {row.get('净额')}亿")

    lines += [
        "",
        "### Facts vs interpretation",
        "",
        "- **事实**: 创业板/科创大幅回调，通信设备与半导体净流出居前；电力、养殖、汽车、煤炭石化净流入。",
        "- **事实**: 生猪现货亏损加深，机构预期周期或延至 2027 才改善。",
        "- **解释**: 这是风格再平衡而非系统性崩盘信号；潜在机会在「低拥挤 + 有逻辑」而非继续追科技反弹。",
        "",
        "## 3. Strategy Details",
        "",
        "| Strategy | Thesis | Spec |",
        "| --- | --- | --- |",
        "| EW_板块袖 | 板块 beta | 各潜在板块内部等权 |",
        "| Sector_Rotation_DualMom | 景气轮动+绝对动量 | 月度持有60d最强且收益>0的潜在板块 |",
        "| Stock_DualMom_Top5 | 个股动量 | 潜在池内月度 Top5 正动量 |",
        "| EW_科技对照 | 拥挤风险对照 | 算力龙头等权 |",
        "",
        "打分因子: RS vs 沪深300、趋势(MA)、距120日高点折扣、波动惩罚、量能确认。",
        "代码: `scripts/potential_sectors_research/run_research.py`。",
        "",
        "## 4. Backtest Results",
        "",
        "| Strategy | CAGR | Sharpe | Sortino | MDD | OOS Sharpe | OOS CAGR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = sorted(
        results.items(),
        key=lambda kv: (kv[1]["full"].get("sharpe") or -99),
        reverse=True,
    )
    for name, blob in order:
        m, o = blob["full"], blob["oos"]
        lines.append(
            f"| {name} | {pct(m.get('cagr'))} | {num(m.get('sharpe'))} | {num(m.get('sortino'))} | "
            f"{pct(m.get('max_drawdown'))} | {num(o.get('sharpe'))} | {pct(o.get('cagr'))} |"
        )

    # Ranking narrative
    lines += [
        "",
        "## 5. Comparison & Ranking",
        "",
        "板块袖相对科技对照的意义：若防御/周期袖夏普更高、回撤更小，则当前再平衡方向成立。",
        "个股推荐以 **综合分** 为主，回测策略验证「潜在池动量」是否具备可交易性。",
        "",
    ]
    best_strats = [n for n, _ in order if n not in ("HS300", "EW_科技对照")][:4]
    for i, n in enumerate(best_strats, 1):
        m = results[n]["full"]
        lines.append(f"{i}. `{n}` — Sharpe {num(m.get('sharpe'))}, MDD {pct(m.get('max_drawdown'))}")

    # Recommendations buckets
    lines += [
        "",
        "## 6. Final Recommendations（个股）",
        "",
        "### A. 进攻弹性（短线，催化驱动，仓位宜小）",
        "",
    ]
    auto = pot[pot["sector"] == "汽车整车"].head(3)
    for r in auto.to_dict("records"):
        lines.append(f"- **{r['name']}({r['code']})** — score {r['score']:.2f}; 关注重组/换电催化失效则退出")

    lines += ["", "### B. 防御配置（降波、承接避险资金）", ""]
    pwr = pot[pot["sector"] == "电力公用"].head(3)
    for r in pwr.to_dict("records"):
        lines.append(f"- **{r['name']}({r['code']})** — score {r['score']:.2f}; 电力公用现金流资产")

    lines += ["", "### C. 中期左侧/红利（胜率框架）", ""]
    for sec in ("能源资源", "生猪养殖", "创新药"):
        sub = pot[pot["sector"] == sec].head(2)
        for r in sub.to_dict("records"):
            lines.append(
                f"- **{r['name']}({r['code']})** [{sec}] — score {r['score']:.2f}; 3M {pct(r['ret_3m'])}"
            )

    lines += [
        "",
        "### D. 不推荐（当前）",
        "",
        "- 高位算力科技继续追涨；对照袖用于说明拥挤风险。",
        "- 生猪期货仍弱时重仓养殖龙头赌V反。",
        "",
        "## 7. Deployment Considerations",
        "",
        "- 组合建议: 防御40%（电力/能源）+ 弹性20%（汽车）+ 中期20%（创新药/养殖）+ 现金20%。",
        "- 单票 ≤ 10–12%；主题催化股（江淮/蓝谷）≤ 5%。",
        "- 触发加仓: 沪深300稳定 + 科技净流出收窄 + 目标板块连续3日净流入。",
        "- 触发减仓: 推荐股放量跌破 MA60 或板块单日净流出转至榜首流出。",
        "",
        "## 8. What Could Go Wrong",
        "",
        "- 节前资金行为噪声大，单日流入不可外推。",
        "- 科技超预期催化导致风格迅速切回，防御袖跑输。",
        "- 汽车重组不及预期；养殖周期进一步延长。",
        "- 打分偏量化技术面，未纳入估值分位数与盈利预测修订。",
        "- **失效**: 潜在池 Top5 组合滚动3个月超额 vs 沪深300 < 0 且最大回撤放大 → 暂停个股推荐清单。",
        "",
        "## Final self-check",
        "",
        "- Edge: 来自风格再平衡与周期位置，而非预测单日涨停。",
        "- 过拟合风险: 股票池人工圈定；已用动量回测约束可交易性。",
        "- 缺失: 一致预期盈利、期权/北向细分、涨跌停成交约束。",
        "",
        "## Sector theses（细节）",
        "",
    ]
    for sec, meta in SECTORS.items():
        lines.append(f"### {sec}")
        lines.append(f"- Thesis: {meta['thesis']}")
        lines.append(f"- Horizon: {meta['horizon']}; Risk: {meta['risk']}")
        lines.append("")

    path = OUT / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    # artifacts
    scores.to_csv(OUT / "stock_scores.csv", index=False)
    eq = pd.DataFrame({k: v["equity"] for k, v in results.items() if "equity" in v})
    eq.to_csv(OUT / "equity_curves.csv")
    metrics = {
        k: {"full": v["full"], "oos": v["oos"], "notes": v["notes"]} for k, v in results.items()
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "fund_flow.json").write_text(json.dumps(flow, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    top_global.to_json(OUT / "top_picks.json", orient="records", force_ascii=False, indent=2)
    return path


def main() -> None:
    import pickle

    cache = OUT / "panel.pkl"
    print("Loading fund flow...")
    flow = live_fund_flow()

    if cache.exists():
        print("Loading prices from cache...")
        panel, idx, names = pickle.load(cache.open("rb"))
    else:
        print("Loading prices...")
        panel, idx, names = load_all()
        pickle.dump((panel, idx, names), cache.open("wb"))

    close = pivot(panel, "close")
    vol = pivot(panel, "volume").reindex(close.index).fillna(0.0)
    idx_close = idx.set_index("date")["close"].reindex(close.index).ffill()
    scores = score_stocks(close, vol, idx_close, names)
    scores.attrs["asof"] = str(close.index[-1].date())
    print("\nTop scores:")
    print(scores.head(15)[["code", "name", "sector", "score", "ret_3m", "rs_3m"]].to_string(index=False))
    print("Backtesting...")
    results = backtest_strategies(close, idx)
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["full"].get("sharpe") or -99, reverse=True):
        m = v["full"]
        print(f"{k}: Sharpe={m.get('sharpe', float('nan')):.2f} CAGR={m.get('cagr', float('nan')):.1%} MDD={m.get('max_drawdown', float('nan')):.1%}")
    path = write_report(scores, results, flow, names)
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
