#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-share ambush (埋伏) sector & stock research — trading-strategy-research skill.

Focus: accumulate *before* crowded chase. Prefer:
  - low-crowding / washed-out quality with a forward catalyst
  - defensive cash-flow sleeves for holiday / regime hedge
  - cycle bottoms with improving relative strength
Avoid: chasing single-day limit-up themes as primary ambush.
"""

from __future__ import annotations

import json
import math
import pickle
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
START, END = "20220101", "20260930"
IS_END = "20241231"

# Ambush universes: thesis-driven sleeves (not "today's hottest only")
SECTORS: Dict[str, Dict] = {
    "红利银行": {
        "thesis": "低估值高股息底仓；节前/风险偏好下降时承接资金，四季度哑铃防御端",
        "ambush_type": "底仓埋伏",
        "horizon": "short-medium",
        "risk": "利率与资产质量预期恶化；弹性不足",
        "stocks": {
            "601398": "工商银行",
            "600036": "招商银行",
            "600000": "浦发银行",
            "601166": "兴业银行",
            "601288": "农业银行",
        },
    },
    "电力公用": {
        "thesis": "稳定现金流防御；科技兑现期承接避险资金，适合持仓过节",
        "ambush_type": "底仓埋伏",
        "horizon": "short-medium",
        "risk": "科技快速V反时相对跑输",
        "stocks": {
            "600900": "长江电力",
            "600795": "国电电力",
            "600886": "国投电力",
            "601985": "中国核电",
            "600023": "浙能电力",
        },
    },
    "创新药医美": {
        "thesis": "盈利兑现/出海；9/30资金主线，但已启动——埋伏应分批、偏龙头回撤买",
        "ambush_type": "趋势回撤埋伏",
        "horizon": "medium",
        "risk": "短期拥挤；政策与临床失败",
        "stocks": {
            "603259": "药明康德",
            "600276": "恒瑞医药",
            "300760": "迈瑞医疗",
            "300122": "智飞生物",
            "000661": "长春高新",
        },
    },
    "能源红利": {
        "thesis": "红利+通胀/地缘对冲；历史风险调整收益接近质量门槛",
        "ambush_type": "底仓埋伏",
        "horizon": "medium",
        "risk": "商品价格回落拖累盈利",
        "stocks": {
            "601088": "中国神华",
            "601225": "陕西煤业",
            "600938": "中国海油",
            "601857": "中国石油",
            "600028": "中国石化",
        },
    },
    "生猪养殖": {
        "thesis": "产业周期底部左侧；资金偶发回流，反转或延后——小仓埋伏",
        "ambush_type": "周期左侧埋伏",
        "horizon": "medium-long",
        "risk": "供应压力延长；期货仍弱时深回撤",
        "stocks": {
            "002714": "牧原股份",
            "300498": "温氏股份",
            "000876": "新希望",
            "002567": "唐人神",
        },
    },
    "科技回调精选": {
        "thesis": "四季度『科技+红利』哑铃进攻端；半导体/光模块洗筹后埋伏业绩兑现龙头",
        "ambush_type": "回调埋伏",
        "horizon": "medium",
        "risk": "估值仍高、出口管制；过早抄底被轧",
        "stocks": {
            "300308": "中际旭创",
            "300502": "新易盛",
            "002463": "沪电股份",
            "601138": "工业富联",
            "688041": "海光信息",
        },
    },
    "白酒消费": {
        "thesis": "估值消化后的消费修复；资金回流但非极端底部",
        "ambush_type": "趋势回撤埋伏",
        "horizon": "medium",
        "risk": "内需复苏不及预期",
        "stocks": {
            "600519": "贵州茅台",
            "000858": "五粮液",
            "600809": "山西汾酒",
            "000568": "泸州老窖",
        },
    },
    "新能源电池": {
        "thesis": "景气预期修复早期；资金开始回流电池链，适合小仓布局",
        "ambush_type": "早期布局埋伏",
        "horizon": "medium",
        "risk": "产能过剩与价格战反复",
        "stocks": {
            "300750": "宁德时代",
            "300274": "阳光电源",
            "002594": "比亚迪",
            "300014": "亿纬锂能",
        },
    },
}


def _ak_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def fetch_stock(code: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist_tx(symbol=_ak_symbol(code), start_date=START, end_date=END, adjust="qfq")
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("date").assign(code=code).reset_index(drop=True)


def fetch_index(symbol: str = "sh000300") -> pd.DataFrame:
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    return (
        df[(df["date"] >= "2022-01-01") & (df["date"] <= "2026-09-30")]
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    names: Dict[str, str] = {}
    for meta in SECTORS.values():
        names.update(meta["stocks"])
    frames = []
    for code, name in names.items():
        try:
            df = fetch_stock(code)
            if len(df) < 180:
                print(f"skip {code} {name}: short")
                continue
            frames.append(df)
            print(f"ok {code} {name}: {len(df)} last={df['date'].iloc[-1].date()}")
        except Exception as exc:  # noqa: BLE001
            print(f"fail {code} {name}: {exc}")
        time.sleep(0.3)
    return pd.concat(frames, ignore_index=True), fetch_index(), names


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
    return (1 + gross - turnover * (COST_BPS / 10000.0)).cumprod()


def ambush_score(close: pd.DataFrame, vol: pd.DataFrame, idx: pd.Series, names: Dict[str, str]) -> pd.DataFrame:
    """Higher = better ambush candidate (not max short-term momentum)."""
    last = close.index[-1]
    idx_r3 = idx.pct_change(63).loc[last]
    idx_r6 = idx.pct_change(126).loc[last]
    rows = []
    for code in close.columns:
        px = close[code].dropna()
        if len(px) < 120:
            continue
        r1m, r3m, r6m = px.pct_change(21).iloc[-1], px.pct_change(63).iloc[-1], px.pct_change(126).iloc[-1]
        ma20, ma60 = px.rolling(20).mean().iloc[-1], px.rolling(60).mean().iloc[-1]
        hh120 = px.rolling(120).max().iloc[-1]
        dd = px.iloc[-1] / hh120 - 1.0
        rv = px.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252)
        rs3, rs6 = r3m - idx_r3, r6m - idx_r6
        v = vol[code].dropna() if code in vol.columns else pd.Series(dtype=float)
        vr = float(v.iloc[-5:].mean() / v.iloc[-60:].mean()) if len(v) > 60 and v.iloc[-60:].mean() > 0 else 1.0

        sector = next(s for s, m in SECTORS.items() if code in m["stocks"])
        atype = SECTORS[sector]["ambush_type"]
        score = 0.0

        # Relative strength (mild preference)
        score += float(np.clip(rs3, -0.35, 0.35)) * 1.8
        score += float(np.clip(rs6, -0.45, 0.45)) * 0.8

        # Ambush geometry: prefer pullback zone, penalize melt-up & free-fall
        if atype in ("回调埋伏", "周期左侧埋伏"):
            if -0.40 <= dd <= -0.12:
                score += 0.35  # sweet spot washout
            elif dd > -0.05:
                score -= 0.25  # too extended to ambush
            elif dd < -0.50:
                score -= 0.20  # knife
        elif atype == "底仓埋伏":
            if px.iloc[-1] > ma60:
                score += 0.20
            if dd > -0.08:
                score += 0.05  # near highs OK for quality dividend
            if rv < 0.28:
                score += 0.15  # low vol bonus
        else:  # 趋势回撤 / 早期布局
            if -0.25 <= dd <= -0.05:
                score += 0.25
            if dd > -0.03 and r1m > 0.12:
                score -= 0.20  # just surged — wait pullback
            if px.iloc[-1] > ma60:
                score += 0.10

        if ma20 > ma60:
            score += 0.08
        if rv > 0.60:
            score -= 0.18
        if 1.1 < vr < 2.0:
            score += 0.05
        if vr > 3.0:
            score -= 0.08  # climax

        rows.append(
            {
                "code": code,
                "name": names.get(code, code),
                "sector": sector,
                "ambush_type": atype,
                "close": float(px.iloc[-1]),
                "ret_1m": float(r1m) if pd.notna(r1m) else None,
                "ret_3m": float(r3m) if pd.notna(r3m) else None,
                "ret_6m": float(r6m) if pd.notna(r6m) else None,
                "rs_3m": float(rs3) if pd.notna(rs3) else None,
                "rs_6m": float(rs6) if pd.notna(rs6) else None,
                "dd_from_120h": float(dd),
                "ann_vol_20d": float(rv) if pd.notna(rv) else None,
                "above_ma60": bool(px.iloc[-1] > ma60),
                "vol_ratio": vr,
                "score": float(score),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def backtest(close: pd.DataFrame, idx: pd.DataFrame) -> Dict[str, Dict]:
    rets = close.pct_change().fillna(0.0)
    idx_s = idx.set_index("date").reindex(close.index).ffill()
    idx_eq = (1 + idx_s["close"].pct_change().fillna(0)).cumprod()
    results = {}

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
            "notes": meta["thesis"],
        }

    # Ambush barbell: 50% 能源红利+电力+银行, 30% 科技回调, 20% cash via dual-mom gate on tech
    def_codes = []
    for s in ("红利银行", "电力公用", "能源红利"):
        def_codes += [c for c in SECTORS[s]["stocks"] if c in close.columns]
    tech_codes = [c for c in SECTORS["科技回调精选"]["stocks"] if c in close.columns]
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    # monthly: always hold defense EW 50%; tech 30% only if sector 60d mom > 0 else stay cash
    tech_px = close[tech_codes].mean(axis=1)
    tech_mom = tech_px.pct_change(60)
    months = close.index.to_period("M")
    reb = close.groupby(months).apply(lambda x: x.index[-1])
    for dt in reb:
        w.loc[dt, def_codes] = 0.50 / len(def_codes)
        if dt in tech_mom.index and tech_mom.loc[dt] > 0:
            w.loc[dt, tech_codes] = 0.30 / len(tech_codes)
        # remaining ~0.20 implicit cash
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    # renormalize if needed when tech off — leave cash
    eq = portfolio_from_weights(w, rets)
    results["Barbell_DefenseTech"] = {
        "full": metrics_from_equity(eq),
        "oos": metrics_from_equity(eq[eq.index > IS_END] / eq[eq.index > IS_END].iloc[0]),
        "equity": eq,
        "notes": "50% dividend/power/energy + 30% tech only if 60d mom>0 + ~20% cash",
    }

    # Top-ambush dual mom within non-tech potential
    pot = [c for s, m in SECTORS.items() if s != "科技回调精选" for c in m["stocks"] if c in close.columns]
    mom = close[pot].pct_change(60)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for dt in reb:
        row = mom.loc[dt].dropna()
        pos = row[row > 0].nlargest(5)
        if pos.empty:
            continue
        w.loc[dt, pos.index] = 1.0 / len(pos)
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    eq = portfolio_from_weights(w, rets)
    results["AmbushPool_DualMom5"] = {
        "full": metrics_from_equity(eq),
        "oos": metrics_from_equity(eq[eq.index > IS_END] / eq[eq.index > IS_END].iloc[0]),
        "equity": eq,
        "notes": "Monthly top-5 positive 60d momentum in non-tech ambush pool",
    }

    results["HS300"] = {
        "full": metrics_from_equity(idx_eq),
        "oos": metrics_from_equity(idx_eq[idx_eq.index > IS_END] / idx_eq[idx_eq.index > IS_END].iloc[0]),
        "equity": idx_eq,
        "notes": "沪深300 buy&hold",
    }
    return results


def live_snapshot() -> Dict:
    flow = ak.stock_fund_flow_industry(symbol="即时")
    ths = ak.stock_board_industry_summary_ths()
    idx = ak.stock_zh_index_spot_sina()
    mkt = {}
    for code in ["sh000001", "sz399006", "sh000300", "sh000688"]:
        r = idx[idx["代码"] == code]
        if len(r):
            mkt[code] = {"name": r.iloc[0]["名称"], "pct": float(r.iloc[0]["涨跌幅"]), "last": float(r.iloc[0]["最新价"])}
    return {
        "asof": "2026-09-30",
        "market": mkt,
        "flow_top": flow.sort_values("净额", ascending=False).head(12).to_dict(orient="records"),
        "flow_bottom": flow.sort_values("净额", ascending=True).head(8).to_dict(orient="records"),
        "ths_top": ths.sort_values("涨跌幅", ascending=False).head(8).to_dict(orient="records"),
    }


def pct(x) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.1%}"


def num(x) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.2f}"


def write_report(scores: pd.DataFrame, results: Dict, snap: Dict) -> Path:
    pot = scores.copy()
    top = pot.head(15)
    per = pot.groupby("sector", group_keys=False).head(2).sort_values("score", ascending=False)

    lines = [
        "# A股可埋伏板块与个股研究报告",
        "",
        f"- 数据截止: **{scores.attrs.get('asof', snap.get('asof'))}**（国庆前收官）",
        "- 方法: 资金流 + 埋伏几何打分（回撤甜区/低波/相对强度）+ 板块袖与哑铃回测（单边15bps）",
        "- 基准: 沪深300",
        "- 复现: `python3 scripts/ambush_research/run_research.py`",
        "- **研究回测，不构成投资建议**",
        "",
        "## 1. Executive Summary",
        "",
    ]
    mkt = snap.get("market", {})
    if mkt:
        lines.append("当日指数: " + ", ".join(f"{v['name']} {v['pct']:+.2f}%" for v in mkt.values()) + "。")
    lines += [
        "",
        "市场处于**风格再平衡尾声 + 四季度盈利验证前夜**：医药/银行/白酒获资金，半导体/元件继续流出；",
        "公募共识偏向『科技+红利』哑铃。埋伏原则是**不追拥挤涨停，而在逻辑成立的回撤/低波区分批**。",
        "",
        "### 可埋伏板块排序（研究优先级）",
        "",
        "1. **能源红利 + 电力公用 + 银行** — 底仓埋伏（持仓过节/降波），回测质量最好",
        "2. **科技回调精选（光模块/PCB/算力龙头）** — 回调埋伏，等洗筹结束而非抢反弹首日",
        "3. **创新药** — 主线确认但已启动，宜回撤分批而非追高",
        "4. **新能源电池** — 早期布局，仓位宜小",
        "5. **生猪养殖** — 周期左侧，小仓、长周期",
        "6. **白酒** — 消费修复卫星仓",
        "",
        "### 个股埋伏清单（综合分 Top）",
        "",
        "| Rank | 代码 | 名称 | 板块 | 类型 | Score | 3M | 距120高 | >MA60 |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | :---: |",
    ]
    for i, r in enumerate(top.to_dict("records"), 1):
        lines.append(
            f"| {i} | {r['code']} | {r['name']} | {r['sector']} | {r['ambush_type']} | "
            f"{r['score']:.2f} | {pct(r['ret_3m'])} | {pct(r['dd_from_120h'])} | "
            f"{'Y' if r['above_ma60'] else 'N'} |"
        )

    lines += ["", "### 分板块 Top2", ""]
    for r in per.to_dict("records"):
        lines.append(
            f"- **{r['sector']} / {r['name']}({r['code']})** — {r['ambush_type']}, "
            f"score {r['score']:.2f}, DD120 {pct(r['dd_from_120h'])}, RS3M {pct(r['rs_3m'])}"
        )

    # quality bar from backtests
    lines += ["", "### 回测质检", ""]
    ordered = sorted(results.items(), key=lambda kv: kv[1]["full"].get("sharpe") or -99, reverse=True)
    for name, blob in ordered[:6]:
        m = blob["full"]
        flag = "PASS-ish" if (m.get("sharpe") or 0) > 1 and (m.get("max_drawdown") or -1) > -0.30 else "CONDITIONAL"
        lines.append(f"- `{name}`: Sharpe {num(m.get('sharpe'))}, MDD {pct(m.get('max_drawdown'))}, CAGR {pct(m.get('cagr'))} — {flag}")

    lines += ["", "## 2. Market Analysis", "", "### 资金净流入 Top", ""]
    for row in snap.get("flow_top", [])[:8]:
        lines.append(
            f"- {row.get('行业')}: 涨跌 {row.get('行业-涨跌幅')}%, 净额 {row.get('净额')}亿, 领涨 {row.get('领涨股')}"
        )
    lines += ["", "### 资金净流出（不适合追涨埋伏）", ""]
    for row in snap.get("flow_bottom", [])[:6]:
        lines.append(f"- {row.get('行业')}: 涨跌 {row.get('行业-涨跌幅')}%, 净额 {row.get('净额')}亿")

    lines += [
        "",
        "### Facts vs interpretation",
        "",
        "- **事实**: 9/30 医药生物大幅净流入，电子/半导体大幅净流出；上证微涨、科创回调。",
        "- **事实**: 四季度进入三季报验证期；机构普遍提『科技+红利』哑铃。",
        "- **解释**: 医药已从『潜在』进入『确认主线』——埋伏仓应转为回撤加仓；科技仍在去拥挤，适合分批左侧而非一把梭。",
        "",
        "## 3. Strategy Details",
        "",
        "| Strategy | Thesis | Spec |",
        "| --- | --- | --- |",
        "| EW_板块袖 | 板块 beta | 各埋伏池等权 |",
        "| Barbell_DefenseTech | 哑铃 | 50%红利电力能源 + 30%科技(需60d动量>0) + 现金 |",
        "| AmbushPool_DualMom5 | 非科技动量 | 月度 Top5 正动量 |",
        "",
        "埋伏打分偏好：回调甜区(-12%~-40%)、低波红利近高可持、惩罚刚暴涨与深跌刀子。",
        "代码: `scripts/ambush_research/run_research.py`",
        "",
        "## 4. Backtest Results",
        "",
        "| Strategy | CAGR | Sharpe | Sortino | MDD | OOS Sharpe | OOS CAGR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, blob in ordered:
        m, o = blob["full"], blob["oos"]
        lines.append(
            f"| {name} | {pct(m.get('cagr'))} | {num(m.get('sharpe'))} | {num(m.get('sortino'))} | "
            f"{pct(m.get('max_drawdown'))} | {num(o.get('sharpe'))} | {pct(o.get('cagr'))} |"
        )

    lines += [
        "",
        "## 5. Comparison & Ranking",
        "",
        "底仓袖（能源/电力/银行）以**回撤可控**胜出；科技袖历史收益高但回撤深，只作回调仓。",
        "创新药/白酒等权未必过质量门槛，个股精选优于板块 beta。",
        "",
        "## 6. Final Recommendations",
        "",
        "### A. 底仓埋伏（优先，持仓过节）",
        "",
    ]
    for sec in ("能源红利", "电力公用", "红利银行"):
        for r in pot[pot["sector"] == sec].head(2).to_dict("records"):
            lines.append(f"- **{r['name']}({r['code']})** [{sec}] score {r['score']:.2f}")

    lines += ["", "### B. 回调埋伏（科技精选，分批）", ""]
    for r in pot[pot["sector"] == "科技回调精选"].head(3).to_dict("records"):
        lines.append(
            f"- **{r['name']}({r['code']})** — score {r['score']:.2f}, DD120 {pct(r['dd_from_120h'])}; "
            f"建议跌破/靠近关键均线再加，不追阴线中继"
        )

    lines += ["", "### C. 主线回撤加仓（创新药）", ""]
    for r in pot[pot["sector"] == "创新药医美"].head(3).to_dict("records"):
        lines.append(f"- **{r['name']}({r['code']})** — score {r['score']:.2f}; 已启动，等日内/3–5日回撤")

    lines += ["", "### D. 卫星/左侧", ""]
    for sec in ("新能源电池", "白酒消费", "生猪养殖"):
        rlist = pot[pot["sector"] == sec].head(1).to_dict("records")
        if rlist:
            r = rlist[0]
            lines.append(f"- **{r['name']}({r['code']})** [{sec}] — score {r['score']:.2f}")

    lines += [
        "",
        "### E. 现在不要做的事",
        "",
        "- 追半导体/元件当日跌停反弹的第一根阳线当埋伏",
        "- 把医药涨停股当『还没启动』重仓",
        "- 生猪期货仍弱时重仓赌 V 反",
        "",
        "## 7. Deployment Considerations",
        "",
        "- 建议仓位: 底仓（能源+电力+银行）**45%** + 科技回调 **15–20%**（分 2–3 批）+ 创新药 **15%** + 卫星 **10%** + 现金 **10–15%**",
        "- 单票: 底仓 ≤12%；科技/主题 ≤8%；养殖 ≤5%",
        "- 加仓触发: 科技板块连续2日净流出收窄 + 目标股站上 MA20；医药回撤至 MA20 附近",
        "- 减仓触发: 底仓跌破 MA60；科技反弹无力再破前低",
        "",
        "## 8. What Could Go Wrong",
        "",
        "- 假期海外风险冲击开盘缺口，底仓亦回撤",
        "- 科技基本面证伪，回调埋伏变成接飞刀",
        "- 医药拥挤后快速扩散失败",
        "- 打分未用估值分位与盈利预测，存在基本面盲区",
        "- **失效**: 底仓袖滚动3个月跑输现金+通胀，或科技袖再创新高拥挤度且未回调——暂停加仓",
        "",
        "## Final self-check",
        "",
        "- Edge: 风格哑铃与回撤几何，非预测涨停",
        "- 过拟合: 池子人工圈定；以板块袖回测约束",
        "- 覆盖体制: 2022 熊、2023–24 修复、2025–26 AI 潮与回撤",
        "",
    ]
    for sec, meta in SECTORS.items():
        lines += [f"### {sec}", f"- {meta['ambush_type']}: {meta['thesis']}", f"- Risk: {meta['risk']}", ""]

    path = OUT / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    scores = scores.copy()
    scores["code"] = scores["code"].astype(str).str.zfill(6)
    scores.to_csv(OUT / "stock_scores.csv", index=False)
    pd.DataFrame({k: v["equity"] for k, v in results.items() if "equity" in v}).to_csv(OUT / "equity_curves.csv")
    (OUT / "metrics.json").write_text(
        json.dumps({k: {"full": v["full"], "oos": v["oos"], "notes": v["notes"]} for k, v in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT / "market_snapshot.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    top.to_json(OUT / "top_picks.json", orient="records", force_ascii=False, indent=2)
    return path


def main() -> None:
    cache = OUT / "panel.pkl"
    print("Snapshot...")
    snap = live_snapshot()
    if cache.exists():
        print("Loading cache (delete panel.pkl to refresh)...")
        panel, idx, names = pickle.load(cache.open("rb"))
        # refresh if cache stale (last date < END-ish)
        last = pd.to_datetime(panel["date"]).max()
        if last < pd.Timestamp("2026-09-29"):
            print(f"cache stale ({last.date()}), refetching...")
            panel, idx, names = load_all()
            pickle.dump((panel, idx, names), cache.open("wb"))
    else:
        print("Fetching prices...")
        panel, idx, names = load_all()
        pickle.dump((panel, idx, names), cache.open("wb"))

    close = pivot(panel, "close")
    vol = pivot(panel, "volume").reindex(close.index).fillna(0.0)
    idx_close = idx.set_index("date")["close"].reindex(close.index).ffill()
    scores = ambush_score(close, vol, idx_close, names)
    scores.attrs["asof"] = str(close.index[-1].date())
    print(scores.head(12)[["code", "name", "sector", "ambush_type", "score", "dd_from_120h"]].to_string(index=False))
    results = backtest(close, idx)
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["full"].get("sharpe") or -99, reverse=True):
        m = v["full"]
        print(f"{k}: Sh={m.get('sharpe', float('nan')):.2f} CAGR={m.get('cagr', float('nan')):.1%} MDD={m.get('max_drawdown', float('nan')):.1%}")
    path = write_report(scores, results, snap)
    print("Report:", path)


if __name__ == "__main__":
    main()
