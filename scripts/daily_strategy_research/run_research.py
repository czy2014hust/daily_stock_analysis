#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily A-share strategy research.

Usage: RESEARCH_DATE=2026-10-21 LAST_BAR=2026-10-20 python3 run_research.py

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
- Sector momentum / low-vol / regime: month-end, top 3, 120-day lookback where applicable.
- Risk parity: month-end inverse 60-day vol across CSI300 / 10Y treasury / gold ETFs.
- Stock momentum + low vol: month-end, average rank of 120d momentum and 60d vol, top 5 stocks.
- Dividend yield (carry): month-end, the 5 stocks with the highest trailing 365-day cash
  dividend per share (Sina, counted from the ex-date) / Eastmoney raw close; a payment
  330+ days older than the newest one in the window is last year's slot and is dropped. Dividends paid
  before a later bonus/transfer issue inside the window are scaled to post-split shares.
- Profitability (quality): month-end, the 5 stocks with the highest PB / PE(TTM) (= TTM
  earnings / book, i.e. ROE) from Eastmoney daily valuation; loss makers (PE <= 0) excluded.
- Residual momentum: month-end, regress each stock's daily return on the CSI300 ETF over
  sessions t-240..t-21, rank by sum(residual) / std(residual); top 5.
- ERP timing: month-end, CSI300 earnings yield (1 / cap-weighted TTM PE, Legulegu) minus the
  CN 10Y yield. Equity weight = percentile rank of the current ERP within the trailing 60
  month-ends; the remainder sits in the 10Y treasury ETF. Benchmark: 50/50 stock/bond.
- No parameter is selected by maximizing in-sample Sharpe. Sensitivity is diagnostic only.

This is research code, not an order-routing system and not investment advice.
"""

from __future__ import annotations

import json
import math
import os
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

ASOF_REQUESTED = os.environ.get("RESEARCH_DATE", "2026-10-21")
LAST_BAR = pd.Timestamp(os.environ.get("LAST_BAR", "2026-10-20"))
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

# Interpretation written after reading this run's tables. It explains results; it never
# feeds back into rules, parameters or the pass/fail checks.
INTERP_NOTES: Dict[str, str] = {
    "Stock_DivYield": (
        "解读：这是风格押注，不是选股 alpha。与 Stock_LowVol 超额相关 0.78、持仓重合 48%；逐年看，2020 年落后股票池等权 70.6 个百分点"
        "（那一年等权靠宁德时代、隆基绿能、比亚迪等成长股大涨，高股息组合持有中国神华、工商银行、长江电力、中国石油基本持平），"
        "2022–2024 年连续三年分别领先 17.3、35.9、27.8 个百分点，2025 年又落后 18.4。最大回撤 -18.7% 是所有个股规则里最浅的，"
        "但 24 次逐只剔除没有一次跑赢等权。当前持仓（中国平安、五粮液、美的、格力、伊利）的高股息率有一部分是股价下跌推高的，"
        "带有消费股「价值陷阱」风险，不同于 2022–2024 年的银行/能源红利。结论：保留观察，定位为熊市/防御期的低回撤风格配置，不记为 edge。"
    ),
    "Stock_Quality": (
        "解读：在这个 24 只的大盘股池里，高 ROE 就是白酒、家电、乳业、药明康德、宁德时代这类 2021–2024 年估值大幅下修的名字，ROE 高没有阻止杀估值；"
        "2021–2025 年连续 5 年跑输等权（-16.7%、-15.3%、-18.9%、-22.9%、-7.7%）。ROE 用到的净资产还有 1–4 个月前视（见数据节），"
        "前视通常会美化结果，真实表现只会更差。淘汰。以后若重测质量因子，需要更宽的宇宙和时点正确的财报数据，而不是在大盘池里做 ROE 单因子。"
    ),
    "Stock_ResMom": (
        "解读：10-16 期引入逐只剔除检验以来第一个稳定跑赢股票池等权的个股规则（23/24 次为正，最差 -0.8%；此前最好 10/24），与低波动（-0.10）和 120 日动量（0.08）的超额都基本不相关，"
        "是真正不同的收益来源。但它在时间上不稳：逐年超额 8.4%、25.6%、-3.1%、-7.5%、10.6%、3.6%、-14.5%，全样本超额主要来自 2021 年；"
        "样本内年化 19.7% 对等权 15.0%，样本外 16.9% 对 19.2%，样本外已落后；2026 年至今 -15.6%。年换手 19.4 倍，成本翻倍后 CAGR 15.1% 低于等权 16.7%。"
        "原始 12-1 动量（不剔除 beta）全样本 Sharpe 0.83、样本外 0.67，剔除 beta 只带来很小的改进；回归窗口改为 120 日时 Sharpe 降到 0.61，持 3 只时样本外降到 0.42。"
        "结论：未过门槛（Sharpe-rf 0.75、回撤 -32.8%、样本外跑输等权），进入观察名单，用前瞻数据检验「跨名字稳健」能否变成「跨时间稳健」。"
    ),
    "ERP_Timing": (
        "解读：相对股债 50/50 的 1.0%/年超额，大部分来自平均多拿 13 个百分点股票；对照同样 63% 股票、不择时的静态组合，择时本身约贡献 1.1 个百分点 CAGR 和 0.07 Sharpe。"
        "这一点几乎全部来自一次信号：2023 年中到 2024 年 8 月 ERP 处于高位，股票权重 82%–98%，2023 年因此落后 EW_SB 3.9 个百分点，2024 年 9 月行情前仍是 95% 股票，当年领先 9.5 个百分点；"
        "其余 6 年合计约 -2.3 个百分点，样本内 Sharpe 0.02 低于 EW_SB 的 0.13。加 10% 波动目标后 Sharpe 从 0.41 降到 0.15。"
        "7 年样本里只有一次有效的大信号，统计上无法与运气区分。淘汰。ERP 可以继续作为市场状态描述（最近月末分位约 52%，股票相对国债估值中性），不作为交易规则。"
    ),
}
EXEC_INTERP: List[str] = [
    "本期第一次引入基本面/估值数据，4 个新候选给出 4 种不同结论：残差动量 Stock_ResMom 是 10-16 期引入逐只剔除检验以来第一个 23/24 次跑赢股票池等权的个股规则（此前最好 10/24），"
    "但超额集中在 2021 年、样本外已落后等权、2026 年至今 -15.6%，且年换手 19 倍；股息率 Stock_DivYield 回撤最浅（-18.7%）但本质是低波动/价值风格（与低波动超额相关 0.78）；"
    "ROE 质量 Stock_Quality 连续 5 年跑输，淘汰；ERP 股债择时的超额来自多拿股票和 2024 年一次信号，淘汰。",
    "相对 10-20 期：新增 2026-10-20 一根日线（沪深300 -0.5%、中证1000 +1.0%），行业 ETF 站上 MA60 的比例从 20% 降到 13%；"
    "新浪前复权的回溯修订让 Stock_MomLowVol 样本外 Sharpe 从 1.44 降到 1.37（约 0.05 来自修订），观察名单排序不变。",
]
RISK_INTERP: List[str] = [
    "- 风格集中：Stock_DivYield、Stock_LowVol、Stock_MomLowVol、Sector_LowVol、Sector_DualMom 当前持仓都偏向银行、红利、家电、电力等防御/价值名字，个股规则之间超额相关 0.66–0.78。"
    "在当前 bear 状态下它们一起占优（2026 年至今 Stock_LowVol +10.0%、Stock_MomLowVol +9.4%，等权 -1.1%），一旦出现 2020 年或 2024-09 那种成长/高 beta 急涨，会一起大幅跑输"
    "（Stock_DivYield 2020 年落后等权 70 个百分点）。",
    "- 残差动量：2026 年至今 -15.6%，当前持仓五粮液、格力电器、工业富联、立讯精密、中航沈飞横跨消费与科技硬件；在指数下跌、行业快速轮动的阶段，过去一年的特质赢家容易被一起抛售。"
    "失效条件：滚动 12 个月相对 EW_Stocks 超额持续为负，或实盘成本高于约 23bps。",
    "- 估值数据时点：东方财富净资产按季末而非披露日生效（ROE 前视），乐咕乐股指数 PE 的利润生效时点未知。基于这些序列的结论（尤其是 ERP_Timing）应视为乐观上限。",
    "- 股息率「价值陷阱」：当前高股息名单里的五粮液、格力、伊利有一部分是股价下跌推高的股息率；若下一次分红被削减，按历史分红算的股息率要到除息后才会反映。",
]
SELF_CHECK_INTERP: List[str] = [
    "- Edge 还是拟合：4 个新候选的规则和参数在跑数前固定，全部未过门槛。首跑后的改动有三处，都只让检验更严或修正实现错误："
    "（1）股息率窗口重复计数的修正（Sharpe 0.84 → 0.85，修正前版本保留在敏感度表）；（2）新增「同平均股票仓位的静态股债组合」作为 ERP_Timing 的 beta 匹配对照；"
    "（3）新增逐年超额表。没有因为结果改任何规则参数。",
    "- 非预注册的显眼结果：DivY_n8 样本外 Sharpe 1.42、ERP_rank120 全样本回撤 -18.6%，都是看完历史后才知道的档位，不升级为候选。",
    "- 本期最值得前瞻跟踪的不是任何一条规则的 Sharpe，而是 Stock_ResMom「跨名字稳健、跨时间不稳健」的矛盾：只有未来 12 个月它相对 EW_Stocks 的超额转正、且逐只剔除仍大多为正，"
    "才能说明 2021 年不是唯一来源。",
    "- 会改变结论的证据：RiskParity_SBG 在国债收益率反转时回撤超过 5%；Stock_MomLowVol 在新增样本外一年里持续跑赢 EW_Stocks；拿到时点正确的财报数据后 ROE 规则结论改变（目前预期只会更差）。",
]
OVERLAP_NOTE = (
    "解读：Stock_DivYield 与 Stock_LowVol 超额相关 0.78、与 Stock_MomLowVol 0.66，持仓与 Stock_LowVol 重合 48%，股息率在这个池子里基本是低波动换了个名字。"
    "Stock_ResMom 与其他所有规则的超额相关都在 -0.15 ~ 0.08 之间（包括 120 日动量 0.08），是本期唯一独立且有正超额的新信号；Stock_Quality 也独立，但独立在亏钱的方向。"
)
FUNDAMENTAL_NOTES: List[str] = [
    "- 估值数据时点抽查（东方财富 `stock_value_em`）：用「原始收盘价 / PE(TTM)」反推 TTM 利润，招商银行 2024 年以来的跳变日为 2024-01-22、2024-04-30、2025-01-15、2025-04-30、"
    "2025-09-01、2026-01-26、2026-04-29、2026-08-31，贵州茅台、中国神华类似，都落在业绩快报/定期报告披露日或之后，利润端基本时点正确。"
    "但「原始收盘价 / PB」反推的净资产在 03-31、06-30、09-30、12-31 季末当天就跳变，即财报披露前就用了季末净资产，有 1–4 个月前视。"
    "ROE = PB / PE(TTM) 因此带前视；这类偏差通常美化 ROE 规则，本期 Stock_Quality 仍大幅跑输，结论只会更差。股息率只用原始收盘价和已实施分红的除息日，没有这个问题。",
    "- 沪深300 TTM PE（乐咕乐股 `stock_index_pe_lg`）只有月末打印，供应商未说明成分股利润按报告期还是披露日生效；若按报告期，ERP 在 4 月、8 月、10 月底前后有最多约 1 个月前视。没有找到第二来源交叉验证，记为缺口。",
    "- 股息率实现修正：首跑用简单 365 天窗口，抽查发现工商银行、中国神华 2020-06-30 的股息率约为正常值两倍（2019-07-03 与 2020-06-30 两次年度分红落在同一窗口）。"
    "修正为「窗口内比最新一笔早 330 天以上的分红剔除」后，评估期 1893 个股票-月中 17 个取值改变，Stock_DivYield 全样本 Sharpe 0.84 → 0.85，结论不变；修正前版本列在敏感度表（DivY_365d_doublecount）。"
    "中间试过的 345 天窗口会在除息日推迟时把 54 个股票-月的股息率错误置零，已放弃。",
    "- 当前股息率（2026-09-30 月末，原始收盘价）：格力电器 7.8%、五粮液 7.4%、中国平安 6.9%、美的集团 5.4%、伊利股份 5.1%、中国移动 4.9%、招商银行 4.9%。"
    "工商银行 2026-10-21 除息（每股 0.151 元）、中国神华 10-26 除息（0.98 元），在 10 月末才进入信号。",
]
REPRO_NOTE = (
    "- 先用上一版（10-20）原始脚本、LAST_BAR=2026-10-19 重跑：16 个序列中 12 个（全部 ETF 序列、EW_Stocks、Stock_LowVol、Stock_AbnTurnover）的 Sharpe、CAGR、最大回撤、样本外 Sharpe 与上一版报告逐一相同；"
    "4 个个股规则小幅变化：Stock_MomLowVol CAGR 16.7% → 16.4%、样本外 Sharpe 1.44 → 1.39，Stock_CS_Mom CAGR 16.0% → 15.9%，Stock_OvernightRev 回撤 -37.5% → -37.8%，Stock_LowMAX CAGR 6.4% → 6.3%。"
    "原因是新浪前复权历史被回溯修订：工商银行 2026-10-21 除息，今天下载的前复权价格按新因子整体缩放、再四舍五入到分，日收益出现微小差异，足以改变个别月末近似平局的排名。"
    "上一版的原始下载没有留存，无法逐行证实是哪只股票，这是推断。本期新增的只是 2026-10-20 一根日线、这次修订和候选集合变化。"
)
REPRO_SELF_CHECK = (
    "- 复现性：上一版（10-20）脚本在 LAST_BAR=2026-10-19 上重跑，16 个序列中 12 个与上一版逐一相同，4 个个股规则有小幅差异（Sharpe 变化 ≤ 0.05），"
    "归因于新浪前复权回溯修订（工商银行 10-21 除息，推断）；观察名单排序不变。"
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
    """Eastmoney daily valuation: raw close, PE(TTM), PB. Missing or non-positive values stay NaN."""
    key = f"{code}_value_em"
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

        raw = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
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


def fetch_index_pe() -> pd.Series:
    """CSI300 cap-weighted TTM PE (Legulegu, month-end prints plus the latest day)."""
    path = _cache_path("csi300_pe_lg")
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        import akshare as ak

        raw = ak.stock_index_pe_lg(symbol="沪深300")
        df = pd.DataFrame({"date": pd.to_datetime(raw["日期"]), "pe_ttm": pd.to_numeric(raw["滚动市盈率"], errors="coerce")})
        df.to_csv(path, index=False)
    df = df.dropna().sort_values("date").drop_duplicates("date")
    return df.set_index("date")["pe_ttm"].loc[:LAST_BAR]


def fetch_cn10y() -> pd.Series:
    """CN 10Y government bond yield (% p.a.), daily."""
    path = _cache_path("cn10y")
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        import akshare as ak

        raw = ak.bond_zh_us_rate(start_date="20100101")
        df = pd.DataFrame({"date": pd.to_datetime(raw["日期"]), "y10": pd.to_numeric(raw["中国国债收益率10年"], errors="coerce")})
        df.to_csv(path, index=False)
    df = df.dropna().sort_values("date").drop_duplicates("date")
    return df.set_index("date")["y10"].loc[:LAST_BAR]


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


def strategy_erp_timing(
    index: pd.DatetimeIndex,
    pe_ttm: pd.Series,
    y10: pd.Series,
    lookback_months: int = 60,
    min_months: int = 36,
    mode: str = "rank",
    equity: str = "510300",
    bond: str = "511260",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Month-end stock/bond split from the equity risk premium (1/PE_TTM - CN10Y).

    The ERP history is built on the Legulegu month-end PE prints (back to 2005) with the 10Y
    yield as of the same date, so the 60-month percentile is available from the first eval month.
    mode="rank": equity weight = share of the trailing window at or below the current ERP.
    mode="median": all equity if the current ERP is above the trailing median, else all bond.
    Returns (weights, monthly ERP series).
    """
    y = y10.sort_index()
    erp = pd.Series(
        {d: 1.0 / pe - float(y.loc[:d].iloc[-1]) / 100.0 for d, pe in pe_ttm.items() if pe > 0 and len(y.loc[:d])},
        dtype=float,
    ).sort_index()
    raw = _empty_weights(index, [equity, bond])
    for dt in month_ends(index):
        hist = erp.loc[:dt].tail(lookback_months)
        if len(hist) < min_months:
            continue
        cur = float(hist.iloc[-1])
        if mode == "rank":
            w = float((hist <= cur).mean())
        else:
            w = 1.0 if cur > float(hist.median()) else 0.0
        raw.loc[dt] = pd.Series({equity: w, bond: 1.0 - w})
    return _materialize(raw), erp


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

    # ERP timing: weight is the percentile of the latest month-end ERP inside the trailing window.
    pe = pd.Series([10.0, 20.0, 12.0, 5.0], index=pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]))
    yy = pd.Series([2.0], index=pd.to_datetime(["2019-12-31"]))
    ew, _ = strategy_erp_timing(pd.bdate_range("2020-01-01", "2020-05-15"), pe, yy, 60, 2)
    assert ew.loc["2020-02-28", "510300"] == 0.5 and ew.loc["2020-03-31", "510300"] == 2.0 / 3.0 and ew.loc["2020-04-30", "510300"] == 1.0
    assert ew.loc["2020-01-31", "510300"] == 0.0 and ew.loc["2020-01-31", "511260"] == 0.0
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
    a("相对 2026-10-20 版本的迭代（依据上一版证据，在跑本次数据之前确定）：")
    a("")
    a(ctx["repro_note"])
    a("- 丢弃 10-20 期 4 个新增且全部淘汰的候选：Stock_AbnTurnover（独立但 24 次逐只剔除 0 次跑赢等权）、Stock_OvernightRev（2024 年起方向反转）、"
      "Stock_LowMAX（与低波动超额相关 0.74，跑输等权 10%/年）、Gold_TSMOM（风险调整后输给金债 50/50）。同时去掉只服务于它们的 BH_Gold、EW_GB 基准。")
    a("- 保留 6 个观察名单候选（规则与参数不变）：RiskParity_SBG、Stock_MomLowVol、Stock_LowVol、Stock_CS_Mom、Sector_LowVol、Sector_DualMom。")
    a("- 新增 4 个收益来源不同、此前各期都没测过的候选。前面 13 期只用价格与成交量；本期第一次引入基本面/估值数据："
      "股息率（现金流 carry）、盈利能力 ROE（质量）、残差动量（剥离市场 beta 后的个股特质动量）、股权风险溢价 ERP 股债择时（估值 vs 利率）。"
      "股息率与低波动在 A 股可能高度重合（银行、电力、能源），继续用「超额相关 + 持仓重合」诊断；残差动量另设原始 12-1 动量作对照，检验剥离 beta 是否带来新信息。"
      "新增股债 50/50 的 EW_SB 作为 ERP_Timing 的宇宙基准。")
    a("")
    a("| ID | 经济逻辑 | 规格 | 成本 |")
    a("| --- | --- | --- | --- |")
    a("| BH_CSI300 | 宽基 beta，作为基准而不是 alpha | 全程持有 510300 沪深300ETF（含分红） | 单边 8bps，仅建仓 |")
    a("| EW_Stocks | 股票池本身的 beta，用来区分「池子选得偏」和「规则有超额」 | 24 只股票里已有足够历史的名字月末等权 | 15bps |")
    a("| EW_Sectors | 行业 ETF 池本身的 beta，行业策略的宇宙基准 | 15 只行业/风格 ETF 月末等权 | 8bps |")
    a("| Sector_DualMom | 部门资金与景气的相对强弱会延续数月；绝对动量过滤用来避免熊市硬做多 | 国内行业/风格 ETF，月末选 120 日收益最高且为正的 3 只，等权 | 8bps |")
    a("| Sector_LowVol | 低波动/红利类资产在 A 股有持有期溢价，且回撤通常浅于高 beta | 月末选 60 日波动最低的 3 只行业 ETF，等权 | 8bps |")
    a("| EW_SBG | 股票/国债/黄金三等分，检验「按波动倒数配权」是否比简单分散更好 | 510300 / 511260 / 518880 月末各 1/3 | 8bps |")
    a("| BH_Bond10Y | 久期 beta，检验股债配置的收益是否只是国债 | 全程持有 511260 十年国债ETF（含分红） | 8bps |")
    a("| RiskParity_SBG | 股票、久期、黄金三类风险溢价长期低相关；按风险而非按金额分配可压低回撤 | 510300 / 511260 / 518880 月末按 60 日波动倒数配权 | 8bps |")
    a("| EW_SB | 股债 50/50，ERP_Timing 的宇宙基准：择时必须好过「两样都拿」 | 510300 / 511260 月末各 50% | 8bps |")
    a("| ERP_Timing | 股权风险溢价（盈利收益率 − 10Y 国债收益率）高时，股票相对债券便宜，后续 1–3 年超额收益的期望更高（Campbell-Shiller 估值回归、Fed model 的相对估值版本）；按分位而不是阈值配权，避免对单一切换点过拟合 | 月末 ERP = 1 / 沪深300 TTM PE（乐咕乐股市值加权）− 中国 10Y 国债收益率；股票权重 = 当前 ERP 在过去 60 个月末中的分位，其余放 511260 | 8bps |")
    a("| Stock_CS_Mom | 个股截面动量来自基本面与资金抱团的缓慢扩散，绝对动量避免逆势抄底 | 24 只跨行业大盘股，月末选 120 日收益为正的前 5，等权 | 15bps |")
    a("| Stock_LowVol | 低波动异象：机构有杠杆约束与彩票偏好，高 beta 被高估、低 beta 被低估 | 同一股票池，月末选 60 日波动最低的 5 只，等权 | 15bps |")
    a("| Stock_MomLowVol | 动量与低波动在 A 股相关性低、回撤期不同；复合排序可保留动量收益并压低动量崩盘 | 同一股票池，月末按「120 日收益排名 + 60 日波动排名（低者优）」之和选前 5，等权 | 15bps |")
    a("| Stock_DivYield | 股息 carry：现金分红是可兑现的回报；高股息大盘股受险资/红利指数配置需求支撑，且分红约束管理层过度投资 | 同一股票池，月末按「过去 365 天每股现金分红（按除息日计入，送转后折算到当前股本；窗口内若某笔分红比最新一笔早 330 天以上，视为上一年度同一档分红而剔除，避免除息日逐年提前时重复计数）/ 原始收盘价」选最高的 5 只，等权 | 15bps |")
    a("| Stock_Quality | 盈利能力溢价（Novy-Marx、Fama-French RMW）：高 ROE 公司现金流更持久，市场对其持续性定价不足 | 同一股票池，月末按 PB / PE(TTM)（= TTM 净利润 / 净资产，即 ROE）选最高的 5 只，亏损股（PE≤0）剔除，等权 | 15bps |")
    a("| Stock_ResMom | 残差动量（Blitz-Huij-Martens）：原始动量混入了市场 beta，在反转时崩盘；剥离 beta 后的特质收益延续来自公司层面信息的缓慢扩散 | 同一股票池，月末对过去 240 日中的前 220 日（跳过最近 20 日）做个股对沪深300ETF 的日收益回归，按「残差之和 / 残差标准差」选最高的 5 只，等权 | 15bps |")
    a("")
    a("成交与流动性约束：")
    a("")
    a("- 信号用 t 日收盘，收益从 t+1 日收盘到再下一根收盘才计入，避免用到信号当日涨跌。")
    a("- ETF 使用新浪未复权成交价，并在除息日加回新浪披露的现金分红（分红 / 前一日原始收盘价）。没有分红记录的 ETF 不估计分红。")
    a("- 股票使用新浪前复权。腾讯前复权在中国神华等标的上出现过不可能的单日涨跌，已整段弃用。")
    a("- ETF 过去 20 日成交额中位数低于 2000 万人民币则当天不可入选；股票阈值为 5000 万。")
    a("- 停牌或单日缺口最多向前填充 5 个交易日；更长缺口不编造价格。")
    a("- 现金收益按 0 计（只影响 Sector_DualMom 无正动量时的空仓）。Sharpe-rf 扣除 3M Shibor，因此对空仓策略更严。")
    a("- 基本面：个股 PE(TTM)、PB 与原始收盘价来自东方财富 `stock_value_em` 日频估值；现金分红与送转来自新浪 `stock_history_dividend_detail`，只用「实施」记录，从除权除息日起计入（公告日更早，因此偏保守）。"
      "沪深300 TTM PE 来自乐咕乐股 `stock_index_pe_lg`（月末打印），10Y 国债收益率来自 `bond_zh_us_rate`。这些估值序列的 TTM 利润是否严格按财报披露日生效未经供应商确认，见数据节抽查。")
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
    a("### 个股规则的因子重叠（新增诊断）")
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
    a("每次从 24 只股票池里去掉 1 只，规则、参数和 EW_Stocks 基准都在同一个缩小后的池子上重跑。")
    a("")
    a(ctx["loo_table"])
    a("")
    a(ctx["loo_note"])
    a("")
    a("### 逐年超额（新增诊断）")
    a("")
    a("每个日历年的策略收益减去其宇宙基准收益（扣费后）。用来区分「每年都有一点」和「只在一两个年份有一大段」。")
    a("")
    a(ctx["yearly_table"])
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
    a("5. 跑赢自身宇宙等权（行业 ETF 策略对 EW_Sectors，个股策略对 EW_Stocks，风险平价对 EW_SBG，ERP_Timing 对股债 50/50 的 EW_SB）。")
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

    weights: Dict[str, pd.DataFrame] = {}
    weights["BH_CSI300"] = pd.DataFrame(1.0, index=etf_close.index, columns=["510300"])
    weights["EW_Stocks"] = strategy_equal_weight(stock_close, stock_amount, STOCK_MIN_AMOUNT)
    weights["EW_Sectors"] = strategy_equal_weight(sector_close, sector_amount, ETF_MIN_AMOUNT)
    weights["Sector_DualMom"] = strategy_sector_dual_mom(sector_close, sector_amount, 120, 3)
    weights["Sector_LowVol"] = strategy_low_vol(sector_close, sector_amount, 60, 3)
    rp_close = etf_close[[c for c in RISK_PARITY_ETFS if c in etf_close.columns]]
    weights["RiskParity_SBG"] = strategy_risk_parity(rp_close, 60)
    # Added after the first run exposed RiskParity_SBG's bond weight: stricter tests, not new alpha.
    weights["EW_SBG"] = _materialize(
        pd.DataFrame({c: np.where(etf_close.index.isin(month_ends(etf_close.index)), 1.0 / len(rp_close.columns), np.nan) for c in rp_close.columns}, index=etf_close.index)
    )
    weights["BH_Bond10Y"] = pd.DataFrame(1.0, index=etf_close.index, columns=["511260"])
    weights["EW_SB"] = static_mix(etf_close.index, {"510300": 0.5, "511260": 0.5})

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
    pe_panel = val_panel("pe_ttm")
    pb_panel = val_panel("pb")
    roe_panel = (pb_panel / pe_panel).where((pe_panel > 0) & (pb_panel > 0))
    divy_panel = trailing_dividend_yield(stock_divs, raw_close)
    mkt_rets = etf_rets["510300"]
    resmom_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

    def resmom(window: int = 240, skip: int = 20) -> pd.DataFrame:
        if (window, skip) not in resmom_cache:
            resmom_cache[(window, skip)] = residual_momentum_score(stock_rets, mkt_rets, window, skip)
        return resmom_cache[(window, skip)]

    def build_divy(c: pd.DataFrame, a: pd.DataFrame, n: int = 5, panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return strategy_stock_rank_low(-(divy_panel if panel is None else panel)[c.columns], c, a, 240, n)

    def build_quality(c: pd.DataFrame, a: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        return strategy_stock_rank_low(-roe_panel[c.columns], c, a, 240, n)

    def build_resmom(c: pd.DataFrame, a: pd.DataFrame, window: int = 240, n: int = 5) -> pd.DataFrame:
        return strategy_stock_rank_low(-resmom(window)[c.columns], c, a, window, n)

    def build_rawmom_12_1(c: pd.DataFrame, a: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        raw_mom = c.shift(20) / c.shift(240) - 1.0
        return strategy_stock_rank_low(-raw_mom, c, a, 240, n)

    index_pe = fetch_index_pe()
    cn10y = fetch_cn10y()
    weights["ERP_Timing"], erp_monthly = strategy_erp_timing(etf_close.index, index_pe, cn10y, 60, 36, "rank")
    erp_avg_w = float(weights["ERP_Timing"].loc[EVAL_START:LAST_BAR, "510300"].mean())

    weights["Stock_CS_Mom"] = strategy_stock_momentum(stock_close, stock_amount, 120, 5)
    weights["Stock_LowVol"] = strategy_stock_low_vol(stock_close, stock_amount, 60, 5)
    weights["Stock_MomLowVol"] = strategy_stock_mom_lowvol(stock_close, stock_amount, 120, 60, 5)
    weights["Stock_DivYield"] = build_divy(stock_close, stock_amount)
    weights["Stock_Quality"] = build_quality(stock_close, stock_amount)
    weights["Stock_ResMom"] = build_resmom(stock_close, stock_amount)

    stock_strats = {
        "EW_Stocks", "Stock_CS_Mom", "Stock_LowVol", "Stock_MomLowVol",
        "Stock_DivYield", "Stock_Quality", "Stock_ResMom",
    }
    cost_map = {n: (STOCK_COST_BPS if n in stock_strats else ETF_COST_BPS) for n in weights}
    ret_map = {n: (stock_rets if n in stock_strats else etf_rets) for n in weights}
    universe_bench = {
        "Sector_DualMom": "EW_Sectors",
        "Sector_LowVol": "EW_Sectors",
        "Stock_CS_Mom": "EW_Stocks",
        "Stock_LowVol": "EW_Stocks",
        "Stock_MomLowVol": "EW_Stocks",
        "Stock_DivYield": "EW_Stocks",
        "Stock_Quality": "EW_Stocks",
        "Stock_ResMom": "EW_Stocks",
        "RiskParity_SBG": "EW_SBG",
        # ERP_Timing always splits between CSI300 and treasuries; timing must beat holding both.
        "ERP_Timing": "EW_SB",
    }

    # Sensitivity diagnostics. Not used for selection. (spec -> weights, kind)
    sensitivity: Dict[str, Tuple[pd.DataFrame, str]] = {
        "DivY_n5": (weights["Stock_DivYield"], "stock"),
        "DivY_n3": (build_divy(stock_close, stock_amount, 3), "stock"),
        "DivY_n8": (build_divy(stock_close, stock_amount, 8), "stock"),
        "DivY_365d_doublecount": (build_divy(stock_close, stock_amount, 5, trailing_dividend_yield(stock_divs, raw_close, 365, None)), "stock"),
        "Quality_n5": (weights["Stock_Quality"], "stock"),
        "Quality_n3": (build_quality(stock_close, stock_amount, 3), "stock"),
        "Quality_n8": (build_quality(stock_close, stock_amount, 8), "stock"),
        "ResMom_w240_n5": (weights["Stock_ResMom"], "stock"),
        "ResMom_w120_n5": (build_resmom(stock_close, stock_amount, 120, 5), "stock"),
        "ResMom_w240_n3": (build_resmom(stock_close, stock_amount, 240, 3), "stock"),
        "ResMom_w240_n8": (build_resmom(stock_close, stock_amount, 240, 8), "stock"),
        "RawMom_12_1_n5": (build_rawmom_12_1(stock_close, stock_amount, 5), "stock"),
        "ERP_rank60": (weights["ERP_Timing"], "etf"),
        "ERP_rank36": (strategy_erp_timing(etf_close.index, index_pe, cn10y, 36, 36, "rank")[0], "etf"),
        "ERP_rank120": (strategy_erp_timing(etf_close.index, index_pe, cn10y, 120, 36, "rank")[0], "etf"),
        "ERP_median60": (strategy_erp_timing(etf_close.index, index_pe, cn10y, 60, 36, "median")[0], "etf"),
        # Same average equity share as ERP_Timing, no timing: isolates timing skill from extra beta.
        "SB_static_at_ERP_avg_w": (static_mix(etf_close.index, {"510300": erp_avg_w, "511260": 1.0 - erp_avg_w}), "etf"),
        "DualMom_60_top3": (strategy_sector_dual_mom(sector_close, sector_amount, 60, 3), "etf"),
        "DualMom_120_top3": (weights["Sector_DualMom"], "etf"),
        "DualMom_180_top3": (strategy_sector_dual_mom(sector_close, sector_amount, 180, 3), "etf"),
        "DualMom_120_top2": (strategy_sector_dual_mom(sector_close, sector_amount, 120, 2), "etf"),
        "DualMom_120_top4": (strategy_sector_dual_mom(sector_close, sector_amount, 120, 4), "etf"),
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
        "StkLowVol_vol120": (strategy_stock_low_vol(stock_close, stock_amount, 120, 5),         "stock"),
    }

    rf_daily: Optional[pd.Series] = None
    rf_note = ""
    try:
        shibor = fetch_shibor_3m()
        rf_daily = (shibor.reindex(etf_rets.index, method="ffill").shift(1) / 100.0 / TRADING_DAYS).loc[EVAL_START:last_date]
        rf_note = (
            f"无风险利率：3M Shibor（AkShare `rate_interbank`），滞后 1 日、按 252 天折算；评估期均值 "
            f"{rf_daily.mean() * TRADING_DAYS:.2%}，最近 {shibor.loc[:last_date].iloc[-1]:.2f}%（{shibor.loc[:last_date].index[-1].date()}）。"
        )
    except Exception as exc:  # noqa: BLE001
        rf_note = f"3M Shibor 获取失败（{type(exc).__name__}），超额现金夏普记为 n/a，门槛回退到 rf=0 夏普。"

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
    stock_rules = ["Stock_LowVol", "Stock_CS_Mom", "Stock_MomLowVol", "Stock_DivYield", "Stock_Quality", "Stock_ResMom"]
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
        "Stock_CS_Mom": lambda c, a: strategy_stock_momentum(c, a, 120, 5),
        "Stock_LowVol": lambda c, a: strategy_stock_low_vol(c, a, 60, 5),
        "Stock_MomLowVol": lambda c, a: strategy_stock_mom_lowvol(c, a, 120, 60, 5),
        "Stock_DivYield": lambda c, a: build_divy(c, a),
        "Stock_Quality": lambda c, a: build_quality(c, a),
        "Stock_ResMom": lambda c, a: build_resmom(c, a),
    }
    loo: Dict[str, Dict[str, object]] = {n: {"cagr": [], "excess": []} for n in loo_builders}
    for drop in stock_close.columns:
        keep = [c for c in stock_close.columns if c != drop]
        c_sub, a_sub, r_sub = stock_close[keep], stock_amount.reindex(columns=keep), stock_rets[keep]
        ew_net, _ = apply_costs(strategy_equal_weight(c_sub, a_sub, STOCK_MIN_AMOUNT), r_sub, STOCK_COST_BPS)
        ew_net = ew_net.loc[EVAL_START:last_date]
        for n, build in loo_builders.items():
            net, _ = apply_costs(build(c_sub, a_sub), r_sub, STOCK_COST_BPS)
            net = net.loc[EVAL_START:last_date]
            loo[n]["cagr"].append((drop, perf_stats(net)["cagr"]))
            loo[n]["excess"].append((drop, excess_cagr(net, ew_net)))
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
        ["Strategy", "Full CAGR", "LOO min CAGR", "LOO max CAGR", "Excess vs EW", "LOO excess vs EW range", "LOO excess>0"],
        loo_rows,
    )

    # Calendar-year return minus the universe benchmark: does an edge hold across regimes or sit in one window?
    year_list = list(range(EVAL_START.year, last_date.year + 1))
    yearly_rows = []
    for n in ["Stock_DivYield", "Stock_Quality", "Stock_ResMom", "ERP_Timing", "Stock_MomLowVol"]:
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

    benchmarks = {"BH_CSI300", "EW_Stocks", "EW_Sectors", "EW_SBG", "BH_Bond10Y", "EW_SB"}
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
            f"逐只剔除后相对 EW_Stocks 超额 CAGR 在 {fmt_pct(d['excess_min'][1])} ~ {fmt_pct(d['excess_max'][1])}，"
            f"{d['excess_positive']}/{d['n']} 次为正。"
        )

    def sens_sharpe(spec: str) -> str:
        return fmt_num(sens_rows[spec]["full"].get("sharpe", float("nan")))

    erp_w = weights["ERP_Timing"].loc[EVAL_START:last_date]
    erp_eq_avg = float(erp_w["510300"].mean())
    erp_eq_last = float(weights["ERP_Timing"].loc[:last_date].iloc[-1]["510300"])
    erp_me = erp_monthly.loc[:month_ends(etf_close.index)[-1]]
    erp_hist60 = erp_me.tail(60)
    erp_last_txt = (
        f"{erp_me.iloc[-1]:.2%}（{erp_me.index[-1].date()}，沪深300 TTM PE {index_pe.loc[:erp_me.index[-1]].iloc[-1]:.2f}，"
        f"近 60 个月区间 {erp_hist60.min():.2%} ~ {erp_hist60.max():.2%}，中位数 {erp_hist60.median():.2%}）"
    )

    def new_stock_note(n: str, specs: List[Tuple[str, str]]) -> str:
        s = stats[n]
        sens = "、".join(f"{lbl} {sens_sharpe(k)}/{sens_oos(k)}" for lbl, k in specs)
        return (
            f"样本内 Sharpe {fmt_num(s['is']['sharpe'])}、样本外 {fmt_num(s['oos']['sharpe'])}；"
            f"2022 年 {fmt_pct(s['stress']['year_2022'])}（EW_Stocks {fmt_pct(stats['EW_Stocks']['stress']['year_2022'])}），"
            f"2024-09~2025-03 {fmt_pct(s['stress']['rally_202409_202503'])}（EW_Stocks {fmt_pct(stats['EW_Stocks']['stress']['rally_202409_202503'])}），"
            f"2026 年至今 {fmt_pct(s['stress']['ytd_2026'])}；年换手 {fmt_num(s['full']['ann_turnover'], 1)} 倍。"
            f"敏感度（全样本/样本外 Sharpe）：{sens}。{loo_txt(n)}"
            f"与 Stock_LowVol 的超额收益相关 {fmt_num(float(excess_corr.at[n, 'Stock_LowVol']))}，持仓重合 {fmt_pct(overlap_with_lowvol(n), 0)}。"
        )

    def sens_oos(spec: str) -> str:
        return fmt_num(sens_rows[spec]["oos"].get("sharpe", float("nan")))

    notes = {
        "RiskParity_SBG": (
            f"评估期平均 {rp_bond:.0%} 仓位在十年国债ETF（当前 {rp_last_bond:.0%}），收益主要来自 2020–2025 年的利率下行；"
            f"单持国债 BH_Bond10Y 的 Sharpe-rf 为 {sc('BH_Bond10Y')}，静态三等分 EW_SBG 的全样本 CAGR {fmt_pct(stats['EW_SBG']['full']['cagr'])} 高于本策略。"
            "这是低波动组合，不是股票替代品；样本外总收益也低于沪深300。"
        ),
        "Stock_LowVol": (
            f"样本外 Sharpe {fmt_num(stats['Stock_LowVol']['oos']['sharpe'])} 但样本内只有 {fmt_num(stats['Stock_LowVol']['is']['sharpe'])}，"
            f"且全样本跑输股票池等权 {fmt_pct(-stats['Stock_LowVol']['excess_vs_universe_cagr'])}/年；它是更低回撤的池子 beta，不是选股 alpha。"
            f"上一版已发现它对 2023-02-28 一个近似平局的选股敏感。{loo_txt('Stock_LowVol')}"
        ),
        "Stock_MomLowVol": (
            f"保留候选里最强的个股规则：样本内 Sharpe {fmt_num(stats['Stock_MomLowVol']['is']['sharpe'])}、样本外 {fmt_num(stats['Stock_MomLowVol']['oos']['sharpe'])}，"
            f"2022 年 {fmt_pct(stats['Stock_MomLowVol']['stress']['year_2022'])}（股票池等权 {fmt_pct(stats['EW_Stocks']['stress']['year_2022'])}），"
            f"2026 年至今 {fmt_pct(stats['Stock_MomLowVol']['stress']['ytd_2026'])}（股票池等权 {fmt_pct(stats['EW_Stocks']['stress']['ytd_2026'])}）。"
            f"持股数 3/5/8 的 Sharpe 为 {sens_sharpe('MomLowVol_n3')} / {sens_sharpe('MomLowVol_n5')} / {sens_sharpe('MomLowVol_n8')}，"
            f"但动量窗口改为 60 日时降到 {sens_sharpe('MomLowVol_m60')}。仍跑输股票池等权，Sharpe-rf 未到 1。{loo_txt('Stock_MomLowVol')}"
        ),
        "Stock_CS_Mom": (
            f"相对股票池等权超额 {fmt_pct(stats['Stock_CS_Mom']['excess_vs_universe_cagr'])}，回撤 {fmt_pct(stats['Stock_CS_Mom']['full']['max_dd'])}；"
            f"超额来自池子构成。{loo_txt('Stock_CS_Mom')}"
        ),
        "Sector_LowVol": (
            f"样本外 Sharpe {fmt_num(stats['Sector_LowVol']['oos']['sharpe'])} 明显好于样本内 {fmt_num(stats['Sector_LowVol']['is']['sharpe'])}，"
            f"2026 年至今 {fmt_pct(stats['Sector_LowVol']['stress']['ytd_2026'])}，但全样本仍跑输行业 ETF 等权；当前持仓与防御板块强势高度重合。"
        ),
        "Sector_DualMom": (
            f"回撤 {fmt_pct(stats['Sector_DualMom']['full']['max_dd'])}，对持仓数参数敏感"
            f"（top2 / top3 / top4 Sharpe {sens_sharpe('DualMom_120_top2')} / {sens_sharpe('DualMom_120_top3')} / {sens_sharpe('DualMom_120_top4')}）。"
        ),
        "Stock_DivYield": new_stock_note(
            "Stock_DivYield",
            [("持 3 只", "DivY_n3"), ("持 8 只", "DivY_n8"), ("首跑的 365 天窗口（含重复计数）", "DivY_365d_doublecount")],
        ),
        "Stock_Quality": new_stock_note(
            "Stock_Quality",
            [("持 3 只", "Quality_n3"), ("持 8 只", "Quality_n8")],
        ),
        "Stock_ResMom": new_stock_note(
            "Stock_ResMom",
            [("回归窗口 120 日", "ResMom_w120_n5"), ("持 3 只", "ResMom_w240_n3"), ("持 8 只", "ResMom_w240_n8"),
             ("对照：原始 12-1 动量", "RawMom_12_1_n5")],
        )
        + f"与 Stock_CS_Mom 的超额收益相关 {fmt_num(float(excess_corr.at['Stock_ResMom', 'Stock_CS_Mom']))}。",
        "ERP_Timing": (
            f"评估期平均股票仓位 {erp_eq_avg:.0%}（股债 50/50 EW_SB 为 50%），当前 {erp_eq_last:.0%}；"
            f"最近一个月末 ERP {erp_last_txt}。"
            f"EW_SB 全样本 CAGR {fmt_pct(stats['EW_SB']['full']['cagr'])}、Sharpe {fmt_num(stats['EW_SB']['full']['sharpe'])}、回撤 {fmt_pct(stats['EW_SB']['full']['max_dd'])}。"
            f"样本内 Sharpe {fmt_num(stats['ERP_Timing']['is']['sharpe'])}、样本外 {fmt_num(stats['ERP_Timing']['oos']['sharpe'])}；"
            f"2022 年 {fmt_pct(stats['ERP_Timing']['stress']['year_2022'])}（EW_SB {fmt_pct(stats['EW_SB']['stress']['year_2022'])}），"
            f"2024-09~2025-03 {fmt_pct(stats['ERP_Timing']['stress']['rally_202409_202503'])}（EW_SB {fmt_pct(stats['EW_SB']['stress']['rally_202409_202503'])}），"
            f"2026 年至今 {fmt_pct(stats['ERP_Timing']['stress']['ytd_2026'])}（EW_SB {fmt_pct(stats['EW_SB']['stress']['ytd_2026'])}）；年换手 {fmt_num(stats['ERP_Timing']['full']['ann_turnover'], 1)} 倍。"
            f"敏感度（全样本/样本外 Sharpe）：分位窗口 36 个月 {sens_sharpe('ERP_rank36')}/{sens_oos('ERP_rank36')}、"
            f"120 个月 {sens_sharpe('ERP_rank120')}/{sens_oos('ERP_rank120')}、60 个月中位数二元切换 {sens_sharpe('ERP_median60')}/{sens_oos('ERP_median60')}。"
            f"同样 {erp_eq_avg:.0%} 平均股票仓位、但不择时的静态股债组合：全样本 Sharpe {sens_sharpe('SB_static_at_ERP_avg_w')}、"
            f"回撤 {fmt_pct(sens_rows['SB_static_at_ERP_avg_w']['full'].get('max_dd', float('nan')))}、样本外 Sharpe {sens_oos('SB_static_at_ERP_avg_w')}、"
            f"CAGR {fmt_pct(sens_rows['SB_static_at_ERP_avg_w']['full'].get('cagr', float('nan')))}。"
        ),
    }
    for k, v in INTERP_NOTES.items():
        notes[k] = notes.get(k, "") + v

    rec_bits = [rec_intro, ""]
    for n in ranked[:5]:
        f, o = stats[n]["full"], stats[n]["oos"]
        rec_bits.append(
            f"- **{n}**：全样本 CAGR {fmt_pct(f['cagr'])}，Sharpe {fmt_num(f['sharpe'])}（扣无风险 {sc(n)}），"
            f"最大回撤 {fmt_pct(f['max_dd'])}，样本外 Sharpe {fmt_num(o['sharpe'])}，"
            f"相对沪深300超额 CAGR {fmt_pct(stats[n]['excess_cagr'])}，{ub_txt(n)}。"
            f"{notes.get(n, '')}当前信号：{('、'.join(holdings.get(n) or [])) or '现金'}。"
        )
    rec_bits.append("")
    rec_bits.append(
        "不作为推荐的做法：用本次敏感度表里 Sharpe 最高的参数替换预注册参数（例如把 Stock_MomLowVol 改成持 8 只、把 Sector_DualMom 改成持 2 只、"
        "或把股息率规则改成持 8 只、把 ERP 分位窗口改成 120 个月这类敏感度表里看起来更好的档位）；把单一行业主题的历史 beta 当成策略 alpha。"
    )
    rec_bits.append("")
    new_cands = ["Stock_DivYield", "Stock_Quality", "Stock_ResMom", "ERP_Timing"]
    rec_bits.append("本期新增 4 个候选的逐项结论（证据见第 4、5 节）：")
    rec_bits.append("")
    for n in new_cands:
        f = stats[n]["full"]
        verdict = "通过门槛" if n in survivors else "未过门槛"
        rec_bits.append(
            f"- **{n}**（{verdict}）：扣费 CAGR {fmt_pct(f['cagr'])}，Sharpe-rf {sc(n)}，最大回撤 {fmt_pct(f['max_dd'])}，{ub_txt(n)}。{notes[n]}"
        )

    deploy = "\n".join(
        [
            "- 交易时点：收盘信号，下一交易日成交。回测用下一根收盘到收盘近似，实盘用开盘或 VWAP 会再吃一点滑点。",
            "- 仓位：单策略名义敞口先按 10% 年化波动目标缩放，且杠杆不超过 1。回撤超过策略自身历史最大回撤的 1.25 倍时降到一半仓位并人工复核。",
            "- 容量：行业 ETF 只交易 20 日成交额中位数 ≥ 2000 万的品种；单日成交不超过该 ETF 20 日均成交额的 2%。股票池限定本次 24 只大盘股，不外推到小盘。"
            "十年国债ETF(511260) 与黄金ETF(518880) 近 20 日成交额中位数均在 30 亿元以上，个人与中小机构规模不构成约束。",
            "- 成本监控：若实盘单边冲击超过回测假设（ETF 8bps / 股票 15bps），停止加仓。新增个股规则的年换手："
            + "、".join(f"{n} {stats[n]['full']['ann_turnover']:.1f} 倍" for n in ("Stock_DivYield", "Stock_Quality", "Stock_ResMom"))
            + f"；ERP_Timing {stats['ERP_Timing']['full']['ann_turnover']:.1f} 倍。成本翻倍的影响见第 4 节。",
            "- 残差动量 Stock_ResMom 年换手约 19 倍（排名在 24 只股票之间频繁交换），扣费后相对 EW_Stocks 的超额与成本同量级；"
            "若实盘单边成本超过约 23bps（按 15bps 与 30bps 两档结果线性插值的盈亏平衡点），这点超额就没有了。股息率规则年换手约 3 倍，对成本不敏感。",
            "- 个股策略的持仓只有 5 只：逐只剔除表显示单一名字能明显改变结果，实盘至少要把单票上限、行业上限写进风控，并接受与回测不同的持仓。",
            "- 低波动组合（RiskParity_SBG）的回撤阈值不能沿用股票口径：其历史最大回撤只有约 4%，按 1.25 倍规则在回撤 5% 时就应减半复核。",
            "- 组合：动量、低波动与股债配置的相关性不同，但都只是弱超额或无超额，不要因为「看起来不同」就叠加杠杆。",
            "- 现金：回测现金收益为 0。实盘空仓可放货币 ETF，但这部分收益不属于策略 alpha。",
            "",
            *hold_lines,
        ]
    )

    cn10 = bond.get("cn_10y")
    risk = "\n".join(
        [
            "- 动量类（Sector_DualMom、Stock_CS_Mom）在快速反转和震荡市会连续止损。失效条件：样本外滚动 12 个月超额持续为负，或 120 日与 60/180 日敏感度符号相反。",
            "- Stock_MomLowVol 依赖「动量与低波动不同时失效」。当前持仓集中在银行、电力、家电、电信等防御名字，与 Stock_LowVol 高度重叠。"
            "失效条件：高 beta 急涨（2024-09 那种）时同时跑输两个因子，或样本外滚动 12 个月相对 EW_Stocks 超额持续为负；60 日动量窗口下 Sharpe 已明显下降，是参数脆弱的信号。",
            (
                f"- RiskParity_SBG 是久期押注。中国 10Y 国债收益率当前约 {cn10:.2f}%，"
                if cn10 is not None
                else "- RiskParity_SBG 是久期押注。"
            )
            + "继续下行空间有限；若收益率上行 1 个百分点，十年国债ETF 价格大约下跌 8–9%（按约 8.5 年久期估算，属解释而非回测）。"
            "股债同时下跌（2022 年海外那种情形）时，按波动倒数配权无法分散。失效条件：国债ETF 滚动 12 个月收益转负且股票与国债相关性转正。"
            "ERP_Timing 的债券腿同样是十年国债，2020 年以来的利率下行也在它的收益里；ERP 本身也随 10Y 收益率下行而被动抬高。",
            "- 低波动（Stock_LowVol、Sector_LowVol）当前持仓集中在银行、电力、红利，与 Sector_DualMom 的防御持仓高度重叠，组合层面并不分散。失效条件：防御板块估值修复结束后的风格反转（如 2024-09 那种高 beta 急涨）。",
            *RISK_INTERP,
            "- 10-19 / 10-20 期淘汰的候选（WeeklyRev、NearHigh、VolMgd、Holiday_Post5、AbnTurnover、OvernightRev、LowMAX、Gold_TSMOM）不进入观察名单；不要换参数复活。",
            "- 幸存者偏差：股票池和 ETF 池都是 2026 年仍在交易的主流工具，没有加入已清盘 ETF 和退市股。结果对「当时可投资宇宙」偏乐观。",
            "- 制度变化：印花税、涨跌停、T+1、行业 ETF 扩容都会改变成本和收益结构。15bps 单边是常数近似；2023-08 印花税减半前的真实成本更高。",
            "- 数据：前复权在大比例分红除权日可能残留跳变，且新浪会回溯修订前复权历史（10-16 期 Stock_LowVol 曾因此变化约 1.8 个百分点 CAGR；本期复现时 4 个个股规则因工商银行 10-21 除息出现小幅变化，见第 3 节）；"
            "ETF 分红依赖新浪披露，部分 ETF 没有记录（见数据节）。",
            "- 宏观打印若与国债期货或财政部曲线不一致，以官方曲线为准，本报告不据此交易利率。",
        ]
    )

    self_check = "\n".join(
        [
            "- 缺了什么：行业资金流、北向净买入明细、期权持仓与隐含波动率、一致预期修正、时点正确的财报（净资产/利润按披露日生效）序列、指数 PE 的第二来源。这些缺失被记为缺口，没有用估计值填上。",
            "- 假设里较弱的一环：现金收益为 0、成本为常数 bps、ETF 收盘可成交、股票池有幸存者偏差、部分 ETF 分红缺失。",
            REPRO_SELF_CHECK,
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
        f"- 基本面覆盖：估值 {len(val_frames)}/{len(stock_frames)} 只、分红 {len(stock_divs)}/{len(stock_frames)} 只；"
        f"评估期月末有 ROE 的股票-月 {int(roe_panel.reindex(me_eval).notna().sum().sum())} 个、有股息率的 {int(divy_panel.reindex(me_eval).notna().sum().sum())} 个、"
        f"有残差动量得分的 {int(resmom().reindex(me_eval).notna().sum().sum())} 个（上限 {len(me_eval) * len(stock_close.columns)}）。"
        f"ERP 月度序列 {erp_monthly.index[0].date()} 至 {erp_monthly.index[-1].date()} 共 {len(erp_monthly)} 个点。"
    )
    gap_lines.append("- QDII（例如纳指 ETF）未纳入，因为溢价和申购限额会让收盘价成交假设失效。")
    gap_lines.append(
        "- 价格源：ETF 为新浪未复权（`fund_etf_hist_sina`）+ 新浪分红（`fund_etf_dividend_sina`），股票为新浪前复权（`stock_zh_a_daily`）。"
        "腾讯前复权与新浪日收益相关性不足，且出现过单日数倍的不可能跳变，本次不使用。"
    )
    with_div = [c for c, k in div_events.items() if k > 0]
    no_div = [c for c, k in div_events.items() if k == 0]
    top_div = sorted(with_div, key=lambda c: -div_total[c])[:5]
    gap_lines.append(
        f"- ETF 分红：评估期内有除息记录的 {len(with_div)} 只，累计加回收益率最高的是 "
        + "、".join(f"{etf_meta.get(c, c)} {div_total[c]:.1%}" for c in top_div)
        + f"。没有记录的 {len(no_div)} 只（"
        + "、".join(etf_meta.get(c, c) for c in no_div)
        + "）保持价格收益；其中银行ETF 等若实际有分红则被低估，没有估计填补。除息日已抽查与原始价格跳空一致（如沪深300ETF 2026-01-19 分红 0.123 元、当日价格 -2.5%）。"
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
        f"{n_cand} 个候选（6 个保留 + 4 个新增），严格质量门槛下合格数量：**{len(survivors)}**。"
        + ("名单：" + "、".join(survivors) + "。" if survivors else "没有任何候选同时满足扣费跑赢沪深300（全样本与样本外）、Sharpe-rf>1、回撤<30%、样本外 Sharpe>0.5、跑赢自身宇宙等权。"),
        "4 个新增候选（扣费后）："
        + "；".join(
            f"{n} CAGR {fmt_pct(stats[n]['full']['cagr'])}、Sharpe-rf {sc(n)}、回撤 {fmt_pct(stats[n]['full']['max_dd'])}、样本外 Sharpe {fmt_num(stats[n]['oos']['sharpe'])}、"
            f"相对 {stats[n]['universe_bench']} {fmt_pct(stats[n]['excess_vs_universe_cagr'])}/年"
            for n in new_cands
        )
        + "。",
        f"保留的观察名单：Stock_MomLowVol 扣费 CAGR {fmt_pct(stats['Stock_MomLowVol']['full']['cagr'])}，Sharpe-rf {sc('Stock_MomLowVol')}，"
        f"样本外 Sharpe {fmt_num(stats['Stock_MomLowVol']['oos']['sharpe'])}，相对股票池等权 {fmt_pct(stats['Stock_MomLowVol']['excess_vs_universe_cagr'])}/年；"
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
