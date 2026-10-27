#!/usr/bin/env python3
"""Post-hoc (not pre-registered) decomposition of Stock_MarginCrowd for the 2026-10-27 report.

Month-end buckets of the 3-month SSE financing-balance change: bottom 5 (the rule), middle, top 5,
next-month simple returns without costs, and the share of the bottom-5 excess that came from 601138.
Run from this directory after run_research.py has filled the cache. Diagnostic only; no rule uses it.
"""
import os, sys, numpy as np, pandas as pd
os.environ.setdefault("RESEARCH_DATE","2026-10-27"); os.environ.setdefault("LAST_BAR","2026-10-26")
sys.path.insert(0, "."); import run_research as R
frames={c:R.fetch_stock_sina(c) for c in R.STOCKS}
for k in frames: frames[k]=frames[k].loc[frames[k]["date"]<=R.LAST_BAR].reset_index(drop=True)
lim={c:(0.23 if c.startswith(("300","688")) else 0.115) for c in frames}
close,rets,amt=R.panel_from_frames(frames,lim)
sse=[c for c in R.SSE_STOCKS if c in close.columns]
me=R.month_ends(close.index); pos=close.index.searchsorted(me)
prev=[close.index[p-1] for p in pos]; m2=dict(zip(prev,me))
dates=[d for d,m in zip(prev,me) if m>=pd.Timestamp("2019-06-01")]
bal,_=R.fetch_stock_margin_sse(dates,sse); bal.index=pd.DatetimeIndex([m2[d] for d in bal.index])
sc=R.margin_change_panel(bal,close.index,3)[sse]
ok=R.eligible_mask(close[sse],amt.reindex(columns=sse),R.STOCK_MIN_AMOUNT,120)
mc=close.reindex(me); fwd=(mc.shift(-1)/mc-1)
rows=[]
for d in [x for x in me if R.EVAL_START<=x<=pd.Timestamp("2026-09-30")]:
    s=sc.loc[d][ok.loc[d].reindex(sse).fillna(False)].dropna()
    if len(s)<10: continue
    r=s.rank(); f=fwd.loc[d,s.index]
    lo=s.nsmallest(5).index; hi=s.nlargest(5).index; mid=[c for c in s.index if c not in lo and c not in hi]
    rows.append(dict(d=d,lo=f[lo].mean(),mid=f[mid].mean(),hi=f[hi].mean(),ew=f.mean(),fii_in_lo="601138" in lo, fii=f.get("601138",np.nan)))
df=pd.DataFrame(rows).set_index("d")
print("months",len(df)); print((df[["lo","mid","hi","ew"]].mean()*12).round(4))
t=lambda x:x.mean()/x.std()*np.sqrt(len(x))
print("lo-ew t",round(t(df.lo-df.ew),2)," hi-ew t",round(t(df.hi-df.ew),2), " lo-hi t", round(t(df.lo-df.hi),2))
print("lo-ew by year"); print(((df.lo-df.ew).groupby(df.index.year).sum()).round(3))
x=df[df.fii_in_lo]; print("months 601138 in bottom5:",len(x), "its mean fwd ret in those months", round(x.fii.mean(),4), " contribution to lo (1/5)*sum:", round((x.fii/5).sum(),3))
print("lo-ew total sum", round((df.lo-df.ew).sum(),3))
