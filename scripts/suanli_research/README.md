# 算力板块策略研究 (Compute Sector Research)

Reproduce the trading-strategy-research run for A-share AI compute names.

```bash
python3 scripts/suanli_research/run_research.py
```

Outputs land in `scripts/suanli_research/output/`:

- `REPORT.md` — ranked research report
- `metrics.json` / `ranking.json` — quantitative results
- `equity_curves.csv` / `equity_curves.png` — equity paths
- `market_snapshot.json` — latest breadth / returns snapshot
