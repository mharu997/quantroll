# quantroll — use-case notebooks

Five end-to-end, executed walkthroughs on **real daily ticker data** (56 US ETFs and large-cap stocks, 2012 → today, cached in `data/` and re-downloadable via `yfinance`). Each notebook demonstrates a failure of the naive workflow, the stable alternative, how to read every output, and what the result is worth in a real process.

| # | Notebook | Question it answers | Tool | Headline result on this data |
|---|---|---|---|---|
| 01 | `01_stable_rolling_pca.ipynb` | What are the market's factors, and is their structure shifting? | `RollingPCA` | 6,256 solver sign-flips removed over a decade of sector-ETF windows; absorption ratio tracks every crisis; real XLRE/XLC inceptions handled as variable universes; 10-year refit in 0.04 s |
| 02 | `02_regime_detection.ipynb` | What regime are we in, with what probability? | `RollingRegimes` | Every marquee stress episode (2016, volmageddon, Q4-2018, COVID, 2022 bear, 2024 carry unwind) flagged in real time; calm vs turbulent SPY ≈ +21%/10% vol vs +8%/26% vol; 3.6× less whipsaw than per-day classification |
| 03 | `03_stock_peer_groups.ipynb` | Which names behave alike, and who is drifting? | `RollingKMeans` | Naive monthly K-Means churns 74% of stock-months; stabilized version 14% — all of it signal. Stable anchors: PG, KO. Style migrants: INTC, BA, TSLA |
| 04 | `04_universe_map.ipynb` | What does the whole universe look like, in one stable picture? | `RollingEmbedding` (UMAP) | 4× calmer maps than naive re-embedding; sector neighborhoods emerge without labels; disparity spikes date the vaccine rotation and the 2022–23 inflation→AI reshuffle |
| 05 | `05_explainable_scorecard.ipynb` | Given a target, what should we hold, and *why*? | `Scorecard`, `measures` | 13 measures → one auditable ranking with per-measure explanation; weight policies (equal vs drawdown-averse) visibly re-rank tech down, defensives up |

## Running them

```bash
pip install -e ".[dev,umap]" jupyter yfinance   # from the repo root
jupyter lab notebooks/
```

The notebooks read cached CSVs from `notebooks/data/` and only hit the network (via `yfinance`) if the cache is missing. Notebook 04 falls back to `backend="pca"` if `umap-learn` isn't installed.

A shared helper, `quantdata.py`, holds the ticker lists, sector map, a colorblind-safe fixed-order palette, and the plot style used throughout.

*All notebooks are educational examples, not investment advice.*
