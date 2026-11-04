# Distance-to-centroid test

**Question:** does a stock drifting away from (or back toward) the center of its cluster predict its future returns?

**Answer:** no. Across two independent map designs and every variant we tried, drift showed no reliable or economically meaningful predictive power. Development period: 2001 to Oct 2022. Everything from Nov 2022 onward is a sealed holdout and was never used.

## What was tested

Each month-end *t*:

1. **Freeze a map** at the month-end three months earlier (*F*).
2. **Place every stock on it twice:** using quarter A (the 63 trading days ending at *F*) and quarter B (*F* to *t*).
3. **Form groups** on the quarter-A positions with a Gaussian mixture (number of groups chosen automatically by BIC). HDBSCAN is run as a comparison.
4. **Drift = distance to group center in quarter B minus distance in quarter A.** Positive means the stock moved away from its group.
5. **Compare drift with beta-adjusted returns** over the next 1 week, 1 month, and 3 months, starting two trading days after *t* (a one-day trading lag).

Two kinds of maps:

| Test | Map axes | Variants |
|---|---|---|
| Returns map | Correlation PCA on 1 year of daily returns (patterns stocks share) | Patterns 2-3 (main), 2-4 (3D), 1-2 (includes the market pattern) |
| Trait map | PCA on nine characteristics measured each quarter: beta, specific risk, crash asymmetry, largest up-day, largest down-day, reaction to yesterday's market, day-to-day bounce, zero-return days, log price | All 9 traits; a "stable 6" set (dropped the 3 traits whose quarter-to-quarter stability averaged 0.15 or less, decided before looking at returns); 2D and 3D |

## Results (1-month horizon, beta-adjusted)

| Signal | IC | t-stat |
|---|---|---|
| Returns map, patterns 2-3 (main) | -0.000 | -0.1 |
| Returns map, patterns 2-4 (3D) | -0.001 | -0.3 |
| Returns map, patterns 1-2 | +0.0045 | +1.7 (t = 0.4 after controls) |
| Trait map, all 9, 2D | -0.002 | -1.1 |
| Trait map, stable 6, 2D | +0.002 | +1.1 (t = 0.3 after controls) |

Full tables for all horizons are in `results/`. Our success bar was an IC of about 0.02 with t > 2. Only one of about 30 drift tests passed a significance test (trait map, all 9, 3D, 3-month horizon: IC -0.008, t = -3.7). It is tiny, appears at only one horizon in only one map, and its quintile spread is not significant, so we treat it as chance.

**The test works.** Known effects run through the same code are detected clearly: volatility IC -0.063 (t = -9.2), specific risk -0.055 (t = -8.2), momentum +0.021 (t = +2.8).

**Why drift fails:** distance ignores direction. Stocks becoming riskier underperform (change in beta: IC -0.020, t = -3.6; change in specific risk: IC -0.011, t = -3.8), and stocks becoming safer do the opposite. Both count as drifting away, so the effects cancel. The maps also have no natural clusters: the mixture usually picks 2-3 groups and HDBSCAN finds about 2. The returns-map groups look like broad sectors; the trait-map groups split mainly into big, steady stocks versus smaller, riskier ones.

## Verification

We checked the pipeline by recomputing several stock-months independently from the raw data rows (beta, last-month return, momentum, next-month return, largest one-day gain, share of zero-return days). Every value matched. The next-month return only matches when it starts two trading days after the signal date, which confirms there's no look-ahead. A full rerun on a second computer reproduced all Gaussian-mixture results exactly. HDBSCAN results differed slightly between computers (borderline stocks get grouped differently) but reached the same conclusion.

## How to run

Requires Python 3.10+ and `pip install polars numpy pandas scipy scikit-learn pyarrow`. From this folder:

```bash
python run_returns_map.py      # ~3-5 min, writes returns_map_panel.parquet
python analyze_returns_map.py  # prints results, writes results/returns_map_ic_table.csv
python run_trait_map.py        # ~7-10 min, writes trait_map_panel.parquet
python analyze_trait_map.py    # prints results, writes results/trait_map_ic_table.csv
```

`config.py` finds the data automatically if `BotQ_2026_data.parquet` (or `.csv`) is in this folder or the folder above it. To use a file somewhere else, set the environment variable `BOTQ_DATA` to its full path. The loader handles the Parquet or CSV version, with dates stored as dates or as text. If the data file came from GitHub's "Download ZIP," it's only a small Git LFS placeholder; you need the real file (about 0.5 GB). Needs about 3 GB of RAM.

## Files

| File | Purpose |
|---|---|
| `config.py` | Finds the data file; sets the cache path and holdout date |
| `data_utils.py` | Loads the panel; builds the daily return matrix and month-end snapshots |
| `returns_map_engine.py` | Monthly map, clusters, and drift for the returns map |
| `trait_map_engine.py` | Same for the trait map, including the nine characteristics |
| `run_*.py` | Build all months of signals, forward returns, and controls |
| `analysis_utils.py` | IC, quintile spreads, Fama-MacBeth regressions, sector check |
| `analyze_*.py` | Produce the result tables |
| `results/` | Saved outputs from our run |

## Caveats

- **Benchmark:** the data has no market caps, so the market is an equal-weighted average of the panel's stocks.
- **Delistings:** stocks leaving the panel mid-period are held at a 0% return (no delisting returns in the data).
- **HDBSCAN reproducibility:** HDBSCAN results can differ slightly between computers and library versions, because borderline stocks get grouped differently. The conclusion is the same everywhere. Gaussian-mixture results reproduce exactly.
- **Scope:** only straight-line distance, quarter-over-quarter drift, and monthly rebalancing were tested. Other definitions remain untested, but each extra variant raises the risk of a chance result.