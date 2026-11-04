"""Shared test functions: IC, quintile spreads, Fama-MacBeth, sector check."""
import pickle, warnings
import numpy as np, pandas as pd
from config import DATA_PATH
from data_utils import tickers_on
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

SECTORS = {
    "tech": ["AAPL", "MSFT", "NVDA", "INTC", "CSCO", "ORCL", "ADBE", "AMD", "QCOM", "TXN"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "HAL", "MPC", "PSX", "VLO"],
    "banks": ["JPM", "BAC", "WFC", "C", "USB", "PNC", "GS", "MS", "TFC", "FITB"],
    "utilities": ["DUK", "SO", "NEE", "D", "AEP", "EXC", "XEL", "ED", "PEG", "WEC"],
    "health": ["JNJ", "PFE", "MRK", "ABT", "LLY", "BMY", "AMGN", "GILD", "CVS", "UNH"],
    "staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MO", "KMB", "GIS", "MDLZ"]}


def nw_t(x, lags=1):
    """t-stat of the mean with a Newey-West correction for overlapping horizons."""
    x = np.asarray(x); x = x[~np.isnan(x)]; n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return m / np.sqrt(v / n)


def monthly(P, sig, ret):
    """Each month: rank correlation (IC) and top-minus-bottom quintile return."""
    ic, spr = [], []
    for d, g in P[["date", sig, ret]].dropna().groupby("date"):
        if len(g) < 200:
            continue
        ic.append(g[sig].rank().corr(g[ret].rank()))
        q = pd.qcut(g[sig].rank(method="first"), 5, labels=False)
        spr.append(g[ret][q == 4].mean() - g[ret][q == 0].mean())
    return np.array(ic), np.array(spr)


def fama_macbeth(P, cols, ret="fwdadj_1m"):
    """Monthly cross-sectional regressions on rank-standardized variables;
    returns the first variable's average coefficient (%/month per 1 sd) and t-stat."""
    co = []
    for d, g in P[["date", ret] + cols].dropna().groupby("date"):
        X = g[cols].rank(pct=True).values - 0.5
        X = np.column_stack([np.ones(len(X)), X / X.std(0)])
        co.append(np.linalg.lstsq(X, g[ret].values, rcond=None)[0][1])
    co = np.array(co)
    return 100 * co.mean(), nw_t(co)


def sector_check(P, M, labels_key, k_key, title):
    target = [m for m in M if str(P[P.mi == m["mi"]].date.iloc[0])[:7] == "2019-12"][0]
    sub = P[P.mi == target["mi"]].copy()
    sub["cluster"] = target[labels_key]
    sub = sub.merge(tickers_on(DATA_PATH, "2019-09-30"), on="barrid", how="left")
    print(f"== Sector check, groups formed for 2019-12 ({title}: k={target[k_key]}, "
          f"sizes {np.bincount(sub.cluster).tolist()}) ==")
    for sec, tk in SECTORS.items():
        counts = sub[sub.ticker.isin(tk)]["cluster"].value_counts()
        print(f"  {sec:9s}", {int(k): int(v) for k, v in counts.items()})
