"""Analysis for the returns-map test (correlation PCA on daily returns)."""
import pickle, numpy as np, pandas as pd
from analysis_utils import nw_t, monthly, fama_macbeth, sector_check

P = pd.read_parquet("returns_map_panel.parquet")
M = pickle.load(open("returns_map_meta.pkl", "rb"))
SIG = ["drift_gmm_PC2-3", "drift_gmm_PC2-4", "drift_gmm_PC1-2", "drift_hdb_PC2-3"]
H = {"1w": 5, "1m": 21, "3m": 63}

rows = []
for s in SIG:
    for hn, h in H.items():
        for rt in ["fwd", "fwdadj"]:
            ic, spr = monthly(P, s, f"{rt}_{hn}")
            lags = max(1, int(np.ceil(h / 21)))
            rows.append(dict(signal=s, horizon=hn, returns="raw" if rt == "fwd" else "beta-adj",
                             months=len(ic), mean_IC=ic.mean(), IC_t=nw_t(ic, lags),
                             Q5_minus_Q1_pct=100 * spr.mean(), spread_t=nw_t(spr, lags)))
T = pd.DataFrame(rows)
print("== Drift vs future returns ==")
print(T.round(4).to_string(index=False))
T.to_csv("results/returns_map_ic_table.csv", index=False)

print("== Positive control: known effects through the same test (1m, beta-adjusted) ==")
for s in ["rev_1m", "mom_12_1", "vol_1y", "srisk"]:
    ic, _ = monthly(P, s, "fwdadj_1m"); print(f"  {s:10s} IC {ic.mean():+.4f}  t {nw_t(ic):+.2f}")
print("== Level distance (not drift), 1m ==")
for s in ["dA_gmm_PC2-3", "dA_gmm_PC1-2"]:
    ic, _ = monthly(P, s, "fwdadj_1m"); print(f"  {s:14s} IC {ic.mean():+.4f}  t {nw_t(ic):+.2f}")
print("== Subperiods, 1m beta-adjusted IC ==")
for s in ["drift_gmm_PC2-3", "drift_gmm_PC1-2"]:
    for lo, hi in [("2001", "2011-12-31"), ("2012", "2022-12-31")]:
        ic, _ = monthly(P[(P.date >= lo) & (P.date <= hi)], s, "fwdadj_1m")
        print(f"  {s:16s} {lo}-{hi[:4]}  IC {ic.mean():+.4f} t {nw_t(ic):+.2f}")
print("== Fama-MacBeth, 1m beta-adjusted (coef in %/month per 1 sd) ==")
ctrl = ["rev_1m", "mom_12_1", "vol_1y", "beta", "srisk"]
for s in ["drift_gmm_PC2-3", "drift_gmm_PC1-2"]:
    for lab, cols in [("alone", [s]), ("with controls", [s, "dA_gmm_" + s.split("_")[-1]] + ctrl)]:
        c, t = fama_macbeth(P, cols); print(f"  {s:16s} {lab:14s} coef {c:+.3f}  t {t:+.2f}")
print("== Map diagnostics ==")
sh = np.array([m["share"] for m in M]); st = np.array([m["stab_axis"] for m in M])
print("  variance share of patterns 1-4 (avg %):", np.round(sh.mean(0) * 100, 1))
print("  quarter-to-quarter position stability, patterns 1-4 (avg corr):", np.round(st.mean(0), 2))
for k in ["k_PC2-3", "k_PC2-4", "k_PC1-2", "k_hdb"]:
    v = np.array([m[k] for m in M]); print(f"  groups chosen {k:8s}: median {np.median(v):.0f}, range {v.min()}-{v.max()}")
sector_check(P, M, "labels_PC2-3", "k_PC2-3", "patterns 2-3 map")
