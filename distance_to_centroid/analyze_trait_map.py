"""Analysis for the trait-map test (PCA on nine stock characteristics)."""
import pickle, numpy as np, pandas as pd
from analysis_utils import nw_t, monthly, fama_macbeth, sector_check
from trait_map_engine import STABLE, MAPS

P = pd.read_parquet("trait_map_panel.parquet")
M = pickle.load(open("trait_map_meta.pkl", "rb"))
SIG = [f"drift_gmm_{n}" for n in MAPS] + ["drift_hdb_all9-PC1-2"]

rows = []
for s in SIG:
    for hn, h in {"1w": 5, "1m": 21, "3m": 63}.items():
        ic, spr = monthly(P, s, f"fwdadj_{hn}")
        lags = max(1, int(np.ceil(h / 21)))
        rows.append(dict(signal=s.replace("drift_", ""), horizon=hn, IC=ic.mean(), IC_t=nw_t(ic, lags),
                         Q5_minus_Q1_pct=100 * spr.mean(), spread_t=nw_t(spr, lags)))
T = pd.DataFrame(rows)
print("== Drift vs future beta-adjusted returns ==")
print(T.round(4).to_string(index=False))
T.to_csv("results/trait_map_ic_table.csv", index=False)

print("== Subperiods, 1m IC ==")
for s in ["drift_gmm_all9-PC1-2", "drift_gmm_stable6-PC1-2"]:
    for lo, hi in [("2001", "2011-12-31"), ("2012", "2022-12-31")]:
        ic, _ = monthly(P[(P.date >= lo) & (P.date <= hi)], s, "fwdadj_1m")
        print(f"  {s:26s} {lo}-{hi[:4]} IC {ic.mean():+.4f} t {nw_t(ic):+.2f}")
print("== Fama-MacBeth, 1m (coef in %/month per 1 sd of signal) ==")
known = ["rev_1m", "mom_12_1", "vol_1y"]
for s in ["drift_gmm_all9-PC1-2", "drift_gmm_stable6-PC1-2"]:
    name = s.split("_")[-1]
    for lab, cols in [("alone", [s]), ("+ momentum/reversal/vol", [s] + known),
                      ("+ trait levels & changes", [s, "dA_gmm_" + name] + known
                       + [f"lvl_{t}" for t in STABLE] + [f"chg_{t}" for t in STABLE])]:
        c, t = fama_macbeth(P, cols); print(f"  {name:16s} {lab:26s} coef {c:+.3f}  t {t:+.2f}")
print("== Positive control: trait levels and changes, 1m IC ==")
for t in STABLE:
    for k in ["lvl", "chg"]:
        ic, _ = monthly(P, f"{k}_{t}", "fwdadj_1m"); print(f"  {k}_{t:12s} IC {ic.mean():+.4f} t {nw_t(ic):+.2f}")
print("== Map diagnostics ==")
st = np.array([m["trait_stab"] for m in M])
print("  trait stability, quarter to quarter (avg corr):")
from trait_map_engine import TRAITS
for j, nm in enumerate(TRAITS):
    print(f"    {nm:13s} {st[:, j].mean():+.2f}")
for n in MAPS:
    sh = np.mean([m[f"share_{n}"][:3] for m in M], 0); ax = np.mean([m[f"axstab_{n}"] for m in M], 0)
    k = [m[f"k_{n}"] for m in M]
    print(f"  {n:15s} variance share PCs {np.round(100 * sh, 1)}  axis stability {np.round(ax, 2)}  "
          f"groups median {np.median(k):.0f} range {min(k)}-{max(k)}")
print("  HDBSCAN groups median", np.median([m["k_hdb"] for m in M]))
for n in ["all9-PC1-2", "stable6-PC1-2"]:
    tr = MAPS[n][0]; ref = M[0][f"load_{n}"]; acc = np.zeros_like(ref)
    for m in M:
        L = m[f"load_{n}"].copy()
        for j in range(L.shape[1]):
            if L[:, j] @ ref[:, j] < 0:
                L[:, j] *= -1
        acc += L
    acc /= len(M)
    print(f"  average loadings, {n}:")
    print(pd.DataFrame(acc[:, :2], index=tr, columns=["PC1", "PC2"]).round(2).to_string())
sector_check(P, M, "labels", "k_all9-PC1-2", "all-9 2D map")
