"""Characteristic-based cluster drift.

At month-end t, with F = month-end three months earlier:
  Quarter A = 63 trading days ending at F,  quarter B = F+1 .. t.
  Nine characteristics are measured separately in each quarter, converted to
  cross-sectional normal scores, and PCA is fit on quarter A (frozen loadings).
  Positions A and B use those same loadings; groups come from quarter A.
  drift = dist(B, center) - dist(A, center)
"""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN

TRAITS = ["beta", "own_risk", "crash_asym", "lottery", "crash_day",
          "reaction_lag", "bounce", "illiquidity", "log_price"]
# map name -> (traits used, PCs kept). The stable set drops the three traits whose
# quarter-to-quarter stability averaged <= 0.15 (decided before looking at returns).
STABLE = ["beta", "own_risk", "lottery", "crash_day", "illiquidity", "log_price"]
MAPS = {"all9-PC1-2": (TRAITS, [0, 1]), "all9-PC1-3": (TRAITS, [0, 1, 2]),
        "stable6-PC1-2": (STABLE, [0, 1]), "stable6-PC1-3": (STABLE, [0, 1, 2])}
KS = [2, 3, 4, 5, 6, 8, 10, 12]
HORIZONS = {"1w": 5, "1m": 21, "3m": 63}


def normal_scores(x):
    """Rank across stocks, then map ranks to a bell curve (mean 0, sd ~1)."""
    r = rankdata(x, method="average")  # tied values share the same rank
    return norm.ppf((r - 0.5) / len(x))


def window_traits(Rw, m, m_lag, beta_snap, srisk_snap, price_snap):
    """Characteristics for every stock from one quarter of daily returns."""
    X = Rw.astype(np.float64)
    ok = ~np.isnan(X)
    Xf = np.where(ok, X, 0.0)
    n = ok.sum(0)

    def masked_beta(days):
        """Beta using only the selected days (e.g. down-market days)."""
        md = m[days] - m[days].mean()
        Xd, okd = X[days], ok[days]
        xbar = np.where(okd, Xd, 0.0).sum(0) / np.maximum(okd.sum(0), 1)
        xd = np.where(okd, Xd - xbar, 0.0)
        return (md @ xd) / max(md @ md, 1e-18)

    crash_asym = masked_beta(m < 0) - masked_beta(m > 0)
    lottery = np.nanmax(X, 0)
    crash_day = np.nanmin(X, 0)
    # reaction speed: coefficient on yesterday's market return in r_t = a + b m_t + c m_{t-1}
    D = np.column_stack([np.ones(len(m)), m, m_lag])
    coef = np.linalg.lstsq(D, Xf - (Xf.sum(0) / np.maximum(n, 1)) * ok, rcond=None)[0]
    reaction_lag = coef[2]
    # bounce: lag-1 autocorrelation of daily returns
    Xc = np.where(ok, X - np.nanmean(X, 0), 0.0)
    bounce = (Xc[1:] * Xc[:-1]).sum(0) / np.maximum((Xc ** 2).sum(0), 1e-18)
    illiquidity = ((X == 0) & ok).sum(0) / np.maximum(n, 1)
    T = np.column_stack([beta_snap, srisk_snap, crash_asym, lottery, crash_day,
                         reaction_lag, bounce, illiquidity, np.log(price_snap)])
    return T


def gmm_bic(X, seed=0):
    best = None
    for k in KS:
        g = GaussianMixture(k, covariance_type="full", random_state=seed, max_iter=200).fit(X)
        b = g.bic(X)
        if best is None or b < best[0]:
            best = (b, g, k)
    return best[1].predict(X), best[1].means_, best[2]


def hdbscan_centers(X, min_size):
    lab = HDBSCAN(min_cluster_size=min_size, copy=True).fit(X).labels_
    cl = [c for c in np.unique(lab) if c >= 0]
    if len(cl) < 2:
        return None, None, len(cl)
    C = np.array([X[lab == c].mean(0) for c in cl])
    lab2 = np.array([cl.index(l) if l >= 0 else -1 for l in lab])
    noise = lab2 < 0
    if noise.any():
        lab2[noise] = np.argmin(((X[noise, None, :] - C[None]) ** 2).sum(-1), 1)
    return lab2, C, len(cl)


def month_signals(R, mkt, me, snap, mi, hold_idx):
    t, F = me[mi], me[mi - 3]
    nB = t - F
    cand = np.where((snap["price"][mi - 3] >= 5) & ~np.isnan(snap["price"][mi])
                    & ~np.isnan(snap["predicted_beta"][mi - 3]) & ~np.isnan(snap["predicted_beta"][mi])
                    & ~np.isnan(snap["specific_risk"][mi - 3]) & ~np.isnan(snap["specific_risk"][mi]))[0]
    okA = (~np.isnan(R[F - 62:F + 1, cand])).sum(0) >= 55
    okB = (~np.isnan(R[F + 1:t + 1, cand])).sum(0) >= nB - 8
    U = cand[okA & okB]

    TA = window_traits(R[F - 62:F + 1, U], mkt[F - 62:F + 1], mkt[F - 63:F],
                       snap["predicted_beta"][mi - 3][U], snap["specific_risk"][mi - 3][U],
                       snap["price"][mi - 3][U])
    TB = window_traits(R[F + 1:t + 1, U], mkt[F + 1:t + 1], mkt[F:t],
                       snap["predicted_beta"][mi][U], snap["specific_risk"][mi][U],
                       snap["price"][mi][U])
    keep = np.isfinite(TA).all(1) & np.isfinite(TB).all(1)
    U, TA, TB = U[keep], TA[keep], TB[keep]
    ZA = np.column_stack([normal_scores(TA[:, j]) for j in range(TA.shape[1])])
    ZB = np.column_stack([normal_scores(TB[:, j]) for j in range(TB.shape[1])])

    out = {"stock": U, "mi": np.full(len(U), mi)}
    meta = {"mi": mi, "n": len(U),
            "trait_stab": [np.corrcoef(ZA[:, j], ZB[:, j])[0, 1] for j in range(ZA.shape[1])]}
    for name, (tr, ax) in MAPS.items():
        idx = [TRAITS.index(x) for x in tr]
        # PCA fit on quarter A (frozen), applied to both quarters
        ev, vec = np.linalg.eigh(np.corrcoef(ZA[:, idx], rowvar=False))
        order = np.argsort(ev)[::-1]
        ev, vec = ev[order], vec[:, order]
        PA, PB = ZA[:, idx] @ vec[:, ax], ZB[:, idx] @ vec[:, ax]
        sc = PA.std(0)
        A, B = PA / sc, PB / sc
        lab, Cn, k = gmm_bic(A)
        dA, dB = np.linalg.norm(A - Cn[lab], axis=1), np.linalg.norm(B - Cn[lab], axis=1)
        out[f"drift_gmm_{name}"] = dB - dA
        out[f"dA_gmm_{name}"] = dA
        meta[f"k_{name}"] = k
        meta[f"share_{name}"] = ev / ev.sum()
        meta[f"load_{name}"] = vec[:, :3]
        meta[f"axstab_{name}"] = [np.corrcoef(PA[:, j], PB[:, j])[0, 1] for j in range(len(ax))]
        if name == "all9-PC1-2":
            meta["labels"] = lab
            lab_h, Ch, nh = hdbscan_centers(A, max(25, len(U) // 100))
            meta["k_hdb"] = nh
            out["drift_hdb_all9-PC1-2"] = (np.full(len(U), np.nan) if lab_h is None else
                                           np.linalg.norm(B - Ch[lab_h], axis=1) - np.linalg.norm(A - Ch[lab_h], axis=1))
    for j, nm in enumerate(TRAITS):  # quarter-B levels and changes, for the disguise check
        out[f"lvl_{nm}"] = ZB[:, j]
        out[f"chg_{nm}"] = ZB[:, j] - ZA[:, j]

    s = t + 2
    for hn, h in HORIZONS.items():
        e = s + h
        if e > hold_idx:
            out[f"fwd_{hn}"] = np.full(len(U), np.nan); out[f"fwdadj_{hn}"] = np.full(len(U), np.nan)
            continue
        r = np.prod(1 + np.nan_to_num(R[s:e, U].astype(np.float64)), 0) - 1
        mr = np.prod(1 + mkt[s:e]) - 1
        out[f"fwd_{hn}"] = r
        out[f"fwdadj_{hn}"] = r - snap["predicted_beta"][mi][U] * mr
    out["rev_1m"] = np.prod(1 + np.nan_to_num(R[t - 20:t + 1, U].astype(np.float64)), 0) - 1
    out["mom_12_1"] = np.prod(1 + np.nan_to_num(R[t - 251:t - 20, U].astype(np.float64)), 0) - 1
    out["vol_1y"] = np.nanstd(R[t - 251:t + 1, U].astype(np.float64), 0)
    return out, meta
