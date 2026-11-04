"""Cluster-drift signal on frozen correlation-PCA maps.

For each month-end t (signal date):
  F  = month-end three months earlier (freeze date)
  Fit window  : 252 trading days ending at F   -> correlation PCA (frozen axes)
  Window A    : 63 trading days ending at F    -> position A
  Window B    : trading days F+1 .. t          -> position B
  Clusters are fit on the position-A map; each stock keeps its A cluster.
  drift = dist(B, center) - dist(A, center)
Forward returns start at t+2 (one-day trading lag) and never extend into the holdout.
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN

CONFIGS = {"PC2-3": [1, 2], "PC2-4": [1, 2, 3], "PC1-2": [0, 1]}  # zero-based pattern indices
KS = [2, 3, 4, 5, 6, 8, 10, 12]
HORIZONS = {"1w": 5, "1m": 21, "3m": 63}


def standardize(W, mu=None, sd=None):
    if mu is None:
        mu, sd = np.nanmean(W, 0), np.nanstd(W, 0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (W - mu) / sd
    return np.nan_to_num(Z, nan=0.0), mu, sd


def place(Zw, Fw):
    """Coordinates of each stock on the frozen map: sensitivity of its
    (standardized) returns to the (standardized) pattern returns in this window."""
    Fs = (Fw - Fw.mean(0)) / Fw.std(0)
    return np.linalg.lstsq(Fs, Zw, rcond=None)[0].T


def gmm_bic(X, seed=0):
    best = None
    for k in KS:
        g = GaussianMixture(k, covariance_type="full", random_state=seed, max_iter=200).fit(X)
        b = g.bic(X)
        if best is None or b < best[0]:
            best = (b, g, k)
    g = best[1]
    return g.predict(X), g.means_, best[2]


def hdbscan_centers(X, min_size):
    lab = HDBSCAN(min_cluster_size=min_size, copy=True).fit(X).labels_
    cl = [c for c in np.unique(lab) if c >= 0]
    if len(cl) < 2:
        return None, None, len(cl)
    C = np.array([X[lab == c].mean(0) for c in cl])
    # stocks HDBSCAN calls "noise" are measured from their nearest group center
    lab2 = np.array([cl.index(l) if l >= 0 else -1 for l in lab])
    noise = lab2 < 0
    if noise.any():
        lab2[noise] = np.argmin(((X[noise, None, :] - C[None]) ** 2).sum(-1), 1)
    return lab2, C, len(cl)


def month_signals(R, mkt, me, snap, mi, hold_idx):
    t, F = me[mi], me[mi - 3]
    nB = t - F
    price_F = snap["price"][mi - 3]
    fit = R[F - 251:F + 1]
    cand = np.where((price_F >= 5) & (np.isnan(fit).sum(0) <= 12) & ~np.isnan(snap["price"][mi]))[0]
    WB = R[F + 1:t + 1, cand]
    cand = cand[(~np.isnan(WB)).sum(0) >= nB - 8]
    U = cand

    Zfit, mu, sd = standardize(R[F - 251:F + 1, U].astype(np.float64))
    _, S, Vt = np.linalg.svd(Zfit, full_matrices=False)
    V = Vt[:4].T
    share = S[:4] ** 2 / (S ** 2).sum()

    WA = R[F - 62:F + 1, U].astype(np.float64)
    WBm = R[F + 1:t + 1, U].astype(np.float64)
    fA = standardize(WA, mu, sd)[0] @ V            # frozen pattern returns, window A
    fB = standardize(WBm, mu, sd)[0] @ V           # frozen pattern returns, window B
    PA = place(standardize(WA)[0], fA)             # N x 4 positions
    PB = place(standardize(WBm)[0], fB)

    out = {"stock": U, "mi": np.full(len(U), mi)}
    meta = {"mi": mi, "n": len(U), "share": share}
    for name, ax in CONFIGS.items():
        A, B = PA[:, ax], PB[:, ax]
        scale = A.std(0)
        A, B = A / scale, B / scale
        lab, C, k = gmm_bic(A)
        dA = np.linalg.norm(A - C[lab], axis=1)
        dB = np.linalg.norm(B - C[lab], axis=1)
        out[f"drift_gmm_{name}"] = dB - dA
        out[f"dA_gmm_{name}"] = dA
        meta[f"k_{name}"] = k
        if name == "PC2-3":
            meta["stab_axis"] = [np.corrcoef(PA[:, j], PB[:, j])[0, 1] for j in range(4)]
            meta["labels_PC2-3"] = lab
            lab_h, C_h, nh = hdbscan_centers(A, max(25, len(U) // 100))
            meta["k_hdb"] = nh
            if lab_h is None:
                out["drift_hdb_PC2-3"] = np.full(len(U), np.nan)
            else:
                out["drift_hdb_PC2-3"] = (np.linalg.norm(B - C_h[lab_h], axis=1)
                                          - np.linalg.norm(A - C_h[lab_h], axis=1))

    # forward returns (start t+2, never into the holdout)
    s = t + 2
    for hn, h in HORIZONS.items():
        e = s + h
        if e > hold_idx:
            out[f"fwd_{hn}"] = np.full(len(U), np.nan)
            out[f"fwdadj_{hn}"] = np.full(len(U), np.nan)
            continue
        r = np.prod(1 + np.nan_to_num(R[s:e, U].astype(np.float64)), 0) - 1
        m = np.prod(1 + mkt[s:e]) - 1
        out[f"fwd_{hn}"] = r
        out[f"fwdadj_{hn}"] = r - snap["predicted_beta"][mi][U] * m

    # known effects, for the "disguise" check
    out["rev_1m"] = np.prod(1 + np.nan_to_num(R[t - 20:t + 1, U].astype(np.float64)), 0) - 1
    out["mom_12_1"] = np.prod(1 + np.nan_to_num(R[t - 251:t - 20, U].astype(np.float64)), 0) - 1
    out["vol_1y"] = np.nanstd(R[t - 251:t + 1, U].astype(np.float64), 0)
    out["beta"] = snap["predicted_beta"][mi][U].astype(float)
    out["srisk"] = snap["specific_risk"][mi][U].astype(float)
    return out, meta
