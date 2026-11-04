"""Load the competition panel into a dense (days x stocks) return matrix."""
import os
import numpy as np
import polars as pl

COLS = ["barrid", "date", "price", "predicted_beta", "return", "specific_risk"]


def load_panel(raw_path, cache_path):
    """Read the panel (Parquet or CSV), keep the needed columns, and cache them.
    Handles dates stored either as real dates or as text."""
    if not os.path.exists(cache_path):
        try:
            lf = pl.scan_parquet(raw_path)
            lf.collect_schema()
        except Exception:
            lf = pl.scan_csv(raw_path, schema_overrides={"barrid": pl.Utf8, "cusip": pl.Utf8})
        lf = lf.select(COLS)
        if lf.collect_schema()["date"] == pl.Utf8:
            lf = lf.with_columns(pl.col("date").str.to_date())
        lf.sink_parquet(cache_path)
    df = pl.read_parquet(cache_path)
    if df["date"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("date").str.to_date())
    return df


def build_arrays(df):
    """Dense daily return matrix (decimals) plus month-end snapshots of
    price, Barra predicted beta, and Barra specific risk."""
    dates = np.sort(df["date"].unique().to_numpy())
    ids = np.sort(df["barrid"].unique().to_numpy())
    d_idx = np.searchsorted(dates, df["date"].to_numpy())
    i_idx = np.searchsorted(ids, df["barrid"].to_numpy())
    R = np.full((len(dates), len(ids)), np.nan, dtype=np.float32)
    R[d_idx, i_idx] = (df["return"].to_numpy() / 100.0).astype(np.float32)  # percent -> decimal

    months = dates.astype("datetime64[M]")
    is_me = np.r_[months[1:] != months[:-1], True]
    me_days = np.where(is_me)[0]
    me_pos = -np.ones(len(dates), dtype=np.int64)
    me_pos[me_days] = np.arange(len(me_days))
    snap = {}
    sel = me_pos[d_idx] >= 0
    rows, cols = me_pos[d_idx][sel], i_idx[sel]
    for c in ["price", "predicted_beta", "specific_risk"]:
        A = np.full((len(me_days), len(ids)), np.nan, dtype=np.float32)
        A[rows, cols] = df[c].to_numpy()[sel].astype(np.float32)
        snap[c] = A
    return dict(dates=dates, ids=ids, R=R, me_days=me_days, snap=snap)


def market_proxy(R):
    """Equal-weighted average return of the panel's stocks each day
    (the file has no market caps, so no cap-weighted index can be built)."""
    return np.nanmean(R, axis=1).astype(np.float64)


def tickers_on(raw_path, date_str):
    """barrid -> ticker on one date (for the sector check). Works for Parquet or
    CSV, with dates stored as dates or as text."""
    try:
        lf = pl.scan_parquet(raw_path); lf.collect_schema()
    except Exception:
        lf = pl.scan_csv(raw_path, schema_overrides={"barrid": pl.Utf8, "cusip": pl.Utf8})
    if lf.collect_schema()["date"] == pl.Utf8:
        lf = lf.with_columns(pl.col("date").str.to_date())
    return (lf.filter(pl.col("date") == pl.lit(date_str).str.to_date())
              .select("barrid", "ticker").collect().to_pandas())
