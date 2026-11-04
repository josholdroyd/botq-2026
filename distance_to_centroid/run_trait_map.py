"""Build the monthly drift signals, forward returns, and controls for the trait-map test."""
import time, pickle, warnings
import numpy as np, pandas as pd
from config import DATA_PATH, CACHE_PATH, HOLDOUT_START
from data_utils import load_panel, build_arrays, market_proxy
from trait_map_engine import month_signals
warnings.filterwarnings("ignore")

t0 = time.time()
df = load_panel(DATA_PATH, CACHE_PATH)
data = build_arrays(df); del df
R, dates, me, snap, ids = data["R"], data["dates"], data["me_days"], data["snap"], data["ids"]
mkt = market_proxy(R)
hold = np.datetime64(HOLDOUT_START)
hold_idx = int(np.searchsorted(dates, hold))
rows, metas = [], []
for mi in range(len(me)):
    if mi < 3 or me[mi - 3] < 252:
        continue
    if dates[me[mi]] >= hold:
        break
    out, meta = month_signals(R, mkt, me, snap, mi, hold_idx)
    rows.append(pd.DataFrame(out)); metas.append(meta)
    if mi % 24 == 0:
        print(dates[me[mi]], f"{time.time() - t0:.0f}s", flush=True)
panel = pd.concat(rows, ignore_index=True)
panel["date"] = dates[me[panel["mi"].values]]
panel["barrid"] = ids[panel["stock"].values]
panel.to_parquet("trait_map_panel.parquet")
pickle.dump(metas, open("trait_map_meta.pkl", "wb"))
print("done:", len(panel), "stock-months,", len(metas), "months,", f"{time.time() - t0:.0f}s")
