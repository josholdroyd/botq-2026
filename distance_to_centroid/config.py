"""Paths used by every script.

The data file is found automatically: the scripts look in this folder and the
folder above it, for the Parquet or CSV version. To use a file somewhere else,
set the environment variable BOTQ_DATA to its full path.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))  # this folder, wherever the terminal is


def _find_data():
    if os.environ.get("BOTQ_DATA"):
        return os.environ["BOTQ_DATA"]
    names = ["BotQ_2026_data.parquet", "BotQ_2026_data.csv"]
    places = [HERE, os.path.dirname(HERE)]
    for place in places:
        for name in names:
            path = os.path.join(place, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        "Couldn't find BotQ_2026_data.parquet (or .csv). Looked in:\n  "
        + "\n  ".join(places)
        + "\nPut the data file in one of those folders, or set BOTQ_DATA to its full path.")


DATA_PATH = _find_data()
if os.path.getsize(DATA_PATH) < 1_000_000:
    raise ValueError(
        f"{DATA_PATH} is only {os.path.getsize(DATA_PATH):,} bytes, so it's a Git LFS placeholder, "
        "not the data. GitHub's 'Download ZIP' doesn't include the real file. Get the real "
        "BotQ_2026_data.parquet (about 0.5 GB) from a teammate or with 'git lfs pull'.")
# A smaller cache written on the first run (only the columns these tests use).
CACHE_PATH = os.environ.get("BOTQ_CACHE", os.path.join(HERE, "botq_panel_compact.parquet"))
# Everything from this date on is the sealed holdout and is never used.
HOLDOUT_START = "2022-11-01"