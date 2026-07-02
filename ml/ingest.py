"""
ingest.py — refresh the datastore with the latest IPL data.

Steps:
  1. download the newest CSVs from the source repo (update_data)
  2. load them into the datastore (Postgres if DATABASE_URL is set, else CSV)

Run locally:      python ingest.py
Run in CI:        DATABASE_URL=... python ingest.py   (loads into Postgres)
"""
from __future__ import annotations
import pandas as pd

import update_data
import datastore


def main():
    print("→ downloading latest source data…")
    summary = update_data.download_latest()   # writes fresh CSVs into ../data
    print(f"  downloaded {summary['matches']} matches, {summary['deliveries']} deliveries")

    matches = pd.read_csv(datastore.MATCH_CSV)
    deliveries = pd.read_csv(datastore.BALL_CSV)

    if datastore.using_db():
        print("→ loading into Postgres (DATABASE_URL detected)…")
    else:
        print("→ no DATABASE_URL — data kept as local CSV (fine for local dev)")
    n_m = datastore.save_matches(matches)
    n_d = datastore.save_deliveries(deliveries)
    print(f"✓ datastore updated: {n_m} matches, {n_d} deliveries")


if __name__ == "__main__":
    main()
