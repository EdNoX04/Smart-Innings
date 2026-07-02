"""
datastore.py — the single storage abstraction for SmartInnings.

Goal: future-proof storage that works both locally (zero setup) and in the
cloud (durable database), with identical code paths.

  - If the DATABASE_URL environment variable is set  -> use Postgres
    (tables: matches, deliveries, artifacts). This is the durable store that
    grows over the years and is written by the GitHub Action / ingest step.
  - Otherwise                                         -> use local CSV files in
    ../data (Match_Info.csv, Ball_By_Ball_Match_Data.csv). Great for local dev
    and for the very first run.

DB libraries (sqlalchemy / psycopg) are imported lazily, so local usage needs
nothing beyond pandas + numpy.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MATCH_CSV = os.path.join(DATA, "Match_Info.csv")
BALL_CSV = os.path.join(DATA, "Ball_By_Ball_Match_Data.csv")


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url and url.startswith("postgres://"):
        # SQLAlchemy wants the postgresql+psycopg2 scheme
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def _engine():
    """Create a SQLAlchemy engine (lazy import). Returns None if no DATABASE_URL."""
    url = database_url()
    if not url:
        return None
    from sqlalchemy import create_engine  # lazy
    # sslmode=require is what Neon/Supabase expect
    connect_args = {}
    if "sslmode" not in url:
        connect_args = {"sslmode": "require"}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def using_db() -> bool:
    return database_url() is not None


# --------------------------------------------------------------------------- #
#  Reads
# --------------------------------------------------------------------------- #
def load_matches() -> pd.DataFrame:
    eng = _engine()
    if eng is not None:
        return pd.read_sql("SELECT * FROM matches", eng)
    return pd.read_csv(MATCH_CSV)


def load_deliveries() -> pd.DataFrame:
    eng = _engine()
    if eng is not None:
        return pd.read_sql("SELECT * FROM deliveries", eng)
    return pd.read_csv(BALL_CSV)


# --------------------------------------------------------------------------- #
#  Writes (used by ingest)
# --------------------------------------------------------------------------- #
def save_matches(df: pd.DataFrame) -> int:
    eng = _engine()
    if eng is not None:
        df.to_sql("matches", eng, if_exists="replace", index=False, chunksize=5000)
    else:
        os.makedirs(DATA, exist_ok=True)
        df.to_csv(MATCH_CSV, index=False)
    return len(df)


def save_deliveries(df: pd.DataFrame) -> int:
    eng = _engine()
    if eng is not None:
        df.to_sql("deliveries", eng, if_exists="replace", index=False, chunksize=10000)
    else:
        os.makedirs(DATA, exist_ok=True)
        df.to_csv(BALL_CSV, index=False)
    return len(df)


# --------------------------------------------------------------------------- #
#  Model artifacts (optional DB persistence)
# --------------------------------------------------------------------------- #
def save_artifact(name: str, obj: dict) -> None:
    """Persist a trained-model JSON. Always writes to ml/artifacts; also to the
    'artifacts' DB table when a database is configured."""
    art_dir = os.path.join(HERE, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    with open(os.path.join(art_dir, name), "w") as f:
        json.dump(obj, f)
    eng = _engine()
    if eng is not None:
        from sqlalchemy import text
        payload = json.dumps(obj)
        with eng.begin() as cx:
            cx.execute(text(
                "CREATE TABLE IF NOT EXISTS artifacts "
                "(name TEXT PRIMARY KEY, body TEXT, updated_at TIMESTAMPTZ DEFAULT now())"))
            cx.execute(text(
                "INSERT INTO artifacts (name, body, updated_at) VALUES (:n, :b, now()) "
                "ON CONFLICT (name) DO UPDATE SET body = :b, updated_at = now()"),
                {"n": name, "b": payload})


def load_artifact(name: str) -> dict | None:
    """Load an artifact, preferring the local file (fast, bundled on Vercel),
    falling back to the DB."""
    path = os.path.join(HERE, "artifacts", name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    eng = _engine()
    if eng is not None:
        from sqlalchemy import text
        with eng.connect() as cx:
            row = cx.execute(text("SELECT body FROM artifacts WHERE name = :n"),
                             {"n": name}).fetchone()
            if row:
                return json.loads(row[0])
    return None


if __name__ == "__main__":
    print("DATABASE_URL set:", using_db())
    m = load_matches()
    print("matches:", len(m), "| deliveries:", len(load_deliveries()))
