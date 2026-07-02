"""
update_data.py — download the latest IPL CSVs from the source repo.

The dataset (ritesh-ojha/IPL-DATASET, Cricsheet-derived) is auto-updated daily.
This module re-downloads the two CSVs into ../data, writing to a temp file first
and only replacing the existing file on success (so a failed download never
corrupts the current data). Uses only the Python standard library.
"""
from __future__ import annotations
import os
import shutil
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

BASE = "https://raw.githubusercontent.com/ritesh-ojha/IPL-DATASET/main/csv"
FILES = {
    "Match_Info.csv": f"{BASE}/Match_Info.csv",
    "Ball_By_Ball_Match_Data.csv": f"{BASE}/Ball_By_Ball_Match_Data.csv",
}


def _download_one(url: str, dest: str) -> int:
    """Stream a URL to dest atomically. Returns bytes written."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "SmartInnings/2.0"})
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    size = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                size += len(chunk)
        if size < 1024:  # sanity: real files are far larger
            raise ValueError(f"downloaded file suspiciously small ({size} bytes)")
        shutil.move(tmp_path, dest)
        return size
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def download_latest() -> dict:
    """Download both CSVs. Returns a small summary dict."""
    result = {}
    for name, url in FILES.items():
        dest = os.path.join(DATA, name)
        result[name] = _download_one(url, dest)
    # quick row counts (subtract header)
    def rows(p):
        with open(p, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    return {
        "matches": rows(os.path.join(DATA, "Match_Info.csv")),
        "deliveries": rows(os.path.join(DATA, "Ball_By_Ball_Match_Data.csv")),
        "bytes": result,
    }


if __name__ == "__main__":
    print(download_latest())
