"""
SmartInnings FastAPI backend.

Serves the three trained models (chase win-probability, 1st-innings score
projection, pre-match winner) plus metadata for the React frontend.

Run:
    cd backend
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations
import os, sys, threading, datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# make the ml package importable
ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
sys.path.insert(0, ML_DIR)
from inference import SmartInnings  # noqa: E402
import update_data  # noqa: E402

app = FastAPI(title="SmartInnings API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SI = SmartInnings()

# --------------------------------------------------------------------------- #
#  Background data-refresh state
# --------------------------------------------------------------------------- #
REFRESH = {
    "running": False,
    "step": "idle",       # idle | downloading | training | reloading | done | error
    "message": "Ready.",
    "error": None,
    "last_updated": None,
    "matches": None,
    "deliveries": None,
}
_refresh_lock = threading.Lock()


def _refresh_worker():
    global SI
    try:
        REFRESH.update(step="downloading", message="Downloading the latest IPL data…", error=None)
        summary = update_data.download_latest()
        REFRESH.update(matches=summary["matches"], deliveries=summary["deliveries"])

        REFRESH.update(step="training",
                       message=f"Cleaning {summary['matches']} matches and retraining models…")
        import train  # heavy import kept local to startup speed
        train.main()

        REFRESH.update(step="reloading", message="Loading refreshed models…")
        SI = SmartInnings()

        REFRESH.update(step="done",
                       message=f"Updated · {summary['matches']} matches, "
                               f"{summary['deliveries']:,} deliveries.",
                       last_updated=datetime.datetime.now().isoformat(timespec="seconds"))
    except Exception as e:  # noqa: BLE001
        REFRESH.update(step="error", error=str(e), message=f"Update failed: {e}")
    finally:
        REFRESH["running"] = False


# --------------------------------------------------------------------------- #
#  Schemas
# --------------------------------------------------------------------------- #
class ChaseReq(BaseModel):
    batting_team: str
    bowling_team: str
    venue: str
    target: int = Field(ge=1, le=350)
    score: int = Field(ge=0, le=350)
    wickets: int = Field(ge=0, le=9)
    overs: int = Field(ge=0, le=19)
    balls: int = Field(ge=0, le=5)


class ScoreReq(BaseModel):
    batting_team: str
    bowling_team: str
    venue: str
    score: int = Field(ge=0, le=350)
    wickets: int = Field(ge=0, le=9)
    overs: int = Field(ge=0, le=19)
    balls: int = Field(ge=0, le=5)


class PreMatchReq(BaseModel):
    team_a: str
    team_b: str
    venue: str
    toss_winner: str
    toss_decision: str  # "bat" | "field"


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.get("/api/meta")
def meta():
    return {
        "teams": SI.spec["current_teams"],
        "all_teams": SI.team_vocab,
        "venues": SI.spec["venues"],
        "seasons": SI.spec["seasons"],
        "models": {
            "chase": {k: SI.chase[k] for k in ("test_acc", "test_auc", "test_logloss") if k in SI.chase},
            "score": {k: SI.score[k] for k in ("test_rmse", "test_mae") if k in SI.score},
            "prematch": {k: SI.prematch[k] for k in ("test_acc", "test_auc") if k in SI.prematch},
        },
        "calibration": SI.chase.get("calibration"),
    }


@app.post("/api/predict/chase")
def predict_chase(r: ChaseReq):
    if r.batting_team == r.bowling_team:
        raise HTTPException(400, "Batting and bowling teams must differ")
    return SI.predict_chase(r.batting_team, r.bowling_team, r.venue, r.target,
                            r.score, r.wickets, r.overs, r.balls)


@app.post("/api/predict/chase/sweep")
def predict_chase_sweep(r: ChaseReq):
    """Win-probability sensitivity: how batting win% changes as wickets in hand
    vary, holding the rest of the situation fixed. Powers the live chart."""
    if r.batting_team == r.bowling_team:
        raise HTTPException(400, "Batting and bowling teams must differ")
    return SI.predict_chase_sweep(r.batting_team, r.bowling_team, r.venue, r.target,
                                  r.score, r.wickets, r.overs, r.balls)


@app.post("/api/predict/score")
def predict_score(r: ScoreReq):
    if r.batting_team == r.bowling_team:
        raise HTTPException(400, "Batting and bowling teams must differ")
    return SI.predict_score(r.batting_team, r.bowling_team, r.venue,
                            r.score, r.wickets, r.overs, r.balls)


@app.post("/api/predict/prematch")
def predict_prematch(r: PreMatchReq):
    if r.team_a == r.team_b:
        raise HTTPException(400, "Teams must differ")
    return SI.predict_prematch(r.team_a, r.team_b, r.venue, r.toss_winner, r.toss_decision)


@app.post("/api/refresh")
def refresh():
    """Kick off a download → clean → retrain → reload cycle in the background."""
    with _refresh_lock:
        if REFRESH["running"]:
            return {"started": False, **REFRESH}
        REFRESH.update(running=True, step="downloading", error=None,
                       message="Starting update…")
        threading.Thread(target=_refresh_worker, daemon=True).start()
    return {"started": True, **REFRESH}


@app.get("/api/refresh/status")
def refresh_status():
    return REFRESH


@app.get("/")
def root():
    return {"status": "ok", "service": "SmartInnings API",
            "docs": "/docs", "endpoints": ["/api/meta", "/api/predict/chase",
                                            "/api/predict/score", "/api/predict/prematch"]}
