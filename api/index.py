"""
Vercel serverless entrypoint for SmartInnings.

Serves predictions from the committed model artifacts (fast, fits serverless
limits) and exposes an /api/refresh that TRIGGERS the GitHub Action (Vercel
cannot retrain itself). Local development uses backend/app.py instead.

Env vars (set in the Vercel dashboard) for the update button:
  GITHUB_REPO   = "owner/repo"
  GITHUB_TOKEN  = a fine-grained PAT with "Actions: write" on that repo
  GITHUB_BRANCH = "main"   (optional, defaults to main)
"""
from __future__ import annotations
import os, sys, json, urllib.request, urllib.error

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
from inference import SmartInnings  # noqa: E402

app = FastAPI(title="SmartInnings API (Vercel)", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

SI = SmartInnings()
WORKFLOW = "update-model.yml"


# ---------- schemas ----------
class ChaseReq(BaseModel):
    batting_team: str; bowling_team: str; venue: str
    target: int = Field(ge=1, le=350); score: int = Field(ge=0, le=350)
    wickets: int = Field(ge=0, le=9); overs: int = Field(ge=0, le=19); balls: int = Field(ge=0, le=5)


class ScoreReq(BaseModel):
    batting_team: str; bowling_team: str; venue: str
    score: int = Field(ge=0, le=350); wickets: int = Field(ge=0, le=9)
    overs: int = Field(ge=0, le=19); balls: int = Field(ge=0, le=5)


class PreMatchReq(BaseModel):
    team_a: str; team_b: str; venue: str; toss_winner: str; toss_decision: str


# ---------- predictions ----------
@app.get("/api/meta")
def meta():
    return {
        "teams": SI.spec["current_teams"], "all_teams": SI.team_vocab,
        "venues": SI.spec["venues"], "seasons": SI.spec["seasons"],
        "models": {
            "chase": {k: SI.chase[k] for k in ("test_acc", "test_auc", "metrics") if k in SI.chase},
            "score": {k: SI.score[k] for k in ("test_rmse", "test_mae") if k in SI.score},
            "prematch": {k: SI.prematch[k] for k in ("test_acc", "test_auc") if k in SI.prematch},
        },
        "calibration": SI.chase.get("calibration"),
        "update_enabled": bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO")),
    }


@app.post("/api/predict/chase")
def chase(r: ChaseReq):
    if r.batting_team == r.bowling_team:
        raise HTTPException(400, "Teams must differ")
    return SI.predict_chase(r.batting_team, r.bowling_team, r.venue, r.target, r.score, r.wickets, r.overs, r.balls)


@app.post("/api/predict/chase/sweep")
def chase_sweep(r: ChaseReq):
    if r.batting_team == r.bowling_team:
        raise HTTPException(400, "Teams must differ")
    return SI.predict_chase_sweep(r.batting_team, r.bowling_team, r.venue, r.target, r.score, r.wickets, r.overs, r.balls)


@app.post("/api/predict/score")
def score(r: ScoreReq):
    if r.batting_team == r.bowling_team:
        raise HTTPException(400, "Teams must differ")
    return SI.predict_score(r.batting_team, r.bowling_team, r.venue, r.score, r.wickets, r.overs, r.balls)


@app.post("/api/predict/prematch")
def prematch(r: PreMatchReq):
    if r.team_a == r.team_b:
        raise HTTPException(400, "Teams must differ")
    return SI.predict_prematch(r.team_a, r.team_b, r.venue, r.toss_winner, r.toss_decision)


# ---------- update: trigger the GitHub Action ----------
def _gh(method, path, body=None):
    token = os.environ["GITHUB_TOKEN"]; repo = os.environ["GITHUB_REPO"]
    url = f"https://api.github.com/repos/{repo}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "SmartInnings"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read()
        return resp.status, (json.loads(raw) if raw else {})


@app.post("/api/refresh")
def refresh():
    if not (os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO")):
        raise HTTPException(400, "Update not configured. Set GITHUB_REPO and GITHUB_TOKEN "
                                 "env vars in Vercel to enable the update button.")
    branch = os.environ.get("GITHUB_BRANCH", "main")
    try:
        _gh("POST", f"/actions/workflows/{WORKFLOW}/dispatches", {"ref": branch})
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"GitHub dispatch failed: {e.read().decode()[:200]}")
    return {"started": True, "running": True, "step": "dispatched",
            "message": "Update started on GitHub — data refresh + retrain running…"}


@app.get("/api/refresh/status")
def refresh_status():
    if not (os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO")):
        return {"running": False, "step": "idle", "message": "Update not configured."}
    try:
        _, data = _gh("GET", f"/actions/workflows/{WORKFLOW}/runs?per_page=1")
    except urllib.error.HTTPError as e:
        return {"running": False, "step": "error", "message": f"GitHub error: {e.code}"}
    runs = data.get("workflow_runs", [])
    if not runs:
        return {"running": False, "step": "idle", "message": "No runs yet."}
    run = runs[0]
    status = run.get("status")          # queued | in_progress | completed
    concl = run.get("conclusion")       # success | failure | ...
    running = status != "completed"
    if running:
        step, msg = "training", "Updating data & retraining on GitHub…"
    elif concl == "success":
        step, msg = "done", "Update complete — Vercel will redeploy the new model shortly."
    else:
        step, msg = "error", f"Update run {concl}."
    return {"running": running, "step": step, "message": msg,
            "last_updated": run.get("updated_at"), "url": run.get("html_url")}


@app.get("/")
def root():
    return {"status": "ok", "service": "SmartInnings API (Vercel)"}
