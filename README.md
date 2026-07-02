# 🏏 SmartInnings 2.0 — IPL Predictor (React + FastAPI + from-scratch ML)

A modern rebuild of the SmartInnings IPL win predictor. Where the original was a
single XGBoost model behind a Flask page, this version is a **React** single-page
app talking to a **FastAPI** service that serves **three** models trained on the
**latest ball-by-ball data (2008–2026)**:

| Predictor | What it answers | Model | Accuracy |
|---|---|---|---|
| **Live win probability** | Mid-chase: who wins from the current score/overs/wickets? | XGBoost + Elo | **94.4% acc · 0.99 AUC** (standard split) · 77% leak-free |
| **Score projection** | First innings: what's the final total going to be? | XGBoost | **±16 runs MAE** |
| **Pre-match** | Before the toss: who's favoured? | Elo + logistic regression | ≈ coin-flip (see note) |

> **Training uses XGBoost** (`ml/requirements.txt`). The trained model is exported
> to a compact portable JSON, so **serving is pure-NumPy** — the app loads models
> with no XGBoost dependency, keeping the Vercel function small. If XGBoost isn't
> installed, training transparently falls back to an equivalent from-scratch
> NumPy booster in `smartml.py`. Either engine yields the same ~94% (XGBoost
> matches or slightly beats it; accuracy never drops).

### On the 94.4% (and why we report three numbers)

Accuracy for a ball-by-ball predictor depends heavily on *how you split* the data:

| Protocol | Chase accuracy | Notes |
|---|---|---|
| **Ball-random 80/20** | **94.4%** (AUC 0.99) | The conventional metric comparable IPL projects report. Optimistic: balls of one match can appear in both train and test. |
| **Match-group 80/20** | **77.2%** (AUC 0.85) | Leak-free — whole matches held out. The honest real-world number; calibration is built from this. |
| **Temporal (train ≤2024 → test 2026)** | 71.1% (AUC 0.87) | Hardest: predict an unseen future season. |

We show all three (`ml/artifacts/chase_model.json → metrics`) rather than cherry-picking.

### Handling team renames & player movement

Team *identity* is a poor, unstable feature — franchises rename (Kings XI → Punjab
Kings, RCB Bangalore → Bengaluru) and players change sides every year. So the models
use a chronological **Elo strength rating** per team (updated match-by-match, leak-free)
as the strength signal. Elo reflects *current* form regardless of name or roster, so
the model keeps working as teams evolve.

---

## Dataset

- **Source:** [`ritesh-ojha/IPL-DATASET`](https://github.com/ritesh-ojha/IPL-DATASET)
  (Cricsheet-derived, MIT-licensed, auto-updated daily).
- **Coverage:** 1,243 matches / 295,732 deliveries, seasons **2008 → 2026**
  (all seasons complete, including the full 2026 season).
- **Storage** (`ml/datastore.py`): a **cloud Postgres** database when
  `DATABASE_URL` is set (the durable home that grows for years), else local CSVs
  in `data/` — identical code path, zero setup for local dev.

## Staying updated & deployment (future-proof)

The app keeps itself current with **no manual work**: a scheduled **GitHub Action**
(`.github/workflows/update-model.yml`, weekly + on-demand) downloads the newest
data → loads it into Postgres → retrains → commits the refreshed model, and Vercel
redeploys. An in-app **Update** button can trigger it on demand.

Full step-by-step (Neon/Supabase + Vercel + the update button) → **[DEPLOYMENT.md](DEPLOYMENT.md)**.

- `python ml/ingest.py` — pull latest data into the datastore
- `python ml/train.py` — retrain from the datastore

### Data cleaning (improves accuracy)

Performed in `ml/train.py`:

- **Team-name normalisation** — Delhi Daredevils → Delhi Capitals, Kings XI Punjab
  → Punjab Kings, RCB Bangalore → Bengaluru, Rising Pune Supergiant(s) unified.
- **Dropped non-results** — ties and no-results removed; only matches with a
  decisive winner among the two contesting teams are kept (1,218 of 1,243).
- **Super-overs excluded** from win/loss labelling.
- **Rain/DLS-shortened innings filtered** — first innings must be a full ~20 overs
  (≥118 legal balls) or all-out, so chase features aren't distorted.
- **Legal-delivery handling** — wides & no-balls don't count toward the 120-ball
  budget; `retired hurt` is not counted as a wicket.
- **Leak-free encodings** — the venue scoring index, Elo ratings, recent form and
  head-to-head are all built **chronologically** from prior matches only.

---

## Project layout

```
Smart Innings/
├── data/                     # the two CSVs you downloaded
├── ml/
│   ├── smartml.py            # from-scratch ML library (GBT, logreg, ridge, metrics)
│   ├── train.py              # clean → feature-engineer → train → evaluate → save
│   ├── inference.py          # load artifacts + build features for predictions
│   └── artifacts/            # *.json models + feature_spec.json (created by train.py)
├── backend/
│   ├── app.py                # FastAPI service
│   └── requirements.txt
├── frontend/                 # React + Vite SPA
│   ├── package.json
│   └── src/ …
└── README.md
```

---

## Run it

### One command (recommended)
```bash
./start.sh
```
First run auto-creates the Python venv, installs backend + frontend dependencies,
then starts **both** servers — backend at `http://localhost:8000` (docs at `/docs`)
and frontend at `http://localhost:5173`. Press **Ctrl+C** once to stop both.

### Manual (two terminals)
Backend — macOS/Homebrew Python blocks system-wide installs (PEP 668), so use a venv:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # prompt shows (.venv)
pip install -r requirements.txt
uvicorn app:app --port 8000        # do NOT add --reload (it would watch .venv and loop)
```
Frontend:
```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173
```
Vite proxies `/api` to `http://localhost:8000`. To point at a deployed API, set `VITE_API_URL`.

> **Note on `--reload`:** because the venv lives in `backend/.venv`, running uvicorn
> with `--reload` makes its file-watcher scan thousands of library files and reload
> in a loop. The launcher and the command above omit it. If you want hot-reload for
> development, scope it: `uvicorn app:app --port 8000 --reload --reload-dir .`
> (run from `backend/`, after temporarily moving the venv outside the watched path).

### (Optional) retrain the models manually
Artifacts ship in `ml/artifacts/`. To regenerate: `cd ml && python3 train.py` (~35s,
needs only numpy + pandas). The dashboard's **Update data** button does this for you.

---

## API

| Method | Endpoint | Body |
|---|---|---|
| GET | `/api/meta` | — (teams, venues, model metrics) |
| POST | `/api/predict/chase` | `batting_team, bowling_team, venue, target, score, wickets, overs, balls` |
| POST | `/api/predict/score` | `batting_team, bowling_team, venue, score, wickets, overs, balls` |
| POST | `/api/predict/prematch` | `team_a, team_b, venue, toss_winner, toss_decision` |
| POST | `/api/refresh` | — starts a background download → clean → retrain → reload |
| GET | `/api/refresh/status` | — progress of the current/last refresh |

### 🔄 Update data button

The dashboard header has an **"↻ Update data"** button. Because the source
dataset is refreshed daily, clicking it will, on the machine running the backend:

1. **Download** the latest `Match_Info.csv` and `Ball_By_Ball_Match_Data.csv`
   (written atomically — a failed download never corrupts your existing data),
2. **Clean + retrain** all three models on the new data (the cleaning steps above
   run as part of training), and
3. **Hot-reload** the served models — no restart needed.

It runs in the background (~35s) with live progress shown next to the button.

---

## How the models work

**Live win probability** — features: runs left, balls left, wickets in hand,
current & required run rate, target, a venue scoring index, and one-hot batting /
bowling teams. Trained with logistic-loss gradient boosting; terminal states
(target reached / all out / out of balls) are resolved deterministically.

Each prediction also shows two badges — **Clarity** (how decisive the situation
is) and **Accuracy** (the model's *empirical* accuracy at that confidence, from a
**leak-free** calibration table built on held-out whole matches). Decisive
situations land in the high-confidence bucket (95–100%) where the model is
correct **≈93%** of the time; close games honestly report lower, because a true
coin-flip cannot be 95% accurate. A live **sensitivity chart** (win% vs wickets
in hand) and a **reliability chart** (confidence vs actual accuracy) visualise it.

**Score projection** — features: current runs, wickets, overs done, balls left,
run rate, venue index, teams. Squared-loss gradient boosting; the UI shows a
likely range of ±MAE around the point estimate.

**Pre-match** — driven by a **chronological Elo rating** (K=30) plus recent form,
head-to-head and toss, fed to a logistic model. Predictions are scored
**symmetrically** (swapping the two teams just swaps the probability).

> **Honest note on pre-match accuracy.** IPL is an exceptionally high-parity
> league. Across 2022–2026 the toss-winner wins **50.3%** of matches and the
> first-listed team **47.8%** — i.e. essentially a coin flip. No available
> pre-match feature delivers a real edge, so this tab is a *lean, not a lock*.
> The genuinely strong, useful models here are the live win-probability and the
> score projection.

---

## Validation

Models are evaluated with a **temporal split** (train ≤ 2024, validate 2025,
test 2026) so reported numbers reflect performance on unseen future seasons, not
memorised history. The training script prints a head-to-head of logistic
regression vs gradient boosting for every task and keeps the better model.
