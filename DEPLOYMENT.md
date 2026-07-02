# 🚀 Deploying SmartInnings (GitHub + Vercel + Postgres) — future-proof setup

This makes the app **self-updating**: a scheduled GitHub Action pulls the newest
IPL data into a cloud database, retrains the models, and commits them back —
Vercel then redeploys automatically. It keeps working for years with no manual
steps, and an in-app button can trigger an update on demand.

```
          ┌─────────────┐   weekly / on-demand    ┌──────────────────────┐
          │ GitHub Action│ ───────────────────────▶│ 1. download latest    │
          │ (CI servers) │                          │ 2. load into Postgres │
          └─────────────┘                          │ 3. retrain models     │
                 │ commits refreshed model JSON     │ 4. commit artifacts   │
                 ▼                                   └──────────────────────┘
          ┌─────────────┐   auto-redeploy on push
          │   Vercel     │  ── React app + /api prediction functions
          └─────────────┘        (reads committed model, talks to nobody heavy)
                 ▲
                 │ predictions
             users' browsers
```

Why this shape? **Vercel can't retrain** — its functions are short-lived with a
read-only filesystem. So training lives in the GitHub Action (no time limit),
Postgres is the durable data home, and Vercel just serves the trained model.

---

## Step 1 — Create a Postgres database (free)

Any Postgres works. Recommended: **Neon** (serverless, generous free tier,
one-click Vercel integration).

1. Sign up at <https://neon.tech> → create a project.
2. Copy the connection string — it looks like:
   `postgresql://user:pass@ep-xxx.aws.neon.tech/neondb?sslmode=require`

(Supabase works identically — use its "Connection string / URI".)

## Step 2 — Push the repo to GitHub

```bash
git init && git add . && git commit -m "SmartInnings"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## Step 3 — Add the database secret to GitHub

Repo → **Settings → Secrets and variables → Actions → New repository secret**:

- Name: `DATABASE_URL`  Value: *(your Postgres URL from step 1)*

Then seed the database + generate the first model by running the workflow once:
**Actions → "Update data & retrain SmartInnings" → Run workflow**. (It also runs
automatically every Monday.)

> Prefer to seed locally? `DATABASE_URL="..." python ml/ingest.py` then
> `DATABASE_URL="..." python ml/train.py`.

## Step 4 — Deploy on Vercel

1. <https://vercel.com> → **Add New → Project** → import your GitHub repo.
2. Vercel reads `vercel.json` automatically (build = Vite, API = Python function).
   No manual build settings needed.
3. Deploy. Your app is live; predictions are served from the committed model.

## Step 5 — (Optional) enable the in-app "Update" button on Vercel

The button triggers the GitHub Action. Give Vercel a token:

1. GitHub → **Settings → Developer settings → Fine-grained tokens** → new token,
   scoped to your repo, permission **Actions: Read and write**.
2. Vercel → Project → **Settings → Environment Variables**:
   - `GITHUB_REPO`  = `owner/repo`
   - `GITHUB_TOKEN` = *(the fine-grained token)*
   - `GITHUB_BRANCH` = `main` (optional)
3. Redeploy. The header button now starts an update run and shows its progress.
   (Without these vars the app simply shows a "Auto-updates weekly" chip — the
   scheduled Action still runs.)

---

## What updates over time

- **Data**: `ml/ingest.py` reloads the newest matches, teams, and ball-by-ball
  deliveries into Postgres — so new seasons, renamed franchises, and new rosters
  all flow in automatically.
- **Model**: `ml/train.py` retrains on the refreshed data. Because team strength
  is modelled with a chronological **Elo rating** (not fixed team identity), the
  model keeps calibrating itself as teams and squads change.
- **Deploy**: the Action commits refreshed `ml/artifacts/*.json`; Vercel redeploys.

## Local development (no cloud needed)

```bash
./start.sh          # backend + frontend; falls back to local CSV data
```
`python ml/ingest.py` (no DATABASE_URL) just refreshes the local CSVs. Set
`DATABASE_URL` in your shell to point local runs at Postgres instead.
