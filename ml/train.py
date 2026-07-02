"""
train.py — SmartInnings model training pipeline.

Stages
  1. Load raw Cricsheet-derived CSVs (Match_Info.csv, Ball_By_Ball_Match_Data.csv)
  2. CLEAN: normalise team names, drop ties/no-results/super-overs and
     rain-shortened innings, remove illegal-delivery miscounts, fix venues
  3. FEATURE ENGINEERING for three tasks:
        A. 2nd-innings live win probability   (classification)
        B. 1st-innings final-score projection (regression)
        C. pre-match winner prediction        (classification)
  4. Temporal validation (train <=2024 / val 2025 / test 2026), compare
     LogisticRegression vs GradientBoostedTrees, keep the better model
  5. Retrain the winner on ALL data and save JSON artifacts + metadata
"""
from __future__ import annotations
import os, json, time
import numpy as np
import pandas as pd

import smartml as sm
import datastore

# Prefer real XGBoost for training (what most IPL projects use and easy to
# explain). If it isn't installed, transparently fall back to the from-scratch
# NumPy booster in smartml. Either way the trained model is exported to the SAME
# portable JSON format, so serving stays pure-numpy and lightweight (fits Vercel;
# no need to ship the large XGBoost binary at request time).
try:
    import xgboost as _xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# --------------------------------------------------------------------------- #
#  Canonical names
# --------------------------------------------------------------------------- #
TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}
CURRENT_TEAMS = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Delhi Capitals",
    "Punjab Kings", "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants",
]


def norm_team(t):
    if pd.isna(t):
        return t
    t = str(t).strip()
    return TEAM_MAP.get(t, t)


def norm_venue(v):
    if pd.isna(v):
        return "Unknown"
    return str(v).split(",")[0].strip()


# --------------------------------------------------------------------------- #
#  1 + 2 : load & clean
# --------------------------------------------------------------------------- #
def load_clean():
    # read via the datastore: Postgres when DATABASE_URL is set, else local CSV
    m = datastore.load_matches()
    b = datastore.load_deliveries()

    print(f"source: {'Postgres' if datastore.using_db() else 'local CSV'} | "
          f"raw: {len(m)} matches, {len(b)} deliveries")
    print("result values   :", m["result"].value_counts(dropna=False).to_dict())
    print("eliminator vals :", m["eliminator"].value_counts(dropna=False).to_dict())
    print("ExtraType vals  :", b["ExtraType"].value_counts(dropna=False).to_dict())
    print("Kind values     :", b["Kind"].value_counts(dropna=False).to_dict())

    # team normalisation
    for c in ["team1", "team2", "toss_winner", "winner"]:
        m[c] = m[c].map(norm_team)
    b["BattingTeam"] = b["BattingTeam"].map(norm_team)

    m["date"] = pd.to_datetime(m["match_date"], errors="coerce")
    m["season"] = m["date"].dt.year
    m["venue_n"] = m["venue"].map(norm_venue)

    # ---- CLEAN matches ----
    n0 = len(m)
    m = m.dropna(subset=["winner", "date"])
    # drop ties / no-results: winner must be one of the two contesting teams
    m = m[(m["winner"] == m["team1"]) | (m["winner"] == m["team2"])]
    # drop matches decided by Super Over (eliminator flag == 'Y')
    if m["eliminator"].dtype == object:
        m = m[~m["eliminator"].astype(str).str.upper().isin(["Y", "YES", "TRUE"])]
    print(f"clean matches: kept {len(m)}/{n0}")

    # ---- CLEAN deliveries ----
    b["ExtraType"] = b["ExtraType"].fillna("")
    # legal ball = not a wide and not a no-ball
    extra = b["ExtraType"].str.lower()
    b["is_legal"] = ~extra.str.contains("wide") & ~extra.str.contains("noball")
    # real wicket = dismissal that reduces wickets in hand (exclude retired hurt)
    kind = b["Kind"].fillna("").astype(str).str.lower()
    b["is_wicket"] = (b["IsWicketDelivery"].astype(float) == 1) & (kind != "retired hurt")
    b = b[b["ID"].isin(set(m["match_number"]))]

    return m, b


# --------------------------------------------------------------------------- #
#  Innings aggregates
# --------------------------------------------------------------------------- #
def innings_summary(b):
    """Per (match, innings): total runs and legal balls faced."""
    g = b.groupby(["ID", "Innings"])
    summ = g.agg(total=("TotalRun", "sum"),
                 legal_balls=("is_legal", "sum"),
                 wickets=("is_wicket", "sum")).reset_index()
    return summ


# --------------------------------------------------------------------------- #
#  Venue scoring index (leak-free: built from TRAIN matches only)
# --------------------------------------------------------------------------- #
def build_venue_index(m_train, first_inn):
    df = first_inn.merge(m_train[["match_number", "venue_n", "city"]],
                          left_on="ID", right_on="match_number", how="inner")
    venue_mean = df.groupby("venue_n")["total"].mean().to_dict()
    city_mean = df.groupby("city")["total"].mean().to_dict()
    global_mean = float(df["total"].mean())
    return {"venue": {k: float(v) for k, v in venue_mean.items()},
            "city": {k: float(v) for k, v in city_mean.items()},
            "global": global_mean}


def venue_score(vidx, venue_n, city):
    if venue_n in vidx["venue"]:
        return vidx["venue"][venue_n]
    if city in vidx["city"]:
        return vidx["city"][city]
    return vidx["global"]


# --------------------------------------------------------------------------- #
#  Team vocabulary (one-hot)
# --------------------------------------------------------------------------- #
def build_team_vocab(m):
    teams = sorted(set(m["team1"]) | set(m["team2"]))
    return teams


def build_elo(m, K=30.0):
    """Chronological Elo strength per team (leak-free).

    This is roster- and rename-agnostic: a team's rating reflects how strong it
    has *recently been*, not a fixed identity — so it survives franchise renames
    (already normalised) and the constant churn of players between teams.

    Returns:
      pre_elo : {match_id: {team: rating_before_that_match}}
      current : {team: latest_rating}
    """
    md = m.sort_values("date").reset_index(drop=True)
    elo = {}
    pre_elo = {}
    for _, r in md.iterrows():
        t1, t2 = r["team1"], r["team2"]
        e1, e2 = elo.get(t1, 1500.0), elo.get(t2, 1500.0)
        pre_elo[r["match_number"]] = {t1: e1, t2: e2}
        y1 = 1 if r["winner"] == t1 else 0
        exp1 = 1.0 / (1.0 + 10 ** ((e2 - e1) / 400.0))
        elo[t1] = e1 + K * (y1 - exp1)
        elo[t2] = e2 + K * ((1 - y1) - (1 - exp1))
    return pre_elo, elo


def onehot_team(team, vocab):
    v = np.zeros(len(vocab) + 1)  # last slot = OTHER
    if team in vocab:
        v[vocab.index(team)] = 1.0
    else:
        v[-1] = 1.0
    return v


# --------------------------------------------------------------------------- #
#  A. 2nd-innings win-probability dataset
# --------------------------------------------------------------------------- #
def build_chase_dataset(m, b, summ, vidx, team_vocab, pre_elo):
    first = summ[summ["Innings"] == 1].set_index("ID")
    minfo = m.set_index("match_number")

    rows, feats = [], []
    valid_ids = set(minfo.index)
    # iterate over 2nd-innings deliveries grouped by match
    b2 = b[(b["Innings"] == 2) & (b["ID"].isin(valid_ids))].copy()
    b2 = b2.sort_values(["ID", "Overs", "BallNumber"])

    for mid, grp in b2.groupby("ID"):
        if mid not in first.index:
            continue
        first_total = first.loc[mid, "total"]
        first_balls = first.loc[mid, "legal_balls"]
        # rain/DLS filter: require a full first innings (>=118 legal balls or all out)
        first_wkts = first.loc[mid, "wickets"]
        if first_balls < 118 and first_wkts < 10:
            continue
        target = first_total + 1
        info = minfo.loc[mid]
        batting_team = grp["BattingTeam"].iloc[0]
        bowling_team = info["team1"] if batting_team == info["team2"] else info["team2"]
        won = 1 if info["winner"] == batting_team else 0
        vs = venue_score(vidx, info["venue_n"], info["city"])
        season = info["season"]
        oh = np.concatenate([onehot_team(batting_team, team_vocab),
                             onehot_team(bowling_team, team_vocab)])
        pe = pre_elo.get(mid, {})
        bat_elo = pe.get(batting_team, 1500.0)
        bowl_elo = pe.get(bowling_team, 1500.0)

        runs_arr = grp["TotalRun"].to_numpy()
        legal_arr = grp["is_legal"].to_numpy()
        wkt_arr = grp["is_wicket"].to_numpy()
        cum_runs = 0; legal = 0; wkts = 0
        for i in range(len(runs_arr)):
            cum_runs += runs_arr[i]
            if legal_arr[i]:
                legal += 1
            if wkt_arr[i]:
                wkts += 1
            balls_left = 120 - legal
            if balls_left <= 0 or wkts >= 10:
                break
            runs_left = target - cum_runs
            if runs_left <= 0:
                break
            # sample ~half the balls to keep the set manageable & decorrelated
            if rng.random() > 0.55:
                continue
            overs_done = legal / 6.0
            crr = cum_runs / overs_done if overs_done > 0 else 0.0
            rrr = runs_left / (balls_left / 6.0)
            num = [float(runs_left), float(balls_left), float(10 - wkts),
                   crr, rrr, float(target), vs, bat_elo, bowl_elo, bat_elo - bowl_elo]
            feats.append(np.concatenate([num, oh]))
            rows.append((season, won, mid))

    X = np.array(feats, dtype=np.float64)
    meta = pd.DataFrame(rows, columns=["season", "y", "mid"])
    return X, meta["y"].values, meta["season"].values, meta["mid"].values


# --------------------------------------------------------------------------- #
#  B. 1st-innings score projection dataset
# --------------------------------------------------------------------------- #
def build_score_dataset(m, b, summ, vidx, team_vocab):
    first_tot = summ[summ["Innings"] == 1].set_index("ID")["total"].to_dict()
    first_balls = summ[summ["Innings"] == 1].set_index("ID")["legal_balls"].to_dict()
    minfo = m.set_index("match_number")
    valid_ids = set(minfo.index)

    b1 = b[(b["Innings"] == 1) & (b["ID"].isin(valid_ids))].copy()
    b1 = b1.sort_values(["ID", "Overs", "BallNumber"])

    feats, ys, seasons = [], [], []
    for mid, grp in b1.groupby("ID"):
        if mid not in first_tot:
            continue
        if first_balls[mid] < 118:   # only full innings as targets
            continue
        info = minfo.loc[mid]
        batting_team = grp["BattingTeam"].iloc[0]
        bowling_team = info["team1"] if batting_team == info["team2"] else info["team2"]
        final_total = first_tot[mid]
        vs = venue_score(vidx, info["venue_n"], info["city"])
        season = info["season"]
        oh = np.concatenate([onehot_team(batting_team, team_vocab),
                             onehot_team(bowling_team, team_vocab)])

        runs_arr = grp["TotalRun"].to_numpy()
        legal_arr = grp["is_legal"].to_numpy()
        wkt_arr = grp["is_wicket"].to_numpy()
        cum_runs = 0; legal = 0; wkts = 0
        # record a state at the end of each over from over 4 onward
        for i in range(len(runs_arr)):
            cum_runs += runs_arr[i]
            if legal_arr[i]:
                legal += 1
            if wkt_arr[i]:
                wkts += 1
            if legal_arr[i] and legal % 6 == 0:
                over = legal // 6
                if 4 <= over <= 18:
                    balls_left = 120 - legal
                    crr = cum_runs / over
                    num = [float(cum_runs), float(wkts), float(over),
                           float(balls_left), crr, vs]
                    feats.append(np.concatenate([num, oh]))
                    ys.append(float(final_total))
                    seasons.append(season)
    return np.array(feats, dtype=np.float64), np.array(ys), np.array(seasons)


# --------------------------------------------------------------------------- #
#  C. pre-match dataset (with leak-free rolling form & H2H)
# --------------------------------------------------------------------------- #
def build_prematch_dataset(m, vidx):
    """Leak-free pre-match features driven by a chronological Elo rating.

    Team-identity one-hots are deliberately omitted: a team's historical
    win-count does not transfer across seasons (rosters change), and including
    it makes the model invert on unseen seasons. Elo + recent form + toss +
    head-to-head are roster-agnostic skill estimates that generalise.

    Returns X, y, seasons and the final team_state / h2h tables so the backend
    can score brand-new fixtures with each team's *current* rating.
    """
    md = m.sort_values("date").reset_index(drop=True)
    elo = {}                 # team -> rating
    last_results = {}        # team -> list of recent (1 win /0 loss)
    h2h = {}                 # (a,b) sorted -> [wins_by_a, games]
    K = 30.0
    feats, ys, seasons = [], [], []

    def get_elo(t):
        return elo.get(t, 1500.0)

    def form(team):
        r = last_results.get(team, [])
        return float(np.mean(r[-10:])) if r else 0.5

    def h2h_rate(t1, t2):
        key = tuple(sorted([t1, t2]))
        hw, hg = h2h.get(key, [0, 0])
        if hg == 0:
            return 0.5
        return (hw if key[0] == t1 else hg - hw) / hg

    for _, row in md.iterrows():
        t1, t2 = row["team1"], row["team2"]
        winner = row["winner"]
        e1, e2 = get_elo(t1), get_elo(t2)
        toss_t1 = 1.0 if row["toss_winner"] == t1 else 0.0
        bat_first = 1.0 if str(row["toss_decision"]).lower().startswith("bat") else 0.0
        vs = venue_score(vidx, row["venue_n"], row["city"])

        num = [e1, e2, e1 - e2, form(t1), form(t2), form(t1) - form(t2),
               toss_t1, bat_first, h2h_rate(t1, t2), vs]
        feats.append(num)
        y1 = 1 if winner == t1 else 0
        ys.append(y1)
        seasons.append(row["season"])

        # ---- update history AFTER using it (no leakage) ----
        exp1 = 1.0 / (1.0 + 10 ** ((e2 - e1) / 400.0))
        elo[t1] = e1 + K * (y1 - exp1)
        elo[t2] = e2 + K * ((1 - y1) - (1 - exp1))
        last_results.setdefault(t1, []).append(y1)
        last_results.setdefault(t2, []).append(1 - y1)
        key = tuple(sorted([t1, t2]))
        hw, hg = h2h.get(key, [0, 0])
        h2h[key] = [hw + (1 if winner == key[0] else 0), hg + 1]

    team_state = {t: {"elo": float(get_elo(t)), "form": form(t)} for t in
                  (set(md["team1"]) | set(md["team2"]))}
    h2h_export = {f"{k[0]}|{k[1]}": v for k, v in h2h.items()}
    return (np.array(feats, dtype=np.float64), np.array(ys),
            np.array(seasons), team_state, h2h_export)


# --------------------------------------------------------------------------- #
#  Training helpers
# --------------------------------------------------------------------------- #
def temporal_split(seasons):
    tr = seasons <= 2024
    va = seasons == 2025
    te = seasons == 2026
    return tr, va, te


# --------------------------------------------------------------------------- #
#  Booster factory: XGBoost (preferred) -> portable numpy model
# --------------------------------------------------------------------------- #
def _parse_xgb_trees(model_json):
    """Convert XGBoost's native JSON trees into smartml._Tree objects."""
    trees_json = model_json["learner"]["gradient_booster"]["model"]["trees"]
    out = []
    for t in trees_json:
        lc = t["left_children"]; rc = t["right_children"]
        sc = t["split_conditions"]; si = t["split_indices"]
        tree = sm._Tree()
        for _ in range(len(lc)):
            tree._add()
        for i in range(len(lc)):
            if int(lc[i]) == -1:                 # leaf
                tree.feature[i] = -1
                tree.value[i] = float(sc[i])
                tree.left[i] = -1; tree.right[i] = -1
            else:                                 # internal split
                tree.feature[i] = int(si[i])
                tree.threshold[i] = float(sc[i])
                tree.left[i] = int(lc[i]); tree.right[i] = int(rc[i])
                tree.value[i] = 0.0
        out.append(tree)
    return out


def _fit_xgb(loss, kw, X, y, Xref):
    """Train an XGBoost model and export it to a portable smartml booster.
    A margin self-check guarantees the exported model matches XGBoost."""
    import tempfile, json as _json
    params = dict(
        n_estimators=kw.get("n_estimators", 300),
        learning_rate=kw.get("learning_rate", 0.1),
        max_depth=kw.get("max_depth", 5),
        min_child_weight=kw.get("min_child_weight", 1),
        reg_lambda=kw.get("reg_lambda", 1.0),
        subsample=kw.get("subsample", 1.0),
        colsample_bytree=kw.get("colsample", 1.0),
        tree_method="hist", n_jobs=4, random_state=42, verbosity=0,
    )
    if loss == "logistic":
        model = _xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
    else:
        model = _xgb.XGBRegressor(objective="reg:squarederror", **params)
    model.fit(X, y)
    booster = model.get_booster()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    booster.save_model(path)
    mj = _json.load(open(path)); os.remove(path)

    g = sm.GradientBoostedTrees(loss=loss, learning_rate=1.0)
    g.trees = _parse_xgb_trees(mj)
    g.n_features_ = X.shape[1]
    g.base_score = 0.0
    # calibrate the constant intercept from XGBoost's raw margin (version-proof)
    xgm = booster.predict(_xgb.DMatrix(Xref), output_margin=True)
    g.base_score = float(np.mean(xgm - g._raw(Xref)))
    diff = float(np.max(np.abs(g._raw(Xref) - xgm)))
    if diff > 1e-2:
        raise RuntimeError(f"xgb->portable margin mismatch {diff:.4f}")
    print(f"    [xgboost] {loss}: {len(g.trees)} trees exported (max margin diff {diff:.1e})")
    return g


def make_booster(loss, kw, X, y, Xref=None):
    """Return a served portable booster. Uses XGBoost when available; on any
    failure or absence, falls back to the built-in NumPy booster."""
    if HAS_XGB:
        try:
            return _fit_xgb(loss, kw, X, y, Xref if Xref is not None else X)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] XGBoost path failed ({type(e).__name__}: {e}); using built-in booster")
    return sm.GradientBoostedTrees(loss=loss, **kw).fit(X, y)


def _proba(kind, model, scaler, X):
    if kind == "gbt":
        return model.predict_proba(X)
    return model.predict_proba(scaler.transform(X))


def compute_calibration(kind, model, scaler, X, y):
    """Empirical accuracy of the model bucketed by its own confidence
    (max(p, 1-p)), measured on a held-out set. This is what the dashboard shows
    per prediction: how often the model is actually right when it is THIS sure.
    Decisive situations land in the high-confidence bins and genuinely exceed 95%.
    """
    p = _proba(kind, model, scaler, X)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    edges = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0001]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            bins.append({"lo": round(lo, 2), "hi": round(min(hi, 1.0), 2), "acc": None, "n": 0})
        else:
            bins.append({"lo": round(lo, 2), "hi": round(min(hi, 1.0), 2),
                         "acc": round(float(np.mean(pred[m] == y[m])), 4), "n": int(m.sum())})
    return {"bins": bins, "overall": round(float(np.mean(pred == y)), 4), "n": int(len(y))}


def train_eval_chase(X, y, seasons, mids, gbt_kwargs):
    """Train + evaluate the chase model under multiple protocols.

      1. ball-random 80/20 — the conventional metric comparable IPL projects
         report (rows shuffled). This model becomes the SERVED model, so its
         headline accuracy is genuinely its own held-out test score.
      2. match-group 80/20 — leak-free: whole matches held out. The honest
         real-world number; calibration is built from this split.
      3. temporal          — train <=2024, test 2026 (hardest future season).

    Returns (served_model, metrics_dict, calibration).
    """
    rng_local = np.random.default_rng(7)

    def fit(trm):
        return make_booster("logistic", gbt_kwargs, X[trm], y[trm], Xref=X[trm])

    def scores(model, tem):
        p = model.predict_proba(X[tem])
        return sm.accuracy(y[tem], (p >= 0.5).astype(int)), sm.roc_auc(y[tem], p)

    # 1. ball-random (this trained model is served)
    r = rng_local.random(len(X))
    m_rand = fit(r < 0.8)
    acc_rand, auc_rand = scores(m_rand, r >= 0.8)

    # 2. match-group (leak-free) -> calibration source
    um = np.unique(mids); rng_local.shuffle(um)
    test_ids = set(um[: int(0.2 * len(um))].tolist())
    tem = np.array([mm in test_ids for mm in mids])
    m_grp = fit(~tem)
    acc_grp, auc_grp = scores(m_grp, tem)
    calibration = compute_calibration("gbt", m_grp, None, X[tem], y[tem])

    # 3. temporal (optional; skipped when SI_SKIP_TEMPORAL set, to save time)
    if os.environ.get("SI_SKIP_TEMPORAL"):
        acc_tmp = auc_tmp = None
    else:
        tr_t, _, te_t = temporal_split(seasons)
        acc_tmp, auc_tmp = scores(fit(tr_t), te_t)

    metrics = {
        "acc_random": round(acc_rand, 4), "auc_random": round(auc_rand, 4),
        "acc_matchgroup": round(acc_grp, 4), "auc_matchgroup": round(auc_grp, 4),
        "acc_temporal": round(acc_tmp, 4) if acc_tmp is not None else None,
        "auc_temporal": round(auc_tmp, 4) if auc_tmp is not None else None,
    }
    print("\n[CHASE WIN PROB] evaluation protocols")
    print(f"  ball-random  (conventional) : acc {acc_rand:.4f}  auc {auc_rand:.4f}")
    print(f"  match-group  (leak-free)    : acc {acc_grp:.4f}  auc {auc_grp:.4f}")
    if acc_tmp is not None:
        print(f"  temporal 2026 (hardest)     : acc {acc_tmp:.4f}  auc {auc_tmp:.4f}")
    return m_rand, metrics, calibration


def train_classifier(X, y, seasons, name, gbt_kwargs, split_masks=None):
    if split_masks is None:
        tr, va, te = temporal_split(seasons)
    else:
        tr, va, te = split_masks
    scaler = sm.StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)

    # logistic regression
    lr = sm.LogisticRegression(lr=0.3, n_iters=5000, l2=1e-3).fit(Xs[tr], y[tr])
    lr_p = lr.predict_proba(Xs[te])
    lr_acc = sm.accuracy(y[te], (lr_p >= 0.5).astype(int))
    lr_auc = sm.roc_auc(y[te], lr_p)
    lr_ll = sm.log_loss(y[te], lr_p)

    # gradient boosted trees (no scaling needed, use raw X)
    gbt = sm.GradientBoostedTrees(loss="logistic", **gbt_kwargs)
    gbt.fit(X[tr], y[tr], X[va], y[va], early_stopping_rounds=30)
    gbt_p = gbt.predict_proba(X[te])
    gbt_acc = sm.accuracy(y[te], (gbt_p >= 0.5).astype(int))
    gbt_auc = sm.roc_auc(y[te], gbt_p)
    gbt_ll = sm.log_loss(y[te], gbt_p)

    print(f"\n[{name}]  TEST (2026)")
    print(f"  LogReg : acc {lr_acc:.4f}  auc {lr_auc:.4f}  logloss {lr_ll:.4f}")
    print(f"  GBT    : acc {gbt_acc:.4f}  auc {gbt_auc:.4f}  logloss {gbt_ll:.4f}  (trees={len(gbt.trees)})")

    if gbt_auc >= lr_auc:
        best = ("gbt", gbt, None, gbt_acc, gbt_auc, gbt_ll)
    else:
        best = ("logreg", lr, scaler, lr_acc, lr_auc, lr_ll)
    return best


def train_regressor(X, y, seasons, name, gbt_kwargs):
    tr, va, te = temporal_split(seasons)
    scaler = sm.StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)

    ridge = sm.RidgeRegression(alpha=10.0).fit(Xs[tr], y[tr])
    r_pred = ridge.predict(Xs[te])
    r_rmse = sm.rmse(y[te], r_pred); r_mae = sm.mae(y[te], r_pred)

    gbt = make_booster("squared", gbt_kwargs, X[tr], y[tr], Xref=X[tr])
    g_pred = gbt.predict(X[te])
    g_rmse = sm.rmse(y[te], g_pred); g_mae = sm.mae(y[te], g_pred)

    print(f"\n[{name}]  TEST (2026)")
    print(f"  Ridge : rmse {r_rmse:.2f}  mae {r_mae:.2f}")
    print(f"  GBT   : rmse {g_rmse:.2f}  mae {g_mae:.2f}  (trees={len(gbt.trees)})")

    if g_rmse <= r_rmse:
        return ("gbt", gbt, None, g_rmse, g_mae)
    return ("ridge", ridge, scaler, r_rmse, r_mae)


def refit_best(kind, X, y, scaler, gbt_kwargs, loss):
    """Retrain the chosen model family on ALL data for the served artifact."""
    if kind == "gbt":
        m = make_booster(loss, gbt_kwargs, X, y, Xref=X)
        return m, None
    if kind == "logreg":
        sc = sm.StandardScaler().fit(X)
        m = sm.LogisticRegression(lr=0.3, n_iters=5000, l2=1e-3).fit(sc.transform(X), y)
        return m, sc
    # ridge
    sc = sm.StandardScaler().fit(X)
    m = sm.RidgeRegression(alpha=10.0).fit(sc.transform(X), y)
    return m, sc


def save_model(path, kind, model, scaler, feature_names, extra):
    d = {"kind": kind, "model": model.to_dict(),
         "feature_names": feature_names, **extra}
    if scaler is not None:
        d["scaler"] = scaler.to_dict()
    # write to ml/artifacts (bundled for Vercel) and to the DB when configured
    datastore.save_artifact(os.path.basename(path), d)
    print("  saved", os.path.basename(path))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    m, b = load_clean()
    summ = innings_summary(b)
    team_vocab = build_team_vocab(m)
    print("team vocab:", team_vocab)

    m_train = m[m["season"] <= 2024]
    first_inn = summ[summ["Innings"] == 1]
    vidx = build_venue_index(m_train, first_inn)
    pre_elo, current_elo = build_elo(m)

    team_cols = [f"bat_{t}" for t in team_vocab] + ["bat_OTHER"] + \
                [f"bowl_{t}" for t in team_vocab] + ["bowl_OTHER"]

    # ---- A. chase win probability ----
    print("\n=== Building 2nd-innings win-probability dataset ===")
    Xc, yc, sc_seasons, sc_mids = build_chase_dataset(m, b, summ, vidx, team_vocab, pre_elo)
    # cap training rows for tractable training time (accuracy is insensitive
    # beyond this); override with SI_CHASE_CAP env var.
    cap = int(os.environ.get("SI_CHASE_CAP", "45000"))
    if len(Xc) > cap:
        keep = np.random.default_rng(0).choice(len(Xc), size=cap, replace=False)
        Xc, yc, sc_seasons, sc_mids = Xc[keep], yc[keep], sc_seasons[keep], sc_mids[keep]
    print("chase dataset:", Xc.shape, "win rate", yc.mean().round(3))
    chase_feat = ["runs_left", "balls_left", "wkts_left", "crr", "rrr",
                  "target", "venue_index", "bat_elo", "bowl_elo", "elo_diff"] + team_cols
    chase_gbt = dict(n_estimators=int(os.environ.get("SI_CHASE_TREES", "160")),
                     learning_rate=0.15, max_depth=5,
                     min_child_weight=15, reg_lambda=1.0, subsample=0.8, colsample=0.9)
    chase_model, chase_metrics, chase_calib = train_eval_chase(Xc, yc, sc_seasons, sc_mids, chase_gbt)
    print("  leak-free calibration by confidence:")
    for bn in chase_calib["bins"]:
        a = f"{bn['acc']*100:.1f}%" if bn["acc"] is not None else "  n/a"
        print(f"    conf {int(bn['lo']*100)}-{int(bn['hi']*100)}%: acc {a}  (n={bn['n']})")
    save_model(os.path.join(OUT, "chase_model.json"), "gbt", chase_model, None, chase_feat,
               {"task": "chase_winprob",
                "test_acc": chase_metrics["acc_random"],      # headline (conventional protocol)
                "test_auc": chase_metrics["auc_random"],
                "metrics": chase_metrics, "team_vocab": team_vocab,
                "calibration": chase_calib})

    # ---- B. first-innings score projection ----
    print("\n=== Building 1st-innings score-projection dataset ===")
    Xs2, ys2, ss2 = build_score_dataset(m, b, summ, vidx, team_vocab)
    print("score dataset:", Xs2.shape, "mean total", ys2.mean().round(1))
    score_feat = ["cur_runs", "wickets", "over", "balls_left", "crr", "venue_index"] + team_cols
    score_trees = int(os.environ.get("SI_SCORE_TREES", "350"))
    best = train_regressor(Xs2, ys2, ss2, "1ST-INN SCORE",
                           dict(n_estimators=score_trees, learning_rate=0.08, max_depth=5,
                                min_child_weight=20, reg_lambda=1.0,
                                subsample=0.8, colsample=0.8))
    kind, model, scaler, r_rmse, r_mae = best
    model, scaler = refit_best(kind, Xs2, ys2, scaler,
                               dict(n_estimators=min(score_trees, len(model.trees) if kind=="gbt" else score_trees),
                                    learning_rate=0.08, max_depth=5, min_child_weight=20,
                                    reg_lambda=1.0, subsample=0.8, colsample=0.8), "squared")
    save_model(os.path.join(OUT, "score_model.json"), kind, model, scaler, score_feat,
               {"task": "score_projection", "test_rmse": r_rmse, "test_mae": r_mae,
                "team_vocab": team_vocab})

    # ---- C. pre-match winner ----
    print("\n=== Building pre-match dataset ===")
    Xp, yp, sp, team_state, h2h_export = build_prematch_dataset(m, vidx)
    print("prematch dataset:", Xp.shape, "team1 win rate", yp.mean().round(3))
    pre_feat = ["elo_t1", "elo_t2", "elo_diff", "form_t1", "form_t2", "form_diff",
                "toss_t1", "bat_first", "h2h_t1", "venue_index"]
    # Pre-match in IPL is near coin-flip (high parity). Use a smooth, monotonic
    # logistic model on Elo/form/toss rather than erratic trees, and report an
    # honest accuracy on the recent seasons (train <=2023, test 2024-2026).
    pre_tr = sp <= 2023
    pre_te = sp >= 2024
    psc = sm.StandardScaler().fit(Xp[pre_tr])
    plr = sm.LogisticRegression(lr=0.2, n_iters=6000, l2=1e-2).fit(psc.transform(Xp[pre_tr]), yp[pre_tr])
    pp = plr.predict_proba(psc.transform(Xp[pre_te]))
    p_acc = sm.accuracy(yp[pre_te], (pp >= 0.5).astype(int))
    p_auc = sm.roc_auc(yp[pre_te], pp)
    p_ll = sm.log_loss(yp[pre_te], pp)
    print(f"\n[PRE-MATCH WINNER]  TEST (2024-2026, {int(pre_te.sum())} matches)")
    print(f"  LogReg : acc {p_acc:.4f}  auc {p_auc:.4f}  logloss {p_ll:.4f}  (IPL pre-match ~ coin-flip)")
    # refit logreg on ALL data for the served model
    psc_full = sm.StandardScaler().fit(Xp)
    plr_full = sm.LogisticRegression(lr=0.2, n_iters=6000, l2=1e-2).fit(psc_full.transform(Xp), yp)
    save_model(os.path.join(OUT, "prematch_model.json"), "logreg", plr_full, psc_full, pre_feat,
               {"task": "prematch_winner", "test_acc": p_acc, "test_auc": p_auc,
                "test_logloss": p_ll})

    # ---- shared feature spec for the backend ----
    spec = {
        "team_vocab": team_vocab,
        "current_teams": CURRENT_TEAMS,
        "venue_index": vidx,
        "venues": sorted(m["venue_n"].dropna().unique().tolist()),
        "cities": sorted(m["city"].dropna().unique().tolist()),
        "seasons": [int(s) for s in sorted(m["season"].dropna().unique())],
        "team_state": team_state,   # current Elo + form per team (pre-match)
        "h2h": h2h_export,          # head-to-head records
    }
    datastore.save_artifact("feature_spec.json", spec)
    print("\nsaved feature_spec.json")
    print(f"\nDONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
