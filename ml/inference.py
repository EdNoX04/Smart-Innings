"""
inference.py — load trained SmartInnings artifacts and produce predictions.

Feature construction here MUST mirror ml/train.py exactly. Shared by the
FastAPI backend (and usable standalone for testing).
"""
from __future__ import annotations
import os, json
import numpy as np

import smartml as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


def _load_model(d):
    t = d["model"]["type"]
    if t == "gbt":
        return sm.GradientBoostedTrees.from_dict(d["model"])
    if t == "logreg":
        return sm.LogisticRegression.from_dict(d["model"])
    if t == "ridge":
        return sm.RidgeRegression.from_dict(d["model"])
    raise ValueError(f"unknown model type {t}")


def _onehot_team(team, vocab):
    v = np.zeros(len(vocab) + 1)
    if team in vocab:
        v[vocab.index(team)] = 1.0
    else:
        v[-1] = 1.0
    return v


class SmartInnings:
    def __init__(self, art_dir=ART):
        self.art = art_dir
        self.spec = json.load(open(os.path.join(art_dir, "feature_spec.json")))
        self.vidx = self.spec["venue_index"]
        self.team_vocab = self.spec["team_vocab"]
        self.team_state = self.spec["team_state"]
        self.h2h = self.spec["h2h"]

        self.chase = json.load(open(os.path.join(art_dir, "chase_model.json")))
        self.score = json.load(open(os.path.join(art_dir, "score_model.json")))
        self.prematch = json.load(open(os.path.join(art_dir, "prematch_model.json")))
        self.chase_m = _load_model(self.chase)
        self.score_m = _load_model(self.score)
        self.prematch_m = _load_model(self.prematch)
        self.chase_sc = sm.StandardScaler.from_dict(self.chase["scaler"]) if "scaler" in self.chase else None
        self.score_sc = sm.StandardScaler.from_dict(self.score["scaler"]) if "scaler" in self.score else None
        self.prematch_sc = sm.StandardScaler.from_dict(self.prematch["scaler"]) if "scaler" in self.prematch else None

    # --- helpers ---
    def venue_score(self, venue, city=None):
        if venue in self.vidx["venue"]:
            return self.vidx["venue"][venue]
        if city and city in self.vidx["city"]:
            return self.vidx["city"][city]
        return self.vidx["global"]

    def _elo(self, t):
        return self.team_state.get(t, {}).get("elo", 1500.0)

    def _form(self, t):
        return self.team_state.get(t, {}).get("form", 0.5)

    def _h2h_rate(self, t1, t2):
        key = "|".join(sorted([t1, t2]))
        rec = self.h2h.get(key)
        if not rec or rec[1] == 0:
            return 0.5
        hw, hg = rec
        first = sorted([t1, t2])[0]
        return (hw if first == t1 else hg - hw) / hg

    def _apply(self, model, scaler, x, proba=True):
        X = x.reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)
        if proba:
            return float(model.predict_proba(X)[0])
        return float(model.predict(X)[0])

    # --- 2nd-innings win probability ---
    def predict_chase(self, batting_team, bowling_team, venue, target,
                      score, wickets, overs, balls, city=None):
        legal = int(overs) * 6 + int(balls)
        legal = max(0, min(legal, 119))
        balls_left = 120 - legal
        runs_left = max(target - score, 0)
        wkts_left = 10 - int(wickets)
        overs_done = legal / 6.0
        crr = score / overs_done if overs_done > 0 else 0.0
        rrr = runs_left / (balls_left / 6.0) if balls_left > 0 else 99.0
        vs = self.venue_score(venue, city)
        bat_elo = self._elo(batting_team)
        bowl_elo = self._elo(bowling_team)
        num = [float(runs_left), float(balls_left), float(wkts_left),
               crr, rrr, float(target), vs, bat_elo, bowl_elo, bat_elo - bowl_elo]
        x = np.concatenate([num, _onehot_team(batting_team, self.team_vocab),
                            _onehot_team(bowling_team, self.team_vocab)])
        # deterministic terminal states
        deterministic = False
        if runs_left <= 0:
            p = 1.0; deterministic = True
        elif wkts_left <= 0 or balls_left <= 0:
            p = 0.0; deterministic = True
        else:
            p = self._apply(self.chase_m, self.chase_sc, x, proba=True)

        conf = max(p, 1 - p)
        if deterministic:
            sit_acc, sit_n = 100.0, None
        else:
            sit_acc, sit_n = self._situation_accuracy(conf)
        return {
            "batting_win_prob": round(p * 100, 1),
            "bowling_win_prob": round((1 - p) * 100, 1),
            "runs_left": runs_left, "balls_left": balls_left,
            "crr": round(crr, 2), "rrr": round(rrr, 2),
            "confidence": round(conf * 100, 1),
            "situation_accuracy": sit_acc,   # % the model is right when this confident
            "situation_n": sit_n,
        }

    def predict_chase_sweep(self, batting_team, bowling_team, venue, target,
                            score, wickets, overs, balls, city=None):
        """Return batting win% for wickets-lost 0..9 at the fixed situation."""
        by_wickets = []
        for w in range(0, 10):
            r = self.predict_chase(batting_team, bowling_team, venue, target,
                                   score, w, overs, balls, city)
            by_wickets.append({"wickets_lost": w, "wickets_in_hand": 10 - w,
                               "prob": r["batting_win_prob"]})
        return {"by_wickets": by_wickets, "current_wickets": int(wickets)}

    def _situation_accuracy(self, conf):
        """Look up empirical held-out accuracy for the model's current confidence."""
        cal = self.chase.get("calibration")
        if not cal:
            return None, None
        for bn in cal["bins"]:
            if bn["acc"] is not None and bn["lo"] <= conf < bn["hi"] + 1e-9:
                return round(bn["acc"] * 100, 1), bn["n"]
        # fall back to nearest populated bin / overall
        populated = [bn for bn in cal["bins"] if bn["acc"] is not None]
        if populated:
            nearest = min(populated, key=lambda bn: abs((bn["lo"] + bn["hi"]) / 2 - conf))
            return round(nearest["acc"] * 100, 1), nearest["n"]
        return round(cal["overall"] * 100, 1), cal["n"]

    # --- 1st-innings projected score ---
    def predict_score(self, batting_team, bowling_team, venue,
                      score, wickets, overs, balls, city=None):
        legal = int(overs) * 6 + int(balls)
        legal = max(1, min(legal, 119))
        over = legal / 6.0
        balls_left = 120 - legal
        crr = score / over if over > 0 else 0.0
        vs = self.venue_score(venue, city)
        num = [float(score), float(wickets), float(over), float(balls_left), crr, vs]
        x = np.concatenate([num, _onehot_team(batting_team, self.team_vocab),
                            _onehot_team(bowling_team, self.team_vocab)])
        proj = self._apply(self.score_m, self.score_sc, x, proba=False)
        proj = max(proj, score)  # can't finish below current score
        rmse = self.score.get("test_rmse", 22.0)
        return {
            "projected_score": int(round(proj)),
            "low": int(round(proj - rmse)),
            "high": int(round(proj + rmse)),
            "current_run_rate": round(crr, 2),
        }

    def _prematch_raw(self, t1, t2, venue, toss_winner, toss_decision, city=None):
        e1, e2 = self._elo(t1), self._elo(t2)
        toss_t1 = 1.0 if toss_winner == t1 else 0.0
        bat_first = 1.0 if str(toss_decision).lower().startswith("bat") else 0.0
        vs = self.venue_score(venue, city)
        num = [e1, e2, e1 - e2, self._form(t1), self._form(t2),
               self._form(t1) - self._form(t2), toss_t1, bat_first,
               self._h2h_rate(t1, t2), vs]
        return self._apply(self.prematch_m, self.prematch_sc, np.array(num), proba=True)

    # --- pre-match winner (symmetric / order-invariant) ---
    def predict_prematch(self, team_a, team_b, venue, toss_winner,
                         toss_decision, city=None):
        p_a = self._prematch_raw(team_a, team_b, venue, toss_winner, toss_decision, city)
        p_b = self._prematch_raw(team_b, team_a, venue, toss_winner, toss_decision, city)
        p = (p_a + (1 - p_b)) / 2.0
        return {
            "team_a_win_prob": round(p * 100, 1),
            "team_b_win_prob": round((1 - p) * 100, 1),
            "team_a_elo": round(self._elo(team_a)),
            "team_b_elo": round(self._elo(team_b)),
        }


if __name__ == "__main__":
    si = SmartInnings()
    print("chase :", si.predict_chase("Chennai Super Kings", "Mumbai Indians",
          "MA Chidambaram Stadium", target=180, score=120, wickets=3, overs=15, balls=0))
    print("score :", si.predict_score("Royal Challengers Bengaluru", "Kolkata Knight Riders",
          "M Chinnaswamy Stadium", score=80, wickets=1, overs=10, balls=0))
    print("pre   :", si.predict_prematch("Chennai Super Kings", "Mumbai Indians",
          "MA Chidambaram Stadium", "Chennai Super Kings", "bat"))
