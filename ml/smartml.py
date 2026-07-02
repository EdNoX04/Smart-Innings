"""
smartml.py — from-scratch ML for SmartInnings (pure numpy).

The sandbox where this was trained has no scikit-learn / xgboost, so the
estimators below are implemented from scratch:

  - StandardScaler          : feature standardisation
  - LogisticRegression      : L2-regularised, full-batch gradient descent
  - RidgeRegression         : closed-form L2 linear regression
  - GradientBoostedTrees    : exact-greedy gradient boosting (XGBoost-style)
                              supporting 'logistic' and 'squared' loss

Every estimator can serialise itself to plain JSON-able dicts so the FastAPI
backend can reload a model with numpy only (no pickle, no version coupling).
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
#  Scaling
# --------------------------------------------------------------------------- #
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_dict(self):
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.mean_ = np.array(d["mean"], dtype=np.float64)
        s.std_ = np.array(d["std"], dtype=np.float64)
        return s


# --------------------------------------------------------------------------- #
#  Logistic regression
# --------------------------------------------------------------------------- #
class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=4000, l2=1e-3, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for i in range(self.n_iters):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            grad_w = (X.T @ err) / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            if self.verbose and i % 500 == 0:
                eps = 1e-12
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                print(f"    iter {i:4d}  logloss {loss:.4f}")
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.asarray(X, dtype=np.float64) @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def to_dict(self):
        return {"type": "logreg", "w": self.w.tolist(), "b": float(self.b)}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.w = np.array(d["w"], dtype=np.float64)
        m.b = float(d["b"])
        return m


# --------------------------------------------------------------------------- #
#  Ridge regression (closed form)
# --------------------------------------------------------------------------- #
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        reg = self.alpha * np.eye(d + 1)
        reg[0, 0] = 0.0  # don't penalise the intercept
        theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        self.b = theta[0]
        self.w = theta[1:]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b

    def to_dict(self):
        return {"type": "ridge", "w": self.w.tolist(), "b": float(self.b)}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.w = np.array(d["w"], dtype=np.float64)
        m.b = float(d["b"])
        return m


# --------------------------------------------------------------------------- #
#  Gradient boosted trees (exact greedy, XGBoost-style)
# --------------------------------------------------------------------------- #
class _Tree:
    """A single regression tree on gradients/hessians. Stored as flat arrays."""
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = []     # internal node split feature (-1 for leaf)
        self.threshold = []   # split threshold
        self.left = []        # left child index (-1 leaf)
        self.right = []       # right child index (-1 leaf)
        self.value = []       # leaf output (0 for internal)

    def _add(self):
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(0.0)
        return len(self.feature) - 1

    def to_dict(self):
        return {
            "feature": self.feature, "threshold": self.threshold,
            "left": self.left, "right": self.right, "value": self.value,
        }

    @classmethod
    def from_dict(cls, d):
        t = cls()
        t.feature = d["feature"]; t.threshold = d["threshold"]
        t.left = d["left"]; t.right = d["right"]; t.value = d["value"]
        return t

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        feat = np.array(self.feature); thr = np.array(self.threshold)
        left = np.array(self.left); right = np.array(self.right)
        val = np.array(self.value)
        node = np.zeros(X.shape[0], dtype=int)
        active = np.ones(X.shape[0], dtype=bool)
        out = np.zeros(X.shape[0])
        # iterate until every row lands on a leaf
        for _ in range(64):
            if not active.any():
                break
            f = feat[node]
            is_leaf = f < 0
            done = is_leaf & active
            out[done] = val[node[done]]
            active = active & ~is_leaf
            if not active.any():
                break
            rows = np.where(active)[0]
            go_left = X[rows, f[rows]] <= thr[node[rows]]
            node[rows[go_left]] = left[node[rows[go_left]]]
            node[rows[~go_left]] = right[node[rows[~go_left]]]
        return out


class GradientBoostedTrees:
    def __init__(self, loss="logistic", n_estimators=300, learning_rate=0.1,
                 max_depth=4, min_child_weight=5.0, reg_lambda=1.0,
                 subsample=1.0, colsample=1.0, min_split_gain=0.0,
                 random_state=42, verbose=False):
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample = colsample
        self.min_split_gain = min_split_gain
        self.random_state = random_state
        self.verbose = verbose
        self.trees = []
        self.base_score = 0.0
        self.n_features_ = None

    # --- loss helpers: return gradient & hessian of loss w.r.t. raw score --- #
    def _grad_hess(self, y, raw):
        if self.loss == "logistic":
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -35, 35)))
            return (p - y), np.maximum(p * (1.0 - p), 1e-6)
        else:  # squared error
            return (raw - y), np.ones_like(y)

    def _build_tree(self, X, g, h, rng):
        tree = _Tree()
        n, d = X.shape
        # column subsample
        if self.colsample < 1.0:
            k = max(1, int(self.colsample * d))
            feats = np.sort(rng.choice(d, size=k, replace=False))
        else:
            feats = np.arange(d)

        root = tree._add()
        # stack of (node_id, row_indices, depth)
        stack = [(root, np.arange(n), 0)]
        lam = self.reg_lambda
        while stack:
            nid, idx, depth = stack.pop()
            G = g[idx].sum(); H = h[idx].sum()
            leaf_val = -G / (H + lam)

            if depth >= self.max_depth or len(idx) < 2 or H < self.min_child_weight:
                tree.value[nid] = leaf_val
                continue

            best_gain = self.min_split_gain
            best_f = -1; best_thr = 0.0; best_mask = None
            for f in feats:
                x = X[idx, f]
                order = np.argsort(x, kind="quicksort")
                xs = x[order]; gs = g[idx][order]; hs = h[idx][order]
                Gl = np.cumsum(gs)[:-1]; Hl = np.cumsum(hs)[:-1]
                Gr = G - Gl; Hr = H - Hl
                # only valid where the value actually changes & children are big enough
                valid = (xs[:-1] != xs[1:]) & (Hl >= self.min_child_weight) & (Hr >= self.min_child_weight)
                if not valid.any():
                    continue
                gain = (Gl * Gl) / (Hl + lam) + (Gr * Gr) / (Hr + lam) - (G * G) / (H + lam)
                gain = 0.5 * gain
                gain[~valid] = -np.inf
                k = int(np.argmax(gain))
                if gain[k] > best_gain:
                    best_gain = gain[k]
                    best_f = int(f)
                    best_thr = (xs[k] + xs[k + 1]) / 2.0
                    best_mask = None  # recompute after loop with chosen feature

            if best_f < 0:
                tree.value[nid] = leaf_val
                continue

            go_left = X[idx, best_f] <= best_thr
            left_idx = idx[go_left]; right_idx = idx[~go_left]
            if len(left_idx) == 0 or len(right_idx) == 0:
                tree.value[nid] = leaf_val
                continue

            lid = tree._add(); rid = tree._add()
            tree.feature[nid] = best_f
            tree.threshold[nid] = best_thr
            tree.left[nid] = lid
            tree.right[nid] = rid
            stack.append((lid, left_idx, depth + 1))
            stack.append((rid, right_idx, depth + 1))
        return tree

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.n_features_ = d
        rng = np.random.default_rng(self.random_state)

        if self.loss == "logistic":
            p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            self.base_score = np.log(p / (1 - p))
        else:
            self.base_score = y.mean()

        raw = np.full(n, self.base_score, dtype=np.float64)
        raw_val = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            raw_val = np.full(X_val.shape[0], self.base_score)
        best_val = np.inf; best_iter = 0; rounds_no_improve = 0

        self.trees = []
        for m in range(self.n_estimators):
            g, h = self._grad_hess(y, raw)
            if self.subsample < 1.0:
                mask = rng.random(n) < self.subsample
                sub = np.where(mask)[0]
            else:
                sub = np.arange(n)
            tree = self._build_tree(X[sub], g[sub], h[sub], rng)
            update = tree.predict(X) * self.learning_rate
            raw += update
            self.trees.append(tree)

            if X_val is not None and early_stopping_rounds:
                raw_val += tree.predict(X_val) * self.learning_rate
                vloss = self._eval_loss(y_val, raw_val)
                if vloss < best_val - 1e-5:
                    best_val = vloss; best_iter = m; rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if self.verbose and m % 25 == 0:
                    print(f"    tree {m:3d}  val_loss {vloss:.4f}")
                if rounds_no_improve >= early_stopping_rounds:
                    self.trees = self.trees[: best_iter + 1]
                    if self.verbose:
                        print(f"    early stop at {m}, best iter {best_iter}")
                    break
        return self

    def _eval_loss(self, y, raw):
        if self.loss == "logistic":
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -35, 35)))
            eps = 1e-12
            return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return np.mean((raw - y) ** 2)

    def _raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        raw = np.full(X.shape[0], self.base_score)
        for t in self.trees:
            raw += t.predict(X) * self.learning_rate
        return raw

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self._raw(X), -35, 35)))

    def predict(self, X):
        if self.loss == "logistic":
            return (self.predict_proba(X) >= 0.5).astype(int)
        return self._raw(X)

    def to_dict(self):
        return {
            "type": "gbt",
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "base_score": float(self.base_score),
            "n_features": int(self.n_features_),
            "trees": [t.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(loss=d["loss"], learning_rate=d["learning_rate"])
        m.base_score = float(d["base_score"])
        m.n_features_ = int(d["n_features"])
        m.trees = [_Tree.from_dict(t) for t in d["trees"]]
        return m


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def accuracy(y, yhat):
    return float(np.mean(np.asarray(y) == np.asarray(yhat)))


def log_loss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    n_pos = (y == 1).sum(); n_neg = (y == 0).sum()
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
