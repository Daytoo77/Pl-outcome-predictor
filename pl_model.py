"""Shared helpers for the PL outcome model (model.ipynb + export).

Pure numpy / pandas / xgboost — no scipy.linalg or scikit-learn, because Windows
Smart App Control blocks their unsigned native DLLs on this machine. Everything
here (isotonic calibration, the Dixon-Coles scoreline matrix, the metrics) is
therefore hand-rolled and short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["H", "D", "A"]
Y_MAP = {"H": 0, "D": 1, "A": 2}
INV = {v: k for k, v in Y_MAP.items()}


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
def confusion(y_true, y_pred, labels=LABELS):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return pd.DataFrame(m, index=[f"actual {l}" for l in labels],
                        columns=[f"pred {l}" for l in labels])


def log_loss(y_true, proba, labels=LABELS):
    idx = {l: i for i, l in enumerate(labels)}
    p = np.clip(np.asarray(proba, float), 1e-15, 1 - 1e-15)
    yi = np.array([idx[t] for t in y_true])
    return float(np.mean(-np.log(p[np.arange(len(yi)), yi])))


def rps(y_true, proba, labels=LABELS):
    """Ranked probability score — the proper scoring rule for ordered H/D/A."""
    idx = {l: i for i, l in enumerate(labels)}
    p = np.asarray(proba, float)
    y = np.zeros_like(p)
    y[np.arange(len(p)), [idx[t] for t in y_true]] = 1.0
    cp, cy = np.cumsum(p, axis=1), np.cumsum(y, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / (p.shape[1] - 1)))


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def class_report(y_true, y_pred, labels=LABELS):
    cm = confusion(y_true, y_pred, labels).to_numpy()
    rows = []
    for i, l in enumerate(labels):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((l, prec, rec, f1, int(cm[i, :].sum())))
    return pd.DataFrame(rows, columns=["class", "precision", "recall", "f1", "support"]).set_index("class")


def to_pred(proba):
    return np.array([INV[i] for i in np.asarray(proba).argmax(1)])


# --------------------------------------------------------------------------- #
#  isotonic calibration  (pool-adjacent-violators, pure numpy)
# --------------------------------------------------------------------------- #
class Isotonic:
    """Monotone 1-D map x -> y fitted by PAV, then linearly interpolated."""

    def __init__(self):
        self.xs = None
        self.ys = None

    def fit(self, x, y, w=None):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        w = np.ones_like(x) if w is None else np.asarray(w, float)
        order = np.argsort(x, kind="mergesort")
        x, y, w = x[order], y[order], w[order]

        # pool adjacent violators
        val = list(y)
        wt = list(w)
        idx = [[i] for i in range(len(y))]
        i = 0
        while i < len(val) - 1:
            if val[i] > val[i + 1] + 1e-12:
                new_w = wt[i] + wt[i + 1]
                new_v = (val[i] * wt[i] + val[i + 1] * wt[i + 1]) / new_w
                val[i] = new_v
                wt[i] = new_w
                idx[i] = idx[i] + idx[i + 1]
                del val[i + 1], wt[i + 1], idx[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        fitted = np.empty(len(y))
        for v, group in zip(val, idx):
            for g in group:
                fitted[g] = v

        # collapse to step knots on the sorted x grid
        self.xs = x
        self.ys = np.clip(fitted, 0.0, 1.0)
        return self

    def predict(self, x):
        x = np.asarray(x, float)
        return np.interp(x, self.xs, self.ys, left=self.ys[0], right=self.ys[-1])


class TemperatureScale:
    """Single-parameter calibration: p -> softmax(log p / T). Robust on small
    validation sets where per-class isotonic would overfit. T > 1 softens, T < 1 sharpens."""

    def __init__(self):
        self.T = 1.0

    def fit(self, proba, y_true, grid=None):
        proba = np.clip(np.asarray(proba, float), 1e-9, 1)
        yi = np.array([Y_MAP[t] for t in y_true])
        grid = np.linspace(0.5, 3.0, 126) if grid is None else grid
        best, best_ll = 1.0, np.inf
        logp = np.log(proba)
        for T in grid:
            z = logp / T
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            ll = float(np.mean(-np.log(np.clip(p[np.arange(len(yi)), yi], 1e-15, 1))))
            if ll < best_ll:
                best, best_ll = float(T), ll
        self.T = best
        return self

    def transform(self, proba):
        logp = np.log(np.clip(np.asarray(proba, float), 1e-9, 1)) / self.T
        logp = logp - logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)


class MultiIsotonic:
    """Per-class isotonic on a 3-class probability matrix, renormalised to sum 1."""

    def __init__(self):
        self.cals = [Isotonic() for _ in LABELS]

    def fit(self, proba, y_true):
        proba = np.asarray(proba, float)
        yi = np.array([Y_MAP[t] for t in y_true])
        for k in range(len(LABELS)):
            self.cals[k].fit(proba[:, k], (yi == k).astype(float))
        return self

    def transform(self, proba):
        proba = np.asarray(proba, float)
        out = np.column_stack([self.cals[k].predict(proba[:, k]) for k in range(len(LABELS))])
        out = np.clip(out, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)

    def to_dict(self):
        return {LABELS[k]: {"x": self.cals[k].xs.tolist(), "y": self.cals[k].ys.tolist()}
                for k in range(len(LABELS))}


# --------------------------------------------------------------------------- #
#  Dixon-Coles scoreline model
# --------------------------------------------------------------------------- #
def _pois_pmf(k, lam):
    """Poisson pmf for integer k (array) and lam (array), no scipy."""
    k = np.asarray(k, float)
    lam = np.asarray(lam, float)
    logp = -lam + k * np.log(np.clip(lam, 1e-12, None)) - _lgamma(k + 1.0)
    return np.exp(logp)


def _lgamma(x):
    # Lanczos approximation — plenty accurate for small integer args
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    x = np.asarray(x, float)
    xm1 = x - 1.0
    a = np.full_like(xm1, c[0])
    t = xm1 + g + 0.5
    for i in range(1, g + 2):
        a = a + c[i] / (xm1 + i)
    return 0.5 * np.log(2 * np.pi) + (xm1 + 0.5) * np.log(t) - t + np.log(a)


def dc_tau(i, j, lam, mu, rho):
    """Dixon-Coles low-score dependence correction."""
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def dc_matrix(lam, mu, rho, max_goals=10):
    """P(home i, away j) scoreline matrix for one match (lam, mu scalars)."""
    ii = np.arange(max_goals + 1)
    ph = _pois_pmf(ii, np.full(max_goals + 1, lam))
    pa = _pois_pmf(ii, np.full(max_goals + 1, mu))
    M = np.outer(ph, pa)
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        M[i, j] *= dc_tau(i, j, lam, mu, rho)
    return M / M.sum()


def dc_outcome_probs(lam, mu, rho, max_goals=10):
    """Vectorised over arrays lam, mu -> (n,3) array of P(H), P(D), P(A)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    mu = np.atleast_1d(np.asarray(mu, float))
    n = len(lam)
    ii = np.arange(max_goals + 1)
    # poisson pmf grids: (n, g+1)
    ph = _pois_pmf(ii[None, :], lam[:, None])
    pa = _pois_pmf(ii[None, :], mu[:, None])
    ph /= ph.sum(1, keepdims=True)
    pa /= pa.sum(1, keepdims=True)
    M = ph[:, :, None] * pa[:, None, :]                      # (n, g+1, g+1)
    corr = np.ones((n, max_goals + 1, max_goals + 1))
    corr[:, 0, 0] = 1.0 - lam * mu * rho
    corr[:, 0, 1] = 1.0 + lam * rho
    corr[:, 1, 0] = 1.0 + mu * rho
    corr[:, 1, 1] = 1.0 - rho
    M = np.clip(M * corr, 0.0, None)
    M /= M.sum((1, 2), keepdims=True)
    iu = np.triu(np.ones((max_goals + 1, max_goals + 1)), 1)   # home < away  -> away win
    il = np.tril(np.ones((max_goals + 1, max_goals + 1)), -1)  # home > away  -> home win
    idm = np.eye(max_goals + 1)
    pH = (M * il).sum((1, 2))
    pD = (M * idm).sum((1, 2))
    pA = (M * iu).sum((1, 2))
    out = np.column_stack([pH, pD, pA])
    return out / out.sum(1, keepdims=True)


def fit_rho(lam, mu, y_true, grid=None):
    """Pick the Dixon-Coles rho that minimises outcome log loss on a holdout slice."""
    grid = np.linspace(-0.25, 0.25, 51) if grid is None else grid
    best, best_ll = 0.0, np.inf
    for rho in grid:
        ll = log_loss(y_true, dc_outcome_probs(lam, mu, rho))
        if ll < best_ll:
            best, best_ll = float(rho), ll
    return best, best_ll


# --------------------------------------------------------------------------- #
#  feature groups
# --------------------------------------------------------------------------- #
SQUAD_DIFF_HINT = ("price", "squad", "att_edge")
MARKET_FEATS = [
    "mkt_p_H", "mkt_p_D", "mkt_p_A", "mkt_pow_H", "mkt_pow_D", "mkt_pow_A",
    "mkt_overround", "mkt_supremacy", "mkt_entropy", "mkt_p_over25",
    "mkt_tot_goals", "mkt_xg_home", "mkt_xg_away",
    "close_p_H", "close_p_D", "close_p_A", "clv_H", "clv_D", "clv_A",
]
ELO_FEATS = ["home_elo_pre", "away_elo_pre", "elo_exp_home", "hfa_used", "is_derby"]


FEATURE_LABELS = {
    "elo_diff": "Elo rating gap", "home_elo_pre": "home Elo", "away_elo_pre": "away Elo",
    "elo_exp_home": "Elo-implied home win", "hfa_used": "home-field advantage", "is_derby": "derby",
    "mkt_p_H": "market P(home)", "mkt_p_D": "market P(draw)", "mkt_p_A": "market P(away)",
    "mkt_pow_H": "market P(home), bias-adj", "mkt_pow_D": "market P(draw), bias-adj",
    "mkt_pow_A": "market P(away), bias-adj", "mkt_overround": "bookmaker margin",
    "mkt_supremacy": "market supremacy", "mkt_entropy": "market uncertainty",
    "mkt_p_over25": "market P(over 2.5)", "mkt_tot_goals": "market total goals",
    "mkt_xg_home": "market home xG", "mkt_xg_away": "market away xG",
    "close_p_H": "closing P(home)", "close_p_D": "closing P(draw)", "close_p_A": "closing P(away)",
    "clv_H": "home line move", "clv_D": "draw line move", "clv_A": "away line move",
    "pts_roll5_diff": "points form (5)", "pts_ewm_diff": "points form (EWMA)",
    "gd_roll5_diff": "goal-diff form (5)", "gd_ewm_diff": "goal-diff form (EWMA)",
    "gf_roll5_diff": "scoring form", "ga_roll5_diff": "conceding form",
    "shots_roll5_diff": "shots form", "shots_ewm_diff": "shots form (EWMA)",
    "sot_roll5_diff": "shots-on-target form", "sot_ewm_diff": "SoT form (EWMA)",
    "corners_roll5_diff": "corners form", "corners_ewm_diff": "corners form (EWMA)",
    "xg_roll5_diff": "xG form", "xg_ewm_diff": "xG form (EWMA)",
    "xga_roll5_diff": "xG-against form", "xga_ewm_diff": "xG-against (EWMA)",
    "xgd_roll5_diff": "xG-diff form", "xgd_ewm_diff": "xG-diff form (EWMA)",
    "finishing_roll5_diff": "finishing vs xG", "def_luck_roll5_diff": "defence vs xG",
    "conv_roll5_diff": "shot conversion", "sot_rate_roll5_diff": "shot accuracy",
    "save_pct_roll5_diff": "save %", "wform_roll5_diff": "opponent-weighted form",
    "pts_std_diff": "season points/game", "gd_std_diff": "season goal-diff/game",
    "gf_std_diff": "season goals/game", "ga_std_diff": "season conceded/game",
    "shots_std_diff": "season shots/game", "sot_std_diff": "season SoT/game",
    "corners_std_diff": "season corners/game",
    "venue_pts_roll5_diff": "home/away form", "days_rest_diff": "rest days",
    "congestion_14d_diff": "fixture congestion", "h2h_pts_roll5_diff": "head-to-head record",
    "cum_pts_diff": "league points", "cum_gd_diff": "league goal difference",
    "rank_diff": "league position gap", "pts_gap_to_safety_diff": "gap to safety",
    "pts_gap_to_top4_diff": "gap to top 4", "match_round_diff": "games played gap",
    "games_played_diff": "games played gap",
    "squad_price_total_diff": "squad value", "squad_price_top11_diff": "first-XI value",
    "bench_price_diff": "bench value", "squad_ppg_diff": "squad prior PPG",
    "gk_price_diff": "keeper value", "def_price_diff": "defence value",
    "mid_price_diff": "midfield value", "fwd_price_diff": "attack value",
    "att_edge_diff": "attack vs defence edge",
}


def label_of(feat):
    return FEATURE_LABELS.get(feat, feat.replace("_diff", "").replace("_", " "))


def feature_groups(df):
    squad = [c for c in df.columns if c.endswith("_diff") and any(h in c for h in SQUAD_DIFF_HINT)]
    form = [c for c in df.columns if c.endswith("_diff") and c not in squad]
    market = [c for c in MARKET_FEATS if c in df.columns]
    elo = [c for c in ELO_FEATS if c in df.columns]
    allf = form + elo + market + squad
    return {"form": form, "elo": elo, "market": market, "squad": squad, "all": allf}
