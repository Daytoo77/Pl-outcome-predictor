"""Inference helpers shared by model.ipynb and predict_upcoming.py.

Everything needed to turn a feature row into the model's output, given the trained boosters and
models/config.json: the market prior as XGBoost base margin, the two Poisson goal models, the
Dixon-Coles outcome probabilities, the blend, temperature calibration, per-row feature
contributions and the scoreline grid. Training stays in the notebook.
"""
from __future__ import annotations

import json
import os

import numpy as np

from pl_model import LABELS, TemperatureScale, dc_matrix, dc_outcome_probs, label_of

MODELS_DIR = "models"


# --------------------------------------------------------------------------- #
#  priors
# --------------------------------------------------------------------------- #
def market_base_margin(frame):
    """base_margin for multi:softprob = log(vig-free market prob); uniform where missing."""
    p = frame[["mkt_p_H", "mkt_p_D", "mkt_p_A"]].to_numpy(float)
    ok = np.isfinite(p).all(axis=1)
    p = np.where(np.isfinite(p), p, 1 / 3)
    p[~ok] = 1 / 3
    p = np.clip(p, 1e-4, 1)
    p /= p.sum(axis=1, keepdims=True)
    return np.log(p)


def goals_base_margin(frame, side):
    """base_margin for count:poisson = log(market implied xG); league mean where missing."""
    col = "mkt_xg_home" if side == "home" else "mkt_xg_away"
    x = frame[col].to_numpy(float)
    fallback = 1.5 if side == "home" else 1.1
    x = np.where(np.isfinite(x) & (x > 0.05), x, fallback)
    return np.log(x)


# --------------------------------------------------------------------------- #
#  DMatrix construction and prediction
# --------------------------------------------------------------------------- #
def clf_dmatrix(frame, feats, label=None):
    import xgboost as xgb
    d = xgb.DMatrix(frame[feats], missing=np.nan, label=label, feature_names=list(feats))
    d.set_base_margin(market_base_margin(frame).ravel())
    return d


def goals_dmatrix(frame, feats, side, label=None):
    import xgboost as xgb
    d = xgb.DMatrix(frame[feats], missing=np.nan, label=label, feature_names=list(feats))
    d.set_base_margin(goals_base_margin(frame, side))
    return d


def best_range(bst):
    """Iteration range that respects early stopping, for a live or a re-loaded booster."""
    best = getattr(bst, "best_iteration", None)
    return (0, best + 1) if best is not None else (0, 0)


def clf_proba(bst, frame, feats):
    return bst.predict(clf_dmatrix(frame, feats), iteration_range=best_range(bst))


def poisson_pred(bst, frame, feats, side):
    return np.clip(bst.predict(goals_dmatrix(frame, feats, side), iteration_range=best_range(bst)), 0.08, 6.0)


def contribs_for(bst, frame, feats, top=6):
    """Top signed feature contributions toward each row's predicted class (classifier)."""
    d = clf_dmatrix(frame, feats)
    C = bst.predict(d, pred_contribs=True, iteration_range=best_range(bst))      # (n, 3, n_feat+1)
    pc = clf_proba(bst, frame, feats)
    out = []
    for i in range(len(frame)):
        cls = int(pc[i].argmax())
        row = C[i, cls, :-1]
        order = np.argsort(-np.abs(row))[:top]
        out.append([{"feat": feats[j], "label": label_of(feats[j]), "effect": round(float(row[j]), 3)}
                    for j in order])
    return out


def score_grid(lam, mu, rho, mx=5):
    """Exact-score distribution for the interface: 6x6 grid, top scorelines, over 2.5, both score."""
    M = dc_matrix(lam, mu, rho, max_goals=8)
    g = M[:mx + 1, :mx + 1]
    tops = sorted(((g[i, j], i, j) for i in range(mx + 1) for j in range(mx + 1)), reverse=True)[:6]
    return {"grid": [[round(float(g[i, j]), 4) for j in range(mx + 1)] for i in range(mx + 1)],
            "top": [{"h": int(i), "a": int(j), "p": round(float(v), 4)} for v, i, j in tops],
            "p_over25": round(float(M[np.add.outer(np.arange(9), np.arange(9)) >= 3].sum()), 4),
            "p_btts": round(float(M[1:, 1:].sum()), 4)}


# --------------------------------------------------------------------------- #
#  the committed bundle
# --------------------------------------------------------------------------- #
def load_bundle(models_dir=MODELS_DIR):
    """The committed models and config as one dict with the same keys model.ipynb's fit_bundle uses."""
    import xgboost as xgb
    cfg = json.load(open(os.path.join(models_dir, "config.json"), encoding="utf-8"))
    def load(name):
        b = xgb.Booster()
        b.load_model(os.path.join(models_dir, name))
        return b
    cal = TemperatureScale()
    cal.T = float(cfg["temperature"])
    return {"clf": load("xgb_clf.json"), "pois_home": load("xgb_pois_home.json"),
            "pois_away": load("xgb_pois_away.json"), "rho": float(cfg["dixon_coles_rho"]),
            "w": float(cfg["blend_weight_w"]), "cal": cal, "feats": list(cfg["features"]),
            "config": cfg}


def predict_bundle(B, frame):
    """A, B, the blend and the calibrated blend for any feature frame."""
    f = B["feats"]
    lam = poisson_pred(B["pois_home"], frame, f, "home")
    mu = poisson_pred(B["pois_away"], frame, f, "away")
    pc = clf_proba(B["clf"], frame, f)
    pdc = dc_outcome_probs(lam, mu, B["rho"])
    pb = B["w"] * pc + (1 - B["w"]) * pdc
    return dict(lam=lam, mu=mu, clf=pc, dc=pdc, blend=pb, cal=B["cal"].transform(pb))


def market_proba(frame, kind="p"):
    """The de-vigged market as a probability matrix (uniform where no odds)."""
    cols = [f"mkt_{kind}_H", f"mkt_{kind}_D", f"mkt_{kind}_A"]
    p = frame[cols].to_numpy(float)
    p = np.where(np.isfinite(p), p, 1 / 3)
    return p / p.sum(axis=1, keepdims=True)


__all__ = ["LABELS", "market_base_margin", "goals_base_margin", "clf_dmatrix", "goals_dmatrix",
           "best_range", "clf_proba", "poisson_pred", "contribs_for", "score_grid", "load_bundle",
           "predict_bundle", "market_proba"]
