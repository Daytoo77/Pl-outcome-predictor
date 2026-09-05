"""Derive the small numbers the site shows that are not already in a committed artefact.

Everything here is computed from files that are committed (season CSVs, data/ui_predictions.json,
models/config.json, data/team_index.json), so every figure on the site traces back to the repo.

    python tools/build_site_stats.py        ->  docs/data/site_stats.json

Stdlib + numpy only.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "docs", "data", "site_stats.json")
LABELS = ["H", "D", "A"]


# ---------------------------------------------------------------------------
# Elo home-field advantage fed per season (same formula as features.ipynb §3.4b)
# ---------------------------------------------------------------------------
def season_files():
    out = []
    for name in sorted(os.listdir(DATA)):
        if re.fullmatch(r"\d{4}-\d{2}\.csv", name):
            out.append((name[:-4], os.path.join(DATA, name)))
    return out


def home_points_rate(path):
    """Mean of {H:1, D:0.5, A:0} over rows that have a result — matches the notebook's
    `matches.assign(hp=FTR.map(...)).groupby('Season').hp.mean()` after load_season's dropna."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc, newline="") as fh:
                rows = list(csv.reader(fh))
            break
        except UnicodeDecodeError:
            continue
    header = rows[0]
    ix = {c: i for i, c in enumerate(header)}
    need = ["Date", "HomeTeam", "AwayTeam", "FTR"]
    total, n = 0.0, 0
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        r = r[: len(header)]
        vals = {k: (r[ix[k]].strip() if ix[k] < len(r) else "") for k in need}
        if any(v == "" for v in vals.values()):
            continue
        total += {"H": 1.0, "D": 0.5, "A": 0.0}[vals["FTR"]]
        n += 1
    return total / n, n


def hfa_from_rate(sbar):
    sbar = min(max(sbar, 0.5001), 0.75)
    return -400.0 * math.log10(1.0 / sbar - 1.0)


def hfa_by_season():
    files = season_files()
    rates = {s: home_points_rate(p) for s, p in files}
    seasons = [s for s, _ in files]
    out = {}
    for i, s in enumerate(seasons):
        prev = seasons[max(0, i - 3):i]
        if prev:
            out[s] = round(hfa_from_rate(sum(rates[p][0] for p in prev) / len(prev)), 2)
        else:
            out[s] = 60.0
    return out, {s: {"home_points_rate": round(r, 4), "n": n} for s, (r, n) in rates.items()}


# ---------------------------------------------------------------------------
# Confusion matrices and per-class recall from the committed predictions
# ---------------------------------------------------------------------------
def confusion(preds, group):
    rows = [m for m in preds if m["eval_group"] == group and m["played"]]
    cm = [[0] * 3 for _ in LABELS]
    for m in rows:
        cm[LABELS.index(m["actual"])][LABELS.index(m["pred"])] += 1
    n = len(rows)
    per = {}
    for i, lab in enumerate(LABELS):
        support = sum(cm[i])
        called = sum(cm[k][i] for k in range(3))
        per[lab] = {
            "support": support,
            "called": called,
            "recall": round(cm[i][i] / support, 4) if support else None,
            "precision": round(cm[i][i] / called, 4) if called else None,
        }
    return {
        "n": n,
        "labels": LABELS,
        "matrix": cm,
        "accuracy": round(sum(cm[i][i] for i in range(3)) / n, 4),
        "per_class": per,
        "predicted": {lab: sum(cm[k][i] for k in range(3)) for i, lab in enumerate(LABELS)},
        "actual": {lab: sum(cm[i]) for i, lab in enumerate(LABELS)},
    }


# ---------------------------------------------------------------------------
# Which blend weight / temperature reproduces the published probabilities?
# ---------------------------------------------------------------------------
def temp(p, T):
    z = np.log(np.clip(p, 1e-9, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def reconstruct_blend(preds, group):
    rows = [m for m in preds if m["eval_group"] == group]
    arr = lambda k: np.array([[m[k][l] for l in LABELS] for m in rows])
    pc, pdc, cal = arr("proba_clf"), arr("proba_dc"), arr("proba")
    best = None
    for w in np.linspace(0, 1, 41):
        for T in np.linspace(0.5, 3.0, 126):
            e = float(np.abs(temp(w * pc + (1 - w) * pdc, T) - cal).mean())
            if best is None or e < best[0]:
                best = (e, float(w), float(T))
    return {"n": len(rows), "w_clf": round(best[1], 3), "w_dc": round(1 - best[1], 3),
            "temperature": round(best[2], 2), "mean_abs_err": round(best[0], 6)}


def main():
    preds = json.load(open(os.path.join(DATA, "ui_predictions.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(DATA, "ui_meta.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(ROOT, "models", "config.json"), encoding="utf-8"))
    teams = json.load(open(os.path.join(DATA, "team_index.json"), encoding="utf-8"))

    hfa, rates = hfa_by_season()
    roll = meta["rolling_origin"]
    beats = sum(r["model_ll"] < r["market_ll"] for r in roll)
    edges = [r["edge"] for r in roll]

    # committed-model reproduction check for the holdout rows, using config.json values directly
    rows = [m for m in preds if m["eval_group"] == "holdout"]
    arr = lambda k: np.array([[m[k][l] for l in LABELS] for m in rows])
    w, T = cfg["blend_weight_w"], cfg["temperature"]
    err_committed = float(np.abs(temp(w * arr("proba_clf") + (1 - w) * arr("proba_dc"), T) - arr("proba")).max())

    latest = max(v["season"] for v in teams.values())
    elo_top = sorted(((v["elo"], k) for k, v in teams.items() if v["season"] == latest), reverse=True)

    played_2627 = rates.get("2026-27", {"n": 0})["n"]

    stats = {
        "_about": "Derived by tools/build_site_stats.py from committed artefacts; not a model output.",
        "hfa_by_season": hfa,
        "home_points_rate_by_season": rates,
        "confusion": {g: confusion(preds, g) for g in ("test", "holdout")},
        "rolling_origin": {"n_seasons": len(roll), "beats_market": int(beats),
                           "mean_edge": round(sum(edges) / len(edges), 4),
                           "min_model_ll": min(r["model_ll"] for r in roll),
                           "max_model_ll": max(r["model_ll"] for r in roll)},
        "blend_reconstructed": {g: reconstruct_blend(preds, g) for g in ("holdout", "test")},
        "committed_model_reproduction_max_abs_err": round(err_committed, 6),
        "elo_top": [{"team": t, "elo": e} for e, t in elo_top[:5]],
        "elo_season": latest,
        "season_2026_27": {"played_matches_in_data": played_2627},
        "draws": {g: {"n_drawn": confusion(preds, g)["actual"]["D"], "n_called": confusion(preds, g)["predicted"]["D"]}
                  for g in ("test", "holdout")},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(stats, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    print("hfa:", {k: v for k, v in list(hfa.items())[-6:]})
    print("rolling-origin beats market:", beats, "/", len(roll), " mean edge", stats["rolling_origin"]["mean_edge"])
    print("blend reconstructed:", stats["blend_reconstructed"])
    print("committed model max abs err on holdout:", err_committed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
