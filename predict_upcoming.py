"""Score upcoming fixtures with the committed model and write the season page's data.

    python predict_upcoming.py                      # data/upcoming/2026-27.csv -> docs/data/upcoming.json
    python predict_upcoming.py --check              # run the parity gate only
    python predict_upcoming.py --fixtures other.csv --season 2026-27

The fixtures file uses football-data.co.uk column names, so a fixtures export from
https://www.football-data.co.uk/englandm.php can be saved as-is.
  required : Date (dd/mm/yyyy), HomeTeam, AwayTeam, and one 1X2 price triple
             (AvgH/AvgD/AvgA, BbAvH/BbAvD/BbAvA or B365H/B365D/B365A; vig included, as quoted)
  optional : Time (UK kickoff), Over/Under 2.5 prices (Avg>2.5/Avg<2.5 or B365>2.5/B365<2.5),
             closing prices (PSCH/PSCD/PSCA, AvgCH/..., B365CH/...)
Fixtures that already appear in the season's results file are scored from their pre-kickoff row and
shown with the result. The model is the committed bundle in models/ (trained through 2022-23,
meta-parameters tuned on 2023-25); nothing is retrained here.

Parity gate, run on every invocation before anything is written: features are rebuilt from the raw
season files, the 2025-26 holdout matches are scored with the committed models, and the result must
match data/ui_predictions.json to within rounding. If it does not, the feature pipeline has drifted
from the one the published numbers came from, and the script refuses to write.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

import pl_features as F
from pl_infer import LABELS, contribs_for, load_bundle, market_proba, predict_bundle, score_grid

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
DOCS_DATA = os.path.join(ROOT, "docs", "data")
PARITY_TOL = 5e-4          # ui_predictions.json stores four decimals
COLD_START_GAMES = 3       # the model was trained and scored on rows with >= 3 prior games each


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
#  parity gate
# --------------------------------------------------------------------------- #
def parity_check(df, bundle):
    """Score the holdout rows and compare with the committed predictions file."""
    pub = json.load(open(os.path.join(DATA, "ui_predictions.json"), encoding="utf-8"))
    pub = [m for m in pub if m["eval_group"] == "holdout"]
    key = pd.DataFrame({"Date": pd.to_datetime([m["date"] for m in pub]),
                        "HomeTeam": [m["home"] for m in pub], "AwayTeam": [m["away"] for m in pub],
                        "_i": range(len(pub))})
    rows = key.merge(df, on=["Date", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one")
    missing = rows["match_id"].isna().sum()
    if missing:
        raise SystemExit(f"parity: {missing} holdout matches not found in the rebuilt feature table")
    pr = predict_bundle(bundle, rows)
    ref = {k: np.array([[m[k][l] for l in LABELS] for m in pub]) for k in ("proba", "proba_clf", "proba_dc")}
    errs = {"calibrated": np.abs(pr["cal"] - ref["proba"]).max(),
            "classifier": np.abs(pr["clf"] - ref["proba_clf"]).max(),
            "dixon_coles": np.abs(pr["dc"] - ref["proba_dc"]).max()}
    ok = all(v <= PARITY_TOL for v in errs.values())
    log(f"parity gate on {len(pub)} holdout matches: max |error| calibrated {errs['calibrated']:.2e}, "
        f"classifier {errs['classifier']:.2e}, dixon-coles {errs['dixon_coles']:.2e}  "
        f"(tolerance {PARITY_TOL:g}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        worst = int(np.abs(pr["cal"] - ref["proba"]).max(axis=1).argmax())
        m = pub[worst]
        log(f"  worst row: {m['date']} {m['home']} v {m['away']}  published {m['proba']}  "
            f"rebuilt {dict(zip(LABELS, np.round(pr['cal'][worst], 4)))}")
        raise SystemExit("parity gate failed: the rebuilt features no longer reproduce the published "
                         "predictions, so nothing was written. Fix the feature pipeline first.")
    return {"n": len(pub), "max_abs_err": float(max(errs.values())), "tolerance": PARITY_TOL}


# --------------------------------------------------------------------------- #
#  fixtures
# --------------------------------------------------------------------------- #
def load_fixtures(path, season, matches):
    fx = F.load_season(path, season=season, require_result=False, extra_cols=("Time",))
    fx = fx.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    played = matches[matches.Season == season]
    known = set(played.HomeTeam) | set(played.AwayTeam)
    strength = pd.read_csv(os.path.join(DATA, "team_season_strength.csv"))
    known |= set(strength.loc[strength.Season == season, "team"])
    unknown = sorted((set(fx.HomeTeam) | set(fx.AwayTeam)) - known)
    if unknown:
        log(f"  ! club names not seen in {season} data or squad table (check spelling against "
            f"football-data.co.uk): {unknown}")
    key = ["Date", "HomeTeam", "AwayTeam"]
    already = fx.merge(played[key], on=key, how="inner")
    if len(already):
        log(f"  {len(already)} fixture(s) already have a result in data/{season}.csv; "
            f"scored from their pre-kickoff row, result attached")
    fx = fx.merge(played[key].assign(_played=True), on=key, how="left")
    fx = fx[fx._played.isna()].drop(columns="_played")
    if len(played):
        last = played.Date.max()
        stale = fx[fx.Date < last]
        if len(stale):
            log(f"  ! {len(stale)} fixture(s) dated before the last played match ({last.date()}) "
                f"have no result in the season file and are skipped: "
                f"{[f'{r.Date.date()} {r.HomeTeam} v {r.AwayTeam}' for r in stale.itertuples()]}")
            fx = fx[fx.Date >= last]
    no_odds = fx[["mkt_H", "mkt_D", "mkt_A"]].isna().any(axis=1).sum()
    if no_odds:
        log(f"  ! {no_odds} fixture(s) have no usable 1X2 prices; the market prior falls back to uniform")
    no_ou = fx[["mkt_over25", "mkt_under25"]].isna().any(axis=1).sum()
    if no_ou:
        log(f"  ! {no_ou} fixture(s) have no over/under 2.5 prices; the goal model's prior falls back to "
            f"league-average expected goals, which pulls its share of the blend toward the middle")
    return fx.reset_index(drop=True)


def uk_zone(date):
    """British Summer Time: last Sunday of March to last Sunday of October."""
    y = date.year
    def last_sunday(month):
        d = dt.date(y, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(y, 12, 31)
        return d - dt.timedelta(days=(d.weekday() + 1) % 7)
    return "BST" if last_sunday(3) <= date.date() < last_sunday(10) else "GMT"


def records_for(df, bundle, season, matches_time):
    part = df[df.Season == season].sort_values(["Date", "match_id"]).reset_index(drop=True)
    if not len(part):
        return []
    pr = predict_bundle(bundle, part)
    ctr = contribs_for(bundle["clf"], part, bundle["feats"])
    mkt = market_proba(part, "p")
    out = []
    for i, r in part.iterrows():
        cal = pr["cal"][i]
        pick = LABELS[int(cal.argmax())]
        played = pd.notna(r.FTHG) and pd.notna(r.FTAG)
        gp_h = int(r.home_games_played) if pd.notna(r.home_games_played) else 0
        gp_a = int(r.away_games_played) if pd.notna(r.away_games_played) else 0
        has_odds = bool(np.isfinite(r[["mkt_H", "mkt_D", "mkt_A"]].to_numpy(float)).all())
        time = matches_time.get((r.Date, r.HomeTeam, r.AwayTeam)) or (r.Time if "Time" in part.columns and isinstance(r.Time, str) else None)
        out.append({
            "id": int(r.match_id),
            "date": r.Date.strftime("%Y-%m-%d"),
            "time": time, "tz": uk_zone(r.Date),
            "matchday": max(gp_h, gp_a) + 1,
            "home": r.HomeTeam, "away": r.AwayTeam,
            "played": bool(played),
            "actual": (r.FTR if played else None),
            "score": ([int(r.FTHG), int(r.FTAG)] if played else None),
            "pred": pick,
            "proba": {k: round(float(v), 4) for k, v in zip(LABELS, cal)},
            "proba_clf": {k: round(float(v), 4) for k, v in zip(LABELS, pr["clf"][i])},
            "proba_dc": {k: round(float(v), 4) for k, v in zip(LABELS, pr["dc"][i])},
            "market": ({k: round(float(v), 4) for k, v in zip(LABELS, mkt[i])} if has_odds else None),
            "odds": ({"H": float(r.mkt_H), "D": float(r.mkt_D), "A": float(r.mkt_A)} if has_odds else None),
            "xg": {"home": round(float(pr["lam"][i]), 2), "away": round(float(pr["mu"][i]), 2)},
            "elo": {"home": round(float(r.home_elo_pre)), "away": round(float(r.away_elo_pre))},
            "grid": score_grid(float(pr["lam"][i]), float(pr["mu"][i]), bundle["rho"]),
            "signals": ctr[i],
            "games_played": {"home": gp_h, "away": gp_a},
            "cold_start": bool(min(gp_h, gp_a) < COLD_START_GAMES),
            "correct": (bool(pick == r.FTR) if played else None),
        })
    return out


def write_outputs(payload):
    os.makedirs(DOCS_DATA, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(DOCS_DATA, "upcoming.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    # The page loads this wrapper with a <script> tag so it also works from a file:// clone,
    # where fetch() of the .json would be blocked. Same content.
    safe = text.replace("</", "<\\/")
    with open(os.path.join(DOCS_DATA, "upcoming.js"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("window.UPCOMING = " + safe + ";\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", default=os.path.join(DATA, "upcoming", "2026-27.csv"))
    ap.add_argument("--season", default=None, help="season label, e.g. 2026-27 (default: from the file name)")
    ap.add_argument("--check", action="store_true", help="run the parity gate only; write nothing")
    args = ap.parse_args(argv)
    season = args.season or F.season_name(args.fixtures)

    log("loading models and the raw season files")
    bundle = load_bundle(os.path.join(ROOT, "models"))
    matches = F.load_matches(DATA)
    matches_time = {(r.Date, r.HomeTeam, r.AwayTeam): r.Time for r in matches.itertuples()
                    if isinstance(getattr(r, "Time", None), str)}
    strength = pd.read_csv(os.path.join(DATA, "team_season_strength.csv"))

    fixtures = None
    if not args.check:
        if os.path.exists(args.fixtures):
            log(f"reading fixtures from {os.path.relpath(args.fixtures, ROOT)}")
            fixtures = load_fixtures(args.fixtures, season, matches)
            log(f"  {len(fixtures)} unplayed fixture(s) to score")
        else:
            log(f"no fixtures file at {os.path.relpath(args.fixtures, ROOT)}; scoring only the "
                f"{season} matches already in the season file")

    combined = matches if fixtures is None or not len(fixtures) else pd.concat([matches, fixtures], ignore_index=True)
    log("building features for every match, including the fixtures")
    df = F.build_features(combined, strength=strength)

    parity = parity_check(df, bundle)
    if args.check:
        log("check only: nothing written")
        return 0

    recs = records_for(df, bundle, season, matches_time)
    if not recs:
        log(f"no {season} rows to score; nothing written (the page will show its empty state)")
        return 0
    meta = bundle["config"]
    ui_meta = json.load(open(os.path.join(DATA, "ui_meta.json"), encoding="utf-8"))
    payload = {
        "season": season,
        "generated": dt.date.today().isoformat(),
        "source_csv": os.path.relpath(args.fixtures, ROOT).replace("\\", "/") if fixtures is not None else None,
        "model": {
            "trained_through": "2022-23",
            "tuned_on": meta["splits"]["test"],
            "blend_weight_w": meta["blend_weight_w"],
            "dixon_coles_rho": meta["dixon_coles_rho"],
            "temperature": meta["temperature"],
            "n_features": len(meta["features"]),
            "holdout": {"season": "2025-26", "n": ui_meta["holdout"]["n"],
                        "model_logloss": ui_meta["holdout"]["model"]["logloss"],
                        "market_logloss": ui_meta["holdout"]["market"]["logloss"]},
        },
        "parity": parity,
        "cold_start_games": COLD_START_GAMES,
        "fixtures": recs,
    }
    write_outputs(payload)
    played = sum(r["played"] for r in recs)
    log(f"wrote docs/data/upcoming.json and upcoming.js: {len(recs)} {season} fixture(s), "
        f"{played} played, {sum(1 for r in recs if r['correct'])} called correctly, "
        f"{sum(r['cold_start'] for r in recs)} flagged early-season")
    return 0


if __name__ == "__main__":
    sys.exit(main())
