"""Loading and feature engineering for the PL outcome model.

Shared by explore.ipynb, features.ipynb and predict_upcoming.py so that there is exactly one
implementation of every feature. The notebooks walk through these functions stage by stage;
the script calls build_features() end to end.

The one rule: a feature for match k may only use matches 1 ... k-1. Every rolling statistic is
shift(1)-ed before it is aggregated, Elo is recorded pre-update, odds are the pre-kickoff price.

Rows without a result (unplayed fixtures) are carried through: they receive pre-kickoff features
like any other row but never contribute to anything downstream of themselves, because their
points, goals and stats are NaN and the Elo loop skips their update.

Pure numpy / pandas.
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  loading (explore.ipynb)
# --------------------------------------------------------------------------- #
CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
STAT_COLS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
XG_COLS = ["HxG", "AxG"]                       # only present 2026-27 onward
KEEP_COLS = CORE_COLS + STAT_COLS + XG_COLS
NUMERIC_COLS = ["FTHG", "FTAG"] + STAT_COLS + XG_COLS

# The bookmaker columns change name and provider across 25 seasons. Collapse them into one
# stable schema, trying each provider group in priority order and taking the first that is
# fully populated for a row.
ODDS_GROUPS = {
    "mkt":   [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")],
    "ou":    [("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("B365>2.5", "B365<2.5")],
    "close": [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"),
              ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")],
}
ODDS_SOURCE_COLS = sorted({c for groups in ODDS_GROUPS.values() for grp in groups for c in grp}
                          | {"B365H", "B365D", "B365A"})
ORDER = (["Season"] + CORE_COLS + STAT_COLS + XG_COLS + ["B365H", "B365D", "B365A"]
         + ["mkt_H", "mkt_D", "mkt_A", "mkt_over25", "mkt_under25", "close_H", "close_D", "close_A"])


def season_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def season_files(data_dir="data"):
    """The per-season football-data.co.uk files, sorted: data/2000-01.csv ... data/2026-27.csv."""
    return sorted(p for p in glob.glob(os.path.join(data_dir, "*.csv"))
                  if re.fullmatch(r"\d{4}-\d{2}", season_name(p)))


def read_season_csv(path):
    """Read one football-data.co.uk CSV robustly.

    Some older files gain extra columns partway through the season without adding headers, so
    rows are wider than the header: read with the csv module and truncate every row to the header
    width. Also handles the one non-UTF-8 file.
    """
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc, newline="") as fh:
                rows = list(csv.reader(fh))
            break
        except UnicodeDecodeError:
            continue
    header = rows[0]
    width = len(header)
    body = [r[:width] for r in rows[1:] if any(c.strip() for c in r)]
    df = pd.DataFrame(body, columns=header)
    return df.replace(r"^\s*$", pd.NA, regex=True)


def harmonise_odds(raw):
    """Collapse the era-specific bookmaker columns into the stable schema."""
    out = pd.DataFrame(index=raw.index)
    num = {c: pd.to_numeric(raw[c], errors="coerce") for c in ODDS_SOURCE_COLS if c in raw.columns}

    def first_complete(groups, names):
        vals = {n: pd.Series(np.nan, index=raw.index) for n in names}
        for grp in groups:
            if not all(c in num for c in grp):
                continue
            block = pd.concat([num[c] for c in grp], axis=1)
            ok = block.notna().all(axis=1) & (block > 1.0).all(axis=1)
            need = vals[names[0]].isna()
            take = ok & need
            for n, c in zip(names, grp):
                vals[n] = vals[n].mask(take, num[c])
        return vals

    for n, v in first_complete(ODDS_GROUPS["mkt"], ["mkt_H", "mkt_D", "mkt_A"]).items():
        out[n] = v
    for n, v in first_complete(ODDS_GROUPS["ou"], ["mkt_over25", "mkt_under25"]).items():
        out[n] = v
    for n, v in first_complete(ODDS_GROUPS["close"], ["close_H", "close_D", "close_A"]).items():
        out[n] = v
    return out


def load_season(path, season=None, require_result=True, extra_cols=()):
    """One season file -> the clean match table for that season.

    require_result=False keeps rows without a full-time result (a fixtures file); their FTHG,
    FTAG and FTR stay NaN. extra_cols are carried through untouched (e.g. kickoff Time).
    """
    raw = read_season_csv(path)
    keep = [c for c in KEEP_COLS if c in raw.columns] + [c for c in extra_cols if c in raw.columns]
    df = raw[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["Season"] = season or season_name(path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    odds = harmonise_odds(raw)
    df = pd.concat([df.reset_index(drop=True), odds.reset_index(drop=True)], axis=1)
    for c in ["B365H", "B365D", "B365A"]:
        df[c] = pd.to_numeric(raw[c], errors="coerce").reset_index(drop=True) if c in raw.columns else np.nan

    need = ["Date", "HomeTeam", "AwayTeam"] + (["FTR"] if require_result else [])
    df = df.dropna(subset=need)
    cols = [c for c in ORDER if c in df.columns] + [c for c in extra_cols if c in df.columns]
    return df[cols]


def load_matches(data_dir="data", extra_cols=("Time",)):
    """All season files -> one chronological table (matches_clean). Kickoff Time is carried
    through where the file has it (2019-20 onward); it is never a feature."""
    matches = pd.concat([load_season(p, extra_cols=extra_cols) for p in season_files(data_dir)],
                        ignore_index=True)
    return matches.sort_values("Date", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  3.1  one row per team per match
# --------------------------------------------------------------------------- #
def points(gf, ga):
    """League points from goals for / against; NaN where the match has no result."""
    pts = pd.Series(1, index=gf.index).mask(gf > ga, 3).mask(gf < ga, 0)
    return pts.where(gf.notna() & ga.notna())


def reshape_team_matches(matches):
    common = dict(match_id=matches.match_id, Season=matches.Season, Date=matches.Date)
    home = pd.DataFrame({**common,
        "team": matches.HomeTeam, "opponent": matches.AwayTeam, "venue": "home",
        "gf": matches.FTHG, "ga": matches.FTAG,
        "shots": matches.HS, "sot": matches.HST, "corners": matches.HC})
    away = pd.DataFrame({**common,
        "team": matches.AwayTeam, "opponent": matches.HomeTeam, "venue": "away",
        "gf": matches.FTAG, "ga": matches.FTHG,
        "shots": matches.AS, "sot": matches.AST, "corners": matches.AC})
    tm = pd.concat([home, away], ignore_index=True)
    tm["pts"] = points(tm.gf, tm.ga)
    tm["gd"] = tm.gf - tm.ga
    return tm.sort_values(["Date", "match_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  3.2  rolling form: shift(1), then aggregate
# --------------------------------------------------------------------------- #
ROLL_STATS = ["pts", "gf", "ga", "shots", "sot", "corners", "gd"]
WINDOW = 5
EWM_HALFLIFE = 4          # continuous decay, no hard 5-game cliff


def _key(*cols):
    """Collapse one or more columns into a single grouping-key Series."""
    if len(cols) == 1:
        return cols[0]
    out = cols[0].astype(str)
    for c in cols[1:]:
        out = out + "||" + c.astype(str)
    return out


def prior_rolling(s, key, window, how="mean"):
    """Aggregate the last `window` PRIOR rows within each group (shift, then roll)."""
    shifted = s.groupby(key).shift(1)
    rolled = shifted.groupby(key).rolling(window, min_periods=1)
    rolled = getattr(rolled, how)().reset_index(level=0, drop=True)
    return rolled.reindex(s.index)


def prior_expanding(s, key, how="mean"):
    """Aggregate ALL prior rows within each group (shift, then expand)."""
    shifted = s.groupby(key).shift(1)
    exp = shifted.groupby(key).expanding(min_periods=1)
    exp = getattr(exp, how)().reset_index(level=0, drop=True)
    return exp.reindex(s.index)


def prior_ewm(s, key, halflife):
    """Exponentially-weighted mean of PRIOR rows (shift, then ewm)."""
    shifted = s.groupby(key).shift(1)
    ewm = shifted.groupby(key).ewm(halflife=halflife).mean()
    return ewm.reset_index(level=0, drop=True).reindex(s.index)


def _team_order(tm):
    return tm.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)


def add_rolling_form(tm):
    tm = _team_order(tm)
    for stat in ROLL_STATS:
        tm[f"{stat}_roll{WINDOW}"] = prior_rolling(tm[stat], _key(tm.team), WINDOW)
        tm[f"{stat}_std"] = prior_expanding(tm[stat], _key(tm.team, tm.Season))
        tm[f"{stat}_ewm"] = prior_ewm(tm[stat], _key(tm.team), EWM_HALFLIFE)
    # games already played this season (0 for the opener); scheduled rows count too
    tm["games_played"] = tm.groupby(["team", "Season"]).cumcount()
    return tm


# --------------------------------------------------------------------------- #
#  3.3  venue split, rest days, congestion, head-to-head
# --------------------------------------------------------------------------- #
def matches_last_n_days(dates, n):
    """For each row, count this team's PRIOR matches within the last n days."""
    d = dates.values.astype("datetime64[D]")
    out = np.zeros(len(d), dtype=int)
    j = 0
    for i in range(len(d)):
        while d[i] - d[j] > np.timedelta64(n, "D"):
            j += 1
        out[i] = i - j
    return out


def add_venue_rest_h2h(tm):
    tm = _team_order(tm)
    tm[f"venue_pts_roll{WINDOW}"] = prior_rolling(tm.pts, _key(tm.team, tm.venue), WINDOW)
    tm["days_rest"] = tm.groupby("team")["Date"].diff().dt.days
    tm["congestion_14d"] = (
        tm.groupby("team", group_keys=False)["Date"]
          .apply(lambda s: pd.Series(matches_last_n_days(s.sort_values(), 14),
                                     index=s.sort_values().index))
          .reindex(tm.index)
    )
    lo = tm[["team", "opponent"]].min(axis=1)
    hi = tm[["team", "opponent"]].max(axis=1)
    tm["pair"] = lo + " v " + hi
    tm["h2h_pts_roll5"] = prior_rolling(tm.pts, _key(tm.team, tm.pair), 5)
    return tm


# --------------------------------------------------------------------------- #
#  3.4a  rolling shot quality / efficiency
# --------------------------------------------------------------------------- #
def _opponent_stat(tm, col, suffix):
    opp = (tm[["match_id", "team", col]]
           .merge(tm[["match_id", "team", col]], on="match_id", suffixes=("", suffix)))
    opp = opp[opp.team != opp["team" + suffix]][["match_id", "team", col + suffix]]
    return tm.merge(opp, on=["match_id", "team"], how="left")


def add_shot_quality(tm):
    for stat in ["gf", "ga", "shots", "sot"]:
        tm[f"{stat}_sum{WINDOW}"] = prior_rolling(tm[stat], _key(tm.team), WINDOW, how="sum")
    tm["conv_roll5"] = tm.gf_sum5 / tm.sot_sum5.replace(0, np.nan)
    tm["sot_rate_roll5"] = tm.sot_sum5 / tm.shots_sum5.replace(0, np.nan)
    tm = _opponent_stat(tm, "sot", "_opp")
    tm["sot_faced_sum5"] = prior_rolling(tm.sot_opp, _key(tm.team), WINDOW, how="sum")
    tm["save_pct_roll5"] = 1 - tm.ga_sum5 / tm.sot_faced_sum5.replace(0, np.nan)
    return tm


# --------------------------------------------------------------------------- #
#  3.4a-bis  expected-goals proxy (OLS on off-target / on-target shots)
# --------------------------------------------------------------------------- #
XG_FIT_MAX_SEASON = "2018-19"


def _det3(A):
    return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))


def ols3(y, x1, x2):
    """Least-squares y ~ a + b*x1 + c*x2 via the 3x3 normal equations (Cramer's rule)."""
    m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    y, x1, x2 = y[m], x1[m], x2[m]
    n = float(len(y))
    S = np.array([
        [n,          x1.sum(),        x2.sum()],
        [x1.sum(),   (x1 * x1).sum(), (x1 * x2).sum()],
        [x2.sum(),   (x1 * x2).sum(), (x2 * x2).sum()],
    ])
    rhs = np.array([y.sum(), (x1 * y).sum(), (x2 * y).sum()])
    d0 = _det3(S)
    out = []
    for i in range(3):
        Ai = S.copy()
        Ai[:, i] = rhs
        out.append(_det3(Ai) / d0)
    return out   # a, b, c


def fit_xg_proxy(tm, max_season=XG_FIT_MAX_SEASON):
    """Fit (a, b, c) on seasons <= max_season only, so nothing downstream of the split leaks in."""
    fit = tm[tm.Season <= max_season]
    return ols3(fit.gf.to_numpy(float), (fit.shots - fit.sot).to_numpy(float), fit.sot.to_numpy(float))


def add_xg(tm, matches, coefs):
    a, b, c = coefs
    tm = _opponent_stat(tm, "shots", "_opp")

    def xg_proxy(shots, sot):
        return np.clip(a + b * (shots - sot) + c * sot, 0.0, None)

    tm["xg"] = xg_proxy(tm.shots, tm.sot)
    tm["xga"] = xg_proxy(tm.shots_opp, tm.sot_opp)

    # real xG where the source provides it: map match xG onto team rows
    if "HxG" in matches.columns and matches["HxG"].notna().any():
        hx = matches.set_index("match_id")[["HxG", "AxG"]]
        for_map = tm["match_id"].map(hx["HxG"]).where(tm.venue == "home", tm["match_id"].map(hx["AxG"]))
        against_map = tm["match_id"].map(hx["AxG"]).where(tm.venue == "home", tm["match_id"].map(hx["HxG"]))
        tm["xg"] = for_map.where(for_map.notna(), tm["xg"])
        tm["xga"] = against_map.where(against_map.notna(), tm["xga"])

    for stat in ["xg", "xga"]:
        tm[f"{stat}_roll{WINDOW}"] = prior_rolling(tm[stat], _key(tm.team), WINDOW)
        tm[f"{stat}_ewm"] = prior_ewm(tm[stat], _key(tm.team), EWM_HALFLIFE)
    tm["xg_sum5"] = prior_rolling(tm.xg, _key(tm.team), WINDOW, how="sum")
    tm["xga_sum5"] = prior_rolling(tm.xga, _key(tm.team), WINDOW, how="sum")
    tm["xgd_roll5"] = tm.xg_roll5 - tm.xga_roll5
    tm["xgd_ewm"] = tm.xg_ewm - tm.xga_ewm
    tm["finishing_roll5"] = (tm.gf_sum5 - tm.xg_sum5) / WINDOW       # + = finishing above xG
    tm["def_luck_roll5"] = (tm.xga_sum5 - tm.ga_sum5) / WINDOW       # + = conceding below xGA
    return tm


# --------------------------------------------------------------------------- #
#  3.4b  Elo: margin-aware, moving home-field advantage, seasonal regression
# --------------------------------------------------------------------------- #
ELO_K = 20
ELO_BASE = 1500.0
ELO_REGRESS = 0.25   # toward BASE at each season boundary
DEFAULT_HFA = 60.0


def hfa_from_rate(sbar):
    sbar = min(max(sbar, 0.5001), 0.75)          # keep the log finite / sane
    return -400.0 * np.log10(1.0 / sbar - 1.0)


def hfa_by_season(matches):
    """Home-field advantage fed to Elo each season: implied by the previous three seasons' results."""
    season_home_rate = (matches.assign(hp=matches.FTR.map({"H": 1.0, "D": 0.5, "A": 0.0}))
                        .groupby("Season").hp.mean())
    seasons_sorted = sorted(matches.Season.unique())
    out = {}
    for i, s in enumerate(seasons_sorted):
        prev = seasons_sorted[max(0, i - 3):i]
        out[s] = hfa_from_rate(season_home_rate.loc[prev].mean()) if prev else DEFAULT_HFA
    return out


def add_elo(matches, return_ratings=False):
    """Pre-match Elo for both sides; unplayed rows read the current rating and update nothing.
    With return_ratings=True also returns the post-update rating of every club."""
    HFA = hfa_by_season(matches)
    elo, seen_season = {}, {}
    home_pre, away_pre, exp_home, hfa_used = [], [], [], []
    for row in matches.itertuples(index=False):
        hfa = HFA[row.Season]
        for t in (row.HomeTeam, row.AwayTeam):
            if t not in elo:
                elo[t] = ELO_BASE
            if seen_season.get(t) != row.Season:            # season rollover for this team
                elo[t] = ELO_BASE + (1 - ELO_REGRESS) * (elo[t] - ELO_BASE)
                seen_season[t] = row.Season
        rh, ra = elo[row.HomeTeam], elo[row.AwayTeam]
        e_h = 1 / (1 + 10 ** (-((rh + hfa) - ra) / 400))
        home_pre.append(rh); away_pre.append(ra)
        exp_home.append(e_h); hfa_used.append(hfa)
        if not isinstance(row.FTR, str):                     # fixture without a result
            continue
        s_h = 1.0 if row.FTR == "H" else (0.5 if row.FTR == "D" else 0.0)
        margin = abs(int(row.FTHG) - int(row.FTAG))
        k_eff = ELO_K * np.sqrt(max(margin, 1))              # margin-of-victory scaling
        elo[row.HomeTeam] = rh + k_eff * (s_h - e_h)
        elo[row.AwayTeam] = ra + k_eff * ((1 - s_h) - (1 - e_h))
    matches = matches.copy()
    matches["home_elo_pre"] = home_pre
    matches["away_elo_pre"] = away_pre
    matches["elo_diff"] = matches.home_elo_pre - matches.away_elo_pre
    matches["elo_exp_home"] = exp_home
    matches["hfa_used"] = hfa_used
    return (matches, elo) if return_ratings else matches


# --------------------------------------------------------------------------- #
#  3.4c  opponent-weighted form + derby flag
# --------------------------------------------------------------------------- #
DERBIES = {
    frozenset({"Arsenal", "Tottenham"}), frozenset({"Liverpool", "Everton"}),
    frozenset({"Man United", "Man City"}), frozenset({"Man United", "Liverpool"}),
    frozenset({"Chelsea", "Tottenham"}), frozenset({"Chelsea", "Arsenal"}),
    frozenset({"Chelsea", "Fulham"}), frozenset({"Arsenal", "Man United"}),
    frozenset({"Newcastle", "Sunderland"}), frozenset({"Aston Villa", "Birmingham"}),
    frozenset({"Aston Villa", "West Brom"}), frozenset({"Wolves", "West Brom"}),
    frozenset({"Southampton", "Portsmouth"}), frozenset({"Crystal Palace", "Brighton"}),
    frozenset({"West Ham", "Tottenham"}), frozenset({"West Ham", "Chelsea"}),
    frozenset({"West Ham", "Millwall"}), frozenset({"Nott'm Forest", "Derby"}),
    frozenset({"Leeds", "Man United"}),
}


def add_opponent_weighted_form(tm, matches):
    eh = matches[["match_id", "HomeTeam", "away_elo_pre"]].rename(
        columns={"HomeTeam": "team", "away_elo_pre": "opp_elo_pre"})
    ea = matches[["match_id", "AwayTeam", "home_elo_pre"]].rename(
        columns={"AwayTeam": "team", "home_elo_pre": "opp_elo_pre"})
    tm = tm.merge(pd.concat([eh, ea], ignore_index=True), on=["match_id", "team"], how="left")
    tm["pts_x_oppstrength"] = tm.pts * (tm.opp_elo_pre / ELO_BASE)
    tm["wform_roll5"] = prior_rolling(tm.pts_x_oppstrength, _key(tm.team), WINDOW)
    return tm


def add_derby(matches):
    matches = matches.copy()
    matches["is_derby"] = [frozenset({h, a}) in DERBIES for h, a in zip(matches.HomeTeam, matches.AwayTeam)]
    return matches


# --------------------------------------------------------------------------- #
#  3.4e  market signal: vig-free probabilities, implied goals, closing-line move
# --------------------------------------------------------------------------- #
MARKET_MATCH_FEATS = [
    "mkt_p_H", "mkt_p_D", "mkt_p_A", "mkt_pow_H", "mkt_pow_D", "mkt_pow_A",
    "mkt_overround", "mkt_supremacy", "mkt_entropy", "mkt_p_over25",
    "mkt_tot_goals", "mkt_xg_home", "mkt_xg_away",
    "close_p_H", "close_p_D", "close_p_A", "clv_H", "clv_D", "clv_A",
]
MARKET_FIT_MAX_SEASON = "2018-19"


def devig_proportional(odds_df):
    inv = 1.0 / odds_df.to_numpy(float)
    return inv / inv.sum(axis=1, keepdims=True)


def fit_power_gamma(odds_df, y_idx, grid=np.linspace(0.80, 1.10, 61)):
    """Favourite-longshot-corrected de-vig: p_i propto (1/o_i)^gamma, gamma by 1X2 log loss."""
    inv = 1.0 / odds_df.to_numpy(float)
    best, best_ll = 1.0, np.inf
    for g in grid:
        p = inv ** g
        p = p / p.sum(axis=1, keepdims=True)
        ll = -np.log(np.clip(p[np.arange(len(p)), y_idx], 1e-12, 1)).mean()
        if ll < best_ll:
            best, best_ll = float(g), ll
    return best


def total_goals_from_over(p_over, lo=0.15, hi=7.0, iters=70):
    """Invert P(N>=3) = 1 - e^-L (1 + L + L^2/2) for the Poisson total-goals mean L."""
    p_over = np.asarray(p_over, float)
    lo = np.full_like(p_over, lo); hi = np.full_like(p_over, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p_mid = 1.0 - np.exp(-mid) * (1 + mid + mid**2 / 2)
        lo = np.where(p_mid < p_over, mid, lo)
        hi = np.where(p_mid < p_over, hi, mid)
    return 0.5 * (lo + hi)


def add_market(matches, fit_max_season=MARKET_FIT_MAX_SEASON):
    matches = matches.copy()
    o = matches[["mkt_H", "mkt_D", "mkt_A"]]
    ok = o.notna().all(axis=1)

    for col in ["mkt_p_H", "mkt_p_D", "mkt_p_A", "mkt_pow_H", "mkt_pow_D", "mkt_pow_A"]:
        matches[col] = np.nan
    matches.loc[ok, ["mkt_p_H", "mkt_p_D", "mkt_p_A"]] = devig_proportional(o[ok])

    fit_mask = ok & (matches.Season <= fit_max_season) & matches.FTR.notna()
    y_fit = matches.loc[fit_mask, "FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy(int)
    gamma = fit_power_gamma(o[fit_mask], y_fit)
    pw = (1.0 / o[ok].to_numpy(float)) ** gamma
    matches.loc[ok, ["mkt_pow_H", "mkt_pow_D", "mkt_pow_A"]] = pw / pw.sum(axis=1, keepdims=True)

    matches["mkt_overround"] = (1.0 / o).sum(axis=1) - 1.0
    matches["mkt_supremacy"] = matches.mkt_p_H - matches.mkt_p_A
    p3 = matches[["mkt_p_H", "mkt_p_D", "mkt_p_A"]].clip(1e-9)
    matches["mkt_entropy"] = -(p3 * np.log(p3)).sum(axis=1)

    c = matches[["close_H", "close_D", "close_A"]]
    ck = c.notna().all(axis=1)
    close_p = pd.DataFrame(np.nan, index=matches.index, columns=["close_p_H", "close_p_D", "close_p_A"])
    close_p.loc[ck, :] = devig_proportional(c[ck])
    matches[["close_p_H", "close_p_D", "close_p_A"]] = close_p
    for s in ["H", "D", "A"]:
        matches[f"clv_{s}"] = matches[f"close_p_{s}"] - matches[f"mkt_p_{s}"]

    ou = matches[["mkt_over25", "mkt_under25"]]
    ouk = ou.notna().all(axis=1)
    p_over = pd.Series(np.nan, index=matches.index)
    p_over.loc[ouk] = devig_proportional(ou[ouk])[:, 0]
    matches["mkt_p_over25"] = p_over
    matches["mkt_tot_goals"] = total_goals_from_over(p_over.fillna(0.5))
    matches.loc[~ouk, "mkt_tot_goals"] = np.nan
    share_home = (0.5 + 0.9 * matches.mkt_supremacy).clip(0.15, 0.85)
    matches["mkt_xg_home"] = matches.mkt_tot_goals * share_home
    matches["mkt_xg_away"] = matches.mkt_tot_goals * (1 - share_home)
    matches.attrs["power_gamma"] = gamma
    return matches


# --------------------------------------------------------------------------- #
#  3.4d  league position & pressure, as the table stood before each match
# --------------------------------------------------------------------------- #
def prematch_standing(g):
    """Within one season's rows: cumulative points / GD / rank BEFORE each team-match."""
    g = g.sort_values(["Date", "match_id"]).copy()
    pts = g["pts"].fillna(0)                     # unplayed rows add nothing
    gd = g["gd"].fillna(0)
    g["cum_pts"] = pts.groupby(g["team"]).cumsum() - pts
    g["cum_gd"] = gd.groupby(g["team"]).cumsum() - gd
    g["rank"] = np.nan
    g["pts_of_18th"] = np.nan
    g["pts_of_4th"] = np.nan
    for dt in g["Date"].unique():
        seen = g[g["Date"] <= dt].sort_values("match_id")
        latest = seen.groupby("team")[["cum_pts", "cum_gd"]].last()
        order = latest.sort_values(["cum_pts", "cum_gd"], ascending=False)
        rk = {t: i + 1 for i, t in enumerate(order.index)}
        sel = g["Date"] == dt
        g.loc[sel, "rank"] = g.loc[sel, "team"].map(rk)
        if len(order) >= 18:
            g.loc[sel, "pts_of_18th"] = order["cum_pts"].iloc[17]
        if len(order) >= 4:
            g.loc[sel, "pts_of_4th"] = order["cum_pts"].iloc[3]
    return g


def add_standings(tm):
    tm = tm.sort_values(["Season", "Date", "match_id"]).reset_index(drop=True)
    tm = pd.concat([prematch_standing(g) for _, g in tm.groupby("Season", sort=False)], ignore_index=True)
    tm["pts_gap_to_safety"] = tm.cum_pts - tm.pts_of_18th
    tm["pts_gap_to_top4"] = tm.cum_pts - tm.pts_of_4th
    tm["match_round"] = tm.games_played.clip(upper=38)
    return tm


# --------------------------------------------------------------------------- #
#  3.5  merge back onto the match table as home_* / away_* / *_diff
# --------------------------------------------------------------------------- #
EWM_STATS = [f"{s}_ewm" for s in ROLL_STATS]
XG_FEATURES = ["xg_roll5", "xga_roll5", "xg_ewm", "xga_ewm",
               "xgd_roll5", "xgd_ewm", "finishing_roll5", "def_luck_roll5"]
TEAM_FEATURES = [
    f"pts_roll{WINDOW}", f"gf_roll{WINDOW}", f"ga_roll{WINDOW}", f"gd_roll{WINDOW}",
    f"shots_roll{WINDOW}", f"sot_roll{WINDOW}", f"corners_roll{WINDOW}",
    *EWM_STATS,
    "pts_std", "gf_std", "ga_std", "gd_std", "shots_std", "sot_std", "corners_std",
    f"venue_pts_roll{WINDOW}", "days_rest", "congestion_14d", "h2h_pts_roll5",
    "conv_roll5", "sot_rate_roll5", "save_pct_roll5", "wform_roll5",
    *XG_FEATURES,
    "cum_pts", "cum_gd", "rank", "pts_gap_to_safety", "pts_gap_to_top4",
    "match_round", "games_played",
]


def merge_features(matches, tm):
    home = tm[tm.venue == "home"].set_index("match_id")[TEAM_FEATURES].add_prefix("home_")
    away = tm[tm.venue == "away"].set_index("match_id")[TEAM_FEATURES].add_prefix("away_")
    model_df = matches.set_index("match_id").join(home).join(away).reset_index()
    diffs = {f"{col}_diff": model_df[f"home_{col}"] - model_df[f"away_{col}"] for col in TEAM_FEATURES}
    model_df = pd.concat([model_df, pd.DataFrame(diffs, index=model_df.index)], axis=1)
    # goal-model targets (post-match, used only as y in model.ipynb, never as X)
    model_df["home_goals"] = model_df["FTHG"]
    model_df["away_goals"] = model_df["FTAG"]
    model_df["target"] = model_df["FTR"]
    return model_df


# --------------------------------------------------------------------------- #
#  3.7  squad strength from the FPL Core Insights pre-season snapshot
# --------------------------------------------------------------------------- #
FPL_TEAM_MAP = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Burnley": "Burnley",
    "Chelsea": "Chelsea", "Coventry City": "Coventry", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Fulham": "Fulham", "Hull City": "Hull",
    "Ipswich": "Ipswich", "Ipswich Town": "Ipswich",
    "Leeds": "Leeds", "Leicester": "Leicester", "Liverpool": "Liverpool",
    "Man City": "Man City", "Man Utd": "Man United", "Newcastle": "Newcastle",
    "Nott'm Forest": "Nott'm Forest", "Southampton": "Southampton", "Spurs": "Tottenham",
    "Sunderland": "Sunderland", "West Ham": "West Ham", "Wolves": "Wolves",
}
POSITION_LINE = {"Goalkeeper": "gk", "Defender": "def", "Midfielder": "mid", "Forward": "fwd"}
FPL_STAT_COLS = ["id", "now_cost", "points_per_game"]   # pre-season-safe signals
LINE_TOP_N = {"def": 5, "mid": 5, "fwd": 3}
SQUAD_FEATURES = [
    "squad_price_total", "squad_price_top11", "bench_price", "squad_ppg",
    "gk_price", "def_price", "mid_price", "fwd_price",
]


def fpl_season_label(folder):
    """FPL season folder '2025-2026' -> our label '2025-26'."""
    a, b = folder.split("-")
    return f"{a}-{b[-2:]}"


def load_fpl_season(fpl_dir, folder, verbose=True):
    """Roster + GW1 snapshot for one FPL season, mapped to football-data team names."""
    base = os.path.join(fpl_dir, folder)
    players = pd.read_csv(os.path.join(base, "players.csv"))
    teams = pd.read_csv(os.path.join(base, "teams.csv"))
    stats = pd.read_csv(os.path.join(base, "playerstats.csv"), low_memory=False)
    gw1 = stats[stats["gw"] == stats["gw"].min()][FPL_STAT_COLS].copy()
    df = players.merge(gw1, left_on="player_id", right_on="id", how="left")
    df = df.merge(teams[["code", "name"]], left_on="team_code", right_on="code", how="left")
    df["team"] = df["name"].map(FPL_TEAM_MAP)
    df["Season"] = fpl_season_label(folder)
    df["line"] = df["position"].map(POSITION_LINE)
    df["price"] = pd.to_numeric(df["now_cost"], errors="coerce")          # already in GBP m
    df["ppg"] = pd.to_numeric(df["points_per_game"], errors="coerce")
    unmapped = sorted(set(df.loc[df["team"].isna(), "name"].dropna()))
    if unmapped and verbose:
        print(f"  ! {folder}: unmapped clubs {unmapped}")
    return df[df["team"].notna()].copy()


def aggregate_fpl_season(df):
    """One FPL season (roster + GW1 snapshot) -> one row per (Season, team)."""
    rows = []
    for (season, team), g in df.groupby(["Season", "team"]):
        g = g.sort_values("price", ascending=False)
        top11 = g.head(11)
        bench = g.iloc[11:20]
        rec = {
            "Season": season, "team": team,
            "squad_size": len(g),
            "squad_price_total": g.price.sum(min_count=1),
            "squad_price_top11": top11.price.sum(min_count=1),
            "bench_price": bench.price.sum(min_count=1) if len(bench) else np.nan,
            "squad_ppg": top11.ppg.mean(),
            "gk_price": g.loc[g.line == "gk", "price"].max(),
        }
        for line, n in LINE_TOP_N.items():
            rec[f"{line}_price"] = g.loc[g.line == line, "price"].head(n).mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def build_strength(fpl_dir, verbose=True):
    """All FPL season folders -> the team_season_strength table."""
    folders = sorted(d for d in os.listdir(fpl_dir) if os.path.isdir(os.path.join(fpl_dir, d)))
    return pd.concat([aggregate_fpl_season(load_fpl_season(fpl_dir, f, verbose)) for f in folders],
                     ignore_index=True)


def add_squad(model_df, strength):
    h = (strength.set_index(["Season", "team"])[SQUAD_FEATURES]
         .add_prefix("home_").rename_axis(index={"team": "HomeTeam"}))
    a = (strength.set_index(["Season", "team"])[SQUAD_FEATURES]
         .add_prefix("away_").rename_axis(index={"team": "AwayTeam"}))
    model_df = (model_df.merge(h.reset_index(), on=["Season", "HomeTeam"], how="left")
                        .merge(a.reset_index(), on=["Season", "AwayTeam"], how="left"))
    new = {f"{f}_diff": model_df[f"home_{f}"] - model_df[f"away_{f}"] for f in SQUAD_FEATURES}
    # FM-style line match-ups: my attack vs your defence
    new["home_fwd_vs_away_def"] = model_df.home_fwd_price - model_df.away_def_price
    new["away_fwd_vs_home_def"] = model_df.away_fwd_price - model_df.home_def_price
    new["att_edge_diff"] = new["home_fwd_vs_away_def"] - new["away_fwd_vs_home_def"]
    return pd.concat([model_df, pd.DataFrame(new, index=model_df.index)], axis=1)


# --------------------------------------------------------------------------- #
#  end to end
# --------------------------------------------------------------------------- #
def build_features(matches, strength=None):
    """Clean match table (played rows and, optionally, unplayed fixtures) -> model_df."""
    matches = matches.sort_values("Date", kind="mergesort").reset_index(drop=True)
    matches["match_id"] = matches.index
    tm = reshape_team_matches(matches)
    tm = add_rolling_form(tm)
    tm = add_venue_rest_h2h(tm)
    tm = add_shot_quality(tm)
    tm = add_xg(tm, matches, fit_xg_proxy(tm))
    matches = add_elo(matches)
    tm = add_opponent_weighted_form(tm, matches)
    matches = add_derby(matches)
    matches = add_market(matches)
    tm = add_standings(tm)
    model_df = merge_features(matches, tm)
    if strength is not None:
        model_df = add_squad(model_df, strength)
    return model_df
