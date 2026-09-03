# Premier League match-outcome model

A three-way classifier (home win / draw / away win) for Premier League fixtures,
trained on 25 seasons of results and priced against the bookmaker's own line.

Every feature is knowable before kickoff. The model starts from the vig-free
market probability as a prior and only tries to improve on it. On held-out
seasons it edges the market on log loss by a hair — which, out of sample against
a settled market, is the whole point.

## The explainer pages

Five standalone HTML pages (no build step, no dependencies — open in a browser or
host as static files):

| Page | What it covers |
|------|----------------|
| [`index.html`](index.html) | Landing page linking the rest |
| [`how-it-works.html`](how-it-works.html) | The real pipeline: pre-kickoff features, the market prior, boosted trees, a Dixon–Coles goal grid, blend and calibration |
| [`pipeline.html`](pipeline.html) | A gentler walk-through: one match → features without leakage → the six-stage pipeline → the confusion matrix |
| [`random-forest.html`](random-forest.html) | Plain-language: how one decision tree works, the two kinds of randomness, and why the vote beats the best single tree |
| [`playground.html`](playground.html) | Why scikit-learn won't load under Windows Smart App Control, the conda-forge fix, and a browser playground for the toy ML concepts |
| [`predictions.html`](predictions.html) | Browse the model's blind predictions for every held-out match — probabilities, likely scoreline, the signals behind each pick, and whether it was right |

`predictions.html` embeds its data inline, so it works offline as a single file.

## Results

Numbers on seasons the model never trained on:

| Slice | Model log loss | Market log loss | Accuracy | Baseline (always home) |
|-------|:--:|:--:|:--:|:--:|
| Test — 2023-24 & 2024-25 (699 matches) | **0.946** | 0.949 | 56.1% | 43.9% |
| Holdout — 2025-26 (350 matches) | 1.019 | 1.016 | 48.3% | 42.0% |

Rolling-origin check (retrain before each season, score it alone): model log loss
stayed in 0.895–1.000 and beat the vig-free market in 5 of 7 seasons.

Draws are the hard class — no pre-kickoff feature correlates with a draw, so the
model treats a well-calibrated P(draw) as the deliverable rather than the hard
label.

## How it's built

```
explore.ipynb    → data/matches_clean.parquet     clean 26 season CSVs, sort by date
features.ipynb   → data/model_df.parquet          rolling form, Elo, xG, market, squad value
model.ipynb      → models/*.json                  train, blend, calibrate, export
pl_model.py                                       shared helpers (metrics, Dixon–Coles, Elo)
teaching-notebook.ipynb                           the toy versions behind playground.html
```

The stack is **numpy + pandas + xgboost** only. Isotonic calibration, the
Dixon–Coles scoreline matrix and the metrics are hand-rolled in numpy so nothing
depends on scipy.linalg or scikit-learn (both blocked by Smart App Control on the
machine this was built on — see `playground.html`).

### Reproduce

```bash
python -m venv venv && venv/Scripts/pip install numpy pandas xgboost pyarrow
# then run the notebooks in order: explore → features → model
```

`*.parquet` files are git-ignored — regenerate them by running the notebooks. The
raw season CSVs and the trained `models/*.json` are committed.

## Model summary

- **78 features**, all `home − away` differences: Elo gap, rolling and
  season-to-date form, xG form, days rest, league position, squad market value,
  and the bookmaker's vig-free price + closing-line move.
- **Model A** — XGBoost multi-class softprob, depth 3, `eta` 0.03, starting from
  the market log-odds as base margin.
- **Model B** — two Poisson regressors for expected home/away goals → a
  Dixon–Coles scoreline matrix (ρ = −0.11); H/D/A fall out of summing the
  triangles.
- **Blend** — `0.30·B + 0.70·A`, weight tuned on 2021–22 & 2022–23.
- **Calibration** — one temperature parameter, T ≈ 1.04.

## Data

- Match results & odds: [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
- Squad values / FPL data: FPL Core Insights (`data/fpl/`)

Both are used here for a non-commercial personal project. Check their terms before
redistributing the raw data.

## License

MIT — see [`LICENSE`](LICENSE). Applies to the code and explainer pages, not to
the third-party match data under `data/`.
