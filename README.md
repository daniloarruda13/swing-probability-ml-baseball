# Swing probability in baseball

This project explores pitch-level swing probability using MLB Statcast data.
The committed notebook compares classifiers, calibrates a LightGBM model, applies
it to a later season, and uses SHAP to explore predicted swing behavior.

The reusable `swing_probability` package adds strict outcome labeling,
schema-controlled feature preparation, train-fitted preprocessing, a lightweight
probability baseline, and validated evaluation metrics.

## Important result context

The notebook is a historical executed analysis, not a self-contained benchmark:

- It fetched 2022, 2023, and part of 2024 from the live Statcast service. No raw
  dataset or trained model is committed.
- Its stored training table contained 1,426,485 pitches before complete-case
  deletion. Re-fetching later can produce different rows.
- The stored tuned LightGBM result reports 0.8503 random-holdout accuracy and the
  calibrated result reports a 0.1059 Brier score.
- The notebook’s original target omitted observed bunt swings (`foul_bunt`,
  `missed_bunt`, and `bunt_foul_tip`). Its cached model outputs therefore reflect
  those events incorrectly labeled as takes and should not be treated as current
  validated results.
- Dummy variables were created before the random split, and batter/pitcher IDs
  were treated as continuous numbers. The package defaults to train-fitted
  transformations and excludes player IDs unless explicitly requested.

Use a future season or another time-based holdout for the strongest estimate of
deployment performance. Random pitch-level splitting places the same players and
nearby games in both sets and answers an easier interpolation question.

## Outcome taxonomy

`swing_probability.labels` explicitly separates swings from takes. Every pitch
description printed by the committed notebook is covered. Unknown or missing
descriptions raise an error instead of silently becoming “no swing,” making
upstream Statcast vocabulary changes visible.

Observed swing outcomes include:

- balls put into play;
- swinging strikes and blocked swinging strikes;
- fouls and foul tips;
- foul bunts, missed bunts, and bunt foul tips.

## Install

Core preparation, testing, and the lightweight baseline require Python 3.10 or
newer:

```bash
python -m pip install -e .
```

Install the full notebook stack only when reproducing downloads, LightGBM,
Bayesian search, plots, or SHAP:

```bash
python -m pip install -r requirements.txt
```

The full analysis stack is intentionally optional because several operations are
large or slow.

## Package example

```python
from swing_probability import (
    evaluate_probabilities,
    make_baseline_pipeline,
    prepare_pitch_data,
)

# Train and evaluation frames should come from different time periods.
X_train, y_train = prepare_pitch_data(training_pitches)
X_future, y_future = prepare_pitch_data(future_pitches)

model = make_baseline_pipeline()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_future)[:, 1]
print(evaluate_probabilities(y_future, probabilities))
```

The pipeline learns numeric imputation, scaling, and categorical vocabularies
from training data only. Unseen pitch categories are handled during prediction.
Set `include_player_ids=True` in both preparation and pipeline construction only
when high-cardinality player identity is genuinely part of the estimand; doing so
can greatly increase memory use and complicate predictions for unseen players.

## Middle-middle pitches

```python
from swing_probability import middle_middle_pitches

subset = middle_middle_pitches(season_frame)
```

This helper removes rows missing plate/zone geometry, calculates each pitch’s
strike-zone center, and applies explicit horizontal and vertical half-widths.

## Tests and bounded validation

```bash
python -m unittest discover -s tests -v
```

The suite uses a small synthetic dataset. It does not query Statcast, train
LightGBM, run Bayesian optimization, perform 20-fold isotonic calibration, or
compute SHAP values.

## Notebook

To inspect or deliberately reproduce the historical workflow:

```bash
git clone https://github.com/daniloarruda13/swing-probability-ml-baseball.git
cd swing-probability-ml-baseball
jupyter notebook swing_probability_modeling.ipynb
```

Long-running sections are clearly identifiable in the notebook:

- three Statcast downloads;
- four-model comparison over more than one million pitches;
- 20-iteration, five-fold Bayesian LightGBM search;
- 20-fold isotonic calibration;
- SHAP calculations and plots.

Cache raw downloads outside Git and record retrieval dates if reproducing the
study. Review Statcast/MLB data terms before redistribution.

## Repository structure

- `swing_probability/labels.py`: strict swing/take definitions.
- `swing_probability/features.py`: stable schema and pitch subsets.
- `swing_probability/modeling.py`: preprocessing baseline and metrics.
- `tests/`: offline unit, model-smoke, and repository tests.
- `swing_probability_modeling.ipynb`: historical exploratory analysis.
- `pyproject.toml`: core and optional analysis dependencies.

## Interpretation

This is an exploratory modeling project, not a production decision system.
Feature importance and SHAP values describe model behavior, not causal effects.
Validate calibration and discrimination on a genuinely later season before using
predictions for scouting or in-game decisions.

## License

No software or data license is currently declared. Copyright remains with the
author; obtain permission before reuse beyond applicable legal rights.
