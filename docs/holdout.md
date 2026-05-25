# Backtest Holdout — LOCKED

Locked 2026-05-25. This is the immutable test window for the project. Do not modify without explicit migration plan and reporting both pre- and post-change metrics for direct comparison.

## Holdout window

**Test seasons: 2021, 2022, 2023** (3 consecutive seasons).

For each test season T, the model trains on seasons through T-1 only. Age curves, regression priors, talent-layer fits, archetype clusters, situation models — every component refits on the train window.

**Reserved lockbox: 2024.** Never used during development. Held back for a single end-of-project validation run to confirm we didn't overfit to the 2021-23 backtest.

## Why this window

- **Three seasons** averages out single-season noise. One-season backtests are easily fooled by an outlier year.
- **2021-2023** is the most recent window where:
  - NGS data exists from 2016, giving 5+ years of training data with advanced metrics for every backtest year.
  - Participation FTN routes data (2023+) is available for at least one test season.
  - The NFL game in this window resembles the current game (post-rule-change passing era, similar pace and personnel trends).
- **2024 lockbox** protects against the natural drift toward unconsciously tuning to the test window. If 2021-23 metrics improve but 2024 metrics don't, we overfit.

## What this constrains

- Pre-2016 history is fine to use for the v1-style model (Marcel history + age curves). Talent-layer features only fit on 2016+.
- Comp pool for Phase 4 comps draws from 2016 through T-1 for each backtest year. Earlier seasons available but with NGS columns null.
- Any change to the backtest harness (new metrics, new variants, new bucketing) is allowed and encouraged. Holdout *seasons* are locked; *evaluation* can evolve.

## Reporting standard

Every model variant must report:
- Per-position MAE, RMSE, bias on FP
- Spearman rank correlation per position
- Top-12 / top-24 / top-36 hit rate per position
- Calibration of 50% and 80% predictive intervals (coverage rate)
- Metrics broken out: with-injuries vs. without-injuries
- Per-component MAE once components exist (Phase 1+)

Averaged across the three test seasons. Single-season numbers are informational but variant comparisons use the 3-season average.
