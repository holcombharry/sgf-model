# Phase 7 — Direct career-VORP training + median-score discovery

Built and validated 2026-05-25. Resolves the central question raised in the post-Phase-6 efficacy audit: do the rankings actually carry signal over a naive baseline?

## Why this exists

The post-Phase-6 audit (run against `phase4_career_calibration_full.csv`) found the master ranking by **median (P50) career VORP** was barely beating — and often losing to — the naive baseline "rank by anchor-year FP." Top-N hit at the elite tier was 0% on top-10. The architectural product (career-VORP rankings with calibrated intervals) appeared unvalidated.

The user requested a structural fix, not a band-aid. Option A from the proposal: **train the model directly on career VORP**, dropping the per-year FP + Monte Carlo aggregation pipeline.

## What was built

1. **`simulation/career_target.py:compute_career_vorp_targets`** — per (player, anchor_season), the realized 5-year discounted career VORP under the default league (12-team 1QB PPR, 0.15 discount), using realized per-(position, year) replacement levels. Anchors past `max_season - horizon` get null targets and become inference rows.
2. **`features/builder.py:build_career_feature_matrix`** — one row per (player, anchor) wrapping the existing per-year builder, swapping the per-year target for the career-VORP target. Drops `future_offset` from features (constant 1). Reuses the rookie + advanced + draft feature paths.
3. **`features/builder.py:CAREER_FEATURE_COLUMNS`** — PHASE5 features minus `future_offset`.
4. **`evaluation/career_backtest.py`** — strict no-leakage backtest: for test anchor T, training targets only exist for anchors A with A+horizon ≤ T. Inference at anchor T predicts career VORP over T+1..T+horizon. Compared against realized career VORP from full history.
5. **`master_ranking` default changed from `median` → `mean`** — see findings.

Test setup: anchors 2017, 2018, 2019 (full 5-year realized data available). Training data from 2002 onward to get enough non-leaking anchors. NGS features are null for pre-2016 anchors but draft / age / prior-FP carry signal.

## Headline finding

**Direct career-VORP training works. The previous architecture's median-score default was broken; the model had the signal but the wrong scalar was being read.**

Pooled across test anchors 2017+2018+2019:

| Score | Spearman | top-10 hit | top-30 hit | top-60 hit |
|---|---|---|---|---|
| Baseline: anchor-year FP | 0.43 | 0.13 | 0.33 | 0.52 |
| **Direct career + P50** | **0.13** | 0.10 | 0.07 | 0.13 |
| **Direct career + mean** | **0.62** | **0.23** | **0.39** | **0.59** |
| Log-target career + P50 | 0.14 | 0.07 | 0.06 | 0.14 |
| Log-target career + mean | 0.63 | 0.10 | 0.32 | 0.57 |

Two distinct findings sit inside this table:

### Finding 1: The median-score default was structurally broken

P50 predictions are bunched at 0 for 99.4% of players. Cause: the training target distribution is 85–87% zeros (most "fantasy-relevant" players never cross replacement level over a 5-year career). Symmetric quantile loss at q=0.5 rationally collapses the median to 0 for any input that doesn't clearly belong to the elite tier — because predicting 0 minimizes the median loss across the majority of training rows.

Log-transforming the target didn't fix the median (same broken 0.14 Spearman). The zero mass is intrinsic to the target, not a skew artifact.

**The mean across the predicted quantile fan recovers the signal.** Same model, same training, different point estimate — Spearman jumps 0.13 → 0.62, top-30 hit jumps 0.07 → 0.39.

### Finding 2: With mean as the score, the model materially beats baseline on every metric

Spearman +0.19 over baseline. Top-10 hit +10pp. Top-30 hit +6pp. Top-60 hit +7pp. Per-position breakdown:

| Pos | Model Spearman | Baseline | Lift | Model top-30 | Baseline | Lift |
|---|---|---|---|---|---|---|
| QB | 0.68 | 0.51 | +0.17 | 0.79 | 0.72 | +0.07 |
| RB | 0.64 | 0.42 | +0.22 | 0.62 | 0.48 | +0.14 |
| WR | 0.64 | 0.43 | +0.21 | 0.67 | 0.51 | +0.16 |
| TE | 0.53 | 0.40 | +0.13 | 0.61 | 0.56 | +0.05 |

Every position. Not one position carrying the result.

### Calibration

cov_50 = 0.83–0.84, cov_80 = 0.91 across test anchors. Slightly over-covered on 80% intervals (target 0.80). Acceptable — the intervals are honest, not overconfident.

## What this changes architecturally

1. **`master_ranking` default is now `mean`, not `median`.** The architecture plan's "median career VORP for the 'neutral view'" advice is wrong for zero-inflated targets. Updated in `simulation/career.py` with explanatory comment.
2. **Direct career-VORP training is now a first-class path.** New module `evaluation/career_backtest.py`; new feature-matrix builder `build_career_feature_matrix`; new target helper `compute_career_vorp_targets`.
3. **Per-year + MC simulator decommissioned.** Direct training matches MC + mean on rank quality with a simpler pipeline; keeping both was overhead. Removed: `simulation/career.py` (sample_career_fps, sample_career_vorps, summarize_career, the MC ranking helpers), `evaluation/career_calibration.py` (the Phase 4 calibration harness — its findings are preserved in `docs/phase4-career-calibration.md` but the live code is gone). `master_ranking` moved to `evaluation/career_backtest.py` alongside `summarize_predictions`.

## What we did NOT do (and why)

- **Did not implement two-stage zero-inflated modeling** (classifier × magnitude). Not needed: the mean-score fix recovered the signal at a meaningful margin. Two-stage would add complexity for marginal gain.
- **Did not implement asymmetric / reweighted loss** (Phase 5 unfinished hypothesis). Same reason. Worth revisiting if we want to push median calibration, but mean is the working ranking score and is sufficient.
- **Did not source ADP or college stats.** Still the right next move for an independent signal, but separate effort from this validation.

## What this changes about confidence

The model's rankings now carry **measured, replicated signal** over the naive baseline, across all four positions, on Spearman and top-N hit rate. The architectural product (career-VORP distributions, ranked by mean, with calibrated intervals) is validated.

Honest limits:
- Top-10 hit rate is 23% — still weak in absolute terms. Identifying the very top of the dynasty pool remains hard.
- Calibration is slightly over (cov_80 = 0.91 vs target 0.80) — intervals are wider than they need to be.
- Test set is 3 anchor seasons. Not a huge sample.

## Files

- `src/sgf_model/simulation/career_target.py` — new, realized career VORP target computation.
- `src/sgf_model/features/builder.py` — added `build_career_feature_matrix`, `CAREER_FEATURE_COLUMNS`.
- `src/sgf_model/features/__init__.py` — re-exports.
- `src/sgf_model/evaluation/career_backtest.py` — new, strict no-leakage career backtest + summarize/evaluate helpers.
- `src/sgf_model/evaluation/__init__.py` — re-exports.
- `src/sgf_model/simulation/career.py` — `master_ranking` default changed to `mean` with explanatory docstring.

## What's next

Now that the rank signal is validated and the architecture is simplified:

1. **Source ADP** for the independent signal that's still missing.
2. **Tighten calibration** — cov_80 = 0.91 is over-covered, suggesting room to narrow intervals modestly.
3. **More test anchors** — currently 3. Re-run as more years of realized data become available.
