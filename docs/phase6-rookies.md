# Phase 6 — Rookie handling (synthetic pre-rookie anchors)

Built and validated 2026-05-25. Closes the Phase 4 deliverable on rookie projection that was deferred.

## The gap

`features/builder.py` filtered out training rows where prior_fp was all-null and experience < 1. Combined with `_build_target_rows` only emitting inference rows for players with a played-in-prior-season anchor, this meant **the model never trained on a rookie season and never projected an incoming rookie.** Rookies were silently absent from both training and inference.

## Fix: synthetic pre-rookie anchors

For each drafted fantasy-position player, manufacture an `anchor_season = draft_year − 1` row.

- **Training (past rookies):** `target_season = anchor + offset`, target_fp = realized FP (or 0 with first-inactive-only mask). The model learns the dynamic from "no NFL history + draft capital + age + position" to year 1/2/3 outcomes from real outcomes.
- **Inference (incoming rookies, `draft_year == inference_season`):** anchor = `inference_season − 1`, target_fp = null. Slots into the existing inference path.

A new `is_rookie` binary feature is added to all `PHASE*_FEATURE_COLUMNS` so gradient boosting can branch cleanly on rookie-vs-veteran instead of inferring it from the pattern of nulls.

No leakage: draft capital is known at the synthetic anchor (April of draft year), strictly before any games are played.

## Results

### Rookie-only MAE vs baseline-of-zero (2021–2023 test seasons)

| Season | n | Model MAE | Baseline-zero MAE | Lift | cov_80 |
|---|---|---|---|---|---|
| 2021 | 77 | 41.7 | 62.4 | **−33%** | 0.86 |
| 2022 | 77 | 37.7 | 55.0 | **−31%** | 0.90 |
| 2023 | 81 | 41.4 | 69.8 | **−41%** | 0.84 |

Coverage is slightly over nominal (target 0.80) — intervals are reasonable, not too tight.

### Face validity (top 5 by predicted FP)

- **2021**: Trevor Lawrence (184 pred → 205 actual), Trey Lance (176→63, injury), Zach Wilson (176→150), Justin Fields (159→137), Javonte Williams (154→207).
- **2022**: Breece Hall (159→117), Walker (147→203), Drake London (142→183), Jameson Williams (140→15, torn ACL), Wan'Dale Robinson (129→52).
- **2023**: Bryce Young (174→164), Anthony Richardson (170→71, injury), C.J. Stroud (170→282 — underrated), Bijan Robinson (146→252), Jahmyr Gibbs (144→244).

Big misses cluster on injuries (Lance, Williams, Richardson) and rookie phenomena the model can't see in features alone (Stroud, Bijan, Gibbs). These are the structural limits of "project from draft capital + age + position only" — addressing them needs college stats or pre-draft scout grades, neither of which we source today.

### End-to-end effect on existing metrics

Running the full v2 backtest on 2022–2023 with rookies plumbed in vs. without:

- **cov_80 improved in 7 of 8 (position × season) cells.** QB cov_80 went 0.65→0.70 (2022) and 0.68→0.75 (2023) — directly attacks the elite-undercoverage from `docs/phase4-career-calibration.md`.
- **top12_hit moved around: 4 cells better, 2 worse, 2 tied.** Biggest wins: QB 2022 +0.17, RB 2023 +0.08, WR 2022 +0.08. Biggest loss: TE 2023 −0.08, WR 2023 −0.08.
- **MAE moved within ±3 FP per position-season** — within run-to-run noise, and rookies are now in the denominator (a harder eval set).

No consistent regression on veteran-driven metrics. The cov_80 improvement is the headline — more training data tightens the distributional claim.

## Known limitations

1. **UDFAs are excluded.** `compute_rookies` requires `draft_year is not null`. Modeling UDFAs would need a separate signal (roster status, training-camp reports) that we don't source.
2. **No college features.** The Phase 4 plan calls for `college stats proxy if cheap`. Not implemented here. Without it, two 1st-round WRs with very different college production look identical to the model — pick number is the only discriminator within a draft cohort.
3. **Year-2+ rookie projections are even thinner.** Year-2 forecasts from a synthetic pre-rookie anchor have no in-NFL signal either, just `future_offset` + draft capital. Once a rookie plays one year, the existing veteran path takes over with prior_fp populated — that handoff is implicit, not designed.
4. **Same model fits rookies + vets.** No separate position-specific rookie loss or reweighting. If post-launch we see rookies systematically under-/over-projected, sample weighting on rookie rows is the next lever.

## Files

- `src/sgf_model/features/advanced.py` — `compute_rookies()` added.
- `src/sgf_model/features/builder.py` — `_build_rookie_anchor_rows()` added, `is_rookie` added to PHASE1 (propagates), master filter updated, `rookies` param added to `build_feature_matrix`.
- `src/sgf_model/features/__init__.py` — re-exports `compute_rookies`.
- `src/sgf_model/evaluation/backtest.py` — `rookies` param plumbed through `project_for_backtest_v2` and `run_backtest_v2`.
- `src/sgf_model/evaluation/career_calibration.py` — `rookies` param plumbed through `career_calibration_at_anchor`.

## What's next

The structural rookie gap is closed. Next opportunities, ranked by likely lift:

1. **Asymmetric quantile loss / elite reweighting** — Phase 5 doc's unfinished hypothesis. Should keep pushing on elite under-coverage even with rookies handled.
2. **Sourcing college stats** — would let the model discriminate within a draft cohort. Currently 1.04 and 1.06 WRs project nearly identically.
3. **Sourcing historical ADP** — independent signal not derivable from per-game stats.
