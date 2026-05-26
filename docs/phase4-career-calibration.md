# Phase 4 — Career VORP calibration validation

Run 2026-05-25. The deferred Phase 3 exit criterion. Asks: when the model emits an 80% interval on career VORP, does the realized career VORP fall in that interval 80% of the time?

## Setup

- **Anchors:** 2017, 2018 (both have ≥5 years of realized data after them).
- **Horizon:** 5 years.
- **Method:** for each anchor, train v2 model on data ≤ anchor, predict career VORP distribution per active player, compute realized career VORP using actual subsequent seasons + actual league replacement levels per (position, year).
- **Test cases:** 1,128 (player, anchor) pairs.

## Headline result

| Cohort | n | cov_50 (target 0.5) | cov_80 (target 0.8) |
|---|---|---|---|
| **All test cases** | 1,128 | 0.86 | 0.95 |
| Realized = 0 | 784 (70%) | 0.99 | 1.00 |
| Realized > 0 (dynasty-relevant) | 346 | **0.57** | **0.84** |
| Realized > 100 | 155 | 0.51 | 0.79 |
| **Realized top 30** | 30 | **0.27** | **0.60** |

**The headline "94% covered" is misleading.** 70% of test cases were "roster filler" players who scored 0 career VORP — these are trivially covered because both the model's P10 and the realized are 0. The actual calibration story emerges when you stratify by realized outcome.

## What the stratification tells us

1. **For dynasty-relevant players (realized > 0): calibration is close to nominal.** cov_50 = 0.57 (target 0.5), cov_80 = 0.84 (target 0.8). Slight over-coverage but within ~5 percentage points of target. The architecture's main claim — that career VORP distributions are meaningfully calibrated — holds for the players who actually matter to a dynasty roster.

2. **At the elite tier (realized > 300, n=60): cov_80 drops to 0.67.** Top 30 by realized: cov_80 = 0.60. The model **under-covers elites by 13-20 percentage points** — same elite over-shrinkage pattern we saw in WR top-12 hit rate.

3. **Per-position (useful cohort):**

| Position | n | cov_50 | cov_80 | realized median | predicted P50 median |
|---|---|---|---|---|---|
| QB | 58 | 0.40 | 0.67 | 121 | 183 |
| RB | 98 | 0.62 | 0.88 | 74 | 0 |
| TE | 66 | 0.52 | 0.73 | 67 | 0 |
| WR | 124 | 0.65 | 0.94 | 92 | 43 |

QB is the worst-calibrated (cov_80 = 0.67) — under-confident below the target. WR is the most over-covered. TE/RB are closer to target.

## Top 15 elite misses (model interval vs. realized)

| Player | Pos | Anchor | P10 | P50 | P90 | Realized | In 80%? |
|---|---|---|---|---|---|---|---|
| T.Kelce | TE | 2017 | 0 | 319 | 434 | 682 | ❌ |
| T.Kelce | TE | 2018 | 0 | 304 | 477 | 625 | ❌ |
| C.McCaffrey | RB | 2017 | 0 | 341 | 663 | 617 | ✓ |
| J.Allen | QB | 2018 | 0 | 29 | 485 | 610 | ❌ |
| D.Adams | WR | 2017 | 0 | 254 | 538 | 605 | ❌ |
| T.Hill | WR | 2017 | 0 | 360 | 554 | 543 | ✓ |
| D.Henry | RB | 2018 | 0 | 162 | 553 | 536 | ✓ |
| A.Ekeler | RB | 2018 | 0 | 88 | 384 | 533 | ❌ |
| P.Mahomes | QB | 2018 | 0 | 226 | 590 | 509 | ✓ |
| E.Elliott | RB | 2017 | 0 | 459 | 943 | 509 | ✓ |

J.Allen 2018 anchor is the most striking miss — model said P90=485, he realized 610. Pre-2019 Allen looked mediocre statistically; his breakout wasn't visible in 2018 data. This is the "breakout we couldn't see coming" class — they're inherently hard.

Kelce twice: even with the retirement fix, the model's intervals on his career VORP from 2017/2018 anchors didn't extend high enough to capture his actual outcomes. The model's P90 ceiling underestimates elite-tier upside.

## What this tells us about the architecture

**The good news.** The architecture's main distributional claim works for the bulk of players. For dynasty-relevant players (realized > 0), 84% fall in the 80% interval — within 5 percentage points of target. We can ship distributional output as a genuine model claim, not just a visualization.

**The bad news.** Elites are systematically under-projected on the upper tail. Same pattern as WR top-12 hit rate: the model's median predictions for elite-looking players are too pulled-to-the-mean, AND the upper tail isn't wide enough to compensate. P90 ceilings are too low.

**Hypothesis for why.** Three reinforcing factors:
1. **Symmetric quantile loss** doesn't reward higher P90 on elite-feature profiles enough.
2. **Sparse data at extreme career outcomes** — only a few Kelce/Mahomes-tier careers exist in training data, so the upper quantile predictions are conservative.
3. **Conformal calibration was per-year, not per-career-VORP.** The conformal widening is correct on per-year FP intervals but doesn't necessarily produce correctly-calibrated career sums.

## Phase 3 exit criterion #2: status

**Conditionally met.** The model's career VORP distributions are calibrated for the bulk of dynasty-relevant players. They are NOT calibrated at the elite top tier. Whether this is "acceptable" depends on use case:

- For ranking the middle of the dynasty pool (ranks 30-150): the model's intervals are trustworthy.
- For separating elites from elites (ranks 1-20): intervals are too narrow at the top; treat with skepticism.
- For ranking the bottom (ranks 150+): irrelevant — most are near zero.

## Files

- `docs/phase4_career_calibration_full.csv` — per-test-case (1,128 rows)
- `docs/phase4_career_calibration_summary.csv` — aggregated coverage stats
- `src/sgf_model/evaluation/career_calibration.py` — the validation function

## What to do about the elite-tier under-coverage

Three options:

1. **Widen elite-feature intervals with elite-stratified conformal calibration.** Fit separate conformal offsets for "high-feature-importance" subsets (e.g., players in top-12 last year). Quick fix. Doesn't address the root cause (symmetric loss).

2. **Asymmetric quantile loss / elite reweighting.** Train upper-quantile models with higher weight on elite training rows so the P90 ceiling extends. Moderate effort.

3. **Add per-route / target-share features (the deferred Phase 2 work).** Better elite discrimination at the feature level should let the model push elite predictions higher organically. Most effort, most upside.

For a short-term ship: option 1 (elite-stratified calibration) is cheapest and tightens the most-visible defect. Option 3 is the right long-term answer but requires the routes-derivation work.
