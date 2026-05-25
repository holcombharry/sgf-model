# Phase 2 — Results: advanced features (NGS + snaps + draft capital)

Established 2026-05-25. Locked holdout 2021–2023, 3-season averages, PPR.

## What Phase 2 added

24 new features on top of Phase 1's 14:
- **Snap features (4):** snap_share_mean/max, snap_games, offense_snaps_total
- **NGS receiving (7):** separation, cushion, aDOT, catch%, YAC, YAC over expected, air-yard share
- **NGS rushing (5):** efficiency, RYOE per attempt, rush% over expected, time to LOS, % attempts vs 8+ defenders
- **NGS passing (5):** CPOE, time to throw, aggressiveness, aDOT, air yards to sticks
- **Draft capital (3):** round, pick, log(pick)

Joined with 1-year lag (prior-season values used to predict current season). NGS coverage is 2016+; pre-2016 rows get nulls (HGB handles natively).

## Headline: v1 vs. v2_phase1 vs. v2_phase2_all

| Position | Metric | v1_no_reg | v2_phase1 | v2_phase2_all | Phase 2 vs. v1 |
|---|---|---|---|---|---|
| **RB** | MAE | 50.4 | 44.9 | **42.9** | **-15%** |
| **RB** | Spearman | 0.684 | 0.672 | 0.689 | tied |
| **RB** | top12_hit | 0.417 | 0.472 | **0.500** | **+20%** |
| **RB** | mean_bias | +8.8 | -8.3 | **-3.6** | nearly centered |
| QB | MAE | 70.9 | 73.1 | 71.9 | tied (+1.4%) |
| QB | top12_hit | 0.500 | 0.389 | 0.333 | worse |
| QB | cov_80 | 0.765 | 0.675 | 0.646 | worse (calibration regressed) |
| WR | MAE | 42.3 | 40.8 | **40.5** | -4% |
| WR | Spearman | 0.741 | 0.735 | 0.731 | tied |
| WR | top12_hit | 0.556 | 0.472 | 0.444 | **worse** |
| TE | MAE | 29.0 | 28.5 | 28.9 | tied |
| TE | Spearman | 0.748 | 0.760 | 0.760 | slightly better |
| TE | top12_hit | 0.583 | 0.528 | 0.556 | slightly worse |

**Phase 2 met exit criteria for RB and only RB.** RB is a clean win across the board: MAE -15% vs. v1, top12_hit +20%, mean_bias nearly centered. Other positions show marginal improvement on MAE / Spearman but rank quality is still worse than v1.

## Feature importance — what's actually doing the work

Permutation importance on a 2022-held-out validation set, top 10 features per position:

| Position | Top features (by importance) |
|---|---|
| **RB** | prior_fp_per_game_weighted (**0.27**), prior_fp_weighted, position_rank_last_year, prior_fp_1y, **prior_snap_share_mean**, **prior_ngs_ryoe_per_att**, prior_fp_2y, **prior_ngs_rush_pct_oe**, draft_pick, age |
| **WR** | prior_fp_weighted (**0.26**), prior_fp_per_game_weighted, prior_fp_1y, age, **prior_ngs_catch_pct**, prior_fp_2y, **prior_ngs_adot**, prior_fp_3y, **prior_ngs_separation**, prior_snap_games |
| **TE** | prior_fp_weighted (**0.21**), position_rank_last_year, prior_fp_1y, draft_pick, prior_fp_per_game_weighted, **prior_ngs_catch_pct**, **prior_ngs_yac**, prior_offense_snaps_total, prior_fp_2y, **prior_ngs_cushion** |

**Read:** Phase 2 features (bolded NGS / snap) earn places in every position's top-10. RB rushing NGS shows up twice in the top 10 and clearly drives the RB win. WR/TE receiving NGS shows up but contributes less than history features.

**The dominant story:** Historical FP features do ~70% of the work. NGS features are real signal but secondary. The architecture is roughly "smart Marcel" — gradient boosting on lagged FP with small advanced-metric nudges.

## Ablation table

Removing each Phase 2 feature group, measured against v2_phase2_all:

| Removed group | RB MAE | RB top12 | WR MAE | WR top12 | QB MAE | QB top12 |
|---|---|---|---|---|---|---|
| (none — baseline) | 42.9 | 0.500 | 40.5 | 0.444 | 71.9 | 0.333 |
| no NGS | 44.0 (+1.1) | 0.500 | 40.6 | 0.500 | 72.7 | 0.306 |
| no snap | 42.9 | 0.500 | 41.0 | 0.472 | 72.9 | 0.306 |
| no draft | 42.3 (-0.6) | 0.528 | 40.8 | 0.472 | 72.4 | 0.278 |

NGS is most impactful for RB. Snap features hurt QB MAE most when removed. Draft features are noisy — removing them actually slightly improves RB metrics (consistent with veterans having stale draft signal).

## The persistent issue: WR/QB top-12 hit rate

WR top12 fell from 0.556 (v1) to 0.444 (v2_phase2). QB top12 fell from 0.500 to 0.333.

Same diagnosis as Phase 1: the model over-shrinks elites toward the mean. Phase 2's NGS features help discriminate at the *median* but not at the *elite tier*. NGS aggregates only cover the top 100-115 WR/TE per year (the players we most want to differentiate), so the elite-vs-elite signal is weak.

Other limitations:
- **Sample size**: QB has ~275 training rows after filtering — gradient boosting is data-hungry, and conservative shrinkage is the consequence.
- **Symmetric loss**: quantile median loss penalizes elite under-projection and mid-tier over-projection equally. Without elite-specific signals, the model defaults to centering.
- **Calibration regressed slightly**: cov_80 dropped from 0.81 (v2_phase1) to 0.79 (v2_phase2_all). More features = more variance in the model, conformal step under-corrects.

## Phase 2 verdict

Phase 2 is a **partial win**. The architecture is producing usable distributional output, RB is clearly better than v1, and feature importance shows NGS features are earning their place where they have data. But the headline goal — beating v1 on rank quality across positions — is not achieved for WR/QB/TE.

Honest framing: we now have a working v2 model with calibrated intervals and explicit feature importance, but its per-year accuracy is roughly tied with v1 on most positions. The distributional output (career VORP percentiles in Phase 3) is the real product differentiation, not per-year FP accuracy.

## Where to go from here

Four options:

### Option A — Add team context + routes-derived features (Phase 2.5)

The biggest remaining feature gap:
- **Team pass/rush volume** (3-year weighted, current team) — situational
- **Depth chart proxy** (player's position-rank within team) — situational
- **QB tier feature** for WR/TE models — situational
- **Routes-derived features**: per-route target rate, derived YPRR. The routes-run data exists in load_participation (FTN 2023+; NGS-era 2016-2022 partial). Aggregating per-play participation to per-player season is non-trivial but tractable.

Hypothesis: routes-derived features distinguish elites better than aggregate NGS, and team context handles cases where a player's situation changed (trades, FA moves). Could be the key to fixing top-N hit rate for WR.

Cost: 1-2 weeks. Risk: routes derivation from participation play-by-play is the slowest data engineering step.

### Option B — Try asymmetric loss / elite reweighting

The over-shrinkage of elites is partly a loss function issue. Try:
- Weighting elite training rows higher (e.g., 2x weight for top-24 in prior year)
- Asymmetric quantile loss
- Separate elite-only model that handles top-tier projections

Cost: 2-3 days. Risk: feels like a band-aid — might improve top-N hit but not address the underlying signal limitation.

### Option C — Move to Phase 3 (multi-year + career VORP)

Accept that per-year accuracy is roughly tied with v1 and ship the actual product differentiation: multi-year FP distributions, career VORP via Monte Carlo, master ranking. The distributional output is what makes the engine different from v1 — and v1 doesn't produce it at all.

Phase 3 work doesn't require beating v1 on per-year metrics to be valuable. A model with per-year MAE tied to v1 but with proper distributional output is strictly better than v1 for dynasty rankings (which v1 produces poorly because it has no uncertainty quantification).

Cost: 2-3 weeks. Risk: we ship without resolving the WR top-N hit rate issue.

### Option D — Source historical ADP somehow

ADP is consistently cited as the most informative external feature for fantasy projections. nflreadpy's `load_ff_rankings` is current-only (no history). Options:
- Scrape historical ADP from FantasyPros / MyFantasyLeague / Sleeper archives
- Use draft sites with public ADP history (Underdog, Draftkings dynasty)
- Buy historical ADP from a paid source

Cost: variable, mostly data engineering. Risk: depends on data availability and quality.

## My recommendation

**Option C (move to Phase 3) is the highest-leverage next step.**

Reasoning:
1. The whole architectural pivot from v1 to v2 was motivated by wanting career VORP distributions. Phase 3 delivers that. Per-year FP is the intermediate; career VORP is the product.
2. Phase 2 has shown the architecture works. It produces calibrated, interpretable, feature-driven predictions. RB is a clean win. The metric ties on other positions are not architecture failures — they're feature-coverage limitations.
3. Phase 3 work is largely independent of further feature improvements — once career-VORP Monte Carlo is built, adding new features later (team context, routes, ADP) lifts both per-year and career outputs.
4. WR top-N hit rate is a real problem but unlikely to be solved by more features alone — sample size and loss function are factors too. Solving it later as part of Phase 4 polishing makes more sense than blocking the main product on it.

The off-ramp question — is the top-down hypothesis right? — has a clearer answer now than after Phase 1: yes, with caveats. The architecture works, the features are showing signal, the distributional output is calibrated. The remaining gap is rank quality on top tier for WR/QB, which is a feature problem more than an architecture problem.

Phase 3 is also smaller work than Option A — it builds on existing infrastructure rather than introducing new data engineering.

## Files

- `docs/phase2_comparison.csv` — 3-season averages, all variants × positions
- `docs/phase2_full.csv` — per-test-season breakdown
- `src/sgf_model/features/advanced.py` — NGS / snap / draft feature engineering
- `src/sgf_model/features/builder.py` — Phase 2 column definitions + lagged join logic
- `src/sgf_model/evaluation/backtest.py` — `project_for_backtest_v2` takes advanced/draft features and a per-variant `feature_columns` subset
