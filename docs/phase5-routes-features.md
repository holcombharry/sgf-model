# Phase 5 — Routes-derived features (Option 3 from Phase 4 follow-ups)

Implemented and validated 2026-05-25. **The fix did not deliver the expected lift.** This is an honest write-up of what we did, what it didn't do, and what we learned.

## What we built

Four new features derived from `load_participation()`:
- **routes_run** — count of pass plays where the player was in `offense_players` AND the play had a logged route concept (filtered to 2016+).
- **yprr** — receiving_yards / routes_run.
- **targets_per_route** — targets / routes_run.
- **td_rate_per_route** — receiving_tds / routes_run.

Joined to the master feature matrix at anchor_season with `prior_` prefix. ~5.8s additional pipeline time. Data quality sanity-checked: Cooper Kupp 2021 = 727 routes / 2.68 YPRR (historic season), Jefferson 2022 = 2.82 YPRR, Kelce 2020 = 2.19 YPRR. These match published PFF-style figures.

## Backtest result vs. Phase 4 (no routes)

| Position | Metric | Phase 4 (no routes) | Phase 5 (routes) | Δ |
|---|---|---|---|---|
| WR | MAE | 40.5 | 40.6 | +0.1 (tied) |
| WR | Spearman | 0.747 | 0.741 | -0.006 (tied) |
| WR | top12_hit | 0.500 | 0.500 | 0 (tied) |
| RB | MAE | 45.4 | 45.0 | -0.4 (tied) |
| RB | top12_hit | 0.528 | 0.500 | -0.028 (slightly worse) |
| TE | top12_hit | 0.667 | 0.611 | -0.056 (slightly worse) |
| QB | Spearman | 0.681 | 0.702 | +0.021 (slightly better) |

**No meaningful improvement on any position.** Some metrics tied, some marginally worse, one marginally better. Overall: the data is in the model, the model didn't use it productively.

## Career calibration (the test Phase 5 was supposed to fix)

| Cohort | Phase 4 cov_80 | Phase 5 cov_80 | Δ |
|---|---|---|---|
| All test cases | 0.95 | 0.94 | -0.01 |
| Useful (realized > 0) | 0.84 | 0.81 | -0.03 |
| **Top 30 (elite)** | **0.60** | **0.57** | **-0.03 (slightly worse)** |

The elite-tier under-coverage that motivated this work got slightly *worse* with routes features added. cov_80 on top 30 fell from 0.60 to 0.57.

## Why the recommendation was wrong

My pre-recommendation was Option 3 over Options 1/2 because routes features address the root cause (elite discrimination). The data is unambiguously informative — YPRR clearly separates elite from average receivers (Jefferson 2.82 vs. league-average ~1.4). So why didn't it help?

Three reinforcing causes:

1. **Redundancy with existing features.** The model already had `prior_fp_per_game_weighted`, `prior_fp_1y`, NGS receiving (catch%, separation, aDOT). These collectively encode most of the same signal as YPRR — "this is an elite receiver." Adding a more granular measure of the same concept doesn't help when gradient boosting already extracts the signal from the existing features.

2. **Sample-size noise > signal.** Training set is ~5,000 rows for the per-year backtest. Adding 4 correlated features tilts toward variance, not signal. With limited data, more features can hurt.

3. **The bottleneck wasn't features.** The persistent WR top-12 gap and elite under-coverage have a different root cause than I diagnosed. Probably the **symmetric quantile loss + sparse elite training examples**. The model has the signal; the loss function and data scarcity don't reward it pushing predictions high enough on elite-feature profiles.

This is a meaningful update to the architecture's understanding. Adding more talent features to address elite under-coverage isn't the right lever — the signal is already there. **The issue is downstream of features.**

## What this changes about next steps

The original ranked Phase 4 follow-up options were:
1. Elite-stratified conformal calibration (band-aid, fast)
2. Asymmetric loss / elite reweighting (medium)
3. Routes-derived features (root cause — most upside) ← we just did this

We did #3 expecting the biggest lift; it gave none. Reordering by what's still likely to help:

**Most likely to work now:**
- **Asymmetric loss / elite reweighting** (was Option 2). The diagnostic above points at the loss function as the actual bottleneck. Training upper-quantile models with higher weight on elite training rows (e.g., 3× for prior-year top-12) should push P90 ceilings higher on elite-feature profiles. Without changing the loss or sample weights, more features won't fix this.
- **Sourcing historical ADP** (deferred earlier). ADP is the only feature that's plausibly *additive* with what we have — it's an external signal, not derived from the same per-game stats the model already uses. If breakouts and busts show up in ADP movements first, the model would learn that.

**Less likely to work now:**
- **Elite-stratified conformal calibration**. Still cheap and would visually fix top-tier cov_80, but it's a calibration patch, not a model fix. Won't improve rank quality (top-12 hit), only intervals.

**Tabled:**
- More feature derivation work. The data has limited additional juice given correlation with existing features.

## Files

- `src/sgf_model/features/advanced.py` — `compute_route_features` added.
- `src/sgf_model/features/builder.py` — `PHASE5_FEATURE_COLUMNS = PHASE3 + routes`.
- `src/sgf_model/features/__init__.py` — re-exports.

The routes features are kept in the codebase (low cost, more discriminating data is rarely bad). They just don't pull the metrics we hoped.

## Honest summary

I recommended Option 3 over Options 1/2 because it addressed the root. The result tells us my diagnosis of "missing elite features" was wrong — the bottleneck is the loss function, not features. The next move should be asymmetric quantile loss or elite reweighting (Option 2). If that doesn't work, the next-next move is sourcing historical ADP for an independent signal.
