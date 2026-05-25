# Phase 3 — Results: multi-year FP + career Monte Carlo + master ranking

Built and tested 2026-05-25. Demonstrates the product differentiation: career VORP distributions with configurable ranking scores.

## What's new

1. **Multi-year feature matrix** — `build_feature_matrix(forecast_horizon=N)` emits one row per (player, anchor_season, future_offset) for offsets 1..N. `future_offset` is itself a feature so a single model handles all horizons.
2. **Per-position quantile FP model with offset feature** — `PHASE3_FEATURE_COLUMNS` adds `future_offset` to the Phase 2 feature set. The same QuantileFPModel trains on multi-offset data and predicts FP distributions for any horizon year.
3. **Career Monte Carlo simulator** — `sample_career_fps()` draws N=1000 rank-coupled trajectories per player (one quantile rank per player, applied across all years) via piecewise-linear inverse CDF interpolation between the model's P10/P25/P50/P75/P90.
4. **Career VORP aggregator** — `sample_career_vorps()` computes per-sample replacement levels (sorting players by sampled FP within position-year), VORPs each player's sampled FP against that, clips negatives, discounts, sums to career VORP.
5. **Master ranking scorer** — configurable: `mean`, `median`, `risk_adjusted_0.5`, `risk_adjusted_1.0`, `p25_floor`, `p_positive`.

## End-to-end results: 2023 projection trained on data through 2022

Trained on `forecast_horizon=5` (project 2023–2027), 1000 Monte Carlo simulations. PPR scoring, 12-team 1QB league, 15% annual discount.

### Top 15 by **median career VORP**

| Rank | Player | Pos | P10 | P50 | P90 | SD | P(>0) |
|---|---|---|---|---|---|---|---|
| 1 | J.Allen | QB | 147 | 667 | 868 | 262 | 1.00 |
| 2 | J.Jefferson | WR | 0 | 623 | 804 | 278 | 0.88 |
| 3 | T.Kelce | TE | 185 | 596 | 741 | 200 | 0.99 |
| 4 | P.Mahomes | QB | 153 | 545 | 811 | 241 | 1.00 |
| 5 | C.Lamb | WR | 57 | 483 | 706 | 230 | 0.97 |
| ... | | | | | | | |
| 12 | T.Higgins | WR | 0 | 373 | 622 | 223 | 0.84 |
| 13 | J.Jacobs | RB | 36 | 371 | 642 | 218 | 0.97 |
| 14 | D.Watson | QB | 0 | 361 | 778 | 317 | 0.73 |
| 15 | J.Herbert | QB | 36 | 360 | 689 | 237 | 0.99 |

Each row shows the configurable percentiles and `p_positive` = probability of finishing with positive career VORP. D.Watson at #14 has the widest spread in the top 15 (SD=317, p_positive=0.73) — consistent with his injury/legal-history uncertainty.

### Different rankings expose different tradeoffs

- **Risk-adjusted 0.5×SD**: Kelce moves to #2 (low variance), Rodgers enters at #12 (veteran QB with very tight distribution), McLaurin breaks top 15.
- **P25 floor**: Kelce takes #1 (highest floor among elites), T.Brady enters at #4 (steady QB with high floor — though see "limitations" below), Kittle and Aiyuk show up (medium ceilings but reliable floors).

The same model produces all three rankings from the same Monte Carlo samples; only the post-aggregation score differs. This is the configurability the architecture was designed for.

### Uncertainty widens for younger players (good)

P90–P10 spread of career VORP (median across players with positive expected VORP), by age tier:

| Position | Young (≤24) | Prime (25-28) | Veteran (29-31) | Aging (32+) |
|---|---|---|---|---|
| QB | 223 | 118 | 14 | 61 |
| WR | 139 | 18 | 0 | 0 |
| TE | 94 | 87 | 22 | 11 |
| RB | (n/a) | (n/a) | (n/a) | 24 |

Young players have materially wider spreads than veterans across positions. This is the expected behavior: a 23-year-old WR could become Justin Jefferson or wash out — that wider distribution carries through to career VORP. A 32-year-old veteran's trajectory is largely settled.

## Phase 3 exit criteria

| Criterion | Status |
|---|---|
| Career VORP distribution emits sensibly (top-30 interpretable, P10/P90 spreads age-aware) | **Met** |
| Career VORP calibration on retired players | **Deferred** — see limitations |
| Master ranking sortable by every configured score with sensible orderings | **Met** |

## Architecture validation

- Multi-year model: 14,361 training rows (vs. 6,243 in Phase 2 single-offset) — 2.3× data. Fit time 7.7s.
- Simulation: 0.2s to produce 1000 trajectories × 609 players × 5 years and aggregate to career VORP. Tiny compared to training.
- Schema: `(player_id, anchor_season, future_offset, target_season, target_fp, ...features)` is clean and extensible.

## Known limitations to address in Phase 4 / 5

1. **No retirement modeling.** T.Brady appears in 2023 rankings (he retired) because the model has no signal for "this player is done." Fix: either an explicit "still playing" hazard model OR position/age caps that mask projections beyond a position-appropriate retirement age. RBs especially need this — a 31-year-old RB with declining production is likely to be cut, not continue at 70% of his peak.

2. **Rank-coupled sampling overstates career correlation.** Pure rank-coupling (default `year_noise_alpha=0`) assumes a player who's 90th percentile in year 1 stays 90th percentile across years 2-5. Reality is somewhere between this and independence. The `year_noise_alpha` parameter exists to dial this; default value of 0 is conservative for career-VORP variance. Calibrate against retired-player career-VORP variance when we add the retired-player holdout (criterion #2 above).

3. **Career VORP calibration not yet validated.** Need a held-out set of retired players (e.g., retired 2010-2020), project their careers as-of their pre-final-season anchor, compare realized career VORP to model's predicted intervals. Confirms whether our 80% interval actually contains 80% of careers. Deferred — pipeline structure works, calibration tuning belongs in Phase 4/5.

4. **Sample-specific replacement is correct but slow at scale.** Sorting players within position per (simulation, year) is O(N log N) per slice; with 1000 sims × 5 years × 4 positions × ~150 players each = 20K sorts. Fast at this scale but if we go to 10k simulations or 10-year horizons, may want to amortize.

5. **D.Watson-style extreme uncertainty isn't fully captured.** His SD of 317 is wide but his actual outcome distribution should be bimodal (suspended vs. playing well). Rank-coupled draws on a unimodal quantile function can't represent that. Multi-modal distributions are a Phase 5+ concern.

## Files

- `src/sgf_model/features/builder.py` — multi-year feature matrix (rewritten to anchor-based logic)
- `src/sgf_model/data/loaders.py` — fixed player-name dedup (group_by player_id + season only)
- `src/sgf_model/models/quantile_fp.py` — unchanged, but now takes `PHASE3_FEATURE_COLUMNS` which includes `future_offset`
- `src/sgf_model/simulation/career.py` — new: sampling, VORP aggregation, master ranking
- `src/sgf_model/simulation/__init__.py` — public API

## Where we stand vs. v1

Direct per-year MAE comparison vs. v1 was the focus of Phases 1-2; Phase 3 adds the actual dynasty product on top. v1 produces:
- Point estimates of future season FP
- Discounted sum → dynasty value
- No uncertainty quantification

v2 (Phase 3) produces:
- Calibrated distributions of future season FP (P10/P25/P50/P75/P90)
- Monte Carlo career trajectories
- Career VORP distributions per player
- Configurable master rankings (median, risk-adjusted, floor, threshold-prob)
- Interpretability via feature importance (Phase 2)

The v2 model's per-year point estimates are roughly tied with v1 (better on RB, tied on WR, slightly worse on QB top-N hit rate). The differentiation is the distributional output — the actual product. v1 cannot produce it at all.

Phase 3 ships the core architecture. Phase 4 will harden it (retirement modeling, calibration validation, rookie integration, ADP if sourceable). Phase 5 will polish (CLI / snapshot schema updates, 2024 lockbox validation).
