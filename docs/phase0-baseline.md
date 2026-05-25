# Phase 0 — Baseline metrics (v1 model)

Established 2026-05-25 on the locked holdout (test seasons 2021–2023, three-season averages, PPR scoring). Future phases will be compared against these numbers.

Full per-season data: `phase0_baseline.csv`. 3-season averages: `phase0_baseline_avg.csv`.

## Headline (3-season average, `all` injury bucket, `no_reg` variant)

| Position | n | MAE | RMSE | Spearman | top12_hit | top24_hit | cov_50 | cov_80 |
|---|---|---|---|---|---|---|---|---|
| QB | 65 | 64.6 | 86.8 | 0.71 | 0.53 | 0.74 | 0.53 | 0.78 |
| RB | 120 | 49.1 | 67.5 | 0.68 | 0.42 | 0.61 | 0.58 | 0.82 |
| WR | 180 | 40.7 | 54.3 | 0.75 | 0.56 | 0.68 | 0.55 | 0.85 |
| TE | 97 | 27.8 | 38.7 | 0.76 | 0.58 | 0.76 | 0.61 | 0.86 |
| ALL | 462 | 43.5 | 61.1 | 0.73 | 0.28 | 0.39 | 0.57 | 0.83 |

Per-position rows are the right comparison target; `ALL` cross-position rows conflate QB-vs-RB scale and are mostly informational.

## Key findings from the baseline

### 1. The current regression layer provides ~zero lift

Variants tested:
- `no_reg` — no empirical-Bayes shrinkage
- `reg_0_5x`, `reg_1x`, `reg_2x` — shrinkage at 0.5×, 1×, 2× the EB-estimated N

Across all four variants, MAE differences are <0.1 FP, Spearman differences are <0.001, top-N hit rate is identical. The shrinkage isn't broken — it's just being applied to raw per-game stats where the noise/signal split doesn't materially differ from a player's own history.

This is the empirical justification for Phase 1: decompose into opportunity (volume) × efficiency (per-opportunity rates) so that shrinkage can be applied with the *correct* stabilization rate per metric (e.g., heavy shrinkage on TD rate, light shrinkage on target share).

### 2. Calibration is close but slightly wide at 50%

`cov_50` is 0.53–0.61 across positions vs. target 0.5. `cov_80` is 0.78–0.86 vs. target 0.8. Intervals are slightly too wide, which is the safer side to err on but indicates room to tighten variance estimation. Phase 4 (comp-based intervals) should improve this materially.

### 3. Rank quality is the upside

Spearman 0.71–0.76 per position is reasonable but top12_hit at 0.42–0.58 means we're missing roughly half the elite players each year. That's where the project's value lies — Phase 2 talent-layer and Phase 4 comp-based forecasting should move these numbers most.

### 4. Mean bias is positive (~5 FP) across the board

Model systematically over-projects. Likely a games-played issue (Phase 3 games-played model should fix) and/or no aging adjustment to top-of-distribution. Worth keeping an eye on as components evolve.

## Improvement targets (post-Phase-4)

These are the numbers we expect to beat. Conservative targets:

| Metric | Baseline | Phase 4 target |
|---|---|---|
| MAE per position | 28–65 | 20% lower |
| Spearman per position | 0.68–0.76 | ≥0.80 |
| top12_hit per position | 0.42–0.58 | ≥0.65 |
| cov_50 deviation from 0.5 | 0.05–0.11 | ≤0.05 |
| cov_80 deviation from 0.8 | 0.02–0.06 | ≤0.04 |
| mean_bias | +5 FP | ±2 FP |

Aggressive targets (if comp-based projection works as hoped):
- top12_hit ≥0.70 per position
- MAE per position 25% lower than baseline

## Reproducing this baseline

```python
from sgf_model.data import load_player_seasons, load_weekly_stats, filter_fantasy_positions
from sgf_model.scoring import PRESETS
from sgf_model.evaluation.backtest import run_backtest

ps = load_player_seasons(start=2010, end=2023)
weekly = filter_fantasy_positions(load_weekly_stats(seasons=list(range(2010, 2024))))

variants = {
    "no_reg":  {"use_regression": False},
    "reg_1x":  {"use_regression": True, "n_multiplier": 1.0},
}
result = run_backtest(
    ps, weekly,
    test_seasons=[2021, 2022, 2023],
    variants=variants,
    scoring=PRESETS["ppr"],
)
```

Runtime: ~35s after the first nflreadpy load (subsequent loads cached).
