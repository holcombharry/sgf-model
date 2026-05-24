"""Out-of-sample backtest: project a test season using only earlier data,
compare to actuals, report MAE/RMSE per position and variant.

For each test season T:
    - Train on player_seasons / weekly through T-1
    - Refit age curves, regression priors, history baselines from training only
    - Project T (one season ahead)
    - Compare projected fantasy points to actuals for players who actually played

This is the gold-standard validation for projection quality. Multiple test
seasons average out single-season noise. Variant comparison (no-regression
vs. shrinkage strengths) directly answers "does the regression layer help,
and is N calibrated correctly?"
"""

from __future__ import annotations

import polars as pl

from sgf_model.curves import fit_age_curves
from sgf_model.projections import fit_regression_priors, project_players
from sgf_model.scoring import ScoringConfig, score_projections

# Stat columns from `load_player_seasons` that need a `_season` suffix added before
# `score_projections` can compute fantasy points. (The function expects column
# names of the form `<stat>_season` and `<stat>_per_game`.)
_ACTUAL_STAT_COLS: tuple[str, ...] = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)


def project_for_backtest(
    player_seasons: pl.DataFrame,
    weekly_stats: pl.DataFrame,
    test_season: int,
    use_regression: bool = True,
    n_multiplier: float = 1.0,
    history_weights: tuple[float, ...] = (5.0, 4.0, 3.0),
) -> pl.DataFrame:
    """One-season-ahead projection using only data through `test_season - 1`.

    Refits age curves and regression priors on training data only — no leakage
    from the test season.
    """
    as_of = test_season - 1
    train_ps = player_seasons.filter(pl.col("season") <= as_of)
    train_weekly = weekly_stats.filter(pl.col("season") <= as_of)

    curves = fit_age_curves(train_ps)
    priors = None
    if use_regression:
        priors = fit_regression_priors(train_weekly, as_of_season=as_of)
        if n_multiplier != 1.0:
            priors = priors.with_columns(regression_n=pl.col("regression_n") * n_multiplier)

    return project_players(
        train_ps,
        curves,
        as_of_season=as_of,
        n_future_seasons=1,
        history_weights=history_weights,
        regression_priors=priors,
    )


def score_actuals_for_backtest(
    player_seasons: pl.DataFrame,
    test_season: int,
    scoring: ScoringConfig,
) -> pl.DataFrame:
    """Compute actual fantasy points for the test season under `scoring`."""
    actuals = player_seasons.filter(pl.col("season") == test_season)
    rename_map = {c: f"{c}_season" for c in _ACTUAL_STAT_COLS if c in actuals.columns}
    return score_projections(actuals.rename(rename_map), scoring)


def evaluate_predictions(
    projections: pl.DataFrame,
    actuals: pl.DataFrame,
) -> pl.DataFrame:
    """Inner-join projections to actuals on player_id and compute per-player FP error.

    Only players who actually played the test season are evaluated — predictions
    for players who retired or missed the whole year don't count as model errors.
    """
    return projections.select(
        "player_id",
        "player_name",
        "position",
        pl.col("fantasy_points_season").alias("proj_fp"),
    ).join(
        actuals.select(
            "player_id",
            pl.col("fantasy_points_season").alias("actual_fp"),
        ),
        on="player_id",
        how="inner",
    ).with_columns(
        error=pl.col("proj_fp") - pl.col("actual_fp"),
        abs_error=(pl.col("proj_fp") - pl.col("actual_fp")).abs(),
        sq_error=(pl.col("proj_fp") - pl.col("actual_fp")) ** 2,
    )


def summarize_errors(merged: pl.DataFrame) -> pl.DataFrame:
    """MAE / RMSE / mean bias of fantasy points per position."""
    per_pos = merged.group_by("position").agg(
        n_players=pl.len(),
        mae=pl.col("abs_error").mean(),
        rmse=(pl.col("sq_error").mean() ** 0.5),
        mean_bias=pl.col("error").mean(),
    )
    overall = merged.select(
        position=pl.lit("ALL"),
        n_players=pl.len(),
        mae=pl.col("abs_error").mean(),
        rmse=(pl.col("sq_error").mean() ** 0.5),
        mean_bias=pl.col("error").mean(),
    )
    return pl.concat([per_pos, overall]).sort("position")


def run_backtest(
    player_seasons: pl.DataFrame,
    weekly_stats: pl.DataFrame,
    test_seasons: list[int],
    variants: dict[str, dict],
    scoring: ScoringConfig,
) -> pl.DataFrame:
    """Run the backtest matrix and return a long-format error summary.

    `variants` maps a variant name to kwargs for `project_for_backtest`
    (typically `{"use_regression": ..., "n_multiplier": ...}`).

    Returns:
        DataFrame with columns:
            test_season | variant | position | n_players | mae | rmse | mean_bias
    """
    parts: list[pl.DataFrame] = []
    for test_season in test_seasons:
        actuals = score_actuals_for_backtest(player_seasons, test_season, scoring)
        for variant_name, variant_kwargs in variants.items():
            proj = project_for_backtest(
                player_seasons, weekly_stats, test_season, **variant_kwargs
            )
            scored = score_projections(proj, scoring)
            merged = evaluate_predictions(scored, actuals)
            summary = summarize_errors(merged).with_columns(
                test_season=pl.lit(test_season),
                variant=pl.lit(variant_name),
            )
            parts.append(summary)

    return pl.concat(parts).select(
        "test_season", "variant", "position", "n_players", "mae", "rmse", "mean_bias"
    )
