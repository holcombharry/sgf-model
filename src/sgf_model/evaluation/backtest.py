"""Out-of-sample backtest: project a test season using only earlier data, compare
to actuals, and report per-position error, rank quality, and calibration coverage.

For each test season T:
    - Train on player_seasons / weekly through T-1.
    - Refit age curves, regression priors, history baselines from training only.
    - Project T (one season ahead).
    - Fit per-position residual sigma on a held-out projection of T-1, build
      predictive intervals on the T projections.
    - Compare to actuals for players who played; split by healthy/injured.

Metrics reported per (test_season, variant, injury_bucket, position):
    - mae / rmse / mean_bias — absolute error level
    - spearman — rank correlation across the per-position field
    - top12_hit / top24_hit / top36_hit — top-N intersection rate
    - cov_50 / cov_80 — fraction of actuals inside the 50% / 80% interval
                       (calibration; well-calibrated → cov_50 ≈ 0.5, cov_80 ≈ 0.8)

The holdout window (test_seasons argument) is locked at 2021–2023 per
docs/holdout.md. 2024 is reserved as final-validation lockbox.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from sgf_model.curves import fit_age_curves
from sgf_model.features import (
    PHASE1_FEATURE_COLUMNS,
    build_feature_matrix,
)
from sgf_model.models import QuantileFPModel
from sgf_model.projections import fit_regression_priors, project_players
from sgf_model.scoring import ScoringConfig, score_projections

DEFAULT_TOP_NS: tuple[int, ...] = (12, 24, 36)

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


HEALTHY_GAMES_THRESHOLD: int = 8

# Gaussian z-scores for two-sided predictive intervals at common levels.
# 50% interval covers ±0.6745 sigma (the interquartile range under a normal),
# 80% covers ±1.2816. We default to these two — narrow tells you "is the median
# centered well?", wide tells you "are we honest about uncertainty?".
INTERVAL_Z: dict[float, float] = {0.5: 0.6745, 0.8: 1.2816}
DEFAULT_INTERVAL_LEVELS: tuple[float, ...] = (0.5, 0.8)


def fit_residual_sigmas(
    player_seasons: pl.DataFrame,
    weekly_stats: pl.DataFrame,
    ref_season: int,
    scoring: ScoringConfig,
    **proj_kwargs,
) -> dict[str, float]:
    """Per-position residual SD from a held-out one-step-ahead projection at `ref_season`.

    The intervals are intentionally simple (one sigma per position, Gaussian) and
    will be replaced by comp-based predictive distributions in Phase 4. The point
    of having them now is so we can track calibration coverage from day 1 — any
    future improvement to the projection model can be measured against this baseline.

    `ref_season` should be the most recent season *before* the test season being
    evaluated, so no information from the test set leaks into the sigma estimate.
    """
    proj = project_for_backtest(
        player_seasons, weekly_stats, test_season=ref_season, **proj_kwargs
    )
    scored = score_projections(proj, scoring)
    actuals = score_actuals_for_backtest(player_seasons, ref_season, scoring)
    merged = evaluate_predictions(scored, actuals)
    sigmas: dict[str, float] = {}
    for (pos,) in merged.select("position").unique().iter_rows():
        sub = merged.filter(pl.col("position") == pos)
        if sub.height < 5:
            continue
        sigmas[pos] = float(np.sqrt((sub["error"].to_numpy() ** 2).mean()))
    return sigmas


def attach_intervals(
    scored: pl.DataFrame,
    sigmas: dict[str, float],
    levels: tuple[float, ...] = DEFAULT_INTERVAL_LEVELS,
) -> pl.DataFrame:
    """Add `proj_fp_lower_{pct}` / `proj_fp_upper_{pct}` columns to a scored projection.

    Uses a Gaussian approximation with per-position sigma. Players in positions
    not present in `sigmas` (rare — only happens if the training sample is tiny)
    get null interval columns.
    """
    if not sigmas:
        return scored
    sigma_df = pl.DataFrame({
        "position": list(sigmas.keys()),
        "_sigma_fp": list(sigmas.values()),
    })
    out = scored.join(sigma_df, on="position", how="left")
    for level in levels:
        if level not in INTERVAL_Z:
            raise ValueError(f"No z-score defined for level {level}. Add to INTERVAL_Z.")
        z = INTERVAL_Z[level]
        pct = int(round(level * 100))
        out = out.with_columns(
            (pl.col("fantasy_points_season") - z * pl.col("_sigma_fp"))
                .alias(f"proj_fp_lower_{pct}"),
            (pl.col("fantasy_points_season") + z * pl.col("_sigma_fp"))
                .alias(f"proj_fp_upper_{pct}"),
        )
    return out.drop("_sigma_fp")


def eligible_player_ids(
    player_seasons: pl.DataFrame,
    test_season: int,
    min_games_prior: int = 3,
) -> list[str]:
    """Players who played at least `min_games_prior` games in `test_season - 1`.

    Used to define a common "evaluation universe" across v1 and v2 backtest
    paths. Without this both paths could project different player sets
    (v1 projects anyone in the historical data; v2 only projects T-1 actives)
    which makes aggregate metrics non-comparable.
    """
    return (
        player_seasons
        .filter(
            (pl.col("season") == test_season - 1)
            & (pl.col("games_played") >= min_games_prior)
        )["player_id"]
        .unique()
        .to_list()
    )


def evaluate_predictions(
    projections: pl.DataFrame,
    actuals: pl.DataFrame,
    eligible_player_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Inner-join projections to actuals on player_id and compute per-player FP error.

    Carries actual `games_played` through so downstream summaries can split into
    healthy vs. injured buckets (a player who missed half the season isn't a
    model error in the same sense as a player who played 17 games at the wrong volume).
    Also carries any `proj_fp_lower_*` / `proj_fp_upper_*` interval columns through
    so the summary can compute calibration coverage.
    Only players who actually played the test season are evaluated — predictions
    for players who retired or missed the whole year don't count as model errors.
    """
    if eligible_player_ids is not None:
        projections = projections.filter(pl.col("player_id").is_in(eligible_player_ids))
    interval_cols = [
        c for c in projections.columns
        if c.startswith("proj_fp_lower_") or c.startswith("proj_fp_upper_")
    ]
    proj_selected = projections.select(
        "player_id",
        "player_name",
        "position",
        pl.col("fantasy_points_season").alias("proj_fp"),
        *interval_cols,
    )
    return proj_selected.join(
        actuals.select(
            "player_id",
            pl.col("fantasy_points_season").alias("actual_fp"),
            "games_played",
        ),
        on="player_id",
        how="inner",
    ).with_columns(
        error=pl.col("proj_fp") - pl.col("actual_fp"),
        abs_error=(pl.col("proj_fp") - pl.col("actual_fp")).abs(),
        sq_error=(pl.col("proj_fp") - pl.col("actual_fp")) ** 2,
        injury_bucket=pl.when(pl.col("games_played") >= HEALTHY_GAMES_THRESHOLD)
            .then(pl.lit("healthy"))
            .otherwise(pl.lit("injured")),
    )


def _position_metrics(
    sub: pl.DataFrame,
    position: str,
    top_ns: tuple[int, ...],
) -> dict[str, float | str | int]:
    """Error + rank metrics for a single position (or 'ALL'). Used by summarize_errors.

    Rank metrics (Spearman, top-N hit rate) measure ordering quality independent
    of absolute error. A model can have decent MAE but still rank top-12 poorly,
    which is what actually matters for dynasty rosters.

    Missing values are returned as NaN (not None) so polars infers Float64 even
    when a whole bucket has too few players to compute a metric.
    """
    nan = float("nan")
    n = sub.height
    proj = sub["proj_fp"].to_numpy()
    actual = sub["actual_fp"].to_numpy()
    err = proj - actual

    row: dict[str, float | str | int] = {
        "position": position,
        "n_players": n,
        "mae": float(np.abs(err).mean()) if n else nan,
        "rmse": float(np.sqrt((err**2).mean())) if n else nan,
        "mean_bias": float(err.mean()) if n else nan,
        "spearman": float(spearmanr(proj, actual).correlation) if n >= 3 else nan,
    }
    proj_order = sub.sort("proj_fp", descending=True)["player_id"].to_list()
    actual_order = sub.sort("actual_fp", descending=True)["player_id"].to_list()
    for top_n in top_ns:
        col = f"top{top_n}_hit"
        if n >= top_n:
            row[col] = len(set(proj_order[:top_n]) & set(actual_order[:top_n])) / top_n
        else:
            row[col] = nan

    # Calibration coverage: fraction of actuals inside each predictive interval.
    # Only emitted if the corresponding interval columns are present.
    actual = sub["actual_fp"]
    for col_name in sub.columns:
        if not col_name.startswith("proj_fp_lower_"):
            continue
        pct = col_name.removeprefix("proj_fp_lower_")
        upper = f"proj_fp_upper_{pct}"
        if upper not in sub.columns:
            continue
        if n == 0:
            row[f"cov_{pct}"] = nan
            continue
        inside = (sub[col_name] <= actual) & (actual <= sub[upper])
        row[f"cov_{pct}"] = float(inside.mean())
    return row


def summarize_errors(
    merged: pl.DataFrame,
    top_ns: tuple[int, ...] = DEFAULT_TOP_NS,
) -> pl.DataFrame:
    """Per-position MAE / RMSE / bias plus Spearman rank correlation and top-N hit rates.

    Top-N hit rate is the fraction of the projected top-N players in this position
    who actually finished in the top-N. Spearman is the rank correlation across the
    full per-position field. Both metrics measure ordering quality and are immune
    to the absolute-error scale.
    """
    positions = sorted(merged["position"].unique().to_list())
    rows = [
        _position_metrics(merged.filter(pl.col("position") == p), p, top_ns) for p in positions
    ]
    rows.append(_position_metrics(merged, "ALL", top_ns))
    return pl.DataFrame(rows)


def summarize_errors_by_bucket(
    merged: pl.DataFrame,
    top_ns: tuple[int, ...] = DEFAULT_TOP_NS,
) -> pl.DataFrame:
    """summarize_errors run separately on healthy / injured / all buckets.

    Adds an `injury_bucket` column to the output. Use this as the default summary
    for variant comparison — comparing models on healthy seasons alone tells you
    whether the model is *getting better at modeling* vs. *getting better at
    predicting who'll get hurt*. Those are different questions.
    """
    if "injury_bucket" not in merged.columns:
        raise ValueError("merged must contain `injury_bucket` (run `evaluate_predictions` first).")

    pieces: list[pl.DataFrame] = []
    for bucket in ("healthy", "injured", "all"):
        sub = merged if bucket == "all" else merged.filter(pl.col("injury_bucket") == bucket)
        if sub.height == 0:
            continue
        summary = summarize_errors(sub, top_ns=top_ns).with_columns(
            injury_bucket=pl.lit(bucket),
        )
        pieces.append(summary)
    return pl.concat(pieces)


def project_for_backtest_v2(
    player_seasons: pl.DataFrame,
    test_season: int,
    scoring: ScoringConfig,
    advanced_features: pl.DataFrame | None = None,
    draft_features: pl.DataFrame | None = None,
    feature_columns: tuple[str, ...] = PHASE1_FEATURE_COLUMNS,
    model_params: dict | None = None,
    random_state: int = 42,
) -> pl.DataFrame:
    """v2 top-down projection: feature matrix + per-position quantile FP model.

    Trains the QuantileFPModel on data through `test_season - 1` (no leakage)
    and emits inference predictions for `test_season`. Output schema matches
    the v1 path: `fantasy_points_season` plus `proj_fp_lower_50` /
    `proj_fp_upper_50` / `proj_fp_lower_80` / `proj_fp_upper_80`.

    The v2 model's intervals are quantile-emitted directly — no `attach_intervals`
    post-processing needed (and `fit_residual_sigmas` is not called).

    `advanced_features` / `draft_features` are passed straight to
    `build_feature_matrix` to enable Phase 2+ feature sets. They can be built
    once and reused across test seasons.

    `feature_columns` selects which columns the model consumes — this is how
    Phase 1 (PHASE1_FEATURE_COLUMNS) and Phase 2 (PHASE2_FEATURE_COLUMNS)
    variants share the same feature-matrix infrastructure.
    """
    train_ps = player_seasons.filter(pl.col("season") <= test_season - 1)
    # Filter advanced features to the training window too — no leakage from T or later.
    advanced_train = (
        advanced_features.filter(pl.col("season") <= test_season - 1)
        if advanced_features is not None else None
    )
    training_matrix = build_feature_matrix(
        train_ps, scoring,
        advanced_features=advanced_train,
        draft_features=draft_features,
    )
    inference_matrix = build_feature_matrix(
        train_ps, scoring,
        inference_season=test_season,
        advanced_features=advanced_train,
        draft_features=draft_features,
    ).filter(pl.col("target_fp").is_null())

    model = QuantileFPModel(
        feature_columns=feature_columns,
        params=model_params,
        random_state=random_state,
    )
    model.fit(training_matrix)
    return model.predict(inference_matrix)


def run_backtest_v2(
    player_seasons: pl.DataFrame,
    test_seasons: list[int],
    variants: dict[str, dict],
    scoring: ScoringConfig,
    advanced_features: pl.DataFrame | None = None,
    draft_features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the v2 backtest matrix.

    `variants` maps a variant name to kwargs for `project_for_backtest_v2`
    (typically `{"feature_columns": PHASE2_FEATURE_COLUMNS}` or
    `{"model_params": {...}}`).

    `advanced_features` / `draft_features` apply to all variants — typically
    built once via `build_advanced_features` and `compute_draft_features` and
    passed in here. Variants then opt into using them via `feature_columns`.

    Same return schema as `run_backtest`. Concatenable with v1 results for
    direct comparison — the `variant` column distinguishes which model produced
    which row.
    """
    parts: list[pl.DataFrame] = []
    for test_season in test_seasons:
        actuals = score_actuals_for_backtest(player_seasons, test_season, scoring)
        eligible = eligible_player_ids(player_seasons, test_season)
        for variant_name, variant_kwargs in variants.items():
            scored = project_for_backtest_v2(
                player_seasons, test_season, scoring,
                advanced_features=advanced_features,
                draft_features=draft_features,
                **variant_kwargs,
            )
            merged = evaluate_predictions(scored, actuals, eligible_player_ids=eligible)
            summary = summarize_errors_by_bucket(merged).with_columns(
                test_season=pl.lit(test_season),
                variant=pl.lit(variant_name),
            )
            parts.append(summary)

    out = pl.concat(parts)
    base_cols = ["test_season", "variant", "injury_bucket", "position", "n_players",
                 "mae", "rmse", "mean_bias", "spearman"]
    hit_cols = [c for c in out.columns if c.startswith("top") and c.endswith("_hit")]
    cov_cols = [c for c in out.columns if c.startswith("cov_")]
    return out.select(base_cols + hit_cols + cov_cols)


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
        eligible = eligible_player_ids(player_seasons, test_season)
        for variant_name, variant_kwargs in variants.items():
            proj = project_for_backtest(
                player_seasons, weekly_stats, test_season, **variant_kwargs
            )
            scored = score_projections(proj, scoring)
            # Fit per-position residual sigma on a held-out projection of test_season - 1.
            # Uses the same variant kwargs so the sigma estimate is variant-specific.
            sigmas = fit_residual_sigmas(
                player_seasons, weekly_stats,
                ref_season=test_season - 1,
                scoring=scoring,
                **variant_kwargs,
            )
            scored_with_intervals = attach_intervals(scored, sigmas)
            merged = evaluate_predictions(
                scored_with_intervals, actuals, eligible_player_ids=eligible
            )
            summary = summarize_errors_by_bucket(merged).with_columns(
                test_season=pl.lit(test_season),
                variant=pl.lit(variant_name),
            )
            parts.append(summary)

    out = pl.concat(parts)
    base_cols = ["test_season", "variant", "injury_bucket", "position", "n_players",
                 "mae", "rmse", "mean_bias", "spearman"]
    hit_cols = [c for c in out.columns if c.startswith("top") and c.endswith("_hit")]
    cov_cols = [c for c in out.columns if c.startswith("cov_")]
    return out.select(base_cols + hit_cols + cov_cols)
