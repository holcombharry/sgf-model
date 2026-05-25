"""Build the (player, season) feature matrix for the FP model.

Each row corresponds to one player-season we may want to model — features come
strictly from data prior to that season, and the target is the FP that player
scored in that season (or null for the inference row, when we don't yet know).

Phase 1 features are intentionally minimal: age, experience, three prior-season
FP lags + weighted summary, prior games, and last-year position rank. The point
of Phase 1 is to validate the top-down architecture beats the v1 Marcel baseline
*before* investing in advanced metric features (Phase 2).
"""

from __future__ import annotations

import polars as pl

from sgf_model.scoring import ScoringConfig, score_projections

# Marcel-style weights matching the v1 baseline. Used to compute weighted prior
# FP / games features so the model sees an aggregate of recent history alongside
# the individual lags, which gives LightGBM both raw and pre-smoothed signals.
_HISTORY_WEIGHTS: tuple[float, float, float] = (5.0, 4.0, 3.0)

# The feature columns the model trains on. Position is handled as a separate
# per-position model (so it's not in this list) — but everything else is.
PHASE1_FEATURE_COLUMNS: tuple[str, ...] = (
    "age",
    "experience",
    "prior_fp_1y",
    "prior_fp_2y",
    "prior_fp_3y",
    "prior_fp_weighted",
    "prior_fp_per_game_weighted",
    "prior_games_1y",
    "prior_games_2y",
    "prior_games_3y",
    "prior_games_weighted",
    "position_rank_last_year",
    "is_top12_last_year",
    "is_top24_last_year",
)

# Phase 2 features are the Phase 1 set plus prior-season advanced metrics
# (snap share + NGS receiving / rushing / passing aggregates) and draft capital.
# All "prior_*" advanced features are last-year values — they describe how the
# player performed last year, used to predict this year.
PHASE2_ADVANCED_COLUMNS: tuple[str, ...] = (
    # Snap features (broad coverage, 2012+)
    "prior_snap_share_mean",
    "prior_snap_share_max",
    "prior_snap_games",
    "prior_offense_snaps_total",
    # NGS receiving (top WR/TE only, 2016+)
    "prior_ngs_separation",
    "prior_ngs_cushion",
    "prior_ngs_adot",
    "prior_ngs_catch_pct",
    "prior_ngs_yac",
    "prior_ngs_yac_oe",
    "prior_ngs_air_yard_share",
    # NGS rushing (top RB only, 2016+)
    "prior_ngs_rush_efficiency",
    "prior_ngs_ryoe_per_att",
    "prior_ngs_rush_pct_oe",
    "prior_ngs_time_to_los",
    "prior_ngs_pct_8plus_box",
    # NGS passing (most starting QBs, 2016+)
    "prior_ngs_cpoe",
    "prior_ngs_time_to_throw",
    "prior_ngs_aggressiveness",
    "prior_ngs_qb_adot",
    "prior_ngs_air_yards_to_sticks",
    # Draft capital (no season dim, joined per-player)
    "draft_round",
    "draft_pick",
    "draft_pick_log",
)
PHASE2_FEATURE_COLUMNS: tuple[str, ...] = PHASE1_FEATURE_COLUMNS + PHASE2_ADVANCED_COLUMNS

# Stat columns we need from player_seasons before applying score_projections.
# score_projections expects `{stat}_season` naming, but load_player_seasons emits
# bare names — we rename inside this module to keep that detail local.
_STAT_COLS_RENAME: dict[str, str] = {
    "passing_yards": "passing_yards_season",
    "passing_tds": "passing_tds_season",
    "passing_interceptions": "passing_interceptions_season",
    "carries": "carries_season",
    "rushing_yards": "rushing_yards_season",
    "rushing_tds": "rushing_tds_season",
    "targets": "targets_season",
    "receptions": "receptions_season",
    "receiving_yards": "receiving_yards_season",
    "receiving_tds": "receiving_tds_season",
}


def _score_history(
    player_seasons: pl.DataFrame, scoring: ScoringConfig
) -> pl.DataFrame:
    """Compute fp_season per (player, season) under the given scoring.

    Returns a slim frame with `player_id, season, position, fp_season, games_played`.
    """
    renamed = player_seasons.rename(
        {k: v for k, v in _STAT_COLS_RENAME.items() if k in player_seasons.columns}
    )
    scored = score_projections(renamed, scoring)
    return scored.select(
        "player_id",
        "player_name",
        "position",
        "season",
        "games_played",
        "birth_date",
        pl.col("fantasy_points_season").alias("fp_season"),
    )


def _add_lag(
    target: pl.DataFrame,
    history: pl.DataFrame,
    lag: int,
) -> pl.DataFrame:
    """Left-join `history` shifted by `lag` calendar years onto `target`.

    Result adds `prior_fp_{lag}y` and `prior_games_{lag}y` columns. Null when
    the player didn't play (or didn't exist) that many years prior — which is
    a legitimate signal, not missing data, so we leave nulls as nulls. LightGBM
    routes nulls natively.
    """
    return target.join(
        history.select(
            "player_id",
            (pl.col("season") + lag).alias("season"),
            pl.col("fp_season").alias(f"prior_fp_{lag}y"),
            pl.col("games_played").cast(pl.Float64).alias(f"prior_games_{lag}y"),
        ),
        on=["player_id", "season"],
        how="left",
    )


def _add_position_rank_last_year(
    target: pl.DataFrame,
    history: pl.DataFrame,
) -> pl.DataFrame:
    """Add the player's prior-season position-rank (1 = top scorer) and flags."""
    ranked = history.with_columns(
        pl.col("fp_season").rank(method="ordinal", descending=True)
        .over(["position", "season"])
        .cast(pl.Int64)
        .alias("position_rank")
    ).select(
        "player_id",
        (pl.col("season") + 1).alias("season"),
        pl.col("position_rank").alias("position_rank_last_year"),
    )
    out = target.join(ranked, on=["player_id", "season"], how="left")
    return out.with_columns(
        is_top12_last_year=(pl.col("position_rank_last_year") <= 12).cast(pl.Int8),
        is_top24_last_year=(pl.col("position_rank_last_year") <= 24).cast(pl.Int8),
    )


def _weighted_history(df: pl.DataFrame) -> pl.DataFrame:
    """Compute Marcel-style weighted prior FP, games, and per-game rate.

    Weights ignore lags where the player didn't play (null gets weight 0). This
    way a player with only 1 prior season still gets a clean weighted estimate
    instead of being dragged down by nulls.
    """
    w1, w2, w3 = _HISTORY_WEIGHTS
    weights = [w1, w2, w3]
    fps = ["prior_fp_1y", "prior_fp_2y", "prior_fp_3y"]
    gps = ["prior_games_1y", "prior_games_2y", "prior_games_3y"]

    # numerator = sum(w_i * x_i) where x_i is null-safe (treated as 0 in product
    # but the denominator below excludes that lag entirely).
    fp_num = sum(w * pl.col(c).fill_null(0.0) for w, c in zip(weights, fps))
    fp_den = sum(w * pl.col(c).is_not_null().cast(pl.Float64) for w, c in zip(weights, fps))
    games_num = sum(w * pl.col(c).fill_null(0.0) for w, c in zip(weights, gps))
    games_den = sum(w * pl.col(c).is_not_null().cast(pl.Float64) for w, c in zip(weights, gps))

    return df.with_columns(
        prior_fp_weighted=pl.when(fp_den > 0).then(fp_num / fp_den).otherwise(None),
        prior_games_weighted=pl.when(games_den > 0).then(games_num / games_den).otherwise(None),
    ).with_columns(
        # Per-game rate uses raw sums (not weighted ratio) so it stays a true rate.
        prior_fp_per_game_weighted=pl.when(
            (games_num > 0) & (fp_den > 0)
        ).then(fp_num / games_num).otherwise(None),
    )


def _experience(df: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """Number of prior NFL seasons in our data — proxy for years in the league.

    Doesn't include the current season (the row being modeled).
    """
    h = history.select("player_id", pl.col("season").alias("hist_season"))
    pairs = df.select("player_id", "season").unique().join(h, on="player_id", how="left")
    counts = (
        pairs.filter(pl.col("hist_season") < pl.col("season"))
        .group_by(["player_id", "season"])
        .agg(pl.len().alias("experience"))
    )
    return df.join(counts, on=["player_id", "season"], how="left").with_columns(
        experience=pl.col("experience").fill_null(0).cast(pl.Int64)
    )


def _join_advanced_lagged(
    df: pl.DataFrame,
    advanced: pl.DataFrame,
) -> pl.DataFrame:
    """Join advanced (player, season) features with a 1-year lag.

    Advanced features describe what a player did *last* season — we use them to
    predict *this* season. So we add 1 to advanced.season before joining, which
    turns "2022 separation" into "the prior_ngs_separation column for the 2023
    target row".
    """
    feature_cols = [c for c in advanced.columns if c not in ("player_id", "season")]
    shifted = advanced.select(
        "player_id",
        (pl.col("season") + 1).alias("season"),
        *[pl.col(c).alias(f"prior_{c}") for c in feature_cols],
    )
    return df.join(shifted, on=["player_id", "season"], how="left")


def _join_draft(df: pl.DataFrame, draft: pl.DataFrame) -> pl.DataFrame:
    """Join per-player draft features (no season dim — joins on player_id only)."""
    return df.join(draft, on="player_id", how="left")


def build_feature_matrix(
    player_seasons: pl.DataFrame,
    scoring: ScoringConfig,
    inference_season: int | None = None,
    min_prior_seasons: int = 1,
    advanced_features: pl.DataFrame | None = None,
    draft_features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the (player, season) feature matrix for training and/or inference.

    Args:
        player_seasons: Output of `load_player_seasons` — one row per
            (player, season) with raw stats, games_played, birth_date.
        scoring: ScoringConfig used to compute FP-based features and the target.
            Both target and historical features must use the same scoring so the
            model sees consistent units.
        inference_season: If provided, emits one row per active player at
            `inference_season` for prediction. Features come from historical
            data through `inference_season - 1`; target is null (we don't know).
            If None, emits training rows only.
        min_prior_seasons: Drop training rows with fewer prior seasons than this.
            Default 1 — rookies (zero prior seasons) are excluded from training
            since their feature vector is mostly null. Inference rows are kept
            regardless so rookies still get predictions (with null-handling by
            the model).
        advanced_features: Optional output of `build_advanced_features()` —
            per-(player, season) snap and NGS features. Joined with a 1-year
            lag so the model sees prior-season values. Columns get a `prior_`
            prefix. Pass None for the Phase 1 feature set.
        draft_features: Optional output of `compute_draft_features()` — per-player
            demographic columns (round, pick, log-pick). Joined per-player
            (every season row gets the same draft features). Pass None for the
            Phase 1 feature set.

    Returns:
        DataFrame with Phase 1 columns plus, if advanced/draft passed in,
        the PHASE2_ADVANCED_COLUMNS feature set.
    """
    history = _score_history(player_seasons, scoring)

    # Build the target row set: every player-season is potentially a training row;
    # for inference, additionally include a row for each player active at
    # inference_season - 1 (we predict their inference_season output).
    training_targets = history.select(
        "player_id", "player_name", "position", "season", "birth_date",
        pl.col("fp_season").alias("target_fp"),
    )

    if inference_season is not None:
        # An "active" player at inference_season is one who played in inference_season - 1.
        # (More elaborate definitions can come later; this matches how the v1 pipeline
        # decides who to project.)
        active_last_year = history.filter(pl.col("season") == inference_season - 1).select(
            "player_id", "player_name", "position", "birth_date",
        )
        inference_rows = active_last_year.with_columns(
            season=pl.lit(inference_season).cast(pl.Int32),
            target_fp=pl.lit(None, dtype=pl.Float64),
        )
        targets = pl.concat([training_targets, inference_rows], how="diagonal")
    else:
        targets = training_targets

    # Add lagged FP and games for the last 3 calendar years.
    out = targets
    for lag in (1, 2, 3):
        out = _add_lag(out, history, lag)

    # Position-rank features.
    out = _add_position_rank_last_year(out, history)

    # Weighted-history features and per-game rate.
    out = _weighted_history(out)

    # Age (in the target season).
    out = out.with_columns(
        age=(pl.col("season") - pl.col("birth_date").dt.year()).cast(pl.Float64)
    ).drop("birth_date")

    # Experience (count of prior seasons in our data).
    out = _experience(out, history)

    # Optional advanced/draft features (Phase 2+).
    if advanced_features is not None:
        out = _join_advanced_lagged(out, advanced_features)
    if draft_features is not None:
        out = _join_draft(out, draft_features)

    # Drop training rows that lack any history (rookies in training). Inference
    # rows are kept regardless — they need predictions even with sparse features.
    out = out.with_columns(
        _has_history=pl.col("prior_fp_1y").is_not_null()
            | pl.col("prior_fp_2y").is_not_null()
            | pl.col("prior_fp_3y").is_not_null(),
        _is_inference=pl.col("target_fp").is_null(),
    )
    out = out.filter(pl.col("_is_inference") | (pl.col("_has_history") &
                     (pl.col("experience") >= min_prior_seasons)))
    return out.drop(["_has_history", "_is_inference"])
