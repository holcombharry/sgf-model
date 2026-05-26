"""Build the (player, season) feature matrix for the FP model.

Each row corresponds to one (player, anchor_season, target_season) we may want
to model — features come strictly from data on or before `anchor_season`, and
the target is the FP that player scored in `target_season` (or null when
inferring). `future_offset = target_season - anchor_season` is itself a feature
so a single model can be trained to predict FP at any horizon.

For Phase 1/2 compatibility, `forecast_horizon=1` produces the same row set
as the original (anchor_season = season - 1, target_season = season). For
Phase 3+, set forecast_horizon=N to also emit rows for offsets 2..N.

Phase 1 features are intentionally minimal: age, experience, three prior-season
FP lags + weighted summary, prior games, and last-year position rank. Phase 2
adds NGS / snap / draft features. Phase 3 adds future_offset.
"""

from __future__ import annotations

import polars as pl

from sgf_model.scoring import ScoringConfig, score_projections

# Marcel-style weights used to compute weighted prior FP / games features so
# the model sees an aggregate of recent history alongside the individual lags.
_HISTORY_WEIGHTS: tuple[float, float, float] = (5.0, 4.0, 3.0)

PHASE1_FEATURE_COLUMNS: tuple[str, ...] = (
    "age",
    "experience",
    "is_rookie",
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

# Phase 2 advanced features — joined at anchor_season so they describe what the
# player did most recently before projection time.
PHASE2_ADVANCED_COLUMNS: tuple[str, ...] = (
    "prior_snap_share_mean",
    "prior_snap_share_max",
    "prior_snap_games",
    "prior_offense_snaps_total",
    "prior_ngs_separation",
    "prior_ngs_cushion",
    "prior_ngs_adot",
    "prior_ngs_catch_pct",
    "prior_ngs_yac",
    "prior_ngs_yac_oe",
    "prior_ngs_air_yard_share",
    "prior_ngs_rush_efficiency",
    "prior_ngs_ryoe_per_att",
    "prior_ngs_rush_pct_oe",
    "prior_ngs_time_to_los",
    "prior_ngs_pct_8plus_box",
    "prior_ngs_cpoe",
    "prior_ngs_time_to_throw",
    "prior_ngs_aggressiveness",
    "prior_ngs_qb_adot",
    "prior_ngs_air_yards_to_sticks",
    "draft_round",
    "draft_pick",
    "draft_pick_log",
)
PHASE2_FEATURE_COLUMNS: tuple[str, ...] = PHASE1_FEATURE_COLUMNS + PHASE2_ADVANCED_COLUMNS

# Phase 3 adds future_offset so the same model can predict at multiple horizons.
PHASE3_FEATURE_COLUMNS: tuple[str, ...] = PHASE2_FEATURE_COLUMNS + ("future_offset",)

# Phase 5 adds routes-derived per-opportunity rates — the talent metrics that
# discriminate elites at the feature level (YPRR, target rate per route, TD rate
# per route). Available 2016+ from participation data. Address the elite-tier
# under-coverage identified in docs/phase4-career-calibration.md.
PHASE5_ROUTES_COLUMNS: tuple[str, ...] = (
    "prior_routes_run",
    "prior_yprr",
    "prior_targets_per_route",
    "prior_td_rate_per_route",
)
PHASE5_FEATURE_COLUMNS: tuple[str, ...] = PHASE3_FEATURE_COLUMNS + PHASE5_ROUTES_COLUMNS

# Career-direct training uses one row per (player, anchor) with target =
# realized career VORP, so `future_offset` is constant 1 and carries no signal.
# Dropped here to avoid wasting a feature slot.
CAREER_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for c in PHASE5_FEATURE_COLUMNS if c != "future_offset"
)


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
    """Compute fp_season per (player, season) under the given scoring."""
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


def _add_lag_from_anchor(
    target: pl.DataFrame,
    history: pl.DataFrame,
    lag: int,
) -> pl.DataFrame:
    """Join `history.fp_season` from `anchor_season - (lag - 1)` years onto each target row.

    Convention:
      lag=1 → FP at anchor_season       (most recent observed)
      lag=2 → FP at anchor_season - 1
      lag=3 → FP at anchor_season - 2

    For the common case where future_offset=1 (target_season = anchor_season+1),
    this matches the Phase 1/2 semantic where prior_fp_1y was "FP at target-1".
    For larger offsets, it correctly clamps to data the model could have at
    projection time (no leakage from years between anchor and target).
    """
    shift_back = lag - 1
    return target.join(
        history.select(
            "player_id",
            (pl.col("season") + shift_back).alias("anchor_season"),
            pl.col("fp_season").alias(f"prior_fp_{lag}y"),
            pl.col("games_played").cast(pl.Float64).alias(f"prior_games_{lag}y"),
        ),
        on=["player_id", "anchor_season"],
        how="left",
    )


def _add_position_rank_at_anchor(
    target: pl.DataFrame,
    history: pl.DataFrame,
) -> pl.DataFrame:
    """Position rank at anchor_season (1 = top scorer that year) and top-N flags."""
    ranked = history.with_columns(
        pl.col("fp_season").rank(method="ordinal", descending=True)
        .over(["position", "season"])
        .cast(pl.Int64)
        .alias("position_rank")
    ).select(
        "player_id",
        pl.col("season").alias("anchor_season"),
        pl.col("position_rank").alias("position_rank_last_year"),
    )
    out = target.join(ranked, on=["player_id", "anchor_season"], how="left")
    return out.with_columns(
        is_top12_last_year=(pl.col("position_rank_last_year") <= 12).cast(pl.Int8),
        is_top24_last_year=(pl.col("position_rank_last_year") <= 24).cast(pl.Int8),
    )


def _weighted_history(df: pl.DataFrame) -> pl.DataFrame:
    """Marcel-style weighted prior FP, games, and per-game rate from the 3 lags."""
    w1, w2, w3 = _HISTORY_WEIGHTS
    weights = [w1, w2, w3]
    fps = ["prior_fp_1y", "prior_fp_2y", "prior_fp_3y"]
    gps = ["prior_games_1y", "prior_games_2y", "prior_games_3y"]

    fp_num = sum(w * pl.col(c).fill_null(0.0) for w, c in zip(weights, fps))
    fp_den = sum(w * pl.col(c).is_not_null().cast(pl.Float64) for w, c in zip(weights, fps))
    games_num = sum(w * pl.col(c).fill_null(0.0) for w, c in zip(weights, gps))
    games_den = sum(w * pl.col(c).is_not_null().cast(pl.Float64) for w, c in zip(weights, gps))

    return df.with_columns(
        prior_fp_weighted=pl.when(fp_den > 0).then(fp_num / fp_den).otherwise(None),
        prior_games_weighted=pl.when(games_den > 0).then(games_num / games_den).otherwise(None),
    ).with_columns(
        prior_fp_per_game_weighted=pl.when(
            (games_num > 0) & (fp_den > 0)
        ).then(fp_num / games_num).otherwise(None),
    )


def _experience_at_anchor(df: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """Number of prior NFL seasons in our data with season <= anchor_season."""
    h = history.select("player_id", pl.col("season").alias("hist_season"))
    pairs = df.select("player_id", "anchor_season").unique().join(
        h, on="player_id", how="left"
    )
    counts = (
        pairs.filter(pl.col("hist_season") <= pl.col("anchor_season"))
        .group_by(["player_id", "anchor_season"])
        .agg(pl.len().alias("experience"))
    )
    return df.join(counts, on=["player_id", "anchor_season"], how="left").with_columns(
        experience=pl.col("experience").fill_null(0).cast(pl.Int64)
    )


def _join_advanced_at_anchor(df: pl.DataFrame, advanced: pl.DataFrame) -> pl.DataFrame:
    """Join advanced (player, season) features at anchor_season.

    Advanced features describe what a player did at `anchor_season`; the model
    uses them to predict future-year FP. Columns get a `prior_` prefix to
    distinguish from raw features (since they're pre-projection signals).
    """
    feature_cols = [c for c in advanced.columns if c not in ("player_id", "season")]
    keyed = advanced.select(
        "player_id",
        pl.col("season").alias("anchor_season"),
        *[pl.col(c).alias(f"prior_{c}") for c in feature_cols],
    )
    return df.join(keyed, on=["player_id", "anchor_season"], how="left")


def _join_draft(df: pl.DataFrame, draft: pl.DataFrame) -> pl.DataFrame:
    """Join per-player draft features (no season dim — joins on player_id only)."""
    return df.join(draft, on="player_id", how="left")


def _build_target_rows(
    history: pl.DataFrame,
    inference_season: int | None,
    forecast_horizon: int,
    include_inactive_targets: bool = True,
) -> pl.DataFrame:
    """Enumerate (player, anchor_season, future_offset) → target_season rows.

    Training rows: every historical (player, anchor_season) gets one row per
    valid offset in 1..forecast_horizon where target_season = anchor_season + offset.

    When `include_inactive_targets=True` (default since Phase 4): if the player
    has no recorded FP at target_season but the season is within the training
    data window, we add the row with `target_fp = 0` — modeling "player was
    inactive that year" (retired, cut, season-long injury). Without this fix,
    the model only sees survivors at older ages, and survivorship bias produces
    unrealistically flat aging curves. See docs/phase4-retirement-fix.md.

    Target seasons beyond the training window (i.e., > max(history.season)) are
    dropped — we can't tell future-season activity from missing data.

    Inference rows: anchor_season = inference_season - 1, one row per offset,
    target_fp null.
    """
    anchors = history.select(
        "player_id", "player_name", "position", "birth_date",
        pl.col("season").alias("anchor_season"),
    ).unique()

    offsets = pl.DataFrame({"future_offset": list(range(1, forecast_horizon + 1))})
    train_rows = anchors.join(offsets, how="cross").with_columns(
        target_season=pl.col("anchor_season") + pl.col("future_offset"),
    )

    target_fps = history.select(
        "player_id",
        pl.col("season").alias("target_season"),
        pl.col("fp_season").alias("target_fp"),
    )

    if include_inactive_targets:
        # Left join to keep all (anchor, offset) pairs; restrict to in-window;
        # fill missing target_fp with 0 to flag inactive seasons.
        last_training_season = int(history["season"].max())
        train_rows = (
            train_rows.join(target_fps, on=["player_id", "target_season"], how="left")
            .filter(pl.col("target_season") <= last_training_season)
            .with_columns(target_fp=pl.col("target_fp").fill_null(0.0))
        )

        # Keep only the FIRST inactive season per (player, anchor_season). The
        # transition from active → inactive is the informative signal; including
        # every subsequent zero is redundant and overwhelms the model with mass
        # zeros, dragging the median below replacement for any retirement-adjacent
        # feature pattern. Active rows are always kept.
        train_rows = train_rows.with_columns(
            _is_inactive=(pl.col("target_fp") == 0).cast(pl.Int8),
        )
        train_rows = train_rows.with_columns(
            _first_inactive_offset=pl.col("future_offset")
                .filter(pl.col("_is_inactive") == 1)
                .min()
                .over(["player_id", "anchor_season"]),
        )
        train_rows = train_rows.filter(
            (pl.col("_is_inactive") == 0)
            | (pl.col("future_offset") == pl.col("_first_inactive_offset"))
        ).drop(["_is_inactive", "_first_inactive_offset"])
    else:
        # Legacy (pre-Phase-4) inner-join semantic — only realized FP rows.
        train_rows = train_rows.join(
            target_fps, on=["player_id", "target_season"], how="inner"
        )

    if inference_season is None:
        return train_rows

    # Inference: active = played in inference_season - 1.
    active = history.filter(pl.col("season") == inference_season - 1).select(
        "player_id", "player_name", "position", "birth_date",
        pl.col("season").alias("anchor_season"),
    )
    inference_rows = active.join(offsets, how="cross").with_columns(
        target_season=pl.col("anchor_season") + pl.col("future_offset"),
        target_fp=pl.lit(None, dtype=pl.Float64),
    )
    return pl.concat([train_rows, inference_rows], how="diagonal")


def _build_rookie_anchor_rows(
    history: pl.DataFrame,
    rookies: pl.DataFrame,
    inference_season: int | None,
    forecast_horizon: int,
    include_inactive_targets: bool,
) -> pl.DataFrame | None:
    """Synthetic pre-rookie anchor rows: anchor_season = draft_year - 1.

    For past rookies (draft_year + future_offset <= last training season):
    target_fp is realized FP (or 0 for inactive — same first-inactive-only
    filter as veterans). Teaches the model what happens given just draft
    capital + age + position with no NFL history.

    For incoming rookies (draft_year == inference_season): target_fp is null,
    one row per future_offset. Lets the simulator project a player who has
    never played a snap.

    All rookie rows carry the `_is_rookie_anchor=1` flag so the master filter
    can pass them through (their `_has_history` is false by construction).
    """
    if rookies is None or rookies.height == 0:
        return None

    last_training_season = int(history["season"].max())
    offsets = pl.DataFrame({"future_offset": list(range(1, forecast_horizon + 1))})

    # Cast anchor_season to match history.season's dtype so downstream
    # diagonal concats and joins line up. target_season is naturally Int64
    # via the offsets DataFrame and matches the veteran target-row schema.
    history_season_dtype = history.schema["season"]
    anchors = rookies.select(
        "player_id", "player_name", "position", "birth_date",
        (pl.col("draft_year") - 1).cast(history_season_dtype).alias("anchor_season"),
    )
    rows = anchors.join(offsets, how="cross").with_columns(
        target_season=pl.col("anchor_season") + pl.col("future_offset"),
    )

    # Inference rookie rows: only those whose draft_year == inference_season,
    # so anchor_season = inference_season - 1 and all targets are post-cutoff.
    if inference_season is not None:
        inference_rows = rows.filter(
            pl.col("anchor_season") == inference_season - 1
        ).with_columns(target_fp=pl.lit(None, dtype=pl.Float64))
    else:
        inference_rows = None

    # Training rookie rows: every (rookie, offset) whose target_season is in our
    # data window, with realized target_fp from history (0 if didn't play).
    target_fps = history.select(
        "player_id",
        pl.col("season").alias("target_season"),
        pl.col("fp_season").alias("target_fp"),
    )

    train_candidates = rows.filter(pl.col("target_season") <= last_training_season)
    if include_inactive_targets:
        train_rows = (
            train_candidates.join(target_fps, on=["player_id", "target_season"], how="left")
            .with_columns(target_fp=pl.col("target_fp").fill_null(0.0))
        )
        # Same first-inactive-only filter as the veteran path: keep all active
        # rows + the first inactive row per (player, anchor) — preserves the
        # active→inactive transition without flooding the model with zero rows.
        train_rows = train_rows.with_columns(
            _is_inactive=(pl.col("target_fp") == 0).cast(pl.Int8),
        )
        train_rows = train_rows.with_columns(
            _first_inactive_offset=pl.col("future_offset")
                .filter(pl.col("_is_inactive") == 1)
                .min()
                .over(["player_id", "anchor_season"]),
        )
        train_rows = train_rows.filter(
            (pl.col("_is_inactive") == 0)
            | (pl.col("future_offset") == pl.col("_first_inactive_offset"))
        ).drop(["_is_inactive", "_first_inactive_offset"])
    else:
        train_rows = train_candidates.join(
            target_fps, on=["player_id", "target_season"], how="inner"
        )

    pieces = [train_rows]
    if inference_rows is not None:
        pieces.append(inference_rows)
    out = pl.concat(pieces, how="diagonal")
    return out.with_columns(_is_rookie_anchor=pl.lit(1, dtype=pl.Int8))


def build_feature_matrix(
    player_seasons: pl.DataFrame,
    scoring: ScoringConfig,
    inference_season: int | None = None,
    forecast_horizon: int = 1,
    min_prior_seasons: int = 1,
    advanced_features: pl.DataFrame | None = None,
    draft_features: pl.DataFrame | None = None,
    rookies: pl.DataFrame | None = None,
    include_inactive_targets: bool = True,
) -> pl.DataFrame:
    """Build the (player, anchor_season, future_offset) feature matrix.

    Args:
        player_seasons: Output of `load_player_seasons` — one row per
            (player, season) with raw stats, games_played, birth_date.
        scoring: ScoringConfig used to compute FP-based features and target.
        inference_season: If set, emits inference rows for this season at each
            offset (anchor_season = inference_season - 1, target_season =
            inference_season .. inference_season + horizon - 1). target_fp null.
        forecast_horizon: Number of future-year offsets to emit per anchor.
            Default 1 reproduces Phase 1/2 behavior. Phase 3 uses larger values.
        min_prior_seasons: Drop training rows with experience < this. Default 1.
        advanced_features: Optional output of `build_advanced_features()`. Joined
            at anchor_season with a `prior_` prefix.
        draft_features: Optional output of `compute_draft_features()`. Joined
            per-player (constant across season rows).
        rookies: Optional output of `compute_rookies()`. When provided, emits
            synthetic pre-rookie anchor rows (anchor_season = draft_year - 1)
            so the model trains on rookie outcomes and can project incoming
            rookies at inference_season. Without this, rookies are silently
            dropped from training and absent from inference.

    Returns:
        DataFrame with columns:
            player_id, player_name, position,
            anchor_season, target_season, future_offset, target_fp,
            <feature columns: age, experience, prior_*, draft_*>

        For backward compatibility, a `season` column (= target_season) is also
        emitted so downstream code that filters/groups on `season` still works.
    """
    history = _score_history(player_seasons, scoring)
    targets = _build_target_rows(
        history, inference_season, forecast_horizon,
        include_inactive_targets=include_inactive_targets,
    ).with_columns(_is_rookie_anchor=pl.lit(0, dtype=pl.Int8))

    rookie_rows = _build_rookie_anchor_rows(
        history, rookies, inference_season, forecast_horizon,
        include_inactive_targets=include_inactive_targets,
    )
    if rookie_rows is not None:
        targets = pl.concat([targets, rookie_rows], how="diagonal")

    # Lagged FP and games from anchor_season backwards.
    out = targets
    for lag in (1, 2, 3):
        out = _add_lag_from_anchor(out, history, lag)

    out = _add_position_rank_at_anchor(out, history)
    out = _weighted_history(out)

    # Age at target_season (so a player at anchor=2022 projected for target=2024
    # is treated as age-at-2024 when ageing-related effects matter).
    out = out.with_columns(
        age=(pl.col("target_season") - pl.col("birth_date").dt.year()).cast(pl.Float64),
        future_offset=pl.col("future_offset").cast(pl.Float64),
    ).drop("birth_date")

    out = _experience_at_anchor(out, history)

    if advanced_features is not None:
        out = _join_advanced_at_anchor(out, advanced_features)
    if draft_features is not None:
        out = _join_draft(out, draft_features)

    # `is_rookie` = 1 for synthetic pre-rookie anchors (experience == 0 by
    # construction), 0 for veterans. Lets the model branch cleanly instead of
    # having to infer rookie-ness from the pattern of nulls.
    out = out.with_columns(
        is_rookie=pl.col("_is_rookie_anchor").fill_null(0).cast(pl.Int8),
    )

    # Master filter: keep inference rows always, rookie-anchor rows (no NFL
    # history by definition), and veteran training rows with sufficient
    # observed history.
    out = out.with_columns(
        _has_history=pl.col("prior_fp_1y").is_not_null()
            | pl.col("prior_fp_2y").is_not_null()
            | pl.col("prior_fp_3y").is_not_null(),
        _is_inference=pl.col("target_fp").is_null(),
    )
    out = out.filter(
        pl.col("_is_inference")
        | (pl.col("is_rookie") == 1)
        | (pl.col("_has_history") & (pl.col("experience") >= min_prior_seasons))
    )

    # `season` alias to target_season for backward compat.
    return out.drop(["_has_history", "_is_inference", "_is_rookie_anchor"]).with_columns(
        season=pl.col("target_season"),
    )


def build_career_feature_matrix(
    player_seasons: pl.DataFrame,
    scoring: ScoringConfig,
    career_targets: pl.DataFrame,
    inference_season: int | None = None,
    min_prior_seasons: int = 1,
    advanced_features: pl.DataFrame | None = None,
    draft_features: pl.DataFrame | None = None,
    rookies: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One-row-per-(player, anchor) matrix for direct career-VORP training.

    Wraps `build_feature_matrix(forecast_horizon=1)` to get the full feature
    set per (player, anchor), then swaps the per-year FP target for the
    realized career VORP target from `career_targets`.

    Training rows: those where the joined `target_career_vorp` is non-null
    (i.e., anchor + horizon falls inside the data window).

    Inference rows: anchor = inference_season - 1, target null. The caller
    consumes these to predict next-anchor career VORP distributions.

    `career_targets`: output of `compute_career_vorp_targets`. Joined on
    (player_id, anchor_season).
    """
    # Reuse the per-year builder for feature construction, then strip the
    # per-year target and substitute the career-VORP target.
    fm = build_feature_matrix(
        player_seasons, scoring,
        inference_season=inference_season,
        forecast_horizon=1,
        min_prior_seasons=min_prior_seasons,
        advanced_features=advanced_features,
        draft_features=draft_features,
        rookies=rookies,
    ).drop("target_fp")

    out = fm.join(career_targets, on=["player_id", "anchor_season"], how="left")
    # Keep training rows with a realized career VORP, plus the explicit
    # inference rows (anchor = inference_season - 1).
    if inference_season is not None:
        is_inference = pl.col("anchor_season") == inference_season - 1
        out = out.filter(is_inference | pl.col("target_career_vorp").is_not_null())
    else:
        out = out.filter(pl.col("target_career_vorp").is_not_null())

    # Standardize the target column name so the existing QuantileFPModel
    # (which reads `target_fp`) can train on it without modification.
    return out.rename({"target_career_vorp": "target_fp"})
