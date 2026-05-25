"""Advanced per-(player, season) features for the v2 model: NGS, snaps, draft capital.

Each `compute_*_features` function returns a slim DataFrame keyed by
`(player_id, season)` with feature columns prefixed by their source
(`ngs_`, `snap_`, `draft_`). Coverage:

- Snap counts: 2012+
- NGS receiving / rushing / passing: 2016+
- Draft / demographic data: all

Pre-coverage seasons get null features; XGBoost/HGB handle nulls natively, so
this works as "use the data when you have it" without imputation tricks.

Features are designed to describe a player's *past performance per opportunity*
(YPRR-like quantities) which are the talent-layer signals identified in the
data audit. They're joined to the master feature matrix with a 1-year lag in
`features.builder`, so the model sees "what this player did last year" when
predicting "what they'll do this year".
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl


def _pfr_to_gsis(players: pl.DataFrame) -> pl.DataFrame:
    """ID crosswalk used to map snap-count rows (pfr_player_id) to our gsis_id key.

    `load_players()` calls the PFR ID `pfr_id` while `load_snap_counts()` calls
    it `pfr_player_id`. We rename to match the snap-count side.
    """
    return players.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("pfr_id").alias("pfr_player_id"),
    ).filter(pl.col("pfr_player_id").is_not_null())


def compute_snap_features(seasons: list[int], players: pl.DataFrame) -> pl.DataFrame:
    """Per (player_id, season) snap-share aggregates from PFR.

    `offense_pct` is the per-game share of team offensive snaps. We aggregate
    to season by:
        snap_share_mean = mean of offense_pct across games played
        snap_share_max  = max of offense_pct (one-week peak — proxy for ceiling)
        snap_games      = number of games with any offense snaps

    Only regular-season games count.
    """
    sc = nfl.load_snap_counts(seasons=seasons)
    sc = sc.filter(
        (pl.col("game_type") == "REG") & (pl.col("offense_snaps") > 0)
    )
    crosswalk = _pfr_to_gsis(players)
    sc = sc.join(crosswalk, on="pfr_player_id", how="inner")
    return (
        sc.group_by(["player_id", "season"])
        .agg(
            snap_share_mean=pl.col("offense_pct").mean(),
            snap_share_max=pl.col("offense_pct").max(),
            snap_games=pl.len(),
            offense_snaps_total=pl.col("offense_snaps").sum(),
        )
        .sort(["player_id", "season"])
    )


NGS_FIRST_SEASON: int = 2016


def _ngs_season_aggs(stat_type: str, seasons: list[int]) -> pl.DataFrame:
    """Load NGS for `stat_type` filtered to regular-season aggregate rows.

    NGS data includes both weekly rows (`week >= 1`) and season aggregates
    (`week == 0`). The season-aggregate rows are pre-computed by nflverse with
    proper opportunity-weighting and are the right shape for our feature
    matrix — no need to re-aggregate weekly.

    Filters the requested seasons to NGS coverage (2016+). If no requested
    seasons are NGS-era, returns an empty frame with the expected schema.
    """
    valid = [s for s in seasons if s >= NGS_FIRST_SEASON]
    if not valid:
        return pl.DataFrame(
            schema={
                "season": pl.Int64, "season_type": pl.String, "week": pl.Int64,
                "player_gsis_id": pl.String,
            }
        )
    ngs = nfl.load_nextgen_stats(seasons=valid, stat_type=stat_type)
    return ngs.filter(
        (pl.col("season_type") == "REG") & (pl.col("week") == 0)
    )


def compute_ngs_receiving_features(seasons: list[int]) -> pl.DataFrame:
    """WR/TE talent features from NGS receiving season aggregates."""
    rec = _ngs_season_aggs("receiving", seasons)
    return rec.select(
        pl.col("player_gsis_id").alias("player_id"),
        "season",
        pl.col("avg_separation").alias("ngs_separation"),
        pl.col("avg_cushion").alias("ngs_cushion"),
        pl.col("avg_intended_air_yards").alias("ngs_adot"),
        pl.col("catch_percentage").alias("ngs_catch_pct"),
        pl.col("avg_yac").alias("ngs_yac"),
        pl.col("avg_yac_above_expectation").alias("ngs_yac_oe"),
        pl.col("percent_share_of_intended_air_yards").alias("ngs_air_yard_share"),
    )


def compute_ngs_rushing_features(seasons: list[int]) -> pl.DataFrame:
    """RB talent features from NGS rushing season aggregates."""
    rush = _ngs_season_aggs("rushing", seasons)
    return rush.select(
        pl.col("player_gsis_id").alias("player_id"),
        "season",
        pl.col("efficiency").alias("ngs_rush_efficiency"),
        pl.col("rush_yards_over_expected_per_att").alias("ngs_ryoe_per_att"),
        pl.col("rush_pct_over_expected").alias("ngs_rush_pct_oe"),
        pl.col("avg_time_to_los").alias("ngs_time_to_los"),
        pl.col("percent_attempts_gte_eight_defenders").alias("ngs_pct_8plus_box"),
    )


def compute_ngs_passing_features(seasons: list[int]) -> pl.DataFrame:
    """QB talent features from NGS passing season aggregates."""
    pas = _ngs_season_aggs("passing", seasons)
    return pas.select(
        pl.col("player_gsis_id").alias("player_id"),
        "season",
        pl.col("completion_percentage_above_expectation").alias("ngs_cpoe"),
        pl.col("avg_time_to_throw").alias("ngs_time_to_throw"),
        pl.col("aggressiveness").alias("ngs_aggressiveness"),
        pl.col("avg_intended_air_yards").alias("ngs_qb_adot"),
        pl.col("avg_air_yards_to_sticks").alias("ngs_air_yards_to_sticks"),
    )


def compute_draft_features(players: pl.DataFrame) -> pl.DataFrame:
    """Per-player demographic features (draft capital). No season dimension —
    these are constant per player and join to every season's row.

    `draft_pick_log` is `log(pick)` for picks 1-262, null for UDFAs and others.
    The log scale captures the heavy diminishing returns past round 2.
    """
    return players.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("draft_round").cast(pl.Float64),
        pl.col("draft_pick").cast(pl.Float64),
        (pl.col("draft_pick").log()).alias("draft_pick_log"),
    )


def build_advanced_features(
    seasons: list[int],
    players: pl.DataFrame,
) -> pl.DataFrame:
    """Join all advanced per-(player, season) feature tables.

    Returns a DataFrame keyed by (player_id, season) with snap and NGS features
    populated when available, null otherwise. Use this output as the
    `advanced_features` argument to `features.builder.build_feature_matrix`.

    Note: `draft_features` is not joined here — it has no season dimension and
    is joined by the master builder directly so it applies to every season row.
    """
    snap = compute_snap_features(seasons, players)
    rec = compute_ngs_receiving_features(seasons)
    rush = compute_ngs_rushing_features(seasons)
    pas = compute_ngs_passing_features(seasons)
    out = snap
    for other in (rec, rush, pas):
        out = out.join(other, on=["player_id", "season"], how="full", coalesce=True)
    return out.sort(["player_id", "season"])
