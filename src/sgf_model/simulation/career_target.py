"""Realized career VORP targets for direct-target training.

Used as the per-(player, anchor) training target when the model is trained on
career VORP directly (instead of per-year FP that's then Monte Carlo-aggregated).

For each (player, anchor_season), `compute_career_vorp_targets` returns the
discounted sum of per-year VORP over years anchor_season+1 .. anchor_season+horizon,
using realized per-(position, year) replacement levels from history. Inactive
years contribute 0.

The target is left-null for anchors where the horizon extends past the data
window — those become inference rows in the feature matrix.
"""

from __future__ import annotations

import polars as pl

from sgf_model.scoring import ScoringConfig, score_projections
from sgf_model.valuation.league import LeagueConfig

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


def compute_career_vorp_targets(
    player_seasons: pl.DataFrame,
    scoring: ScoringConfig,
    league: LeagueConfig,
    horizon: int = 5,
    discount_rate: float = 0.15,
) -> pl.DataFrame:
    """Realized career VORP per (player_id, anchor_season).

    For each player and each candidate anchor_season such that
    anchor_season + horizon <= max(player_seasons.season), compute the
    discounted sum of per-year VORP over years anchor+1..anchor+horizon.
    Per-year VORP uses realized league replacement levels from the actual
    field that year (matches the Phase 4 calibration setup).

    Returns: (player_id, anchor_season, target_career_vorp).
    """
    renamed = player_seasons.rename(
        {k: v for k, v in _STAT_COLS_RENAME.items() if k in player_seasons.columns}
    )
    scored = score_projections(renamed, scoring).select(
        "player_id", "position", "season",
        pl.col("fantasy_points_season").alias("fp"),
    )

    repl_ranks = league.replacement_rank()
    max_season = int(scored["season"].max())

    # Per-(position, year) replacement FP from the actual field that year.
    repl_records = []
    for year in scored["season"].unique().sort().to_list():
        for pos, rank_N in repl_ranks.items():
            year_pos = scored.filter(
                (pl.col("season") == year) & (pl.col("position") == pos)
            ).sort("fp", descending=True)
            if year_pos.height == 0:
                continue
            idx = min(rank_N - 1, year_pos.height - 1)
            repl_records.append({
                "season": year, "position": pos,
                "replacement_fp": float(year_pos["fp"][idx]),
            })
    replacement = pl.DataFrame(repl_records).with_columns(
        season=pl.col("season").cast(scored.schema["season"]),
    )

    # Per-(player, year) per-year VORP, clipped at 0.
    yearly_vorp = (
        scored.join(replacement, on=["season", "position"], how="left")
        .with_columns(
            year_vorp=(pl.col("fp") - pl.col("replacement_fp")).clip(lower_bound=0.0),
        )
        .select("player_id", "season", "year_vorp")
    )

    # Cross every (player_id, anchor_season) where anchor+horizon <= max_season
    # with each future year offset, sum discounted realized VORP.
    players_seasons = scored.select("player_id").unique()
    candidate_anchors = pl.DataFrame({
        "anchor_season": [s for s in range(scored["season"].min(), max_season - horizon + 1)]
    }).with_columns(anchor_season=pl.col("anchor_season").cast(scored.schema["season"]))

    anchors = players_seasons.join(candidate_anchors, how="cross")

    offsets = pl.DataFrame({"offset": list(range(1, horizon + 1))})
    expanded = anchors.join(offsets, how="cross").with_columns(
        season=(pl.col("anchor_season") + pl.col("offset")).cast(scored.schema["season"]),
        discount=(1.0 + discount_rate) ** -pl.col("offset"),
    )

    joined = (
        expanded.join(yearly_vorp, on=["player_id", "season"], how="left")
        .with_columns(year_vorp=pl.col("year_vorp").fill_null(0.0))
        .with_columns(discounted=pl.col("year_vorp") * pl.col("discount"))
    )

    return (
        joined.group_by(["player_id", "anchor_season"])
        .agg(target_career_vorp=pl.col("discounted").sum())
        .sort(["player_id", "anchor_season"])
    )
