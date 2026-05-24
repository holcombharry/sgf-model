"""VORP per (position, season) and discounted dynasty value.

VORP (value over replacement) — for each future season, a player's fantasy
points minus the replacement-level player's at the same position. The exact
replacement rank is set by `LeagueConfig.replacement_rank()`; this is where
league format (superflex, roster size, bench depth) actually moves rankings.

Dynasty value = sum of discounted future VORP. By default we clip negative
season VORP to zero — a player you wouldn't start has zero starter-value;
you'd just play someone else. Discount rate is a knob: lower (0.05-0.10)
favors young players (rebuild outlook), higher (0.20-0.25) favors current
producers (contention outlook).
"""

from __future__ import annotations

import polars as pl

from sgf_model.valuation.league import LeagueConfig


def compute_vorp(
    scored_projections: pl.DataFrame,
    league: LeagueConfig,
    fp_col: str = "fantasy_points_season",
) -> pl.DataFrame:
    """Add `replacement_fp_season` and `vorp_season` to scored projections.

    For each (position, season), the replacement-level fantasy points are
    taken from the player at `replacement_rank[position]` after sorting
    descending. Players outside the four scored positions (QB/RB/WR/TE) are
    dropped — we have no projection for them.
    """
    repl_ranks = league.replacement_rank()
    valid_positions = list(repl_ranks.keys())

    df = scored_projections.filter(pl.col("position").is_in(valid_positions))

    # Within each (position, season), rank by FP descending and grab the Nth row.
    df = df.with_columns(
        _pos_rank=pl.col(fp_col).rank("ordinal", descending=True).over(["position", "season"]),
    )
    # Per-position replacement: pick the row where _pos_rank == replacement_rank[position].
    repl_map_df = pl.DataFrame(
        {"position": list(repl_ranks.keys()), "_repl_rank": list(repl_ranks.values())}
    )
    df = df.join(repl_map_df, on="position")

    # The replacement player's FP for (position, season): grab it via window.
    df = df.with_columns(
        replacement_fp_season=pl.when(pl.col("_pos_rank") == pl.col("_repl_rank"))
        .then(pl.col(fp_col))
        .otherwise(None)
        .max()
        .over(["position", "season"]),
    )
    df = df.with_columns(
        vorp_season=pl.col(fp_col) - pl.col("replacement_fp_season"),
    )
    return df.drop(["_pos_rank", "_repl_rank"])


def compute_dynasty_value(
    vorp_table: pl.DataFrame,
    as_of_season: int = 2024,
    discount_rate: float = 0.15,
    clip_negative: bool = True,
) -> pl.DataFrame:
    """Discount future-season VORP and aggregate to one row per player.

    Args:
        vorp_table: Output of `compute_vorp`.
        as_of_season: Anchor season — projections for `as_of_season + 1` are
            discounted at year 1, etc.
        discount_rate: Annual discount. 0.15 = 15%/yr (default). Lower values
            increase young-player dynasty values; higher favors current producers.
        clip_negative: If True (default), negative season VORP is treated as 0
            for dynasty value purposes — you simply wouldn't play that player.
            Setting False is useful for diagnostics.

    Returns:
        Per-player summary, sorted descending by dynasty_value:
            player_id | player_name | position | current_age |
            dynasty_value | total_vorp_undiscounted | years_projected |
            peak_vorp_season | peak_vorp_year_offset
    """
    df = vorp_table.with_columns(
        year_offset=(pl.col("season") - as_of_season),
    )
    # Discount factor per row.
    df = df.with_columns(
        discount=pl.lit(1.0) / ((1.0 + discount_rate) ** pl.col("year_offset").cast(pl.Float64)),
    )

    vorp_for_value = (
        pl.max_horizontal(pl.col("vorp_season"), pl.lit(0.0))
        if clip_negative
        else pl.col("vorp_season")
    )
    df = df.with_columns(
        discounted_vorp=vorp_for_value * pl.col("discount"),
    )

    # Player's current age = age in the earliest projected season minus its offset.
    summary = (
        df.group_by(["player_id", "player_name", "position"])
        .agg(
            current_age=(pl.col("age") - pl.col("year_offset")).min(),
            dynasty_value=pl.col("discounted_vorp").sum(),
            total_vorp_undiscounted=pl.col("vorp_season").sum(),
            years_projected=pl.len(),
            peak_vorp_season=pl.col("vorp_season").max(),
            peak_vorp_year_offset=pl.col("year_offset")
            .filter(pl.col("vorp_season") == pl.col("vorp_season").max())
            .first(),
        )
        .sort("dynasty_value", descending=True)
    )
    return summary
