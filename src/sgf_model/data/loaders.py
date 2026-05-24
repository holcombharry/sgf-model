"""Thin typed wrappers around `nflreadpy` for the inputs we actually use.

`nflreadpy` handles filesystem caching transparently, so these are intentionally
shallow — they exist to pin the call shape (regular-season weekly granularity,
fantasy positions only, etc.) so the rest of the codebase doesn't repeat that.

We deliberately ignore `fantasy_points` / `fantasy_points_ppr` columns from the
weekly stats: the whole point of this project is to project raw stats and apply
scoring later via a pluggable config.
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl

FANTASY_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")


def load_weekly_stats(
    seasons: int | list[int] | bool | None,
    include_postseason: bool = False,
) -> pl.DataFrame:
    """Weekly player stats.

    Args:
        seasons: Season(s) to load. `True` pulls all available history;
            `None` uses the current season.
        include_postseason: If False (default), filters to regular-season weeks
            only — postseason has selection bias for projection modeling.
    """
    df = nfl.load_player_stats(seasons=seasons, summary_level="week")
    if not include_postseason:
        df = df.filter(pl.col("season_type") == "REG")
    return df


def load_players() -> pl.DataFrame:
    """Player metadata — includes birthdate, draft year, college, etc.

    Required for age curves and rookie modeling.
    """
    return nfl.load_players()


def filter_fantasy_positions(
    df: pl.DataFrame,
    positions: tuple[str, ...] = FANTASY_POSITIONS,
) -> pl.DataFrame:
    """Restrict to skill positions used in standard fantasy formats."""
    return df.filter(pl.col("position").is_in(positions))


# Volume/efficiency stats we aggregate to player-season level. Defensive/special-teams
# columns from load_player_stats are dropped here — they're not relevant for fantasy
# offensive projections.
_SEASON_SUM_STATS: tuple[str, ...] = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles_lost",
    "receiving_air_yards",
)


def load_player_seasons(
    start: int = 1999,
    end: int = 2024,
) -> pl.DataFrame:
    """Player-season aggregates with age, games played, and key fantasy stats.

    This is the primary input for age curves and player-level projection models —
    aggregated to one row per (player, season) with games_played and stat totals,
    joined to birth_date so age is computable.

    Age is computed as `season_year - birth_year` (the player's age during that
    season's calendar year). This is an approximation — a more exact "age on
    Sep 1" calc is straightforward later if it matters.
    """
    weekly = filter_fantasy_positions(load_weekly_stats(seasons=list(range(start, end + 1))))

    agg_exprs = [pl.len().alias("games_played")] + [
        pl.col(c).sum().alias(c) for c in _SEASON_SUM_STATS
    ]
    season = weekly.group_by(["player_id", "player_name", "position", "season"]).agg(agg_exprs)

    players = load_players().select(
        pl.col("gsis_id"),
        pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False),
    )
    season = season.join(players, left_on="player_id", right_on="gsis_id", how="left")
    season = season.with_columns(
        age=(pl.col("season") - pl.col("birth_date").dt.year()).cast(pl.Int32),
    )
    return season
