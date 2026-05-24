"""Team-context adjustments for player projections.

The vet projection layer treats each player as a closed system: their per-game
stats come from their own history, age-curved forward. That ignores the team
environment those stats were produced in. A WR on the Bengals (38 pass/g) and
on the Ravens (28 pass/g) face fundamentally different opportunity.

This module adds a multiplicative adjustment:

    pass_factor[player] = projected_pass_pg[NEW team] / weighted_history_pass_pg[OLD team]
    rush_factor[player] = projected_carries_pg[NEW team] / weighted_history_carries_pg[OLD team]

**Critical methodology point.** The "old team" anchor uses the *same weighted
history window* as the player's own baseline (default 5/4/3 over the last
three seasons). That way, for a player who stayed on the same team, the
numerator and denominator are computed identically and the factor is exactly
1.0 — no spurious adjustment from single-year team-volume noise. The factor
only deviates from 1.0 when the player actually changes teams (via the
`team_overrides` argument).

This is by design: year-to-year team volume changes for a *continuing* player
are already baked into their own per-game baseline. The team factor is for
**relocation** — moving the player into a different team environment than the
one their stats came from.

What this v1 captures:
    - Player team changes via user-provided overrides (FA, trades, retirements)
      → the major real-world use case

What it does NOT capture (documented gaps):
    - Forward-looking team-volume trends (e.g. new HC implementing pass-heavy
      scheme) — Marcel weighting alone is conservative; a trend model would
      require explicitly modeling team-level seasonality / coaching changes.
    - Target/touch share drift when teammates leave or arrive (share assumed stable)
    - QB-quality effects on receiver efficiency (yards/target invariant)
    - Schedule strength / opponent defense
"""

from __future__ import annotations

import polars as pl

# Stat groups for routing the right factor to the right column.
_RECEIVING_STATS: tuple[str, ...] = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)
_RUSHING_STATS: tuple[str, ...] = (
    "carries",
    "rushing_yards",
    "rushing_tds",
)


def compute_team_volumes(weekly_stats: pl.DataFrame) -> pl.DataFrame:
    """Per (team, season) totals + per-game averages for team-level volume.

    Sums across ALL positions on the team (not just fantasy) — `attempts` totals
    all QBs' passes, `carries` totals all RBs' + scramblers' rushes, etc.
    Returns games played for that team-season so per-game can be computed cleanly.
    """
    team_week = weekly_stats.group_by(["team", "season", "week"]).agg(
        team_pass_attempts=pl.col("attempts").sum(),
        team_carries=pl.col("carries").sum(),
    )
    return (
        team_week.group_by(["team", "season"])
        .agg(
            games=pl.len(),
            team_pass_attempts=pl.col("team_pass_attempts").sum(),
            team_carries=pl.col("team_carries").sum(),
        )
        .with_columns(
            pass_attempts_pg=pl.col("team_pass_attempts") / pl.col("games"),
            carries_pg=pl.col("team_carries") / pl.col("games"),
        )
        .sort(["team", "season"])
    )


def project_team_volumes(
    team_volumes: pl.DataFrame,
    as_of_season: int,
    n_future_seasons: int = 5,
    history_weights: tuple[float, ...] = (5.0, 4.0, 3.0),
) -> pl.DataFrame:
    """Marcel-style forward projection of team-level pass/carry rate per game.

    For each team, takes the most recent `len(history_weights)` seasons of
    pass_attempts_pg and carries_pg and computes a games-weighted mean
    (weights × games_played). The same projection is used for every future
    season — we don't model team-level age curves or trends here.

    Returns one row per (team, future_season).
    """
    history = team_volumes.filter(pl.col("season") <= as_of_season)
    out_rows: list[dict] = []

    for team, group in history.group_by("team"):
        team_str = team[0] if isinstance(team, tuple) else team
        sorted_group = group.sort("season", descending=True).head(len(history_weights))
        if sorted_group.is_empty():
            continue

        weights = list(history_weights[: sorted_group.height])
        games = sorted_group["games"].to_list()
        combined = [w * g for w, g in zip(weights, games)]
        total_combined = sum(combined)
        if total_combined == 0:
            continue

        pass_pg = sorted_group["pass_attempts_pg"].to_list()
        carries_pg = sorted_group["carries_pg"].to_list()
        proj_pass_pg = sum(c * v for c, v in zip(combined, pass_pg)) / total_combined
        proj_carries_pg = sum(c * v for c, v in zip(combined, carries_pg)) / total_combined

        for offset in range(1, n_future_seasons + 1):
            out_rows.append(
                {
                    "team": team_str,
                    "season": as_of_season + offset,
                    "projected_pass_attempts_pg": proj_pass_pg,
                    "projected_carries_pg": proj_carries_pg,
                }
            )

    return pl.DataFrame(out_rows)


def get_player_team_mapping(
    weekly_stats: pl.DataFrame,
    as_of_season: int,
) -> pl.DataFrame:
    """Each active player's most recent team — the max-week team in `as_of_season`.

    Mid-season trades: a player who finished the year on Team B (even if started
    on Team A) maps to Team B. The historical-team that anchors the pass/rush
    factor comes from the same season.
    """
    in_season = weekly_stats.filter(pl.col("season") == as_of_season)
    # For each player, find the row with the max week (their last game) → that's their team.
    return (
        in_season.sort(["player_id", "week"], descending=[False, True])
        .group_by("player_id")
        .agg(team_as_of=pl.col("team").first())
    )


def _team_weighted_anchor(
    team_volumes: pl.DataFrame,
    as_of_season: int,
    history_weights: tuple[float, ...],
) -> pl.DataFrame:
    """Weighted-history team pass/rush rate — matches `_weighted_history` scheme.

    For each team, takes the most recent `len(history_weights)` seasons through
    `as_of_season` and computes a games-weighted mean using
    `history_weights[i] * games_played[i]` — identical to how player baselines
    are weighted. This ensures factor = 1.0 for same-team players.
    """
    history = team_volumes.filter(pl.col("season") <= as_of_season)
    rows: list[dict] = []
    for team, group in history.group_by("team"):
        team_str = team[0] if isinstance(team, tuple) else team
        recent = group.sort("season", descending=True).head(len(history_weights))
        if recent.is_empty():
            continue
        weights = list(history_weights[: recent.height])
        games = recent["games"].to_list()
        combined = [w * g for w, g in zip(weights, games)]
        total = sum(combined)
        if total == 0:
            continue
        rows.append(
            {
                "team": team_str,
                "anchor_pass_pg": sum(
                    c * v for c, v in zip(combined, recent["pass_attempts_pg"].to_list())
                ) / total,
                "anchor_carries_pg": sum(
                    c * v for c, v in zip(combined, recent["carries_pg"].to_list())
                ) / total,
            }
        )
    return pl.DataFrame(rows)


def apply_team_context(
    projections: pl.DataFrame,
    team_volumes_historical: pl.DataFrame,
    team_volumes_projected: pl.DataFrame,
    player_team_mapping: pl.DataFrame,
    as_of_season: int,
    team_overrides: dict[str, str] | None = None,
    history_weights: tuple[float, ...] = (5.0, 4.0, 3.0),
) -> pl.DataFrame:
    """Apply multiplicative team-volume adjustments to player projections.

    Args:
        projections: Output of `project_players` or `project_rookies`.
        team_volumes_historical: Output of `compute_team_volumes`. Used to
            compute the weighted-history "old team volume" baseline.
        team_volumes_projected: Output of `project_team_volumes`. Used as the
            "new team volume" target.
        player_team_mapping: Output of `get_player_team_mapping(as_of_season)`.
        as_of_season: Anchor season; team history is summarized through this season.
        team_overrides: Optional `{player_id: new_team}` to handle trades / FA
            moves the model can't infer from data.
        history_weights: Must match the `history_weights` passed to
            `project_players` so anchor and player baseline windows align.
            Default matches the default in `project_players`.

    Returns:
        Same shape as `projections`, with `_per_game` and `_season` receiving
        and rushing stats scaled. New columns: `pass_factor`, `rush_factor`.
    """
    overrides_df = pl.DataFrame(
        {
            "player_id": list((team_overrides or {}).keys()),
            "team_override": list((team_overrides or {}).values()),
        },
        schema={"player_id": pl.Utf8, "team_override": pl.Utf8},
    )

    teams = (
        player_team_mapping.join(overrides_df, on="player_id", how="left")
        .with_columns(team=pl.coalesce("team_override", "team_as_of"))
        .select("player_id", "team")
    )

    anchor = _team_weighted_anchor(
        team_volumes_historical, as_of_season=as_of_season, history_weights=history_weights
    )

    # Old-team baseline: each player's recent-history team using the same weighted window.
    player_old_team = player_team_mapping.rename({"team_as_of": "team"}).join(
        anchor, on="team", how="left"
    ).select(
        "player_id",
        pl.col("anchor_pass_pg").alias("old_pass_pg"),
        pl.col("anchor_carries_pg").alias("old_carries_pg"),
    )

    # New-team projection (potentially overridden).
    player_new_team = teams.join(
        team_volumes_projected.rename({"team": "team"}),
        on="team",
        how="left",
    ).select(
        "player_id",
        "season",
        pl.col("projected_pass_attempts_pg").alias("new_pass_pg"),
        pl.col("projected_carries_pg").alias("new_carries_pg"),
    )

    factors = (
        projections.join(player_old_team, on="player_id", how="left")
        .join(player_new_team, on=["player_id", "season"], how="left")
        .with_columns(
            pass_factor=pl.when(
                pl.col("old_pass_pg").is_not_null()
                & pl.col("new_pass_pg").is_not_null()
                & (pl.col("old_pass_pg") > 0)
            )
            .then(pl.col("new_pass_pg") / pl.col("old_pass_pg"))
            .otherwise(1.0),
            rush_factor=pl.when(
                pl.col("old_carries_pg").is_not_null()
                & pl.col("new_carries_pg").is_not_null()
                & (pl.col("old_carries_pg") > 0)
            )
            .then(pl.col("new_carries_pg") / pl.col("old_carries_pg"))
            .otherwise(1.0),
        )
    )

    # Scale per-game and per-season columns for receiving and rushing stats.
    rec_per_game = [f"{s}_per_game" for s in _RECEIVING_STATS if f"{s}_per_game" in factors.columns]
    rec_season = [f"{s}_season" for s in _RECEIVING_STATS if f"{s}_season" in factors.columns]
    rush_per_game = [f"{s}_per_game" for s in _RUSHING_STATS if f"{s}_per_game" in factors.columns]
    rush_season = [f"{s}_season" for s in _RUSHING_STATS if f"{s}_season" in factors.columns]

    scale_exprs: list[pl.Expr] = []
    for col in rec_per_game + rec_season:
        scale_exprs.append((pl.col(col) * pl.col("pass_factor")).alias(col))
    for col in rush_per_game + rush_season:
        scale_exprs.append((pl.col(col) * pl.col("rush_factor")).alias(col))

    if scale_exprs:
        factors = factors.with_columns(scale_exprs)

    # Drop intermediate cols, keep the factors visible for inspection.
    return factors.drop(["old_pass_pg", "old_carries_pg", "new_pass_pg", "new_carries_pg"])
