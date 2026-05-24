"""Project per-game and per-season stats for active players.

Method (v1):
    1. Active players = anyone with games in `as_of_season`.
    2. Baseline per-game stat = weighted mean over the player's most recent
       `len(history_weights)` seasons, where the weight on each season is
       `history_weights[i] * games_played[i]`. This is Marcel-style: more
       recent seasons matter more, and a 17-game season carries more
       weight than a 4-game one.
    3. For each future season, age multipliers from `curves` are applied
       multiplicatively to project per-game stats forward.
    4. `games_played` is projected as a games-weighted mean over the
       same recent seasons. (A games-played-vs-age model would be a
       natural refinement.)

Known gaps to fix later:
    - **No regression to the mean.** A WR who caught 20 TDs in his only
      NFL season will project to ~20 TDs forever. Marcel's full form
      adds N "league average" plate appearances; we should do the same
      with N league-average games per position.
    - **No team context.** A player who lost their starting QB or got
      traded to a worse offense is projected as if nothing changed.
      Team-level projections need to feed back in here as opportunity
      adjustments.
    - **No rookie handling.** A first-year player has only 1 season of
      data and the weighted mean is just that one season. Needs a
      separate rookie-projection sub-model.
"""

from __future__ import annotations

import polars as pl

from sgf_model.curves import age_multiplier
from sgf_model.projections.regression import priors_as_dict

# Stats we project (must have age curves). Anything else from the player_seasons
# table is dropped — we have no aging information for it.
_PROJECTED_STATS_BY_POSITION: dict[str, list[str]] = {
    "QB": [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],
    "RB": [
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
    "WR": [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
    "TE": [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
}


def _weighted_history(
    player_history: pl.DataFrame,
    stats: list[str],
    history_weights: tuple[float, ...],
    priors: dict[tuple[str, str], tuple[float, float]] | None = None,
    position: str | None = None,
) -> dict[str, float] | None:
    """Return weighted per-game baselines + projected games_played for one player.

    `player_history` is the player's rows from the player_seasons table,
    already sorted descending by season (most recent first). The most recent
    `len(history_weights)` seasons are used.

    If `priors` is provided, Marcel-style empirical-Bayes shrinkage is applied
    to each per-game baseline:
        shrunk = (sample_weight * raw + N * pop_mean) / (sample_weight + N)
    where `sample_weight` is the player's total weighted game-units of evidence
    and `N` is the per-stat regression amount derived in `fit_regression_priors`.
    """
    recent = player_history.head(len(history_weights))
    if recent.is_empty():
        return None

    # Align weights to however many seasons the player actually has.
    weights = list(history_weights[: recent.height])
    games = recent["games_played"].to_list()
    combined = [w * g for w, g in zip(weights, games)]
    total_weight = sum(combined)
    if total_weight == 0:
        return None

    baselines: dict[str, float] = {}
    for stat in stats:
        per_game_values = [
            row[stat] / row["games_played"] if row["games_played"] else 0.0
            for row in recent.iter_rows(named=True)
        ]
        raw = sum(w * v for w, v in zip(combined, per_game_values)) / total_weight
        if priors is not None and position is not None and (position, stat) in priors:
            pop_mean, n = priors[(position, stat)]
            baselines[stat] = (total_weight * raw + n * pop_mean) / (total_weight + n)
        else:
            baselines[stat] = raw

    # Games played is its own projection — straight weighted mean over recent seasons.
    # Not shrunk: games-played has its own dynamics (injury history, age) that the
    # population mean would obscure. A games-played-vs-age model is the better fix.
    baselines["games_played"] = sum(w * g for w, g in zip(weights, games)) / sum(weights)
    return baselines


def project_players(
    player_seasons: pl.DataFrame,
    curves: pl.DataFrame,
    as_of_season: int = 2024,
    n_future_seasons: int = 5,
    history_weights: tuple[float, ...] = (5.0, 4.0, 3.0),
    regression_priors: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Project per-game and season-total stats forward for every active player.

    Args:
        player_seasons: Output of `sgf_model.data.load_player_seasons`. Must
            include rows through `as_of_season`.
        curves: Delta-method age curves from `fit_age_curves`.
        as_of_season: The most recent completed season. Players with games in
            this season are considered active and get projections.
        n_future_seasons: How many seasons forward to project (typical dynasty
            horizon is 3-5).
        history_weights: Marcel-style weights on the most recent N seasons,
            most recent first. Defaults to (5, 4, 3): the last season is
            weighted 5x relative to two seasons ago at 3x, before multiplying
            by games played.
        regression_priors: Optional output of `fit_regression_priors`. When
            provided, applies empirical-Bayes shrinkage toward the population
            mean per (position, stat) — recommended for projection quality,
            especially for small-sample players (rookies, injury-shortened
            careers). Omit for the raw weighted-history baseline.

    Returns:
        Long-format DataFrame with one row per (player, projected_season):
            player_id | player_name | position | season | age |
            games_played_proj | <stat>_per_game | <stat>_season
        Per-game and per-season values are provided for every stat with a
        curve for that position.
    """
    priors_dict = priors_as_dict(regression_priors) if regression_priors is not None else None

    active_ids = (
        player_seasons.filter(pl.col("season") == as_of_season)
        .select("player_id")
        .unique()
    )
    history = (
        player_seasons.join(active_ids, on="player_id", how="inner")
        .filter(pl.col("season") <= as_of_season)
        .sort(["player_id", "season"], descending=[False, True])
    )

    out_rows: list[dict] = []
    for player_id, group in history.group_by("player_id"):
        player_id_str = player_id[0] if isinstance(player_id, tuple) else player_id
        latest = group.row(0, named=True)
        position = latest["position"]
        if position not in _PROJECTED_STATS_BY_POSITION:
            continue

        stats = _PROJECTED_STATS_BY_POSITION[position]
        baselines = _weighted_history(
            group, stats, history_weights, priors=priors_dict, position=position
        )
        if baselines is None or latest["age"] is None:
            continue

        current_age = int(latest["age"])
        games_proj = baselines["games_played"]

        for offset in range(1, n_future_seasons + 1):
            future_season = as_of_season + offset
            future_age = current_age + offset
            row: dict = {
                "player_id": player_id_str,
                "player_name": latest["player_name"],
                "position": position,
                "season": future_season,
                "age": future_age,
                "games_played_proj": games_proj,
            }
            for stat in stats:
                mult = age_multiplier(
                    curves, position, stat, from_age=current_age, to_age=future_age
                )
                # If we have no curve for this stat (NaN), hold the baseline flat —
                # better than dropping the column entirely. This shouldn't happen
                # for stats in _PROJECTED_STATS_BY_POSITION but is a safe fallback.
                if mult != mult:  # NaN check
                    mult = 1.0
                per_game = baselines[stat] * mult
                row[f"{stat}_per_game"] = per_game
                row[f"{stat}_season"] = per_game * games_proj
            out_rows.append(row)

    return pl.DataFrame(out_rows)
