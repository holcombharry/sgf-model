"""Rookie projection sub-model.

The main `project_players` function can't handle rookies — they have no NFL
history to feed the weighted-history baseline. This module fills that gap:
predict rookie-year per-game stats from pre-NFL features (draft pick + age),
then hand off to the same age-curve / scoring / valuation machinery that
projects veterans.

Method (v1):
    1. Training set = every drafted fantasy-position rookie whose first NFL
       season had >= `min_games_train` games played. Features: log(draft_pick),
       age at rookie season.
    2. For each (position, stat), fit a ridge regression of per-game stat on
       the features. Numpy `lstsq` with explicit ridge penalty — no extra dep.
    3. Same machinery for projecting games_played in the rookie year.
    4. For a new rookie class, predict per-game stats + games_played, then
       output in the same long-format shape as `project_players` so the
       outputs can simply be concatenated.

Deliberately out of scope for v1 (documented gaps):
    - **College production stats.** Adds signal but requires a separate data
      pipeline (cfbfastR/cfbd). Future addition.
    - **Combine measurements.** Not all rookies combine; marginal extra signal.
    - **UDFAs.** Survivorship-bias-heavy (we only see UDFAs who made rosters);
      need their own treatment. Currently dropped from training and prediction.
    - **Sophomore-leap correction.** Age curves are calibrated against the
      mix of rookies/2nd-year/3rd-year players at each age, so they implicitly
      capture *some* experience effect. Could be improved with an explicit
      experience curve on top of age curve.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from sgf_model.curves import age_multiplier

# Same stats projected for vets — keeps output shape compatible with project_players.
_ROOKIE_STATS_BY_POSITION: dict[str, list[str]] = {
    "QB": [
        "attempts", "completions", "passing_yards", "passing_tds",
        "passing_interceptions", "carries", "rushing_yards", "rushing_tds",
    ],
    "RB": [
        "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ],
    "WR": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ],
    "TE": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ],
}

# Restrict to standard 7-round drafts (modern era). Earlier drafts had more rounds
# but those mappings to current NFL economics are sketchy.
_MAX_DRAFT_ROUND: int = 7


def build_rookie_training_data(
    player_seasons: pl.DataFrame,
    players_meta: pl.DataFrame,
) -> pl.DataFrame:
    """Per-rookie training row: features + rookie-year stats.

    A "rookie season" is a player's first NFL season that appears in
    `player_seasons` AND matches their `draft_year` from `players_meta`
    (so we exclude players who sat out / didn't play their draft year and
    then started later — different selection dynamics).
    """
    draft_info = players_meta.filter(
        pl.col("draft_pick").is_not_null()
        & pl.col("draft_round").is_not_null()
        & (pl.col("draft_round") <= _MAX_DRAFT_ROUND)
    ).select(
        pl.col("gsis_id"),
        pl.col("draft_year"),
        pl.col("draft_round"),
        pl.col("draft_pick"),
    )

    rookies = (
        player_seasons.join(
            draft_info, left_on="player_id", right_on="gsis_id", how="inner"
        )
        .filter(pl.col("season") == pl.col("draft_year"))
        .filter(pl.col("position").is_in(list(_ROOKIE_STATS_BY_POSITION.keys())))
    )
    return rookies


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge_alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression. Intercept (first column) is not penalized."""
    XtX = X.T @ X
    penalty = ridge_alpha * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(XtX + penalty, X.T @ y)


def _features(draft_pick: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Design matrix: [intercept, log(draft_pick), age]."""
    return np.column_stack([
        np.ones(len(draft_pick)),
        np.log(draft_pick.astype(float)),
        age.astype(float),
    ])


def fit_rookie_models(
    training_data: pl.DataFrame,
    stats_by_position: dict[str, list[str]] | None = None,
    min_games_train: int = 4,
    ridge_alpha: float = 1.0,
) -> dict[tuple[str, str], np.ndarray]:
    """Fit one ridge model per (position, stat).

    Returns:
        {(position, stat): np.array([intercept, coef_log_pick, coef_age])}
        Plus the special key `(position, "games_played")` for the games model.
    """
    stats_by_position = stats_by_position or _ROOKIE_STATS_BY_POSITION
    qualified = training_data.filter(pl.col("games_played") >= min_games_train)

    models: dict[tuple[str, str], np.ndarray] = {}
    for position, stats in stats_by_position.items():
        pos_df = qualified.filter(pl.col("position") == position)
        if pos_df.height < 10:
            continue

        draft_pick = pos_df["draft_pick"].to_numpy()
        age = pos_df["age"].to_numpy()
        games = pos_df["games_played"].to_numpy().astype(float)
        X = _features(draft_pick, age)

        # Per-stat models on per-game rates
        for stat in stats:
            y = pos_df[stat].to_numpy().astype(float) / games
            models[(position, stat)] = _fit_ridge(X, y, ridge_alpha)

        # Games-played model — fit on ALL drafted rookies (no min_games filter)
        # to avoid the obvious selection bias of "rookies who played a lot played a lot".
        all_pos = training_data.filter(pl.col("position") == position)
        X_all = _features(all_pos["draft_pick"].to_numpy(), all_pos["age"].to_numpy())
        y_games = all_pos["games_played"].to_numpy().astype(float)
        models[(position, "games_played")] = _fit_ridge(X_all, y_games, ridge_alpha)

    return models


def _predict(coefs: np.ndarray, draft_pick: float, age: float) -> float:
    """Apply a fitted ridge model to a single rookie."""
    intercept, log_pick_coef, age_coef = coefs
    return float(intercept + log_pick_coef * np.log(draft_pick) + age_coef * age)


def project_rookies(
    rookie_class: pl.DataFrame,
    models: dict[tuple[str, str], np.ndarray],
    curves: pl.DataFrame,
    as_of_season: int,
    n_future_seasons: int = 5,
    stats_by_position: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """Project per-game + per-season stats for an incoming rookie class.

    Args:
        rookie_class: One row per rookie. Required columns:
            player_id | player_name | position | draft_pick | age
            where `age` is the rookie's age at the rookie season.
        models: Output of `fit_rookie_models`.
        curves: Delta-method age curves (same as for `project_players`).
        as_of_season: Season *before* the rookie season. Rookies' first
            projected season is `as_of_season + 1`. (This matches the
            semantics of `project_players` for clean concat.)
        n_future_seasons: How many seasons forward to project, starting at
            the rookie season.

    Returns:
        DataFrame with the same columns as `project_players` output, so the
        two can be concatenated directly:
            player_id | player_name | position | season | age |
            games_played_proj | <stat>_per_game | <stat>_season
    """
    stats_by_position = stats_by_position or _ROOKIE_STATS_BY_POSITION
    out_rows: list[dict] = []

    for row in rookie_class.iter_rows(named=True):
        position = row["position"]
        if position not in stats_by_position:
            continue
        if (position, "games_played") not in models:
            continue

        rookie_age = int(row["age"])
        draft_pick = float(row["draft_pick"])

        games_proj = max(0.0, min(17.0, _predict(models[(position, "games_played")], draft_pick, rookie_age)))

        # Baseline per-game rates predicted by the rookie model (negative predictions
        # floor at 0 — happens for rare stats with late-round picks).
        baselines: dict[str, float] = {}
        for stat in stats_by_position[position]:
            key = (position, stat)
            if key not in models:
                continue
            baselines[stat] = max(0.0, _predict(models[key], draft_pick, rookie_age))

        for offset in range(1, n_future_seasons + 1):
            future_season = as_of_season + offset
            future_age = rookie_age + (offset - 1)  # offset=1 → rookie season at rookie_age
            out_row: dict = {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "position": position,
                "season": future_season,
                "age": future_age,
                "games_played_proj": games_proj,
            }
            for stat, baseline in baselines.items():
                mult = age_multiplier(curves, position, stat, from_age=rookie_age, to_age=future_age)
                if mult != mult:  # NaN
                    mult = 1.0
                per_game = baseline * mult
                out_row[f"{stat}_per_game"] = per_game
                out_row[f"{stat}_season"] = per_game * games_proj
            out_rows.append(out_row)

    return pl.DataFrame(out_rows)


def make_rookie_class_from_history(
    player_seasons: pl.DataFrame,
    players_meta: pl.DataFrame,
    rookie_season: int,
) -> pl.DataFrame:
    """Helper: assemble a `rookie_class` DataFrame for a given historical season.

    Useful for backtesting the rookie model. For forward projection (e.g.
    projecting the upcoming 2025 class), the user constructs `rookie_class`
    from their own draft data.
    """
    draft_info = players_meta.filter(
        (pl.col("draft_year") == rookie_season)
        & pl.col("draft_pick").is_not_null()
        & (pl.col("draft_round") <= _MAX_DRAFT_ROUND)
    ).select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("draft_pick"),
        pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False).alias("birth_date"),
    )

    # Player display info + position come from any player_seasons row we can find,
    # falling back to players_meta if they never showed up in the stats table.
    name_pos_from_seasons = (
        player_seasons.group_by("player_id").agg(
            player_name=pl.col("player_name").first(),
            position=pl.col("position").first(),
        )
    )
    name_pos_from_meta = players_meta.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("display_name").alias("player_name_meta"),
        pl.col("position").alias("position_meta"),
    )

    out = (
        draft_info
        .join(name_pos_from_seasons, on="player_id", how="left")
        .join(name_pos_from_meta, on="player_id", how="left")
        .with_columns(
            player_name=pl.coalesce("player_name", "player_name_meta"),
            position=pl.coalesce("position", "position_meta"),
        )
        .filter(pl.col("position").is_in(list(_ROOKIE_STATS_BY_POSITION.keys())))
        .with_columns(
            age=(rookie_season - pl.col("birth_date").dt.year()).cast(pl.Int32),
        )
        .filter(pl.col("age").is_not_null())
        .select(["player_id", "player_name", "position", "draft_pick", "age"])
    )
    return out
