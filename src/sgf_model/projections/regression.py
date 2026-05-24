"""Empirical Bayes regression priors for projections.

Marcel-style shrinkage requires a regression amount `N` (in game-units of
league-average production to add to each player's sample). Hand-tuning N
is fine but ad-hoc; empirical Bayes derives the *optimal* N from the data
under a Gaussian-prior approximation:

    observed_per_game = true_player_rate + noise
    true_player_rate  ~ Normal(population_mean, between_var)
    noise per game    ~ Normal(0, within_var_per_game)

The posterior mean of the player's true rate becomes a weighted average of
their observed rate and the population mean, with weight on the player
proportional to `games_played` and weight on the prior equal to
`N = within_var_per_game / between_var`.

**Critical scale point.** `within_var_per_game` is the variance of a single
*game's* per-game stat value around the player's true rate — not the variance
of *season means* around the player mean. These differ by a factor of
~`games_per_season`. We compute the per-game variance directly from weekly
data (variance of weekly stats around each player-season mean), which is the
mathematically correct σ² for the formula.

The between-player variance is also corrected for sampling noise: observed
season means are noisy estimates of true rate, so the naive between-variance
overestimates τ² by `mean(σ²/games_per_season)`. We subtract that off.

Stats with high `N`:
    - TDs (rare events, high game-to-game noise relative to player skill
      differences). Strongly regressed.
Stats with low `N`:
    - Targets, yards (large player-to-player skill gap, lower relative noise).
      Lightly regressed.

The Gaussian approximation is loose for count stats (TDs, receptions), but
it's the standard empirical-Bayes form and produces sensible N values; a
Negative Binomial / Beta-Binomial treatment would be a future refinement.
"""

from __future__ import annotations

import polars as pl

# Floor relative to per-game variance so it scales with the stat's natural noise.
# Prevents pathological N when sampling correction drives between_var near zero.
_BETWEEN_VAR_FLOOR_RATIO = 0.01
# Cap N to avoid pathological shrinkage when between_var collapses near the floor.
# 200 game-units ≈ 12 full NFL seasons of regression — heavy but bounded.
_N_CAP = 200.0


def fit_regression_priors(
    weekly_stats: pl.DataFrame,
    stats_by_position: dict[str, list[str]] | None = None,
    as_of_season: int = 2024,
    recent_seasons: int = 5,
    min_games_per_season: int = 8,
    n_cap: float = _N_CAP,
) -> pl.DataFrame:
    """Fit empirical Bayes shrinkage priors per (position, stat) from weekly data.

    Args:
        weekly_stats: Output of `sgf_model.data.load_weekly_stats` (regular
            season only), already filtered to fantasy positions. Must have
            one row per (player, season, week).
        stats_by_position: Which (position, stat) pairs to fit. Defaults to
            the same set the projection layer projects.
        as_of_season: Anchor season; prior window ends here.
        recent_seasons: Number of seasons of history to use (default 5 —
            recent enough to reflect current NFL passing rates, deep enough
            for stable variance estimates).
        min_games_per_season: Drop player-seasons with fewer games (noise floor
            for both variance estimation and the population mean).
        n_cap: Maximum regression amount (in game-units) to prevent unstable
            shrinkage when between-player variance is estimated near zero.

    Returns:
        DataFrame with one row per (position, stat):
            position | stat | pop_mean_per_game | within_var_per_game |
            between_var | regression_n
    """
    # Late import to avoid circularity with player.py.
    from sgf_model.projections.player import _PROJECTED_STATS_BY_POSITION

    stats_by_position = stats_by_position or _PROJECTED_STATS_BY_POSITION
    season_lo = as_of_season - recent_seasons + 1

    weekly = weekly_stats.filter(pl.col("season").is_between(season_lo, as_of_season))

    # Per-(player, season) game counts; restrict to qualified player-seasons.
    games_per_ps = weekly.group_by(["player_id", "season"]).agg(games=pl.len())
    qualified = games_per_ps.filter(pl.col("games") >= min_games_per_season)
    weekly = weekly.join(qualified.select("player_id", "season"), on=["player_id", "season"])

    rows: list[dict] = []
    for position, stats in stats_by_position.items():
        pos_weekly = weekly.filter(pl.col("position") == position)
        if pos_weekly.is_empty():
            continue

        # Per-(player, season) summary — needed for both within- and between-variance.
        ps_summary = pos_weekly.group_by(["player_id", "season"]).agg(
            games=pl.len(),
            **{f"{s}_mean": pl.col(s).mean() for s in stats},
            **{f"{s}_var": pl.col(s).var(ddof=1) for s in stats},
        )

        for stat in stats:
            ps_with_var = ps_summary.filter(pl.col(f"{stat}_var").is_not_null())
            if ps_with_var.is_empty():
                continue

            total_games = float(ps_with_var["games"].sum())

            # σ² (per-game): games-weighted mean of within-(player,season) variance.
            sigma2 = float(
                (ps_with_var[f"{stat}_var"] * ps_with_var["games"]).sum() / total_games
            )

            # Population mean of per-game rate, games-weighted across player-seasons.
            pop_mean = float(
                (ps_with_var[f"{stat}_mean"] * ps_with_var["games"]).sum() / total_games
            )

            # Naive variance of season means around pop_mean (games-weighted).
            naive_between = float(
                ((ps_with_var[f"{stat}_mean"] - pop_mean) ** 2 * ps_with_var["games"]).sum()
                / total_games
            )

            # Subtract average sampling variance of each season mean (σ²/games_in_that_season).
            # This is the standard method-of-moments correction so we don't double-count
            # within-season noise as between-player variance.
            mean_sampling_var = float(
                (sigma2 / ps_with_var["games"] * ps_with_var["games"]).sum() / total_games
            )
            # Note: simplifies to sigma2 * n_player_seasons / total_games, but written
            # explicitly to keep the formula's intent visible.

            tau2 = max(naive_between - mean_sampling_var, sigma2 * _BETWEEN_VAR_FLOOR_RATIO)
            n = min(sigma2 / tau2, n_cap)

            rows.append(
                {
                    "position": position,
                    "stat": stat,
                    "pop_mean_per_game": pop_mean,
                    "within_var_per_game": sigma2,
                    "between_var": tau2,
                    "regression_n": n,
                }
            )

    return pl.DataFrame(rows)


def priors_as_dict(
    priors: pl.DataFrame,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Convert a priors DataFrame to a dict for fast per-stat lookup.

    Returns: {(position, stat): (pop_mean_per_game, regression_n)}
    """
    return {
        (row["position"], row["stat"]): (row["pop_mean_per_game"], row["regression_n"])
        for row in priors.iter_rows(named=True)
    }
