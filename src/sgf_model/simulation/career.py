"""Career Monte Carlo simulator: turn multi-year quantile predictions into
career VORP distributions and a configurable master ranking.

Three steps:
    1. `sample_career_fps`: rank-coupled sampling — per simulation, draw a
       single quantile rank u per player, use it to query the quantile model
       at each future year (with piecewise-linear interpolation between the
       five known quantiles, linear extrapolation beyond P10/P90). Optional
       `year_noise_alpha` blends in fresh per-year randomness to dial down
       the comonotonic assumption.
    2. `sample_career_vorps`: per-sample replacement levels — within each
       simulation, rank players' sampled FPs per (position, year), grab the
       replacement-rank player's FP, subtract, clip negatives, discount, sum.
    3. `summarize_career`: percentiles + configurable ranking scores
       (mean, median, risk-adjusted median, P25 floor, P(VORP > threshold)).

Rank-coupling sampling assumes a player's "true skill" sits at a fixed
quantile across the horizon — comonotonic, an over-correlation of reality.
Independent sampling per year would under-correlate. The `year_noise_alpha`
parameter blends the two; default 0 (pure rank-coupling) is conservative for
career-VORP variance and a reasonable starting point.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from sgf_model.valuation.league import LeagueConfig

DEFAULT_N_SIMULATIONS: int = 1000

# Quantile levels emitted by QuantileFPModel — used as the known points for
# the piecewise-linear inverse CDF interpolation.
_KNOWN_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
_QUANTILE_COL_MAP: dict[float, str] = {
    0.10: "proj_fp_lower_80",
    0.25: "proj_fp_lower_50",
    0.50: "fantasy_points_season",
    0.75: "proj_fp_upper_50",
    0.90: "proj_fp_upper_80",
}

# Score names exposed by master_ranking. Each maps to a function taking a
# (n_simulations,) array of one player's career VORP samples and returning
# a scalar score used for sorting.
DEFAULT_RANKING_SCORES: dict[str, str] = {
    "mean": "Mean career VORP — risk-neutral",
    "median": "P50 career VORP — robust central tendency",
    "risk_adjusted_0.5": "median − 0.5 × SD — mild risk aversion",
    "risk_adjusted_1.0": "median − 1.0 × SD — strong risk aversion",
    "p25_floor": "P25 career VORP — pessimistic floor",
    "p_positive": "P(career VORP > 0) — usefulness probability",
}


def _quantile_tensor(predictions: pl.DataFrame, horizon: int) -> tuple[
    np.ndarray, np.ndarray, list[str], list[str]
]:
    """Reshape predictions DataFrame into a (n_players, horizon, K) quantile tensor.

    Returns:
        quantiles: ndarray of shape (n_players, horizon, 5) — sorted ascending
            across the last axis (P10..P90).
        positions: ndarray of shape (n_players,) — string position per player.
        player_ids: list of player ids in matrix order.
        player_names: list of player names in matrix order.
    """
    needed_cols = ["player_id", "player_name", "position", "future_offset"] + list(
        _QUANTILE_COL_MAP.values()
    )
    df = predictions.select(needed_cols).with_columns(
        future_offset=pl.col("future_offset").cast(pl.Int32),
    )

    # Validate one row per (player, offset) and full coverage.
    offsets = sorted(df["future_offset"].unique().to_list())
    if offsets != list(range(1, horizon + 1)):
        raise ValueError(
            f"Expected offsets 1..{horizon} in predictions; got {offsets}. "
            "Pass a `predictions` frame that covers every horizon year."
        )

    df = df.sort(["player_id", "future_offset"])
    players = (
        df.select("player_id", "player_name", "position")
        .unique()
        .sort("player_id")
    )
    player_ids = players["player_id"].to_list()
    player_names = players["player_name"].to_list()
    positions = players["position"].to_numpy()

    # Build (n_players, horizon, 5) tensor by sorting and reshaping.
    qs = np.stack(
        [df[c].to_numpy().reshape(len(player_ids), horizon) for c in _QUANTILE_COL_MAP.values()],
        axis=-1,
    )
    # Sort each row across the quantile axis to enforce monotonicity
    # (model output is already sorted but be defensive).
    qs.sort(axis=-1)
    return qs, positions, player_ids, player_names


def _inverse_cdf_sample(
    quantiles: np.ndarray,
    levels: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Piecewise-linear inverse CDF sampling with linear-extrapolation tails.

    quantiles: (..., K) of FP values at corresponding `levels`.
    levels: (K,) sorted ascending in [0, 1].
    u: (...,) uniform draws to sample at.

    Returns: (...,) sampled FP values.
    """
    K = len(levels)
    # Per-row bracket index: which (k, k+1) pair surrounds u.
    k_lo = np.searchsorted(levels, u, side="right") - 1
    k_lo = np.clip(k_lo, 0, K - 2)
    k_hi = k_lo + 1

    # Gather quantile values at k_lo / k_hi for each row.
    # quantiles shape: (..., K); k_lo shape: (...,)
    # Use take_along_axis for efficient row-wise lookup.
    k_lo_exp = k_lo[..., None]
    k_hi_exp = k_hi[..., None]
    q_lo = np.take_along_axis(quantiles, k_lo_exp, axis=-1).squeeze(-1)
    q_hi = np.take_along_axis(quantiles, k_hi_exp, axis=-1).squeeze(-1)
    l_lo = levels[k_lo]
    l_hi = levels[k_hi]

    t = (u - l_lo) / (l_hi - l_lo)
    return q_lo + t * (q_hi - q_lo)


def sample_career_fps(
    predictions: pl.DataFrame,
    horizon: int,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    year_noise_alpha: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Sample N career FP trajectories per player via rank-coupled draws.

    For each simulation `s` and player `p`:
        - Draw u_p ~ Uniform(0, 1) (the player's career quantile rank)
        - For each year y: u_{p,y} = (1-α) * u_p + α * Uniform()
        - Sample FP at quantile u_{p,y} via piecewise-linear interpolation
          between the known P10/P25/P50/P75/P90 values for that year.

    `year_noise_alpha = 0` → fully comonotonic (a P90 player stays P90 every year).
    `year_noise_alpha = 1` → independent draws per year (loses career correlation).

    Returns:
        fps: (n_simulations, n_players, horizon) — sampled FP per (sim, player, year)
        positions: (n_players,) string position per player
        player_ids: list of player ids in matrix order
        player_names: list of player names in matrix order
    """
    if not 0 <= year_noise_alpha <= 1:
        raise ValueError(f"year_noise_alpha must be in [0, 1]; got {year_noise_alpha}")

    qs, positions, player_ids, player_names = _quantile_tensor(predictions, horizon)
    n_players = qs.shape[0]
    levels = np.array(_KNOWN_QUANTILES)

    rng = np.random.default_rng(seed)
    u_player = rng.uniform(size=(n_simulations, n_players))[..., None]  # (S, P, 1)
    if year_noise_alpha > 0:
        u_year = rng.uniform(size=(n_simulations, n_players, horizon))
        u = (1 - year_noise_alpha) * u_player + year_noise_alpha * u_year
    else:
        u = np.broadcast_to(u_player, (n_simulations, n_players, horizon))
    u = np.clip(u, 1e-6, 1 - 1e-6)

    # Broadcast qs (P, H, K) across simulations: result is (S, P, H, K) via expand
    # — but we don't need to materialize that; use broadcasting in _inverse_cdf_sample.
    # _inverse_cdf_sample's quantiles arg can be (P, H, K) and u can be (S, P, H);
    # however take_along_axis requires matching shape. So we broadcast qs first.
    qs_broadcast = np.broadcast_to(qs[None, ...], (n_simulations, n_players, horizon, qs.shape[-1]))
    fps = _inverse_cdf_sample(qs_broadcast, levels, u)
    return fps, positions, player_ids, player_names


def sample_career_vorps(
    fp_samples: np.ndarray,
    positions: np.ndarray,
    league: LeagueConfig,
    discount_rate: float = 0.15,
    clip_negative: bool = True,
) -> np.ndarray:
    """Per-sample career VORP using sample-specific replacement levels.

    For each (simulation, position, year), the replacement-rank FP is the
    value at `league.replacement_rank()[position]` after sorting that
    (sim, position, year) slice descending. Per-player per-year VORP is
    sample-fp minus that replacement; clipped at zero (unstarted player =
    zero value), discounted, summed across years.

    Args:
        fp_samples: (n_simulations, n_players, horizon) — output of sample_career_fps.
        positions: (n_players,) — string position per player.
        league: LeagueConfig with `replacement_rank()`.
        discount_rate: Annual discount (default 0.15).
        clip_negative: If True, treat negative per-year VORP as 0 for the sum.

    Returns:
        (n_simulations, n_players) — career VORP samples.
    """
    n_sims, n_players, horizon = fp_samples.shape
    replacement_ranks = league.replacement_rank()

    vorps_by_year = np.zeros_like(fp_samples)
    for pos, rank_N in replacement_ranks.items():
        idx = np.where(positions == pos)[0]
        if len(idx) == 0:
            continue
        fps_pos = fp_samples[:, idx, :]  # (S, n_pos, H)
        # Sort descending along the n_pos axis (per-sim, per-year).
        fps_sorted = -np.sort(-fps_pos, axis=1)
        rep_idx = min(rank_N - 1, fps_sorted.shape[1] - 1)
        replacement_fps = fps_sorted[:, rep_idx, :]  # (S, H)
        vorps_pos = fps_pos - replacement_fps[:, None, :]
        vorps_by_year[:, idx, :] = vorps_pos

    if clip_negative:
        vorps_by_year = np.maximum(vorps_by_year, 0)

    # Discount factors per year offset (1..horizon).
    year_offsets = np.arange(1, horizon + 1)
    discount = (1.0 + discount_rate) ** (-year_offsets)
    return (vorps_by_year * discount[None, None, :]).sum(axis=2)


def summarize_career(
    player_ids: list[str],
    player_names: list[str],
    positions: np.ndarray,
    career_vorps: np.ndarray,
    percentiles: tuple[int, ...] = (10, 25, 50, 75, 90),
) -> pl.DataFrame:
    """Per-player career VORP summary stats.

    Returns a DataFrame with one row per player and columns:
        player_id, player_name, position,
        mean, std,
        p10, p25, p50, p75, p90 (or whatever `percentiles` requests),
        risk_adjusted_0.5, risk_adjusted_1.0,
        p_positive  (P(career VORP > 0) across simulations)
    """
    mean_v = career_vorps.mean(axis=0)
    std_v = career_vorps.std(axis=0)
    pcts = {f"p{p}": np.percentile(career_vorps, p, axis=0) for p in percentiles}
    p_positive = (career_vorps > 0).mean(axis=0)

    cols = {
        "player_id": player_ids,
        "player_name": player_names,
        "position": positions.tolist(),
        "mean": mean_v,
        "std": std_v,
        **pcts,
        "risk_adjusted_0.5": pcts["p50"] - 0.5 * std_v,
        "risk_adjusted_1.0": pcts["p50"] - 1.0 * std_v,
        "p_positive": p_positive,
    }
    return pl.DataFrame(cols)


def master_ranking(
    summary: pl.DataFrame,
    score: str = "median",
    descending: bool = True,
) -> pl.DataFrame:
    """Sort the career summary by a configurable score column.

    `score` must be one of:
        - "mean"
        - "median" (== p50)
        - "risk_adjusted_0.5"  (median - 0.5 * SD)
        - "risk_adjusted_1.0"  (median - 1.0 * SD)
        - "p25_floor"          (== p25)
        - "p_positive"

    Returns the summary with an added `rank` column (1-indexed) sorted by score.
    """
    score_col = {
        "mean": "mean",
        "median": "p50",
        "risk_adjusted_0.5": "risk_adjusted_0.5",
        "risk_adjusted_1.0": "risk_adjusted_1.0",
        "p25_floor": "p25",
        "p_positive": "p_positive",
    }.get(score)
    if score_col is None:
        raise ValueError(
            f"Unknown score {score!r}. Options: {list(DEFAULT_RANKING_SCORES.keys())}"
        )
    out = summary.sort(score_col, descending=descending).with_columns(
        rank=pl.int_range(1, summary.height + 1, eager=True),
    )
    return out.select("rank", *[c for c in summary.columns if c != "rank"])
