"""Career VORP calibration backtest — the Phase 3 exit criterion we deferred.

For one or more historical anchor seasons:
    1. Train the v2 model on data through `anchor_season` only (no leakage).
    2. Sample career-VORP distributions for each player active at `anchor_season`
       over years anchor_season+1 .. anchor_season+horizon.
    3. Compute realized career VORP for the same players from actual data over
       those years (using actual league replacement levels per year).
    4. Check what fraction of realized career VORPs fall inside the model's
       predicted 50% / 80% intervals.

Well-calibrated → coverage matches the target (cov_50 ≈ 0.5, cov_80 ≈ 0.8).
Under-covered → intervals too narrow (model overconfident).
Over-covered → intervals too wide (model under-confident).

This validates the architecture's main claim: that career-VORP distributional
output is meaningful, not just sensible-looking.
"""

from __future__ import annotations

import polars as pl

from sgf_model.features import (
    PHASE3_FEATURE_COLUMNS,
    build_feature_matrix,
)
from sgf_model.models import QuantileFPModel
from sgf_model.scoring import ScoringConfig, score_projections
from sgf_model.simulation import (
    sample_career_fps,
    sample_career_vorps,
    summarize_career,
)
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


def _score_actuals(player_seasons: pl.DataFrame, scoring: ScoringConfig) -> pl.DataFrame:
    """FP per (player_id, season) under the given scoring, for all available seasons."""
    renamed = player_seasons.rename(
        {k: v for k, v in _STAT_COLS_RENAME.items() if k in player_seasons.columns}
    )
    scored = score_projections(renamed, scoring)
    return scored.select(
        "player_id", "player_name", "position", "season",
        pl.col("fantasy_points_season").alias("actual_fp"),
    )


def _realized_career_vorp(
    actual_fps: pl.DataFrame,
    test_players: list[str],
    anchor_season: int,
    horizon: int,
    league: LeagueConfig,
    discount_rate: float,
) -> dict[str, float]:
    """Compute realized career VORP per player over years anchor+1..anchor+horizon.

    For each (position, year), the replacement FP is taken from the actual
    position rank N among players who played that year. Each test player's
    per-year VORP is `max(their actual FP - replacement, 0)`, then summed with
    annual discount. A player who didn't play in year Y contributes 0 to that
    year's VORP (= clipped negative).

    Returns: {player_id: realized_career_vorp}
    """
    repl_ranks = league.replacement_rank()
    discount = {y: (1.0 + discount_rate) ** -(y - anchor_season)
                for y in range(anchor_season + 1, anchor_season + horizon + 1)}

    # Per-year replacement FP per position from actuals.
    realized: dict[str, float] = {pid: 0.0 for pid in test_players}
    for year in range(anchor_season + 1, anchor_season + horizon + 1):
        year_actuals = actual_fps.filter(pl.col("season") == year)
        if year_actuals.height == 0:
            continue
        for position, rep_rank in repl_ranks.items():
            pos_year = year_actuals.filter(pl.col("position") == position)
            if pos_year.height == 0:
                continue
            sorted_fps = pos_year.sort("actual_fp", descending=True)
            replacement_fp = (
                sorted_fps["actual_fp"][rep_rank - 1]
                if pos_year.height >= rep_rank
                else sorted_fps["actual_fp"][-1]
            )
            # For each test player in this position who played this year:
            pos_test_fps = pos_year.filter(
                pl.col("player_id").is_in(test_players)
            )
            for row in pos_test_fps.iter_rows(named=True):
                vorp_year = max(row["actual_fp"] - replacement_fp, 0.0)
                realized[row["player_id"]] += vorp_year * discount[year]
    return realized


def career_calibration_at_anchor(
    player_seasons: pl.DataFrame,
    advanced_features: pl.DataFrame,
    draft_features: pl.DataFrame,
    anchor_season: int,
    horizon: int,
    league: LeagueConfig,
    scoring: ScoringConfig,
    rookies: pl.DataFrame | None = None,
    n_simulations: int = 1000,
    discount_rate: float = 0.15,
    model_params: dict | None = None,
    random_state: int = 42,
) -> pl.DataFrame:
    """Run career-VORP calibration at one anchor season.

    Returns a per-player DataFrame:
        player_id, player_name, position, anchor_season,
        predicted_p10/p25/p50/p75/p90 (career VORP),
        realized_career_vorp,
        inside_50, inside_80   (booleans: in [P25,P75] / in [P10,P90])
    """
    train_ps = player_seasons.filter(pl.col("season") <= anchor_season)
    train_adv = advanced_features.filter(pl.col("season") <= anchor_season)
    rookies_train = (
        rookies.filter(pl.col("draft_year") <= anchor_season)
        if rookies is not None else None
    )

    fm_train = build_feature_matrix(
        train_ps, scoring,
        advanced_features=train_adv, draft_features=draft_features,
        rookies=rookies_train,
        forecast_horizon=horizon, include_inactive_targets=True,
    )
    fm_inf = build_feature_matrix(
        train_ps, scoring,
        advanced_features=train_adv, draft_features=draft_features,
        rookies=rookies,
        inference_season=anchor_season + 1, forecast_horizon=horizon,
        include_inactive_targets=True,
    )
    inference_rows = fm_inf.filter(pl.col("target_fp").is_null())

    model = QuantileFPModel(
        feature_columns=PHASE3_FEATURE_COLUMNS,
        params=(model_params or {"max_depth": 3}),
        random_state=random_state,
    ).fit(fm_train)
    preds = model.predict(inference_rows)

    fp_samples, positions, ids, names = sample_career_fps(
        preds, horizon=horizon, n_simulations=n_simulations, seed=random_state,
    )
    career_vorps = sample_career_vorps(
        fp_samples, positions, league, discount_rate=discount_rate,
    )
    summary = summarize_career(ids, names, positions, career_vorps)

    actuals = _score_actuals(player_seasons, scoring)
    realized = _realized_career_vorp(
        actuals, ids, anchor_season, horizon, league, discount_rate,
    )
    realized_df = pl.DataFrame({
        "player_id": list(realized.keys()),
        "realized_career_vorp": list(realized.values()),
    })

    out = (
        summary
        .join(realized_df, on="player_id", how="inner")
        .with_columns(
            anchor_season=pl.lit(anchor_season).cast(pl.Int32),
            inside_50=(pl.col("realized_career_vorp") >= pl.col("p25"))
                & (pl.col("realized_career_vorp") <= pl.col("p75")),
            inside_80=(pl.col("realized_career_vorp") >= pl.col("p10"))
                & (pl.col("realized_career_vorp") <= pl.col("p90")),
        )
    )
    return out


def career_calibration_backtest(
    player_seasons: pl.DataFrame,
    advanced_features: pl.DataFrame,
    draft_features: pl.DataFrame,
    anchor_seasons: list[int],
    horizon: int,
    league: LeagueConfig,
    scoring: ScoringConfig,
    n_simulations: int = 1000,
    discount_rate: float = 0.15,
    model_params: dict | None = None,
) -> pl.DataFrame:
    """Run calibration at multiple anchor seasons; concat results.

    Use multiple anchors so a single cohort's idiosyncrasies don't drive the
    coverage estimate. The locked holdout is 2021-2023; pick anchors with at
    least `horizon` years of realized data after them (e.g., anchor=2017
    gives 2018-2022 realized; anchor=2018 gives 2019-2023).
    """
    parts: list[pl.DataFrame] = []
    for anchor in anchor_seasons:
        part = career_calibration_at_anchor(
            player_seasons, advanced_features, draft_features,
            anchor_season=anchor,
            horizon=horizon, league=league, scoring=scoring,
            n_simulations=n_simulations, discount_rate=discount_rate,
            model_params=model_params,
        )
        parts.append(part)
    return pl.concat(parts)


def summarize_calibration(calibration_df: pl.DataFrame) -> pl.DataFrame:
    """Per-(anchor, position) coverage summary plus overall.

    Coverage at level L = fraction of test cases where realized career VORP
    falls inside the model's L% predictive interval. Well-calibrated means
    coverage matches L.
    """
    rows: list[dict] = []
    for (anchor, position), sub in calibration_df.group_by(
        ["anchor_season", "position"]
    ):
        rows.append({
            "anchor_season": anchor,
            "position": position,
            "n": sub.height,
            "cov_50": float(sub["inside_50"].mean()),
            "cov_80": float(sub["inside_80"].mean()),
            "realized_median": float(sub["realized_career_vorp"].median()),
            "predicted_median_p50": float(sub["p50"].median()),
        })
    # Overall (collapsing position) per anchor
    for (anchor,), sub in calibration_df.group_by(["anchor_season"]):
        rows.append({
            "anchor_season": anchor,
            "position": "ALL",
            "n": sub.height,
            "cov_50": float(sub["inside_50"].mean()),
            "cov_80": float(sub["inside_80"].mean()),
            "realized_median": float(sub["realized_career_vorp"].median()),
            "predicted_median_p50": float(sub["p50"].median()),
        })
    # Grand overall
    rows.append({
        "anchor_season": None,
        "position": "ALL",
        "n": calibration_df.height,
        "cov_50": float(calibration_df["inside_50"].mean()),
        "cov_80": float(calibration_df["inside_80"].mean()),
        "realized_median": float(calibration_df["realized_career_vorp"].median()),
        "predicted_median_p50": float(calibration_df["p50"].median()),
    })
    return pl.DataFrame(rows).sort(["anchor_season", "position"])
