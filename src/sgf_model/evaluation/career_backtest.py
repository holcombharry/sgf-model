"""Direct career-VORP backtest: trains on career VORP targets, evaluates rank
quality + calibration against realized career VORP over a held-out anchor.

Leakage discipline:
    For each test anchor T (horizon h):
        - Training data: player_seasons.filter(season <= T).
        - Training targets: computed from same filtered data, so only anchors
          A with A + h <= T contribute a non-null target. The model sees no
          actual outcomes from years T+1..T+h.
        - Test target: realized career VORP at anchor T computed from full
          player_seasons (which extends past T by h years).

This is the strictest setup we can run with the data we have. Per-position
quantile models are trained directly on career-VORP target; output is the
career-VORP distribution per player (no Monte Carlo).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from sgf_model.features import (
    CAREER_FEATURE_COLUMNS,
    build_career_feature_matrix,
)
from sgf_model.models import QuantileFPModel
from sgf_model.scoring import ScoringConfig
from sgf_model.simulation.career_target import compute_career_vorp_targets
from sgf_model.valuation.league import LeagueConfig

DEFAULT_TOP_NS: tuple[int, ...] = (10, 30, 60, 100)

DEFAULT_RANKING_SCORES: dict[str, str] = {
    "mean": "Mean across the predicted quantile fan — default, robust to zero-inflated targets",
    "median": "P50 — broken for zero-inflated targets, do not use",
    "risk_adjusted_0.5": "p50 − 0.5 × std — mild risk aversion",
    "risk_adjusted_1.0": "p50 − 1.0 × std — strong risk aversion",
    "p25_floor": "P25 career VORP — pessimistic floor",
    "p_positive": "P(career VORP > 0) — usefulness probability",
}


def master_ranking(
    summary: pl.DataFrame,
    score: str = "mean",
    descending: bool = True,
) -> pl.DataFrame:
    """Sort a career-summary DataFrame (output of `summarize_predictions`) by a
    configurable score column.

    Default is `mean` — NOT median. The career-VORP target is heavily zero-
    inflated (~85% of players score 0 career VORP), which collapses the
    quantile model's P50 to 0 for nearly everyone under symmetric quantile loss.
    Mean across the predicted quantile fan is the correct point estimate.
    See docs/phase7-career-direct.md.

    Returns the summary with an added `rank` column (1-indexed).
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


def project_career_for_backtest(
    player_seasons: pl.DataFrame,
    test_anchor: int,
    horizon: int,
    scoring: ScoringConfig,
    league: LeagueConfig,
    discount_rate: float = 0.15,
    advanced_features: pl.DataFrame | None = None,
    draft_features: pl.DataFrame | None = None,
    rookies: pl.DataFrame | None = None,
    feature_columns: tuple[str, ...] = CAREER_FEATURE_COLUMNS,
    model_params: dict | None = None,
    random_state: int = 42,
) -> pl.DataFrame:
    """Train on data through `test_anchor`, predict career VORP at `test_anchor`."""
    train_ps = player_seasons.filter(pl.col("season") <= test_anchor)
    advanced_train = (
        advanced_features.filter(pl.col("season") <= test_anchor)
        if advanced_features is not None else None
    )
    rookies_train = (
        rookies.filter(pl.col("draft_year") <= test_anchor)
        if rookies is not None else None
    )

    # Training targets are computed strictly from training-window data, so
    # they only exist for anchors A where A + horizon <= test_anchor.
    train_targets = compute_career_vorp_targets(
        train_ps, scoring, league, horizon=horizon, discount_rate=discount_rate,
    )

    fm_train = build_career_feature_matrix(
        train_ps, scoring, train_targets,
        advanced_features=advanced_train,
        draft_features=draft_features,
        rookies=rookies_train,
    )

    # Inference rows for the test anchor: anchor_season = test_anchor.
    fm_inf = build_career_feature_matrix(
        train_ps, scoring, train_targets,
        inference_season=test_anchor + 1,
        advanced_features=advanced_train,
        draft_features=draft_features,
        rookies=rookies,
    ).filter(pl.col("target_fp").is_null())

    model = QuantileFPModel(
        feature_columns=feature_columns,
        params=model_params,
        random_state=random_state,
    ).fit(fm_train)
    return model.predict(fm_inf)


def summarize_predictions(predictions: pl.DataFrame) -> pl.DataFrame:
    """Convert direct-model quantile predictions to the summary schema used by
    `simulation.career.master_ranking`.

    Mean is computed as the unweighted average of the 5 emitted quantile values
    (P10/P25/P50/P75/P90) — a discretized expectation of the predicted
    distribution. P25/P50/P75 are passed through. p_positive is the implied
    probability that the lowest non-trivial quantile (P10) is above 0; reads
    as "the model's intervals say career VORP > 0 with at least 90% confidence".
    """
    p10 = pl.col("proj_fp_lower_80")
    p25 = pl.col("proj_fp_lower_50")
    p50 = pl.col("fantasy_points_season")
    p75 = pl.col("proj_fp_upper_50")
    p90 = pl.col("proj_fp_upper_80")
    return predictions.with_columns(
        mean=(p10 + p25 + p50 + p75 + p90) / 5,
        p10=p10, p25=p25, p50=p50, p75=p75, p90=p90,
        p_positive=(p10 > 0).cast(pl.Float64),
        std=((p90 - p10) / 2.563).abs(),  # rough sigma from 80% interval width
    ).with_columns(
        risk_adjusted_0_5=pl.col("p50") - 0.5 * pl.col("std"),
        risk_adjusted_1_0=pl.col("p50") - 1.0 * pl.col("std"),
    ).select(
        "player_id", "player_name", "position",
        "mean", "std", "p10", "p25", "p50", "p75", "p90",
        pl.col("risk_adjusted_0_5").alias("risk_adjusted_0.5"),
        pl.col("risk_adjusted_1_0").alias("risk_adjusted_1.0"),
        "p_positive",
    )


def evaluate_career_ranking(
    predictions: pl.DataFrame,
    realized: pl.DataFrame,
    baseline_score: pl.DataFrame | None = None,
    top_ns: tuple[int, ...] = DEFAULT_TOP_NS,
) -> dict[str, dict[str, float]]:
    """Compute ranking + calibration metrics for one test anchor.

    Args:
        predictions: output of project_career_for_backtest. Has
            `fantasy_points_season` (P50), `proj_fp_*` quantile interval cols,
            player metadata.
        realized: (player_id, target_career_vorp) for the test anchor.
        baseline_score: optional (player_id, baseline_score) to compute lift
            against. Typical baseline: anchor-year FP.
        top_ns: top-N cuts for hit-rate computation.

    Returns dict with per-score sub-dicts of metrics (model + baseline).
    """
    merged = predictions.join(
        realized.select("player_id", "target_career_vorp"),
        on="player_id", how="inner",
    )
    if baseline_score is not None:
        merged = merged.join(baseline_score, on="player_id", how="left").with_columns(
            baseline_score=pl.col("baseline_score").fill_null(0.0),
        )

    realized_arr = merged["target_career_vorp"].to_numpy()
    n = merged.height

    def rank_metrics(score_arr: np.ndarray) -> dict[str, float]:
        rho = float(spearmanr(score_arr, realized_arr).correlation) if n >= 3 else float("nan")
        out = {"n": n, "spearman": rho}
        for N in top_ns:
            pred_top = set(np.argsort(-score_arr)[:N])
            true_top = set(np.argsort(-realized_arr)[:N])
            out[f"top{N}_hit"] = len(pred_top & true_top) / N if n >= N else float("nan")
        return out

    results = {
        "model_p50": rank_metrics(merged["fantasy_points_season"].to_numpy()),
    }
    if baseline_score is not None:
        results["baseline"] = rank_metrics(merged["baseline_score"].to_numpy())

    # Calibration: are realized VORPs inside the model's intervals at nominal rate?
    p10 = merged["proj_fp_lower_80"].to_numpy()
    p25 = merged["proj_fp_lower_50"].to_numpy()
    p75 = merged["proj_fp_upper_50"].to_numpy()
    p90 = merged["proj_fp_upper_80"].to_numpy()
    cov = {
        "cov_50": float(((realized_arr >= p25) & (realized_arr <= p75)).mean()),
        "cov_80": float(((realized_arr >= p10) & (realized_arr <= p90)).mean()),
    }
    results["calibration"] = cov

    return results


def run_career_backtest(
    player_seasons: pl.DataFrame,
    test_anchors: list[int],
    scoring: ScoringConfig,
    league: LeagueConfig,
    horizon: int = 5,
    discount_rate: float = 0.15,
    advanced_features: pl.DataFrame | None = None,
    draft_features: pl.DataFrame | None = None,
    rookies: pl.DataFrame | None = None,
    feature_columns: tuple[str, ...] = CAREER_FEATURE_COLUMNS,
) -> dict[int, dict[str, dict]]:
    """Run the career-VORP backtest across multiple anchors.

    Returns: {test_anchor: {score_name: metrics_dict, "calibration": {...}}}
    """
    # Realized targets computed on full data — used for test-side scoring at
    # every anchor. Training-side targets are recomputed per anchor inside
    # project_career_for_backtest with the training-window restriction.
    realized_full = compute_career_vorp_targets(
        player_seasons, scoring, league, horizon=horizon, discount_rate=discount_rate,
    )
    # Baseline score: anchor-year FP per player (naive 'rank by last year').
    # Compute it once from the scored history (target_career_vorp is keyed by
    # anchor_season; we want fp at season == anchor_season).
    from sgf_model.simulation.career_target import _STAT_COLS_RENAME
    from sgf_model.scoring import score_projections
    renamed = player_seasons.rename(
        {k: v for k, v in _STAT_COLS_RENAME.items() if k in player_seasons.columns}
    )
    anchor_fp = score_projections(renamed, scoring).select(
        "player_id", "season", pl.col("fantasy_points_season").alias("anchor_fp"),
    )

    results: dict[int, dict[str, dict]] = {}
    for test_anchor in test_anchors:
        preds = project_career_for_backtest(
            player_seasons, test_anchor, horizon, scoring, league,
            discount_rate=discount_rate,
            advanced_features=advanced_features,
            draft_features=draft_features,
            rookies=rookies,
            feature_columns=feature_columns,
        )
        anchor_realized = realized_full.filter(pl.col("anchor_season") == test_anchor)
        anchor_baseline = anchor_fp.filter(pl.col("season") == test_anchor).select(
            "player_id", pl.col("anchor_fp").alias("baseline_score"),
        )
        results[test_anchor] = evaluate_career_ranking(
            preds, anchor_realized, baseline_score=anchor_baseline,
        )
    return results
