"""Career Monte Carlo simulator and master ranking — the Phase 3 product.

Takes multi-year quantile FP predictions and produces:
    - sampled career FP trajectories (rank-coupled across years)
    - per-sample career VORP (using sample-specific position replacement levels)
    - master ranking with configurable risk-adjusted score
"""

from sgf_model.simulation.career import (
    DEFAULT_N_SIMULATIONS,
    DEFAULT_RANKING_SCORES,
    sample_career_fps,
    sample_career_vorps,
    summarize_career,
    master_ranking,
)

__all__ = [
    "DEFAULT_N_SIMULATIONS",
    "DEFAULT_RANKING_SCORES",
    "sample_career_fps",
    "sample_career_vorps",
    "summarize_career",
    "master_ranking",
]
