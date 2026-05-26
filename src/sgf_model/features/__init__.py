"""Feature engineering for the top-down FP model.

Builds the (player_id, season, features..., target_fp) matrix consumed by the
quantile FP model. Features are computed from data strictly prior to `season`
so the training set has no leakage from the target.
"""

from sgf_model.features.advanced import (
    build_advanced_features,
    compute_draft_features,
    compute_ngs_passing_features,
    compute_ngs_receiving_features,
    compute_ngs_rushing_features,
    compute_route_features,
    compute_rookies,
    compute_snap_features,
)
from sgf_model.features.builder import (
    PHASE1_FEATURE_COLUMNS,
    PHASE2_FEATURE_COLUMNS,
    PHASE3_FEATURE_COLUMNS,
    PHASE5_FEATURE_COLUMNS,
    PHASE5_ROUTES_COLUMNS,
    build_feature_matrix,
)

__all__ = [
    "build_advanced_features",
    "build_feature_matrix",
    "compute_draft_features",
    "compute_ngs_passing_features",
    "compute_ngs_receiving_features",
    "compute_ngs_rushing_features",
    "compute_route_features",
    "compute_rookies",
    "compute_snap_features",
    "PHASE1_FEATURE_COLUMNS",
    "PHASE2_FEATURE_COLUMNS",
    "PHASE3_FEATURE_COLUMNS",
    "PHASE5_FEATURE_COLUMNS",
    "PHASE5_ROUTES_COLUMNS",
]
