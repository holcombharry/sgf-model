"""Per-player and per-team stat projections."""

from sgf_model.projections.player import project_players
from sgf_model.projections.regression import fit_regression_priors
from sgf_model.projections.rookie import (
    build_rookie_training_data,
    fit_rookie_models,
    make_rookie_class_from_history,
    project_rookies,
)
from sgf_model.projections.team import (
    apply_team_context,
    compute_team_volumes,
    get_player_team_mapping,
    project_team_volumes,
)

__all__ = [
    "apply_team_context",
    "build_rookie_training_data",
    "compute_team_volumes",
    "fit_regression_priors",
    "fit_rookie_models",
    "get_player_team_mapping",
    "make_rookie_class_from_history",
    "project_players",
    "project_rookies",
    "project_team_volumes",
]
