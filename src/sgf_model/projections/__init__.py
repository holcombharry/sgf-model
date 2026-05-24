"""Per-player and per-team stat projections."""

from sgf_model.projections.player import project_players
from sgf_model.projections.regression import fit_regression_priors

__all__ = ["fit_regression_priors", "project_players"]
