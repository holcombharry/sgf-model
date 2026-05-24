"""Pluggable league scoring: convert stat-space projections to fantasy points."""

from sgf_model.scoring.config import PRESETS, ScoringConfig
from sgf_model.scoring.score import score_projections

__all__ = ["PRESETS", "ScoringConfig", "score_projections"]
