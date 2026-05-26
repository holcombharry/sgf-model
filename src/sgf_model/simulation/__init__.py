"""Career-VORP target computation from realized history.

Module name is historical — the Monte Carlo simulator was decommissioned in
Phase 7 once direct career-VORP training matched its rank quality with a
simpler pipeline. What remains is the realized-target helper that the
direct-training feature builder consumes.
"""

from sgf_model.simulation.career_target import compute_career_vorp_targets

__all__ = ["compute_career_vorp_targets"]
