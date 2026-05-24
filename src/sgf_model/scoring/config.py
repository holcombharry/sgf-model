"""Scoring configuration — the layer where league rules become concrete.

This is intentionally separated from the projection layer: stats are projected
once and then scored under any number of rule sets. Adding a new format (e.g.
a custom IDP league, big-yardage bonuses) means adding a `ScoringConfig` and
adjusting the apply function — no re-projection needed.

Only the offensive stats actually projected upstream are scored here. 2-point
conversions, fumbles lost, and yardage bonuses (100/200/300 yd) are common
real-league scoring items but aren't projected at the per-game level yet —
they'd need their own modeling first. The fields exist conceptually but
aren't applied until projections include them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    """Per-stat scoring coefficients. Construct directly or use a preset.

    Coefficients are points per unit of the named stat (e.g. `passing_yards=0.04`
    is 1 point per 25 passing yards). Negative values for penalties (INTs).
    """

    name: str = "custom"

    # Passing
    passing_yards: float = 0.04          # standard: 1 / 25 yd
    passing_tds: float = 4.0
    passing_interceptions: float = -2.0

    # Rushing
    rushing_yards: float = 0.10          # standard: 1 / 10 yd
    rushing_tds: float = 6.0

    # Receiving
    receiving_yards: float = 0.10
    receiving_tds: float = 6.0
    receptions: float = 1.0              # 0 = standard, 0.5 = half PPR, 1.0 = full PPR

    # Position bonus on top of `receptions` — applied only when position == "TE".
    # e.g. TE-premium half: receptions=0.5, te_premium_per_reception=0.5 → TEs get 1.0
    te_premium_per_reception: float = 0.0


PRESETS: dict[str, ScoringConfig] = {
    "standard": ScoringConfig(name="standard", receptions=0.0),
    "half_ppr": ScoringConfig(name="half_ppr", receptions=0.5),
    "ppr": ScoringConfig(name="ppr", receptions=1.0),
    "te_premium_half_ppr": ScoringConfig(
        name="te_premium_half_ppr",
        receptions=0.5,
        te_premium_per_reception=0.5,
    ),
    "te_premium_ppr": ScoringConfig(
        name="te_premium_ppr",
        receptions=1.0,
        te_premium_per_reception=0.5,
    ),
    # 6-point passing TD variant (common in dynasty leagues)
    "ppr_6ptpasstd": ScoringConfig(name="ppr_6ptpasstd", receptions=1.0, passing_tds=6.0),
}
