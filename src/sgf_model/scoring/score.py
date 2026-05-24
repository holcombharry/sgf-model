"""Apply a ScoringConfig to a projection DataFrame."""

from __future__ import annotations

import polars as pl

from sgf_model.scoring.config import ScoringConfig

# Mapping from scoring config attribute -> projected stat column root.
# (config_attr, stat_root) — column lookup is "{stat_root}_{per_game|season}".
_STAT_TO_COEF: tuple[tuple[str, str], ...] = (
    ("passing_yards", "passing_yards"),
    ("passing_tds", "passing_tds"),
    ("passing_interceptions", "passing_interceptions"),
    ("rushing_yards", "rushing_yards"),
    ("rushing_tds", "rushing_tds"),
    ("receiving_yards", "receiving_yards"),
    ("receiving_tds", "receiving_tds"),
    ("receptions", "receptions"),
)


def _points_expr(scoring: ScoringConfig, columns: list[str], suffix: str) -> pl.Expr:
    """Build the fantasy-points expression for one suffix ('per_game' or 'season').

    Sums each available `<stat>_<suffix>` column times its scoring coefficient,
    plus the TE-premium bonus on receptions when applicable. Missing columns
    (because a stat wasn't projected for some positions) are simply skipped —
    polars fills the gap with NaN at the row level, and we fill those with 0
    via `coalesce` so a missing column doesn't poison the sum for other rows.
    """
    parts: list[pl.Expr] = []
    for coef_attr, stat_root in _STAT_TO_COEF:
        coef = getattr(scoring, coef_attr)
        col = f"{stat_root}_{suffix}"
        if coef != 0 and col in columns:
            parts.append(pl.col(col).fill_null(0.0) * coef)

    reception_col = f"receptions_{suffix}"
    if scoring.te_premium_per_reception != 0 and reception_col in columns:
        parts.append(
            pl.when(pl.col("position") == "TE")
            .then(pl.col(reception_col).fill_null(0.0) * scoring.te_premium_per_reception)
            .otherwise(0.0)
        )

    if not parts:
        return pl.lit(0.0)
    expr = parts[0]
    for p in parts[1:]:
        expr = expr + p
    return expr


def score_projections(
    projections: pl.DataFrame,
    scoring: ScoringConfig,
) -> pl.DataFrame:
    """Add fantasy_points_per_game + fantasy_points_season columns.

    Applies `scoring` to each row of `projections`. Position-specific rules
    (TE premium) are handled inside the expression so the result is consistent
    across positions in one pass.
    """
    cols = projections.columns
    return projections.with_columns(
        fantasy_points_per_game=_points_expr(scoring, cols, "per_game"),
        fantasy_points_season=_points_expr(scoring, cols, "season"),
    )
