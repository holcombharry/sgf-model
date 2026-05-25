"""End-to-end ranking pipeline — wraps every step into one callable.

Kept in the CLI subpackage because it exists specifically to back the
`sgf rank` command. Library users typically call the underlying functions
directly so they can inspect intermediate outputs.
"""

from __future__ import annotations

from dataclasses import asdict

import polars as pl

from sgf_model.curves import fit_age_curves
from sgf_model.data import (
    filter_fantasy_positions,
    load_player_seasons,
    load_weekly_stats,
)
from sgf_model.projections import (
    apply_team_context,
    compute_team_volumes,
    fit_regression_priors,
    get_player_team_mapping,
    project_players,
    project_team_volumes,
)
from sgf_model.scoring import PRESETS as SCORING_PRESETS
from sgf_model.scoring import ScoringConfig, score_projections
from sgf_model.storage import MODEL_VERSION, Snapshot
from sgf_model.valuation import LEAGUE_PRESETS, LeagueConfig, compute_dynasty_value, compute_vorp


def run_pipeline(
    league: LeagueConfig,
    scoring: ScoringConfig,
    as_of_season: int,
    n_future_seasons: int = 5,
    history_weights: tuple[float, ...] = (5.0, 4.0, 3.0),
    discount_rate: float = 0.15,
    use_regression: bool = True,
    team_overrides: dict[str, str] | None = None,
    data_start_season: int = 1999,
    notes: str = "",
) -> Snapshot:
    """Run the full pipeline: data → curves → projections → scoring → VORP → dynasty value.

    Returns a fully populated Snapshot ready to persist.
    """
    ps = load_player_seasons(start=data_start_season, end=as_of_season)
    weekly_all = load_weekly_stats(seasons=list(range(data_start_season, as_of_season + 1)))
    weekly_fantasy = filter_fantasy_positions(weekly_all)

    curves = fit_age_curves(ps)
    priors = fit_regression_priors(weekly_fantasy, as_of_season=as_of_season) if use_regression else None
    proj = project_players(
        ps,
        curves,
        as_of_season=as_of_season,
        n_future_seasons=n_future_seasons,
        history_weights=history_weights,
        regression_priors=priors,
    )

    tv_hist = compute_team_volumes(weekly_all)
    tv_proj = project_team_volumes(
        tv_hist,
        as_of_season=as_of_season,
        n_future_seasons=n_future_seasons,
        history_weights=history_weights,
    )
    team_map = get_player_team_mapping(weekly_all, as_of_season=as_of_season)
    proj = apply_team_context(
        proj,
        tv_hist,
        tv_proj,
        team_map,
        as_of_season=as_of_season,
        team_overrides=team_overrides,
        history_weights=history_weights,
    )

    scored = score_projections(proj, scoring)
    vorp = compute_vorp(scored, league)
    rankings_df = compute_dynasty_value(vorp, as_of_season=as_of_season, discount_rate=discount_rate)

    rankings = rankings_df.with_columns(
        pl.col("dynasty_value").round(1),
        pl.col("total_vorp_undiscounted").round(1),
        pl.col("peak_vorp_season").round(1),
    ).to_dicts()

    return Snapshot(
        as_of_season=as_of_season,
        n_future_seasons=n_future_seasons,
        league_config=asdict(league),
        scoring_config=asdict(scoring),
        history_weights=list(history_weights),
        discount_rate=discount_rate,
        use_regression=use_regression,
        team_overrides=team_overrides or {},
        rankings=rankings,
        notes=notes,
        model_version=MODEL_VERSION,
    )


def resolve_league(name_or_custom: str) -> LeagueConfig:
    if name_or_custom in LEAGUE_PRESETS:
        return LEAGUE_PRESETS[name_or_custom]
    raise ValueError(
        f"Unknown league preset {name_or_custom!r}. "
        f"Available: {sorted(LEAGUE_PRESETS.keys())}"
    )


def resolve_scoring(name_or_custom: str) -> ScoringConfig:
    if name_or_custom in SCORING_PRESETS:
        return SCORING_PRESETS[name_or_custom]
    raise ValueError(
        f"Unknown scoring preset {name_or_custom!r}. "
        f"Available: {sorted(SCORING_PRESETS.keys())}"
    )
