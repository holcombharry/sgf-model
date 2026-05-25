"""sgf — Dynasty fantasy football valuation CLI.

Three top-level commands:

    sgf rank            Run the full pipeline and (optionally) save a snapshot.
    sgf snapshots list  Show recent snapshots.
    sgf snapshots show  Display a saved snapshot's metadata and top rankings.

Storage backend is chosen by the SGF_STORAGE env var (defaults to filesystem
under ~/.sgf-model/snapshots). See `sgf_model.storage` for backend details.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from sgf_model.cli.pipeline import resolve_league, resolve_scoring, run_pipeline
from sgf_model.scoring import PRESETS as SCORING_PRESETS
from sgf_model.storage import default_storage
from sgf_model.valuation import LEAGUE_PRESETS

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Dynasty fantasy football valuation.",
)
snapshots_app = typer.Typer(no_args_is_help=True, help="Browse saved snapshots.")
app.add_typer(snapshots_app, name="snapshots")

console = Console()


@app.command()
def rank(
    league: str = typer.Option(
        "12_team_1qb",
        "--league",
        "-l",
        help=f"League preset. Options: {sorted(LEAGUE_PRESETS.keys())}",
    ),
    scoring: str = typer.Option(
        "ppr",
        "--scoring",
        "-s",
        help=f"Scoring preset. Options: {sorted(SCORING_PRESETS.keys())}",
    ),
    as_of: int = typer.Option(2024, "--as-of", help="Most recent completed season."),
    start_season: int = typer.Option(
        1999, "--start-season", help="Earliest season to load. Narrow to skip flaky early files."
    ),
    n_future: int = typer.Option(5, "--n-future", help="Years of dynasty horizon."),
    discount: float = typer.Option(0.15, "--discount", help="Annual discount rate."),
    no_regression: bool = typer.Option(
        False, "--no-regression", help="Disable empirical-Bayes regression to mean."
    ),
    top_n: int = typer.Option(25, "--top", "-n", help="How many rows to display."),
    save: bool = typer.Option(True, "--save/--no-save", help="Save snapshot."),
    notes: str = typer.Option("", "--notes", help="Free-text notes attached to the snapshot."),
    overrides_json: str = typer.Option(
        "", "--overrides", help='JSON `{player_id: new_team}` for FA/trades.'
    ),
) -> None:
    """Run the full ranking pipeline and (by default) persist the result."""
    league_cfg = resolve_league(league)
    scoring_cfg = resolve_scoring(scoring)
    overrides = json.loads(overrides_json) if overrides_json else None

    console.print(
        f"[dim]Running pipeline:[/dim] "
        f"[bold]{league}[/bold] | [bold]{scoring}[/bold] | "
        f"as_of={as_of} | n_future={n_future} | discount={discount} | "
        f"regression={'off' if no_regression else 'on'}"
    )
    snapshot = run_pipeline(
        league=league_cfg,
        scoring=scoring_cfg,
        as_of_season=as_of,
        n_future_seasons=n_future,
        discount_rate=discount,
        use_regression=not no_regression,
        team_overrides=overrides,
        notes=notes,
        data_start_season=start_season,
    )

    table = Table(title=f"Top {top_n} — {league} / {scoring} / as_of {as_of}")
    for col in ["rank", "player", "pos", "age", "dynasty_value", "peak_yr_offset"]:
        table.add_column(col, justify="right" if col != "player" else "left")
    for i, row in enumerate(snapshot.rankings[:top_n], start=1):
        table.add_row(
            str(i),
            row["player_name"],
            row["position"],
            str(row["current_age"]),
            f"{row['dynasty_value']:.1f}",
            str(row["peak_vorp_year_offset"]),
        )
    console.print(table)

    if save:
        storage = default_storage()
        sid = storage.save(snapshot)
        console.print(f"[green]saved[/green] snapshot_id=[bold]{sid}[/bold]")
    else:
        console.print("[yellow]not saved[/yellow] (--no-save)")


@snapshots_app.command("list")
def list_snapshots(
    limit: int = typer.Option(20, "--limit", "-n", help="Max snapshots to show."),
) -> None:
    """Show recent saved snapshots."""
    storage = default_storage()
    rows = storage.list(limit=limit)
    if not rows:
        console.print("[dim]no snapshots[/dim]")
        return
    table = Table(title=f"{len(rows)} most recent snapshots")
    for col in ["snapshot_id", "created_at", "as_of", "league", "scoring", "n", "notes"]:
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["snapshot_id"][:8] + "…",
            r["created_at"][:19],
            str(r["as_of_season"]),
            r["league_name"],
            r["scoring_name"],
            str(r["n_players"]),
            (r["notes"] or "")[:40],
        )
    console.print(table)


@snapshots_app.command("show")
def show_snapshot(
    snapshot_id: str = typer.Argument(..., help="Full or unique-prefix snapshot id."),
    top_n: int = typer.Option(25, "--top", "-n", help="How many rankings to display."),
) -> None:
    """Display a snapshot's configs and top rankings."""
    storage = default_storage()
    snap = _resolve_id(storage, snapshot_id)

    console.print(
        f"[bold]Snapshot[/bold] {snap.snapshot_id}\n"
        f"  created_at: {snap.created_at}\n"
        f"  as_of_season: {snap.as_of_season}, n_future: {snap.n_future_seasons}\n"
        f"  league: {snap.league_config.get('name', 'custom')}\n"
        f"  scoring: {snap.scoring_config.get('name', 'custom')}\n"
        f"  discount: {snap.discount_rate}, regression: {snap.use_regression}\n"
        f"  team_overrides: {len(snap.team_overrides)} entries\n"
        f"  notes: {snap.notes or '(none)'}\n"
        f"  model_version: {snap.model_version}\n"
    )
    table = Table(title=f"Top {top_n} rankings")
    for col in ["rank", "player", "pos", "age", "dynasty_value", "peak_yr_offset"]:
        table.add_column(col, justify="right" if col != "player" else "left")
    for i, row in enumerate(snap.rankings[:top_n], start=1):
        table.add_row(
            str(i),
            row["player_name"],
            row["position"],
            str(row["current_age"]),
            f"{row['dynasty_value']:.1f}",
            str(row["peak_vorp_year_offset"]),
        )
    console.print(table)


def _resolve_id(storage, prefix_or_full: str):
    """Allow short IDs (any unique prefix) without requiring the full UUID."""
    try:
        return storage.get(prefix_or_full)
    except KeyError:
        pass
    # Try prefix match against the recent listing.
    candidates = [r for r in storage.list(limit=200) if r["snapshot_id"].startswith(prefix_or_full)]
    if not candidates:
        raise typer.BadParameter(f"no snapshot matches id prefix {prefix_or_full!r}")
    if len(candidates) > 1:
        raise typer.BadParameter(
            f"ambiguous id prefix {prefix_or_full!r} — {len(candidates)} matches"
        )
    return storage.get(candidates[0]["snapshot_id"])


if __name__ == "__main__":
    app()
