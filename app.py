"""Streamlit frontend for sgf-model.

Launch:
    uv run streamlit run app.py

Storage backend is picked from the SGF_STORAGE env var (same as the CLI).
For MongoDB:
    SGF_STORAGE=mongodb://localhost:27017 uv run streamlit run app.py
"""

from __future__ import annotations

import json

import polars as pl
import streamlit as st

from sgf_model.cli.pipeline import resolve_league, resolve_scoring, run_pipeline
from sgf_model.scoring import PRESETS as SCORING_PRESETS
from sgf_model.storage import Snapshot, default_storage
from sgf_model.valuation import LEAGUE_PRESETS

st.set_page_config(page_title="sgf-model", layout="wide")

POSITIONS = ("QB", "RB", "WR", "TE")
DISPLAY_COLS = [
    "rank",
    "player_name",
    "position",
    "current_age",
    "dynasty_value",
    "total_vorp_undiscounted",
    "years_projected",
    "peak_vorp_year_offset",
    "peak_vorp_season",
]


@st.cache_resource
def get_storage():
    """Cached per session — storage clients (esp. Mongo) shouldn't reconnect on every interaction."""
    return default_storage()


@st.cache_data(show_spinner="Running pipeline...")
def cached_run(
    league_name: str,
    scoring_name: str,
    as_of_season: int,
    n_future_seasons: int,
    discount_rate: float,
    use_regression: bool,
    start_season: int,
    overrides_json: str,
    notes: str,
) -> Snapshot:
    """Streamlit-friendly wrapper. All args are simple, hashable types so the cache works."""
    overrides = json.loads(overrides_json) if overrides_json.strip() else None
    return run_pipeline(
        league=resolve_league(league_name),
        scoring=resolve_scoring(scoring_name),
        as_of_season=as_of_season,
        n_future_seasons=n_future_seasons,
        discount_rate=discount_rate,
        use_regression=use_regression,
        team_overrides=overrides,
        data_start_season=start_season,
        notes=notes,
    )


def render_rankings(snap: Snapshot, top_n: int, positions: list[str]) -> None:
    """Render header + filterable rankings table for a snapshot."""
    st.subheader(
        f"{snap.league_config.get('name', 'custom')} | "
        f"{snap.scoring_config.get('name', 'custom')} | "
        f"as_of {snap.as_of_season}"
    )
    meta_bits = [
        f"discount={snap.discount_rate}",
        f"n_future={snap.n_future_seasons}",
        f"regression={'on' if snap.use_regression else 'off'}",
        f"overrides={len(snap.team_overrides)}",
        f"created={snap.created_at[:19]}",
    ]
    st.caption(" | ".join(meta_bits) + (f" | notes: {snap.notes}" if snap.notes else ""))

    df = pl.from_dicts(snap.rankings).filter(pl.col("position").is_in(positions))
    # Re-rank after filtering so ranks reflect the visible set.
    df = df.with_columns(rank=pl.int_range(1, df.height + 1)).head(top_n)
    # Pick the available subset of DISPLAY_COLS (defensive against schema changes).
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    st.dataframe(df.select(cols), use_container_width=True, hide_index=True, height=600)


# Top-level mode toggle.
st.title("Dynasty Rankings")
mode = st.sidebar.radio("Mode", ["Run new ranking", "Load saved snapshot"], horizontal=False)

storage = get_storage()


if mode == "Run new ranking":
    league_options = sorted(LEAGUE_PRESETS.keys())
    scoring_options = sorted(SCORING_PRESETS.keys())

    with st.sidebar:
        st.header("Configuration")
        league_name = st.selectbox("League", league_options, index=league_options.index("12_team_1qb"))
        scoring_name = st.selectbox("Scoring", scoring_options, index=scoring_options.index("ppr"))
        as_of_season = st.number_input("As-of season", min_value=2000, max_value=2030, value=2024, step=1)
        n_future_seasons = st.slider("Years projected", 1, 7, 5)
        discount_rate = st.slider("Discount rate", 0.0, 0.5, 0.15, 0.01)
        use_regression = st.toggle("Empirical-Bayes regression", value=True)
        start_season = st.number_input(
            "Data start season",
            min_value=1999,
            max_value=2024,
            value=2010,
            step=1,
            help="Narrow to skip flaky early files; 1999 = full history.",
        )
        overrides_json = st.text_area(
            "Team overrides (JSON)",
            value="",
            placeholder='{"00-0036355": "BAL"}',
            help="Map player_id → new team for FA/trades.",
            height=80,
        )
        notes = st.text_input("Notes", value="", placeholder="optional description")
        top_n = st.slider("Display top N", 10, 250, 50)
        run_button = st.button("Run", type="primary")

    if run_button:
        try:
            snap = cached_run(
                league_name=league_name,
                scoring_name=scoring_name,
                as_of_season=as_of_season,
                n_future_seasons=n_future_seasons,
                discount_rate=discount_rate,
                use_regression=use_regression,
                start_season=start_season,
                overrides_json=overrides_json,
                notes=notes,
            )
            st.session_state["last_snap"] = snap
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

    if "last_snap" in st.session_state:
        snap = st.session_state["last_snap"]
        positions = st.multiselect("Position filter", POSITIONS, default=list(POSITIONS))
        render_rankings(snap, top_n=top_n, positions=positions)

        save_col, sid_col = st.columns([1, 4])
        with save_col:
            if st.button("Save to storage"):
                sid = storage.save(snap)
                st.success(f"Saved snapshot_id={sid}")
        with sid_col:
            st.caption(f"snapshot_id (preview): {snap.snapshot_id}")
    else:
        st.info("Configure on the left and click **Run**.")


else:  # Load saved snapshot
    snapshots = storage.list(limit=100)

    if not snapshots:
        st.info("No saved snapshots yet. Use **Run new ranking** to create one.")
    else:
        with st.sidebar:
            st.header("Saved snapshots")
            options = {
                f"{s['created_at'][:19]}  |  {s['league_name']} / {s['scoring_name']}  "
                f"|  as_of {s['as_of_season']}"
                + (f"  |  {s['notes']}" if s["notes"] else ""): s["snapshot_id"]
                for s in snapshots
            }
            selected_label = st.selectbox(
                f"Select ({len(snapshots)} available)", list(options.keys())
            )
            top_n = st.slider("Display top N", 10, 250, 50)
            delete_col, _ = st.columns([1, 1])
            with delete_col:
                if st.button("Delete this snapshot", type="secondary"):
                    sid = options[selected_label]
                    storage.delete(sid)
                    st.cache_resource.clear()  # refresh storage view
                    st.rerun()

        snap = storage.get(options[selected_label])
        positions = st.multiselect("Position filter", POSITIONS, default=list(POSITIONS))
        render_rankings(snap, top_n=top_n, positions=positions)
