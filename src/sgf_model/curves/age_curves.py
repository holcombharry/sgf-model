"""Position-specific age curves for fantasy stats.

Two approaches are exposed:

- `fit_age_curves_raw`: weighted per-game means by age. Easy to explain, but
  severely biased by survivorship at the tails — only elite players appear at
  age 21 or 33, so observed production overstates the true age effect. Use for
  diagnostics, not projection.

- `fit_age_curves` (default): the within-player **delta method**. For each
  player, pairs consecutive seasons and measures the age-over-age change in
  per-game production, holding player identity constant. Aging factors are
  then chained from an anchor age into a smooth multiplier curve. This is
  the canonical sabermetric approach (Tango et al.) and controls for the
  selection effect that breaks the raw method.

The output curves are *multipliers* relative to a chosen anchor age: by
construction the curve is 1.0 at `anchor_age`, and a player whose per-game
production at age A is X is expected to produce X * (curve[A+n] / curve[A])
at age A+n, holding everything else equal.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

# Stats worth age-curving per position. Volume + efficiency inputs that downstream
# projections will need. Omitted: defensive/special-teams columns (irrelevant for
# fantasy offense) and pre-baked fantasy_points (we score from raw stats).
DEFAULT_STATS_BY_POSITION: dict[str, list[str]] = {
    "QB": [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],
    "RB": [
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
    # WR rushing stats omitted — too sparse for the delta method (most WRs have
    # 0 rushes most weeks, so per-game ratios blow up). Modeled at a per-player
    # level downstream if needed.
    "WR": [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
    "TE": [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
}


def fit_age_curves_raw(
    player_seasons: pl.DataFrame,
    stats_by_position: dict[str, list[str]] | None = None,
    min_games: int = 4,
    min_n_per_age: int = 20,
    age_range: tuple[int, int] = (21, 37),
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> pl.DataFrame:
    """Survivorship-biased reference curves: weighted per-game mean by age.

    Returns long-format DataFrame:
        position | stat | age | value_per_game | smoothed | n_player_seasons

    Not recommended for projection — provided for diagnostics and to make the
    bias in this approach visible by comparison with `fit_age_curves`.
    """
    stats_by_position = stats_by_position or DEFAULT_STATS_BY_POSITION

    df = player_seasons.filter(
        (pl.col("games_played") >= min_games)
        & (pl.col("age").is_between(age_range[0], age_range[1]))
    )

    rows: list[pl.DataFrame] = []
    for position, stats in stats_by_position.items():
        pos_df = df.filter(pl.col("position") == position)
        if pos_df.is_empty():
            continue

        for stat in stats:
            per_age = (
                pos_df.with_columns(per_game=pl.col(stat) / pl.col("games_played"))
                .group_by("age")
                .agg(
                    value_per_game=(pl.col("per_game") * pl.col("games_played")).sum()
                    / pl.col("games_played").sum(),
                    n_player_seasons=pl.len(),
                )
                .filter(pl.col("n_player_seasons") >= min_n_per_age)
                .sort("age")
            )

            if per_age.height < savgol_window:
                smoothed = per_age["value_per_game"].to_numpy()
            else:
                smoothed = savgol_filter(
                    per_age["value_per_game"].to_numpy(),
                    window_length=savgol_window,
                    polyorder=savgol_polyorder,
                )

            rows.append(
                per_age.with_columns(
                    position=pl.lit(position),
                    stat=pl.lit(stat),
                    smoothed=pl.Series(smoothed),
                ).select(
                    ["position", "stat", "age", "value_per_game", "smoothed", "n_player_seasons"]
                )
            )

    return pl.concat(rows) if rows else pl.DataFrame()


def fit_age_curves(
    player_seasons: pl.DataFrame,
    stats_by_position: dict[str, list[str]] | None = None,
    min_games: int = 4,
    min_pairs_per_age: int = 15,
    age_range: tuple[int, int] = (21, 37),
    anchor_age: int = 27,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> pl.DataFrame:
    """Delta-method age curves: multiplier vs `anchor_age`, controlling for player identity.

    For each (position, stat):
      1. Pair each player's consecutive seasons (age a -> age a+1).
      2. Aging factor at age a = weighted_sum(per_game at a+1) / weighted_sum(per_game at a),
         with weight = min(games_a, games_{a+1}) — limits the influence of small-sample seasons.
      3. Chain the factors out from `anchor_age` (curve[anchor_age]=1) to produce a
         multiplier curve over the full age range.
      4. Smooth across age with Savitzky-Golay to remove noise.

    Returns long-format DataFrame:
        position | stat | age | multiplier | smoothed | n_pairs

    `smoothed` is the column projections should consume.
    """
    stats_by_position = stats_by_position or DEFAULT_STATS_BY_POSITION

    base = player_seasons.filter(
        (pl.col("games_played") >= min_games)
        & (pl.col("age").is_between(age_range[0], age_range[1]))
    )

    rows: list[pl.DataFrame] = []
    for position, stats in stats_by_position.items():
        pos_df = base.filter(pl.col("position") == position).sort(["player_id", "age"])
        if pos_df.is_empty():
            continue

        for stat in stats:
            pairs = pos_df.with_columns(
                per_game=pl.col(stat) / pl.col("games_played"),
            ).with_columns(
                per_game_next=pl.col("per_game").shift(-1).over("player_id"),
                games_next=pl.col("games_played").shift(-1).over("player_id"),
                age_next=pl.col("age").shift(-1).over("player_id"),
            )
            pairs = pairs.filter(pl.col("age_next") == pl.col("age") + 1)
            if pairs.is_empty():
                continue

            factors = (
                pairs.with_columns(
                    weight=pl.min_horizontal("games_played", "games_next"),
                )
                .group_by("age")
                .agg(
                    num=(pl.col("weight") * pl.col("per_game_next")).sum(),
                    den=(pl.col("weight") * pl.col("per_game")).sum(),
                    n_pairs=pl.len(),
                )
                .filter((pl.col("n_pairs") >= min_pairs_per_age) & (pl.col("den") > 0))
                .with_columns(factor=pl.col("num") / pl.col("den"))
                .sort("age")
            )

            if factors.is_empty():
                continue

            # Chain factors out from anchor_age. `factor[a]` is the multiplier going
            # from age a to age a+1, so curve[a+1] = curve[a] * factor[a] and
            # curve[a] = curve[a+1] / factor[a] when walking backwards.
            factor_by_age = dict(zip(factors["age"].to_list(), factors["factor"].to_list()))
            covered_ages = sorted(
                set(factors["age"].to_list()) | {a + 1 for a in factors["age"].to_list()}
            )
            if anchor_age not in covered_ages:
                continue

            curve: dict[int, float] = {anchor_age: 1.0}
            for a in range(anchor_age, max(covered_ages)):
                if a in factor_by_age:
                    curve[a + 1] = curve[a] * factor_by_age[a]
                else:
                    break
            for a in range(anchor_age, min(covered_ages), -1):
                if (a - 1) in factor_by_age:
                    curve[a - 1] = curve[a] / factor_by_age[a - 1]
                else:
                    break

            curve_df = pl.DataFrame(
                {
                    "age": sorted(curve.keys()),
                    "multiplier": [curve[a] for a in sorted(curve.keys())],
                }
            )
            # Attach n_pairs as the count of pairs *ending* at this age (a-1 -> a).
            curve_df = curve_df.join(
                factors.select(
                    (pl.col("age") + 1).alias("age"),
                    pl.col("n_pairs"),
                ),
                on="age",
                how="left",
            ).with_columns(pl.col("n_pairs").fill_null(0))

            # Drop any non-finite multipliers (can happen if a factor produced 0 or
            # inf earlier in the chain) before smoothing.
            curve_df = curve_df.filter(
                pl.col("multiplier").is_finite() & (pl.col("multiplier") > 0)
            )
            if curve_df.is_empty():
                continue

            if curve_df.height < savgol_window:
                smoothed = curve_df["multiplier"].to_numpy()
            else:
                smoothed = savgol_filter(
                    curve_df["multiplier"].to_numpy(),
                    window_length=savgol_window,
                    polyorder=savgol_polyorder,
                )

            rows.append(
                curve_df.with_columns(
                    position=pl.lit(position),
                    stat=pl.lit(stat),
                    smoothed=pl.Series(smoothed),
                ).select(["position", "stat", "age", "multiplier", "smoothed", "n_pairs"])
            )

    return pl.concat(rows) if rows else pl.DataFrame()


def curve_value(
    curves: pl.DataFrame,
    position: str,
    stat: str,
    age: float,
    extrapolation_tail: int = 3,
) -> float:
    """Linearly interpolate the smoothed curve to a fractional age.

    Within the fitted age range, performs linear interpolation between adjacent
    integer ages. Outside the range, extrapolates linearly using the slope of
    the last `extrapolation_tail` fitted points at the appropriate end, floored
    at 0 (production can't be negative). Returns NaN if no curve exists for
    (position, stat).

    Works on the output of either `fit_age_curves` (multipliers) or
    `fit_age_curves_raw` (per-game values).
    """
    sub = curves.filter((pl.col("position") == position) & (pl.col("stat") == stat)).sort("age")
    if sub.is_empty():
        return float("nan")
    ages = sub["age"].to_numpy().astype(float)
    vals = sub["smoothed"].to_numpy().astype(float)

    if ages.min() <= age <= ages.max():
        return float(np.interp(age, ages, vals))

    n_tail = min(extrapolation_tail, len(ages))
    if age > ages.max():
        slope = np.polyfit(ages[-n_tail:], vals[-n_tail:], 1)[0]
        return float(max(0.0, vals[-1] + slope * (age - ages[-1])))
    slope = np.polyfit(ages[:n_tail], vals[:n_tail], 1)[0]
    return float(max(0.0, vals[0] - slope * (ages[0] - age)))


def age_multiplier(
    curves: pl.DataFrame,
    position: str,
    stat: str,
    from_age: float,
    to_age: float,
) -> float:
    """Expected production multiplier going from `from_age` to `to_age`.

    `curve(to_age) / curve(from_age)`. The natural input to projection: given
    a player's current per-game stat at `from_age`, multiply by this to get
    the expected per-game stat at `to_age` (holding everything else equal).
    Returns NaN if `from_age` value is zero or curve is missing.
    """
    base = curve_value(curves, position, stat, from_age)
    target = curve_value(curves, position, stat, to_age)
    if base == 0 or np.isnan(base):
        return float("nan")
    return target / base
