"""League configuration: roster shape + replacement-level depth.

Replacement level is the single most important league-format variable for
dynasty value: it's why a QB2 is irrelevant in a 1-QB league and a top-12
asset in superflex, and why a TE3 matters in TE-premium 12-team formats but
not in shallow flex leagues.

We model it from roster composition:
    starters[pos] = n_teams * (slots[pos] + flex_share[pos] * flex_slots
                                          + sflex_share[pos] * sflex_slots)
    replacement_rank[pos] = starters[pos] + n_teams * bench_buffer

`flex_share` and `superflex_share` are the empirical fraction of those slots
filled by each position; defaults are tuned to common league patterns (most
flex slots go to RB/WR, most superflex slots go to QB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueConfig:
    """League roster + replacement-level definition.

    Defaults are 12-team standard PPR with 1QB/2RB/3WR/1TE/1FLEX, no superflex.
    """

    name: str = "custom"
    n_teams: int = 12

    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 3
    te_slots: int = 1
    flex_slots: int = 1
    superflex_slots: int = 0

    # Empirical share of FLEX (RB/WR/TE only) slots filled by each position.
    # Tuned to roughly match observed start% in mainstream leagues.
    flex_share_rb: float = 0.35
    flex_share_wr: float = 0.55
    flex_share_te: float = 0.10

    # Empirical share of SUPERFLEX (QB/RB/WR/TE) slots filled by each.
    # In practice, superflex is overwhelmingly a 2nd QB slot.
    superflex_share_qb: float = 0.85
    superflex_share_rb: float = 0.05
    superflex_share_wr: float = 0.08
    superflex_share_te: float = 0.02

    # Bench depth beyond the last starter that still counts as "rostered."
    # 1.0 = one player per team beyond starters, i.e. replacement is the
    # first guy on waivers in a deep-ish league.
    bench_buffer: float = 1.0

    def starters_per_position(self) -> dict[str, float]:
        return {
            "QB": self.n_teams
            * (self.qb_slots + self.superflex_slots * self.superflex_share_qb),
            "RB": self.n_teams
            * (
                self.rb_slots
                + self.flex_slots * self.flex_share_rb
                + self.superflex_slots * self.superflex_share_rb
            ),
            "WR": self.n_teams
            * (
                self.wr_slots
                + self.flex_slots * self.flex_share_wr
                + self.superflex_slots * self.superflex_share_wr
            ),
            "TE": self.n_teams
            * (
                self.te_slots
                + self.flex_slots * self.flex_share_te
                + self.superflex_slots * self.superflex_share_te
            ),
        }

    def replacement_rank(self) -> dict[str, int]:
        """The rank at which a player is considered replacement-level per position."""
        return {
            pos: math.ceil(starters + self.n_teams * self.bench_buffer)
            for pos, starters in self.starters_per_position().items()
        }


LEAGUE_PRESETS: dict[str, LeagueConfig] = {
    "12_team_1qb": LeagueConfig(name="12_team_1qb"),
    "12_team_superflex": LeagueConfig(
        name="12_team_superflex", superflex_slots=1,
    ),
    "10_team_1qb": LeagueConfig(name="10_team_1qb", n_teams=10),
    "10_team_superflex": LeagueConfig(
        name="10_team_superflex", n_teams=10, superflex_slots=1,
    ),
    "12_team_2qb": LeagueConfig(name="12_team_2qb", qb_slots=2),
}
