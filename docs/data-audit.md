# Data Audit — `nflreadpy` Coverage for the Talent / Situation Model

Verified 2026-05-25 against `nflreadpy` 0.1.5 with 2023 season pulls.

## What we have for free

| Source | Function | Coverage | Key columns we'll use |
|---|---|---|---|
| Snap counts | `load_snap_counts()` | 2012+ | `offense_snaps`, `offense_pct`, `defense_snaps`, `st_snaps` |
| NGS receiving | `load_nextgen_stats(stat_type="receiving")` | 2016+ | `avg_separation`, `avg_cushion`, `avg_intended_air_yards` (aDOT), `percent_share_of_intended_air_yards`, `avg_yac`, `avg_yac_above_expectation`, `catch_percentage` |
| NGS rushing | `load_nextgen_stats(stat_type="rushing")` | 2016+ | `efficiency`, `rush_yards_over_expected`, `rush_yards_over_expected_per_att`, `rush_pct_over_expected`, `percent_attempts_gte_eight_defenders`, `avg_time_to_los` |
| NGS passing | `load_nextgen_stats(stat_type="passing")` | 2016+ | `avg_time_to_throw`, `aggressiveness`, `completion_percentage_above_expectation` (CPOE), `avg_air_yards_to_sticks`, `avg_air_distance` |
| PFR adv passing | `load_pfr_advstats(stat_type="pass")` | 2018+ | `times_pressured`, `pressure_pct`, `times_blitzed`, `times_hurried`, `pocket_time`, `drops`, `bad_throws`, `on_tgt_pct`, `pass_yards_after_catch` |
| PFR adv rushing | `load_pfr_advstats(stat_type="rush")` | 2018+ | `ybc` (yards before contact), `yac` (yards after contact), `brk_tkl` (broken tackles), `att_br` |
| PFR adv receiving | `load_pfr_advstats(stat_type="rec")` | 2018+ | `adot`, `ybc`, `yac`, `brk_tkl`, `drop`, `drop_percent` |
| Participation (play-level) | `load_participation()` | 2016+ (NGS source), 2023+ (FTN source) | `route`, `offense_formation`, `offense_personnel`, `offense_players` (GSIS list), `was_pressure` |
| Play-by-play | `load_pbp()` | 1999+ | `epa`, `cpoe`, `air_yards`, `qb_dropback`, `pass`, `rush` |
| Player metadata | `load_players()` | all | `birth_date`, `draft_year`, `draft_round`, `draft_pick`, `college` |

## Derived metrics we'll compute

| Metric | Derivation |
|---|---|
| Routes run (per player-week/season) | Count plays where player's `gsis_id` is in `participation.offense_players` AND play is a pass (`pbp.pass == 1`). FTN-era 2023+; pre-2023 use NGS-era participation joined to PBP. |
| YPRR | `receiving_yards / routes_run` (in-house figure; differs slightly from PFF's by ~5-10% due to route-definition differences but directionally correct). |
| Route participation rate | `routes_run / team_pass_attempts_while_on_field` |
| Target rate per route | `targets / routes_run` |
| Snap share | `offense_snaps / team_offense_snaps` |

## Gaps remaining (deferred unless backtest demands)

| Metric | Why missing | Workaround / decision |
|---|---|---|
| PFF canonical YPRR | Proprietary route definitions | Use derived in-house YPRR. Revisit PFF ELITE ($199.99/yr) after Phase 2 if backtest demands. |
| Per-player slot rate | Participation has formation/personnel but not per-player alignment | Approximate from offense_formation + receiver position when in 11/12 personnel. Crude. PFF ELITE for clean figure. |
| Charted missed-tackles-forced for RBs | PFR has `brk_tkl` but PFF/SIS chart this more granularly | Use PFR `brk_tkl` and `att_br`. Adequate. |
| PFF pressure split (hurry vs. hit vs. sack) | PFR has totals only | PFR `times_hurried` + `times_hit` + `sacks` is close enough. |

## Data window for each layer

- **Talent layer (Phase 2):** advanced metrics available 2016+ for NGS, 2018+ for PFR. Talent estimates fit on 2016+. Pre-2016 players excluded from talent-conditioned analysis.
- **Comp pool (Phase 4):** receiver comps draw from 2016+ for separation/aDOT; 2023+ for routes-run. Earlier comps available but with fewer features.
- **Backtest holdout (locked, see `holdout.md`):** 2021–2023 inclusive, three seasons. 2024 reserved as final lockbox.
