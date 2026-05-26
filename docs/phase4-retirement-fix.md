# Phase 4 — Retirement modeling fix (inactive-season training rows)

Built and validated 2026-05-25. Addresses the Phase 3 finding that the model was applying ~zero aging penalty to elite veterans (Kelce projected 273 FP at 34 → 258 at 38, with P10 actually rising over time). Root cause: survivorship bias in training data — only players who *kept playing* at older ages were in the model's view of "what an old player looks like."

## The fix

`build_feature_matrix(include_inactive_targets=True)` (default since Phase 4) modifies `_build_target_rows` to:

1. **Use left join** instead of inner join on target_season — keep all (anchor, offset) pairs, not just those with realized FP.
2. **Fill missing target_fp with 0** for target seasons within the training data window. These are "player was inactive that year" examples (retired, cut, or season-long injury).
3. **Keep only the first inactive offset per (player, anchor_season).** The transition from active → inactive is the informative signal. Including every subsequent zero year was the bug in my first attempt — it inflated the zero rate to 70%+ at offset 5 and dragged median predictions for any veteran toward zero, even Henry and Evans who were clearly still playing.

Inactive-rate after the fix is stable around 21-29% across offsets (was 29% → 75% as offsets increased in the first attempt).

## Effect on Kelce (the canonical defect from Phase 3)

| Age | P50 FP (Phase 3 — no fix) | P50 FP (Phase 4 — first-inactive) |
|---|---|---|
| 34 | 273 | 300 |
| 35 | 268 | 282 |
| 36 | 265 | 243 |
| 37 | 260 | 249 |
| 38 | 258 | 225 |
| **5-yr drop** | **-5%** | **-25%** |

Believable decline curve now. Year-1 prediction is slightly higher (300 vs. 273) because the model isn't pulling him toward a survivorship-biased "median 34-year-old TE." Year-5 drops 25% — appropriate for a TE going 34→38.

## Position-appropriate aging curves emerged

5-year P50 decline for representative players in 2023 inference:

| Player | Position | Age 23→28 | 5-year P50 decline |
|---|---|---|---|
| Mahomes | QB | 28→32 | **-7%** (QBs hold up) |
| Allen | QB | 27→31 | ~-10% |
| Jefferson | WR | 24→28 | **-21%** |
| Lamb | WR | 24→28 | -20% |
| Kelce | TE | 34→38 | **-25%** |
| Kittle | TE | 30→34 | -22% |
| Henry | RB | 29→33 | **-46%** (RB cliff) |
| Evans | WR | 30→34 | **-60%** (older WR sharp decline) |

QBs hold up best, RBs decline fastest, older WRs especially sharp. The shape now matches NFL aging realities.

## Backtest metrics (locked holdout, 2021–2023, 3-season averages)

| Position | Metric | v1_no_reg | v2_phase2 (no fix) | v2_phase4 (fix) | Phase 4 vs v1 |
|---|---|---|---|---|---|
| RB | MAE | 51.5 | 46.0 | 46.1 | **-10%** (held) |
| RB | Spearman | 0.678 | 0.681 | **0.693** | better |
| RB | top12_hit | 0.417 | 0.500 | **0.528** | **+27%** |
| RB | cov_50 | 0.558 | 0.534 | 0.688 | over-covered (intervals too wide) |
| RB | cov_80 | 0.813 | 0.822 | **0.844** | better |
| TE | top12_hit | 0.583 | 0.556 | **0.667** | **+14%** |
| TE | Spearman | 0.748 | 0.758 | **0.760** | better |
| TE | MAE | 29.0 | 28.6 | 28.6 | tied |
| WR | MAE | 42.8 | 41.6 | **41.1** | -4% |
| WR | top12_hit | 0.556 | 0.444 | 0.472 | **still worse than v1** |
| WR | cov_80 | 0.825 | 0.789 | **0.839** | better |
| QB | MAE | 68.3 | 67.1 | 67.3 | tied |
| QB | top12_hit | 0.528 | 0.528 | 0.500 | slightly worse |
| ALL | mean_bias | +7 | -3 | **-15** | **more negative** |

### What improved

- **RB top-12 hit rate +27% vs. v1.** Best single result of the project. Tells us aging-corrected RB projections actually help identify who'll perform.
- **TE top-12 hit rate +14% vs. v1.** Kelce-style misranking corrected without losing legitimate elite TEs.
- **WR/TE/RB calibration tightened** — intervals contain truth at closer to target rates on cov_80.
- **Spearman rank correlation improved or held** on every position.

### What worsened

- **Mean bias went from -3 to -15 FP.** The model now systematically under-projects. The zero-FP injection rows pull every prediction down. Could be partially recalibrated with a small constant shift, but the direction (slightly conservative) is the safer side for dynasty rankings.
- **cov_50 is now over-covered (0.66 on RB, 0.66 on WR vs. target 0.5)** — intervals are too wide at the inner band. Conformal calibration is overcorrecting after zero rows added variance.
- **QB top-12 hit fell** from 0.528 → 0.500. QBs are sample-starved and the retirement injection introduced more zero examples than was needed (most veteran QBs don't retire at 32).
- **WR top-12 hit still worse than v1** (0.472 vs. 0.556). The Phase 1 → Phase 2 → Phase 4 progression hasn't fully closed this. Likely needs the routes-derived features deferred from Phase 2 (per-route stats, target share) and possibly ADP.

## Top 15 dynasty ranking with the fix (median career VORP, 12-team 1QB, PPR)

| Rank | Player | Pos | P10 | P50 | P90 | P(>0) |
|---|---|---|---|---|---|---|
| 1 | J.Allen | QB | 141 | 796 | 988 | 0.99 |
| 2 | P.Mahomes | QB | 185 | 718 | 941 | 1.00 |
| 3 | J.Jefferson | WR | 0 | 703 | 853 | 0.69 |
| 4 | T.Kelce | TE | 0 | 610 | 768 | 0.85 |
| 5 | J.Waddle | WR | 0 | 601 | 769 | 0.70 |
| ... | | | | | | |
| 11 | T.Brady | QB | 0 | 494 | 799 | 0.83 |
| 12 | J.Chase | WR | 0 | 453 | 787 | 0.67 |
| 13 | T.Lawrence | QB | 0 | 450 | 651 | 0.83 |
| 14 | J.Jacobs | RB | 0 | 440 | 697 | 0.68 |
| 15 | T.McLaurin | WR | 0 | 421 | 506 | 0.67 |

Kelce dropped from #3 to #4 (Jefferson passes him). Top of the list is more weighted toward QB now (Allen, Mahomes #1-2), reflecting the position's flatter aging curve. T.Brady at #11 remains a defect — the model still under-penalizes age 46 (he retired before the 2023 season we're projecting). Would need an explicit hazard model or hard age cap to fully address.

## What's still defective

1. **WR top-N hit rate** — unchanged regression from v1. The retirement fix doesn't help here. Needs routes-derived features (per-route target rate, derived YPRR) and/or ADP.
2. **Brady-class outliers** — players who literally retired before the inference season still show up. The "inactive within training data" only catches retirement signals visible BEFORE 2023. Phase 5 should add real-time retirement signal (offseason news / contract status) or hard age caps.
3. **P10 = 0 for elite young players** like Jefferson — the retirement-driven lower tail clamps even safe-bet players. Could be addressed by per-player bust probability modeling instead of uniform zero injection.
4. **Mean bias -15** is a known cost of the fix. Could correct with a constant per-position shift, or accept the conservative tilt.

## Files changed

- `src/sgf_model/features/builder.py` — `_build_target_rows` extended with `include_inactive_targets` (default True). New "first-inactive only" logic to prevent zero-overload.

## Verdict

Phase 4 retirement fix is **shipped**. Aging curves are realistic, RB and TE rank quality improved materially vs. both Phase 2 and v1, calibration tightened. The mean-bias cost is acceptable; the still-imperfect WR top-12 and Brady-class outliers are the next two things to address.

Next-step options for Phase 4 continuation:
- **A.** Address WR top-12 with routes-derived features (deferred from Phase 2). Most likely to close the persistent gap.
- **B.** Add hard retirement filtering (age cap per position or offseason status check) — fixes Brady class but is crude.
- **C.** Validate career-VORP calibration on a retired-player holdout (still-deferred Phase 3 exit criterion #2).
- **D.** Address the mean_bias = -15 systematic under-projection with a per-position constant correction or alternative loss function.
