# SGF Model — Architecture & Implementation Plan (v2, top-down)

## Vision

A dynasty fantasy football engine that produces **genuinely predictive** player valuations with full uncertainty quantification. The master ranking output is a **career VORP distribution** per player, ordered by a configurable score combining magnitude and confidence.

Specifically the model must:

- Predict next-year and multi-year FP **distributions**, not point estimates.
- Aggregate year-by-year FP samples into **career VORP distributions** via Monte Carlo over correlated trajectories.
- Distinguish breakout candidates from fool's-gold using talent/situation features that drive the FP forecast.
- Provide interpretability (which features drove a projection, which historical players are most similar).
- Quantify uncertainty honestly — well-calibrated 50% / 80% intervals on next-year FP and career VORP.
- Be backtestable component-by-component and produce a single sortable master ranking.

## Pivot from v1

Earlier drafts targeted a bottom-up architecture: project individual stats per player, decompose into opportunity × efficiency, multiply through scoring. After review, the project is pivoting to **top-down direct FP modeling**:

- Target is FP directly, not stats. Scoring config still matters but only for VORP computation, not as a transformation over projected components.
- Talent and situation factors (YPRR, separation, target share, team volume, QB context, etc.) become **features** on the right-hand side, not intermediate projection targets.
- Age becomes a feature, not a fitted curve.
- Predictive intervals come from quantile regression (or Bayesian posteriors), not bolted-on post-hoc residual sigmas.
- Career VORP comes from Monte Carlo over correlated year samples, not summed point estimates.

Why: less error compounding, fewer hyperparameters, smaller codebase, better aligned with modern ML, and the distributional output is structurally built in rather than reverse-engineered. League-format flexibility is preserved by either training per common scoring or by predicting component FP (rec/rush/pass) and combining.

## Architecture overview

```
                                        ┌─────────────────────────────────────┐
                                        │  Features                            │
                                        │  ─────────                           │
                                        │  talent:    YPRR, separation,        │
                                        │             aDOT, success rate, ...  │
                                        │  situation: team volume, depth chart │
                                        │             QB tier, scheme proxy    │
                                        │  demo:      age, position, exp,      │
                                        │             draft capital            │
                                        │  market:    ADP, dynasty ADP         │
                                        │  history:   weighted prior FP        │
                                        └─────────────────────────────────────┘
                                                          ↓
                                        ┌─────────────────────────────────────┐
                                        │  Per-position quantile FP model     │
                                        │  (gradient boosting w/ quantile loss│
                                        │   OR probabilistic regression)      │
                                        │                                     │
                                        │  Output: FP distribution per future │
                                        │  year offset (t+1 ... t+horizon)    │
                                        └─────────────────────────────────────┘
                                                          ↓
                                        ┌─────────────────────────────────────┐
                                        │  Career Monte Carlo simulator       │
                                        │  - Sample N=1000 correlated         │
                                        │    trajectories per player          │
                                        │  - Convert each year to VORP via    │
                                        │    league config                    │
                                        │  - Sum to career VORP per sample    │
                                        └─────────────────────────────────────┘
                                                          ↓
                                        ┌─────────────────────────────────────┐
                                        │  Career VORP distribution + ranking │
                                        │  - per-player percentiles (P10..P90)│
                                        │  - configurable master-rank score:  │
                                        │     mean / median / risk-adjusted / │
                                        │     P25 floor / P(VORP > threshold) │
                                        └─────────────────────────────────────┘
```

## Key design decisions

1. **Top-down FP target.** Predict FP directly; talent/situation are features.
2. **Per-position models** (initial). One quantile model each for QB/RB/WR/TE. May consolidate later if cross-position learning helps.
3. **Quantile gradient boosting** as the default (LightGBM/XGBoost). Bayesian regression considered later if calibration demands it.
4. **Monte Carlo with correlated samples.** Latent-skill parameterization: draw a per-player "skill" scalar per simulation, then each year's FP draws conditional on skill. Years correlate via shared skill; structural decline applied multiplicatively via age feature.
5. **Career horizon** capped at age 40 (or position-specific: 35 for RBs, 38 for WRs/TEs, 42 for QBs). Beyond that, prob(still playing) ≈ 0 in the data.
6. **Master ranking score** configurable. Default: median career VORP for the "neutral" view, plus a risk-adjusted view (`median − 0.5 × SD`) and a floor view (P25). Surface all three; user picks the lens.
7. **League flexibility** via VORP/dynasty post-processing, not the FP model. The FP model is trained on one scoring system; VORP/career-VORP computed per league config.

## Phased implementation plan

Phase 0 (backtest harden + data audit) is complete; assets at `docs/phase0-baseline.md`, `docs/data-audit.md`, `docs/holdout.md` are reusable as-is. All later phases compare against the locked 2021–2023 holdout.

### Phase 1 — Feature pipeline + baseline FP model

Build the feature pipeline and a per-position quantile model using only data we already have (no advanced metrics yet). Goal: prove the top-down approach beats the v1 Marcel baseline before investing in advanced features.

Deliverables:
- `src/sgf_model/features/` — feature engineering module: takes player_seasons, weekly stats, players, team mapping; emits a (player_id, season, feature1, ..., featureN, target_FP) feature matrix per position.
- Initial feature set: age, position, experience, weighted historical FP (1y / 2y / 3y), historical games played, position-tier indicators (top-12 last year, etc.).
- Quantile gradient boosting models (P10/P25/P50/P75/P90) per position. LightGBM or XGBoost.
- Wire into existing backtest. Add distributional calibration metrics (we have these from Phase 0; quantile predictions slot in directly).

Exit criteria:
- Backtest shows clear improvement on at least two of: (a) per-position MAE, (b) per-position Spearman, (c) cov_50/cov_80 calibration deviation from target.
- Calibration intervals are model-emitted (quantile predictions), not bolt-on residual sigmas.

### Phase 2 — Talent + situation + market features

Add the advanced metric features identified in the data audit. The model architecture doesn't change; only the feature set expands.

Deliverables:
- Talent features: YPRR (derived), routes run, target rate per route, separation, aDOT, CPOE, RYOE, success rate, snap share. Per-position relevance.
- Situation features: team pass/rush volume (3-year weighted), team rank context, depth-chart proxy (position-rank within team in prior year), QB tier feature for WR/TE models.
- Market feature: ADP / dynasty ADP. Treated as a feature, not a target. Disagreement flag computed downstream.
- Demographic features: draft capital (round/pick log), college tier proxy if cheap.
- Per-feature ablation in backtest to measure marginal lift.

Exit criteria:
- Material improvement on rank quality (Spearman, top-N hit rate) for at least 2 positions vs. Phase 1.
- Ablation table identifies the high-value features for each position.
- ADP disagreement output works: flag players where model and ADP differ by >X percentile rank.

### Phase 3 — Multi-year FP + career Monte Carlo + master ranking

Extend the FP model from t+1 to a horizon, build the career simulator, emit the career VORP distribution and master ranking.

Deliverables:
- Multi-year FP model: train per-(position, future-year-offset) quantile models, OR train a single model with future_year_offset as a feature. Decision based on backtest.
- Career simulator: latent-skill Monte Carlo. Per player, draw N=1000 trajectories, each a sequence of FP samples across remaining career.
- Career-length handling: either explicit hazard model on "still playing" per year, OR let FP decay handle it (FP → 0 as age increases beyond position-typical retirement age, validated against retired-player data).
- Career VORP distribution: apply league config replacement level per year, sum, percentile.
- Master ranking scorer: configurable (mean / median / median − k×SD / P25 / P(VORP > threshold)).

Exit criteria:
- Career VORP distribution emits sensibly: top-30 players' median career VORPs are interpretable, P10/P90 spreads are wider for young/rookie players than for established veterans.
- Career VORP calibration on retired players: realized career VORP for players who retired between 2010-2020 (held out from any training) falls within model's predicted interval at expected rate.
- Master ranking sortable by every configured score with sensible orderings.

### Phase 4 — Specialty handling: rookies, injuries, trades, comps

Cases the base model doesn't handle well, plus interpretability outputs.

Deliverables:
- Rookie features: draft capital (round, pick log) + college stats proxy + cohort indicators. Same model architecture; rookie features fill in for the "history" features that don't exist yet.
- Games-played model: either as separate per-year sub-model (logistic on age, position, past games missed) or as a feature/output of the FP model.
- Trade / FA tool: swap situation features (new team's volume, new QB, new depth chart) for a player, re-run projection, show before/after distribution. Built on top of the feature pipeline.
- Comparable-player output: for interpretability, find K nearest historical players in feature space at the same age. Returns "most similar past players" list with their actual career outcomes. Used for the ranking UI, not as the projection mechanism.
- SHAP outputs for top-N players: which features drove the projection.

Exit criteria:
- Rookies projected sensibly; cohort-level rookie projection accuracy compares favorably to v1 ridge regression.
- Trade tool produces plausible outputs on held-out historical trades.
- Comparable-player lists pass sanity checks ("this rookie's comps are A/B/C" — A/B/C are believable).

### Phase 5 — Production hardening + 2024 lockbox validation

Final validation and shipping polish.

Deliverables:
- Full backtest sweep across feature variants, model hyperparameter grid, and ranking-score choices.
- 2024 lockbox: single end-of-project validation. Document expected metrics, run once, compare. Do not iterate after seeing 2024 results.
- CLI / UI updates: master ranking with configurable score, drill-down to per-player distribution, comparable-player list, ADP disagreement flagging.
- Storage schema for distributional snapshots (currently stores point rankings; needs to handle quantile / sample-based output).

Exit criteria:
- 2024 lockbox metrics meet success criteria (below) on at least 2 of 3 positions.
- Production pipeline runs end-to-end and persists distributional snapshots.

## Success criteria

Compared against the Phase 0 baseline (`docs/phase0-baseline.md`):

| Metric | Phase 0 baseline | Phase 5 target |
|---|---|---|
| Per-position MAE | 28–65 FP | 20–25% reduction |
| Per-position Spearman | 0.71–0.76 | ≥0.80 |
| Per-position top12_hit | 0.42–0.58 | ≥0.65 |
| cov_50 deviation from 0.5 | 0.05–0.11 | ≤0.05 |
| cov_80 deviation from 0.8 | 0.02–0.06 | ≤0.04 |
| Mean bias | +5 FP | ±2 FP |
| Career VORP calibration | (new) | retired-player career VORP in model's 80% interval ≥75% of the time |
| Top-30 master ranking hit rate | (new) | 60%+ overlap with held-out top-30 by realized 3-year FP |

## What we keep from v1

- `src/sgf_model/data/` — entirely.
- `src/sgf_model/storage/` — entirely (extend schema for distributional output in Phase 5).
- `src/sgf_model/scoring/` — entirely. Still needed to convert historical stats → training-target FP and to compute VORP per league.
- `src/sgf_model/valuation/` — entirely. VORP and dynasty value calculations are league-config-driven; the career simulator feeds samples in, the existing logic produces VORP.
- `src/sgf_model/evaluation/backtest.py` — fully reusable. Quantile-model predictions slot into the existing calibration scoring (no longer need bolt-on residual sigmas).
- `docs/data-audit.md`, `docs/holdout.md`, `docs/phase0-baseline.md` — all valid.

## What v1 we retire

- `src/sgf_model/curves/` — age becomes a feature, not a fitted curve.
- `src/sgf_model/projections/regression.py` — empirical-Bayes shrinkage is replaced by quantile regression's implicit handling.
- `src/sgf_model/projections/player.py` — Marcel-style projection replaced by feature-driven model.
- `src/sgf_model/projections/rookie.py` — rookies handled by the same model with rookie-appropriate features.
- `src/sgf_model/projections/team.py` — team volume becomes a feature; explicit team-volume projection sub-model not needed in v2 (Phase 3 may introduce a small team-volume sub-model if the FP model has trouble learning it implicitly).

Retirement happens lazily — keep the v1 code in place until v2 equivalents are validated. Phase 1 introduces new modules alongside; v1 modules are deleted when their replacements clear backtest gates.

## Cross-cutting principles

- **Distributions, not points.** Every model output is a distribution.
- **Component-level backtest as ground truth.** No model change ships without backtest evidence it improves the metric it claims to improve.
- **ADP as a feature with disagreement-flagging.** The model uses ADP as input; disagreement with ADP is a downstream alert.
- **Schema reflects distributions.** Stored snapshots include sample arrays or percentile summaries, not just point estimates.
- **Calibration honest.** If 80% intervals contain truth 60% of the time, we ship the calibration deficit alongside the rankings rather than hiding it.

## Risks & open decisions

- **Quantile vs. probabilistic regression.** Quantile boosting is simpler and well-supported; Bayesian regression gives better joint distribution properties but is heavier. Start with quantile; revisit if calibration is poor.
- **Correlated-samples parameterization.** Latent-skill scalar is the simplest; if FP variance is highly age-driven, may need a richer joint structure. Validate against retired-player career-VORP calibration.
- **Per-position vs. global models.** Per-position is simpler and likely fine. Global with position as a feature could be considered if RB/TE samples are too small for stable training.
- **Career length modeling.** Hazard vs. natural-decay. Try natural-decay first; revisit if model overestimates careers of marginal players.
- **Comp pool data depth.** Talent features only exist back to ~2016. Career-VORP calibration on retired players needs the retired player to have NGS-era seasons; this constrains the calibration sample. Acceptable for now; flag if it becomes limiting.
- **Train-target scoring.** Train on PPR by default (most common dynasty format). Provide secondary models for half-PPR / TE-premium if backtest shows they meaningfully differ; otherwise approximate by adjusting VORP at inference.

## Sequencing & off-ramps

The plan is built to allow off-ramps at every phase:

- After **Phase 1**: if the feature-engineered baseline doesn't beat v1, the top-down hypothesis is wrong. Stop and revisit.
- After **Phase 2**: if advanced features don't add lift, scope down the data-sourcing effort.
- After **Phase 3**: if career VORP distributions are poorly calibrated on retired players, revisit the correlated-sampling design before shipping master ranking.

Each phase exit criteria is a real decision gate, not a checklist.
