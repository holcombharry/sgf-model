"""Per-position scikit-learn HistGradientBoostingRegressor quantile model for FP.

Trains one model per (position, quantile). Default quantiles are 0.10 / 0.25 /
0.50 / 0.75 / 0.90 so the model directly emits 50% and 80% predictive intervals
with the median as the point estimate.

HistGradientBoostingRegressor handles null features natively — no imputation
needed, which is the whole reason we leave prior_fp_*y as null instead of
zero-filling: the model learns "no signal here" from the null directly rather
than from a sentinel.

We use sklearn rather than LightGBM/XGBoost because the latter two require a
system-installed libomp matching the Python build's architecture, which is a
fragile macOS dependency. sklearn ships pure Python + numpy/scipy. The
algorithm (histogram-based gradient boosting) is the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from sgf_model.features import PHASE1_FEATURE_COLUMNS

DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Hyperparameters for the Phase 1 baseline. Intentionally not tuned — we want
# to know how much lift the architecture itself gives before tuning is added
# as a confounder. Phase 2+ will revisit.
_PARAMS: dict = {
    "loss": "quantile",
    "max_iter": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "early_stopping": False,
}


# Interval-level → (lower-quantile, upper-quantile) mapping used by conformal
# calibration. The model exposes 50% and 80% intervals built from these quantile
# pairs after applying per-position calibration offsets.
_INTERVAL_QUANTILES: dict[int, tuple[float, float]] = {
    50: (0.25, 0.75),
    80: (0.10, 0.90),
}


@dataclass
class _PositionModels:
    """All quantile models for a single position plus conformal calibration offsets.

    `offsets[level]` is the per-interval-level (50 or 80) additive widening from
    Conformalized Quantile Regression (Romano, Patterson & Candès 2019). At
    inference time the calibrated interval is `(q_lo - offset, q_hi + offset)`,
    which empirically achieves the target coverage on the calibration set.
    """
    models: dict[float, HistGradientBoostingRegressor]
    offsets: dict[int, float]


class QuantileFPModel:
    """Per-position quantile gradient boosting for next-year FP.

    Usage:
        model = QuantileFPModel()
        model.fit(training_feature_matrix)
        predictions = model.predict(inference_feature_matrix)

    `predictions` is a polars DataFrame with `fantasy_points_season` (P50) and
    `proj_fp_lower_50` / `proj_fp_upper_50` / `proj_fp_lower_80` / `proj_fp_upper_80`
    columns, plus the player metadata columns from the input.
    """

    def __init__(
        self,
        feature_columns: Iterable[str] = PHASE1_FEATURE_COLUMNS,
        quantiles: Iterable[float] = DEFAULT_QUANTILES,
        positions: Iterable[str] = DEFAULT_POSITIONS,
        params: dict | None = None,
        calibrate: bool = True,
        calibration_fraction: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        self.quantiles = tuple(quantiles)
        self.positions = tuple(positions)
        self.params = {**_PARAMS, **(params or {})}
        self.calibrate = calibrate
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state
        self._by_position: dict[str, _PositionModels] = {}

        for q in (0.5, 0.10, 0.25, 0.75, 0.90):
            if q not in self.quantiles:
                raise ValueError(
                    f"Required interval/median quantile {q} not in {self.quantiles}. "
                    "Pass a `quantiles` set that includes 0.10/0.25/0.50/0.75/0.90 "
                    "or update the predict() interval logic."
                )

    def fit(self, training: pl.DataFrame) -> "QuantileFPModel":
        """Train one model per (position, quantile) on rows with a non-null target.

        When `calibrate=True`, holds out a random fraction of training rows
        per position to compute Conformalized Quantile Regression offsets so
        the emitted intervals achieve target coverage.
        """
        train = training.filter(pl.col("target_fp").is_not_null())
        rng = np.random.default_rng(self.random_state)
        for position in self.positions:
            sub = train.filter(pl.col("position") == position)
            if sub.height < 50:
                raise ValueError(
                    f"Position {position!r} has only {sub.height} training rows; "
                    "need at least 50 for stable HGB fits."
                )

            # Split into fit and calibration subsets if calibration is enabled.
            n = sub.height
            if self.calibrate:
                calib_n = max(30, int(round(n * self.calibration_fraction)))
                if calib_n >= n - 30:
                    raise ValueError(
                        f"Position {position!r}: calibration_fraction={self.calibration_fraction} "
                        f"leaves only {n - calib_n} fit rows out of {n}; need at least 30 each. "
                        "Reduce calibration_fraction or set calibrate=False."
                    )
                perm = rng.permutation(n)
                calib_idx = perm[:calib_n]
                fit_idx = perm[calib_n:]
            else:
                fit_idx = np.arange(n)
                calib_idx = np.array([], dtype=np.int64)

            X_all = sub.select(self.feature_columns).to_numpy()
            y_all = sub["target_fp"].to_numpy()
            X_fit, y_fit = X_all[fit_idx], y_all[fit_idx]

            models: dict[float, HistGradientBoostingRegressor] = {}
            for q in self.quantiles:
                est = HistGradientBoostingRegressor(
                    **self.params,
                    quantile=q,
                    random_state=self.random_state,
                )
                est.fit(X_fit, y_fit)
                models[q] = est

            # Compute conformal offsets per interval level on the calibration set.
            offsets: dict[int, float] = {50: 0.0, 80: 0.0}
            if self.calibrate and len(calib_idx) > 0:
                X_cal, y_cal = X_all[calib_idx], y_all[calib_idx]
                offsets = self._fit_conformal_offsets(models, X_cal, y_cal)

            self._by_position[position] = _PositionModels(models=models, offsets=offsets)
        return self

    @staticmethod
    def _fit_conformal_offsets(
        models: dict[float, HistGradientBoostingRegressor],
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> dict[int, float]:
        """Conformalized Quantile Regression offsets per interval level.

        For interval (q_lo, q_hi) covering level `c`, the conformity score is
            s_i = max(q_lo_hat(x_i) - y_i, y_i - q_hi_hat(x_i))
        and the offset is the `(1 - alpha) * (n + 1) / n`-th quantile of the
        scores, where `alpha = 1 - c`. Calibrated intervals
            (q_lo_hat - offset, q_hi_hat + offset)
        achieve coverage at least `c` on exchangeable data.
        """
        offsets: dict[int, float] = {}
        n = len(y_cal)
        for level, (q_lo, q_hi) in _INTERVAL_QUANTILES.items():
            lo_pred = models[q_lo].predict(X_cal)
            hi_pred = models[q_hi].predict(X_cal)
            scores = np.maximum(lo_pred - y_cal, y_cal - hi_pred)
            target_coverage = level / 100.0
            # Finite-sample-corrected quantile rank.
            rank = int(np.ceil((n + 1) * target_coverage))
            rank = min(max(rank, 1), n)
            offsets[level] = float(np.sort(scores)[rank - 1])
        return offsets

    def predict(self, inference: pl.DataFrame) -> pl.DataFrame:
        """Predict FP quantiles for each row in `inference`.

        Returns a DataFrame with the input's metadata columns plus:
            fantasy_points_season   (median, P50 — the point estimate)
            proj_fp_lower_50, proj_fp_upper_50   (50% interval from P25/P75)
            proj_fp_lower_80, proj_fp_upper_80   (80% interval from P10/P90)
        """
        if not self._by_position:
            raise RuntimeError("Model is not fit. Call .fit() before .predict().")

        pieces: list[pl.DataFrame] = []
        for position, pos_models in self._by_position.items():
            sub = inference.filter(pl.col("position") == position)
            if sub.height == 0:
                continue
            X = sub.select(self.feature_columns).to_numpy()

            preds_by_q: dict[float, np.ndarray] = {
                q: model.predict(X) for q, model in pos_models.models.items()
            }

            # Apply conformal calibration offsets per interval level (zero when
            # calibrate=False). Widening happens BEFORE the row-wise sort so
            # monotonicity is enforced on calibrated quantiles.
            calibrated = {
                0.50: preds_by_q[0.50],
                0.25: preds_by_q[0.25] - pos_models.offsets.get(50, 0.0),
                0.75: preds_by_q[0.75] + pos_models.offsets.get(50, 0.0),
                0.10: preds_by_q[0.10] - pos_models.offsets.get(80, 0.0),
                0.90: preds_by_q[0.90] + pos_models.offsets.get(80, 0.0),
            }

            # Enforce monotonicity across calibrated quantiles. Independent
            # quantile models + conformal calibration don't guarantee ordering,
            # so sort row-wise.
            ordered_qs = sorted(calibrated.keys())
            stacked = np.stack([calibrated[q] for q in ordered_qs], axis=1)
            stacked.sort(axis=1)
            for i, q in enumerate(ordered_qs):
                calibrated[q] = stacked[:, i]

            out = sub.with_columns(
                pl.Series("fantasy_points_season", calibrated[0.50]),
                pl.Series("proj_fp_lower_50", calibrated[0.25]),
                pl.Series("proj_fp_upper_50", calibrated[0.75]),
                pl.Series("proj_fp_lower_80", calibrated[0.10]),
                pl.Series("proj_fp_upper_80", calibrated[0.90]),
            )
            pieces.append(out)

        if not pieces:
            return inference
        return pl.concat(pieces, how="diagonal")

    def feature_importance(
        self,
        position: str,
        validation: pl.DataFrame,
        quantile: float = 0.5,
        n_repeats: int = 5,
    ) -> pl.DataFrame:
        """Permutation feature importance for one (position, quantile) model.

        Uses sklearn.inspection.permutation_importance, which shuffles each
        feature on a held-out set and measures how much the score degrades.
        Slower than tree-based gain importance but more reliable for HGB,
        which doesn't expose gain importance directly.

        `validation` should be a feature-matrix slice **not** used for training
        (e.g., a single test season). Pass enough rows for stable estimates;
        50+ recommended.
        """
        from sklearn.inspection import permutation_importance

        if position not in self._by_position:
            raise KeyError(f"Position {position!r} not fit.")
        model = self._by_position[position].models[quantile]
        sub = validation.filter(
            (pl.col("position") == position) & pl.col("target_fp").is_not_null()
        )
        if sub.height < 20:
            raise ValueError(
                f"Validation set has {sub.height} rows for {position} — need ≥20."
            )
        X = sub.select(self.feature_columns).to_numpy()
        y = sub["target_fp"].to_numpy()
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=self.random_state
        )
        return pl.DataFrame(
            {
                "feature": list(self.feature_columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort("importance_mean", descending=True)
