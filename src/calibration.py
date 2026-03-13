"""
Probability Calibration Module
================================
Wraps raw neural-network sigmoid outputs and aligns them with observed
win frequencies using isotonic regression or Platt (logistic) scaling.

Raw NN scores are often poorly calibrated – the model may be systematically
overconfident or underconfident.  This module fixes that so that when the
model says "70 % chance Player A wins", it really means 70 % historically.

Usage
-----
    from src.calibration import ProbabilityCalibrator

    cal = ProbabilityCalibrator(method="isotonic")
    cal.fit(raw_probs_train, y_train)

    calibrated = cal.transform(raw_probs_test)
    # or equivalently
    calibrated = cal.fit_transform(raw_probs_train, y_train, raw_probs_test)

    # Inspect calibration quality
    metrics = cal.calibration_metrics(raw_probs_test, calibrated, y_test)
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


class ProbabilityCalibrator(BaseEstimator, TransformerMixin):
    """
    Post-hoc probability calibrator for binary classification.

    Parameters
    ----------
    method : {"isotonic", "platt"}
        - "isotonic"  : non-parametric, monotone (more flexible, needs ~1000+ samples)
        - "platt"     : logistic sigmoid fit (works well with fewer samples)
    clip : bool
        Clip output probabilities to [eps, 1-eps] to avoid log(0) downstream.
    eps : float
        Clipping bound when clip=True.
    """

    def __init__(self, method: str = "isotonic", clip: bool = True, eps: float = 1e-6):
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self.clip = clip
        self.eps = eps
        self._calibrator = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit the calibrator on a held-out calibration set.

        Parameters
        ----------
        raw_probs : array-like of shape (n,)
            Uncalibrated probabilities from the NN (values in [0, 1]).
        y : array-like of shape (n,)
            True binary labels (0 or 1).

        Returns
        -------
        self
        """
        raw_probs = np.asarray(raw_probs, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_probs, y)
        else:  # platt
            # Reshape for sklearn's LogisticRegression
            X = raw_probs.reshape(-1, 1)
            self._calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self._calibrator.fit(X, y)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new raw probabilities.

        Parameters
        ----------
        raw_probs : array-like of shape (n,)
            Uncalibrated NN output probabilities.

        Returns
        -------
        calibrated : np.ndarray of shape (n,)
            Calibrated probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        raw_probs = np.asarray(raw_probs, dtype=float).ravel()

        if self.method == "isotonic":
            calibrated = self._calibrator.predict(raw_probs)
        else:
            X = raw_probs.reshape(-1, 1)
            calibrated = self._calibrator.predict_proba(X)[:, 1]

        if self.clip:
            calibrated = np.clip(calibrated, self.eps, 1.0 - self.eps)

        return calibrated

    def fit_transform(
        self,
        raw_probs_calib: np.ndarray,
        y_calib: np.ndarray,
        raw_probs_new: np.ndarray,
    ) -> np.ndarray:
        """Convenience: fit on calibration set, transform new set."""
        self.fit(raw_probs_calib, y_calib)
        return self.transform(raw_probs_new)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def calibration_metrics(
        self,
        raw_probs: np.ndarray,
        calibrated_probs: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Compute Expected Calibration Error (ECE) before and after calibration.

        A well-calibrated model has ECE near 0.

        Returns
        -------
        dict with keys:
            "ece_before"  : ECE of raw probabilities
            "ece_after"   : ECE of calibrated probabilities
            "brier_before": Brier score before calibration
            "brier_after" : Brier score after calibration
        """
        raw_probs = np.asarray(raw_probs).ravel()
        calibrated_probs = np.asarray(calibrated_probs).ravel()
        y_true = np.asarray(y_true).ravel()

        def _ece(probs, labels, n_bins):
            frac_pos, mean_pred = calibration_curve(
                labels, probs, n_bins=n_bins, strategy="uniform"
            )
            bin_sizes = np.histogram(probs, bins=n_bins, range=(0, 1))[0]
            weights = bin_sizes / bin_sizes.sum()
            ece = float(np.sum(np.abs(frac_pos - mean_pred) * weights[: len(frac_pos)]))
            return ece

        def _brier(probs, labels):
            return float(np.mean((probs - labels) ** 2))

        return {
            "ece_before": _ece(raw_probs, y_true, n_bins),
            "ece_after": _ece(calibrated_probs, y_true, n_bins),
            "brier_before": _brier(raw_probs, y_true),
            "brier_after": _brier(calibrated_probs, y_true),
        }

    def plot_reliability_diagram(
        self,
        raw_probs: np.ndarray,
        calibrated_probs: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> None:
        """Plot reliability diagram (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        raw_probs = np.asarray(raw_probs).ravel()
        calibrated_probs = np.asarray(calibrated_probs).ravel()
        y_true = np.asarray(y_true).ravel()

        frac_before, mean_before = calibration_curve(y_true, raw_probs, n_bins=n_bins)
        frac_after, mean_after = calibration_curve(y_true, calibrated_probs, n_bins=n_bins)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.plot(mean_before, frac_before, "s-", label="Before calibration")
        ax.plot(mean_after, frac_after, "o-", label=f"After ({self.method})")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Reliability Diagram")
        ax.legend()
        plt.tight_layout()
        plt.show()
