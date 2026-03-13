"""
Out-of-Distribution (OOD) Detection Module
============================================
Detects when a live match's feature vector is far from the training
distribution, i.e., a situation the model has not seen before.

For unseen / rare situations the model's output should not be trusted
at full weight.  This module assigns each incoming match an OOD score
and a corresponding shrinkage factor that the pipeline uses to reduce
position size or skip the bet entirely.

Two detectors are provided:
    1. MahalanobisOOD  – parametric, based on Gaussian fit to training data.
       Fast, interpretable, works well when features are roughly Gaussian.
    2. IsolationForestOOD – non-parametric ensemble method.
       Handles non-Gaussian distributions and outliers better.

Usage
-----
    from src.ood_detector import MahalanobisOOD

    detector = MahalanobisOOD(threshold_pct=95)
    detector.fit(X_train)

    scores = detector.score(X_live)         # higher = more OOD
    flags  = detector.is_ood(X_live)        # bool array
    shrink = detector.shrinkage(X_live)     # float in [0, 1]
"""

import numpy as np
from typing import Optional


class MahalanobisOOD:
    """
    Mahalanobis-distance based OOD detector.

    Measures how many standard-deviation-equivalents a new point is from the
    centroid of the training distribution, accounting for feature correlations.

    Parameters
    ----------
    threshold_pct : float
        Training-set percentile used as the in-distribution boundary.
        E.g. 95 means: any new point with a score above the 95th percentile
        of training scores is flagged as OOD.
    shrink_min : float
        Minimum shrinkage factor applied to a maximally OOD point (0 = no bet).
    """

    def __init__(self, threshold_pct: float = 95.0, shrink_min: float = 0.0):
        self.threshold_pct = threshold_pct
        self.shrink_min = shrink_min
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None
        self._train_max: Optional[float] = None

    def fit(self, X_train: np.ndarray) -> "MahalanobisOOD":
        """
        Fit detector on training feature matrix.

        Parameters
        ----------
        X_train : np.ndarray of shape (n, d)
            Training feature vectors (already scaled / imputed).
        """
        X = np.asarray(X_train, dtype=float)
        # Replace any remaining NaN/Inf with column means
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            mask = ~np.isfinite(X[:, j])
            X[mask, j] = col_means[j]

        self._mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        # Regularise to ensure invertibility
        cov += np.eye(cov.shape[0]) * 1e-6
        self._cov_inv = np.linalg.inv(cov)

        train_scores = self._mahalanobis(X)
        self._threshold = float(np.percentile(train_scores, self.threshold_pct))
        self._train_max = float(np.max(train_scores))
        return self

    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        diff = X - self._mean
        left = diff @ self._cov_inv
        scores = np.sqrt(np.sum(left * diff, axis=1))
        return scores

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distance for each row (higher = more OOD)."""
        self._check_fitted()
        X = self._clean(X)
        return self._mahalanobis(X)

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        """Return bool array: True where the match is out-of-distribution."""
        return self.score(X) > self._threshold

    def shrinkage(self, X: np.ndarray) -> np.ndarray:
        """
        Return a per-match multiplier in [shrink_min, 1.0].

        - In-distribution match → 1.0 (no shrinkage)
        - Maximally OOD match  → shrink_min
        """
        scores = self.score(X)
        # Normalise: 0 at threshold, 1 at train_max (clamped beyond)
        above = np.maximum(scores - self._threshold, 0.0)
        span = max(self._train_max - self._threshold, 1e-9)
        ood_frac = np.clip(above / span, 0.0, 1.0)
        return 1.0 - ood_frac * (1.0 - self.shrink_min)

    def _clean(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        col_means = self._mean  # use training mean for imputation
        for j in range(X.shape[1]):
            mask = ~np.isfinite(X[:, j])
            X[mask, j] = col_means[j]
        return X

    def _check_fitted(self):
        if self._mean is None:
            raise RuntimeError("Call fit() before score().")


class IsolationForestOOD:
    """
    Isolation Forest based OOD detector.

    Non-parametric; works well for multi-modal or skewed feature distributions.

    Parameters
    ----------
    contamination : float
        Expected fraction of OOD points in training data (sklearn default 0.1).
    shrink_min : float
        Minimum shrinkage factor for maximally anomalous points.
    n_estimators : int
        Number of trees in the isolation forest.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        shrink_min: float = 0.0,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.shrink_min = shrink_min
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None
        self._threshold = None

    def fit(self, X_train: np.ndarray) -> "IsolationForestOOD":
        from sklearn.ensemble import IsolationForest

        X = np.asarray(X_train, dtype=float)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(X)
        # score_samples returns negative anomaly scores; flip so higher = more OOD
        raw = -self._model.score_samples(X)
        # threshold = score at the (1 - contamination) percentile of training
        self._threshold = float(np.percentile(raw, (1.0 - self.contamination) * 100))
        self._train_max = float(np.max(raw))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score (higher = more OOD)."""
        self._check_fitted()
        return -self._model.score_samples(np.asarray(X, dtype=float))

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self._threshold

    def shrinkage(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        above = np.maximum(scores - self._threshold, 0.0)
        span = max(self._train_max - self._threshold, 1e-9)
        ood_frac = np.clip(above / span, 0.0, 1.0)
        return 1.0 - ood_frac * (1.0 - self.shrink_min)

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Call fit() before score().")


# ---------------------------------------------------------------------------
# Hierarchical fallback (rare-state pooling)
# ---------------------------------------------------------------------------

class HierarchicalPrior:
    """
    When a state is OOD, fall back through a hierarchy of broader priors
    rather than inventing a forced prediction.

    Level 0 (most specific): model output
    Level 1: surface-level win rate from historical data
    Level 2: overall historical win rate (≈ 0.5 after randomisation)
    Level 3: market price (if available)

    Usage
    -----
        hp = HierarchicalPrior()
        hp.fit(surface_priors={"Clay": 0.52, "Grass": 0.50, ...}, overall_prior=0.50)

        # Returns blended probability: partially reverts to broader priors
        p_blended = hp.blend(
            p_model=0.73,
            ood_score=0.8,       # 0=in-dist, 1=max OOD
            surface="Clay",
            market_price=0.65,   # optional
        )
    """

    def __init__(self):
        self._surface_priors: dict = {}
        self._overall_prior: float = 0.5

    def fit(
        self,
        surface_priors: Optional[dict] = None,
        overall_prior: float = 0.5,
    ) -> "HierarchicalPrior":
        self._surface_priors = surface_priors or {}
        self._overall_prior = overall_prior
        return self

    def blend(
        self,
        p_model: float,
        ood_score: float,
        surface: Optional[str] = None,
        market_price: Optional[float] = None,
    ) -> float:
        """
        Blend model probability with broader priors proportional to OOD score.

        ood_score = 0  →  return p_model unchanged
        ood_score = 1  →  return market/surface/overall prior

        Parameters
        ----------
        p_model : float
            Model's calibrated probability.
        ood_score : float in [0, 1]
            0 means in-distribution, 1 means maximally OOD.
        surface : str or None
            Surface name for surface-level prior lookup.
        market_price : float or None
            Market implied probability. If provided, used as a prior anchor.

        Returns
        -------
        float: blended probability in (0, 1)
        """
        # Build prior from most-to-least specific
        if market_price is not None:
            prior = market_price
        elif surface and surface in self._surface_priors:
            prior = self._surface_priors[surface]
        else:
            prior = self._overall_prior

        blended = (1.0 - ood_score) * p_model + ood_score * prior
        return float(np.clip(blended, 1e-6, 1.0 - 1e-6))
