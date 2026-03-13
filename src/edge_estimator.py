"""
Edge Estimator Module
======================
Translates a calibrated probability + market price + uncertainty into a
single actionable "usable edge" value.

The fundamental formula is:

    usable_edge = p_calibrated - q - lambda * uncertainty

Where:
    p_calibrated  = calibrated model probability (Player A wins)
    q             = market implied probability    (derived from decimal odds)
    lambda        = uncertainty penalty coefficient (default 0.5)
    uncertainty   = MC Dropout std or similar measure

If usable_edge <= 0 → no bet.
If usable_edge >  0 → pass to sizing engine.

Market Odds Conversion
-----------------------
Decimal odds (e.g. 1.85) → implied probability = 1 / decimal_odds
If the book has two-sided odds, the sum of implied probs > 1 (the overround).
We strip the overround before computing edge.

Usage
-----
    from src.edge_estimator import EdgeEstimator

    estimator = EdgeEstimator(uncertainty_lambda=0.5, min_edge=0.03)

    result = estimator.estimate(
        p_calibrated=0.68,
        decimal_odds_A=1.72,   # market odds for player A
        decimal_odds_B=2.20,   # market odds for player B (for overround removal)
        uncertainty=0.05,
    )
    # result.usable_edge, result.bet, result.raw_edge, result.q_fair
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EdgeResult:
    """Full breakdown of an edge computation."""
    p_calibrated: float       # calibrated model probability
    q_market: float           # market implied prob (raw, with overround)
    q_fair: float             # market implied prob after overround removal
    raw_edge: float           # p_calibrated - q_fair
    uncertainty: float        # model uncertainty (std)
    uncertainty_penalty: float  # lambda * uncertainty
    usable_edge: float        # raw_edge - uncertainty_penalty
    bet: bool                 # True if usable_edge > min_edge
    side: str                 # "A" or "none"


class EdgeEstimator:
    """
    Compute usable edge for a binary tennis match market.

    Parameters
    ----------
    uncertainty_lambda : float
        Penalty multiplier on uncertainty.  0 = ignore uncertainty.
        Higher values make the system more conservative under uncertainty.
        Reasonable range: 0.25 – 1.0.
    min_edge : float
        Minimum usable edge to trigger a bet recommendation.
        E.g. 0.03 means we need at least 3 percentage points of edge.
    fee_rate : float
        Commission / vig rate to subtract from edge.  E.g. 0.02 = 2 % fee.
    """

    def __init__(
        self,
        uncertainty_lambda: float = 0.5,
        min_edge: float = 0.03,
        fee_rate: float = 0.0,
    ):
        self.uncertainty_lambda = uncertainty_lambda
        self.min_edge = min_edge
        self.fee_rate = fee_rate

    # ------------------------------------------------------------------
    # Single-match edge estimate
    # ------------------------------------------------------------------
    def estimate(
        self,
        p_calibrated: float,
        decimal_odds_A: float,
        decimal_odds_B: Optional[float] = None,
        uncertainty: float = 0.0,
    ) -> EdgeResult:
        """
        Compute usable edge for a single match.

        Parameters
        ----------
        p_calibrated : float
            Calibrated model probability that Player A wins (in [0, 1]).
        decimal_odds_A : float
            Market decimal odds for Player A (e.g. 1.85 means bet 1, win 0.85).
        decimal_odds_B : float or None
            Market decimal odds for Player B.  Providing both enables
            overround-adjusted fair probability.  If None, the raw implied
            probability from decimal_odds_A is used without adjustment.
        uncertainty : float
            Model uncertainty estimate (e.g. MC Dropout std).

        Returns
        -------
        EdgeResult
        """
        # Raw market implied probability for A
        q_market = 1.0 / decimal_odds_A

        # Remove overround if both sides provided
        if decimal_odds_B is not None:
            q_B_market = 1.0 / decimal_odds_B
            overround = q_market + q_B_market  # should be > 1
            q_fair = q_market / overround
        else:
            q_fair = q_market

        # Raw edge before uncertainty and fees
        raw_edge = float(p_calibrated) - q_fair

        # Uncertainty penalty
        penalty = self.uncertainty_lambda * float(uncertainty)

        # Fee deduction (applied to gross edge)
        usable_edge = raw_edge - penalty - self.fee_rate

        bet = usable_edge > self.min_edge

        return EdgeResult(
            p_calibrated=float(p_calibrated),
            q_market=float(q_market),
            q_fair=float(q_fair),
            raw_edge=float(raw_edge),
            uncertainty=float(uncertainty),
            uncertainty_penalty=float(penalty),
            usable_edge=float(usable_edge),
            bet=bool(bet),
            side="A" if bet else "none",
        )

    # ------------------------------------------------------------------
    # Batch version
    # ------------------------------------------------------------------
    def estimate_batch(
        self,
        p_calibrated: np.ndarray,
        decimal_odds_A: np.ndarray,
        decimal_odds_B: Optional[np.ndarray] = None,
        uncertainty: Optional[np.ndarray] = None,
    ) -> list:
        """
        Vectorised version – returns a list of EdgeResult objects.

        Parameters
        ----------
        p_calibrated : array-like of shape (n,)
        decimal_odds_A : array-like of shape (n,)
        decimal_odds_B : array-like of shape (n,) or None
        uncertainty : array-like of shape (n,) or None

        Returns
        -------
        list of EdgeResult
        """
        p_arr = np.asarray(p_calibrated, dtype=float).ravel()
        odds_A = np.asarray(decimal_odds_A, dtype=float).ravel()
        n = len(p_arr)

        odds_B = (
            np.asarray(decimal_odds_B, dtype=float).ravel()
            if decimal_odds_B is not None
            else np.full(n, np.nan)
        )
        unc = (
            np.asarray(uncertainty, dtype=float).ravel()
            if uncertainty is not None
            else np.zeros(n)
        )

        results = []
        for i in range(n):
            ob = None if np.isnan(odds_B[i]) else float(odds_B[i])
            results.append(
                self.estimate(
                    p_calibrated=float(p_arr[i]),
                    decimal_odds_A=float(odds_A[i]),
                    decimal_odds_B=ob,
                    uncertainty=float(unc[i]),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Utility: convert moneyline / fractional odds
    # ------------------------------------------------------------------
    @staticmethod
    def moneyline_to_decimal(moneyline: float) -> float:
        """Convert US moneyline to decimal odds."""
        if moneyline > 0:
            return (moneyline / 100.0) + 1.0
        else:
            return (100.0 / abs(moneyline)) + 1.0

    @staticmethod
    def fractional_to_decimal(numerator: float, denominator: float) -> float:
        """Convert fractional odds (e.g. 3/2) to decimal."""
        return (numerator / denominator) + 1.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def edge_summary(self, results: list) -> dict:
        """Summarise a batch of EdgeResult objects."""
        usable = np.array([r.usable_edge for r in results])
        bets = [r for r in results if r.bet]
        return {
            "total_matches": len(results),
            "bet_count": len(bets),
            "bet_rate": len(bets) / max(len(results), 1),
            "mean_usable_edge": float(usable.mean()),
            "mean_edge_when_betting": float(
                np.mean([r.usable_edge for r in bets]) if bets else 0.0
            ),
            "max_usable_edge": float(usable.max()),
        }
