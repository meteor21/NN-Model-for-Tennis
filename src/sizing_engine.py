"""
Sizing Engine Module
=====================
Converts a usable edge into a concrete stake (fraction of bankroll)
using fractional Kelly criterion with hard policy caps.

The Kelly criterion maximises expected log-wealth growth.  Full Kelly is too
aggressive in practice because our probabilities are estimated and noisy.
We use fractional Kelly (e.g. 0.1 – 0.25) and add hard caps.

Stake formula
-------------
    kelly_fraction = edge / (decimal_odds - 1)
    raw_stake      = kelly_coeff * kelly_fraction
    confidence_adj = raw_stake * confidence_multiplier   # from MC Dropout
    ood_adj        = confidence_adj * ood_shrinkage      # from OOD detector
    final_stake    = min(ood_adj, max_per_bet_fraction)  # hard cap

Policy defaults (conservative)
-------------------------------
    kelly_coeff           = 0.25   (quarter Kelly)
    max_per_bet_fraction  = 0.010  (1 % of bankroll max per bet)
    max_per_match         = 0.030  (3 % across all bets on one match)
    max_daily_loss_frac   = 0.050  (stop for the day if down 5 %)
    min_edge              = 0.030  (3 pp edge floor)

Usage
-----
    from src.sizing_engine import SizingEngine, BettingPolicy

    policy = BettingPolicy(kelly_coeff=0.25, max_per_bet_fraction=0.01)
    engine = SizingEngine(policy=policy)

    decision = engine.size(
        usable_edge=0.06,
        decimal_odds=1.80,
        confidence_multiplier=0.85,
        ood_shrinkage=1.0,
        current_match_exposure=0.0,
        bankroll=10000.0,
    )
    # decision.stake_fraction, decision.stake_dollars, decision.action
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

@dataclass
class BettingPolicy:
    """
    All configurable limits for the sizing engine.

    Attributes
    ----------
    kelly_coeff : float
        Fraction of full Kelly to apply.  0.25 is a common conservative choice.
    max_per_bet_fraction : float
        Maximum fraction of bankroll on a single bet (hard cap).
    max_per_match_fraction : float
        Maximum total exposure across all bets on the same match.
    max_daily_loss_fraction : float
        Halt betting for the day if cumulative loss exceeds this fraction.
    min_edge : float
        Minimum usable edge to place any bet.  Overrides the EdgeEstimator
        threshold at the sizing layer as an extra safety check.
    min_confidence_multiplier : float
        Do not bet if MC Dropout confidence multiplier is below this threshold.
    min_ood_shrinkage : float
        Do not bet if OOD shrinkage factor is below this threshold
        (i.e. state is too unfamiliar).
    """
    kelly_coeff: float = 0.25
    max_per_bet_fraction: float = 0.010
    max_per_match_fraction: float = 0.030
    max_daily_loss_fraction: float = 0.050
    min_edge: float = 0.030
    min_confidence_multiplier: float = 0.30
    min_ood_shrinkage: float = 0.20


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass
class SizingDecision:
    """Full breakdown of a sizing decision."""
    action: str               # "BET" or "NO_BET"
    reason: str               # why action was taken
    stake_fraction: float     # fraction of bankroll to wager
    stake_dollars: float      # concrete dollar amount (if bankroll provided)
    kelly_fraction: float     # full Kelly fraction (before scaling)
    raw_kelly_stake: float    # kelly_coeff * kelly_fraction (before caps)
    confidence_adj: float     # after confidence multiplier
    ood_adj: float            # after OOD shrinkage
    usable_edge: float
    decimal_odds: float


# ---------------------------------------------------------------------------
# Sizing engine
# ---------------------------------------------------------------------------

class SizingEngine:
    """
    Converts usable edge + contextual multipliers into a concrete stake.

    Parameters
    ----------
    policy : BettingPolicy
        Risk management parameters.
    """

    def __init__(self, policy: Optional[BettingPolicy] = None):
        self.policy = policy or BettingPolicy()

    def size(
        self,
        usable_edge: float,
        decimal_odds: float,
        confidence_multiplier: float = 1.0,
        ood_shrinkage: float = 1.0,
        current_match_exposure: float = 0.0,
        bankroll: float = 10_000.0,
        daily_loss_so_far: float = 0.0,
    ) -> SizingDecision:
        """
        Compute stake for a single bet.

        Parameters
        ----------
        usable_edge : float
            Edge from EdgeEstimator (after calibration + uncertainty penalty).
        decimal_odds : float
            Decimal market odds for the side we want to bet.
        confidence_multiplier : float in [0, 1]
            From MCDropoutEstimator.confidence_multiplier().  1 = full size.
        ood_shrinkage : float in [0, 1]
            From OOD detector.shrinkage().  1 = in-distribution, 0 = skip.
        current_match_exposure : float
            Fraction of bankroll already committed to this match (for cap).
        bankroll : float
            Current total bankroll in dollars.
        daily_loss_so_far : float
            Cumulative loss today in dollars (for daily stop-loss).

        Returns
        -------
        SizingDecision
        """
        p = self.policy

        # ---- Guard: daily stop-loss ----------------------------------------
        if daily_loss_so_far / max(bankroll, 1.0) >= p.max_daily_loss_fraction:
            return self._no_bet(
                "DAILY_STOP_LOSS",
                usable_edge=usable_edge,
                decimal_odds=decimal_odds,
            )

        # ---- Guard: minimum edge -------------------------------------------
        if usable_edge <= p.min_edge:
            return self._no_bet(
                "EDGE_TOO_SMALL",
                usable_edge=usable_edge,
                decimal_odds=decimal_odds,
            )

        # ---- Guard: confidence ---------------------------------------------
        if confidence_multiplier < p.min_confidence_multiplier:
            return self._no_bet(
                "LOW_CONFIDENCE",
                usable_edge=usable_edge,
                decimal_odds=decimal_odds,
            )

        # ---- Guard: OOD ----------------------------------------------------
        if ood_shrinkage < p.min_ood_shrinkage:
            return self._no_bet(
                "OOD_STATE",
                usable_edge=usable_edge,
                decimal_odds=decimal_odds,
            )

        # ---- Kelly formula -------------------------------------------------
        # For a binary bet with decimal odds d (pays d-1 per unit staked):
        #   Kelly fraction = edge / (d - 1)
        net_odds = decimal_odds - 1.0
        if net_odds <= 0:
            return self._no_bet("INVALID_ODDS", usable_edge=usable_edge, decimal_odds=decimal_odds)

        kelly_fraction = usable_edge / net_odds
        kelly_fraction = max(0.0, kelly_fraction)  # floor at 0

        # ---- Apply fractional Kelly ----------------------------------------
        raw_kelly_stake = p.kelly_coeff * kelly_fraction

        # ---- Apply confidence multiplier -----------------------------------
        confidence_adj = raw_kelly_stake * confidence_multiplier

        # ---- Apply OOD shrinkage -------------------------------------------
        ood_adj = confidence_adj * ood_shrinkage

        # ---- Hard cap per bet ----------------------------------------------
        capped = min(ood_adj, p.max_per_bet_fraction)

        # ---- Hard cap per match --------------------------------------------
        remaining_match_budget = max(
            p.max_per_match_fraction - current_match_exposure, 0.0
        )
        final_fraction = min(capped, remaining_match_budget)

        if final_fraction <= 0:
            return self._no_bet(
                "MATCH_EXPOSURE_EXCEEDED",
                usable_edge=usable_edge,
                decimal_odds=decimal_odds,
            )

        stake_dollars = final_fraction * bankroll

        return SizingDecision(
            action="BET",
            reason="OK",
            stake_fraction=float(final_fraction),
            stake_dollars=float(stake_dollars),
            kelly_fraction=float(kelly_fraction),
            raw_kelly_stake=float(raw_kelly_stake),
            confidence_adj=float(confidence_adj),
            ood_adj=float(ood_adj),
            usable_edge=float(usable_edge),
            decimal_odds=float(decimal_odds),
        )

    def size_batch(
        self,
        usable_edges: np.ndarray,
        decimal_odds: np.ndarray,
        confidence_multipliers: Optional[np.ndarray] = None,
        ood_shrinkages: Optional[np.ndarray] = None,
        bankroll: float = 10_000.0,
        daily_loss_so_far: float = 0.0,
    ) -> list:
        """
        Size a batch of bets independently (no cross-match aggregation).

        Returns list of SizingDecision.
        """
        n = len(usable_edges)
        conf = confidence_multipliers if confidence_multipliers is not None else np.ones(n)
        ood = ood_shrinkages if ood_shrinkages is not None else np.ones(n)

        decisions = []
        for i in range(n):
            d = self.size(
                usable_edge=float(usable_edges[i]),
                decimal_odds=float(decimal_odds[i]),
                confidence_multiplier=float(conf[i]),
                ood_shrinkage=float(ood[i]),
                bankroll=bankroll,
                daily_loss_so_far=daily_loss_so_far,
            )
            decisions.append(d)
        return decisions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_bet(reason: str, usable_edge: float, decimal_odds: float) -> SizingDecision:
        return SizingDecision(
            action="NO_BET",
            reason=reason,
            stake_fraction=0.0,
            stake_dollars=0.0,
            kelly_fraction=0.0,
            raw_kelly_stake=0.0,
            confidence_adj=0.0,
            ood_adj=0.0,
            usable_edge=usable_edge,
            decimal_odds=decimal_odds,
        )

    def decision_summary(self, decisions: list) -> dict:
        """Summarise a batch of SizingDecision objects."""
        bets = [d for d in decisions if d.action == "BET"]
        reasons = {}
        for d in decisions:
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
        return {
            "total": len(decisions),
            "bets": len(bets),
            "bet_rate": len(bets) / max(len(decisions), 1),
            "mean_stake_pct": float(
                np.mean([d.stake_fraction * 100 for d in bets]) if bets else 0.0
            ),
            "total_exposure_pct": float(
                np.sum([d.stake_fraction * 100 for d in bets])
            ),
            "reasons": reasons,
        }
