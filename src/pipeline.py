"""
Betting Pipeline Orchestrator
================================
Ties together all modules into a single, event-driven decision pipeline:

    NN probability
        ↓
    Calibration          (ProbabilityCalibrator)
        ↓
    MC Dropout           (MCDropoutEstimator)  → confidence_multiplier
        ↓
    OOD Detection        (MahalanobisOOD)      → ood_shrinkage + hierarchical blend
        ↓
    Edge Estimation      (EdgeEstimator)       → usable_edge
        ↓
    Sizing Engine        (SizingEngine)        → stake_fraction / stake_dollars

The pipeline separates a SLOW layer (computed once per session / tournament)
from a FAST layer (computed per match event):

    SLOW (offline / session start):
        - calibrator.fit(calib_probs, calib_labels)
        - ood_detector.fit(X_train)

    FAST (per match):
        - calibrator.transform(raw_prob)
        - mc_estimator.predict(inputs)      → mean_prob, std_prob
        - ood_detector.shrinkage(X_live)    → ood_factor
        - edge_estimator.estimate(...)      → EdgeResult
        - sizing_engine.size(...)           → SizingDecision

Usage
-----
    from src.pipeline import TennisBettingPipeline, PipelineConfig

    config = PipelineConfig(
        kelly_coeff=0.25,
        max_per_bet_fraction=0.01,
        uncertainty_lambda=0.5,
        min_edge=0.03,
    )
    pipeline = TennisBettingPipeline(model, config)

    # One-time offline fit
    pipeline.fit(X_train, y_train, calib_probs_val, y_val)

    # Per-match decision
    decision = pipeline.decide(
        X_live=X_match,
        player_A_id=A_id,
        player_B_id=B_id,
        decimal_odds_A=1.85,
        decimal_odds_B=2.10,
    )
    print(decision.sizing.action, decision.sizing.stake_dollars)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

from src.calibration import ProbabilityCalibrator
from src.uncertainty import MCDropoutEstimator
from src.ood_detector import MahalanobisOOD, HierarchicalPrior
from src.edge_estimator import EdgeEstimator, EdgeResult
from src.sizing_engine import SizingEngine, SizingDecision, BettingPolicy


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Centralised configuration for the full pipeline."""

    # Calibration
    calibration_method: str = "isotonic"   # "isotonic" or "platt"

    # Uncertainty (MC Dropout)
    mc_samples: int = 100
    mc_low_std: float = 0.03
    mc_high_std: float = 0.12

    # OOD
    ood_threshold_pct: float = 95.0
    ood_shrink_min: float = 0.0

    # Edge
    uncertainty_lambda: float = 0.5
    min_edge: float = 0.03
    fee_rate: float = 0.0

    # Sizing / Kelly
    kelly_coeff: float = 0.25
    max_per_bet_fraction: float = 0.010
    max_per_match_fraction: float = 0.030
    max_daily_loss_fraction: float = 0.050
    min_confidence_multiplier: float = 0.30
    min_ood_shrinkage: float = 0.20

    # Hierarchical prior surface win rates
    surface_priors: dict = field(default_factory=lambda: {
        "Clay": 0.50,
        "Grass": 0.50,
        "Hard_outdoor": 0.50,
        "Hard_indoor": 0.50,
    })
    overall_prior: float = 0.50


# ---------------------------------------------------------------------------
# Full pipeline decision output
# ---------------------------------------------------------------------------

@dataclass
class PipelineDecision:
    """Complete decision record for a single match."""
    raw_prob: float
    calibrated_prob: float
    mc_mean_prob: float
    mc_std: float
    confidence_multiplier: float
    ood_score: float
    ood_shrinkage: float
    is_ood: bool
    blended_prob: float
    edge: EdgeResult
    sizing: SizingDecision
    match_meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class TennisBettingPipeline:
    """
    End-to-end betting decision pipeline for the tennis NN model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model (with player embeddings or dense-only).
    config : PipelineConfig
        All pipeline hyperparameters.
    has_player_ids : bool
        True when using the embedding model (inputs are [X, A_id, B_id]).
    """

    def __init__(
        self,
        model,
        config: Optional[PipelineConfig] = None,
        has_player_ids: bool = True,
    ):
        self.model = model
        self.config = config or PipelineConfig()
        self.has_player_ids = has_player_ids

        cfg = self.config

        self.calibrator = ProbabilityCalibrator(method=cfg.calibration_method)
        self.mc_estimator = MCDropoutEstimator(model, n_samples=cfg.mc_samples)
        self.ood_detector = MahalanobisOOD(
            threshold_pct=cfg.ood_threshold_pct,
            shrink_min=cfg.ood_shrink_min,
        )
        self.hier_prior = HierarchicalPrior()
        self.edge_estimator = EdgeEstimator(
            uncertainty_lambda=cfg.uncertainty_lambda,
            min_edge=cfg.min_edge,
            fee_rate=cfg.fee_rate,
        )
        policy = BettingPolicy(
            kelly_coeff=cfg.kelly_coeff,
            max_per_bet_fraction=cfg.max_per_bet_fraction,
            max_per_match_fraction=cfg.max_per_match_fraction,
            max_daily_loss_fraction=cfg.max_daily_loss_fraction,
            min_edge=cfg.min_edge,
            min_confidence_multiplier=cfg.min_confidence_multiplier,
            min_ood_shrinkage=cfg.min_ood_shrinkage,
        )
        self.sizing_engine = SizingEngine(policy=policy)

        self._is_fitted = False

    # ------------------------------------------------------------------
    # SLOW layer: offline fit (run once per training session)
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        raw_probs_calib: np.ndarray,
        y_calib: np.ndarray,
        A_train: Optional[np.ndarray] = None,
        B_train: Optional[np.ndarray] = None,
    ) -> "TennisBettingPipeline":
        """
        Fit all offline components.

        Parameters
        ----------
        X_train : np.ndarray
            Scaled + imputed feature matrix used for OOD detector fitting.
        y_train : np.ndarray
            Match outcomes (0/1) for X_train.
        raw_probs_calib : np.ndarray
            Raw NN probabilities on a held-out calibration set.
        y_calib : np.ndarray
            True outcomes for the calibration set.
        A_train, B_train : np.ndarray or None
            Player ID arrays for training set (needed only to build surface priors).
        """
        cfg = self.config

        # 1. Fit calibrator
        self.calibrator.fit(raw_probs_calib, y_calib)

        # 2. Fit OOD detector on training features
        self.ood_detector.fit(X_train)

        # 3. Fit hierarchical prior
        self.hier_prior.fit(
            surface_priors=cfg.surface_priors,
            overall_prior=cfg.overall_prior,
        )

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # FAST layer: per-match decision
    # ------------------------------------------------------------------

    def decide(
        self,
        X_live: np.ndarray,
        decimal_odds_A: float,
        decimal_odds_B: Optional[float] = None,
        player_A_id: Optional[int] = None,
        player_B_id: Optional[int] = None,
        surface: Optional[str] = None,
        current_match_exposure: float = 0.0,
        bankroll: float = 10_000.0,
        daily_loss_so_far: float = 0.0,
        match_meta: Optional[dict] = None,
    ) -> PipelineDecision:
        """
        Full pipeline for a single live match.

        Parameters
        ----------
        X_live : np.ndarray of shape (1, n_features)
            Scaled + imputed feature vector for this match.
        decimal_odds_A : float
            Market decimal odds for Player A.
        decimal_odds_B : float or None
            Market decimal odds for Player B (for overround removal).
        player_A_id, player_B_id : int or None
            Player IDs for embedding model.  Required if has_player_ids=True.
        surface : str or None
            Surface name (for hierarchical prior fallback).
        current_match_exposure : float
            Fraction of bankroll already on this match.
        bankroll : float
            Current bankroll in dollars.
        daily_loss_so_far : float
            Cumulative loss today in dollars.
        match_meta : dict or None
            Arbitrary metadata stored in the decision record (e.g. player names).

        Returns
        -------
        PipelineDecision
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before decide().")

        X = np.asarray(X_live, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # ---- Step 1: Raw NN probability ------------------------------------
        if self.has_player_ids and player_A_id is not None:
            A = np.array([player_A_id])
            B = np.array([player_B_id])
            inputs = [X, A, B]
        else:
            inputs = X

        raw_prob = float(self.model.predict(inputs, verbose=0).ravel()[0])

        # ---- Step 2: Calibration -------------------------------------------
        calibrated_prob = float(self.calibrator.transform(np.array([raw_prob]))[0])

        # ---- Step 3: MC Dropout uncertainty --------------------------------
        mc_mean, mc_std_arr = self.mc_estimator.predict(inputs, has_player_ids=self.has_player_ids and player_A_id is not None)
        mc_mean_prob = float(mc_mean[0])
        mc_std = float(mc_std_arr[0])
        conf_mult = float(self.mc_estimator.confidence_multiplier(
            mc_std_arr,
            low_std=self.config.mc_low_std,
            high_std=self.config.mc_high_std,
        )[0])

        # ---- Step 4: OOD detection + hierarchical blending -----------------
        ood_score_raw = float(self.ood_detector.score(X)[0])
        ood_flag = bool(self.ood_detector.is_ood(X)[0])
        ood_shrink = float(self.ood_detector.shrinkage(X)[0])

        # Normalise ood_score to [0,1] for blending
        ood_norm = 1.0 - ood_shrink  # 0 = in-dist, 1 = max OOD
        blended_prob = self.hier_prior.blend(
            p_model=calibrated_prob,
            ood_score=ood_norm,
            surface=surface,
            market_price=(1.0 / decimal_odds_A) if decimal_odds_A else None,
        )

        # ---- Step 5: Edge estimation ---------------------------------------
        edge_result = self.edge_estimator.estimate(
            p_calibrated=blended_prob,
            decimal_odds_A=decimal_odds_A,
            decimal_odds_B=decimal_odds_B,
            uncertainty=mc_std,
        )

        # ---- Step 6: Sizing ------------------------------------------------
        sizing_decision = self.sizing_engine.size(
            usable_edge=edge_result.usable_edge,
            decimal_odds=decimal_odds_A,
            confidence_multiplier=conf_mult,
            ood_shrinkage=ood_shrink,
            current_match_exposure=current_match_exposure,
            bankroll=bankroll,
            daily_loss_so_far=daily_loss_so_far,
        )

        return PipelineDecision(
            raw_prob=raw_prob,
            calibrated_prob=calibrated_prob,
            mc_mean_prob=mc_mean_prob,
            mc_std=mc_std,
            confidence_multiplier=conf_mult,
            ood_score=ood_score_raw,
            ood_shrinkage=ood_shrink,
            is_ood=ood_flag,
            blended_prob=blended_prob,
            edge=edge_result,
            sizing=sizing_decision,
            match_meta=match_meta or {},
        )

    # ------------------------------------------------------------------
    # Batch decision (backtest / simulation)
    # ------------------------------------------------------------------

    def decide_batch(
        self,
        X_batch: np.ndarray,
        decimal_odds_A: np.ndarray,
        decimal_odds_B: Optional[np.ndarray] = None,
        A_ids: Optional[np.ndarray] = None,
        B_ids: Optional[np.ndarray] = None,
        surfaces: Optional[list] = None,
        bankroll: float = 10_000.0,
    ) -> List[PipelineDecision]:
        """
        Run the full pipeline on a batch of matches (for backtesting).

        Returns a list of PipelineDecision objects.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before decide_batch().")

        n = len(X_batch)
        odds_B_arr = decimal_odds_B if decimal_odds_B is not None else [None] * n
        surf_arr = surfaces if surfaces is not None else [None] * n

        # Bulk MC Dropout (faster than one-at-a-time)
        if self.has_player_ids and A_ids is not None:
            inputs_bulk = [X_batch, A_ids, B_ids]
        else:
            inputs_bulk = X_batch

        mc_means, mc_stds = self.mc_estimator.predict(inputs_bulk, has_player_ids=(self.has_player_ids and A_ids is not None))
        conf_mults = self.mc_estimator.confidence_multiplier(mc_stds, self.config.mc_low_std, self.config.mc_high_std)
        ood_scores = self.ood_detector.score(X_batch)
        ood_flags = self.ood_detector.is_ood(X_batch)
        ood_shrinkages = self.ood_detector.shrinkage(X_batch)

        # Calibrate raw NN probs in bulk
        raw_probs = self.model.predict(inputs_bulk, verbose=0).ravel()
        calib_probs = self.calibrator.transform(raw_probs)

        decisions = []
        cumulative_loss = 0.0

        for i in range(n):
            ood_norm = 1.0 - float(ood_shrinkages[i])
            mkt_price = 1.0 / float(decimal_odds_A[i]) if decimal_odds_A[i] else None
            blended = self.hier_prior.blend(
                p_model=float(calib_probs[i]),
                ood_score=ood_norm,
                surface=surf_arr[i],
                market_price=mkt_price,
            )
            edge_result = self.edge_estimator.estimate(
                p_calibrated=blended,
                decimal_odds_A=float(decimal_odds_A[i]),
                decimal_odds_B=float(odds_B_arr[i]) if odds_B_arr[i] is not None else None,
                uncertainty=float(mc_stds[i]),
            )
            sizing = self.sizing_engine.size(
                usable_edge=edge_result.usable_edge,
                decimal_odds=float(decimal_odds_A[i]),
                confidence_multiplier=float(conf_mults[i]),
                ood_shrinkage=float(ood_shrinkages[i]),
                bankroll=bankroll,
                daily_loss_so_far=cumulative_loss,
            )
            decisions.append(PipelineDecision(
                raw_prob=float(raw_probs[i]),
                calibrated_prob=float(calib_probs[i]),
                mc_mean_prob=float(mc_means[i]),
                mc_std=float(mc_stds[i]),
                confidence_multiplier=float(conf_mults[i]),
                ood_score=float(ood_scores[i]),
                ood_shrinkage=float(ood_shrinkages[i]),
                is_ood=bool(ood_flags[i]),
                blended_prob=float(blended),
                edge=edge_result,
                sizing=sizing,
            ))
        return decisions

    # ------------------------------------------------------------------
    # Postmortem / logging
    # ------------------------------------------------------------------

    @staticmethod
    def log_decisions(decisions: List[PipelineDecision], outcomes: Optional[np.ndarray] = None) -> dict:
        """
        Aggregate statistics across a list of decisions.

        Parameters
        ----------
        decisions : list of PipelineDecision
        outcomes : np.ndarray of 0/1 (optional)
            Actual match outcomes for computing PnL.

        Returns
        -------
        dict with keys: bet_count, mean_edge, mean_stake_pct,
                        mean_calibrated_prob, mean_uncertainty,
                        pct_ood, pnl (if outcomes provided)
        """
        bets = [d for d in decisions if d.sizing.action == "BET"]
        n_total = len(decisions)
        stats = {
            "total_evaluated": n_total,
            "bet_count": len(bets),
            "bet_rate": len(bets) / max(n_total, 1),
            "mean_calibrated_prob": float(np.mean([d.calibrated_prob for d in decisions])),
            "mean_uncertainty": float(np.mean([d.mc_std for d in decisions])),
            "pct_ood": float(np.mean([d.is_ood for d in decisions])),
            "mean_edge_all": float(np.mean([d.edge.usable_edge for d in decisions])),
            "mean_edge_bets": float(np.mean([d.edge.usable_edge for d in bets]) if bets else 0.0),
            "mean_stake_pct": float(np.mean([d.sizing.stake_fraction * 100 for d in bets]) if bets else 0.0),
        }

        if outcomes is not None and len(bets) > 0:
            # Map bets back to outcome indices – requires tracking original index
            # Here we assume all decisions are returned in order with outcomes
            bet_indices = [i for i, d in enumerate(decisions) if d.sizing.action == "BET"]
            pnl = 0.0
            for idx in bet_indices:
                d = decisions[idx]
                won = int(outcomes[idx])
                stake = d.sizing.stake_fraction
                net_odds = d.sizing.decimal_odds - 1.0
                pnl += stake * net_odds * won - stake * (1 - won)
            stats["pnl_fraction"] = float(pnl)
            stats["roi"] = float(pnl / max(sum(decisions[i].sizing.stake_fraction for i in bet_indices), 1e-9))

        return stats
