"""
Tennis Betting Pipeline – source package.

Modules
-------
calibration   : Post-hoc probability calibration (isotonic / Platt)
uncertainty   : MC Dropout epistemic uncertainty estimation
ood_detector  : Out-of-distribution detection + hierarchical fallback prior
edge_estimator: Usable edge = calibrated_prob - market_price - uncertainty_penalty
sizing_engine : Fractional Kelly stake sizing with hard policy caps
pipeline      : End-to-end TennisBettingPipeline orchestrator
"""

from src.calibration import ProbabilityCalibrator
from src.uncertainty import MCDropoutEstimator
from src.ood_detector import MahalanobisOOD, IsolationForestOOD, HierarchicalPrior
from src.edge_estimator import EdgeEstimator, EdgeResult
from src.sizing_engine import SizingEngine, SizingDecision, BettingPolicy
from src.pipeline import TennisBettingPipeline, PipelineConfig, PipelineDecision

__all__ = [
    "ProbabilityCalibrator",
    "MCDropoutEstimator",
    "MahalanobisOOD",
    "IsolationForestOOD",
    "HierarchicalPrior",
    "EdgeEstimator",
    "EdgeResult",
    "SizingEngine",
    "SizingDecision",
    "BettingPolicy",
    "TennisBettingPipeline",
    "PipelineConfig",
    "PipelineDecision",
]
