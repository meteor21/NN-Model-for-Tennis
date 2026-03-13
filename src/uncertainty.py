"""
Uncertainty Estimation Module
================================
Estimates epistemic (model) uncertainty via Monte Carlo Dropout.

During training, Dropout randomly zeroes neurons.  Normally it is disabled
at inference time.  MC Dropout keeps it *on* at inference time and runs the
forward pass N times.  The variance across those N passes measures how
uncertain the model is about that particular input.

High variance  →  model is uncertain  →  shrink bet or skip.
Low variance   →  model is confident  →  proceed with normal sizing.

Usage
-----
    from src.uncertainty import MCDropoutEstimator

    estimator = MCDropoutEstimator(model, n_samples=100)

    # For the embedding model (two extra player-ID inputs):
    mean_prob, std_prob = estimator.predict(
        [X_test, A_test, B_test],
        has_player_ids=True
    )

    # For the simple dense model (single feature array):
    mean_prob, std_prob = estimator.predict(X_test)
"""

import numpy as np
import tensorflow as tf
from typing import Union, Tuple, List


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimator for Keras models.

    Parameters
    ----------
    model : tf.keras.Model
        A trained Keras model that contains Dropout layers.
    n_samples : int
        Number of stochastic forward passes.  More samples = more accurate
        uncertainty estimate but slower.  50–200 is a good range.
    batch_size : int
        Mini-batch size for inference (avoids OOM on large test sets).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        n_samples: int = 100,
        batch_size: int = 512,
    ):
        self.model = model
        self.n_samples = n_samples
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        has_player_ids: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run N stochastic forward passes and return mean + std of predictions.

        Parameters
        ----------
        inputs : np.ndarray or list of np.ndarray
            Model inputs.  Pass a list [X, A_ids, B_ids] when using the
            player-embedding model; pass a single array for the dense model.
        has_player_ids : bool
            Set True when passing [X, A_ids, B_ids] (embedding model).

        Returns
        -------
        mean_prob : np.ndarray of shape (n,)
            Mean probability across MC samples.
        std_prob  : np.ndarray of shape (n,)
            Standard deviation – use as uncertainty proxy.
        """
        # Determine number of samples in inputs
        if has_player_ids or isinstance(inputs, (list, tuple)):
            n = len(inputs[0])
        else:
            n = len(inputs)

        all_preds = np.zeros((self.n_samples, n), dtype=np.float32)

        for i in range(self.n_samples):
            preds = self._forward_pass_with_dropout(inputs, has_player_ids)
            all_preds[i] = preds.ravel()

        mean_prob = all_preds.mean(axis=0)
        std_prob = all_preds.std(axis=0)
        return mean_prob, std_prob

    def _forward_pass_with_dropout(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        has_player_ids: bool,
    ) -> np.ndarray:
        """
        Single forward pass with dropout *enabled* (training=True).

        Keras Dropout uses training=True to stay active; this is the key
        trick that makes MC Dropout work at inference time.
        """
        if has_player_ids or isinstance(inputs, (list, tuple)):
            # Batched inference across player-embedding model inputs
            X_feat, A_ids, B_ids = inputs[0], inputs[1], inputs[2]
            n = len(X_feat)
            results = []
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch_inputs = [
                    X_feat[start:end],
                    A_ids[start:end],
                    B_ids[start:end],
                ]
                out = self.model(batch_inputs, training=True)  # dropout ON
                results.append(out.numpy())
            return np.concatenate(results, axis=0)
        else:
            n = len(inputs)
            results = []
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                out = self.model(inputs[start:end], training=True)
                results.append(out.numpy())
            return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Confidence multiplier
    # ------------------------------------------------------------------
    def confidence_multiplier(
        self,
        std_prob: np.ndarray,
        low_std: float = 0.03,
        high_std: float = 0.12,
    ) -> np.ndarray:
        """
        Convert std to a [0, 1] multiplier for bet sizing.

        - std ≤ low_std  → multiplier = 1.0  (full confidence)
        - std ≥ high_std → multiplier = 0.0  (no bet)
        - linear ramp in between

        Parameters
        ----------
        std_prob : np.ndarray
            Uncertainty (std) per match.
        low_std : float
            Threshold below which we treat the model as fully confident.
        high_std : float
            Threshold above which we treat uncertainty as too high to bet.

        Returns
        -------
        multiplier : np.ndarray in [0, 1]
        """
        std_prob = np.asarray(std_prob)
        multiplier = 1.0 - (std_prob - low_std) / (high_std - low_std)
        return np.clip(multiplier, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    def uncertainty_summary(self, std_prob: np.ndarray) -> dict:
        """Return basic stats on uncertainty distribution."""
        return {
            "mean_std": float(np.mean(std_prob)),
            "median_std": float(np.median(std_prob)),
            "p90_std": float(np.percentile(std_prob, 90)),
            "pct_high_uncertainty": float(np.mean(std_prob > 0.10)),
        }
