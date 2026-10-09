"""Unit tests for Temporal Order Shuffling Leakage Sanity Check."""

import unittest
import numpy as np

from src.models.lstm_world_model import NumpyLSTMWorldModel, StateDistribution
from src.models.leakage_checker import (
    check_temporal_leakage_by_shuffling,
    TemporalOrderShuffleResult,
)


class MockOrderInvariantModel:
    """Mock model that ignores temporal sequence ordering (e.g., leaky or static)."""
    def predict_next_state(self, x_seq: np.ndarray) -> StateDistribution:
        # Predicts using the unordered time-average of past states
        mean_pred = np.mean(x_seq, axis=1)
        log_var_pred = np.zeros_like(mean_pred)
        return StateDistribution(mean=mean_pred, log_var=log_var_pred)


class TestTemporalLeakageSanityCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.num_samples = 40
        cls.seq_len = 6
        cls.num_features = 8

        # Create temporal sequence where S_{t+1} depends strongly on the immediate past S_t
        # and has a directional trend: S_{t} = S_{t-1} + delta
        x_data = np.zeros((cls.num_samples, cls.seq_len, cls.num_features), dtype=np.float32)
        y_data = np.zeros((cls.num_samples, cls.num_features), dtype=np.float32)

        for i in range(cls.num_samples):
            trend = np.linspace(0.1, 1.0, cls.seq_len)[:, None] * np.ones((cls.seq_len, cls.num_features))
            noise = np.random.randn(cls.seq_len, cls.num_features) * 0.05
            seq = trend + noise
            x_data[i] = seq
            # Next state continues the trend:
            y_data[i] = seq[-1] + 0.15

        cls.x_seq = x_data
        cls.y_next = y_data

    def test_warning_triggered_for_order_invariant_leaky_surrogate(self):
        """Verify that a model invariant to time order triggers the TEMPORAL LEAKAGE WARNING."""
        leaky_model = MockOrderInvariantModel()

        # Run sanity check
        res = check_temporal_leakage_by_shuffling(
            model=leaky_model,
            x_history=self.x_seq,
            y_future=self.y_next,
            min_relative_drop=0.15,
            n_permutations=5,
            random_state=42,
            verbose=False,
        )

        self.assertIsInstance(res, TemporalOrderShuffleResult)
        # For an order-invariant model (mean across time), shuffled MSE is identical to original MSE
        self.assertAlmostEqual(res.original_mse, res.shuffled_mse, places=5)
        self.assertAlmostEqual(res.relative_mse_increase, 0.0, places=4)
        self.assertTrue(res.warning_triggered)
        self.assertFalse(res.meaningful_drop_detected)

        # Check that the diagnostic summary table contains the warning banner
        summary = res.summary_table()
        self.assertIn("CRITICAL TEMPORAL LEAKAGE WARNING", summary)
        self.assertIn("Temporal Lookahead Leakage", summary)
        self.assertIn("WARNING: LEAKAGE SUSPECTED", summary)

    def test_world_model_degradation_on_shuffled_temporal_order(self):
        """Verify that an autoregressive sequence model degrades when temporal order is scrambled."""
        from src.models.trainer import train_world_model

        model = NumpyLSTMWorldModel(
            input_dim=self.num_features,
            hidden_dim=32,
            seed=42,
        )

        train_world_model(
            model=model,
            train_x=self.x_seq,
            train_y=self.y_next,
            num_epochs=15,
            batch_size=8,
            lr=0.01,
            verbose=False,
        )

        res = check_temporal_leakage_by_shuffling(
            model=model,
            x_history=self.x_seq,
            y_future=self.y_next,
            min_relative_drop=0.05,
            n_permutations=5,
            random_state=42,
            verbose=False,
        )

        self.assertIsInstance(res, TemporalOrderShuffleResult)
        # Error must increase when chronological order is scrambled
        self.assertGreater(res.shuffled_mse, res.original_mse)
        self.assertGreaterEqual(res.relative_mse_increase, 0.05)
        self.assertFalse(res.warning_triggered)
        self.assertTrue(res.meaningful_drop_detected)

        summary = res.summary_table()
        self.assertIn("PASSED: CAUSAL DYNAMICS VERIFIED", summary)
        self.assertIn("Zero upstream temporal leakage detected", summary)

    def test_invalid_dimensions_raise_error(self):
        """Verify input dimensional assertions."""
        model = MockOrderInvariantModel()
        # 2D input should fail
        with self.assertRaises(ValueError):
            check_temporal_leakage_by_shuffling(model, np.zeros((10, 5)), np.zeros(10))
        # Sequence length 1 should fail (cannot shuffle length 1)
        with self.assertRaises(ValueError):
            check_temporal_leakage_by_shuffling(model, np.zeros((10, 1, 5)), np.zeros((10, 5)))


if __name__ == "__main__":
    unittest.main()
