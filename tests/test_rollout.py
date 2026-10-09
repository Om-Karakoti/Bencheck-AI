"""Unit tests for K-step autoregressive rollout and time-series infiltration scoring."""

import unittest
import numpy as np

from src.models.lstm_world_model import NumpyLSTMWorldModel
from src.models.forecaster import rollout_and_score, RolloutForecaster, RolloutForecastResult
from src.data.causal_windowing import NETWORK_STATE_FEATURES


class TestRolloutAndScoring(unittest.TestCase):
    def setUp(self):
        self.input_dim = len(NETWORK_STATE_FEATURES)
        self.hidden_dim = 32
        self.history_len = 6
        self.model = NumpyLSTMWorldModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            seed=42,
        )

        np.random.seed(42)
        self.history = np.random.randn(self.history_len, self.input_dim).astype(np.float32)

    def test_rollout_k_steps_structure(self):
        """Verify K-step rollout produces exactly K future states and scores."""
        k = 5
        curr_t = 120.0
        dt = 30.0

        res = rollout_and_score(
            model=self.model,
            history_states=self.history,
            k_steps=k,
            current_timestamp=curr_t,
            time_step_sec=dt,
        )

        self.assertIsInstance(res, RolloutForecastResult)
        self.assertEqual(res.horizon_k, k)

        # Shapes verification
        self.assertEqual(res.predicted_states.shape, (k, self.input_dim))
        self.assertEqual(res.predicted_variances.shape, (k, self.input_dim))
        self.assertEqual(res.infiltration_scores.shape, (k,))
        self.assertEqual(len(res.threat_levels), k)
        self.assertEqual(len(res.dominant_stages), k)
        self.assertEqual(len(res.timestamps), k)

        # Time series timestamps
        expected_times = [150.0, 180.0, 210.0, 240.0, 270.0]
        np.testing.assert_allclose(res.timestamps, expected_times)

        # Infiltration scores must be valid probabilities [0, 1]
        for score in res.infiltration_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_rollout_to_dataframe(self):
        """Verify export to pandas DataFrame."""
        k = 4
        res = rollout_and_score(
            model=self.model,
            history_states=self.history,
            k_steps=k,
            current_timestamp=0.0,
        )

        df = res.to_dataframe()
        self.assertEqual(len(df), k)
        self.assertIn("infiltration_probability", df.columns)
        self.assertIn("threat_level", df.columns)
        self.assertIn("predicted_stage", df.columns)
        self.assertIn("mean_state_uncertainty", df.columns)

    def test_rollout_forecaster_wrapper(self):
        """Test the stateful RolloutForecaster class."""
        k = 3
        forecaster = RolloutForecaster(
            model=self.model,
            k_steps=k,
            time_step_sec=30.0,
        )

        res = forecaster.forecast(self.history, current_timestamp=90.0)
        self.assertEqual(res.horizon_k, k)
        self.assertEqual(res.timestamps[0], 120.0)
        self.assertEqual(res.timestamps[-1], 180.0)

    def test_invalid_k_raises_error(self):
        """Verify non-positive K raises ValueError."""
        with self.assertRaises(ValueError):
            rollout_and_score(self.model, self.history, k_steps=0)
        with self.assertRaises(ValueError):
            rollout_and_score(self.model, self.history, k_steps=-3)


if __name__ == "__main__":
    unittest.main()
