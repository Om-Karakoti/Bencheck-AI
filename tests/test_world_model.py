"""Unit tests for LSTM World Model, distinct prediction & scoring functions, and training loop."""

import unittest
import numpy as np

from src.models.lstm_world_model import (
    NumpyLSTMWorldModel,
    StateDistribution,
    GaussianNLLLoss,
    create_world_model,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from src.models.infiltration_scorer import (
    derive_infiltration_score,
    InfiltrationAssessment,
)
from src.models.trainer import train_world_model, TrainingHistory
from src.data.causal_windowing import NETWORK_STATE_FEATURES


class TestWorldModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = len(NETWORK_STATE_FEATURES)  # 36
        self.hidden_dim = 32
        self.seq_len = 6
        self.batch_size = 8

        # Create model (NumPy reference guaranteed to run in all environments)
        self.model = NumpyLSTMWorldModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            seed=42,
        )

        # Synthetic history sequences: (B, L, D)
        np.random.seed(42)
        self.x_history = np.random.randn(self.batch_size, self.seq_len, self.input_dim).astype(np.float32)
        self.y_target = np.random.randn(self.batch_size, self.input_dim).astype(np.float32)

    def test_predict_next_state_distribution(self):
        """Verify model learns P(S_{t+1} | S_{<=t}) and outputs distribution parameters."""
        dist = self.model.predict_next_state(self.x_history)

        self.assertIsInstance(dist, StateDistribution)
        # Check shapes
        self.assertEqual(dist.mean.shape, (self.batch_size, self.input_dim))
        self.assertEqual(dist.log_var.shape, (self.batch_size, self.input_dim))
        self.assertEqual(dist.variance.shape, (self.batch_size, self.input_dim))

        # Variance must be strictly positive
        self.assertTrue(np.all(dist.variance > 0))

        # Test sampling
        sample = dist.sample()
        self.assertEqual(sample.shape, (self.batch_size, self.input_dim))

        # Test Gaussian NLL loss
        loss = GaussianNLLLoss(self.y_target, dist)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))

    def test_multi_step_rollout(self):
        """Test autoregressive future state rollout over horizon H."""
        horizon = 3
        forecasts = self.model.rollout_future_states(self.x_history, horizon=horizon)

        self.assertEqual(len(forecasts), horizon)
        for step_dist in forecasts:
            self.assertEqual(step_dist.mean.shape, (self.batch_size, self.input_dim))

    def test_two_distinct_callable_functions(self):
        """CRITICAL TEST: Ensure 'predict_next_state' and 'derive_infiltration_score'
        are two distinct, separately callable functions.
        
        The model must not collapse into an end-to-end classifier directly predicting attack_label.
        """
        # Step 1: Call predict_next_state independently
        pred_dist = self.model.predict_next_state(self.x_history[0:1])
        predicted_state = pred_dist.mode()[0]
        self.assertEqual(predicted_state.shape, (self.input_dim,))

        # Step 2: Call derive_infiltration_score independently on a benign state
        benign_state = np.zeros(self.input_dim)
        benign_assessment = derive_infiltration_score(benign_state)
        self.assertIsInstance(benign_assessment, InfiltrationAssessment)
        self.assertLess(benign_assessment.infiltration_score, 0.30)
        self.assertEqual(benign_assessment.threat_level, "Benign")

        # Step 3: Call derive_infiltration_score on a predicted attack state (Exfiltration)
        exfil_state = np.zeros(self.input_dim)
        # Inject high byte rate and high payload size
        byte_rate_idx = NETWORK_STATE_FEATURES.index("byte_rate")
        payload_idx = NETWORK_STATE_FEATURES.index("payload_size_max")
        exfil_state[byte_rate_idx] = 200000.0
        exfil_state[payload_idx] = 1400.0

        exfil_assessment = derive_infiltration_score(exfil_state)
        self.assertGreater(exfil_assessment.infiltration_score, 0.70)
        self.assertEqual(exfil_assessment.dominant_stage, "Exfiltration")
        self.assertIn(exfil_assessment.threat_level, ["Elevated", "High", "Critical"])

    def test_training_loop_with_loss_logging(self):
        """Verify train_world_model executes epochs, logs train/val loss, and records history."""
        num_epochs = 4
        train_x = self.x_history[:6]
        train_y = self.y_target[:6]
        val_x = self.x_history[6:]
        val_y = self.y_target[6:]

        history = train_world_model(
            model=self.model,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            num_epochs=num_epochs,
            batch_size=2,
            lr=0.01,
            verbose=False,
        )

        self.assertIsInstance(history, TrainingHistory)
        self.assertEqual(len(history.epochs), num_epochs)
        self.assertEqual(len(history.train_losses), num_epochs)
        self.assertEqual(len(history.val_losses), num_epochs)

        # Check all losses are finite
        for loss in history.train_losses:
            self.assertFalse(np.isnan(loss))
            self.assertFalse(np.isinf(loss))
        for loss in history.val_losses:
            self.assertFalse(np.isnan(loss))
            self.assertFalse(np.isinf(loss))

        self.assertGreater(history.best_epoch, 0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional and not installed")
    def test_pytorch_lstm_world_model_training(self):
        """Verify PyTorch LSTM World Model trains with autograd and Gaussian NLL."""
        import torch
        from src.models.lstm_world_model import _TorchLSTMWorldModel

        torch_model = _TorchLSTMWorldModel(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=1,
        )

        train_x = self.x_history[:6]
        train_y = self.y_target[:6]
        val_x = self.x_history[6:]
        val_y = self.y_target[6:]

        history = train_world_model(
            model=torch_model,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            num_epochs=3,
            batch_size=2,
            lr=0.005,
            verbose=False,
        )

        self.assertEqual(len(history.train_losses), 3)
        self.assertEqual(len(history.val_losses), 3)
        for loss in history.train_losses:
            self.assertFalse(np.isnan(loss))

        # Test predict_next_state on torch tensor
        pred_dist = torch_model.predict_next_state(train_x[:2])
        self.assertEqual(pred_dist.mean.shape, (2, self.input_dim))


if __name__ == "__main__":
    unittest.main()
