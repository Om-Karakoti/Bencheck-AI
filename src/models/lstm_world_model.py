"""LSTM-Based World Model for Network State Dynamics.

This module learns the temporal transition probability P(S_{t+1} | S_{<=t}):
Given a sequence of past network state vectors [s_{t-L+1}, ..., s_t], it models
the continuous distribution over the subsequent state s_{t+1}.

CRITICAL ARCHITECTURAL DISTINCTION:
This model learns network physics and temporal dynamics via Gaussian Negative
Log-Likelihood (NLL). It is completely decoupled from attack labels.
State forecasting and infiltration scoring are separate, independently callable
functions:
1. predict_next_state(): Predicts distribution P(S_{t+1} | S_{<=t}).
2. derive_infiltration_score(): Downstream security evaluation on predicted states.
This separation prevents the model from collapsing into a mere static classifier.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class StateDistribution:
    """Represents a diagonal Gaussian distribution over next network states."""
    mean: Union[np.ndarray, "torch.Tensor"]
    log_var: Union[np.ndarray, "torch.Tensor"]

    @property
    def variance(self) -> Union[np.ndarray, "torch.Tensor"]:
        if HAS_TORCH and isinstance(self.log_var, torch.Tensor):
            return torch.exp(self.log_var)
        return np.exp(self.log_var)

    @property
    def std(self) -> Union[np.ndarray, "torch.Tensor"]:
        if HAS_TORCH and isinstance(self.log_var, torch.Tensor):
            return torch.exp(0.5 * self.log_var)
        return np.exp(0.5 * self.log_var)

    def sample(self, num_samples: int = 1) -> Union[np.ndarray, "torch.Tensor"]:
        """Draw samples from the predictive Gaussian distribution."""
        if HAS_TORCH and isinstance(self.mean, torch.Tensor):
            eps = torch.randn_like(self.mean)
            return self.mean + eps * self.std
        eps = np.random.randn(*self.mean.shape)
        return self.mean + eps * self.std

    def mode(self) -> Union[np.ndarray, "torch.Tensor"]:
        """Return the expected (mean) next network state."""
        return self.mean

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert distribution parameters to NumPy arrays."""
        if HAS_TORCH and isinstance(self.mean, torch.Tensor):
            m = self.mean.detach().cpu().numpy()
            lv = self.log_var.detach().cpu().numpy()
            return m, lv
        return np.asarray(self.mean), np.asarray(self.log_var)


if HAS_TORCH:
    class _TorchLSTMWorldModel(nn.Module):
        """PyTorch implementation of LSTM World Model."""

        def __init__(
            self,
            input_dim: int = 36,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
            min_log_var: float = -6.0,
            max_log_var: float = 3.0,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.min_log_var = min_log_var
            self.max_log_var = max_log_var

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # Distribution parameter heads
            self.fc_mean = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )
            self.fc_log_var = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x: torch.Tensor) -> StateDistribution:
            """Forward pass returning distribution over S_{t+1}.
            
            Args:
                x: Past trajectory tensor of shape (batch_size, seq_len, input_dim).
            """
            lstm_out, _ = self.lstm(x)
            # Final step representation h_t
            h_t = lstm_out[:, -1, :]

            mean = self.fc_mean(h_t)
            raw_log_var = self.fc_log_var(h_t)
            # Bound log_var for numerical stability in NLL
            log_var = torch.clamp(raw_log_var, self.min_log_var, self.max_log_var)

            return StateDistribution(mean=mean, log_var=log_var)

        def predict_next_state(self, history_states: torch.Tensor) -> StateDistribution:
            """Explicit function: Predicts distribution P(S_{t+1} | S_{<=t}).
            
            Kept strictly separate from infiltration scoring.
            """
            self.eval()
            with torch.no_grad():
                if not isinstance(history_states, torch.Tensor):
                    history_states = torch.tensor(history_states, dtype=torch.float32)
                if history_states.ndim == 2:
                    history_states = history_states.unsqueeze(0)
                return self.forward(history_states)

        def rollout_future_states(
            self,
            history_states: torch.Tensor,
            horizon: int = 3,
            use_mean: bool = True,
        ) -> List[StateDistribution]:
            """Autoregressively roll out predictions over multiple future time steps.

            Args:
                history_states: Initial context of shape (batch_size, seq_len, input_dim).
                horizon: Number of future time steps H to forecast.
                use_mean: If True, feeds predicted mean forward; else samples.

            Returns:
                List of H StateDistribution instances for steps t+1, ..., t+H.
            """
            self.eval()
            with torch.no_grad():
                if not isinstance(history_states, torch.Tensor):
                    history_states = torch.tensor(history_states, dtype=torch.float32)
                if history_states.ndim == 2:
                    history_states = history_states.unsqueeze(0)

                curr_seq = history_states.clone()
                predictions = []

                for _ in range(horizon):
                    dist = self.forward(curr_seq)
                    predictions.append(dist)
                    next_step = dist.mode() if use_mean else dist.sample()
                    next_step = next_step.unsqueeze(1)  # (B, 1, D)
                    curr_seq = torch.cat([curr_seq[:, 1:, :], next_step], dim=1)

                return predictions


class NumpyLSTMWorldModel:
    """Vectorized NumPy reference implementation of LSTM World Model.

    Ensures the pipeline is fully functional and self-contained even in environments
    without a PyTorch C-extension runtime.
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 64,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)

        # LSTM weights [i, f, g, o]
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W_x = rng.randn(4 * hidden_dim, input_dim) * scale
        self.W_h = rng.randn(4 * hidden_dim, hidden_dim) * scale
        self.b_lstm = np.zeros(4 * hidden_dim)
        # Forget gate bias init to 1.0 (standard best practice)
        self.b_lstm[hidden_dim:2 * hidden_dim] = 1.0

        # Mean head
        self.W_mean = rng.randn(input_dim, hidden_dim) * scale
        self.b_mean = np.zeros(input_dim)

        # Log-var head
        self.W_var = rng.randn(input_dim, hidden_dim) * scale
        self.b_var = np.zeros(input_dim)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def forward(self, x: np.ndarray) -> StateDistribution:
        """Forward pass over sequence (B, L, D)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        batch_size, seq_len, _ = x.shape
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

        H = self.hidden_dim
        for t in range(seq_len):
            xt = x[:, t, :]  # (B, D)
            gates = xt @ self.W_x.T + h @ self.W_h.T + self.b_lstm  # (B, 4H)

            i = self._sigmoid(gates[:, :H])
            f = self._sigmoid(gates[:, H:2 * H])
            g = np.tanh(gates[:, 2 * H:3 * H])
            o = self._sigmoid(gates[:, 3 * H:])

            c = f * c + i * g
            h = o * np.tanh(c)

        mean = h @ self.W_mean.T + self.b_mean
        log_var = np.clip(h @ self.W_var.T + self.b_var, -6.0, 3.0)

        return StateDistribution(mean=mean, log_var=log_var)

    def predict_next_state(self, history_states: np.ndarray) -> StateDistribution:
        """Explicit function: Predicts distribution P(S_{t+1} | S_{<=t})."""
        return self.forward(history_states)

    def rollout_future_states(
        self,
        history_states: np.ndarray,
        horizon: int = 3,
        use_mean: bool = True,
    ) -> List[StateDistribution]:
        """Autoregressively roll out predictions over multiple future time steps."""
        x = np.asarray(history_states, dtype=np.float32)
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        curr_seq = x.copy()
        predictions = []

        for _ in range(horizon):
            dist = self.forward(curr_seq)
            predictions.append(dist)
            next_step = dist.mode() if use_mean else dist.sample()
            next_step = np.expand_dims(next_step, axis=1)  # (B, 1, D)
            curr_seq = np.concatenate([curr_seq[:, 1:, :], next_step], axis=1)

        return predictions


def GaussianNLLLoss(
    y_true: Union[np.ndarray, "torch.Tensor"],
    dist: StateDistribution,
    eps: float = 1e-6,
) -> Union[float, "torch.Tensor"]:
    """Gaussian Negative Log-Likelihood Loss for transition dynamics:
    
    L = 0.5 * sum_d (log(sigma_d^2) + (y_d - mu_d)^2 / sigma_d^2)
    """
    if HAS_TORCH and isinstance(y_true, torch.Tensor):
        var = torch.exp(dist.log_var) + eps
        nll = 0.5 * (dist.log_var + ((y_true - dist.mean) ** 2) / var)
        return torch.mean(torch.sum(nll, dim=-1))

    y_arr = np.asarray(y_true)
    mean_arr = np.asarray(dist.mean)
    log_var_arr = np.asarray(dist.log_var)
    var_arr = np.exp(log_var_arr) + eps
    nll = 0.5 * (log_var_arr + ((y_arr - mean_arr) ** 2) / var_arr)
    return float(np.mean(np.sum(nll, axis=-1)))


# Dynamic factory: Uses PyTorch if installed, falls back to NumPy implementation
def create_world_model(
    input_dim: int = 36,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    force_numpy: bool = False,
):
    """Factory creating an LSTM World Model."""
    if HAS_TORCH and not force_numpy:
        return _TorchLSTMWorldModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    return NumpyLSTMWorldModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )


# Standard alias
LSTMWorldModel = _TorchLSTMWorldModel if HAS_TORCH else NumpyLSTMWorldModel
