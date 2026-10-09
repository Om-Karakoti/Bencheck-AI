"""World Model Sequence Dataset and Feature Normalization.

This module formats windowed network state vectors into sequential trajectories
for training World Models.

Given a continuous sequence of states per session [s_1, s_2, ..., s_T], it builds:
- Context / History: X = [s_{t-L+1}, ..., s_t] in R^{L x D}
- Future State Targets: Y_state = [s_{t+1}, ..., s_{t+H}] in R^{H x D}
- Future Stage Targets: Y_stage = [stage_{t+1}, ..., stage_{t+H}] in {0..5}^H
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.config import WindowConfig
from src.data.causal_windowing import NETWORK_STATE_FEATURES


class StateNormalizer:
    """Standardizes network state feature vectors using Z-score normalization.

    To prevent data leakage, statistics (mean and std) must only be computed
    on training sessions.
    """

    def __init__(self, feature_names: Optional[List[str]] = None, eps: float = 1e-6):
        self.feature_names = feature_names or NETWORK_STATE_FEATURES
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, state_df_or_array: Union[pd.DataFrame, np.ndarray]) -> "StateNormalizer":
        """Fit normalization parameters exclusively on training data."""
        if isinstance(state_df_or_array, pd.DataFrame):
            data = state_df_or_array[self.feature_names].to_numpy(dtype=np.float32)
        else:
            data = np.asarray(state_df_or_array, dtype=np.float32)

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Prevent division by zero for invariant features
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self.is_fitted = True
        return self

    def transform(self, state_df_or_array: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Standardize state features using fitted mean and std."""
        if not self.is_fitted:
            raise RuntimeError("StateNormalizer must be fitted before transforming.")

        if isinstance(state_df_or_array, pd.DataFrame):
            data = state_df_or_array[self.feature_names].to_numpy(dtype=np.float32)
        else:
            data = np.asarray(state_df_or_array, dtype=np.float32)

        return (data - self.mean) / self.std

    def inverse_transform(self, normalized_array: np.ndarray) -> np.ndarray:
        """Reconstruct original un-normalized state vector values."""
        if not self.is_fitted:
            raise RuntimeError("StateNormalizer must be fitted before inverse transforming.")
        return (normalized_array * self.std) + self.mean


class WorldModelSequenceDataset:
    """Prepares causal sequence pairs (History X -> Forecast Targets Y) for World Model training.

    Supports both pure NumPy slicing and standard PyTorch Dataset usage.
    """

    def __init__(
        self,
        windowed_df: pd.DataFrame,
        config: Optional[WindowConfig] = None,
        normalizer: Optional[StateNormalizer] = None,
        fit_normalizer: bool = False,
    ):
        self.config = config or WindowConfig()
        self.history_len = self.config.history_length
        self.horizon_len = self.config.forecast_horizon
        self.feature_names = NETWORK_STATE_FEATURES

        # Normalization
        if normalizer is None:
            self.normalizer = StateNormalizer(self.feature_names)
            if fit_normalizer:
                self.normalizer.fit(windowed_df)
        else:
            self.normalizer = normalizer
            if fit_normalizer and not self.normalizer.is_fitted:
                self.normalizer.fit(windowed_df)

        self.sequences: List[Dict[str, np.ndarray]] = []
        self._build_sequences(windowed_df)

    def _build_sequences(self, windowed_df: pd.DataFrame) -> None:
        """Extract continuous trajectory slices per session."""
        required_len = self.history_len + self.horizon_len

        for session_id, session_df in windowed_df.groupby("session_id", sort=False):
            # Sort by step index or window end timestamp
            s_df = session_df.sort_values(by="window_end").reset_index(drop=True)
            n_steps = len(s_df)

            if n_steps < required_len:
                continue

            # Extract raw or normalized feature matrix
            if self.normalizer.is_fitted:
                states = self.normalizer.transform(s_df)
            else:
                states = s_df[self.feature_names].to_numpy(dtype=np.float32)

            stages = s_df["attack_stage"].to_numpy(dtype=np.int64)
            is_attack = s_df["is_attack"].to_numpy(dtype=np.float32)
            timestamps = s_df["window_end"].to_numpy(dtype=np.float64)

            # Slide over valid causal sequence points
            for i in range(n_steps - required_len + 1):
                hist_end = i + self.history_len
                future_end = hist_end + self.horizon_len

                # History slice: [t - L + 1, ..., t]
                x_hist = states[i:hist_end]  # Shape: (L, D)
                x_stages = stages[i:hist_end]  # Shape: (L,)

                # Future forecast target: [t + 1, ..., t + H]
                y_future_state = states[hist_end:future_end]  # Shape: (H, D)
                y_future_stage = stages[hist_end:future_end]  # Shape: (H,)
                y_future_attack = is_attack[hist_end:future_end]  # Shape: (H,)

                self.sequences.append(
                    {
                        "session_id": session_id,
                        "current_timestamp": timestamps[hist_end - 1],
                        "history_states": x_hist,
                        "history_stages": x_stages,
                        "future_states": y_future_state,
                        "future_stages": y_future_stage,
                        "future_is_attack": y_future_attack,
                    }
                )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.sequences[idx]

    def get_batch(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """Assemble a mini-batch of trajectory sequences as NumPy arrays."""
        batch_x = np.stack([self.sequences[i]["history_states"] for i in indices], axis=0)
        batch_y_state = np.stack([self.sequences[i]["future_states"] for i in indices], axis=0)
        batch_y_stage = np.stack([self.sequences[i]["future_stages"] for i in indices], axis=0)
        batch_y_attack = np.stack([self.sequences[i]["future_is_attack"] for i in indices], axis=0)

        return {
            "x_history": batch_x,           # (B, L, D)
            "y_future_state": batch_y_state, # (B, H, D)
            "y_future_stage": batch_y_stage, # (B, H)
            "y_future_attack": batch_y_attack # (B, H)
        }

    def to_torch_dataset(self):
        """Convert into a native torch.utils.data.Dataset if PyTorch is available."""
        try:
            import torch
            from torch.utils.data import Dataset

            class _TorchAdapter(Dataset):
                def __init__(self, seqs):
                    self.seqs = seqs

                def __len__(self):
                    return len(self.seqs)

                def __getitem__(self, i):
                    item = self.seqs[i]
                    return {
                        "x_history": torch.from_numpy(item["history_states"]),
                        "future_states": torch.from_numpy(item["future_states"]),
                        "future_stages": torch.from_numpy(item["future_stages"]),
                        "future_is_attack": torch.from_numpy(item["future_is_attack"]),
                    }

            return _TorchAdapter(self.sequences)
        except ImportError:
            raise ImportError("PyTorch is not installed in the current environment.")
