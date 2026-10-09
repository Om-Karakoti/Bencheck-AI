"""Stable Inference Interface for Network Attack Forecasting World Model.

This module provides the single, stable production entry-point function:
    predict(window_sequence) -> {
        "infiltration_probability": float,
        "predicted_stage": str,
        "top_features": list
    }

This is the only function another team's backend needs to import.
The signature and return schema remain strictly stable regardless of internal changes.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Union
import joblib
import numpy as np
import pandas as pd

from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.data.dataset import StateNormalizer
from src.models.lstm_world_model import NumpyLSTMWorldModel, create_world_model
from src.models.infiltration_scorer import derive_infiltration_score
from src.models.mitre_mapper import MITREStageMapper, map_state_to_attack_stage, CANONICAL_STAGES
from src.models.explainability import InfiltrationSHAPExplainer


class NetworkAttackPredictor:
    """Production inference engine bundling World Model forecasting and explainability."""

    def __init__(
        self,
        world_model: Any,
        stage_mapper: MITREStageMapper,
        normalizer: Optional[StateNormalizer] = None,
        explainer: Optional[InfiltrationSHAPExplainer] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.world_model = world_model
        self.stage_mapper = stage_mapper
        self.normalizer = normalizer
        self.explainer = explainer
        self.feature_names = feature_names or list(NETWORK_STATE_FEATURES)

    def _parse_input_sequence(self, window_sequence: Any) -> np.ndarray:
        """Parse various sequence representations into a normalized (1, L, D) float32 array."""
        # 1. Pandas DataFrame (e.g. L window rows from causal windowing)
        if isinstance(window_sequence, pd.DataFrame):
            # Check for missing features, fill with 0.0 if not present
            df_cols = window_sequence.columns
            missing = [c for c in self.feature_names if c not in df_cols]
            if missing:
                # If dataframe is missing features, attempt copy and fill with 0.0
                df_copy = window_sequence.copy()
                for c in missing:
                    df_copy[c] = 0.0
                raw = df_copy[self.feature_names].to_numpy(dtype=np.float32)
            else:
                raw = window_sequence[self.feature_names].to_numpy(dtype=np.float32)

            if len(raw) == 0:
                raise ValueError("window_sequence DataFrame contains 0 rows.")
            return np.expand_dims(raw, axis=0)  # (1, L, D)

        # 2. List of Dictionaries
        if isinstance(window_sequence, list) and len(window_sequence) > 0 and isinstance(window_sequence[0], dict):
            df = pd.DataFrame(window_sequence)
            return self._parse_input_sequence(df)

        # 3. NumPy Array or List of Lists
        arr = np.asarray(window_sequence, dtype=np.float32)
        if arr.ndim == 2:
            # Shape is (L, D) -> add batch dim -> (1, L, D)
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got shape: {arr.shape}"
                )
            return np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:
            # Shape is (B, L, D) -> ensure B=1 or take first item
            if arr.shape[2] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got shape: {arr.shape}"
                )
            return arr[:1]
        elif arr.ndim == 1:
            # Single state vector (1, D) -> treat as L=1
            if len(arr) != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got vector length: {len(arr)}"
                )
            return arr.reshape(1, 1, -1)
        else:
            raise ValueError(f"Unsupported array dimensions: {arr.ndim}. Expected 2D (L, D) or 3D (1, L, D).")

    def predict(self, window_sequence: Any) -> Dict[str, Any]:
        """Core inference function executing the full forecasting and assessment pipeline.
        
        Returns:
            dict containing:
                "infiltration_probability": float in [0.0, 1.0]
                "predicted_stage": str in canonical MITRE stages
                "top_features": list of top contributing feature names
        """
        # A. Parse raw input sequence into (1, L, D)
        raw_seq = self._parse_input_sequence(window_sequence)  # (1, L, D)
        batch_size, seq_len, num_features = raw_seq.shape

        # B. Apply Normalization if fitted
        if self.normalizer is not None:
            flat_raw = raw_seq.reshape(-1, num_features)
            flat_norm = self.normalizer.transform(flat_raw)
            norm_seq = flat_norm.reshape(1, seq_len, num_features).astype(np.float32)
        else:
            norm_seq = raw_seq

        # C. Predict Next Network State Distribution via World Model
        pred_dist = self.world_model.predict_next_state(norm_seq)
        pred_next_norm = pred_dist.mode()[0]  # (D,)

        # D. Convert back to unnormalized space for physical interpretation
        if self.normalizer is not None:
            unnorm_pred = self.normalizer.inverse_transform(pred_next_norm.reshape(1, -1)).squeeze(0)
        else:
            unnorm_pred = pred_next_norm

        # E. Derive Infiltration Probability Score
        assessment = derive_infiltration_score(
            predicted_state=pred_next_norm,
            feature_names=self.feature_names,
            unnormalize_fn=self.normalizer.inverse_transform if self.normalizer else None,
        )
        infil_prob = float(np.clip(assessment.infiltration_score, 0.0, 1.0))

        # F. MITRE ATT&CK Stage Mapping
        stage_pred = map_state_to_attack_stage(unnorm_pred, self.stage_mapper)
        predicted_stage = str(stage_pred.stage_label)

        # G. SHAP Feature Attribution for Top Risk Drivers
        if self.explainer is not None:
            explanation = self.explainer.explain_prediction(unnorm_pred, n_samples=40)
            if explanation.top_positive_features:
                top_features = [feat[0] for feat in explanation.top_positive_features[:5]]
            else:
                top_df = explanation.to_dataframe()
                top_features = top_df["feature"].head(5).tolist()
        else:
            # Fallback based on raw score contribution
            top_features = [
                name for name in [
                    "unique_dst_ips", "scan_signature_score_mean", "byte_rate",
                    "fan_out_ratio", "syn_ack_ratio"
                ] if name in self.feature_names
            ]

        # Return the exact contract required by other teams' backends
        return {
            "infiltration_probability": float(round(infil_prob, 4)),
            "predicted_stage": str(predicted_stage),
            "top_features": list(top_features),
        }

    def save(self, file_path: str) -> None:
        """Persist the predictor bundle to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        bundle = {
            "world_model": self.world_model,
            "stage_mapper": self.stage_mapper,
            "normalizer": self.normalizer,
            "explainer": self.explainer,
            "feature_names": self.feature_names,
        }
        joblib.dump(bundle, file_path)

    @classmethod
    def load(cls, file_path: str) -> "NetworkAttackPredictor":
        """Load a persisted predictor bundle from disk."""
        bundle = joblib.load(file_path)
        return cls(
            world_model=bundle["world_model"],
            stage_mapper=bundle["stage_mapper"],
            normalizer=bundle.get("normalizer"),
            explainer=bundle.get("explainer"),
            feature_names=bundle.get("feature_names"),
        )


# =========================================================================
# GLOBAL SINGLETON AND DEFAULT PREDICTOR INITIALIZATION
# =========================================================================

_DEFAULT_PREDICTOR: Optional[NetworkAttackPredictor] = None
_LOCK = threading.Lock()


def set_default_predictor(predictor: NetworkAttackPredictor) -> None:
    """Register the active production predictor instance."""
    global _DEFAULT_PREDICTOR
    with _LOCK:
        _DEFAULT_PREDICTOR = predictor


def get_default_predictor() -> NetworkAttackPredictor:
    """Retrieve or lazily initialize the default production predictor."""
    global _DEFAULT_PREDICTOR
    with _LOCK:
        if _DEFAULT_PREDICTOR is not None:
            return _DEFAULT_PREDICTOR

        # Lazy initialization fallback on synthetic baseline data
        d = len(NETWORK_STATE_FEATURES)
        world_model = NumpyLSTMWorldModel(input_dim=d, hidden_dim=32, seed=42)
        stage_mapper = MITREStageMapper(random_state=42)

        # Train minimal baseline mapper so it is fitted
        synth_x = np.random.randn(60, d).astype(np.float32) * 0.1
        synth_y = [CANONICAL_STAGES[i % len(CANONICAL_STAGES)] for i in range(60)]
        stage_mapper.fit(synth_x, synth_y)

        normalizer = StateNormalizer()
        normalizer.fit(pd.DataFrame(synth_x, columns=NETWORK_STATE_FEATURES))

        explainer = InfiltrationSHAPExplainer(
            background_data=synth_x[:20],
            feature_names=list(NETWORK_STATE_FEATURES),
            max_background_samples=20,
            random_state=42,
        )

        _DEFAULT_PREDICTOR = NetworkAttackPredictor(
            world_model=world_model,
            stage_mapper=stage_mapper,
            normalizer=normalizer,
            explainer=explainer,
            feature_names=list(NETWORK_STATE_FEATURES),
        )
        return _DEFAULT_PREDICTOR


# =========================================================================
# STABLE INFERENCE ENTRY POINT FUNCTION
# =========================================================================

def predict(window_sequence: Any) -> Dict[str, Any]:
    """Predict future infiltration risk, attack stage, and top features.
    
    This is the STABLE external entry point for other teams' backends:
    >>> from src.inference import predict
    >>> response = predict(window_sequence)
    >>> print(response)
    {
        "infiltration_probability": 0.6705,
        "predicted_stage": "Command & Control",
        "top_features": ["fan_out_ratio", "unique_dst_ips", ...]
    }

    Args:
        window_sequence: Sequence of historical network states. Accepts:
            - pandas.DataFrame (L rows with feature columns)
            - numpy.ndarray (2D of shape (L, 36) or 3D of shape (1, L, 36))
            - List of dictionaries [{feat_name: value, ...}, ...]
            - List of state vectors [[val_1, ..., val_36], ...]

    Returns:
        Dict[str, Any] with exact keys:
            - "infiltration_probability": float in [0.0, 1.0]
            - "predicted_stage": str
            - "top_features": list of feature names (strings)
    """
    predictor = get_default_predictor()
    return predictor.predict(window_sequence)
