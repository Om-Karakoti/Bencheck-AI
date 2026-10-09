"""K-Step Autoregressive Rollout and Temporal Infiltration Scoring.

This module rolls the trained World Model forward K steps into the future:
1. Autoregressively simulates future network state transitions:
   S_t -> S_{t+1} -> S_{t+2} -> ... -> S_{t+K}
2. Converts the rolled-out state trajectory into a time-series of attack
   infiltration probability scores p(infiltrate | S_{t+k}) and threat tiers.

This enables proactive, pre-incident forecasting: SOC teams can observe attack
progression 60-300 seconds before damaging lateral movement or exfiltration occurs.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union
import numpy as np

from src.models.lstm_world_model import LSTMWorldModel, NumpyLSTMWorldModel, StateDistribution
from src.models.infiltration_scorer import derive_infiltration_score, InfiltrationAssessment
from src.data.causal_windowing import NETWORK_STATE_FEATURES


@dataclass
class RolloutForecastResult:
    """Container for multi-step autoregressive forecast and infiltration trajectory."""
    horizon_k: int
    timestamps: List[float]                  # Forecasted time points [t + dt, ..., t + K*dt]
    step_indices: List[int]                  # Relative future steps [1, 2, ..., K]
    predicted_states: np.ndarray             # Forecasted state vectors (K, D)
    predicted_variances: np.ndarray          # Epistemic & aleatoric variance (K, D)
    infiltration_scores: np.ndarray          # Time-series infiltration probability [0, 1] (K,)
    threat_levels: List[str]                 # Threat tier per future step
    dominant_stages: List[str]               # Dominant kill-chain stage per future step
    explanations: List[str]                  # Per-step diagnosis
    component_time_series: Dict[str, List[float]] = field(default_factory=dict)
    stage_probabilities: Optional[List[Dict[str, float]]] = None

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert forecast results into a clean pandas DataFrame."""
        import pandas as pd
        data = {
            "future_step": self.step_indices,
            "forecast_timestamp": self.timestamps,
            "infiltration_probability": self.infiltration_scores,
            "threat_level": self.threat_levels,
            "predicted_stage": self.dominant_stages,
            "mean_state_uncertainty": np.mean(self.predicted_variances, axis=-1),
            "recon_score": self.component_time_series.get("reconnaissance", []),
            "access_score": self.component_time_series.get("initial_access", []),
            "lateral_score": self.component_time_series.get("lateral_movement", []),
            "c2_score": self.component_time_series.get("command_and_control", []),
            "exfil_score": self.component_time_series.get("exfiltration", []),
            "explanation": self.explanations,
        }
        return pd.DataFrame(data)

    def summary(self) -> Dict[str, Union[int, float, str]]:
        """Summary of peak forecasted threat."""
        max_idx = int(np.argmax(self.infiltration_scores))
        return {
            "horizon_k": self.horizon_k,
            "peak_infiltration_score": round(float(self.infiltration_scores[max_idx]), 4),
            "peak_step": self.step_indices[max_idx],
            "peak_timestamp": self.timestamps[max_idx],
            "peak_threat_level": self.threat_levels[max_idx],
            "peak_stage": self.dominant_stages[max_idx],
        }


def rollout_and_score(
    model: Union[LSTMWorldModel, NumpyLSTMWorldModel],
    history_states: Union[np.ndarray, "torch.Tensor"],
    k_steps: int = 5,
    current_timestamp: float = 0.0,
    time_step_sec: float = 30.0,
    unnormalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    stage_mapper: Optional[object] = None,
    use_mean: bool = True,
) -> RolloutForecastResult:
    """Roll the World Model forward K steps and compute time-series infiltration probabilities.

    # =========================================================================
    # TWO-PHASE SEPARATION:
    # Phase 1: Autoregressive state transition rollout P(S_{t+k} | S_{t+k-1}).
    #          The World Model predicts purely how network dynamics evolve.
    # Phase 2: Convert forecasted physical states into infiltration probabilities
    #          and data-grounded MITRE ATT&CK stages.
    # =========================================================================

    Args:
        model: Trained LSTMWorldModel or NumpyLSTMWorldModel instance.
        history_states: Past state trajectory of shape (L, D) or (1, L, D).
        k_steps: Number of future steps K to roll forward (e.g. 5 steps of 30s = 150s).
        current_timestamp: Base observation timestamp t at the end of the history.
        time_step_sec: Duration Delta_t of each time window in seconds (default: 30.0s).
        unnormalize_fn: Function to map normalized states back to raw scale for scoring.
        stage_mapper: Optional trained MITREStageMapper for data-grounded stage prediction.
        use_mean: If True, feeds expected state mu forward; else samples from distribution.

    Returns:
        RolloutForecastResult containing the time-series state predictions,
        uncertainty projections, infiltration probability scores, and MITRE stages.
    """
    if k_steps <= 0:
        raise ValueError(f"k_steps must be positive, got {k_steps}")

    # Phase 1: Rollout future state distributions
    # Supports both PyTorch and NumPy implementations via their rollout_future_states API
    dist_list: List[StateDistribution] = model.rollout_future_states(
        history_states=history_states,
        horizon=k_steps,
        use_mean=use_mean,
    )

    pred_states_list = []
    pred_vars_list = []
    timestamps = []
    step_indices = []

    # Phase 2: Score each rolled-out future state
    infil_scores = []
    threat_tiers = []
    dom_stages = []
    explanations = []
    stage_probs_list: List[Dict[str, float]] = []
    component_series: Dict[str, List[float]] = {
        "reconnaissance": [],
        "initial_access": [],
        "lateral_movement": [],
        "command_and_control": [],
        "exfiltration": [],
    }

    for step_k, dist in enumerate(dist_list, start=1):
        m, lv = dist.to_numpy()
        # Extract 1D state vector
        mean_vec = m[0] if m.ndim > 1 else m
        var_vec = np.exp(lv[0]) if lv.ndim > 1 else np.exp(lv)

        pred_states_list.append(mean_vec)
        pred_vars_list.append(var_vec)

        future_t = current_timestamp + (step_k * time_step_sec)
        timestamps.append(round(future_t, 2))
        step_indices.append(step_k)

        # Unnormalize if function provided
        raw_vec = unnormalize_fn(np.expand_dims(mean_vec, 0)).squeeze(0) if unnormalize_fn else mean_vec

        # Call distinct infiltration scorer on forecasted state
        assessment: InfiltrationAssessment = derive_infiltration_score(
            predicted_state=mean_vec,
            unnormalize_fn=unnormalize_fn,
        )

        # If data-grounded stage mapper is provided, use its learned prediction
        if stage_mapper is not None:
            stage_res = stage_mapper.predict_stage(raw_vec)
            stage_label = stage_res.stage_label
            stage_probs_list.append(stage_res.stage_probabilities)
        else:
            stage_label = assessment.dominant_stage

        infil_scores.append(assessment.infiltration_score)
        threat_tiers.append(assessment.threat_level)
        dom_stages.append(stage_label)
        explanations.append(assessment.explanation)

        for key in component_series:
            component_series[key].append(assessment.sub_scores.get(key, 0.0))

    return RolloutForecastResult(
        horizon_k=k_steps,
        timestamps=timestamps,
        step_indices=step_indices,
        predicted_states=np.stack(pred_states_list, axis=0),
        predicted_variances=np.stack(pred_vars_list, axis=0),
        infiltration_scores=np.array(infil_scores, dtype=np.float32),
        threat_levels=threat_tiers,
        dominant_stages=dom_stages,
        explanations=explanations,
        component_time_series=component_series,
        stage_probabilities=stage_probs_list if stage_probs_list else None,
    )


class RolloutForecaster:
    """Stateful wrapper for performing rolling horizon forecasts in streaming/monitoring pipelines."""

    def __init__(
        self,
        model: Union[LSTMWorldModel, NumpyLSTMWorldModel],
        k_steps: int = 5,
        time_step_sec: float = 30.0,
        unnormalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        stage_mapper: Optional[object] = None,
    ):
        self.model = model
        self.k_steps = k_steps
        self.time_step_sec = time_step_sec
        self.unnormalize_fn = unnormalize_fn
        self.stage_mapper = stage_mapper

    def forecast(
        self,
        history_states: Union[np.ndarray, "torch.Tensor"],
        current_timestamp: float = 0.0,
    ) -> RolloutForecastResult:
        """Execute K-step rollout and generate infiltration time series."""
        return rollout_and_score(
            model=self.model,
            history_states=history_states,
            k_steps=self.k_steps,
            current_timestamp=current_timestamp,
            time_step_sec=self.time_step_sec,
            unnormalize_fn=self.unnormalize_fn,
            stage_mapper=self.stage_mapper,
        )
