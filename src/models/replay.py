"""Causal replay utilities for analyst-facing escalation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.models.forecaster import rollout_and_score
from src.models.infiltration_scorer import derive_infiltration_score
from src.models.mitre_mapper import CANONICAL_STAGES


ATTACK_STAGES = CANONICAL_STAGES[1:]


@dataclass
class ReplayResult:
    """Bounded, causal replay output used by the Streamlit SOC console."""

    replay_df: pd.DataFrame
    arrival_df: pd.DataFrame
    escalation_alert: Dict[str, Any] | None


def _history_at_step(features: np.ndarray, step: int, history_length: int) -> np.ndarray:
    """Return history ending at `step`, padding only with already-observed state."""
    available = features[: step + 1]
    history = available[-history_length:]
    if len(history) < history_length:
        history = np.concatenate([np.repeat(history[:1], history_length - len(history), axis=0), history])
    return history


def _arrival_rows(stage_probabilities, timestamps, threshold: float) -> List[Dict[str, Any]]:
    rows = []
    for stage in ATTACK_STAGES:
        arrival = next(
            (
                (timestamp, probabilities.get(stage, 0.0))
                for timestamp, probabilities in zip(timestamps, stage_probabilities or [])
                if probabilities.get(stage, 0.0) >= threshold
            ),
            None,
        )
        rows.append({
            "Attack Stage": stage,
            "Predicted Arrival": "Not predicted" if arrival is None else f"+{int(arrival[0])}s",
            "Confidence (%)": 0.0 if arrival is None else round(arrival[1] * 100, 1),
        })
    return rows


def replay_recent_windows(
    windowed: pd.DataFrame,
    normalizer,
    model,
    stage_mapper,
    history_length: int,
    forecast_steps: int,
    time_step_sec: float = 30.0,
    max_replay_windows: int = 20,
    arrival_threshold: float = 0.50,
) -> ReplayResult:
    """Replay at most 20 observation windows without ever reading future states.

    Every forecast at replay step *t* is built from windows available through
    *t* only. The arrival table is taken from the final (most recent) replay
    point, while the alert is the first escalation found across the replay and
    is deliberately emitted once.
    """
    if windowed.empty:
        return ReplayResult(pd.DataFrame(), pd.DataFrame(), None)

    ordered = windowed.sort_values(["window_end", "session_id"], kind="stable").reset_index(drop=True)
    start_index = max(0, len(ordered) - max_replay_windows)
    replay_window = ordered.iloc[start_index:].reset_index(drop=True)
    features = ordered[NETWORK_STATE_FEATURES].to_numpy(dtype=np.float32)
    replay_rows: List[Dict[str, Any]] = []
    escalation_alert = None
    final_arrival_rows: List[Dict[str, Any]] = []
    stage_rank = {stage: rank for rank, stage in enumerate(CANONICAL_STAGES)}

    for replay_step, step in enumerate(range(start_index, len(ordered))):
        history_raw = _history_at_step(features, step, history_length)
        history_norm = normalizer.transform(history_raw)
        observed_state = history_raw[-1]
        observed_prediction = stage_mapper.predict_stage(observed_state)
        timestamp = float(ordered.loc[step, "window_end"])
        forecast = rollout_and_score(
            model=model,
            history_states=history_norm,
            k_steps=forecast_steps,
            current_timestamp=timestamp,
            time_step_sec=time_step_sec,
            unnormalize_fn=normalizer.inverse_transform,
            stage_mapper=stage_mapper,
        )

        arrivals = _arrival_rows(forecast.stage_probabilities, [i * time_step_sec for i in forecast.step_indices], arrival_threshold)
        if replay_step == len(replay_window) - 1:
            final_arrival_rows = arrivals

        escalation = next(
            (
                (future_step, future_stage, probabilities)
                for future_step, (future_stage, probabilities) in enumerate(
                    zip(forecast.dominant_stages, forecast.stage_probabilities or []), start=1
                )
                if stage_rank.get(future_stage, 0) > stage_rank.get(observed_prediction.stage_label, 0)
                and probabilities.get(future_stage, 0.0) >= arrival_threshold
            ),
            None,
        )
        if escalation_alert is None and escalation is not None:
            future_step, future_stage, probabilities = escalation
            escalation_alert = {
                "stage": future_stage,
                "arrival_seconds": int(future_step * time_step_sec),
                "confidence": round(probabilities.get(future_stage, 0.0) * 100, 1),
                "replay_window_end": timestamp,
            }

        replay_rows.append({
            "Window End": timestamp,
            "Observed Stage": observed_prediction.stage_label,
            "Observed Risk (%)": round(float(derive_infiltration_score(observed_state).infiltration_score * 100), 1),
            "Forecast Peak Risk (%)": round(float(np.max(forecast.infiltration_scores) * 100), 1),
            "Escalation Predicted": "YES" if escalation is not None else "NO",
        })

    return ReplayResult(pd.DataFrame(replay_rows), pd.DataFrame(final_arrival_rows), escalation_alert)
