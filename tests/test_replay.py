import unittest

import numpy as np
import pandas as pd

from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.models.lstm_world_model import StateDistribution
from src.models.mitre_mapper import CANONICAL_STAGES, StagePredictionResult
from src.models.replay import replay_recent_windows


class IdentityNormalizer:
    def transform(self, value):
        return value

    def inverse_transform(self, value):
        return value


class RecordingModel:
    def __init__(self):
        self.histories = []

    def rollout_future_states(self, history_states, horizon, use_mean=True):
        self.histories.append(np.asarray(history_states).copy())
        future = np.zeros(len(NETWORK_STATE_FEATURES), dtype=np.float32)
        future[0] = 100.0  # Fake mapper treats this as Exfiltration.
        return [StateDistribution(future[None, :], np.zeros((1, len(future)), dtype=np.float32)) for _ in range(horizon)]


class ReplayMapper:
    def predict_stage(self, state):
        is_future = float(np.asarray(state)[0]) >= 100.0
        stage = "Exfiltration" if is_future else "Benign"
        probs = {name: 0.0 for name in CANONICAL_STAGES}
        probs[stage] = 0.9
        return StagePredictionResult(stage, CANONICAL_STAGES.index(stage), 0.9, probs)


class SwitchingMapper:
    """Benign for the observed zero state; escalation stage comes from the rollout."""

    def predict_stage(self, state):
        value = float(np.asarray(state)[0])
        if value >= 100.0:
            stage = "Exfiltration"
        elif value >= 50.0:
            stage = "Lateral Movement"
        else:
            stage = "Benign"
        probs = {name: 0.0 for name in CANONICAL_STAGES}
        probs[stage] = 0.9
        return StagePredictionResult(stage, CANONICAL_STAGES.index(stage), 0.9, probs)


class GatedModel:
    """Escalates only while the observed state's second feature acts as the gate."""

    def rollout_future_states(self, history_states, horizon, use_mean=True):
        gate = float(np.asarray(history_states)[-1][1])
        future = np.zeros(len(NETWORK_STATE_FEATURES), dtype=np.float32)
        if gate >= 2.0:
            future[0] = 100.0
        elif gate >= 1.0:
            future[0] = 50.0
        return [
            StateDistribution(future[None, :], np.zeros((1, len(future)), dtype=np.float32))
            for _ in range(horizon)
        ]


class StubExplanation:
    def __init__(self, drivers):
        self.top_positive_features = drivers


class StubExplainer:
    def __init__(self):
        self.calls = []

    def explain_prediction(self, state_vector, n_samples=200):
        self.calls.append((np.asarray(state_vector).copy(), n_samples))
        return StubExplanation([
            ("fan_out_ratio", 0.31, 12.0),
            ("byte_rate", 0.18, 9500.0),
            ("packet_rate", 0.04, 33.0),
        ])


def _windowed_frame(gates):
    rows = []
    for index, gate in enumerate(gates):
        row = {feature: 0.0 for feature in NETWORK_STATE_FEATURES}
        row[NETWORK_STATE_FEATURES[1]] = gate
        row.update({"window_end": (index + 1) * 30.0, "session_id": "session-1"})
        rows.append(row)
    return pd.DataFrame(rows)


class TestCausalReplay(unittest.TestCase):
    def test_replays_latest_twenty_windows_and_emits_one_alert(self):
        rows = []
        for index in range(25):
            row = {feature: 0.0 for feature in NETWORK_STATE_FEATURES}
            row.update({"window_end": (index + 1) * 30.0, "session_id": "session-1"})
            rows.append(row)
        model = RecordingModel()

        result = replay_recent_windows(
            pd.DataFrame(rows), IdentityNormalizer(), model, ReplayMapper(),
            history_length=4, forecast_steps=3, time_step_sec=30.0,
        )

        self.assertEqual(len(result.replay_df), 20)
        self.assertEqual(len(model.histories), 20)
        self.assertEqual(result.replay_df.iloc[0]["Window End"], 180.0)
        self.assertEqual(result.escalation_alert["stage"], "Exfiltration")
        self.assertEqual(result.escalation_alert["arrival_seconds"], 30)
        exfil = result.arrival_df[result.arrival_df["Attack Stage"] == "Exfiltration"].iloc[0]
        self.assertEqual(exfil["ETA"], "30 seconds")
        self.assertEqual(exfil["Confidence (%)"], 90.0)

    def test_alert_log_carries_time_stage_eta_and_top_two_shap_drivers(self):
        explainer = StubExplainer()

        result = replay_recent_windows(
            _windowed_frame([2.0, 2.0]), IdentityNormalizer(), GatedModel(), SwitchingMapper(),
            history_length=2, forecast_steps=3, time_step_sec=30.0,
            explainer=explainer, shap_samples=17,
        )

        self.assertEqual(len(result.alert_log), 1)
        entry = result.alert_log[0]
        self.assertEqual(entry["replay_time_sec"], 30.0)
        self.assertEqual(entry["stage"], "Exfiltration")
        self.assertEqual(entry["eta_seconds"], 30)
        self.assertEqual(entry["confidence_pct"], 90.0)
        self.assertEqual(
            entry["drivers"],
            [{"feature": "fan_out_ratio", "impact": 0.31, "value": 12.0},
             {"feature": "byte_rate", "impact": 0.18, "value": 9500.0}],
        )
        # Explainers receive the raw observed state, not the normalised rollout.
        self.assertEqual(len(explainer.calls), 1)
        state, n_samples = explainer.calls[0]
        self.assertEqual(n_samples, 17)
        self.assertEqual(float(state[1]), 2.0)

    def test_alert_log_omits_drivers_without_an_explainer(self):
        result = replay_recent_windows(
            _windowed_frame([2.0]), IdentityNormalizer(), GatedModel(), SwitchingMapper(),
            history_length=2, forecast_steps=3, time_step_sec=30.0,
        )

        self.assertEqual(len(result.alert_log), 1)
        self.assertEqual(result.alert_log[0]["drivers"], [])

    def test_alert_log_does_not_repeat_while_the_same_stage_persists(self):
        result = replay_recent_windows(
            _windowed_frame([2.0] * 6), IdentityNormalizer(), GatedModel(), SwitchingMapper(),
            history_length=2, forecast_steps=3, time_step_sec=30.0,
        )

        self.assertEqual([entry["stage"] for entry in result.alert_log], ["Exfiltration"])
        self.assertEqual([entry["replay_time_sec"] for entry in result.alert_log], [30.0])

    def test_alert_logs_a_new_episode_after_the_escalation_clears(self):
        gates = [1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0]

        result = replay_recent_windows(
            _windowed_frame(gates), IdentityNormalizer(), GatedModel(), SwitchingMapper(),
            history_length=2, forecast_steps=3, time_step_sec=30.0,
        )

        self.assertEqual(
            [(entry["stage"], entry["replay_time_sec"], entry["eta_seconds"]) for entry in result.alert_log],
            [("Lateral Movement", 30.0, 30), ("Exfiltration", 150.0, 30), ("Lateral Movement", 210.0, 30)],
        )

    def test_alert_log_is_empty_when_no_escalation_is_forecast(self):
        result = replay_recent_windows(
            _windowed_frame([0.0] * 5), IdentityNormalizer(), GatedModel(), SwitchingMapper(),
            history_length=2, forecast_steps=3, time_step_sec=30.0,
            explainer=StubExplainer(),
        )

        self.assertEqual(result.alert_log, [])
        self.assertIsNone(result.escalation_alert)
