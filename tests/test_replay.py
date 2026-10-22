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
