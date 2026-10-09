"""Unit tests for data-grounded MITRE ATT&CK stage mapping and evaluation."""

import os
import unittest
import numpy as np
import pandas as pd

from src.models.mitre_mapper import (
    MITREStageMapper,
    map_state_to_attack_stage,
    StagePredictionResult,
    StageEvaluationReport,
    CANONICAL_STAGES,
)
from src.models.lstm_world_model import NumpyLSTMWorldModel
from src.models.forecaster import rollout_and_score
from src.data.config import AttackStage
from src.data.causal_windowing import NETWORK_STATE_FEATURES


class TestMITREStageMapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.num_features = len(NETWORK_STATE_FEATURES)
        cls.num_samples = 120

        # Create realistic synthetic training states for all 6 stages
        records = []
        labels = []

        for i in range(cls.num_samples):
            stage_idx = i % len(CANONICAL_STAGES)
            stage_name = CANONICAL_STAGES[stage_idx]
            labels.append(stage_name)

            state = np.random.randn(cls.num_features).astype(np.float32) * 0.1

            # Inject strong stage-specific characteristics into known feature indices
            if stage_name == "Reconnaissance":
                state[NETWORK_STATE_FEATURES.index("scan_signature_score_mean")] = 0.85
                state[NETWORK_STATE_FEATURES.index("unique_dst_ports")] = 45.0
                state[NETWORK_STATE_FEATURES.index("syn_ack_ratio")] = 3.5
            elif stage_name == "Initial Access":
                state[NETWORK_STATE_FEATURES.index("flow_duration_mean")] = 4.2
                state[NETWORK_STATE_FEATURES.index("retransmit_rate")] = 0.25
                state[NETWORK_STATE_FEATURES.index("payload_size_mean_avg")] = 250.0
            elif stage_name == "Lateral Movement":
                state[NETWORK_STATE_FEATURES.index("unique_dst_ips")] = 12.0
                state[NETWORK_STATE_FEATURES.index("fan_out_ratio")] = 6.0
            elif stage_name == "Command & Control":
                state[NETWORK_STATE_FEATURES.index("iat_var_avg")] = 0.0001
                state[NETWORK_STATE_FEATURES.index("iat_mean_avg")] = 0.04
            elif stage_name == "Exfiltration":
                state[NETWORK_STATE_FEATURES.index("byte_rate")] = 350000.0
                state[NETWORK_STATE_FEATURES.index("payload_size_max")] = 1450.0
            # Benign remains baseline near zero

            records.append(state)

        cls.x_train = np.array(records[:90], dtype=np.float32)
        cls.y_train = labels[:90]
        cls.x_test = np.array(records[90:], dtype=np.float32)
        cls.y_test = labels[90:]

        cls.mapper = MITREStageMapper(random_state=42)
        cls.mapper.fit(cls.x_train, cls.y_train)

    def test_fit_and_classes(self):
        """Verify mapper trains on ground-truth stages and tracks canonical classes."""
        self.assertTrue(self.mapper.is_fitted)
        for stage in CANONICAL_STAGES:
            self.assertIn(stage, self.mapper.classes_)

    def test_predict_stage_single(self):
        """Verify function mapping predicted state vector to an attack_stage label."""
        recon_state = np.zeros(self.num_features, dtype=np.float32)
        recon_state[NETWORK_STATE_FEATURES.index("scan_signature_score_mean")] = 0.90
        recon_state[NETWORK_STATE_FEATURES.index("unique_dst_ports")] = 50.0
        recon_state[NETWORK_STATE_FEATURES.index("syn_ack_ratio")] = 4.0

        res = map_state_to_attack_stage(recon_state, self.mapper)

        self.assertIsInstance(res, StagePredictionResult)
        self.assertEqual(res.stage_label, "Reconnaissance")
        self.assertEqual(res.stage_id, int(AttackStage.RECONNAISSANCE))
        self.assertGreater(res.confidence, 0.40)
        self.assertTrue(res.is_attack())

        # Check that probabilities sum to approximately 1.0
        prob_sum = sum(res.stage_probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_predict_exfiltration(self):
        """Verify mapper identifies exfiltration state characteristics."""
        exfil_state = np.zeros(self.num_features, dtype=np.float32)
        exfil_state[NETWORK_STATE_FEATURES.index("byte_rate")] = 400000.0
        exfil_state[NETWORK_STATE_FEATURES.index("payload_size_max")] = 1460.0

        res = self.mapper.predict_stage(exfil_state)
        self.assertEqual(res.stage_label, "Exfiltration")
        self.assertEqual(res.stage_id, int(AttackStage.EXFILTRATION))

    def test_evaluate_against_actual_ground_truth(self):
        """Verify rigorous evaluation against test set actual attack_stage labels."""
        report = self.mapper.evaluate(self.x_test, self.y_test)

        self.assertIsInstance(report, StageEvaluationReport)
        self.assertGreaterEqual(report.accuracy, 0.70)
        self.assertGreaterEqual(report.macro_f1, 0.70)
        self.assertGreaterEqual(report.macro_precision, 0.70)
        self.assertGreaterEqual(report.macro_recall, 0.70)

        # Verify confusion matrix dataframe structure
        self.assertIsInstance(report.confusion_matrix_df, pd.DataFrame)
        self.assertGreater(len(report.confusion_matrix_df), 0)

        summary_text = report.summary_table()
        self.assertIn("Overall Accuracy", summary_text)
        self.assertIn("Macro F1-Score", summary_text)

    def test_invalid_stages_raise_error(self):
        """Verify invalid invented labels are rejected."""
        invalid_labels = ["Invented_Label_A", "Malware_Bad"] * 5
        fake_states = np.zeros((10, self.num_features))
        bad_mapper = MITREStageMapper()
        with self.assertRaises(ValueError):
            bad_mapper.fit(fake_states, invalid_labels)

    def test_integration_with_rollout_and_score(self):
        """Verify K-step rollout leverages data-grounded stage mapper."""
        model = NumpyLSTMWorldModel(input_dim=self.num_features, hidden_dim=32, seed=42)
        history = np.random.randn(6, self.num_features).astype(np.float32)

        res = rollout_and_score(
            model=model,
            history_states=history,
            k_steps=4,
            current_timestamp=60.0,
            stage_mapper=self.mapper,
        )

        self.assertEqual(len(res.dominant_stages), 4)
        for stage_name in res.dominant_stages:
            self.assertIn(stage_name, CANONICAL_STAGES)

        self.assertIsNotNone(res.stage_probabilities)
        self.assertEqual(len(res.stage_probabilities), 4)

    def test_save_and_load(self):
        """Verify model persistence to disk."""
        save_path = "checkpoints/test_stage_mapper.joblib"
        try:
            self.mapper.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            loaded_mapper = MITREStageMapper.load(save_path)
            self.assertTrue(loaded_mapper.is_fitted)

            # Assert identical prediction
            test_vec = self.x_test[0]
            pred_orig = self.mapper.predict_stage(test_vec)
            pred_loaded = loaded_mapper.predict_stage(test_vec)
            self.assertEqual(pred_orig.stage_label, pred_loaded.stage_label)
            self.assertAlmostEqual(pred_orig.confidence, pred_loaded.confidence, places=4)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


if __name__ == "__main__":
    unittest.main()
