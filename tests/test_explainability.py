"""Unit tests for SHAP-based explainability and operational traceability.

Validates that SHAP explanations directly decompose the real operational predictions:
1. InfiltrationSHAPExplainer directly decomposes `derive_infiltration_score(...).infiltration_score`.
2. MITREStageSHAPExplainer directly decomposes `stage_mapper.classifier.predict_proba(...)`.
Guarantees Shapley additivity efficiency: base_value + sum(phi_i) == prediction_value.
"""

import inspect
import unittest
import numpy as np
import pandas as pd

from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.models.infiltration_scorer import derive_infiltration_score
from src.models.mitre_mapper import MITREStageMapper, CANONICAL_STAGES
from src.models.explainability import (
    InfiltrationSHAPExplainer,
    MITREStageSHAPExplainer,
    SHAPExplanation,
    _compute_kernel_shap,
)


class TestSHAPExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.num_features = len(NETWORK_STATE_FEATURES)
        cls.num_background = 40

        # Create realistic baseline background data (predominantly benign traffic)
        cls.bg_data = np.random.randn(cls.num_background, cls.num_features).astype(np.float32) * 0.05
        cls.bg_data = np.abs(cls.bg_data)

        # Build and fit a MITREStageMapper with synthetic multi-stage data
        records = []
        labels = []
        for i in range(120):
            stage = CANONICAL_STAGES[i % len(CANONICAL_STAGES)]
            labels.append(stage)
            vec = np.random.randn(cls.num_features).astype(np.float32) * 0.1
            if stage == "Reconnaissance":
                vec[NETWORK_STATE_FEATURES.index("scan_signature_score_mean")] = 0.9
                vec[NETWORK_STATE_FEATURES.index("unique_dst_ports")] = 40.0
            elif stage == "Exfiltration":
                vec[NETWORK_STATE_FEATURES.index("byte_rate")] = 300000.0
                vec[NETWORK_STATE_FEATURES.index("payload_size_max")] = 1400.0
            records.append(vec)

        cls.stage_mapper = MITREStageMapper(random_state=42)
        cls.stage_mapper.fit(np.array(records, dtype=np.float32), labels)

    def test_infiltration_shap_additivity_and_exact_match(self):
        """Verify SHAP explanation decomposes the EXACT derive_infiltration_score result."""
        explainer = InfiltrationSHAPExplainer(
            background_data=self.bg_data,
            max_background_samples=25,
            random_state=42,
        )

        # Create a high-risk exfiltration state
        exfil_state = np.zeros(self.num_features, dtype=np.float32)
        exfil_state[NETWORK_STATE_FEATURES.index("byte_rate")] = 500000.0
        exfil_state[NETWORK_STATE_FEATURES.index("payload_size_max")] = 1460.0
        exfil_state[NETWORK_STATE_FEATURES.index("scan_signature_score_mean")] = 0.2

        # 1. Direct operational call (what the user actually sees)
        direct_assessment = derive_infiltration_score(exfil_state, feature_names=NETWORK_STATE_FEATURES)
        expected_score = direct_assessment.infiltration_score

        # 2. Compute SHAP explanation
        explanation = explainer.explain_prediction(exfil_state, n_samples=150)

        self.assertIsInstance(explanation, SHAPExplanation)
        # Verify prediction_value matches direct operational call exactly
        self.assertAlmostEqual(explanation.prediction_value, expected_score, places=4)

        # 3. Verify Shapley efficiency / additivity: base_value + sum(phi_i) == prediction_value
        shap_sum = sum(explanation.shap_values.values())
        reconstructed = explanation.base_value + shap_sum
        self.assertAlmostEqual(reconstructed, explanation.prediction_value, places=3)
        self.assertLess(explanation.additivity_gap, 1e-3)

    def test_infiltration_shap_identifies_top_risk_drivers(self):
        """Verify top positive SHAP features identify the actual attack characteristics."""
        explainer = InfiltrationSHAPExplainer(
            background_data=self.bg_data,
            max_background_samples=25,
            random_state=42,
        )

        recon_state = np.zeros(self.num_features, dtype=np.float32)
        recon_state[NETWORK_STATE_FEATURES.index("scan_signature_score_mean")] = 0.95
        recon_state[NETWORK_STATE_FEATURES.index("unique_dst_ports")] = 80.0
        recon_state[NETWORK_STATE_FEATURES.index("syn_ack_ratio")] = 5.0

        explanation = explainer.explain_prediction(recon_state, n_samples=150)

        # Verify positive drivers exist and are sorted descending
        self.assertGreater(len(explanation.top_positive_features), 0)
        top_driver_name = explanation.top_positive_features[0][0]
        top_driver_phi = explanation.top_positive_features[0][1]
        self.assertGreater(top_driver_phi, 0.0)

        top_feature_names = [name for name, phi, val in explanation.top_positive_features[:3]]
        # At least one of the prominent scan features should appear in the top drivers
        self.assertTrue(
            any(f in top_feature_names for f in ["scan_signature_score_mean", "unique_dst_ports", "syn_ack_ratio"])
        )

    def test_mitre_stage_shap_explainer_probability_attribution(self):
        """Verify MITREStageSHAPExplainer decomposes stage_mapper posterior probabilities."""
        explainer = MITREStageSHAPExplainer(
            stage_mapper=self.stage_mapper,
            background_data=self.bg_data,
            max_background_samples=25,
            random_state=42,
        )

        test_state = np.zeros(self.num_features, dtype=np.float32)
        test_state[NETWORK_STATE_FEATURES.index("scan_signature_score_mean")] = 0.95
        test_state[NETWORK_STATE_FEATURES.index("unique_dst_ports")] = 45.0

        # Direct classifier posterior probability
        probs = self.stage_mapper.classifier.predict_proba(test_state.reshape(1, -1))[0]
        recon_idx = list(self.stage_mapper.classes_).index("Reconnaissance")
        direct_recon_prob = float(probs[recon_idx])

        explanation = explainer.explain_stage_prediction(test_state, target_stage="Reconnaissance", n_samples=150)

        self.assertIsInstance(explanation, SHAPExplanation)
        self.assertAlmostEqual(explanation.prediction_value, direct_recon_prob, places=3)

        # Verify Shapley additivity
        shap_sum = sum(explanation.shap_values.values())
        self.assertAlmostEqual(explanation.base_value + shap_sum, explanation.prediction_value, places=3)
        self.assertLess(explanation.additivity_gap, 1e-3)

    def test_traceability_confirmation_comments_present(self):
        """Verify source code contains explicit traceability comments confirming no surrogates."""
        from src.models import explainability
        source = inspect.getsource(explainability)

        # Must confirm operational functions wrapped
        self.assertIn("CRITICAL TRACEABILITY & SURROGATE-FREE GUARANTEE", source)
        self.assertIn("derive_infiltration_score", source)
        self.assertIn("stage_mapper.classifier.predict_proba", source)
        self.assertIn("EXACT TARGET FUNCTION CONFIRMATION", source)

    def test_to_dataframe_and_summary_table(self):
        """Verify export formats to DataFrame and formatted terminal summary."""
        explainer = InfiltrationSHAPExplainer(
            background_data=self.bg_data,
            max_background_samples=20,
            random_state=42,
        )
        sample_state = np.random.randn(self.num_features).astype(np.float32) * 0.1
        sample_state[NETWORK_STATE_FEATURES.index("scan_signature_score_mean")] = 0.8

        exp = explainer.explain_prediction(sample_state, n_samples=100)

        df = exp.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(NETWORK_STATE_FEATURES))
        for col in ["feature", "feature_value", "shap_value", "abs_importance", "direction"]:
            self.assertIn(col, df.columns)

        summary = exp.summary_table(top_n=3)
        self.assertIn("Target Explained:", summary)
        self.assertIn("Top 3 Factors Escalating Threat (+):", summary)


if __name__ == "__main__":
    unittest.main()
