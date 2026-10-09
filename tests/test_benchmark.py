"""Unit tests for Baseline & Benchmark module and Per-Class False Positive Rate (FPR)."""

import unittest
import numpy as np
import pandas as pd

from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.models.mitre_mapper import MITREStageMapper, CANONICAL_STAGES
from src.models.baseline_benchmark import (
    LogisticRegressionBaseline,
    ClassImbalanceReport,
    ModelEvaluationSummary,
    BenchmarkReport,
    analyze_class_imbalance,
    compute_per_class_metrics,
    benchmark_models,
)


class TestBaselineAndBenchmark(unittest.TestCase):
    def test_fpr_calculation_accuracy(self):
        """Verify mathematical precision of False Positive Rate (FPR = FP / (FP + TN))."""
        # Define deterministic confusion scenario:
        # Class A: 4 samples. Class B: 4 samples. Total: 8 samples.
        y_true = ["Class_A", "Class_A", "Class_A", "Class_A", "Class_B", "Class_B", "Class_B", "Class_B"]
        # For Class_B:
        # Pred: 2 True Positives, 2 False Negatives (actual B predicted as A)
        # 1 False Positive (actual A predicted as B)
        # 3 True Negatives (actual A predicted as A)
        y_pred = ["Class_A", "Class_A", "Class_A", "Class_B", "Class_B", "Class_B", "Class_A", "Class_A"]

        summary = compute_per_class_metrics(y_true, y_pred, class_names=["Class_A", "Class_B"])

        m_b = summary.per_class_metrics["Class_B"]
        self.assertEqual(m_b.tp, 2)
        self.assertEqual(m_b.fp, 1)
        self.assertEqual(m_b.fn, 2)
        self.assertEqual(m_b.tn, 3)
        self.assertEqual(m_b.support, 4)

        # FPR = FP / (FP + TN) = 1 / (1 + 3) = 0.25
        self.assertAlmostEqual(m_b.false_positive_rate, 0.25, places=5)
        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3
        self.assertAlmostEqual(m_b.precision, 2.0 / 3.0, places=5)
        # Recall = TP / (TP + FN) = 2 / (2 + 2) = 0.50
        self.assertAlmostEqual(m_b.recall, 0.50, places=5)
        # F1 = 2 * (2/3 * 1/2) / (2/3 + 1/2) = (2/3) / (7/6) = 4/7 = 0.571428
        self.assertAlmostEqual(m_b.f1_score, 4.0 / 7.0, places=5)

    def test_identical_feature_set_enforcement(self):
        """Verify LogisticRegressionBaseline is trained on the exact identical 36 features."""
        baseline = LogisticRegressionBaseline(random_state=42)
        self.assertEqual(len(baseline.feature_names), len(NETWORK_STATE_FEATURES))
        self.assertEqual(baseline.feature_names, list(NETWORK_STATE_FEATURES))

        # Generate synthetic state data
        np.random.seed(42)
        n_samples = 60
        x_synth = np.random.randn(n_samples, len(NETWORK_STATE_FEATURES)).astype(np.float32)
        y_synth = np.random.choice(CANONICAL_STAGES, size=n_samples)

        baseline.fit(x_synth, y_synth)
        self.assertTrue(baseline.is_fitted)

        preds = baseline.predict(x_synth[:5])
        self.assertEqual(len(preds), 5)
        probs = baseline.predict_proba(x_synth[:5])
        self.assertEqual(probs.shape, (5, len(baseline.classes_)))
        # Verify probability rows sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

    def test_imbalance_analyzer_flagging(self):
        """Verify class imbalance analyzer correctly flags heavy benign window dominance."""
        # Create heavily imbalanced dataset (90 benign, 2 each for 5 attack stages = 100 total)
        labels = ["Benign"] * 90
        for st in ["Reconnaissance", "Initial Access", "Lateral Movement", "Command & Control", "Exfiltration"]:
            labels.extend([st] * 2)

        report = analyze_class_imbalance(labels, class_names=CANONICAL_STAGES)

        self.assertIsInstance(report, ClassImbalanceReport)
        self.assertEqual(report.total_samples, 100)
        self.assertEqual(report.majority_class, "Benign")
        self.assertEqual(report.majority_percentage, 90.0)
        self.assertTrue(report.is_heavily_imbalanced)
        self.assertGreaterEqual(report.imbalance_ratio, 40.0)

        banner = report.format_banner()
        self.assertIn("DATASET CLASS IMBALANCE DIAGNOSTIC", banner)
        self.assertIn("90.00%", banner)
        self.assertIn("accuracy is inherently misleading", banner)
        self.assertIn("False Positive Rate", banner)

    def test_benchmark_models_side_by_side(self):
        """Verify side-by-side benchmark report comparing Logistic Regression vs MITREStageMapper."""
        np.random.seed(42)
        num_features = len(NETWORK_STATE_FEATURES)
        num_samples = 120

        # Create multi-stage training data
        records = []
        labels = []
        for i in range(num_samples):
            st = CANONICAL_STAGES[i % len(CANONICAL_STAGES)]
            labels.append(st)
            vec = np.random.randn(num_features).astype(np.float32) * 0.1
            if st == "Reconnaissance":
                vec[NETWORK_STATE_FEATURES.index("scan_signature_score_mean")] = 0.9
            elif st == "Exfiltration":
                vec[NETWORK_STATE_FEATURES.index("byte_rate")] = 300000.0
            records.append(vec)

        x_arr = np.array(records, dtype=np.float32)

        train_x, test_x = x_arr[:90], x_arr[90:]
        train_y, test_y = labels[:90], labels[90:]

        # Train both models on identical features
        baseline = LogisticRegressionBaseline(random_state=42).fit(train_x, train_y)
        primary = MITREStageMapper(random_state=42).fit(train_x, train_y)

        report = benchmark_models(
            baseline_model=baseline,
            primary_model=primary,
            x_test=test_x,
            y_test=test_y,
            class_names=CANONICAL_STAGES,
        )

        self.assertIsInstance(report, BenchmarkReport)
        self.assertIsInstance(report.baseline_summary, ModelEvaluationSummary)
        self.assertIsInstance(report.primary_summary, ModelEvaluationSummary)

        # Verify all classes have FPR and metrics computed
        for st in CANONICAL_STAGES:
            self.assertIn(st, report.baseline_summary.per_class_metrics)
            self.assertIn(st, report.primary_summary.per_class_metrics)
            b_m = report.baseline_summary.per_class_metrics[st]
            p_m = report.primary_summary.per_class_metrics[st]
            self.assertGreaterEqual(b_m.false_positive_rate, 0.0)
            self.assertLessEqual(b_m.false_positive_rate, 1.0)
            self.assertGreaterEqual(p_m.false_positive_rate, 0.0)
            self.assertLessEqual(p_m.false_positive_rate, 1.0)

        # Verify DataFrame conversion
        df = report.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(CANONICAL_STAGES))
        self.assertIn("Baseline FPR", df.columns)
        self.assertIn("Primary FPR", df.columns)
        self.assertIn("Baseline F1", df.columns)
        self.assertIn("Primary F1", df.columns)

        # Verify terminal summary table format
        table_str = report.summary_table()
        self.assertIn("BENCHMARK COMPARISON", table_str)
        self.assertIn("Identical Feature Set", table_str)
        self.assertIn("Macro FPR", table_str)


if __name__ == "__main__":
    unittest.main()
