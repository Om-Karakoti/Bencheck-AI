import unittest

from src.models.alerting import SEVERE_STAGES, evaluate_alert


class TestAlertThreshold(unittest.TestCase):
    def test_alerts_when_forecast_risk_reaches_threshold(self):
        result = evaluate_alert(
            [0.10, 0.40, 0.62, 0.30],
            ["Benign", "Reconnaissance", "Initial Access", "Reconnaissance"],
            threshold=0.50,
        )

        self.assertTrue(result.triggered)
        self.assertEqual(result.peak_risk, 0.62)
        self.assertEqual(result.risk_breach["step"], 3)
        self.assertEqual(result.risk_breach["seconds"], 90)
        self.assertIsNone(result.stage_breach)

    def test_equal_to_threshold_alerts(self):
        result = evaluate_alert([0.50], ["Benign"], threshold=0.50)

        self.assertTrue(result.triggered)
        self.assertEqual(result.risk_breach["step"], 1)

    def test_severe_stage_alerts_below_the_risk_threshold(self):
        result = evaluate_alert(
            [0.05, 0.08],
            ["Benign", "Command & Control"],
            threshold=0.90,
        )

        self.assertTrue(result.triggered)
        self.assertIsNone(result.risk_breach)
        self.assertEqual(result.stage_breach["stage"], "Command & Control")
        self.assertEqual(result.stage_breach["seconds"], 60)

    def test_reports_the_earliest_severe_stage(self):
        result = evaluate_alert(
            [0.1, 0.1, 0.1],
            ["Lateral Movement", "Command & Control", "Exfiltration"],
            threshold=0.90,
        )

        self.assertEqual(result.stage_breach["step"], 1)
        self.assertEqual(result.stage_breach["stage"], "Lateral Movement")

    def test_no_alert_below_threshold_with_no_severe_stage(self):
        result = evaluate_alert(
            [0.10, 0.22],
            ["Benign", "Reconnaissance"],
            threshold=0.50,
        )

        self.assertFalse(result.triggered)
        self.assertEqual(result.message, "")
        self.assertEqual(result.reasons, [])

    def test_empty_forecast_does_not_alert(self):
        result = evaluate_alert([], [], threshold=0.50)

        self.assertFalse(result.triggered)
        self.assertEqual(result.peak_risk, 0.0)

    def test_custom_time_step_scales_eta(self):
        result = evaluate_alert([0.9], ["Exfiltration"], threshold=0.50, time_step_sec=45.0)

        self.assertEqual(result.risk_breach["seconds"], 45)
        self.assertEqual(result.stage_breach["seconds"], 45)

    def test_both_conditions_are_reported(self):
        result = evaluate_alert(
            [0.80],
            ["Exfiltration"],
            threshold=0.50,
        )

        self.assertEqual(len(result.reasons), 2)
        self.assertTrue(result.message.startswith("ALERT:"))

    def test_out_of_range_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_alert([0.5], ["Benign"], threshold=1.5)

    def test_monitored_severe_stages(self):
        self.assertEqual(
            SEVERE_STAGES,
            ("Lateral Movement", "Command & Control", "Exfiltration"),
        )


if __name__ == "__main__":
    unittest.main()
