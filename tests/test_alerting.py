import unittest

from src.models.alerting import (
    RISK_ESCALATION,
    AlertGate,
    DEFAULT_HYSTERESIS_MARGIN,
    SEVERE_STAGES,
    evaluate_alert,
)


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


class TestAlertDeduplication(unittest.TestCase):
    def setUp(self):
        self.gate = AlertGate(margin=0.10)

    def test_raises_once_then_suppresses_the_same_escalation(self):
        first = self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)
        second = self.gate.decide([0.62], ["Lateral Movement"], threshold=0.50)

        self.assertTrue(first.notify)
        self.assertFalse(second.notify)
        self.assertTrue(second.ongoing)
        self.assertEqual(second.latched, ["Lateral Movement"])

    def test_holds_the_latch_while_the_stage_persists_even_if_risk_dips(self):
        self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)
        dipped = self.gate.decide([0.10], ["Lateral Movement"], threshold=0.50)

        self.assertFalse(dipped.rearmed)
        self.assertFalse(dipped.notify)
        self.assertTrue(dipped.ongoing)

    def test_rearms_only_after_risk_clears_the_margin_and_the_stage_gone(self):
        self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)
        cleared = self.gate.decide([0.35], ["Reconnaissance"], threshold=0.50)
        re_escalated = self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)

        self.assertTrue(cleared.rearmed)
        self.assertEqual(cleared.latched, [])
        self.assertTrue(re_escalated.notify)

    def test_risk_inside_the_hysteresis_band_does_not_rearm(self):
        self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)
        in_band = self.gate.decide([0.42], ["Reconnaissance"], threshold=0.50)

        self.assertFalse(in_band.rearmed)
        self.assertEqual(in_band.latched, ["Lateral Movement"])
        self.assertEqual(in_band.arm_level, 0.40)

    def test_a_different_stage_is_its_own_escalation(self):
        self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)
        escalated = self.gate.decide([0.61], ["Exfiltration"], threshold=0.50)

        self.assertTrue(escalated.notify)
        self.assertEqual(escalated.latched, ["Lateral Movement", "Exfiltration"])

    def test_risk_only_breach_latches_under_its_own_identity(self):
        first = self.gate.decide([0.60], ["Reconnaissance"], threshold=0.50)
        repeat = self.gate.decide([0.65], ["Reconnaissance"], threshold=0.50)

        self.assertEqual(first.latched, [RISK_ESCALATION])
        self.assertFalse(repeat.notify)
        self.assertTrue(repeat.ongoing)

    def test_no_breach_never_announces(self):
        decision = self.gate.decide([0.10], ["Benign"], threshold=0.50)

        self.assertFalse(decision.notify)
        self.assertFalse(decision.ongoing)
        self.assertEqual(decision.latched, [])

    def test_reset_forgets_every_announced_escalation(self):
        self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)
        self.gate.reset()
        after_reset = self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)

        self.assertTrue(after_reset.notify)

    def test_gate_uses_the_module_default_margin(self):
        self.assertEqual(AlertGate().margin, DEFAULT_HYSTERESIS_MARGIN)

    def test_negative_margin_is_rejected(self):
        with self.assertRaises(ValueError):
            AlertGate(margin=-0.1)

    def test_ongoing_banner_reports_the_rearm_level(self):
        self.gate.decide([0.60], ["Lateral Movement"], threshold=0.50)
        ongoing = self.gate.decide([0.55], ["Lateral Movement"], threshold=0.50)

        self.assertEqual(ongoing.arm_level, 0.40)
        self.assertEqual(ongoing.reasons_text, " ".join(ongoing.evaluation.reasons))
        self.assertFalse(ongoing.reasons_text.startswith("ALERT"))


if __name__ == "__main__":
    unittest.main()
