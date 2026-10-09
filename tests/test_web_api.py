"""Unit and integration tests for the Flask Web API."""

import json
import unittest
from web.app import create_app


class TestWebAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        # Initialize pipeline state once so all dependent endpoints can be tested
        resp = cls.client.post("/api/pipeline/run_all")
        assert resp.status_code == 200

    def test_status_endpoint(self):
        """Verify GET /api/status returns 200 OK and pipeline status."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("num_features", data)
        self.assertEqual(data["num_features"], 36)

    def test_presets_endpoint(self):
        """Verify GET /api/presets returns attack presets."""
        resp = self.client.get("/api/presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("presets", data)
        self.assertGreaterEqual(len(data["presets"]), 5)

    def test_predict_endpoint(self):
        """Verify POST /api/predict returns the exact stable contract."""
        pred_resp = self.client.post("/api/predict", json={})
        self.assertEqual(pred_resp.status_code, 200)
        pred_data = pred_resp.get_json()
        self.assertTrue(pred_data["success"])
        res = pred_data["result"]
        self.assertIn("infiltration_probability", res)
        self.assertIn("predicted_stage", res)
        self.assertIn("top_features", res)
        self.assertIsInstance(res["infiltration_probability"], float)
        self.assertIsInstance(res["predicted_stage"], str)
        self.assertIsInstance(res["top_features"], list)

    def test_model_rollout_endpoint(self):
        """Verify POST /api/model/rollout returns K future steps."""
        resp = self.client.post("/api/model/rollout", json={"k_steps": 3})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["rollout_steps"]), 3)

    def test_shap_explain_endpoint(self):
        """Verify POST /api/explain/shap returns exact attributions and zero gap."""
        resp = self.client.post("/api/explain/shap", json={"step_index": 1})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("top_positive_drivers", data)
        self.assertIn("additivity_gap", data)
        self.assertLess(data["additivity_gap"], 1e-4)

    def test_benchmark_endpoint(self):
        """Verify POST /api/benchmark returns comparison table and imbalance banner."""
        resp = self.client.post("/api/benchmark")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("comparison_table", data)
        self.assertIn("imbalance_banner", data)

    def test_leakage_check_endpoint(self):
        """Verify POST /api/leakage/check executes temporal order shuffling."""
        resp = self.client.post("/api/leakage/check", json={"min_relative_drop": 0.08})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("relative_mse_increase", data)

    def test_static_index_serving(self):
        """Verify GET / serves the HTML frontend."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"CYBER WORLD MODEL", resp.data)

    def test_dataset_template_endpoint(self):
        """Verify GET /api/dataset/template returns columns and sample CSV."""
        resp = self.client.get("/api/dataset/template")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("columns", data)
        self.assertIn("sample_csv", data)

    def test_dataset_load_sample_endpoint(self):
        """Verify POST /api/dataset/load-sample loads scenarios correctly."""
        for scenario in ["kill_chain", "benign", "exfil"]:
            resp = self.client.post("/api/dataset/load-sample", json={"scenario": scenario})
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertGreater(data["num_flows"], 0)
            self.assertIn("preview", data)

    def test_dataset_upload_json_endpoint(self):
        """Verify POST /api/dataset/upload parses CSV text payload."""
        template_resp = self.client.get("/api/dataset/template")
        sample_csv = template_resp.get_json()["sample_csv"]
        resp = self.client.post("/api/dataset/upload", json={"csv_content": sample_csv})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["num_flows"], 2)

    def test_dataset_analyze_endpoint(self):
        """Verify POST /api/dataset/analyze runs end-to-end ML World Model analysis."""
        # Load sample kill_chain first
        self.client.post("/api/dataset/load-sample", json={"scenario": "kill_chain"})
        resp = self.client.post("/api/dataset/analyze")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("summary", data)
        self.assertIn("current_risk_score", data["summary"])
        self.assertIn("predicted_stage", data["summary"])
        self.assertIn("forecast_steps", data)
        self.assertEqual(len(data["forecast_steps"]), 5)
        self.assertIn("top_threat_drivers", data)
        self.assertIn("timeline", data)


if __name__ == "__main__":
    unittest.main()

