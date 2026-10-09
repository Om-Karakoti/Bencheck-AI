"""Unit tests verifying strict causal windowing, zero future leakage, and World Model dataset preparation."""

import unittest
import numpy as np
import pandas as pd

from src.data.config import AttackStage, WindowConfig, SCHEMA_COLUMNS
from src.data.causal_windowing import (
    CausalNetworkStateExtractor,
    NETWORK_STATE_FEATURES,
    load_and_window_csv,
)
from src.data.dataset import WorldModelSequenceDataset, StateNormalizer
from src.data.synthetic import generate_synthetic_attack_scenario


class TestCausalWindowing(unittest.TestCase):
    def setUp(self):
        self.config = WindowConfig(
            window_size_sec=30.0,
            step_size_sec=30.0,
            fill_empty_windows=True,
            history_length=4,
            forecast_horizon=2,
        )
        self.extractor = CausalNetworkStateExtractor(self.config)

    def _create_minimal_row(
        self,
        session_id: str = "sess_1",
        timestamp: float = 10.0,
        bytes_per_flow: int = 1000,
        packets_per_flow: int = 10,
        syn_count: int = 1,
        ack_count: int = 5,
        attack_stage: str = "Benign",
        attack_label: str = "Normal",
        dst_port: int = 80,
    ) -> dict:
        return {
            "session_id": session_id,
            "timestamp": timestamp,
            "src_ip": "192.168.1.50",
            "dst_ip": "10.0.0.1",
            "src_port": 54321,
            "dst_port": dst_port,
            "protocol": 6,
            "syn_count": syn_count,
            "ack_count": ack_count,
            "fin_count": 1,
            "rst_count": 0,
            "bytes_per_flow": bytes_per_flow,
            "packets_per_flow": packets_per_flow,
            "flow_duration": 1.5,
            "iat_mean": 0.1,
            "iat_var": 0.01,
            "iat_max": 0.3,
            "ttl_variance": 0.05,
            "tcp_window_size": 65535,
            "fragment_flag": 0,
            "payload_size_mean": 100.0,
            "retransmit_count": 0,
            "scan_signature_score": 0.05,
            "attack_label": attack_label,
            "attack_stage": attack_stage,
        }

    def test_strict_causality_zero_future_leakage(self):
        """CRITICAL TEST: Ensure records at t+1 or later NEVER alter state computed at time t.
        
        This verifies that any windowed statistic for state at time t only uses
        data with timestamp <= t.
        """
        # Session A: Has events at t=5.0, 15.0, 25.0 (within window 1: (0, 30])
        rows_base = [
            self._create_minimal_row(timestamp=5.0, bytes_per_flow=500),
            self._create_minimal_row(timestamp=15.0, bytes_per_flow=1500),
            self._create_minimal_row(timestamp=25.0, bytes_per_flow=2000),
        ]
        df_base = pd.DataFrame(rows_base)[SCHEMA_COLUMNS]

        res_base = self.extractor.process(df_base)
        state_at_t30_base = res_base[res_base["window_end"] == 30.0].iloc[0]

        # Session B: Identical to Session A up to t=30.0, but injects a massive
        # future attack burst at t=30.0001, t=35.0, t=55.0
        rows_with_future = rows_base + [
            self._create_minimal_row(
                timestamp=30.0001,
                bytes_per_flow=9999999,
                syn_count=500,
                attack_stage="Exfiltration",
                attack_label="MassiveLeak",
            ),
            self._create_minimal_row(
                timestamp=35.0,
                bytes_per_flow=5000000,
                attack_stage="Exfiltration",
            ),
            self._create_minimal_row(
                timestamp=59.0,
                bytes_per_flow=8000000,
                attack_stage="Exfiltration",
            ),
        ]
        df_with_future = pd.DataFrame(rows_with_future)[SCHEMA_COLUMNS]

        res_with_future = self.extractor.process(df_with_future)
        state_at_t30_with_future = res_with_future[res_with_future["window_end"] == 30.0].iloc[0]

        # The state vector at t=30.0 MUST BE 100% IDENTICAL across all features
        for feat in NETWORK_STATE_FEATURES:
            val_base = state_at_t30_base[feat]
            val_future = state_at_t30_with_future[feat]
            self.assertAlmostEqual(
                val_base,
                val_future,
                places=5,
                msg=f"Causality violation! Feature '{feat}' at t=30 was modified by data at t > 30 ({val_base} != {val_future})",
            )

        # Labels at t=30.0 must remain Benign and unaffected by future stages
        self.assertEqual(state_at_t30_base["attack_stage"], state_at_t30_with_future["attack_stage"])
        self.assertEqual(state_at_t30_with_future["attack_stage"], int(AttackStage.BENIGN))

        # But the state at t=60.0 SHOULD reflect the future flows
        state_at_t60 = res_with_future[res_with_future["window_end"] == 60.0].iloc[0]
        self.assertEqual(state_at_t60["attack_stage"], int(AttackStage.EXFILTRATION))
        self.assertGreater(state_at_t60["total_bytes"], 10_000_000)

    def test_window_boundary_edge_cases(self):
        """Test exact window boundaries: [0, 30] for first window, (30, 60] for next.
        
        An event at exactly timestamp t_k is included in the window ending at t_k.
        An event just past t_k is included in the subsequent window.
        """
        rows = [
            self._create_minimal_row(timestamp=10.0, bytes_per_flow=500),  # Inside Window 1: [0, 30]
            self._create_minimal_row(timestamp=30.0, bytes_per_flow=700),  # Exactly at boundary t=30 (Inside Window 1)
            self._create_minimal_row(timestamp=30.001, bytes_per_flow=900),# Just past t=30 (Inside Window 2: (30, 60])
            self._create_minimal_row(timestamp=60.0, bytes_per_flow=300),  # Exactly at boundary t=60 (Inside Window 2)
        ]
        df = pd.DataFrame(rows)[SCHEMA_COLUMNS]
        res = self.extractor.process(df)

        win1 = res[res["window_end"] == 30.0].iloc[0]
        # Should include t=10.0 and t=30.0 -> 500 + 700 = 1200 bytes, flow_count = 2
        self.assertEqual(win1["flow_count"], 2)
        self.assertEqual(win1["total_bytes"], 1200)

        win2 = res[res["window_end"] == 60.0].iloc[0]
        # Should include t=30.001 and t=60.0 -> 900 + 300 = 1200 bytes, flow_count = 2
        self.assertEqual(win2["flow_count"], 2)
        self.assertEqual(win2["total_bytes"], 1200)

    def test_session_isolation(self):
        """Test that different session_ids are processed completely independently."""
        rows = [
            self._create_minimal_row(session_id="session_A", timestamp=10.0, bytes_per_flow=100),
            self._create_minimal_row(session_id="session_B", timestamp=10.0, bytes_per_flow=9999),
        ]
        df = pd.DataFrame(rows)[SCHEMA_COLUMNS]
        res = self.extractor.process(df)

        res_a = res[res["session_id"] == "session_A"].iloc[0]
        res_b = res[res["session_id"] == "session_B"].iloc[0]

        self.assertEqual(res_a["total_bytes"], 100)
        self.assertEqual(res_b["total_bytes"], 9999)

    def test_empty_window_handling(self):
        """Test that quiescent intervals generate clean 0-traffic vectors when fill_empty_windows=True."""
        rows = [
            self._create_minimal_row(timestamp=10.0, bytes_per_flow=500),
            # Gap of 60 seconds (empty window between 30 and 60)
            self._create_minimal_row(timestamp=70.0, bytes_per_flow=800),
        ]
        df = pd.DataFrame(rows)[SCHEMA_COLUMNS]
        res = self.extractor.process(df)

        # We expect windows at t=30, t=60, t=90 (or t_max)
        self.assertGreaterEqual(len(res), 3)
        win_empty = res[res["window_end"] == 60.0].iloc[0]
        self.assertEqual(win_empty["flow_count"], 0)
        self.assertEqual(win_empty["total_bytes"], 0)
        self.assertEqual(win_empty["attack_stage"], int(AttackStage.BENIGN))

    def test_world_model_dataset_sequences(self):
        """Test sequence generation (L steps history -> H steps forecast) for World Models."""
        # Generate a synthetic multi-stage scenario
        df_synthetic = generate_synthetic_attack_scenario(
            session_id="test_session",
            start_time=0.0,
            stage_duration_sec=60.0,  # 6 stages * 60s = 360s -> ~12 windows of 30s
            flows_per_minute=100,
            seed=42,
        )
        windowed_df = self.extractor.process(df_synthetic)
        self.assertGreater(len(windowed_df), 10)

        # Fit normalizer and build dataset
        normalizer = StateNormalizer().fit(windowed_df)
        dataset = WorldModelSequenceDataset(
            windowed_df=windowed_df,
            config=self.config,  # L=4, H=2
            normalizer=normalizer,
        )

        self.assertGreater(len(dataset), 0)
        sample = dataset[0]

        # Verify shapes
        # L = 4 history steps, D = 33 features
        num_features = len(NETWORK_STATE_FEATURES)
        self.assertEqual(sample["history_states"].shape, (4, num_features))
        self.assertEqual(sample["history_stages"].shape, (4,))
        # H = 2 forecast horizon steps
        self.assertEqual(sample["future_states"].shape, (2, num_features))
        self.assertEqual(sample["future_stages"].shape, (2,))

        # Verify batch extraction
        batch = dataset.get_batch([0, 1])
        self.assertEqual(batch["x_history"].shape, (2, 4, num_features))
        self.assertEqual(batch["y_future_state"].shape, (2, 2, num_features))
        self.assertEqual(batch["y_future_stage"].shape, (2, 2))

        # Verify PyTorch Dataset adapter if torch is available
        try:
            torch_ds = dataset.to_torch_dataset()
            self.assertEqual(len(torch_ds), len(dataset))
            item = torch_ds[0]
            self.assertEqual(item["x_history"].shape, (4, num_features))
            self.assertEqual(item["future_states"].shape, (2, num_features))
        except (ImportError, ModuleNotFoundError):
            pass


if __name__ == "__main__":
    unittest.main()
