"""Unit tests for session-based train/validation/test splitting."""

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.split import (
    session_train_test_split,
    assert_no_session_leakage,
    SessionSplitResult,
)
from src.data.dataset import StateNormalizer
from src.data.causal_windowing import NETWORK_STATE_FEATURES


class TestSessionSplit(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset with 10 distinct sessions
        records = []
        for s_idx in range(10):
            session_id = f"session_{s_idx:03d}"
            # 20 windows per session
            for w_idx in range(20):
                row = {
                    "session_id": session_id,
                    "window_start": w_idx * 30.0,
                    "window_end": (w_idx + 1) * 30.0,
                    "attack_stage": (w_idx // 4) % 6,
                    "is_attack": 1 if (w_idx // 4) % 6 > 0 else 0,
                }
                for feat in NETWORK_STATE_FEATURES:
                    row[feat] = float(s_idx * 100 + w_idx)
                records.append(row)
        self.df = pd.DataFrame(records)

    def test_session_isolation_zero_leakage(self):
        """Verify that train, validation, and test partitions share ZERO session IDs."""
        res = session_train_test_split(
            self.df,
            test_size=0.2,
            val_size=0.2,
            seed=42,
        )

        # Assert no session leakage
        assert_no_session_leakage(res)

        train_sessions = set(res.train_sessions)
        test_sessions = set(res.test_sessions)
        val_sessions = set(res.val_sessions)

        self.assertEqual(len(train_sessions.intersection(test_sessions)), 0)
        self.assertEqual(len(train_sessions.intersection(val_sessions)), 0)
        self.assertEqual(len(val_sessions.intersection(test_sessions)), 0)

        # Check that all records belong strictly to assigned sessions
        self.assertTrue(res.train_df["session_id"].isin(train_sessions).all())
        self.assertTrue(res.test_df["session_id"].isin(test_sessions).all())
        self.assertTrue(res.val_df["session_id"].isin(val_sessions).all())

        # Total rows match original
        total_rows = len(res.train_df) + len(res.test_df) + len(res.val_df)
        self.assertEqual(total_rows, len(self.df))

    def test_contrast_with_naive_random_window_split(self):
        """Demonstrate that naive random window shuffling severely leaks sessions.
        
        This test proves why random shuffling must NEVER be used for attack forecasting.
        """
        # Naive random split on rows/windows
        train_naive, test_naive = train_test_split(
            self.df,
            test_size=0.2,
            random_state=42,
        )

        naive_train_sessions = set(train_naive["session_id"])
        naive_test_sessions = set(test_naive["session_id"])
        leaked_sessions = naive_train_sessions.intersection(naive_test_sessions)

        # In naive window-level shuffling, almost 100% of sessions leak across train and test!
        self.assertGreater(
            len(leaked_sessions),
            0,
            "Naive random shuffling leaked sessions as expected.",
        )
        print(
            f"\n[Leakage Demonstration] Naive random shuffling caused {len(leaked_sessions)} "
            f"out of {len(naive_test_sessions)} test sessions to leak into training data!"
        )

    def test_state_normalizer_isolated_fitting(self):
        """Verify normalizer parameters are fit strictly on training sessions."""
        res = session_train_test_split(self.df, test_size=0.3, seed=42)

        normalizer = StateNormalizer().fit(res.train_df)

        # Transform both splits
        train_norm = normalizer.transform(res.train_df)
        test_norm = normalizer.transform(res.test_df)

        # Training data should have mean ~ 0 and std ~ 1
        np.testing.assert_allclose(np.mean(train_norm, axis=0), 0.0, atol=1e-3)
        np.testing.assert_allclose(np.std(train_norm, axis=0), 1.0, atol=1e-3)

        # Test data mean and std should differ from exactly 0 and 1, reflecting natural shift
        self.assertFalse(np.allclose(np.mean(test_norm, axis=0), 0.0, atol=1e-3))

    def test_invalid_parameters_raise_errors(self):
        """Test boundary conditions and error handling."""
        # Single session error
        single_df = self.df[self.df["session_id"] == "session_000"]
        with self.assertRaises(ValueError):
            session_train_test_split(single_df, test_size=0.2)

        # Invalid test_size
        with self.assertRaises(ValueError):
            session_train_test_split(self.df, test_size=1.5)

        # Sum of test_size and val_size >= 1.0
        with self.assertRaises(ValueError):
            session_train_test_split(self.df, test_size=0.6, val_size=0.5)


if __name__ == "__main__":
    unittest.main()
