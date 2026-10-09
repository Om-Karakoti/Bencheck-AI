"""Unit tests for the stable inference interface: predict(window_sequence)."""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.models.mitre_mapper import CANONICAL_STAGES
from src.inference import (
    predict,
    NetworkAttackPredictor,
    set_default_predictor,
    get_default_predictor,
)


class TestInferenceInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.num_features = len(NETWORK_STATE_FEATURES)
        cls.seq_len = 6

        # 1. 2D NumPy array (L, 36)
        cls.arr_2d = np.random.randn(cls.seq_len, cls.num_features).astype(np.float32) * 0.1
        # 2. 3D NumPy array (1, L, 36)
        cls.arr_3d = np.expand_dims(cls.arr_2d, axis=0)
        # 3. Pandas DataFrame (L, 36)
        cls.df_seq = pd.DataFrame(cls.arr_2d, columns=NETWORK_STATE_FEATURES)
        # 4. List of Dictionaries
        cls.list_dicts = cls.df_seq.to_dict(orient="records")
        # 5. List of Lists
        cls.list_lists = cls.arr_2d.tolist()

    def test_stable_contract_and_keys(self):
        """CRITICAL TEST: Assert exact dictionary contract and types expected by backend."""
        res = predict(self.df_seq)

        self.assertIsInstance(res, dict)
        self.assertIn("infiltration_probability", res)
        self.assertIn("predicted_stage", res)
        self.assertIn("top_features", res)

        # Assert types
        self.assertIsInstance(res["infiltration_probability"], float)
        self.assertIsInstance(res["predicted_stage"], str)
        self.assertIsInstance(res["top_features"], list)

        # Assert value bounds
        self.assertGreaterEqual(res["infiltration_probability"], 0.0)
        self.assertLessEqual(res["infiltration_probability"], 1.0)
        self.assertIn(res["predicted_stage"], CANONICAL_STAGES)
        self.assertGreater(len(res["top_features"]), 0)
        for feat in res["top_features"]:
            self.assertIsInstance(feat, str)
            self.assertIn(feat, NETWORK_STATE_FEATURES)

    def test_input_types_flexibility(self):
        """Verify predict() accepts DataFrame, 2D array, 3D array, List of dicts, List of lists."""
        # 1. 2D Array
        res_2d = predict(self.arr_2d)
        self.assertEqual(sorted(res_2d.keys()), ["infiltration_probability", "predicted_stage", "top_features"])

        # 2. 3D Array
        res_3d = predict(self.arr_3d)
        self.assertEqual(sorted(res_3d.keys()), ["infiltration_probability", "predicted_stage", "top_features"])

        # 3. List of Dictionaries
        res_dicts = predict(self.list_dicts)
        self.assertEqual(sorted(res_dicts.keys()), ["infiltration_probability", "predicted_stage", "top_features"])

        # 4. List of Lists
        res_lists = predict(self.list_lists)
        self.assertEqual(sorted(res_lists.keys()), ["infiltration_probability", "predicted_stage", "top_features"])

    def test_predictor_persistence_and_reload(self):
        """Verify save() and load() for NetworkAttackPredictor."""
        predictor = get_default_predictor()
        res_orig = predictor.predict(self.arr_2d)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_predictor.joblib")
            predictor.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            loaded_predictor = NetworkAttackPredictor.load(save_path)
            res_loaded = loaded_predictor.predict(self.arr_2d)

            self.assertAlmostEqual(res_orig["infiltration_probability"], res_loaded["infiltration_probability"], places=3)
            self.assertEqual(res_orig["predicted_stage"], res_loaded["predicted_stage"])
            self.assertEqual(res_orig["top_features"], res_loaded["top_features"])

    def test_invalid_dimensions_raise_error(self):
        """Verify mismatched feature dimensions raise clear ValueErrors."""
        bad_arr = np.zeros((self.seq_len, 10))  # Only 10 features instead of 36
        with self.assertRaises(ValueError):
            predict(bad_arr)


if __name__ == "__main__":
    unittest.main()
