import unittest

import numpy as np
import pandas as pd

from prediction_model import prepare_features, predict_prices, time_based_split


class DummyModel:
    def __init__(self, value: float = 0.0):
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)


class PredictionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "Timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
                "Price": [10.0, 11.0, 12.0, 13.0, 14.0],
                "MaxTemp": [1.0, 2.0, 3.0, 4.0, 5.0],
                "IsWeekend": [False, False, False, False, False],
                "Humidity": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )

    def test_prepare_features_drops_timestamp_and_target(self):
        features, target = prepare_features(self.df, target_col="Price")
        self.assertNotIn("Timestamp", features.columns)
        self.assertNotIn("Price", features.columns)
        self.assertEqual(len(target), len(self.df))

    def test_prepare_features_casts_bool(self):
        features, _ = prepare_features(self.df, target_col="Price")
        self.assertTrue(np.issubdtype(features["IsWeekend"].dtype, np.integer))

    def test_time_based_split_invalid(self):
        features, target = prepare_features(self.df, target_col="Price")
        with self.assertRaises(ValueError):
            time_based_split(features, target, test_size=1.0)

    def test_predict_prices_uses_inference_prep(self):
        model = DummyModel(value=42.0)
        preds = predict_prices(model, self.df, target_col="Price")
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(preds == 42.0))


if __name__ == "__main__":
    unittest.main()
