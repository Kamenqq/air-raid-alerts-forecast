import unittest

import numpy as np
import pandas as pd

from alerts.model import predict, train, training_rows
from tests.helpers import synthetic_rows


class ModelTest(unittest.TestCase):
    def test_training_rows_respect_cutoff(self):
        rows = synthetic_rows()
        cutoff = pd.Timestamp("2026-02-01")
        used = training_rows(rows, cutoff)
        self.assertLess(used["target_day"].max(), cutoff)
        self.assertGreater(len(used), 0)

    def test_train_and_predict(self):
        rows = synthetic_rows()
        model = train(rows, pd.Timestamp("2026-02-20"))
        test = rows[rows["t0"] == pd.Timestamp("2026-02-20")]
        pred = predict(model, test)
        self.assertTrue((pred > 0).all())
        self.assertGreater(np.corrcoef(pred, test["cnt_7d"])[0, 1], 0.5)  # learned the real signal

    def test_too_little_history_is_an_error(self):
        with self.assertRaises(ValueError):
            train(synthetic_rows(), pd.Timestamp("2026-01-03"))

    def test_all_empty_feature_is_left_out(self):
        # scikit-learn >= 1.7 cannot bin a column with no known values (crashed CI on Python 3.13)
        rows = synthetic_rows()
        rows["tg_drones_1d"] = np.nan
        model = train(rows, pd.Timestamp("2026-02-20"))
        self.assertNotIn("tg_drones_1d", model.features_)
        self.assertTrue((predict(model, rows[rows["t0"] == pd.Timestamp("2026-02-20")]) > 0).all())
