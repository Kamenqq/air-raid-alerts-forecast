import unittest
from unittest import mock

import pandas as pd

from alerts import model as model_module
from alerts.evaluate import backtest, bias, calibration, chance_scores, mae_table, naive_check, test_period
from tests.helpers import synthetic_rows


class EvaluateTest(unittest.TestCase):
    def test_backtest_retrains_before_each_forecast_day(self):
        rows = synthetic_rows()
        days = test_period(pd.Timestamp("2026-02-28"))[-3:]
        with mock.patch("alerts.evaluate.train", side_effect=model_module.train) as spy:
            bt = backtest(rows, days)
        cutoffs = sorted(pd.Timestamp(call.kwargs["before_day"]) for call in spy.call_args_list)
        self.assertEqual(cutoffs, list(pd.DatetimeIndex(bt["t0"].unique()).sort_values()))
        self.assertTrue(bt["target_day"].isin(days).all())
        self.assertTrue((bt["t0"] <= bt["target_day"]).all())
        self.assertEqual(len(bt), 3 * 3 * 7)

    def test_mae_table_and_bias(self):
        bt = pd.DataFrame({"actual": [2, 4], "pred": [3, 4], "b1": [0, 4], "b2": [2, 0], "lead_days": [1, 2]})
        table = mae_table(bt)
        self.assertEqual(list(table.loc[0, ["model", "b1", "b2"]]), [0.5, 1.0, 2.0])
        self.assertAlmostEqual(table.loc[0, "skill_vs_best"], 0.5)
        self.assertEqual(bias(bt), 0.5)
        self.assertEqual(list(mae_table(bt, "lead_days")["model"]), [1.0, 0.0])

    def test_skill_undefined_when_a_baseline_is_perfect(self):
        # e.g. Zakarpattia: 0 alerts every day, both baselines say 0 -> skill is not defined (not -inf)
        bt = pd.DataFrame({"actual": [0, 0], "pred": [0.1, 0.1], "b1": [0, 0], "b2": [0, 0]})
        self.assertTrue(pd.isna(mae_table(bt).loc[0, "skill_vs_best"]))

    def test_naive_check(self):
        vol = pd.DataFrame({"oblast": ["Kyiv City", "Kyiv City"], "naive": [False, True],
                            "start": pd.to_datetime(["2026-01-05 10:00", "2026-01-06 10:00"]),
                            "end": pd.to_datetime(["2026-01-05 10:50", "2026-01-06 10:30"])})
        table = naive_check(vol, {"Kyiv City": 40.0}, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"))
        self.assertEqual(list(table["MAE, minutes"]), [20.0, 10.0])
        self.assertEqual(list(table["alerts"]), [1, 1])




class ChanceScoreTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from alerts.config import OBLASTS
        self.days = pd.date_range("2026-03-01", periods=40)
        self.cube = np.zeros((len(OBLASTS), 40, 24), dtype=bool)
        self.cube[0, :31, 22] = True        # Cherkaska: hour 22 on every day, incl. the test day
        self.bt = pd.DataFrame({"oblast": [0], "t0": [self.days[30]], "target_day": [self.days[30]],
                                "lead_days": [1], "pred": [1.0], "cnt_28d": [1.0]})

    def test_brier_scores(self):
        scores = chance_scores(self.bt, self.cube, self.days, leads=(1,))
        self.assertAlmostEqual(scores.loc[1, "hourly"], 0.0)              # predicted 100% at 22, 0% elsewhere
        flat = ((1 - 1 / 24) ** 2 + 23 * (1 / 24) ** 2) / 24              # "every hour the same" baseline
        self.assertAlmostEqual(scores.loc[1, "flat"], flat)
        at_22, elsewhere = 0.5 + 0.5 / 24, 0.5 / 24                       # shown = half pattern, half flat
        self.assertAlmostEqual(scores.loc[1, "shown"], ((1 - at_22) ** 2 + 23 * elsewhere ** 2) / 24)

    def test_calibration_table(self):
        table = calibration(self.bt, self.cube, self.days)
        self.assertEqual(int(table["hours"].sum()), 24)
        self.assertEqual(table.loc["50–60%", "observed"], 1.0)           # hour 22 shown as ~52%
        self.assertEqual(table.loc["0–10%", "observed"], 0.0)


class WindowCalibrationTest(unittest.TestCase):
    def test_two_hour_windows(self):
        from tests.test_main import small_state
        from alerts.evaluate import backtest, calibration
        state = small_state()
        bt = backtest(state["rows"], pd.date_range("2026-01-12", "2026-01-12"))
        table = calibration(bt, state["cube"], state["days"], two_hour=True)
        self.assertIn("windows", table.columns)
        self.assertEqual(table["windows"].sum(), 12 * (bt["lead_days"] == 1).sum())
        self.assertTrue(((table["observed"] >= 0) & (table["observed"] <= 1)).all())
