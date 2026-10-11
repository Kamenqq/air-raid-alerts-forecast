import unittest
from unittest import mock

import pandas as pd

from alerts import model as model_module
from alerts.evaluate import backtest, bias, mae_table, naive_check, peak_hit_rate, test_period
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

    def test_peak_hit_rate(self):
        starts = pd.DataFrame({"oblast": ["Kyiv City"] * 4, "hour": [22, 23, 21, 5],
                               "day": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-10"])})
        self.assertEqual(peak_hit_rate(starts, pd.DatetimeIndex(["2026-03-10"])), 0.0)
        starts.loc[3, "hour"] = 22
        self.assertEqual(peak_hit_rate(starts, pd.DatetimeIndex(["2026-03-10"])), 1.0)

    def test_naive_check(self):
        vol = pd.DataFrame({"oblast": ["Kyiv City", "Kyiv City"], "naive": [False, True],
                            "start": pd.to_datetime(["2026-01-05 10:00", "2026-01-06 10:00"]),
                            "end": pd.to_datetime(["2026-01-05 10:50", "2026-01-06 10:30"])})
        table = naive_check(vol, {"Kyiv City": 40.0}, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"))
        self.assertEqual(list(table["MAE, minutes"]), [20.0, 10.0])
        self.assertEqual(list(table["alerts"]), [1, 1])
