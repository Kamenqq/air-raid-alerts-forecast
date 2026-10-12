import unittest

import numpy as np
import pandas as pd

from alerts.details import WINDOW_COLUMNS, durations, minutes_text


def alert(oblast, start, minutes, naive=False):
    start = pd.Timestamp(start)
    return (oblast, start, start + pd.Timedelta(minutes=minutes), naive)


class DurationsTest(unittest.TestCase):
    # forecast made on 2026-01-10 -> moment = 00:00 Kyiv = 2026-01-09 22:00 UTC
    VOL = pd.DataFrame([
        alert("Kyiv City", "2026-01-05 10:00", 10),
        alert("Kyiv City", "2026-01-06 10:00", 30),
        alert("Kyiv City", "2026-01-07 10:00", 50),
        alert("Kyiv City", "2026-01-08 10:00", 5, naive=True),   # made-up end: left out
        alert("Kyiv City", "2026-01-09 21:30", 60),              # still running at the moment: left out
        alert("Kyiv City", "2026-01-09 23:00", 7),               # after the moment: left out (leakage)
        alert("Kyiv City", "2025-12-01 10:00", 100),             # older than 28 days
    ], columns=["oblast", "start", "end", "naive"])

    def test_last_28_days_without_leakage(self):
        kyiv = durations(self.VOL, pd.Timestamp("2026-01-10")).loc["Kyiv City"]
        self.assertEqual(kyiv["alerts"], 3)
        self.assertEqual((kyiv["typical"], kyiv["shortest"], kyiv["average"], kyiv["longest"]), (30, 10, 30, 50))
        self.assertEqual(kyiv["p10"], 14)   # 10th percentile of 10, 30, 50

    def test_all_history_and_no_alerts(self):
        table = durations(self.VOL, pd.Timestamp("2026-01-10"), window=None)
        self.assertEqual(table.loc["Kyiv City", "alerts"], 4)
        self.assertEqual(table.loc["Lvivska oblast", "alerts"], 0)
        self.assertTrue(np.isnan(table.loc["Lvivska oblast", "typical"]))

    def test_minutes_text(self):
        self.assertEqual(minutes_text(17), "17 min")
        self.assertEqual(minutes_text(201), "3 h 21 min")
        self.assertEqual(minutes_text(float("nan")), "—")

    def test_window_columns_are_ascii(self):
        self.assertEqual(WINDOW_COLUMNS[0], "w00_02")
        self.assertEqual(WINDOW_COLUMNS[-1], "w22_00")
        self.assertTrue(all(name.isascii() for name in WINDOW_COLUMNS))
