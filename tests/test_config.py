import unittest

import pandas as pd

from alerts.config import kyiv_day, kyiv_day_start_utc, to_kyiv


class KyivTimeTest(unittest.TestCase):
    def test_summer_and_winter_offsets(self):
        utc = pd.Series(pd.to_datetime(["2026-07-01 21:30", "2026-01-01 22:30"]))
        self.assertEqual(list(to_kyiv(utc)), list(pd.to_datetime(["2026-07-02 00:30", "2026-01-02 00:30"])))

    def test_day_helpers(self):
        self.assertEqual(kyiv_day(pd.Timestamp("2026-07-01 21:30")), pd.Timestamp("2026-07-02"))
        self.assertEqual(kyiv_day_start_utc("2026-07-02"), pd.Timestamp("2026-07-01 21:00"))
        self.assertEqual(kyiv_day_start_utc("2026-01-02"), pd.Timestamp("2026-01-01 22:00"))
