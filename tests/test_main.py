import unittest

import numpy as np
import pandas as pd

from alerts.main import check_start, parse_start, peak_hours_text


class StartDateTest(unittest.TestCase):
    def test_bad_format(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_start("10/12/2026")
        self.assertIn("YYYY-MM-DD", str(ctx.exception.code))

    def test_range(self):
        end_day = pd.Timestamp("2026-10-11")
        check_start(pd.Timestamp("2026-10-11"), end_day)  # last allowed day: no error
        for bad in ("2025-12-31", "2026-10-12"):
            with self.assertRaises(SystemExit) as ctx:
                check_start(pd.Timestamp(bad), end_day)
            self.assertIn("between 2026-01-01 and 2026-10-11", str(ctx.exception.code))


class PeakHoursTest(unittest.TestCase):
    def test_top_three_hours_in_order(self):
        profile = np.zeros(24)
        profile[[23, 2, 19]] = [0.5, 0.3, 0.2]
        self.assertEqual(peak_hours_text(profile), "02:00, 19:00, 23:00")

    def test_no_recent_alerts_gives_dash(self):
        self.assertEqual(peak_hours_text(np.full(24, 1 / 24)), "—")
