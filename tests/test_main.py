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


class WindowTextTest(unittest.TestCase):
    def test_text(self):
        from alerts.main import window_text
        self.assertEqual(window_text((22, 2)), "22:00–00:00")
        self.assertEqual(window_text((16, 5)), "16:00–21:00")
        self.assertEqual(window_text(None), "—")


class TargetTest(unittest.TestCase):
    def test_target_is_volunteer_only(self):
        from alerts.main import build
        days = pd.date_range("2025-09-01", periods=40)
        vol = pd.DataFrame({"oblast": "Kyiv City", "start": days + pd.Timedelta(hours=10),
                            "end": days + pd.Timedelta(hours=10, minutes=30), "source": "volunteer_csv"})
        official = pd.DataFrame({"oblast": ["Kyiv City"], "start": [pd.Timestamp("2025-09-05 02:00")],
                                 "end": [pd.Timestamp("2025-09-05 03:00")], "source": ["official_csv"]})
        inputs = {"target": vol, "intervals": pd.concat([vol, official], ignore_index=True),
                  "chomu": pd.DataFrame({"id": [], "time": pd.to_datetime([]), "text": []})}
        daily = build(inputs)["daily"]
        self.assertEqual(daily.loc["2025-09-05", "Kyiv City"], 1.0)  # the official alert is not counted


class ConsoleTest(unittest.TestCase):
    def test_printing_on_a_non_unicode_console_does_not_crash(self):
        # Windows consoles often use cp1252, which has no '→': the app crashed there on CI.
        import io
        import sys
        from unittest import mock
        from alerts.main import make_console_safe
        raw = io.BytesIO()
        console = io.TextIOWrapper(raw, encoding="cp1252")
        with mock.patch.object(sys, "stdout", console):
            make_console_safe()
            print("Backtest: 2026-09-13 → 2026-10-10 | 16:00–21:00 | —")
            console.flush()
        self.assertTrue(raw.getvalue().startswith(b"Backtest: 2026-09-13 ? 2026-10-10"))
