import unittest

import numpy as np
import pandas as pd

from alerts.main import check_start, parse_slot, parse_start


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


class TargetTest(unittest.TestCase):
    def test_target_is_volunteer_only(self):
        from alerts.main import build
        days = pd.date_range("2025-09-01", periods=40)
        vol = pd.DataFrame({"oblast": "Kyiv City", "start": days + pd.Timedelta(hours=10),
                            "end": days + pd.Timedelta(hours=10, minutes=30), "source": "volunteer_csv"})
        official = pd.DataFrame({"oblast": ["Kyiv City"], "start": [pd.Timestamp("2025-09-05 02:00")],
                                 "end": [pd.Timestamp("2025-09-05 03:00")], "source": ["official_csv"]})
        inputs = {"intervals": pd.concat([vol, official], ignore_index=True),  # build() must filter it
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


class EmptyStartTest(unittest.TestCase):
    def test_empty_start_is_an_error_not_a_full_run(self):
        from alerts.main import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--start", ""])
        self.assertIn("YYYY-MM-DD", str(ctx.exception.code))


class DaysOptionTest(unittest.TestCase):
    def test_days_also_applies_without_start(self):
        from unittest import mock
        import alerts.main as M
        with mock.patch.object(M, "load_inputs", return_value={}), mock.patch.object(M, "build", return_value={}), \
                mock.patch.object(M, "run_evaluation") as evaluation:
            M.main(["--days", "3"])
        self.assertEqual(evaluation.call_args.args[2], 3)



def small_state():
    """Kyiv City has 1-3 alerts every day from 2025-11-01 to 2026-01-20; other oblasts have none."""
    from alerts.main import build
    rng = np.random.RandomState(0)
    rows = []
    for day in pd.date_range("2025-11-01", "2026-01-20"):
        for n in range(rng.randint(1, 4)):
            start = day + pd.Timedelta(hours=8 + 3 * n)
            rows.append(("Kyiv City", start, start + pd.Timedelta(minutes=20), "volunteer_csv"))
    intervals = pd.DataFrame(rows, columns=["oblast", "start", "end", "source"])
    return build({"intervals": intervals, "chomu": pd.DataFrame({"id": [], "time": pd.to_datetime([]), "text": []})})


class ForecastCutoffTest(unittest.TestCase):
    def test_past_forecast_trains_only_on_days_before_start(self):
        from unittest import mock
        import alerts.main as M
        from alerts import model as model_module
        start = pd.Timestamp("2026-01-10")
        with mock.patch.object(M, "train", side_effect=model_module.train) as spy:
            fc = M.forecast(small_state(), start, 3)
        self.assertEqual(spy.call_args.kwargs["before_day"], start)
        self.assertTrue(fc["actual"].notna().all())  # a past start: the real outcome is shown next to it


class SmallFixesTest(unittest.TestCase):
    def test_skill_text(self):
        from alerts.main import skill_text
        self.assertEqual(skill_text(-0.001), "+0%")
        self.assertEqual(skill_text(0.123), "+12%")
        self.assertEqual(skill_text(float("nan")), "n/a")

    def test_forecast_csv_opens_in_excel(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        import alerts.main as M
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M.C, "REPORTS_DIR", Path(tmp)):
            path = M.save_forecast(pd.DataFrame({"main": ["16:00–21:00"]}), pd.Timestamp("2026-10-12"))
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))  # UTF-8 mark Excel needs


class SlotTest(unittest.TestCase):
    def test_parse_slot(self):
        self.assertEqual(parse_slot("08:00-09:30"), [8, 9])
        self.assertEqual(parse_slot("08:00-10:00"), [8, 9])
        self.assertEqual(parse_slot("23:00-01:00"), [23, 0])
        self.assertEqual(parse_slot("8:15-8:45"), [8])
        for bad in ("09:00-09:00", "25:00-26:00", "evening"):
            with self.assertRaises(SystemExit):
                parse_slot(bad)


class ForecastOutputTest(unittest.TestCase):
    def test_hourly_chances_slots_and_slot_query(self):
        import alerts.main as M
        fc = M.forecast(small_state(), pd.Timestamp("2026-01-10"), 2, slot=[8, 9])
        for column in ["chance_00", "chance_23", "quietest_2h", "riskiest_2h", "slot_chance"]:
            self.assertIn(column, fc.columns)
        kyiv = fc[fc["oblast"] == "Kyiv City"].iloc[0]
        hours = [kyiv[f"chance_{h:02d}"] for h in range(24)]
        self.assertEqual(int(np.argmax(hours)), 10)        # an alert starts at 10:00 Kyiv every day
        self.assertTrue(0 <= kyiv["slot_chance"] <= 1)
        from alerts.features import blended_chance        # day 2: same chances, no count scaling
        state = small_state()
        day2 = fc[(fc["oblast"] == "Kyiv City") & (fc["day"] == fc["day"].max())].iloc[0]
        expected = blended_chance(state["cube"], state["days"], "Kyiv City", pd.Timestamp("2026-01-10"))
        np.testing.assert_allclose([day2[f"chance_{h:02d}"] for h in range(24)], expected)
        quiet = fc[fc["oblast"] == "Lvivska oblast"].iloc[0]
        self.assertEqual(quiet["quietest_2h"], "—")       # no alerts at all in its history


def run_menu(answers, state=None, **patches):
    """Run the menu with typed answers; returns (printed text, mocks)."""
    import contextlib
    import io
    import tempfile
    from pathlib import Path
    from unittest import mock
    import alerts.main as M
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M.C, "REPORTS_DIR", Path(tmp)), \
            mock.patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(out), \
            mock.patch.object(M, "run_evaluation") as evaluation, \
            mock.patch.object(M, "load_inputs", **patches.get("load_inputs", {})) as load:
        M.menu({}, state if state is not None else small_state())
    return out.getvalue(), evaluation, load


class MenuTest(unittest.TestCase):
    def test_no_menu_without_a_keyboard(self):
        # CI and scripts have no person typing: they must get the full run, never a waiting menu
        from unittest import mock
        import alerts.main as M
        with mock.patch.object(M, "load_inputs", return_value={}), mock.patch.object(M, "build", return_value={}), \
                mock.patch.object(M, "run_evaluation") as evaluation, mock.patch.object(M, "menu") as menu, \
                mock.patch.object(M.sys.stdin, "isatty", return_value=False):
            M.main([])
        evaluation.assert_called_once()
        menu.assert_not_called()

    def test_menu_for_a_person_without_options(self):
        from unittest import mock
        import alerts.main as M
        with mock.patch.object(M, "load_inputs", return_value={}), mock.patch.object(M, "build", return_value={}), \
                mock.patch.object(M, "menu") as menu, mock.patch.object(M, "run_evaluation") as evaluation, \
                mock.patch.object(M, "keyboard", return_value=True):
            M.main([])
        menu.assert_called_once()
        evaluation.assert_not_called()  # the menu decides what runs

    def test_enter_runs_the_evaluation_and_q_quits(self):
        text, evaluation, _ = run_menu(["", "q"])
        evaluation.assert_called_once()
        self.assertIn("1  Test the model on the last 28 days", text)

    def test_forecast_from_a_date_asks_again_after_wrong_answers(self):
        text, _, _ = run_menu(["2", "2025-01-01", "2026-01-10", "9", "2", "", "q"])
        self.assertIn("between 2026-01-01 and 2026-01-20", text)  # out-of-range date: asked again
        self.assertIn("from 1 to 7", text)                        # 9 days: asked again
        self.assertIn("→ same as: python3 run.py --start 2026-01-10 --days 2\n", text)
        self.assertIn("Expected number of alerts per day", text)

    def test_time_slot_with_default_date_and_days(self):
        text, _, _ = run_menu(["3", "8-9", "08:00-09:30", "", "", "q"])
        self.assertIn("--slot must look like 08:00-09:30", text)
        self.assertIn("→ same as: python3 run.py --start 2026-01-20 --days 7 --slot 08:00-09:30", text)
        self.assertIn("Chance of an alert at some moment during 08:00-09:30", text)

    def test_unknown_choice_failed_download_and_ctrl_d_do_not_crash(self):
        text, evaluation, load = run_menu(["7", "4", EOFError()],
                                          load_inputs={"side_effect": RuntimeError("Download failed after 3 attempts")})
        self.assertIn("Please type 1-5 or q", text)
        self.assertIn("Download failed after 3 attempts", text)
        load.assert_called_once_with(True)
        evaluation.assert_not_called()  # no fresh data: nothing to evaluate
