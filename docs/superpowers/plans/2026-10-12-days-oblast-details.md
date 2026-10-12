# Choose the days, one oblast, details — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a person ask "Wednesday, Lviv, and how long do alerts last there?": choose the forecast days (`--for`,
start picked automatically), one oblast (`--oblast`), and details (12 two-hour windows + alert durations,
`--details`), with both new outputs checked once on the test period.

**Architecture:** Forecasting itself does not change. `alerts/main.py` gains the options, an automatic start
(`pick_days`), oblast matching (`match_oblast`), one place that filters/prints/saves what was asked (`present`),
and the menu questions. New `alerts/details.py` holds the window columns, duration statistics and the duration
check; `alerts/evaluate.calibration` learns 2-hour windows.

**Tech Stack:** Python 3.9+, pandas, numpy, unittest (`python3 run.py test`), GitHub Actions CI.

**Spec:** `docs/superpowers/specs/2026-10-12-days-oblast-details-design.md`

## Global Constraints

- Python 3.9+ on macOS / Linux / Windows; no new dependencies (pandas, numpy, scikit-learn, matplotlib only).
- No leakage: a forecast made on `start` uses only information from before 00:00 Kyiv of `start`.
- The model forecasts at most `C.MAX_LEAD = 7` days ahead (start day + 6).
- Never tune on the test period: the two new checks are measured once and reported as they are.
- `python3 run.py` without a keyboard (CI, scripts) must keep doing the evaluation run with the same numbers.
- CI green on 3 OS × Python 3.9 / 3.13 / 3.14 / 3.15 before every merge into `main`.
- Each task ends with a risk report + self-check commands to the user; wait for "go" before the next task.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A common English oblast name that is not the official spelling (`odesa`, `sumy`) → a message listing all
  oblast names, not a crash (test in Task 2).
- A range that starts in the past and ends beyond reach (`--for 2026-10-05 2026-10-14`, data to 10-11) →
  refused with the furthest reachable day (test in Task 1).
- An oblast with no alerts in the last 28 days (Zakarpatska) in the details → `—` and a "No alerts" line, no
  NaN crash (test in Task 3).
- `--for` before 2026-01-01 → the usual range message, not a traceback (test in Task 1).
- New CSV column names stay ASCII (`w00_02` …, `dur_…`), so Excel and Windows consoles cope (test in Task 3).

---

### Task 0: Review and merge the menu branch

**Files:** none planned (fixes only if the review finds something).

- [ ] **Step 1:** Run the `code-review` skill on the branch diff `main..menu` (effort: medium).
- [ ] **Step 2:** For each confirmed finding: write a failing test in `tests/test_main.py`, fix in `alerts/main.py`,
  run `python3 run.py test` (expect all OK), commit.
- [ ] **Step 3:** Push `menu`, wait for CI (12 jobs green), then:

```bash
git switch main && git merge --ff-only menu && git push origin main && git switch -c features
```

- [ ] **Step 4:** Report to the user (findings, fixes, CI), wait for "go".

---

### Task 1: Choose the days (`--for`) with an automatic start

**Files:**
- Modify: `alerts/main.py` (`parse_start`, new `pick_days`, `show_forecast`, new `present`, `run_evaluation` end,
  `menu_forecast`, `main`; delete `parse_days`)
- Test: `tests/test_main.py`
- Docs: `README.md` (options table), `HOW_TO_USE.md` (Step 3 menu text, Step 5)

**Interfaces:**
- Produces: `parse_start(text, option="--start") -> pd.Timestamp`;
  `pick_days(first, last, start, end_day) -> (pd.Timestamp start, int n_days)`;
  `present(fc, start, slot_label=None, first=None) -> pd.DataFrame` (the shown rows);
  `show_forecast(state, start, n_days, slot_label=None, first=None) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_main.py`)

```python
class PickDaysTest(unittest.TestCase):
    END = pd.Timestamp("2026-10-11")

    def pick(self, first, last=None, start=None):
        from alerts.main import pick_days
        return pick_days(pd.Timestamp(first), pd.Timestamp(last or first),
                         pd.Timestamp(start) if start else None, self.END)

    def refused(self, *args):
        with self.assertRaises(SystemExit) as ctx:
            self.pick(*args)
        return str(ctx.exception.code)

    def test_future_day_starts_at_the_last_data_day(self):
        self.assertEqual(self.pick("2026-10-14"), (self.END, 4))   # forecast 11..14 Oct, show 14 Oct

    def test_past_days_start_at_the_first_chosen_day(self):
        self.assertEqual(self.pick("2026-10-01", "2026-10-03"), (pd.Timestamp("2026-10-01"), 3))

    def test_manual_start(self):
        self.assertEqual(self.pick("2026-10-14", None, "2026-10-10"), (pd.Timestamp("2026-10-10"), 5))
        self.assertIn("after the first chosen day", self.refused("2026-10-14", None, "2026-10-15"))

    def test_too_far(self):
        self.assertIn("reaches 2026-10-17 at most", self.refused("2026-10-18"))
        self.assertIn("reaches 2026-10-11 at most", self.refused("2026-10-05", "2026-10-14"))

    def test_bad_ranges(self):
        self.assertIn("before the first day", self.refused("2026-10-14", "2026-10-13"))
        self.assertIn("between 2026-01-01 and 2026-10-11", self.refused("2025-12-30"))


def run_main(args):
    """main() on the small test state; returns (printed text, train spy, saved file names)."""
    import contextlib
    import io
    import tempfile
    from pathlib import Path
    from unittest import mock
    import alerts.main as M
    from alerts import model as model_module
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M.C, "REPORTS_DIR", Path(tmp)), \
            mock.patch.object(M, "load_inputs", return_value={}), \
            mock.patch.object(M, "build", return_value=small_state()), \
            mock.patch.object(M, "train", side_effect=model_module.train) as spy, contextlib.redirect_stdout(out):
        M.main(args)
        saved = sorted(p.name for p in Path(tmp).iterdir())
    return out.getvalue(), spy, saved


class ForDaysTest(unittest.TestCase):
    def test_future_day_uses_the_newest_data_and_shows_only_that_day(self):
        text, spy, saved = run_main(["--for", "2026-01-22"])       # small_state data ends 2026-01-20
        self.assertEqual(spy.call_args.kwargs["before_day"], pd.Timestamp("2026-01-20"))
        self.assertIn("Forecast made on 2026-01-20", text)
        self.assertNotIn("2026-01-21", text)
        self.assertEqual(saved, ["forecast_2026-01-22.csv"])

    def test_past_days_use_only_data_before_them(self):
        _, spy, _ = run_main(["--for", "2026-01-10", "2026-01-12"])
        self.assertEqual(spy.call_args.kwargs["before_day"], pd.Timestamp("2026-01-10"))  # leakage guard

    def test_wrong_combinations_are_refused(self):
        for args, message in ((["--for", "2026-01-22", "--days", "3"], "not both"),
                              (["--for", "2026-01-10", "2026-01-11", "2026-01-12"], "one day or two"),
                              (["--for", "14.10.2026"], "--for must be a date")):
            with self.assertRaises(SystemExit) as ctx:
                run_main(args)
            self.assertIn(message, str(ctx.exception.code))
```

Also replace the two menu tests that used the old "How many days" question:

```python
    def test_forecast_from_a_date_asks_again_after_wrong_answers(self):
        text, _, _ = run_menu(["2", "2025-01-01", "2026-01-28", "2026-01-10 2026-01-11", "2026-01-12", "", "", "q"])
        self.assertIn("between 2026-01-01 and 2026-01-20", text)   # day before the allowed range
        self.assertIn("reaches 2026-01-26 at most", text)          # too far ahead
        self.assertIn("after the first chosen day", text)          # manual start after the chosen days
        self.assertIn("→ same as: python3 run.py --for 2026-01-10 2026-01-11\n", text)
        self.assertIn("Expected number of alerts per day", text)

    def test_time_slot_with_default_days_and_start(self):
        text, _, _ = run_menu(["3", "8-9", "08:00-09:30", "", "", "q"])
        self.assertIn("--slot must look like 08:00-09:30", text)
        self.assertIn("→ same as: python3 run.py --for 2026-01-20 2026-01-26 --slot 08:00-09:30", text)
        self.assertIn("Chance of an alert at some moment during 08:00-09:30", text)
```

(Delete the old `test_time_slot_with_default_date_and_days`.)

- [ ] **Step 2: Run, expect failures**

Run: `python3 run.py test 2>&1 | grep -E "^(FAIL|ERROR)|^Ran|^OK|^FAILED"`
Expected: FAILED — `ImportError: cannot import name 'pick_days'`, `unrecognized arguments: --for`, menu texts.

- [ ] **Step 3: Implement in `alerts/main.py`**

`parse_start` takes the option name for its message:

```python
def parse_start(text, option="--start"):
    try:
        day = pd.to_datetime(text, format="%Y-%m-%d")
    except (ValueError, TypeError):
        day = pd.NaT
    if pd.isna(day):  # e.g. "" gives NaT instead of an error
        raise SystemExit(f"{option} must be a date like 2026-10-05 (YYYY-MM-DD), got: {text!r}")
    return pd.Timestamp(day).normalize()
```

New, after `check_start`:

```python
def pick_days(first, last, start, end_day):
    """--for FIRST LAST -> (start, number of days to forecast). Without a manual start the forecast is made at
    the latest moment that is not after the first chosen day: future days get the newest data, past days only
    data from before them."""
    if last < first:
        raise SystemExit(f"--for: the last day {last.date()} is before the first day {first.date()}.")
    if start is None:
        start = min(first, end_day)
    elif start > first:
        raise SystemExit(f"--start {start.date()} is after the first chosen day {first.date()}: the forecast "
                         "would use information from a chosen day.")
    check_start(start, end_day)
    furthest = start + pd.Timedelta(days=C.MAX_LEAD - 1)
    if last > furthest:
        raise SystemExit(f"The model forecasts at most {C.MAX_LEAD} days ahead: a forecast made on {start.date()} "
                         f"reaches {furthest.date()} at most. For later days use newer data (menu 4 / --fresh).")
    return start, (last - start).days + 1
```

Delete `parse_days` (only the old menu used it). Replace `show_forecast` and add `present`:

```python
def show_forecast(state, start, n_days, slot_label=None, first=None):
    """Forecast only (no evaluation)."""
    fc = forecast(state, start, n_days, parse_slot(slot_label) if slot_label else None)
    print(f"Forecast made on {start.date()} at 00:00 Kyiv, using only data from before then.")
    return present(fc, start, slot_label, first)


def present(fc, start, slot_label=None, first=None):
    """Keep the chosen days (from `first`, default the start day), print them and save the CSV."""
    first = first if first is not None else start
    fc = fc[fc["day"] >= first.date()]
    print_forecast(fc, slot_label)
    print(f"Saved: {save_forecast(fc, first)}")
    return fc
```

In `run_evaluation`, delete the line `save_forecast(fc, fc_start)` and replace the line
`print_forecast(fc, slot_label)` near the end with:

```python
    present(fc, fc_start, slot_label)
```

Replace `menu_forecast`:

```python
def menu_forecast(state, slot_first, fresh):
    end_day = state["end_day"]

    def days_answer(text):
        dates = [parse_start(day, "Days") for day in text.split()] or \
            [end_day, end_day + pd.Timedelta(days=C.MAX_LEAD - 1)]
        if len(dates) > 2:
            raise SystemExit("Type one day or two (first and last).")
        pick_days(dates[0], dates[-1], None, end_day)  # refuses days the model cannot reach
        return dates[0], dates[-1]

    def slot_answer(text, required):
        if text or required:
            parse_slot(text)  # wrong format: its usual message, then the question again
        return text or None

    slot = ask("Time slot, Kyiv time, e.g. 08:00-09:30", lambda a: slot_answer(a, True)) if slot_first else None
    first, last = ask("Which days? e.g. 2026-10-14, or 2026-10-14 2026-10-16 [Enter = next 7 days]", days_answer)
    auto = min(first, end_day)
    start, n_days = ask(f"Forecast made on, YYYY-MM-DD [automatic = {auto.date()}]",
                        lambda a: pick_days(first, last, parse_start(a) if a else None, end_day))
    if not slot_first:
        slot = ask("Time slot, Kyiv time, e.g. 08:00-09:30 (Enter = skip)", lambda a: slot_answer(a, False))
    command = ("python3 run.py" + (" --fresh" if fresh else "") + f" --for {first.date()}"
               + (f" {last.date()}" if last != first else "") + (f" --start {start.date()}" if start != auto else "")
               + (f" --slot {slot}" if slot else ""))
    print(f"→ same as: {command}")
    return show_forecast(state, start, n_days, slot, first=first)
```

In `main`, the options (replace the `--days` line, add `--for`):

```python
    parser.add_argument("--for", dest="for_days", nargs="+", metavar="DAY",
                        help="the day(s) to forecast, YYYY-MM-DD or FIRST LAST; the start is chosen automatically")
    parser.add_argument("--days", type=int, choices=range(1, C.MAX_LEAD + 1), metavar="1-7")
```

After `make_console_safe()`:

```python
    for_days = [parse_start(day, "--for") for day in args.for_days] if args.for_days else None
    if for_days and len(for_days) > 2:
        raise SystemExit(f"--for takes one day or two (FIRST LAST), got {len(for_days)}.")
    if for_days and args.days is not None:
        raise SystemExit("Use either --for (which days) or --days (how many), not both.")
    n_days = args.days or C.MAX_LEAD
```

and the end of `main`:

```python
    if with_menu:
        menu(inputs, state)
    elif for_days:
        start, n = pick_days(for_days[0], for_days[-1], start, state["end_day"])
        show_forecast(state, start, n, args.slot, first=for_days[0])
    elif start is None:
        run_evaluation(inputs, state, n_days, slot, args.slot)
    else:
        show_forecast(state, start, n_days, args.slot)
```

- [ ] **Step 4: Run all tests, expect OK**

Run: `python3 run.py test 2>&1 | grep -E "^(FAIL|ERROR)|^Ran|^OK|^FAILED"`
Expected: `OK`. Then by hand: `python3 run.py --for 2026-10-14` (shows only 14 Oct, "Forecast made on
2026-10-11") and `python3 run.py --for 2026-10-20` (refused: reaches 2026-10-17 at most).

- [ ] **Step 5: Docs**

`README.md`, options table — add after the `--start` row:

```markdown
| `python3 run.py --for 2026-10-14` or `--for 2026-10-14 2026-10-16` | Forecast only for the chosen day(s). The start (forecast moment) is chosen automatically: the last data day for future days, the first chosen day for past days, so only earlier data is used. Up to 7 days ahead of the start; `--start` may be added by hand |
```

`HOW_TO_USE.md`: in the Step 3 menu paragraph replace "Choices 2 and 3 ask for the date, the number of days and
the time slot one by one" with "Choices 2 and 3 ask which days you want, when the forecast is made (Enter =
automatic) and the time slot one by one". Insert at the top of Step 5, before the `--start` example:

````markdown
**The easy way — say which days you want:**

```bash
python3 run.py --for 2026-10-14
python3 run.py --for 2026-10-14 2026-10-16
```

The app picks the start (the moment the forecast is made) itself: the last day in the data when the days are in
the future (newest information), or the first chosen day when they are in the past (then it shows what the model
would have said back then, using only earlier data). The model looks at most 7 days ahead of the start; further
days are refused with the furthest possible day. Only the chosen days are printed and saved
(`reports/forecast_<first day>.csv`).

**Choosing the start yourself:**
````

- [ ] **Step 6: Commit, push, CI, merge**

```bash
git add alerts/main.py tests/test_main.py README.md HOW_TO_USE.md
git commit -m "Choose the forecast days with --for; the start is picked automatically before the first day

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin features
```

Wait for CI (12 green), then `git switch main && git merge --ff-only features && git push origin main && git switch features`.

- [ ] **Step 7: Leakage audit + report.** Run the `leakage-auditor` agent on `git diff main~1 -- alerts/` (the
  automatic start). Report risks + self-check commands to the user; wait for "go".

---

### Task 2: One oblast (`--oblast`)

**Files:**
- Modify: `alerts/main.py` (new `match_oblast`; `present`, `show_forecast`, `run_evaluation`, `menu_forecast`, `main`)
- Test: `tests/test_main.py`
- Docs: `README.md`, `HOW_TO_USE.md`

**Interfaces:**
- Consumes: `present`, `show_forecast`, `pick_days` from Task 1.
- Produces: `match_oblast(text) -> str` (full name from `C.OBLASTS`);
  `present(fc, start, slot_label=None, first=None, oblast=None)`;
  `show_forecast(state, start, n_days, slot_label=None, first=None, oblast=None)`;
  `run_evaluation(inputs, state, n_days=C.MAX_LEAD, slot=None, slot_label=None, oblast=None)`.

- [ ] **Step 1: Write the failing tests**

```python
class OblastTest(unittest.TestCase):
    def test_match(self):
        from alerts.main import match_oblast
        self.assertEqual(match_oblast("kyiv city"), "Kyiv City")
        self.assertEqual(match_oblast("  LVIV "), "Lvivska oblast")
        self.assertEqual(match_oblast("ivano"), "Ivano-Frankivska oblast")
        for text, message in (("kyiv", "several oblasts: Kyiv City, Kyivska oblast"),
                              ("odesa", "fits no oblast"), ("", "fits no oblast")):
            with self.assertRaises(SystemExit) as ctx:
                match_oblast(text)
            self.assertIn(message, str(ctx.exception.code))

    def test_forecast_for_one_oblast(self):
        text, _, _ = run_main(["--for", "2026-01-22", "--oblast", "kyiv city"])
        self.assertIn("Kyiv City", text)
        self.assertNotIn("Lvivska oblast", text)

    def test_evaluation_run_gets_the_oblast(self):
        from unittest import mock
        import alerts.main as M
        with mock.patch.object(M, "load_inputs", return_value={}), mock.patch.object(M, "build", return_value={}), \
                mock.patch.object(M, "run_evaluation") as evaluation:
            M.main(["--oblast", "lviv"])
        self.assertEqual(evaluation.call_args.kwargs["oblast"], "Lvivska oblast")

    def test_menu_asks_for_the_oblast(self):
        text, _, _ = run_menu(["2", "2026-01-22", "", "kyiv", "kyiv city", "", "q"])
        self.assertIn("fits several oblasts", text)
        self.assertIn('→ same as: python3 run.py --for 2026-01-22 --oblast "Kyiv City"\n', text)
        self.assertNotIn("Lvivska oblast", text)
```

The menu now asks for the oblast after the start, so the two menu tests from Task 1 get one more typed answer
(`""` = all oblasts) — replace their `run_menu([...])` lists with:

```python
        text, _, _ = run_menu(["2", "2025-01-01", "2026-01-28", "2026-01-10 2026-01-11", "2026-01-12", "", "", "", "q"])
```

```python
        text, _, _ = run_menu(["3", "8-9", "08:00-09:30", "", "", "", "q"])
```

- [ ] **Step 2: Run, expect failures**

Run: `python3 run.py test 2>&1 | grep -E "^(FAIL|ERROR)|^Ran|^OK|^FAILED"`
Expected: FAILED — `cannot import name 'match_oblast'`, `unrecognized arguments: --oblast`.

- [ ] **Step 3: Implement in `alerts/main.py`**

After `parse_slot`:

```python
def match_oblast(text):
    """Part of an oblast name, any case -> its full name. Several or no matches: a message listing them.
    shortcut: matches the official spelling only ('odes', not 'odesa'); add city names if people trip on it."""
    wanted = " ".join(text.lower().split())
    found = [o for o in C.OBLASTS if o.lower() == wanted] or \
        [o for o in C.OBLASTS if wanted and wanted in o.lower()]
    if len(found) == 1:
        return found[0]
    if found:
        raise SystemExit(f"--oblast {text!r} fits several oblasts: {', '.join(found)}. Type more of the name.")
    raise SystemExit(f"--oblast {text!r} fits no oblast. Oblasts: {', '.join(C.OBLASTS)} "
                     "(Luhansk and Crimea are not forecast: alerts there never stop).")
```

`present` and `show_forecast` gain `oblast`:

```python
def show_forecast(state, start, n_days, slot_label=None, first=None, oblast=None):
    """Forecast only (no evaluation)."""
    fc = forecast(state, start, n_days, parse_slot(slot_label) if slot_label else None)
    print(f"Forecast made on {start.date()} at 00:00 Kyiv, using only data from before then.")
    return present(fc, start, slot_label, first, oblast)


def present(fc, start, slot_label=None, first=None, oblast=None):
    """Keep the chosen days (from `first`, default the start day) and oblast, print them and save the CSV."""
    first = first if first is not None else start
    fc = fc[fc["day"] >= first.date()]
    if oblast:
        fc = fc[fc["oblast"] == oblast]
    print_forecast(fc, slot_label)
    print(f"Saved: {save_forecast(fc, first)}")
    return fc
```

`run_evaluation`: signature `def run_evaluation(inputs, state, n_days=C.MAX_LEAD, slot=None, slot_label=None,
oblast=None):` and its `present(fc, fc_start, slot_label)` line becomes
`present(fc, fc_start, slot_label, oblast=oblast)` (the HTML report keeps all oblasts).

`menu_forecast`: after the `start, n_days = ask(...)` line add

```python
    oblast = ask("Oblast, part of the name, e.g. lviv [Enter = all]", lambda a: match_oblast(a) if a else None)
```

extend `command` with `+ (f' --oblast "{oblast}"' if oblast else "")` placed before the `--slot` part, and
return `show_forecast(state, start, n_days, slot, first=first, oblast=oblast)`.

`main`: add the option and pass it on:

```python
    parser.add_argument("--oblast", help="one oblast, part of the name (e.g. lviv); default: all")
```

after the `n_days = ...` line: `oblast = match_oblast(args.oblast) if args.oblast is not None else None`, then
`show_forecast(..., first=for_days[0], oblast=oblast)`, `run_evaluation(inputs, state, n_days, slot, args.slot,
oblast=oblast)` and `show_forecast(state, start, n_days, args.slot, oblast=oblast)`.

- [ ] **Step 4: Run all tests, expect OK** — `python3 run.py test 2>&1 | grep -E "^(FAIL|ERROR)|^Ran|^OK|^FAILED"`;
  by hand: `python3 run.py --for 2026-10-14 --oblast lviv`, `python3 run.py --oblast kyiv` (refused, lists both).

- [ ] **Step 5: Docs.** `README.md` options table, new row:

```markdown
| `python3 run.py --for 2026-10-14 --oblast lviv` | Only one oblast (part of its name, any case; `kyiv` is ambiguous: use `kyiv city` or `kyivska`). Works with every run; the HTML report always covers all oblasts |
```

`HOW_TO_USE.md`, new step after Step 6:

````markdown
## Step 6b. Only your oblast

```bash
python3 run.py --for 2026-10-14 --oblast lviv
```

Type part of the oblast's name in any case (`lviv`, `kharkiv`, `ivano`). If it fits several (`kyiv` → Kyiv City
and Kyivska oblast) or none, the app lists the names; type more of the name. Names follow the dataset's spelling
(`odeska`, `sumska`), so `odes` works and `odesa` does not. In the menu, choices 2 and 3 ask for it (Enter = all).
````

- [ ] **Step 6: Commit, push, CI, merge** — as in Task 1 Step 6, message
  "Forecast for one oblast with --oblast (part of the name); menu asks for it".

- [ ] **Step 7: Report** risks + self-check commands; wait for "go".

---

### Task 3: Details — 2-hour windows and alert durations (`--details`)

**Files:**
- Create: `alerts/details.py`, `tests/test_details.py`
- Modify: `alerts/main.py` (`build`, `forecast`, new `print_details`, `present`, `show_forecast`,
  `run_evaluation`, `menu_forecast`, `main`), `tests/test_main.py` (`small_state`, `TargetTest`)
- Docs: `README.md`, `HOW_TO_USE.md`

**Interfaces:**
- Consumes: `present`, `show_forecast`, `match_oblast` from Tasks 1–2; `C.kyiv_day_start_utc`.
- Produces (`alerts/details.py`): `WINDOWS` (list of start hours 0, 2, …, 22), `WINDOW_COLUMNS`
  (`["w00_02", …, "w22_00"]`), `DURATION_COLUMNS` (`["alerts", "typical", "p10", "p90", "shortest", "average",
  "longest"]`), `durations(vol, start, window=28) -> DataFrame` (index `C.OBLASTS`, columns `DURATION_COLUMNS`,
  minutes), `minutes_text(minutes) -> str`. State gains `state["vol"]`. Forecast frames gain the
  `WINDOW_COLUMNS` and `dur_<name>` columns. `present(..., oblast=None, details=False)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_details.py`:

```python
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
```

In `tests/test_main.py`: `small_state` must pass the volunteer alerts — replace its last line with

```python
    return build({"intervals": intervals, "vol": intervals.drop(columns="source").assign(naive=False),
                  "chomu": pd.DataFrame({"id": [], "time": pd.to_datetime([]), "text": []})})
```

and in `TargetTest` add `"vol": vol.drop(columns="source").assign(naive=False),` to its `inputs` dict. Then add:

```python
class DetailsTest(unittest.TestCase):
    def test_forecast_has_windows_and_durations(self):
        import alerts.main as M
        from alerts.details import WINDOWS, durations
        from alerts.features import blended_slot_chances
        state, start = small_state(), pd.Timestamp("2026-01-10")
        fc = M.forecast(state, start, 2)
        day2 = fc[(fc["oblast"] == "Kyiv City") & (fc["day"] == fc["day"].max())].iloc[0]
        expected = blended_slot_chances(state["cube"], state["days"], "Kyiv City", start, 2)[WINDOWS]
        np.testing.assert_allclose(day2[["w00_02", "w02_04", "w04_06", "w06_08", "w08_10", "w10_12", "w12_14",
                                         "w14_16", "w16_18", "w18_20", "w20_22", "w22_00"]].astype(float), expected)
        self.assertEqual(day2["dur_alerts"], durations(state["vol"], start).loc["Kyiv City", "alerts"])

    def test_one_oblast_prints_details_all_oblasts_only_when_asked(self):
        text, _, _ = run_main(["--for", "2026-01-22", "--oblast", "kyiv city"])
        self.assertIn("each 2-hour window", text)
        self.assertIn("Alert durations", text)
        text, _, _ = run_main(["--for", "2026-01-22"])
        self.assertNotIn("Alert durations", text)
        text, _, _ = run_main(["--for", "2026-01-22", "--details"])
        self.assertIn("Alert durations", text)
        self.assertIn("No alerts in the last 28 days: Cherkaska oblast", text)  # oblasts without alerts

    def test_note_when_long_records_pull_the_average_up(self):
        import contextlib
        import io
        import alerts.main as M
        from alerts.details import DURATION_COLUMNS, WINDOW_COLUMNS
        fc = pd.DataFrame([{"oblast": "Ternopilska oblast", "day": pd.Timestamp("2026-10-14").date(),
                            **{name: 0.1 for name in WINDOW_COLUMNS},
                            **{f"dur_{k}": v for k, v in zip(DURATION_COLUMNS, (9, 27, 12, 60, 12, 590, 4740))}}])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            M.print_details(fc, pd.Timestamp("2026-10-11"))
        self.assertIn("Ternopilska oblast: the average is more than twice the typical", out.getvalue())
        self.assertIn("79 h 0 min", out.getvalue())  # the longest record, 4740 minutes

    def test_menu_offers_details_for_all_oblasts(self):
        text, _, _ = run_menu(["2", "2026-01-22", "", "", "", "y", "q"])
        self.assertIn("→ same as: python3 run.py --for 2026-01-22 --details", text)
        self.assertIn("Alert durations", text)
```

After an all-oblast forecast the menu now asks "Show details …? [y/N]", so the two menu tests from Task 1 get
one more answer (`""` = no) — their lists become:

```python
        text, _, _ = run_menu(["2", "2025-01-01", "2026-01-28", "2026-01-10 2026-01-11", "2026-01-12", "", "", "", "", "q"])
```

```python
        text, _, _ = run_menu(["3", "8-9", "08:00-09:30", "", "", "", "", "q"])
```

(Task 2's menu test chooses one oblast, so it gets no details question and stays as it is.)

- [ ] **Step 2: Run, expect failures** — `No module named 'alerts.details'`, `KeyError: 'vol'`, missing texts.

- [ ] **Step 3: Implement.** Create `alerts/details.py`:

```python
"""Details for a forecast: the chance of an alert in each fixed 2-hour window, and recent alert durations.
Both use only history from before the forecast moment; the report checks them once on the test period."""
import pandas as pd

from .config import OBLASTS, kyiv_day_start_utc

WINDOWS = list(range(0, 24, 2))  # start hours of the windows 00–02, 02–04, … 22–00
WINDOW_COLUMNS = [f"w{h:02d}_{(h + 2) % 24:02d}" for h in WINDOWS]
DURATION_COLUMNS = ["alerts", "typical", "p10", "p90", "shortest", "average", "longest"]


def durations(vol, start, window=28):
    """Alert durations in minutes per oblast: eTryvoga alerts that started in the `window` days before the
    forecast moment (00:00 Kyiv of `start`; window=None: all history) and ended before it. Naive alerts (no
    'all clear' message, made-up end) and alerts still running at that moment are left out."""
    moment = kyiv_day_start_utc(start)
    known = vol[~vol["naive"] & (vol["end"] <= moment)]
    if window is not None:
        known = known[known["start"] >= moment - pd.Timedelta(days=window)]
    minutes = (known["end"] - known["start"]).dt.total_seconds() / 60
    by = minutes.groupby(known["oblast"])
    table = pd.DataFrame({"alerts": by.size(), "typical": by.median(), "p10": by.quantile(0.1),
                          "p90": by.quantile(0.9), "shortest": by.min(), "average": by.mean(), "longest": by.max()})
    return table.reindex(OBLASTS).fillna({"alerts": 0}).astype({"alerts": int})


def minutes_text(minutes):
    if pd.isna(minutes):
        return "—"
    minutes = int(round(minutes))
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"
```

In `alerts/main.py`: import `from .details import DURATION_COLUMNS, WINDOW_COLUMNS, WINDOWS, durations,
minutes_text`. In `build` add `"vol": inputs["vol"],` to the returned dict. In `forecast`, before
`records = []` add `recent = durations(state["vol"], start)`, and right after `record = {...}` is created add:

```python
        record.update(dict(zip(WINDOW_COLUMNS, slots[WINDOWS])))
        record.update({f"dur_{name}": recent.loc[oblast, name] for name in DURATION_COLUMNS})
```

New function after `print_forecast`:

```python
def print_details(fc, start):
    """Every fixed 2-hour window (one oblast: each shown day; all oblasts: the first shown day, the rest is in
    the CSV) and alert durations of the last 28 days before the forecast moment."""
    one = fc["oblast"].nunique() == 1
    rows = fc if one else fc[fc["day"] == fc["day"].min()]
    windows = rows.set_index("day" if one else "oblast")[WINDOW_COLUMNS].mul(100).round().astype(int)
    windows = windows.astype(str).add("%")
    windows.columns = [f"{h:02d}–{(h + 2) % 24:02d}" for h in WINDOWS]
    where = fc["oblast"].iloc[0] if one else f"{rows['day'].iloc[0]} (other days: in the CSV)"
    print(f"\nChance of an alert at some moment in each 2-hour window (Kyiv time), {where}:")
    print(windows.to_string())
    dur = rows.drop_duplicates("oblast").set_index("oblast")[[f"dur_{k}" for k in DURATION_COLUMNS]]
    dur.columns = DURATION_COLUMNS
    table = pd.DataFrame({
        "alerts": dur["alerts"].astype(int),
        "typical": dur["typical"].map(minutes_text),
        "most last (10-90%)": [f"{minutes_text(a)} – {minutes_text(b)}" for a, b in zip(dur["p10"], dur["p90"])],
        "shortest": dur["shortest"].map(minutes_text),
        "average": dur["average"].map(minutes_text),
        "longest": dur["longest"].map(minutes_text)})
    print(f"\nAlert durations (eTryvoga alerts with a real end time, last 28 days before {start.date()}):")
    print(table.to_string())
    for oblast in dur.index[dur["average"] > 2 * dur["typical"]]:
        print(f"Note: {oblast}: the average is more than twice the typical — a few very long records "
              "(possibly missing 'all clear' messages) pull it up.")
    quiet = list(dur.index[dur["alerts"] == 0])
    if quiet:
        print(f"No alerts in the last 28 days: {', '.join(quiet)}.")
```

`present`, `show_forecast`, `run_evaluation` gain `details=False` and pass it on; `present` prints details for one
oblast or when asked:

```python
def show_forecast(state, start, n_days, slot_label=None, first=None, oblast=None, details=False):
    """Forecast only (no evaluation)."""
    fc = forecast(state, start, n_days, parse_slot(slot_label) if slot_label else None)
    print(f"Forecast made on {start.date()} at 00:00 Kyiv, using only data from before then.")
    return present(fc, start, slot_label, first, oblast, details)


def present(fc, start, slot_label=None, first=None, oblast=None, details=False):
    """Keep the chosen days (from `first`, default the start day) and oblast, print them (with the details
    for one oblast or when asked) and save the CSV."""
    first = first if first is not None else start
    fc = fc[fc["day"] >= first.date()]
    if oblast:
        fc = fc[fc["oblast"] == oblast]
    print_forecast(fc, slot_label)
    if oblast or details:
        print_details(fc, start)
    print(f"Saved: {save_forecast(fc, first)}")
    return fc
```

`run_evaluation(..., oblast=None, details=False)` → `present(fc, fc_start, slot_label, oblast=oblast,
details=details)`. `main`: `parser.add_argument("--details", action="store_true", help="every 2-hour window and
recent alert durations (automatic with --oblast)")`, pass `details=args.details` to the three calls.
`menu_forecast`: replace its last line `return show_forecast(...)` with

```python
    fc = show_forecast(state, start, n_days, slot, first=first, oblast=oblast)
    if not oblast and ask("Show details for all oblasts (every 2-hour window, alert durations)? [y/N]",
                          lambda a: a.lower() in ("y", "yes")):
        print(f"→ same as: {command} --details")
        print_details(fc, start)
    return fc
```

- [ ] **Step 4: Run all tests, expect OK**; by hand: `python3 run.py --for 2026-10-14 --oblast ternopil` (note
  about the average), `python3 run.py --for 2026-10-14 --oblast zakarpat` (0 alerts, `—`, "No alerts" line),
  `python3 run.py --details` (evaluation + details for all oblasts).

- [ ] **Step 5: Docs.** `README.md` options table, new row:

```markdown
| `python3 run.py --for 2026-10-14 --details` | Adds the chance for each 2-hour window (00–02 … 22–00) and alert durations of the last 28 days (typical, usual range 10–90 %, shortest, average, longest; only alerts with a real end time). Automatic with `--oblast`. Also in the CSV (`w00_02` …, `dur_…`) |
```

`HOW_TO_USE.md`, new step after Step 6b:

````markdown
## Step 6c. Details: every 2-hour window and how long alerts last

```bash
python3 run.py --for 2026-10-14 --oblast lviv     # one oblast: details are shown automatically
python3 run.py --for 2026-10-14 --details         # all oblasts
```

- **2-hour windows** — the chance of an alert at some moment in 00–02, 02–04 … 22–00. Only the start day's
  chances are adjusted by the predicted number of alerts; later days show the oblast's usual pattern, so they
  repeat. (The "quietest 2 hours" may start at an odd hour, e.g. 13:00–15:00, so it need not be one of these 12.)
- **Alert durations** — from alerts of the last 28 days before the forecast moment that have a real "all clear"
  message: *typical* (half the alerts are shorter), *most last* (the middle 80 %), *shortest*, *average*,
  *longest*. A note appears when a few very long records (often a missing "all clear") pull the average up.
  This is recent history, not a model; how well it predicts is in the report (Step 4, "Details checks").
- In the menu, after a forecast for all oblasts: "Show details for all oblasts? [y/N]".
````

- [ ] **Step 6: Commit, push, CI, merge** — as in Task 1 Step 6, files `alerts/details.py alerts/main.py
  tests/test_details.py tests/test_main.py README.md HOW_TO_USE.md`, message
  "Details: every 2-hour window and recent alert durations (--details, automatic for one oblast)".

- [ ] **Step 7: Leakage audit + report.** `leakage-auditor` on `alerts/details.py` and the `forecast` change;
  report risks + self-check commands; wait for "go".

---

### Task 4: Honesty checks in the report (measured once)

**Files:**
- Modify: `alerts/evaluate.py` (`calibration` learns 2-hour windows), `alerts/details.py` (new `duration_check`),
  `alerts/main.py` (`run_evaluation` report section)
- Test: `tests/test_details.py`, `tests/test_evaluate.py`
- Docs: `README.md` (results), `HOW_TO_USE.md` (Step 4 section list)

**Interfaces:**
- Consumes: `WINDOWS`, `durations` (Task 3); `backtest`, `blended_slot_chances`, `forecast_chance`.
- Produces: `calibration(bt, cube, days, lead=1, two_hour=False)` (column `windows` instead of `hours` when
  `two_hour`); `duration_check(vol, days) -> (DataFrame[method, "MAE, minutes", alerts], float share_inside)`.

- [ ] **Step 1: Write the failing tests.** `tests/test_details.py`:

```python
class DurationCheckTest(unittest.TestCase):
    def test_typical_of_last_28_days_vs_all_history(self):
        from alerts.details import duration_check
        vol = pd.DataFrame([
            *(alert("Kyiv City", f"2025-11-2{i} 10:00", 100) for i in range(4)),   # old history: long alerts
            alert("Kyiv City", "2026-01-01 10:00", 10),
            alert("Kyiv City", "2026-01-02 10:00", 20),
            alert("Kyiv City", "2026-01-03 10:00", 30),
            alert("Kyiv City", "2026-01-10 10:00", 25),                             # the test-day alert
        ], columns=["oblast", "start", "end", "naive"])
        table, inside = duration_check(vol, pd.date_range("2026-01-10", "2026-01-10"))
        mae = table.set_index("method")["MAE, minutes"]
        self.assertEqual(mae["typical of the last 28 days"], 5)     # |25 - 20|
        self.assertEqual(mae["typical over all history"], 75)        # |25 - median(10,20,30,100,100,100,100)|
        self.assertEqual(inside, 1.0)                                # 25 is inside 12–28 (10–90 % of 10, 20, 30)
```

`tests/test_evaluate.py` (append):

```python
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
```

(If `tests/test_evaluate.py` lacks `import pandas as pd` / `import unittest` at the top, add them.)

- [ ] **Step 2: Run, expect failures** — `cannot import name 'duration_check'`, `unexpected keyword 'two_hour'`.

- [ ] **Step 3: Implement.** `alerts/evaluate.py`: import `from .details import WINDOWS` and
`from .features import blended_chance, blended_slot_chances, forecast_chance, hourly_chance`; replace `calibration`:

```python
def calibration(bt, cube, days, lead=1, two_hour=False):
    """When the forecast said 30-40%, how often was an alert really active? Shown chances in 10% bins, for
    each hour, or (two_hour) for each fixed 2-hour window 00-02 ... 22-00."""
    predicted, observed = [], []
    for r in bt[bt["lead_days"] == lead].itertuples():
        oblast = OBLASTS[r.oblast]
        chance = (blended_slot_chances(cube, days, oblast, r.t0, 2)[WINDOWS] if two_hour
                  else blended_chance(cube, days, oblast, r.t0))
        predicted.append(forecast_chance(chance, r.lead_days, r.pred, r.cnt_28d))
        seen = cube[r.oblast, days.get_loc(r.target_day)]
        observed.append(seen.reshape(12, 2).any(axis=1) if two_hour else seen)
    labels = [f"{10 * i}–{10 * (i + 1)}%" for i in range(10)]
    pairs = pd.DataFrame({"predicted": np.concatenate(predicted), "observed": np.concatenate(observed).astype(float)})
    pairs["bin"] = pd.cut(pairs["predicted"], np.linspace(0, 1, 11), labels=labels, include_lowest=True)
    table = pairs.groupby("bin", observed=True).agg(predicted=("predicted", "mean"), observed=("observed", "mean"),
                                                   hours=("observed", "size"))
    return table.rename(columns={"hours": "windows"}) if two_hour else table
```

`alerts/details.py`, append:

```python
def duration_check(vol, days):
    """Test period, measured once: every alert with a real end, its duration predicted at 00:00 Kyiv of its
    day by the typical duration of the last 28 days and by the typical duration over all earlier history."""
    parts = []
    for day in days:
        moment, next_moment = kyiv_day_start_utc(day), kyiv_day_start_utc(day + pd.Timedelta(days=1))
        today = vol[~vol["naive"] & (vol["start"] >= moment) & (vol["start"] < next_moment)]
        recent, history = durations(vol, day), durations(vol, day, window=None)
        parts.append(pd.DataFrame({
            "minutes": ((today["end"] - today["start"]).dt.total_seconds() / 60).to_numpy(),
            "recent": recent["typical"].reindex(today["oblast"]).to_numpy(),
            "p10": recent["p10"].reindex(today["oblast"]).to_numpy(),
            "p90": recent["p90"].reindex(today["oblast"]).to_numpy(),
            "history": history["typical"].reindex(today["oblast"]).to_numpy()}))
    alerts = pd.concat(parts, ignore_index=True).dropna()  # oblasts with no earlier alerts: no prediction
    table = pd.DataFrame({
        "method": ["typical of the last 28 days", "typical over all history"],
        "MAE, minutes": [(alerts["minutes"] - alerts["recent"]).abs().mean(),
                         (alerts["minutes"] - alerts["history"]).abs().mean()],
        "alerts": [len(alerts), len(alerts)]})
    return table, float(alerts["minutes"].between(alerts["p10"], alerts["p90"]).mean())
```

`alerts/main.py`: import `duration_check` from `.details`. In `run_evaluation`, after `cal = calibration(...)`:

```python
    window_cal = calibration(bt, state["cube"], state["days"], two_hour=True)
    dur_table, inside = duration_check(inputs["vol"], days)
```

and in section "4. Test results", after `+ table_html(cal, 2)`:

```python
         + "<h3>Details checks (measured once on the test period; nothing was changed because of them)</h3>"
           "<p class='note'>2-hour windows (00–02 … 22–00), 1 day ahead: when we said X%, how often was there "
           "an alert at some moment in the window?</p>" + table_html(window_cal, 2)
         + "<p class='note'>Alert durations: each test-period alert with a real end, its duration predicted at "
           "00:00 Kyiv of its day. Mean absolute error in minutes (lower is better):</p>"
         + table_html(dur_table.set_index("method"), 1)
         + f"<p class='note'>Share of these alerts inside the usual range (10–90%) of the last 28 days: "
           f"{inside:.0%} (ideal ≈ 80%).</p>"
```

- [ ] **Step 4: Run all tests, expect OK.** Then run `python3 run.py < /dev/null` once, open
  `reports/report.html`, and copy the two new results (window calibration rows, duration MAEs, share inside).
  Do not change anything because of them.

- [ ] **Step 5: Docs.** `README.md` results section, after the hourly calibration paragraph, add a paragraph
  "**Details checks** (measured once, nothing changed because of them):" with the numbers from Step 4 in plain
  words (windows: "said X % → happened Y %" for 2–3 bins; durations: MAE last-28-days vs all-history in minutes,
  share inside the usual range vs ideal 80 %), and one honest sentence on what they mean. `HOW_TO_USE.md` Step 4,
  item 4: append "; then 'Details checks' — the 2-hour windows' calibration and how well recent durations
  predicted real ones".

- [ ] **Step 6: Commit, push, CI, merge** — message "Report: details checks (2-hour window calibration,
  duration error) measured once on the test period".

- [ ] **Step 7: Report** the measured numbers honestly + self-check commands; wait for "go".

---

### Task 5: Final checks and status

**Files:** `CLAUDE.md` (status + decisions), memory note.

- [ ] **Step 1:** `repro-tester` agent: fresh clone, run as the README says (`python3 run.py < /dev/null`,
  `--for`, `--oblast`, `--details`, `test`).
- [ ] **Step 2:** `CLAUDE.md` "Current status": the menu, `--for` / `--oblast` / `--details`, new test count,
  the details-check numbers, Python 3.15 support; "Agreed design": decisions of 2026-10-12 (dates only, refuse
  beyond 7 days, one oblast → details at once, 12 windows, all duration numbers, measure once).
- [ ] **Step 3:** Commit, push, CI green, merge into `main`; report the final state to the user.
