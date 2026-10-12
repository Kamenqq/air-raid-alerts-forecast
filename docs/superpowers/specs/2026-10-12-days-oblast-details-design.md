# Choose the days, one oblast, details — design

Date: 2026-10-12 (Kyiv). Builds on the interactive menu (branch `menu`). Agreed with the user in the main
conversation; this file records the decisions.

## Why

A resident or reviewer testing the app wants answers like "what about Wednesday, in Lviv, and how long do alerts
usually last there?". Today they must pick a *start date* (the forecast moment), get all 24 oblasts, and see only
the quietest / riskiest 2 hours. Three additions, built and merged one at a time in this order.

## 1. Choose the days: `--for`

- `--for DAY` or `--for FIRST LAST` (dates only, `YYYY-MM-DD`, Kyiv calendar days). Only these days are printed
  and saved.
- **Automatic start** (the forecast moment, 00:00 Kyiv of the start day; the model sees only data before it):
  `start = min(FIRST, last data day)` — the latest moment that is not after the first chosen day. Future days →
  the last data day (most recent information); past days → FIRST ("what would the model have said").
  Leak-free by the existing rule: a forecast made at start uses only data before 00:00 Kyiv of start.
- `--start` may still be given by hand together with `--for`.
- Checks (each a clear message; in the menu the question is asked again):
  - start after FIRST → refused (it would use information from a chosen day);
  - LAST more than 6 days after start (the model forecasts at most 7 days ahead, tested only that far) →
    "with data up to D the furthest day is D+6; for newer data download first (menu 4 / `--fresh`)";
  - LAST before FIRST; start outside 2026-01-01 … last data day (existing check);
  - `--for` together with `--days` → "use one or the other".
- Without `--for`, `--start` / `--days` work exactly as now.
- **Menu (choices 2 and 3):** "Which days? e.g. 2026-10-14, or 2026-10-14 2026-10-16 [Enter = next 7 days]" →
  "Forecast made on [automatic = …]" → (oblast, slot as below). The "→ same as" line shows the equivalent command.

## 2. One oblast: `--oblast`

- Part of the name, case-insensitive (`lviv`, `kyiv city`). Several matches (`kyiv` → Kyiv City, Kyivska
  oblast) or none → message listing the matches / all oblasts; in the menu, asked again. Menu: Enter = all.
- Filters the printed tables and the forecast CSV, in the forecast run and in the evaluation run. The HTML report
  stays all-oblast.
- **One oblast → its details (section 3) are printed right away.**

## 3. Details: `--details` (automatic for one oblast)

- **12 two-hour windows** 00–02 … 22–00 for each chosen day: the chance of an alert at some moment in the
  window — the existing 2-hour slot chances (`blended_slot_chances(..., 2)`) at even start hours, with the same
  1-day-ahead count scaling as the hourly chances. The quietest / riskiest 2 hours stay as the summary (they
  may start at an odd hour, so they need not be one of the 12 windows; said in the docs).
- **Alert durations** per oblast, from eTryvoga alerts that started in the 28 days before the forecast moment
  and have a real end before it (naive alerts — no "all clear" message — and alerts still running are left
  out, so nothing after the forecast moment is used): number of alerts, typical (median), usual range
  (10th–90th percentile), shortest, average, longest. If the average is more than twice the typical: a note
  that a few very long records (possibly missing "all clear" messages) pull it up. No alerts → "no alerts in the
  last 28 days".
- In the menu, after an all-oblast forecast: "Show details for all oblasts? [y/N]".
- Forecast CSV gains the 12 window columns (`w00_02` … `w22_00`) and the duration columns.

### Honesty checks (report, test section; measured once, nothing changed because of them)

Committed in advance: whatever these show is reported as is; no setting is changed after seeing them (that
would be tuning on the test period).

- **2-hour windows:** calibration on the test period, 1 day ahead, like the existing hourly table — "when we
  said X %, an alert happened Y % of the time".
- **Durations:** for every test-period alert with a real end, the prediction made at 00:00 Kyiv of its day:
  mean absolute error in minutes of "typical of the last 28 days" vs the simpler "typical over all history before
  that moment"; and the share of alerts inside the predicted usual range (10–90 %; ideal ≈ 80 %).

## Code

- New `alerts/details.py`: window chances, duration statistics, the two checks — small functions with tests.
- `alerts/main.py`: `--for`, `--oblast`, `--details`, automatic start, oblast name matching, menu questions,
  printing.
- Docs: README (options table, results), `HOW_TO_USE.md`, CLAUDE.md status.

## Tests

- Automatic start: future days → last data day; past days → first chosen day; refusals above.
- **Leakage:** a forecast via `--for` trains only on days before the start; duration statistics ignore alerts
  that start or end after the forecast moment.
- Oblast matching: exact, partial, ambiguous, unknown.
- Windows: 12 values equal to the even-hour 2-hour slot chances; durations: known toy data, no-alert case,
  distortion note.
- Menu: the new questions, wrong answers asked again, "→ same as" line.
- CI on 3 OS × Python 3.9 / 3.13 / 3.14 / 3.15 before each merge.

## Not doing

Weekday words ("wed"), forecasts beyond 7 days, any change to the count model or the hourly chances, a
duration *model* (durations are recent history, checked once).
