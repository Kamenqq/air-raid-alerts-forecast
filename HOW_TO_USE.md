# How to use the air raid alerts forecast — step by step

This guide assumes no programming experience. Every command is typed into a **terminal**:
**macOS** — open the *Terminal* app; **Windows** — open *PowerShell* or *Command Prompt*;
**Linux** — any terminal. Press Enter after each command.

> On **Windows**, type `python` everywhere this guide says `python3`.

---

## Step 1. Check that Python is installed

```bash
python3 --version
```

You need **Python 3.9 or newer** (e.g. `Python 3.9.6`, `Python 3.13.1`).
If you see "command not found" or an older version, install Python from
[python.org/downloads](https://www.python.org/downloads/) (on Windows, tick **"Add Python to PATH"** in the
installer), then open a new terminal and check again.

## Step 2. Download the project

**With git:**

```bash
git clone https://github.com/Kamenqq/air-raid-alerts-forecast.git
cd air-raid-alerts-forecast
```

**Without git:** on the GitHub page click **Code → Download ZIP**, unpack it, and in the terminal go into the
unpacked folder, e.g. `cd Downloads/air-raid-alerts-forecast-main`.

All following commands are run **inside this folder**.

## Step 3. First run — the menu, then evaluation, report and a 7-day forecast

```bash
python3 run.py
```

What happens:

1. **First time only (~1 minute, needs internet):** it creates a private Python environment in a folder
   called `.venv` and installs four libraries into it (pandas, numpy, scikit-learn, matplotlib). Nothing is
   installed into your system Python. Later runs skip this step.
2. **The menu** lists everything the app can do. Type a number and press Enter; at the start just **Enter** = choice 1:

   ```
   Air raid alerts forecast. Data: stored snapshot.
     1  Test the model on the last 28 days + report + 7-day forecast   (Enter at the start)
     2  Forecast for the days you choose (up to 7 days ahead)
     3  Chance of an alert during your own time slot (commute, shopping...)
     4  Download the newest data, then do 1
     5  Run the automatic tests
     q  Quit
   Choose [1]:
   ```

   Choices 2 and 3 ask which days you want, when the forecast is made (Enter = automatic) and the time slot one by one (Enter takes the value in
   `[brackets]`); a wrong answer shows what is wrong and asks again. Each choice prints the **equivalent
   command** (`→ same as: python3 run.py --start …`), which is what Steps 5–8 describe. After a choice the menu
   comes back; Ctrl-C stops a running choice, `q` quits. The data is loaded (~2 s) on the first choice that needs it. The menu appears only when you run `python3 run.py` without options in a
   terminal.
3. **Evaluation (~20 seconds, choice 1):** using the data stored in the project (`data/snapshot/`), it tests the model
   honestly on the latest 28 days: for every day it pretends to be on the morning of that day, retrains the
   model on earlier days only, forecasts, and compares with what really happened.
4. It prints a **one-paragraph result**, the **forecast tables**, and saves files to the `reports/` folder.

### How to read the forecast

```
Expected number of alerts per day:
oblast              2026-10-11  2026-10-12  2026-10-13 ...
Kyiv City                  4.3         4.4         5.3
Volynska oblast            4.0         3.9         5.0

Chance of an alert at some moment in a 2-hour slot, 2026-10-11 (Kyiv time):
oblast              quietest 2 hours  riskiest 2 hours
Kyiv City            02:00–04:00 36%   20:00–22:00 46%
Lvivska oblast       07:00–09:00 33%   20:00–22:00 48%
Zakarpatska oblast                 —                 —
```

- **`4.3`** — expected number of air raid alerts starting in that oblast on that day (an average, so it can be
  fractional).
- **`02:00–04:00 36%`** — of all 2-hour slots of that day, this one has the lowest chance of an alert being
  active at some moment during it: 36 %. The riskiest slot is shown next to it.
- **`—`** — the oblast had no alerts at all in the last 56 days.
- The chance for **every hour** of every forecast day is in the CSV file (`chance_00` … `chance_23`) and as a
  heatmap in the report.

**How much to trust it:** the chance **levels** are reliable — when the forecast said 35 %, an alert really happened
about 35 % of the time. The **differences between hours** are only a weak hint: in most oblasts alerts happen in
every part of the day. So read "Kyiv City 02:00–04:00 36 % vs 20:00–22:00 46 %" as "roughly a 40 % chance in any
2 hours, a bit higher in the evening".

### How to read the one-paragraph result

It compares the model with simple rules on the latest 28 days: the number of alerts with "average of the last 7
days" and "same weekday last week", and the hourly chances with "the same chance every hour". The current honest
result: both **tie** the simple rule (error ≈ 1.97 alerts per oblast per day; Brier score ≈ 0.187). Details: the
"Results" section of [README.md](README.md).

## Step 4. Open the report

Open **`reports/report.html`** in any web browser (double-click it in your file manager, or on macOS run
`open reports/report.html`). Sections:

1. **Data** — which sources were used and for what period.
2. **Exploration** — when alerts usually happen in each oblast (heatmap), which oblasts have the most alerts,
   whether busy days follow busy days.
3. **H6** — whether filling in missing alert end times with each oblast's median duration beats "+30 minutes".
4. **Test results** — the model vs the two simple rules, by days ahead and by oblast; then "When: hourly alert
   chances" — the Brier score table and a calibration table ("when we said X %, how often was there an alert?");
   then "Details checks" — the 2-hour windows' calibration and how well recent durations predicted real ones.
5. **Forecast** — expected alerts per day, a heatmap of the chance of an alert in each hour of the first day, and the
   quietest / riskiest 2-hour slots.

Other files in `reports/`: `backtest.csv` (every test forecast with the real outcome) and
`forecast_<date>.csv` (the forecast table, see Step 5).

## Step 5. Forecast for the days you choose

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

```bash
python3 run.py --start 2026-10-05 --days 3
```

- `--start` — the first day to forecast, format **YYYY-MM-DD**, Kyiv calendar day. Allowed range: from
  2026-01-01 to the last day in the data (the program tells you the range if you go outside it).
- `--days` — how many days to forecast, **1 to 7** (default 7).

This prints the table and saves `reports/forecast_2026-10-05.csv`. It does **not** re-run the evaluation or
update `report.html`.

**Checking the model on the past:** if you pick a past date, the CSV also contains the column `actual` — what
really happened — so you can compare. The forecast still uses only information available on the morning of
`--start`.

Columns of the CSV: `oblast`, `day`, `expected` (alerts), `actual` (if known), `quietest_2h`, `riskiest_2h`, the 2-hour windows `w00_02` … `w22_00` and durations `dur_…` (Step 6c), `made_on`,
`slot_chance` (only with `--slot`, Step 6), `chance_00` … `chance_23` (chance of an alert at some moment in each hour,
0–1). The CSV opens correctly in Excel.

## Step 6. Ask about your own time slot (commute, shopping, a meeting)

```bash
python3 run.py --start 2026-10-11 --days 2 --slot 08:00-09:30
```

`--slot` takes a Kyiv time span `HH:MM-HH:MM` and prints, for every oblast and day, the **chance that an alert is
active at some moment during it** — e.g. "will an alert catch me on my way to university between 8:00 and 9:30?":

```
Chance of an alert at some moment during 08:00-09:30:
oblast                   2026-10-11  2026-10-12
Chernivetska oblast              6%          8%
Cherkaska oblast                47%         46%
Dnipropetrovska oblast          94%         93%
```

Slots are counted in whole hours (`08:00-09:30` covers the hours 8 and 9) and may cross midnight
(`23:00-01:00`). It also works with the default run: `python3 run.py --slot 15:00-17:00`.

## Step 6b. Only your oblast

```bash
python3 run.py --for 2026-10-14 --oblast lviv
```

Type part of the oblast's name in any case (`lviv`, `kharkiv`, `ivano`). If it fits several (`kyiv` → Kyiv City
and Kyivska oblast) or none, the app lists the names; type more of the name. Names follow the dataset's spelling
(`odeska`, `sumska`), so `odes` works and `odesa` does not. In the menu, choices 2 and 3 ask for it (Enter = all).

## Step 6c. Details: every 2-hour window and how long alerts last

```bash
python3 run.py --for 2026-10-14 --oblast lviv     # one oblast: details are shown automatically
python3 run.py --for 2026-10-14 --details         # all oblasts
```

- **2-hour windows** — the chance of an alert at some moment in 00–02, 02–04 … 22–00. Only the start day's
  chances are adjusted by the predicted number of alerts; later days show the oblast's usual pattern, so they
  repeat. (The "quietest 2 hours" may start at an odd hour, e.g. 13:00–15:00, so it need not be one of these 12.)
  For all oblasts the screen shows the first chosen day; every day is in the CSV.
- **Alert durations** — from alerts of the last 28 days before the forecast moment that have a real "all clear"
  message: *typical* (half the alerts are shorter), *most last* (the middle 80 %), *shortest*, *average*,
  *longest*. A note appears when a few very long records (often a missing "all clear") pull the average up.
  This is recent history, not a model; how well it predicts is in the report (Step 4, "Details checks").
- In the menu, after a forecast for all oblasts: "Show details for all oblasts? [y/N]".
- The CSV also has `made_on`: the day the forecast was made (its start).

## Step 7. Use the newest data

```bash
python3 run.py --fresh
```

Downloads the latest alert files from the
[source repository](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (~40 MB) and new posts from
the Telegram channels, then does the same as menu choice 1 (Step 3; with an option given, no menu is shown). Takes about half a minute plus download time. The fresh
data goes to `data/fresh/`; the stored snapshot is not changed, so `python3 run.py` without `--fresh` still
reproduces the numbers in the README. You can combine it, e.g. `python3 run.py --fresh --days 3` (the forecast
starts from the last day in the fresh data).

Note: the source repository is updated once a day (around 08:00 Kyiv time), so `--fresh` can only be as
recent as that.

## Step 8. (Optional) Run the tests

```bash
python3 run.py test
```

Runs the 70 automatic checks — including the **leakage test**, which proves the forecast never uses information
from after the moment it is made. Expected last line: `OK`. A few lines like `Creating a private environment…`
or `Downloading https://example.invalid/x.csv` may appear in between — they come from tests that simulate
those situations and are not errors.

---

## Troubleshooting

| You see | What to do |
|---|---|
| `python3: command not found` | Install Python (Step 1). On Windows use `python`. |
| `Python 3.9 or newer is required.` | Install a newer Python from python.org. |
| `Could not create .venv (...)` | On Debian/Ubuntu run `sudo apt install python3-venv`, then try again. Or install the libraries yourself: `pip install -r requirements.txt`, then run `python3 -m alerts.main` (same options as `run.py`). |
| `Installing the libraries failed ...` | Check the internet connection and run the same command again. |
| `--start must be a date like 2026-10-05 (YYYY-MM-DD)` | Write the date as year-month-day with dashes. |
| `--start must be between 2026-01-01 and ...` | Pick a date inside the range shown. |
| `argument --days: invalid choice` | Use a number from 1 to 7. |
| `--slot must look like 08:00-09:30` | Write the slot as two Kyiv times with a dash, e.g. `07:30-08:15`. |
| An error ending in `Download failed after 3 attempts` (with `--fresh`) | No internet or the source is unavailable; run without `--fresh` to use the stored data. |
| A `?` instead of a symbol in the output (Windows) | Harmless: the console cannot show that symbol. |
| A "DeprecationWarning" about NumPy (Python 3.13) | Harmless warning from the libraries; the results are not affected. |

## What the forecast is — and is not

- It is a **statistical forecast for planning**, built from alert history and public Telegram posts. It
  is **not** a warning system: always follow the official air raid alerts.
- Its honest accuracy is in the README: on the latest 28 days the number of alerts is about as good as "the
  average of the last 7 days", and the hourly chances are reliable as levels but only a weak guide to which hour
  is quieter.
