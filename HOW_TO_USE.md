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

## Step 3. First run — evaluation, report and a 7-day forecast

```bash
python3 run.py
```

What happens:

1. **First time only (~1 minute, needs internet):** it creates a private Python environment in a folder
   called `.venv` and installs four libraries into it (pandas, numpy, scikit-learn, matplotlib). Nothing is
   installed into your system Python. Later runs skip this step.
2. **Evaluation (~20 seconds):** using the data stored in the project (`data/snapshot/`), it tests the model
   honestly on the latest 28 days: for every day it pretends to be on the morning of that day, retrains the
   model on earlier days only, forecasts, and compares with what really happened.
3. It prints a **one-paragraph result** and a **7-day forecast table**, and saves files to the `reports/` folder.

### How to read the forecast table

```
Expected alerts per day | hours expected to hold half of them (Kyiv time):
day                        2026-10-11           2026-10-12   ...
Kyivska oblast           12.4 | 09:00–21:00   14.2 | 09:00–21:00
Volynska oblast           4.2 | 17:00–04:00    4.0 | 17:00–04:00
Zakarpatska oblast        0.0 | —              0.0 | —
```

- **`12.4`** — expected number of air raid alerts starting in that oblast on that day (an average, so it
  can be fractional).
- **`09:00–21:00`** — the shortest span of hours (Kyiv time) expected to contain **half** of that day's alerts.
  It can cross midnight: `17:00–04:00` means from 5 pm until 4 am the next morning.
- **`—`** — the oblast had no alerts in the last 28 days, so there is no meaningful time window.

### How to read the one-paragraph result

It compares the model with two simple rules ("average of the last 7 days" and "same weekday last week")
on the latest 28 days. The current honest result: the model **ties** the 7-day average (error ≈ 1.97 alerts
per oblast per day) and beats "same weekday last week". The time windows hold about half of real alerts, as
promised, but are wide (≈ 10 hours). Details: the "Results" section of [README.md](README.md).

## Step 4. Open the report

Open **`reports/report.html`** in any web browser (double-click it in your file manager, or on macOS run
`open reports/report.html`). Sections:

1. **Data** — which sources were used and for what period.
2. **Exploration** — when alerts usually happen in each oblast (heatmap), which oblasts have the most alerts,
   whether busy days follow busy days.
3. **H6** — whether filling in missing alert end times with each oblast's median duration beats "+30 minutes".
4. **Test results** — the model vs the two simple rules, by days ahead and by oblast.
5. **Forecast** — the 7-day forecast as a heatmap and a table.

Other files in `reports/`: `backtest.csv` (every test forecast with the real outcome) and
`forecast_<date>.csv` (the forecast table, see Step 5).

## Step 5. Forecast from a date you choose

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

Columns of the CSV: `oblast`, `day`, `expected` (alerts), `actual` (if known), `peak_hours` (the 3 most likely
start hours), `main_window` (hours holding half of the alerts), `h00` … `h23` (expected alerts in each hour).

## Step 6. Use the newest data

```bash
python3 run.py --fresh
```

Downloads the latest alert files from the
[source repository](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (~40 MB) and new posts from
the Telegram channels, then does the same as Step 3. Takes about half a minute plus download time. The fresh
data goes to `data/fresh/`; the stored snapshot is not changed, so `python3 run.py` without `--fresh` still
reproduces the numbers in the README. You can combine it, e.g. `python3 run.py --fresh --days 3` (the forecast
starts from the last day in the fresh data).

Note: the source repository is updated once a day (around 08:00 Kyiv time), so `--fresh` can only be as
recent as that.

## Step 7. (Optional) Run the tests

```bash
python3 run.py test
```

Runs the 53 automatic checks — including the **leakage test**, which proves the forecast never uses information
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
| An error ending in `Download failed after 3 attempts` (with `--fresh`) | No internet or the source is unavailable; run without `--fresh` to use the stored data. |
| A `?` instead of a symbol in the output (Windows) | Harmless: the console cannot show that symbol. |
| A "DeprecationWarning" about NumPy (Python 3.13) | Harmless warning from the libraries; the results are not affected. |

## What the forecast is — and is not

- It is a **statistical forecast for planning**, built from alert history and public Telegram posts. It
  is **not** a warning system: always follow the official air raid alerts.
- Its honest accuracy is in the README: on the latest 28 days it is about as good as "the average of the last
  7 days", and its time windows are wide.
