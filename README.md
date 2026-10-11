# Air raid alerts forecast — Ukraine

A Python app that forecasts, **for each of 24 oblasts and each of the next 1–7 days**:

- **how many air raid alerts** will start that day;
- **from what time to what time** most of them are expected — the shortest span of hours holding half of the
  day's alerts (e.g. `16:00–21:00`), plus the top-3 single hours.

It is evaluated honestly on the latest 28 days against two simple baselines, with no information from after the
moment a forecast is made. Built for the KSE Agentic AI School (Stage 2).

## Quick start

```bash
git clone https://github.com/Kamenqq/air-raid-alerts-forecast.git
cd air-raid-alerts-forecast
python3 run.py            # Windows: python run.py
```

The first run creates a private environment in `.venv` and installs pandas, numpy, scikit-learn and matplotlib
(needs internet, ~1 minute). Then it runs the 28-day evaluation on the data snapshot in the repo (~20 seconds),
prints a 7-day forecast and writes `reports/report.html`.

| Command | What it does |
|---|---|
| `python3 run.py` | Evaluation + HTML report + 7-day forecast from the last data day |
| `python3 run.py --start 2026-10-01 --days 3` | Forecast from any day (2026-01-01 … last data day), 1–7 days. For past days the CSV also shows what actually happened |
| `python3 run.py --fresh` | Downloads the latest data first (repo CSVs + new Telegram posts), then the same |
| `python3 run.py test` | Runs the 48 unit tests |

Requirements: Python 3.9+ on macOS, Linux or Windows. Fallback if `.venv` cannot be created
(e.g. Debian/Ubuntu without `python3-venv`): `pip install -r requirements.txt`, then `python -m alerts.main`.
CI runs the tests and the full app on Ubuntu, macOS and Windows with Python 3.9 and 3.13.

Example output (expected alerts | hours expected to hold half of them, Kyiv time):

```
Kyivska oblast    12.4 | 09:00–21:00
Volynska oblast    4.2 | 17:00–04:00
Zakarpatska oblast 0.0 | —            (no alerts in the last 28 days)
```

## Results on the latest 28 days (2026-09-13 … 2026-10-10)

Average error in alerts per oblast per day (MAE, lower is better), over 4,704 forecasts:

| Days ahead | Model | Baseline: last-7-days average | Baseline: same weekday last week |
|---|---|---|---|
| 1 | **1.83** | 1.87 | 2.46 |
| 2 | **1.91** | 1.94 | 2.46 |
| 3 | **1.95** | 1.96 | 2.46 |
| 4 | 2.01 | **1.97** | 2.46 |
| 5 | 2.04 | **2.00** | 2.46 |
| 6 | 2.03 | **2.02** | 2.46 |
| 7 | **2.03** | 2.04 | 2.46 |
| **All** | **1.97** | **1.97** | 2.46 |

**In plain words:**
- The count model **ties the "average of the last 7 days" baseline** (skill ≈ 0%): slightly better 1–3 days ahead,
  slightly worse 4–6 days ahead. Both clearly beat "same weekday last week". Bias −0.09 alerts/day (almost none).
  Daily alert counts behave mostly like a persistent level; recent history explains most of what is predictable.
- **"Half of the alerts" windows held 49% of real alerts** (they promise 50% — well calibrated), but they are wide:
  **9.8 hours on average**. A randomly placed window of the same length would hold about 41%.
- **Top-3 hours** caught 15% of alerts (a random guess: 12.5%) — barely better than random.
- Per oblast: best vs baseline Dnipropetrovska (+11%) and Kyivska (+8%); worst Kyiv City (−33%).

## How it works

1. **Data** (`alerts/parse.py`, `alerts/sources.py`): eTryvoga alert CSV (the target), official CSV and the
   `air_alert_ua` Telegram channel (data checks), and the `tryvoga_chomu` Telegram channel (threat-text features).
2. **Episodes** (`alerts/episodes.py`): alerts per oblast, counted per **Kyiv** calendar day. Alerts without an
   "all clear" message get the oblast's median duration (hypothesis H6).
3. **Features** (`alerts/features.py`): what is known at 00:00 Kyiv of the forecast day — counts yesterday /
   last 7 / last 28 days, same weekday last week, national totals, Telegram mentions of the oblast and of
   threat groups (drones, missiles, aviation, massive attack) in the last day / 7 days.
4. **Model** (`alerts/model.py`): `HistGradientBoostingRegressor(loss="poisson")`, default settings (no tuning),
   **retrained for every forecast date** on days whose outcome was already known.
5. **When** (`alerts/features.py`): a 28-day hourly profile per oblast → top-3 hours and the shortest window
   holding half of the alerts.
6. **Evaluation** (`alerts/evaluate.py`) and **report** (`alerts/report.py`).

**No data leakage — and how it is proven.** A forecast for day D is made at 00:00 Kyiv of D and uses only alerts
that started before then, Telegram posts published before then, and a model trained on earlier days. The leakage
test computes all features once with the full data and once with everything after the forecast moment deleted —
they must be identical. It was checked by deliberately breaking the code three ways (each break made the test fail).

## Data sources

| Source | Used for | Period in the snapshot |
|---|---|---|
| [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) `volunteer_data_en.csv` (eTryvoga) | **forecast target** | 2025-09-01 → 2026-10-11 |
| same repo, `official_data_en.csv` | data checks; H6 comparison | 2025-09-01 → 2026-09-07 (file no longer updated) |
| Telegram [`air_alert_ua`](https://t.me/air_alert_ua) (official alert channel) | data checks | 2026-08-25 → 2026-09-07 (channel stopped posting) |
| Telegram [`tryvoga_chomu`](https://t.me/tryvoga_chomu) («Чому тривога?») | text features | 2025-11-24 → 2026-10-11 (22,059 posts) |

The default run uses the frozen snapshot in `data/snapshot/` so results are reproducible; `--fresh` refreshes it.
Telegram data comes from the public web preview (`t.me/s/<channel>`); no login or API key.

## Hypotheses checked (details in `reports/report.html`)

- **H1** alerts follow an hour-of-day pattern that differs by oblast — heatmap in the report.
- **H2** oblasts differ strongly in alert frequency — bar chart (frontline vs western oblasts).
- **H3** busy days are followed by busy days — lag-1 autocorrelation per oblast.
- **H6** missing alert ends: per-oblast **median** duration vs the default +30 min, tested on 27,846 alerts whose
  real end is known: error 23.9 vs 24.9 minutes — slightly better.

## Known limitations

- The target is eTryvoga data only. A first version combined all sources, but the official CSV and `air_alert_ua`
  stopped on 2026-09-07, so the target changed meaning between training and testing (−16% skill, −0.57 bias).
  Using one consistent source fixed that; the other sources are kept as checks.
- `air_alert_ua` and the official CSV both have an outage on 2026-08-30 … 09-01.
- Since July 2026 eTryvoga records far more alerts in Lviv and Volyn than the official data (cause unknown).
- Telegram keyword matching is approximate (e.g. "дніпр" matches the city and the river); posts may have been
  edited after publication and only the final text is visible.
- The live repo updates once a day (~05:00 UTC); this delay is not simulated in the backtest.
- One 28-day test period: results on another period may differ.

## Project structure

```
run.py                one-command launcher (standard library only)
alerts/               config, parse, sources, snapshot, episodes, features, model, evaluate, report, main
data/snapshot/        frozen data used by the default run; data/region_map.csv (Ukrainian raion → oblast)
tests/                48 unit tests (python3 run.py test)
docs/superpowers/     design spec and implementation plan
.claude/agents/       project agents used for independent checks (leakage, reproducibility, data)
```

## How this was built

With AI as the main engineering tool (Claude Code): requirements and design were agreed step by step
(`docs/superpowers/specs/`), turned into a test-first implementation plan (`docs/superpowers/plans/`), and built
one task at a time with risk checks after each task. The full conversation log is part of the submission.
