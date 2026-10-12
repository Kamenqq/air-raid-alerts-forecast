# Air raid alerts forecast — Ukraine

[![run](https://github.com/Kamenqq/air-raid-alerts-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Kamenqq/air-raid-alerts-forecast/actions/workflows/ci.yml)

A Python app that forecasts, **for each of 24 oblasts and each of the next 1–7 days**:

- **how many air raid alerts** will start that day;
- **when** — the **chance that an alert is active at some moment in each hour** (Kyiv time), the **quietest and
  riskiest 2-hour slots**, and the chance for **any time slot you ask about** (e.g. your commute: `--slot 08:00-09:30`).

It is evaluated honestly on the latest 28 days against simple baselines, with no information from after the
moment a forecast is made. Built for the KSE Agentic AI School (Stage 2).

## Quick start

**New to the command line? Follow the step-by-step guide: [HOW_TO_USE.md](HOW_TO_USE.md).**

```bash
git clone https://github.com/Kamenqq/air-raid-alerts-forecast.git
cd air-raid-alerts-forecast
python3 run.py            # Windows: python run.py
```

The first run creates a private environment in `.venv` and installs pandas, numpy, scikit-learn and matplotlib
(needs internet, ~1 minute). Then it runs the 28-day evaluation on the data snapshot in the repo (~20 seconds),
prints a 7-day forecast and writes the outputs to the `reports/` folder (`report.html`, `backtest.csv`,
`forecast_<date>.csv`; the folder is created by the run and not stored in git).

| Command | What it does |
|---|---|
| `python3 run.py` | Evaluation + HTML report + 7-day forecast from the last data day (`--days N` for fewer days) |
| `python3 run.py --start 2026-10-01 --days 3` | Forecast only (no evaluation, report not updated) from any day (2026-01-01 … last data day), 1–7 days → `reports/forecast_<date>.csv`. For past days the CSV also shows what actually happened |
| `python3 run.py --start 2026-10-11 --slot 15:00-17:00` | Adds the chance of an alert at some moment during that Kyiv time slot, for every oblast and day |
| `python3 run.py --fresh` | Downloads the latest data first (repo CSVs + new Telegram posts), then the same |
| `python3 run.py test` | Runs the 64 unit tests |

Requirements: Python 3.9+ on macOS, Linux or Windows. Fallback if `.venv` cannot be created
(e.g. Debian/Ubuntu without `python3-venv`): `pip install -r requirements.txt`, then `python -m alerts.main`.
CI runs the tests and the full app on Ubuntu, macOS and Windows with Python 3.9, 3.13, 3.14 and 3.15.

Example output (forecast made for 2026-10-11):

```
Expected number of alerts per day:
oblast              2026-10-11  2026-10-12  2026-10-13 ...
Kyiv City                  4.3         4.4         5.3
Volynska oblast            4.0         3.9         5.0

Chance of an alert at some moment in a 2-hour slot, 2026-10-11 (Kyiv time):
oblast              quietest 2 hours  riskiest 2 hours
Kyiv City            02:00–04:00 36%   20:00–22:00 46%
Lvivska oblast       07:00–09:00 33%   20:00–22:00 48%
Zakarpatska oblast                 —                 —      (no alerts in the last 56 days)
```

## Results on the latest 28 days (2026-09-13 … 2026-10-10)

**How many alerts** — average error in alerts per oblast per day (MAE, lower is better), 4,704 forecasts:

| Days ahead | Model | Baseline: last-7-days average | Baseline: same weekday last week |
|---|---|---|---|
| 1 | **1.84** | 1.87 | 2.46 |
| 2 | **1.90** | 1.94 | 2.46 |
| 3 | **1.94** | 1.96 | 2.46 |
| 4 | 2.01 | **1.97** | 2.46 |
| 5 | 2.06 | **2.00** | 2.46 |
| 6 | **2.02** | 2.02 | 2.46 |
| 7 | **2.03** | 2.04 | 2.46 |
| **All** | **1.97** | **1.97** | 2.46 |

**When** — hourly alert chances, Brier score (average of (chance − what happened)², lower is better):

| Days ahead | Shown in the forecast | Same chance every hour | Each hour's own 28-day share |
|---|---|---|---|
| 1 | 0.1871 | **0.1870** | 0.1875 |
| 3 | 0.1892 | **0.1881** | 0.1887 |
| 7 | 0.1904 | **0.1893** | 0.1902 |

Calibration of the shown chances (1 day ahead): when the forecast said 35 %, an alert happened 35 % of the time;
45 % → 46 %; 74 % → 76 %; only the lowest chances run a bit low (3 % → 6 %).

**In plain words:**
- The count model **ties** "average of the last 7 days" (skill ≈ 0 %): slightly better 1–3 days ahead, slightly
  worse 4–5 days ahead; both clearly beat "same weekday last week". Bias −0.09 alerts/day (−0.11 on Python 3.13–3.15, which install a newer
  scikit-learn; the errors and Brier scores are the same). Per oblast: best
  Chernivetska (+11 %) and Dnipropetrovska (+10 %), worst Kyiv City (−32 %) and Ivano-Frankivska (−25 %).
- The **hourly chances are trustworthy as levels** (well calibrated): "about 46 % chance of an alert during an
  evening walk in Kyiv" is a reliable statement. But **which hour is quieter is only a weak hint**: on the test
  period the hour-by-hour chances tie the flat "same chance every hour" baseline 1 day ahead and are slightly worse
  3–7 days ahead. In most oblasts alerts happen in every part of almost every day, so there is little hour-of-day
  pattern to find.
- Daily alert counts behave mostly like a persistent level; recent history explains most of what is predictable.
  Telegram keyword features and neighbouring oblasts added little.

## How it works

1. **Data** (`alerts/parse.py`, `alerts/sources.py`): eTryvoga alert CSV (the target), official CSV and the
   `air_alert_ua` Telegram channel (data checks), and the `tryvoga_chomu` Telegram channel (threat-text features).
2. **Alerts per day** (`alerts/episodes.py`): alerts per oblast counted per **Kyiv** calendar day. Alerts without an
   "all clear" message get the oblast's median duration (hypothesis H6).
3. **Features** (`alerts/features.py`): what is known at 00:00 Kyiv of the forecast day — counts yesterday / last 7 /
   last 28 days, same weekday last week, national totals, **neighbouring oblasts** yesterday / last 7 days (H4),
   Telegram mentions of the oblast and of threat groups (drones, missiles, aviation, massive attack).
4. **Model** (`alerts/model.py`): `HistGradientBoostingRegressor(loss="poisson")`, default settings (no tuning),
   **retrained for every forecast date** on days whose outcome was already known.
5. **When** (`alerts/features.py`): an "active hours" grid records in which Kyiv hours each oblast had an alert
   active. The shown chance for an hour = half the oblast's own share of days with an alert in that hour + half its
   average over all hours, over the last 56 days; for 1 day ahead it is scaled by the predicted count (busier day →
   higher chances). Slots (2-hour, `--slot`) use the same rule.
6. **Evaluation** (`alerts/evaluate.py`) and **report** (`alerts/report.py`).

**Choices were made on a validation period, not on the test.** Whether to keep the neighbour features (H4) and which
"when" method to show were decided on the 28 days *before* the test period (2026-08-16 … 09-12); the test period was
then measured once. On validation, H4 improved the count error from 2.028 to 2.010, and the shown chance method
beat the flat baseline at every horizon — on the test period both advantages shrank to a tie (reported above).

**No data leakage — and how it is proven.** A forecast for day D is made at 00:00 Kyiv of D and uses only alerts
that started before then, Telegram posts published before then, and a model trained on earlier days. The leakage
test computes all features and hourly chances once with the full data and once with everything after the forecast
moment deleted — they must be identical. It was checked by deliberately breaking the code (each break made a test
fail) and by two independent review agents that rebuilt 8 real forecast days from cut data (identical results).

## Data sources

| Source | Used for | Period in the snapshot |
|---|---|---|
| [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) `volunteer_data_en.csv` (eTryvoga) | **forecast target** | 2025-09-01 → 2026-10-11 |
| same repo, `official_data_en.csv` | data checks; H6 comparison | 2025-09-01 → 2026-09-07 (file no longer updated) |
| Telegram [`air_alert_ua`](https://t.me/air_alert_ua) (official alert channel) | data checks | 2026-08-25 → 2026-09-07 (channel stopped posting) |
| Telegram [`tryvoga_chomu`](https://t.me/tryvoga_chomu) («Чому тривога?») | text features | 2025-11-24 → 2026-10-11 (22,059 posts) |

The default run uses the frozen snapshot in `data/snapshot/` so results are reproducible; `--fresh` refreshes it.
Telegram data comes from the public web preview (`t.me/s/<channel>`); no login or API key. In the stored and
downloaded posts, payment card numbers, links, emails and @usernames are replaced with placeholders (`[card]`,
`[link]`, …); the model only uses place and threat words, which are unchanged. All data belongs to its sources.

## Hypotheses checked (details in `reports/report.html`)

- **H1** alerts follow an hour-of-day pattern that differs by oblast — heatmap in the report; the pattern is weak
  in most oblasts (see "When" above).
- **H2** oblasts differ strongly in alert frequency — bar chart (frontline vs western oblasts).
- **H3** busy days are followed by busy days — lag-1 autocorrelation per oblast.
- **H4** neighbouring oblasts help — small gain on validation (count error 2.028 → 2.010), no visible effect on test.
- **H6** missing alert ends: per-oblast **median** duration vs the default +30 min, tested on 27,846 alerts whose
  real end is known: error 23.9 vs 24.9 minutes — slightly better.

## Known limitations

- The count model does **not beat** the 7-day average overall — it ties. The hourly chances are well calibrated but
  their hour-to-hour differences add little over a flat chance.
- The target is eTryvoga data only. A first version combined all sources, but the official CSV and `air_alert_ua`
  stopped on 2026-09-07, so the target changed meaning between training and testing (−16 % skill, −0.57 bias).
  Using one consistent source fixed that. **Honest caveat:** this choice was made after seeing results on the same
  28 test days (later choices used the validation period instead).
- A time slot crossing midnight (e.g. `23:00-01:00`) uses the oblast's typical late-evening and early-morning hours
  of a day, not specifically "tonight into tomorrow morning".
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
tests/                64 unit tests (python3 run.py test)
docs/superpowers/     design spec and implementation plan
.claude/agents/       project agents: leakage-auditor, repro-tester, data-checker, interview-coach
```

## How this was built

With AI as the main engineering tool (Claude Code): requirements and design were agreed step by step
(`docs/superpowers/specs/`), turned into a test-first implementation plan (`docs/superpowers/plans/`), and built
one task at a time with risk checks after each task. It was then independently checked by review agents — a
leakage audit (no leakage found, verified on 8 real forecast days), a reproducibility test (fresh clone → run), a
code review and a security review — and their findings were fixed test-first. The full conversation log is
submitted separately with the application.
