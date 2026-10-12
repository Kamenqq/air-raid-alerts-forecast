# Air Raid Alerts Forecast — Prototype Design

Date: 2026-10-11 · Status: approved; revised 2026-10-12 (target = volunteer data, main window added — see §10)
Deadline: Mon 12 Oct 2026, 23:59 Kyiv

## 1. Goal

A Python app that forecasts, **per oblast and per day, for 1–7 days ahead**:
- **how many air raid alerts** will start that day, and
- **at which hours** they are most likely (top-3 hours + a 24-hour profile).

It is honest by construction: it uses only information available before the forecast is made, it is
compared against simple baselines on the most recent period, and the report states plainly whether the
model beats them.

Non-goals for the prototype: raion-level forecasts, hyperparameter tuning, LLM-based text understanding,
neighbour/lead-lag features (H4), regime comparison (H5), daylight-saving edge cases, simulating the
repo's daily update lag, tracking Telegram edit times.

## 2. What the user does

| Command | What happens |
|---|---|
| `python3 run.py` | Creates `.venv` if missing, installs `requirements.txt`, then runs the **reproducible mode**: loads the frozen snapshot, evaluates on the test period, writes `reports/report.html` + CSVs, and forecasts 7 days starting 2026-10-10 (the day the snapshot ends). |
| `python3 run.py --start 2026-10-01 --days 3` | Forecast made at 00:00 Kyiv of `--start`, for `--days` days (1–7, default 7). Prints a table and writes `reports/forecast_<start>.csv` + heatmap. |
| `python3 run.py --fresh` | Downloads the latest repo CSVs and recent Telegram posts first, then the same as above. Default `--start` (both modes) = the last, partly covered day in the data, so features only use complete days. |

Rules for `--start`:
- before 2026-01-01 → error (needs ≥ 1 month of training examples and 28 days of feature history);
- the forecast always uses data available at 00:00 Kyiv of `--start` and a model retrained on days before it
  (see §5), so backtests for "last week" are honest; if `--start` is after the end of the data, the output
  says which date the data ends on.

Windows users type `python` instead of `python3`; the README says so.

## 3. Data sources (all treated as equally trustworthy and combined)

| Source | Content | Period in snapshot | Role |
|---|---|---|---|
| `official_data_en.csv` (Vadimkin repo) | alert intervals at oblast/raion/hromada level | 2025-09-01 → 2026-09-07 | alert history |
| `volunteer_data_en.csv` (Vadimkin repo, eTryvoga) | oblast-level alert intervals, `naive` flag | 2025-09-01 → 2026-10-10 05:00 UTC | alert history |
| Telegram `air_alert_ua` (public preview) | fixed-format 🔴 start / 🟢 end posts per raion/hromada/oblast | 2026-08-25 → 2026-09-07 (channel's last post) | alert history; covers the CSV's outage 2026-08-29…09-02 if the channel has posts there; agreement check vs CSV |
| Telegram `tryvoga_chomu` (public preview) | free-text threat reports | 2025-11-24 → snapshot cutoff | text features |

Facts behind these choices are in CLAUDE.md ("Data facts"). air_alert_ua stopped posting on
2026-09-07 06:09 UTC (confirmed by the user in the app), and its history very likely duplicates
`official_data_en.csv`, so only a two-week window is collected. The fetcher still runs in `--fresh` and
picks up new posts if the channel resumes.

Snapshot cutoff: **2026-10-10 05:00 UTC**. Last full Kyiv day in data: **2026-10-09**.

### 3.1 Cleaning
- Read with pandas (handles CRLF). Drop exact duplicate rows (the official file duplicates 2022–2025 rows).
- Exclude Luhanska oblast and Crimea. Keep 24 oblasts (Kyiv City separate from Kyivska oblast).
- Map every record to its oblast: CSV rows have an `oblast`/`region` column; air_alert_ua raion/hromada
  names (Ukrainian) are mapped with `data/region_map.csv` (Ukrainian raion/hromada name → oblast), built
  once from `official_data_uk.csv` by `alerts/snapshot.py` and committed.
- **Naive alerts (H6, user version):** `finished_at = started_at + median duration of non-naive eTryvoga
  alerts of the same oblast`. The median is computed once, from 2025-09-01 → 2025-11-30 — before any training
  example — so it cannot leak into training or test.

### 3.2 Combining into alert episodes
Per oblast: collect intervals from all sources, sort by start, merge any that overlap or touch → **episodes**.
- **Daily count** for oblast *o*, day *D* = episodes whose start falls on Kyiv calendar day *D*.
- Heads-up (not handled in the prototype): in frontline oblasts raion alerts overlap almost continuously,
  so episodes become long and counts low. Fallback if misleading: "hours under alert per day".

### 3.3 As-of rule (leakage protection) — as implemented
There is no single `available_at()` function; the rule is enforced in two places and proven by a test:
- **Features** (`features.make_rows`): every statistic for forecast day t0 is shifted with `shift(1)` (or `shift(7)`
  read at the target day D = t0 + k − 1 ≤ t0 + 6), so it only uses Kyiv days before t0. Daily counts and hourly
  profiles use alert **starts**, so a day before t0 is complete at 00:00 Kyiv of t0; the last, partly covered data
  day is NaN. Telegram counts use posts published before t0.
- **Training** (`model.training_rows`): only rows whose target day is before t0; the model is retrained per t0.
- **Proof** (`tests/test_features.py::LeakageTest`): `episodes.clip_as_of(intervals, T)` builds the data as it was
  known at T (alerts started before T, running alerts cut at T); features and profiles from that data must equal the
  ones from the full data. `clip_as_of` and `config.kyiv_day_start_utc` exist for this proof.
- Residual risk: Telegram posts may have been edited after publication (only the final text is visible).

## 4. Forecast definition

A forecast is made at **T = 00:00 Kyiv of the start day**. For lead *k* ∈ 1…7 it targets day
*D = start + (k − 1)* days.

### 4.1 Part A — how many alerts (count model)
Model: scikit-learn `HistGradientBoostingRegressor(loss="poisson", random_state=0)`, default settings,
oblast as a categorical feature. No tuning (avoids tuning on the test period).

Features for (oblast *o*, forecast time *T*, lead *k*), all known before 00:00 Kyiv of t0 (see §3.3):
- calendar: `lead_days` (k), `weekday` of D;
- `oblast` (categorical);
- own history: episodes on the day before T; mean per day over the last 7 and 28 days;
  episodes on the same weekday one week before D (always before T for k ≤ 7);
- national: total episodes across oblasts on the day before T and mean per day over the last 7 days;
- tryvoga_chomu (last 24 h and last 7 days before T): posts mentioning oblast *o* (stem dictionary,
  e.g. "Київщин", "Харківщин"); national counts of four threat groups: drones (шахед, бпла, дрон),
  missiles (ракет, балістик, калібр, кинджал, іскандер), aviation (міг-31, ту-95, ту-22, авіац),
  massive (масован, масштабн). Lower-case substring matching after stripping the channel's boilerplate.

### 4.2 Part B — when (hourly profile)
For oblast *o*: the share of episode starts in each Kyiv hour 0–23 over the 28 days before T
(uniform if there were none). Expected alerts in hour *h* of day D = Part A count × share(h).
**Peak hours** = the 3 hours with the largest share.

## 5. Training and evaluation

### 5.1 Training rule
For a forecast at T, the model is trained only on examples whose target day ends before T
(D_train < day of T), each example built with its own as-of rule. The model is **retrained for every
forecast time**; it is small (≈24 oblasts × ~300 days × 7 leads ≈ 50k rows), so this takes seconds.
Training examples start at 2025-12-01 (after the switch to raion-level alerts; features need 28 days
of history before that).

### 5.2 Test protocol
- Test days: the last 28 full Kyiv days of the snapshot, **2026-09-12 → 2026-10-09**.
- For every test day D and lead k ∈ 1…7, make the forecast at T = D − (k − 1) days and compare with the
  actual count. (The earliest forecast time is 2026-09-06.)
- Baselines, same as-of rule: **B1** mean daily count over the last 7 days; **B2** count on the same
  weekday one week before D.

### 5.3 Metrics
- **MAE** (average absolute error, alerts/day) for model, B1, B2 — overall, per lead day, per oblast.
- **Skill** = 1 − MAE_model / MAE_best_baseline (positive = model better).
- **Bias** = mean predicted − mean actual (checks the predicted over-forecast after 2026-09-07,
  when the official CSV ends and the test period becomes eTryvoga-only).
- **Peak-hour hit rate**: share of actual test-day episodes that started in the predicted top-3 hours
  (random guess ≈ 12.5%).
- README table in plain words: "on average we miss by X alerts per day (baseline: Y);
  Z% of alerts started in our predicted peak hours".

## 6. Report (`reports/report.html`, charts as embedded PNG)
1. Data summary: sources, periods, rows per source, air_alert_ua vs CSV agreement for its two-week window.
2. Exploration on the **training period only**: H1 heatmap (oblast × Kyiv hour, share of starts);
   H2 mean daily count per oblast; H3 day-to-day autocorrelation of daily counts per oblast.
3. H6 check: on non-naive eTryvoga alerts in the training period, hide the real end and compare the error
   of "+30 min" vs "per-oblast median" (median from earlier alerts only).
4. Test results: metrics from §5.3, MAE by lead day chart, per-oblast table, bias.
5. The forecast for the requested days: table + heatmap (oblast × hour, expected alerts).

## 7. Code layout
```
run.py                launcher, stdlib only: creates .venv, installs requirements, runs alerts.main
requirements.txt      pandas, numpy, scikit-learn, matplotlib (version ranges that support 3.9–3.13)
alerts/
  sources.py          download repo CSVs; fetch Telegram preview pages (urllib, html.parser, 2 s delay)
  parse.py            CSV cleaning, air_alert_ua post parser, tryvoga_chomu text cleaning + keyword counts
  episodes.py         naive-end imputation, region mapping, merging into episodes, daily counts
  features.py         feature rows (shift-based as-of rule), hourly profiles
  model.py            train (with cutoff) / predict
  evaluate.py         test protocol and metrics
  report.py           HTML report + charts
  snapshot.py         maintainer tool (run once, not by reviewers): builds data/snapshot/* and
                      data/region_map.csv. One-time cost: tryvoga_chomu ≈ 1,300 requests (~45 min),
                      air_alert_ua ≈ 350 requests (~12 min), official_data_uk.csv download (41 MB).
  main.py             CLI (--fresh, --start, --days)
data/snapshot/        official_data_en.csv (deduplicated), volunteer_data_en.csv (both trimmed ≥ 2025-09-01),
                      air_alert_ua_posts.csv, tryvoga_chomu_posts.csv
data/region_map.csv
tests/                unittest (stdlib)
.github/workflows/ci.yml
```

## 8. Testing
Using `unittest` from the standard library, written test-first for the critical parts:
- **Leakage test:** features, profiles and baselines at T are identical when computed on the full data and on
  data with everything after T deleted.
- **Training cutoff test:** the training set for T contains no target day ≥ day of T.
- Episode merging on a small hand-made example.
- air_alert_ua parser on saved sample posts (single region, bullet list, threat-only posts ignored).
- tryvoga_chomu cleaning: boilerplate stripped, keyword groups counted.

**CI** (GitHub Actions): ubuntu / macOS / windows × Python 3.9 and 3.13 → `python run.py` (reproducible mode)
and `python -m unittest`.

## 9. Known limitations (stated in README)
- The test period is eTryvoga-only (other sources end 2026-09-07); combined-source counts before that date
  are higher, so a bias is expected and measured.
- Episode counts in frontline oblasts are distorted by merging (see 3.2).
- Lviv/Volyn: eTryvoga records many more alerts than the raion-level file since July 2026 (cause unknown).
- Telegram text may have been edited after publication.
- The live repo updates once a day (~05:00 UTC); this lag is not simulated in backtests.

## 10. Revision 2026-10-12 (after the first full run)
- **Target changed to eTryvoga (volunteer CSV) alerts only.** With all sources combined, the official CSV and
  air_alert_ua ended on 2026-09-07, so the target meant something different in training (merging absorbed
  volunteer alerts: ~87 vs ~130 alerts/day nationally) than in the test period. Result: skill -16%, bias -0.57.
  A throwaway experiment with the volunteer target everywhere gave skill ≈ 0%, bias -0.09; the user chose it.
  The other sources stay imported for the data checks in report section 1.
- **New output, "main window":** the shortest continuous span of hours (may cross midnight) expected to hold at
  least half of the day's alerts, from the same 28-day hourly profile. Evaluated by coverage (share of real
  alerts inside the window; promised 50%) and mean window length, compared with a random window of the same
  length (length / 24).

## 11. Revision 2026-10-12 (second round, after independent reviews)
- **H4 neighbouring oblasts:** border table for the 24 oblasts (Luhansk and Crimea excluded: permanent alerts, no
  daily variation, not in the data) and two features — neighbours' alerts yesterday and 7-day average. Decided on a
  **validation period** = the 28 days before the test period (2026-08-16 … 09-12): count MAE 2.028 → 2.010 → kept.
  Test period: unchanged tie with the last-7-days average.
- **"When" for a resident planning the day:** the main window and top-3 hours are retired. New output: the chance
  that an alert is active at some moment of each Kyiv hour (from an active-hours grid), the quietest / riskiest
  2-hour slots, and any slot via `--slot HH:MM-HH:MM`. On validation, per-hour chances from 28 days (with or without
  scaling by the predicted count) lost to "the same chance every hour"; the shown rule — half the oblast's hourly
  pattern + half its flat level over 56 days, scaled by the predicted count for 1 day ahead only — beat it at every
  horizon. Test period (measured once): Brier 0.1871 vs 0.1870 (1 day), 0.1904 vs 0.1893 (7 days); calibration
  close. Evaluated with the Brier score and a calibration table.
- **Security review fixes:** payment cards, links, emails and @usernames are replaced with placeholders in stored and
  downloaded Telegram posts (model inputs proven identical); upper version limits for libraries; read-only CI token.
- **Code review fixes:** tests now guard the `--start` training cutoff and the volunteer-only target filter (filter
  moved into `build()`); clean `.venv` rebuild and Ctrl-C cleanup; retry of cut-off downloads; Excel-friendly CSV.
