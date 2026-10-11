# Air Raid Alerts Forecast — KSE Agentic AI School, Stage 2

## Task
Python mini-project: time series forecasting of air raid alerts in Ukraine.
Deadline: **Mon 12 Oct 2026, 23:59 Kyiv time**.
Submission: (1) GitHub repo with working code, (2) full AI conversation log,
(3) ~100-word plain-text reflection (what went wrong / how we adjusted / why final > first).

**Full design: `docs/superpowers/specs/2026-10-11-air-raid-alerts-forecast-design.md`** (source of truth for details).

## Hard rules
- Use the most recent data; evaluate on the latest period (time-based split, never random).
- **No data leakage.** A forecast made at T uses only information available before T — features, baselines,
  imputation, training examples (target day before T) and Telegram posts (published before T).
  Never explore or tune on the test period.
- The result matters as much as the process: the product must do what we claim and be useful.
  Always compare against simple baselines; report honestly if the model does not beat them.
- **Reproducibility:** the review committee must be able to simply clone the repo and run it.
  No manual steps, logins, API keys or machine-specific paths. Python 3.9+ on macOS/Linux/Windows.
  `python3 run.py` (stdlib launcher → `.venv` → `requirements.txt`) is the one command; GitHub Actions CI
  (3 OS × Python 3.9/3.13) is the proof; manual `pip install -r requirements.txt` is the README fallback.
  If a library can't work everywhere and can't be fixed quickly → standard library only.

## Agreed design (summary)
- Forecast **per oblast per day, 1–7 days ahead**: number of alerts + top-3 most likely hours (24h profile).
  User picks `--start` and `--days`; `--fresh` downloads the latest data; default run = frozen snapshot.
- **All sources are equally trustworthy and combined** (no "official" vs "unofficial"): the two Vadimkin repo
  CSVs + Telegram `air_alert_ua` → alert intervals merged per oblast into episodes. Telegram `tryvoga_chomu`
  → threat-text features. Luhansk and Crimea excluded (permanent alerts).
- Naive alerts (no end message): end = start + per-oblast median duration (user's revised H6).
- Count model: `HistGradientBoostingRegressor(loss="poisson")`, retrained for each forecast time, no tuning.
  Baselines: last-7-days mean; same weekday last week. Metrics: MAE, skill vs best baseline, bias,
  peak-hour hit rate. Test = last 28 full days of the snapshot.
- Prototype philosophy (user): build a first working version, test, then correct. Don't over-engineer
  (DST edge cases, hour-precise data availability). Give heads-ups; don't block on details.

## Key data facts (measured 2026-10-11)
- Repo github.com/Vadimkin/ukrainian-air-raid-sirens-dataset, times in UTC, CRLF line endings.
  `official_data_en.csv` ends **2026-09-07** (raion/hromada level since Aug–Nov 2025; 2022–2025 rows
  duplicated; outage 2026-08-29…09-02). `volunteer_data_en.csv` (eTryvoga, oblast-level) is updated daily.
  Lesson: a repo's last-commit date says nothing about whether a specific file is fresh.
- Telegram `air_alert_ua` stopped posting 2026-09-07 06:09 UTC (confirmed in the app); its history very
  likely duplicates `official_data_en.csv`. `tryvoga_chomu`: ~80 posts/day, posts are edited after
  publication, full history reachable via `t.me/s/<channel>?before=<id>`.
- The two CSVs agree on ~83% of oblast-hours; in frontline oblasts the raion-level file is ~90–100% "under
  alert". Lviv/Volyn differ strongly since Jul 2026 (cause unknown). Naive alerts ≈ 4–9% of eTryvoga alerts.
- H6 first look: past-mean imputation was worse than +30 min; median ≈ +30 min (small sample) → re-test in code.

## About the user
- First-year Economics & Big Data student at KSE; little Python, no time-series background.
- Must understand and be able to defend every key decision in an interview.

## Working agreements
- No code before the plan for that step is explained and confirmed. Before any big step: explain, wait.
- **Explain every step before/while taking it** (what and why), including file reads and commands.
- Ask the user questions throughout, one at a time; don't decide silently on their behalf.
- For any data source: state confidence that it exists and is up to date, so the user can verify.
- Say "I'm not sure" instead of guessing. Flag possible hallucinations and own mistakes openly.
- Explain decisions in plain language; keep code simple and readable.
- **Proactively suggest the right skill / plugin / agent for each task** (and why): writing-plans after the
  spec, test-driven-development for leakage tests, systematic-debugging on bugs, verification-before-completion
  before saying "done", `/code-review` + `leakage-auditor` after each feature, `repro-tester` per milestone.
- Core decisions stay in this main conversation (it is the submitted AI log); agents do checks and reviews.

## Project agents (`.claude/agents/`)
Loaded only when a session starts; in the session that created them, run their instructions via a
general-purpose agent.
- `leakage-auditor` — independent read-only audit of code/diffs for time leakage.
- `repro-tester` — clones the repo into a temp dir and runs it like a reviewer would.
- `data-checker` — checks data source freshness, schema and cross-source agreement.
- `interview-coach` — quizzes the user on the project's decisions to prepare for Stage 3.
