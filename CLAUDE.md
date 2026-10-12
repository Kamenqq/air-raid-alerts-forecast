# Air Raid Alerts Forecast — KSE Agentic AI School, Stage 2

## Task
Python mini-project: time series forecasting of air raid alerts in Ukraine.
Deadline: **Mon 12 Oct 2026, 23:59 Kyiv time**.
Submission: (1) GitHub repo with working code, (2) full AI conversation log,
(3) ~100-word plain-text reflection (what went wrong / how we adjusted / why final > first).

**Full design: `docs/superpowers/specs/2026-10-11-air-raid-alerts-forecast-design.md`** (source of truth for details).

## Current status & how to resume (updated 2026-10-12, Kyiv morning)
- **Done and verified:** `main` on GitHub (github.com/Kamenqq/air-raid-alerts-forecast) at commit `cb16697` (+ this
  CLAUDE.md update); 64 tests pass; CI green on Ubuntu/macOS/Windows × Python 3.9/3.13. App = daily alert count per
  oblast (1–7 days) + hourly alert chances, quietest/riskiest 2-hour slots, `--slot HH:MM-HH:MM`; H4 neighbours;
  security fixes (redacted Telegram posts, version limits, read-only CI token); code-review fixes.
- **Final results (test 2026-09-13…10-10, measured once):** count MAE 1.97 = last-7-days average 1.97 (tie),
  bias −0.09; hourly chances Brier 0.1871 vs flat 0.1870 (1 day), 0.1904 vs 0.1893 (7 days), well calibrated
  (35%→35%, 74%→76%). Validation (2026-08-16…09-12) was used for choices: H4 2.028→2.010; "when" variant A.
- **Left for the user (deadline Mon 12 Oct 2026, 23:59 Kyiv):** submit the form with (1) repo link, (2) the AI log —
  exported by the app's session export as a zip in ~/Downloads (made 2026-10-12), (3) ~100-word reflection in the
  user's own words.
- **Reflection material:** combining all sources changed the target's meaning on 2026-09-07 (−16% skill; Claude's
  predicted bias direction was wrong) → experiment → volunteer-only target; H6 mean→median; main window not useful
  → hourly chances for residents; both approved "when" options lost to a flat baseline on validation → shrinkage
  variant A; reviewer agents found an untested `--start` cutoff and card numbers in stored posts → fixed; risk checks
  caught Ukrainian word forms, partial last day, Windows console crash, half-made `.venv`, a test deleting `.venv`,
  scikit-learn 1.9 crash on an all-empty column (found by CI on Python 3.13).
- **Open questions (ask the user, don't act alone):** delete the stale remote branch `prototype`? run the
  `interview-coach` agent (new session) for Stage 3 prep? Option 3 for "when" (separate hourly model) was skipped.
- **Deferred minors (documented, not fixed):** `--fresh` with a too-early `--start` downloads before rejecting;
  Telegram first/last collected day counted as complete; numpy "deprecated" warnings on Python 3.13; card numbers
  remain in old git history (user chose no rewrite); old commits use the user's Gmail (user: leave it).
- **Local-only files (not in git):** `docs/PROJECT_DESCRIPTION.md` (full project description for the user's review;
  listed in `.git/info/exclude`), root `official_data_en.csv` / `volunteer_data_en.csv` (ignored).
- **To resume:** read this file, `README.md`, `docs/PROJECT_DESCRIPTION.md`; run `python3 run.py test`
  (expect 64 OK) and `python3 run.py`. Keep the working agreements below (one task at a time, risk report +
  self-check commands, wait for "go").

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
- Forecast **per oblast per day, 1–7 days ahead**: number of alerts + **chance of an alert in each Kyiv hour**,
  quietest / riskiest 2-hour slots, and any time slot via `--slot` (main window and top-3 hours retired).
  User picks `--start` and `--days`; `--fresh` downloads the latest data; default run = frozen snapshot.
- **Target (user decision 2026-10-12, after an experiment):** eTryvoga (volunteer CSV) alerts only, merged per
  oblast into episodes. Combining all sources made the target change on 2026-09-07 (the other sources stopped),
  which caused a -16% skill and -0.57 bias; with the volunteer target: skill ≈ 0%, bias -0.09.
  The official CSV and `air_alert_ua` are still imported and shown as data checks in the report.
  Telegram `tryvoga_chomu` → threat-text features. Luhansk and Crimea excluded (permanent alerts).
- **"When" outputs:** top-3 peak hours, and the **main window** = shortest continuous span of hours (may cross
  midnight) expected to hold ≥ 50% of the day's alerts. Checked by coverage (target 50%) and mean length.
- Naive alerts (no end message): end = start + per-oblast median duration (user's revised H6).
- **Decisions 2026-10-12 (round 2):** (1) H4 approved — add neighbouring-oblast features (border table +
  neighbours' yesterday / 7-day averages). (2) "When" for a resident planning the day (mall trip, commute):
  replace the main window with the **chance that an alert is active in each hour / any time slot**, from recent
  history (option 1), and test **scaling it by the predicted daily count** (option 2). On validation both lost to
  a flat chance; user chose **variant A**: half hourly pattern + half flat level over 56 days, × predicted count
  for 1 day ahead only (still shown for all 7 days). Test: ties flat, well calibrated. Option 3 (separate hourly
  model) reminded and **skipped by the user** (2026-10-12). (3) Decide what to keep on a
  **validation period = the 28 days before the test period**, then report the test period once.
- Count model: `HistGradientBoostingRegressor(loss="poisson")`, retrained for each forecast time, no tuning.
  Baselines: last-7-days mean; same weekday last week. Metrics: MAE, skill vs best baseline, bias;
  hourly chances: Brier score vs 'same chance every hour', calibration table. Test = last 28 full days of the snapshot.
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
- **Build mode (agreed): native execution of the plan, one task at a time.** After each task: test its
  risks, then report them to the user with concrete commands the user can run to check the work themselves.
  Wait for the user before starting the next task.

## Project agents (`.claude/agents/`)
Loaded only when a session starts; in the session that created them, run their instructions via a
general-purpose agent.
- `leakage-auditor` — independent read-only audit of code/diffs for time leakage.
- `repro-tester` — clones the repo into a temp dir and runs it like a reviewer would.
- `data-checker` — checks data source freshness, schema and cross-source agreement.
- `interview-coach` — quizzes the user on the project's decisions to prepare for Stage 3.
