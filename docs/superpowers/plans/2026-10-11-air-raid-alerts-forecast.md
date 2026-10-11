# Air Raid Alerts Forecast — Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A one-command Python app that forecasts, per Ukrainian oblast and per day (1–7 days ahead), how many air raid alerts will start and in which hours they are most likely — evaluated honestly against simple baselines on the latest 28 days.

**Architecture:** Alert intervals from all sources (two repo CSVs + the air_alert_ua Telegram channel) are merged per oblast into episodes and counted per Kyiv day. Feature rows describe what is known at 00:00 Kyiv of the forecast day (alert history + tryvoga_chomu keyword counts). A Poisson gradient-boosting model, retrained for every forecast date on earlier days only, predicts the daily count; a 28-day hourly profile turns it into peak hours. An HTML report shows exploration, the H6 check, and the backtest.

**Tech Stack:** Python 3.9+, pandas, numpy, scikit-learn, matplotlib; standard library `urllib`, `unittest`, `venv`; GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-11-air-raid-alerts-forecast-design.md`

## Global Constraints

- Python **3.9+** compatible syntax (no `match`, no `X | Y` type hints, no `list[int]` at runtime in annotations evaluated at import).
- Must run on macOS, Linux, Windows; reviewers run **`python3 run.py`** (Windows: `python run.py`) with no manual steps, logins or API keys.
- `run.py` uses the standard library only; libraries go in `requirements.txt` (pandas, numpy, scikit-learn, matplotlib).
- All timestamps inside the app are **naive UTC**; Kyiv time (`Europe/Kyiv`) only for day/hour labels.
- Oblasts: the 24 in `config.OBLASTS`; Luhansk and Crimea excluded.
- No leakage: a forecast for day `t0` (made at 00:00 Kyiv of `t0`) uses only alerts that started before it, posts published before it, and a model trained on target days `< t0`.
- No hyperparameter tuning; model is `HistGradientBoostingRegressor(loss="poisson", random_state=0)`.
- Snapshot data from `2025-09-01`; naive-alert medians from `2025-09-01`–`2025-11-30`; training targets from `2025-12-01`; earliest `--start` `2026-01-01`; test = last 28 full days; leads 1–7.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Explain each step to the user (beginner) in plain language while executing.

## Review Focus

1. **Bad `--start` input** (wrong format, before 2026-01-01, after the data ends + 1 day) → one clear sentence, no Python traceback. Pinned in Task 9 (`test_main.py`).
2. **Telegram pages with media-only posts or an empty last page** → pagination keeps going / stops cleanly, no crash. Pinned in Task 3 (`test_parse_media_only_post`) and Task 4 (`test_fetch_stops_on_empty_page`).
3. **air_alert_ua posts we can't use** (region not in the map, ambiguous name, end without start, threat-only posts) → ignored, never crash. Pinned in Task 3 (`test_air_alert_intervals_skip_unusable`).
4. **Oblast with no alerts in the last 28 days** (common in the west) → uniform hourly profile, no division by zero, peak hours still printed. Pinned in Task 6 (`test_profile_without_history_is_uniform`).
5. **`.venv` cannot be created** (Debian/Ubuntu without `python3-venv`) → message telling the user exactly what to install or how to use the pip fallback. Pinned in Task 1 (`test_venv_failure_gives_instructions`).

---

## File map

| File | Responsibility |
|---|---|
| `run.py` | stdlib launcher: create `.venv`, install requirements, route `test` / `snapshot` / app args |
| `requirements.txt` | library list with version floors |
| `alerts/config.py` | paths, dates, oblast list, Kyiv time helpers |
| `alerts/parse.py` | read repo CSVs; parse Telegram pages; air_alert_ua posts → intervals; tryvoga_chomu keyword flags; region map |
| `alerts/sources.py` | HTTP downloads, Telegram pagination, `--fresh` refresh |
| `alerts/snapshot.py` | maintainer tool that builds `data/snapshot/*` and `data/region_map.csv` |
| `alerts/episodes.py` | naive-end imputation, as-of clipping, episode merging, daily counts, episode start table |
| `alerts/features.py` | Telegram daily table, feature rows, hourly profile |
| `alerts/model.py` | training rows with cutoff, train, predict |
| `alerts/evaluate.py` | test period, backtest, MAE/skill/bias tables, peak-hour hit rate, H6 check |
| `alerts/report.py` | charts and HTML report writer |
| `alerts/main.py` | CLI: load → build → evaluate/report or forecast |
| `tests/…` | unittest suites per module |
| `.github/workflows/ci.yml` | 3 OS × Python 3.9/3.13 |
| `README.md` | how to run, results, limitations |

---

### Task 1: Project skeleton, launcher and config

**Why:** reviewers must get from `git clone` to a running app with one command. This task makes `python3 run.py test` create the environment and run tests.

**Files:**
- Create: `requirements.txt`, `run.py`, `alerts/__init__.py`, `alerts/config.py`, `tests/__init__.py`, `tests/test_launcher.py`, `tests/test_config.py`
- Modify: `.gitignore` (append)

**Interfaces:**
- Produces: `run.command(args) -> list[str]`, `run.ensure_env() -> Path`; `alerts.config` constants (`ROOT, SNAPSHOT_DIR, FRESH_DIR, REGION_MAP, REPORTS_DIR, REPO_RAW, KYIV, OBLASTS, DATA_START, NAIVE_MEDIAN_END, TRAIN_START, EARLIEST_START, CHOMU_SINCE, AIR_ALERT_SINCE, AIR_ALERT_UNTIL, TEST_DAYS, MAX_LEAD`) and helpers `to_kyiv(series) -> Series`, `kyiv_day(ts) -> Timestamp`, `kyiv_day_start_utc(day) -> Timestamp`.

- [ ] **Step 1: Write `requirements.txt`**

```
pandas>=2.0,<3
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
```

- [ ] **Step 2: Write `run.py`**

```python
"""One-command launcher (standard library only).

Creates a private environment in .venv, installs requirements.txt into it once,
then runs the app inside it:
    python3 run.py [--start YYYY-MM-DD] [--days N] [--fresh]
    python3 run.py test        # run the unit tests
    python3 run.py snapshot    # maintainer only: rebuild data/snapshot (slow, uses the network)
"""
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / "installed-requirements.txt"


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_env():
    python = venv_python()
    if not python.exists():
        print("Creating a private environment in .venv (first run only)...")
        try:
            venv.EnvBuilder(with_pip=True).create(VENV)
        except Exception as error:  # e.g. Debian/Ubuntu without the python3-venv package
            sys.exit(
                f"Could not create .venv ({error}).\n"
                "Fix: install your system's venv package (Debian/Ubuntu: sudo apt install python3-venv),\n"
                "or install the libraries yourself: pip install -r requirements.txt, then run: python -m alerts.main"
            )
    wanted = REQUIREMENTS.read_text()
    if not STAMP.exists() or STAMP.read_text() != wanted:
        print("Installing libraries from requirements.txt (first run only, needs internet)...")
        pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-q"]
        subprocess.check_call(pip + ["--upgrade", "pip"])
        subprocess.check_call(pip + ["-r", str(REQUIREMENTS)])
        STAMP.write_text(wanted)
    return python


def command(args):
    if args[:1] == ["test"]:
        return ["-m", "unittest", "discover", "-s", "tests", "-t", "."]
    if args[:1] == ["snapshot"]:
        return ["-m", "alerts.snapshot"]
    return ["-m", "alerts.main", *args]


def main():
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")
    python = ensure_env()
    sys.exit(subprocess.call([str(python), *command(sys.argv[1:])], cwd=ROOT))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `alerts/__init__.py` (empty) and `tests/__init__.py` (empty)**

- [ ] **Step 4: Write `alerts/config.py`**

```python
"""Shared constants and time helpers. All timestamps inside the app are naive UTC."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "data" / "snapshot"
FRESH_DIR = ROOT / "data" / "fresh"
REGION_MAP = ROOT / "data" / "region_map.csv"
REPORTS_DIR = ROOT / "reports"
REPO_RAW = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/"

KYIV = "Europe/Kyiv"
OBLASTS = [
    "Cherkaska oblast", "Chernihivska oblast", "Chernivetska oblast", "Dnipropetrovska oblast",
    "Donetska oblast", "Ivano-Frankivska oblast", "Kharkivska oblast", "Khersonska oblast",
    "Khmelnytska oblast", "Kirovohradska oblast", "Kyiv City", "Kyivska oblast", "Lvivska oblast",
    "Mykolaivska oblast", "Odeska oblast", "Poltavska oblast", "Rivnenska oblast", "Sumska oblast",
    "Ternopilska oblast", "Vinnytska oblast", "Volynska oblast", "Zakarpatska oblast",
    "Zaporizka oblast", "Zhytomyrska oblast",
]

DATA_START = pd.Timestamp("2025-09-01")        # snapshot keeps alerts from here (UTC)
NAIVE_MEDIAN_END = pd.Timestamp("2025-12-01")  # naive-alert medians use [DATA_START, this)
TRAIN_START = pd.Timestamp("2025-12-01")       # first target day used for training
EARLIEST_START = pd.Timestamp("2026-01-01")    # earliest allowed --start
CHOMU_SINCE = pd.Timestamp("2025-11-24")       # tryvoga_chomu posts kept from here
AIR_ALERT_SINCE = pd.Timestamp("2026-08-25")   # air_alert_ua posts kept from here
AIR_ALERT_UNTIL = pd.Timestamp("2026-09-08")   # channel stopped posting on 2026-09-07
TEST_DAYS = 28
MAX_LEAD = 7


def to_kyiv(utc):
    """Naive-UTC Series -> naive Kyiv local time Series."""
    return utc.dt.tz_localize("UTC").dt.tz_convert(KYIV).dt.tz_localize(None)


def kyiv_day(ts):
    """Naive-UTC Timestamp -> the Kyiv calendar day it falls on (midnight Timestamp)."""
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert(KYIV).tz_localize(None).normalize()


def kyiv_day_start_utc(day):
    """Kyiv calendar day -> naive-UTC Timestamp of its 00:00 Kyiv."""
    return pd.Timestamp(day).normalize().tz_localize(KYIV).tz_convert("UTC").tz_localize(None)
```

- [ ] **Step 5: Write the tests `tests/test_launcher.py` and `tests/test_config.py`**

```python
# tests/test_launcher.py
import unittest
from unittest import mock

import run


class LauncherTest(unittest.TestCase):
    def test_command_routes_subcommands(self):
        self.assertEqual(run.command(["test"])[:2], ["-m", "unittest"])
        self.assertEqual(run.command(["snapshot"]), ["-m", "alerts.snapshot"])
        self.assertEqual(run.command(["--days", "3"]), ["-m", "alerts.main", "--days", "3"])

    def test_venv_failure_gives_instructions(self):
        missing = mock.Mock()
        missing.exists.return_value = False
        with mock.patch.object(run, "venv_python", return_value=missing), \
                mock.patch("venv.EnvBuilder.create", side_effect=OSError("no ensurepip")):
            with self.assertRaises(SystemExit) as ctx:
                run.ensure_env()
        self.assertIn("pip install -r requirements.txt", str(ctx.exception.code))
```

```python
# tests/test_config.py
import unittest

import pandas as pd

from alerts.config import kyiv_day, kyiv_day_start_utc, to_kyiv


class KyivTimeTest(unittest.TestCase):
    def test_summer_and_winter_offsets(self):
        utc = pd.Series(pd.to_datetime(["2026-07-01 21:30", "2026-01-01 22:30"]))
        self.assertEqual(list(to_kyiv(utc)), list(pd.to_datetime(["2026-07-02 00:30", "2026-01-02 00:30"])))

    def test_day_helpers(self):
        self.assertEqual(kyiv_day(pd.Timestamp("2026-07-01 21:30")), pd.Timestamp("2026-07-02"))
        self.assertEqual(kyiv_day_start_utc("2026-07-02"), pd.Timestamp("2026-07-01 21:00"))
        self.assertEqual(kyiv_day_start_utc("2026-01-02"), pd.Timestamp("2026-01-01 22:00"))
```

- [ ] **Step 6: Append to `.gitignore`**

```
# project
data/fresh/
reports/
/*.csv
```

- [ ] **Step 7: Run `python3 run.py test`**

Expected: first run prints "Creating a private environment…" and "Installing libraries…", then `Ran 4 tests … OK`.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt run.py alerts/__init__.py alerts/config.py tests/__init__.py tests/test_launcher.py tests/test_config.py .gitignore
git commit -m "Add launcher, config and Kyiv time helpers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Read the repo CSVs

**Why:** the two repo files are the main alert history. They have CRLF line endings, duplicated rows and regions we exclude.

**Files:**
- Create: `alerts/parse.py`, `tests/test_parse_csv.py`

**Interfaces:**
- Consumes: `config.OBLASTS`
- Produces: `load_official(path) -> DataFrame[oblast, start, end, source]`, `load_volunteer(path) -> DataFrame[oblast, start, end, naive]` (times naive UTC, `naive` bool)

- [ ] **Step 1: Write the failing test `tests/test_parse_csv.py`**

```python
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alerts.parse import load_official, load_volunteer


def write(tmp, name, lines):
    path = Path(tmp) / name
    path.write_bytes("\r\n".join(lines).encode("utf-8") + b"\r\n")  # CRLF like the real files
    return path


class CsvTest(unittest.TestCase):
    def test_official_dedupes_and_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = "Kharkivska oblast,Kharkivskyi raion,,raion,2026-01-01 10:00:00+00:00,2026-01-01 11:00:00+00:00,official"
            path = write(tmp, "o.csv", [
                "oblast,raion,hromada,level,started_at,finished_at,source", row, row,
                "Luhanska oblast,,,oblast,2026-01-01 10:00:00+00:00,2026-01-01 11:00:00+00:00,official",
            ])
            df = load_official(path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "oblast"], "Kharkivska oblast")
        self.assertEqual(df.loc[0, "start"], pd.Timestamp("2026-01-01 10:00"))
        self.assertEqual(df.loc[0, "source"], "official_csv")

    def test_volunteer_reads_naive_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "v.csv", [
                "region,started_at,finished_at,naive",
                "Kyiv City,2026-01-01 10:00:00+00:00,2026-01-01 10:30:00+00:00,True",
                "Lvivska oblast,2026-01-02 10:00:00+00:00,2026-01-02 10:45:00+00:00,False",
            ])
            df = load_volunteer(path)
        self.assertEqual(list(df["naive"]), [True, False])
        self.assertEqual(list(df["oblast"]), ["Kyiv City", "Lvivska oblast"])
        self.assertEqual(df.loc[1, "end"], pd.Timestamp("2026-01-02 10:45"))
```

- [ ] **Step 2: Run it — expect failure**

Run: `python3 run.py test` → Expected: `ModuleNotFoundError: No module named 'alerts.parse'`.

- [ ] **Step 3: Write `alerts/parse.py` (CSV part)**

```python
"""Read and clean every data source into simple pandas tables."""
import html
import re

import pandas as pd

from .config import OBLASTS


def _utc(column):
    return pd.to_datetime(column, utc=True).dt.tz_localize(None).to_numpy()


def load_official(path):
    """official_data_en.csv -> one row per alert interval (any level: oblast, raion, hromada)."""
    df = pd.read_csv(path).drop_duplicates()
    df = df[df["oblast"].isin(OBLASTS)]
    return pd.DataFrame({
        "oblast": df["oblast"].to_numpy(),
        "start": _utc(df["started_at"]),
        "end": _utc(df["finished_at"]),
        "source": "official_csv",
    })


def load_volunteer(path):
    """volunteer_data_en.csv (eTryvoga) -> one row per oblast alert, with the naive flag."""
    df = pd.read_csv(path).drop_duplicates()
    df = df[df["region"].isin(OBLASTS)]
    return pd.DataFrame({
        "oblast": df["region"].to_numpy(),
        "start": _utc(df["started_at"]),
        "end": _utc(df["finished_at"]),
        "naive": df["naive"].astype(str).str.strip().eq("True").to_numpy(),
    })
```

- [ ] **Step 4: Run tests — expect PASS** (`python3 run.py test` → `OK`)

- [ ] **Step 5: Commit**

```bash
git add alerts/parse.py tests/test_parse_csv.py
git commit -m "Read repo alert CSVs (dedupe, filter oblasts, naive flag)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Parse Telegram posts

**Why:** air_alert_ua gives alert intervals in a fixed format (🔴 start / 🟢 end per raion); tryvoga_chomu gives threat text. Both arrive as HTML pages from `t.me/s/<channel>`.

**Files:**
- Modify: `alerts/parse.py` (append)
- Create: `tests/test_parse_telegram.py`

**Interfaces:**
- Produces:
  - `parse_tg_page(page_html) -> list[dict(id:int, time:Timestamp naive UTC, text:str)]` (media-only posts have `text == ""`)
  - `read_posts(path) -> DataFrame[id, time, text]`
  - `norm_name(name) -> str`
  - `OBLAST_UK: dict[str, str|None]`, `build_region_map(uk_df) -> dict[str, str]`, `load_region_map(path) -> dict[str, str]`
  - `air_alert_events(posts) -> DataFrame[time, kind, region]`, `air_alert_intervals(events, region_map) -> DataFrame[oblast, start, end, source]`
  - `OBLAST_PATTERNS`, `THREAT_PATTERNS`, `TG_COLUMNS`, `clean_chomu(text) -> str`, `chomu_flags(posts) -> DataFrame[time, *TG_COLUMNS]`

- [ ] **Step 1: Write the failing test `tests/test_parse_telegram.py`**

```python
import unittest

import pandas as pd

from alerts.parse import (TG_COLUMNS, air_alert_events, air_alert_intervals, build_region_map,
                          chomu_flags, clean_chomu, parse_tg_page)


def post(pid, iso, inner=None):
    text = f'<div class="tgme_widget_message_text js-message_text" dir="auto">{inner}</div>' if inner else ""
    return (f'<div class="tgme_widget_message_wrap js-widget_message_wrap">'
            f'<div class="tgme_widget_message" data-post="air_alert_ua/{pid}">{text}'
            f'<a class="tgme_widget_message_date"><time datetime="{iso}" class="time">x</time></a></div></div>')


START = '<i class="emoji"><b>🔴</b></i> <b>08:44 Повітряна тривога в Харківський район</b><br/>Слідкуйте за подальшими повідомленнями.<br/><a href="?q=x">#Харківський_район</a>'
END = '<b><i class="emoji"><b>🟢</b></i> 09:30 Відбій тривоги в Харківський район.</b><br/>Слідкуйте за подальшими повідомленнями.'
BULLETS = '<i class="emoji"><b>🔴</b></i> <b>08:54 Повітряна тривога в <br/>• Кам’янський район<br/>• Невідомий район</b><br/>Слідкуйте за подальшими повідомленнями.'
THREAT = '🔴🔴 08:38 Загроза ударних БпЛА в м. Запоріжжя та Запорізька територіальна громада&#33;'

REGION_MAP = {"Харківський район": "Kharkivska oblast", "Кам'янський район": "Dnipropetrovska oblast"}


class TelegramPageTest(unittest.TestCase):
    def test_parse_page(self):
        posts = parse_tg_page(post(10, "2026-09-07T05:44:06+00:00", START))
        self.assertEqual(posts[0]["id"], 10)
        self.assertEqual(posts[0]["time"], pd.Timestamp("2026-09-07 05:44:06"))
        self.assertTrue(posts[0]["text"].startswith("🔴 08:44 Повітряна тривога в Харківський район\n"))

    def test_parse_media_only_post(self):
        posts = parse_tg_page(post(11, "2026-09-07T05:44:06+00:00"))
        self.assertEqual(posts, [{"id": 11, "time": pd.Timestamp("2026-09-07 05:44:06"), "text": ""}])


class AirAlertTest(unittest.TestCase):
    def posts(self, items):
        rows = [p for pid, iso, inner in items for p in parse_tg_page(post(pid, iso, inner))]
        return pd.DataFrame(rows)

    def test_events_single_bullets_and_threats(self):
        events = air_alert_events(self.posts([
            (1, "2026-09-07T05:44:00+00:00", START),
            (2, "2026-09-07T05:50:00+00:00", BULLETS),
            (3, "2026-09-07T05:51:00+00:00", THREAT),
            (4, "2026-09-07T06:30:00+00:00", END),
        ]))
        self.assertEqual(list(events["kind"]), ["start", "start", "start", "end"])
        self.assertEqual(list(events["region"]),
                         ["Харківський район", "Кам'янський район", "Невідомий район", "Харківський район"])

    def test_air_alert_intervals_skip_unusable(self):
        events = air_alert_events(self.posts([
            (1, "2026-09-07T05:00:00+00:00", END),      # end without start -> ignored
            (2, "2026-09-07T05:44:00+00:00", START),
            (3, "2026-09-07T05:50:00+00:00", BULLETS),  # Кам'янський never ends, Невідомий not in map
            (4, "2026-09-07T06:30:00+00:00", END),
        ]))
        intervals = air_alert_intervals(events, REGION_MAP)
        self.assertEqual(len(intervals), 1)
        row = intervals.iloc[0]
        self.assertEqual((row["oblast"], row["start"], row["end"], row["source"]),
                         ("Kharkivska oblast", pd.Timestamp("2026-09-07 05:44"),
                          pd.Timestamp("2026-09-07 06:30"), "air_alert_ua"))


class RegionMapTest(unittest.TestCase):
    def test_build_region_map(self):
        uk = pd.DataFrame({
            "oblast": ["Харківська область", "Дніпропетровська область", "Луганська область",
                       "Чернівецька область", "Одеська область"],
            "raion": ["Харківський район", "Кам’янський район", "Щастинський район", None, None],
            "hromada": [None, None, None, "Тестова територіальна громада", "Тестова територіальна громада"],
        })
        mapping = build_region_map(uk)
        self.assertEqual(mapping["Харківський район"], "Kharkivska oblast")
        self.assertEqual(mapping["Кам'янський район"], "Dnipropetrovska oblast")  # apostrophe normalised
        self.assertEqual(mapping["Харківська область"], "Kharkivska oblast")
        self.assertNotIn("Щастинський район", mapping)                     # Luhansk excluded
        self.assertNotIn("Тестова територіальна громада", mapping)          # ambiguous -> dropped

    def test_unknown_oblast_name_fails_loudly(self):
        with self.assertRaises(ValueError):
            build_region_map(pd.DataFrame({"oblast": ["Невідома область"], "raion": [None], "hromada": [None]}))


class ChomuTest(unittest.TestCase):
    def test_clean_and_flags(self):
        text = "❗️Реактивний Шахед на Київщину\n\nПідтримати канал☕️\n\n✅️ Чому тривога?"
        self.assertEqual(clean_chomu(text), "❗️реактивний шахед на київщину")
        posts = pd.DataFrame({"time": pd.to_datetime(["2026-09-01 10:00", "2026-09-01 11:00"]),
                              "text": [text, "Балістика на Київ!"]})
        flags = chomu_flags(posts)
        self.assertEqual(list(flags.columns), ["time"] + TG_COLUMNS)
        self.assertEqual(list(flags["m_Kyivska oblast"]), [1, 0])
        self.assertEqual(list(flags["m_Kyiv City"]), [0, 1])
        self.assertEqual(list(flags["drones"]), [1, 0])
        self.assertEqual(list(flags["missiles"]), [0, 1])
```

- [ ] **Step 2: Run it — expect failure** (`ImportError: cannot import name 'TG_COLUMNS'`)

- [ ] **Step 3: Append the Telegram code to `alerts/parse.py`**

```python
# ---------------------------------------------------------------- Telegram pages
_POST_ID = re.compile(r'data-post="[^/"]+/(\d+)"')
_TIME = re.compile(r'<time datetime="([^"]+)"')
_TEXT = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)


def html_to_text(fragment):
    text = re.sub(r"<br\s*/?>", "\n", fragment)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def parse_tg_page(page_html):
    """One t.me/s/<channel> page -> posts (oldest first). Media-only posts get text ''."""
    posts = []
    for block in page_html.split("tgme_widget_message_wrap")[1:]:
        post_id, time = _POST_ID.search(block), _TIME.search(block)
        if not (post_id and time):
            continue
        text = _TEXT.search(block)
        posts.append({
            "id": int(post_id.group(1)),
            "time": pd.Timestamp(time.group(1)).tz_convert("UTC").tz_localize(None),
            "text": html_to_text(text.group(1)) if text else "",
        })
    return posts


def read_posts(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df["text"] = df["text"].fillna("")
    return df


# ---------------------------------------------------------------- air_alert_ua
OBLAST_UK = {
    "Вінницька область": "Vinnytska oblast", "Волинська область": "Volynska oblast",
    "Дніпропетровська область": "Dnipropetrovska oblast", "Донецька область": "Donetska oblast",
    "Житомирська область": "Zhytomyrska oblast", "Закарпатська область": "Zakarpatska oblast",
    "Запорізька область": "Zaporizka oblast", "Івано-Франківська область": "Ivano-Frankivska oblast",
    "Київська область": "Kyivska oblast", "м. Київ": "Kyiv City", "Київ": "Kyiv City",
    "Кіровоградська область": "Kirovohradska oblast", "Луганська область": None,
    "Львівська область": "Lvivska oblast", "Миколаївська область": "Mykolaivska oblast",
    "Одеська область": "Odeska oblast", "Полтавська область": "Poltavska oblast",
    "Рівненська область": "Rivnenska oblast", "Сумська область": "Sumska oblast",
    "Тернопільська область": "Ternopilska oblast", "Харківська область": "Kharkivska oblast",
    "Херсонська область": "Khersonska oblast", "Хмельницька область": "Khmelnytska oblast",
    "Черкаська область": "Cherkaska oblast", "Чернівецька область": "Chernivetska oblast",
    "Чернігівська область": "Chernihivska oblast", "Автономна Республіка Крим": None,
}


def norm_name(name):
    for apostrophe in "’ʼ`":
        name = name.replace(apostrophe, "'")
    return name.strip().rstrip(".!").strip()


def build_region_map(uk):
    """official_data_uk.csv rows -> {Ukrainian oblast/raion/hromada name: English oblast}.
    Names that belong to more than one oblast are dropped (ambiguous)."""
    unknown = sorted(set(uk["oblast"].dropna()) - set(OBLAST_UK))
    if unknown:
        raise ValueError(f"Unknown oblast names in official_data_uk.csv, add them to OBLAST_UK: {unknown}")
    owners = {}
    for column in ("oblast", "raion", "hromada"):
        for name, oblast_uk in zip(uk[column], uk["oblast"]):
            oblast = OBLAST_UK[oblast_uk]
            if isinstance(name, str) and oblast:
                owners.setdefault(norm_name(name), set()).add(oblast)
    return {name: next(iter(oblasts)) for name, oblasts in owners.items() if len(oblasts) == 1}


def load_region_map(path):
    df = pd.read_csv(path)
    return dict(zip(df["name"], df["oblast"]))


_ALERT_START, _ALERT_END = "Повітряна тривога в", "Відбій тривоги в"


def air_alert_events(posts):
    """Posts -> one row per (time, 'start'/'end', region). Threat-only posts are ignored."""
    rows = []
    for time, text in zip(posts["time"], posts["text"]):
        lines = text.split("\n")
        first = lines[0]
        if first.startswith("🔴") and _ALERT_START in first:
            kind, marker = "start", _ALERT_START
        elif first.startswith("🟢") and _ALERT_END in first:
            kind, marker = "end", _ALERT_END
        else:
            continue
        rest = first.split(marker, 1)[1].strip()
        regions = [rest] if rest else []
        if not regions:  # bullet list on the following lines
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("•"):
                    regions.append(line.lstrip("•").strip())
                elif regions:
                    break
        rows.extend((time, kind, norm_name(region)) for region in regions)
    return pd.DataFrame(rows, columns=["time", "kind", "region"])


def air_alert_intervals(events, region_map):
    """Pair each region's start with its next end. Unpaired or unmapped regions are skipped."""
    open_alerts, rows = {}, []
    for time, kind, region in events.sort_values("time", kind="stable").itertuples(index=False):
        if kind == "start":
            open_alerts.setdefault(region, time)
        elif region in open_alerts:
            rows.append((region, open_alerts.pop(region), time))
    df = pd.DataFrame(rows, columns=["region", "start", "end"])
    df["oblast"] = df["region"].map(region_map)
    df = df.dropna(subset=["oblast"]).reset_index(drop=True)
    df["source"] = "air_alert_ua"
    return df[["oblast", "start", "end", "source"]]


# ---------------------------------------------------------------- tryvoga_chomu
# Lower-case regexes. Prototype keyword matching: some noise is expected (e.g. "дніпр" = city and river).
OBLAST_PATTERNS = {
    "Cherkaska oblast": r"черкас",
    "Chernihivska oblast": r"чернігів",
    "Chernivetska oblast": r"чернівец|чернівеч|чернівц|буковин",
    "Dnipropetrovska oblast": r"дніпр|кривий ріг|кривого рогу|кривому розі|павлоград|нікопол|кам'янськ",
    "Donetska oblast": r"донеч|донецьк|краматорськ|слов'янськ|покровськ|костянтинівк|дружківк",
    "Ivano-Frankivska oblast": r"франків|прикарпат",
    "Kharkivska oblast": r"харків|куп'янськ|ізюм|чугуїв",
    "Khersonska oblast": r"херсон",
    "Khmelnytska oblast": r"хмельниц|хмельнич|кам'янець|шепетівк|старокостянтин",
    "Kirovohradska oblast": r"кіровоград|кропивниц",
    "Kyiv City": r"\bкиїв\b|\bкиєв|столиц",
    "Kyivska oblast": r"київщин|бровар|бориспіл|біла церкв|білої церкв|вишгород|обухів|фастів",
    "Lvivska oblast": r"львів",
    "Mykolaivska oblast": r"миколаїв",
    "Odeska oblast": r"одес|ізмаїл",
    "Poltavska oblast": r"полтав|кременчук|миргород",
    "Rivnenska oblast": r"рівненщ|рівненськ|\bрівне\b|\bрівного\b|\bрівному\b",
    "Sumska oblast": r"сумщин|\bсуми\b|\bсумах\b|сумськ|шостк|конотоп|охтирк",
    "Ternopilska oblast": r"тернопіл",
    "Vinnytska oblast": r"вінниц|вінничч|козятин",
    "Volynska oblast": r"волин|луцьк|ковел",
    "Zakarpatska oblast": r"закарпат|ужгород|мукачев",
    "Zaporizka oblast": r"запоріж|запоріз",
    "Zhytomyrska oblast": r"житомир|звягел|коростен|бердичів",
}
THREAT_PATTERNS = {
    "drones": r"шахед|бпла|дрон|герань|гербер",
    "missiles": r"ракет|балістик|калібр|кинджал|іскандер|циркон",
    "aviation": r"міг-31|ту-95|ту-22|ту-160|су-34|су-35|авіабомб|стратегічн",
    "massive": r"масован|масштабн|комбінован",
}
TG_COLUMNS = [f"m_{oblast}" for oblast in OBLASTS] + list(THREAT_PATTERNS)


def clean_chomu(text):
    """Drop the channel's boilerplate lines, normalise apostrophes, lower-case."""
    lines = [line for line in text.split("\n")
             if line.strip() and "Підтримати канал" not in line and "Чому тривога?" not in line]
    return norm_name("\n".join(lines)).lower()


def chomu_flags(posts):
    """One row per post: 1 if the post mentions each oblast / threat group, else 0."""
    if posts.empty:
        return pd.DataFrame(columns=["time"] + TG_COLUMNS)
    text = posts["text"].fillna("").map(clean_chomu)
    flags = {f"m_{o}": text.str.contains(OBLAST_PATTERNS[o], regex=True) for o in OBLASTS}
    flags.update({g: text.str.contains(p, regex=True) for g, p in THREAT_PATTERNS.items()})
    out = pd.DataFrame(flags).astype(int)
    out.insert(0, "time", posts["time"].to_numpy())
    return out
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add alerts/parse.py tests/test_parse_telegram.py
git commit -m "Parse Telegram pages: air_alert_ua intervals, tryvoga_chomu keyword flags, region map" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Downloads, Telegram pagination, snapshot builder — then build the snapshot (background)

**Why:** reviewers use a frozen snapshot in the repo. Building it takes ~1 hour (Telegram), so it runs in the background while Tasks 5–8 are coded.

**Files:**
- Create: `alerts/sources.py`, `alerts/snapshot.py`, `tests/test_sources.py`
- Output (committed): `data/snapshot/official_data_en.csv`, `data/snapshot/volunteer_data_en.csv`, `data/snapshot/air_alert_ua_posts.csv`, `data/snapshot/tryvoga_chomu_posts.csv`, `data/region_map.csv`

**Interfaces:**
- Consumes: `parse.parse_tg_page`, `parse.read_posts`, `parse.build_region_map`, config constants
- Produces: `get_bytes(url, retries=3, timeout=30) -> bytes`, `get_text(url) -> str`, `download(url, dest) -> Path`, `fetch_channel(channel, since, after_id=0, delay=1.5, get=get_text, log=print) -> DataFrame[id, time, text]`, `refresh() -> Path` (folder with the same four file names as the snapshot); `snapshot.build_snapshot()`

- [ ] **Step 1: Write the failing test `tests/test_sources.py`**

```python
import unittest

import pandas as pd

from alerts.sources import fetch_channel


def page(ids, newest_time):
    """Fake t.me/s page: posts with the given ids, one hour apart, newest id = newest_time."""
    blocks = []
    for pid in sorted(ids):
        iso = (pd.Timestamp(newest_time) - pd.Timedelta(hours=max(ids) - pid)).isoformat() + "+00:00"
        blocks.append(f'<div class="tgme_widget_message_wrap"><div data-post="ch/{pid}">'
                      f'<div class="tgme_widget_message_text js-message_text">post {pid}</div>'
                      f'<time datetime="{iso}"></time></div></div>')
    return "".join(blocks)


PAGES = {
    "https://t.me/s/ch": page(range(41, 51), "2026-01-10 12:00"),
    "https://t.me/s/ch?before=41": page(range(31, 41), "2026-01-10 02:00"),
    "https://t.me/s/ch?before=31": page(range(21, 31), "2026-01-09 16:00"),
    "https://t.me/s/ch?before=21": "",
}


class FetchTest(unittest.TestCase):
    def test_fetch_until_since(self):
        posts = fetch_channel("ch", since=pd.Timestamp("2026-01-09 20:00"), delay=0, get=PAGES.__getitem__, log=lambda *_: None)
        self.assertEqual(list(posts["id"]), list(range(34, 51)))  # id p is at 2026-01-10 12:00 minus (50 - p) hours; id 34 = 2026-01-09 20:00
        self.assertTrue((posts["time"] >= pd.Timestamp("2026-01-09 20:00")).all())

    def test_fetch_stops_at_known_id(self):
        posts = fetch_channel("ch", since=pd.Timestamp("2020-01-01"), after_id=44, delay=0, get=PAGES.__getitem__, log=lambda *_: None)
        self.assertEqual(list(posts["id"]), list(range(45, 51)))

    def test_fetch_stops_on_empty_page(self):
        posts = fetch_channel("ch", since=pd.Timestamp("2020-01-01"), delay=0, get=PAGES.__getitem__, log=lambda *_: None)
        self.assertEqual(posts["id"].min(), 21)
```

- [ ] **Step 2: Run it — expect failure** (`No module named 'alerts.sources'`)

- [ ] **Step 3: Write `alerts/sources.py`**

```python
"""Network access: repo CSV downloads and Telegram public-preview pagination."""
import time
import urllib.request

import pandas as pd

from .config import FRESH_DIR, REPO_RAW, SNAPSHOT_DIR
from .parse import parse_tg_page, read_posts

HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_bytes(url, retries=3, timeout=30):
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as response:
                return response.read()
        except OSError as error:
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {url} ({error})")
            time.sleep(5 * attempt)


def get_text(url):
    return get_bytes(url).decode("utf-8")


def download(url, dest):
    """Save the file byte-for-byte (text mode would turn CRLF into CR CR LF on Windows)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    dest.write_bytes(get_bytes(url, timeout=120))
    return dest


def fetch_channel(channel, since, after_id=0, delay=1.5, get=get_text, log=print):
    """Walk t.me/s/<channel> backwards (newest first) until posts are older than `since`
    or reach `after_id`. Returns posts with time >= since and id > after_id, oldest first."""
    url, posts, pages, last_oldest = f"https://t.me/s/{channel}", {}, 0, None
    while True:
        batch = parse_tg_page(get(url))
        pages += 1
        if not batch:
            break
        posts.update({p["id"]: p for p in batch})
        oldest = min(batch, key=lambda p: p["id"])
        if oldest["time"] < since or oldest["id"] <= after_id + 1 or oldest["id"] == last_oldest:
            break
        last_oldest = oldest["id"]
        if pages % 50 == 0:
            log(f"  {channel}: {len(posts)} posts so far, back to {oldest['time']}")
        url = f"https://t.me/s/{channel}?before={oldest['id']}"
        time.sleep(delay)
    df = pd.DataFrame(list(posts.values()), columns=["id", "time", "text"])
    df = df[(df["time"] >= since) & (df["id"] > after_id)]
    return df.sort_values("id").reset_index(drop=True)


def refresh():
    """--fresh: latest repo CSVs + Telegram posts newer than the snapshot -> data/fresh/."""
    for name in ("official_data_en.csv", "volunteer_data_en.csv"):
        download(REPO_RAW + name, FRESH_DIR / name)
    for channel in ("air_alert_ua", "tryvoga_chomu"):
        old = read_posts(SNAPSHOT_DIR / f"{channel}_posts.csv")
        new = fetch_channel(channel, since=old["time"].max(), after_id=int(old["id"].max()))
        print(f"{channel}: {len(new)} new posts")
        pd.concat([old, new]).drop_duplicates("id").to_csv(FRESH_DIR / f"{channel}_posts.csv", index=False)
    return FRESH_DIR
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Write `alerts/snapshot.py`**

```python
"""Maintainer tool: build the frozen snapshot committed to the repo.
Run with: python3 run.py snapshot   (slow: ~1 hour of polite Telegram requests)"""
import tempfile
from pathlib import Path

import pandas as pd

from .config import AIR_ALERT_SINCE, CHOMU_SINCE, DATA_START, REGION_MAP, REPO_RAW, SNAPSHOT_DIR
from .parse import build_region_map
from .sources import download, fetch_channel


def build_snapshot():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("official_data_en.csv", "volunteer_data_en.csv", "official_data_uk.csv"):
            download(REPO_RAW + name, tmp / name)
        region_map = build_region_map(pd.read_csv(tmp / "official_data_uk.csv"))
        pd.DataFrame(sorted(region_map.items()), columns=["name", "oblast"]).to_csv(REGION_MAP, index=False)
        print(f"region map: {len(region_map)} names")
        for name in ("official_data_en.csv", "volunteer_data_en.csv"):
            df = pd.read_csv(tmp / name).drop_duplicates()
            started = pd.to_datetime(df["started_at"], utc=True).dt.tz_localize(None)
            df[started >= DATA_START].to_csv(SNAPSHOT_DIR / name, index=False)
            print(f"{name}: {int((started >= DATA_START).sum())} rows kept")
    for channel, since in (("air_alert_ua", AIR_ALERT_SINCE), ("tryvoga_chomu", CHOMU_SINCE)):
        posts = fetch_channel(channel, since)
        posts.to_csv(SNAPSHOT_DIR / f"{channel}_posts.csv", index=False)
        print(f"{channel}: {len(posts)} posts saved ({posts['time'].min()} → {posts['time'].max()})")


if __name__ == "__main__":
    build_snapshot()
```

- [ ] **Step 6: Commit the code**

```bash
git add alerts/sources.py alerts/snapshot.py tests/test_sources.py
git commit -m "Add downloads, Telegram pagination, --fresh refresh and snapshot builder" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Ask the user for permission, then build the snapshot in the background**

Tell the user: downloads `official_data_en.csv` (31 MB), `volunteer_data_en.csv` (9 MB), `official_data_uk.csv` (41 MB) from `raw.githubusercontent.com/Vadimkin/…`, plus ~1,700 Telegram preview pages (1.5 s apart). Only after a yes:

Run (background): `python3 run.py snapshot`
Expected at the end: `region map: N names`, two `rows kept` lines, `air_alert_ua: … posts saved (2026-08-25… → 2026-09-07 06:09…)`, `tryvoga_chomu: … posts saved (2025-11-24… → 2026-10-1…)`.
If it stops with `Unknown oblast names … add them to OBLAST_UK: [...]`: add those names to `OBLAST_UK` in `parse.py`, re-run.

- [ ] **Step 8: When finished, check sizes and commit the snapshot**

Run: `ls -la data/snapshot data/region_map.csv` (each file < 50 MB).

```bash
git add data/snapshot data/region_map.csv
git commit -m "Add frozen data snapshot" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Episodes and daily counts

**Why:** all sources are combined by merging overlapping intervals per oblast into episodes; the model predicts episodes per Kyiv day.

**Files:**
- Create: `alerts/episodes.py`, `tests/test_episodes.py`

**Interfaces:**
- Consumes: `config` constants/helpers; interval tables `DataFrame[oblast, start, end, …]`
- Produces: `naive_medians(vol) -> dict[str, float]`, `impute_naive(vol, medians) -> DataFrame[oblast, start, end, source]`, `clip_as_of(intervals, T) -> DataFrame`, `merge_episodes(intervals) -> DataFrame[oblast, start, end]`, `daily_counts(episodes, days, data_end) -> DataFrame(index=days, columns=OBLASTS, float, NaN after data end)`, `episode_starts(episodes) -> DataFrame[oblast, day, hour]`

- [ ] **Step 1: Write the failing test `tests/test_episodes.py`**

```python
import unittest

import numpy as np
import pandas as pd

from alerts.episodes import (clip_as_of, daily_counts, episode_starts, impute_naive, merge_episodes,
                             naive_medians)

T = pd.Timestamp


def intervals(rows):
    return pd.DataFrame(rows, columns=["oblast", "start", "end"]).assign(
        start=lambda d: pd.to_datetime(d["start"]), end=lambda d: pd.to_datetime(d["end"]))


class EpisodesTest(unittest.TestCase):
    def test_merge_overlapping_and_touching(self):
        eps = merge_episodes(intervals([
            ("Kyiv City", "2026-03-01 10:00", "2026-03-01 10:30"),
            ("Kyiv City", "2026-03-01 10:20", "2026-03-01 11:00"),  # overlaps
            ("Kyiv City", "2026-03-01 11:00", "2026-03-01 11:10"),  # touches
            ("Kyiv City", "2026-03-01 12:00", "2026-03-01 12:05"),  # separate
            ("Lvivska oblast", "2026-03-01 10:10", "2026-03-01 10:20"),
        ]))
        kyiv = eps[eps["oblast"] == "Kyiv City"].sort_values("start")
        self.assertEqual(list(zip(kyiv["start"], kyiv["end"])),
                         [(T("2026-03-01 10:00"), T("2026-03-01 11:10")), (T("2026-03-01 12:00"), T("2026-03-01 12:05"))])
        self.assertEqual(int((eps["oblast"] == "Lvivska oblast").sum()), 1)

    def test_clip_as_of(self):
        clipped = clip_as_of(intervals([
            ("Kyiv City", "2026-03-01 10:00", "2026-03-01 12:00"),  # running at T -> clipped
            ("Kyiv City", "2026-03-01 11:30", "2026-03-01 11:40"),  # starts after T -> dropped
        ]), T("2026-03-01 11:00"))
        self.assertEqual(len(clipped), 1)
        self.assertEqual(clipped.iloc[0]["end"], T("2026-03-01 11:00"))

    def test_naive_imputation_uses_oblast_median(self):
        vol = intervals([
            ("Kyiv City", "2025-10-01 10:00", "2025-10-01 10:20"),
            ("Kyiv City", "2025-10-02 10:00", "2025-10-02 10:40"),
            ("Kyiv City", "2025-10-03 10:00", "2025-10-03 11:00"),
            ("Kyiv City", "2026-01-01 10:00", "2026-01-01 10:30"),  # naive
            ("Lvivska oblast", "2026-01-01 10:00", "2026-01-01 10:30"),  # naive, no median -> 30 min
        ]).assign(naive=[False, False, False, True, True])
        medians = naive_medians(vol)
        self.assertEqual(medians, {"Kyiv City": 40.0})
        out = impute_naive(vol, medians)
        self.assertEqual(out.iloc[3]["end"], T("2026-01-01 10:40"))
        self.assertEqual(out.iloc[4]["end"], T("2026-01-01 10:30"))
        self.assertEqual(out.iloc[0]["end"], T("2025-10-01 10:20"))

    def test_daily_counts_use_kyiv_days(self):
        eps = intervals([
            ("Kyiv City", "2026-07-01 21:30", "2026-07-01 22:00"),  # 00:30 Kyiv on 2 July
            ("Kyiv City", "2026-07-01 20:30", "2026-07-01 20:40"),  # 23:30 Kyiv on 1 July
        ])
        days = pd.date_range("2026-07-01", "2026-07-04")
        daily = daily_counts(eps, days, data_end=T("2026-07-02 12:00"))
        self.assertEqual(list(daily["Kyiv City"].iloc[:2]), [1.0, 1.0])
        self.assertTrue(np.isnan(daily.loc["2026-07-03", "Kyiv City"]))  # after data end
        self.assertEqual(daily.loc["2026-07-01", "Lvivska oblast"], 0.0)
        starts = episode_starts(eps)
        self.assertEqual(sorted(zip(starts["day"], starts["hour"])),
                         [(T("2026-07-01"), 23), (T("2026-07-02"), 0)])
```

- [ ] **Step 2: Run it — expect failure** (`No module named 'alerts.episodes'`)

- [ ] **Step 3: Write `alerts/episodes.py`**

```python
"""Combine alert intervals from all sources into episodes and count them per Kyiv day."""
import numpy as np
import pandas as pd

from .config import DATA_START, NAIVE_MEDIAN_END, OBLASTS, kyiv_day, to_kyiv


def naive_medians(vol):
    """Median duration in minutes of non-naive eTryvoga alerts per oblast, Sep-Nov 2025 only
    (before any training example, so it cannot leak)."""
    window = vol[~vol["naive"] & (vol["start"] >= DATA_START) & (vol["start"] < NAIVE_MEDIAN_END)]
    minutes = (window["end"] - window["start"]).dt.total_seconds() / 60
    return minutes.groupby(window["oblast"]).median().to_dict()


def impute_naive(vol, medians):
    """H6: alerts without an 'end' message get start + the oblast's median duration (30 min if unknown)."""
    minutes = vol["oblast"].map(medians).fillna(30.0)
    imputed_end = vol["start"] + pd.to_timedelta(minutes, unit="min")
    return pd.DataFrame({
        "oblast": vol["oblast"].to_numpy(),
        "start": vol["start"].to_numpy(),
        "end": imputed_end.where(vol["naive"], vol["end"]).to_numpy(),
        "source": "volunteer_csv",
    })


def clip_as_of(intervals, T):
    """What was known at time T: alerts that started before T; a still-running alert ends at T."""
    known = intervals[intervals["start"] < T].copy()
    known["end"] = known["end"].where(known["end"] <= T, T)
    return known


def merge_episodes(intervals):
    """Per oblast, merge intervals that overlap or touch into one episode."""
    rows = []
    for oblast, group in intervals.sort_values("start").groupby("oblast", sort=False):
        cur_start = cur_end = None
        for start, end in zip(group["start"], group["end"]):
            if cur_start is not None and start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                if cur_start is not None:
                    rows.append((oblast, cur_start, cur_end))
                cur_start, cur_end = start, end
        rows.append((oblast, cur_start, cur_end))
    return pd.DataFrame(rows, columns=["oblast", "start", "end"])


def daily_counts(episodes, days, data_end):
    """Episodes starting on each Kyiv day per oblast. Days after the data ends are NaN (unknown)."""
    if len(episodes):
        day = to_kyiv(episodes["start"]).dt.normalize()
        counts = pd.crosstab(day.to_numpy(), episodes["oblast"].to_numpy())
    else:
        counts = pd.DataFrame()
    counts = counts.reindex(index=days, columns=OBLASTS, fill_value=0).astype(float)
    counts.loc[counts.index > kyiv_day(data_end)] = np.nan
    counts.index.name, counts.columns.name = "day", None
    return counts


def episode_starts(episodes):
    """Kyiv day and hour of every episode start (for hourly profiles)."""
    local = to_kyiv(episodes["start"])
    return pd.DataFrame({"oblast": episodes["oblast"].to_numpy(),
                         "day": local.dt.normalize().to_numpy(), "hour": local.dt.hour.to_numpy()})
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add alerts/episodes.py tests/test_episodes.py
git commit -m "Merge alert intervals into episodes and count them per Kyiv day" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Features, hourly profile and the leakage test (test-first)

**Why:** this is where leakage would happen. The leakage test is written first: features computed from all data must equal features computed after deleting everything after the forecast time.

**Files:**
- Create: `alerts/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `episodes.*`, `parse.chomu_flags`, `parse.THREAT_PATTERNS`, `parse.TG_COLUMNS`, `config.*`
- Produces: `FEATURES: list[str]`, `tg_daily(flags, days) -> DataFrame(index=days, columns=TG_COLUMNS)`, `make_rows(daily, tg, t0_days, leads=range(1, 8)) -> DataFrame[oblast(int), t0, target_day, lead_days, weekday, cnt_1d, cnt_7d, cnt_28d, cnt_same_wd, nat_1d, nat_7d, tg_mention_1d, tg_mention_7d, tg_<group>_1d, tg_<group>_7d…, actual]`, `hourly_profile(starts, oblast, t0, window=28) -> np.ndarray(24)`

- [ ] **Step 1: Write the failing test `tests/test_features.py`**

```python
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from alerts.config import OBLASTS, kyiv_day_start_utc
from alerts.episodes import clip_as_of, daily_counts, episode_starts, merge_episodes
from alerts.features import FEATURES, hourly_profile, make_rows, tg_daily
from alerts.parse import chomu_flags


def synthetic(seed=0):
    rng = np.random.RandomState(seed)
    rows, posts = [], []
    for day in pd.date_range("2026-03-01", "2026-05-09"):
        for oblast in ("Kyivska oblast", "Lvivska oblast", "Kyiv City"):
            for _ in range(rng.poisson(3)):
                start = day + pd.Timedelta(minutes=int(rng.randint(0, 24 * 60)))
                rows.append((oblast, start, start + pd.Timedelta(minutes=int(rng.randint(10, 300)))))
        for _ in range(rng.poisson(5)):
            posts.append((day + pd.Timedelta(minutes=int(rng.randint(0, 24 * 60))),
                          rng.choice(["Шахед на Київщину", "Балістика на Київ", "Ракети на Львів", "Тихо"])))
    iv = pd.DataFrame(rows, columns=["oblast", "start", "end"])
    tg = pd.DataFrame(posts, columns=["time", "text"])
    return iv, tg


class LeakageTest(unittest.TestCase):
    def test_features_ignore_everything_after_forecast_time(self):
        iv, posts = synthetic()
        days = pd.date_range("2026-03-01", "2026-05-16")
        t0 = pd.Timestamp("2026-04-10")
        T = kyiv_day_start_utc(t0)

        full_eps = merge_episodes(iv)
        cut_eps = merge_episodes(clip_as_of(iv, T))
        full_daily = daily_counts(full_eps, days, data_end=iv["start"].max())
        cut_daily = daily_counts(cut_eps, days, data_end=T)
        self.assertFalse(full_daily.loc[t0].equals(cut_daily.loc[t0]))  # the future really differs

        full = make_rows(full_daily, tg_daily(chomu_flags(posts), days), [t0])
        cut = make_rows(cut_daily, tg_daily(chomu_flags(posts[posts["time"] < T]), days), [t0])
        assert_frame_equal(full[FEATURES], cut[FEATURES])

        for oblast in ("Kyivska oblast", "Lvivska oblast"):
            np.testing.assert_array_equal(hourly_profile(episode_starts(full_eps), oblast, t0),
                                          hourly_profile(episode_starts(cut_eps), oblast, t0))


class RowsTest(unittest.TestCase):
    def test_row_values(self):
        days = pd.date_range("2026-03-01", "2026-04-30")
        daily = pd.DataFrame(0.0, index=days, columns=OBLASTS)
        daily.loc["2026-04-09", "Kyiv City"] = 7
        daily.loc["2026-04-05", "Kyiv City"] = 2
        tg = tg_daily(chomu_flags(pd.DataFrame({"time": [], "text": []})), days)
        rows = make_rows(daily, tg, [pd.Timestamp("2026-04-10")], leads=[1, 3])
        kyiv = rows[(rows["oblast"] == 10)].set_index("lead_days")
        self.assertEqual(kyiv.loc[1, "cnt_1d"], 7)
        self.assertAlmostEqual(kyiv.loc[1, "cnt_7d"], 9 / 7)
        self.assertEqual(kyiv.loc[3, "target_day"], pd.Timestamp("2026-04-12"))
        self.assertEqual(kyiv.loc[3, "cnt_same_wd"], 2)  # 2026-04-12 minus 7 days = 2026-04-05
        self.assertEqual(kyiv.loc[1, "nat_1d"], 7)
        self.assertTrue(np.isnan(kyiv.loc[1, "tg_mention_1d"]))  # no Telegram coverage -> unknown


class ProfileTest(unittest.TestCase):
    def test_profile_counts_last_28_days(self):
        starts = pd.DataFrame({"oblast": ["Kyiv City"] * 3, "hour": [22, 22, 3],
                               "day": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-10"])})
        profile = hourly_profile(starts, "Kyiv City", pd.Timestamp("2026-04-10"))
        self.assertEqual(profile[22], 1.0)  # the 04-10 start is not before t0
        self.assertAlmostEqual(profile.sum(), 1.0)

    def test_profile_without_history_is_uniform(self):
        starts = pd.DataFrame({"oblast": [], "hour": [], "day": pd.to_datetime([])})
        profile = hourly_profile(starts, "Zakarpatska oblast", pd.Timestamp("2026-04-10"))
        np.testing.assert_allclose(profile, np.full(24, 1 / 24))
```

- [ ] **Step 2: Run it — expect failure** (`No module named 'alerts.features'`)

- [ ] **Step 3: Write `alerts/features.py`**

```python
"""Feature rows: what the model may know at 00:00 Kyiv of day t0 (the forecast time).
Every statistic for day t0 uses only days before t0 (shift(1)); the leakage test checks this."""
import numpy as np
import pandas as pd

from .config import MAX_LEAD, OBLASTS, to_kyiv
from .parse import TG_COLUMNS, THREAT_PATTERNS

FEATURES = (["oblast", "lead_days", "weekday", "cnt_1d", "cnt_7d", "cnt_28d", "cnt_same_wd",
             "nat_1d", "nat_7d", "tg_mention_1d", "tg_mention_7d"]
            + [f"tg_{group}_{window}" for group in THREAT_PATTERNS for window in ("1d", "7d")])


def tg_daily(flags, days):
    """Posts per Kyiv day mentioning each oblast / threat group. NaN outside the collected period."""
    if flags.empty:
        return pd.DataFrame(np.nan, index=days, columns=TG_COLUMNS)
    day = to_kyiv(flags["time"]).dt.normalize()
    daily = flags[TG_COLUMNS].groupby(day.to_numpy()).sum().reindex(days).astype(float)
    covered = (days >= day.min()) & (days <= day.max())
    daily.loc[covered] = daily.loc[covered].fillna(0.0)
    return daily


def make_rows(daily, tg, t0_days, leads=range(1, MAX_LEAD + 1)):
    """One row per (oblast, forecast day t0, lead k). Target day D = t0 + k - 1.
    `daily` and `tg` must share one continuous daily index that covers every target day."""
    t0_days = pd.DatetimeIndex(t0_days)
    national = daily.sum(axis=1, min_count=1)
    cnt_1d, cnt_7d, cnt_28d = daily.shift(1), daily.rolling(7).mean().shift(1), daily.rolling(28).mean().shift(1)
    nat_1d, nat_7d = national.shift(1), national.rolling(7).mean().shift(1)
    tg_1d, tg_7d = tg.shift(1), tg.rolling(7).sum().shift(1)
    same_weekday = daily.shift(7)  # read at target day D -> count on D - 7, always before t0 for k <= 7
    parts = []
    for k in leads:
        target_days = t0_days + pd.Timedelta(days=k - 1)
        for i, oblast in enumerate(OBLASTS):
            part = pd.DataFrame({
                "oblast": i, "t0": t0_days, "target_day": target_days, "lead_days": k,
                "weekday": target_days.weekday,
                "cnt_1d": cnt_1d[oblast].reindex(t0_days).to_numpy(),
                "cnt_7d": cnt_7d[oblast].reindex(t0_days).to_numpy(),
                "cnt_28d": cnt_28d[oblast].reindex(t0_days).to_numpy(),
                "cnt_same_wd": same_weekday[oblast].reindex(target_days).to_numpy(),
                "nat_1d": nat_1d.reindex(t0_days).to_numpy(),
                "nat_7d": nat_7d.reindex(t0_days).to_numpy(),
                "tg_mention_1d": tg_1d[f"m_{oblast}"].reindex(t0_days).to_numpy(),
                "tg_mention_7d": tg_7d[f"m_{oblast}"].reindex(t0_days).to_numpy(),
            })
            for group in THREAT_PATTERNS:
                part[f"tg_{group}_1d"] = tg_1d[group].reindex(t0_days).to_numpy()
                part[f"tg_{group}_7d"] = tg_7d[group].reindex(t0_days).to_numpy()
            part["actual"] = daily[oblast].reindex(target_days).to_numpy()
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def hourly_profile(starts, oblast, t0, window=28):
    """Share of the oblast's episode starts in each Kyiv hour over the `window` days before t0."""
    t0 = pd.Timestamp(t0)
    recent = starts[(starts["oblast"] == oblast) & (starts["day"] < t0)
                    & (starts["day"] >= t0 - pd.Timedelta(days=window))]
    counts = np.bincount(recent["hour"].to_numpy(dtype=int), minlength=24).astype(float)
    return counts / counts.sum() if counts.sum() else np.full(24, 1 / 24)
```

- [ ] **Step 4: Run tests — expect PASS.** If the leakage test fails, stop and use superpowers:systematic-debugging — do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add alerts/features.py tests/test_features.py
git commit -m "Add as-of feature rows, hourly profile and leakage test" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Model with a training cutoff

**Why:** a forecast for day t0 may only learn from days whose answer was known before t0.

**Files:**
- Create: `alerts/model.py`, `tests/helpers.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `features.FEATURES`, `config.TRAIN_START`
- Produces: `training_rows(rows, before_day) -> DataFrame`, `train(rows, before_day) -> HistGradientBoostingRegressor`, `predict(model, rows) -> np.ndarray`; test helper `synthetic_rows(n_days=60, start="2026-01-01") -> DataFrame`

- [ ] **Step 1: Write `tests/helpers.py` and the failing `tests/test_model.py`**

```python
# tests/helpers.py
import numpy as np
import pandas as pd

from alerts.features import FEATURES


def synthetic_rows(n_days=60, start="2026-01-01"):
    """Fake feature rows for 3 oblasts x 7 leads where actual depends on cnt_7d."""
    rng = np.random.RandomState(0)
    t0 = pd.date_range(start, periods=n_days)
    parts = []
    for k in range(1, 8):
        for oblast in range(3):
            part = pd.DataFrame({name: rng.rand(n_days) for name in FEATURES})
            part["oblast"], part["lead_days"] = oblast, k
            part["t0"], part["target_day"] = t0, t0 + pd.Timedelta(days=k - 1)
            part["actual"] = rng.poisson(1 + 6 * part["cnt_7d"])
            parts.append(part)
    return pd.concat(parts, ignore_index=True)
```

```python
# tests/test_model.py
import unittest

import numpy as np
import pandas as pd

from alerts.model import predict, train, training_rows
from tests.helpers import synthetic_rows


class ModelTest(unittest.TestCase):
    def test_training_rows_respect_cutoff(self):
        rows = synthetic_rows()
        cutoff = pd.Timestamp("2026-02-01")
        used = training_rows(rows, cutoff)
        self.assertLess(used["target_day"].max(), cutoff)
        self.assertGreater(len(used), 0)

    def test_train_and_predict(self):
        rows = synthetic_rows()
        model = train(rows, pd.Timestamp("2026-02-20"))
        test = rows[rows["t0"] == pd.Timestamp("2026-02-20")]
        pred = predict(model, test)
        self.assertTrue((pred > 0).all())
        self.assertGreater(np.corrcoef(pred, test["cnt_7d"])[0, 1], 0.5)  # learned the real signal

    def test_too_little_history_is_an_error(self):
        with self.assertRaises(ValueError):
            train(synthetic_rows(), pd.Timestamp("2026-01-03"))
```

- [ ] **Step 2: Run it — expect failure** (`No module named 'alerts.model'`)

- [ ] **Step 3: Write `alerts/model.py`**

```python
"""Daily alert-count model: Poisson gradient boosting, retrained for every forecast date."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import TRAIN_START
from .features import FEATURES


def training_rows(rows, before_day):
    """Only examples whose target day ended before the forecast day (answer already known)."""
    mask = (rows["target_day"] < before_day) & (rows["target_day"] >= TRAIN_START) & rows["actual"].notna()
    return rows[mask]


def train(rows, before_day):
    data = training_rows(rows, before_day)
    if len(data) < 500:
        raise ValueError(f"Not enough training data before {pd.Timestamp(before_day).date()} ({len(data)} rows).")
    model = HistGradientBoostingRegressor(loss="poisson", random_state=0,
                                          categorical_features=[name == "oblast" for name in FEATURES])
    model.fit(data[FEATURES], data["actual"])
    return model


def predict(model, rows):
    return model.predict(rows[FEATURES])
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add alerts/model.py tests/helpers.py tests/test_model.py
git commit -m "Add Poisson count model with training cutoff" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Evaluation

**Why:** the honest test — rolling forecasts over the last 28 days, compared with two baselines; plus the peak-hour hit rate and the H6 check.

**Files:**
- Create: `alerts/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `model.train`, `model.predict`, `features.hourly_profile`, `config.*`
- Produces: `test_period(last_full_day) -> DatetimeIndex`, `backtest(rows, days) -> DataFrame[…rows columns, pred, b1, b2]`, `mae_table(bt, by=None) -> DataFrame[model, b1, b2, skill_vs_best]`, `bias(bt) -> float`, `peak_hit_rate(starts, days) -> float`, `naive_check(vol, medians, start, end) -> DataFrame[method, MAE, minutes, alerts]`

- [ ] **Step 1: Write the failing test `tests/test_evaluate.py`**

```python
import unittest
from unittest import mock

import pandas as pd

from alerts import model as model_module
from alerts.evaluate import backtest, bias, mae_table, naive_check, peak_hit_rate, test_period
from tests.helpers import synthetic_rows


class EvaluateTest(unittest.TestCase):
    def test_backtest_retrains_before_each_forecast_day(self):
        rows = synthetic_rows()
        days = test_period(pd.Timestamp("2026-02-28"))[-3:]
        with mock.patch("alerts.evaluate.train", side_effect=model_module.train) as spy:
            bt = backtest(rows, days)
        cutoffs = sorted(pd.Timestamp(call.kwargs["before_day"]) for call in spy.call_args_list)
        self.assertEqual(cutoffs, list(pd.DatetimeIndex(bt["t0"].unique()).sort_values()))
        self.assertTrue(bt["target_day"].isin(days).all())
        self.assertTrue((bt["t0"] <= bt["target_day"]).all())
        self.assertEqual(len(bt), 3 * 3 * 7)

    def test_mae_table_and_bias(self):
        bt = pd.DataFrame({"actual": [2, 4], "pred": [3, 4], "b1": [0, 4], "b2": [2, 0], "lead_days": [1, 2]})
        table = mae_table(bt)
        self.assertEqual(list(table.loc[0, ["model", "b1", "b2"]]), [0.5, 1.0, 2.0])
        self.assertAlmostEqual(table.loc[0, "skill_vs_best"], 0.5)
        self.assertEqual(bias(bt), 0.5)
        self.assertEqual(list(mae_table(bt, "lead_days")["model"]), [1.0, 0.0])

    def test_peak_hit_rate(self):
        starts = pd.DataFrame({"oblast": ["Kyiv City"] * 4, "hour": [22, 23, 21, 5],
                               "day": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-10"])})
        self.assertEqual(peak_hit_rate(starts, pd.DatetimeIndex(["2026-03-10"])), 0.0)
        starts.loc[3, "hour"] = 22
        self.assertEqual(peak_hit_rate(starts, pd.DatetimeIndex(["2026-03-10"])), 1.0)

    def test_naive_check(self):
        vol = pd.DataFrame({"oblast": ["Kyiv City", "Kyiv City"], "naive": [False, True],
                            "start": pd.to_datetime(["2026-01-05 10:00", "2026-01-06 10:00"]),
                            "end": pd.to_datetime(["2026-01-05 10:50", "2026-01-06 10:30"])})
        table = naive_check(vol, {"Kyiv City": 40.0}, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"))
        self.assertEqual(list(table["MAE, minutes"]), [20.0, 10.0])
        self.assertEqual(list(table["alerts"]), [1, 1])
```

- [ ] **Step 2: Run it — expect failure** (`No module named 'alerts.evaluate'`)

- [ ] **Step 3: Write `alerts/evaluate.py`**

```python
"""Honest test: rolling forecasts over the last TEST_DAYS days vs two simple baselines."""
import numpy as np
import pandas as pd

from .config import OBLASTS, TEST_DAYS
from .features import hourly_profile
from .model import predict, train


def test_period(last_full_day):
    return pd.date_range(last_full_day - pd.Timedelta(days=TEST_DAYS - 1), last_full_day)


def backtest(rows, days):
    """For every forecast day t0 that targets a test day, retrain on earlier days and predict.
    Baselines: b1 = mean of the last 7 days, b2 = same weekday one week earlier."""
    test = rows[rows["target_day"].isin(days)]
    parts = []
    for t0, group in test.groupby("t0"):
        model = train(rows, before_day=t0)
        parts.append(group.assign(pred=predict(model, group)))
    result = pd.concat(parts, ignore_index=True)
    return result.assign(b1=result["cnt_7d"], b2=result["cnt_same_wd"])


def mae_table(bt, by=None):
    """Mean absolute error (alerts/day) of model and baselines; skill = 1 - model / best baseline."""
    errors = pd.DataFrame({name: (bt[name] - bt["actual"]).abs() for name in ("pred", "b1", "b2")})
    errors = errors.rename(columns={"pred": "model"})
    if by is None:
        table = errors.mean().to_frame().T
    else:
        table = errors.groupby(bt[by]).mean()
    table["skill_vs_best"] = 1 - table["model"] / table[["b1", "b2"]].min(axis=1)
    return table


def bias(bt):
    """Mean predicted minus mean actual (positive = over-forecast)."""
    return float((bt["pred"] - bt["actual"]).mean())


def peak_hit_rate(starts, days):
    """Share of real episode starts on the test days that fell in the predicted top-3 hours."""
    hits = total = 0
    for day in days:
        for oblast in OBLASTS:
            top3 = np.argsort(hourly_profile(starts, oblast, day))[-3:]
            hours = starts.loc[(starts["oblast"] == oblast) & (starts["day"] == day), "hour"]
            hits += int(hours.isin(top3).sum())
            total += len(hours)
    return hits / total if total else float("nan")


def naive_check(vol, medians, start, end):
    """H6: hide the real end of non-naive alerts and compare '+30 min' with the per-oblast median."""
    real = vol[~vol["naive"] & (vol["start"] >= start) & (vol["start"] < end)]
    minutes = (real["end"] - real["start"]).dt.total_seconds() / 60
    median = real["oblast"].map(medians).fillna(30.0)
    return pd.DataFrame({
        "method": ["+30 min (default)", "per-oblast median (H6)"],
        "MAE, minutes": [(minutes - 30).abs().mean(), (minutes - median).abs().mean()],
        "alerts": [len(real), len(real)],
    })
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add alerts/evaluate.py tests/test_evaluate.py
git commit -m "Add backtest, baselines, MAE/skill/bias, peak-hour hit rate and H6 check" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Report and command-line app (end-to-end run on the snapshot)

**Why:** this connects everything into the product the user runs. Load the **dataviz** skill before writing chart code.

**Files:**
- Create: `alerts/report.py`, `alerts/main.py`, `tests/test_report.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: everything above
- Produces: `report.figure_html(fig) -> str`, `report.heatmap(table, title, label) -> Figure`, `report.bar_chart(series, title, ylabel) -> Figure`, `report.line_chart(table, title, xlabel, ylabel) -> Figure`, `report.table_html(df, digits=2) -> str`, `report.write_report(path, title, sections)`; `main.parse_start(text) -> Timestamp`, `main.check_start(start, end_day)`, `main.load_inputs(fresh) -> dict`, `main.build(inputs) -> dict`, `main.forecast(state, start, n_days) -> DataFrame`, `main.run_evaluation(inputs, state)`, `main.main(argv=None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report.py
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alerts.report import figure_html, heatmap, table_html, write_report


class ReportTest(unittest.TestCase):
    def test_write_report(self):
        fig = heatmap(pd.DataFrame([[0.1, 0.9]], index=["Kyiv City"], columns=[0, 1]), "t", "share")
        image = figure_html(fig)
        self.assertTrue(image.startswith('<img src="data:image/png;base64,'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.html"
            write_report(path, "Report", [("Section", image + table_html(pd.DataFrame({"a": [1.234]})))])
            text = path.read_text(encoding="utf-8")
        self.assertIn("<h2>Section</h2>", text)
        self.assertIn("1.23", text)
```

```python
# tests/test_main.py
import unittest

import pandas as pd

from alerts.main import check_start, parse_start


class StartDateTest(unittest.TestCase):
    def test_bad_format(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_start("10/12/2026")
        self.assertIn("YYYY-MM-DD", str(ctx.exception.code))

    def test_range(self):
        end_day = pd.Timestamp("2026-10-11")
        check_start(pd.Timestamp("2026-10-12"), end_day)  # last allowed day: no error
        for bad in ("2025-12-31", "2026-10-13"):
            with self.assertRaises(SystemExit) as ctx:
                check_start(pd.Timestamp(bad), end_day)
            self.assertIn("between 2026-01-01 and 2026-10-12", str(ctx.exception.code))
```

- [ ] **Step 2: Run — expect failure** (`No module named 'alerts.report'`)

- [ ] **Step 3: Write `alerts/report.py`**

```python
"""Charts (matplotlib, embedded as PNG) and a single self-contained HTML report."""
import base64
import html
import io

import matplotlib

matplotlib.use("Agg")  # no screen needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402


def figure_html(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}">'


def heatmap(table, title, label):
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * table.shape[1] + 3), 0.32 * table.shape[0] + 1.5))
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="Reds")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], fontsize=8, rotation=90 if table.shape[1] > 12 else 0)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index], fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    return fig


def bar_chart(series, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    series.plot.bar(ax=ax, color="#b03a2e")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8)
    return fig


def line_chart(table, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    table.plot(ax=ax, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig


def table_html(df, digits=2):
    return df.round(digits).to_html(border=0)


def write_report(path, title, sections):
    body = "".join(f"<h2>{html.escape(heading)}</h2>{content}" for heading, content in sections)
    style = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;line-height:1.45}"
             "table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px;"
             "text-align:right}img{max-width:100%}p.note{color:#555}")
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                    f"<style>{style}</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>",
                    encoding="utf-8")
```

- [ ] **Step 4: Write `alerts/main.py`**

```python
"""python3 run.py [--start YYYY-MM-DD] [--days N] [--fresh]

Without --start: full evaluation on the last 28 days + HTML report + forecast from the last data day.
With --start: only the forecast for --days days starting that Kyiv date."""
import argparse

import numpy as np
import pandas as pd

from . import config as C
from .episodes import daily_counts, episode_starts, impute_naive, merge_episodes, naive_medians
from .evaluate import backtest, bias, mae_table, naive_check, peak_hit_rate, test_period
from .features import hourly_profile, make_rows, tg_daily
from .model import predict, train
from .parse import (air_alert_events, air_alert_intervals, chomu_flags, load_official, load_region_map,
                    load_volunteer, read_posts)
from .report import bar_chart, figure_html, heatmap, line_chart, table_html, write_report


def parse_start(text):
    try:
        return pd.Timestamp(pd.to_datetime(text, format="%Y-%m-%d")).normalize()
    except (ValueError, TypeError):
        raise SystemExit(f"--start must be a date like 2026-10-05 (YYYY-MM-DD), got: {text}")


def check_start(start, end_day):
    latest = end_day + pd.Timedelta(days=1)
    if not C.EARLIEST_START <= start <= latest:
        raise SystemExit(f"--start must be between {C.EARLIEST_START.date()} and {latest.date()} "
                         f"(the data ends on {end_day.date()}).")


def load_inputs(fresh):
    folder = C.SNAPSHOT_DIR
    if fresh:
        from .sources import refresh  # network code only when asked for
        folder = refresh()
    official = load_official(folder / "official_data_en.csv")
    vol = load_volunteer(folder / "volunteer_data_en.csv")
    medians = naive_medians(vol)
    air_alert = air_alert_intervals(air_alert_events(read_posts(folder / "air_alert_ua_posts.csv")),
                                    load_region_map(C.REGION_MAP))
    intervals = pd.concat([official, impute_naive(vol, medians), air_alert], ignore_index=True)
    intervals = intervals[intervals["start"] >= C.DATA_START].reset_index(drop=True)
    return {"intervals": intervals, "vol": vol, "medians": medians,
            "chomu": read_posts(folder / "tryvoga_chomu_posts.csv")}


def build(inputs):
    episodes = merge_episodes(inputs["intervals"])
    data_end = inputs["intervals"]["start"].max()
    end_day = C.kyiv_day(data_end)
    days = pd.date_range(C.kyiv_day(C.DATA_START), end_day + pd.Timedelta(days=C.MAX_LEAD + 1))
    daily = daily_counts(episodes, days, data_end)
    tg = tg_daily(chomu_flags(inputs["chomu"]), days)
    return {"episodes": episodes, "starts": episode_starts(episodes), "daily": daily,
            "rows": make_rows(daily, tg, days), "end_day": end_day}


def forecast(state, start, n_days):
    check_start(start, state["end_day"])
    model = train(state["rows"], before_day=start)
    rows = state["rows"]
    rows = rows[(rows["t0"] == start) & (rows["lead_days"] <= n_days)].copy()
    rows["expected"] = predict(model, rows)
    profiles = {o: hourly_profile(state["starts"], o, start) for o in C.OBLASTS}
    records = []
    for row in rows.itertuples():
        oblast = C.OBLASTS[row.oblast]
        profile = profiles[oblast]
        record = {"oblast": oblast, "day": row.target_day.date(), "expected": row.expected,
                  "actual": row.actual,
                  "peak_hours": ", ".join(f"{h:02d}:00" for h in sorted(np.argsort(profile)[-3:]))}
        record.update({f"h{h:02d}": row.expected * profile[h] for h in range(24)})
        records.append(record)
    return pd.DataFrame(records)


def print_forecast(fc):
    cells = fc.assign(cell=fc["expected"].round(1).astype(str) + " (" + fc["peak_hours"] + ")")
    table = cells.pivot(index="oblast", columns="day", values="cell")
    with pd.option_context("display.width", 250, "display.max_columns", 10):
        print("Expected alerts per day (top-3 most likely start hours, Kyiv time):")
        print(table.to_string())


def save_forecast(fc, start):
    C.REPORTS_DIR.mkdir(exist_ok=True)
    path = C.REPORTS_DIR / f"forecast_{start.date()}.csv"
    fc.round(3).to_csv(path, index=False)
    return path


def run_evaluation(inputs, state):
    C.REPORTS_DIR.mkdir(exist_ok=True)
    rows, starts, daily, intervals = state["rows"], state["starts"], state["daily"], inputs["intervals"]
    last_full_day = state["end_day"] - pd.Timedelta(days=1)
    days = test_period(last_full_day)
    explore_end = days[0] - pd.Timedelta(days=C.MAX_LEAD - 1)  # earliest test forecast time
    print(f"Backtest: {days[0].date()} → {days[-1].date()} (retraining for every forecast day)...")
    bt = backtest(rows, days)
    bt.to_csv(C.REPORTS_DIR / "backtest.csv", index=False)
    overall, by_lead = mae_table(bt), mae_table(bt, "lead_days")
    by_oblast = mae_table(bt, "oblast").rename(index=dict(enumerate(C.OBLASTS)))
    peak = peak_hit_rate(starts, days)
    model_bias = bias(bt)
    summary = (f"On {days[0].date()}–{days[-1].date()} the model misses by {overall.loc[0, 'model']:.2f} alerts per "
               f"oblast-day on average (best baseline: {overall.loc[0, ['b1', 'b2']].min():.2f}; skill "
               f"{overall.loc[0, 'skill_vs_best']:+.0%}). Bias {model_bias:+.2f} alerts/day. "
               f"{peak:.0%} of alerts started in the predicted top-3 hours (random guess: 12.5%).")
    print(summary)

    train_starts = starts[(starts["day"] >= C.TRAIN_START) & (starts["day"] < explore_end)]
    h1 = pd.crosstab(train_starts["oblast"], train_starts["hour"], normalize="index")
    train_daily = daily.loc[C.TRAIN_START:explore_end - pd.Timedelta(days=1)]
    h3 = pd.DataFrame({"lag-1 autocorrelation": {o: train_daily[o].autocorr(1) for o in C.OBLASTS}})
    window = intervals[(intervals["start"] >= C.AIR_ALERT_SINCE) & (intervals["start"] < C.AIR_ALERT_UNTIL)]
    by_source = pd.crosstab(C.to_kyiv(window["start"]).dt.date, window["source"])
    sources = intervals.groupby("source")["start"].agg(["count", "min", "max"])

    fc_start = state["end_day"]
    fc = forecast(state, fc_start, C.MAX_LEAD)
    save_forecast(fc, fc_start)
    fc_table = fc.pivot(index="oblast", columns="day", values="expected")

    sections = [
        ("Result in one paragraph", f"<p>{summary}</p>"),
        ("1. Data", table_html(sources) + "<p class='note'>Alert starts per day by source while air_alert_ua "
         "was still posting (it covers the official CSV outage of 2026-08-29…09-02 if its column is non-zero):</p>"
         + table_html(by_source)),
        ("2. Exploration (training period only)",
         figure_html(heatmap(h1, "H1: share of alert starts by Kyiv hour", "share")) +
         figure_html(bar_chart(train_daily.mean().sort_values(ascending=False),
                               "H2: mean alerts per day by oblast", "alerts/day")) + table_html(h3)),
        ("3. H6: imputing missing alert ends", table_html(
            naive_check(inputs["vol"], inputs["medians"], C.TRAIN_START, explore_end).set_index("method"))),
        ("4. Test results", table_html(overall) +
         figure_html(line_chart(by_lead[["model", "b1", "b2"]], "MAE by days ahead", "days ahead", "alerts/day"))
         + table_html(by_oblast)),
        (f"5. Forecast from {fc_start.date()}", figure_html(heatmap(fc_table, "Expected alerts per day", "alerts"))
         + table_html(fc[["oblast", "day", "expected", "peak_hours"]].set_index(["oblast", "day"]), 1)),
    ]
    write_report(C.REPORTS_DIR / "report.html", "Air raid alerts forecast — evaluation", sections)
    print_forecast(fc)
    print(f"Report: {C.REPORTS_DIR / 'report.html'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Forecast air raid alerts per oblast and day (1-7 days ahead).")
    parser.add_argument("--start", help="first forecast day (Kyiv), YYYY-MM-DD. Without it: evaluation + report.")
    parser.add_argument("--days", type=int, default=C.MAX_LEAD, choices=range(1, C.MAX_LEAD + 1), metavar="1-7")
    parser.add_argument("--fresh", action="store_true", help="download the latest data first (slow)")
    args = parser.parse_args(argv)
    start = parse_start(args.start) if args.start else None
    inputs = load_inputs(args.fresh)
    state = build(inputs)
    if start is None:
        run_evaluation(inputs, state)
    else:
        fc = forecast(state, start, args.days)
        print_forecast(fc)
        print(f"Saved: {save_forecast(fc, start)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the unit tests — expect PASS** (`python3 run.py test`)

- [ ] **Step 6: Run the app end-to-end on the snapshot**

Run: `python3 run.py`
Expected: a "Backtest: … → …" line, the one-paragraph summary, a forecast table for 24 oblasts × 7 days, and `Report: …/reports/report.html`. Total time: a few minutes. Open the report and check every section renders.

Run: `python3 run.py --start 2026-10-01 --days 3`
Expected: table with `expected (hh:00, hh:00, hh:00)` cells; `reports/forecast_2026-10-01.csv` has an `actual` column filled (this is a backtest of a past week).

Run: `python3 run.py --start 2025-01-01` → one-line error, no traceback.

- [ ] **Step 7: Commit**

```bash
git add alerts/report.py alerts/main.py tests/test_report.py tests/test_main.py
git commit -m "Add HTML report and command-line app" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: README and CI

**Why:** reviewers read the README first; CI proves "runs on every machine".

**Files:**
- Create: `README.md` (replace the one-line file), `.github/workflows/ci.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: run
on: [push, pull_request]
jobs:
  run:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.9", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: python run.py test
      - run: python run.py
```

- [ ] **Step 2: Write `README.md`** with these sections, filling the numbers from the Task 9 run (copy the one-paragraph summary verbatim):
  1. What it does (2–3 sentences, the per-oblast daily forecast + peak hours).
  2. Quick start: `git clone …`, `cd air-raid-alerts-forecast`, `python3 run.py` (Windows: `python run.py`); then `python3 run.py --start 2026-10-01 --days 3`; `python3 run.py --fresh`; `python3 run.py test`. Fallback: `pip install -r requirements.txt` then `python -m alerts.main`.
  3. Results on the latest 28 days — the summary paragraph + MAE table (model / last-7-days mean / same weekday last week) + by-lead-days line.
  4. How it works — sources combined into episodes; as-of rule; model retrained per forecast date; baselines.
  5. Data sources with links and periods (spec §3), noting air_alert_ua stopped on 2026-09-07.
  6. Known limitations (spec §9).
  7. How this was built — AI-assisted (Claude Code), with the design doc and plan in `docs/superpowers/`.

- [ ] **Step 3: Commit**

```bash
git add README.md .github/workflows/ci.yml
git commit -m "Add README and CI on 3 OS x Python 3.9/3.13" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Ask the user before pushing** (`git push -u origin prototype`), then check CI results. If a job fails, use superpowers:systematic-debugging.

---

### Task 11: Independent checks

**Why:** the user's plan: catch what we both got used to.

- [ ] **Step 1:** Run the `leakage-auditor` instructions (via a general-purpose agent if project agents are not loaded) on the branch. Fix confirmed findings test-first.
- [ ] **Step 2:** Run the `repro-tester` instructions against the pushed branch (fresh clone → `python3 run.py`).
- [ ] **Step 3:** Use superpowers:verification-before-completion: re-run `python3 run.py test` and `python3 run.py`, compare numbers with the README.
- [ ] **Step 4:** Report results to the user; they decide on merging `prototype` into `main`.
