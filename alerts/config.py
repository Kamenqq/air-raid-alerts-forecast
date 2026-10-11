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
