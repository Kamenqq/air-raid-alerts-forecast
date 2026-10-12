"""Maintainer tool: build the frozen snapshot committed to the repo.
Run with: python3 run.py snapshot   (slow: ~1 hour of polite Telegram requests)"""
import tempfile
from pathlib import Path

import pandas as pd

from .config import AIR_ALERT_SINCE, CHOMU_SINCE, DATA_START, REGION_MAP, REPO_RAW, SNAPSHOT_DIR
from .parse import build_region_map, redact
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
        posts["text"] = posts["text"].map(redact)
        posts.to_csv(SNAPSHOT_DIR / f"{channel}_posts.csv", index=False)
        print(f"{channel}: {len(posts)} posts saved ({posts['time'].min()} → {posts['time'].max()})")


if __name__ == "__main__":
    build_snapshot()
