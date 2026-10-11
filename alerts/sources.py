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
