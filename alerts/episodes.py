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
    """Episodes starting on each Kyiv day per oblast. The last, partly covered day and all later
    days are NaN (unknown), so a half-day count is never mistaken for a full day."""
    if len(episodes):
        day = to_kyiv(episodes["start"]).dt.normalize()
        counts = pd.crosstab(day.to_numpy(), episodes["oblast"].to_numpy())
    else:
        counts = pd.DataFrame()
    counts = counts.reindex(index=days, columns=OBLASTS, fill_value=0).astype(float)
    counts.loc[counts.index >= kyiv_day(data_end)] = np.nan
    counts.index.name, counts.columns.name = "day", None
    return counts


def episode_starts(episodes):
    """Kyiv day and hour of every episode start (for hourly profiles)."""
    local = to_kyiv(episodes["start"])
    return pd.DataFrame({"oblast": episodes["oblast"].to_numpy(),
                         "day": local.dt.normalize().to_numpy(), "hour": local.dt.hour.to_numpy()})
