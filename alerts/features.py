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


def main_window(profile, share=0.5):
    """Shortest continuous span of hours (may cross midnight) expected to hold at least `share`
    of the oblast's alerts; among equally short spans, the one holding the most.
    Returns (start_hour, hours), or None for a flat profile (no recent alerts)."""
    if np.allclose(profile, profile[0]):
        return None
    doubled = np.concatenate([profile, profile])  # lets a window run past midnight
    for hours in range(1, 25):
        sums = np.array([doubled[start:start + hours].sum() for start in range(24)])
        if sums.max() >= share - 1e-9:
            return int(sums.argmax()), hours
