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
    best = table[["b1", "b2"]].min(axis=1)
    table["skill_vs_best"] = 1 - table["model"] / best.where(best > 0)  # undefined if a baseline is perfect
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
