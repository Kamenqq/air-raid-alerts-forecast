"""Honest test: rolling forecasts over the last TEST_DAYS days vs two simple baselines."""
import numpy as np
import pandas as pd

from .config import OBLASTS, TEST_DAYS
from .details import WINDOWS
from .features import blended_chance, blended_slot_chances, forecast_chance, hourly_chance
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


def chance_scores(bt, cube, days, leads=(1, 7)):
    """Brier score (mean squared gap between chance and what happened, 0 or 1) of hourly alert chances
    for the backtest's forecasts: 'flat' (same chance every hour, last 28 days), 'hourly' (each hour's own
    28-day share), 'shown' (what the forecast shows). Lower is better. Uses each forecast's own t0."""
    rows = []
    for r in bt[bt["lead_days"].isin(leads)].itertuples():
        chance = hourly_chance(cube, days, OBLASTS[r.oblast], r.t0)
        observed = cube[r.oblast, days.get_loc(r.target_day)]
        shown = forecast_chance(blended_chance(cube, days, OBLASTS[r.oblast], r.t0), r.lead_days, r.pred, r.cnt_28d)
        variants = {"flat": np.full(24, chance.mean()), "hourly": chance, "shown": shown}
        rows += [(r.lead_days, name, float(((p - observed) ** 2).mean())) for name, p in variants.items()]
    scores = pd.DataFrame(rows, columns=["lead_days", "method", "brier"])
    return scores.pivot_table(index="lead_days", columns="method", values="brier")[["flat", "hourly", "shown"]]


def calibration(bt, cube, days, lead=1, two_hour=False):
    """When the forecast said 30-40%, how often was an alert really active? Shown chances in 10% bins, for
    each hour, or (two_hour) for each fixed 2-hour window 00-02 ... 22-00."""
    predicted, observed = [], []
    for r in bt[bt["lead_days"] == lead].itertuples():
        oblast = OBLASTS[r.oblast]
        chance = (blended_slot_chances(cube, days, oblast, r.t0, 2)[WINDOWS] if two_hour
                  else blended_chance(cube, days, oblast, r.t0))
        predicted.append(forecast_chance(chance, r.lead_days, r.pred, r.cnt_28d))
        seen = cube[r.oblast, days.get_loc(r.target_day)]
        observed.append(seen.reshape(12, 2).any(axis=1) if two_hour else seen)
    labels = [f"{10 * i}–{10 * (i + 1)}%" for i in range(10)]
    pairs = pd.DataFrame({"predicted": np.concatenate(predicted), "observed": np.concatenate(observed).astype(float)})
    pairs["bin"] = pd.cut(pairs["predicted"], np.linspace(0, 1, 11), labels=labels, include_lowest=True)
    table = pairs.groupby("bin", observed=True).agg(predicted=("predicted", "mean"), observed=("observed", "mean"),
                                                   hours=("observed", "size"))
    return table.rename(columns={"hours": "windows"}) if two_hour else table
