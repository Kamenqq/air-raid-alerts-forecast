"""Details for a forecast: the chance of an alert in each fixed 2-hour window, and recent alert durations.
Both use only history from before the forecast moment; the report checks them once on the test period."""
import pandas as pd

from .config import OBLASTS, kyiv_day_start_utc

WINDOWS = list(range(0, 24, 2))  # start hours of the windows 00–02, 02–04, … 22–00
WINDOW_COLUMNS = [f"w{h:02d}_{(h + 2) % 24:02d}" for h in WINDOWS]
DURATION_COLUMNS = ["alerts", "typical", "p10", "p90", "shortest", "average", "longest"]


def durations(vol, start, window=28):
    """Alert durations in minutes per oblast: eTryvoga alerts that started in the `window` days before the
    forecast moment (00:00 Kyiv of `start`; window=None: all history) and ended before it. Naive alerts (no
    'all clear' message, made-up end) and alerts still running at that moment are left out."""
    moment = kyiv_day_start_utc(start)
    known = vol[~vol["naive"] & (vol["end"] <= moment)]
    if window is not None:
        known = known[known["start"] >= moment - pd.Timedelta(days=window)]
    minutes = (known["end"] - known["start"]).dt.total_seconds() / 60
    by = minutes.groupby(known["oblast"])
    table = pd.DataFrame({"alerts": by.size(), "typical": by.median(), "p10": by.quantile(0.1),
                          "p90": by.quantile(0.9), "shortest": by.min(), "average": by.mean(), "longest": by.max()})
    return table.reindex(OBLASTS).fillna({"alerts": 0}).astype({"alerts": int})


def minutes_text(minutes):
    if pd.isna(minutes):
        return "—"
    minutes = int(round(minutes))
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"
