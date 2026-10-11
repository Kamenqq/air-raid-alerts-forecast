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
