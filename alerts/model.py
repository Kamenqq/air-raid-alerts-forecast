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
