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
    # Leave out inputs with no known value at all: newer scikit-learn cannot bin an all-empty column.
    features = [name for name in FEATURES if data[name].notna().any()]
    model = HistGradientBoostingRegressor(loss="poisson", random_state=0,
                                          categorical_features=[name == "oblast" for name in features])
    model.fit(data[features], data["actual"])
    model.features_ = features
    return model


def predict(model, rows):
    return model.predict(rows[model.features_])
