"""Read and clean every data source into simple pandas tables."""
import pandas as pd

from .config import OBLASTS


def _utc(column):
    return pd.to_datetime(column, utc=True).dt.tz_localize(None).to_numpy()


def load_official(path):
    """official_data_en.csv -> one row per alert interval (any level: oblast, raion, hromada)."""
    df = pd.read_csv(path).drop_duplicates()
    df = df[df["oblast"].isin(OBLASTS)]
    return pd.DataFrame({
        "oblast": df["oblast"].to_numpy(),
        "start": _utc(df["started_at"]),
        "end": _utc(df["finished_at"]),
        "source": "official_csv",
    })


def load_volunteer(path):
    """volunteer_data_en.csv (eTryvoga) -> one row per oblast alert, with the naive flag."""
    df = pd.read_csv(path).drop_duplicates()
    df = df[df["region"].isin(OBLASTS)]
    return pd.DataFrame({
        "oblast": df["region"].to_numpy(),
        "start": _utc(df["started_at"]),
        "end": _utc(df["finished_at"]),
        "naive": df["naive"].astype(str).str.strip().eq("True").to_numpy(),
    })
