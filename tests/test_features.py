import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from alerts.config import OBLASTS, kyiv_day_start_utc
from alerts.episodes import clip_as_of, daily_counts, episode_starts, merge_episodes
from alerts.features import FEATURES, hourly_profile, make_rows, tg_daily
from alerts.parse import chomu_flags


def synthetic(seed=0):
    rng = np.random.RandomState(seed)
    rows, posts = [], []
    for day in pd.date_range("2026-03-01", "2026-05-09"):
        for oblast in ("Kyivska oblast", "Lvivska oblast", "Kyiv City"):
            for _ in range(rng.poisson(3)):
                start = day + pd.Timedelta(minutes=int(rng.randint(0, 24 * 60)))
                rows.append((oblast, start, start + pd.Timedelta(minutes=int(rng.randint(10, 300)))))
        for _ in range(rng.poisson(5)):
            posts.append((day + pd.Timedelta(minutes=int(rng.randint(0, 24 * 60))),
                          rng.choice(["Шахед на Київщину", "Балістика на Київ", "Ракети на Львів", "Тихо"])))
    iv = pd.DataFrame(rows, columns=["oblast", "start", "end"])
    tg = pd.DataFrame(posts, columns=["time", "text"])
    return iv, tg


class LeakageTest(unittest.TestCase):
    def test_features_ignore_everything_after_forecast_time(self):
        iv, posts = synthetic()
        days = pd.date_range("2026-03-01", "2026-05-16")
        t0 = pd.Timestamp("2026-04-10")
        T = kyiv_day_start_utc(t0)

        full_eps = merge_episodes(iv)
        cut_eps = merge_episodes(clip_as_of(iv, T))
        full_daily = daily_counts(full_eps, days, data_end=iv["start"].max())
        cut_daily = daily_counts(cut_eps, days, data_end=T)
        self.assertFalse(full_daily.loc[t0].equals(cut_daily.loc[t0]))  # the future really differs

        full = make_rows(full_daily, tg_daily(chomu_flags(posts), days), [t0])
        cut = make_rows(cut_daily, tg_daily(chomu_flags(posts[posts["time"] < T]), days), [t0])
        assert_frame_equal(full[FEATURES], cut[FEATURES])

        for oblast in ("Kyivska oblast", "Lvivska oblast"):
            np.testing.assert_array_equal(hourly_profile(episode_starts(full_eps), oblast, t0),
                                          hourly_profile(episode_starts(cut_eps), oblast, t0))


class RowsTest(unittest.TestCase):
    def test_row_values(self):
        days = pd.date_range("2026-03-01", "2026-04-30")
        daily = pd.DataFrame(0.0, index=days, columns=OBLASTS)
        daily.loc["2026-04-09", "Kyiv City"] = 7
        daily.loc["2026-04-05", "Kyiv City"] = 2
        tg = tg_daily(chomu_flags(pd.DataFrame({"time": [], "text": []})), days)
        rows = make_rows(daily, tg, [pd.Timestamp("2026-04-10")], leads=[1, 3])
        kyiv = rows[(rows["oblast"] == 10)].set_index("lead_days")
        self.assertEqual(kyiv.loc[1, "cnt_1d"], 7)
        self.assertAlmostEqual(kyiv.loc[1, "cnt_7d"], 9 / 7)
        self.assertEqual(kyiv.loc[3, "target_day"], pd.Timestamp("2026-04-12"))
        self.assertEqual(kyiv.loc[3, "cnt_same_wd"], 2)  # 2026-04-12 minus 7 days = 2026-04-05
        self.assertEqual(kyiv.loc[1, "nat_1d"], 7)
        self.assertTrue(np.isnan(kyiv.loc[1, "tg_mention_1d"]))  # no Telegram coverage -> unknown


class ProfileTest(unittest.TestCase):
    def test_profile_counts_last_28_days(self):
        starts = pd.DataFrame({"oblast": ["Kyiv City"] * 3, "hour": [22, 22, 3],
                               "day": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-10"])})
        profile = hourly_profile(starts, "Kyiv City", pd.Timestamp("2026-04-10"))
        self.assertEqual(profile[22], 1.0)  # the 04-10 start is not before t0
        self.assertAlmostEqual(profile.sum(), 1.0)

    def test_profile_without_history_is_uniform(self):
        starts = pd.DataFrame({"oblast": [], "hour": [], "day": pd.to_datetime([])})
        profile = hourly_profile(starts, "Zakarpatska oblast", pd.Timestamp("2026-04-10"))
        np.testing.assert_allclose(profile, np.full(24, 1 / 24))
