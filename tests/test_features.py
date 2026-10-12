import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from alerts.config import OBLASTS, kyiv_day_start_utc
from alerts.episodes import clip_as_of, daily_counts, merge_episodes
from alerts.features import (FEATURES, active_cube, blended_chance, forecast_chance, hourly_chance, make_rows,
                             scale_chance, slot_chance, tg_daily)
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

        full_cube, cut_cube = active_cube(full_eps, days), active_cube(cut_eps, days)
        for oblast in ("Kyivska oblast", "Lvivska oblast"):
            np.testing.assert_array_equal(hourly_chance(full_cube, days, oblast, t0),
                                          hourly_chance(cut_cube, days, oblast, t0))
            self.assertEqual(slot_chance(full_cube, days, oblast, t0, [23, 0]),
                             slot_chance(cut_cube, days, oblast, t0, [23, 0]))
            np.testing.assert_array_equal(blended_chance(full_cube, days, oblast, t0),
                                          blended_chance(cut_cube, days, oblast, t0))


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


class NeighboursTest(unittest.TestCase):
    def test_table_is_complete_and_symmetric(self):
        from alerts.features import NEIGHBOURS
        self.assertEqual(set(NEIGHBOURS), set(OBLASTS))
        for oblast, neighbours in NEIGHBOURS.items():
            self.assertNotIn(oblast, neighbours)
            for other in neighbours:
                self.assertIn(oblast, NEIGHBOURS[other], f"{oblast} -> {other} is not mutual")
        self.assertEqual(NEIGHBOURS["Kyiv City"], ["Kyivska oblast"])

    def test_neighbour_features(self):
        days = pd.date_range("2026-03-01", "2026-04-30")
        daily = pd.DataFrame(0.0, index=days, columns=OBLASTS)
        daily.loc["2026-04-09", "Kyiv City"] = 6       # both border Kyivska oblast (6 neighbours in total)
        daily.loc["2026-04-09", "Zhytomyrska oblast"] = 4
        tg = tg_daily(chomu_flags(pd.DataFrame({"time": [], "text": []})), days)
        row = make_rows(daily, tg, [pd.Timestamp("2026-04-10")], leads=[1])
        kyivska = row[row["oblast"] == OBLASTS.index("Kyivska oblast")].iloc[0]
        self.assertAlmostEqual(kyivska["nb_1d"], 10 / 6)
        self.assertAlmostEqual(kyivska["nb_7d"], (10 / 7) / 6)


def episodes_at(rows):
    return pd.DataFrame(rows, columns=["oblast", "start", "end"]).assign(
        start=lambda d: pd.to_datetime(d["start"]), end=lambda d: pd.to_datetime(d["end"]))


class ActiveCubeTest(unittest.TestCase):
    def test_marks_every_kyiv_hour_an_alert_touches(self):
        days = pd.date_range("2026-07-01", "2026-07-03")
        cube = active_cube(episodes_at([
            ("Kyiv City", "2026-07-01 21:30", "2026-07-01 23:10"),   # 00:30-02:10 Kyiv on 2 July
            ("Kyiv City", "2026-07-02 07:00", "2026-07-02 07:00"),   # zero length, 10:00 Kyiv
        ]), days)
        kyiv = OBLASTS.index("Kyiv City")
        self.assertEqual(list(np.flatnonzero(cube[kyiv, 1])), [0, 1, 2, 10])
        self.assertFalse(cube[kyiv, 0].any())
        self.assertEqual(int(cube.sum()), 4)


class ChanceTest(unittest.TestCase):
    def setUp(self):
        self.days = pd.date_range("2026-03-01", periods=40)
        self.cube = np.zeros((len(OBLASTS), 40, 24), dtype=bool)
        self.lviv = OBLASTS.index("Lvivska oblast")

    def test_hourly_chance_is_share_of_recent_days(self):
        self.cube[self.lviv, [5, 10, 12, 15, 20, 25, 29], 22] = True   # 7 of the 28 days before day 30
        self.cube[self.lviv, 30, 22] = True                             # the forecast day itself: ignored
        chance = hourly_chance(self.cube, self.days, "Lvivska oblast", self.days[30])
        self.assertAlmostEqual(chance[22], 0.25)
        self.assertEqual(chance[3], 0.0)

    def test_slot_chance_counts_days_with_any_alert_in_the_slot(self):
        self.cube[self.lviv, 20, 23] = True
        self.cube[self.lviv, 21, 0] = True
        self.cube[self.lviv, 22, [23, 0]] = True   # both hours on one day counts once
        self.assertAlmostEqual(slot_chance(self.cube, self.days, "Lvivska oblast", self.days[30], [23, 0]), 3 / 28)

    def test_scale_chance(self):
        self.assertAlmostEqual(scale_chance(np.array([0.5]), 2.0)[0], 0.75)   # twice as busy: 1 - 0.5**2
        self.assertAlmostEqual(scale_chance(np.array([0.5]), 1.0)[0], 0.5)

    def test_blended_chance_is_half_pattern_half_flat_over_56_days(self):
        days = pd.date_range("2026-01-01", periods=70)
        cube = np.zeros((len(OBLASTS), 70, 24), dtype=bool)
        lviv = OBLASTS.index("Lvivska oblast")
        cube[lviv, 60 - 56:60:2, 19] = True                          # hour 19 on 28 of the 56 days before day 60
        cube[lviv, 0:4, 3] = True                                     # older than 56 days: ignored
        chance = blended_chance(cube, days, "Lvivska oblast", days[60])
        self.assertAlmostEqual(chance[19], 0.5 * 0.5 + 0.5 * (0.5 / 24))
        self.assertAlmostEqual(chance[3], 0.5 * (0.5 / 24))

    def test_count_scaling_only_for_one_day_ahead(self):
        chance = np.array([0.5])
        self.assertAlmostEqual(forecast_chance(chance, lead=1, pred=20.0, cnt_28d=10.0)[0], 0.75)
        self.assertAlmostEqual(forecast_chance(chance, lead=2, pred=20.0, cnt_28d=10.0)[0], 0.5)
