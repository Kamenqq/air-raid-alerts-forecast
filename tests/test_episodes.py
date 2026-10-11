import unittest

import numpy as np
import pandas as pd

from alerts.episodes import (clip_as_of, daily_counts, episode_starts, impute_naive, merge_episodes,
                             naive_medians)

T = pd.Timestamp


def intervals(rows):
    return pd.DataFrame(rows, columns=["oblast", "start", "end"]).assign(
        start=lambda d: pd.to_datetime(d["start"]), end=lambda d: pd.to_datetime(d["end"]))


class EpisodesTest(unittest.TestCase):
    def test_merge_overlapping_and_touching(self):
        eps = merge_episodes(intervals([
            ("Kyiv City", "2026-03-01 10:00", "2026-03-01 10:30"),
            ("Kyiv City", "2026-03-01 10:20", "2026-03-01 11:00"),  # overlaps
            ("Kyiv City", "2026-03-01 11:00", "2026-03-01 11:10"),  # touches
            ("Kyiv City", "2026-03-01 12:00", "2026-03-01 12:05"),  # separate
            ("Lvivska oblast", "2026-03-01 10:10", "2026-03-01 10:20"),
        ]))
        kyiv = eps[eps["oblast"] == "Kyiv City"].sort_values("start")
        self.assertEqual(list(zip(kyiv["start"], kyiv["end"])),
                         [(T("2026-03-01 10:00"), T("2026-03-01 11:10")), (T("2026-03-01 12:00"), T("2026-03-01 12:05"))])
        self.assertEqual(int((eps["oblast"] == "Lvivska oblast").sum()), 1)

    def test_clip_as_of(self):
        clipped = clip_as_of(intervals([
            ("Kyiv City", "2026-03-01 10:00", "2026-03-01 12:00"),  # running at T -> clipped
            ("Kyiv City", "2026-03-01 11:30", "2026-03-01 11:40"),  # starts after T -> dropped
        ]), T("2026-03-01 11:00"))
        self.assertEqual(len(clipped), 1)
        self.assertEqual(clipped.iloc[0]["end"], T("2026-03-01 11:00"))

    def test_naive_imputation_uses_oblast_median(self):
        vol = intervals([
            ("Kyiv City", "2025-10-01 10:00", "2025-10-01 10:20"),
            ("Kyiv City", "2025-10-02 10:00", "2025-10-02 10:40"),
            ("Kyiv City", "2025-10-03 10:00", "2025-10-03 11:00"),
            ("Kyiv City", "2026-01-01 10:00", "2026-01-01 10:30"),  # naive
            ("Lvivska oblast", "2026-01-01 10:00", "2026-01-01 10:30"),  # naive, no median -> 30 min
        ]).assign(naive=[False, False, False, True, True])
        medians = naive_medians(vol)
        self.assertEqual(medians, {"Kyiv City": 40.0})
        out = impute_naive(vol, medians)
        self.assertEqual(out.iloc[3]["end"], T("2026-01-01 10:40"))
        self.assertEqual(out.iloc[4]["end"], T("2026-01-01 10:30"))
        self.assertEqual(out.iloc[0]["end"], T("2025-10-01 10:20"))

    def test_daily_counts_use_kyiv_days(self):
        eps = intervals([
            ("Kyiv City", "2026-07-01 21:30", "2026-07-01 22:00"),  # 00:30 Kyiv on 2 July
            ("Kyiv City", "2026-07-01 20:30", "2026-07-01 20:40"),  # 23:30 Kyiv on 1 July
        ])
        days = pd.date_range("2026-07-01", "2026-07-04")
        daily = daily_counts(eps, days, data_end=T("2026-07-03 12:00"))
        self.assertEqual(list(daily["Kyiv City"].iloc[:2]), [1.0, 1.0])
        self.assertTrue(np.isnan(daily.loc["2026-07-03", "Kyiv City"]))  # last day only partly covered
        self.assertTrue(np.isnan(daily.loc["2026-07-04", "Kyiv City"]))  # after data end
        self.assertEqual(daily.loc["2026-07-01", "Lvivska oblast"], 0.0)
        starts = episode_starts(eps)
        self.assertEqual(sorted(zip(starts["day"], starts["hour"])),
                         [(T("2026-07-01"), 23), (T("2026-07-02"), 0)])
