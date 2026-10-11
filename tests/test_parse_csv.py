import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alerts.parse import load_official, load_volunteer


def write(tmp, name, lines):
    path = Path(tmp) / name
    path.write_bytes("\r\n".join(lines).encode("utf-8") + b"\r\n")  # CRLF like the real files
    return path


class CsvTest(unittest.TestCase):
    def test_official_dedupes_and_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = "Kharkivska oblast,Kharkivskyi raion,,raion,2026-01-01 10:00:00+00:00,2026-01-01 11:00:00+00:00,official"
            path = write(tmp, "o.csv", [
                "oblast,raion,hromada,level,started_at,finished_at,source", row, row,
                "Luhanska oblast,,,oblast,2026-01-01 10:00:00+00:00,2026-01-01 11:00:00+00:00,official",
            ])
            df = load_official(path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "oblast"], "Kharkivska oblast")
        self.assertEqual(df.loc[0, "start"], pd.Timestamp("2026-01-01 10:00"))
        self.assertEqual(df.loc[0, "source"], "official_csv")

    def test_volunteer_reads_naive_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "v.csv", [
                "region,started_at,finished_at,naive",
                "Kyiv City,2026-01-01 10:00:00+00:00,2026-01-01 10:30:00+00:00,True",
                "Lvivska oblast,2026-01-02 10:00:00+00:00,2026-01-02 10:45:00+00:00,False",
            ])
            df = load_volunteer(path)
        self.assertEqual(list(df["naive"]), [True, False])
        self.assertEqual(list(df["oblast"]), ["Kyiv City", "Lvivska oblast"])
        self.assertEqual(df.loc[1, "end"], pd.Timestamp("2026-01-02 10:45"))
