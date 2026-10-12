import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alerts.parse import (TG_COLUMNS, air_alert_events, air_alert_intervals, build_region_map,
                          chomu_flags, clean_chomu, load_region_map, parse_tg_page, read_posts)


def post(pid, iso, inner=None):
    text = f'<div class="tgme_widget_message_text js-message_text" dir="auto">{inner}</div>' if inner else ""
    return (f'<div class="tgme_widget_message_wrap js-widget_message_wrap">'
            f'<div class="tgme_widget_message" data-post="air_alert_ua/{pid}">{text}'
            f'<a class="tgme_widget_message_date"><time datetime="{iso}" class="time">x</time></a></div></div>')


START = '<i class="emoji"><b>🔴</b></i> <b>08:44 Повітряна тривога в Харківський район</b><br/>Слідкуйте за подальшими повідомленнями.<br/><a href="?q=x">#Харківський_район</a>'
END = '<b><i class="emoji"><b>🟢</b></i> 09:30 Відбій тривоги в Харківський район.</b><br/>Слідкуйте за подальшими повідомленнями.'
BULLETS = '<i class="emoji"><b>🔴</b></i> <b>08:54 Повітряна тривога в <br/>• Кам’янський район<br/>• Невідомий район</b><br/>Слідкуйте за подальшими повідомленнями.'
THREAT = '🔴🔴 08:38 Загроза ударних БпЛА в м. Запоріжжя та Запорізька територіальна громада&#33;'

REGION_MAP = {"Харківський район": "Kharkivska oblast", "Кам'янський район": "Dnipropetrovska oblast"}


class TelegramPageTest(unittest.TestCase):
    def test_parse_page(self):
        posts = parse_tg_page(post(10, "2026-09-07T05:44:06+00:00", START))
        self.assertEqual(posts[0]["id"], 10)
        self.assertEqual(posts[0]["time"], pd.Timestamp("2026-09-07 05:44:06"))
        self.assertTrue(posts[0]["text"].startswith("🔴 08:44 Повітряна тривога в Харківський район\n"))

    def test_parse_media_only_post(self):
        posts = parse_tg_page(post(11, "2026-09-07T05:44:06+00:00"))
        self.assertEqual(posts, [{"id": 11, "time": pd.Timestamp("2026-09-07 05:44:06"), "text": ""}])


class AirAlertTest(unittest.TestCase):
    def posts(self, items):
        rows = [p for pid, iso, inner in items for p in parse_tg_page(post(pid, iso, inner))]
        return pd.DataFrame(rows)

    def test_events_single_bullets_and_threats(self):
        events = air_alert_events(self.posts([
            (1, "2026-09-07T05:44:00+00:00", START),
            (2, "2026-09-07T05:50:00+00:00", BULLETS),
            (3, "2026-09-07T05:51:00+00:00", THREAT),
            (4, "2026-09-07T06:30:00+00:00", END),
        ]))
        self.assertEqual(list(events["kind"]), ["start", "start", "start", "end"])
        self.assertEqual(list(events["region"]),
                         ["Харківський район", "Кам'янський район", "Невідомий район", "Харківський район"])

    def test_air_alert_intervals_skip_unusable(self):
        events = air_alert_events(self.posts([
            (1, "2026-09-07T05:00:00+00:00", END),      # end without start -> ignored
            (2, "2026-09-07T05:44:00+00:00", START),
            (3, "2026-09-07T05:50:00+00:00", BULLETS),  # Кам'янський never ends, Невідомий not in map
            (4, "2026-09-07T06:30:00+00:00", END),
        ]))
        intervals = air_alert_intervals(events, REGION_MAP)
        self.assertEqual(len(intervals), 1)
        row = intervals.iloc[0]
        self.assertEqual((row["oblast"], row["start"], row["end"], row["source"]),
                         ("Kharkivska oblast", pd.Timestamp("2026-09-07 05:44"),
                          pd.Timestamp("2026-09-07 06:30"), "air_alert_ua"))


class RenamedRaionTest(unittest.TestCase):
    def test_old_raion_name_still_maps(self):
        # Новомосковський район was renamed (Самарівський) in 2024; the channel kept the old name.
        start = START.replace("Харківський район", "Новомосковський район")
        end = END.replace("Харківський район", "Новомосковський район")
        posts = pd.DataFrame(parse_tg_page(post(1, "2026-09-01T10:00:00+00:00", start))
                             + parse_tg_page(post(2, "2026-09-01T10:30:00+00:00", end)))
        intervals = air_alert_intervals(air_alert_events(posts), {})
        self.assertEqual(list(intervals["oblast"]), ["Dnipropetrovska oblast"])


class RegionMapTest(unittest.TestCase):
    def test_build_region_map(self):
        uk = pd.DataFrame({
            "oblast": ["Харківська область", "Дніпропетровська область", "Луганська область",
                       "Чернівецька область", "Одеська область"],
            "raion": ["Харківський район", "Кам’янський район", "Щастинський район", None, None],
            "hromada": [None, None, None, "Тестова територіальна громада", "Тестова територіальна громада"],
        })
        mapping = build_region_map(uk)
        self.assertEqual(mapping["Харківський район"], "Kharkivska oblast")
        self.assertEqual(mapping["Кам'янський район"], "Dnipropetrovska oblast")  # apostrophe normalised
        self.assertEqual(mapping["Харківська область"], "Kharkivska oblast")
        self.assertNotIn("Щастинський район", mapping)                     # Luhansk excluded
        self.assertNotIn("Тестова територіальна громада", mapping)          # ambiguous -> dropped

    def test_unknown_oblast_name_fails_loudly(self):
        with self.assertRaises(ValueError):
            build_region_map(pd.DataFrame({"oblast": ["Невідома область"], "raion": [None], "hromada": [None]}))


class FilesTest(unittest.TestCase):
    def test_read_posts_and_region_map_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            posts_path, map_path = Path(tmp) / "posts.csv", Path(tmp) / "map.csv"
            pd.DataFrame({"id": [1, 2], "time": ["2026-09-07 05:44:06", "2026-09-07 06:00:00"],
                          "text": ["рядок 1\nрядок 2", None]}).to_csv(posts_path, index=False)
            pd.DataFrame({"name": ["Харківський район"], "oblast": ["Kharkivska oblast"]}).to_csv(map_path, index=False)
            posts = read_posts(posts_path)
            mapping = load_region_map(map_path)
        self.assertEqual(posts.loc[0, "time"], pd.Timestamp("2026-09-07 05:44:06"))
        self.assertEqual(list(posts["text"]), ["рядок 1\nрядок 2", ""])  # multi-line kept, empty -> ""
        self.assertEqual(mapping, {"Харківський район": "Kharkivska oblast"})


class ChomuTest(unittest.TestCase):
    def test_clean_and_flags(self):
        text = "❗️Реактивний Шахед на Київщину\n\nПідтримати канал☕️\n\n✅️ Чому тривога?"
        self.assertEqual(clean_chomu(text), "❗️реактивний шахед на київщину")
        posts = pd.DataFrame({"time": pd.to_datetime(["2026-09-01 10:00", "2026-09-01 11:00"]),
                              "text": [text, "Балістика на Київ!"]})
        flags = chomu_flags(posts)
        self.assertEqual(list(flags.columns), ["time"] + TG_COLUMNS)
        self.assertEqual(list(flags["m_Kyivska oblast"]), [1, 0])
        self.assertEqual(list(flags["m_Kyiv City"]), [0, 1])
        self.assertEqual(list(flags["drones"]), [1, 0])
        self.assertEqual(list(flags["missiles"]), [0, 1])

    def test_city_names_in_other_grammatical_cases(self):
        # Ukrainian city stems change in other cases: Харків -> Харкова, Львів -> Львові, ...
        cases = {"шахеди біля Харкова": "Kharkivska oblast", "вибухи у Львові": "Lvivska oblast",
                 "дрон на Миколаєва": "Mykolaivska oblast", "над Черніговом": "Chernihivska oblast",
                 "в Тернополі": "Ternopilska oblast", "дрон над Сумами": "Sumska oblast"}
        flags = chomu_flags(pd.DataFrame({"time": pd.to_datetime(["2026-09-01"] * len(cases)),
                                          "text": list(cases)}))
        for i, oblast in enumerate(cases.values()):
            self.assertEqual(flags.loc[i, f"m_{oblast}"], 1, list(cases)[i])


class RedactTest(unittest.TestCase):
    def test_redact_removes_payment_and_personal_details(self):
        from alerts.parse import redact
        text = ("Підтримати канал: 4441 1144 2233 4455, https://send.monobank.ua/jar/abc, адмін @some_admin, "
                "пошта info@example.gov.ua\n❗️Шахед на Київщину")
        out = redact(text)
        for secret in ("4441", "monobank", "@some_admin", "info@example"):
            self.assertNotIn(secret, out)
        self.assertIn("Шахед на Київщину", out)

    def test_redact_keeps_keyword_flags(self):
        from alerts.parse import redact
        texts = ["Шахед на Київщину https://t.me/x 4441114422334455", "Балістика на Київ! @user"]
        posts = pd.DataFrame({"time": pd.to_datetime(["2026-09-01 10:00"] * 2), "text": texts})
        pd.testing.assert_frame_equal(chomu_flags(posts), chomu_flags(posts.assign(text=posts["text"].map(redact))))
