"""Read and clean every data source into simple pandas tables."""
import html
import re

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


# ---------------------------------------------------------------- Telegram pages
_POST_ID = re.compile(r'data-post="[^/"]+/(\d+)"')
_TIME = re.compile(r'<time datetime="([^"]+)"')
_TEXT = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)


def html_to_text(fragment):
    text = re.sub(r"<br\s*/?>", "\n", fragment)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def parse_tg_page(page_html):
    """One t.me/s/<channel> page -> posts (oldest first). Media-only posts get text ''."""
    posts = []
    for block in page_html.split("tgme_widget_message_wrap")[1:]:
        post_id, time = _POST_ID.search(block), _TIME.search(block)
        if not (post_id and time):
            continue
        text = _TEXT.search(block)
        posts.append({
            "id": int(post_id.group(1)),
            "time": pd.Timestamp(time.group(1)).tz_convert("UTC").tz_localize(None),
            "text": html_to_text(text.group(1)) if text else "",
        })
    return posts


def read_posts(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df["text"] = df["text"].fillna("")
    return df


# ---------------------------------------------------------------- air_alert_ua
OBLAST_UK = {
    "Вінницька область": "Vinnytska oblast", "Волинська область": "Volynska oblast",
    "Дніпропетровська область": "Dnipropetrovska oblast", "Донецька область": "Donetska oblast",
    "Житомирська область": "Zhytomyrska oblast", "Закарпатська область": "Zakarpatska oblast",
    "Запорізька область": "Zaporizka oblast", "Івано-Франківська область": "Ivano-Frankivska oblast",
    "Київська область": "Kyivska oblast", "м. Київ": "Kyiv City", "Київ": "Kyiv City",
    "Кіровоградська область": "Kirovohradska oblast", "Луганська область": None,
    "Львівська область": "Lvivska oblast", "Миколаївська область": "Mykolaivska oblast",
    "Одеська область": "Odeska oblast", "Полтавська область": "Poltavska oblast",
    "Рівненська область": "Rivnenska oblast", "Сумська область": "Sumska oblast",
    "Тернопільська область": "Ternopilska oblast", "Харківська область": "Kharkivska oblast",
    "Херсонська область": "Khersonska oblast", "Хмельницька область": "Khmelnytska oblast",
    "Черкаська область": "Cherkaska oblast", "Чернівецька область": "Chernivetska oblast",
    "Чернігівська область": "Chernihivska oblast", "Автономна Республіка Крим": None,
}


def norm_name(name):
    for apostrophe in "’ʼ`":
        name = name.replace(apostrophe, "'")
    return name.strip().rstrip(".!").strip()


def build_region_map(uk):
    """official_data_uk.csv rows -> {Ukrainian oblast/raion/hromada name: English oblast}.
    Names that belong to more than one oblast are dropped (ambiguous)."""
    unknown = sorted(set(uk["oblast"].dropna()) - set(OBLAST_UK))
    if unknown:
        raise ValueError(f"Unknown oblast names in official_data_uk.csv, add them to OBLAST_UK: {unknown}")
    owners = {}
    for column in ("oblast", "raion", "hromada"):
        for name, oblast_uk in zip(uk[column], uk["oblast"]):
            oblast = OBLAST_UK[oblast_uk]
            if isinstance(name, str) and oblast:
                owners.setdefault(norm_name(name), set()).add(oblast)
    return {name: next(iter(oblasts)) for name, oblasts in owners.items() if len(oblasts) == 1}


def load_region_map(path):
    df = pd.read_csv(path)
    return dict(zip(df["name"], df["oblast"]))


_ALERT_START, _ALERT_END = "Повітряна тривога в", "Відбій тривоги в"

# Old raion names the channel still used after official renaming (the CSV only has the new names).
RENAMED_RAIONS = {"Новомосковський район": "Dnipropetrovska oblast"}  # now Самарівський район


def air_alert_events(posts):
    """Posts -> one row per (time, 'start'/'end', region). Threat-only posts are ignored."""
    rows = []
    for time, text in zip(posts["time"], posts["text"]):
        lines = text.split("\n")
        first = lines[0]
        if first.startswith("🔴") and _ALERT_START in first:
            kind, marker = "start", _ALERT_START
        elif first.startswith("🟢") and _ALERT_END in first:
            kind, marker = "end", _ALERT_END
        else:
            continue
        rest = first.split(marker, 1)[1].strip()
        regions = [rest] if rest else []
        if not regions:  # bullet list on the following lines
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("•"):
                    regions.append(line.lstrip("•").strip())
                elif regions:
                    break
        rows.extend((time, kind, norm_name(region)) for region in regions)
    return pd.DataFrame(rows, columns=["time", "kind", "region"])


def air_alert_intervals(events, region_map):
    """Pair each region's start with its next end. Unpaired or unmapped regions are skipped."""
    open_alerts, rows = {}, []
    for time, kind, region in events.sort_values("time", kind="stable").itertuples(index=False):
        if kind == "start":
            open_alerts.setdefault(region, time)
        elif region in open_alerts:
            rows.append((region, open_alerts.pop(region), time))
    df = pd.DataFrame(rows, columns=["region", "start", "end"])
    df["oblast"] = df["region"].map({**RENAMED_RAIONS, **region_map})
    df = df.dropna(subset=["oblast"]).reset_index(drop=True)
    df["source"] = "air_alert_ua"
    return df[["oblast", "start", "end", "source"]]


# ---------------------------------------------------------------- tryvoga_chomu
# Lower-case regexes. Prototype keyword matching: some noise is expected (e.g. "дніпр" = city and river).
# City stems change in other grammatical cases (Харків -> Харкова, Львів -> Львові), so both stems are listed.
OBLAST_PATTERNS = {
    "Cherkaska oblast": r"черкас",
    "Chernihivska oblast": r"чернігів|чернігов",
    "Chernivetska oblast": r"чернівец|чернівеч|чернівц|буковин",
    "Dnipropetrovska oblast": r"дніпр|кривий ріг|кривого рогу|кривому розі|павлоград|нікопол|кам'янськ",
    "Donetska oblast": r"донеч|донецьк|краматорськ|слов'янськ|покровськ|костянтинівк|дружківк",
    "Ivano-Frankivska oblast": r"франків|прикарпат",
    "Kharkivska oblast": r"харків|харков|куп'янськ|ізюм|чугуїв",
    "Khersonska oblast": r"херсон",
    "Khmelnytska oblast": r"хмельниц|хмельнич|кам'янець|шепетівк|старокостянтин",
    "Kirovohradska oblast": r"кіровоград|кропивниц",
    "Kyiv City": r"\bкиїв\b|\bкиєв|столиц",
    "Kyivska oblast": r"київщин|бровар|бориспіл|біла церкв|білої церкв|вишгород|обухів|фастів",
    "Lvivska oblast": r"львів|львов",
    "Mykolaivska oblast": r"миколаїв|миколаєв",
    "Odeska oblast": r"одес|ізмаїл",
    "Poltavska oblast": r"полтав|кременчук|миргород",
    "Rivnenska oblast": r"рівненщ|рівненськ|\bрівне\b|\bрівного\b|\bрівному\b",
    "Sumska oblast": r"сумщин|\bсуми\b|\bсумах\b|\bсумам|сумськ|шостк|конотоп|охтирк",
    "Ternopilska oblast": r"тернопіл|тернопол",
    "Vinnytska oblast": r"вінниц|вінничч|козятин",
    "Volynska oblast": r"волин|луцьк|ковел",
    "Zakarpatska oblast": r"закарпат|ужгород|мукачев",
    "Zaporizka oblast": r"запоріж|запоріз",
    "Zhytomyrska oblast": r"житомир|звягел|коростен|бердичів",
}
THREAT_PATTERNS = {
    "drones": r"шахед|бпла|дрон|герань|гербер",
    "missiles": r"ракет|балістик|калібр|кинджал|іскандер|циркон",
    "aviation": r"міг-31|ту-95|ту-22|ту-160|су-34|су-35|авіабомб|стратегічн",
    "massive": r"масован|масштабн|комбінован",
}
TG_COLUMNS = [f"m_{oblast}" for oblast in OBLASTS] + list(THREAT_PATTERNS)


def clean_chomu(text):
    """Drop the channel's boilerplate lines, normalise apostrophes, lower-case."""
    lines = [line for line in text.split("\n")
             if line.strip() and "Підтримати канал" not in line and "Чому тривога?" not in line]
    return norm_name("\n".join(lines)).lower()


def chomu_flags(posts):
    """One row per post: 1 if the post mentions each oblast / threat group, else 0."""
    if posts.empty:
        return pd.DataFrame(columns=["time"] + TG_COLUMNS)
    text = posts["text"].fillna("").map(clean_chomu)
    flags = {f"m_{o}": text.str.contains(OBLAST_PATTERNS[o], regex=True) for o in OBLASTS}
    flags.update({g: text.str.contains(p, regex=True) for g, p in THREAT_PATTERNS.items()})
    out = pd.DataFrame(flags).astype(int)
    out.insert(0, "time", posts["time"].to_numpy())
    return out
