"""python3 run.py [--start YYYY-MM-DD] [--days N] [--fresh]

Without --start: full evaluation on the last 28 days + HTML report + forecast from the last data day.
With --start: only the forecast for --days days starting that Kyiv date."""
import argparse
import re
import sys

import numpy as np
import pandas as pd

from . import config as C
from .episodes import daily_counts, episode_starts, impute_naive, merge_episodes, naive_medians
from .evaluate import backtest, bias, calibration, chance_scores, mae_table, naive_check, test_period
from .features import active_cube, blended_chance, blended_slot_chances, forecast_chance, make_rows, tg_daily
from .model import predict, train
from .parse import (air_alert_events, air_alert_intervals, chomu_flags, load_official, load_region_map,
                    load_volunteer, read_posts)
from .report import bar_chart, figure_html, heatmap, line_chart, table_html, write_report


def make_console_safe():
    """Some Windows consoles use an 8-bit encoding (e.g. cp1252) without characters like '→':
    print '?' for those instead of crashing."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def parse_start(text):
    try:
        day = pd.to_datetime(text, format="%Y-%m-%d")
    except (ValueError, TypeError):
        day = pd.NaT
    if pd.isna(day):  # e.g. "" gives NaT instead of an error
        raise SystemExit(f"--start must be a date like 2026-10-05 (YYYY-MM-DD), got: {text!r}")
    return pd.Timestamp(day).normalize()


def check_start(start, end_day):
    """The last allowed start is the last (partly covered) data day: features then use only full days."""
    if not C.EARLIEST_START <= start <= end_day:
        raise SystemExit(f"--start must be between {C.EARLIEST_START.date()} and {end_day.date()} "
                         f"(the last day in the data).")


def parse_slot(text):
    """'08:00-09:30' -> Kyiv hours [8, 9]; '23:00-01:00' -> [23, 0] (a slot may cross midnight)."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})", text.strip())
    if match:
        h1, m1, h2, m2 = map(int, match.groups())
        if h1 < 24 and h2 < 24 and m1 < 60 and m2 < 60 and (h1, m1) != (h2, m2):
            begin, end = h1 * 60 + m1, h2 * 60 + m2
            end += 24 * 60 if end < begin else 0
            return [(minute // 60) % 24 for minute in range(begin - begin % 60, end, 60)]
    raise SystemExit(f"--slot must look like 08:00-09:30 (Kyiv time), got: {text!r}")


def slot_text(chances, hour):
    return f"{hour:02d}:00–{(hour + 2) % 24:02d}:00 {chances[hour]:.0%}"


def skill_text(skill):
    """'+12%'; tiny values print as '+0%' (not '-0%'); an undefined skill prints 'n/a'."""
    if pd.isna(skill):
        return "n/a"
    return f"{0.0 if abs(skill) < 0.005 else skill:+.0%}"


def load_inputs(fresh):
    folder = C.SNAPSHOT_DIR
    if fresh:
        from .sources import refresh  # network code only when asked for
        folder = refresh()
    official = load_official(folder / "official_data_en.csv")
    vol = load_volunteer(folder / "volunteer_data_en.csv")
    medians = naive_medians(vol)
    air_alert = air_alert_intervals(air_alert_events(read_posts(folder / "air_alert_ua_posts.csv")),
                                    load_region_map(C.REGION_MAP))
    intervals = pd.concat([official, impute_naive(vol, medians), air_alert], ignore_index=True)
    intervals = intervals[intervals["start"] >= C.DATA_START].reset_index(drop=True)
    return {"intervals": intervals, "vol": vol, "medians": medians,
            "chomu": read_posts(folder / "tryvoga_chomu_posts.csv")}


def build(inputs):
    # Target = eTryvoga (volunteer) alerts only: the one source that covers train AND test periods the same way.
    # Other sources stopped on 2026-09-07; they stay in inputs["intervals"] for the data checks in the report.
    target = inputs["intervals"][inputs["intervals"]["source"] == "volunteer_csv"]
    episodes = merge_episodes(target)
    data_end = target["start"].max()
    end_day = C.kyiv_day(data_end)
    days = pd.date_range(C.kyiv_day(C.DATA_START), end_day + pd.Timedelta(days=C.MAX_LEAD + 1))
    daily = daily_counts(episodes, days, data_end)
    tg = tg_daily(chomu_flags(inputs["chomu"]), days)
    return {"episodes": episodes, "starts": episode_starts(episodes), "daily": daily, "days": days,
            "cube": active_cube(episodes, days), "rows": make_rows(daily, tg, days), "end_day": end_day}


def forecast(state, start, n_days, slot=None):
    """Expected alerts per oblast and day, plus the chance of an alert in each Kyiv hour, the quietest
    and riskiest 2-hour slots, and (with --slot) the chance of an alert during the given slot."""
    check_start(start, state["end_day"])
    model = train(state["rows"], before_day=start)
    rows = state["rows"]
    rows = rows[(rows["t0"] == start) & (rows["lead_days"] <= n_days)].copy()
    rows["expected"] = predict(model, rows)
    cube, days = state["cube"], state["days"]
    history = {o: (blended_chance(cube, days, o, start), blended_slot_chances(cube, days, o, start, 2),
                   blended_slot_chances(cube, days, o, start, len(slot))[slot[0]] if slot else None)
               for o in C.OBLASTS}
    records = []
    for row in rows.itertuples():
        oblast = C.OBLASTS[row.oblast]
        hourly, slots, in_slot = history[oblast]
        def shown(chance):  # day 1: scaled by the predicted count; days 2-7: the history-based chance
            return forecast_chance(chance, row.lead_days, row.expected, row.cnt_28d)

        slots = shown(slots)
        quiet = (slot_text(slots, int(np.argmin(slots))), slot_text(slots, int(np.argmax(slots)))) \
            if slots.max() > 0 else ("—", "—")
        record = {"oblast": oblast, "day": row.target_day.date(), "expected": row.expected, "actual": row.actual,
                  "quietest_2h": quiet[0], "riskiest_2h": quiet[1]}
        if slot:
            record["slot_chance"] = shown(in_slot)
        record.update({f"chance_{h:02d}": c for h, c in enumerate(shown(hourly))})
        records.append(record)
    return pd.DataFrame(records)


def print_forecast(fc, slot_label=None):
    with pd.option_context("display.width", 250, "display.max_columns", 10):
        print("Expected number of alerts per day:")
        print(fc.pivot(index="oblast", columns="day", values="expected").round(1).to_string())
        first = fc[fc["day"] == fc["day"].min()].set_index("oblast")
        print(f"\nChance of an alert at some moment in a 2-hour slot, {fc['day'].min()} (Kyiv time):")
        print(first[["quietest_2h", "riskiest_2h"]].rename(
            columns={"quietest_2h": "quietest 2 hours", "riskiest_2h": "riskiest 2 hours"}).to_string())
        if "slot_chance" in fc:
            print(f"\nChance of an alert at some moment during {slot_label}:")
            percent = fc.pivot(index="oblast", columns="day", values="slot_chance").mul(100).round().astype(int)
            print(percent.astype(str).add("%").to_string())


def save_forecast(fc, start):
    C.REPORTS_DIR.mkdir(exist_ok=True)
    path = C.REPORTS_DIR / f"forecast_{start.date()}.csv"
    fc.round(3).to_csv(path, index=False, encoding="utf-8-sig")  # the BOM makes Excel read it as UTF-8
    return path


def run_evaluation(inputs, state, n_days=C.MAX_LEAD, slot=None, slot_label=None):
    C.REPORTS_DIR.mkdir(exist_ok=True)
    rows, starts, daily, intervals = state["rows"], state["starts"], state["daily"], inputs["intervals"]
    last_full_day = state["end_day"] - pd.Timedelta(days=1)
    days = test_period(last_full_day)
    explore_end = days[0] - pd.Timedelta(days=C.MAX_LEAD - 1)  # earliest test forecast time
    print(f"Backtest: {days[0].date()} to {days[-1].date()} (retraining for every forecast day)...")
    bt = backtest(rows, days)
    bt.to_csv(C.REPORTS_DIR / "backtest.csv", index=False)
    overall, by_lead = mae_table(bt), mae_table(bt, "lead_days")
    by_oblast = mae_table(bt, "oblast").rename(index=dict(enumerate(C.OBLASTS)))
    model_bias = bias(bt)
    scores = chance_scores(bt, state["cube"], state["days"])
    chosen = "shown"
    cal = calibration(bt, state["cube"], state["days"])
    summary = (f"On {days[0].date()}–{days[-1].date()} the model misses by {overall.loc[0, 'model']:.2f} alerts per "
               f"oblast-day on average (best baseline: {overall.loc[0, ['b1', 'b2']].min():.2f}; skill "
               f"{skill_text(overall.loc[0, 'skill_vs_best'])}). Bias {model_bias:+.2f} alerts/day. "
               f"Hourly alert chances, 1 day ahead: Brier score {scores.loc[1, chosen]:.3f} vs "
               f"{scores.loc[1, 'flat']:.3f} for 'the same chance every hour' (lower is better; "
               f"7 days ahead: {scores.loc[7, chosen]:.3f} vs {scores.loc[7, 'flat']:.3f}).")
    print(summary)
    labels = {"b1": "last-7-days average", "b2": "same weekday last week", "skill_vs_best": "skill vs best baseline"}
    overall, by_lead, by_oblast = (t.rename(columns=labels) for t in (overall, by_lead, by_oblast))

    train_starts = starts[(starts["day"] >= C.TRAIN_START) & (starts["day"] < explore_end)]
    h1 = pd.crosstab(train_starts["oblast"], train_starts["hour"], normalize="index")
    train_daily = daily.loc[C.TRAIN_START:explore_end - pd.Timedelta(days=1)]
    h3 = pd.DataFrame({"lag-1 autocorrelation": {o: train_daily[o].autocorr(1) for o in C.OBLASTS}})
    window = intervals[(intervals["start"] >= C.AIR_ALERT_SINCE) & (intervals["start"] < C.AIR_ALERT_UNTIL)]
    by_source = pd.crosstab(C.to_kyiv(window["start"]).dt.date, window["source"])
    sources = intervals.groupby("source")["start"].agg(["count", "min", "max"])

    fc_start = state["end_day"]
    fc = forecast(state, fc_start, n_days, slot)
    save_forecast(fc, fc_start)
    first_day = fc[fc["day"] == fc["day"].min()].set_index("oblast")
    chance_map = first_day[[f"chance_{h:02d}" for h in range(24)]].mul(100)
    chance_map.columns = [f"{h:02d}" for h in range(24)]

    sections = [
        ("Result in one paragraph", f"<p>{summary}</p>"),
        ("1. Data", "<p class='note'>The forecast target is eTryvoga (volunteer) alerts only; the other "
         "sources stopped on 2026-09-07 and are shown here as data checks.</p>" + table_html(sources) +
         "<p class='note'>Alert starts per day by source while air_alert_ua "
         "was still posting (it covers the official CSV outage of 2026-08-29…09-02 if its column is non-zero):</p>"
         + table_html(by_source)),
        ("2. Exploration (training period only)",
         figure_html(heatmap(h1, "H1: share of alert starts by Kyiv hour", "share")) +
         figure_html(bar_chart(train_daily.mean().sort_values(ascending=False),
                               "H2: mean alerts per day by oblast", "alerts/day")) + table_html(h3)),
        ("3. H6: imputing missing alert ends", table_html(
            naive_check(inputs["vol"], inputs["medians"], C.TRAIN_START, explore_end).set_index("method"))),
        ("4. Test results", table_html(overall) +
         figure_html(line_chart(by_lead[["model", labels["b1"], labels["b2"]]],
                                "Average error (MAE) by days ahead", "days ahead", "alerts per oblast-day"))
         + table_html(by_oblast)
         + "<h3>When: hourly alert chances</h3><p class='note'>Brier score = average of (chance − what happened)², "
           "what happened = 1 if an alert was active at some moment of the hour, else 0. Lower is better. "
           "<b>shown</b> = what the forecast shows: half the oblast's hourly pattern + half its flat level over the "
           "last 56 days, scaled by the predicted count for 1 day ahead (chosen on the validation period "
           "2026-08-16…09-12); <b>flat</b> = the same chance every hour; <b>hourly</b> = each hour's own "
           "28-day share.</p>" + table_html(scores, 3)
         + f"<p class='note'>Calibration ({chosen}, 1 day ahead): when we said X%, how often was there an alert?</p>"
         + table_html(cal, 2)),
        (f"5. Forecast from {fc_start.date()}",
         table_html(fc.pivot(index="oblast", columns="day", values="expected"), 1)
         + figure_html(heatmap(chance_map, f"Chance of an alert in each Kyiv hour, {fc['day'].min()} (%)", "%"))
         + table_html(first_day[["expected", "quietest_2h", "riskiest_2h"]], 1)),
    ]
    write_report(C.REPORTS_DIR / "report.html", "Air raid alerts forecast — evaluation", sections)
    print_forecast(fc, slot_label)
    print(f"Report: {C.REPORTS_DIR / 'report.html'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Forecast air raid alerts per oblast and day (1-7 days ahead).")
    parser.add_argument("--start", help="first forecast day (Kyiv), YYYY-MM-DD. Without it: evaluation + report.")
    parser.add_argument("--days", type=int, default=C.MAX_LEAD, choices=range(1, C.MAX_LEAD + 1), metavar="1-7")
    parser.add_argument("--fresh", action="store_true", help="download the latest data first (slow)")
    parser.add_argument("--slot", help="Kyiv time slot like 08:00-09:30: chance of an alert during it")
    args = parser.parse_args(argv)
    make_console_safe()
    start = parse_start(args.start) if args.start is not None else None  # "" is an invalid date, not "absent"
    slot = parse_slot(args.slot) if args.slot is not None else None
    inputs = load_inputs(args.fresh)
    state = build(inputs)
    if start is None:
        run_evaluation(inputs, state, args.days, slot, args.slot)
    else:
        fc = forecast(state, start, args.days, slot)
        print_forecast(fc, args.slot)
        print(f"Saved: {save_forecast(fc, start)}")


if __name__ == "__main__":
    main()
