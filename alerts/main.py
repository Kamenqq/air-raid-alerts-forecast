"""python3 run.py [--start YYYY-MM-DD] [--days N] [--fresh]

Without --start: full evaluation on the last 28 days + HTML report + forecast from the last data day.
With --start: only the forecast for --days days starting that Kyiv date."""
import argparse
import sys

import numpy as np
import pandas as pd

from . import config as C
from .episodes import daily_counts, episode_starts, impute_naive, merge_episodes, naive_medians
from .evaluate import backtest, bias, mae_table, naive_check, peak_hit_rate, test_period, window_coverage
from .features import hourly_profile, main_window, make_rows, tg_daily
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


def peak_hours_text(profile):
    """Top-3 most likely start hours, or '—' when the oblast had no alerts in the window (flat profile)."""
    if np.allclose(profile, profile[0]):
        return "—"
    return ", ".join(f"{h:02d}:00" for h in sorted(np.argsort(profile)[-3:]))


def window_text(window):
    """(start_hour, hours) -> '16:00–21:00'; None -> '—'."""
    if window is None:
        return "—"
    start, hours = window
    return f"{start:02d}:00–{(start + hours) % 24:02d}:00"


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
    # Target = eTryvoga (volunteer) alerts only: the one source that covers train AND test periods the same way.
    # Other sources stopped on 2026-09-07; they stay in `intervals` for the data checks in the report.
    target = intervals[intervals["source"] == "volunteer_csv"].reset_index(drop=True)
    return {"intervals": intervals, "target": target, "vol": vol, "medians": medians,
            "chomu": read_posts(folder / "tryvoga_chomu_posts.csv")}


def build(inputs):
    episodes = merge_episodes(inputs["target"])
    data_end = inputs["target"]["start"].max()
    end_day = C.kyiv_day(data_end)
    days = pd.date_range(C.kyiv_day(C.DATA_START), end_day + pd.Timedelta(days=C.MAX_LEAD + 1))
    daily = daily_counts(episodes, days, data_end)
    tg = tg_daily(chomu_flags(inputs["chomu"]), days)
    return {"episodes": episodes, "starts": episode_starts(episodes), "daily": daily,
            "rows": make_rows(daily, tg, days), "end_day": end_day}


def forecast(state, start, n_days):
    check_start(start, state["end_day"])
    model = train(state["rows"], before_day=start)
    rows = state["rows"]
    rows = rows[(rows["t0"] == start) & (rows["lead_days"] <= n_days)].copy()
    rows["expected"] = predict(model, rows)
    profiles = {o: hourly_profile(state["starts"], o, start) for o in C.OBLASTS}
    records = []
    for row in rows.itertuples():
        oblast = C.OBLASTS[row.oblast]
        profile = profiles[oblast]
        record = {"oblast": oblast, "day": row.target_day.date(), "expected": row.expected,
                  "actual": row.actual, "peak_hours": peak_hours_text(profile),
                  "main_window": window_text(main_window(profile))}
        record.update({f"h{h:02d}": row.expected * profile[h] for h in range(24)})
        records.append(record)
    return pd.DataFrame(records)


def print_forecast(fc):
    cells = fc.assign(cell=fc["expected"].round(1).astype(str) + " | " + fc["main_window"])
    table = cells.pivot(index="oblast", columns="day", values="cell")
    with pd.option_context("display.width", 250, "display.max_columns", 10):
        print("Expected alerts per day | hours expected to hold half of them (Kyiv time):")
        print(table.to_string())


def save_forecast(fc, start):
    C.REPORTS_DIR.mkdir(exist_ok=True)
    path = C.REPORTS_DIR / f"forecast_{start.date()}.csv"
    fc.round(3).to_csv(path, index=False)
    return path


def run_evaluation(inputs, state):
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
    peak = peak_hit_rate(starts, days)
    model_bias = bias(bt)
    coverage, window_hours = window_coverage(starts, days)
    summary = (f"On {days[0].date()}–{days[-1].date()} the model misses by {overall.loc[0, 'model']:.2f} alerts per "
               f"oblast-day on average (best baseline: {overall.loc[0, ['b1', 'b2']].min():.2f}; skill "
               f"{round(overall.loc[0, 'skill_vs_best'], 2) or 0.0:+.0%}). Bias {model_bias:+.2f} alerts/day. "
               f"{peak:.0%} of alerts started in the predicted top-3 hours (random guess: 12.5%). "
               f"The predicted 'half of the alerts' windows ({window_hours:.1f} h long on average) "
               f"held {coverage:.0%} of real alerts (they promise 50%; a randomly placed window of the same "
               f"length would hold about {window_hours / 24:.0%}).")
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
    fc = forecast(state, fc_start, C.MAX_LEAD)
    save_forecast(fc, fc_start)
    fc_table = fc.pivot(index="oblast", columns="day", values="expected")

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
         + table_html(by_oblast)),
        (f"5. Forecast from {fc_start.date()}", figure_html(heatmap(fc_table, "Expected alerts per day", "alerts"))
         + table_html(fc[["oblast", "day", "expected", "main_window", "peak_hours"]].set_index(["oblast", "day"]), 1)),
    ]
    write_report(C.REPORTS_DIR / "report.html", "Air raid alerts forecast — evaluation", sections)
    print_forecast(fc)
    print(f"Report: {C.REPORTS_DIR / 'report.html'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Forecast air raid alerts per oblast and day (1-7 days ahead).")
    parser.add_argument("--start", help="first forecast day (Kyiv), YYYY-MM-DD. Without it: evaluation + report.")
    parser.add_argument("--days", type=int, default=C.MAX_LEAD, choices=range(1, C.MAX_LEAD + 1), metavar="1-7")
    parser.add_argument("--fresh", action="store_true", help="download the latest data first (slow)")
    args = parser.parse_args(argv)
    make_console_safe()
    start = parse_start(args.start) if args.start is not None else None  # "" is an invalid date, not "absent"
    inputs = load_inputs(args.fresh)
    state = build(inputs)
    if start is None:
        run_evaluation(inputs, state)
    else:
        fc = forecast(state, start, args.days)
        print_forecast(fc)
        print(f"Saved: {save_forecast(fc, start)}")


if __name__ == "__main__":
    main()
