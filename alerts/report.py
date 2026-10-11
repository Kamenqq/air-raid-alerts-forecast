"""Charts (matplotlib, embedded as PNG) and a single self-contained HTML report.
Colours follow the dataviz reference palette: one blue ramp for magnitudes (heatmaps),
validated categorical slots (blue, orange, aqua) for the model and the two baselines."""
import base64
import html
import io

import matplotlib

matplotlib.use("Agg")  # no screen needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])  # light = near zero
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order, validated for colour-blind separation
MARKERS = ["o", "s", "^"]                   # second cue so colour is never the only one
INK, MUTED = "#1f1f1e", "#6b6a64"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)


def figure_html(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}">'


def heatmap(table, title, label):
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * table.shape[1] + 3), 0.32 * table.shape[0] + 1.5))
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap=BLUE_RAMP)
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=90 if table.shape[1] > 12 else 0)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index])
    ax.set_title(title)
    _style(ax)
    fig.colorbar(image, ax=ax, label=label)
    return fig


def bar_chart(series, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(series)), series.to_numpy(), color=SERIES[0], width=0.7)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([str(i) for i in series.index], rotation=90)
    ax.set_title(title)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return fig


def line_chart(table, title, xlabel, ylabel):
    """One line per column, colours/markers in fixed order, legend + label at each line's end."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, column in enumerate(table.columns):
        ax.plot(table.index, table[column], color=SERIES[i], marker=MARKERS[i], markersize=6,
                linewidth=2, label=str(column))
        ax.annotate(str(column), (table.index[-1], table[column].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_title(title)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=len(table.columns))  # below the plot, never on top of a line
    _style(ax)
    return fig


def table_html(df, digits=2):
    return df.round(digits).to_html(border=0)


def write_report(path, title, sections):
    body = "".join(f"<h2>{html.escape(heading)}</h2>{content}" for heading, content in sections)
    style = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;line-height:1.45;"
             "color:#1f1f1e;background:#fcfcfb}table{border-collapse:collapse;font-size:13px;margin:8px 0}"
             "td,th{border:1px solid #d9d8d2;padding:3px 6px;text-align:right}img{max-width:100%}"
             "p.note{color:#6b6a64}")
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                    f"<style>{style}</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>",
                    encoding="utf-8")
