import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alerts.report import figure_html, heatmap, line_chart, table_html, write_report


class ReportTest(unittest.TestCase):
    def test_write_report(self):
        fig = heatmap(pd.DataFrame([[0.1, 0.9]], index=["Kyiv City"], columns=[0, 1]), "t", "share")
        image = figure_html(fig)
        self.assertTrue(image.startswith('<img src="data:image/png;base64,'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.html"
            write_report(path, "Report", [("Section", image + table_html(pd.DataFrame({"a": [1.234]})))])
            text = path.read_text(encoding="utf-8")
        self.assertIn("<h2>Section</h2>", text)
        self.assertIn("1.23", text)

    def test_line_chart_labels_every_series(self):
        table = pd.DataFrame({"model": [2.1, 2.5], "b1": [1.9, 2.2], "b2": [2.5, 2.5]}, index=[1, 2])
        fig = line_chart(table, "MAE", "days ahead", "alerts/day")
        ax = fig.axes[0]
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ["model", "b1", "b2"])
        self.assertEqual(sorted(t.get_text() for t in ax.texts), ["b1", "b2", "model"])  # direct end labels
        figure_html(fig)  # closes the figure
