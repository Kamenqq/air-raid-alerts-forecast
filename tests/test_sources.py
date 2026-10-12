import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from alerts import sources
from alerts.sources import fetch_channel


def page(ids, newest_time):
    """Fake t.me/s page: posts with the given ids, one hour apart, newest id = newest_time."""
    blocks = []
    for pid in sorted(ids):
        iso = (pd.Timestamp(newest_time) - pd.Timedelta(hours=max(ids) - pid)).isoformat() + "+00:00"
        blocks.append(f'<div class="tgme_widget_message_wrap"><div data-post="ch/{pid}">'
                      f'<div class="tgme_widget_message_text js-message_text">post {pid}</div>'
                      f'<time datetime="{iso}"></time></div></div>')
    return "".join(blocks)


PAGES = {
    "https://t.me/s/ch": page(range(41, 51), "2026-01-10 12:00"),
    "https://t.me/s/ch?before=41": page(range(31, 41), "2026-01-10 02:00"),
    "https://t.me/s/ch?before=31": page(range(21, 31), "2026-01-09 16:00"),
    "https://t.me/s/ch?before=21": "",
}


class FetchTest(unittest.TestCase):
    def test_fetch_until_since(self):
        posts = fetch_channel("ch", since=pd.Timestamp("2026-01-09 20:00"), delay=0, get=PAGES.__getitem__, log=lambda *_: None)
        self.assertEqual(list(posts["id"]), list(range(34, 51)))  # id p is at 2026-01-10 12:00 minus (50 - p) hours; id 34 = 2026-01-09 20:00
        self.assertTrue((posts["time"] >= pd.Timestamp("2026-01-09 20:00")).all())

    def test_fetch_stops_at_known_id(self):
        posts = fetch_channel("ch", since=pd.Timestamp("2020-01-01"), after_id=44, delay=0, get=PAGES.__getitem__, log=lambda *_: None)
        self.assertEqual(list(posts["id"]), list(range(45, 51)))

    def test_fetch_stops_on_empty_page(self):
        posts = fetch_channel("ch", since=pd.Timestamp("2020-01-01"), delay=0, get=PAGES.__getitem__, log=lambda *_: None)
        self.assertEqual(posts["id"].min(), 21)


class DownloadTest(unittest.TestCase):
    def test_download_keeps_bytes(self):
        # Text-mode writing would turn CRLF into CR CR LF on Windows; the file must be byte-identical.
        payload = "region,started_at\r\nKyiv City,2026-01-01\r\n".encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sources, "get_bytes", return_value=payload):
            dest = sources.download("https://example.invalid/x.csv", Path(tmp) / "sub" / "x.csv")
            self.assertEqual(dest.read_bytes(), payload)


class RetryTest(unittest.TestCase):
    def test_cut_off_download_is_retried(self):
        import http.client
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"ok"

        def flaky_urlopen(request, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise http.client.IncompleteRead(b"partial")
            return Response()

        with mock.patch("urllib.request.urlopen", flaky_urlopen), mock.patch("time.sleep"):
            self.assertEqual(sources.get_bytes("https://example.invalid/x"), b"ok")
        self.assertEqual(len(calls), 3)
