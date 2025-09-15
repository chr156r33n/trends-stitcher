from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from stitcher import TrendsFetcher


class DummyResponse:
    def __init__(self):
        self.status_code = 200
        self.headers = {}
        self.text = ""
        # mimic requests' elapsed attribute
        class _Elapsed:
            def total_seconds(self_inner):
                return 0
        self.elapsed = _Elapsed()

    def json(self):
        # minimal structure so _parse_timeseries succeeds
        return {
            "interest_over_time": {
                "timeline_data": [
                    {
                        "time": "2024-01-01",
                        "values": [
                            {"query": "nike", "value": 1},
                            {"query": "adidas", "value": 2},
                        ],
                    }
                ]
            }
        }

    def raise_for_status(self):
        pass


def test_brightdata_payload(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        codex/fix-bright-data-api-timeouts-in-app-b26meu
        captured["timeout"] = timeout


    monkeypatch.setattr("stitcher.requests.post", fake_post)

    fetcher = TrendsFetcher(
        serpapi_key="dummy",
        provider="brightdata",
        cache_dir=str(tmp_path),
        use_cache=False,
        sleep_ms=0,
        brightdata_zone="serp_api6",
    )

    fetcher.fetch_batch(["nike", "adidas"])

    assert captured["url"] == "https://api.brightdata.com/request"
    assert captured["json"]["format"] == "raw"
    assert "brd_trends=timeseries" in captured["json"]["url"]
    assert "geo_map" not in captured["json"]["url"]
    assert captured["json"]["zone"] == "serp_api6"
    assert captured["timeout"] == 180
