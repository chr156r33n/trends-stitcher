from pathlib import Path
import sys

import pytest
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from stitcher import TrendsFetcher


class DummyResponse:
    def __init__(self):
        self.status_code = 429
        self.headers = {"X-RateLimit-Remaining": "0"}
        self.text = "rate limit exceeded"

    def json(self):
        return {}

    def raise_for_status(self):
        raise requests.HTTPError("Too Many Requests")


def test_fetch_batch_enriched_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "stitcher.requests.get", lambda *a, **k: DummyResponse()
    )
    fetcher = TrendsFetcher(
        serpapi_key="dummy",
        cache_dir=str(tmp_path),
        use_cache=False,
        sleep_ms=0,
    )
    with pytest.raises(RuntimeError) as exc:
        fetcher.fetch_batch(["nike"])
    msg = str(exc.value)
    assert "status=429" in msg
    assert "X-RateLimit-Remaining=0" in msg
    assert "rate limit exceeded" in msg
