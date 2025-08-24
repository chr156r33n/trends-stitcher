from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from stitcher import stitch_terms


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


PAYLOAD = {
    "interest_over_time": {
        "timeline_data": [
            {"time": "2024-01-01", "value": [1, 2]},
            {"time": "2024-01-02", "value": [3, 4]},
        ]
    }
}


def setup_mocks(monkeypatch, calls):
    monkeypatch.setattr(
        "stitcher.requests.get", lambda *args, **kwargs: DummyResponse(PAYLOAD)
    )
    monkeypatch.setattr("stitcher.os.makedirs", lambda *a, **k: None)

    def fake_load_cache(path):
        calls["load"] += 1
        return None

    def fake_save_cache(path, data):
        calls["save"] += 1

    monkeypatch.setattr("stitcher._load_cache", fake_load_cache)
    monkeypatch.setattr("stitcher._save_cache", fake_save_cache)


def test_use_cache_true_invokes_cache(monkeypatch):
    calls = {"load": 0, "save": 0}
    setup_mocks(monkeypatch, calls)
    stitch_terms(
        serpapi_key="dummy",
        terms=["nike", "adidas"],
        use_cache=True,
        verbose=False,
        sleep_ms=0,
    )
    assert calls["load"] == 1
    assert calls["save"] == 1


def test_use_cache_false_bypasses_cache(monkeypatch):
    calls = {"load": 0, "save": 0}
    setup_mocks(monkeypatch, calls)
    stitch_terms(
        serpapi_key="dummy",
        terms=["nike", "adidas"],
        use_cache=False,
        verbose=False,
        sleep_ms=0,
    )
    assert calls["load"] == 0
    assert calls["save"] == 0
