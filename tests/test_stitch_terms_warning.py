from pathlib import Path
import sys
from types import ModuleType

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from stitcher import TrendsFetcher, stitch_terms


def test_stitch_terms_warns_on_empty_pairwise(monkeypatch):
    # Stub out streamlit to capture warnings
    record = {}
    stub = ModuleType("streamlit")

    def warning(msg):
        record["msg"] = msg

    stub.warning = warning
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    # Return all-zero values so pairwise_ratios is empty
    def fake_fetch_batch(self, batch):
        rows = [
            {"date": "2024-01-01", "term": t, "value": 0}
            for t in batch
        ]
        return pd.DataFrame(rows)

    monkeypatch.setattr(TrendsFetcher, "fetch_batch", fake_fetch_batch)

    _, _, scales = stitch_terms(
        "dummy", ["alpha", "beta"], verbose=False, use_cache=False
    )

    assert record["msg"] == "No overlapping data found; default scale of 1 used."
    assert scales.to_dict() == {"alpha": 1.0, "beta": 1.0}

