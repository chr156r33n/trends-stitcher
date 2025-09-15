from pathlib import Path
import sys
import datetime as dt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from stitcher import TrendsFetcher


def test_parse_dataforseo_year_month():
    payload = {
        "items": [
            {
                "type": "google_trends_graph",
                "keywords": ["nike"],
                "data": [
                    {"year": 2023, "month": 1, "values": [10]},
                    {"year": 2024, "month": 1, "values": [20]},
                ],
            }
        ]
    }
    rows = TrendsFetcher._parse_dataforseo_trends(payload)
    df = pd.DataFrame(rows)
    assert df.shape[0] == 2
    assert list(df["date"]) == [dt.date(2023, 1, 1), dt.date(2024, 1, 1)]
    assert df["value"].tolist() == [10.0, 20.0]
