from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from stitcher import TrendsFetcher


def test_parse_timeseries_nested_structure():
    payload = {
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
    df = TrendsFetcher._parse_timeseries(payload, ["nike", "adidas"])
    assert set(df.columns) == {"date", "term", "value"}
    assert set(df["term"]) == {"nike", "adidas"}


def test_parse_timeseries_flat_structure():
    payload = {
        "timeline_data": [
            {"time": "2024-01-01", "value": [1, 2]},
            {"time": "2024-01-02", "value": [3, 4]},
        ]
    }
    df = TrendsFetcher._parse_timeseries(payload, ["nike", "adidas"])
    assert len(df) == 4
    nike_values = df[df["term"] == "nike"]["value"].tolist()
    assert nike_values == [1, 3]


def test_parse_timeseries_missing_raises():
    with pytest.raises(RuntimeError):
        TrendsFetcher._parse_timeseries({}, ["nike"])


def test_parse_timeseries_brightdata_widgets():
    payload = {
        "widgets": [
            {
                "id": "TIMESERIES",
                "data": {
                    "default": {
                        "timelineData": [
                            {
                                "time": "1599955200",
                                "value": [15, 72],
                                "formattedValue": ["15", "72"],
                                "hasData": [True, True],
                            },
                            {
                                "time": "1600560000",
                                "value": [14, 76],
                                "formattedValue": ["14", "76"],
                                "hasData": [True, True],
                            },
                        ]
                    }
                },
            }
        ]
    }

    df = TrendsFetcher._parse_timeseries(payload, ["nike", "adidas"])
    assert not df.empty
    assert set(df["term"]) == {"nike", "adidas"}
    # Ensure unix timestamps are converted to dates
    assert df["date"].min().year >= 2020

