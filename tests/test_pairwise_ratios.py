from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from stitcher import pairwise_ratios


def test_pairwise_ratios_handles_duplicates():
    data = [
        {"batch_id": "b1", "date": "2024-01-01", "term": "alpha", "value": 10},
        {"batch_id": "b1", "date": "2024-01-01", "term": "alpha", "value": 20},
        {"batch_id": "b1", "date": "2024-01-01", "term": "beta", "value": 30},
    ]
    df = pd.DataFrame(data)
    result = pairwise_ratios(df)

    ratio_ab = result[(result.term_i == "alpha") & (result.term_j == "beta")]["ratio_med"].iloc[0]
    ratio_ba = result[(result.term_i == "beta") & (result.term_j == "alpha")]["ratio_med"].iloc[0]

    assert ratio_ab == 0.5
    assert ratio_ba == 2.0
