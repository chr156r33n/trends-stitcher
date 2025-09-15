import ast
from pathlib import Path
import pandas as pd
import datetime as dt
import numpy as np

# Extract yoy_table and infer_cadence from app.py without importing streamlit
app_path = Path(__file__).resolve().parents[1] / 'app.py'
source = app_path.read_text()
module = ast.parse(source)
functions = {}
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in {'infer_cadence', 'yoy_table'}:
        functions[node.name] = ast.get_source_segment(source, node)
namespace = {'pd': pd, 'np': np}
for name, src in functions.items():
    exec(src, namespace)
yoy_table = namespace['yoy_table']


def test_yoy_monthly_merge():
    rows = []
    values = {2023: 100, 2024: 120, 2025: 180}
    for year, val in values.items():
        rows.append({'date': dt.date(year, 1, 1), 'term': 'nike', 'value': val})
    df = pd.DataFrame(rows)
    yt = yoy_table(df, 'nike')

    r2024 = yt[yt['date'] == pd.Timestamp('2024-01-01')].iloc[0]
    assert r2024['previous_year'] == 100
    assert round(r2024['pct_diff'], 2) == 20.0

    r2025 = yt[yt['date'] == pd.Timestamp('2025-01-01')].iloc[0]
    assert r2025['previous_year'] == 120
    assert round(r2025['pct_diff'], 2) == 50.0
