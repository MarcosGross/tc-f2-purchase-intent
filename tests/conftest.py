"""Fixtures pequenas e locais para o pipeline de dados."""

from pathlib import Path

import pandas as pd
import pytest

from purchase_intent.data_schema import EXPECTED_COLUMNS, NUMERIC_COLUMNS


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows = 40
    data = {
        name: [0.12345678901234568 + index for index in range(rows)] for name in NUMERIC_COLUMNS
    }
    data.update(
        {
            "Month": ["Jan"] * rows,
            "VisitorType": ["Returning_Visitor"] * rows,
            "Weekend": [False, True] * 20,
            "Revenue": [False, True] * 20,
        }
    )
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


@pytest.fixture
def raw_path(tmp_path: Path, raw_frame: pd.DataFrame) -> Path:
    path = tmp_path / "raw.csv"
    raw_frame.to_csv(path, index=False, lineterminator="\n")
    return path
