import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app.tabular_automl.services import (
    load_table,
    validate_tabular_inputs,
)


@pytest.fixture
def fake_data():
    return pd.DataFrame.from_dict(
        {"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]}
    )


@pytest.fixture(params=["csv", "excel", "parquet", "json"], ids=lambda x: f"{x}_file")
def file_fixture(request, fake_data, tmp_path):
    file = tmp_path / f"test.{request.param if request.param != 'excel' else 'xlsx'}"

    if request.param == "csv":
        fake_data.to_csv(file, index=False)
    elif request.param == "excel":
        fake_data.to_excel(file, index=False)
    elif request.param == "parquet":
        fake_data.to_parquet(file)
    elif request.param == "json":
        fake_data.to_json(file, orient="records")

    return file


def test_load_table(file_fixture, fake_data):
    df = load_table(file_fixture)

    assert isinstance(df, pd.DataFrame)
    assert list(df["col_1"]) == list(fake_data["col_1"])


@pytest.mark.parametrize(
    "task_type , target_col, expected",
    [
        ("tabular_regression", "col_1", None),
        ("tabular_classification", "col_1", None),
        ("tabular_time_series", "col_1", None),
        (
            "tabular_classification",
            "col_not",
            "Target column 'col_not' not found. Available columns: col_1, col_2",
        ),
    ],
)
def test_validate_tabular_inputs_task_type(
    task_type, target_col, expected, file_fixture
):
    temp = validate_tabular_inputs(
        Path(file_fixture), target_col, task_type=task_type, time_stamp_column_name=None
    )
    assert temp == expected
