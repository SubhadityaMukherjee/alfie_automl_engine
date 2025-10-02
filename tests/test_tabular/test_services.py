import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app.tabular_automl.services import (
    create_session_directory,
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


def test_create_session_directory():
    tmp_dir = tempfile.mkdtemp()
    session_id, session_dir = create_session_directory(upload_root=Path(tmp_dir))
    assert type(session_id) == str
    assert os.path.exists(session_dir)


@pytest.mark.parametrize(
    "task_type , target_col, expected",
    [
        ("regression", "col_1", None),
        ("classification", "col_1", None),
        ("time series", "col_1", None),
        ("random", "col_1", "Invalid task_type 'random'"),
        ("classification", "col_not", "Target column 'col_not' not found."),
    ],
)
def test_validate_tabular_inputs_task_type(
    task_type, target_col, expected, file_fixture
):
    temp = validate_tabular_inputs(
        Path(file_fixture), target_col, task_type=task_type, time_stamp_column_name=None
    )
    assert temp == expected
