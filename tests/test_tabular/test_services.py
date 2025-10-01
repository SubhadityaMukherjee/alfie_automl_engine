import pytest
import tempfile
import pandas as pd
from app.tabular_automl.services import load_table, create_session_directory, validate_tabular_inputs
from pathlib import Path
import os


@pytest.fixture
def fake_data():
    return pd.DataFrame.from_dict(
        {"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]}
    )


@pytest.fixture
def csv_file(fake_data):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        fake_data.to_csv(tmp.name, index=False)
        yield Path(tmp.name)


@pytest.fixture
def excel_file(fake_data):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        fake_data.to_excel(tmp.name, index=False)
        yield Path(tmp.name)


@pytest.fixture
def parquet_file(fake_data):
    with tempfile.NamedTemporaryFile(suffix=".pq", delete=False) as tmp:
        fake_data.to_parquet(tmp.name)
        yield Path(tmp.name)


@pytest.fixture
def json_file(fake_data):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        fake_data.to_json(tmp.name, orient="records")
        yield Path(tmp.name)


@pytest.mark.parametrize(
    "file_fixture", ["csv_file", "excel_file", "parquet_file", "json_file"]
)
def test_load_table(file_fixture, request, fake_data):
    file = request.getfixturevalue(file_fixture)
    df = load_table(file)

    assert isinstance(df, pd.DataFrame)
    assert list(df["col_1"]) == list(fake_data["col_1"])


def test_create_session_directory():
    tmp_dir = tempfile.mkdtemp()
    session_id, session_dir = create_session_directory(upload_root=Path(tmp_dir))
    assert type(session_id) == str
    assert os.path.exists(session_dir)

@pytest.mark.parametrize(
    "task_types", ["regression", "classification", "time series"])
def test_validate_tabular_inputs_task_type(task_types, csv_file, target_column_name = "col_1"):
    temp = validate_tabular_inputs(Path(csv_file), target_column_name, task_type = task_types, time_stamp_column_name=None)
    print(temp)
    assert temp is not None

