import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.core.exceptions import (
    AutoMLDataError,
    AutoMLRuntimeError,
    AutoMLSerializationError,
    AutoMLValidationError,
)
from app.tabular_automl.services import (
    build_upload_payload,
    convert_leaderboard_safely,
    load_table,
    serialize_and_zip_predictor,
    validate_tabular_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# load_table
# ---------------------------------------------------------------------------


def test_load_table(file_fixture, fake_data):
    df = load_table(file_fixture)
    assert isinstance(df, pd.DataFrame)
    assert list(df["col_1"]) == list(fake_data["col_1"])


@pytest.mark.parametrize(
    "make_path",
    [
        pytest.param(lambda tmp: tmp / "nonexistent.csv", id="file_not_found"),
        pytest.param(lambda tmp: tmp, id="directory"),
        pytest.param(
            lambda tmp: (_f := tmp / "empty.csv", _f.write_text(""), _f)[2],
            id="empty_file",
        ),
    ],
)
def test_load_table_raises_data_error(tmp_path, make_path):
    with pytest.raises(AutoMLDataError):
        load_table(make_path(tmp_path))


def test_load_table_malformed_csv(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text('col_1\n"unclosed')
    with pytest.raises((AutoMLDataError, AutoMLRuntimeError)):
        load_table(bad)


def test_load_table_unsupported_extension_fallback(tmp_path, fake_data):
    file = tmp_path / "data.txt"
    fake_data.to_csv(file, index=False)
    df = load_table(file)
    assert isinstance(df, pd.DataFrame)
    assert list(df["col_1"]) == list(fake_data["col_1"])


# ---------------------------------------------------------------------------
# validate_tabular_inputs
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    "task_type, target_col, keyword, setup",
    [
        pytest.param(
            "tabular_classification",
            "col_1",
            "not found",
            lambda tmp: tmp / "no_file.csv",
            id="missing_file",
        ),
        pytest.param(
            "invalid_task",
            "col_1",
            "Invalid",
            None,
            id="invalid_task_type",
        ),
        pytest.param(
            "tabular_classification",
            "",
            "target_column_name",
            None,
            id="empty_target",
        ),
        pytest.param(
            "tabular_classification",
            None,
            "target_column_name",
            None,
            id="none_target",
        ),
    ],
)
def test_validate_tabular_inputs_returns_error(
    task_type, target_col, keyword, setup, file_fixture, tmp_path
):
    path = setup(tmp_path) if setup else file_fixture
    result = validate_tabular_inputs(path, target_col, task_type=task_type)
    assert result is not None
    assert keyword in result


def test_validate_empty_dataframe(tmp_path):
    empty_csv = tmp_path / "empty_data.csv"
    pd.DataFrame(columns=["col_1", "col_2"]).to_csv(empty_csv, index=False)
    result = validate_tabular_inputs(
        empty_csv, "col_1", task_type="tabular_classification"
    )
    assert result is not None
    assert "empty" in result.lower()


@pytest.mark.parametrize(
    "ts_col, should_error",
    [
        pytest.param("nonexistent_ts", True, id="ts_not_found"),
        pytest.param("col_2", False, id="ts_found"),
    ],
)
def test_validate_timestamp_column(file_fixture, ts_col, should_error):
    result = validate_tabular_inputs(
        file_fixture,
        "col_1",
        task_type="tabular_time_series",
        time_stamp_column_name=ts_col,
    )
    if should_error:
        assert result is not None
        assert "Timestamp column" in result
    else:
        assert result is None


# ---------------------------------------------------------------------------
# convert_leaderboard_safely
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_val, expect_list, expect_str",
    [
        pytest.param(
            pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]}),
            True,
            None,
            id="dataframe",
        ),
        pytest.param(
            "some string result",
            False,
            "some string result",
            id="string",
        ),
    ],
)
def test_convert_leaderboard_safely(input_val, expect_list, expect_str):
    json_out, str_out = convert_leaderboard_safely(input_val)
    if expect_list:
        assert isinstance(json_out, list)
        assert len(json_out) == 2
    else:
        assert isinstance(json_out, dict)
        assert "result" in json_out
    if expect_str:
        assert str_out == expect_str


# ---------------------------------------------------------------------------
# serialize_and_zip_predictor
# ---------------------------------------------------------------------------


def test_serialize_and_zip_creates_zip(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    zip_path = serialize_and_zip_predictor({"model": True}, model_dir, tmp_path)
    assert zip_path.exists()
    assert zip_path.suffix == ".zip"
    assert zipfile.is_zipfile(zip_path)


def test_serialize_and_zip_contains_pickle(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    zip_path = serialize_and_zip_predictor({"foo": "bar"}, model_dir, tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        assert "predictor.pkl" in zf.namelist()


@pytest.mark.parametrize(
    "predictor, model_path_factory",
    [
        pytest.param(
            None,
            lambda tmp: (tmp / "model").mkdir() or tmp / "model",
            id="none_predictor",
        ),
        pytest.param(
            MagicMock, lambda tmp: tmp / "does_not_exist", id="missing_model_path"
        ),
    ],
)
def test_serialize_and_zip_raises_validation_error(
    tmp_path, predictor, model_path_factory
):
    model_path = model_path_factory(tmp_path)
    pred = predictor() if isinstance(predictor, type) else predictor
    with pytest.raises(AutoMLValidationError):
        serialize_and_zip_predictor(pred, model_path, tmp_path)


# ---------------------------------------------------------------------------
# build_upload_payload
# ---------------------------------------------------------------------------


def test_build_upload_payload_returns_tuple():
    model_id, data = build_upload_payload(
        dataset_id="ds-123",
        dataset_version="v2",
        metadata={},
        task_type="tabular_classification",
        leaderboard_json=[{"model": "A", "score": 0.9}],
    )
    assert isinstance(model_id, str)
    assert model_id.startswith("automl_ds-123_")
    assert data["model_id"] == model_id
    assert data["framework"] == "sklearn"
    assert data["model_type"] == "tabular_classification"
    assert "leaderboard" in data


def test_build_upload_payload_without_version():
    _, data = build_upload_payload(
        dataset_id="ds-456",
        dataset_version=None,
        metadata={"version": "v3"},
        task_type="tabular_regression",
        leaderboard_json={},
    )
    assert data["training_dataset_version"] == "v3"


def test_build_upload_payload_includes_deployment_instructions(monkeypatch):
    from app.tabular_automl import services as svc

    monkeypatch.setattr(svc, "deployment_instructions", lambda: "deploy me")
    _, data = build_upload_payload(
        dataset_id="ds-789",
        dataset_version="v1",
        metadata={},
        task_type="tabular_classification",
        leaderboard_json=[],
    )
    assert data.get("deployment_instructions") == "deploy me"


@pytest.mark.parametrize(
    "kwargs, expected_exc",
    [
        pytest.param(
            dict(
                dataset_id="", task_type="tabular_classification", leaderboard_json=[]
            ),
            AutoMLValidationError,
            id="empty_dataset_id",
        ),
        pytest.param(
            dict(
                dataset_id=None, task_type="tabular_classification", leaderboard_json=[]
            ),  # type: ignore[arg-type]
            AutoMLValidationError,
            id="none_dataset_id",
        ),
        pytest.param(
            dict(dataset_id="ds-1", task_type="", leaderboard_json=[]),
            AutoMLValidationError,
            id="empty_task_type",
        ),
        pytest.param(
            dict(
                dataset_id="ds-1",
                task_type="tabular_classification",
                leaderboard_json=type("Bad", (), {"__str__": lambda s: "bad"})(),
            ),
            AutoMLSerializationError,
            id="non_serializable_leaderboard",
        ),
    ],
)
def test_build_upload_payload_raises(kwargs, expected_exc):
    with pytest.raises(expected_exc):
        build_upload_payload(
            dataset_version="v1",
            metadata={},
            **kwargs,
        )
