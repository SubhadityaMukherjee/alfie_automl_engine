import io
from pathlib import Path

import pandas as pd
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.tabular_automl import services
from app.tabular_automl.db import Base


def test_create_session_directory_uses_custom_upload_root(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "UPLOAD_ROOT", tmp_path)
    session_id, session_dir = services.create_session_directory()
    assert isinstance(session_id, str) and len(session_id) > 0
    assert session_dir.exists() and session_dir.is_dir()
    assert session_dir.parent == tmp_path


def test_save_upload_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "UPLOAD_ROOT", tmp_path)
    _sid, session_dir = services.create_session_directory()
    content = b"col1,col2\n1,2\n3,4\n"
    upload = UploadFile(filename="train.csv", file=io.BytesIO(content))
    destination = session_dir / "train.csv"

    services.save_upload(upload, destination)

    assert destination.exists()
    assert destination.read_bytes() == content


def test_load_table_csv_and_json(tmp_path):
    csv_path = tmp_path / "data.csv"
    df_csv_expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_csv_expected.to_csv(csv_path, index=False)
    df_csv = services.load_table(csv_path)
    pd.testing.assert_frame_equal(df_csv, df_csv_expected)

    json_path = tmp_path / "data.json"
    df_json_expected = pd.DataFrame({"x": ["u", "v"], "y": [10, 20]})
    df_json_expected.to_json(json_path, orient="records")
    # read_json without orient will still parse the records list
    df_json = services.load_table(json_path)
    # Order of columns may differ; sort columns for comparison
    pd.testing.assert_frame_equal(df_json[df_json_expected.columns], df_json_expected)


def test_validate_tabular_inputs_success_and_errors(tmp_path):
    # success
    path_ok = tmp_path / "train.csv"
    pd.DataFrame({"target": [0, 1], "ts": [1, 2], "f": [3, 4]}).to_csv(path_ok, index=False)
    err = services.validate_tabular_inputs(path_ok, "target", "ts", "classification")
    assert err is None

    # missing target
    path_missing_target = tmp_path / "train2.csv"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(path_missing_target, index=False)
    err = services.validate_tabular_inputs(path_missing_target, "target", None, "regression")
    assert err and "Target column" in err

    # missing timestamp
    err = services.validate_tabular_inputs(path_ok, "target", "missing_ts", "time series")
    assert err and "Timestamp column" in err

    # invalid task type
    err = services.validate_tabular_inputs(path_ok, "target", None, "not_a_task")
    assert err and "Invalid task_type" in err

    # unreadable file
    err = services.validate_tabular_inputs(path_ok.with_suffix(".doesnotexist"), "target", None, "classification")
    assert err and "Could not read" in err


def test_store_and_get_session_with_in_memory_db(tmp_path, monkeypatch):
    # Prepare in-memory database and bind to services.SessionLocal
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(services, "SessionLocal", TestSessionLocal)

    # Prepare files
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"target": [1, 0]}).to_csv(train, index=False)
    pd.DataFrame({"target": [0, 1]}).to_csv(test, index=False)

    # Create a session entry
    session_id = "test-session-123"
    services.store_session_in_db(
        session_id=session_id,
        train_path=train,
        test_path=test,
        target_column_name="target",
        time_stamp_column_name=None,
        task_type="classification",
        time_budget=120,
    )

    # Retrieve and validate
    sd = services.get_session(session_id)
    assert sd is not None
    assert sd.session_id == session_id
    assert Path(sd.train_file_path) == train
    assert sd.test_file_path is not None
    assert Path(sd.test_file_path) == test
    assert sd.target_column == "target"
    assert sd.time_stamp_column_name is None
    assert sd.task_type == "classification"
    assert sd.time_budget == 120

    # Missing record should return None
    assert services.get_session("nonexistent") is None

