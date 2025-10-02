import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from fastapi import UploadFile

from .db import AutoMLSession, SessionLocal

UPLOAD_ROOT = Path("uploaded_data")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def create_session_directory(upload_root=UPLOAD_ROOT) -> Tuple[str, Path]:
    """Create and return a new session id and directory path."""
    session_id = str(uuid.uuid4())
    session_dir = upload_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


def save_upload(file: UploadFile, destination: Path) -> None:
    """Persist an uploaded file to the given destination path."""
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


def load_table(file_path: Path) -> pd.DataFrame:
    """Load a table file into a DataFrame based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(file_path)
    if suffix in [".xls", ".xlsx", ".xlsm", ".xlsb"]:
        return pd.read_excel(file_path)
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file_path)
    if suffix in [".json"]:
        return pd.read_json(file_path)
    # Fallback: try csv to keep previous behavior
    return pd.read_csv(file_path)


def validate_tabular_inputs(
    train_path: Path,
    target_column_name: str,
    time_stamp_column_name: Optional[str] = None,
    task_type: str = "classification",
) -> Optional[str]:
    """Validate required columns and task type for tabular training."""
    try:
        train_df = load_table(train_path)
    except Exception as e:
        return f"Could not read training data: {e}"

    if target_column_name not in train_df.columns:
        return f"Target column '{target_column_name}' not found."

    if time_stamp_column_name and time_stamp_column_name not in train_df.columns:
        return f"Timestamp column '{time_stamp_column_name}' not found."

    if task_type not in ["classification", "regression", "time series"]:
        return f"Invalid task_type '{task_type}'"

    return None


def store_session_in_db(
    session_id: str,
    train_path: Path,
    test_path: Optional[Path],
    target_column_name: str,
    time_stamp_column_name: Optional[str],
    task_type: str,
    time_budget: int,
) -> None:
    """Persist a new AutoML session in the database."""
    db = SessionLocal()
    try:
        new_session = AutoMLSession(
            session_id=session_id,
            train_file_path=str(train_path),
            test_file_path=str(test_path) if test_path else None,
            target_column=target_column_name,
            time_stamp_column_name=time_stamp_column_name,
            task_type=task_type,
            time_budget=time_budget,
        )
        db.add(new_session)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@dataclass
class SessionData:
    """Lightweight container for session metadata retrieved from DB."""

    session_id: str
    train_file_path: str
    test_file_path: Optional[str]
    target_column: str
    time_stamp_column_name: Optional[str]
    task_type: str
    time_budget: int


def get_session(session_id: str) -> Optional[SessionData]:
    """Fetch a session by id, returning typed `SessionData` or None."""
    db = SessionLocal()
    try:
        rec = db.query(AutoMLSession).filter_by(session_id=session_id).first()
        if rec is None:
            return None
        tb_raw = rec.__dict__.get("time_budget")
        return SessionData(
            session_id=str(rec.session_id),
            train_file_path=str(rec.train_file_path),
            test_file_path=(
                str(rec.test_file_path) if rec.test_file_path is not None else None
            ),
            target_column=str(rec.target_column),
            time_stamp_column_name=(
                str(rec.time_stamp_column_name)
                if rec.time_stamp_column_name is not None
                else None
            ),
            task_type=str(rec.task_type),
            time_budget=int(tb_raw) if tb_raw is not None else 0,
        )
    finally:
        db.close()
