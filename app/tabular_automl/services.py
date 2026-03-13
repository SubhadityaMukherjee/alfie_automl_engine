import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import UploadFile
import requests
import os
from datetime import datetime
import pickle
from .db import AutoMLSession, SessionLocal
import json

from app.tabular_automl.modules import AutoMLTrainer

logger = logging.getLogger(__name__)

UPLOAD_ROOT = Path("uploaded_data")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_FILE_TYPES = {"csv", "tsv", "parquet"}


def create_session_directory(upload_root: Path = UPLOAD_ROOT) -> tuple[str, Path]:
    """Create and return a new session id and directory path."""
    session_id = str(uuid.uuid4())
    session_dir = upload_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Session directory created at {session_dir}")
    return session_id, session_dir


def save_upload(file: UploadFile, destination: Path) -> None:
    """Persist an uploaded file to the given destination path."""
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        logging.debug(f"File saved to {destination}")


def load_table(file_path: Path) -> pd.DataFrame:
    """Load a table file into a DataFrame based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix in [".csv"]:
        logging.debug("csv file loaded")
        return pd.read_csv(file_path)
    if suffix in [".xls", ".xlsx", ".xlsm", ".xlsb"]:
        logging.debug("excel file loaded")
        return pd.read_excel(file_path)
    if suffix in [".parquet", ".pq"]:
        logging.debug("Parquet file loaded")
        return pd.read_parquet(file_path)
    if suffix in [".json"]:
        logging.debug("Json file loaded")
        return pd.read_json(file_path)
    # Fallback: try csv to keep previous behavior
    return pd.read_csv(file_path)


def validate_tabular_inputs(
    train_path: Path,
    target_column_name: str,
    time_stamp_column_name: str | None = None,
    task_type: str = "classification",
) -> str | None:
    """Validate required columns and task type for tabular training."""
    try:
        train_df = load_table(train_path)
    except Exception as e:
        logging.error(f"Could not read training data {e}")
        return f"Could not read training data: {e}"

    if target_column_name not in train_df.columns:
        logger.error(f"Target column '{target_column_name}' not found.")
        return f"Target column '{target_column_name}' not found."

    if time_stamp_column_name and time_stamp_column_name not in train_df.columns:
        logger.error(f"Timestampl column '{time_stamp_column_name}' not found.")
        return f"Timestamp column '{time_stamp_column_name}' not found."

    if task_type not in ["classification", "regression", "time series"]:
        logger.error(f"Invalid task type {task_type}")
        return f"Invalid task_type '{task_type}'"

    return None


def store_session_in_db(
    session_id: str,
    train_path: Path,
    test_path: Path | None,
    target_column_name: str,
    time_stamp_column_name: str | None,
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
    test_file_path: str | None
    target_column: str
    time_stamp_column_name: str | None
    task_type: str
    time_budget: int


def get_session(session_id: str) -> SessionData | None:
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
                str(rec.test_file_path) if rec.test_file_path != "" else None
            ),
            target_column=str(rec.target_column),
            time_stamp_column_name=(
                str(rec.time_stamp_column_name)
                if rec.time_stamp_column_name != ""
                else None
            ),
            task_type=str(rec.task_type),
            time_budget=int(tb_raw) if tb_raw is not None else 0,
        )
    finally:
        db.close()


def convert_leaderboard_safely(leaderboard):
    if isinstance(leaderboard, pd.DataFrame):
        leaderboard_json = leaderboard.to_dict(orient="records")
        leaderboard_str = leaderboard.to_markdown()
    else:
        leaderboard_json = {"result": str(leaderboard)}
        leaderboard_str = str(leaderboard)
    return leaderboard_json, leaderboard_str


def _build_metadata_url(
    autodw_base: str, user_id: str, dataset_id: str, dataset_version: str | None
) -> str:
    url = f"{autodw_base}/datasets/{user_id}/{dataset_id}"
    if dataset_version:
        url = f"{url}/version/{dataset_version}"
    return url


def fetch_dataset_metadata(
    autodw_base: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str | None,
) -> dict:
    """Fetch and return dataset metadata from AutoDW."""
    metadata_url = _build_metadata_url(
        autodw_base, user_id, dataset_id, dataset_version
    )
    logger.debug(f"Fetching dataset metadata: {metadata_url}")
    resp = requests.get(metadata_url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def resolve_download_url(
    autodw_base: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    dataset_split: str | None,
) -> str:
    """Determine the correct dataset download URL, accounting for splits."""
    base_url = _build_metadata_url(autodw_base, user_id, dataset_id, dataset_version)
    download_url = f"{base_url}/download"

    has_split = bool(metadata.get("custom_metadata", {}).get("split"))
    effective_split = (
        dataset_split
        if (has_split and dataset_split in ("train", "test", "drift"))
        else None
    )

    if effective_split:
        download_url = f"{download_url}?split={effective_split}"
        logger.info(
            f"Dataset has splits; downloading '{effective_split}' split from: {download_url}"
        )
    else:
        if dataset_split and not has_split:
            logger.warning(
                f"dataset_split='{dataset_split}' was requested but dataset has no splits; "
                "downloading full dataset."
            )
        logger.debug(f"Downloading full dataset file: {download_url}")

    return download_url


def download_dataset(download_url: str, dest_path: Path) -> None:
    """Stream-download a dataset file to dest_path."""
    with requests.get(download_url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
    logger.info(f"Dataset saved to {dest_path}")


def train_automl(
    dataset_path: Path,
    save_model_path: Path,
    target_column_name: str,
    task_type: str,
    time_budget: int,
):
    """Train an AutoML model and return (leaderboard, predictor)."""
    os.makedirs(save_model_path, exist_ok=True)
    trainer = AutoMLTrainer(save_model_path=save_model_path)
    train_df = load_table(dataset_path)
    return trainer.train(
        train_df=train_df,
        test_df=None,
        target_column=target_column_name,
        time_limit=int(time_budget),
    )


def serialize_and_zip_predictor(
    predictor, save_model_path: Path, tmp_path: Path
) -> Path:
    """Pickle the predictor and zip the model directory. Returns the zip path."""
    predictor_path = save_model_path / "predictor.pkl"
    with open(predictor_path, "wb") as f:
        pickle.dump(predictor, f)

    zip_path = tmp_path / "automl_predictor.zip"
    shutil.make_archive(
        base_name=str(zip_path).replace(".zip", ""),
        format="zip",
        root_dir=save_model_path,
    )
    return zip_path


def build_upload_payload(
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    task_type: str,
    leaderboard_json: list | dict,
) -> tuple[str, dict]:
    """Return (model_id, form_data_dict) for the AutoDW upload request."""
    model_id = f"automl_{dataset_id}_{int(datetime.utcnow().timestamp())}"
    data = {
        "model_id": model_id,
        "name": f"AutoML Model - {model_id}",
        "description": "AutoML trained model for tabular data",
        "framework": "sklearn",
        "model_type": task_type,
        "training_dataset": str(dataset_id),
        "training_dataset_version": dataset_version or metadata.get("version", "v1"),
        "leaderboard": json.dumps(leaderboard_json),
    }
    return model_id, data


def upload_model(
    upload_url: str,
    zip_path: Path,
    payload: dict,
    task_id: str | None,
) -> requests.Response:
    """Upload the zipped model to AutoDW. Returns the raw response."""
    headers = {"X-Task-ID": task_id} if task_id else {}
    if task_id:
        logger.debug(f"Including X-Task-ID header: {task_id}")

    with open(zip_path, "rb") as f:
        files = {"file": (zip_path.name, f, "application/octet-stream")}
        logger.debug(f"Uploading model to {upload_url}")
        return requests.post(
            upload_url, headers=headers, files=files, data=payload, timeout=120
        )
