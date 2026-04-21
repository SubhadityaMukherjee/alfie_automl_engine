import json
import logging
import os
import pickle
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from fastapi import UploadFile

from app.tabular_automl.modules import AutoMLTrainer
from app.tabular_automl.models import SUPPORTED_TABULAR_TASK_TYPES

from app.core.utils import render_template

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_jinja_path = os.getenv("JINJAPATH")
if not _jinja_path:
    raise RuntimeError("JINJAPATH environment variable is not set")

jinja_environment = Environment(loader=FileSystemLoader(_jinja_path))


UPLOAD_ROOT = Path("uploaded_data")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_FILE_TYPES = {"csv", "tsv", "parquet"}


def create_session_directory(upload_root: Path = UPLOAD_ROOT) -> tuple[str, Path]:
    """Create and return a new session id and directory path."""

    session_id = str(uuid.uuid4())
    session_dir = upload_root / session_id

    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create session directory {session_dir}: {e}")
        raise RuntimeError(f"Failed to create session directory: {e}") from e

    logging.debug(f"Session directory created at {session_dir}")
    return session_id, session_dir


def save_upload(file: UploadFile, destination: Path) -> None:
    """Persist an uploaded file to the given destination path."""
    if not hasattr(file, "file"):
        raise ValueError("file must have a 'file' attribute")

    if destination.parent:
        destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.debug(f"File saved to {destination}")
    except IOError as e:
        logging.error(f"Failed to write file to {destination}: {e}")
        raise RuntimeError(f"Failed to save uploaded file: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error saving file to {destination}: {e}")
        raise RuntimeError(f"Unexpected error saving file: {e}") from e


def load_table(file_path: Path) -> pd.DataFrame:
    """Load a table file into a DataFrame based on file extension."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    suffix = file_path.suffix.lower()

    try:
        if suffix in [".csv"]:
            logging.debug("csv file loaded")
            return pd.read_csv(file_path)
        if suffix in [".tsv"]:
            logging.debug("tsv file loaded")
            return pd.read_csv(file_path, sep="\t")
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
    except pd.errors.EmptyDataError:
        logging.error(f"File is empty: {file_path}")
        raise ValueError(f"File is empty: {file_path}") from None
    except pd.errors.ParserError as e:
        logging.error(f"Failed to parse file {file_path}: {e}")
        raise ValueError(f"Failed to parse file: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error loading table from {file_path}: {e}")
        raise RuntimeError(f"Failed to load table: {e}") from e


def validate_tabular_inputs(
    train_path: Path,
    target_column_name: str,
    time_stamp_column_name: str | None = None,
    task_type: str = "tabular_classification",
) -> str | None:
    """Validate required columns and task type for tabular training."""

    if not train_path.exists():
        logger.error(f"Training file not found: {train_path}")
        return f"Training file not found: {train_path}"

    if not target_column_name or not isinstance(target_column_name, str):
        logger.error("target_column_name must be a non-empty string")
        return "target_column_name must be a non-empty string"

    if task_type not in SUPPORTED_TABULAR_TASK_TYPES:
        logger.error(f"Invalid task type {task_type}")
        return f"Invalid task_type '{task_type}'. Must be one of: {SUPPORTED_TABULAR_TASK_TYPES}"

    try:
        train_df = load_table(train_path)
    except FileNotFoundError as e:
        logging.error(f"Training file not found: {e}")
        return f"Training file not found: {e}"
    except ValueError as e:
        logging.error(f"Could not read training data: {e}")
        return f"Could not read training data: {e}"
    except Exception as e:
        logging.error(f"Unexpected error reading training data: {e}")
        return f"Unexpected error reading training data: {e}"

    if train_df.empty:
        logger.error("Training dataframe is empty")
        return "Training dataframe is empty"

    if target_column_name not in train_df.columns:
        available_columns = ", ".join(train_df.columns.tolist())
        logger.error(
            f"Target column '{target_column_name}' not found. Available columns: {available_columns}"
        )
        return f"Target column '{target_column_name}' not found. Available columns: {available_columns}"

    if time_stamp_column_name and time_stamp_column_name not in train_df.columns:
        available_columns = ", ".join(train_df.columns.tolist())
        logger.error(
            f"Timestamp column '{time_stamp_column_name}' not found. Available columns: {available_columns}"
        )
        return f"Timestamp column '{time_stamp_column_name}' not found. Available columns: {available_columns}"

    return None


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

    if not autodw_base or not isinstance(autodw_base, str):
        raise ValueError("autodw_base must be a non-empty string")

    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string")

    if not dataset_id or not isinstance(dataset_id, str):
        raise ValueError("dataset_id must be a non-empty string")

    metadata_url = _build_metadata_url(
        autodw_base, user_id, dataset_id, dataset_version
    )
    logger.debug(f"Fetching dataset metadata: {metadata_url}")

    try:
        resp = requests.get(metadata_url, timeout=15)
        resp.raise_for_status()
    except requests.Timeout:
        logger.error(f"Timeout fetching metadata from {metadata_url}")
        raise RuntimeError("Timeout fetching dataset metadata from AutoDW") from None
    except requests.ConnectionError as e:
        logger.error(f"Connection error fetching metadata: {e}")
        raise RuntimeError(f"Failed to connect to AutoDW: {e}") from e
    except requests.HTTPError as e:
        logger.error(f"HTTP error fetching metadata: {e}")
        raise RuntimeError(f"AutoDW returned HTTP error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error fetching metadata: {e}")
        raise RuntimeError(f"Unexpected error fetching metadata: {e}") from e

    try:
        metadata = resp.json()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from AutoDW: {e}")
        raise RuntimeError(f"Invalid JSON response from AutoDW: {e}") from e

    if not isinstance(metadata, dict):
        logger.error(f"Metadata is not a dict: {type(metadata)}")
        raise RuntimeError(
            f"Invalid metadata format: expected dict, got {type(metadata)}"
        )

    return metadata


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

    if not download_url or not isinstance(download_url, str):
        raise ValueError("download_url must be a non-empty string")

    if dest_path.parent and not dest_path.parent.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(download_url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
    except requests.RequestException as e:
        if isinstance(e, requests.Timeout):
            logger.error(f"Timeout downloading from {download_url}")
            raise RuntimeError("Timeout downloading dataset") from e
        elif isinstance(e, requests.ConnectionError):
            logger.error(f"Connection error downloading dataset: {e}")
            raise RuntimeError(f"Failed to connect to download URL: {e}") from e
        elif isinstance(e, requests.HTTPError):
            logger.error(f"HTTP error downloading dataset: {e}")
            raise RuntimeError(f"HTTP error downloading dataset: {e}") from e
        else:
            logger.error(f"Request error downloading dataset: {e}")
            raise RuntimeError(f"Request error downloading dataset: {e}") from e
    except OSError as e:
        logger.error(f"Failed to write dataset to {dest_path}: {e}")
        raise RuntimeError(f"Failed to save dataset file: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error downloading dataset: {e}")
        raise RuntimeError(f"Unexpected error downloading dataset: {e}") from e

    if not dest_path.exists():
        raise RuntimeError(f"Download completed but file not created at {dest_path}")

    if dest_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {dest_path}")

    logger.info(f"Dataset saved to {dest_path}")


def train_automl(
    dataset_path: Path,
    save_model_path: Path,
    target_column_name: str,
    task_type: str,
    time_budget: int,
):
    """Train an AutoML model and return (leaderboard, predictor)."""

    if not target_column_name or not isinstance(target_column_name, str):
        raise ValueError("target_column_name must be a non-empty string")

    if not isinstance(time_budget, int) or time_budget <= 0:
        raise ValueError("time_budget must be a positive integer")

    try:
        os.makedirs(save_model_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create model directory {save_model_path}: {e}")
        raise RuntimeError(f"Failed to create model directory: {e}") from e

    try:
        trainer = AutoMLTrainer(save_model_path=save_model_path)
    except ValueError as e:
        logger.error(f"Failed to initialize AutoML trainer: {e}")
        raise RuntimeError(f"Failed to initialize trainer: {e}") from e

    try:
        train_df = load_table(dataset_path)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise RuntimeError(f"Failed to load training data: {e}") from e

    try:
        return trainer.train(
            train_df=train_df,
            test_df=None,
            target_column=target_column_name,
            time_limit=int(time_budget),
        )
    except ValueError as e:
        logger.error(f"Training validation error: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Training runtime error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise RuntimeError(f"Unexpected error during training: {e}") from e


def deployment_instructions() -> str:
    if jinja_environment is not None:
        try:
            return render_template(
                jinja_environment, "tabular_deployment_instructions.md"
            )
        except Exception as e:
            logger.error(f"Failed to render deployment instructions: {e}")
            return "No instructions available"
    else:
        logger.warning("jinja_environment is None, returning default instructions")
        return "No instructions found"


def serialize_and_zip_predictor(
    predictor, save_model_path: Path, tmp_path: Path
) -> Path:
    """Pickle the predictor and zip the model directory. Returns the zip path."""

    if predictor is None:
        raise ValueError("predictor cannot be None")

    if not save_model_path.exists():
        raise ValueError(f"save_model_path does not exist: {save_model_path}")

    predictor_path = save_model_path / "predictor.pkl"

    try:
        with open(predictor_path, "wb") as f:
            pickle.dump(predictor, f)
        logger.debug(f"Predictor serialized to {predictor_path}")
    except IOError as e:
        logger.error(f"Failed to write predictor pickle: {e}")
        raise RuntimeError(f"Failed to serialize predictor: {e}") from e
    except pickle.PicklingError as e:
        logger.error(f"Failed to pickle predictor: {e}")
        raise RuntimeError(f"Failed to pickle predictor: {e}") from e

    try:
        instructions_path = save_model_path / "tabular_deployment_instructions.md"
        with open(instructions_path, "w") as f:
            f.write(deployment_instructions())
        logger.debug(f"Deployment instructions written to {instructions_path}")
    except Exception as e:
        logger.debug(f"No deployment_instructions found, {e}")

    zip_path = tmp_path / "automl_predictor.zip"

    try:
        base_name = str(zip_path).replace(".zip", "")
        shutil.make_archive(
            base_name=base_name,
            format="zip",
            root_dir=save_model_path,
        )
        logger.debug(f"Model zipped to {zip_path}")
    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
        raise RuntimeError(f"Failed to zip model: {e}") from e

    if not zip_path.exists():
        raise RuntimeError(f"Zip file was not created at {zip_path}")

    return zip_path


def build_upload_payload(
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    task_type: str,
    leaderboard_json: list | dict,
) -> tuple[str, dict]:
    """Return (model_id, form_data_dict) for the AutoDW upload request."""

    if not dataset_id or not isinstance(dataset_id, str):
        raise ValueError("dataset_id must be a non-empty string")

    if not task_type or not isinstance(task_type, str):
        raise ValueError("task_type must be a non-empty string")

    try:
        model_id = f"automl_{dataset_id}_{int(datetime.utcnow().timestamp())}"
    except Exception as e:
        logger.error(f"Failed to generate model_id: {e}")
        raise RuntimeError(f"Failed to generate model_id: {e}") from e

    try:
        leaderboard_str = json.dumps(leaderboard_json)
    except TypeError as e:
        logger.error(f"Failed to serialize leaderboard_json: {e}")
        raise RuntimeError(f"Failed to serialize leaderboard: {e}") from e

    version = dataset_version or metadata.get("version", "v1")
    if not isinstance(version, str):
        version = "v1"

    data = {
        "model_id": model_id,
        "name": f"AutoML Model - {model_id}",
        "description": "AutoML trained model for tabular data",
        "framework": "sklearn",
        "model_type": task_type,
        "training_dataset": str(dataset_id),
        "training_dataset_version": version,
        "leaderboard": leaderboard_str,
        "deployment_instructions": deployment_instructions(),
    }

    if not isinstance(data["deployment_instructions"], str):
        data["deployment_instructions"] = ""

    return model_id, data


def upload_model(
    upload_url: str,
    zip_path: Path,
    payload: dict,
    task_id: str | None,
) -> requests.Response:
    """Upload the zipped model to AutoDW. Returns the raw response."""

    if not upload_url or not isinstance(upload_url, str):
        raise ValueError("upload_url must be a non-empty string")

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if not isinstance(payload, dict) or not payload:
        raise ValueError("payload must be a non-empty dict")

    headers = {"X-Task-ID": task_id} if task_id else {}
    if task_id:
        logger.debug(f"Including X-Task-ID header: {task_id}")

    try:
        with open(zip_path, "rb") as f:
            files = {"file": (zip_path.name, f, "application/octet-stream")}
            logger.debug(f"Uploading model to {upload_url}")
            return requests.post(
                upload_url, headers=headers, files=files, data=payload, timeout=120
            )
    except requests.RequestException as e:
        if isinstance(e, requests.Timeout):
            logger.error(f"Timeout uploading to {upload_url}")
            raise RuntimeError("Timeout uploading model to AutoDW") from e
        elif isinstance(e, requests.ConnectionError):
            logger.error(f"Connection error uploading model: {e}")
            raise RuntimeError(f"Failed to connect to upload URL: {e}") from e
        elif isinstance(e, requests.HTTPError):
            logger.error(f"HTTP error uploading model: {e}")
            raise RuntimeError(f"HTTP error uploading model: {e}") from e
        else:
            logger.error(f"Request error uploading model: {e}")
            raise RuntimeError(f"Request error uploading model: {e}") from e
    except OSError as e:
        logger.error(f"Failed to read zip file {zip_path}: {e}")
        raise RuntimeError(f"Failed to read zip file: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error uploading model: {e}")
        raise RuntimeError(f"Unexpected error uploading model: {e}") from e
