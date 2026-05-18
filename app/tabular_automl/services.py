import logging
import os
import pickle
import shutil
from pathlib import Path

import pandas as pd

from app.core.exceptions import (
    AutoMLConfigError,
    AutoMLDataError,
    AutoMLRuntimeError,
    AutoMLSerializationError,
    AutoMLValidationError,
)
from app.core.service_helpers import build_upload_payload as _core_build_upload_payload
from app.core.service_helpers import download_dataset as _core_download_dataset
from app.core.service_helpers import (
    fetch_dataset_metadata,
    resolve_download_url,
    upload_model,
)
from app.core.utils import jinja_environment, render_template
from app.tabular_automl.models import SUPPORTED_TABULAR_TASK_TYPES
from app.tabular_automl.modules import AutoMLTrainer

logger = logging.getLogger(__name__)


UPLOAD_ROOT = Path("uploaded_data")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_FILE_TYPES = {"csv", "tsv", "parquet"}


def load_table(file_path: Path) -> pd.DataFrame:
    """Load a table file into a DataFrame based on file extension."""
    if not file_path.exists():
        raise AutoMLDataError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise AutoMLDataError(f"Path is not a file: {file_path}")

    suffix = file_path.suffix.lower()

    try:
        if suffix in [".csv"]:
            logging.debug("csv file loaded")
            return pd.read_csv(file_path)
        if suffix in [".tsv"]:
            logging.debug("tsv file loaded")
            return pd.read_csv(file_path, sep="\t")
        if suffix in [".xlsx"]:
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
        raise AutoMLDataError(f"File is empty: {file_path}") from None
    except pd.errors.ParserError as e:
        logging.error(f"Failed to parse file {file_path}: {e}")
        raise AutoMLDataError(f"Failed to parse file: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error loading table from {file_path}: {e}")
        raise AutoMLRuntimeError(f"Failed to load table: {e}") from e


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
    except AutoMLDataError as e:
        logging.error(f"Training file not found: {e}")
        return f"Training file not found: {e}"
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


def download_dataset(download_url: str, dest_path: Path) -> None:
    """Stream-download a dataset file to dest_path."""
    _core_download_dataset(download_url, dest_path)


def train_automl(
    dataset_path: Path,
    save_model_path: Path,
    target_column_name: str,
    task_type: str,
    time_budget: int,
):
    """Train an AutoML model and return (leaderboard, predictor)."""

    if not target_column_name or not isinstance(target_column_name, str):
        raise AutoMLDataError("target_column_name must be a non-empty string")

    if not isinstance(time_budget, int) or time_budget <= 0:
        raise AutoMLConfigError("time_budget must be a positive integer")

    try:
        os.makedirs(save_model_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create model directory {save_model_path}: {e}")
        raise AutoMLRuntimeError(f"Failed to create model directory: {e}") from e

    try:
        trainer = AutoMLTrainer(save_model_path=save_model_path)
    except AutoMLConfigError as e:
        logger.error(f"Failed to initialize AutoML trainer: {e}")
        raise AutoMLRuntimeError(f"Failed to initialize trainer: {e}") from e

    try:
        train_df = load_table(dataset_path)
    except AutoMLDataError as e:
        logger.error(f"Failed to load training data: {e}")
        raise AutoMLRuntimeError(f"Failed to load training data: {e}") from e

    try:
        return trainer.train(
            train_df=train_df,
            test_df=None,
            target_column=target_column_name,
            time_limit=int(time_budget),
        )
    except AutoMLValidationError as e:
        logger.error(f"Training validation error: {e}")
        raise
    except AutoMLRuntimeError as e:
        logger.error(f"Training runtime error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise AutoMLRuntimeError(f"Unexpected error during training: {e}") from e


def deployment_instructions() -> str:
    """Return the deployment instructions from a markdown file"""
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


def tabular_data_instructions() -> str:
    """Return the instructions from what kind of data is accepted by the tabular AutoML engine"""
    if jinja_environment is not None:
        try:
            return render_template(jinja_environment, "tabular_accepted_format.md")
        except Exception as e:
            logger.error(f"Failed to render accepted format instructions: {e}")
            return "No accepted format instructions available"
    else:
        logger.warning("jinja_environment is None, returning default formats")
        return "Ask the agent for help"


def serialize_and_zip_predictor(
    predictor, save_model_path: Path, tmp_path: Path
) -> Path:
    """Pickle the predictor and zip the model directory. Returns the zip path."""

    if predictor is None:
        raise AutoMLValidationError("predictor cannot be None")

    if not save_model_path.exists():
        raise AutoMLValidationError(
            f"save_model_path does not exist: {save_model_path}"
        )

    predictor_path = save_model_path / "predictor.pkl"

    try:
        with open(predictor_path, "wb") as f:
            pickle.dump(predictor, f)
        logger.debug(f"Predictor serialized to {predictor_path}")
    except IOError as e:
        logger.error(f"Failed to write predictor pickle: {e}")
        raise AutoMLSerializationError(f"Failed to serialize predictor: {e}") from e
    except pickle.PicklingError as e:
        logger.error(f"Failed to pickle predictor: {e}")
        raise AutoMLSerializationError(f"Failed to pickle predictor: {e}") from e

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
        raise AutoMLSerializationError(f"Failed to zip model: {e}") from e

    if not zip_path.exists():
        raise AutoMLSerializationError(f"Zip file was not created at {zip_path}")

    return zip_path


def build_upload_payload(
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    task_type: str,
    leaderboard_json: list | dict,
) -> tuple[str, dict]:
    """Return (model_id, form_data_dict) for the AutoDW upload request."""
    instructions = deployment_instructions()
    extra = {}
    if isinstance(instructions, str):
        extra["deployment_instructions"] = instructions

    return _core_build_upload_payload(
        dataset_id,
        dataset_version,
        metadata,
        task_type,
        leaderboard_json,
        name=f"AutoML Model - {dataset_id}",
        description="AutoML trained model for tabular data",
        framework="sklearn",
        extra_fields=extra,
    )
