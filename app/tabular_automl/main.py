"""FastAPI endpoints for tabular AutoML workflows.

Provides endpoints to accept user data/config, validate inputs, store
session metadata, and trigger AutoML training using AutoGluon.
"""

import json
import logging
import os
import pickle
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.chat_handler import ChatHandler
from app.tabular_automl.modules import AutoMLTrainer
from app.tabular_automl.services import (create_session_directory, load_table,
                                         save_upload, store_session_in_db,
                                         validate_tabular_inputs)

logger = logging.getLogger(__name__)


load_dotenv(find_dotenv())

app = FastAPI()

TABULAR_AUTOML_PORT = os.getenv("TABULAR_AUTOML_PORT", "http://localhost:8001")
autodw_port_url = os.getenv("AUTODW_DATASETS_PORT", 8000)
autodw_url = os.getenv("AUTODW_URL", "http://localhost:8000")


def convert_leaderboard_safely(leaderboard):
    if isinstance(leaderboard, pd.DataFrame):
        leaderboard_json = leaderboard.to_dict(orient="records")
        leaderboard_str = leaderboard.to_markdown()
    else:
        leaderboard_json = {"result": str(leaderboard)}
    leaderboard_str = str(leaderboard)
    return leaderboard_json, leaderboard_str


@app.post("/automl_tabular/best_model/")
async def find_best_model_for_mvp(
    request: Request,
    user_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_version: Annotated[
        str | None,
        Form(..., description="Optional dataset version selection from AutoDW"),
    ] = None,
    target_column_name: Annotated[
        str, Form(..., description="Name of the target column")
    ] = "",
    time_stamp_column_name: Annotated[
        str | None,
        Form(..., description="Timestamp column (required for time-series tasks)"),
    ] = None,
    task_type: Annotated[
        str,
        Form(
            ...,
            description="Type of ML task",
            examples=["classification", "regression", "time_series"],
        ),
    ] = "classification",
    time_budget: Annotated[int, Form(..., description="Time budget in seconds")] = 10,
    # Task ID will be eventually deprecated. Currently, it is sent to the AutoDW
    # when creating model because AutoDW then sends Kafka task complete message.
) -> JSONResponse:
    """
    Fetch dataset metadata and file from AutoDW, validate it,
    and run AutoML training to find and upload the best model.
    Train and upload an AutoML model for a given tabular dataset retrieved from AutoDW.

    This endpoint:
      1. Fetches dataset metadata and file from AutoDW using the provided user and dataset IDs.
      2. Validates dataset integrity and user-specified parameters such as target column,
         timestamp column (if applicable), and task type.
      3. Trains an AutoML model on the dataset within a specified time budget.
      4. Serializes and uploads the best-performing model and leaderboard results back to AutoDW.

    Args:
        user_id (str): Unique user identifier from AutoDW.
        dataset_id (str): Unique dataset identifier from AutoDW.
        dataset_version (str): Unique dataset version from AutoDW.
        target_column_name (str): Name of the target column in the dataset.
        time_stamp_column_name (str | None): Name of the timestamp column for time-series tasks.
        task_type (str): Type of ML task. One of {"classification", "regression", "time_series"}.
        time_budget (int): Time budget for AutoML training, in seconds.

    Returns:
        JSONResponse: A structured response indicating success or failure.
            - On success (200): Returns a success message and leaderboard summary.
            - On validation error (400): Returns an error message describing the invalid input.
            - On AutoDW communication failure (502): Returns an error indicating network issues.
            - On unexpected failure (500): Returns a general error description.

    Raises:
        requests.RequestException: If AutoDW metadata or dataset requests fail.
        Exception: For unexpected runtime or training-related errors.

    Example:
        HTTP POST /automl_tabular/best_model/
        Form data:
            user_id=101
            dataset_id=55
            target_column_name="label"
            task_type="classification"
            time_budget=600

    Notes:
        - Only "csv", "tsv", and "parquet" dataset formats are supported.
        - A temporary directory is used for dataset download, model training,
          and serialization before upload.
        - The leaderboard is returned both as a markdown string and as JSON
          when uploaded to AutoDW.
    """

    autodw_base = autodw_url
    metadata_url = f"{autodw_base}/datasets/{user_id}/{dataset_id}"

    if dataset_version is not None:
        metadata_url = f"{metadata_url}/version/{dataset_version}"

    download_url = f"{metadata_url}/download"
    upload_url = f"{autodw_base}/ai-models/upload/single/{user_id}"

    try:
        # --- 1. Fetch dataset metadata ---
        logger.debug(f"Fetching dataset metadata: {metadata_url}")
        metadata_response = requests.get(metadata_url, timeout=15)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()

        file_type = metadata.get("file_type")
        original_filename = metadata.get("original_filename", "train.csv")

        if file_type not in {"csv", "tsv", "parquet"}:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type '{file_type}'."},
            )

        # --- 2. Download dataset file ---
        logger.debug(f"Downloading dataset file: {download_url}")
        with requests.get(download_url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                dataset_path = tmp_path / original_filename

                with open(dataset_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)

                logger.info(f"Dataset saved to {dataset_path}")

                # --- 3. Validate inputs ---
                validation_error = validate_tabular_inputs(
                    train_path=dataset_path,
                    target_column_name=target_column_name,
                    time_stamp_column_name=time_stamp_column_name,
                    task_type=task_type,
                )
                if validation_error:
                    return JSONResponse(
                        status_code=400, content={"error": validation_error}
                    )

                # --- 4. Train AutoML model ---
                save_model_path = tmp_path / "automl_model"
                os.makedirs(save_model_path, exist_ok=True)

                trainer = AutoMLTrainer(save_model_path=save_model_path)
                train_df = load_table(dataset_path)

                leaderboard, predictor = trainer.train(
                    train_df=train_df,
                    test_df=None,
                    target_column=target_column_name,
                    time_limit=int(time_budget),
                )

                # --- 5. Serialize predictor ---
                predictor_path = save_model_path / "predictor.pkl"
                with open(predictor_path, "wb") as f:
                    pickle.dump(predictor, f)

                zip_path = tmp_path / "automl_predictor.zip"
                _ = shutil.make_archive(
                    base_name=str(zip_path).replace(".zip", ""),
                    format="zip",
                    root_dir=save_model_path,
                )

                leaderboard_json, leaderboard_str = convert_leaderboard_safely(
                    leaderboard
                )

                # --- 6. Upload trained model to AutoDW ---
                model_id = f"automl_{dataset_id}_{int(datetime.utcnow().timestamp())}"
                # Get X-Task-ID from request headers if present
                task_id = request.headers.get("X-Task-ID")
                headers = {}
                if task_id:
                    headers["X-Task-ID"] = task_id
                    logger.debug(f"Including X-Task-ID header: {task_id}")


                with open(zip_path, "rb") as f:
                    files = {"file": (zip_path.name, f, "application/octet-stream")}
                    data = {
                        "model_id": model_id,
                        "name": f"AutoML Model - {model_id}",
                        "description": "AutoML trained model for tabular data",
                        "framework": "sklearn",
                        "model_type": task_type,
                        "training_dataset": str(dataset_id),
                        "leaderboard": json.dumps(leaderboard_json),  # ensure JSON-safe
                    }

                    logger.debug(f"Uploading model to {upload_url}")
                    upload_resp = requests.post(
                        upload_url, headers=headers, files=files, data=data, timeout=120
                    )

                    if upload_resp.status_code >= 400:
                        logger.error(f"Model upload failed: {upload_resp.text}")
                        return JSONResponse(
                            status_code=upload_resp.status_code,
                            content={
                                "error": f"Failed to upload model: {upload_resp.text}"
                            },
                        )

        logger.info("AutoML training completed and model uploaded successfully.")
        return JSONResponse(
            status_code=200,
            content={
                "message": "AutoML training completed successfully and model uploaded to AutoDW",
                "leaderboard": leaderboard_str,
            },
        )

    except requests.RequestException as e:
        logger.exception("Network or HTTP error during AutoDW communication")
        return JSONResponse(
            status_code=502, content={"error": f"AutoDW communication failed: {e}"}
        )
    except Exception as e:
        logger.exception("Unexpected error during AutoML training or upload")
        return JSONResponse(status_code=500, content={"error": str(e)})
