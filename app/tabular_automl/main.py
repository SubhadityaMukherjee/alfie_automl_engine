"""FastAPI endpoints for tabular AutoML workflows.

Provides endpoints to accept user data/config, validate inputs, store
session metadata, and trigger AutoML training using AutoGluon.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated

import requests
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse

from app.tabular_automl.services import (
    validate_tabular_inputs,
    fetch_dataset_metadata,
    SUPPORTED_FILE_TYPES,
    resolve_download_url,
    download_dataset,
    train_automl,
    serialize_and_zip_predictor,
    convert_leaderboard_safely,
    build_upload_payload,
    upload_model,
)

logger = logging.getLogger(__name__)


load_dotenv(find_dotenv())

app = FastAPI()

TABULAR_AUTOML_PORT = os.getenv("TABULAR_AUTOML_PORT", "http://localhost:8001")
autodw_url = os.getenv("AUTODW_URL", "http://localhost:8000")


@app.post("/automl_tabular/best_model/")
async def find_best_model_for_mvp(
    request: Request,
    user_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_id: Annotated[str, Form(..., description="Dataset id from AutoDW")],
    dataset_version: Annotated[
        str | None, Form(description="Dataset version (e.g., 'v1', 'v2')")
    ] = "",
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
    dataset_split: Annotated[
        str | None,
        Form(description="Dataset split to use for training (e.g., 'train')."),
    ] = None,
) -> JSONResponse:
    """
    Fetch a tabular dataset from AutoDW, run AutoML training, and upload the best model.

    Steps:
      1. Fetch dataset metadata from AutoDW.
      2. Resolve the correct download URL (respecting splits if present).
      3. Download the dataset file to a temporary directory.
      4. Validate user-supplied parameters against the dataset.
      5. Train an AutoML model within the given time budget.
      6. Serialize and zip the best predictor.
      7. Upload the model and leaderboard back to AutoDW.

    Returns:
        200 – success message and leaderboard summary.
        400 – validation error (bad inputs or unsupported file type).
        502 – AutoDW communication failure.
        500 – unexpected runtime error.
    """
    autodw_base = autodw_url
    upload_url = f"{autodw_base}/ai-models/upload/single/{user_id}"

    try:
        # 1. Metadata
        metadata = fetch_dataset_metadata(
            autodw_base, user_id, dataset_id, dataset_version
        )

        file_type = metadata.get("file_type")
        if file_type not in SUPPORTED_FILE_TYPES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type '{file_type}'."},
            )

        # 2. Download URL
        download_url = resolve_download_url(
            autodw_base, user_id, dataset_id, dataset_version, metadata, dataset_split
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / metadata.get("original_filename", "train.csv")

            # 3. Download
            download_dataset(download_url, dataset_path)

            # 4. Validate
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

            # 5. Train
            save_model_path = tmp_path / "automl_model"
            leaderboard, predictor = train_automl(
                dataset_path,
                save_model_path,
                target_column_name,
                task_type,
                time_budget,
            )

            # 6. Serialize
            zip_path = serialize_and_zip_predictor(predictor, save_model_path, tmp_path)
            leaderboard_json, leaderboard_str = convert_leaderboard_safely(leaderboard)

            # 7. Upload
            _, payload = build_upload_payload(
                dataset_id, dataset_version, metadata, task_type, leaderboard_json
            )
            upload_resp = upload_model(
                upload_url, zip_path, payload, request.headers.get("X-Task-ID")
            )

            if upload_resp.status_code >= 400:
                logger.error(f"Model upload failed: {upload_resp.text}")
                return JSONResponse(
                    status_code=upload_resp.status_code,
                    content={"error": f"Failed to upload model: {upload_resp.text}"},
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
