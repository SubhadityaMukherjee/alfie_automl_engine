"""FastAPI endpoints for vision AutoML workflows.

Handles dataset fetch, validation, model training, and upload.
Structure mirrors tabular_automl/main.py exactly so both pipelines
share a consistent API shape.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated
import shutil
import tempfile

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse

from app.vision_automl.services import (
    DatasetValidationError,
    AutodwError,
    fetch_dataset_metadata,
    resolve_download_url,
    download_dataset,
    extract_and_locate_dataset,
    validate_vision_inputs,
    train_automl,
    serialize_and_zip_model,
    convert_leaderboard_safely,
    build_upload_payload,
    upload_model,
)

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

app = FastAPI()

VISION_AUTOML_PORT = os.getenv("VISION_AUTOML_PORT", "http://localhost:8002")
autodw_url = os.getenv("AUTODW_URL", "http://localhost:8000")


@contextmanager
def dataset_workspace(prefix: str):
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@app.post("/automl_vision/best_model/")
async def find_best_model_for_vision(
    request: Request,
    user_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_id: Annotated[str, Form(..., description="Dataset id from AutoDW")],
    dataset_version: Annotated[
        str | None, Form(description="Optional dataset version")
    ] = None,
    filename_column: Annotated[
        str, Form(..., description="Filename column in labels.csv")
    ] = "filename",
    label_column: Annotated[
        str, Form(..., description="Label column in labels.csv")
    ] = "label",
    task_type: Annotated[
        str, Form(..., description="Vision task type")
    ] = "classification",
    time_budget: Annotated[int, Form(..., description="Time budget in seconds")] = 60,
    model_size: Annotated[
        str, Form(..., description="Model size: small / medium / large")
    ] = "small",
    dataset_split: Annotated[
        str | None,
        Form(description="Dataset split to use for training (e.g., 'train')."),
    ] = None,
) -> JSONResponse:
    """
    Fetch a vision dataset from AutoDW, run AutoML training, and upload the best model.

    Steps:
      1. Fetch dataset metadata from AutoDW.
      2. Resolve the correct download URL (respecting splits if present).
      3. Download the dataset ZIP to a temporary directory and extract it.
      4. Validate CSV structure and image file presence.
      5. Train a vision AutoML model within the given time budget.
      6. Zip the model artifacts.
      7. Upload the model and leaderboard back to AutoDW.

    Returns:
        200 – success message and leaderboard summary.
        400 – validation error (bad inputs or unsupported dataset).
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

        if metadata.get("file_type") != "zip":
            return JSONResponse(
                status_code=400,
                content={"error": "Vision AutoML requires a ZIP dataset."},
            )

        # 2. Download URL
        download_url = resolve_download_url(
            autodw_base, user_id, dataset_id, dataset_version, metadata, dataset_split
        )

        with dataset_workspace(f"automl_{dataset_id}") as workdir:
            # 3. Download & extract
            zip_path = download_dataset(
                download_url, workdir, metadata.get("original_filename", "dataset.zip")
            )
            csv_path, images_dir = extract_and_locate_dataset(zip_path, workdir)

            # 4. Validate
            validation_error = validate_vision_inputs(
                csv_path, images_dir, filename_column, label_column
            )
            if validation_error:
                return JSONResponse(
                    status_code=400, content={"error": validation_error}
                )

            # 5. Train
            optuna_result = await train_automl(
                csv_path,
                images_dir,
                filename_column,
                label_column,
                time_budget,
                model_size,
                workdir=workdir,
            )

            # 6. Serialize
            zip_path = serialize_and_zip_model(optuna_result, workdir)
            leaderboard_json, leaderboard_str = convert_leaderboard_safely(
                optuna_result
            )

            # 7. Upload
            _, payload = build_upload_payload(
                dataset_id, dataset_version, metadata, task_type, leaderboard_json
            )
            upload_resp = upload_model(
                upload_url, zip_path, payload, request.headers.get("X-Task-ID")
            )

            if upload_resp.status_code >= 400:
                logger.error("Model upload failed: %s", upload_resp.text)
                return JSONResponse(
                    status_code=upload_resp.status_code,
                    content={"error": f"Failed to upload model: {upload_resp.text}"},
                )

        logger.info("Vision AutoML training completed and model uploaded successfully.")
        return JSONResponse(
            status_code=200,
            content={
                "message": "Vision AutoML training completed successfully and model uploaded to AutoDW",
                "leaderboard": leaderboard_str,
            },
        )

    except DatasetValidationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except AutodwError as e:
        return JSONResponse(status_code=502, content={"error": f"AutoDW error: {e}"})
    except Exception as e:
        logger.exception("Unexpected error during vision AutoML")
        return JSONResponse(status_code=500, content={"error": str(e)})
