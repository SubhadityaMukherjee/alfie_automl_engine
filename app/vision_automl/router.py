"""Route definitions for the vision AutoML service."""

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AutoDWDownloadError, AutoMLValidationError
from app.vision_automl.models import SUPPORTED_VISION_TASK_TYPES
from app.vision_automl.services import (
    build_upload_payload,
    convert_leaderboard_safely,
    deployment_instructions,
    download_dataset,
    extract_and_locate_dataset,
    fetch_dataset_metadata,
    resolve_download_url,
    serialize_and_zip_model,
    train_automl,
    train_automl_multimodal,
    upload_model,
    validate_multimodal_inputs,
    validate_vision_inputs,
    vision_data_instructions,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/automl_vision", tags=["vision"])


@contextmanager
def dataset_workspace(prefix: str):
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@router.post("/deployment_instructions/")
async def show_deployment_instructions() -> JSONResponse:
    """Show deployment instructions from a template"""
    try:
        return JSONResponse(
            content={"instructions": deployment_instructions()}, status_code=200
        )
    except Exception as e:
        logger.exception(
            "Unexpected error in finding deployment instructions in vision"
        )
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/accepted_format/")
async def show_accepted_format_instructions() -> JSONResponse:
    """Show accepted format instructions from a template"""
    try:
        return JSONResponse(
            content={"instructions": vision_data_instructions()}, status_code=200
        )
    except Exception as e:
        logger.exception(
            "Unexpected error in finding data format instructions in vision"
        )
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/best_model/")
async def find_best_model_for_vision(
    request: Request,
    user_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_id: Annotated[str, Form(..., description="Dataset id from AutoDW")],
    dataset_version: Annotated[
        str | None, Form(description="Optional dataset version")
    ] = "v1",
    filename_column: Annotated[
        str, Form(..., description="Filename column in labels.csv")
    ] = "filename",
    label_column: Annotated[
        str, Form(..., description="Label column in labels.csv")
    ] = "label",
    task_type: Annotated[
        str,
        Form(
            description=(
                "Vision task type. One of: "
                + ", ".join(sorted(SUPPORTED_VISION_TASK_TYPES))
            )
        ),
    ] = "image_classification",
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
    autodw_base = os.getenv("AUTODW_URL", "http://localhost:8000")
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
            if task_type not in SUPPORTED_VISION_TASK_TYPES:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unsupported task_type '{task_type}'. "
                        f"Supported: {sorted(SUPPORTED_VISION_TASK_TYPES)}"
                    },
                )

            validation_error = validate_vision_inputs(
                csv_path, images_dir, filename_column, label_column, task_type
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
                task_type=task_type,
            )

            # 6. Serialize
            zip_path = serialize_and_zip_model(workdir)
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

    except AutoMLValidationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except AutoDWDownloadError as e:
        return JSONResponse(status_code=502, content={"error": f"AutoDW error: {e}"})
    except Exception as e:
        logger.exception("Unexpected error during vision AutoML")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/multimodal_best_model/")
async def find_best_model_for_multimodal_vision(
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
    exclude_columns: Annotated[
        str | None,
        Form(
            description=(
                "Comma-separated list of CSV columns to exclude from auxiliary features "
                "(in addition to filename_column and label_column)."
            )
        ),
    ] = None,
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
    Fetch a multimodal vision dataset from AutoDW, run AutoML training
    using both image data and auxiliary tabular metadata columns, and
    upload the best model.

    Auxiliary columns are auto-detected from the CSV: all columns except
    ``filename_column``, ``label_column``, and any columns listed in
    ``exclude_columns`` are used as tabular features.  Numeric columns
    are standard-scaled and categorical columns are ordinal-encoded.

    Steps:
      1. Fetch dataset metadata from AutoDW.
      2. Resolve the correct download URL (respecting splits if present).
      3. Download the dataset ZIP to a temporary directory and extract it.
      4. Auto-discover auxiliary columns and validate dataset structure.
      5. Train a multimodal vision AutoML model within the given time budget.
      6. Zip the model artifacts.
      7. Upload the model and leaderboard back to AutoDW.

    Returns:
        200 – success message and leaderboard summary.
        400 – validation error (bad inputs or unsupported dataset).
        502 – AutoDW communication failure.
        500 – unexpected runtime error.
    """
    autodw_base = os.getenv("AUTODW_URL", "http://localhost:8000")
    upload_url = f"{autodw_base}/ai-models/upload/single/{user_id}"

    try:
        metadata = fetch_dataset_metadata(
            autodw_base, user_id, dataset_id, dataset_version
        )

        if metadata.get("file_type") != "zip":
            return JSONResponse(
                status_code=400,
                content={"error": "Vision AutoML requires a ZIP dataset."},
            )

        download_url = resolve_download_url(
            autodw_base, user_id, dataset_id, dataset_version, metadata, dataset_split
        )

        with dataset_workspace(f"multimodal_{dataset_id}") as workdir:
            zip_path = download_dataset(
                download_url, workdir, metadata.get("original_filename", "dataset.zip")
            )
            csv_path, images_dir = extract_and_locate_dataset(zip_path, workdir)

            exclude_cols = (
                [c.strip() for c in exclude_columns.split(",") if c.strip()]
                if exclude_columns
                else None
            )

            validation_error, auxiliary_columns = validate_multimodal_inputs(
                csv_path, images_dir, filename_column, label_column, exclude_cols
            )
            if validation_error:
                return JSONResponse(
                    status_code=400, content={"error": validation_error}
                )

            optuna_result = await train_automl_multimodal(
                csv_path,
                images_dir,
                filename_column,
                label_column,
                auxiliary_columns,
                time_budget,
                model_size,
                workdir=workdir,
            )

            zip_path = serialize_and_zip_model(workdir)
            leaderboard_json, leaderboard_str = convert_leaderboard_safely(
                optuna_result
            )

            task_type = "image_classification_multimodal"
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

        logger.info(
            "Multimodal vision AutoML training completed and model uploaded successfully."
        )
        return JSONResponse(
            status_code=200,
            content={
                "message": "Multimodal vision AutoML training completed successfully and model uploaded to AutoDW",
                "leaderboard": leaderboard_str,
                "auxiliary_columns": auxiliary_columns,
            },
        )

    except AutoMLValidationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except AutoDWDownloadError as e:
        return JSONResponse(status_code=502, content={"error": f"AutoDW error: {e}"})
    except Exception as e:
        logger.exception("Unexpected error during multimodal vision AutoML")
        return JSONResponse(status_code=500, content={"error": str(e)})
