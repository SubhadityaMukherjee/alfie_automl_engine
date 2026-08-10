"""Route definitions for the tabular AutoML service."""

import logging
import tempfile
from pathlib import Path
from typing import Annotated, Any

import requests
from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse

from app.core.concurrency import offload
from app.core.config import get_settings
from app.core.exceptions import AutoMLRuntimeError, AutoMLValidationError
from app.core.schemas.responses import (
    ErrorResponse,
    InstructionsResponse,
    TrainingSuccessResponse,
)
from app.tabular_automl.models import SUPPORTED_TABULAR_TASK_TYPES
from app.tabular_automl.services import (
    SUPPORTED_FILE_TYPES,
    build_upload_payload,
    convert_leaderboard_safely,
    deployment_instructions,
    download_dataset,
    fetch_dataset_metadata,
    resolve_download_url,
    serialize_and_zip_predictor,
    tabular_data_instructions,
    train_automl,
    upload_model,
    validate_tabular_inputs,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/automl_tabular", tags=["tabular"])

_COMMON_RESPONSES: dict[int | str, dict[str, Any]] = {
    500: {"description": "Internal server error", "model": ErrorResponse},
}


@router.post(
    "/deployment_instructions/",
    response_model=InstructionsResponse,
    responses=_COMMON_RESPONSES,
)
async def show_deployment_instructions() -> JSONResponse:
    """Show deployment instructions from a template"""
    try:
        return JSONResponse(
            content={"instructions": deployment_instructions()}, status_code=200
        )
    except Exception as e:
        logger.exception(
            "Unexpected error in finding deployment instructions in tabular"
        )
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post(
    "/accepted_format/",
    response_model=InstructionsResponse,
    responses=_COMMON_RESPONSES,
)
async def show_accepted_format_instructions() -> JSONResponse:
    """Show accepted format instructions from a template"""
    try:
        return JSONResponse(
            content={"instructions": tabular_data_instructions()}, status_code=200
        )
    except Exception as e:
        logger.exception(
            "Unexpected error in finding data format instructions in tabular"
        )
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post(
    "/best_model/",
    response_model=TrainingSuccessResponse,
    responses={
        400: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
        502: {"description": "AutoDW communication failure", "model": ErrorResponse},
    },
)
async def find_best_model_for_mvp(
    request: Request,
    user_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_id: Annotated[str, Form(..., description="Dataset id from AutoDW")],
    dataset_version: Annotated[
        str | None, Form(description="Dataset version (e.g., 'v1', 'v2')")
    ] = "v1",
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
            examples=SUPPORTED_TABULAR_TASK_TYPES,
        ),
    ] = "classification",
    time_budget: Annotated[int, Form(..., description="Time budget in seconds")] = 10,
    num_cpus: Annotated[
        int | str,
        Form(
            ...,
            description="Number of CPUs to use for AutoML. Can be a number or 'auto' ",
        ),
    ] = "auto",
    num_gpus: Annotated[
        int | str,
        Form(
            ...,
            description="Number of GPUs to use for AutoML. Can be a number or 'auto' ",
        ),
    ] = "auto",
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
    if not user_id or not isinstance(user_id, str) or not user_id.strip():
        return JSONResponse(
            status_code=400, content={"error": "user_id must be a non-empty string"}
        )

    if not dataset_id or not isinstance(dataset_id, str) or not dataset_id.strip():
        return JSONResponse(
            status_code=400, content={"error": "dataset_id must be a non-empty string"}
        )
    if isinstance(num_cpus, str):
        if num_cpus != "auto":
            return JSONResponse(
                status_code=400,
                content={"error": "num_cpus must be either a number or auto"},
            )
    else:
        if num_cpus < 1:
            return JSONResponse(
                status_code=400,
                content={"error": "num_cpus must be a positive integer (at least 1)"},
            )

    if isinstance(num_gpus, str):
        if num_gpus != "auto":
            return JSONResponse(
                status_code=400,
                content={"error": "num_gpus must be either a number or auto"},
            )
    else:
        if num_gpus < 1:
            return JSONResponse(
                status_code=400,
                content={"error": "num_gpus must be a positive integer (at least 1)"},
            )

    if (
        not target_column_name
        or not isinstance(target_column_name, str)
        or not target_column_name.strip()
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "target_column_name must be a non-empty string"},
        )

    if task_type not in SUPPORTED_TABULAR_TASK_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid task_type '{task_type}'. Must be one of: {SUPPORTED_TABULAR_TASK_TYPES}"
            },
        )

    if not isinstance(time_budget, int) or time_budget <= 0:
        return JSONResponse(
            status_code=400, content={"error": "time_budget must be a positive integer"}
        )

    if dataset_split is not None and dataset_split not in ("train", "test", "drift"):
        return JSONResponse(
            status_code=400,
            content={"error": "dataset_split must be one of: 'train', 'test', 'drift'"},
        )
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
    autodw_base = get_settings().autodw_url
    upload_url = f"{autodw_base}/ai-models/upload/single/{user_id}"

    try:
        if not autodw_base:
            return JSONResponse(
                status_code=500,
                content={"error": "AUTODW_URL environment variable is not set"},
            )

        # 1. Metadata
        try:
            metadata = await offload(
                fetch_dataset_metadata,
                autodw_base,
                user_id,
                dataset_id,
                dataset_version,
            )
        except requests.RequestException as e:
            logger.error("Failed to fetch dataset metadata: %s", e)
            return JSONResponse(
                status_code=502,
                content={"error": f"Failed to fetch dataset metadata from AutoDW: {e}"},
            )
        except Exception as e:
            logger.error("Unexpected error fetching metadata: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": f"Unexpected error fetching metadata: {e}"},
            )

        if not isinstance(metadata, dict) or not metadata:
            return JSONResponse(
                status_code=502,
                content={"error": "Invalid or empty metadata received from AutoDW"},
            )

        file_type = metadata.get("file_type")
        if not file_type:
            return JSONResponse(
                status_code=400,
                content={"error": "file_type not found in dataset metadata"},
            )

        if file_type not in SUPPORTED_FILE_TYPES:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Unsupported file type '{file_type}'. Supported types: {SUPPORTED_FILE_TYPES}"
                },
            )

        # 2. Download URL
        try:
            download_url = resolve_download_url(
                autodw_base,
                user_id,
                dataset_id,
                dataset_version,
                metadata,
                dataset_split,
            )
        except Exception as e:
            logger.error("Failed to resolve download URL: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to resolve download URL: {e}"},
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            original_filename = metadata.get("original_filename", "train.csv")
            if not isinstance(original_filename, str) or not original_filename:
                original_filename = "train.csv"
            dataset_path = tmp_path / original_filename

            # 3. Download
            try:
                await offload(download_dataset, download_url, dataset_path)
            except requests.RequestException as e:
                logger.error("Failed to download dataset: %s", e)
                return JSONResponse(
                    status_code=502,
                    content={"error": f"Failed to download dataset from AutoDW: {e}"},
                )
            except Exception as e:
                logger.error("Unexpected error downloading dataset: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Unexpected error downloading dataset: {e}"},
                )

            if not dataset_path.exists():
                return JSONResponse(
                    status_code=500,
                    content={"error": "Dataset file was not created after download"},
                )

            # 4. Validate
            try:
                validation_error = await offload(
                    validate_tabular_inputs,
                    train_path=dataset_path,
                    target_column_name=target_column_name,
                    time_stamp_column_name=time_stamp_column_name,
                    task_type=task_type,
                )
                if validation_error:
                    return JSONResponse(
                        status_code=400, content={"error": validation_error}
                    )
            except Exception as e:
                logger.error("Unexpected error during validation: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Unexpected error during validation: {e}"},
                )

            # 5. Train
            try:
                save_model_path = tmp_path / "automl_model"
                leaderboard, predictor = await offload(
                    train_automl,
                    dataset_path=dataset_path,
                    save_model_path=save_model_path,
                    target_column_name=target_column_name,
                    task_type=task_type,
                    time_budget=time_budget,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                )
            except AutoMLValidationError as e:
                logger.error("Validation error during training: %s", e)
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Training validation failed: {e}"},
                )
            except AutoMLRuntimeError as e:
                logger.error("Training runtime error: %s", e)
                return JSONResponse(
                    status_code=500, content={"error": f"Model training failed: {e}"}
                )
            except Exception as e:
                logger.error("Unexpected error during training: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Unexpected error during training: {e}"},
                )

            if leaderboard is None:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Training completed but leaderboard is empty"},
                )

            # 6. Serialize
            try:
                zip_path = await offload(
                    serialize_and_zip_predictor, predictor, save_model_path, tmp_path
                )
                leaderboard_json, leaderboard_str = convert_leaderboard_safely(
                    leaderboard
                )
            except Exception as e:
                logger.error("Failed to serialize and zip model: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to serialize model: {e}"},
                )

            if not zip_path.exists():
                return JSONResponse(
                    status_code=500, content={"error": "Model zip file was not created"}
                )

            # 7. Upload
            try:
                _, payload = build_upload_payload(
                    dataset_id, dataset_version, metadata, task_type, leaderboard_json
                )
            except Exception as e:
                logger.error("Failed to build upload payload: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to build upload payload: {e}"},
                )

            try:
                upload_resp = await offload(
                    upload_model,
                    upload_url,
                    zip_path,
                    payload,
                    request.headers.get("X-Task-ID"),
                )

                if upload_resp.status_code >= 400:
                    logger.error("Model upload failed: %s", upload_resp.text)
                    return JSONResponse(
                        status_code=upload_resp.status_code,
                        content={
                            "error": f"Failed to upload model: {upload_resp.text}"
                        },
                    )
            except requests.RequestException as e:
                logger.error("Network error uploading model: %s", e)
                return JSONResponse(
                    status_code=502,
                    content={"error": f"Failed to upload model to AutoDW: {e}"},
                )
            except Exception as e:
                logger.error("Unexpected error uploading model: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Unexpected error uploading model: {e}"},
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
