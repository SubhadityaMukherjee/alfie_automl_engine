"""FastAPI endpoints for vision AutoML workflows.

Handles session intake (CSV + images), validation, storage, model
selection from Hugging Face Hub, and time-budgeted training.
"""

import logging
import os
from typing import Annotated

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
from app.vision_automl.utils import (
    AutodwError,
    DatasetValidationError,
    _fetch_and_extract_dataset,
    _package_model_artifacts,
    _prepare_model_metadata,
    _run_automl_optimization,
    _upload_model_to_autodw,
    _validate_dataset_structure,
)

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


app = FastAPI()

VISION_AUTOML_PORT = os.getenv("VISION_AUTOML_PORT", "http://localhost:8002")
MAX_MODELS_HF = int(os.getenv("MAX_MODELS_HF", 1))

autodw_port_url = os.getenv("AUTODW_DATASETS_PORT", 8000)
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
        str | None, Form(..., description="Optional dataset version")
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
        str, Form(..., description="Model size: small/medium/large")
    ] = "small",
) -> JSONResponse:
    """
    Optimized Vision AutoML endpoint that finds and trains the best model.
    """
    try:
        with dataset_workspace(f"automl_{dataset_id}") as workdir:
            csv_path, images_dir = await _fetch_and_extract_dataset(
                user_id, dataset_id, dataset_version, workdir
            )

            csv_path, images_dir = _validate_dataset_structure(
                csv_path, images_dir, filename_column, label_column
            )

            optuna_result = await _run_automl_optimization(
                csv_path,
                images_dir,
                filename_column,
                label_column,
                time_budget,
                model_size,
                workdir=workdir,
            )

            model_zip_path = _package_model_artifacts(optuna_result, workdir)

            model_metadata = _prepare_model_metadata(
                dataset_id, optuna_result, task_type
            )

            upload_result = _upload_model_to_autodw(
                model_zip_path, model_metadata, request.headers.get("X-Task-ID")
            )
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Vision AutoML training completed successfully",
                    "best_loss": optuna_result["best_value"],
                    "best_params": optuna_result["best_params"],
                    "trials": optuna_result["n_trials"],
                    "model_id": upload_result["model_id"],
                },
            )

    except DatasetValidationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except AutodwError as e:
        return JSONResponse(status_code=502, content={"error": f"AutoDW error: {e}"})
    except Exception as e:
        logger.exception("Unexpected error during vision AutoML")
        return JSONResponse(status_code=500, content={"error": str(e)})
