"""FastAPI endpoints for vision AutoML workflows.

Handles session intake (CSV + images), validation, storage, model
selection from Hugging Face Hub, and time-budgeted training.
"""

import datetime
import logging
import os
import shutil
import uuid
from collections.abc import Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker
from torch import nn, optim

from app.core.chat_handler import ChatHandler
from app.vision_automl.ml_engine import (ClassificationData,
                                         ClassificationModel, FabricTrainer)
from app.vision_automl.utils import (
    collect_missing_files, normalize_dataframe_filenames, resolve_images_root,
    save_upload, search_hf_for_pytorch_models_with_estimated_parameters,
    sort_models_by_size)

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


app = FastAPI()

VISION_AUTOML_PORT = os.getenv("VISION_AUTOML_PORT", "http://localhost:8002")
MAX_MODELS_HF = int(os.getenv("MAX_MODELS_HF", 1))

autodw_port_url = os.getenv("AUTODW_DATASETS_PORT", 8000)
autodw_url = os.getenv("AUTODW_URL", "http://localhost:8000")


@app.post("/automl_vision/best_model/")
async def find_best_model_for_vision_mvp(
    user_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_version: Annotated[
        str | None,
        Form(..., description="Optional dataset version selection from AutoDW"),
    ] = None,
    target_column_name: Annotated[
        str, Form(..., description="Name of the target column")
    ] = "",
    filename_column: Annotated[
        str, Form(..., description="Name of the CSV column containing image filenames")
    ] = "",
    target_column_name: Annotated[
        str,
        Form(..., description="Name of the CSV column containing labels or targets"),
    ] = "",
    task_type: Annotated[
        str,
        Form(
            ...,
            description="Type of ML task",
            examples=["classification"],
        ),
    ] = "classification",
    model_size: Annotated[
        str,
        Form(
            ..., description="Desired model size hint (e.g., 'small', 'base', 'large')"
        ),
    ] = "base",
    time_budget: Annotated[int, Form(..., description="Time budget in seconds")] = 10,
    # Task ID will be eventually deprecated. Currently, it is sent to the AutoDW
    # when creating model because AutoDW then sends Kafka task complete message.
    task_id: Annotated[str | None, Header(alias="X-Task-ID")] = None,
) -> JSONResponse:

    autodw_base = autodw_url
    metadata_url = f"{autodw_base}/datasets/{user_id}/{dataset_id}"

    if dataset_version is not None:
        metadata_url = f"{metadata_url}/version/{dataset_version}"

    download_url = f"{metadata_url}/download"
    upload_url = f"{autodw_base}/ai-models/upload/single/{user_id}"

    try:
        # --- Save and validate CSV ---
        csv_path = session_dir / "labels.csv"
        save_upload(csv_file, csv_path)
        df = pd.read_csv(csv_path)

        for col, name in [(filename_column, "Filename"), (label_column, "Label")]:
            if col not in df.columns:
                msg = f"{name} column '{col}' not found in CSV"
                logger.error(msg)
                return JSONResponse(status_code=400, content={"error": msg})

        # --- Save and extract images ---
        images_zip_path = session_dir / "images.zip"
        save_upload(images_zip, images_zip_path)
        images_dir = session_dir / "images"
        images_dir.mkdir(exist_ok=True)
        shutil.unpack_archive(images_zip_path, images_dir)
        images_dir = resolve_images_root(images_dir)

        # --- Normalize and validate ---
        df = normalize_dataframe_filenames(df, filename_column, csv_path)
        missing_files = collect_missing_files(
            df, images_dir, filename_column, label_column
        )
        if missing_files:
            preview = missing_files[:5]
            msg = f"Missing image files: {preview}{'...' if len(missing_files) > 5 else ''}"
            logger.warning("%s (%d missing)", msg, len(missing_files))
            return JSONResponse(
                status_code=400,
                content={"error": msg, "missing_count": len(missing_files)},
            )

        # --- Store session in DB ---
        session_record = AutoMLVisionSession(
            session_id=session_id,
            csv_file_path=str(csv_path),
            images_dir_path=str(images_dir),
            filename_column=filename_column,
            label_column=label_column,
            task_type=task_type,
            time_budget=time_budget,
            model_size=model_size,
        )
        db.add(session_record)
        db.commit()
        logger.info("Session %s stored successfully", session_id)

        # --- Model search and selection ---
        candidates = search_hf_for_pytorch_models_with_estimated_parameters(
            filter="image-classification",
            limit=MAX_MODELS_HF,
        )
        chosen = sort_models_by_size(candidates, model_size)[0] if candidates else None
        model_id = chosen["model_id"] if chosen else "google/vit-base-patch16-224"
        logger.info("Chosen model for session %s: %s", session_id, model_id)

        # --- Initialize DataModule and Trainer ---
        datamodule = ClassificationData(
            csv_file=str(csv_path),
            root_dir=str(images_dir),
            img_col=filename_column,
            label_col=label_column,
            batch_size=64,
            hf_model_id=model_id,
        )
        trainer = FabricTrainer(
            datamodule=datamodule,
            model_class=ClassificationModel,
            model_kwargs={
                "model_id": model_id,
                "num_classes": datamodule.num_classes,
                "id2label": datamodule.id2label,
                "label2id": datamodule.label2id,
            },
            optimizer_class=optim.AdamW,
            optimizer_kwargs={"lr": 0.001},
            loss_fn=nn.CrossEntropyLoss(),
            epochs=50,
            time_limit=float(time_budget),
        )

        # --- Train and return metrics ---
        logger.info("Starting training for session %s", session_id)
        test_loss, test_acc = trainer.fit()
        logger.info("Training completed: loss=%.4f, acc=%.4f", test_loss, test_acc)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Vision AutoML training completed successfully.",
                "session_id": session_id,
                "chosen_model": model_id,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "num_classes": datamodule.num_classes,
            },
        )

    except Exception as e:
        db.rollback()
        logger.exception("Error during vision AutoML MVP processing: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# @app.post("/automl_vision/best_model_mvp/")
# async def find_best_model_for_vision_mvp(
#     csv_file: Annotated[
#         UploadFile,
#         File(..., description="CSV file containing image filenames and labels")
#     ],
#     images_zip: Annotated[
#         UploadFile,
#         File(..., description="ZIP file containing image dataset")
#     ],
#     filename_column: Annotated[
#         str,
#         Form(..., description="Name of the CSV column containing image filenames")
#     ],
#     label_column: Annotated[
#         str,
#         Form(..., description="Name of the CSV column containing labels or targets")
#     ],
#     task_type: Annotated[
#         str,
#         Form(..., description="Type of ML task (e.g., 'classification')", examples=["classification"])
#     ] = "classification",
#     time_budget: Annotated[
#         int,
#         Form(..., description="Time budget for AutoML training in seconds")
#     ] = 600,
#     model_size: Annotated[
#         str,
#         Form(..., description="Desired model size hint (e.g., 'small', 'base', 'large')")
#     ] = "base",
#     db: Session = Depends(get_db),
# ) -> JSONResponse:
#     """
#     ### Endpoint: `/automl_vision/best_model_mvp/`
#
#     **Purpose:**
#     Single-step AutoML pipeline for image datasets.
#     Upload CSV + ZIP + metadata, run validation, choose model, train, and return metrics.
#
#     ---
#     #### Input Parameters
#
#     * **csv_file:** CSV containing image filenames and labels.
#     * **images_zip:** ZIP archive of image files.
#     * **filename_column:** Column in CSV that maps to image filenames.
#     * **label_column:** Column in CSV with class labels.
#     * **task_type:** `"classification"` (others can be extended later).
#     * **time_budget:** Training time in seconds.
#     * **model_size:** `"small" | "base" | "large"`, selects model candidates.
#     ---
#     """
#     session_id = str(uuid.uuid4())
#     session_dir = UPLOAD_ROOT / session_id
#     session_dir.mkdir(parents=True, exist_ok=True)
#     logger.info("Starting new vision AutoML MVP session: %s", session_id)
#
#     try:
#         # --- Save and validate CSV ---
#         csv_path = session_dir / "labels.csv"
#         save_upload(csv_file, csv_path)
#         df = pd.read_csv(csv_path)
#
#         for col, name in [(filename_column, "Filename"), (label_column, "Label")]:
#             if col not in df.columns:
#                 msg = f"{name} column '{col}' not found in CSV"
#                 logger.error(msg)
#                 return JSONResponse(status_code=400, content={"error": msg})
#
#         # --- Save and extract images ---
#         images_zip_path = session_dir / "images.zip"
#         save_upload(images_zip, images_zip_path)
#         images_dir = session_dir / "images"
#         images_dir.mkdir(exist_ok=True)
#         shutil.unpack_archive(images_zip_path, images_dir)
#         images_dir = resolve_images_root(images_dir)
#
#         # --- Normalize and validate ---
#         df = normalize_dataframe_filenames(df, filename_column, csv_path)
#         missing_files = collect_missing_files(df, images_dir, filename_column, label_column)
#         if missing_files:
#             preview = missing_files[:5]
#             msg = f"Missing image files: {preview}{'...' if len(missing_files) > 5 else ''}"
#             logger.warning("%s (%d missing)", msg, len(missing_files))
#             return JSONResponse(
#                 status_code=400,
#                 content={"error": msg, "missing_count": len(missing_files)},
#             )
#
#         # --- Store session in DB ---
#         session_record = AutoMLVisionSession(
#             session_id=session_id,
#             csv_file_path=str(csv_path),
#             images_dir_path=str(images_dir),
#             filename_column=filename_column,
#             label_column=label_column,
#             task_type=task_type,
#             time_budget=time_budget,
#             model_size=model_size,
#         )
#         db.add(session_record)
#         db.commit()
#         logger.info("Session %s stored successfully", session_id)
#
#         # --- Model search and selection ---
#         candidates = search_hf_for_pytorch_models_with_estimated_parameters(
#             filter="image-classification",
#             limit=MAX_MODELS_HF,
#         )
#         chosen = sort_models_by_size(candidates, model_size)[0] if candidates else None
#         model_id = chosen["model_id"] if chosen else "google/vit-base-patch16-224"
#         logger.info("Chosen model for session %s: %s", session_id, model_id)
#
#         # --- Initialize DataModule and Trainer ---
#         datamodule = ClassificationData(
#             csv_file=str(csv_path),
#             root_dir=str(images_dir),
#             img_col=filename_column,
#             label_col=label_column,
#             batch_size=64,
#             hf_model_id=model_id,
#         )
#         trainer = FabricTrainer(
#             datamodule=datamodule,
#             model_class=ClassificationModel,
#             model_kwargs={
#                 "model_id": model_id,
#                 "num_classes": datamodule.num_classes,
#                 "id2label": datamodule.id2label,
#                 "label2id": datamodule.label2id,
#             },
#             optimizer_class=optim.AdamW,
#             optimizer_kwargs={"lr": 0.001},
#             loss_fn=nn.CrossEntropyLoss(),
#             epochs=50,
#             time_limit=float(time_budget),
#         )
#
#         # --- Train and return metrics ---
#         logger.info("Starting training for session %s", session_id)
#         test_loss, test_acc = trainer.fit()
#         logger.info("Training completed: loss=%.4f, acc=%.4f", test_loss, test_acc)
#
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "message": "Vision AutoML training completed successfully.",
#                 "session_id": session_id,
#                 "chosen_model": model_id,
#                 "test_loss": test_loss,
#                 "test_accuracy": test_acc,
#                 "num_classes": datamodule.num_classes,
#             },
#         )
#
#     except Exception as e:
#         db.rollback()
#         logger.exception("Error during vision AutoML MVP processing: %s", e)
#         return JSONResponse(status_code=500, content={"error": str(e)})
