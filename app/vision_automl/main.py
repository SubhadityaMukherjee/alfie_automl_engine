"""FastAPI endpoints for vision AutoML workflows.

Handles session intake (CSV + images), validation, storage, model
selection from Hugging Face Hub, and time-budgeted training.
"""

import datetime
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast
from collections.abc import Generator

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, Mapped, mapped_column
from torch import nn, optim
from typing import Annotated, Any

from app.core.chat_handler import ChatHandler
from app.vision_automl.ml_engine import (ClassificationData,
                                         ClassificationModel, FabricTrainer)

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

# app initialized after lifespan definition below

VISION_AUTOML_PORT = os.getenv("VISION_AUTOML_PORT", "http://localhost:8002")
DATABASE_URL = os.getenv("VISION_DATABASE_CONFIG", "sqlite:///automl_sessions.db")
MAX_MODELS_HF = int(os.getenv("MAX_MODELS_HF", 1))
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# NOTE: Tables are created after all models are declared further below

# NOTE: replace with autodw path later
UPLOAD_ROOT = Path("uploaded_data")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    await ChatHandler.init()
    yield
    # Cleanup resources
    pass


# Initialize FastAPI app after lifespan function is defined
app = FastAPI(lifespan=lifespan)


# NOTE: I AM NOT SURE IF THE AUTODW WILL HANDLE THIS PART FIRST :/
class SessionRequest(BaseModel):
    """Payload for initiating model search/training for a vision session."""

    session_id: str

class AutoMLVisionSession(Base):
    """SQLAlchemy model for vision AutoML session metadata."""

    __tablename__: str = "automl_vision_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    csv_file_path: Mapped[str] = mapped_column(String, nullable=False)
    images_dir_path: Mapped[str] = mapped_column(String, nullable=False)
    filename_column: Mapped[str] = mapped_column(String, nullable=False)
    label_column: Mapped[str] = mapped_column(String, nullable=False)
    task_type: Mapped[str] = mapped_column(String, nullable=False)  # e.g. classification
    time_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    model_size: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
# Create database tables after all models are defined
Base.metadata.create_all(bind=engine)


# -----------------------------
# Database dependency
# -----------------------------
def get_db() -> Generator[Session, None, None]:
    """Provide a SQLAlchemy session scoped to the request lifespan."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_dataframe_filenames(
    df: pd.DataFrame, filename_column: str, csv_path: Path
) -> pd.DataFrame:
    """Normalize filenames to basenames and persist CSV back to disk."""
    logger.info("Normalizing filenames in column '%s'", filename_column)
    if filename_column in df.columns:
        df[filename_column] = (
            df[filename_column]
            .astype(str)
            .map(lambda s: os.path.basename(str(s).replace("\\", "/")))
        )
        df.to_csv(csv_path, index=False)
        logger.debug("Normalized filenames saved to %s", csv_path)
    else:
        logger.warning("Filename column '%s' not found during normalization", filename_column)
    return df


def resolve_images_root(images_dir: Path) -> Path:
    """Resolve common nested packaging patterns inside uploaded image zips."""
    logger.info("Resolving image directory structure at %s", images_dir)
    nested_images_dir = images_dir / "images"
    if nested_images_dir.exists() and nested_images_dir.is_dir():
        logger.debug("Detected nested 'images' folder, using it as root")
        images_dir = nested_images_dir

    try:
        top_level_entries = list(images_dir.iterdir())
        only_dirs = [p for p in top_level_entries if p.is_dir()]
        only_files = [p for p in top_level_entries if p.is_file()]
        if len(only_files) == 0 and len(only_dirs) == 1:
            logger.debug("Detected single top-level directory: %s", only_dirs[0])
            images_dir = only_dirs[0]
    except Exception as e:
        logger.warning("Error resolving image root: %s", e)

    return images_dir


def collect_missing_files(
    df: pd.DataFrame,
    images_dir: Path,
    filename_column: str,
    label_column: str,
) -> list[str]:
    """Return list of filenames that do not exist in the extracted images."""
    logger.info("Checking for missing image files...")
    missing_files: list[str] = []
    for _, row in df.iterrows():
        raw_filename = str(row[filename_column])
        label = str(row[label_column])
        basename = os.path.basename(raw_filename.replace("\\", "/"))

        candidates = [
            images_dir / label / basename,
            images_dir / basename,
            images_dir / raw_filename,
        ]

        if any(path.exists() for path in candidates):
            continue

        try:
            found_any = next(images_dir.rglob(basename), None) is not None
        except Exception as e:
            logger.debug("Error searching recursively for %s: %s", basename, e)
            found_any = False

        if not found_any:
            missing_files.append(raw_filename)

    logger.info("Missing %d files", len(missing_files))
    return missing_files


def get_num_params_if_available(repo_id: str, revision: str|None = None) -> int|None:
    """Try to retrieve number of parameters for a HF model, if available."""
    logger.debug("Fetching parameter count for model %s", repo_id)
    api = HfApi()
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        num_params = getattr(info, "safetensors", None)
        if num_params is not None:
            return num_params.total
    except Exception as e:
        logger.warning("Failed to retrieve num_params for %s: %s", repo_id, e)
    return None


def search_hf_for_pytorch_models_with_estimated_parameters(
    filter: str = "image-classification", limit: int = 3, sort: str = "downloads"
) -> list[dict[str, Any]]:
    """Search HF for PyTorch models and annotate with estimated parameters."""
    logger.info("Searching Hugging Face models for filter='%s'", filter)
    api = HfApi()
    models = api.list_models(
        filter=filter,
        library="pytorch",
        sort=sort,
        direction=-1,
        limit=limit,
    )

    results: list[dict[str, Any]] = []
    for m in models:
        num_params = get_num_params_if_available(m.id)
        if num_params:
            results.append(
                {
                    "model_id": m.id,
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                    "last_modified": getattr(m, "lastModified", None),
                    "private": getattr(m, "private", None),
                    "num_params": num_params,
                }
            )

    logger.info("Found %d models with parameter info", len(results))
    return results

def save_upload(upload_file: UploadFile, destination: Path) -> None:
    """
    Save a FastAPI UploadFile to a destination path.

    Args:
        upload_file (UploadFile): The uploaded file from a multipart/form-data request.
        destination (Path): The path (including filename) where the file should be saved.

    Raises:
        Exception: If the file cannot be saved.
    """
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise RuntimeError(f"Failed to save uploaded file {upload_file.filename}: {e}")

def sort_models_by_size(models: list[dict[str, Any]], size_tier: str) -> list[dict[str, Any]]:
    """Filter and sort models by size tier based on estimated parameter counts."""
    logger.info("Sorting models by size tier: %s", size_tier)
    tier = str(size_tier).strip().lower()

    SMALL_MAX: int = int(os.getenv("MODEL_SMALL_MAX_PARAM_SIZE", 50_000_000))
    MEDIUM_MIN: int = SMALL_MAX + 1
    MEDIUM_MAX: int = int(os.getenv("MODEL_MEDIUM_MAX_PARAM_SIZE", 200_000_000))
    LARGE_MIN: int = MEDIUM_MAX + 1

    def in_tier(m: dict[str, Any]) -> bool:
        n = m.get("num_params")
        if n is None:
            return False
        if tier == "small":
            return 0 <= n <= SMALL_MAX
        if tier == "medium":
            return MEDIUM_MIN <= n <= MEDIUM_MAX
        if tier == "large":
            return n >= LARGE_MIN
        return True

    filtered = [m for m in models if in_tier(m)]
    if not filtered:
        logger.warning("No models matched tier '%s'; falling back to all models", tier)
        filtered = models

    return sorted(filtered, key=lambda m: (m.get("num_params") is None, m.get("num_params", 0)))

@app.post("/automl_vision/best_model_mvp/")
async def find_best_model_for_vision_mvp(
    csv_file: Annotated[
        UploadFile, 
        File(..., description="CSV file containing image filenames and labels")
    ],
    images_zip: Annotated[
        UploadFile, 
        File(..., description="ZIP file containing image dataset")
    ],
    filename_column: Annotated[
        str, 
        Form(..., description="Name of the CSV column containing image filenames")
    ],
    label_column: Annotated[
        str, 
        Form(..., description="Name of the CSV column containing labels or targets")
    ],
    task_type: Annotated[
        str, 
        Form(..., description="Type of ML task (e.g., 'classification')", examples=["classification"])
    ] = "classification",
    time_budget: Annotated[
        int, 
        Form(..., description="Time budget for AutoML training in seconds")
    ] = 600,
    model_size: Annotated[
        str, 
        Form(..., description="Desired model size hint (e.g., 'small', 'base', 'large')")
    ] = "base",
    db: Session = Depends(get_db),
) -> JSONResponse:
    """
    ### Endpoint: `/automl_vision/best_model_mvp/`

    **Purpose:**  
    Single-step AutoML pipeline for image datasets.  
    Upload CSV + ZIP + metadata, run validation, choose model, train, and return metrics.

    ---
    #### Input Parameters

    * **csv_file:** CSV containing image filenames and labels.
    * **images_zip:** ZIP archive of image files.
    * **filename_column:** Column in CSV that maps to image filenames.
    * **label_column:** Column in CSV with class labels.
    * **task_type:** `"classification"` (others can be extended later).
    * **time_budget:** Training time in seconds.
    * **model_size:** `"small" | "base" | "large"`, selects model candidates.
    ---
    """

    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting new vision AutoML MVP session: %s", session_id)

    try:
        # --- Save and validate CSV ---
        csv_path = session_dir / "labels.csv"
        save_upload(csv_file, csv_path)
        df = pd.read_csv(csv_path)

        if filename_column not in df.columns:
            msg = f"Filename column '{filename_column}' not found in CSV"
            logger.error(msg)
            return JSONResponse(status_code=400, content={"error": msg})
        if label_column not in df.columns:
            msg = f"Label column '{label_column}' not found in CSV"
            logger.error(msg)
            return JSONResponse(status_code=400, content={"error": msg})

        # --- Save and extract images ---
        images_zip_path = session_dir / "images.zip"
        save_upload(images_zip, images_zip_path)
        images_dir = session_dir / "images"
        images_dir.mkdir(exist_ok=True)
        shutil.unpack_archive(images_zip_path, images_dir)
        images_dir = resolve_images_root(images_dir)

        # Normalize and validate
        df = normalize_dataframe_filenames(df, filename_column, csv_path)
        missing_files = collect_missing_files(df, images_dir, filename_column, label_column)
        if missing_files:
            preview = missing_files[:5]
            suffix = "..." if len(missing_files) > 5 else ""
            msg = f"Missing image files: {preview}{suffix}"
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
        sorted_candidates = sort_models_by_size(candidates, model_size)
        chosen = sorted_candidates[0] if sorted_candidates else None
        model_id = chosen["model_id"] if chosen else "google/vit-base-patch16-224"
        logger.info("Chosen model for session %s: %s", session_id, model_id)

        # --- Initialize DataModule and Trainer ---
        datamodule = ClassificationData(
            csv_file=str(csv_path),
            root_dir=str(images_dir),
            img_col=str(filename_column),
            label_col=str(label_column),
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
