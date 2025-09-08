import datetime
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Generator, cast

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from torch import nn, optim

from app.core.chat_handler import ChatHandler
from app.vision_automl.ml_engine import (ClassificationData,
                                         ClassificationModel, FabricTrainer)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# app initialized after lifespan definition below

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8002")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///automl_sessions.db")
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
    session_id: str


class AutoMLVisionSession(Base):
    __tablename__ = "automl_vision_sessions"

    session_id = Column(String, primary_key=True, index=True)
    csv_file_path = Column(String, nullable=False)
    images_dir_path = Column(String, nullable=False)
    filename_column = Column(String, nullable=False)
    label_column = Column(String, nullable=False)
    task_type = Column(String, nullable=False)  # e.g., classification
    time_budget = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# Create database tables after all models are defined
Base.metadata.create_all(bind=engine)


# -----------------------------
# Database dependency
# -----------------------------
def get_db() -> Generator[Session, None, None]:
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# Helpers
# -----------------------------
def normalize_dataframe_filenames(
    df: pd.DataFrame, filename_column: str, csv_path: Path
) -> pd.DataFrame:
    if filename_column in df.columns:
        df[filename_column] = (
            df[filename_column]
            .astype(str)
            .map(lambda s: os.path.basename(str(s).replace("\\", "/")))
        )
        df.to_csv(csv_path, index=False)
    return df


def resolve_images_root(images_dir: Path) -> Path:
    # Handle common zip packaging patterns
    nested_images_dir = images_dir / "images"
    if nested_images_dir.exists() and nested_images_dir.is_dir():
        images_dir = nested_images_dir

    try:
        top_level_entries = [p for p in images_dir.iterdir()]
        only_dirs = [p for p in top_level_entries if p.is_dir()]
        only_files = [p for p in top_level_entries if p.is_file()]
        if len(only_files) == 0 and len(only_dirs) == 1:
            images_dir = only_dirs[0]
    except Exception:
        pass

    return images_dir


def collect_missing_files(
    df: pd.DataFrame,
    images_dir: Path,
    filename_column: str,
    label_column: str,
) -> list[str]:
    missing_files: list[str] = []
    for _, row in df.iterrows():
        raw_filename = str(row[filename_column])
        label = str(row[label_column])

        normalized = raw_filename.replace("\\", "/")
        basename = os.path.basename(normalized)

        candidates = [
            images_dir / label / basename,
            images_dir / basename,
            images_dir / normalized,
        ]

        if any(path.exists() for path in candidates):
            continue

        try:
            found_any = next(images_dir.rglob(basename), None) is not None
        except Exception:
            found_any = False

        if not found_any:
            missing_files.append(raw_filename)

    return missing_files


@app.post("/automl_vision/get_user_input/")
async def get_vision_user_input(
    csv_file: UploadFile = File(...),
    images_zip: UploadFile = File(...),  # zipped folder with images
    filename_column: str = Form(...),
    label_column: str = Form(...),
    task_type: str = Form(..., examples=["classification"]),
    time_budget: int = Form(...),
    db: Session = Depends(get_db),
) -> JSONResponse:
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save CSV file
        csv_path = session_dir / "labels.csv"
        with open(csv_path, "wb") as buffer:
            shutil.copyfileobj(csv_file.file, buffer)
        df = pd.read_csv(csv_path)

        # Validate CSV columns
        if filename_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Filename column '{filename_column}' not found"},
            )
        if label_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Label column '{label_column}' not found"},
            )

        # add a new filename column with absolute file paths

        # Save and extract images
        images_zip_path = session_dir / "images.zip"
        with open(images_zip_path, "wb") as buffer:
            shutil.copyfileobj(images_zip.file, buffer)

        images_dir = session_dir / "images"
        images_dir.mkdir(exist_ok=True)
        shutil.unpack_archive(images_zip_path, images_dir)

        # Resolve common top-level folder patterns in zips
        images_dir = resolve_images_root(images_dir)

        # Normalize filename column to just basenames
        df = normalize_dataframe_filenames(df, filename_column, csv_path)

        # Ensure all files listed in CSV exist
        missing_files = collect_missing_files(
            df=df,
            images_dir=images_dir,
            filename_column=filename_column,
            label_column=label_column,
        )

        if missing_files:
            preview = missing_files[:5]
            suffix = "..." if len(missing_files) > 5 else ""
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Missing image files: {preview}{suffix}",
                    "missing_count": len(missing_files),
                },
            )

        # Save to DB
        new_session = AutoMLVisionSession(
            session_id=session_id,
            csv_file_path=str(csv_path),
            images_dir_path=str(images_dir),
            filename_column=filename_column,
            label_column=label_column,
            task_type=task_type,
            time_budget=time_budget,
        )
        db.add(new_session)
        db.commit()

        return JSONResponse(
            status_code=200,
            content={
                "message": "Vision session stored in DB",
                "session_id": session_id,
            },
        )

    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/automl_vision/find_best_model/")
def find_best_model(request: SessionRequest, db: Session = Depends(get_db)):
    session_record = (
        db.query(AutoMLVisionSession).filter_by(session_id=request.session_id).first()
    )

    if not session_record:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    # Build data module
    # choose a candidate HF model
    api = HfApi()
    models = list(
        api.list_models(
            filter="image-classification",
            library="pytorch",
            sort="downloads",
            direction=-1,
            limit=MAX_MODELS_HF,
        )
    )
    model_id = models[0].id if len(models) > 0 else "google/vit-base-patch16-224"

    datamodule = ClassificationData(
        csv_file=str(session_record.csv_file_path),
        root_dir=str(session_record.images_dir_path),
        img_col=str(session_record.filename_column),
        label_col=str(session_record.label_column),
        batch_size=64,
        hf_model_id=model_id,
    )

    # Train with FabricTrainer under a time budget
    # ensure proper type for time limit
    time_limit: float = float(cast(int, session_record.time_budget))

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
        epochs=50,  # upper bound; time_limit will cut earlier
        time_limit=time_limit,
    )
    test_loss, test_acc = trainer.fit()

    return JSONResponse(
        status_code=200,
        content={
            "message": "Training completed",
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "num_classes": datamodule.num_classes,
        },
    )
