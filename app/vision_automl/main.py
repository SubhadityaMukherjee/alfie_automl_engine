import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sqlalchemy import Column, String, Integer, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import uuid

from app.core.chat_handler import ChatHandler
from torch import nn, optim
from typing import cast
from app.vision_automl.ml_engine import (
    ClassificationData,
    ClassificationModel,
    FabricTrainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8002")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///automl_sessions.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Create database tables
Base.metadata.create_all(bind=engine)

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

@app.post("/automl_vision/get_user_input/")
async def get_vision_user_input(
    csv_file: UploadFile = File(...),
    images_zip: UploadFile = File(...),  # zipped folder with images
    filename_column: str = Form(...),
    label_column: str = Form(...),
    task_type: str = Form(..., examples=["classification"]),
    time_budget: int = Form(...)
) -> JSONResponse:
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()

    try:
        # Save CSV file
        csv_path = session_dir / "labels.csv"
        with open(csv_path, "wb") as buffer:
            shutil.copyfileobj(csv_file.file, buffer)
        df = pd.read_csv(csv_path)

        # Validate CSV columns
        if filename_column not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Filename column '{filename_column}' not found"})
        if label_column not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Label column '{label_column}' not found"})
        
        # add a new filename column with absolute file paths

        # Save and extract images
        images_zip_path = session_dir / "images.zip"
        with open(images_zip_path, "wb") as buffer:
            shutil.copyfileobj(images_zip.file, buffer)

        images_dir = session_dir / "images"
        images_dir.mkdir(exist_ok=True)
        shutil.unpack_archive(images_zip_path, images_dir)

        # Ensure all files listed in CSV exist
        missing_files = []
        for _, row in df.iterrows():
            filename = row[filename_column]
            label = row[label_column]
            # Check if file exists in label subdirectory
            file_path = images_dir / label / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing image files: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}"}
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
            content={"message": "Vision session stored in DB", "session_id": session_id}
        )

    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        db.close()

@app.post("/automl_vision/find_best_model/")
def find_best_model(request: SessionRequest):
    db = SessionLocal()
    session_record = (
        db.query(AutoMLVisionSession).filter_by(session_id=request.session_id).first()
    )
    db.close()

    if not session_record:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    # Build data module
    datamodule = ClassificationData(
        csv_file=str(session_record.csv_file_path),
        root_dir=str(session_record.images_dir_path),
        img_col=str(session_record.filename_column),
        label_col=str(session_record.label_column),
        batch_size=64,
    )

    # Train with FabricTrainer under a time budget
    # ensure proper type for time limit
    time_limit: float = float(cast(int, session_record.time_budget))

    trainer = FabricTrainer(
        datamodule=datamodule,
        model_class=ClassificationModel,
        model_kwargs={
            "model_name": "resnet34",
            "num_classes": datamodule.num_classes,
            "pretrained": True,
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
