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
from app.tabular_automl.modules import AutoMLTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8001")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///automl_sessions.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

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


# # NOTE : I AM NOT SURE IF THE AUTODW WILL HANDLE THIS PART FIRST :/
class SessionRequest(BaseModel):
    session_id: str

class AutoMLSession(Base):
    __tablename__ = "automl_sessions"

    session_id = Column(String, primary_key=True, index=True)
    train_file_path = Column(String, nullable=False)
    test_file_path = Column(String, nullable=True)
    target_column = Column(String, nullable=False)
    time_stamp_column_name = Column(String, nullable=True)
    task_type = Column(String, nullable=False)
    time_budget = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)


@app.post("/automl_tabular/get_user_input/")
async def get_user_input(
    train_csv: UploadFile = File(...),
    test_csv: Optional[UploadFile] = None,
    target_column_name: str = Form(...),
    time_stamp_column_name: Optional[str] = None,
    task_type: str = Form(
        ..., examples=["classification", "regression", "time series"]
    ),
    time_budget: int = Form(...),
) -> JSONResponse:
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()

    try:
        # Save train file
        train_path = session_dir / "train.csv"
        with open(train_path, "wb") as buffer:
            shutil.copyfileobj(train_csv.file, buffer)
        train_df = pd.read_csv(train_path)

        # Save test file if provided
        test_path = None
        if test_csv:
            test_path = session_dir / "test.csv"
            with open(test_path, "wb") as buffer:
                shutil.copyfileobj(test_csv.file, buffer)
            # pd.read_csv(test_path)

        # Validate columns
        if target_column_name not in train_df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Target column '{target_column_name}' not found."},
            )
        if time_stamp_column_name and time_stamp_column_name not in train_df.columns:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Timestamp column '{time_stamp_column_name}' not found."
                },
            )
        if task_type not in ["classification", "regression", "time series"]:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid task_type '{task_type}'"}
            )

        # Save to DB
        new_session = AutoMLSession(
            session_id=session_id,
            train_file_path=str(train_path),
            test_file_path=str(test_path) if test_path else None,
            target_column=target_column_name,
            time_stamp_column_name=time_stamp_column_name,
            task_type=task_type,
            time_budget=time_budget,
        )
        db.add(new_session)
        db.commit()

        return JSONResponse(
            status_code=200,
            content={
                "message": "Session stored in DB.",
                "session_id": session_id,
            },
        )
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        db.close()


@app.post("/automl_tabular/find_best_model/")
def find_best_model(request: SessionRequest):
    db = SessionLocal()
    session_record = (
        db.query(AutoMLSession).filter_by(session_id=request.session_id).first()
    )
    db.close()

    if not session_record:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    save_model_path = Path(str(session_record.train_file_path)).parent / "automl_data_path"
    os.makedirs(save_model_path, exist_ok=True)

    trainer = AutoMLTrainer(save_model_path=save_model_path)

    leaderboard = trainer.train(
        train_file=str(session_record.train_file_path),
        test_file=str(session_record.test_file_path) or str(session_record.train_file_path),
        target_column=str(session_record.target_column),
        time_limit=session_record.time_budget,
    )

    return JSONResponse(
        content=(
            leaderboard.to_markdown()
            if isinstance(leaderboard, pd.DataFrame)
            else leaderboard
        )
    )
