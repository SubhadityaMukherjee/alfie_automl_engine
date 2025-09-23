"""FastAPI endpoints for tabular AutoML workflows.

Provides endpoints to accept user data/config, validate inputs, store
session metadata, and trigger AutoML training using AutoGluon.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.chat_handler import ChatHandlerOllama
from app.tabular_automl.modules import AutoMLTrainer
from app.tabular_automl.services import (create_session_directory, get_session,
                                         load_table, save_upload,
                                         store_session_in_db,
                                         validate_tabular_inputs)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

app = FastAPI()

TABULAR_AUTOML_PORT = os.getenv("TABULAR_AUTOML_PORT", "http://localhost:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    await ChatHandlerOllama.init()
    yield
    # Cleanup resources
    pass


# # NOTE : I AM NOT SURE IF THE AUTODW WILL HANDLE THIS PART FIRST :/
class SessionRequest(BaseModel):
    """Payload for initiating model search/training for a session."""

    session_id: str


@app.post("/automl_tabular/get_user_input/")
async def get_user_input(
    train_file: Optional[UploadFile] = File(None),
    train_csv: Optional[UploadFile] = File(None),
    test_file: Optional[UploadFile] = None,
    target_column_name: str = Form(...),
    time_stamp_column_name: Optional[str] = None,
    task_type: str = Form(
        ..., examples=["classification", "regression", "time series"]
    ),
    time_budget: int = Form(...),
) -> JSONResponse:
    """Create a session, upload data, validate inputs, and store metadata."""
    session_id, session_dir = create_session_directory()

    # Prefer 'train_file' but fallback to legacy 'train_csv'
    provided_train: Optional[UploadFile] = train_file or train_csv
    if provided_train is None:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Field 'train_file' is required (or legacy 'train_csv')."
            },
        )

    try:
        provided_filename = provided_train.filename or "train.csv"
        train_suffix = Path(provided_filename).suffix or ".csv"
        train_path = session_dir / f"train{train_suffix}"
        save_upload(provided_train, train_path)

        test_path = None
        if test_file:
            test_filename = test_file.filename or "test.csv"
            test_suffix = Path(test_filename).suffix or ".csv"
            test_path = session_dir / f"test{test_suffix}"
            save_upload(test_file, test_path)

        validation_error = validate_tabular_inputs(
            train_path=train_path,
            target_column_name=target_column_name,
            time_stamp_column_name=time_stamp_column_name,
            task_type=task_type,
        )
        if validation_error:
            return JSONResponse(status_code=400, content={"error": validation_error})

        store_session_in_db(
            session_id=session_id,
            train_path=train_path,
            test_path=test_path,
            target_column_name=target_column_name,
            time_stamp_column_name=time_stamp_column_name,
            task_type=task_type,
            time_budget=time_budget,
        )

        return JSONResponse(
            status_code=200,
            content={"message": "Session stored in DB.", "session_id": session_id},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/automl_tabular/find_best_model/")
def find_best_model(request: SessionRequest):
    """Train AutoML on stored session data and return leaderboard."""
    session_record = get_session(request.session_id)

    if not session_record:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    save_model_path = (
        Path(str(session_record.train_file_path)).parent / "automl_data_path"
    )
    os.makedirs(save_model_path, exist_ok=True)

    trainer = AutoMLTrainer(save_model_path=save_model_path)

    # Load dataframes
    train_df = load_table(Path(session_record.train_file_path))
    test_df = (
        load_table(Path(session_record.test_file_path))
        if session_record.test_file_path
        else None
    )

    leaderboard = trainer.train(
        train_df=train_df,
        test_df=test_df,
        target_column=str(session_record.target_column),
        time_limit=int(session_record.time_budget),
    )

    return JSONResponse(
        content=(
            leaderboard.to_markdown()
            if isinstance(leaderboard, pd.DataFrame)
            else leaderboard
        )
    )
