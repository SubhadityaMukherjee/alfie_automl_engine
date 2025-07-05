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

from app.core.chat_handler import ChatHandler
from app.tabular_automl.modules import AutoMLTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8001")

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


# NOTE : I AM NOT SURE IF THE AUTODW WILL HANDLE THIS PART FIRST :/
@app.post("/automl_tabular/get_user_input/")
async def get_user_input(
    train_csv: UploadFile = File(...),
    test_csv: Optional[UploadFile] = None,
    target_column_name: str = Form(...),
    time_stamp_column_name: Optional[str] = None,
    task_type: str = Form(
        ...,
        description="Type of task classification, regression, time series (Must be a selectbox or match exactly)",
        examples=["classification", "regression", "time series"],
    ),
    time_budget: int = Form(
        ..., description="Time budget to train automl system in seconds"
    ),
) -> JSONResponse:
    """
    This just gets the user input and saves it to a temporary file. This feature might be replaced by autodw later on but for now this is what it is.
    """
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save train file
        train_path = session_dir / "train.csv"
        with open(train_path, "wb") as buffer:
            shutil.copyfileobj(train_csv.file, buffer)
        train_df = pd.read_csv(train_path)

        # Save and validate optional test file
        test_df = None
        test_path = None
        if test_csv:
            test_path = session_dir / "test.csv"
            with open(test_path, "wb") as buffer:
                shutil.copyfileobj(test_csv.file, buffer)
            test_df = pd.read_csv(test_path)

        # Validate columns
        if target_column_name not in train_df.columns:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Target column '{target_column_name}' not in train file."
                },
            )

        if time_stamp_column_name and time_stamp_column_name not in train_df.columns:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Timestamp column '{time_stamp_column_name}' not in train file."
                },
            )

        if task_type not in ["classification", "regression", "time series"]:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid task_type '{task_type}'"}
            )

        content = {
            "message": "Files validated and stored.",
            "session_id": session_id,
            "train_file_path": str(train_path),
            "test_file_path": str(test_path) if test_path else None,
            "time_stamp_column_name": (
                str(time_stamp_column_name) if time_stamp_column_name else None
            ),
            "target_column": str(target_column_name),
            "task_type": str(task_type),
            "time_budget": int(time_budget),
        }

        # save config
        with open(session_dir / "config.json", "w+") as fp:
            json.dump(content, fp)

        return JSONResponse(
            status_code=200,
            content=content,
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Processing failed: {str(e)}"}
        )


@app.post("/automl_tabular/find_best_model/")
async def find_best_model(session_id: str) -> JSONResponse:
    """Output :
            - automl : Json {“leaderboard” with best models/metrics, best model as pkl/dataset id information on how to use them}
                    - Further complications : Might have to return the “model_id” that was uploaded to autodw instead of the actual model as a pkl
    - Exceptions :
            - something wrong with the data, so couldnt train a model
            - ran out of time
    """
    session_dir = UPLOAD_ROOT / session_id
    save_model_path = session_dir / "automl_data_path"
    os.makedirs(save_model_path, exist_ok=True)
    with open(session_dir / "config.json", "r") as fp:
        session_config = json.load(fp)

    trainer = AutoMLTrainer(save_model_path=save_model_path)

    # TODO add support for timestamp column here
    leaderboard = trainer.train(
        train_file=session_config.get("train_file_path", ""),
        test_file=session_config.get("train_file_path", ""),
        target_column=session_config.get("target_column", "label"),
        time_limit=session_config.get("time_budget", 10),
    )

    content = (
        leaderboard.to_markdown()
        if isinstance(leaderboard, pd.DataFrame)
        else leaderboard
    )

    return JSONResponse(content=content)
