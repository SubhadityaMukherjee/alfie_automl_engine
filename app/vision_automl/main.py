import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, FilePath, field_validator

from app.core.chat_handler import ChatHandler
from app.core.utils import render_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8002")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    await ChatHandler.init()
    yield
    # Cleanup resources
    pass


# NOTE: I AM NOT SURE IF THE AUTODW WILL HANDLE THIS PART FIRST :/
@app.post("/vision_automl/get_user_input/")
async def get_user_input(
    train_folder: UploadFile = File(...),
    test_folder: Optional[UploadFile] = File(...),
    label_format: str = Form(
        ...,
        description="Format of labels. eg: folder if sub folder name is the label, csv if theres a csv with path and labels",
        examples=["folder", "csv"],
    ),
    label_file: Optional[UploadFile] = File(
        ...,
        description="If label_format is not folder, this is the required label file",
    ),
    task_type: str = Form(
        ...,
        description="Type of task - for now only classification)",
        examples=["classification"],
    ),
    time_budget: int = Form(
        ..., description="Time budget to train automl system in seconds"
    ),
) -> JSONResponse: ...


@app.post("/vision_automl/find_best_model/")
async def find_best_model(session_id: str) -> JSONResponse:
    """Output :
            - automl : Json {“leaderboard” with best models/metrics, best model as pkl/dataset id information on how to use them}
                    - Further complications : Might have to return the “model_id” that was uploaded to autodw instead of the actual model as a pkl
    - Exceptions :
            - something wrong with the data, so couldnt train a model
            - ran out of time
    """
    ...
