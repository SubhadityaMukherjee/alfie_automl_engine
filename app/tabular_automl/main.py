import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.chat_handler import ChatHandler
from app.core.utils import render_template

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from pydantic import BaseModel, FilePath, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    await ChatHandler.init()
    yield
    # Cleanup resources
    pass


@app.post("/automl_tabular/get_user_input/")
async def get_user_input(
    train_csv: UploadFile = File(...),
    test_csv: Optional[UploadFile] = None,
    target_column_name: str = Form(...),
    time_stamp_column_name: Optional[str] = None,
    task_type: str = Form(..., description="Type of task classification, regression, time series (Must be a selectbox or match exactly)", examples=["classification", "regression", "time series"]),
) -> JSONResponse: ...


@app.post("/automl_tabular/find_best_model/")
async def find_best_model() -> JSONResponse:
    """Output :
            - automl : Json {“leaderboard” with best models/metrics, best model as pkl/dataset id information on how to use them}
                    - Further complications : Might have to return the “model_id” that was uploaded to autodw instead of the actual model as a pkl
    - Exceptions :
            - something wrong with the data, so couldnt train a model
            - ran out of time
    """
    ...

