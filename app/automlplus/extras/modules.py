import logging
import os
from typing import Any, Dict, List

import torch  # type: ignore
from dotenv import find_dotenv, load_dotenv  # type: ignore
from fastapi import FastAPI, File, Form, UploadFile  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from pydantic import BaseModel

from app.general_inference_tools.modules import get_engine

torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

app = FastAPI()
BACKEND = os.getenv("GENERAL_INFERENCE_BACKEND_URL", "http://localhost:8004")

from transformers import pipeline

TASK_MODEL_MAP = {
    "visual-question-answering": "openbmb/MiniCPM-V-2",
    "document-question-answering": "naver-clova-ix/donut-base-finetuned-docvqa",
}

# pipeline(1, model =2, trust_remote_code=)


class HFModelTaskConnector(BaseModel):
    """
    hf_task_tag: str - Actual task type on HF (eg: visual-question-answering
    subtask : str - Sub task (multiple subtasks can use the same model)
    model: str - Model name
    input_types: List[str] - Types of accepted inputs eg : *.jpeg, *.png
    """

    hf_task_tag: str
    subtask: str
    model: str
    input_types: List[str]
