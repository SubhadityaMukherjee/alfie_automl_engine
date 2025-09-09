
from typing import Any, Dict, List, Optional, Tuple, Type

import torch  # type: ignore
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import logging
import os

from modules import get_engine, RequirementsRequest, VideoUnderstandingRequest, DocumentQARequest,GeneralInferenceEngine
torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = FastAPI()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8004")

@app.post("/general_inference/instruction_to_webpage")
def instruction_to_webpage(req: RequirementsRequest) -> JSONResponse:
    try:
        engine = get_engine()
        if "instruction_to_webpage" not in engine.registry:
            return JSONResponse(status_code=400, content={
                "error": "instruction_to_webpage task not available"
            })
        gen_kwargs: Dict[str, Any] = req.gen_kwargs or {}
        response, _ = engine.run_task(
            "instruction_to_webpage",
            query=req.requirements,
            inputs=[],
            **gen_kwargs,
        )
        return JSONResponse(status_code=200, content={
            "response": response,
            "task_name": "instruction_to_webpage",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/general_inference/screenshot_to_webpage")
def screenshot_to_webpage(req: RequirementsRequest) -> JSONResponse:
    """Generate a Tailwind HTML webpage from natural-language requirements.

    This uses the instruction-aware webpage generation model; the endpoint name
    matches the UX intent while the underlying task is text -> webpage.
    """
    try:
        engine = get_engine()
        if "instruction_to_webpage" not in engine.registry:
            return JSONResponse(status_code=400, content={
                "error": "instruction_to_webpage task not available"
            })
        gen_kwargs: Dict[str, Any] = req.gen_kwargs or {}
        response, _ = engine.run_task(
            "instruction_to_webpage",
            query=req.requirements,
            inputs=[],
            **gen_kwargs,
        )
        return JSONResponse(status_code=200, content={
            "response": response,
            "task_name": "instruction_to_webpage",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/general_inference/video_understanding")
def video_understanding(req: VideoUnderstandingRequest) -> JSONResponse:
    try:
        engine = get_engine()
        if "video_understanding" not in engine.registry:
            return JSONResponse(status_code=400, content={
                "error": "video_understanding task not available"
            })
        gen_kwargs: Dict[str, Any] = req.gen_kwargs or {}
        response, _ = engine.run_task(
            "video_understanding",
            query=req.query,
            inputs=req.inputs,
            history=req.history,
            **gen_kwargs,
        )
        return JSONResponse(status_code=200, content={
            "response": response,
            "task_name": "video_understanding",
            "used_inputs": req.inputs,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/general_inference/document_qa")
def document_qa(req: DocumentQARequest) -> JSONResponse:
    try:
        engine = get_engine()
        if "document_qa" not in engine.registry:
            return JSONResponse(status_code=400, content={
                "error": "document_qa task not available"
            })
        gen_kwargs: Dict[str, Any] = req.gen_kwargs or {}
        response, _ = engine.run_task(
            "document_qa",
            query=req.question,
            inputs=[req.document],
            **gen_kwargs,
        )
        return JSONResponse(status_code=200, content={
            "response": response,
            "task_name": "document_qa",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    engine = GeneralInferenceEngine(precision="bf16-mixed", devices=1)

    query = "Here are some frames of a video. Describe this video in detail"
    video_inputs = [
        "./examples/liuxiang.mp4",
    ]
    response, history = engine.run_task(
        "video_understanding",
        query,
        video_inputs,
        do_sample=False,
        num_beams=3,
        use_meta=True,
    )
    print(response)

    follow_up = "tell me the athlete code of Liu Xiang"
    response2, _ = engine.run_task(
        "video_understanding",
        follow_up,
        video_inputs,
        history=history,
        do_sample=False,
        num_beams=3,
        use_meta=True,
    )
    print(response2)
