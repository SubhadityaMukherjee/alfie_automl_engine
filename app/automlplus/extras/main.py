import logging
import os
from typing import Any, Dict

import torch  # type: ignore
from dotenv import find_dotenv, load_dotenv  # type: ignore
from fastapi import FastAPI, File, Form, UploadFile  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore

from app.general_inference_tools.modules import get_engine

torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

app = FastAPI()
BACKEND = os.getenv("GENERAL_INFERENCE_BACKEND_URL", "http://localhost:8004")

_READY: bool = False


@app.on_event("startup")
async def preload_general_inference_models() -> None:
    """Preload heavy HF models and pipelines so the process waits until ready."""
    logger.info("[startup] Preloading General Inference models...")
    engine = get_engine()
    for task_name, task_cls in engine.task_classes.items():
        try:
            if task_cls.requires_hf_model():
                engine._get_model_and_tokenizer(task_name)
                logger.info(f"[startup] Loaded model/tokenizer for task '{task_name}'")
            else:
                # Optionally warm up heavy lazy tasks to avoid first-request timeout
                if task_name in ("document_qa",):
                    cfg = engine.registry[task_name]
                    task = task_cls(None, None, engine.fabric, cfg)
                    try:
                        # Trigger Donut lazy load by calling run with a tiny blank image
                        import tempfile

                        from PIL import Image  # type: ignore

                        img = Image.new("RGB", (8, 8), color=(255, 255, 255))
                        with tempfile.NamedTemporaryFile(
                            suffix=".png", delete=True
                        ) as tmp:
                            img.save(tmp.name)
                            _ = task.run(query="warmup", inputs=[tmp.name])
                        logger.info(f"[startup] Warmed up task '{task_name}'")
                    except Exception as warm_exc:
                        logger.warning(
                            f"[startup] Skipping warmup for '{task_name}': {warm_exc}"
                        )
        except Exception as exc:
            logger.exception(f"[startup] Failed to preload task '{task_name}': {exc}")
            # Abort startup so process does not accept traffic until ready
            raise
    global _READY
    _READY = True


@app.get("/health")
def health() -> JSONResponse:
    status = {"ready": _READY}
    return JSONResponse(status_code=200 if _READY else 503, content=status)


@app.post("/general_inference/instruction_to_webpage")
def instruction_to_webpage(
    requirements: str = Form(...), gen_kwargs: str = Form("{}")
) -> JSONResponse:
    try:
        import json

        engine = get_engine()
        if "instruction_to_webpage" not in engine.registry:
            return JSONResponse(
                status_code=400,
                content={"error": "instruction_to_webpage task not available"},
            )
        gen_kwargs_dict: Dict[str, Any] = json.loads(gen_kwargs) if gen_kwargs else {}
        response, _ = engine.run_task(
            "instruction_to_webpage",
            query=requirements,
            inputs=[],
            **gen_kwargs_dict,
        )
        return JSONResponse(
            status_code=200,
            content={
                "response": response,
                "task_name": "instruction_to_webpage",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/general_inference/screenshot_to_webpage")
def screenshot_to_webpage(
    requirements: str = Form(...), gen_kwargs: str = Form("{}")
) -> JSONResponse:
    """Generate a Tailwind HTML webpage from natural-language requirements.

    This uses the instruction-aware webpage generation model; the endpoint name
    matches the UX intent while the underlying task is text -> webpage.
    """
    try:
        import json

        engine = get_engine()
        if "instruction_to_webpage" not in engine.registry:
            return JSONResponse(
                status_code=400,
                content={"error": "instruction_to_webpage task not available"},
            )
        gen_kwargs_dict: Dict[str, Any] = json.loads(gen_kwargs) if gen_kwargs else {}
        response, _ = engine.run_task(
            "instruction_to_webpage",
            query=requirements,
            inputs=[],
            **gen_kwargs_dict,
        )
        return JSONResponse(
            status_code=200,
            content={
                "response": response,
                "task_name": "instruction_to_webpage",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/general_inference/video_understanding")
def video_understanding(
    query: str = Form(...),
    video_file: UploadFile = File(...),
    gen_kwargs: str = Form("{}"),
) -> JSONResponse:
    try:
        import json
        import os

        from app.tabular_automl.services import create_session_directory

        engine = get_engine()
        if "video_understanding" not in engine.registry:
            return JSONResponse(
                status_code=400,
                content={"error": "video_understanding task not available"},
            )

        # Save uploaded video file
        session_id, session_dir = create_session_directory()
        video_path = os.path.join(str(session_dir), video_file.filename)
        with open(video_path, "wb") as buffer:
            content = video_file.file.read()
            buffer.write(content)

        gen_kwargs_dict: Dict[str, Any] = json.loads(gen_kwargs) if gen_kwargs else {}
        response, _ = engine.run_task(
            "video_understanding",
            query=query,
            inputs=[video_path],
            **gen_kwargs_dict,
        )
        return JSONResponse(
            status_code=200,
            content={
                "response": response,
                "task_name": "video_understanding",
                "used_inputs": [video_path],
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/general_inference/document_qa")
def document_qa(
    question: str = Form(...),
    document_file: UploadFile = File(...),
    gen_kwargs: str = Form("{}"),
) -> JSONResponse:
    try:
        import json
        import os

        from app.tabular_automl.services import create_session_directory

        engine = get_engine()
        if "document_qa" not in engine.registry:
            return JSONResponse(
                status_code=400, content={"error": "document_qa task not available"}
            )

        # Save uploaded document file
        session_id, session_dir = create_session_directory()
        document_path = os.path.join(str(session_dir), document_file.filename)
        with open(document_path, "wb") as buffer:
            content = document_file.file.read()
            buffer.write(content)

        gen_kwargs_dict: Dict[str, Any] = json.loads(gen_kwargs) if gen_kwargs else {}
        response, _ = engine.run_task(
            "document_qa",
            query=question,
            inputs=[document_path],
            **gen_kwargs_dict,
        )
        return JSONResponse(
            status_code=200,
            content={
                "response": response,
                "task_name": "document_qa",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
