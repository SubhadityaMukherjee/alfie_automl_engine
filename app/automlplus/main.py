"""FastAPI endpoints for website accessibility analysis and chat utilities."""
import logging
import os
from contextlib import asynccontextmanager

import requests
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader

from app.automlplus.imagetools import ImagePromptRunner
from app.automlplus.website_accessibility.modules import (
    AltTextChecker,
    ReadabilityAnalyzer,
)
from app.automlplus.website_accessibility.services import (
    extract_text_from_html_bytes,
    run_accessibility_pipeline,
    resolve_coroutines,
)
from app.core.chat_handler import ChatHandler
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

app = FastAPI()
jinja_path = os.getenv("JINJAPATH")
if not jinja_path:
    raise RuntimeError("JINJAPATH environment variable is not set")


jinja_environment = Environment(loader=FileSystemLoader(jinja_path))

LLM_BACKEND = os.getenv("MODEL_BACKEND", "ollama")
DEFAULT_MODEL = os.getenv("WEB_ACCESSIBILITY_CHAT_MODEL", "gemma3:4b")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    await ChatHandler.init()
    yield
    # Cleanup resources
    pass

def json_safe(data: dict) -> dict:
    """
    Recursively convert all string values in a dict to JSON-safe strings
    (escape quotes, line breaks, etc.) so JSONResponse won't fail.
    """
    if isinstance(data, dict):
        return {k: json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_safe(v) for v in data]
    elif isinstance(data, str):
        # Escape quotes and newlines
        return data.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    else:
        return data

@app.post("/automlplus/image_tools/image_to_website/")
async def image_to_website(
    image_file: UploadFile | None = File(default=None),
) -> JSONResponse:
    logger.info("Converting image to a website")
    try:
        # result = AltTextChecker.check(jinja_environment, image_url, alt_text)
        logger.info("Conversion completed.")
        return JSONResponse(content={})
    except Exception as e:
        logger.exception("Error during conversion")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/automlplus/web_access/check-alt-text/")
async def check_alt_text(
    image_url: str = Form(...), alt_text: str = Form(...)
) -> JSONResponse:
    """Evaluate provided alt text against the referenced image using an LLM."""
    logger.info("Checking alt-text for image: %s", image_url)
    try:
        result = AltTextChecker.check(jinja_environment, image_url, alt_text)
        logger.info("Alt-text evaluation completed.")
        safe_result = json_safe({
            "src": image_url,
            "alt_text": alt_text,
            "evaluation": result
        })
        return JSONResponse(content=safe_result)
    except Exception as e:
        logger.exception("Error during alt-text check")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/automlplus/image_tools/run_on_image/")
async def run_on_image(
    prompt: str = Form(...),
    model: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    image_url: str | None = Form(default=None),
) -> JSONResponse:
    if image_file is None and not image_url:
        return JSONResponse({"error": "Provide image_file or image_url"}, status_code=400)
    try:
        image_bytes = await image_file.read() if image_file else None
        if image_file:
            await image_file.close()

        result = ImagePromptRunner.run(
            image_bytes=image_bytes,
            image_path_or_url=image_url,
            prompt=prompt,
            model=model,
            jinja_environment=jinja_environment,
        )

        safe_result = json_safe({"response": result})
        return JSONResponse(content=safe_result)
    except Exception as e:
        logger.exception("Error during image prompt run")
        return JSONResponse({"error": str(e)}, status_code=500)

        
@app.post("/automlplus/image_tools/run_on_image_stream/")
async def run_on_image_stream(
    prompt: str = Form(...),
    model: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    image_url: str | None = Form(default=None),
):
    """Stream a VLM on an image and prompt.

    Returns text chunks as they arrive using text/plain streaming.
    """
    if image_file is None and not image_url:
        return JSONResponse(
            content={"error": "Provide image_file or image_url"}, status_code=400
        )
    try:
        image_bytes: bytes | None = None
        if image_file is not None:
            try:
                image_bytes = await image_file.read()
            finally:
                try:
                    await image_file.close()
                except Exception:
                    pass

        def generator():
            for chunk in ImagePromptRunner.run_stream(
                image_bytes=image_bytes,
                image_path_or_url=image_url,
                prompt=prompt,
                model=model,
                jinja_environment=jinja_environment,
            ):
                yield chunk

        return StreamingResponse(generator(), media_type="text/plain")
    except Exception as e:
        logger.exception("Error during image prompt run (stream)")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/automlplus/web_access/analyze/")
async def analyze_web_accessibility_and_readability(
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    extra_file_input: UploadFile | None = File(default=None),
):
    """
    Run WCAG-inspired accessibility checks and optional readability analysis on HTML.
    Accepts an uploaded file or a URL, with optional guidelines file for context.
    """
    content: str | None = None
    source_name: str = "uploaded.html"
    timeout: int = int(os.getenv("WEB_ACCESSIBILITY_URL_RETRY_TIMEOUT", 10))

    # --- Load HTML content ---
    if file is not None:
        try:
            content = (await file.read()).decode("utf-8", errors="replace")
            source_name = file.filename or source_name
        finally:
            try:
                await file.close()
            except Exception:
                pass
    elif url is not None:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            content = resp.text
            source_name = url
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to fetch URL: {e}"}, status_code=400
            )
    else:
        return JSONResponse(
            content={"error": "Either 'file' or 'url' must be provided."},
            status_code=400,
        )

    if not content or not str(content).strip():
        return JSONResponse(
            content={"error": "Resolved content is empty"}, status_code=400
        )
    content_str: str = str(content)

    # --- Load guidelines file if provided ---
    context_str: str = ""
    if extra_file_input is not None:
        try:
            guidelines_bytes = await extra_file_input.read()
            guidelines_text = guidelines_bytes.decode("utf-8", errors="replace")
            context_str = f"Accessibility guidelines to follow (user-provided):\n\n{guidelines_text}"
        finally:
            try:
                await extra_file_input.close()
            except Exception:
                pass

    # --- Run accessibility pipeline ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE_FOR_ACCESSIBILITY", 3000))
    concurrency_num: int = int(os.getenv("CONCURRENCY_NUM_FOR_ACCESSIBILITY", 4))
    results = await run_accessibility_pipeline(
        content=content_str,
        filename=source_name,
        jinja_environment=jinja_environment,
        chunk_size=chunk_size,
        concurrency=concurrency_num,
        context=context_str,
    )

    # --- Aggregate results into a single JSON payload (include average + readability) ---
    # Convert dataclass objects to dicts, resolving any coroutine fields
    resolved_results = [await resolve_coroutines(r) for r in results]

    # Compute average score
    scores = [r.get("score") for r in resolved_results if isinstance(r.get("score"), (int, float))]
    average_score = (sum(scores)/len(scores)) if scores else None

    # Readability analysis
    readability_scores = None
    try:
        text = extract_text_from_html_bytes(content_str.encode("utf-8"))
        if text.strip():
            readability_scores = ReadabilityAnalyzer.analyze(text)
    except Exception as e:
        readability_scores = {"error": str(e)}

    payload = {
        "source": source_name,
        "average_score": average_score,
        "results": resolved_results,
        "readability": readability_scores,
    }

    # Make all text fields JSON-safe
    safe_payload = json_safe(payload)
    return JSONResponse(content=safe_payload)