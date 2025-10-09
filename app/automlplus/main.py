"""FastAPI endpoints for website accessibility analysis and chat utilities."""
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import requests
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, JSONResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader

from app.automlplus.imagetools import ImagePromptRunner
from app.automlplus.website_accessibility.modules import (
    AltTextChecker,
    ReadabilityAnalyzer,
)
from typing import Annotated
from app.automlplus.website_accessibility.services import (
    extract_text_from_html_bytes,
    run_accessibility_pipeline,
    resolve_coroutines,
)
from app.core.chat_handler import ChatHandler

# Module logger
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


def json_safe(data: Any) -> Any:
    """
    Recursively convert all string values in a dict or list to JSON-safe strings
    (escape quotes, line breaks, etc.) so JSONResponse won't fail.
    """
    if isinstance(data, dict):
        logger.debug("Processing dict for JSON safety")
        return {k: json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        logger.debug("Processing list for JSON safety")
        return [json_safe(v) for v in data]
    elif isinstance(data, str):
        logger.debug("Escaping special characters in string for JSON safety")
        return (
            data.replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
        )
    else:
        return data


@app.post("/automlplus/image_tools/image_to_website/")
async def image_to_website(
    image_file: UploadFile | None = File(default=None),
) -> JSONResponse:
    """Convert an uploaded image into a basic HTML website structure."""
    logger.info("Starting image-to-website conversion")
    try:
        # TODO: Implement image-to-website logic (currently placeholder)
        logger.info("Image-to-website conversion completed successfully")
        return JSONResponse(content={})
    except Exception as e:
        logger.exception("Error during image-to-website conversion: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/automlplus/web_access/check-alt-text/")
async def check_alt_text(
    image_url: str = Form(...),
    alt_text: str = Form(...),
) -> JSONResponse:
    """Evaluate provided alt text against the referenced image using an LLM."""
    logger.info(f"Checking alt text for image URL: {image_url}")
    try:
        result: dict[str, Any] = AltTextChecker.check(jinja_environment, image_url, alt_text)
        logger.info("Alt-text evaluation completed successfully")

        safe_result = json_safe({
            "src": image_url,
            "alt_text": alt_text,
            "evaluation": result,
        })
        return JSONResponse(content=safe_result)
    except Exception as e:
        logger.exception("Error during alt-text check: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/automlplus/image_tools/run_on_image/")
async def run_on_image(
    prompt: str = Form(...),
    model: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    image_url: str | None = Form(default=None),
) -> JSONResponse:
    """Run a vision-language model on an image and return the text output."""
    logger.info("Running model on image with prompt: %s", prompt)

    if image_file is None and not image_url:
        logger.error("Missing both image_file and image_url")
        return JSONResponse({"error": "Provide image_file or image_url"}, status_code=400)

    try:
        image_bytes: bytes | None = await image_file.read() if image_file else None
        if image_file:
            await image_file.close()
            logger.debug("Image file successfully read and closed")

        result = ImagePromptRunner.run(
            image_bytes=image_bytes,
            image_path_or_url=image_url,
            prompt=prompt,
            model=model,
            jinja_environment=jinja_environment,
        )

        safe_result = json_safe({"response": result})
        logger.info("Image prompt run completed successfully")
        return JSONResponse(content=safe_result)
    except Exception as e:
        logger.exception("Error during image prompt run: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post(
    "/automlplus/image_tools/run_on_image_stream/",
    response_model=None,
)
async def run_on_image_stream(
    prompt: Annotated[str, Form(..., description="Prompt to apply on the image")] = "",
    model: Annotated[str | None, Form(..., description="Model to apply on the image")] = None,
    image_file: Annotated[UploadFile | None, File(..., description="Image file if not a URL")] = None,
    image_url: Annotated[str | None, Form(..., description="Image URL if not a file but an URL")] = None,
) -> Response:
    """Stream a vision-language model's output on an image and prompt."""
    logger.info("Streaming model output for image prompt: %s", prompt)

    if image_file is None and not image_url:
        logger.error("No image or URL provided for streaming run")
        return JSONResponse(
            content={"error": "Provide image_file or image_url"}, status_code=400
        )

    try:
        image_bytes: bytes | None = None
        if image_file is not None:
            try:
                image_bytes = await image_file.read()
                logger.debug("Image file successfully read for streaming")
            finally:
                try:
                    await image_file.close()
                except Exception:
                    logger.warning("Failed to properly close image file", exc_info=True)

        def generator():
            logger.debug("Starting stream generator for image model run")
            for chunk in ImagePromptRunner.run_stream(
                image_bytes=image_bytes,
                image_path_or_url=image_url,
                prompt=prompt,
                model=model,
                jinja_environment=jinja_environment,
            ):
                yield chunk

        logger.info("Image stream initiated successfully")
        return StreamingResponse(generator(), media_type="text/plain")

    except Exception as e:
        logger.exception("Error during image prompt streaming run: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/automlplus/web_access/analyze/")
async def analyze_web_accessibility_and_readability(
    file: Annotated[UploadFile, File(..., description="HTML file")],
    url: Annotated[str | None, Form(..., description="URL of website")] = None,
    extra_file_input: Annotated[UploadFile | None, File(..., description="Extra file for LLM context")] = None,
) -> JSONResponse:
    """Run WCAG-inspired accessibility checks and optional readability analysis on HTML."""
    logger.info("Starting web accessibility and readability analysis")

    content: str | None = None
    source_name: str = "uploaded.html"
    timeout: int = int(os.getenv("WEB_ACCESSIBILITY_URL_RETRY_TIMEOUT", 10))

    # --- LoaOd HTML content ---
    if file:
        try:
            content = (await file.read()).decode("utf-8", errors="replace")
            source_name = file.filename or source_name
            logger.debug(f"HTML file '{source_name}' successfully loaded")
        finally:
            try:
                await file.close()
            except Exception:
                logger.warning("Failed to close uploaded HTML file", exc_info=True)

    if url:
        try:
            logger.debug(f"Fetching HTML from URL: {url}")
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            content = resp.text
            source_name = url
            logger.debug("HTML successfully fetched from URL")
        except Exception as e:
            logger.error(f"Failed to fetch HTML from URL: {e}")
            return JSONResponse(
                content={"error": f"Failed to fetch URL: {e}"}, status_code=400
            )

    
    if not content or not str(content).strip():
        logger.error("Resolved HTML content is empty")
        return JSONResponse(content={"error": "Resolved content is empty"}, status_code=400)

    content_str: str = str(content)

    # --- Load guidelines file if provided ---
    context_str: str = ""
    if extra_file_input is not None:
        try:
            logger.debug("Reading extra context file for accessibility analysis")
            guidelines_bytes = await extra_file_input.read()
            guidelines_text = guidelines_bytes.decode("utf-8", errors="replace")
            context_str = f"Accessibility guidelines to follow (user-provided):\n\n{guidelines_text}"
            logger.debug("Extra context file successfully loaded")
        finally:
            try:
                await extra_file_input.close()
            except Exception:
                logger.warning("Failed to close extra context file", exc_info=True)

    # --- Run accessibility pipeline ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE_FOR_ACCESSIBILITY", 3000))
    concurrency_num: int = int(os.getenv("CONCURRENCY_NUM_FOR_ACCESSIBILITY", 4))
    logger.debug(f"Running accessibility pipeline with chunk size {chunk_size}, concurrency {concurrency_num}")

    results = await run_accessibility_pipeline(
        content=content_str,
        filename=source_name,
        jinja_environment=jinja_environment,
        chunk_size=chunk_size,
        concurrency=concurrency_num,
        context=context_str,
    )
    logger.info("Accessibility pipeline completed successfully")

    # --- Aggregate results ---
    resolved_results = [await resolve_coroutines(r) for r in results]

    scores = [r.get("score") for r in resolved_results if isinstance(r.get("score"), (int, float))]
    average_score: float | None = (sum(scores) / len(scores)) if scores else None
    logger.debug(f"Computed average accessibility score: {average_score}")

    # --- Readability analysis ---
    readability_scores: dict[str, Any] | None = None
    try:
        text = extract_text_from_html_bytes(content_str.encode("utf-8"))
        if text.strip():
            readability_scores = ReadabilityAnalyzer.analyze(text)
            logger.debug("Readability analysis completed successfully")
    except Exception as e:
        logger.warning(f"Error during readability analysis: {e}")
        readability_scores = {"error": str(e)}

    payload = {
        "source": source_name,
        "average_score": average_score,
        "results": resolved_results,
        "readability": readability_scores,
    }

    safe_payload = json_safe(payload)
    logger.info("Web accessibility and readability analysis finished successfully")
    return JSONResponse(content=safe_payload)
