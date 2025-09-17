"""FastAPI endpoints for website accessibility analysis and chat utilities."""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, File, Form, UploadFile
import requests
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader

from app.core.chat_handler import ChatHandler
from app.website_accessibility.modules import (AltTextChecker,
                                               ReadabilityAnalyzer)
from app.website_accessibility.services import (extract_text_from_html_bytes,
                                                run_accessibility_pipeline,
                                                stream_accessibility_results)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

app = FastAPI()
jinja_path = os.getenv("JINJAPATH")
if not jinja_path:
    raise RuntimeError("JINJAPATH environment variable is not set")


jinja_environment = Environment(loader=FileSystemLoader(jinja_path))

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    await ChatHandler.init()
    yield
    # Cleanup resources
    pass


@app.post("/web_access/readability/")
async def analyze_readability(file: UploadFile = File(...)) -> JSONResponse:
    """Parse uploaded HTML and return readability scores."""
    logger.info("Received file for readability analysis: %s", file.filename)
    try:
        content = await file.read()
        # Validate non-empty content
        if not content or not content.strip():
            raise ValueError("Uploaded file is empty")
        text = extract_text_from_html_bytes(content)
        # Validate extracted text is non-empty
        if not text.strip():
            raise ValueError("No readable text found in uploaded file")

        scores = ReadabilityAnalyzer.analyze(text)
        logger.info("Readability scores computed successfully.")
        return JSONResponse(content=scores)
    except Exception as e:
        logger.exception("Error during readability analysis")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/web_access/check-alt-text/")
async def check_alt_text(
    image_url: str = Form(...), alt_text: str = Form(...)
) -> JSONResponse:
    """Evaluate provided alt text against the referenced image using an LLM."""
    logger.info("Checking alt-text for image: %s", image_url)
    try:
        result = AltTextChecker.check(jinja_environment, image_url, alt_text)
        logger.info("Alt-text evaluation completed.")
        return JSONResponse(
            content={"src": image_url, "alt_text": alt_text, "evaluation": result}
        )
    except Exception as e:
        logger.exception("Error during alt-text check")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/web_access/chat/")
async def chat_endpoint(prompt: str):
    """Stream chat completions from the configured LLM for a prompt."""
    logger.info("Chat prompt received")
    try:
        stream = await ChatHandler.chat(prompt, context="", stream=True)

        async def stream_response():
            async for chunk in stream:
                yield chunk

        return StreamingResponse(stream_response(), media_type="text/plain")
    except Exception as e:
        logger.exception("Chat streaming error")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/web_access/accessibility/")
async def check_accessibility(
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    guidelines_file: UploadFile | None = File(default=None),
):
    """Run WCAG-inspired checks, readability, and alt-text validation on HTML."""
    content: str | None = None
    source_name: str = "uploaded.html"

    # Prefer uploaded file if provided; otherwise fetch from URL
    if file is not None:
        try:
            content = (await file.read()).decode("utf-8")
            source_name = file.filename or source_name
        finally:
            try:
                await file.close()
            except Exception:
                pass
    elif url:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            # Use response text (requests handles encoding detection)
            content = resp.text
            source_name = url
        except Exception as e:
            return JSONResponse(content={"error": f"Failed to fetch URL: {e}"}, status_code=400)
    else:
        return JSONResponse(content={"error": "Either 'file' or 'url' must be provided."}, status_code=400)

    # Validate content and normalize type
    if content is None or not str(content).strip():
        return JSONResponse(content={"error": "Resolved content is empty"}, status_code=400)
    content_str: str = str(content)

    # Optional: read guidelines file and forward as LLM context
    context_str: str = ""
    if guidelines_file is not None:
        try:
            guidelines_bytes = await guidelines_file.read()
            guidelines_text = guidelines_bytes.decode("utf-8", errors="replace")
            context_str = f"Accessibility guidelines to follow (user-provided):\n\n{guidelines_text}"
        finally:
            try:
                await guidelines_file.close()
            except Exception:
                pass

    chunk_size: int = int(os.getenv("CHUNK_SIZE_FOR_ACCESSIBILITY", 3000))
    results = await run_accessibility_pipeline(
        content=content_str,
        filename=source_name,
        jinja_environment=jinja_environment,
        chunk_size=chunk_size,
        concurrency=4,
        context=context_str,
    )
    return StreamingResponse(
        stream_accessibility_results(results), media_type="application/jsonlines"
    )
