"""FastAPI endpoints for website accessibility analysis and chat utilities."""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
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

load_dotenv()

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
async def check_accessibility(file: UploadFile = File(...)):
    """Run WCAG-inspired checks, readability, and alt-text validation on HTML."""
    try:
        content = (await file.read()).decode("utf-8")
    finally:
        await file.close()

    chunk_size: int = int(os.getenv("CHUNK_SIZE_FOR_ACCESSIBILITY", 3000))
    results = await run_accessibility_pipeline(
        content=content,
        filename=file.filename or "uploaded.html",
        jinja_environment=jinja_environment,
        chunk_size=chunk_size,
        concurrency=4,
    )
    return StreamingResponse(
        stream_accessibility_results(results), media_type="application/jsonlines"
    )
