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
from jinja2 import Environment, FileSystemLoader

from app.core.chat_handler import ChatHandler
from app.core.utils import render_template
from app.website_accessibility.modules import (AltTextChecker,
                                               ReadabilityAnalyzer,
                                               split_chunks)

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
    """Get html, parse the text only, get readability scores"""
    logger.info("Received file for readability analysis: %s", file.filename)
    try:
        content = await file.read()
        soup = BeautifulSoup(content, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        lines = (line.strip() for line in soup.get_text().splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

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
    """Given a base64 encoded image or a url to an image, check for alt text"""
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
    """Send prompts to a running LLM"""
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
    """Check accessibility given a html/css/js file based on WCAG guidelines ,
    readability scores and checking if the alt text in an image matches the actual image
    """
    try:
        content = (await file.read()).decode("utf-8")
    finally:
        await file.close()

    chunk_size: int = int(os.getenv("CHUNK_SIZE_FOR_ACCESSIBILITY", 3000))
    chunks, ranges = split_chunks(content, chunk_size)

    sem = asyncio.Semaphore(4)  # Concurrency control

    async def process_chunk(i, chunk, start, end):
        async with sem:
            logger.info(f"[Chunk {i}] Processing lines {start}-{end}")
            try:
                prompt = render_template(
                    jinja_environment=jinja_environment,
                    template_name="build_chunk_prompt.txt",
                    filename=file.filename,
                    chunk=chunk,
                    idx=i,
                    total=len(chunks),
                    start_line=start,
                    end_line=end,
                )

                # Get response
                response = await ChatHandler.chat(prompt, context="", stream=False)

                # Parse score
                score_match = re.search(
                    r"\bScore[:\s]*([0-9]+(?:\.[0-9]+)?)", response, re.IGNORECASE
                )
                score = float(score_match.group(1)) if score_match else None

                # Alt text checks
                images = re.findall(r'<img[^>]+src="([^"]+)"[^>]*alt="([^"]+)"', chunk)
                image_feedback = []
                for src, alt in images:
                    try:
                        result = AltTextChecker.check(jinja_environment, src, alt)
                        image_feedback.append(
                            {"src": src, "alt_text": alt, "result": result}
                        )
                    except Exception as e:
                        error_msg = str(e)
                        image_feedback.append(
                            {"src": src, "alt_text": alt, "error": error_msg}
                        )

                return {
                    "chunk": i,
                    "start_line": start,
                    "end_line": end,
                    "score": score,
                    "image_feedback": image_feedback,
                    "llm_response": response,
                }
            except Exception as e:
                logger.exception(f"[Chunk {i}] Error during chunk processing")
                return {
                    "chunk": i,
                    "start_line": start,
                    "end_line": end,
                    "score": None,
                    "error": str(e),
                    "image_feedback": [],
                    "llm_response": None,
                }

    # Run all chunk tasks in parallel
    tasks = [
        process_chunk(i, chunk, start, end)
        for i, (chunk, (start, end)) in enumerate(zip(chunks, ranges))
    ]
    logger.info(f"Spawning {len(tasks)} chunk tasks")
    results = await asyncio.gather(*tasks)

    async def accessibility_stream():
        scores = [r["score"] for r in results if r.get("score") is not None]
        for r in results:
            yield json.dumps(r) + "\n"
        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        logger.info(f"Average score: {avg_score}")
        yield json.dumps({"average_score": avg_score}) + "\n"

    return StreamingResponse(accessibility_stream(), media_type="application/jsonlines")
