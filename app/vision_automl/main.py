"""FastAPI app init for the vision AutoML service."""

import logging
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

from app.core.health import router as health_router
from app.core.logging import configure_service_logging
from app.vision_automl.router import router

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_service_logging("vision_automl")
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(health_router)
app.include_router(router)
