"""FastAPI app init for the AutoML+ service."""

import logging
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

from app.automlplus.router import router
from app.core.health import router as health_router
from app.core.logging import configure_service_logging

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_service_logging("automlplus")
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(health_router)
app.include_router(router)
