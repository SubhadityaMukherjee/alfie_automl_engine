"""Unified API router for all ALFIE AutoML engine services.

Every service router is mounted under a single ``/automl`` prefix so the
combined app exposes one URL scheme:

- ``/automl/tabular/...``     — tabular AutoML
- ``/automl/vision/...``      — vision AutoML
- ``/automl/automl_plus/...`` — AutoML+ (LLM/VLM tools)
"""

from fastapi import APIRouter

from app.automlplus.router import router as automl_plus_router
from app.tabular_automl.router import router as tabular_router
from app.vision_automl.router import router as vision_router

router = APIRouter(prefix="/automl")

router.include_router(tabular_router, prefix="/tabular")
router.include_router(vision_router, prefix="/vision")
router.include_router(automl_plus_router, prefix="/automl_plus")
