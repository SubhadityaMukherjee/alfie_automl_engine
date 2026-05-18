"""Pydantic response models for all API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str


class InstructionsResponse(BaseModel):
    instructions: str


# ---------------------------------------------------------------------------
# AutoML+  (/automlplus)
# ---------------------------------------------------------------------------


class AltTextCheckResponse(BaseModel):
    src: str
    alt_text: str
    evaluation: str


class ImagePromptResponse(BaseModel):
    response: str


class WebAccessibilityResponse(BaseModel):
    source: str
    average_score: float | None = None
    results: list[dict[str, Any]]
    readability: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Tabular AutoML  (/automl_tabular)
# ---------------------------------------------------------------------------


class TrainingSuccessResponse(BaseModel):
    message: str
    leaderboard: str


# ---------------------------------------------------------------------------
# Vision AutoML  (/automl_vision)
# ---------------------------------------------------------------------------


class VisionTrainingSuccessResponse(BaseModel):
    message: str
    leaderboard: str


class MultimodalTrainingSuccessResponse(BaseModel):
    message: str
    leaderboard: str
    auxiliary_columns: list[str]
