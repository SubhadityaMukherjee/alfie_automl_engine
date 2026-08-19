"""Translate AutoML domain exceptions into HTTP responses.

The FastAPI routers in this project are kept intentionally thin: they bind
HTTP form parameters, delegate the multi-step pipeline work to an orchestrator
in the ``services``/``orchestrator`` layer, and map the typed exceptions raised
there to HTTP status codes via ``automl_exception_to_response``.

Keeping this mapping in one place means every AutoML endpoint translates
failures the same way and the status-code policy has a single source of truth.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AutoDWDownloadError,
    AutoDWUploadError,
    AutoMLError,
    AutoMLValidationError,
)


def automl_exception_to_response(exc: Exception) -> JSONResponse:
    """Map a raised exception onto a JSON error response.

    Mapping policy:

    * AutoMLValidationError (and its subclasses, e.g. bad inputs, unsupported
      file/task type) → ``400``.
    * AutoDWUploadError → the upstream status code carried on the exception
      (``502`` by default). Checked before the download branch because both
      share the AutoMLError hierarchy.
    * AutoDWDownloadError → ``502``.
    * Any other AutoMLError (runtime failures) → ``500``.
    * Any unrelated Exception → ``500``.

    The exception's ``str()`` is preserved verbatim as the ``error`` field so
    callers and tests can rely on stable, context-rich error text.
    """
    if isinstance(exc, AutoMLValidationError):
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if isinstance(exc, AutoDWUploadError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})
    if isinstance(exc, AutoDWDownloadError):
        return JSONResponse(status_code=502, content={"error": str(exc)})
    if isinstance(exc, AutoMLError):
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(status_code=500, content={"error": str(exc)})
