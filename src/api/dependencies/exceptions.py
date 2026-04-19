"""
Centralized exception handlers for the inference API.

This module defines handlers for validation errors and unexpected internal
errors. All responses include a request ID for traceability.
"""

import logging
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

logger = logging.getLogger("api")

async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI/Pydantic validation errors."""
    request_id: str | None = getattr(request.state, "request_id", None)
    logger.warning(f"Validation error: {exc} (request_id={request_id})")

    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "detail": exc.errors(),
            "request_id": request_id,
        },
    )

async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected internal server errors."""
    request_id: str | None = getattr(request.state, "request_id", None)
    logger.error(f"Unhandled error: {exc} (request_id={request_id})", exc_info=True)

    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred.",
            "request_id": request_id,
        },
    )
