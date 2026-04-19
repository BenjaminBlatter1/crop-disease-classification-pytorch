"""
Logging utilities and middleware for the inference API.

This module configures JSON-based logging and provides a middleware that
injects a unique request ID into each request. The middleware logs request
metadata including method, path, status code, and latency.
"""

import json
import logging
import time
import uuid
from typing import Callable, Awaitable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

def setup_logging(level: str = "INFO") -> None:
    """Configure global JSON logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )

class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique request ID and logs request metadata."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id: str = str(uuid.uuid4())
        request.state.request_id = request_id

        start: float = time.time()
        response: Response = await call_next(request)
        duration_ms: float = (time.time() - start) * 1000

        log_record = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }
        logging.getLogger("api").info(json.dumps(log_record))

        response.headers["X-Request-ID"] = request_id
        return response
