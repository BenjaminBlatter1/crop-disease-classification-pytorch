"""
Main FastAPI application for the crop disease inference service.

This module initializes the API, registers middleware, exception handlers,
and routes, and defines startup/shutdown lifecycle events.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from .dependencies.config import get_settings, Settings
from .dependencies.logging import setup_logging, RequestIdMiddleware
from .dependencies import exceptions as exc_handlers
from .routes import health as health_routes

settings: Settings = get_settings()

app: FastAPI = FastAPI(
    title=settings.api_name,
    version=settings.api_version,
)

# --- Middleware ---
setup_logging(settings.log_level)
app.add_middleware(RequestIdMiddleware)

# --- Exception handlers ---
app.add_exception_handler(
    RequestValidationError, exc_handlers.validation_exception_handler
)
app.add_exception_handler(Exception, exc_handlers.generic_exception_handler)

# --- Routers ---
app.include_router(health_routes.router, tags=["system"])

# --- Lifecycle events ---
@app.on_event("startup")
async def on_startup() -> None:
    """Run startup tasks such as model loading (Day 2)."""
    from .routes.health import READY_STATE
    READY_STATE["ready"] = True

@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Run cleanup tasks on shutdown."""
    pass
