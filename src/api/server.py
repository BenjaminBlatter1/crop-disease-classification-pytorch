"""
Main FastAPI application for the crop disease inference service.

This module initializes the API, registers middleware, exception handlers,
and routes, and defines a lifespan context manager that handles startup and
shutdown events. During startup, the model is loaded, preprocessing is
initialized, and the readiness probe is activated.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from .dependencies.config import get_settings, Settings
from .dependencies.logging import setup_logging, RequestIdMiddleware
from .dependencies import exceptions as exc_handlers
from .routes import health as health_routes
from .routes import predict as predict_routes

from .services.model_loader import load_model
from .services.preprocessing import get_preprocess_function

settings: Settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.

    Startup:
        - Load model
        - Initialize preprocessing
        - Store components in app.state
        - Mark readiness probe as ready

    Shutdown:
        - Reserved for future cleanup tasks
    """

    # --- Startup ---
    model = load_model(settings)
    preprocess = get_preprocess_function()

    app.state.MODEL = model
    app.state.PREPROCESS = preprocess

    # Extract class names if available (checkpoint models)
    if hasattr(model.model, "class_names"):
        app.state.CLASS_NAMES = model.model.class_names
    else:
        app.state.CLASS_NAMES = None
        
    print("Loaded class names:", app.state.CLASS_NAMES)

    # Mark API as ready
    from .routes.health import READY_STATE
    READY_STATE["ready"] = True

    yield  # <-- API runs here

    # --- Shutdown ---
    # Reserved for future cleanup (e.g., closing ONNX sessions, GPU memory cleanup)
    pass

app: FastAPI = FastAPI(
    title=settings.api_name,
    version=settings.api_version,
    lifespan=lifespan,
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
app.include_router(predict_routes.router, tags=["inference"])
