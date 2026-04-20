"""
Dependency providers for inference components.

This module exposes lightweight dependency functions that retrieve the
model, preprocessing function, and class names from the FastAPI application
state. This avoids circular imports and keeps routers clean.
"""

from fastapi import Request, HTTPException


def get_inference_components(request: Request):
    """Retrieve model, preprocessing function, and class names from app state."""
    model = request.app.state.MODEL
    preprocess = request.app.state.PREPROCESS
    class_names = request.app.state.CLASS_NAMES

    if model is None or preprocess is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    return model, preprocess, class_names
