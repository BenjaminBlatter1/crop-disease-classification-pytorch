"""
Prediction endpoints for the inference API.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from ..dependencies.inference import get_inference_components
from ..services.image_utils import decode_image
from ..services.inference_service import run_inference

router = APIRouter()

@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    request_id = getattr(request.state, "request_id", None)
    model, preprocess, class_names = get_inference_components(request)

    img_bytes = await file.read()
    img = decode_image(img_bytes)

    result = run_inference(img, model, preprocess, class_names)
    result["request_id"] = request_id
    return result


@router.post("/predict/batch")
async def predict_batch(request: Request, files: list[UploadFile] = File(...)):
    request_id = getattr(request.state, "request_id", None)
    model, preprocess, class_names = get_inference_components(request)

    results = []
    for f in files:
        img_bytes = await f.read()
        img = decode_image(img_bytes)
        results.append(run_inference(img, model, preprocess, class_names))

    return {"request_id": request_id, "results": results}


@router.post("/predict/url")
async def predict_url(request: Request, url: str):
    request_id = getattr(request.state, "request_id", None)
    model, preprocess, class_names = get_inference_components(request)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not download image")

    img = decode_image(resp.content)
    result = run_inference(img, model, preprocess, class_names)
    result["request_id"] = request_id
    return result
