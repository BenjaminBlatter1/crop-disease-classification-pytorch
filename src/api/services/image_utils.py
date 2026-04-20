"""
Image decoding and validation utilities.
"""

import io
from fastapi import HTTPException
from PIL import Image


def decode_image(file_bytes: bytes) -> Image.Image:
    """Decode raw bytes into a PIL RGB image."""
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
