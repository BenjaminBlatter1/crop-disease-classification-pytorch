"""
System-level health and readiness endpoints.

- `/health` returns immediately and is used for liveness checks.
- `/ready` returns True only after the model is fully loaded (set in startup).
"""

from fastapi import APIRouter

router = APIRouter()

READY_STATE: dict[str, bool] = {"ready": False}

@router.get("/health")
def health() -> dict[str, str]:
    """Return basic liveness status."""
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, bool]:
    """Return readiness status (model loaded, warmup complete)."""
    return {"ready": READY_STATE["ready"]}
