"""Results Router — GET /results/{run_id}."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from storage.result_store import ResultStore

router = APIRouter()


@router.get("/{run_id}")
async def get_result(run_id: str):
    """
    Poll for the result of an async simulation run.

    Returns:
        - `{"status": "running"}` while in progress.
        - `{"status": "completed", "result": {...}}` when done.
        - `{"status": "failed", "error": "..."}` on failure.
        - HTTP 404 if run_id unknown.
    """
    data = await ResultStore.load(run_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run ID '{run_id}' not found. It may have expired or never existed.",
        )
    return data
