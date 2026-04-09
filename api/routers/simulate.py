"""
Simulation Router — POST /simulate/{domain}.

Handles both:
  - Synchronous runs (small N, fast): returns result immediately
  - Asynchronous runs (large N, slow): queues Celery task, returns run_id

Threshold: n_iterations <= 1000 → sync | > 1000 → async via Celery
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from agents.orchestrator import OrchestratorAgent
from storage.result_store import ResultStore

router = APIRouter()
orchestrator = OrchestratorAgent()


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Generic simulation request payload."""
    inputs: Dict[str, Any]
    n_iterations: Optional[int] = None
    seed: Optional[int] = None
    async_mode: bool = False      # Force async even for small runs


class SimulationResponse(BaseModel):
    """Synchronous simulation response."""
    run_id: str
    domain: str
    status: str = "completed"
    result: Dict[str, Any]


class AsyncSimulationResponse(BaseModel):
    """Async simulation enqueue response."""
    run_id: str
    domain: str
    status: str = "queued"
    poll_url: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

ASYNC_THRESHOLD = 1000   # n_iterations above which we go async


@router.post("/{domain}", response_model=None)
async def run_simulation(
    domain: str,
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Run a simulation for the specified domain.

    - **domain**: Domain identifier (e.g., `supply_chain`, `energy_roi`).
    - **inputs**: Domain-specific input parameters (see `/docs` for examples).
    - **n_iterations**: Override number of Monte Carlo iterations.
    - **seed**: Random seed for reproducibility.
    - **async_mode**: Force async processing (returns run_id to poll).

    **Synchronous** (n_iterations ≤ 1000): Returns full result immediately.
    **Asynchronous** (n_iterations > 1000 or async_mode=True): Returns `run_id`.
    """
    from registry.domain_registry import DomainRegistry

    # Validate domain exists before spinning up work
    if not DomainRegistry.is_registered(domain):
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain}' not found. Available: {[d['key'] for d in DomainRegistry.list_domains()]}",
        )

    run_id      = str(uuid.uuid4())
    n_iter      = request.n_iterations or request.inputs.get("n_iterations", 1000)
    use_async   = request.async_mode or n_iter > ASYNC_THRESHOLD

    context = {
        "domain":       domain,
        "inputs":       request.inputs,
        "n_iterations": n_iter,
        "seed":         request.seed,
        "run_id":       run_id,
    }

    if use_async:
        # Queue as background task (swap for Celery task in production)
        background_tasks.add_task(_run_and_store, context, run_id)
        return AsyncSimulationResponse(
            run_id   = run_id,
            domain   = domain,
            status   = "queued",
            poll_url = f"/results/{run_id}",
        )
    else:
        # Synchronous execution
        try:
            result = await orchestrator.run(context)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Cache result
        await ResultStore.save(run_id, {"status": "completed", "result": result})

        return SimulationResponse(
            run_id = run_id,
            domain = domain,
            result = result,
        )


async def _run_and_store(context: Dict[str, Any], run_id: str) -> None:
    """Background coroutine: run simulation and store result."""
    store = ResultStore
    try:
        await store.save(run_id, {"status": "running"})
        result = await orchestrator.run(context)
        await store.save(run_id, {"status": "completed", "result": result})
    except Exception as exc:
        await store.save(run_id, {"status": "failed", "error": str(exc)})
