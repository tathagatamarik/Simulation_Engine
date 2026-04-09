"""
Celery Application — Async Task Worker.

For large simulation runs (n_iterations > 1000), tasks are dispatched
to Celery workers backed by Redis, enabling horizontal scaling.

Usage:
    Start worker: celery -A tasks.celery_app worker --loglevel=info
    Submit task:  run_simulation_task.delay(context_dict)
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "simulation_engine",
    broker  = REDIS_URL,
    backend = REDIS_URL,
)

celery_app.conf.update(
    task_serializer         = "json",
    result_serializer       = "json",
    accept_content          = ["json"],
    result_expires          = 3600,         # 1 hour TTL
    task_track_started      = True,
    worker_prefetch_multiplier = 1,         # One task at a time per worker
    task_acks_late          = True,         # Ack after completion, not on receive
)


@celery_app.task(bind=True, name="run_simulation")
def run_simulation_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task: runs a simulation synchronously inside a worker process.

    Args:
        context: Full orchestration context dict (domain, inputs, n_iterations, seed).

    Returns:
        SimulationResult.to_dict()
    """
    from agents.orchestrator import OrchestratorAgent

    orchestrator = OrchestratorAgent()

    # Run async orchestrator in a new event loop (Celery workers are sync processes)
    loop   = asyncio.new_event_loop()
    result = loop.run_until_complete(orchestrator.run(context))
    loop.close()

    return result
