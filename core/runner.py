"""
Simulation Runner.

Async wrapper over MonteCarloEngine.
Handles both small-N (in-process) and large-N (thread-offloaded) runs
without blocking the FastAPI event loop.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from core.interfaces import SimulationModule
from core.monte_carlo import MonteCarloEngine


class SimulationRunner:
    """
    Async simulation runner.

    Delegates to MonteCarloEngine while ensuring the FastAPI async event loop
    is never blocked by CPU-intensive simulation work.
    """

    PARALLEL_THRESHOLD = 200  # Use process pool above this count

    def __init__(self, max_workers: Optional[int] = None):
        self._engine = MonteCarloEngine(max_workers=max_workers)

    # ------------------------------------------------------------------
    # Async interface (used by FastAPI routes and agents)
    # ------------------------------------------------------------------

    async def run_async(
        self,
        module: SimulationModule,
        inputs: Dict[str, Any],
        n_iterations: int = 1000,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run simulation asynchronously.

        Offloads CPU-bound work to a thread pool executor so the
        FastAPI event loop stays responsive during large runs.

        Args:
            module:       Domain simulator instance.
            inputs:       Validated inputs dict.
            n_iterations: Number of Monte Carlo iterations.
            seed:         Master random seed.

        Returns:
            List of per-iteration result dicts.
        """
        loop = asyncio.get_event_loop()
        use_parallel = n_iterations >= self.PARALLEL_THRESHOLD

        results: List[Dict[str, Any]] = await loop.run_in_executor(
            None,  # Default ThreadPoolExecutor
            lambda: self._engine.run(
                module=module,
                inputs=inputs,
                n_iterations=n_iterations,
                seed=seed,
                use_parallel=use_parallel,
            ),
        )
        return results

    # ------------------------------------------------------------------
    # Sync interface (used by Celery workers)
    # ------------------------------------------------------------------

    def run_sync(
        self,
        module: SimulationModule,
        inputs: Dict[str, Any],
        n_iterations: int = 1000,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous run for Celery workers (no event loop needed).
        """
        use_parallel = n_iterations >= self.PARALLEL_THRESHOLD
        return self._engine.run(
            module=module,
            inputs=inputs,
            n_iterations=n_iterations,
            seed=seed,
            use_parallel=use_parallel,
        )
