"""
Simulation Runner Agent.

Executes the Monte Carlo iterations for any domain simulator.
Uses SimulationRunner (async wrapper over MonteCarloEngine) to
offload CPU-bound computation without blocking the event loop.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.interfaces import Agent
from core.runner import SimulationRunner


class SimulationRunnerAgent(Agent):
    """
    Executes N iterations of a domain simulator.

    Expects context keys:
        - simulator:       BaseDomainSimulator instance
        - validated_inputs: dict of validated domain inputs
        - n_iterations:    int (or read from validated_inputs)
        - seed:            int | None

    Produces context key:
        - raw_results: List[Dict[str, Any]] — one dict per iteration
    """

    name = "simulation_runner_agent"

    def __init__(self, max_workers: Optional[int] = None):
        self._runner = SimulationRunner(max_workers=max_workers)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the simulation and attach raw_results to context.

        Args:
            context: Agent pipeline context dict.

        Returns:
            Context with 'raw_results' key added.
        """
        simulator       = context["simulator"]
        inputs          = context["validated_inputs"]
        n_iterations    = context.get("n_iterations") or inputs.get("n_iterations", 1000)
        seed            = context.get("seed") or inputs.get("seed")

        raw_results = await self._runner.run_async(
            module=simulator,
            inputs=inputs,
            n_iterations=n_iterations,
            seed=seed,
        )

        return {
            **context,
            "raw_results":   raw_results,
            "n_iterations":  n_iterations,
        }
