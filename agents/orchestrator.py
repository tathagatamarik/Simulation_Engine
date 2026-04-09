"""
Orchestrator Agent — Central Coordinator.

The OrchestratorAgent chains all sub-agents together to execute a complete
simulation run from raw user request to final visualization-ready output.

Pipeline:
    User Request
        → DomainAgent       (validate inputs, attach simulator)
        → ScenarioAgent     (build/pass-through ScenarioModel)
        → RunnerAgent       (execute N iterations)
        → AnalysisAgent     (aggregate, compute risk metrics)
        → VisualizationAgent (build chart-ready JSON)
        → SimulationResult  (standardized output)

The Orchestrator is domain-agnostic: it discovers the correct DomainAgent
from the registry and chains a fixed pipeline of specialist agents.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from core.interfaces import Agent, SimulationResult
from agents.scenario_agent import ScenarioGeneratorAgent
from agents.runner_agent import SimulationRunnerAgent
from agents.analysis_agent import AnalysisAgent
from agents.visualization_agent import VisualizationAgent


class OrchestratorAgent(Agent):
    """
    Top-level orchestrator for the simulation engine.

    Usage:
        orchestrator = OrchestratorAgent()
        result = await orchestrator.run({
            "domain": "supply_chain",
            "inputs": { ... },
        })
        # result is a SimulationResult.to_dict()
    """

    name = "orchestrator_agent"

    def __init__(self, max_workers: Optional[int] = None):
        self._scenario_agent  = ScenarioGeneratorAgent()
        self._runner_agent    = SimulationRunnerAgent(max_workers=max_workers)
        self._analysis_agent  = AnalysisAgent()
        self._viz_agent       = VisualizationAgent()

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full agent pipeline for one simulation request.

        Args:
            context: Must contain:
                - domain:        str   (e.g., "supply_chain")
                - inputs:        dict  (raw user inputs before validation)
                - n_iterations:  int   (optional override)
                - seed:          int   (optional master seed)

        Returns:
            SimulationResult.to_dict() — standardized output payload.

        Raises:
            KeyError:   If domain not found in registry.
            ValueError: If input validation fails.
        """
        from registry.domain_registry import DomainRegistry  # Late import avoids circular

        domain   = context["domain"]
        run_id   = context.get("run_id", str(uuid.uuid4()))

        # 1. Resolve domain agent from registry
        domain_agent = DomainRegistry.get_agent(domain)

        # 2. Domain agent: validate inputs, attach simulator
        ctx = await domain_agent.run({**context, "run_id": run_id})

        # 3. Scenario agent: build ScenarioModel
        ctx = await self._scenario_agent.run(ctx)

        # 4. Runner agent: execute N iterations
        ctx = await self._runner_agent.run(ctx)

        # 5. Analysis agent: aggregate + risk metrics
        ctx = await self._analysis_agent.run(ctx)

        # 6. Visualization agent: build chart-ready JSON
        ctx = await self._viz_agent.run(ctx)

        # 7. Assemble standardized result
        result = SimulationResult(
            run_id        = run_id,
            domain        = domain,
            n_iterations  = ctx["n_iterations"],
            summary       = ctx["analysis"]["summary"],
            risk_metrics  = ctx["analysis"]["risk_metrics"],
            time_series   = ctx["analysis"]["time_series"],
            visualizations= ctx["visualizations"],
            metadata      = {
                "domain_metadata": ctx.get("domain_metadata", {}),
                "seed":            context.get("seed"),
            },
        )

        return result.to_dict()

    # ------------------------------------------------------------------
    # Convenience: run multiple domains in one call
    # ------------------------------------------------------------------

    async def run_multi(self, requests: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Run multiple simulation requests concurrently.

        Args:
            requests: List of context dicts, each with 'domain' and 'inputs'.

        Returns:
            List of SimulationResult dicts.
        """
        import asyncio
        tasks = [self.run(req) for req in requests]
        return await asyncio.gather(*tasks)
