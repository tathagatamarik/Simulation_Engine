"""
Scenario Generator Agent.

Wraps the ScenarioSampler to build a ScenarioModel from domain context.
In the agent pipeline, this agent enriches context with variable distributions
before the SimulationRunnerAgent executes iterations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.interfaces import Agent, DistributionSpec, ScenarioModel
from core.scenario_generator import ScenarioSampler


class ScenarioGeneratorAgent(Agent):
    """
    Builds a ScenarioModel from simulation context.

    Currently passes through domain-provided scenario config.
    In future phases, this agent will:
      - Learn distributions from historical data
      - Apply scenario stress-tests (bull/bear/base cases)
      - Generate correlated scenario families
    """

    name = "scenario_generator_agent"

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build or pass through a ScenarioModel.

        If the domain provides a pre-built scenario_model in context,
        it's used as-is. Otherwise, a minimal default is constructed.

        Args:
            context: Must contain 'validated_inputs' and 'simulator'.

        Returns:
            Context enriched with 'scenario_model'.
        """
        # If domain already supplied a scenario model, pass it through
        if "scenario_model" in context:
            return context

        inputs = context.get("validated_inputs", {})

        # Build a minimal scenario model capturing the time horizon
        simulation_days   = inputs.get("simulation_days", 90)
        simulation_months = inputs.get("simulation_months", 12)
        simulation_years  = inputs.get("simulation_years", 5)

        if "simulation_days" in inputs:
            horizon, unit = simulation_days, "days"
        elif "simulation_months" in inputs:
            horizon, unit = simulation_months, "months"
        else:
            horizon, unit = simulation_years, "years"

        scenario_model = ScenarioModel(
            variables={},        # Specific variables managed per-domain simulator
            time_horizon=horizon,
            time_unit=unit,
        )

        return {**context, "scenario_model": scenario_model}

    # ------------------------------------------------------------------
    # Utility: build scenario model from explicit variable specs
    # ------------------------------------------------------------------

    @staticmethod
    def build_from_specs(
        variable_specs: Dict[str, Dict],
        time_horizon: int,
        time_unit: str = "days",
        correlations: Optional[Dict] = None,
    ) -> ScenarioModel:
        """
        Helper to build a ScenarioModel from a plain dict of spec configs.

        Args:
            variable_specs: {name: {"dist": ..., "params": {...}}}
            time_horizon:   Number of time steps.
            time_unit:      "days" | "months" | "years".
            correlations:   Optional correlation groups.

        Returns:
            ScenarioModel
        """
        variables = {
            name: DistributionSpec(
                dist=spec["dist"],
                params=spec["params"],
                correlation_group=spec.get("correlation_group"),
                description=spec.get("description"),
            )
            for name, spec in variable_specs.items()
        }
        return ScenarioModel(
            variables=variables,
            time_horizon=time_horizon,
            time_unit=time_unit,
            correlations=correlations,
        )
