"""
Domain Agent Base Class.

Provides the standard domain agent implementation:
  1. Validate inputs against domain schema
  2. Instantiate the domain simulator
  3. Pass through to the runner (called by OrchestratorAgent)

All domain agents subclass this with just two class attributes:
    name         = "my_domain_agent"
    simulator_cls = MyDomainSimulator
"""
from __future__ import annotations

from typing import Any, Dict, Type

from core.interfaces import Agent
from domains.base import BaseDomainSimulator


class DomainAgent(Agent):
    """
    Generic domain agent.

    Responsibilities:
      - Validate domain-specific inputs via the simulator's schema.
      - Expose the instantiated simulator for the runner.
      - Surface domain metadata for the orchestrator.
    """

    name: str = "domain_agent"
    simulator_cls: Type[BaseDomainSimulator]

    def __init__(self):
        self._simulator = self.simulator_cls()

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate inputs and attach simulator to context.

        The actual simulation is delegated to SimulationRunnerAgent,
        which receives the simulator instance via context.
        """
        raw_inputs = context.get("inputs", {})

        # Validate inputs — raises Pydantic ValidationError on bad input
        validated = self._simulator.validate_inputs(raw_inputs)
        validated_dict = validated.model_dump()

        return {
            **context,
            "validated_inputs": validated_dict,
            "simulator": self._simulator,
            "domain_metadata": self._simulator.metadata(),
        }

    @property
    def simulator(self) -> BaseDomainSimulator:
        return self._simulator
