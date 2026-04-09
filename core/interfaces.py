"""
Core Simulation Engine — Abstract Interfaces.

Defines the contracts that all domain modules, agents, and scenario models must satisfy.
These ABCs are the central contracts that make the plugin architecture work.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Type

import numpy as np
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Simulation Module (Domain Plugin Contract)
# ---------------------------------------------------------------------------

class SimulationModule(ABC):
    """
    Abstract base class for all domain simulation modules.

    Every domain (Supply Chain, Energy ROI, etc.) must implement this interface.
    The Monte Carlo engine calls `simulate_once` N times in parallel.

    Usage:
        class MyDomainSimulator(SimulationModule):
            domain = "my_domain"
            schema = MyInputSchema

            def simulate_once(self, inputs, rng):
                ...
                return {"metric_a": value, "metric_b": value}

            def describe_outputs(self):
                return ["metric_a", "metric_b"]
    """

    domain: str                       # Unique domain identifier (e.g., "supply_chain")
    schema: Type[BaseModel]           # Pydantic model for input validation

    @abstractmethod
    def simulate_once(
        self,
        inputs: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """
        Execute a single Monte Carlo iteration.

        Args:
            inputs: Validated domain inputs (dict form of schema).
            rng:    NumPy random generator — use exclusively for stochastic draws
                    to ensure reproducibility.

        Returns:
            Dict mapping output metric names → scalar values.
        """

    @abstractmethod
    def describe_outputs(self) -> List[str]:
        """Declare the list of output keys returned by simulate_once."""

    def validate_inputs(self, raw_inputs: Dict[str, Any]) -> BaseModel:
        """Validate raw input dict against the domain schema."""
        return self.schema.model_validate(raw_inputs)


# ---------------------------------------------------------------------------
# Agent Contract
# ---------------------------------------------------------------------------

class Agent(ABC):
    """
    Abstract base class for all orchestration agents.

    Agents are async actors that receive a context dict and return a result dict.
    They are composable — the Orchestrator chains multiple agents together.
    """

    name: str  # Human-readable agent identifier

    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this agent's responsibilities.

        Args:
            context: Input data / shared state from previous agents.

        Returns:
            Updated or new context dict with this agent's outputs.
        """


# ---------------------------------------------------------------------------
# Distribution & Scenario Model
# ---------------------------------------------------------------------------

@dataclass
class DistributionSpec:
    """
    Specifies a named statistical distribution for a stochastic variable.

    Supported distributions:
        normal      : params = {mean, std}
        lognormal   : params = {mean, std}  (underlying normal params)
        uniform     : params = {low, high}
        poisson     : params = {lam}
        triangular  : params = {left, mode, right}
        bernoulli   : params = {p}
        weibull     : params = {shape, scale}
        beta        : params = {alpha, beta}
        exponential : params = {scale}
    """

    dist: Literal[
        "normal", "lognormal", "uniform", "poisson",
        "triangular", "bernoulli", "weibull", "beta", "exponential",
    ]
    params: Dict[str, float]
    correlation_group: Optional[str] = None   # Group key for Cholesky correlation
    description: Optional[str] = None


@dataclass
class ScenarioModel:
    """
    Describes the complete stochastic variable model for a simulation scenario.

    Contains:
        - Named distribution specs for each variable
        - Optional correlation structure (for correlated draws)
        - Time horizon metadata
    """

    variables: Dict[str, DistributionSpec]     # variable_name → DistributionSpec
    time_horizon: int                           # How many steps to simulate
    time_unit: Literal["days", "months", "years"]
    correlations: Optional[Dict[str, Any]] = None  # {group: np.ndarray corr matrix}
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulation Result (Standardized Output Container)
# ---------------------------------------------------------------------------

class SimulationResult:
    """
    Standardized, serializable container for simulation run outputs.

    Produced by the AnalysisAgent after aggregating raw Monte Carlo results.
    This is the final API response payload.
    """

    def __init__(
        self,
        run_id: str,
        domain: str,
        n_iterations: int,
        summary: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        time_series: List[Dict[str, Any]],
        visualizations: Dict[str, Any],
        raw_samples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.run_id = run_id
        self.domain = domain
        self.n_iterations = n_iterations
        self.summary = summary
        self.risk_metrics = risk_metrics
        self.time_series = time_series
        self.visualizations = visualizations
        self.raw_samples = raw_samples
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "domain": self.domain,
            "n_iterations": self.n_iterations,
            "summary": self.summary,
            "risk_metrics": self.risk_metrics,
            "time_series": self.time_series,
            "visualizations": self.visualizations,
            "raw_samples": self.raw_samples,
            "metadata": self.metadata,
        }
