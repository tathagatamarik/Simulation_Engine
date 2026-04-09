"""
Base Domain Simulator.

Extends SimulationModule with domain-level metadata, failure threshold
configuration, and time-series config hooks that agents consume.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
from pydantic import BaseModel

from core.interfaces import SimulationModule


class BaseDomainSimulator(SimulationModule):
    """
    Extended base class for all domain simulators.

    Beyond the core SimulationModule contract, provides:
      - display_name / description / version metadata
      - failure_threshold configuration (used by AnalysisEngine)
      - time_series config hook (used by VisualizationAgent)
      - standardized metadata() export

    All domain modules must subclass this, not SimulationModule directly.
    """

    domain: str = "base"
    display_name: str = "Base Simulator"
    description: str = ""
    version: str = "1.0.0"

    @abstractmethod
    def simulate_once(
        self,
        inputs: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Single Monte Carlo iteration."""

    @abstractmethod
    def describe_outputs(self) -> List[str]:
        """Declare output metric keys."""

    def get_failure_thresholds(self) -> Optional[Dict[str, float]]:
        """
        Override to define failure thresholds for risk metric calculation.

        Returns:
            Dict mapping output metric keys to threshold values.
            AnalysisEngine will compute P(metric >= threshold) for each.
        """
        return None

    def get_time_series_config(self) -> Optional[Dict[str, Any]]:
        """
        Override to configure time-series extraction.

        Returns:
            {
              "time_key":   str,         # key identifying the time step in results
              "value_keys": list[str],   # metric keys to aggregate over time
            }
        """
        return None

    def metadata(self) -> Dict[str, Any]:
        """Export domain metadata for the registry and API /domains endpoint."""
        return {
            "domain":       self.domain,
            "display_name": self.display_name,
            "description":  self.description,
            "version":      self.version,
            "outputs":      self.describe_outputs(),
        }
