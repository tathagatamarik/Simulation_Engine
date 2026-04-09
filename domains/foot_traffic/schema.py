"""Foot traffic domain — input schema."""
from __future__ import annotations
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class FootTrafficInput(BaseModel):
    """Input schema for Foot Traffic Agent-Based Model."""

    grid_width: int = Field(
        20, ge=5, le=100, description="Floor plan grid width (cells)"
    )
    grid_height: int = Field(
        20, ge=5, le=100, description="Floor plan grid height (cells)"
    )
    num_entry_points: int = Field(
        2, ge=1, le=10, description="Number of entry/exit points"
    )
    num_customers: int = Field(
        50, ge=5, le=500, description="Number of customer agents per run"
    )
    simulation_steps: int = Field(
        100, ge=10, le=1000, description="Number of ABM time steps"
    )
    obstacle_density: float = Field(
        0.1, ge=0.0, le=0.5, description="Fraction of grid cells blocked (aisles, shelves)"
    )
    n_iterations: int = Field(
        500, ge=50, le=5000,
        description="Number of ABM simulation runs (lower for ABM due to cost)"
    )
    seed: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "grid_width": 20,
                "grid_height": 20,
                "num_entry_points": 3,
                "num_customers": 60,
                "simulation_steps": 120,
                "obstacle_density": 0.12,
                "n_iterations": 200,
                "seed": 42,
            }
        }
    }
