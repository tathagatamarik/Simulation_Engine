"""Machine maintenance domain — input schema."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class MachineMaintInput(BaseModel):
    """Input schema for Industrial Machine Maintenance simulator."""

    machine_age_years: float = Field(
        ..., ge=0, description="Current age of the machine (years)"
    )
    daily_usage_hours: float = Field(
        ..., gt=0, le=24, description="Average daily operating hours"
    )
    machine_type: Literal["motor", "pump", "compressor", "conveyor", "cnc"] = Field(
        ..., description="Machine category (affects Weibull failure parameters)"
    )
    last_maintenance_days_ago: int = Field(
        ..., ge=0, description="Days since last maintenance"
    )
    maintenance_cost_usd: float = Field(
        ..., gt=0, description="Cost of a scheduled maintenance service (USD)"
    )
    failure_repair_cost_usd: float = Field(
        ..., gt=0, description="Cost of emergency failure repair (USD)"
    )
    simulation_days: int = Field(
        365, ge=30, le=1825, description="Time horizon (days)"
    )
    n_iterations: int = Field(1000, ge=100, le=10_000)
    seed: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "machine_age_years": 5.0,
                "daily_usage_hours": 16.0,
                "machine_type": "compressor",
                "last_maintenance_days_ago": 45,
                "maintenance_cost_usd": 800,
                "failure_repair_cost_usd": 5000,
                "simulation_days": 365,
                "n_iterations": 2000,
                "seed": 42,
            }
        }
    }
