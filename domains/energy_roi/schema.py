"""Energy ROI domain — input schema."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class EnergyROIInput(BaseModel):
    """Input schema for Renewable Energy ROI simulation."""

    location: Literal["tropical", "temperate", "arid", "polar"] = Field(
        ..., description="Climate zone (determines solar irradiance profile)"
    )
    building_size_sqm: float = Field(
        ..., gt=0, description="Building floor area in square meters"
    )
    monthly_kwh_usage: float = Field(
        ..., gt=0, description="Average monthly electricity consumption (kWh)"
    )
    system_cost_usd: float = Field(
        ..., gt=0, description="Total solar installation cost (USD)"
    )
    electricity_tariff_usd_per_kwh: float = Field(
        0.15, gt=0, description="Current electricity tariff (USD/kWh)"
    )
    feed_in_tariff_usd_per_kwh: float = Field(
        0.08, ge=0, description="Feed-in tariff for excess solar (USD/kWh)"
    )
    simulation_years: int = Field(
        25, ge=5, le=50, description="Investment time horizon (years)"
    )
    n_iterations: int = Field(1000, ge=100, le=10_000)
    seed: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "location": "tropical",
                "building_size_sqm": 250,
                "monthly_kwh_usage": 800,
                "system_cost_usd": 15000,
                "electricity_tariff_usd_per_kwh": 0.18,
                "feed_in_tariff_usd_per_kwh": 0.08,
                "simulation_years": 25,
                "n_iterations": 2000,
                "seed": 42,
            }
        }
    }
