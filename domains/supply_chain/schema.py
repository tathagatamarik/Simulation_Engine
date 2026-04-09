"""
Supply Chain Domain — Input Schema.

Pydantic v2 model for validating supply chain simulation inputs.
Validation happens at the API gateway before any compute begins.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SupplyChainInput(BaseModel):
    """Input schema for the Supply Chain Monte Carlo simulator."""

    # ---------- Inventory parameters ----------
    stock_level: float = Field(
        ..., gt=0, description="Current on-hand inventory (units)"
    )
    reorder_point: float = Field(
        ..., gt=0, description="Reorder trigger inventory level (units)"
    )
    lead_time_days: int = Field(
        ..., gt=0, le=365, description="Baseline supplier lead time (days)"
    )

    # ---------- Supplier & shipping ----------
    supplier_country: Literal[
        "china", "india", "usa", "germany", "mexico", "vietnam", "bangladesh"
    ] = Field(..., description="Primary supplier country (determines congestion profile)")

    shipping_mode: Literal["air", "sea", "land"] = Field(
        ..., description="Primary shipping mode"
    )

    # ---------- Demand parameters ----------
    mean_daily_demand: float = Field(
        ..., gt=0, description="Average daily demand (units/day)"
    )
    demand_cv: float = Field(
        0.2, ge=0.0, le=1.0,
        description="Coefficient of variation for daily demand (0 = deterministic)"
    )

    # ---------- Simulation config ----------
    simulation_days: int = Field(
        90, ge=7, le=730,
        description="Time horizon for simulation (days)"
    )
    n_iterations: int = Field(
        1000, ge=100, le=10_000,
        description="Number of Monte Carlo iterations"
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility (None = random)"
    )

    @field_validator("stock_level")
    @classmethod
    def stock_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("stock_level must be > 0")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "stock_level": 500,
                "reorder_point": 150,
                "lead_time_days": 21,
                "supplier_country": "china",
                "shipping_mode": "sea",
                "mean_daily_demand": 20,
                "demand_cv": 0.25,
                "simulation_days": 90,
                "n_iterations": 2000,
                "seed": 42,
            }
        }
    }
