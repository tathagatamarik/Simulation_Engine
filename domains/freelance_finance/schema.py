"""Freelance finance domain — input schema."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class FreelanceFinanceInput(BaseModel):
    """Input schema for Freelance Financial Risk simulator."""

    monthly_expenses: float = Field(
        ..., gt=0, description="Fixed monthly expenses (rent, subscriptions, etc.) in USD"
    )
    mean_monthly_income: float = Field(
        ..., gt=0, description="Expected average monthly client income (USD)"
    )
    income_std: float = Field(
        ..., ge=0, description="Standard deviation of monthly income (USD)"
    )
    savings_buffer: float = Field(
        ..., ge=0, description="Starting cash savings / runway buffer (USD)"
    )
    monthly_churn_prob: float = Field(
        0.05, ge=0.0, le=0.5,
        description="Probability of losing a client each month"
    )
    tax_rate: float = Field(
        0.25, ge=0.0, le=0.6,
        description="Effective tax rate on income (fraction)"
    )
    inflation_annual: float = Field(
        0.05, ge=0.0, le=0.3,
        description="Annual expense inflation rate"
    )
    simulation_months: int = Field(
        24, ge=3, le=120,
        description="Time horizon (months)"
    )
    n_iterations: int = Field(1000, ge=100, le=10_000)
    seed: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "monthly_expenses": 3000,
                "mean_monthly_income": 6500,
                "income_std": 2000,
                "savings_buffer": 15000,
                "monthly_churn_prob": 0.07,
                "tax_rate": 0.28,
                "inflation_annual": 0.06,
                "simulation_months": 24,
                "n_iterations": 2000,
                "seed": 42,
            }
        }
    }
