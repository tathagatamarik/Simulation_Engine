"""Marketing strategy domain — input schema."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class MarketingStrategyInput(BaseModel):
    """Input schema for Marketing Strategy Monte Carlo Simulator."""

    total_budget_usd: float = Field(
        ..., gt=0, description="Total marketing budget (USD)"
    )
    ads_budget_pct: float = Field(
        0.4, ge=0.0, le=1.0, description="Fraction allocated to paid ads"
    )
    influencer_budget_pct: float = Field(
        0.3, ge=0.0, le=1.0, description="Fraction allocated to influencer marketing"
    )
    seo_budget_pct: float = Field(
        0.3, ge=0.0, le=1.0, description="Fraction allocated to SEO / content"
    )

    # Audience
    target_audience_size: int = Field(
        ..., gt=0, description="Total addressable audience size"
    )
    avg_ltv_usd: float = Field(
        ..., gt=0, description="Average customer lifetime value (USD)"
    )

    # Channel assumptions
    ads_conversion_rate: float = Field(
        0.02, ge=0.0, le=1.0, description="Paid ads baseline conversion rate"
    )
    influencer_conversion_rate: float = Field(
        0.035, ge=0.0, le=1.0, description="Influencer baseline conversion rate"
    )
    seo_conversion_rate: float = Field(
        0.015, ge=0.0, le=1.0, description="SEO baseline conversion rate"
    )
    virality_factor: float = Field(
        1.1, ge=1.0, le=5.0, description="Organic amplification multiplier (1.0 = none)"
    )

    # Cost assumptions
    cost_per_click_usd: float = Field(
        2.5, gt=0, description="Average CPC for paid ads"
    )

    simulation_months: int = Field(
        12, ge=1, le=60, description="Campaign horizon (months)"
    )
    n_iterations: int = Field(1000, ge=100, le=10_000)
    seed: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_budget_usd": 50000,
                "ads_budget_pct": 0.4,
                "influencer_budget_pct": 0.35,
                "seo_budget_pct": 0.25,
                "target_audience_size": 100000,
                "avg_ltv_usd": 250,
                "ads_conversion_rate": 0.025,
                "influencer_conversion_rate": 0.04,
                "seo_conversion_rate": 0.02,
                "virality_factor": 1.3,
                "cost_per_click_usd": 2.0,
                "simulation_months": 12,
                "n_iterations": 2000,
                "seed": 42,
            }
        }
    }
