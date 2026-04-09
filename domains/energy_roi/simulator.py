"""
Energy ROI Monte Carlo Simulator.

Models a solar PV system investment over a multi-year horizon with:
  - Stochastic solar irradiance (Beta distribution by climate zone)
  - Panel degradation over time (linear + random shock)
  - Electricity tariff drift (geometric random walk)
  - Weather efficiency loss events (Bernoulli)

Outputs: ROI %, payback period (months), cumulative energy savings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from domains.base import BaseDomainSimulator
from domains.energy_roi.schema import EnergyROIInput


# Climate zone → solar irradiance distribution (Beta α, β scaled to kWh/m²/day)
# Tropical: high/consistent; Polar: low/variable
IRRADIANCE_PARAMS: Dict[str, Dict[str, float]] = {
    "tropical":  {"alpha": 8.0, "beta": 2.0, "scale": 7.0},   # mean ~5.6 kWh/m²/day
    "temperate": {"alpha": 4.0, "beta": 4.0, "scale": 6.0},   # mean ~3.0 kWh/m²/day
    "arid":      {"alpha": 7.0, "beta": 3.0, "scale": 7.5},   # mean ~5.25 kWh/m²/day
    "polar":     {"alpha": 2.0, "beta": 6.0, "scale": 4.0},   # mean ~1.0 kWh/m²/day
}

PANEL_EFFICIENCY          = 0.20   # 20% panel efficiency (industry standard)
SYSTEM_EFFICIENCY         = 0.85   # Inverter + wiring losses
ANNUAL_DEGRADATION_RATE   = 0.005  # 0.5%/year panel degradation (lognormal center)
DEGRADATION_STD           = 0.002  # Stochastic degradation std
TARIFF_DRIFT_ANNUAL       = 0.03   # 3% annual tariff increase (drift)
TARIFF_VOLATILITY         = 0.05   # Tariff random walk volatility
WEATHER_LOSS_PROB         = 0.15   # P(adverse weather year)
WEATHER_LOSS_FACTOR       = 0.12   # 12% output reduction in bad year


class EnergyROISimulator(BaseDomainSimulator):
    """
    Renewable Energy ROI Monte Carlo Simulator.

    Simulates year-by-year solar generation and financial returns,
    accounting for stochastic irradiance, panel degradation, and tariff changes.
    """

    domain       = "energy_roi"
    display_name = "Renewable Energy ROI Simulator"
    description  = (
        "Stochastic ROI, payback period, and energy savings simulation "
        "for solar PV investments with climate-zone-specific irradiance profiles."
    )
    schema = EnergyROIInput

    def simulate_once(
        self,
        inputs: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:

        years            = inputs["simulation_years"]
        system_cost      = inputs["system_cost_usd"]
        bldg_sqm         = inputs["building_size_sqm"]
        monthly_kwh      = inputs["monthly_kwh_usage"]
        base_tariff      = inputs["electricity_tariff_usd_per_kwh"]
        feed_in_tariff   = inputs["feed_in_tariff_usd_per_kwh"]
        location         = inputs["location"]

        irr_params = IRRADIANCE_PARAMS[location]

        # Approximate panel area (30% of building footprint)
        panel_area_sqm = bldg_sqm * 0.30

        cumulative_savings = 0.0
        cumulative_revenue = 0.0
        tariff             = base_tariff
        payback_month      = None
        panel_efficiency   = PANEL_EFFICIENCY

        for year in range(1, years + 1):
            # --- Stochastic irradiance (Beta shaped by climate zone) ---
            raw_irr   = rng.beta(irr_params["alpha"], irr_params["beta"])
            irr_daily = raw_irr * irr_params["scale"]                       # kWh/m²/day
            annual_irr = irr_daily * 365

            # --- Stochastic panel degradation ---
            degradation = max(0.0, rng.normal(ANNUAL_DEGRADATION_RATE, DEGRADATION_STD))
            panel_efficiency = max(0.01, panel_efficiency * (1 - degradation))

            # --- Weather event ---
            weather_loss = WEATHER_LOSS_FACTOR if bool(rng.binomial(1, WEATHER_LOSS_PROB)) else 0.0

            # --- Annual solar generation ---
            annual_gen_kwh = (
                annual_irr
                * panel_area_sqm
                * panel_efficiency
                * SYSTEM_EFFICIENCY
                * (1 - weather_loss)
            )

            # --- Energy consumed vs exported ---
            annual_demand_kwh = monthly_kwh * 12
            direct_use_kwh    = min(annual_gen_kwh, annual_demand_kwh)
            exported_kwh      = max(0.0, annual_gen_kwh - annual_demand_kwh)

            # --- Tariff drift (geometric random walk) ---
            tariff_shock = rng.normal(TARIFF_DRIFT_ANNUAL, TARIFF_VOLATILITY)
            tariff       = tariff * (1 + tariff_shock)
            tariff       = max(0.01, tariff)

            # --- Savings & revenue ---
            year_savings = direct_use_kwh * tariff
            year_revenue = exported_kwh * feed_in_tariff

            cumulative_savings += year_savings + year_revenue
            cumulative_revenue += year_revenue

            # --- Payback detection ---
            if payback_month is None and cumulative_savings >= system_cost:
                payback_month = (year - 1) * 12 + 12 * (
                    1 - (cumulative_savings - system_cost) / max(1.0, year_savings + year_revenue)
                )

        roi_pct = ((cumulative_savings - system_cost) / system_cost) * 100.0

        return {
            "roi_pct":              roi_pct,
            "payback_months":       float(payback_month) if payback_month else float(years * 12),
            "cumulative_savings":   cumulative_savings,
            "cumulative_revenue":   cumulative_revenue,
            "net_profit":           cumulative_savings - system_cost,
            "did_break_even":       float(cumulative_savings >= system_cost),
        }

    def describe_outputs(self) -> List[str]:
        return [
            "roi_pct", "payback_months", "cumulative_savings",
            "cumulative_revenue", "net_profit", "did_break_even",
        ]

    def get_failure_thresholds(self) -> Dict[str, float]:
        return {
            "roi_pct":       0.0,   # Risk of negative ROI
            "did_break_even": 0.5,   # Break-even probability threshold
        }
