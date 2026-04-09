"""
Supply Chain Monte Carlo Simulator — Reference Implementation.

Models inventory dynamics over a configurable time horizon with:
  - Port congestion (Poisson, country-specific λ)
  - Customs delays (lognormal, shipping-mode-specific)
  - Weather disruption events (Bernoulli + Poisson severity)
  - Stochastic daily demand (Normal with configurable CV)
  - Reorder cycle logic (single reorder per simulation run)

This is the reference domain implementation demonstrating the full
plugin pattern: schema → simulator → agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from domains.base import BaseDomainSimulator
from domains.supply_chain.schema import SupplyChainInput


# ---------------------------------------------------------------------------
# Domain-Specific Parameter Tables
# ---------------------------------------------------------------------------

# Country-specific average port congestion delay (Poisson λ in extra days)
COUNTRY_CONGESTION: Dict[str, float] = {
    "china":      3.5,
    "india":      2.8,
    "usa":        1.2,
    "germany":    0.8,
    "mexico":     2.0,
    "vietnam":    3.0,
    "bangladesh": 3.2,
}

# Shipping-mode customs delay: lognormal params (underlying normal μ, σ)
# These produce ~exp(μ) median extra days
SHIPPING_DELAY: Dict[str, Dict[str, float]] = {
    "air":  {"mean": 0.3, "std": 0.4},   # median ~1.4 days
    "land": {"mean": 1.0, "std": 0.5},   # median ~2.7 days
    "sea":  {"mean": 2.0, "std": 0.7},   # median ~7.4 days
}

# Probability of a weather disruption event per shipment
WEATHER_DISRUPTION_PROB: Dict[str, float] = {
    "air":  0.05,
    "land": 0.08,
    "sea":  0.15,
}

# Severity of weather disruption (Poisson λ of extra days added)
WEATHER_SEVERITY_LAM: Dict[str, float] = {
    "air":  1.5,
    "land": 2.5,
    "sea":  6.0,
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class SupplyChainSimulator(BaseDomainSimulator):
    """
    Supply Chain & Logistics Monte Carlo Simulator.

    Single-period inventory model with stochastic lead time components
    and stochastic demand. Tracks:
      - Daily inventory level
      - Stockout events and duration
      - Units short (unmet demand)
      - Full lead time decomposition

    The reorder policy is a simple (s, Q) policy:
        When stock ≤ reorder_point, place one order.
        Order arrives after `effective_lead_time` days.
        Order quantity = safety stock + lead_time × mean_demand.
    """

    domain       = "supply_chain"
    display_name = "Supply Chain & Logistics Simulator"
    description  = (
        "Monte Carlo simulation of inventory risk, supplier delays, "
        "and stockout probability across a configurable time horizon."
    )
    schema = SupplyChainInput

    # ------------------------------------------------------------------
    # Core Simulation
    # ------------------------------------------------------------------

    def simulate_once(
        self,
        inputs: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """
        Single Monte Carlo iteration.

        Steps:
            1. Draw stochastic delay components.
            2. Simulate daily inventory loop.
            3. Return outcome metrics.
        """
        # Unpack inputs
        days           = inputs["simulation_days"]
        stock          = float(inputs["stock_level"])
        reorder_pt     = float(inputs["reorder_point"])
        lead_time      = int(inputs["lead_time_days"])
        mean_demand    = float(inputs["mean_daily_demand"])
        demand_cv      = float(inputs["demand_cv"])
        country        = inputs["supplier_country"]
        mode           = inputs["shipping_mode"]

        # --- Stochastic lead-time components ---
        congestion_days = int(rng.poisson(COUNTRY_CONGESTION.get(country, 2.0)))

        customs_days = float(
            rng.lognormal(SHIPPING_DELAY[mode]["mean"], SHIPPING_DELAY[mode]["std"])
        )

        weather_hit     = bool(rng.binomial(1, WEATHER_DISRUPTION_PROB[mode]))
        weather_days    = int(rng.poisson(WEATHER_SEVERITY_LAM[mode])) if weather_hit else 0

        effective_lead  = lead_time + congestion_days + int(customs_days) + weather_days

        # --- Daily inventory simulation ---
        current_stock    = stock
        stockout_days    = 0
        total_short      = 0.0
        reorder_placed   = False
        arrival_day      = None

        daily_demand_std = mean_demand * demand_cv  # std for normal demand draw

        for day in range(days):
            # Sample daily demand (clipped at 0)
            demand = max(0.0, float(rng.normal(mean_demand, daily_demand_std)))

            # Trigger reorder if not already outstanding
            if current_stock <= reorder_pt and not reorder_placed:
                reorder_placed = True
                arrival_day    = day + max(1, effective_lead)

            # Receive order
            if reorder_placed and arrival_day == day:
                replenishment  = reorder_pt + mean_demand * lead_time
                current_stock += replenishment
                reorder_placed = False

            # Consume demand — track shortfall
            if demand > current_stock:
                total_short   += demand - current_stock
                current_stock  = 0.0
                stockout_days += 1
            else:
                current_stock -= demand

        return {
            "stockout_event":            float(stockout_days > 0),
            "stockout_days":             float(stockout_days),
            "total_units_short":         total_short,
            "effective_lead_time_days":  float(effective_lead),
            "congestion_delay_days":     float(congestion_days),
            "customs_delay_days":        customs_days,
            "weather_delay_days":        float(weather_days),
            "weather_disruption":        float(weather_hit),
            "ending_stock":              max(0.0, current_stock),
            "fill_rate":                 1.0 - (total_short / max(1.0, mean_demand * days)),
        }

    # ------------------------------------------------------------------
    # Metadata Hooks
    # ------------------------------------------------------------------

    def describe_outputs(self) -> List[str]:
        return [
            "stockout_event",
            "stockout_days",
            "total_units_short",
            "effective_lead_time_days",
            "congestion_delay_days",
            "customs_delay_days",
            "weather_delay_days",
            "weather_disruption",
            "ending_stock",
            "fill_rate",
        ]

    def get_failure_thresholds(self) -> Dict[str, float]:
        return {
            "stockout_event": 0.5,   # P(stockout > 0)
            "stockout_days":  5.0,   # Risk of 5+ stockout days
            "fill_rate":      0.95,  # Risk of fill rate < 95% (inverted)
        }

    def get_time_series_config(self) -> Optional[Dict[str, Any]]:
        return None  # Supply chain uses cross-iteration aggregation, not time steps
