"""
Industrial Machine Maintenance Monte Carlo Simulator.

Uses a Weibull reliability model (industry-standard for wear-out failures)
to simulate machine failure probability and optimal maintenance windows.

Stochastic components:
  - Weibull time-to-failure (shape/scale by machine type)
  - Random shock events (Poisson, e.g., power surges, operator error)
  - Maintenance effectiveness (Beta, not every service achieves 100%)

Outputs: failure probability, optimal maintenance window, expected costs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from domains.base import BaseDomainSimulator
from domains.machine_maintenance.schema import MachineMaintInput


# Weibull parameters by machine type
# shape (β > 1 = wear-out failures), scale (η = characteristic life in hours)
WEIBULL_PARAMS: Dict[str, Dict[str, float]] = {
    "motor":      {"shape": 2.5, "scale": 15_000},
    "pump":       {"shape": 2.0, "scale": 12_000},
    "compressor": {"shape": 2.8, "scale": 10_000},
    "conveyor":   {"shape": 1.8, "scale": 20_000},
    "cnc":        {"shape": 3.0, "scale": 8_000},
}

SHOCK_RATE            = 0.002   # Poisson λ: shock events per operating hour
SHOCK_LIFE_REDUCTION  = 0.15    # Each shock reduces remaining life by ~15%
MAINTENANCE_RESTORE   = 0.70    # Maintenance restores 70% of lost life (deterministic center)
MAINTENANCE_STD       = 0.10    # Stochastic ± around maintenance effectiveness


class MachineMaintSimulator(BaseDomainSimulator):
    """
    Industrial Machine Maintenance Monte Carlo Simulator.

    Models accumulated wear, shock events, and the value of
    preventive vs reactive maintenance strategies.
    """

    domain       = "machine_maintenance"
    display_name = "Industrial Machine Maintenance Simulator"
    description  = (
        "Weibull-based reliability simulation for industrial machinery. "
        "Computes failure probability curves, optimal maintenance windows, "
        "and cost comparison between preventive and reactive strategies."
    )
    schema = MachineMaintInput

    def simulate_once(
        self,
        inputs: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:

        days              = inputs["simulation_days"]
        age_years         = float(inputs["machine_age_years"])
        daily_hours       = float(inputs["daily_usage_hours"])
        machine_type      = inputs["machine_type"]
        last_maint_days   = int(inputs["last_maintenance_days_ago"])
        maint_cost        = float(inputs["maintenance_cost_usd"])
        repair_cost       = float(inputs["failure_repair_cost_usd"])

        wb = WEIBULL_PARAMS[machine_type]
        wb_shape = wb["shape"]
        wb_scale = wb["scale"]

        # Accumulated operating hours at start of simulation
        base_hours  = age_years * 365 * daily_hours
        hours_since_maint = last_maint_days * daily_hours

        # Effective accumulated hours (maintenance partially resets wear)
        maint_effectiveness = max(0.1, rng.normal(MAINTENANCE_RESTORE, MAINTENANCE_STD))
        effective_hours     = base_hours - hours_since_maint * maint_effectiveness

        # Weibull survival: S(t) = exp(-(t/η)^β)
        # Map current hours to a "pseudo-age" in terms of hazard
        current_wear = (max(0.0, effective_hours) / wb_scale) ** wb_shape

        # --- Simulate day by day ---
        failed            = False
        failure_day       = None
        optimal_maint_day = None
        total_cost        = 0.0
        shock_count       = 0

        for day in range(1, days + 1):
            daily_hrs   = daily_hours
            daily_wear  = (daily_hrs / wb_scale) ** wb_shape

            # --- Random shock events ---
            n_shocks = int(rng.poisson(SHOCK_RATE * daily_hrs))
            if n_shocks > 0:
                shock_count  += n_shocks
                current_wear += n_shocks * daily_wear * SHOCK_LIFE_REDUCTION

            current_wear += daily_wear

            # --- Failure probability this day (Weibull hazard increment) ---
            survival_now  = np.exp(-current_wear)
            fail_prob_day = 1.0 - survival_now

            if bool(rng.binomial(1, min(0.99, fail_prob_day))):
                if not failed:
                    failed      = True
                    failure_day = day
                    total_cost += repair_cost
                break

            # --- Suggest maintenance window: when failure prob > 20% ---
            if optimal_maint_day is None and fail_prob_day > 0.20:
                optimal_maint_day = day

        # Survival probability over the horizon
        final_survival = np.exp(-current_wear)
        failure_prob   = 1.0 - final_survival

        if not failed:
            total_cost = 0.0  # No unplanned repair needed

        return {
            "failed":                 float(failed),
            "failure_day":            float(failure_day) if failure_day else float(days),
            "failure_probability":    failure_prob,
            "optimal_maintenance_day": float(optimal_maint_day) if optimal_maint_day else float(days),
            "shock_events":           float(shock_count),
            "repair_cost":            total_cost,
            "preventive_cost_saving": max(0.0, repair_cost - maint_cost) if not failed else 0.0,
        }

    def describe_outputs(self) -> List[str]:
        return [
            "failed", "failure_day", "failure_probability",
            "optimal_maintenance_day", "shock_events",
            "repair_cost", "preventive_cost_saving",
        ]

    def get_failure_thresholds(self) -> Dict[str, float]:
        return {
            "failed":              0.5,   # P(failure)
            "failure_probability": 0.5,   # Critical failure risk
        }
