"""
Freelance Financial Risk Monte Carlo Simulator.

Models a freelancer's monthly cash flow over a time horizon with:
  - Stochastic client income (Normal with CV)
  - Client churn events (Bernoulli, with recovery delay)
  - Expense inflation (geometric drift)
  - Tax liability
  - Sudden income spike opportunities (Poisson)

Outputs: runway (months until insolvency), insolvency probability, net cash.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from domains.base import BaseDomainSimulator
from domains.freelance_finance.schema import FreelanceFinanceInput


class FreelanceFinanceSimulator(BaseDomainSimulator):
    """
    Freelance Financial Risk Monte Carlo Simulator.

    Simulates month-by-month cash position. Insolvency is defined as
    cash dropping to zero. Runway = months before insolvency (if applicable).
    """

    domain       = "freelance_finance"
    display_name = "Freelance Financial Risk Simulator"
    description  = (
        "Monte Carlo simulation of freelancer cash flow, runway, "
        "and insolvency risk under stochastic income and expense conditions."
    )
    schema = FreelanceFinanceInput

    def simulate_once(
        self,
        inputs: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:

        months          = inputs["simulation_months"]
        expenses        = float(inputs["monthly_expenses"])
        mean_income     = float(inputs["mean_monthly_income"])
        income_std      = float(inputs["income_std"])
        cash            = float(inputs["savings_buffer"])
        churn_prob      = float(inputs["monthly_churn_prob"])
        tax_rate        = float(inputs["tax_rate"])
        inflation_ann   = float(inputs["inflation_annual"])

        monthly_inflation = (1 + inflation_ann) ** (1 / 12) - 1

        insolvent        = False
        runway_months    = float(months)  # Default: survived full horizon
        total_income     = 0.0
        total_expenses   = 0.0
        income_multiplier = 1.0  # Client base multiplier (affected by churn)

        for month in range(1, months + 1):
            # --- Client churn event ---
            if bool(rng.binomial(1, churn_prob)):
                income_multiplier = max(0.1, income_multiplier * rng.uniform(0.5, 0.85))

            # --- Opportunistic client acquisition ---
            new_clients = int(rng.poisson(0.3))
            if new_clients > 0:
                income_multiplier = min(2.0, income_multiplier + new_clients * 0.1)

            # --- Stochastic monthly income ---
            gross_income = max(0.0, rng.normal(mean_income * income_multiplier, income_std))
            net_income   = gross_income * (1 - tax_rate)

            # --- Expense inflation ---
            expenses *= (1 + monthly_inflation)

            # --- Cash update ---
            net_cash_flow = net_income - expenses
            cash          = cash + net_cash_flow

            total_income   += net_income
            total_expenses += expenses

            # --- Insolvency check ---
            if cash <= 0 and not insolvent:
                insolvent     = True
                runway_months = float(month - 1)  # Survived through previous month
                cash          = 0.0

        ending_cash = max(0.0, cash)

        return {
            "runway_months":         runway_months,
            "insolvent":             float(insolvent),
            "ending_cash":           ending_cash,
            "total_net_income":      total_income,
            "total_expenses":        total_expenses,
            "net_surplus_deficit":   total_income - total_expenses,
            "income_client_mult_end": income_multiplier,
        }

    def describe_outputs(self) -> List[str]:
        return [
            "runway_months", "insolvent", "ending_cash",
            "total_net_income", "total_expenses",
            "net_surplus_deficit", "income_client_mult_end",
        ]

    def get_failure_thresholds(self) -> Dict[str, float]:
        return {
            "insolvent":     0.5,   # P(insolvency)
            "runway_months": 6.0,   # Risk of <6 month runway
        }
