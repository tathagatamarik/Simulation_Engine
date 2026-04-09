"""
Marketing Strategy Monte Carlo Simulator — Advanced CAC/LTV Modeling.

Models a multi-channel marketing campaign with stochastic:
  - Conversion rates (Beta distribution — captures uncertainty around baselines)
  - Channel saturation decay (diminishing returns at high spend)
  - Virality amplification (lognormal — fat-tailed organic growth)
  - LTV uncertainty (lognormal — customer value varies)

Outputs: ROI distribution, CAC vs LTV, campaign success probability.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from domains.base import BaseDomainSimulator
from domains.marketing_strategy.schema import MarketingStrategyInput


# Channel saturation: at what spend fraction does diminishing returns kick in?
SATURATION_THRESHOLD = 0.6   # Above 60% of optimal budget → saturation
SATURATION_DECAY     = 0.25  # 25% conversion rate decay in saturation zone

# Beta distribution shape parameters for conversion rate uncertainty
# α, β chosen so that Beta(α, β) has mean ≈ baseline conversion rate
# and meaningful variance to model uncertainty
CONV_BETA_CONCENTRATION = 20.0  # Higher = tighter around mean (less uncertainty)


def _beta_params_from_mean(mean: float, concentration: float) -> tuple:
    """Convert a mean and concentration to Beta(α, β) params."""
    alpha = mean * concentration
    beta  = (1 - mean) * concentration
    return max(0.1, alpha), max(0.1, beta)


class MarketingStrategySimulator(BaseDomainSimulator):
    """
    Marketing Strategy Monte Carlo Simulator.

    Simulates a multi-channel campaign over a monthly horizon,
    aggregating customers acquired, total CAC, and ROI vs LTV.
    """

    domain       = "marketing_strategy"
    display_name = "Marketing Strategy Simulator"
    description  = (
        "Stochastic multi-channel marketing simulation modeling CAC, LTV, "
        "ROI distribution, virality effects, and campaign success probability."
    )
    schema = MarketingStrategyInput

    def simulate_once(
        self,
        inputs: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:

        budget          = float(inputs["total_budget_usd"])
        ads_pct         = float(inputs["ads_budget_pct"])
        inf_pct         = float(inputs["influencer_budget_pct"])
        seo_pct         = float(inputs["seo_budget_pct"])
        audience        = int(inputs["target_audience_size"])
        avg_ltv         = float(inputs["avg_ltv_usd"])
        base_ads_cr     = float(inputs["ads_conversion_rate"])
        base_inf_cr     = float(inputs["influencer_conversion_rate"])
        base_seo_cr     = float(inputs["seo_conversion_rate"])
        virality        = float(inputs["virality_factor"])
        cpc             = float(inputs["cost_per_click_usd"])
        months          = int(inputs["simulation_months"])

        # Monthly budget allocation
        monthly_budget = budget / months
        ads_spend      = monthly_budget * ads_pct
        inf_spend      = monthly_budget * inf_pct
        seo_spend      = monthly_budget * seo_pct

        # --- Stochastic conversion rates (Beta, uncertainty around baseline) ---
        def sample_cr(baseline: float) -> float:
            alpha, beta = _beta_params_from_mean(
                max(0.001, min(0.999, baseline)), CONV_BETA_CONCENTRATION
            )
            return float(rng.beta(alpha, beta))

        # --- Stochastic LTV (lognormal — fat tail of high-value customers) ---
        ltv_log_mean = np.log(avg_ltv)
        ltv_log_std  = 0.4  # ~41% coefficient of variation
        actual_ltv   = float(rng.lognormal(ltv_log_mean, ltv_log_std))

        # --- Virality amplification (lognormal centered on input) ---
        viral_actual = float(rng.lognormal(np.log(virality), 0.3))
        viral_actual = max(1.0, viral_actual)

        total_customers  = 0
        total_spend      = 0.0
        monthly_revenue  = 0.0

        for month in range(months):
            # --- Saturation: diminishing returns if spend is high ---
            ads_saturation  = 1.0 - SATURATION_DECAY * max(0.0, ads_pct - SATURATION_THRESHOLD)
            seo_saturation  = 1.0 - SATURATION_DECAY * max(0.0, seo_pct - SATURATION_THRESHOLD)

            # Sample this month's effective conversion rates
            ads_cr = sample_cr(base_ads_cr) * ads_saturation
            inf_cr = sample_cr(base_inf_cr)
            seo_cr = sample_cr(base_seo_cr) * seo_saturation

            # --- Reach calculations ---
            # Ads: clicks = spend / CPC, conversions = clicks × CR
            ads_clicks    = ads_spend / max(0.01, cpc)
            ads_customers = ads_clicks * ads_cr

            # Influencer: treat spend as direct reach proxy
            inf_reach     = (inf_spend / 100) * 10          # $100 → 10 impressions (rough)
            inf_customers = inf_reach * inf_cr

            # SEO: compound effect (traffic grows month over month)
            seo_base_traffic = (seo_spend / 50) * (1 + month * 0.05)   # Compounding
            seo_customers    = seo_base_traffic * seo_cr

            # Organic / viral amplification
            organic_customers = (ads_customers + inf_customers + seo_customers) * (viral_actual - 1)

            month_customers  = int(ads_customers + inf_customers + seo_customers + organic_customers)
            total_customers += month_customers
            total_spend     += monthly_budget
            monthly_revenue += month_customers * actual_ltv

        # --- Final metrics ---
        cac = total_spend / max(1, total_customers)
        roi = (monthly_revenue - total_spend) / max(1.0, total_spend) * 100.0
        ltv_to_cac = actual_ltv / max(0.01, cac)
        campaign_success = float(roi > 0 and ltv_to_cac >= 3.0)  # LTV:CAC ≥ 3 = success

        return {
            "total_customers_acquired": float(total_customers),
            "cac_usd":                  cac,
            "actual_ltv_usd":           actual_ltv,
            "ltv_to_cac_ratio":         ltv_to_cac,
            "roi_pct":                  roi,
            "total_revenue_usd":        monthly_revenue,
            "campaign_success":         campaign_success,
            "viral_amplification":      viral_actual,
        }

    def describe_outputs(self) -> List[str]:
        return [
            "total_customers_acquired", "cac_usd", "actual_ltv_usd",
            "ltv_to_cac_ratio", "roi_pct", "total_revenue_usd",
            "campaign_success", "viral_amplification",
        ]

    def get_failure_thresholds(self) -> Dict[str, float]:
        return {
            "roi_pct":          0.0,    # Risk of negative ROI
            "campaign_success": 0.5,    # Campaign success probability
            "ltv_to_cac_ratio": 3.0,    # P(LTV:CAC < 3) = risky
        }
