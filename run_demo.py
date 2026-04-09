"""
Simulation Engine — Local Demo.

A comprehensive demonstration of the full simulation platform:
1. Monte Carlo execution of all 6 domains.
2. Agent orchestration (Orchestrator → Sub-agents).
3. Optimization agent (Decision support).

Usage:
    python run_demo.py
"""
import asyncio
import json
from agents.orchestrator import OrchestratorAgent
from agents.optimization_agent import OptimizationAgent
from registry.domain_registry import DomainRegistry

async def run_demo():
    print("🚀 Starting Simulation Engine Demo...")
    orchestrator = OrchestratorAgent()
    optimizer = OptimizationAgent()

    # 1. Supply Chain Simulation
    print("\n--- [1] Supply Chain Simulation ---")
    sc_result = await orchestrator.run({
        "domain": "supply_chain",
        "inputs": {
            "stock_level": 500,
            "reorder_point": 400,
            "lead_time_days": 10,
            "supplier_country": "china",
            "shipping_mode": "sea",
            "mean_daily_demand": 20,
            "demand_cv": 0.2
        },
        "n_iterations": 1000
    })
    print(f"Stockout Probability: {sc_result['summary']['stockout_event']['mean']*100:.1f}%")
    print(f"Fill Rate (Mean): {sc_result['summary']['fill_rate']['mean']*100:.1f}%")

    # 2. Marketing Strategy (The NEW domain)
    print("\n--- [2] Marketing Strategy Simulation ---")
    mkt_result = await orchestrator.run({
        "domain": "marketing_strategy",
        "inputs": {
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
            "simulation_months": 12
        },
        "n_iterations": 1000
    })
    print(f"Mean ROI: {mkt_result['summary']['roi_pct']['mean']:.1f}%")
    print(f"LTV:CAC Ratio: {mkt_result['summary']['ltv_to_cac_ratio']['mean']:.2f}")

    # 3. Foot Traffic (ABM)
    print("\n--- [3] Foot Traffic ABM Simulation ---")
    ft_result = await orchestrator.run({
        "domain": "foot_traffic",
        "inputs": {
            "grid_width": 20,
            "grid_height": 20,
            "num_entry_points": 2,
            "num_customers": 50,
            "simulation_steps": 100,
            "obstacle_density": 0.1
        },
        "n_iterations": 100
    })
    print(f"Mean Dwell Time: {ft_result['summary']['mean_dwell_time']['mean']:.1f} steps")
    print(f"Congestion Score: {ft_result['summary']['congestion_score']['mean']:.2f}")

    # 4. Optimization Layer (Decision Support)
    print("\n--- [4] Optimization: Finding Best Marketing Mix ---")
    opt_result = await optimizer.run({
        "domain": "marketing_strategy",
        "goal_metric": "roi_pct",
        "goal_direction": "max",
        "fixed_inputs": {
            "total_budget_usd": 50000,
            "target_audience_size": 100000,
            "avg_ltv_usd": 250,
            "simulation_months": 12
        },
        "search_space": {
            "ads_budget_pct": [0.1, 0.6],
            "influencer_budget_pct": [0.1, 0.6],
            "seo_budget_pct": [0.1, 0.6],
            "virality_factor": [1.0, 2.0]
        },
        "n_iterations": 200,
        "n_trials": 8
    })
    print(f"Best ROI Found: {opt_result['best_value']:.1f}%")
    print(f"Optimal Inputs: {json.dumps(opt_result['best_inputs'], indent=2)}")

    print("\n✅ Demo Complete.")

if __name__ == "__main__":
    asyncio.run(run_demo())
