"""
Tests — Supply Chain Domain Simulator.
"""
from __future__ import annotations

import numpy as np
import pytest

from domains.supply_chain.simulator import SupplyChainSimulator
from domains.supply_chain.schema import SupplyChainInput
from core.monte_carlo import MonteCarloEngine
from core.analysis import AnalysisEngine


BASE_INPUTS = {
    "stock_level":       500.0,
    "reorder_point":     400.0,
    "lead_time_days":    10,
    "supplier_country":  "china",
    "shipping_mode":     "sea",
    "mean_daily_demand": 20.0,
    "demand_cv":         0.2,
    "simulation_days":   90,
    "n_iterations":      1000,
    "seed":              42,
}



class TestSupplyChainSimulator:

    def setup_method(self):
        self.sim    = SupplyChainSimulator()
        self.engine = MonteCarloEngine()

    def test_schema_validates_valid_inputs(self):
        parsed = SupplyChainInput.model_validate(BASE_INPUTS)
        assert parsed.stock_level == 500.0
        assert parsed.supplier_country == "china"

    def test_schema_rejects_zero_stock(self):
        with pytest.raises(Exception):
            SupplyChainInput.model_validate({**BASE_INPUTS, "stock_level": 0})

    def test_simulate_once_returns_all_keys(self):
        rng    = np.random.default_rng(0)
        result = self.sim.simulate_once(BASE_INPUTS, rng)
        for key in self.sim.describe_outputs():
            assert key in result, f"Missing output key: {key}"

    def test_stockout_event_is_binary(self):
        rng    = np.random.default_rng(0)
        result = self.sim.simulate_once(BASE_INPUTS, rng)
        assert result["stockout_event"] in (0.0, 1.0)

    def test_fill_rate_in_range(self):
        rng    = np.random.default_rng(0)
        result = self.sim.simulate_once(BASE_INPUTS, rng)
        assert 0.0 <= result["fill_rate"] <= 1.0

    def test_ending_stock_non_negative(self):
        for seed in range(20):
            rng    = np.random.default_rng(seed)
            result = self.sim.simulate_once(BASE_INPUTS, rng)
            assert result["ending_stock"] >= 0.0

    def test_deterministic_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1   = self.sim.simulate_once(BASE_INPUTS, rng1)
        r2   = self.sim.simulate_once(BASE_INPUTS, rng2)
        assert r1 == r2

    def test_monte_carlo_run_produces_plausible_stockout_rate(self):
        """
        With sea shipment from China and low safety stock, expect non-trivial
        stockout probability (> 5%, < 80% for this config).
        """
        results = self.engine.run(
            self.sim, BASE_INPUTS, n_iterations=1000, seed=42, use_parallel=False
        )
        agg = AnalysisEngine.aggregate(results)
        p_stockout = agg["summary"]["stockout_event"]["mean"]
        assert 0.05 < p_stockout < 0.80, f"Unexpected stockout rate: {p_stockout}"

    def test_high_stock_reduces_stockout_risk(self):
        """More safety stock → lower stockout probability."""
        low_stock_inputs  = {**BASE_INPUTS, "stock_level":  200.0}
        high_stock_inputs = {**BASE_INPUTS, "stock_level": 1500.0}

        r_low  = self.engine.run(self.sim, low_stock_inputs,  n_iterations=500, seed=1, use_parallel=False)
        r_high = self.engine.run(self.sim, high_stock_inputs, n_iterations=500, seed=1, use_parallel=False)

        mean_low  = np.mean([r["stockout_event"] for r in r_low])
        mean_high = np.mean([r["stockout_event"] for r in r_high])
        assert mean_high < mean_low, "Higher stock should reduce stockouts"

    def test_analysis_aggregation_structure(self):
        results = self.engine.run(
            self.sim, BASE_INPUTS, n_iterations=200, seed=5, use_parallel=False
        )
        agg = AnalysisEngine.aggregate(results, self.sim.get_failure_thresholds())
        assert "summary"      in agg
        assert "risk_metrics" in agg
        assert "stockout_event" in agg["summary"]
        assert "p90" in agg["summary"]["stockout_event"]
        assert "failure_probability" in agg["risk_metrics"]["stockout_event"]
