"""
Tests — Core Monte Carlo Engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.monte_carlo import MonteCarloEngine
from core.interfaces import SimulationModule
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Minimal test simulator (no domain needed)
# ---------------------------------------------------------------------------

class _SimpleInput(BaseModel):
    mu: float = 0.0
    sigma: float = 1.0
    n_iterations: int = 100
    seed: int = 42


class _SimpleSimulator(SimulationModule):
    """Draws a single normal sample per iteration — completely deterministic given seed."""

    domain = "_simple"
    schema = _SimpleInput

    def simulate_once(self, inputs, rng) -> dict:
        value = float(rng.normal(inputs["mu"], inputs["sigma"]))
        return {"value": value}

    def describe_outputs(self):
        return ["value"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMonteCarloEngine:

    def setup_method(self):
        self.engine = MonteCarloEngine()
        self.sim    = _SimpleSimulator()
        self.inputs = {"mu": 10.0, "sigma": 2.0, "n_iterations": 500, "seed": 42}

    def test_returns_correct_count(self):
        results = self.engine.run(self.sim, self.inputs, n_iterations=500, seed=42)
        assert len(results) == 500

    def test_each_result_has_expected_keys(self):
        results = self.engine.run(self.sim, self.inputs, n_iterations=100, seed=1)
        for r in results:
            assert "value" in r
            assert isinstance(r["value"], float)

    def test_deterministic_with_same_seed(self):
        """Same master seed must produce identical results."""
        r1 = self.engine.run(self.sim, self.inputs, n_iterations=200, seed=99)
        r2 = self.engine.run(self.sim, self.inputs, n_iterations=200, seed=99)
        vals1 = sorted([r["value"] for r in r1])
        vals2 = sorted([r["value"] for r in r2])
        assert vals1 == vals2

    def test_different_seeds_produce_different_results(self):
        r1 = self.engine.run(self.sim, self.inputs, n_iterations=200, seed=1)
        r2 = self.engine.run(self.sim, self.inputs, n_iterations=200, seed=2)
        vals1 = set([round(r["value"], 4) for r in r1])
        vals2 = set([round(r["value"], 4) for r in r2])
        # Very unlikely to be identical
        assert vals1 != vals2

    def test_statistical_accuracy(self):
        """Empirical mean should converge to μ=10 with large N."""
        results = self.engine.run(self.sim, self.inputs, n_iterations=5000, seed=7)
        values  = np.array([r["value"] for r in results])
        assert abs(np.mean(values) - 10.0) < 0.15    # Within 0.15 of true mean
        assert abs(np.std(values) - 2.0) < 0.15

    def test_sequential_vs_parallel_equivalence(self):
        """Sequential and parallel execution should produce same distribution shape."""
        r_seq = self.engine.run(self.sim, self.inputs, n_iterations=1000, seed=55, use_parallel=False)
        r_par = self.engine.run(self.sim, self.inputs, n_iterations=1000, seed=55, use_parallel=True)
        mean_seq = np.mean([r["value"] for r in r_seq])
        mean_par = np.mean([r["value"] for r in r_par])
        # Means should be close (same seed, same distribution)
        assert abs(mean_seq - mean_par) < 0.5

    def test_correlated_sampling(self):
        """Cholesky-correlated samples should have the specified correlation."""
        rng      = np.random.default_rng(42)
        means    = np.array([0.0, 0.0])
        stds     = np.array([1.0, 1.0])
        corr     = np.array([[1.0, 0.9], [0.9, 1.0]])

        samples  = np.array([
            MonteCarloEngine.sample_correlated(rng, means, stds, corr)
            for _ in range(2000)
        ])

        empirical_corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        assert abs(empirical_corr - 0.9) < 0.05   # Within 5% of target correlation
