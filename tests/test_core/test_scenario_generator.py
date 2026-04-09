"""
Tests — Scenario Generator (ScenarioSampler).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from core.scenario_generator import ScenarioSampler
from core.interfaces import DistributionSpec, ScenarioModel


class TestScenarioSampler:

    def setup_method(self):
        self.rng = np.random.default_rng(42)

    # --- Distribution type coverage ---

    def test_normal_sampling(self):
        spec    = DistributionSpec(dist="normal", params={"mean": 5.0, "std": 1.0})
        samples = ScenarioSampler.sample(spec, self.rng, 2000)
        assert abs(np.mean(samples) - 5.0) < 0.1
        assert abs(np.std(samples) - 1.0) < 0.1

    def test_lognormal_sampling(self):
        spec    = DistributionSpec(dist="lognormal", params={"mean": 1.0, "std": 0.5})
        samples = ScenarioSampler.sample(spec, self.rng, 2000)
        assert np.all(samples > 0), "Lognormal must be strictly positive"

    def test_uniform_sampling(self):
        spec    = DistributionSpec(dist="uniform", params={"low": 2.0, "high": 8.0})
        samples = ScenarioSampler.sample(spec, self.rng, 2000)
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 8.0)
        assert abs(np.mean(samples) - 5.0) < 0.2

    def test_poisson_sampling(self):
        spec    = DistributionSpec(dist="poisson", params={"lam": 3.0})
        samples = ScenarioSampler.sample(spec, self.rng, 2000)
        assert np.all(samples >= 0)
        assert abs(np.mean(samples) - 3.0) < 0.2

    def test_bernoulli_sampling(self):
        spec    = DistributionSpec(dist="bernoulli", params={"p": 0.3})
        samples = ScenarioSampler.sample(spec, self.rng, 2000)
        assert set(np.unique(samples)).issubset({0.0, 1.0})
        assert abs(np.mean(samples) - 0.3) < 0.05

    def test_weibull_sampling(self):
        spec    = DistributionSpec(dist="weibull", params={"shape": 2.0, "scale": 1000.0})
        samples = ScenarioSampler.sample(spec, self.rng, 2000)
        assert np.all(samples > 0), "Weibull must be strictly positive"

    def test_beta_sampling(self):
        spec    = DistributionSpec(dist="beta", params={"alpha": 2.0, "beta": 5.0})
        samples = ScenarioSampler.sample(spec, self.rng, 2000)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_exponential_sampling(self):
        spec    = DistributionSpec(dist="exponential", params={"scale": 5.0})
        samples = ScenarioSampler.sample(spec, self.rng, 2000)
        assert np.all(samples > 0)
        assert abs(np.mean(samples) - 5.0) < 0.3

    def test_unknown_dist_raises(self):
        spec = DistributionSpec(dist="invalid_dist", params={})  # type: ignore
        with pytest.raises(ValueError):
            ScenarioSampler.sample(spec, self.rng)

    # --- Correlated sampling ---

    def test_correlated_group_preserves_correlation(self):
        """Correlated group samples should exhibit the intended correlation."""
        specs = {
            "x": DistributionSpec(dist="normal", params={"mean": 0.0, "std": 1.0}),
            "y": DistributionSpec(dist="normal", params={"mean": 0.0, "std": 1.0}),
        }
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        rng  = np.random.default_rng(100)

        pairs = [
            ScenarioSampler.sample_correlated_group(specs, corr, rng)
            for _ in range(2000)
        ]
        xs = np.array([p["x"] for p in pairs])
        ys = np.array([p["y"] for p in pairs])

        empirical_corr = np.corrcoef(xs, ys)[0, 1]
        assert abs(empirical_corr - 0.8) < 0.06

    # --- Random walk ---

    def test_random_walk_arithmetic(self):
        series = ScenarioSampler.generate_random_walk(
            self.rng, start=100.0, drift=0.0, volatility=5.0, steps=50, log_scale=False
        )
        assert len(series) == 50

    def test_random_walk_geometric_positive(self):
        series = ScenarioSampler.generate_random_walk(
            self.rng, start=100.0, drift=0.01, volatility=0.05, steps=100, log_scale=True
        )
        assert np.all(series > 0), "Geometric random walk must stay positive"

    # --- Scenario model ---

    def test_generate_scenario_from_model_independent(self):
        model = ScenarioModel(
            variables={
                "demand": DistributionSpec(dist="normal", params={"mean": 100, "std": 20}),
                "price":  DistributionSpec(dist="uniform", params={"low": 10, "high": 30}),
            },
            time_horizon=30,
            time_unit="days",
        )
        rng    = np.random.default_rng(0)
        result = ScenarioSampler.generate_scenario_from_model(model, rng)
        assert "demand" in result
        assert "price"  in result
        assert 10 <= result["price"] <= 30
