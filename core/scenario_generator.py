"""
Scenario Generator.

Samples stochastic variables from DistributionSpec configurations.
Supports all distribution types used across domain modules,
including correlated multi-variate sampling via Gaussian copula.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

from core.interfaces import DistributionSpec, ScenarioModel


class ScenarioSampler:
    """
    Samples individual variables from DistributionSpec objects.

    Supports 9 distribution types, correlated group sampling via
    Gaussian copula, and time-series scenario generation.
    """

    # ------------------------------------------------------------------
    # Single-variable sampling
    # ------------------------------------------------------------------

    @staticmethod
    def sample(
        spec: DistributionSpec,
        rng: np.random.Generator,
        size: int = 1,
    ) -> np.ndarray:
        """
        Sample `size` values from a DistributionSpec.

        Args:
            spec: Distribution specification.
            rng:  NumPy random generator (caller-owned for reproducibility).
            size: Number of samples.

        Returns:
            Array of shape (size,).
        """
        p = spec.params

        match spec.dist:
            case "normal":
                return rng.normal(p["mean"], p["std"], size)

            case "lognormal":
                # mean/std are params of the underlying normal distribution
                return rng.lognormal(p["mean"], p["std"], size)

            case "uniform":
                return rng.uniform(p["low"], p["high"], size)

            case "poisson":
                return rng.poisson(p["lam"], size).astype(float)

            case "triangular":
                return rng.triangular(p["left"], p["mode"], p["right"], size)

            case "bernoulli":
                return rng.binomial(1, p["p"], size).astype(float)

            case "weibull":
                # Scale * (-ln U)^(1/shape) — equivalent to scipy.weibull_min
                u = rng.uniform(1e-12, 1.0, size)
                return p["scale"] * (-np.log(u)) ** (1.0 / p["shape"])

            case "beta":
                return rng.beta(p["alpha"], p["beta"], size)

            case "exponential":
                return rng.exponential(p["scale"], size)

            case _:
                raise ValueError(f"Unknown distribution type: '{spec.dist}'")

    # ------------------------------------------------------------------
    # Correlated group sampling (Gaussian copula)
    # ------------------------------------------------------------------

    @staticmethod
    def sample_correlated_group(
        specs: Dict[str, DistributionSpec],
        corr_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """
        Sample a group of potentially non-normal variables with correlation.

        Uses a Gaussian copula:
            1. Draw correlated standard normals using Cholesky.
            2. Map to uniform via normal CDF.
            3. Map uniform → target marginals via inverse CDF.

        Args:
            specs:       Dict of variable_name → DistributionSpec.
            corr_matrix: Correlation matrix (n × n). Must be PSD.
            rng:         Random generator.

        Returns:
            Dict of variable_name → sampled scalar value.
        """
        keys = list(specs.keys())
        n = len(keys)

        # Step 1: Correlated standard normals
        corr_matrix = corr_matrix + np.eye(n) * 1e-8  # Regularize
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            L = np.eye(n)  # Fallback to independence
        z = rng.standard_normal(n)
        corr_z = L @ z  # shape (n,)

        # Step 2: Gaussian CDF → uniform
        u = scipy_stats.norm.cdf(corr_z)

        # Step 3: Quantile transform to target marginals
        results: Dict[str, float] = {}
        for i, key in enumerate(keys):
            results[key] = float(ScenarioSampler._ppf(specs[key], float(u[i])))

        return results

    @staticmethod
    def _ppf(spec: DistributionSpec, u: float) -> float:
        """Inverse CDF (quantile function) for a distribution spec."""
        p = spec.params
        u = float(np.clip(u, 1e-9, 1 - 1e-9))  # Avoid ±∞ at boundaries

        match spec.dist:
            case "normal":
                return float(scipy_stats.norm.ppf(u, p["mean"], p["std"]))
            case "lognormal":
                return float(scipy_stats.lognorm.ppf(u, p["std"], scale=np.exp(p["mean"])))
            case "uniform":
                return float(scipy_stats.uniform.ppf(u, p["low"], p["high"] - p["low"]))
            case "poisson":
                return float(scipy_stats.poisson.ppf(u, p["lam"]))
            case "beta":
                return float(scipy_stats.beta.ppf(u, p["alpha"], p["beta"]))
            case "exponential":
                return float(scipy_stats.expon.ppf(u, scale=p["scale"]))
            case "weibull":
                return float(scipy_stats.weibull_min.ppf(u, p["shape"], scale=p["scale"]))
            case _:
                # Fallback: direct sample (loses correlation accuracy)
                tmp_rng = np.random.default_rng()
                return float(ScenarioSampler.sample(spec, tmp_rng, 1)[0])

    # ------------------------------------------------------------------
    # Time-series scenario generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_random_walk(
        rng: np.random.Generator,
        start: float,
        drift: float,
        volatility: float,
        steps: int,
        log_scale: bool = False,
    ) -> np.ndarray:
        """
        Generate a geometric or arithmetic random walk.

        Args:
            start:      Starting value.
            drift:      Per-step drift (e.g., inflation rate).
            volatility: Per-step standard deviation of shocks.
            steps:      Number of time steps.
            log_scale:  If True, geometric (multiplicative) random walk.

        Returns:
            Array of shape (steps,).
        """
        shocks = rng.normal(drift, volatility, steps)
        if log_scale:
            # Geometric Brownian Motion
            log_path = np.cumsum(shocks)
            return start * np.exp(log_path)
        else:
            return start + np.cumsum(shocks)

    @staticmethod
    def generate_scenario_from_model(
        model: ScenarioModel,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """
        Draw one complete scenario from a ScenarioModel.

        Handles both independent and correlated variable groups.

        Returns:
            Dict of variable_name → sampled value.
        """
        results: Dict[str, Any] = {}

        # Group variables by correlation_group
        groups: Dict[Optional[str], Dict[str, DistributionSpec]] = {}
        for name, spec in model.variables.items():
            grp = spec.correlation_group
            groups.setdefault(grp, {})[name] = spec

        for grp, specs in groups.items():
            if grp is None or model.correlations is None or grp not in model.correlations:
                # Independent sampling
                for name, spec in specs.items():
                    results[name] = float(ScenarioSampler.sample(spec, rng, 1)[0])
            else:
                # Correlated sampling
                corr = np.array(model.correlations[grp])
                corr_results = ScenarioSampler.sample_correlated_group(specs, corr, rng)
                results.update(corr_results)

        return results
