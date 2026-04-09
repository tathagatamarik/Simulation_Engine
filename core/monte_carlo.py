"""
Monte Carlo Engine.

Executes N iterations of any SimulationModule, managing:
  - Deterministic per-run seeding (from a master seed)
  - Parallel execution via ProcessPoolExecutor (CPU-bound work)
  - Sequential fallback for small N or debugging
  - Correlated variable sampling via Cholesky decomposition
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.interfaces import SimulationModule


# ---------------------------------------------------------------------------
# Top-level picklable worker function (required for multiprocessing)
# ---------------------------------------------------------------------------

def _worker(args: Tuple) -> Dict[str, Any]:
    """
    Multiprocessing worker. Must be a top-level function (picklable).

    Reconstructs the module instance per-process (modules are stateless).
    """
    module_cls, inputs, seed = args
    rng = np.random.default_rng(int(seed))
    module = module_cls()
    return module.simulate_once(inputs, rng)


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """
    Core Monte Carlo simulation engine.

    Design:
      - Master seed → child seeds via SeedSequence (BitGenerator inheritance)
        so each iteration has a unique, reproducible random stream.
      - ProcessPoolExecutor for parallelism on CPU-bound iterations.
      - Cholesky method for sampling correlated multi-variate normals.
    """

    PARALLEL_THRESHOLD = 200  # Use parallel above this iteration count

    def __init__(self, max_workers: Optional[int] = None):
        """
        Args:
            max_workers: Max CPU workers. None = os.cpu_count().
        """
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        module: SimulationModule,
        inputs: Dict[str, Any],
        n_iterations: int = 1000,
        seed: Optional[int] = None,
        use_parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run n_iterations of module.simulate_once() and return raw results.

        Args:
            module:       Instantiated domain simulator.
            inputs:       Validated inputs dict.
            n_iterations: Number of Monte Carlo draws.
            seed:         Master seed for full reproducibility. None = random.
            use_parallel: If True and n >= threshold, use multiprocessing.

        Returns:
            List of dicts, one per iteration, with output metrics.
        """
        # Derive per-iteration seeds from master seed
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_iterations)
        child_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]

        if use_parallel and n_iterations >= self.PARALLEL_THRESHOLD:
            return self._run_parallel(module, inputs, child_ints)
        return self._run_sequential(module, inputs, child_ints)

    # ------------------------------------------------------------------
    # Execution Backends
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        module: SimulationModule,
        inputs: Dict[str, Any],
        seeds: List[int],
    ) -> List[Dict[str, Any]]:
        results = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            results.append(module.simulate_once(inputs, rng))
        return results

    def _run_parallel(
        self,
        module: SimulationModule,
        inputs: Dict[str, Any],
        seeds: List[int],
    ) -> List[Dict[str, Any]]:
        module_cls = type(module)
        args_list = [(module_cls, inputs, s) for s in seeds]

        results: List[Dict[str, Any]] = [{}] * len(seeds)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all at once, preserve index for ordering
            future_to_idx = {
                executor.submit(_worker, args): idx
                for idx, args in enumerate(args_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    # ------------------------------------------------------------------
    # Correlated Sampling Utility
    # ------------------------------------------------------------------

    @staticmethod
    def sample_correlated(
        rng: np.random.Generator,
        means: np.ndarray,
        stds: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Sample correlated Gaussian variables using Cholesky decomposition.

        Algorithm:
            1. Build covariance from stds and corr_matrix.
            2. Cholesky-factorize: Σ = L Lᵀ.
            3. Draw z ~ N(0, I), return μ + Lz.

        Args:
            rng:         NumPy random generator.
            means:       Array of means, shape (n,).
            stds:        Array of standard deviations, shape (n,).
            corr_matrix: Correlation matrix, shape (n, n). Must be PSD.

        Returns:
            Array of correlated samples, shape (n,).
        """
        n = len(means)
        cov = np.outer(stds, stds) * corr_matrix
        # Regularize for numerical stability
        cov += np.eye(n) * 1e-10
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: independent draws if matrix not PSD
            return rng.normal(means, stds)
        z = rng.standard_normal(n)
        return means + L @ z
