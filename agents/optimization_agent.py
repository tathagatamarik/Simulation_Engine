"""
Optimization Agent.

Provides a layer for automated decision support by running multiple
simulation scenarios to find the optimal input parameters for a goal.

Example goals:
- Maximize ROI (Energy ROI, Marketing Strategy)
- Minimize Stockout Probability (Supply Chain)
- Maximize Runway (Freelance Finance)
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from core.interfaces import Agent
from agents.orchestrator import OrchestratorAgent


class OptimizationAgent(Agent):
    """
    Search-based optimization agent.

    Uses a simple grid search or random search across a parameter space
    to find the inputs that optimize a specific output metric.

    Input context:
        - domain:         str
        - goal_metric:    str  (e.g., "roi_pct")
        - goal_direction: str  ("max" or "min")
        - fixed_inputs:   dict (parameters that stay constant)
        - search_space:   dict {param: [min, max, steps]}
        - n_iterations:   int  (per-sim iterations)
    """

    name = "optimization_agent"

    def __init__(self, max_workers: Optional[int] = None):
        self._orchestrator = OrchestratorAgent(max_workers=max_workers)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        domain         = context["domain"]
        goal_metric    = context["goal_metric"]
        direction      = context.get("goal_direction", "max")
        fixed_inputs   = context.get("fixed_inputs", {})
        search_space   = context.get("search_space", {})
        n_iter         = context.get("n_iterations", 500)
        n_trials       = context.get("n_trials", 10)  # Total points in space to check

        results = []

        # Generate search points
        search_points = self._generate_search_points(search_space, n_trials)

        # Run trials concurrently
        tasks = []
        for point in search_points:
            trial_inputs = {**fixed_inputs, **point}
            tasks.append(self._orchestrator.run({
                "domain": domain,
                "inputs": trial_inputs,
                "n_iterations": n_iter,
            }))

        trial_results = await asyncio.gather(*tasks)

        # Find best point
        best_val = -float('inf') if direction == "max" else float('inf')
        best_point = None
        best_full_result = None

        for i, res in enumerate(trial_results):
            # Extract the mean value of the goal metric from the summary
            val = res["summary"].get(goal_metric, {}).get("mean")
            if val is None:
                continue

            is_better = (val > best_val) if direction == "max" else (val < best_val)
            if is_better:
                best_val = val
                best_point = search_points[i]
                best_full_result = res

        return {
            "best_inputs": best_point,
            "best_value":  best_val,
            "goal_metric": goal_metric,
            "trials":      len(trial_results),
            "optimization_result": best_full_result
        }

    def _generate_search_points(self, space: Dict[str, List], n: int) -> List[Dict[str, Any]]:
        """Generate N random points within the search space."""
        points = []
        for _ in range(n):
            point = {}
            for param, bounds in space.items():
                if len(bounds) == 2:  # [min, max]
                    point[param] = np.random.uniform(bounds[0], bounds[1])
                    # If the parameter is likely an int in the schema, we could round it here
                    # But the domain agents handle validation anyway.
                elif len(bounds) == 3: # [min, max, step]
                    # Simple discrete choices
                    choices = np.arange(bounds[0], bounds[1] + bounds[2], bounds[2])
                    point[param] = float(np.random.choice(choices))
            points.append(point)
        return points
