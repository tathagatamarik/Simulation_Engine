"""
Foot Traffic Agent-Based Model (ABM) Simulator.

Implements a lightweight grid-based ABM without Mesa dependency,
making it easily testable and portable. A Mesa-based version
can replace this by subclassing BaseDomainSimulator with the same interface.

Model:
  - Grid: W×H cells, with randomly placed obstacles
  - CustomerAgent: random-walk until reaching a destination zone
  - Metrics: congestion heatmap, mean dwell time, flow efficiency

Each simulate_once() runs `simulation_steps` steps of the ABM.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from domains.base import BaseDomainSimulator
from domains.foot_traffic.schema import FootTrafficInput


class FootTrafficSimulator(BaseDomainSimulator):
    """
    Foot Traffic ABM Simulator.

    Lightweight custom grid ABM — no external Mesa dependency required.
    Customers enter through entry points, perform bounded random walks,
    and exit through the same entry points when 'done'.

    Outputs:
      - mean_dwell_time: average steps a customer spends on the floor
      - max_concurrent_customers: peak crowd count
      - congestion_score: normalized occupancy variance (proxy for heatmap entropy)
      - flow_efficiency: fraction of customers who exited vs total spawned
    """

    domain       = "foot_traffic"
    display_name = "Foot Traffic & Spatial Flow ABM Simulator"
    description  = (
        "Agent-based model of customer movement on a retail floor plan. "
        "Computes congestion scores, dwell time, and flow efficiency."
    )
    schema = FootTrafficInput

    def simulate_once(
        self,
        inputs: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:

        W       = int(inputs["grid_width"])
        H       = int(inputs["grid_height"])
        n_entry = int(inputs["num_entry_points"])
        n_cust  = int(inputs["num_customers"])
        steps   = int(inputs["simulation_steps"])
        obs_den = float(inputs["obstacle_density"])

        # --- Build obstacle grid ---
        obstacles = rng.uniform(0, 1, (H, W)) < obs_den

        # Entry points: spread along bottom edge
        entry_xs = np.linspace(0, W - 1, n_entry, dtype=int)
        entry_points: List[Tuple[int, int]] = [(int(x), 0) for x in entry_xs]
        for ep in entry_points:
            obstacles[ep[1], ep[0]] = False  # Clear entry points

        # --- Initialize agents ---
        # Each agent: [x, y, steps_on_floor, exited, destination_x, destination_y]
        agents: List[Dict[str, Any]] = []
        for i in range(n_cust):
            ep  = entry_points[i % len(entry_points)]
            dst = (int(rng.integers(0, W)), int(rng.integers(H // 2, H)))
            agents.append({
                "x":       ep[0], "y": ep[1],
                "dwell":   0,
                "exited":  False,
                "dest_x":  dst[0], "dest_y": dst[1],
            })

        # --- Heatmap accumulator ---
        heatmap = np.zeros((H, W), dtype=int)

        # --- Directions: N, S, E, W + diagonals ---
        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        max_concurrent = 0
        exit_count     = 0

        for step in range(steps):
            active_count = sum(1 for a in agents if not a["exited"])
            max_concurrent = max(max_concurrent, active_count)

            for agent in agents:
                if agent["exited"]:
                    continue

                agent["dwell"] += 1
                x, y = agent["x"], agent["y"]
                heatmap[y, x] += 1

                # Check if reached destination → move toward exit
                at_dest = (abs(x - agent["dest_x"]) <= 1 and abs(y - agent["dest_y"]) <= 1)
                if at_dest and agent["dwell"] > 5:
                    # Head back to entry
                    agent["dest_x"] = entry_points[0][0]
                    agent["dest_y"] = entry_points[0][1]

                # Exit condition: reached entry point after visiting destination
                if at_dest and y == 0:
                    agent["exited"] = True
                    exit_count += 1
                    continue

                # --- Biased random walk toward destination ---
                dx = np.sign(agent["dest_x"] - x)
                dy = np.sign(agent["dest_y"] - y)

                # Candidate moves: bias toward destination + random noise
                candidates = [(dx, dy)] + MOVES
                rng.shuffle(candidates)    # type: ignore[arg-type]

                moved = False
                for ddx, ddy in candidates:
                    nx, ny = x + ddx, y + ddy
                    if 0 <= nx < W and 0 <= ny < H and not obstacles[ny, nx]:
                        agent["x"] = nx
                        agent["y"] = ny
                        moved = True
                        break

        # --- Metrics ---
        dwell_times  = [a["dwell"] for a in agents]
        mean_dwell   = float(np.mean(dwell_times)) if dwell_times else 0.0
        flow_eff     = exit_count / max(1, n_cust)

        # Congestion score: normalized variance of heatmap cell counts
        occupied     = heatmap[heatmap > 0].flatten()
        cong_score   = float(np.std(occupied) / (np.mean(occupied) + 1e-9)) if len(occupied) > 0 else 0.0

        return {
            "mean_dwell_time":          mean_dwell,
            "max_concurrent_customers": float(max_concurrent),
            "flow_efficiency":          flow_eff,
            "congestion_score":         cong_score,
            "exit_count":               float(exit_count),
            "heatmap_peak_cell":        float(np.max(heatmap)),
        }

    def describe_outputs(self) -> List[str]:
        return [
            "mean_dwell_time", "max_concurrent_customers",
            "flow_efficiency", "congestion_score",
            "exit_count", "heatmap_peak_cell",
        ]

    def get_failure_thresholds(self) -> Dict[str, float]:
        return {
            "flow_efficiency":  0.8,   # Risk of < 80% flow efficiency
            "congestion_score": 2.0,   # High congestion risk
        }
