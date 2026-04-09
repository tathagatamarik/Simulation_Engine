"""
Analysis Engine.

Aggregates raw Monte Carlo results into:
  - Statistical summaries (mean, std, percentiles)
  - Risk metrics (VaR, CVaR, failure probability)
  - Time-series aggregation
  - Histogram data for visualization
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class AnalysisEngine:
    """
    Stateless analysis utilities for Monte Carlo result aggregation.

    All methods are static, making AnalysisEngine usable without instantiation,
    or as a dependency injected into the AnalysisAgent.
    """

    # ------------------------------------------------------------------
    # Core Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(
        raw_results: List[Dict[str, Any]],
        failure_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate per-iteration result dicts into summary statistics.

        Args:
            raw_results:        List of dicts from simulate_once().
            failure_thresholds: {metric_key: threshold} — compute P(X >= threshold).

        Returns:
            {
              "summary":      {metric: {mean, std, min, p25, p50, p75, p90, p95, p99, max}},
              "risk_metrics": {metric: {var_95, cvar_95, failure_probability?, ...}},
            }
        """
        if not raw_results:
            return {"summary": {}, "risk_metrics": {}}

        # Collect numeric keys from first result
        sample = raw_results[0]
        keys = [k for k, v in sample.items() if isinstance(v, (int, float, bool))]

        # Build arrays
        arrays: Dict[str, np.ndarray] = {
            key: np.array([r.get(key, np.nan) for r in raw_results], dtype=float)
            for key in keys
        }

        summary: Dict[str, Any] = {}
        risk_metrics: Dict[str, Any] = {}

        for key, arr in arrays.items():
            # Remove NaN values
            clean = arr[~np.isnan(arr)]
            if len(clean) == 0:
                continue

            # Descriptive statistics
            summary[key] = {
                "mean":  float(np.mean(clean)),
                "std":   float(np.std(clean)),
                "min":   float(np.min(clean)),
                "p10":   float(np.percentile(clean, 10)),
                "p25":   float(np.percentile(clean, 25)),
                "p50":   float(np.percentile(clean, 50)),
                "p75":   float(np.percentile(clean, 75)),
                "p90":   float(np.percentile(clean, 90)),
                "p95":   float(np.percentile(clean, 95)),
                "p99":   float(np.percentile(clean, 99)),
                "max":   float(np.max(clean)),
                "count": int(len(clean)),
            }

            # Risk metrics
            var_95 = float(np.percentile(clean, 95))
            beyond = clean[clean >= var_95]
            cvar_95 = float(np.mean(beyond)) if len(beyond) > 0 else var_95

            rm: Dict[str, Any] = {
                "var_95":  var_95,
                "cvar_95": cvar_95,
            }

            if failure_thresholds and key in failure_thresholds:
                thr = failure_thresholds[key]
                rm["failure_threshold"]   = thr
                rm["failure_probability"] = float(np.mean(clean >= thr))

            risk_metrics[key] = rm

        return {"summary": summary, "risk_metrics": risk_metrics}

    # ------------------------------------------------------------------
    # Time-Series Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def build_time_series(
        raw_results: List[Dict[str, Any]],
        time_key: str,
        value_keys: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Build time-aggregated series from results containing a time dimension.

        Each result dict must include `time_key` identifying its time step.
        Returns one entry per unique time step with mean/p50/p90 aggregates.
        """
        by_t: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: {k: [] for k in value_keys}
        )

        for result in raw_results:
            t = int(result.get(time_key, 0))
            for key in value_keys:
                val = result.get(key)
                if val is not None and not np.isnan(val):
                    by_t[t][key].append(float(val))

        time_series: List[Dict[str, Any]] = []
        for t in sorted(by_t.keys()):
            entry: Dict[str, Any] = {"t": t}
            for key in value_keys:
                vals = by_t[t][key]
                if vals:
                    arr = np.array(vals)
                    entry[f"{key}_mean"] = float(np.mean(arr))
                    entry[f"{key}_p50"]  = float(np.percentile(arr, 50))
                    entry[f"{key}_p90"]  = float(np.percentile(arr, 90))
            time_series.append(entry)

        return time_series

    # ------------------------------------------------------------------
    # Visualization Data Builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_histogram(
        arr: np.ndarray,
        bins: int = 40,
    ) -> Dict[str, Any]:
        """Build histogram data (bin centers + counts) for chart rendering."""
        clean = arr[~np.isnan(arr)]
        if len(clean) == 0:
            return {"bin_centers": [], "bin_edges": [], "counts": []}
        counts, edges = np.histogram(clean, bins=bins)
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        return {
            "bin_centers": centers,
            "bin_edges":   edges.tolist(),
            "counts":      counts.tolist(),
        }

    @staticmethod
    def build_all_histograms(
        raw_results: List[Dict[str, Any]],
        bins: int = 40,
    ) -> Dict[str, Any]:
        """Build histograms for all numeric output metrics."""
        if not raw_results:
            return {}

        keys = [k for k, v in raw_results[0].items() if isinstance(v, (int, float, bool))]
        histograms: Dict[str, Any] = {}

        for key in keys:
            arr = np.array([r.get(key, np.nan) for r in raw_results], dtype=float)
            histograms[key] = AnalysisEngine.build_histogram(arr, bins)

        return histograms

    @staticmethod
    def build_percentile_envelope(
        raw_results: List[Dict[str, Any]],
        key: str,
        time_key: str = "t",
        percentiles: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Build a percentile envelope for a metric over time.
        Useful for 'fan chart' visualizations.
        """
        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90]

        ts = AnalysisEngine.build_time_series(raw_results, time_key, [key])
        t_vals = [e["t"] for e in ts]
        envelopes: Dict[str, List[float]] = {f"p{p}": [] for p in percentiles}

        by_t: Dict[int, List[float]] = defaultdict(list)
        for result in raw_results:
            t = int(result.get(time_key, 0))
            val = result.get(key)
            if val is not None:
                by_t[t].append(float(val))

        for t in sorted(by_t.keys()):
            arr = np.array(by_t[t])
            for p in percentiles:
                envelopes[f"p{p}"].append(float(np.percentile(arr, p)))

        return {"t": t_vals, "percentiles": percentiles, "envelopes": envelopes}
