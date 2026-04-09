"""
Analysis Agent.

Aggregates raw Monte Carlo results into:
  - Statistical summaries (mean, std, percentiles)
  - Risk metrics (VaR, CVaR, failure probabilities)
  - Time-series aggregation (if domain supports it)
  - Histogram data for all numeric outputs
"""
from __future__ import annotations

from typing import Any, Dict

from core.interfaces import Agent
from core.analysis import AnalysisEngine


class AnalysisAgent(Agent):
    """
    Aggregates raw Monte Carlo results into structured analysis.

    Expects context keys:
        - raw_results:  List[Dict[str, Any]]
        - simulator:    BaseDomainSimulator (for failure thresholds + ts config)
        - n_iterations: int

    Produces context key:
        - analysis: {summary, risk_metrics, time_series, histograms}
    """

    name = "analysis_agent"

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate raw results and enrich context with analysis."""
        raw_results  = context["raw_results"]
        simulator    = context["simulator"]
        n_iterations = context.get("n_iterations", len(raw_results))

        # Pull failure thresholds from domain module
        failure_thresholds = simulator.get_failure_thresholds()

        # Core aggregation: summary + risk metrics
        agg = AnalysisEngine.aggregate(raw_results, failure_thresholds)

        # Histograms for all numeric outputs
        histograms = AnalysisEngine.build_all_histograms(raw_results)

        # Time-series (if domain configures it)
        time_series = []
        ts_config = simulator.get_time_series_config()
        if ts_config:
            time_series = AnalysisEngine.build_time_series(
                raw_results,
                time_key=ts_config["time_key"],
                value_keys=ts_config["value_keys"],
            )

        analysis = {
            "summary":      agg["summary"],
            "risk_metrics": agg["risk_metrics"],
            "time_series":  time_series,
            "histograms":   histograms,
            "n_iterations": n_iterations,
        }

        return {**context, "analysis": analysis}
