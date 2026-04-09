"""
Visualization Agent.

Transforms analysis output into chart-ready JSON payloads.
Suggests appropriate chart types per output variable and
produces standard chart data structures consumable by any
frontend (Chart.js, D3, Plotly, etc.).
"""
from __future__ import annotations

from typing import Any, Dict, List

from core.interfaces import Agent


# Chart type suggestions per metric name pattern
_CHART_HINTS: Dict[str, str] = {
    "stockout":      "histogram",
    "delay":         "histogram",
    "distribution":  "histogram",
    "roi":           "histogram",
    "runway":        "histogram",
    "probability":   "bar",
    "heatmap":       "heatmap",
    "time_series":   "line",
    "curve":         "line",
    "cac":           "line",
    "ltv":           "line",
    "payback":       "line",
}


class VisualizationAgent(Agent):
    """
    Produces chart-ready JSON for frontend consumption.

    Input context keys:
        - analysis: {summary, risk_metrics, time_series, histograms}
        - domain:   str

    Output context key:
        - visualizations: {charts: [...], suggested_charts: [...]}
    """

    name = "visualization_agent"

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        analysis = context.get("analysis", {})
        domain   = context.get("domain", "unknown")

        charts           = []
        suggested_charts: List[str] = []

        # ---- Histogram charts ----
        histograms = analysis.get("histograms", {})
        for metric, hist_data in histograms.items():
            chart_type = self._suggest_chart_type(metric)
            charts.append({
                "type":   "histogram",
                "metric": metric,
                "data":   hist_data,
                "title":  self._humanize(metric),
            })
            if chart_type not in suggested_charts:
                suggested_charts.append(chart_type)

        # ---- Time-series line charts ----
        time_series = analysis.get("time_series", [])
        if time_series:
            charts.append({
                "type":   "line",
                "metric": "time_series",
                "data":   {
                    "x":    [e["t"] for e in time_series],
                    "series": self._extract_ts_series(time_series),
                },
                "title": f"{domain.replace('_', ' ').title()} — Time Series",
            })
            if "line" not in suggested_charts:
                suggested_charts.append("line")

        # ---- Summary percentile bar chart ----
        summary = analysis.get("summary", {})
        if summary:
            primary_keys = list(summary.keys())[:3]   # Top 3 metrics
            charts.append({
                "type":   "percentile_bars",
                "metric": "summary_overview",
                "data": {
                    key: {
                        "p50": summary[key]["p50"],
                        "p90": summary[key]["p90"],
                        "p95": summary[key]["p95"],
                        "mean": summary[key]["mean"],
                    }
                    for key in primary_keys if key in summary
                },
                "title": "Key Metrics — Percentile Overview",
            })

        visualizations = {
            "charts":          charts,
            "suggested_charts": list(set(suggested_charts)) or ["histogram"],
            "domain":          domain,
        }

        return {**context, "visualizations": visualizations}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_chart_type(metric: str) -> str:
        metric_lower = metric.lower()
        for pattern, chart_type in _CHART_HINTS.items():
            if pattern in metric_lower:
                return chart_type
        return "histogram"

    @staticmethod
    def _humanize(name: str) -> str:
        """Convert snake_case to Title Case."""
        return name.replace("_", " ").title()

    @staticmethod
    def _extract_ts_series(time_series: List[Dict[str, Any]]) -> Dict[str, List]:
        """Extract series arrays from time-series list of dicts."""
        if not time_series:
            return {}
        keys = [k for k in time_series[0] if k != "t"]
        return {key: [e.get(key) for e in time_series] for key in keys}
