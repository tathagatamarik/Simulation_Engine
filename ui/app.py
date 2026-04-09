import streamlit as st
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from ui.api_client import SimulationAPIClient

# --- Page Config ---
st.set_page_config(
    page_title="Simulation Engine Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- Style ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- App Initialization ---
client = SimulationAPIClient(base_url="http://localhost:8000")

# --- Sidebar ---
st.sidebar.title("🛠️ Simulation Control")
st.sidebar.markdown("---")

# Domain Selection
domains = client.get_domains()
if not domains:
    st.sidebar.warning("⚠️ API is not reachable. Ensure uvicorn is running on port 8000.")
    selected_domain_meta = None
else:
    domain_names = [d["display_name"] for d in domains]
    selected_domain_name = st.sidebar.selectbox("Select Domain", domain_names)
    selected_domain_meta = next(d for d in domains if d["display_name"] == selected_domain_name)

st.sidebar.markdown("---")
global_n_iterations = st.sidebar.slider("Global Max Iterations", 100, 5000, 1000)
global_seed = st.sidebar.number_input("Random Seed (Optional)", value=0, min_value=0) or None

# --- Main View ---
st.title("🚀 Multi-Domain Simulation Engine")
st.caption("Monte Carlo & Agent-Based Modeling for Enterprise Decision Support")

if selected_domain_meta:
    domain_key = selected_domain_meta["key"]
    st.header(f"{selected_domain_meta['display_name']} domain")
    st.write(selected_domain_meta["description"])

    tab_config, tab_results, tab_opt = st.tabs(["⚙️ Configuration", "📈 Results", "🎯 Optimization"])

    with tab_config:
        st.subheader("Simulation Parameters")
        inputs = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            if domain_key == "supply_chain":
                inputs["stock_level"] = st.number_input("Stock Level (Units)", value=500)
                inputs["reorder_point"] = st.number_input("Reorder Point (Units)", value=150)
                inputs["lead_time_days"] = st.slider("Lead Time (Days)", 1, 90, 21)
                inputs["supplier_country"] = st.selectbox("Supplier Source", ["china", "india", "usa", "germany", "mexico", "vietnam", "bangladesh"])
                inputs["shipping_mode"] = st.selectbox("Shipping Mode", ["sea", "air", "land"])
            
            elif domain_key == "energy_roi":
                inputs["location"] = st.selectbox("Climate Zone", ["tropical", "temperate", "arid", "polar"])
                inputs["building_size_sqm"] = st.number_input("Building Size (sqm)", value=250)
                inputs["monthly_kwh_usage"] = st.number_input("Monthly Usage (kWh)", value=800)
                inputs["system_cost_usd"] = st.number_input("System Cost (USD)", value=15000)
            
            elif domain_key == "marketing_strategy":
                inputs["total_budget_usd"] = st.number_input("Total Budget (USD)", value=50000)
                inputs["ads_budget_pct"] = st.slider("Ads Budget %", 0.0, 1.0, 0.4)
                inputs["influencer_budget_pct"] = st.slider("Influencer Budget %", 0.0, 1.0, 0.3)
                inputs["seo_budget_pct"] = st.slider("SEO Budget %", 0.0, 1.0, 0.3)
                inputs["target_audience_size"] = st.number_input("Addressable Audience", value=100000)
                inputs["avg_ltv_usd"] = st.number_input("Customer LTV (USD)", value=250)
            
            elif domain_key == "freelance_finance":
                inputs["monthly_expenses"] = st.number_input("Monthly Expenses (USD)", value=3000)
                inputs["mean_monthly_income"] = st.number_input("Avg Monthly Income (USD)", value=6500)
                inputs["income_std"] = st.number_input("Income Volatility (USD)", value=2000)
                inputs["savings_buffer"] = st.number_input("Initial Savings (USD)", value=15000)
            
            elif domain_key == "machine_maintenance":
                inputs["machine_type"] = st.selectbox("Machine Type", ["motor", "pump", "compressor", "conveyor", "cnc"])
                inputs["machine_age_years"] = st.slider("Current Age (Years)", 0.0, 20.0, 5.0)
                inputs["daily_usage_hours"] = st.slider("Daily Usage (Hours)", 1.0, 24.0, 16.0)
                inputs["maintenance_cost_usd"] = st.number_input("Maintenance Cost (USD)", value=800)
                inputs["failure_repair_cost_usd"] = st.number_input("Failure Repair Cost (USD)", value=5000)
            
            elif domain_key == "foot_traffic":
                inputs["grid_width"] = st.slider("Grid Width", 10, 100, 20)
                inputs["grid_height"] = st.slider("Grid Height", 10, 100, 20)
                inputs["num_customers"] = st.slider("Number of Agents", 10, 500, 50)
                inputs["simulation_steps"] = st.slider("Simulation Steps", 50, 1000, 100)
                inputs["obstacle_density"] = st.slider("Obstacle Density", 0.0, 0.5, 0.1)

        with col2:
            st.info("💡 **Domain Insight**\n" + selected_domain_meta.get("description", "No detailed description available."))
            n_iterations = st.number_input("Iterations for this run", value=global_n_iterations, min_value=100, max_value=10000)
            # Add placeholders for other inputs if needed
            if domain_key == "supply_chain":
                inputs["mean_daily_demand"] = st.number_input("Avg Daily Demand", value=20)
                inputs["demand_cv"] = st.slider("Demand Volatility (CV)", 0.0, 1.0, 0.2)
                inputs["simulation_days"] = st.slider("Horizon (Days)", 30, 730, 90)
            elif domain_key == "energy_roi":
                inputs["electricity_tariff_usd_per_kwh"] = st.number_input("Electricity Tariff ($/kWh)", value=0.18)
                inputs["feed_in_tariff_usd_per_kwh"] = st.number_input("Feed-in Tariff ($/kWh)", value=0.08)
                inputs["simulation_years"] = st.slider("Horizon (Years)", 5, 50, 25)
            elif domain_key == "marketing_strategy":
                inputs["ads_conversion_rate"] = st.slider("Ads Conv. Rate", 0.0, 0.1, 0.025, format="%.3f")
                inputs["virality_factor"] = st.slider("Virality Multiplier", 1.0, 3.0, 1.3)
                inputs["simulation_months"] = st.slider("Horizon (Months)", 1, 60, 12)
            elif domain_key == "freelance_finance":
                inputs["tax_rate"] = st.slider("Tax Rate", 0.0, 0.6, 0.25)
                inputs["monthly_churn_prob"] = st.slider("Churn Prob/Mo", 0.0, 0.5, 0.05)
                inputs["simulation_months"] = st.slider("Horizon (Months)", 3, 60, 24)
            elif domain_key == "machine_maintenance":
                inputs["last_maintenance_days_ago"] = st.number_input("Days Since Last Maint.", value=30)
                inputs["simulation_days"] = st.slider("Horizon (Days)", 30, 1825, 365)

        if st.button("🚀 Run Simulation", use_container_width=True):
            with st.spinner(f"Simulating {n_iterations} scenarios..."):
                result = client.run_simulation(domain_key, inputs, n_iterations)
                if result:
                    st.session_state["last_result"] = result
                    st.success("Simulation Complete! View results in the Results tab.")
                    # Automatically switch to results tab if possible (not easily in Streamlit without state dance)
                else:
                    st.error("Simulation failed to start.")

    with tab_results:
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            summary = result.get("summary", {})
            metrics = result.get("risk_metrics", {})
            
            st.subheader("Key Performance Indicators")
            cols = st.columns(len(summary))
            for i, (m_key, m_val) in enumerate(summary.items()):
                with cols[i]:
                    disp_name = m_key.replace("_", " ").title()
                    st.metric(disp_name, f"{m_val['mean']:.2f}")
                    st.caption(f"P95: {m_val['p95']:.2f}")

            st.markdown("---")
            
            col_chart1, col_chart2 = st.columns(2)
            
            # Risk Metrics Table
            with col_chart1:
                st.subheader("⚠️ Risk Analysis")
                if metrics:
                    risk_df = pd.DataFrame.from_dict(metrics, orient='index')
                    st.dataframe(risk_df.style.highlight_max(axis=0, subset=['failure_probability'], color='lightpink'))
                else:
                    st.write("No specific risk metrics available.")

            # Distribution Chart
            with col_chart2:
                st.subheader("📊 Distribution of Outcomes")
                primary_metric = list(summary.keys())[0] if summary else None
                if primary_metric:
                    # Mocking distribution if actual samples aren't returned (API usually returns summary only)
                    # If the API returns raw samples, we'd use them. Let's assume summary only for now.
                    st.info(f"Summary for {primary_metric.replace('_', ' ')}")
                    # For a "wow" UI, we'd want samples. If samples aren't there, we'll show a summary bar chart.
                    m_data = summary[primary_metric]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=["Mean", "P50", "P90", "P95"],
                        y=[m_data["mean"], m_data["p50"], m_data["p90"], m_data["p95"]],
                        marker_color='rgb(55, 83, 109)'
                    ))
                    fig.update_layout(title=f"{primary_metric.replace('_', ' ').title()} Percentiles")
                    st.plotly_chart(fig, use_container_width=True)

            # Time Series (if available)
            time_series = result.get("time_series", [])
            if time_series:
                st.subheader("⏳ Time Series Projection (Mean Path)")
                ts_df = pd.DataFrame(time_series)
                fig_ts = px.line(ts_df, x=ts_df.columns[0], y=ts_df.columns[1:], title="Simulation Path Over Time")
                st.plotly_chart(fig_ts, use_container_width=True)

        else:
            st.info("Run a simulation in the Configuration tab to see results here.")

    with tab_opt:
        st.subheader("🎯 Intelligent Decision Support")
        st.write("Find the optimal configuration to maximize or minimize a specific outcome.")
        
        goal_col1, goal_col2 = st.columns(2)
        with goal_col1:
            goal_metric = st.selectbox("Goal Metric", selected_domain_meta["outputs"])
            goal_direction = st.radio("Direction", ["Maximize", "Minimize"], horizontal=True)
        
        with goal_col2:
            n_trials = st.slider("Optimization Trials", 5, 50, 10)
            st.info("The optimizer will run multiple simulation sets across a search space to find the best performing parameters.")

        if st.button("🔍 Find Optimal Setup", type="primary"):
            with st.spinner("Agent exploring parameter space..."):
                # Define search space based on domain
                search_space = {}
                if domain_key == "supply_chain":
                    search_space = {"reorder_point": [100, 400], "stock_level": [200, 800]}
                elif domain_key == "marketing_strategy":
                    search_space = {"ads_budget_pct": [0.1, 0.6], "influencer_budget_pct": [0.1, 0.6]}
                elif domain_key == "energy_roi":
                    search_space = {"building_size_sqm": [100, 500]}
                elif domain_key == "freelance_finance":
                    search_space = {"savings_buffer": [5000, 30000]}
                elif domain_key == "machine_maintenance":
                    search_space = {"maintenance_cost_usd": [500, 2000]}

                opt_payload = {
                    "domain": domain_key,
                    "goal_metric": goal_metric,
                    "goal_direction": goal_direction.lower()[:3],
                    "fixed_inputs": inputs,
                    "search_space": search_space,
                    "n_iterations": 200,
                    "n_trials": n_trials
                }
                
                opt_result = client.run_optimization(opt_payload)
                if opt_result:
                    st.success(f"Best Configuration Found! Optimal {goal_metric}: {opt_result['best_value']:.2f}")
                    st.json(opt_result["best_inputs"])
                    
                    if st.button("Apply these settings"):
                        st.session_state["last_result"] = opt_result["optimization_result"]
                        st.rerun()
                else:
                    st.error("Optimization failed.")

else:
    st.info("👈 Select a domain in the sidebar to get started.")

# Footer
st.markdown("---")
st.caption("Mindgraph Simulation Engine v0.1.0 | Built with Streamlit & Plotly")
