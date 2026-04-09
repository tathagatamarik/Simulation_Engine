# Simulation Engine

A **modular, multi-domain Monte Carlo & Agent-Based Modeling (ABM) simulation engine** supporting 6 industry verticals, fully API-first and agent-orchestrated.

---

## 🎯 Supported Domains

| Domain | Key Outputs |
|--------|-------------|
| **Supply Chain & Logistics** | Stockout probability, delay distribution, fill rate |
| **Renewable Energy ROI** | ROI %, payback period, cumulative savings |
| **Freelance Financial Risk** | Runway months, insolvency probability, cash position |
| **Machine Maintenance** | Failure probability, optimal maintenance window, cost savings |
| **Foot Traffic ABM** | Congestion score, dwell time, flow efficiency |
| **Marketing Strategy** | CAC, LTV:CAC ratio, ROI distribution, campaign success |

---

## ⚡ Quick Start

### Local Development (no Docker)

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API server
uvicorn api.main:app --reload --port 8000

# 4. Start the Dashboard (separate terminal)
streamlit run ui/app.py
```

API docs: **http://localhost:8000/docs**
Dashboard: **http://localhost:8501**

### Docker Compose (full stack)

```bash
docker compose up --build
```

Services:
- API: http://localhost:8000
- Flower (Celery monitor): http://localhost:5555

---

## 📡 API Usage

### List all domains
```http
GET /domains
```

### Run a simulation (synchronous)
```http
POST /simulate/supply_chain
Content-Type: application/json

{
  "inputs": {
    "stock_level": 500,
    "reorder_point": 150,
    "lead_time_days": 21,
    "supplier_country": "china",
    "shipping_mode": "sea",
    "mean_daily_demand": 20,
    "demand_cv": 0.25,
    "simulation_days": 90,
    "n_iterations": 1000,
    "seed": 42
  }
}
```

### Run a large async simulation
```http
POST /simulate/marketing_strategy
Content-Type: application/json

{
  "inputs": { ... },
  "n_iterations": 5000,
  "async_mode": true
}
```

Response → `{ "run_id": "...", "status": "queued", "poll_url": "/results/{run_id}" }`

### Poll for async result
```http
GET /results/{run_id}
```

---

## 📊 Standardized Output Format

All domains produce this structure:

```json
{
  "run_id":       "uuid",
  "domain":       "supply_chain",
  "n_iterations": 1000,
  "summary": {
    "stockout_event": { "mean": 0.31, "p50": 0.0, "p90": 1.0, "p95": 1.0, ... }
  },
  "risk_metrics": {
    "stockout_event": { "var_95": 1.0, "cvar_95": 1.0, "failure_probability": 0.31 }
  },
  "time_series": [],
  "visualizations": { "charts": [...], "suggested_charts": ["histogram"] }
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

Expected: **~20 tests** across core engine and supply chain domain.

---

## 🏗️ Architecture

```
User Request
    ↓
FastAPI Gateway (/simulate/{domain})
    ↓
OrchestratorAgent
    ├── DomainAgent      (validate inputs)
    ├── ScenarioAgent    (build ScenarioModel)
    ├── RunnerAgent      (execute N iterations via MonteCarloEngine)
    ├── AnalysisAgent    (percentiles, VaR, CVaR)
    └── VisualizationAgent (chart-ready JSON)
    ↓
SimulationResult → Redis → API Response
```

---

## 🔌 Adding a New Domain

1. Create `domains/my_domain/` with `schema.py`, `simulator.py`, `__init__.py`
2. Implement `class MySimulator(BaseDomainSimulator)` with `simulate_once()` + `describe_outputs()`
3. Register in `registry/domain_registry.py`:
   ```python
   from domains.my_domain.simulator import MySimulator
   cls.register("my_domain", MySimulator)
   ```
4. New domain is immediately available at `POST /simulate/my_domain` ✅

---

## 🚀 SaaS Evolution Roadmap

| Stage | Feature |
|-------|---------|
| **No-Code Builder** | JSON-schema driven sim config, drag-and-drop UI |
| **SaaS MVP** | Multi-tenancy, quota system, Stripe billing |
| **Marketplace** | Versioned module registry, community contributions |

---

## License

MIT
