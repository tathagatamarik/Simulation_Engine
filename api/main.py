"""
FastAPI Application — Simulation Engine API.

Endpoints:
    GET  /                     — Health check + welcome
    GET  /health               — Service health
    GET  /domains              — List registered simulation domains
    POST /simulate/{domain}    — Run a simulation (sync or async)
    GET  /results/{run_id}     — Poll for async simulation result

Authentication: Placeholder (add API key middleware for SaaS deployment).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from api.routers import simulate, results, optimize

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("simulation_engine_starting", version="0.1.0")
    # Pre-warm registry (trigger auto-registration)
    from registry.domain_registry import DomainRegistry
    domains = DomainRegistry.list_domains()
    logger.info("domains_registered", count=len(domains), domains=[d["key"] for d in domains])
    yield
    logger.info("simulation_engine_shutdown")


# ---------------------------------------------------------------------------
# App Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Simulation Engine API",
    description = (
        "Modular multi-domain Monte Carlo & ABM simulation engine. "
        "Supports Supply Chain, Energy ROI, Freelance Finance, "
        "Machine Maintenance, Foot Traffic, and Marketing Strategy simulations."
    ),
    version     = "0.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Tighten for production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(simulate.router, prefix="/simulate", tags=["Simulation"])
app.include_router(results.router,  prefix="/results",  tags=["Results"])
app.include_router(optimize.router, prefix="/optimize", tags=["Optimization"])


# ---------------------------------------------------------------------------
# Root Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Simulation Engine",
        "version": "0.1.0",
        "docs":    "/docs",
        "status":  "running",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@app.get("/domains", tags=["Registry"])
async def list_domains():
    """List all registered simulation domain modules."""
    from registry.domain_registry import DomainRegistry
    return {"domains": DomainRegistry.list_domains()}
