from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.optimization_agent import OptimizationAgent

router = APIRouter()
optimizer = OptimizationAgent()

class OptimizationRequest(BaseModel):
    domain: str
    goal_metric: str
    goal_direction: str = "max"
    fixed_inputs: Dict[str, Any] = {}
    search_space: Dict[str, List[float]] = {}  # {param: [min, max]}
    n_iterations: int = 500
    n_trials: int = 10

@router.post("/")
async def run_optimization(request: OptimizationRequest):
    """
    Run an optimization search to find best parameters.
    """
    try:
        result = await optimizer.run(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
