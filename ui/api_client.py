import httpx
import streamlit as st
from typing import Dict, List, Any, Optional

class SimulationAPIClient:
    """Helper client to interact with the Simulation Engine API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def get_domains(self) -> List[Dict[str, Any]]:
        """Fetch registered simulation domains."""
        try:
            response = httpx.get(f"{self.base_url}/domains", timeout=5.0)
            response.raise_for_status()
            return response.json().get("domains", [])
        except Exception as e:
            st.error(f"Failed to fetch domains: {e}")
            return []

    def run_simulation(self, domain: str, inputs: Dict[str, Any], n_iterations: int = 1000) -> Optional[Dict[str, Any]]:
        """Run a simulation and return results."""
        payload = {
            "inputs": inputs,
            "n_iterations": n_iterations,
            "async_mode": False
        }
        try:
            response = httpx.post(f"{self.base_url}/simulate/{domain}", json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data.get("result")
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return None

    def get_poll_result(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Poll for an async result."""
        try:
            response = httpx.get(f"{self.base_url}/results/{run_id}", timeout=5.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Polling failed: {e}")
            return None

    def run_optimization(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run an optimization task."""
        try:
            response = httpx.post(f"{self.base_url}/optimize/", json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            return None
