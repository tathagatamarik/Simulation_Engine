"""Supply chain domain agent — delegates to SupplyChainSimulator."""
from __future__ import annotations

from typing import Any, Dict

from agents.domain_agent import DomainAgent
from domains.supply_chain.simulator import SupplyChainSimulator


class SupplyChainAgent(DomainAgent):
    """Domain agent for the Supply Chain & Logistics simulator."""

    name = "supply_chain_agent"
    simulator_cls = SupplyChainSimulator
