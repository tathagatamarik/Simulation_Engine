"""
Domain Registry — Plugin Discovery & Resolution.

The registry is the glue between the orchestrator and domain modules.
It maps domain identifiers → (simulator class, agent class).

New domains are registered by calling DomainRegistry.register().
The registry auto-loads all built-in domains on first access.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from agents.domain_agent import DomainAgent
from domains.base import BaseDomainSimulator


class DomainRegistry:
    """
    Central registry for domain simulation modules and their agents.

    Supports:
        - Manual registration: DomainRegistry.register(domain_key, SimCls, AgentCls)
        - Auto-registration:   All built-in domains registered on first access
        - Lookup:              DomainRegistry.get_agent(domain_key) → DomainAgent
        - Listing:             DomainRegistry.list_domains() → [{metadata}]
    """


    # Internal registry: domain_key → (SimulatorClass, AgentClass)
    _registry: Dict[str, Tuple[Type[BaseDomainSimulator], Type[DomainAgent]]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Initialize built-in domains if not already done."""
        if not cls._initialized:
            cls._initialized = True
            cls._auto_register()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        domain_key: str,
        simulator_cls: Type[BaseDomainSimulator],
        agent_cls: Optional[Type[DomainAgent]] = None,
    ) -> None:
        """
        Register a domain module.

        Args:
            domain_key:     Unique identifier, e.g. "supply_chain".
            simulator_cls:  Domain simulator class (subclass of BaseDomainSimulator).
            agent_cls:      Domain agent class. If None, creates a generic DomainAgent.
        """
        if agent_cls is None:
            # Create a dynamic agent class pointing to this simulator
            agent_cls = type(
                f"{simulator_cls.__name__}Agent",
                (DomainAgent,),
                {"name": f"{domain_key}_agent", "simulator_cls": simulator_cls},
            )
        cls._registry[domain_key] = (simulator_cls, agent_cls)

    @classmethod
    def _auto_register(cls) -> None:
        """Register all built-in domain modules."""
        # --- Supply Chain ---
        from domains.supply_chain.simulator import SupplyChainSimulator
        from domains.supply_chain.agent import SupplyChainAgent
        cls.register("supply_chain", SupplyChainSimulator, SupplyChainAgent)

        # --- Energy ROI ---
        from domains.energy_roi.simulator import EnergyROISimulator
        cls.register("energy_roi", EnergyROISimulator)

        # --- Freelance Finance ---
        from domains.freelance_finance.simulator import FreelanceFinanceSimulator
        cls.register("freelance_finance", FreelanceFinanceSimulator)

        # --- Machine Maintenance ---
        from domains.machine_maintenance.simulator import MachineMaintSimulator
        cls.register("machine_maintenance", MachineMaintSimulator)

        # --- Foot Traffic ABM ---
        from domains.foot_traffic.simulator import FootTrafficSimulator
        cls.register("foot_traffic", FootTrafficSimulator)

        # --- Marketing Strategy ---
        from domains.marketing_strategy.simulator import MarketingStrategySimulator
        cls.register("marketing_strategy", MarketingStrategySimulator)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get_agent(cls, domain_key: str) -> DomainAgent:
        """
        Instantiate and return the domain agent for a given domain.
        """
        cls._ensure_initialized()
        if domain_key not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Domain '{domain_key}' not found. "
                f"Available domains: {available}"
            )
        _, agent_cls = cls._registry[domain_key]
        return agent_cls()

    @classmethod
    def get_simulator(cls, domain_key: str) -> BaseDomainSimulator:
        """Return an instantiated simulator for a domain."""
        cls._ensure_initialized()
        if domain_key not in cls._registry:
            raise KeyError(f"Domain '{domain_key}' not registered.")
        sim_cls, _ = cls._registry[domain_key]
        return sim_cls()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @classmethod
    def list_domains(cls) -> List[Dict]:
        """Return metadata for all registered domains."""
        cls._ensure_initialized()
        result = []
        for domain_key, (sim_cls, _) in cls._registry.items():
            sim = sim_cls()
            meta = sim.metadata()
            meta["key"] = domain_key
            result.append(meta)
        return result

    @classmethod
    def is_registered(cls, domain_key: str) -> bool:
        cls._ensure_initialized()
        return domain_key in cls._registry
