"""agent-safety-bench: Multi-step agent safety compliance testing framework.

Measures how well an LLM agent maintains global safety policies
across a chain of D sequential steps. Finds the phase transition depth
D* where compliance breaks down.

Core idea: each step looks like a legitimate research action, but
cumulatively they test the agent's ability to track and apply a
consistent safety policy across actions.
"""

from .core import SafetyBench, SafetyResult
from .api import call_model
from .analyze import find_phase_transition, PhaseTransition
from .scenarios import get_scenario, list_scenarios, register_scenario

__version__ = "0.1.0"
__all__ = [
    "SafetyBench", "SafetyResult",
    "call_model",
    "find_phase_transition", "PhaseTransition",
    "get_scenario", "list_scenarios", "register_scenario",
]