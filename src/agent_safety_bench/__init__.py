"""agent-safety-bench: Multi-step agent safety compliance testing framework.

Measures how well an LLM agent maintains global safety policies
across a chain of D sequential steps. Finds the phase transition depth
D* where compliance breaks down.
"""

from .core import SafetyBench, SafetyResult
from .api import call_model, call_ollama
from .analyze import find_phase_transition, PhaseTransition, aggregate_results, print_summary, save_results
from .scenarios import get_scenario, list_scenarios, register_scenario

__version__ = "0.1.0"
__all__ = [
    "SafetyBench", "SafetyResult",
    "call_model", "call_ollama",
    "find_phase_transition", "PhaseTransition", "aggregate_results",
    "print_summary", "save_results",
    "get_scenario", "list_scenarios", "register_scenario",
]