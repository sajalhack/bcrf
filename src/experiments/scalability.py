"""
BCRF Experiment - Scalability Experiment
Tests BCRF performance as the number of execution states scales.
"""

from typing import List, Dict, Any, Optional
from src.experiments.main_experiment import run_single_trial


def run_scalability_trial(
    scenario: str,
    num_states: int,
    seed: int,
    fault_rate: float = 0.1,
    **kwargs,
) -> Dict[str, Any]:
    """Run a single scalability trial for a given state count."""
    return run_single_trial(
        scenario=scenario,
        num_states=num_states,
        seed=seed,
        fault_rate=fault_rate,
        **kwargs,
    )
