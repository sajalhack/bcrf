"""
BCRF Experiment - Ablation Study
Tests BCRF with individual components removed.

Ablation conditions:
  A: Without FCI
  B: Without BCR
  C: Without External Operation Registry
  D: Without CIV
  E: Without adaptive CRF expansion
  Full: All components enabled
"""

from typing import List, Dict, Any, Optional
from src.experiments.main_experiment import run_single_trial


ABLATION_CONDITIONS = {
    "full_bcrf": {
        "use_fci": True,
        "use_bcr": True,
        "use_registry": True,
        "use_civ": True,
        "use_crf_expansion": True,
        "label": "Full BCRF",
    },
    "no_fci": {
        "use_fci": False,
        "use_bcr": True,
        "use_registry": True,
        "use_civ": True,
        "use_crf_expansion": True,
        "label": "Without FCI",
    },
    "no_bcr": {
        "use_fci": True,
        "use_bcr": False,
        "use_registry": True,
        "use_civ": True,
        "use_crf_expansion": True,
        "label": "Without BCR",
    },
    "no_registry": {
        "use_fci": True,
        "use_bcr": True,
        "use_registry": False,
        "use_civ": True,
        "use_crf_expansion": True,
        "label": "Without Registry",
    },
    "no_civ": {
        "use_fci": True,
        "use_bcr": True,
        "use_registry": True,
        "use_civ": False,
        "use_crf_expansion": True,
        "label": "Without CIV",
    },
    "no_expansion": {
        "use_fci": True,
        "use_bcr": True,
        "use_registry": True,
        "use_civ": True,
        "use_crf_expansion": False,
        "label": "Without CRF Expansion",
    },
}


def run_ablation_trial(
    scenario: str,
    num_states: int,
    seed: int,
    fault_rate: float = 0.1,
    fault_type: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run all ablation conditions on the same workload.
    Returns dict: {condition_name: bcrf_result}
    """
    ablation_results = {}

    for condition_name, flags in ABLATION_CONDITIONS.items():
        trial_result = run_single_trial(
            scenario=scenario,
            num_states=num_states,
            seed=seed,
            fault_rate=fault_rate,
            fault_type=fault_type,
            use_fci=flags["use_fci"],
            use_bcr=flags["use_bcr"],
            use_registry=flags["use_registry"],
            use_civ=flags["use_civ"],
            use_crf_expansion=flags["use_crf_expansion"],
            **kwargs,
        )
        ablation_results[condition_name] = {
            "label": flags["label"],
            "bcrf_result": trial_result.get("bcrf", {}),
            "num_states": num_states,
            "scenario": scenario,
            "seed": seed,
            "condition": condition_name,
        }

    return ablation_results
