"""
BCRF Experiment - Main Experiment Runner
Orchestrates a single trial: runs all 4 methods on the same workload.
"""

import time
import copy
import random
from typing import Dict, List, Optional, Any

from src.agent.agent_simulator import AgentSimulator
from src.agent.state import AgentState
from src.integrity.integrity_monitor import IntegrityMonitor
from src.integrity.fault_injection import FaultInjector

from src.bcrf.fci import compute_fci
from src.bcrf.bcr import compute_bcr
from src.bcrf.causal_frontier import construct_crf, is_non_contiguous
from src.bcrf.src_manager import SRCManager
from src.bcrf.operation_registry import ExternalOperationRegistry
from src.bcrf.civ import compute_civ, validate_civ
from src.bcrf.recovery_controller import BCRFRecoveryController
from src.bcrf.resource_evaluation import measure_resource_usage

from src.baselines.full_rollback import FullRollbackRecovery
from src.baselines.checkpoint_recovery import CheckpointRecovery
from src.baselines.dependency_recovery import DependencyRecovery


def run_single_trial(
    scenario: str,
    num_states: int,
    seed: int,
    fault_rate: float = 0.1,
    fault_type: Optional[str] = None,
    checkpoint_interval: int = 10,
    fci_threshold_method: str = "percentile",
    fci_threshold_value: float = 0.6,
    bcr_threshold_method: str = "percentile",
    bcr_threshold_value: float = 0.5,
    civ_weights=(0.4, 0.35, 0.25),
    civ_threshold: float = 0.65,
    max_recovery_iterations: int = 5,
    expansion_factor: float = 0.15,
    # Ablation flags for BCRF
    use_fci: bool = True,
    use_bcr: bool = True,
    use_registry: bool = True,
    use_civ: bool = True,
    use_crf_expansion: bool = True,
) -> Dict[str, Any]:
    """
    Run a single experimental trial with all recovery methods.

    Returns a dict with results for all methods.
    """
    # ── 1. Build and execute the agent scenario ──────────────────────────────
    sim = AgentSimulator(
        scenario=scenario,
        num_states=num_states,
        seed=seed,
        checkpoint_interval=checkpoint_interval,
    )
    original_states, graph = sim.run()

    # ── 2. Create SRCs from clean states ─────────────────────────────────────
    src_manager = SRCManager()
    src_manager.create_all_capsules(original_states)

    # ── 3. Register all external operations in the registry ──────────────────
    registry = ExternalOperationRegistry()
    for entry in sim.get_external_operations():
        if entry["operation_id"] is not None:
            registry.register_operation(
                operation_id=entry["operation_id"],
                operation_type=entry.get("state_type", "external"),
                state_id=entry["state_id"],
                result={"status": "completed", "step": entry["state_id"]},
                args={"state_id": entry["state_id"]},
            )

    # ── 4. Set up integrity monitor ───────────────────────────────────────────
    monitor = IntegrityMonitor(graph)
    monitor.register_baseline(original_states)

    # ── 5. Inject faults (on a COPY of the states) ───────────────────────────
    corrupted_states = [s.clone() for s in original_states]
    # Update graph to use corrupted states
    injector = FaultInjector(seed=seed + 1)
    corrupted_list, fault_records = injector.inject_random_faults(
        corrupted_states, fault_rate=fault_rate, fault_type=fault_type
    )
    corrupted_state_ids = [s.state_id for s in corrupted_list]

    if not corrupted_state_ids:
        # Ensure at least one fault
        fault_idx = max(1, num_states // 4)
        ft = fault_type or "memory_corruption"
        injector.inject_single_fault(corrupted_states, fault_idx, ft)
        corrupted_state_ids = [corrupted_states[fault_idx].state_id]

    # Update graph with corrupted states
    for cs in corrupted_states:
        if cs.state_id in graph.states:
            graph.states[cs.state_id] = cs

    n_total = len(original_states)
    n_external = len(sim.get_external_operations())

    results = {
        "scenario": scenario,
        "num_states": num_states,
        "seed": seed,
        "fault_type": fault_type or "random",
        "fault_rate": fault_rate,
        "n_corrupted": len(corrupted_state_ids),
        "n_external_ops": n_external,
    }

    # ── 6. Baseline 1: Full Rollback ─────────────────────────────────────────
    def run_full_rollback():
        bl1 = FullRollbackRecovery(registry=registry)
        return bl1.recover(original_states, corrupted_states, corrupted_state_ids)

    bl1_result, bl1_resources = measure_resource_usage(run_full_rollback)
    bl1_result.update({
        "cpu_time_ms": bl1_resources["cpu_time_ms"],
        "wall_time_ms": bl1_resources["wall_time_ms"],
        "memory_delta_bytes": bl1_resources["memory_delta_bytes"],
        "pct_reconstructed": round(bl1_result["states_reconstructed"] / n_total * 100, 2),
    })
    results["full_rollback"] = bl1_result

    # ── 7. Baseline 2: Checkpoint Recovery ───────────────────────────────────
    def run_checkpoint():
        bl2 = CheckpointRecovery(
            checkpoints=sim.checkpoints,
            checkpoint_interval=checkpoint_interval,
            registry=registry,
        )
        return bl2.recover(original_states, corrupted_states, corrupted_state_ids)

    bl2_result, bl2_resources = measure_resource_usage(run_checkpoint)
    bl2_result.update({
        "cpu_time_ms": bl2_resources["cpu_time_ms"],
        "wall_time_ms": bl2_resources["wall_time_ms"],
        "memory_delta_bytes": bl2_resources["memory_delta_bytes"],
        "pct_reconstructed": round(bl2_result["states_reconstructed"] / n_total * 100, 2),
    })
    results["checkpoint"] = bl2_result

    # ── 8. Baseline 3: Dependency Recovery ───────────────────────────────────
    def run_dependency():
        bl3 = DependencyRecovery(graph=graph, registry=registry)
        return bl3.recover(original_states, corrupted_states, corrupted_state_ids)

    bl3_result, bl3_resources = measure_resource_usage(run_dependency)
    bl3_result.update({
        "cpu_time_ms": bl3_resources["cpu_time_ms"],
        "wall_time_ms": bl3_resources["wall_time_ms"],
        "memory_delta_bytes": bl3_resources["memory_delta_bytes"],
        "pct_reconstructed": round(bl3_result["states_reconstructed"] / n_total * 100, 2),
    })
    results["dependency"] = bl3_result

    # ── 9. BCRF: Complete Method ──────────────────────────────────────────────
    def run_bcrf():
        controller = BCRFRecoveryController(
            graph=graph,
            src_manager=src_manager,
            registry=registry,
            monitor=monitor,
            fci_threshold_method=fci_threshold_method,
            fci_threshold_value=fci_threshold_value,
            bcr_threshold_method=bcr_threshold_method,
            bcr_threshold_value=bcr_threshold_value,
            civ_weights=civ_weights,
            civ_threshold=civ_threshold,
            max_iterations=max_recovery_iterations,
            expansion_factor_per_iter=expansion_factor,
            use_fci=use_fci,
            use_bcr=use_bcr,
            use_registry=use_registry,
            use_civ=use_civ,
            use_crf_expansion=use_crf_expansion,
        )
        return controller.recover(
            original_states=original_states,
            corrupted_states=corrupted_states,
            corrupted_state_ids=corrupted_state_ids,
        )

    bcrf_rec_result, bcrf_resources = measure_resource_usage(run_bcrf)
    bcrf_dict = bcrf_rec_result.to_dict()
    bcrf_dict.update({
        "cpu_time_ms": bcrf_resources["cpu_time_ms"],
        "wall_time_ms": bcrf_resources["wall_time_ms"],
        "memory_delta_bytes": bcrf_resources["memory_delta_bytes"],
        "pct_reconstructed": round(bcrf_dict["total_reconstructed"] / n_total * 100, 2),
        "states_reconstructed": bcrf_dict["total_reconstructed"],
        "states_skipped": bcrf_dict["total_skipped"],
        "duplicate_side_effects": 0 if use_registry else (
            bcrf_dict.get("total_external_ops_prevented", 0)
        ),
        "recovery_time_ms": bcrf_dict["total_time_ms"],
        "recovery_iterations": bcrf_dict["total_attempts"],
        "crf_is_non_contiguous": bcrf_dict["crf_is_non_contiguous"],
        "crf_size": bcrf_dict["crf_final_size"],
        "external_ops_prevented": bcrf_dict["total_external_ops_prevented"],
    })
    results["bcrf"] = bcrf_dict

    return results
