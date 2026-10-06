"""
BCRF Experiment - Baseline 3: Dependency-Guided Selective Recovery
Selects states using dependency relationships, WITHOUT FCI/BCR/CRF/Registry.
"""

import time
from typing import List, Dict, Set, Optional
from src.agent.state import AgentState, StateStatus
from src.agent.execution_graph import ExecutionGraph
from src.bcrf.operation_registry import ExternalOperationRegistry


class DependencyRecovery:
    """
    Baseline 3: Dependency-Guided Selective Recovery.

    Strategy: Starting from corrupted states, follow dependency edges backward
    to identify states to reconstruct. Select all ancestors of corrupted states.

    Uses: dependency graph traversal
    Does NOT use:
    - FCI (no counterfactual analysis)
    - BCR (no backward causal responsibility weighting)
    - CRF threshold selection (takes all ancestors)
    - External Operation Registry (re-executes external ops)
    - CIV validation

    This isolates the contribution of bidirectional causal analysis.
    """

    def __init__(
        self,
        graph: ExecutionGraph,
        registry: Optional[ExternalOperationRegistry] = None,
    ):
        self.graph = graph
        self.registry = registry
        self.recovery_log: List[Dict] = []

    def _get_dependency_recovery_set(
        self,
        corrupted_state_ids: List[str],
        all_states: List[AgentState],
    ) -> Set[str]:
        """
        Identify states to recover using pure dependency traversal.
        Returns all ancestors of corrupted states (plus the corrupted states).
        """
        recovery_set: Set[str] = set(corrupted_state_ids)

        for cid in corrupted_state_ids:
            ancestors = self.graph.get_ancestors(cid)
            recovery_set.update(ancestors)

        return recovery_set

    def recover(
        self,
        original_states: List[AgentState],
        corrupted_states: List[AgentState],
        corrupted_state_ids: List[str],
    ) -> Dict:
        """Execute dependency-guided selective recovery."""
        t_start = time.time()

        original_map = {s.state_id: s for s in original_states}

        # Get recovery set via dependency traversal
        recovery_set = self._get_dependency_recovery_set(corrupted_state_ids, corrupted_states)

        # Reconstruct selected states
        reconstructed_ids = []
        skipped_ids = []
        external_ops_re_executed = 0
        duplicate_ops = 0

        for state in corrupted_states:
            if state.state_id in recovery_set:
                reconstructed_ids.append(state.state_id)
                # External ops are re-executed (no registry check)
                if state.tool.side_effect_produced or state.tool.is_external:
                    external_ops_re_executed += 1
                    if self.registry is not None:
                        existing = self.registry.check_already_completed(
                            state.tool.operation_id,
                            state.tool.tool_name,
                            state.state_id,
                            state.tool.tool_args,
                        )
                        if existing is not None:
                            duplicate_ops += 1
            else:
                skipped_ids.append(state.state_id)

        recovery_time_ms = (time.time() - t_start) * 1000

        result = {
            "method": "dependency",
            "success": True,
            "states_reconstructed": len(reconstructed_ids),
            "states_skipped": len(skipped_ids),
            "recovery_set_size": len(recovery_set),
            "external_ops_re_executed": external_ops_re_executed,
            "duplicate_side_effects": duplicate_ops,
            "recovery_time_ms": round(recovery_time_ms, 3),
            "civ_score": 1.0,
            "recovery_iterations": 1,
        }
        self.recovery_log.append(result)
        return result
