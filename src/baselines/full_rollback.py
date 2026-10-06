"""
BCRF Experiment - Baseline 1: Full Rollback and Replay
Restores execution to the very beginning and replays all operations.
"""

import time
from typing import List, Dict, Optional
from src.agent.state import AgentState, StateStatus
from src.bcrf.operation_registry import ExternalOperationRegistry


class FullRollbackRecovery:
    """
    Baseline 1: Full Rollback and Replay.

    Strategy: Roll back to the beginning of execution and replay
    ALL states from scratch. This is the most conservative recovery
    method and serves as the upper bound for recovery cost.

    Does NOT use:
    - FCI / BCR causal analysis
    - CRF selective reconstruction
    - External Operation Registry (re-executes all external ops)
    - CIV validation
    """

    def __init__(self, registry: Optional[ExternalOperationRegistry] = None):
        # Registry NOT used in this baseline (to show duplicate side effects)
        self.registry = registry
        self.recovery_log: List[Dict] = []

    def recover(
        self,
        original_states: List[AgentState],
        corrupted_states: List[AgentState],
        corrupted_state_ids: List[str],
        checkpoint_states: Optional[List[AgentState]] = None,
    ) -> Dict:
        """
        Execute full rollback and replay from the beginning.

        Returns recovery metrics dict.
        """
        t_start = time.time()

        # Identify the first corrupted state index
        corrupted_indices = []
        for i, s in enumerate(corrupted_states):
            if s.state_id in corrupted_state_ids:
                corrupted_indices.append(i)

        rollback_point = 0  # Full rollback: go to beginning

        # Count ALL states that need to be replayed
        states_to_replay = corrupted_states[rollback_point:]
        n_replayed = len(states_to_replay)

        # Simulate replay: mark all states as replayed
        reconstructed = []
        external_ops_re_executed = 0
        duplicate_ops = 0

        for state in states_to_replay:
            cloned = original_states[corrupted_states.index(state)].clone() if state.state_id in [
                s.state_id for s in original_states
            ] else state.clone()
            cloned.status = StateStatus.RECOVERED
            reconstructed.append(cloned)

            # External operations ARE re-executed (no registry check)
            if state.tool.side_effect_produced or state.tool.is_external:
                external_ops_re_executed += 1
                # Check if this would be a duplicate
                if self.registry is not None:
                    existing = self.registry.check_already_completed(
                        state.tool.operation_id,
                        state.tool.tool_name,
                        state.state_id,
                        state.tool.tool_args,
                    )
                    if existing is not None:
                        duplicate_ops += 1  # Would be duplicate but we do it anyway

        recovery_time_ms = (time.time() - t_start) * 1000

        result = {
            "method": "full_rollback",
            "success": True,  # Full rollback always "succeeds" structurally
            "rollback_point": rollback_point,
            "states_replayed": n_replayed,
            "states_reconstructed": n_replayed,
            "states_skipped": 0,
            "external_ops_re_executed": external_ops_re_executed,
            "duplicate_side_effects": duplicate_ops,
            "recovery_time_ms": round(recovery_time_ms, 3),
            "civ_score": 1.0,  # Not computed; assumed valid for this baseline
            "recovery_iterations": 1,
        }
        self.recovery_log.append(result)
        return result
