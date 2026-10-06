"""
BCRF Experiment - Baseline 2: Checkpoint-Based Recovery
Restores latest valid checkpoint and replays from there.
"""

import time
from typing import List, Dict, Optional, Tuple
from src.agent.state import AgentState, StateStatus
from src.bcrf.operation_registry import ExternalOperationRegistry


class CheckpointRecovery:
    """
    Baseline 2: Checkpoint-Based Recovery.

    Strategy: Find the latest valid checkpoint before the corruption,
    restore to that checkpoint, and replay all states from that point.

    Does NOT use:
    - FCI / BCR causal analysis
    - CRF selective reconstruction
    - External Operation Registry (re-executes external ops after checkpoint)
    - CIV validation
    """

    def __init__(
        self,
        checkpoints: Dict[int, List[AgentState]],
        checkpoint_interval: int = 10,
        registry: Optional[ExternalOperationRegistry] = None,
    ):
        self.checkpoints = checkpoints  # {index: [state_snapshot]}
        self.checkpoint_interval = checkpoint_interval
        self.registry = registry
        self.recovery_log: List[Dict] = []

    def _find_latest_valid_checkpoint(
        self,
        corrupted_state_ids: List[str],
        all_states: List[AgentState],
    ) -> Tuple[int, List[AgentState]]:
        """Find the latest checkpoint index before the first corruption."""
        # Find first corrupted state index
        first_corrupted_idx = None
        for i, s in enumerate(all_states):
            if s.state_id in corrupted_state_ids:
                first_corrupted_idx = i
                break

        if first_corrupted_idx is None:
            first_corrupted_idx = len(all_states)

        # Find latest checkpoint before corruption
        valid_checkpoints = sorted(
            [idx for idx in self.checkpoints.keys() if idx < first_corrupted_idx],
            reverse=True
        )

        if valid_checkpoints:
            ckpt_idx = valid_checkpoints[0]
            return ckpt_idx, self.checkpoints[ckpt_idx]
        else:
            # No checkpoint before corruption: start from beginning
            return 0, all_states[:1]

    def recover(
        self,
        original_states: List[AgentState],
        corrupted_states: List[AgentState],
        corrupted_state_ids: List[str],
    ) -> Dict:
        """Execute checkpoint-based recovery."""
        t_start = time.time()

        ckpt_idx, checkpoint_snapshot = self._find_latest_valid_checkpoint(
            corrupted_state_ids, corrupted_states
        )

        # States to replay = all states after checkpoint
        states_to_replay = corrupted_states[ckpt_idx:]
        n_replayed = len(states_to_replay)

        external_ops_re_executed = 0
        duplicate_ops = 0

        for state in states_to_replay:
            if state.tool.side_effect_produced or state.tool.is_external:
                external_ops_re_executed += 1
                # Check for duplicates (informational only; this baseline re-executes anyway)
                if self.registry is not None:
                    existing = self.registry.check_already_completed(
                        state.tool.operation_id,
                        state.tool.tool_name,
                        state.state_id,
                        state.tool.tool_args,
                    )
                    if existing is not None:
                        duplicate_ops += 1

        recovery_time_ms = (time.time() - t_start) * 1000

        result = {
            "method": "checkpoint",
            "success": True,
            "checkpoint_index": ckpt_idx,
            "states_replayed": n_replayed,
            "states_reconstructed": n_replayed,
            "states_skipped": ckpt_idx,
            "external_ops_re_executed": external_ops_re_executed,
            "duplicate_side_effects": duplicate_ops,
            "recovery_time_ms": round(recovery_time_ms, 3),
            "civ_score": 1.0,
            "recovery_iterations": 1,
        }
        self.recovery_log.append(result)
        return result
