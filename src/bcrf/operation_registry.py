"""
BCRF Experiment - External Operation Registry
Records all external side-effect-producing operations.
Prevents duplicate re-execution during recovery.

This is a CORE component of the BCRF invention.
"""

import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class OperationStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REPLAYED = "replayed"


@dataclass
class ExternalOperation:
    """Record of a side-effect-producing external operation."""
    operation_id: str
    operation_type: str
    timestamp: float
    state_id: str
    status: OperationStatus
    side_effect_status: str       # "applied", "not_applied", "idempotent"
    idempotency_key: str          # Unique key for deduplication
    result: Any = None
    replay_prevented: bool = False
    replay_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "timestamp": self.timestamp,
            "state_id": self.state_id,
            "status": self.status.value,
            "side_effect_status": self.side_effect_status,
            "idempotency_key": self.idempotency_key,
            "result": str(self.result),
            "replay_prevented": self.replay_prevented,
            "replay_count": self.replay_count,
        }


class ExternalOperationRegistry:
    """
    Registry for all external side-effect-producing operations.

    Before recovery reconstruction, the system checks this registry:
    - If operation already COMPLETED → do NOT execute again
    - Instead: mark as already_completed, restore recorded result
    - Log replay-prevention events

    Prevents duplicate side effects (e.g., double database writes,
    double inventory reservations, double notifications).
    """

    def __init__(self):
        self.operations: Dict[str, ExternalOperation] = {}
        self.idempotency_map: Dict[str, str] = {}  # idempotency_key → operation_id
        self.replay_prevention_log: List[Dict] = []
        self._duplicate_prevention_count: int = 0

    def _compute_idempotency_key(
        self, operation_type: str, state_id: str, args: Dict
    ) -> str:
        """Compute a deterministic idempotency key for an operation."""
        content = {
            "operation_type": operation_type,
            "state_id": state_id,
            "args": args,
        }
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def register_operation(
        self,
        operation_id: str,
        operation_type: str,
        state_id: str,
        result: Any = None,
        args: Dict = None,
    ) -> ExternalOperation:
        """Register a completed external operation."""
        args = args or {}
        idem_key = self._compute_idempotency_key(operation_type, state_id, args)
        op = ExternalOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            timestamp=time.time(),
            state_id=state_id,
            status=OperationStatus.COMPLETED,
            side_effect_status="applied",
            idempotency_key=idem_key,
            result=result,
        )
        self.operations[operation_id] = op
        self.idempotency_map[idem_key] = operation_id
        return op

    def check_already_completed(
        self,
        operation_id: Optional[str],
        operation_type: str,
        state_id: str,
        args: Dict = None,
    ) -> Optional[ExternalOperation]:
        """
        Check if an operation has already been completed.

        Returns the existing operation record if it has been completed,
        or None if it needs to be executed.

        This is the CORE duplicate-prevention mechanism.
        """
        args = args or {}

        # Check by operation_id
        if operation_id and operation_id in self.operations:
            op = self.operations[operation_id]
            if op.status == OperationStatus.COMPLETED:
                return op

        # Check by idempotency key
        idem_key = self._compute_idempotency_key(operation_type, state_id, args)
        if idem_key in self.idempotency_map:
            op_id = self.idempotency_map[idem_key]
            op = self.operations.get(op_id)
            if op and op.status == OperationStatus.COMPLETED:
                return op

        return None

    def prevent_replay(
        self,
        operation_id: str,
        reason: str = "already_completed",
    ) -> bool:
        """
        Record that a replay was prevented for an operation.

        Returns True if prevention was recorded successfully.
        """
        if operation_id in self.operations:
            op = self.operations[operation_id]
            op.replay_prevented = True
            op.replay_count += 1
            self._duplicate_prevention_count += 1

            self.replay_prevention_log.append({
                "operation_id": operation_id,
                "operation_type": op.operation_type,
                "state_id": op.state_id,
                "prevented_at": time.time(),
                "reason": reason,
                "replay_count": op.replay_count,
            })
            return True
        return False

    def execute_or_skip(
        self,
        operation_id: Optional[str],
        operation_type: str,
        state_id: str,
        args: Dict = None,
        execute_fn=None,
    ) -> Dict:
        """
        Check registry and either:
        1. Skip execution (already completed) — return recorded result
        2. Execute and register (first time)

        This is the main entry point for external operations during recovery.
        """
        args = args or {}
        existing = self.check_already_completed(operation_id, operation_type, state_id, args)

        if existing is not None:
            # PREVENT DUPLICATE EXECUTION
            self.prevent_replay(existing.operation_id, "already_completed_during_recovery")
            return {
                "skipped": True,
                "operation_id": existing.operation_id,
                "result": existing.result,
                "reason": "operation_already_completed",
                "duplicate_prevented": True,
            }

        # First-time execution
        result = None
        if execute_fn is not None:
            result = execute_fn()
        else:
            result = {"executed": True, "timestamp": time.time()}

        # Register the operation
        eff_op_id = operation_id or f"auto_{len(self.operations):06d}"
        self.register_operation(eff_op_id, operation_type, state_id, result, args)

        return {
            "skipped": False,
            "operation_id": eff_op_id,
            "result": result,
            "reason": "first_execution",
            "duplicate_prevented": False,
        }

    def get_duplicate_prevention_count(self) -> int:
        """Return total number of replay preventions."""
        return self._duplicate_prevention_count

    def get_completed_operations(self) -> List[ExternalOperation]:
        return [op for op in self.operations.values()
                if op.status == OperationStatus.COMPLETED]

    def reset(self) -> None:
        """Reset the registry (for new trials)."""
        self.operations.clear()
        self.idempotency_map.clear()
        self.replay_prevention_log.clear()
        self._duplicate_prevention_count = 0

    def summary(self) -> Dict:
        return {
            "total_operations": len(self.operations),
            "completed": len(self.get_completed_operations()),
            "replay_preventions": self._duplicate_prevention_count,
            "replay_log_entries": len(self.replay_prevention_log),
        }
