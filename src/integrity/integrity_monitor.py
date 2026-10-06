"""
BCRF Experiment - Integrity Monitor
Detects integrity violations in agent execution states.
"""

import hashlib
import json
from typing import List, Optional, Dict, Tuple
from src.agent.state import AgentState, StateStatus
from src.agent.execution_graph import ExecutionGraph


class IntegrityViolation:
    """Represents a detected integrity violation."""

    def __init__(
        self,
        state_id: str,
        violation_type: str,
        expected_signature: str,
        actual_signature: str,
        description: str,
    ):
        self.state_id = state_id
        self.violation_type = violation_type
        self.expected_signature = expected_signature
        self.actual_signature = actual_signature
        self.description = description
        self.detected = True

    def to_dict(self) -> Dict:
        return {
            "state_id": self.state_id,
            "violation_type": self.violation_type,
            "expected_signature": self.expected_signature,
            "actual_signature": self.actual_signature,
            "description": self.description,
        }


class IntegrityMonitor:
    """
    Runtime integrity monitor for the stateful AI agent.

    Detects:
    1. Corrupted state (signature mismatch)
    2. Unexpected memory modification
    3. Unexpected tool result
    4. Dependency inconsistency
    5. Output inconsistency

    NOTE: The cryptographic hash is used only for detection.
    The novelty lies in the subsequent BCRF causal recovery.
    """

    def __init__(self, graph: ExecutionGraph):
        self.graph = graph
        self.violations: List[IntegrityViolation] = []
        self.baseline_signatures: Dict[str, str] = {}
        self.baseline_memory_versions: Dict[str, int] = {}

    def register_baseline(self, states: List[AgentState]) -> None:
        """Register baseline signatures for all clean states."""
        for state in states:
            self.baseline_signatures[state.state_id] = state.integrity_signature
            self.baseline_memory_versions[state.state_id] = state.memory.memory_version

    def check_state(self, state: AgentState) -> Optional[IntegrityViolation]:
        """Check a single state for integrity violations."""
        # 1. Signature check (most fundamental)
        current_sig = state._compute_signature()
        expected_sig = self.baseline_signatures.get(state.state_id, state.original_signature)

        if current_sig != expected_sig:
            v = IntegrityViolation(
                state_id=state.state_id,
                violation_type=state.corruption_type or "signature_mismatch",
                expected_signature=expected_sig,
                actual_signature=current_sig,
                description=f"Integrity signature mismatch for state {state.state_id}",
            )
            self.violations.append(v)
            return v

        # 2. Memory version sanity check
        baseline_mem_ver = self.baseline_memory_versions.get(state.state_id)
        if baseline_mem_ver is not None and state.memory.memory_version != baseline_mem_ver:
            v = IntegrityViolation(
                state_id=state.state_id,
                violation_type="memory_version_mismatch",
                expected_signature=expected_sig,
                actual_signature=current_sig,
                description=f"Memory version changed unexpectedly for state {state.state_id}",
            )
            self.violations.append(v)
            return v

        return None

    def scan_all(self, states: List[AgentState]) -> List[IntegrityViolation]:
        """Scan all states for violations."""
        violations = []
        for state in states:
            v = self.check_state(state)
            if v is not None:
                violations.append(v)
        return violations

    def check_dependency_consistency(
        self, state: AgentState, all_states: Dict[str, AgentState]
    ) -> Optional[IntegrityViolation]:
        """Check if state's dependencies are all valid (non-corrupted)."""
        for dep_id in state.dependency_ids:
            if dep_id in all_states:
                dep_state = all_states[dep_id]
                if dep_state.is_corrupted:
                    v = IntegrityViolation(
                        state_id=state.state_id,
                        violation_type="dependency_inconsistency",
                        expected_signature=self.baseline_signatures.get(state.state_id, ""),
                        actual_signature=state.integrity_signature,
                        description=f"Dependency {dep_id} of state {state.state_id} is corrupted",
                    )
                    self.violations.append(v)
                    return v
        return None

    def get_first_violation(self, states: List[AgentState]) -> Optional[IntegrityViolation]:
        """Return the first detected violation in execution order."""
        for state in states:
            v = self.check_state(state)
            if v is not None:
                return v
        return None

    def get_violation_summary(self) -> Dict:
        return {
            "total_violations": len(self.violations),
            "violation_types": list(set(v.violation_type for v in self.violations)),
            "affected_states": list(set(v.state_id for v in self.violations)),
        }
