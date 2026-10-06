"""
BCRF Experiment - State Reconstruction Capsule (SRC) Manager
Implements SRC creation, storage, retrieval, and state reconstruction.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any
from src.agent.state import AgentState, StateStatus


class StateReconstructionCapsule:
    """
    A capsule containing all information needed to reconstruct an execution state.

    Contains:
    - state identifier
    - full state snapshot
    - relevant memory references
    - dependency information
    - tool outputs
    - execution metadata
    - integrity signature of the capsule itself
    - creation time
    - anchor validity flag
    """

    def __init__(self, state: AgentState):
        self.capsule_id = f"SRC_{state.state_id}"
        self.state_id = state.state_id
        self.state_snapshot = state.clone()
        self.original_integrity_signature = state.original_signature
        self.memory_snapshot = state.memory.clone()
        self.planning_snapshot = state.planning.clone()
        self.tool_snapshot = state.tool.clone()
        self.dependency_ids = list(state.dependency_ids)
        self.parent_ids = list(state.parent_ids)
        self.output_snapshot = state.output
        self.metadata_snapshot = state.metadata.to_dict()
        self.creation_time = time.time()
        self.is_anchor_valid = True
        self.reconstruction_count = 0

        # Capsule integrity signature
        self.capsule_signature = self._compute_capsule_signature()

    def _compute_capsule_signature(self) -> str:
        content = {
            "capsule_id": self.capsule_id,
            "state_id": self.state_id,
            "original_integrity_signature": self.original_integrity_signature,
            "memory": self.memory_snapshot.to_dict(),
            "dependency_ids": sorted(self.dependency_ids),
            "output": str(self.output_snapshot),
        }
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def verify_capsule_integrity(self) -> bool:
        """Verify that the capsule itself hasn't been tampered with."""
        expected = self._compute_capsule_signature()
        return expected == self.capsule_signature

    def reconstruct_state(self) -> AgentState:
        """Reconstruct the original clean state from the capsule."""
        reconstructed = self.state_snapshot.clone()
        reconstructed.memory = self.memory_snapshot.clone()
        reconstructed.planning = self.planning_snapshot.clone()
        reconstructed.tool = self.tool_snapshot.clone()
        reconstructed.output = self.output_snapshot
        reconstructed.dependency_ids = set(self.dependency_ids)
        reconstructed.parent_ids = list(self.parent_ids)
        reconstructed.is_corrupted = False
        reconstructed.corruption_type = None
        reconstructed.status = StateStatus.RECOVERED
        reconstructed.integrity_signature = self.original_integrity_signature
        self.reconstruction_count += 1
        return reconstructed

    def to_dict(self) -> Dict:
        return {
            "capsule_id": self.capsule_id,
            "state_id": self.state_id,
            "original_integrity_signature": self.original_integrity_signature,
            "capsule_signature": self.capsule_signature,
            "creation_time": self.creation_time,
            "is_anchor_valid": self.is_anchor_valid,
            "reconstruction_count": self.reconstruction_count,
            "dependency_ids": self.dependency_ids,
            "parent_ids": self.parent_ids,
        }


class SRCManager:
    """
    Manager for State Reconstruction Capsules.

    Stores capsules for all states pre-corruption.
    During recovery, retrieves and reconstructs selected states.
    """

    def __init__(self):
        self.capsules: Dict[str, StateReconstructionCapsule] = {}
        self.reconstruction_log: List[Dict] = []

    def create_capsule(self, state: AgentState) -> StateReconstructionCapsule:
        """Create and store a capsule for a given state."""
        capsule = StateReconstructionCapsule(state)
        self.capsules[state.state_id] = capsule
        return capsule

    def create_all_capsules(self, states: List[AgentState]) -> int:
        """Create capsules for all states. Returns number of capsules created."""
        for state in states:
            self.create_capsule(state)
        return len(states)

    def get_capsule(self, state_id: str) -> Optional[StateReconstructionCapsule]:
        """Retrieve a capsule by state_id."""
        return self.capsules.get(state_id)

    def verify_capsule(self, state_id: str) -> bool:
        """Verify the integrity of a stored capsule."""
        capsule = self.capsules.get(state_id)
        if capsule is None:
            return False
        return capsule.verify_capsule_integrity()

    def reconstruct_state(self, state_id: str) -> Optional[AgentState]:
        """
        Reconstruct a state from its capsule.

        Returns the reconstructed state or None if capsule is missing/invalid.
        """
        capsule = self.capsules.get(state_id)
        if capsule is None:
            self.reconstruction_log.append({
                "state_id": state_id,
                "success": False,
                "reason": "capsule_not_found",
                "timestamp": time.time(),
            })
            return None

        if not capsule.verify_capsule_integrity():
            self.reconstruction_log.append({
                "state_id": state_id,
                "success": False,
                "reason": "capsule_integrity_failed",
                "timestamp": time.time(),
            })
            capsule.is_anchor_valid = False
            return None

        reconstructed = capsule.reconstruct_state()
        self.reconstruction_log.append({
            "state_id": state_id,
            "success": True,
            "reason": "reconstructed_from_capsule",
            "timestamp": time.time(),
            "reconstruction_count": capsule.reconstruction_count,
        })
        return reconstructed

    def reconstruct_crf_states(
        self, crf_state_ids: set
    ) -> Dict[str, Optional[AgentState]]:
        """Reconstruct all states in the CRF. Returns dict of state_id → state."""
        results = {}
        for sid in crf_state_ids:
            results[sid] = self.reconstruct_state(sid)
        return results

    def get_reconstruction_success_rate(self) -> float:
        if not self.reconstruction_log:
            return 1.0
        successes = sum(1 for r in self.reconstruction_log if r["success"])
        return successes / len(self.reconstruction_log)

    def summary(self) -> Dict:
        return {
            "total_capsules": len(self.capsules),
            "reconstruction_attempts": len(self.reconstruction_log),
            "successful_reconstructions": sum(
                1 for r in self.reconstruction_log if r["success"]
            ),
            "success_rate": self.get_reconstruction_success_rate(),
        }
