"""
BCRF Experiment - Fault Injection Module
Controlled, reproducible fault injection for experimental validation.
"""

import random
import time
import copy
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from src.agent.state import AgentState, StateStatus


FAULT_TYPES = [
    "memory_corruption",
    "planning_corruption",
    "tool_corruption",
    "dependency_corruption",
    "metadata_corruption",
    "combined_corruption",
]


@dataclass
class FaultRecord:
    """Records details of an injected fault."""
    fault_id: str
    state_id: str
    fault_type: str
    corruption_location: str
    timestamp: float
    original_value: Any
    corrupted_value: Any
    affected_downstream_count: int
    seed: int

    def to_dict(self) -> Dict:
        return {
            "fault_id": self.fault_id,
            "state_id": self.state_id,
            "fault_type": self.fault_type,
            "corruption_location": self.corruption_location,
            "timestamp": self.timestamp,
            "original_value": str(self.original_value),
            "corrupted_value": str(self.corrupted_value),
            "affected_downstream_count": self.affected_downstream_count,
            "seed": self.seed,
        }


class FaultInjector:
    """
    Controlled fault injection for the stateful AI agent.

    All injections use fixed random seeds for reproducibility.

    Fault types:
    1. memory_corruption    - corrupts working memory values
    2. planning_corruption  - modifies planning state / goal / subtasks
    3. tool_corruption      - alters tool result
    4. dependency_corruption- corrupts dependency metadata
    5. metadata_corruption  - corrupts execution metadata fields
    6. combined_corruption  - applies multiple fault types simultaneously
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.fault_records: List[FaultRecord] = []
        self._fault_counter = 0

    def _next_fault_id(self) -> str:
        self._fault_counter += 1
        return f"FAULT_{self._fault_counter:06d}"

    def inject_fault(
        self,
        state: AgentState,
        fault_type: str,
        downstream_count: int = 0,
    ) -> FaultRecord:
        """Inject a fault into a state and return the fault record."""
        fault_id = self._next_fault_id()
        original_value = None
        corrupted_value = None
        location = ""

        if fault_type == "memory_corruption":
            if state.memory.working_memory:
                key = self.rng.choice(list(state.memory.working_memory.keys()))
                original_value = state.memory.working_memory[key]
                corrupted_value = self._corrupt_value(original_value)
                state.memory.working_memory[key] = corrupted_value
                location = f"memory.working_memory[{key}]"
            else:
                state.memory.working_memory["corrupted_key"] = "CORRUPTED"
                original_value = None
                corrupted_value = "CORRUPTED"
                location = "memory.working_memory[corrupted_key]"

        elif fault_type == "planning_corruption":
            original_value = state.planning.current_goal
            corrupted_value = "CORRUPTED_GOAL_" + str(self.rng.randint(1000, 9999))
            state.planning.current_goal = corrupted_value
            location = "planning.current_goal"

        elif fault_type == "tool_corruption":
            original_value = state.tool.tool_result
            if isinstance(state.tool.tool_result, dict):
                corrupted_value = dict(state.tool.tool_result)
                corrupted_value["_corrupted"] = True
                corrupted_value["value"] = self.rng.uniform(-1000, -1)
            else:
                corrupted_value = "TOOL_ERROR_" + str(self.rng.randint(1000, 9999))
            state.tool.tool_result = corrupted_value
            location = "tool.tool_result"

        elif fault_type == "dependency_corruption":
            # Remove a valid dependency or add a spurious one
            original_value = list(state.dependency_ids)
            if state.dependency_ids:
                removed = self.rng.choice(list(state.dependency_ids))
                state.dependency_ids.discard(removed)
                corrupted_value = list(state.dependency_ids)
            else:
                state.dependency_ids.add("SPURIOUS_DEP_9999")
                corrupted_value = list(state.dependency_ids)
            location = "state.dependency_ids"

        elif fault_type == "metadata_corruption":
            original_value = state.metadata.logical_clock
            corrupted_value = -abs(self.rng.randint(0, 9999))
            state.metadata.logical_clock = corrupted_value
            location = "metadata.logical_clock"

        elif fault_type == "combined_corruption":
            # Apply memory + tool corruption
            record1 = self.inject_fault(state, "memory_corruption", downstream_count)
            record2 = self.inject_fault(state, "tool_corruption", downstream_count)
            original_value = f"mem:{record1.original_value}, tool:{record2.original_value}"
            corrupted_value = f"mem:{record1.corrupted_value}, tool:{record2.corrupted_value}"
            location = "memory + tool"

        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

        # Mark state as corrupted and recompute its signature (which will now differ)
        state.is_corrupted = True
        state.corruption_type = fault_type
        state.status = StateStatus.CORRUPTED
        state.recompute_signature()  # Now differs from original_signature

        record = FaultRecord(
            fault_id=fault_id,
            state_id=state.state_id,
            fault_type=fault_type,
            corruption_location=location,
            timestamp=time.time(),
            original_value=original_value,
            corrupted_value=corrupted_value,
            affected_downstream_count=downstream_count,
            seed=self.seed,
        )
        self.fault_records.append(record)
        return record

    def _corrupt_value(self, value: Any) -> Any:
        """Corrupt a value in a type-appropriate way."""
        if isinstance(value, (int, float)):
            return value * self.rng.choice([-999, 0, 999]) + self.rng.uniform(-1e6, 1e6)
        elif isinstance(value, str):
            return "CORRUPTED_" + value[::-1] + str(self.rng.randint(0, 9999))
        elif isinstance(value, list):
            return ["CORRUPTED"] * len(value)
        elif isinstance(value, dict):
            return {k: "CORRUPTED" for k in value}
        elif value is None:
            return "CORRUPTED_NONE"
        else:
            return "CORRUPTED_" + str(value)

    def inject_random_faults(
        self,
        states: List[AgentState],
        fault_rate: float = 0.1,
        fault_type: Optional[str] = None,
        min_index: int = 1,  # Don't corrupt the very first state
    ) -> Tuple[List[AgentState], List[FaultRecord]]:
        """
        Inject faults into a random subset of states.

        Args:
            states: List of agent states to potentially corrupt
            fault_rate: Fraction of states to corrupt (0.0–1.0)
            fault_type: Specific fault type or None for random selection
            min_index: Skip states before this index

        Returns:
            (corrupted_states_list, fault_records)
        """
        corrupted = []
        records = []
        eligible = [s for i, s in enumerate(states) if i >= min_index]
        n_faults = max(1, int(len(eligible) * fault_rate))
        targets = self.rng.sample(eligible, min(n_faults, len(eligible)))

        for state in targets:
            ftype = fault_type or self.rng.choice(FAULT_TYPES)
            # Estimate downstream count from state index
            idx = next((i for i, s in enumerate(states) if s.state_id == state.state_id), 0)
            downstream_count = len(states) - idx - 1
            record = self.inject_fault(state, ftype, downstream_count)
            corrupted.append(state)
            records.append(record)

        return corrupted, records

    def inject_single_fault(
        self,
        states: List[AgentState],
        fault_index: int,
        fault_type: str,
    ) -> FaultRecord:
        """Inject a single fault at a specific state index."""
        state = states[fault_index]
        downstream_count = len(states) - fault_index - 1
        return self.inject_fault(state, fault_type, downstream_count)

    def get_records(self) -> List[Dict]:
        return [r.to_dict() for r in self.fault_records]

    def reset(self) -> None:
        self.fault_records.clear()
        self._fault_counter = 0
        self.rng = random.Random(self.seed)
