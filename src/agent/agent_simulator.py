"""
BCRF Experiment - Agent Simulator
Orchestrates full agent execution, records execution history,
and provides a clean interface for fault injection and recovery.
"""

import time
import random
from typing import List, Optional, Dict, Any, Tuple
from src.agent.state import AgentState, StateType, StateStatus
from src.agent.execution_graph import ExecutionGraph
from src.agent.scenarios import build_scenario, SCENARIO_BUILDERS


class AgentSimulator:
    """
    Synthetic stateful AI agent execution environment.

    Executes a sequence S₁ → S₂ → ... → Sₙ, maintaining:
    - Full execution history
    - Dependency graph
    - Checkpoints at regular intervals
    - External operation log
    """

    def __init__(
        self,
        scenario: str = "order_processing",
        num_states: int = 50,
        seed: int = 42,
        checkpoint_interval: int = 10,
    ):
        self.scenario = scenario
        self.num_states = num_states
        self.seed = seed
        self.checkpoint_interval = checkpoint_interval
        self.rng = random.Random(seed)

        self.states: List[AgentState] = []
        self.graph: ExecutionGraph = ExecutionGraph()
        self.checkpoints: Dict[int, List[AgentState]] = {}  # index → snapshot
        self.execution_log: List[Dict] = []
        self.is_executed: bool = False

    def run(self) -> Tuple[List[AgentState], ExecutionGraph]:
        """Execute the agent scenario and build the execution graph."""
        self.states, self.graph = build_scenario(
            self.scenario, self.num_states, self.seed
        )

        # Record checkpoints
        for i, state in enumerate(self.states):
            if i % self.checkpoint_interval == 0:
                snapshot = [s.clone() for s in self.states[: i + 1]]
                self.checkpoints[i] = snapshot

            self.execution_log.append({
                "index": i,
                "state_id": state.state_id,
                "state_type": state.state_type.value,
                "logical_clock": state.metadata.logical_clock,
                "is_external": state.tool.is_external,
                "side_effect": state.tool.side_effect_produced,
                "operation_id": state.tool.operation_id,
            })

        self.is_executed = True
        return self.states, self.graph

    def get_state(self, state_id: str) -> Optional[AgentState]:
        """Retrieve a state by ID."""
        for s in self.states:
            if s.state_id == state_id:
                return s
        return None

    def get_states_by_index(self, start: int, end: int) -> List[AgentState]:
        """Get states in index range [start, end]."""
        return self.states[start: end + 1]

    def get_latest_checkpoint_before(self, state_index: int) -> Tuple[int, List[AgentState]]:
        """Return (checkpoint_index, snapshot) for the latest checkpoint before state_index."""
        valid_checkpoints = [
            idx for idx in self.checkpoints if idx <= state_index
        ]
        if not valid_checkpoints:
            return 0, [self.states[0].clone()]
        latest = max(valid_checkpoints)
        return latest, self.checkpoints[latest]

    def get_external_operations(self) -> List[Dict]:
        """Return all logged external operations."""
        return [
            entry for entry in self.execution_log if entry["side_effect"]
        ]

    def compute_execution_value(self, state: AgentState) -> float:
        """
        Compute a numerical 'execution value' for a state.
        This represents the observed outcome quality / integrity score.
        Based on: output consistency, memory coherence, tool result validity.
        """
        value = 0.0
        # Output contribution
        if isinstance(state.output, dict):
            value += state.output.get("value", 0.0) * 0.5
        # Memory coherence contribution
        mem_size = len(state.memory.working_memory)
        value += min(mem_size * 2.0, 20.0)
        # Planning confidence contribution
        value += state.planning.confidence * 30.0
        # Tool result contribution
        if state.tool.tool_result is not None:
            value += 10.0
        # External operation penalty (side effects cost more to re-execute)
        if state.tool.side_effect_produced:
            value += 15.0
        return round(value, 4)

    def summary(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "num_states": len(self.states),
            "seed": self.seed,
            "checkpoints": list(self.checkpoints.keys()),
            "external_operations": len(self.get_external_operations()),
            "graph_summary": self.graph.summary(),
        }
