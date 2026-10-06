"""
BCRF Experiment - Agent State Module
Defines the execution state data structure for the stateful AI agent simulator.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set
from enum import Enum


class StateType(Enum):
    NORMAL = "normal"
    TOOL_CALL = "tool_call"
    MEMORY_UPDATE = "memory_update"
    PLANNING = "planning"
    DECISION = "decision"
    EXTERNAL_OPERATION = "external_operation"
    CHECKPOINT = "checkpoint"
    RECOVERY = "recovery"


class StateStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CORRUPTED = "corrupted"
    RECOVERED = "recovered"
    SKIPPED = "skipped"


@dataclass
class MemoryState:
    """Represents the memory/context of the agent at a given state."""
    working_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    context_window: List[str] = field(default_factory=list)
    memory_version: int = 0

    def to_dict(self) -> Dict:
        return {
            "working_memory": self.working_memory,
            "long_term_memory": self.long_term_memory,
            "context_window": self.context_window,
            "memory_version": self.memory_version
        }

    def clone(self) -> "MemoryState":
        return MemoryState(
            working_memory=dict(self.working_memory),
            long_term_memory=dict(self.long_term_memory),
            context_window=list(self.context_window),
            memory_version=self.memory_version
        )


@dataclass
class PlanningState:
    """Represents the planning/decision context of the agent."""
    current_goal: str = ""
    subtasks: List[str] = field(default_factory=list)
    completed_subtasks: List[str] = field(default_factory=list)
    decision_history: List[Dict] = field(default_factory=list)
    plan_version: int = 0
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "current_goal": self.current_goal,
            "subtasks": self.subtasks,
            "completed_subtasks": self.completed_subtasks,
            "decision_history": self.decision_history,
            "plan_version": self.plan_version,
            "confidence": self.confidence
        }

    def clone(self) -> "PlanningState":
        return PlanningState(
            current_goal=self.current_goal,
            subtasks=list(self.subtasks),
            completed_subtasks=list(self.completed_subtasks),
            decision_history=[dict(d) for d in self.decision_history],
            plan_version=self.plan_version,
            confidence=self.confidence
        )


@dataclass
class ToolInteraction:
    """Represents a tool call and its result."""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Any = None
    tool_status: str = "none"
    is_external: bool = False
    side_effect_produced: bool = False
    operation_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "tool_status": self.tool_status,
            "is_external": self.is_external,
            "side_effect_produced": self.side_effect_produced,
            "operation_id": self.operation_id
        }

    def clone(self) -> "ToolInteraction":
        return ToolInteraction(
            tool_name=self.tool_name,
            tool_args=dict(self.tool_args),
            tool_result=self.tool_result,
            is_external=self.is_external,
            side_effect_produced=self.side_effect_produced,
            operation_id=self.operation_id,
            tool_status=self.tool_status
        )


@dataclass
class ExecutionMetadata:
    """Metadata about the execution of a state."""
    cpu_time_ms: float = 0.0
    memory_bytes: int = 0
    logical_clock: int = 0
    wall_time: float = 0.0
    scenario: str = ""
    step_index: int = 0
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
            "logical_clock": self.logical_clock,
            "wall_time": self.wall_time,
            "scenario": self.scenario,
            "step_index": self.step_index,
            "retry_count": self.retry_count,
            "tags": self.tags
        }


class AgentState:
    """
    Represents a single execution state Sᵢ = {Mᵢ, Pᵢ, Tᵢ, Dᵢ, Oᵢ, Xᵢ}

    M = Memory state
    P = Planning state
    T = Tool interaction
    D = Dependency information
    O = Observed output
    X = Execution metadata
    """

    def __init__(
        self,
        state_id: str,
        state_type: StateType = StateType.NORMAL,
        memory: Optional[MemoryState] = None,
        planning: Optional[PlanningState] = None,
        tool: Optional[ToolInteraction] = None,
        output: Any = None,
        metadata: Optional[ExecutionMetadata] = None,
        parent_ids: Optional[List[str]] = None,
        child_ids: Optional[List[str]] = None,
        dependency_ids: Optional[List[str]] = None,
    ):
        self.state_id = state_id
        self.state_type = state_type
        self.memory = memory or MemoryState()
        self.planning = planning or PlanningState()
        self.tool = tool or ToolInteraction()
        self.output = output
        self.metadata = metadata or ExecutionMetadata()
        self.parent_ids: List[str] = parent_ids or []
        self.child_ids: List[str] = child_ids or []
        self.dependency_ids: Set[str] = set(dependency_ids or [])
        self.status: StateStatus = StateStatus.COMPLETED
        self.timestamp: float = time.time()

        # Integrity
        self.integrity_signature: str = ""
        self._compute_signature()

        # Corruption tracking
        self.is_corrupted: bool = False
        self.corruption_type: Optional[str] = None
        self.original_signature: str = self.integrity_signature

    def _compute_signature(self) -> str:
        """Compute a deterministic cryptographic integrity signature for this state."""
        content = {
            "state_id": self.state_id,
            "state_type": self.state_type.value,
            "memory": self.memory.to_dict(),
            "planning": self.planning.to_dict(),
            "tool": self.tool.to_dict(),
            "output": str(self.output),
            "parent_ids": sorted(self.parent_ids),
            "dependency_ids": sorted(list(self.dependency_ids)),
            "metadata_clock": self.metadata.logical_clock,
            "metadata_step": self.metadata.step_index,
        }
        serialized = json.dumps(content, sort_keys=True, default=str)
        self.integrity_signature = hashlib.sha256(serialized.encode()).hexdigest()
        return self.integrity_signature

    def recompute_signature(self) -> str:
        """Recompute signature after state modification."""
        return self._compute_signature()

    def verify_integrity(self) -> bool:
        """Verify that the current state matches its original signature."""
        current_sig = self._compute_signature()
        return current_sig == self.original_signature

    def clone(self) -> "AgentState":
        """Deep clone this state."""
        cloned = AgentState(
            state_id=self.state_id,
            state_type=self.state_type,
            memory=self.memory.clone(),
            planning=self.planning.clone(),
            tool=self.tool.clone(),
            output=self.output,
            metadata=ExecutionMetadata(
                cpu_time_ms=self.metadata.cpu_time_ms,
                memory_bytes=self.metadata.memory_bytes,
                logical_clock=self.metadata.logical_clock,
                wall_time=self.metadata.wall_time,
                scenario=self.metadata.scenario,
                step_index=self.metadata.step_index,
                retry_count=self.metadata.retry_count,
                tags=list(self.metadata.tags),
            ),
            parent_ids=list(self.parent_ids),
            child_ids=list(self.child_ids),
            dependency_ids=list(self.dependency_ids),
        )
        cloned.status = self.status
        cloned.timestamp = self.timestamp
        cloned.integrity_signature = self.integrity_signature
        cloned.original_signature = self.original_signature
        cloned.is_corrupted = self.is_corrupted
        cloned.corruption_type = self.corruption_type
        return cloned

    def to_dict(self) -> Dict:
        return {
            "state_id": self.state_id,
            "state_type": self.state_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "memory": self.memory.to_dict(),
            "planning": self.planning.to_dict(),
            "tool": self.tool.to_dict(),
            "output": str(self.output),
            "metadata": self.metadata.to_dict(),
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "dependency_ids": list(self.dependency_ids),
            "integrity_signature": self.integrity_signature,
            "original_signature": self.original_signature,
            "is_corrupted": self.is_corrupted,
            "corruption_type": self.corruption_type,
        }

    def __repr__(self) -> str:
        return (f"AgentState(id={self.state_id}, type={self.state_type.value}, "
                f"corrupted={self.is_corrupted}, sig={self.integrity_signature[:8]})")
