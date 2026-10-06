"""
BCRF Experiment - Execution Graph Module
Directed state/dependency graph for agent execution.
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any
from src.agent.state import AgentState, StateType


class ExecutionGraph:
    """
    Represents agent execution as a directed state/dependency graph.

    Nodes = execution states
    Edges = dependencies (temporal, data, memory, tool, causal)

    Supports: linear, branching, merging, non-contiguous causal paths.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.states: Dict[str, AgentState] = {}
        self.edge_types: Dict[Tuple[str, str], str] = {}
        self.execution_order: List[str] = []

    def add_state(self, state: AgentState) -> None:
        """Add an execution state to the graph."""
        self.states[state.state_id] = state
        self.graph.add_node(
            state.state_id,
            state_type=state.state_type.value,
            is_corrupted=state.is_corrupted,
            logical_clock=state.metadata.logical_clock,
            is_external=(state.state_type == StateType.EXTERNAL_OPERATION),
        )
        if state.state_id not in self.execution_order:
            self.execution_order.append(state.state_id)

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str = "temporal",
        weight: float = 1.0
    ) -> None:
        """
        Add a directed dependency edge.

        edge_type options:
          temporal    - sequential execution order
          data        - data produced by from_id consumed by to_id
          memory      - memory dependency
          tool        - tool output dependency
          causal      - explicit causal relationship
        """
        self.graph.add_edge(from_id, to_id, edge_type=edge_type, weight=weight)
        self.edge_types[(from_id, to_id)] = edge_type
        # Update state dependency info
        if to_id in self.states:
            self.states[to_id].dependency_ids.add(from_id)
        if from_id in self.states and to_id not in self.states[from_id].child_ids:
            self.states[from_id].child_ids.append(to_id)
        if to_id in self.states and from_id not in self.states[to_id].parent_ids:
            self.states[to_id].parent_ids.append(from_id)

    def get_ancestors(self, state_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """Get all ancestor states (backward traversal)."""
        if state_id not in self.graph:
            return set()
        ancestors = set()
        queue = [state_id]
        depth_map = {state_id: 0}
        while queue:
            current = queue.pop(0)
            for pred in self.graph.predecessors(current):
                if pred not in ancestors:
                    depth = depth_map[current] + 1
                    if max_depth is None or depth <= max_depth:
                        ancestors.add(pred)
                        depth_map[pred] = depth
                        queue.append(pred)
        return ancestors

    def get_descendants(self, state_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """Get all descendant states (forward traversal)."""
        if state_id not in self.graph:
            return set()
        descendants = set()
        queue = [state_id]
        depth_map = {state_id: 0}
        while queue:
            current = queue.pop(0)
            for succ in self.graph.successors(current):
                if succ not in descendants:
                    depth = depth_map[current] + 1
                    if max_depth is None or depth <= max_depth:
                        descendants.add(succ)
                        depth_map[succ] = depth
                        queue.append(succ)
        return descendants

    def get_causal_paths(self, from_id: str, to_id: str) -> List[List[str]]:
        """Find all simple paths from from_id to to_id in the graph."""
        try:
            paths = list(nx.all_simple_paths(self.graph, from_id, to_id))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_backward_distance(self, source_id: str, target_id: str) -> int:
        """
        Get shortest path length (backward) from target to source.
        Returns how many hops source is upstream from target.
        """
        try:
            return nx.shortest_path_length(self.graph, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return -1

    def get_topological_order(self) -> List[str]:
        """Get states in topological order."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            return self.execution_order

    def get_corrupted_states(self) -> List[str]:
        """Return list of state_ids that are marked as corrupted."""
        return [sid for sid, state in self.states.items() if state.is_corrupted]

    def get_external_operation_states(self) -> List[str]:
        """Return state_ids that have external operations."""
        return [
            sid for sid, state in self.states.items()
            if state.state_type == StateType.EXTERNAL_OPERATION
               or state.tool.is_external
        ]

    def get_dependency_subgraph(self, state_ids: Set[str]) -> "ExecutionGraph":
        """Extract a subgraph containing only the given states and their edges."""
        sub = ExecutionGraph()
        for sid in state_ids:
            if sid in self.states:
                sub.add_state(self.states[sid])
        for (u, v), etype in self.edge_types.items():
            if u in state_ids and v in state_ids:
                sub.add_edge(u, v, etype)
        return sub

    def compute_influence_set(self, corrupted_state_id: str) -> Set[str]:
        """
        Compute the set of states whose execution was influenced by the
        corrupted state (all downstream states).
        """
        return self.get_descendants(corrupted_state_id)

    def get_node_count(self) -> int:
        return self.graph.number_of_nodes()

    def get_edge_count(self) -> int:
        return self.graph.number_of_edges()

    def summary(self) -> Dict[str, Any]:
        return {
            "nodes": self.get_node_count(),
            "edges": self.get_edge_count(),
            "corrupted_states": len(self.get_corrupted_states()),
            "external_states": len(self.get_external_operation_states()),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
        }
