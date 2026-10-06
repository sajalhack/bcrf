"""
BCRF Experiment - Backward Causal Responsibility (BCR)
Computes BCRᵢ = P(Sᵢ causally contributed to V | E)

BCR is derived from:
- Backward graph traversal from the violation
- Number and length of causal paths
- Dependency strength (edge type weighting)
- Distance from violation
- Whether modifying the state changes the violation outcome
"""

import math
from typing import List, Dict, Set, Optional
from src.agent.state import AgentState
from src.agent.execution_graph import ExecutionGraph


# Edge type weights: how strongly does each edge type imply causation?
EDGE_TYPE_WEIGHTS = {
    "temporal": 0.5,
    "data": 0.9,
    "memory": 0.85,
    "tool": 0.8,
    "causal": 1.0,
}

# Distance decay factor: more distant states have lower BCR
DISTANCE_DECAY = 0.75


def _count_causal_paths(
    graph: ExecutionGraph,
    source_id: str,
    violation_id: str,
) -> int:
    """Count simple paths from source to violation (forward direction)."""
    paths = graph.get_causal_paths(source_id, violation_id)
    return len(paths)


def _compute_dependency_strength(
    graph: ExecutionGraph,
    source_id: str,
    violation_id: str,
) -> float:
    """
    Compute average edge-weight-based dependency strength along paths
    from source to violation.
    """
    paths = graph.get_causal_paths(source_id, violation_id)
    if not paths:
        return 0.0

    path_strengths = []
    for path in paths:
        strength = 1.0
        for i in range(len(path) - 1):
            edge_type = graph.edge_types.get((path[i], path[i + 1]), "temporal")
            weight = EDGE_TYPE_WEIGHTS.get(edge_type, 0.5)
            strength *= weight
        path_strengths.append(strength)

    return sum(path_strengths) / len(path_strengths)


def _compute_counterfactual_impact(
    state: AgentState,
    violation_state: AgentState,
    graph: ExecutionGraph,
) -> float:
    """
    Estimate whether modifying state changes the violation.

    Heuristic: if the state is corrupted, its counterfactual impact is 1.0.
    If the state directly influences the violation state's memory/tool inputs, higher impact.
    """
    # Direct corruption → maximum impact
    if state.is_corrupted:
        return 1.0

    # Direct dependency → high impact
    if state.state_id in violation_state.dependency_ids:
        return 0.85

    # Indirect dependency
    ancestors_of_violation = graph.get_ancestors(violation_state.state_id)
    if state.state_id in ancestors_of_violation:
        return 0.6

    return 0.0


def compute_bcr(
    states: List[AgentState],
    graph: ExecutionGraph,
    corrupted_state_ids: List[str],
    violation_states: Optional[List[AgentState]] = None,
) -> List[Dict]:
    """
    Compute Backward Causal Responsibility for all states.

    BCRᵢ = P(Sᵢ causally contributed to V | E)

    Combines:
    - Causal path count (how many ways does Sᵢ reach the violation)
    - Dependency strength (edge-type weighted path strength)
    - Distance decay (closer to violation = higher BCR)
    - Counterfactual impact (does modifying Sᵢ change the violation?)

    Normalized to [0, 1].

    Returns list of dicts with:
      state_id, backward_distance, dependency_score, causal_path_score, BCR
    """
    # Determine the primary violation state
    if violation_states:
        v_states = violation_states
    else:
        v_states = [s for s in states if s.state_id in corrupted_state_ids]

    if not v_states:
        # No violation identified; all BCR = 0
        return [{"state_id": s.state_id, "backward_distance": -1,
                 "dependency_score": 0.0, "causal_path_score": 0.0, "BCR": 0.0}
                for s in states]

    # Use the first violation state as primary reference
    primary_violation = v_states[0]
    v_id = primary_violation.state_id

    # All ancestors of the violation are candidates
    ancestors = graph.get_ancestors(v_id)
    ancestor_ids = ancestors | {v_id}

    raw_scores: Dict[str, Dict] = {}

    for state in states:
        sid = state.state_id

        if sid == v_id:
            # The violation state itself has BCR = 1.0 (trivially causal)
            raw_scores[sid] = {
                "state_id": sid,
                "backward_distance": 0,
                "dependency_score": 1.0,
                "causal_path_score": 1.0,
                "raw_bcr": 1.0,
            }
            continue

        if sid not in ancestor_ids:
            # Not an ancestor → BCR = 0
            raw_scores[sid] = {
                "state_id": sid,
                "backward_distance": -1,
                "dependency_score": 0.0,
                "causal_path_score": 0.0,
                "raw_bcr": 0.0,
            }
            continue

        # Compute backward distance (hops from this state to violation)
        dist = graph.get_backward_distance(sid, v_id)
        if dist < 0:
            dist = len(states)  # unreachable

        # Distance decay component
        distance_component = DISTANCE_DECAY ** dist

        # Dependency strength component
        dep_strength = _compute_dependency_strength(graph, sid, v_id)

        # Causal path score (normalized by log to avoid extreme values)
        n_paths = _count_causal_paths(graph, sid, v_id)
        path_score = math.log1p(n_paths) / math.log1p(10)  # normalize against 10 paths

        # Counterfactual impact
        cf_impact = _compute_counterfactual_impact(state, primary_violation, graph)

        # Combined raw BCR
        raw_bcr = (
            0.25 * distance_component +
            0.30 * dep_strength +
            0.25 * path_score +
            0.20 * cf_impact
        )

        raw_scores[sid] = {
            "state_id": sid,
            "backward_distance": dist,
            "dependency_score": round(dep_strength, 6),
            "causal_path_score": round(path_score, 6),
            "raw_bcr": round(raw_bcr, 6),
        }

    # Normalize BCR to [0, 1] using max value
    max_bcr = max((v["raw_bcr"] for v in raw_scores.values()), default=1.0)
    if max_bcr == 0:
        max_bcr = 1.0

    results = []
    for state in states:
        sid = state.state_id
        entry = raw_scores[sid]
        bcr = entry["raw_bcr"] / max_bcr
        bcr = round(min(1.0, max(0.0, bcr)), 6)  # clamp to [0, 1]
        results.append({
            "state_id": sid,
            "backward_distance": entry["backward_distance"],
            "dependency_score": entry["dependency_score"],
            "causal_path_score": entry["causal_path_score"],
            "BCR": bcr,
        })

    # Sanity check: BCR must be in [0, 1]
    for r in results:
        assert 0.0 <= r["BCR"] <= 1.0, f"BCR out of range for {r['state_id']}: {r['BCR']}"

    return results
