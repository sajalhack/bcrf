"""
BCRF Experiment - Forward Counterfactual Influence (FCI)
Computes FCIᵢ = |V(Sᵢ) − V(Sᵢᶜᶠ)| for each candidate state.

The counterfactual modification actually simulates the execution with
the state removed/replaced and recomputes the downstream execution value.
"""

import copy
import math
from typing import List, Dict, Tuple, Optional
from src.agent.state import AgentState
from src.agent.execution_graph import ExecutionGraph


def _compute_state_value(state: AgentState) -> float:
    """
    Compute the execution outcome value V(Sᵢ) associated with a state.

    This combines:
    - Output numerical value
    - Memory coherence (number of valid memory entries)
    - Planning confidence
    - Tool result quality
    - Whether the state has side effects
    """
    value = 0.0

    # Output contribution
    if isinstance(state.output, dict):
        raw_val = state.output.get("value", 0.0)
        if isinstance(raw_val, (int, float)) and math.isfinite(raw_val):
            value += float(raw_val) * 0.4
    elif isinstance(state.output, (int, float)) and math.isfinite(float(state.output)):
        value += float(state.output) * 0.4

    # Memory coherence
    valid_mem = sum(
        1 for v in state.memory.working_memory.values()
        if v is not None and str(v) != "CORRUPTED"
           and not str(v).startswith("CORRUPTED_")
    )
    value += valid_mem * 3.0

    # Planning confidence
    if math.isfinite(state.planning.confidence):
        value += state.planning.confidence * 20.0

    # Tool result quality
    if state.tool.tool_result is not None:
        if isinstance(state.tool.tool_result, dict):
            # Penalize corrupted tool results
            if state.tool.tool_result.get("_corrupted", False):
                value -= 30.0
            else:
                value += 8.0
        else:
            value += 5.0

    # Side-effect states have higher importance
    if state.tool.side_effect_produced:
        value += 12.0

    # Integrity bonus
    if not state.is_corrupted:
        value += 10.0

    return round(value, 6)


def _compute_counterfactual_value(
    state: AgentState,
    all_states: List[AgentState],
    graph: ExecutionGraph,
    mode: str = "remove",
) -> float:
    """
    Compute V(Sᵢᶜᶠ) by simulating counterfactual removal/replacement.

    mode options:
      "remove"   - simulate removing the state and propagating neutral values downstream
      "baseline" - replace state output with a fixed baseline value
      "perturb"  - replace output with a perturbed (zeroed) version
    """
    if mode == "remove":
        # Downstream states that depended on this state now lose its contribution
        descendants = graph.get_descendants(state.state_id)
        cf_value = 0.0

        # Simulate: downstream states lose the value this state contributed
        for sid in descendants:
            dep_state = graph.states.get(sid)
            if dep_state is not None and state.state_id in dep_state.dependency_ids:
                # Each direct dependent loses a portion proportional to its dependency
                cf_value -= _compute_state_value(dep_state) * 0.15

        # The state itself contributes zero in the counterfactual
        return round(cf_value, 6)

    elif mode == "baseline":
        # Replace with a fixed neutral baseline (mean state value)
        all_values = [_compute_state_value(s) for s in all_states]
        mean_val = sum(all_values) / max(len(all_values), 1)
        return round(mean_val, 6)

    elif mode == "perturb":
        # Zero out the output and recompute
        cf_value = 0.0
        # Memory and planning parts remain; output zeroed
        valid_mem = sum(
            1 for v in state.memory.working_memory.values()
            if v is not None and not str(v).startswith("CORRUPTED")
        )
        cf_value += valid_mem * 3.0
        cf_value += state.planning.confidence * 20.0
        # No output, no tool result contribution
        return round(cf_value, 6)

    else:
        raise ValueError(f"Unknown counterfactual mode: {mode}")


def compute_fci(
    states: List[AgentState],
    graph: ExecutionGraph,
    corrupted_state_ids: List[str],
    mode: str = "remove",
) -> List[Dict]:
    """
    Compute Forward Counterfactual Influence for all states.

    FCIᵢ = |V(Sᵢ) − V(Sᵢᶜᶠ)|

    Only states that are ancestors of (or equal to) the corrupted state
    are considered as candidates for the CRF.

    Returns a list of dicts with:
      state_id, observed_value, counterfactual_value, FCI
    """
    # Determine candidate states: all states that causally precede any corruption
    candidate_ids = set()
    for cid in corrupted_state_ids:
        candidate_ids.add(cid)
        candidate_ids.update(graph.get_ancestors(cid))

    results = []
    for state in states:
        if state.state_id not in candidate_ids:
            # State has no causal connection to the violation — assign FCI=0
            results.append({
                "state_id": state.state_id,
                "observed_value": _compute_state_value(state),
                "counterfactual_value": _compute_state_value(state),
                "FCI": 0.0,
                "is_candidate": False,
            })
            continue

        observed = _compute_state_value(state)
        cf_value = _compute_counterfactual_value(state, states, graph, mode)
        fci = abs(observed - cf_value)

        results.append({
            "state_id": state.state_id,
            "observed_value": observed,
            "counterfactual_value": cf_value,
            "FCI": round(fci, 6),
            "is_candidate": True,
        })

    # Sanity check: FCI must be non-negative
    for r in results:
        assert r["FCI"] >= 0.0, f"FCI is negative for state {r['state_id']}: {r['FCI']}"

    return results
