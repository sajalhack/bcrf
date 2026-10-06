"""
BCRF Experiment - Counterfactual Integrity Validation (CIV)
Implements CIV = w₁O + w₂I + w₃C

O = outcome consistency
I = reconstructed-state integrity consistency
C = causal consistency

All weights and threshold are configurable.
"""

import math
from typing import List, Dict, Set, Optional, Tuple
from src.agent.state import AgentState
from src.agent.execution_graph import ExecutionGraph


def compute_outcome_consistency(
    original_states: List[AgentState],
    reconstructed_states: Dict[str, AgentState],
    all_states: List[AgentState],
) -> float:
    """
    O = outcome consistency.

    Measures how well the reconstructed states restore the original
    execution outcomes. Compares outputs of reconstructed states to
    their original (pre-corruption) values.
    """
    if not reconstructed_states:
        return 0.0

    scores = []
    for state_id, reconstructed in reconstructed_states.items():
        if reconstructed is None:
            scores.append(0.0)
            continue

        # Find original state
        original = next((s for s in original_states if s.state_id == state_id), None)
        if original is None:
            scores.append(0.5)  # Unknown original, neutral score
            continue

        # Compare output values
        orig_out = original.output
        rec_out = reconstructed.output

        if orig_out == rec_out:
            scores.append(1.0)
        elif isinstance(orig_out, dict) and isinstance(rec_out, dict):
            # Compare overlapping keys
            common_keys = set(orig_out.keys()) & set(rec_out.keys())
            if not common_keys:
                scores.append(0.5)
            else:
                matches = sum(1 for k in common_keys if orig_out[k] == rec_out[k])
                scores.append(matches / len(common_keys))
        elif isinstance(orig_out, (int, float)) and isinstance(rec_out, (int, float)):
            if orig_out == 0 and rec_out == 0:
                scores.append(1.0)
            else:
                denom = max(abs(orig_out), abs(rec_out), 1e-9)
                diff = abs(orig_out - rec_out) / denom
                scores.append(max(0.0, 1.0 - diff))
        else:
            scores.append(0.0 if orig_out != rec_out else 1.0)

    return round(sum(scores) / len(scores), 6) if scores else 0.0


def compute_integrity_consistency(
    reconstructed_states: Dict[str, AgentState],
    src_signatures: Dict[str, str],
) -> float:
    """
    I = integrity consistency.

    Verifies that reconstructed states match their original integrity signatures.
    """
    if not reconstructed_states:
        return 0.0

    scores = []
    for state_id, reconstructed in reconstructed_states.items():
        if reconstructed is None:
            scores.append(0.0)
            continue

        expected_sig = src_signatures.get(state_id, reconstructed.original_signature)
        current_sig = reconstructed._compute_signature()

        # After reconstruction, the state should match the original signature
        # We check if the state content is consistent with the original
        if current_sig == expected_sig or reconstructed.integrity_signature == expected_sig:
            scores.append(1.0)
        else:
            # Partial match: check if at least corruption is cleared
            if not reconstructed.is_corrupted:
                scores.append(0.7)
            else:
                scores.append(0.0)

    return round(sum(scores) / len(scores), 6) if scores else 0.0


def compute_causal_consistency(
    reconstructed_states: Dict[str, AgentState],
    graph: ExecutionGraph,
    corrupted_state_ids: Set[str],
) -> float:
    """
    C = causal consistency.

    Verifies that the reconstructed states causally resolve the violation:
    - All corrupted states in the CRF have been reconstructed
    - Dependencies are satisfied
    """
    if not reconstructed_states:
        return 0.0

    # Check how many originally corrupted states are now reconstructed
    covered_corruptions = sum(
        1 for cid in corrupted_state_ids
        if cid in reconstructed_states and reconstructed_states[cid] is not None
    )
    coverage_score = covered_corruptions / max(len(corrupted_state_ids), 1)

    # Check dependency satisfaction
    dep_scores = []
    for state_id, reconstructed in reconstructed_states.items():
        if reconstructed is None:
            dep_scores.append(0.0)
            continue
        # Check that dependencies of this state were either:
        # (a) also reconstructed, or (b) not corrupted
        dep_ids = reconstructed.dependency_ids
        if not dep_ids:
            dep_scores.append(1.0)
            continue
        satisfied = 0
        for dep_id in dep_ids:
            if dep_id in reconstructed_states and reconstructed_states[dep_id] is not None:
                satisfied += 1
            elif dep_id in graph.states and not graph.states[dep_id].is_corrupted:
                satisfied += 1
        dep_scores.append(satisfied / len(dep_ids))

    dep_score = sum(dep_scores) / len(dep_scores) if dep_scores else 1.0

    return round(0.6 * coverage_score + 0.4 * dep_score, 6)


def compute_civ(
    original_states: List[AgentState],
    reconstructed_states: Dict[str, AgentState],
    graph: ExecutionGraph,
    corrupted_state_ids: Set[str],
    src_signatures: Dict[str, str],
    w1: float = 0.4,
    w2: float = 0.35,
    w3: float = 0.25,
) -> Dict:
    """
    Compute Counterfactual Integrity Validation score.

    CIV = w₁O + w₂I + w₃C

    Returns dict with O, I, C, CIV, and component details.
    """
    # Validate weights sum to ~1
    assert abs(w1 + w2 + w3 - 1.0) < 1e-6, \
        f"CIV weights must sum to 1.0, got {w1+w2+w3}"

    O = compute_outcome_consistency(original_states, reconstructed_states, original_states)
    I = compute_integrity_consistency(reconstructed_states, src_signatures)
    C = compute_causal_consistency(reconstructed_states, graph, corrupted_state_ids)

    civ = w1 * O + w2 * I + w3 * C

    return {
        "outcome_consistency": round(O, 6),
        "integrity_consistency": round(I, 6),
        "causal_consistency": round(C, 6),
        "CIV": round(civ, 6),
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "n_reconstructed": sum(1 for v in reconstructed_states.values() if v is not None),
        "n_corrupted": len(corrupted_state_ids),
    }


def validate_civ(civ_score: float, threshold: float = 0.65) -> bool:
    """
    Validate whether CIV passes the threshold.

    Returns True if CIV ≥ threshold (RESUME execution).
    Returns False if CIV < threshold (EXPAND CRF and repair again).
    """
    return civ_score >= threshold
