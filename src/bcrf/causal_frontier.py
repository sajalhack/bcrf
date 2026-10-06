"""
BCRF Experiment - Causal Repair Frontier (CRF)
Constructs CRF = {Sᵢ | FCIᵢ ≥ T_F AND BCRᵢ ≥ T_B}

Supports:
  - fixed threshold
  - percentile-based threshold
  - adaptive threshold (based on distribution shape)

The CRF is allowed to be NON-CONTIGUOUS.
"""

import math
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from src.agent.state import AgentState


def _compute_threshold(values: List[float], method: str, value: float) -> float:
    """
    Compute a threshold given a list of values and a method.

    method:
      "fixed"      → return `value` directly
      "percentile" → return the (value*100)-th percentile of values
      "adaptive"   → use mean + k*std where k is derived from value
    """
    if not values or all(v == 0 for v in values):
        return 0.0

    arr = np.array(values)

    if method == "fixed":
        return float(value)
    elif method == "percentile":
        percentile = float(value) * 100.0  # value in [0,1] → percentile [0,100]
        return float(np.percentile(arr, percentile))
    elif method == "adaptive":
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        # value interpreted as multiplier for std above mean
        return mean + float(value) * std
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def compute_crs(fci_results: List[Dict], bcr_results: List[Dict]) -> List[Dict]:
    """
    Compute Combined Causal Relevance Score:
      CRSᵢ = FCIᵢ × BCRᵢ

    Returns ranked table with:
      state_id, FCI, BCR, CRS, rank
    """
    fci_map = {r["state_id"]: r["FCI"] for r in fci_results}
    bcr_map = {r["state_id"]: r["BCR"] for r in bcr_results}

    all_ids = list(fci_map.keys())
    results = []
    for sid in all_ids:
        fci = fci_map.get(sid, 0.0)
        bcr = bcr_map.get(sid, 0.0)
        crs = fci * bcr
        results.append({
            "state_id": sid,
            "FCI": round(fci, 6),
            "BCR": round(bcr, 6),
            "CRS": round(crs, 6),
        })

    # Rank by CRS descending
    results.sort(key=lambda x: x["CRS"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Sanity check: CRS must be non-negative
    for r in results:
        assert r["CRS"] >= 0.0, f"CRS is negative for {r['state_id']}: {r['CRS']}"

    return results


def construct_crf(
    states: List[AgentState],
    fci_results: List[Dict],
    bcr_results: List[Dict],
    fci_threshold_method: str = "percentile",
    fci_threshold_value: float = 0.6,
    bcr_threshold_method: str = "percentile",
    bcr_threshold_value: float = 0.5,
    expansion_factor: float = 0.0,  # Expand CRF by lowering thresholds by this factor
) -> Tuple[Set[str], Dict, List[Dict]]:
    """
    Construct the Causal Repair Frontier (CRF).

    CRF = {Sᵢ | FCIᵢ ≥ T_F AND BCRᵢ ≥ T_B}

    The CRF is ALLOWED to be non-contiguous.

    Args:
        states: All execution states
        fci_results: FCI computation results
        bcr_results: BCR computation results
        fci_threshold_method: "fixed", "percentile", or "adaptive"
        fci_threshold_value: threshold parameter
        bcr_threshold_method: same
        bcr_threshold_value: same
        expansion_factor: if > 0, lower thresholds by this fraction (for CRF expansion)

    Returns:
        (crf_state_ids, threshold_info, crs_table)
    """
    fci_map = {r["state_id"]: r["FCI"] for r in fci_results}
    bcr_map = {r["state_id"]: r["BCR"] for r in bcr_results}

    fci_values = list(fci_map.values())
    bcr_values = list(bcr_map.values())

    # Compute base thresholds
    T_F_base = _compute_threshold(fci_values, fci_threshold_method, fci_threshold_value)
    T_B_base = _compute_threshold(bcr_values, bcr_threshold_method, bcr_threshold_value)

    # Apply expansion (lower thresholds to include more states)
    T_F = T_F_base * (1.0 - expansion_factor)
    T_B = T_B_base * (1.0 - expansion_factor)

    # Ensure thresholds are non-negative
    T_F = max(0.0, T_F)
    T_B = max(0.0, T_B)

    # Select CRF states
    crf_ids: Set[str] = set()
    crs_table = compute_crs(fci_results, bcr_results)

    for entry in crs_table:
        sid = entry["state_id"]
        fci = fci_map.get(sid, 0.0)
        bcr = bcr_map.get(sid, 0.0)

        selected = (fci >= T_F) and (bcr >= T_B)
        entry["selected"] = selected
        entry["T_F"] = round(T_F, 6)
        entry["T_B"] = round(T_B, 6)

        if selected:
            crf_ids.add(sid)

    threshold_info = {
        "T_F": round(T_F, 6),
        "T_B": round(T_B, 6),
        "T_F_base": round(T_F_base, 6),
        "T_B_base": round(T_B_base, 6),
        "fci_method": fci_threshold_method,
        "bcr_method": bcr_threshold_method,
        "expansion_factor": expansion_factor,
        "crf_size": len(crf_ids),
        "total_states": len(states),
    }

    # Sanity check: CRF states must satisfy thresholds
    for sid in crf_ids:
        fci = fci_map.get(sid, 0.0)
        bcr = bcr_map.get(sid, 0.0)
        assert fci >= T_F, f"CRF state {sid} has FCI={fci} < T_F={T_F}"
        assert bcr >= T_B, f"CRF state {sid} has BCR={bcr} < T_B={T_B}"

    return crf_ids, threshold_info, crs_table


def is_non_contiguous(
    crf_ids: Set[str],
    execution_order: List[str],
) -> bool:
    """
    Check whether the CRF is non-contiguous in the execution order.

    A non-contiguous CRF has gaps — states not in the CRF appear
    between two CRF states in execution order.
    """
    crf_in_order = [sid for sid in execution_order if sid in crf_ids]
    if len(crf_in_order) < 2:
        return False

    for i in range(len(crf_in_order) - 1):
        idx_a = execution_order.index(crf_in_order[i])
        idx_b = execution_order.index(crf_in_order[i + 1])
        if idx_b - idx_a > 1:
            # There's a gap
            return True
    return False


def expand_crf(
    states: List[AgentState],
    fci_results: List[Dict],
    bcr_results: List[Dict],
    current_crf_ids: Set[str],
    fci_threshold_method: str,
    fci_threshold_value: float,
    bcr_threshold_method: str,
    bcr_threshold_value: float,
    expansion_step: float = 0.15,
    current_expansion: float = 0.0,
) -> Tuple[Set[str], Dict, List[Dict]]:
    """
    Expand the CRF by lowering thresholds by expansion_step.
    Used during the CIV failure → expand CRF → retry loop.
    """
    new_expansion = current_expansion + expansion_step
    return construct_crf(
        states, fci_results, bcr_results,
        fci_threshold_method, fci_threshold_value,
        bcr_threshold_method, bcr_threshold_value,
        expansion_factor=new_expansion,
    )
