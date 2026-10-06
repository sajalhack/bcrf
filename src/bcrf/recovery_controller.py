"""
BCRF Experiment - Recovery Controller
Implements the complete autonomous closed-loop BCRF recovery pipeline:

  Integrity Detection
  → FCI + BCR
  → CRF Construction
  → External Operation Registry Check
  → SRC Reconstruction
  → CIV
  → Resume (if CIV passes) OR Expand CRF + Repair again (if CIV fails)
"""

import time
from typing import List, Dict, Set, Optional, Any, Tuple
from src.agent.state import AgentState
from src.agent.execution_graph import ExecutionGraph
from src.integrity.integrity_monitor import IntegrityMonitor
from src.bcrf.fci import compute_fci
from src.bcrf.bcr import compute_bcr
from src.bcrf.causal_frontier import construct_crf, expand_crf, is_non_contiguous
from src.bcrf.src_manager import SRCManager
from src.bcrf.operation_registry import ExternalOperationRegistry
from src.bcrf.civ import compute_civ, validate_civ


class RecoveryAttempt:
    """Record of a single recovery attempt."""

    def __init__(self, attempt_num: int):
        self.attempt_num = attempt_num
        self.crf_size = 0
        self.reconstructed_states: List[str] = []
        self.skipped_states: List[str] = []
        self.skipped_external_ops: List[str] = []
        self.civ_score = 0.0
        self.civ_components = {}
        self.recovery_time_ms = 0.0
        self.result = "pending"
        self.failure_reason = ""
        self.expansion_factor = 0.0

    def to_dict(self) -> Dict:
        return {
            "attempt_num": self.attempt_num,
            "crf_size": self.crf_size,
            "reconstructed_count": len(self.reconstructed_states),
            "reconstructed_states": self.reconstructed_states,
            "skipped_count": len(self.skipped_states),
            "skipped_states": self.skipped_states,
            "skipped_external_ops": len(self.skipped_external_ops),
            "civ_score": round(self.civ_score, 6),
            "civ_components": self.civ_components,
            "recovery_time_ms": round(self.recovery_time_ms, 3),
            "result": self.result,
            "failure_reason": self.failure_reason,
            "expansion_factor": self.expansion_factor,
        }


class RecoveryResult:
    """Complete result of a BCRF recovery run."""

    def __init__(self):
        self.success = False
        self.total_attempts = 0
        self.total_time_ms = 0.0
        self.final_civ_score = 0.0
        self.total_reconstructed = 0
        self.total_skipped = 0
        self.total_external_ops_prevented = 0
        self.crf_final_size = 0
        self.crf_is_non_contiguous = False
        self.attempts: List[RecoveryAttempt] = []
        self.fci_results: List[Dict] = []
        self.bcr_results: List[Dict] = []
        self.crs_table: List[Dict] = []
        self.threshold_info: Dict = {}
        self.error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "total_attempts": self.total_attempts,
            "total_time_ms": round(self.total_time_ms, 3),
            "final_civ_score": round(self.final_civ_score, 6),
            "total_reconstructed": self.total_reconstructed,
            "total_skipped": self.total_skipped,
            "total_external_ops_prevented": self.total_external_ops_prevented,
            "crf_final_size": self.crf_final_size,
            "crf_is_non_contiguous": self.crf_is_non_contiguous,
            "error": self.error,
        }


class BCRFRecoveryController:
    """
    Complete BCRF autonomous closed-loop recovery controller.

    Orchestrates:
    1. Integrity detection
    2. FCI computation
    3. BCR computation
    4. CRF construction
    5. External Operation Registry check
    6. SRC reconstruction
    7. CIV validation
    8. CRF expansion on failure
    """

    def __init__(
        self,
        graph: ExecutionGraph,
        src_manager: SRCManager,
        registry: ExternalOperationRegistry,
        monitor: IntegrityMonitor,
        fci_threshold_method: str = "percentile",
        fci_threshold_value: float = 0.6,
        bcr_threshold_method: str = "percentile",
        bcr_threshold_value: float = 0.5,
        civ_weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
        civ_threshold: float = 0.65,
        max_iterations: int = 5,
        expansion_factor_per_iter: float = 0.15,
        # Ablation flags
        use_fci: bool = True,
        use_bcr: bool = True,
        use_registry: bool = True,
        use_civ: bool = True,
        use_crf_expansion: bool = True,
    ):
        self.graph = graph
        self.src_manager = src_manager
        self.registry = registry
        self.monitor = monitor
        self.fci_threshold_method = fci_threshold_method
        self.fci_threshold_value = fci_threshold_value
        self.bcr_threshold_method = bcr_threshold_method
        self.bcr_threshold_value = bcr_threshold_value
        self.civ_w1, self.civ_w2, self.civ_w3 = civ_weights
        self.civ_threshold = civ_threshold
        self.max_iterations = max_iterations
        self.expansion_factor_per_iter = expansion_factor_per_iter

        # Ablation flags
        self.use_fci = use_fci
        self.use_bcr = use_bcr
        self.use_registry = use_registry
        self.use_civ = use_civ
        self.use_crf_expansion = use_crf_expansion

    def recover(
        self,
        original_states: List[AgentState],
        corrupted_states: List[AgentState],
        corrupted_state_ids: List[str],
    ) -> RecoveryResult:
        """
        Execute the full BCRF closed-loop recovery.

        Args:
            original_states: Clean pre-corruption states
            corrupted_states: States after fault injection (some corrupted)
            corrupted_state_ids: IDs of the corrupted states

        Returns:
            RecoveryResult with full details of the recovery process
        """
        result = RecoveryResult()
        t_start = time.time()

        try:
            # ── Step 1: Compute FCI ──────────────────────────────────────────
            if self.use_fci:
                fci_results = compute_fci(
                    corrupted_states, self.graph, corrupted_state_ids, mode="remove"
                )
            else:
                # Without FCI: assign uniform FCI=1 to ancestors of violation
                from src.agent.execution_graph import ExecutionGraph
                ancestors = set()
                for cid in corrupted_state_ids:
                    ancestors.update(self.graph.get_ancestors(cid))
                    ancestors.add(cid)
                fci_results = [
                    {
                        "state_id": s.state_id,
                        "observed_value": 0.0,
                        "counterfactual_value": 0.0,
                        "FCI": 1.0 if s.state_id in ancestors else 0.0,
                        "is_candidate": s.state_id in ancestors,
                    }
                    for s in corrupted_states
                ]

            result.fci_results = fci_results

            # ── Step 2: Compute BCR ──────────────────────────────────────────
            if self.use_bcr:
                violation_states = [s for s in corrupted_states if s.state_id in corrupted_state_ids]
                bcr_results = compute_bcr(
                    corrupted_states, self.graph, corrupted_state_ids, violation_states
                )
            else:
                # Without BCR: assign uniform BCR=1 to ancestors of violation
                ancestors = set()
                for cid in corrupted_state_ids:
                    ancestors.update(self.graph.get_ancestors(cid))
                    ancestors.add(cid)
                bcr_results = [
                    {
                        "state_id": s.state_id,
                        "backward_distance": 0,
                        "dependency_score": 1.0,
                        "causal_path_score": 1.0,
                        "BCR": 1.0 if s.state_id in ancestors else 0.0,
                    }
                    for s in corrupted_states
                ]

            result.bcr_results = bcr_results

            # Precompute SRC signatures for CIV
            src_signatures = {
                sid: capsule.original_integrity_signature
                for sid, capsule in self.src_manager.capsules.items()
            }

            # ── Recovery Loop ─────────────────────────────────────────────────
            current_expansion = 0.0
            current_crf_ids: Set[str] = set()
            crs_table: List[Dict] = []

            for iteration in range(self.max_iterations):
                attempt = RecoveryAttempt(iteration + 1)
                t_attempt = time.time()

                # ── Step 3: Construct CRF ────────────────────────────────────
                current_crf_ids, threshold_info, crs_table = construct_crf(
                    corrupted_states,
                    fci_results,
                    bcr_results,
                    self.fci_threshold_method,
                    self.fci_threshold_value,
                    self.bcr_threshold_method,
                    self.bcr_threshold_value,
                    expansion_factor=current_expansion,
                )
                attempt.crf_size = len(current_crf_ids)
                attempt.expansion_factor = current_expansion
                result.threshold_info = threshold_info

                # Ensure corrupted states are in CRF
                for cid in corrupted_state_ids:
                    current_crf_ids.add(cid)

                # ── Step 4: External Operation Registry Check ─────────────────
                external_ops_states = [
                    sid for sid in current_crf_ids
                    if sid in self.graph.states
                    and (self.graph.states[sid].tool.side_effect_produced
                         or self.graph.states[sid].tool.is_external)
                ]

                skipped_ops = []
                if self.use_registry:
                    for sid in external_ops_states:
                        state = self.graph.states[sid]
                        existing = self.registry.check_already_completed(
                            state.tool.operation_id,
                            state.tool.tool_name,
                            sid,
                            state.tool.tool_args,
                        )
                        if existing is not None:
                            self.registry.prevent_replay(
                                existing.operation_id, "recovery_skip"
                            )
                            skipped_ops.append(existing.operation_id)

                attempt.skipped_external_ops = skipped_ops

                # ── Step 5: SRC Reconstruction ────────────────────────────────
                reconstructed_map = self.src_manager.reconstruct_crf_states(current_crf_ids)

                successfully_reconstructed = [
                    sid for sid, s in reconstructed_map.items() if s is not None
                ]
                failed_reconstruction = [
                    sid for sid, s in reconstructed_map.items() if s is None
                ]

                attempt.reconstructed_states = successfully_reconstructed
                attempt.skipped_states = failed_reconstruction

                # ── Step 6: CIV Validation ────────────────────────────────────
                if self.use_civ:
                    valid_reconstructed = {
                        sid: s for sid, s in reconstructed_map.items() if s is not None
                    }
                    civ_result = compute_civ(
                        original_states=original_states,
                        reconstructed_states=valid_reconstructed,
                        graph=self.graph,
                        corrupted_state_ids=set(corrupted_state_ids),
                        src_signatures=src_signatures,
                        w1=self.civ_w1,
                        w2=self.civ_w2,
                        w3=self.civ_w3,
                    )
                    civ_score = civ_result["CIV"]
                    attempt.civ_score = civ_score
                    attempt.civ_components = civ_result
                    civ_pass = validate_civ(civ_score, self.civ_threshold)
                else:
                    # Without CIV: always pass after first reconstruction
                    civ_score = 1.0
                    attempt.civ_score = 1.0
                    attempt.civ_components = {"CIV": 1.0}
                    civ_pass = True

                attempt.recovery_time_ms = (time.time() - t_attempt) * 1000

                if civ_pass:
                    attempt.result = "success"
                    result.attempts.append(attempt)
                    # Recovery succeeded
                    result.success = True
                    result.total_attempts = iteration + 1
                    result.final_civ_score = civ_score
                    result.total_reconstructed = len(successfully_reconstructed)
                    result.total_skipped = len(failed_reconstruction)
                    result.total_external_ops_prevented = len(skipped_ops)
                    result.crf_final_size = len(current_crf_ids)
                    result.crs_table = crs_table
                    result.crf_is_non_contiguous = is_non_contiguous(
                        current_crf_ids, self.graph.execution_order
                    )
                    break
                else:
                    attempt.result = "civ_fail"
                    attempt.failure_reason = f"CIV={civ_score:.4f} < threshold={self.civ_threshold}"
                    result.attempts.append(attempt)

                    if not self.use_crf_expansion or iteration >= self.max_iterations - 1:
                        # Cannot expand further
                        result.success = False
                        result.total_attempts = iteration + 1
                        result.final_civ_score = civ_score
                        result.total_reconstructed = len(successfully_reconstructed)
                        result.total_skipped = len(failed_reconstruction)
                        result.total_external_ops_prevented = len(skipped_ops)
                        result.crf_final_size = len(current_crf_ids)
                        result.crs_table = crs_table
                        result.crf_is_non_contiguous = is_non_contiguous(
                            current_crf_ids, self.graph.execution_order
                        )
                        break

                    # ── Expand CRF and retry ─────────────────────────────────
                    current_expansion += self.expansion_factor_per_iter

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.total_time_ms = (time.time() - t_start) * 1000
        return result
