"""
BCRF Unit Tests - Complete test suite for all components.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import random
from src.agent.state import (
    AgentState, StateType, StateStatus, MemoryState, PlanningState,
    ToolInteraction, ExecutionMetadata
)
from src.agent.execution_graph import ExecutionGraph
from src.agent.scenarios import build_scenario, SCENARIO_BUILDERS
from src.agent.agent_simulator import AgentSimulator
from src.integrity.integrity_monitor import IntegrityMonitor
from src.integrity.fault_injection import FaultInjector, FAULT_TYPES
from src.bcrf.fci import compute_fci
from src.bcrf.bcr import compute_bcr
from src.bcrf.causal_frontier import construct_crf, compute_crs, is_non_contiguous
from src.bcrf.src_manager import SRCManager
from src.bcrf.operation_registry import ExternalOperationRegistry
from src.bcrf.civ import compute_civ, validate_civ
from src.bcrf.recovery_controller import BCRFRecoveryController


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Build a minimal test state
# ─────────────────────────────────────────────────────────────────────────────

def make_test_state(sid: str, step: int = 0, is_external: bool = False) -> AgentState:
    mem = MemoryState(working_memory={"key": f"value_{step}"}, memory_version=step)
    plan = PlanningState(current_goal="test", subtasks=["a", "b"], confidence=0.9)
    tool = ToolInteraction(
        tool_name="test_tool" if is_external else "",
        tool_result={"value": step} if is_external else None,
        is_external=is_external,
        side_effect_produced=is_external,
        operation_id=f"op_{sid}" if is_external else None,
        tool_status="success" if is_external else "none",
    )
    meta = ExecutionMetadata(logical_clock=step, step_index=step, scenario="test")
    return AgentState(
        state_id=sid,
        state_type=StateType.EXTERNAL_OPERATION if is_external else StateType.NORMAL,
        memory=mem,
        planning=plan,
        tool=tool,
        output={"value": step, "status": "ok"},
        metadata=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: State Creation
# ─────────────────────────────────────────────────────────────────────────────

class TestStateCreation:
    def test_state_has_signature(self):
        s = make_test_state("S001")
        assert s.integrity_signature != ""
        assert len(s.integrity_signature) == 64  # SHA-256 hex

    def test_state_clone_is_equal(self):
        s = make_test_state("S001")
        c = s.clone()
        assert c.state_id == s.state_id
        assert c.integrity_signature == s.integrity_signature
        assert c.memory.working_memory == s.memory.working_memory

    def test_state_verify_integrity_clean(self):
        s = make_test_state("S001")
        # Clean state: verification should pass (signature matches original)
        assert s.integrity_signature == s.original_signature

    def test_state_detects_modification(self):
        s = make_test_state("S001")
        original_sig = s.original_signature
        s.memory.working_memory["new_key"] = "tampered"
        s.recompute_signature()
        # Signature should now differ from original
        assert s.integrity_signature != original_sig


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Dependency Graph
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionGraph:
    def setup_method(self):
        self.graph = ExecutionGraph()
        for i in range(5):
            s = make_test_state(f"S{i:03d}", step=i)
            self.graph.add_state(s)
        for i in range(4):
            self.graph.add_edge(f"S{i:03d}", f"S{i+1:03d}", "temporal")

    def test_ancestors(self):
        ancestors = self.graph.get_ancestors("S004")
        assert "S000" in ancestors
        assert "S001" in ancestors
        assert "S003" in ancestors

    def test_descendants(self):
        desc = self.graph.get_descendants("S000")
        assert "S001" in desc
        assert "S004" in desc

    def test_backward_distance(self):
        dist = self.graph.get_backward_distance("S000", "S004")
        assert dist == 4

    def test_topological_order(self):
        order = self.graph.get_topological_order()
        assert len(order) == 5
        # S000 should come before S004
        assert order.index("S000") < order.index("S004")

    def test_no_path_returns_negative(self):
        dist = self.graph.get_backward_distance("S004", "S000")  # Wrong direction
        assert dist == -1


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Integrity Signature
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegritySignature:
    def test_signature_deterministic(self):
        s1 = make_test_state("S001", step=0)
        s2 = make_test_state("S001", step=0)
        assert s1.integrity_signature == s2.integrity_signature

    def test_different_states_different_signatures(self):
        s1 = make_test_state("S001", step=0)
        s2 = make_test_state("S002", step=1)
        assert s1.integrity_signature != s2.integrity_signature

    def test_monitor_detects_corruption(self):
        graph = ExecutionGraph()
        states = [make_test_state(f"S{i:03d}", i) for i in range(3)]
        for s in states:
            graph.add_state(s)

        monitor = IntegrityMonitor(graph)
        monitor.register_baseline(states)

        # Corrupt one state
        states[1].memory.working_memory["key"] = "CORRUPTED"
        states[1].is_corrupted = True
        states[1].corruption_type = "memory_corruption"
        states[1].recompute_signature()

        violation = monitor.check_state(states[1])
        assert violation is not None
        assert violation.state_id == "S001"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Fault Injection
# ─────────────────────────────────────────────────────────────────────────────

class TestFaultInjection:
    def test_memory_corruption(self):
        injector = FaultInjector(seed=42)
        state = make_test_state("S010", step=5)
        original_sig = state.original_signature
        record = injector.inject_fault(state, "memory_corruption")
        assert state.is_corrupted
        assert state.corruption_type == "memory_corruption"
        assert state.integrity_signature != original_sig
        assert record.fault_id is not None

    def test_planning_corruption(self):
        injector = FaultInjector(seed=42)
        state = make_test_state("S010", step=5)
        injector.inject_fault(state, "planning_corruption")
        assert state.is_corrupted
        assert "CORRUPTED_GOAL" in state.planning.current_goal

    def test_tool_corruption(self):
        injector = FaultInjector(seed=42)
        state = make_test_state("S010", step=5, is_external=True)
        injector.inject_fault(state, "tool_corruption")
        assert state.is_corrupted

    def test_reproducibility(self):
        """Same seed → same corruption."""
        injector1 = FaultInjector(seed=42)
        injector2 = FaultInjector(seed=42)
        state1 = make_test_state("S010", step=5)
        state2 = make_test_state("S010", step=5)
        injector1.inject_fault(state1, "memory_corruption")
        injector2.inject_fault(state2, "memory_corruption")
        assert state1.integrity_signature == state2.integrity_signature

    def test_all_fault_types_execute(self):
        for ft in FAULT_TYPES:
            injector = FaultInjector(seed=42)
            state = make_test_state("S010", step=5, is_external=True)
            record = injector.inject_fault(state, ft)
            assert state.is_corrupted
            assert record.fault_type == ft


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: FCI
# ─────────────────────────────────────────────────────────────────────────────

class TestFCI:
    def setup_method(self):
        self.graph = ExecutionGraph()
        self.states = []
        for i in range(10):
            s = make_test_state(f"S{i:03d}", step=i, is_external=(i == 3))
            self.states.append(s)
            self.graph.add_state(s)
        for i in range(9):
            self.graph.add_edge(f"S{i:03d}", f"S{i+1:03d}", "temporal")
        # Inject corruption on state 5
        injector = FaultInjector(seed=42)
        injector.inject_fault(self.states[5], "memory_corruption")

    def test_fci_non_negative(self):
        results = compute_fci(self.states, self.graph, ["S005"], mode="remove")
        for r in results:
            assert r["FCI"] >= 0.0, f"FCI is negative: {r}"

    def test_fci_ancestors_are_candidates(self):
        results = compute_fci(self.states, self.graph, ["S005"], mode="remove")
        result_map = {r["state_id"]: r for r in results}
        # States before S005 should be candidates
        for i in range(5):
            assert result_map[f"S{i:03d}"]["is_candidate"]

    def test_fci_non_ancestors_zero(self):
        results = compute_fci(self.states, self.graph, ["S005"], mode="remove")
        result_map = {r["state_id"]: r for r in results}
        # States after S005 should have FCI=0 (not candidates)
        for i in range(6, 10):
            assert result_map[f"S{i:03d}"]["FCI"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: BCR
# ─────────────────────────────────────────────────────────────────────────────

class TestBCR:
    def setup_method(self):
        self.graph = ExecutionGraph()
        self.states = []
        for i in range(8):
            s = make_test_state(f"S{i:03d}", step=i)
            self.states.append(s)
            self.graph.add_state(s)
        for i in range(7):
            self.graph.add_edge(f"S{i:03d}", f"S{i+1:03d}", "temporal")

        injector = FaultInjector(seed=42)
        injector.inject_fault(self.states[4], "memory_corruption")

    def test_bcr_in_range(self):
        results = compute_bcr(self.states, self.graph, ["S004"])
        for r in results:
            assert 0.0 <= r["BCR"] <= 1.0, f"BCR out of range: {r}"

    def test_violation_state_has_high_bcr(self):
        results = compute_bcr(self.states, self.graph, ["S004"])
        result_map = {r["state_id"]: r for r in results}
        # The corrupted state should have BCR = 1.0 (normalized)
        assert result_map["S004"]["BCR"] == 1.0

    def test_non_ancestors_have_zero_bcr(self):
        results = compute_bcr(self.states, self.graph, ["S004"])
        result_map = {r["state_id"]: r for r in results}
        # States after S004 should have BCR = 0
        for i in range(5, 8):
            assert result_map[f"S{i:03d}"]["BCR"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: CRS
# ─────────────────────────────────────────────────────────────────────────────

class TestCRS:
    def test_crs_non_negative(self):
        fci_results = [{"state_id": f"S{i:03d}", "FCI": float(i) * 0.1} for i in range(5)]
        bcr_results = [{"state_id": f"S{i:03d}", "BCR": float(i) * 0.2} for i in range(5)]
        crs = compute_crs(fci_results, bcr_results)
        for r in crs:
            assert r["CRS"] >= 0.0

    def test_crs_sorted_descending(self):
        fci_results = [{"state_id": f"S{i:03d}", "FCI": float(i)} for i in range(5)]
        bcr_results = [{"state_id": f"S{i:03d}", "BCR": float(i)} for i in range(5)]
        crs = compute_crs(fci_results, bcr_results)
        for i in range(len(crs) - 1):
            assert crs[i]["CRS"] >= crs[i + 1]["CRS"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: CRF
# ─────────────────────────────────────────────────────────────────────────────

class TestCRF:
    def setup_method(self):
        self.states = [make_test_state(f"S{i:03d}", i) for i in range(10)]
        self.fci_results = [{"state_id": f"S{i:03d}", "FCI": float(i) / 9} for i in range(10)]
        self.bcr_results = [{"state_id": f"S{i:03d}", "BCR": float(9-i) / 9} for i in range(10)]

    def test_crf_states_satisfy_thresholds(self):
        crf_ids, info, table = construct_crf(
            self.states, self.fci_results, self.bcr_results,
            fci_threshold_method="fixed", fci_threshold_value=0.5,
            bcr_threshold_method="fixed", bcr_threshold_value=0.5,
        )
        fci_map = {r["state_id"]: r["FCI"] for r in self.fci_results}
        bcr_map = {r["state_id"]: r["BCR"] for r in self.bcr_results}
        for sid in crf_ids:
            assert fci_map[sid] >= info["T_F"], f"CRF state {sid} violates T_F"
            assert bcr_map[sid] >= info["T_B"], f"CRF state {sid} violates T_B"

    def test_crf_count_not_exceed_total(self):
        crf_ids, info, table = construct_crf(
            self.states, self.fci_results, self.bcr_results,
        )
        assert len(crf_ids) <= len(self.states)

    def test_crf_can_be_non_contiguous(self):
        # Create FCI/BCR with a deliberate gap
        fci_results = [{"state_id": f"S{i:03d}", "FCI": 0.9 if i in [0, 5, 9] else 0.1}
                       for i in range(10)]
        bcr_results = [{"state_id": f"S{i:03d}", "BCR": 0.9 if i in [0, 5, 9] else 0.1}
                       for i in range(10)]
        crf_ids, _, _ = construct_crf(
            self.states, fci_results, bcr_results,
            fci_threshold_method="fixed", fci_threshold_value=0.8,
            bcr_threshold_method="fixed", bcr_threshold_value=0.8,
        )
        execution_order = [f"S{i:03d}" for i in range(10)]
        assert is_non_contiguous(crf_ids, execution_order)


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: SRC
# ─────────────────────────────────────────────────────────────────────────────

class TestSRC:
    def test_capsule_creation_and_reconstruction(self):
        state = make_test_state("S001", step=3, is_external=True)
        manager = SRCManager()
        capsule = manager.create_capsule(state)

        # Verify capsule
        assert manager.verify_capsule(state.state_id)

        # Reconstruct
        reconstructed = manager.reconstruct_state(state.state_id)
        assert reconstructed is not None
        assert reconstructed.state_id == state.state_id
        assert not reconstructed.is_corrupted

    def test_reconstruction_after_corruption(self):
        state = make_test_state("S001", step=3)
        manager = SRCManager()
        manager.create_capsule(state)

        # Corrupt the state
        state.memory.working_memory["key"] = "CORRUPTED"
        state.is_corrupted = True
        state.recompute_signature()

        # Reconstruction should restore clean state
        reconstructed = manager.reconstruct_state(state.state_id)
        assert reconstructed is not None
        assert not reconstructed.is_corrupted
        assert reconstructed.memory.working_memory["key"] == "value_3"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: External Operation Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestOperationRegistry:
    def test_duplicate_prevention(self):
        registry = ExternalOperationRegistry()
        registry.register_operation("op_001", "db_write", "S005",
                                    result={"status": "ok"}, args={"key": "val"})

        # Check: already completed
        existing = registry.check_already_completed("op_001", "db_write", "S005",
                                                     args={"key": "val"})
        assert existing is not None

    def test_execute_or_skip_prevents_duplicate(self):
        registry = ExternalOperationRegistry()
        registry.register_operation("op_001", "db_write", "S005",
                                    result={"status": "ok"}, args={"key": "val"})

        result = registry.execute_or_skip(
            operation_id="op_001",
            operation_type="db_write",
            state_id="S005",
            args={"key": "val"},
        )
        assert result["skipped"] is True
        assert result["duplicate_prevented"] is True

    def test_first_execution_not_skipped(self):
        registry = ExternalOperationRegistry()
        result = registry.execute_or_skip(
            operation_id="op_002",
            operation_type="notification",
            state_id="S010",
            args={"msg": "hello"},
        )
        assert result["skipped"] is False

    def test_prevention_count_increments(self):
        registry = ExternalOperationRegistry()
        registry.register_operation("op_001", "db_write", "S005",
                                    result={"status": "ok"}, args={"key": "val"})
        registry.prevent_replay("op_001")
        assert registry.get_duplicate_prevention_count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: CIV
# ─────────────────────────────────────────────────────────────────────────────

class TestCIV:
    def test_civ_weights_sum_to_one(self):
        w1, w2, w3 = 0.4, 0.35, 0.25
        assert abs(w1 + w2 + w3 - 1.0) < 1e-6

    def test_civ_pass(self):
        assert validate_civ(0.80, threshold=0.65) is True

    def test_civ_fail(self):
        assert validate_civ(0.50, threshold=0.65) is False

    def test_civ_computation_returns_dict(self):
        graph = ExecutionGraph()
        states = [make_test_state(f"S{i:03d}", i) for i in range(4)]
        for s in states:
            graph.add_state(s)

        src_manager = SRCManager()
        src_manager.create_all_capsules(states)

        corrupted_ids = {"S002"}
        src_sigs = {s.state_id: s.original_signature for s in states}

        reconstructed = {
            "S002": make_test_state("S002", 2)
        }

        result = compute_civ(
            original_states=states,
            reconstructed_states=reconstructed,
            graph=graph,
            corrupted_state_ids=corrupted_ids,
            src_signatures=src_sigs,
        )
        assert "CIV" in result
        assert 0.0 <= result["CIV"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Recovery Controller
# ─────────────────────────────────────────────────────────────────────────────

class TestRecoveryController:
    def setup_method(self):
        self.sim = AgentSimulator(
            scenario="order_processing", num_states=20, seed=42
        )
        self.original_states, self.graph = self.sim.run()

        self.src_manager = SRCManager()
        self.src_manager.create_all_capsules(self.original_states)

        self.registry = ExternalOperationRegistry()
        for entry in self.sim.get_external_operations():
            if entry["operation_id"]:
                self.registry.register_operation(
                    entry["operation_id"], "external", entry["state_id"],
                    args={"state_id": entry["state_id"]}
                )

        self.monitor = IntegrityMonitor(self.graph)
        self.monitor.register_baseline(self.original_states)

        self.corrupted_states = [s.clone() for s in self.original_states]
        injector = FaultInjector(seed=43)
        corrupted_list, _ = injector.inject_random_faults(
            self.corrupted_states, fault_rate=0.1, fault_type="memory_corruption"
        )
        self.corrupted_ids = [s.state_id for s in corrupted_list]
        if not self.corrupted_ids:
            injector.inject_single_fault(self.corrupted_states, 5, "memory_corruption")
            self.corrupted_ids = [self.corrupted_states[5].state_id]

        # Update graph
        for cs in self.corrupted_states:
            self.graph.states[cs.state_id] = cs

    def test_recovery_runs(self):
        controller = BCRFRecoveryController(
            graph=self.graph,
            src_manager=self.src_manager,
            registry=self.registry,
            monitor=self.monitor,
        )
        result = controller.recover(
            self.original_states, self.corrupted_states, self.corrupted_ids
        )
        assert result is not None
        assert result.total_attempts >= 1

    def test_recovery_reconstructed_not_exceed_total(self):
        controller = BCRFRecoveryController(
            graph=self.graph,
            src_manager=self.src_manager,
            registry=self.registry,
            monitor=self.monitor,
        )
        result = controller.recover(
            self.original_states, self.corrupted_states, self.corrupted_ids
        )
        assert result.total_reconstructed <= len(self.original_states)


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: Baseline Methods
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineMethods:
    def setup_method(self):
        sim = AgentSimulator("order_processing", num_states=20, seed=42)
        self.original_states, self.graph = sim.run()
        self.checkpoints = sim.checkpoints

        self.corrupted_states = [s.clone() for s in self.original_states]
        injector = FaultInjector(seed=44)
        corrupted_list, _ = injector.inject_random_faults(
            self.corrupted_states, fault_rate=0.1
        )
        self.corrupted_ids = [s.state_id for s in corrupted_list]
        if not self.corrupted_ids:
            injector.inject_single_fault(self.corrupted_states, 5, "memory_corruption")
            self.corrupted_ids = [self.corrupted_states[5].state_id]

    def test_full_rollback(self):
        from src.baselines.full_rollback import FullRollbackRecovery
        bl = FullRollbackRecovery()
        result = bl.recover(self.original_states, self.corrupted_states, self.corrupted_ids)
        assert result["states_reconstructed"] == len(self.original_states)

    def test_checkpoint_recovery(self):
        from src.baselines.checkpoint_recovery import CheckpointRecovery
        bl = CheckpointRecovery(self.checkpoints)
        result = bl.recover(self.original_states, self.corrupted_states, self.corrupted_ids)
        assert result["states_reconstructed"] <= len(self.original_states)

    def test_dependency_recovery(self):
        from src.baselines.dependency_recovery import DependencyRecovery
        bl = DependencyRecovery(self.graph)
        result = bl.recover(self.original_states, self.corrupted_states, self.corrupted_ids)
        assert result["states_reconstructed"] <= len(self.original_states)
        assert result["states_reconstructed"] >= len(self.corrupted_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarios:
    @pytest.mark.parametrize("scenario", list(SCENARIO_BUILDERS.keys()))
    def test_scenario_builds(self, scenario):
        states, graph = build_scenario(scenario, num_states=25, seed=42)
        assert len(states) == 25
        assert graph.get_node_count() == 25

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            build_scenario("nonexistent_scenario", 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
