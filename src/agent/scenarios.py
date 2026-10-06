"""
BCRF Experiment - Agent Scenarios
Four reproducible agent task scenarios with different dependency structures.
"""

import random
from typing import List, Tuple, Dict
from src.agent.state import (
    AgentState, StateType, MemoryState, PlanningState,
    ToolInteraction, ExecutionMetadata
)
from src.agent.execution_graph import ExecutionGraph


def build_state(
    state_id: str,
    step_index: int,
    state_type: StateType,
    goal: str,
    subtasks: List[str],
    completed: List[str],
    working_mem: Dict,
    output: object,
    tool_name: str = "",
    tool_args: Dict = None,
    tool_result=None,
    is_external: bool = False,
    side_effect: bool = False,
    operation_id: str = None,
    scenario: str = "",
    logical_clock: int = 0,
    rng: random.Random = None,
) -> AgentState:
    """Helper to construct an AgentState."""
    rng = rng or random.Random(0)
    mem = MemoryState(
        working_memory=dict(working_mem),
        long_term_memory={"scenario": scenario},
        context_window=[f"Step {step_index}: {goal}"],
        memory_version=step_index,
    )
    plan = PlanningState(
        current_goal=goal,
        subtasks=list(subtasks),
        completed_subtasks=list(completed),
        plan_version=step_index // 5,
        confidence=round(rng.uniform(0.7, 1.0), 3),
    )
    tool = ToolInteraction(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        tool_status="success" if tool_result is not None else "none",
        is_external=is_external,
        side_effect_produced=side_effect,
        operation_id=operation_id,
    )
    meta = ExecutionMetadata(
        cpu_time_ms=round(rng.uniform(1.0, 50.0), 3),
        memory_bytes=rng.randint(1024, 65536),
        logical_clock=logical_clock,
        wall_time=logical_clock * 0.01,
        scenario=scenario,
        step_index=step_index,
        tags=[state_type.value],
    )
    state = AgentState(
        state_id=state_id,
        state_type=state_type,
        memory=mem,
        planning=plan,
        tool=tool,
        output=output,
        metadata=meta,
    )
    return state


def _make_id(scenario_prefix: str, index: int) -> str:
    return f"{scenario_prefix}_S{index:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO A: Order-processing agent
# Linear with several external side-effect operations
# ─────────────────────────────────────────────────────────────────────────────

def scenario_a_order_processing(
    num_states: int,
    rng: random.Random,
) -> Tuple[List[AgentState], ExecutionGraph]:
    """
    Order-processing agent:
    user_request → retrieve_customer → calculate_order → reserve_inventory
    → create_order → update_database → send_notification → [more steps...]

    Dependency structure: largely linear with some data-dependency skip edges.
    """
    prefix = "OA"
    states: List[AgentState] = []
    graph = ExecutionGraph()

    # Step templates (cycle if num_states > len(templates))
    step_templates = [
        ("receive_user_request", StateType.NORMAL, False, False),
        ("retrieve_customer_info", StateType.TOOL_CALL, False, False),
        ("validate_customer", StateType.NORMAL, False, False),
        ("calculate_order_total", StateType.NORMAL, False, False),
        ("apply_discounts", StateType.PLANNING, False, False),
        ("check_inventory", StateType.TOOL_CALL, False, False),
        ("reserve_inventory", StateType.EXTERNAL_OPERATION, True, True),
        ("create_order_record", StateType.EXTERNAL_OPERATION, True, True),
        ("process_payment", StateType.EXTERNAL_OPERATION, True, True),
        ("update_database", StateType.EXTERNAL_OPERATION, True, True),
        ("generate_invoice", StateType.NORMAL, False, False),
        ("send_notification", StateType.EXTERNAL_OPERATION, True, True),
        ("log_transaction", StateType.TOOL_CALL, False, False),
        ("update_analytics", StateType.NORMAL, False, False),
        ("finalize_order", StateType.NORMAL, False, False),
    ]

    goal = "Process customer order end-to-end"
    subtasks = [t[0] for t in step_templates]
    completed = []
    working_mem: Dict = {"order_id": None, "customer_id": None, "total": 0}
    op_counter = 0

    for i in range(num_states):
        tmpl = step_templates[i % len(step_templates)]
        name, stype, is_ext, side_eff = tmpl
        sid = _make_id(prefix, i)

        if side_eff:
            op_counter += 1
            op_id = f"op_{prefix}_{op_counter:04d}"
        else:
            op_id = None

        # Simulate accumulated state
        working_mem = dict(working_mem)
        if "order" in name:
            working_mem["order_id"] = f"ORD-{i:04d}"
        if "customer" in name:
            working_mem["customer_id"] = f"CUST-{rng.randint(1000, 9999)}"
        if "total" in name or "payment" in name:
            working_mem["total"] = round(rng.uniform(10.0, 5000.0), 2)

        output = {
            "step": name,
            "index": i,
            "value": rng.uniform(0, 100),
            "status": "ok",
        }

        state = build_state(
            state_id=sid,
            step_index=i,
            state_type=stype,
            goal=goal,
            subtasks=subtasks,
            completed=completed[:],
            working_mem=working_mem,
            output=output,
            tool_name=name if stype in (StateType.TOOL_CALL, StateType.EXTERNAL_OPERATION) else "",
            tool_args={"step": name, "index": i},
            tool_result=output if stype in (StateType.TOOL_CALL, StateType.EXTERNAL_OPERATION) else None,
            is_external=is_ext,
            side_effect=side_eff,
            operation_id=op_id,
            scenario="order_processing",
            logical_clock=i,
            rng=rng,
        )
        states.append(state)
        graph.add_state(state)
        completed.append(name)

        # Temporal edges
        if i > 0:
            graph.add_edge(_make_id(prefix, i - 1), sid, "temporal")
        # Skip data-dependency edges (non-contiguous)
        if i == 6:  # reserve_inventory depends on calculate_order (step 3)
            graph.add_edge(_make_id(prefix, 3), sid, "data")
        if i == 9 and num_states > 9:  # update_db depends on create_order
            graph.add_edge(_make_id(prefix, 7), sid, "data")

    return states, graph


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO B: Workflow automation agent
# Branching and merging dependency structure
# ─────────────────────────────────────────────────────────────────────────────

def scenario_b_workflow_automation(
    num_states: int,
    rng: random.Random,
) -> Tuple[List[AgentState], ExecutionGraph]:
    """
    Workflow automation agent with branching sub-workflows.

    Structure: planning → parallel data-retrieval branches → merge → transform
               → tool invocation → external update → final response
    """
    prefix = "WB"
    states: List[AgentState] = []
    graph = ExecutionGraph()

    step_templates = [
        ("parse_input", StateType.NORMAL, False, False),
        ("create_workflow_plan", StateType.PLANNING, False, False),
        ("fetch_data_source_a", StateType.TOOL_CALL, False, False),
        ("fetch_data_source_b", StateType.TOOL_CALL, False, False),
        ("validate_data_a", StateType.NORMAL, False, False),
        ("validate_data_b", StateType.NORMAL, False, False),
        ("merge_data", StateType.MEMORY_UPDATE, False, False),
        ("transform_data", StateType.NORMAL, False, False),
        ("apply_business_rules", StateType.DECISION, False, False),
        ("invoke_external_tool", StateType.EXTERNAL_OPERATION, True, True),
        ("store_intermediate", StateType.TOOL_CALL, False, False),
        ("run_secondary_workflow", StateType.PLANNING, False, False),
        ("external_system_update", StateType.EXTERNAL_OPERATION, True, True),
        ("generate_report", StateType.NORMAL, False, False),
        ("send_final_response", StateType.EXTERNAL_OPERATION, True, True),
    ]

    goal = "Automate multi-branch workflow"
    subtasks = [t[0] for t in step_templates]
    completed = []
    working_mem: Dict = {"input": None, "merged_data": None}
    op_counter = 0

    for i in range(num_states):
        tmpl = step_templates[i % len(step_templates)]
        name, stype, is_ext, side_eff = tmpl
        sid = _make_id(prefix, i)

        if side_eff:
            op_counter += 1
            op_id = f"op_{prefix}_{op_counter:04d}"
        else:
            op_id = None

        working_mem = dict(working_mem)
        working_mem[f"step_{i}"] = rng.random()

        output = {"step": name, "index": i, "branch": "A" if i % 2 == 0 else "B", "value": rng.uniform(0, 100)}

        state = build_state(
            state_id=sid, step_index=i, state_type=stype,
            goal=goal, subtasks=subtasks, completed=completed[:],
            working_mem=working_mem, output=output,
            tool_name=name if stype in (StateType.TOOL_CALL, StateType.EXTERNAL_OPERATION) else "",
            tool_args={"step": name}, tool_result=output if stype != StateType.NORMAL else None,
            is_external=is_ext, side_effect=side_eff, operation_id=op_id,
            scenario="workflow_automation", logical_clock=i, rng=rng,
        )
        states.append(state)
        graph.add_state(state)
        completed.append(name)

        if i > 0:
            graph.add_edge(_make_id(prefix, i - 1), sid, "temporal")

        # Branching: step 6 (merge) depends on both branch A (step 2) and branch B (step 3)
        if i == 6 and num_states > 6:
            if 2 < i: graph.add_edge(_make_id(prefix, 2), sid, "data")
            if 3 < i: graph.add_edge(_make_id(prefix, 3), sid, "data")

        # Merging: step 8 (business rules) also depends on step 4/5 validation
        if i == 8 and num_states > 8:
            if 4 < i: graph.add_edge(_make_id(prefix, 4), sid, "data")
            if 5 < i: graph.add_edge(_make_id(prefix, 5), sid, "data")

    return states, graph


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO C: Data-analysis agent
# Dense memory dependencies, non-contiguous causal links
# ─────────────────────────────────────────────────────────────────────────────

def scenario_c_data_analysis(
    num_states: int,
    rng: random.Random,
) -> Tuple[List[AgentState], ExecutionGraph]:
    """
    Data-analysis agent with dense memory dependencies.

    Structure: input → preprocessing → feature_calc → analysis →
               memory_updates → tool_call → final_result

    Many non-contiguous memory dependencies.
    """
    prefix = "DC"
    states: List[AgentState] = []
    graph = ExecutionGraph()

    step_templates = [
        ("ingest_dataset", StateType.TOOL_CALL, False, False),
        ("data_cleaning", StateType.NORMAL, False, False),
        ("normalize_features", StateType.NORMAL, False, False),
        ("compute_statistics", StateType.NORMAL, False, False),
        ("feature_engineering", StateType.MEMORY_UPDATE, False, False),
        ("dimensionality_reduction", StateType.NORMAL, False, False),
        ("update_feature_memory", StateType.MEMORY_UPDATE, False, False),
        ("run_analysis_model", StateType.TOOL_CALL, False, False),
        ("evaluate_results", StateType.NORMAL, False, False),
        ("persist_intermediate", StateType.EXTERNAL_OPERATION, True, True),
        ("refine_features", StateType.NORMAL, False, False),
        ("rerun_analysis", StateType.TOOL_CALL, False, False),
        ("summarize_findings", StateType.NORMAL, False, False),
        ("export_results", StateType.EXTERNAL_OPERATION, True, True),
        ("generate_report", StateType.NORMAL, False, False),
    ]

    goal = "Perform complete data analysis pipeline"
    subtasks = [t[0] for t in step_templates]
    completed = []
    working_mem: Dict = {"dataset_size": 0, "features": [], "model_accuracy": 0}
    op_counter = 0

    for i in range(num_states):
        tmpl = step_templates[i % len(step_templates)]
        name, stype, is_ext, side_eff = tmpl
        sid = _make_id(prefix, i)

        if side_eff:
            op_counter += 1
            op_id = f"op_{prefix}_{op_counter:04d}"
        else:
            op_id = None

        working_mem = dict(working_mem)
        working_mem["dataset_size"] = 1000 + i * 10
        if "feature" in name:
            working_mem["features"] = [f"f{j}" for j in range(i % 10)]
        if "model" in name or "analysis" in name:
            working_mem["model_accuracy"] = round(rng.uniform(0.6, 0.99), 4)

        output = {"step": name, "index": i, "accuracy": working_mem["model_accuracy"],
                  "n_features": len(working_mem.get("features", []))}

        state = build_state(
            state_id=sid, step_index=i, state_type=stype,
            goal=goal, subtasks=subtasks, completed=completed[:],
            working_mem=working_mem, output=output,
            tool_name=name if stype in (StateType.TOOL_CALL, StateType.EXTERNAL_OPERATION) else "",
            tool_args={"step": name}, tool_result=output if stype != StateType.NORMAL else None,
            is_external=is_ext, side_effect=side_eff, operation_id=op_id,
            scenario="data_analysis", logical_clock=i, rng=rng,
        )
        states.append(state)
        graph.add_state(state)
        completed.append(name)

        if i > 0:
            graph.add_edge(_make_id(prefix, i - 1), sid, "temporal")

        # Non-contiguous memory dependencies
        if i == 7 and num_states > 7:  # run_analysis depends on feature_engineering (4)
            graph.add_edge(_make_id(prefix, 4), sid, "memory")
        if i == 10 and num_states > 10:  # refine_features depends on evaluate (8)
            graph.add_edge(_make_id(prefix, 8), sid, "memory")
        if i == 11 and num_states > 11:  # rerun depends on refine (10) AND original features (4)
            graph.add_edge(_make_id(prefix, 4), sid, "memory")

    return states, graph


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO D: Multi-step planning agent
# Complex goal/sub-goal hierarchy with decision revisions
# ─────────────────────────────────────────────────────────────────────────────

def scenario_d_multi_step_planning(
    num_states: int,
    rng: random.Random,
) -> Tuple[List[AgentState], ExecutionGraph]:
    """
    Multi-step planning agent with goal hierarchy and decision revisions.

    Structure: goal → planning → subtask_execution → intermediate_states →
               tool_interactions → decision_revision → final_result

    Complex dependency: decision revisions create backward edges (simulated
    as new planning states depending on earlier decision states).
    """
    prefix = "MP"
    states: List[AgentState] = []
    graph = ExecutionGraph()

    step_templates = [
        ("define_goal", StateType.PLANNING, False, False),
        ("decompose_goal", StateType.PLANNING, False, False),
        ("allocate_resources", StateType.DECISION, False, False),
        ("execute_subtask_1", StateType.NORMAL, False, False),
        ("execute_subtask_2", StateType.NORMAL, False, False),
        ("check_progress", StateType.DECISION, False, False),
        ("call_planning_tool", StateType.TOOL_CALL, False, False),
        ("revise_plan", StateType.PLANNING, False, False),
        ("execute_revised_subtask", StateType.NORMAL, False, False),
        ("external_resource_acquire", StateType.EXTERNAL_OPERATION, True, True),
        ("integrate_subtask_results", StateType.MEMORY_UPDATE, False, False),
        ("validate_intermediate", StateType.DECISION, False, False),
        ("external_commit", StateType.EXTERNAL_OPERATION, True, True),
        ("synthesize_result", StateType.NORMAL, False, False),
        ("finalize_plan", StateType.PLANNING, False, False),
    ]

    goal = "Achieve multi-step complex goal through adaptive planning"
    subtasks = [t[0] for t in step_templates]
    completed = []
    working_mem: Dict = {"goal": goal, "subtask_results": {}, "revision_count": 0}
    op_counter = 0

    for i in range(num_states):
        tmpl = step_templates[i % len(step_templates)]
        name, stype, is_ext, side_eff = tmpl
        sid = _make_id(prefix, i)

        if side_eff:
            op_counter += 1
            op_id = f"op_{prefix}_{op_counter:04d}"
        else:
            op_id = None

        working_mem = dict(working_mem)
        if "subtask" in name:
            working_mem["subtask_results"][f"task_{i}"] = rng.random()
        if "revise" in name:
            working_mem["revision_count"] = working_mem.get("revision_count", 0) + 1

        output = {"step": name, "index": i,
                  "revision_count": working_mem.get("revision_count", 0),
                  "value": rng.uniform(0, 100)}

        state = build_state(
            state_id=sid, step_index=i, state_type=stype,
            goal=goal, subtasks=subtasks, completed=completed[:],
            working_mem=working_mem, output=output,
            tool_name=name if stype in (StateType.TOOL_CALL, StateType.EXTERNAL_OPERATION) else "",
            tool_args={"step": name}, tool_result=output if stype != StateType.NORMAL else None,
            is_external=is_ext, side_effect=side_eff, operation_id=op_id,
            scenario="multi_step_planning", logical_clock=i, rng=rng,
        )
        states.append(state)
        graph.add_state(state)
        completed.append(name)

        if i > 0:
            graph.add_edge(_make_id(prefix, i - 1), sid, "temporal")

        # Decision revision creates dependency on earlier decision (non-contiguous)
        if i == 7 and num_states > 7:  # revise_plan depends on check_progress (5)
            graph.add_edge(_make_id(prefix, 5), sid, "causal")
        if i == 11 and num_states > 11:  # validate_intermediate depends on allocate (2)
            graph.add_edge(_make_id(prefix, 2), sid, "causal")
        if i == 10 and num_states > 10:  # integrate results depends on subtask1 (3) and subtask2 (4)
            if 3 < i: graph.add_edge(_make_id(prefix, 3), sid, "data")
            if 4 < i: graph.add_edge(_make_id(prefix, 4), sid, "data")

    return states, graph


# ─────────────────────────────────────────────────────────────────────────────
# Generic scenario builder for arbitrary N
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_BUILDERS = {
    "order_processing": scenario_a_order_processing,
    "workflow_automation": scenario_b_workflow_automation,
    "data_analysis": scenario_c_data_analysis,
    "multi_step_planning": scenario_d_multi_step_planning,
}


def build_scenario(
    scenario_name: str,
    num_states: int,
    seed: int = 42,
) -> Tuple[List[AgentState], ExecutionGraph]:
    """Build a named scenario with a given number of states and random seed."""
    rng = random.Random(seed)
    builder = SCENARIO_BUILDERS.get(scenario_name)
    if builder is None:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                         f"Available: {list(SCENARIO_BUILDERS.keys())}")
    return builder(num_states, rng)
