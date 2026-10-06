"""
BCRF Experiment - Patent-Quality Figure Generation
Generates all 12 required figures at 300 DPI.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Any, Optional


# ── Publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFC",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#DDE3EA",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})

# Vibrant, distinct color palette (colorblind-friendly)
METHOD_COLORS = {
    "full_rollback": "#E84855",   # Bold red
    "checkpoint":    "#F4A261",   # Warm orange
    "dependency":    "#2A9D8F",   # Teal green
    "bcrf":          "#264653",   # Deep navy (proposed method, stands out)
}
METHOD_MARKERS = {
    "full_rollback": "s",
    "checkpoint": "^",
    "dependency": "D",
    "bcrf": "o",
}
METHOD_LABELS = {
    "full_rollback": "Full Rollback",
    "checkpoint": "Checkpoint",
    "dependency": "Dependency-Guided",
    "bcrf": "BCRF (Proposed)",
}
METHOD_LINESTYLES = {
    "full_rollback": "--",
    "checkpoint": "-.",
    "dependency": ":",
    "bcrf": "-",
}

DPI = 300
FORMATS = ["png", "pdf"]


def _save_figure(fig, figures_dir: str, stem: str) -> List[str]:
    """Save a figure in all configured formats."""
    paths = []
    os.makedirs(figures_dir, exist_ok=True)
    for fmt in FORMATS:
        path = os.path.join(figures_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", format=fmt)
        paths.append(path)
    return paths


def fig1_recovery_success_comparison(
    df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG1: Recovery Success Rate Comparison (bar chart)."""
    methods = ["full_rollback", "checkpoint", "dependency", "bcrf"]
    labels = [METHOD_LABELS[m] for m in methods]

    success_rates = []
    errors = []
    for m in methods:
        col = f"{m}_success"
        if col in df.columns:
            vals = df[col].dropna().astype(float) * 100
            success_rates.append(vals.mean())
            errors.append(vals.std() / np.sqrt(len(vals)) * 1.96 if len(vals) > 1 else 0)
        else:
            success_rates.append(0)
            errors.append(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    colors = [METHOD_COLORS[m] for m in methods]
    bars = ax.bar(x, success_rates, yerr=errors, capsize=4,
                  color=colors, width=0.55, edgecolor="black", linewidth=0.7)

    ax.set_xlabel("Recovery Method")
    ax.set_ylabel("Recovery Success Rate (%)")
    ax.set_title("Figure 1: Recovery Success Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 115)

    for bar, val, err in zip(bars, success_rates, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG1_recovery_success_comparison")
    plt.close(fig)
    return paths[0]


def fig2_recovery_latency_vs_states(
    scalability_df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG2: Recovery Latency vs Number of Execution States."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = ["full_rollback", "checkpoint", "dependency", "bcrf"]

    for m in methods:
        col = f"{m}_recovery_time_ms"
        if col not in scalability_df.columns:
            continue
        grouped = scalability_df.groupby("num_states")[col]
        means = grouped.mean()
        sems = grouped.sem()
        x = means.index.values
        y = means.values
        err = sems.values * 1.96

        ax.plot(x, y, marker=METHOD_MARKERS[m], linestyle=METHOD_LINESTYLES[m],
                color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.fill_between(x, y - err, y + err, alpha=0.12, color=METHOD_COLORS[m])

    ax.set_xlabel("Number of Execution States")
    ax.set_ylabel("Recovery Latency (ms)")
    ax.set_title("Figure 2: Recovery Latency vs. Number of Execution States")
    ax.legend()
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG2_recovery_latency_vs_states")
    plt.close(fig)
    return paths[0]


def fig3_reconstructed_states(
    scalability_df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG3: Number of Reconstructed States."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ["full_rollback", "checkpoint", "dependency", "bcrf"]

    for m in methods:
        col = f"{m}_states_reconstructed"
        if col not in scalability_df.columns:
            continue
        grouped = scalability_df.groupby("num_states")[col]
        means = grouped.mean()
        x = means.index.values
        ax.plot(x, means.values, marker=METHOD_MARKERS[m], linestyle=METHOD_LINESTYLES[m],
                color=METHOD_COLORS[m], label=METHOD_LABELS[m])

    ax.set_xlabel("Number of Execution States")
    ax.set_ylabel("Number of States Reconstructed")
    ax.set_title("Figure 3: States Reconstructed by Recovery Method")
    ax.legend()
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG3_reconstructed_states")
    plt.close(fig)
    return paths[0]


def fig4_reconstruction_percentage(
    scalability_df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG4: Percentage of States Reconstructed."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ["full_rollback", "checkpoint", "dependency", "bcrf"]

    for m in methods:
        col = f"{m}_pct_reconstructed"
        if col not in scalability_df.columns:
            continue
        grouped = scalability_df.groupby("num_states")[col]
        means = grouped.mean()
        x = means.index.values
        ax.plot(x, means.values, marker=METHOD_MARKERS[m], linestyle=METHOD_LINESTYLES[m],
                color=METHOD_COLORS[m], label=METHOD_LABELS[m])

    ax.set_xlabel("Number of Execution States")
    ax.set_ylabel("States Reconstructed (%)")
    ax.set_title("Figure 4: Relative Recovery Scope by Method")
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG4_reconstruction_percentage")
    plt.close(fig)
    return paths[0]


def fig5_duplicate_side_effects(
    df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG5: Duplicate External Operations."""
    methods = ["full_rollback", "checkpoint", "dependency", "bcrf"]
    labels = [METHOD_LABELS[m] for m in methods]

    dup_means = []
    dup_errors = []
    for m in methods:
        col = f"{m}_duplicate_side_effects"
        if col in df.columns:
            vals = df[col].dropna().astype(float)
            dup_means.append(vals.mean())
            dup_errors.append(vals.std() / np.sqrt(len(vals)) * 1.96 if len(vals) > 1 else 0)
        else:
            dup_means.append(0)
            dup_errors.append(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    colors = [METHOD_COLORS[m] for m in methods]
    bars = ax.bar(x, dup_means, yerr=dup_errors, capsize=4,
                  color=colors, width=0.55, edgecolor="black", linewidth=0.7)

    ax.set_xlabel("Recovery Method")
    ax.set_ylabel("Mean Duplicate Side Effects")
    ax.set_title("Figure 5: Duplicate External Operations by Recovery Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    for bar, val in zip(bars, dup_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG5_duplicate_side_effects")
    plt.close(fig)
    return paths[0]


def fig6_causal_repair_frontier(
    fci_results: List[Dict],
    bcr_results: List[Dict],
    crf_ids: set,
    graph_nx: nx.DiGraph,
    execution_order: List[str],
    figures_dir: str,
) -> str:
    """FIG6: Causal Frontier Formation — state graph with highlighted CRF."""
    fig, ax = plt.subplots(figsize=(10, 5))

    fci_map = {r["state_id"]: r["FCI"] for r in fci_results}
    bcr_map = {r["state_id"]: r["BCR"] for r in bcr_results}

    # Layout: linear by execution order
    nodes = list(graph_nx.nodes())
    if not nodes:
        ax.text(0.5, 0.5, "No graph data", ha="center", va="center")
        paths = _save_figure(fig, figures_dir, "FIG6_causal_repair_frontier")
        plt.close(fig)
        return paths[0]

    # Use subset of nodes for clarity if too many
    max_display = 30
    if len(nodes) > max_display:
        nodes = nodes[:max_display]
        node_set = set(nodes)
        subgraph = graph_nx.subgraph(node_set)
    else:
        subgraph = graph_nx

    pos = {n: (i, 0) for i, n in enumerate(nodes)}

    # Node colors — vibrant scheme
    node_colors = []
    node_sizes = []
    for n in subgraph.nodes():
        if n in crf_ids:
            fci = fci_map.get(n, 0)
            bcr = bcr_map.get(n, 0)
            if fci > 0 and bcr > 0:
                node_colors.append("#264653")   # Deep navy: CRF state (FCI+BCR)
                node_sizes.append(320)
            else:
                node_colors.append("#2A9D8F")   # Teal: CRF (partial)
                node_sizes.append(220)
        else:
            node_colors.append("#A8D5E2")       # Light blue: non-CRF
            node_sizes.append(120)

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, ax=ax, arrows=True,
                           arrowsize=12, edge_color="#546E7A",
                           connectionstyle="arc3,rad=0.1")

    # Draw node labels (abbreviated)
    labels_map = {n: n.split("_")[-1] if "_" in n else n for n in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels=labels_map, ax=ax,
                            font_size=6, font_color="white")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#264653", label="CRF State (selected)"),
        mpatches.Patch(facecolor="#A8D5E2", label="Non-CRF State"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_title("Figure 6: Causal Repair Frontier Formation\n"
                 "(Navy nodes = CRF selected; gaps show non-contiguous selection)")
    ax.axis("off")
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG6_causal_repair_frontier")
    plt.close(fig)
    return paths[0]


def fig7_fci_bcr_scatter(
    fci_results: List[Dict],
    bcr_results: List[Dict],
    crf_ids: set,
    T_F: float,
    T_B: float,
    figures_dir: str,
) -> str:
    """FIG7: FCI vs BCR scatter plot with CRF threshold lines."""
    fig, ax = plt.subplots(figsize=(6, 5))

    fci_map = {r["state_id"]: r["FCI"] for r in fci_results}
    bcr_map = {r["state_id"]: r["BCR"] for r in bcr_results}
    all_ids = list(fci_map.keys())

    selected_fci = [fci_map[sid] for sid in all_ids if sid in crf_ids]
    selected_bcr = [bcr_map[sid] for sid in all_ids if sid in crf_ids]
    non_selected_fci = [fci_map[sid] for sid in all_ids if sid not in crf_ids]
    non_selected_bcr = [bcr_map[sid] for sid in all_ids if sid not in crf_ids]

    ax.scatter(non_selected_fci, non_selected_bcr, c="#A8D5E2", s=40,
               marker="o", label="Non-CRF States", alpha=0.8, edgecolors="#546E7A", linewidths=0.5)
    ax.scatter(selected_fci, selected_bcr, c="#264653", s=80,
               marker="o", label="CRF States (selected)", zorder=5, edgecolors="white", linewidths=0.8)

    # Threshold lines
    ax.axvline(x=T_F, color="#E84855", linestyle="--", linewidth=1.6, label=f"T_F = {T_F:.3f}")
    ax.axhline(y=T_B, color="#F4A261", linestyle=":", linewidth=1.6, label=f"T_B = {T_B:.3f}")

    ax.set_xlabel("Forward Counterfactual Influence (FCI)")
    ax.set_ylabel("Backward Causal Responsibility (BCR)")
    ax.set_title("Figure 7: FCI vs BCR — Bidirectional Causal State Selection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG7_fci_bcr_scatter")
    plt.close(fig)
    return paths[0]


def fig8_ablation_study(
    ablation_df: pd.DataFrame, figures_dir: str, metric: str = "success"
) -> str:
    """FIG8: Ablation Study."""
    condition_labels = {
        "full_bcrf": "Full BCRF",
        "no_fci": "Without FCI",
        "no_bcr": "Without BCR",
        "no_registry": "Without Registry",
        "no_civ": "Without CIV",
        "no_expansion": "Without CRF Expansion",
    }
    conditions = list(condition_labels.keys())
    labels = [condition_labels[c] for c in conditions]

    col = f"bcrf_{metric}"
    if col not in ablation_df.columns:
        col = "bcrf_success"

    means = []
    errors = []
    for c in conditions:
        if "condition" in ablation_df.columns:
            subset = ablation_df[ablation_df["condition"] == c][col].dropna()
        else:
            subset = pd.Series([])
        if len(subset) > 0:
            means.append(subset.astype(float).mean() * (100 if "success" in col else 1))
            errors.append(subset.astype(float).std() / np.sqrt(len(subset)) * 1.96 if len(subset) > 1 else 0)
        else:
            means.append(0)
            errors.append(0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    # Each ablation condition gets its own color; full BCRF is deep navy (hero)
    ablation_colors = {
        "full_bcrf":    "#264653",  # Deep navy  — Full BCRF (proposed)
        "no_fci":       "#E84855",  # Bold red   — missing FCI
        "no_bcr":       "#F4A261",  # Orange     — missing BCR
        "no_registry":  "#2A9D8F",  # Teal       — missing Registry
        "no_civ":       "#E9C46A",  # Yellow     — missing CIV
        "no_expansion": "#8ECAE6",  # Sky blue   — missing CRF expansion
    }
    colors = [ablation_colors.get(c, "#888888") for c in conditions]
    bars = ax.bar(x, means, yerr=errors, capsize=4,
                  color=colors, width=0.6, edgecolor="white", linewidth=0.8)

    y_label = "Recovery Success Rate (%)" if "success" in col else metric
    ax.set_ylabel(y_label)
    ax.set_title("Figure 8: Ablation Study of BCRF Components")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errors) * 0.5 + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG8_ablation_study")
    plt.close(fig)
    return paths[0]


def fig9_civ_validation(
    civ_df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG9: CIV scores for recovery attempts."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if "attempt_num" in civ_df.columns and "civ_score" in civ_df.columns:
        for trial_id, group in civ_df.groupby("trial_id") if "trial_id" in civ_df.columns else [(0, civ_df)]:
            group = group.sort_values("attempt_num")
            ax.plot(group["attempt_num"], group["civ_score"],
                    marker="o", color="#2A9D8F", alpha=0.65, linewidth=1.4)

    # Threshold line
    threshold = civ_df.get("threshold", pd.Series([0.65])).iloc[0] if "threshold" in civ_df.columns else 0.65
    ax.axhline(y=threshold, color="#E84855", linestyle="--", linewidth=2.0,
               label=f"CIV Threshold = {threshold:.2f}")

    ax.set_xlabel("Recovery Attempt Number")
    ax.set_ylabel("CIV Score")
    ax.set_title("Figure 9: Counterfactual Integrity Validation Scores")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG9_civ_validation")
    plt.close(fig)
    return paths[0]


def fig10_recovery_iterations(
    iteration_df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG10: CIV score across recovery iterations (closed-loop)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if "attempt_num" in iteration_df.columns and "civ_score" in iteration_df.columns:
        grouped = iteration_df.groupby("attempt_num")["civ_score"]
        means = grouped.mean()
        sems = grouped.sem() * 1.96
        x = means.index.values

        ax.plot(x, means.values, "o-", color="#264653", linewidth=2.4, markersize=7,
                label="Mean CIV Score")
        ax.fill_between(x, means.values - sems.values, means.values + sems.values,
                        alpha=0.18, color="#2A9D8F")

    threshold = 0.65
    ax.axhline(y=threshold, color="#E84855", linestyle="--", linewidth=2.0,
               label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("Recovery Iteration (CRF Expansion Step)")
    ax.set_ylabel("CIV Score")
    ax.set_title("Figure 10: Closed-Loop Recovery Progression (CIV vs. Iteration)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG10_recovery_iterations")
    plt.close(fig)
    return paths[0]


def fig11_scalability(
    scalability_df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG11: Scalability — Recovery Time vs State Count."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ["full_rollback", "checkpoint", "dependency", "bcrf"]

    for m in methods:
        col = f"{m}_recovery_time_ms"
        if col not in scalability_df.columns:
            continue
        grouped = scalability_df.groupby("num_states")[col]
        means = grouped.mean()
        sems = grouped.sem() * 1.96
        x = means.index.values

        ax.plot(x, means.values, marker=METHOD_MARKERS[m],
                linestyle=METHOD_LINESTYLES[m],
                color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.fill_between(x, means.values - sems.values, means.values + sems.values,
                        alpha=0.1, color=METHOD_COLORS[m])

    ax.set_xlabel("Number of Execution States")
    ax.set_ylabel("Recovery Time (ms)")
    ax.set_title("Figure 11: Scalability — Recovery Time vs. State Count")
    ax.legend()
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG11_scalability")
    plt.close(fig)
    return paths[0]


def fig12_fault_robustness(
    fault_df: pd.DataFrame, figures_dir: str
) -> str:
    """FIG12: Fault-Type Robustness."""
    fault_types = fault_df["fault_type"].unique() if "fault_type" in fault_df.columns else []
    methods = ["full_rollback", "checkpoint", "dependency", "bcrf"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fault_types))
    width = 0.2

    for i, m in enumerate(methods):
        col = f"{m}_success"
        if col not in fault_df.columns:
            continue
        vals = []
        for ft in fault_types:
            subset = fault_df[fault_df["fault_type"] == ft][col].dropna().astype(float) * 100
            vals.append(subset.mean() if len(subset) > 0 else 0)

        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=METHOD_LABELS[m],
               color=METHOD_COLORS[m], edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Recovery Success Rate (%)")
    ax.set_title("Figure 12: Fault-Type Robustness of Recovery Methods")
    ax.set_xticks(x)
    ax.set_xticklabels([ft.replace("_", "\n") for ft in fault_types],
                       fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 120)
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "FIG12_fault_robustness")
    plt.close(fig)
    return paths[0]


def generate_figure_captions(figures_dir: str) -> str:
    """Generate figure_captions.txt with patent-ready captions."""
    captions = """BCRF Experimental Validation — Patent-Ready Figure Captions
============================================================

FIG1_recovery_success_comparison
Caption: "Comparison of recovery success rates among baseline recovery mechanisms
(Full Rollback, Checkpoint-Based, and Dependency-Guided) and the proposed BCRF
framework, measured across all experimental trials. Error bars represent 95%
confidence intervals."

FIG2_recovery_latency_vs_states
Caption: "Recovery latency as a function of execution-state count for all four
evaluated methods. BCRF selective reconstruction limits the growth of recovery
cost as state count increases."

FIG3_reconstructed_states
Caption: "Absolute number of states reconstructed by each recovery method across
varying execution lengths. BCRF selectively reconstructs only causally relevant
states identified by the Causal Repair Frontier."

FIG4_reconstruction_percentage
Caption: "Relative recovery scope of the evaluated recovery mechanisms, expressed
as the percentage of total execution states requiring reconstruction."

FIG5_duplicate_side_effects
Caption: "Effect of the External Operation Registry-based non-replay control on
duplicate side-effect-producing operations. BCRF with the registry enabled
prevents re-execution of already-completed external operations during recovery."

FIG6_causal_repair_frontier
Caption: "Example formation of the Causal Repair Frontier through bidirectional
causal analysis. Black nodes indicate states selected for reconstruction; the
non-contiguous nature of the CRF demonstrates that causally irrelevant
intermediate states are excluded from recovery."

FIG7_fci_bcr_scatter
Caption: "Bidirectional causal state selection using Forward Counterfactual
Influence (FCI) and Backward Causal Responsibility (BCR). States in the upper-right
quadrant (above T_F and T_B threshold lines) are selected into the Causal Repair
Frontier."

FIG8_ablation_study
Caption: "Ablation analysis of major BCRF components. Each bar corresponds to a
variant with one component removed relative to the full BCRF configuration,
demonstrating the contribution of each component to overall recovery effectiveness."

FIG9_civ_validation
Caption: "Counterfactual Integrity Validation (CIV) scores for individual recovery
attempts. The dashed line indicates the CIV acceptance threshold; attempts below
the threshold trigger CRF expansion and additional reconstruction."

FIG10_recovery_iterations
Caption: "Closed-loop recovery progression showing CIV score evolution across
iterative Causal Repair Frontier expansion steps. The mean CIV score and 95%
confidence band are shown; recovery terminates when CIV exceeds the threshold."

FIG11_scalability
Caption: "Scalability of recovery methods expressed as recovery time versus
execution-state count. BCRF computational cost grows more slowly than full
rollback/replay as state count increases."

FIG12_fault_robustness
Caption: "Recovery success rate of all evaluated methods across six fault types:
memory corruption, planning corruption, tool-output corruption, dependency
corruption, metadata corruption, and combined corruption."
"""
    path = os.path.join(figures_dir, "figure_captions.txt")
    with open(path, "w") as f:
        f.write(captions)
    return path
