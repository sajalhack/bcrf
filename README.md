# BCRF: Bidirectional Causal Repair Frontier for Autonomous State Recovery in AI Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending%20(IPO%20%2F%20USPTO)-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/pytest-48%2F48%20passed%20(100%25)-success.svg)]()
[![Experimental Trials](https://img.shields.io/badge/Evaluated%20Trials-600%20Deterministic%20Runs-blue.svg)]()
[![Duplicate Side Effects](https://img.shields.io/badge/Duplicate%20Side%20Effects-0.000%20(Zero)-success.svg)]()

> **Full Patent Title:**  
> *System and Method for Autonomous Causal Recovery of Execution States in Stateful Artificial Intelligence Agents*  
> **Short Name:** `BCRF` (Bidirectional Causal Repair Frontier)

---

## 📌 Executive Summary & Abstract

Stateful Artificial Intelligence (AI) agents maintain evolving internal memory, hierarchical planning contexts, tool execution traces, and external side-effect operations (e.g., database writes, financial transactions, API calls). When an in-flight runtime integrity fault occurs (such as memory corruption, invalid tool results, or dependency divergence):
* **Conventional Full Rollback & Replay** forces the agent to restart from the beginning, wasting execution budget and repeating irreversible external operations (causing duplicate side effects).
* **Checkpoint-Based Recovery** can only restore coarse snapshots, re-executing uncorrupted intermediate states and failing to isolate non-contiguous causal dependencies.
* **Dependency-Only Rollback** blindly traverses all graph ancestors, failing to differentiate genuine causal root causes from incidental ancestors.

**BCRF (Bidirectional Causal Repair Frontier)** solves this fundamental limitation. By combining **Forward Counterfactual Influence (FCI)** with **Backward Causal Responsibility (BCR)**, BCRF dynamically constructs a minimal, non-contiguous **Causal Repair Frontier (CRF)**. It restores execution states using tamper-evident **State Reconstruction Capsules (SRC)**, eliminates duplicate side effects via an **External Operation Registry**, and validates state recovery using a closed-loop **Counterfactual Integrity Validation (CIV)** gate.

---

## 🔬 Core Mathematical Formulation

```
                                      ┌───────────────────────────────┐
                                      │  Forward Counterfactual (FCI) │
                                      │  FCI_i = |V(S_i) - V(S_i^cf)| │
                                      └──────────────┬────────────────┘
                                                     │
[In-Flight Integrity Fault Detected] ────────────────┼──► [Causal Repair Frontier: CRF = {S_i | FCI_i >= T_F AND BCR_i >= T_B}]
                                                     │
                                      ┌──────────────┴────────────────┐
                                      │ Backward Responsibility (BCR) │
                                      │ BCR_i = P(S_i causal to V | E)│
                                      └───────────────────────────────┘
```

### 1. Forward Counterfactual Influence (FCI)
Quantifies downstream impact if state $S_i$ is counterfactually perturbed or removed:
$$FCI_i = |V(S_i) - V(S_i^{cf})|$$

### 2. Backward Causal Responsibility (BCR)
Quantifies the path-weighted structural probability that state $S_i$ caused the detected violation $V$:
$$BCR_i = P(S_i \text{ is causal to } V \mid E)$$

### 3. Causal Repair Frontier (CRF)
Minimal set of states requiring targeted reconstruction:
$$CRF = \left\{ S_i \mid FCI_i \ge T_F \land BCR_i \ge T_B \right\}$$
*(where $T_F$ and $T_B$ are adaptive percentile-based thresholds).*

### 4. Counterfactual Integrity Validation (CIV)
Post-repair closed-loop validation gate:
$$CIV = w_1 O + w_2 I + w_3 C$$
$$\text{where: } w_1 = 0.40 \text{ (Outcome Consistency)}, \; w_2 = 0.35 \text{ (State Integrity)}, \; w_3 = 0.25 \text{ (Causal Path Consistency)}$$
$$\text{Acceptance Criterion: } CIV \ge T_{CIV} \; (T_{CIV} = 0.65)$$

---

## 🏛️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   STATEFUL AI AGENT EXECUTION                                    │
│    Memory State (v_t)  │  Planning Context (P_t)  │  Tool Action (A_t)  │  External API Call     │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                               │ In-flight Execution Stream
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 RUNTIME INTEGRITY MONITORING                                     │
│  • Cryptographic State Hash Signature Verification                                               │
│  • In-flight Violation Detection (Memory, Tool, Planning, Dependency, Metadata)                 │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                               │ Integrity Violation Detected
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            BIDIRECTIONAL CAUSAL ANALYSIS (FCI + BCR)                             │
│  • Forward Counterfactual Simulation: FCI_i = |V(S_i) - V(S_i^cf)|                               │
│  • Backward Causal Responsibility: BCR_i = Path-weighted structural analysis                     │
│  • Adaptive Thresholds: T_F (60th percentile), T_B (50th percentile)                             │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                               │ Causal Repair Frontier (CRF)
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              AUTONOMOUS RECOVERY & DEDUPLICATION                                 │
│  • External Operation Registry: Checks idempotency keys -> ZERO duplicate API calls / writes     │
│  • State Reconstruction Capsules (SRC): Restores verified snapshots of CRF states only           │
│  • Non-Contiguous State Skipping: Excludes causally irrelevant intermediate states               │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                               │ Reconstructed Execution State
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          COUNTERFACTUAL INTEGRITY VALIDATION (CIV) GATE                          │
│  • Computes CIV = 0.40*O + 0.35*I + 0.25*C                                                      │
│  • If CIV >= 0.65 ──► RESUME EXECUTION (Success)                                                 │
│  • If CIV < 0.65  ──► EXPAND CRF (+15% frontier margin) & RE-EXECUTE REPAIR                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Measured Experimental Results (600 Trials, N=25 to 500)

Evaluated across **4 diverse real-world agent scenarios** (*Order Processing, Workflow Automation, Data Analysis, Multi-Step Planning*) and **5 execution sizes** (*N = 25, 50, 100, 250, 500 states*), 30 deterministic trials per configuration (Seed = 42).

### 🏆 Master Comparison Table

| Metric | BCRF (Proposed) | Full Rollback & Replay | Checkpoint Recovery | Dependency-Guided | Technical Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **Recovery Success Rate** | **100.0%** | 100.0% | 100.0% | 100.0% | All methods recover; efficiency differs |
| **States Reconstructed (Mean)** | **87.81 (46.5%)** | 185.00 (100.0%) | 178.07 (91.4%) | 176.37 (88.8%) | **53.5% Fewer States Replayed** |
| **Duplicate Side Effects / Trial** | **0.000** | 36.450 | 35.250 | 34.600 | **100% Duplicate Elimination** |
| **External Operations Prevented** | **25.85 / trial** | 0.00 | 0.00 | 0.00 | Up to 71.6 ops saved per run |
| **Mean CIV Validation Score** | **0.8961** | — (N/A) | — (N/A) | — (N/A) | High confidence ($>0.65$ threshold) |
| **Non-Contiguous CRF Selection** | **Demonstrated** | Impossible | Impossible | Impossible | Skips uncorrupted intermediate steps |

---

## 📈 Ablation Study (Component Contributions)

Evaluated on Order Processing ($N = 50$, Seed = 42):

| Ablation Condition | States Reconstructed | Duplicate Side Effects | External Ops Prevented | Mean CIV Score | Finding |
|:---|:---:|:---:|:---:|:---:|:---|
| **Full BCRF (All Components)** | **22.17** | **0.000** | **11.33** | **0.8953** | **Optimal selective repair** |
| **Without FCI (No Forward Simulation)** | 42.17 (+90.2%) | 0.000 | 13.33 | 0.8976 | FCI is essential to eliminate false ancestors |
| **Without BCR (No Backward Analysis)** | 22.97 (+3.6%) | 0.000 | 11.67 | 0.8953 | BCR ensures path-responsibility isolation |
| **Without External Operation Registry** | 22.17 | 11.33 (+100%) | **0.00** | 0.8953 | **Registry is strictly necessary for zero duplicates** |
| **Without CIV Validation** | 22.17 | 0.000 | 11.33 | 1.0000* | Eliminates safety verification gate |
| **Without CRF Expansion** | 22.17 | 0.000 | 11.33 | 0.8953 | First-iteration repair sufficient for 10% fault |

---

## 🖼️ Publication & Patent Figures (FIG1 to FIG12)

All figures are generated in publication quality (300 DPI PNG + vector PDF) in `results/figures/`:

| Figure | Thumbnail / Title | Key Technical Insight |
|:---:|:---|:---|
| **FIG. 1** | **Recovery Success Rate** | 100% success across all baselines; establishes validity baseline. |
| **FIG. 2** | **Recovery Latency vs. State Count** | BCRF latency scaling across $N = 25$ to $500$ states. |
| **FIG. 3** | **States Reconstructed (Absolute)** | BCRF reconstructs $87.8$ states vs. $185.0$ for Full Rollback. |
| **FIG. 4** | **Reconstruction Scope (%)** | BCRF maintains consistent $44\%\text{--}48\%$ selectivity across all scales. |
| **FIG. 5** | **Duplicate External Operations** | BCRF produces **0.000 duplicates** vs. $36.45$ for Full Rollback. |
| **FIG. 6** | **Causal Repair Frontier Graph** | Visualizes non-contiguous state selection on execution graph. |
| **FIG. 7** | **FCI vs. BCR Scatter Plot** | State classification into CRF quadrant above $T_F$ and $T_B$. |
| **FIG. 8** | **Ablation Study Breakdown** | Component contributions of FCI, BCR, Registry, CIV, and Expansion. |
| **FIG. 9** | **CIV Score Distribution** | Measured CIV scores consistently passing the $T_{CIV} = 0.65$ gate. |
| **FIG. 10** | **Closed-Loop Progression** | CIV convergence across iterative repair attempts. |
| **FIG. 11** | **Scalability Analysis** | Multi-method resource tradeoffs up to 500 execution states. |
| **FIG. 12** | **Fault-Type Robustness** | Zero duplicates across Memory, Tool, Planning, Dependency, Metadata faults. |

---

## 🚀 Quickstart & Reproducibility Guide

### 1. Installation
```bash
# Clone repository
git clone https://github.com/sajalhack/bcrf.git
cd bcrf

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Instant 1-Second Live Demo
```bash
python3 demo.py
```

### 3. Run Unit Test Suite (48 Tests)
```bash
pytest tests/ -v
```

### 4. Execute Full 600-Trial Experimental Validation Pipeline
```bash
# Quick Mode (60 trials: N=25, 50, 100; ~10 seconds)
python3 run_all_experiments.py --quick --seed 42

# Full Mode (600 trials: N=25, 50, 100, 250, 500; ~3 minutes)
python3 run_all_experiments.py --seed 42
```

---

## 📁 Repository Structure

```
bcrf/
├── README.md                            # Comprehensive project overview & patent results
├── demo.py                              # 1-second live demonstration script
├── requirements.txt                     # Python dependencies
├── config.yaml                          # Experiment hyperparameters & thresholds
├── docs/
│   └── BCRF_Patent_Experimental_Validation_Section.md  # Formal patent specification section
├── src/
│   ├── bcrf/
│   │   ├── fci.py                       # Forward Counterfactual Influence (FCI)
│   │   ├── bcr.py                       # Backward Causal Responsibility (BCR)
│   │   ├── causal_frontier.py           # CRF Construction & Adaptive Expansion
│   │   ├── src_manager.py               # State Reconstruction Capsules (SRC)
│   │   ├── operation_registry.py        # External Operation Idempotency Registry
│   │   ├── civ.py                       # Counterfactual Integrity Validation (CIV)
│   │   └── recovery_controller.py       # Closed-Loop Recovery Orchestrator
│   ├── agent/
│   │   ├── state.py                     # Execution State Data Object & Signatures
│   │   ├── execution_graph.py           # Directed Causal Dependency Graph
│   │   ├── agent_simulator.py           # Stateful AI Agent Simulation Engine
│   │   └── scenarios.py                 # 4 Production Agent Scenarios
│   ├── integrity/
│   │   ├── integrity_monitor.py         # Runtime Integrity & Signature Verification
│   │   └── fault_injection.py           # 6 Controlled In-Flight Fault Injectors
│   ├── baselines/
│   │   ├── full_rollback.py             # Baseline 1: Full Rollback & Replay
│   │   ├── checkpoint_recovery.py       # Baseline 2: Checkpoint-Based Recovery
│   │   └── dependency_recovery.py       # Baseline 3: Dependency-Guided Rollback
│   ├── experiments/
│   │   ├── main_experiment.py           # Master 600-trial comparative suite
│   │   ├── ablation.py                  # 6-condition component ablation
│   │   ├── scalability.py               # Scaling evaluation across N=25..500
│   │   └── statistics.py                # Welch t-test, Mann-Whitney U, Cohen's d
│   └── visualization/
│       └── patent_figures.py            # High-resolution patent figure renderer (300 DPI)
├── tests/
│   └── test_bcrf.py                     # 48/48 comprehensive pytest validation suite
└── results/
    ├── master_results.csv               # 600 individual trial data records
    ├── summary_statistics.csv           # Mean, std, 95% confidence intervals
    ├── ablation_results.csv             # Component ablation data
    ├── fault_robustness.csv             # 6 fault type evaluation data
    ├── scalability_results.csv          # Scalability data across state sizes
    ├── experiment_metadata.json         # Python, OS, package versions & runtime
    ├── figures/                         # FIG1 through FIG12 (PNG + PDF)
    ├── tables/                          # Patent comparison tables (CSV + Markdown)
    └── reports/                         # Generated experimental reports
```

---

## ⚖️ Indian Patents Act Section 3(k) Compliance

Under the **CRI (Computer-Related Inventions) Guidelines** of the Indian Patent Office (IPO):
1. **Technical Problem Solved:** Resolves cascading state corruption, non-contiguous dependency recovery failures, and duplicate side-effect execution in distributed stateful AI systems.
2. **Technical Effect & Real-World Transformation:** Isolates causal dependencies via counterfactual graph perturbation, dynamically constructs non-contiguous execution boundaries, deduplicates external API side effects, and validates execution state integrity via closed-loop feedback.
3. **Hardware Synergy:** Interoperates with persistent cryptographic state storage, hardware registers, network transceivers, and distributed agent execution runtime engines.

---

## 📄 Citation

```bibtex
@patent{bcrf2026,
  title={System and Method for Autonomous Causal Recovery of Execution States in Stateful Artificial Intelligence Agents},
  author={Sajal},
  year={2026},
  note={Patent Pending (IPO / USPTO)}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
