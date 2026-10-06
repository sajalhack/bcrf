# BCRF Experimental Validation Report
## System and Method for Autonomous Causal Recovery of Execution States in Stateful AI Agents
### Bidirectional Causal Repair Frontier (BCRF)

---

**Generated:** 2026-09-14T00:26:10.537620
**Experiment Seed:** 42
**Python Version:** 3.11.5 (v3.11.5:cce6ba91b3, Aug 24 2023, 10:50:31) [Clang 13.0.0 (clang-1300.0.29.30)]
**Total Trials:** 600

---

## 1. Experimental Objective

This report presents the experimental validation of the BCRF (Bidirectional Causal Repair
Frontier) framework — a system for autonomous causal recovery of execution states in stateful
AI agents. The experiments answer nine research questions (RQ1–RQ9) pertaining to recovery
scope, latency, duplicate side-effect prevention, causal accuracy, validation correctness,
scalability, fault robustness, and resource tradeoffs.

---

## 2. System Description

The experimental system is a synthetic stateful AI-agent simulator implementing the complete
BCRF pipeline:

```
Stateful Agent Execution
→ Runtime Integrity Monitoring
→ Integrity Violation Detection
→ Forward Counterfactual Influence (FCI) + Backward Causal Responsibility (BCR)
→ Causal Repair Frontier (CRF)
→ External Operation Registry Check
→ State Reconstruction Capsule (SRC) Reconstruction
→ Counterfactual Integrity Validation (CIV)
→ Resume or Expand CRF + Repair Again
```

---

## 3. BCRF Implementation

| Component | Implementation |
|-----------|---------------|
| FCI | `FCIᵢ = |V(Sᵢ) − V(Sᵢᶜᶠ)|` via graph-based counterfactual simulation |
| BCR | `BCRᵢ = P(Sᵢ causal to V|E)` via backward traversal + path analysis |
| CRF | `{Si | FCIi >= T_F AND BCRi >= T_B}` with adaptive thresholds |
| SRC | State snapshots with capsule integrity verification |
| Registry | Idempotency-key deduplication preventing duplicate external ops |
| CIV | `CIV = w₁O + w₂I + w₃C`, w₁=0.4, w₂=0.35, w₃=0.25 |

---

## 4. Baseline Methods

| Baseline | Description |
|----------|-------------|
| Full Rollback | Rolls back to start; replays ALL states |
| Checkpoint Recovery | Restores latest checkpoint before corruption |
| Dependency-Guided | Selects ancestors by dependency only (no FCI/BCR) |
| **BCRF** | Full proposed method with all components |

---

## 5. Experimental Setup

- **Scenarios:** Order Processing, Workflow Automation, Data Analysis, Multi-Step Planning
- **State sizes:** [25, 50, 100, 250, 500]
- **Trials per configuration:** 30
- **Fault types:** memory, planning, tool, dependency, metadata, combined
- **Fault rate:** 0.1
- **Base seed:** 42

---

## 6. Main Results

### Recovery Success Rate

| Method | Success Rate |
|--------|-------------|
| Full Rollback | 100.0% |
| Checkpoint | 100.0% |
| Dependency-Guided | 100.0% |
| **BCRF** | **100.0%** |

### Recovery Latency (ms)

| Method | Mean ± Std |
|--------|-----------|
| Full Rollback | 13.759 ± 21.816 |
| **BCRF** | **54.287 ± 94.887** |

### States Reconstructed

| Method | Mean ± Std |
|--------|-----------|
| Full Rollback | 185.000 ± 175.931 |
| **BCRF** | **87.812 ± 84.808** |

### Duplicate Side Effects

| Method | Mean ± Std |
|--------|-----------|
| Full Rollback | 36.450 ± 40.746 |
| **BCRF** | **0.000 ± 0.000** |

### CIV Score (BCRF only)

| Metric | Value |
|--------|-------|
| Mean CIV Score | 0.896 ± 0.003 |

---

## 7. Research Questions — Summary

**RQ1 (Recovery scope):** BCRF reconstructed fewer states than full rollback/replay
(mean: 87.812 ± 84.808 vs 185.000 ± 175.931).

**RQ2 (Recovery latency):** BCRF recovery latency was 54.287 ± 94.887 ms vs
13.759 ± 21.816 ms for full rollback.

**RQ3 (Unnecessary re-execution):** BCRF's CRF threshold mechanism limits
reconstruction to causally relevant states only.

**RQ4 (Duplicate side effects):** BCRF with External Operation Registry:
0.000 ± 0.000 duplicates; Full Rollback: 36.450 ± 40.746.

**RQ5 (Bidirectional analysis):** See ablation study — removing FCI or BCR
changes CRF composition and recovery metrics.

**RQ6 (CIV correctness):** Mean CIV score = 0.896 ± 0.003. Trials requiring
CRF expansion are logged in recovery attempt records.

**RQ7 (Scalability):** See Figure 11 and scalability_results.csv.

**RQ8 (Fault robustness):** See Figure 12 and fault_robustness.csv.

**RQ9 (Resource tradeoff):** See patent_comparison_table.csv.

---

## 8. Ablation Results

See results/ablation_results.csv and Figure 8 for detailed component contribution analysis.

Ablation conditions tested:
- Full BCRF (all components)
- Without FCI
- Without BCR
- Without External Operation Registry
- Without CIV
- Without CRF Expansion

---

## 9. Fault Robustness

See results/fault_robustness.csv and Figure 12.

Fault types evaluated:
memory_corruption, planning_corruption, tool_corruption, dependency_corruption, metadata_corruption, combined_corruption

---

## 10. Statistical Analysis

Statistical tests applied:
- **Welch's t-test**: Two-sample comparison of BCRF vs. each baseline (unequal variance assumed).
- **Mann-Whitney U**: Non-parametric confirmatory test.
- **Cohen's d**: Effect size measurement.
- **Kruskal-Wallis**: Multi-group non-parametric comparison.

See results/processed/statistical_analysis.json for complete test results.

---

## 11. Reproducibility

All experiments use deterministic random seeds. To reproduce:

```bash
python run_all_experiments.py --seed 42
```

Experiment metadata: results/experiment_metadata.json

---

## 12. Limitations

1. The agent simulator is synthetic; results may differ on real LLM-based agents.
2. Recovery success is defined as CIV >= threshold; the threshold is configurable.
3. External operation registry effectiveness depends on correct idempotency key assignment.
4. Scalability experiments use single-machine execution.

---

## 13. Conclusion

The BCRF experimental validation demonstrates the technical effects of bidirectional causal
analysis, selective state reconstruction, external operation deduplication, and closed-loop
integrity validation in a controlled stateful AI agent environment. All results are derived
from actual experiment execution; no values are fabricated or hard-coded.

---

*Report auto-generated by run_all_experiments.py*
