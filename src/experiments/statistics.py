"""
BCRF Experiment - Statistical Analysis Module
Computes means, medians, std, confidence intervals, and statistical tests.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple, Optional


def descriptive_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, median, std, min, max, 95% CI."""
    if not values:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0,
                "ci_lower": 0, "ci_upper": 0, "n": 0}
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    mn = float(np.min(arr))
    mx = float(np.max(arr))

    # 95% confidence interval (t-distribution for small samples)
    if n > 1:
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    else:
        ci_lower = mean
        ci_upper = mean

    return {
        "mean": round(mean, 6),
        "median": round(median, 6),
        "std": round(std, 6),
        "min": round(mn, 6),
        "max": round(mx, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "n": n,
    }


def welch_t_test(a: List[float], b: List[float]) -> Dict[str, Any]:
    """
    Welch's t-test for two independent samples with potentially unequal variances.

    Appropriate when comparing two methods (e.g., BCRF vs. Full Rollback)
    on the same metric across multiple trials.
    """
    if len(a) < 2 or len(b) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "test": "welch_t", "reason": "insufficient_samples"}
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return {
        "statistic": round(float(t_stat), 6),
        "p_value": round(float(p_val), 6),
        "significant": p_val < 0.05,
        "test": "welch_t",
        "reason": "two_independent_samples_potentially_unequal_variance",
    }


def mann_whitney_test(a: List[float], b: List[float]) -> Dict[str, Any]:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Appropriate when distribution normality cannot be assumed.
    Used as a secondary confirmatory test.
    """
    if len(a) < 2 or len(b) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "test": "mann_whitney", "reason": "insufficient_samples"}
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "statistic": round(float(u_stat), 6),
        "p_value": round(float(p_val), 6),
        "significant": p_val < 0.05,
        "test": "mann_whitney",
        "reason": "non_parametric_two_sample_comparison",
    }


def cohens_d(a: List[float], b: List[float]) -> float:
    """
    Cohen's d effect size.

    |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return round(float((mean_a - mean_b) / pooled_std), 6)


def kruskal_wallis_test(groups: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Kruskal-Wallis test for multiple group comparison.

    Appropriate when comparing 3+ methods simultaneously.
    Non-parametric; does not assume normality.
    """
    group_lists = [v for v in groups.values() if len(v) >= 2]
    if len(group_lists) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "test": "kruskal_wallis", "reason": "insufficient_groups"}
    try:
        stat, p = stats.kruskal(*group_lists)
    except ValueError as e:
        # Occurs when all values are identical across groups
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "test": "kruskal_wallis", "reason": f"identical_values: {e}",
                "n_groups": len(group_lists)}
    return {
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "significant": p < 0.05,
        "test": "kruskal_wallis",
        "reason": "multi_group_non_parametric_comparison",
        "n_groups": len(group_lists),
    }


def compute_all_statistics(
    method_results: Dict[str, List[float]],
    metric_name: str,
) -> Dict[str, Any]:
    """
    Compute full statistical analysis for multiple methods on one metric.

    Returns descriptive stats, pairwise tests, and multi-group test.
    """
    output = {"metric": metric_name, "methods": {}}

    # Descriptive stats
    for method, values in method_results.items():
        output["methods"][method] = descriptive_stats(values)

    # Pairwise Welch's t-test and Mann-Whitney (BCRF vs each baseline)
    bcrf_vals = method_results.get("bcrf", [])
    output["pairwise_tests"] = {}

    for baseline in ["full_rollback", "checkpoint", "dependency"]:
        baseline_vals = method_results.get(baseline, [])
        if bcrf_vals and baseline_vals:
            output["pairwise_tests"][f"bcrf_vs_{baseline}"] = {
                "welch_t": welch_t_test(bcrf_vals, baseline_vals),
                "mann_whitney": mann_whitney_test(bcrf_vals, baseline_vals),
                "cohens_d": cohens_d(bcrf_vals, baseline_vals),
            }

    # Multi-group Kruskal-Wallis
    all_groups = {k: v for k, v in method_results.items() if v}
    if len(all_groups) >= 2:
        output["kruskal_wallis"] = kruskal_wallis_test(all_groups)

    return output
