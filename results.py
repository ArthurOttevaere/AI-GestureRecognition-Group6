"""
11. RESULT SAVING & STATISTICAL TESTS
=======================================
CSV export of fold accuracies, Wilcoxon signed-rank tests, and p-value tables.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import DATA_DIR


def save_fold_results(fold_accs: list, method: str,
                      setting: str, domain: int) -> None:
    """Save per-fold accuracies to CSV for later statistical tests."""
    filename = os.path.join(DATA_DIR,
                            f"results_domain{domain}_{setting}_{method}.csv")
    pd.DataFrame({"accuracy": fold_accs}).to_csv(filename, index=False)
    print(f"  Results saved → {filename}")


def wilcoxon_test(folds_a: list, folds_b: list,
                  name_a: str, name_b: str, domain: int) -> None:
    """Wilcoxon signed-rank test; skips when fold counts differ."""
    if len(folds_a) != len(folds_b):
        print(f"\n  [SKIP] Wilcoxon {name_a} vs {name_b}: "
              f"unequal fold counts ({len(folds_a)} vs {len(folds_b)})")
        return

    stat, p = wilcoxon(folds_a, folds_b)
    print(f"\n  Wilcoxon – {name_a} vs {name_b} | Domain {domain}")
    print(f"  statistic = {stat:.4f} | p-value = {p:.4f}")
    if p < 0.05:
        better = name_a if np.mean(folds_a) > np.mean(folds_b) else name_b
        print(f"  → Significant difference (p<0.05): {better} is better")
    else:
        print(f"  → No significant difference (p={p:.4f})")


def generate_pvalue_table(methods_results: dict, domain: int) -> pd.DataFrame:
    """
    All-pairs Wilcoxon signed-rank tests with Bonferroni correction.

    Skips pairs whose fold vectors have different lengths.
    Prints Bonferroni-corrected significance threshold alongside the
    raw p-value matrix.

    Parameters
    ----------
    methods_results : dict  method_name → list[float]  (fold accuracies)
    domain          : 1 or 4
    """
    names = list(methods_results.keys())
    n     = len(names)

    eligible_pairs = [
        (i, j)
        for i in range(n) for j in range(i + 1, n)
        if len(methods_results[names[i]]) == len(methods_results[names[j]])
    ]
    n_comparisons   = max(len(eligible_pairs), 1)
    alpha_corrected = 0.05 / n_comparisons

    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)

    for i, j in eligible_pairs:
        _, p = wilcoxon(methods_results[names[i]], methods_results[names[j]])
        matrix[i, j] = p
        matrix[j, i] = p

    df_pvals = pd.DataFrame(matrix, index=names, columns=names)
    print(f"\n=== P-value table (Wilcoxon) | Domain {domain} ===")
    print(f"  Eligible comparisons   : {n_comparisons}")
    print(f"  Bonferroni-corrected α : {alpha_corrected:.4f}")
    print(df_pvals.round(4))
    df_pvals.to_csv(os.path.join(DATA_DIR, f"p_values_domain{domain}.csv"))
    return df_pvals
