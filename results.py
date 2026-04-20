"""
11. RESULT SAVING & STATISTICAL TESTS
=======================================
CSV export of fold accuracies, Wilcoxon signed-rank tests, and p-value tables
with Benjamini-Hochberg FDR correction.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from config import DATA_DIR


def save_fold_results(fold_accs: list, method: str,
                      setting: str, domain: int) -> None:
    """Save per-fold accuracies to CSV for later statistical tests."""
    filename = os.path.join(DATA_DIR,
                            f"results_domain{domain}_{setting}_{method}.csv")
    pd.DataFrame({"accuracy": fold_accs}).to_csv(filename, index=False)
    print(f"  Results saved -> {filename}")


def generate_pvalue_table(methods_results: dict,
                           domain: int) -> pd.DataFrame:
    """
    All-pairs Wilcoxon signed-rank tests with raw p-values and
    Benjamini-Hochberg (BH) FDR correction.

    Required by the course guidelines for the user-independent setting.

    Why Benjamini-Hochberg rather than Bonferroni
    ---------------------------------------------
    Bonferroni controls the Family-Wise Error Rate: threshold alpha/m =
    0.05/6 ~= 0.0083.  It is conservative and loses statistical power
    when m is moderate.  BH controls the False Discovery Rate (expected
    proportion of false positives among rejected hypotheses).  It is the
    standard correction for exploratory method comparisons in the ML
    literature and offers more power while maintaining principled error
    control.  Both corrections are printed for full transparency.

    Parameters
    ----------
    methods_results : dict  method_name -> list[float]  (fold accuracies)
    domain          : 1 or 4

    Returns
    -------
    DataFrame of raw p-values (symmetric, diagonal = 1.0).
    """
    names = list(methods_results.keys())
    n     = len(names)

    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if len(methods_results[names[i]]) ==
           len(methods_results[names[j]])
    ]
    n_comp     = max(len(pairs), 1)
    alpha_bonf = 0.05 / n_comp

    pair_labels, raw_pvals = [], []
    for i, j in pairs:
        _, p = wilcoxon(methods_results[names[i]],
                        methods_results[names[j]])
        pair_labels.append((names[i], names[j]))
        raw_pvals.append(float(p))

    # Benjamini-Hochberg correction
    if raw_pvals:
        reject_bh, pvals_bh, _, _ = multipletests(
            raw_pvals, alpha=0.05, method="fdr_bh")
    else:
        reject_bh = []
        pvals_bh  = []

    # Build symmetric raw p-value matrix
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)
    for k_idx, (i, j) in enumerate(pairs):
        matrix[i, j] = raw_pvals[k_idx]
        matrix[j, i] = raw_pvals[k_idx]
    df_raw = pd.DataFrame(matrix, index=names, columns=names)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Statistical tests | Domain {domain} | User-independent")
    print(f"  Pairs : {n_comp}  (C(4,2)=6)  |  n=10 per method")
    print(f"  Bonferroni threshold : alpha/{n_comp} = {alpha_bonf:.4f}")
    print(f"  BH correction        : FDR controlled at 5%")
    print(sep)

    print("\n  RAW p-value matrix (Wilcoxon signed-rank test):")
    print(df_raw.round(4).to_string())

    print(f"\n  {'Method A':<20} {'Method B':<20} "
          f"{'p-value':>9}  {'p-BH':>9}  "
          f"{'Bonf. sig.':>12}  {'BH sig.':>9}")
    print("  " + "-" * 78)
    for k_idx, (i, j) in enumerate(pairs):
        na, nb   = names[i], names[j]
        p_raw    = raw_pvals[k_idx]
        p_bh_val = float(pvals_bh[k_idx])
        bonf_ok  = "YES *" if p_raw < alpha_bonf  else "no"
        bh_ok    = "YES *" if reject_bh[k_idx]    else "no"
        print(f"  {na:<20} {nb:<20} {p_raw:>9.4f}  "
              f"{p_bh_val:>9.4f}  {bonf_ok:>12}  {bh_ok:>9}")

    # Identify best method and check significance
    means    = {name: np.mean(folds) for name, folds in methods_results.items()}
    best     = max(means, key=means.get)
    best_idx = names.index(best)
    print(f"\n  Best mean accuracy: {best} ({means[best]:.3f})")

    all_sig = True
    for k_idx, (i, j) in enumerate(pairs):
        if best_idx in (i, j) and not reject_bh[k_idx]:
            other = names[j] if i == best_idx else names[i]
            print(f"  -> {best} NOT significantly better than {other} "
                  f"(BH p={pvals_bh[k_idx]:.4f})")
            all_sig = False
    if all_sig and len(pairs) > 0:
        print(f"  -> {best} is significantly better than ALL others (BH)")

    out_path = os.path.join(DATA_DIR,
                            f"p_values_domain{domain}_user_independent.csv")
    df_raw.to_csv(out_path)
    print(f"  Saved -> {out_path}")
    return df_raw
