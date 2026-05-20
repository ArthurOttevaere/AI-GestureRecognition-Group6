import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.config import DIR_TBL_STATS, DIR_FIG_STATS


# ==============================================================================
# 10b.  STATISTICAL TESTS
# ==============================================================================

def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """
    Wrapper around scipy.stats.wilcoxon (signed-rank) that handles the
    degenerate case where all paired differences are zero (raises in
    scipy) by returning p=1.0.
    NaNs are dropped pairwise prior to the test.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0 or np.allclose(a, b):
        warnings.warn("Wilcoxon: zero differences -> p set to 1.0",
                      RuntimeWarning, stacklevel=2)
        return 1.0
    try:
        _, p = wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        warnings.warn("Wilcoxon raised ValueError -> p set to 1.0",
                      RuntimeWarning, stacklevel=2)
        return 1.0
    return float(p)


def generate_pvalue_table(methods_gu: dict,
                           domain: int,
                           show: bool = False) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank test on the n=100 vectors of
    per-(gesture, user) accuracies (10 gestures x 10 users).
    Benjamini-Hochberg FDR correction (BH) is applied. Bonferroni is not
    used: BH is more appropriate here because the pairwise tests share
    methods and are therefore not independent (Benjamini & Hochberg 1995).
    Saves a square symmetric CSV of raw p-values and a heatmap PNG.

    Parameters
    ----------
    methods_gu : dict[str, np.ndarray]
        method name -> 100-vector of per-(gesture, user) accuracies.
    """
    names = list(methods_gu.keys())
    n     = len(names)

    pairs  = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_comp = max(len(pairs), 1)
    alpha  = 0.05

    raw_pvals = []
    for i, j in pairs:
        raw_pvals.append(_safe_wilcoxon(methods_gu[names[i]],
                                          methods_gu[names[j]]))

    if raw_pvals:
        reject_bh, pvals_bh, _, _ = multipletests(
            raw_pvals, alpha=alpha, method="fdr_bh")
    else:
        reject_bh, pvals_bh = [], []

    matrix = np.full((n, n), np.nan)
    for k_idx, (i, j) in enumerate(pairs):
        matrix[i, j] = raw_pvals[k_idx]
        matrix[j, i] = raw_pvals[k_idx]
    df_raw = pd.DataFrame(matrix, index=names, columns=names)

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Statistical tests | Domain {domain} | User-independent")
    print(f"  Pairs : {n_comp}  |  n = 100 (10 gestures x 10 users)")
    print(f"  Test  : Paired Wilcoxon signed-rank on per-(gesture, user) "
          f"accuracies")
    print(f"  Correction : Benjamini-Hochberg FDR @ {alpha*100:.0f}%")
    print(sep)
    print("\n  RAW Wilcoxon p-value matrix (symmetric):")
    print(df_raw.round(4).to_string())

    hdr = (f"\n  {'Method A':<18} {'Method B':<18} "
           f"{'p_raw':>9}  {'p_BH':>9}  {'BH sig':>7}")
    print(hdr)
    print("  " + "-" * 70)
    for k_idx, (i, j) in enumerate(pairs):
        na, nb = names[i], names[j]
        p_r    = raw_pvals[k_idx]
        p_bh   = float(pvals_bh[k_idx])
        bh_ok  = "YES *" if reject_bh[k_idx] else "no"
        print(f"  {na:<18} {nb:<18} {p_r:>9.4f}  {p_bh:>9.4f}  {bh_ok:>7}")

    means = {name: float(np.nanmean(gu))
             for name, gu in methods_gu.items()}
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

    csv_path = os.path.join(
        DIR_TBL_STATS,
        f"p_values_domain{domain}_user_independent.csv")
    df_raw.to_csv(csv_path)
    print(f"  Saved -> {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis_r", vmin=0.0, vmax=0.1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}",
                        ha="center", va="center",
                        color="white" if matrix[i, j] < 0.05 else "black",
                        fontsize=8)
    ax.set_title(f"Wilcoxon p-values (n=100) - Domain {domain}")
    plt.colorbar(im, ax=ax, label="p-value")
    plt.tight_layout()
    heatmap_path = os.path.join(
        DIR_FIG_STATS, f"p_values_heatmap_domain{domain}.png")
    plt.savefig(heatmap_path, dpi=150)
    print(f"  Saved -> {heatmap_path}")
    if show:
        plt.show()

    return df_raw
