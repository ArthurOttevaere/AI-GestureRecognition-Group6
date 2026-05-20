import os

import numpy as np
import pandas as pd

from src.config import (
    K_CLUSTERS, KNN_K,
    DIR_TBL_ABLATION, DIR_TBL_FOLDS, DIR_TBL_OVERFITTING,
)
from src.evaluation.crossval import (
    _ui_fold_indices, _ud_fold_indices,
    crossval_ui_edit, crossval_ud_edit,
    crossval_ui_dtw, crossval_ud_dtw,
    crossval_ui_dt, crossval_ud_dt,
    crossval_ui_rf, crossval_ud_rf,
    crossval_ui_lr, crossval_ud_lr,
    crossval_ui_dollar, crossval_ud_dollar,
)


# ==============================================================================
# 7.  ABLATION STUDY
# ==============================================================================

METHODS_ORDER = ["Edit Distance", "DTW", "DT", "RF", "LR", "$1"]
METHOD_KEY    = {"Edit Distance": "edit", "DTW": "dtw", "DT": "dt",
                  "RF": "rf", "LR": "lr", "$1": "dollar"}


def run_ablation_study(data_raw: list, data_std: list,
                        data_denoised: list, evr_list: list,
                        labels: list, users: list,
                        domain: int,
                        setting: str = "UI",
                        k_clusters: int = K_CLUSTERS,
                        k_clusters_raw: int = K_CLUSTERS,
                        knn_k: int = KNN_K
                        ) -> tuple[pd.DataFrame, dict]:
    """
    Compare six methods (Edit, DTW, DT, RF, LR, $1) under three
    preprocessing conditions on either the user-independent (UI) or the
    user-dependent (UD) cross-validation.

    Conditions
    ----------
    (a) No preprocessing
    (b) Standardisation only
    (c) Standardisation + per-gesture PCA denoising (full pipeline)

    Parameters
    ----------
    setting : str
        "UI" (default) or "UD".  Determines whether the leave-one-user-out
        folds (UI) or the leave-one-sample-out folds (UD) are used, and
        which `crossval_{ui,ud}_*` functions are called.
    k_clusters : int
        k-means K for Edit Distance in conditions (b) and (c), optimised
        via validation curve on standardised data.
    k_clusters_raw : int
        k-means K for Edit Distance in condition (a) only, optimised via
        validation curve on raw data. The two spaces have different distance
        scales, so the optimal K may differ (Linde et al. 1980, VQ theory).
    knn_k : int
        kNN K used by the DTW pipeline. Forced to 1 elsewhere in the
        pipeline; the parameter is kept for API stability.
    """
    setting = setting.upper()
    assert setting in ("UI", "UD"), f"Unknown setting {setting!r}"

    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY | Domain {domain} | "
          f"{'User-independent' if setting=='UI' else 'User-dependent'}")
    print(f"{'='*65}")

    if setting == "UI":
        folds = _ui_fold_indices(users)
        ed_fn, dtw_fn, dt_fn, rf_fn, lr_fn, dollar_fn = (
            crossval_ui_edit, crossval_ui_dtw, crossval_ui_dt,
            crossval_ui_rf, crossval_ui_lr,
            crossval_ui_dollar,
        )
    else:
        folds = _ud_fold_indices(labels, users)
        ed_fn, dtw_fn, dt_fn, rf_fn, lr_fn, dollar_fn = (
            crossval_ud_edit, crossval_ud_dtw, crossval_ud_dt,
            crossval_ud_rf, crossval_ud_lr,
            crossval_ud_dollar,
        )

    rows = []
    methods   = ["Edit Distance", "DTW", "DT", "RF", "LR", "$1"]
    cond_data = {
        "(a) No preprocessing" : (data_raw,      None),
        "(b) Standardisation"  : (data_std,       None),
        "(c) Std + PCA denoise": (data_denoised,  evr_list),
    }
    results: dict = {m: {} for m in methods}

    def _record(cond, method, mean, std, folds_accs, gu, train_accs=None,
                 note=""):
        rows.append({"Preprocessing": cond, "Method": method,
                     "Mean": mean, "Std": std, "Note": note})
        note_str = f"  [{note}]" if note else ""
        print(f"    [{cond}] {method}: {mean:.3f} +/- {std:.3f}{note_str}")
        results[method][cond] = (mean, std, folds_accs, gu, train_accs)

    def _unpack(ret):
        """Normalise 4- and 5-tuple returns (parametric classifiers
        return a 5-tuple including train_accs; baseline 1-NN methods
        return a 4-tuple). Returns (mean, std, folds, gu, train_accs|None).
        """
        if len(ret) == 5:
            return ret[0], ret[1], ret[2], ret[3], ret[4]
        return ret[0], ret[1], ret[2], ret[3], None

    # ---- Condition (a) -----------------------------------------------------
    print("\n  (a) No preprocessing")
    m, s, f, gu, tr_accs = _unpack(ed_fn(data_raw, labels, users, folds,
                                  k_clusters=k_clusters_raw))
    _record("(a) No preprocessing", "Edit Distance", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(dtw_fn(data_raw, labels, users, folds,
                                   knn_k=knn_k))
    _record("(a) No preprocessing", "DTW", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(dt_fn(data_raw, labels, users, folds,
                                  evr_list=None, tag=" [no EVR]", domain=domain))
    _record("(a) No preprocessing", "DT", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(rf_fn(data_raw, labels, users, folds,
                                  evr_list=None, tag=" [no EVR]", domain=domain))
    _record("(a) No preprocessing", "RF", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(lr_fn(data_raw, labels, users, folds,
                                  evr_list=None, tag=" [no EVR]", domain=domain))
    _record("(a) No preprocessing", "LR", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(dollar_fn(data_raw, labels, users, folds))
    _record("(a) No preprocessing", "$1", m, s, f, gu, train_accs=tr_accs)

    # ---- Condition (b) -----------------------------------------------------
    print("\n  (b) Standardisation only")
    m, s, f, gu, tr_accs = _unpack(ed_fn(data_std, labels, users, folds,
                                  k_clusters=k_clusters))
    _record("(b) Standardisation", "Edit Distance", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(dtw_fn(data_std, labels, users, folds,
                                   knn_k=knn_k))
    _record("(b) Standardisation", "DTW", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(dt_fn(data_std, labels, users, folds,
                                  evr_list=None, tag=" [no EVR]", domain=domain))
    _record("(b) Standardisation", "DT", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(rf_fn(data_std, labels, users, folds,
                                  evr_list=None, tag=" [no EVR]", domain=domain))
    _record("(b) Standardisation", "RF", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(lr_fn(data_std, labels, users, folds,
                                  evr_list=None, tag=" [no EVR]", domain=domain))
    _record("(b) Standardisation", "LR", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(dollar_fn(data_std, labels, users, folds))
    _record("(b) Standardisation", "$1", m, s, f, gu, train_accs=tr_accs)

    # ---- Condition (c) -----------------------------------------------------
    print("\n  (c) Standardisation + PCA denoising 3D->2D->3D")
    m, s, f, gu, tr_accs = _unpack(ed_fn(data_denoised, labels, users, folds,
                                  k_clusters=k_clusters))
    _record("(c) Std + PCA denoise", "Edit Distance", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(dtw_fn(data_denoised, labels, users, folds,
                                   knn_k=knn_k))
    _record("(c) Std + PCA denoise", "DTW", m, s, f, gu, train_accs=tr_accs)
    m, s, f, gu, tr_accs = _unpack(dt_fn(data_denoised, labels, users, folds,
                                  evr_list=evr_list, tag=" [+EVR]", domain=domain))
    _record("(c) Std + PCA denoise", "DT", m, s, f, gu, train_accs=tr_accs,
            note="3 PCA EVR values added to DT feature vector")
    m, s, f, gu, tr_accs = _unpack(rf_fn(data_denoised, labels, users, folds,
                                  evr_list=evr_list, tag=" [+EVR]", domain=domain))
    _record("(c) Std + PCA denoise", "RF", m, s, f, gu, train_accs=tr_accs,
            note="3 PCA EVR values added to RF feature vector")
    m, s, f, gu, tr_accs = _unpack(lr_fn(data_denoised, labels, users, folds,
                                  evr_list=evr_list, tag=" [+EVR]", domain=domain))
    _record("(c) Std + PCA denoise", "LR", m, s, f, gu, train_accs=tr_accs,
            note="3 PCA EVR values added to LR feature vector")
    m, s, f, gu, tr_accs = _unpack(dollar_fn(data_denoised, labels, users, folds))
    _record("(c) Std + PCA denoise", "$1", m, s, f, gu, train_accs=tr_accs)

    best_preprocessing: dict = {}

    print(f"\n  {'-'*60}")
    print(f"  Best preprocessing per method - Domain {domain} ({setting}):")
    print(f"  {'-'*60}")

    for method in methods:
        best_cond = max(results[method], key=lambda c: results[method][c][0])
        (best_mean, best_std, best_folds,
         best_gu, best_train_accs) = results[method][best_cond]
        best_data, best_evr = cond_data[best_cond]

        if method in ("RF", "DT", "LR") and best_cond != "(c) Std + PCA denoise":
            best_evr = None

        best_preprocessing[method] = {
            "condition" : best_cond,
            "data"      : best_data,
            "evr"       : best_evr,
            "mean"      : best_mean,
            "std"       : best_std,
            "folds"     : best_folds,
            "gu"        : best_gu,
            "train_accs": best_train_accs,
            "all_results": results[method],
        }
        print(f"    {method:<16}: {best_cond}  "
              f"(mean acc = {best_mean:.3f} +/- {best_std:.3f})")

    df = pd.DataFrame(rows)
    df["Result"] = (df["Mean"].map("{:.3f}".format)
                    + " +/- " + df["Std"].map("{:.3f}".format))
    pivot = df.pivot_table(index="Preprocessing", columns="Method",
                           values="Result", aggfunc="first")
    print(f"\n  Ablation summary - Domain {domain} ({setting}):")
    print(pivot.to_string())
    setting_tag = setting.lower()
    csv_path = os.path.join(
        DIR_TBL_ABLATION,
        f"ablation_domain{domain}_{setting_tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    return df, best_preprocessing


# ==============================================================================
# 12.  RESULT SAVING
# ==============================================================================

def save_overfitting_table(rows: list, csv_path: str) -> pd.DataFrame:
    """
    Persist the train-vs-test gap analysis as a CSV and print a tidy
    summary. `rows` is a list of dicts with keys:
        Domain, Setting, Method, TrainAcc, TestAcc, Gap.
    """
    df = pd.DataFrame(rows)
    df["TrainAcc"] = df["TrainAcc"].astype(float)
    df["TestAcc"]  = df["TestAcc"].astype(float)
    df["Gap"]      = df["TrainAcc"] - df["TestAcc"]
    df = df[["Domain", "Setting", "Method",
             "TrainAcc", "TestAcc", "Gap"]]
    df.to_csv(csv_path, index=False)

    print("\n  Train-vs-test gap (only parametric classifiers; "
          "DTW/Edit/$1 are 1-NN, train acc = 1.0 by construction):")
    print("  " + "-" * 65)
    for _, r in df.iterrows():
        flag = "  <-- higher gap" if r["Gap"] > 0.10 else ""
        print(f"   D{int(r['Domain'])}-{r['Setting']:<2} {r['Method']:<3} "
              f"train={r['TrainAcc']:.3f}  test={r['TestAcc']:.3f}  "
              f"gap={r['Gap']:+.3f}{flag}")
    print(f"  Saved -> {csv_path}")
    return df


def save_fold_results(fold_accs: list, method: str,
                       setting: str, domain: int) -> None:
    fname = os.path.join(
        DIR_TBL_FOLDS,
        f"results_domain{domain}_{setting}_{method}.csv")
    pd.DataFrame({"accuracy": fold_accs}).to_csv(fname, index=False)
    print(f"  Saved -> {fname}")


# ==============================================================================
# Helpers extracted from __main__ (no closed-over variables)
# ==============================================================================

def _force_dollar_raw(best_prep: dict, data_raw_dom: list) -> None:
    # Wobbrock et al. (2007) and Kratz & Rohs (2010) prescribe that $1
    # operates on raw input points (the algorithm contains its own
    # internal preprocessing: resample, centroid translation, indicative-
    # axis rotation, uniform cube scaling).  Feeding externally-standardised
    # data into $1 alters the geometry used by the cross-product rotation
    # step and is methodologically inconsistent.  We therefore override
    # the ablation auto-selection for $1 and force condition (a) -- the
    # ablation table is kept for transparency.
    if "$1" not in best_prep:
        return
    raw_cond = "(a) No preprocessing"
    raw_entry = best_prep["$1"]["all_results"].get(raw_cond)
    if raw_entry is None:
        return
    m_raw, s_raw, f_raw, gu_raw, tr_raw = raw_entry
    prev = best_prep["$1"]["condition"]
    if prev != raw_cond:
        print(f"  [policy] $1 forced to '{raw_cond}' (was '{prev}') "
              f"per Wobbrock 2007 / Kratz & Rohs 2010.")
    best_prep["$1"] = {
        "condition" : raw_cond,
        "data"      : data_raw_dom,
        "evr"       : None,
        "mean"      : m_raw,
        "std"       : s_raw,
        "folds"     : f_raw,
        "gu"        : gu_raw,
        "train_accs": tr_raw,
        "all_results": best_prep["$1"]["all_results"],
    }


def _save_preproc_comparison(bp_ui: dict, bp_ud: dict,
                               domain: int) -> None:
    rows = []
    for m in ["Edit Distance", "DTW", "DT", "RF", "LR", "$1"]:
        rows.append({
            "Domain"      : domain,
            "Method"      : m,
            "BestPrepUI"  : bp_ui[m]["condition"],
            "MeanUI"      : bp_ui[m]["mean"],
            "BestPrepUD"  : bp_ud[m]["condition"],
            "MeanUD"      : bp_ud[m]["mean"],
            "Divergent"   : bp_ui[m]["condition"] != bp_ud[m]["condition"],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(DIR_TBL_ABLATION,
                         f"preproc_ui_vs_ud_domain{domain}.csv")
    df.to_csv(path, index=False)
    print(f"\n  UI-vs-UD best-preprocessing comparison (Domain {domain}):")
    for _, r in df.iterrows():
        flag = "  <-- DIVERGENT" if r["Divergent"] else ""
        print(f"    {r['Method']:<14} UI: {r['BestPrepUI']:<22}"
              f" UD: {r['BestPrepUD']:<22}{flag}")
    print(f"  Saved -> {path}")


def _print_and_save_phase(domain: int, setting_tag: str,
                            best_prep: dict) -> dict:
    """Print the per-method summary and save fold accuracies. Returns
    a flat dict for downstream stats/summary use."""
    out = {}
    print(f"\n=== Main Evaluation - Domain {domain} - "
          f"{'User-Independent' if setting_tag=='UI' else 'User-Dependent'} "
          f"(from ablation) ===")
    for m in METHODS_ORDER:
        entry = best_prep[m]
        print(f"  {m:<16}: {entry['condition']:<22} "
              f"mean = {entry['mean']:.3f} +/- {entry['std']:.3f}")
        save_fold_results(
            entry["folds"], METHOD_KEY[m],
            "user_independent" if setting_tag == "UI" else "user_dependent",
            domain)
        out[m] = entry
    return out
