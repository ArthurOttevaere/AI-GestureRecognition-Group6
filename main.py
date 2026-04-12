"""
MLSM2154 – Artificial Intelligence: Gesture Recognition Project
===============================================================
Phase 1 : Data loading & exploratory analysis
Phase 2 : Pre-processing (standardisation, PCA, k-means clustering)
Phase 3 : Baseline methods (DTW, Edit Distance) + k-NN classifier
Phase 4 : Advanced methods (Random Forest, LSTM)
Phase 5 : Cross-validation (user-independent & user-dependent)
Phase 6 : Statistical tests (Wilcoxon + Bonferroni)

Authors  : <Andry Lenny / El Mohcine Mohamed / Ottevaere Arthur>
Group    : <Group 6>
Date     : 2026
"""

import os
import numpy as np
import pandas as pd

from config import DOMAIN1_DIR, DOMAIN4_DIR, DATA_DIR
from data_loading import load_domain1, load_domain4, print_dataset_info
from visualization import plot_sequence_lengths, plot_gesture_samples
from preprocessing import standardize_gestures, apply_pca, cluster_and_encode
from distance_metrics import dtw_distance, edit_distance
from crossvalidation import crossval_user_independent, crossval_user_dependent
from random_forest import (random_forest_evaluation,
                            random_forest_evaluation_user_dependent)
from lstm_model import lstm_evaluation, lstm_evaluation_user_dependent
from evaluation import draw_best_model_cm
from results import save_fold_results, wilcoxon_test, generate_pvalue_table


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 0.  Warm-up numba JIT
    # ------------------------------------------------------------------
    print("Warming up numba JIT …", end=" ", flush=True)
    _d = np.random.randn(10, 3)
    dtw_distance(_d, _d)
    _s = np.zeros(10, dtype=np.int64)
    edit_distance(_s, _s)
    print("done.")

    # ------------------------------------------------------------------
    # 1.  Load data
    # ------------------------------------------------------------------
    print("\n=== Loading Domain 1 ===")
    data1, labels1, users1 = load_domain1(DOMAIN1_DIR)
    print_dataset_info(data1, labels1, users1, "Domain 1")

    print("\n=== Loading Domain 4 ===")
    data4, labels4, users4 = load_domain4(DOMAIN4_DIR)
    print_dataset_info(data4, labels4, users4, "Domain 4")

    # ------------------------------------------------------------------
    # 2.  Exploratory visualisation – Domain 1 AND Domain 4
    # ------------------------------------------------------------------
    print("\n=== Exploratory visualisation – Domain 1 ===")
    plot_sequence_lengths(data1, labels1, "Domain 1",
                          save_path=os.path.join(DATA_DIR, "d1_sequence_lengths.png"))
    plot_gesture_samples(data1, labels1, users1, "Domain 1",
                         save_path=os.path.join(DATA_DIR, "d1_gesture_samples.png"))

    print("\n=== Exploratory visualisation – Domain 4 ===")
    plot_sequence_lengths(data4, labels4, "Domain 4",
                          save_path=os.path.join(DATA_DIR, "d4_sequence_lengths.png"))
    plot_gesture_samples(data4, labels4, users4, "Domain 4",
                         save_path=os.path.join(DATA_DIR, "d4_gesture_samples.png"))

    # ------------------------------------------------------------------
    # 3.  Pre-processing – Domain 1
    # ------------------------------------------------------------------
    print("\n=== Pre-processing (Domain 1) ===")
    data1_std = standardize_gestures(data1)
    print("  Standardisation done.")

    print("  Fitting PCA …")
    data1_pca, pca1 = apply_pca(data1_std, n_components=2)

    K = 20
    print(f"  Fitting k-means (k={K}) …")
    sequences1, kmeans1 = cluster_and_encode(data1_pca, k=K)
    print(f"  Example sequence (first 20): {sequences1[0][:20]}")

    # ------------------------------------------------------------------
    # 4.  Baselines – Domain 1 – USER-INDEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Edit Distance | User-independent | Domain 1 ===")
    mean_ed, std_ed, folds_ed = crossval_user_independent(
        sequences1, labels1, users1, edit_distance, k=1)
    print(f"  → Mean accuracy: {mean_ed:.3f} ± {std_ed:.3f}")
    save_fold_results(folds_ed, "edit", "user_independent", 1)

    print("\n=== DTW | User-independent | Domain 1 ===")
    mean_dtw, std_dtw, folds_dtw = crossval_user_independent(
        data1_std, labels1, users1, dtw_distance, k=1)
    print(f"  → Mean accuracy: {mean_dtw:.3f} ± {std_dtw:.3f}")
    save_fold_results(folds_dtw, "dtw", "user_independent", 1)

    # ------------------------------------------------------------------
    # 5.  Advanced methods – Domain 1 – USER-INDEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Random Forest | User-independent | Domain 1 ===")
    mean_rf, std_rf, folds_rf = random_forest_evaluation(
        data1_std, labels1, users1)
    print(f"  → Mean accuracy: {mean_rf:.3f} ± {std_rf:.3f}")
    save_fold_results(folds_rf, "rf", "user_independent", 1)

    print("\n=== LSTM | User-independent | Domain 1 ===")
    mean_lstm, std_lstm, folds_lstm = lstm_evaluation(
        data1_std, labels1, users1)
    print(f"  → Mean accuracy: {mean_lstm:.3f} ± {std_lstm:.3f}")
    save_fold_results(folds_lstm, "lstm", "user_independent", 1)

    # ------------------------------------------------------------------
    # 6.  Baselines – Domain 1 – USER-DEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Edit Distance | User-dependent | Domain 1 ===")
    mean_ed_ud, std_ed_ud, folds_ed_ud = crossval_user_dependent(
        sequences1, labels1, users1, edit_distance, k=1)
    print(f"  → Mean accuracy: {mean_ed_ud:.3f} ± {std_ed_ud:.3f}")
    save_fold_results(folds_ed_ud, "edit", "user_dependent", 1)

    print("\n=== DTW | User-dependent | Domain 1 ===")
    mean_dtw_ud, std_dtw_ud, folds_dtw_ud = crossval_user_dependent(
        data1_std, labels1, users1, dtw_distance, k=1)
    print(f"  → Mean accuracy: {mean_dtw_ud:.3f} ± {std_dtw_ud:.3f}")
    save_fold_results(folds_dtw_ud, "dtw", "user_dependent", 1)

    # ------------------------------------------------------------------
    # 7.  Advanced methods – Domain 1 – USER-DEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Random Forest | User-dependent | Domain 1 ===")
    mean_rf_ud, std_rf_ud, folds_rf_ud = \
        random_forest_evaluation_user_dependent(data1_std, labels1, users1)
    print(f"  → Mean accuracy: {mean_rf_ud:.3f} ± {std_rf_ud:.3f}")
    save_fold_results(folds_rf_ud, "rf", "user_dependent", 1)

    print("\n=== LSTM | User-dependent | Domain 1 ===")
    mean_lstm_ud, std_lstm_ud, folds_lstm_ud = \
        lstm_evaluation_user_dependent(data1_std, labels1, users1)
    print(f"  → Mean accuracy: {mean_lstm_ud:.3f} ± {std_lstm_ud:.3f}")
    save_fold_results(folds_lstm_ud, "lstm", "user_dependent", 1)

    # ------------------------------------------------------------------
    # 8.  Statistical tests – Domain 1 – user-independent
    # ------------------------------------------------------------------
    results_ui_d1 = {
        "Edit" : folds_ed,
        "DTW"  : folds_dtw,
        "RF"   : folds_rf,
        "LSTM" : folds_lstm,
    }
    generate_pvalue_table(results_ui_d1, domain=1)
    wilcoxon_test(folds_ed, folds_dtw, "Edit Distance", "DTW", domain=1)

    # ------------------------------------------------------------------
    # 9.  Confusion matrix – best model across ALL methods – Domain 1
    # ------------------------------------------------------------------
    all_methods_ui_d1 = {
        "Edit Distance" : mean_ed,
        "DTW"           : mean_dtw,
        "RF"            : mean_rf,
        "LSTM"          : mean_lstm,
    }
    best_name_d1 = max(all_methods_ui_d1, key=all_methods_ui_d1.get)
    print(f"\n  Best model – Domain 1 (UI): {best_name_d1} "
          f"(acc={all_methods_ui_d1[best_name_d1]:.3f})")

    draw_best_model_cm(best_name_d1,
                       data_std=data1_std, sequences=sequences1,
                       labels=labels1, users=users1, domain=1)

    # ================================================================
    # DOMAIN 4
    # ================================================================

    # ------------------------------------------------------------------
    # 10.  Pre-processing – Domain 4
    # ------------------------------------------------------------------
    print("\n=== Pre-processing (Domain 4) ===")
    data4_std = standardize_gestures(data4)
    print("  Standardisation done.")

    print("  Fitting PCA …")
    data4_pca, pca4 = apply_pca(data4_std, n_components=2)

    print(f"  Fitting k-means (k={K}) …")
    sequences4, kmeans4 = cluster_and_encode(data4_pca, k=K)
    print(f"  Example sequence (first 20): {sequences4[0][:20]}")

    # ------------------------------------------------------------------
    # 11.  Baselines – Domain 4 – USER-INDEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Edit Distance | User-independent | Domain 4 ===")
    mean_ed4, std_ed4, folds_ed4 = crossval_user_independent(
        sequences4, labels4, users4, edit_distance, k=1)
    print(f"  → Mean accuracy: {mean_ed4:.3f} ± {std_ed4:.3f}")
    save_fold_results(folds_ed4, "edit", "user_independent", 4)

    print("\n=== DTW | User-independent | Domain 4 ===")
    mean_dtw4, std_dtw4, folds_dtw4 = crossval_user_independent(
        data4_std, labels4, users4, dtw_distance, k=1)
    print(f"  → Mean accuracy: {mean_dtw4:.3f} ± {std_dtw4:.3f}")
    save_fold_results(folds_dtw4, "dtw", "user_independent", 4)

    # ------------------------------------------------------------------
    # 12.  Advanced methods – Domain 4 – USER-INDEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Random Forest | User-independent | Domain 4 ===")
    mean_rf4, std_rf4, folds_rf4 = random_forest_evaluation(
        data4_std, labels4, users4)
    print(f"  → Mean accuracy: {mean_rf4:.3f} ± {std_rf4:.3f}")
    save_fold_results(folds_rf4, "rf", "user_independent", 4)

    print("\n=== LSTM | User-independent | Domain 4 ===")
    mean_lstm4, std_lstm4, folds_lstm4 = lstm_evaluation(
        data4_std, labels4, users4)
    print(f"  → Mean accuracy: {mean_lstm4:.3f} ± {std_lstm4:.3f}")
    save_fold_results(folds_lstm4, "lstm", "user_independent", 4)

    # ------------------------------------------------------------------
    # 13.  Baselines – Domain 4 – USER-DEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Edit Distance | User-dependent | Domain 4 ===")
    mean_ed4_ud, std_ed4_ud, folds_ed4_ud = crossval_user_dependent(
        sequences4, labels4, users4, edit_distance, k=1)
    print(f"  → Mean accuracy: {mean_ed4_ud:.3f} ± {std_ed4_ud:.3f}")
    save_fold_results(folds_ed4_ud, "edit", "user_dependent", 4)

    print("\n=== DTW | User-dependent | Domain 4 ===")
    mean_dtw4_ud, std_dtw4_ud, folds_dtw4_ud = crossval_user_dependent(
        data4_std, labels4, users4, dtw_distance, k=1)
    print(f"  → Mean accuracy: {mean_dtw4_ud:.3f} ± {std_dtw4_ud:.3f}")
    save_fold_results(folds_dtw4_ud, "dtw", "user_dependent", 4)

    # ------------------------------------------------------------------
    # 14.  Advanced methods – Domain 4 – USER-DEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Random Forest | User-dependent | Domain 4 ===")
    mean_rf4_ud, std_rf4_ud, folds_rf4_ud = \
        random_forest_evaluation_user_dependent(data4_std, labels4, users4)
    print(f"  → Mean accuracy: {mean_rf4_ud:.3f} ± {std_rf4_ud:.3f}")
    save_fold_results(folds_rf4_ud, "rf", "user_dependent", 4)

    print("\n=== LSTM | User-dependent | Domain 4 ===")
    mean_lstm4_ud, std_lstm4_ud, folds_lstm4_ud = \
        lstm_evaluation_user_dependent(data4_std, labels4, users4)
    print(f"  → Mean accuracy: {mean_lstm4_ud:.3f} ± {std_lstm4_ud:.3f}")
    save_fold_results(folds_lstm4_ud, "lstm", "user_dependent", 4)

    # ------------------------------------------------------------------
    # 15.  Statistical tests – Domain 4
    # ------------------------------------------------------------------
    results_ui_d4 = {
        "Edit" : folds_ed4,
        "DTW"  : folds_dtw4,
        "RF"   : folds_rf4,
        "LSTM" : folds_lstm4,
    }
    generate_pvalue_table(results_ui_d4, domain=4)
    wilcoxon_test(folds_ed4, folds_dtw4, "Edit Distance", "DTW", domain=4)

    # ------------------------------------------------------------------
    # 16.  Confusion matrix – best model across ALL methods – Domain 4
    # ------------------------------------------------------------------
    all_methods_ui_d4 = {
        "Edit Distance" : mean_ed4,
        "DTW"           : mean_dtw4,
        "RF"            : mean_rf4,
        "LSTM"          : mean_lstm4,
    }
    best_name_d4 = max(all_methods_ui_d4, key=all_methods_ui_d4.get)
    print(f"\n  Best model – Domain 4 (UI): {best_name_d4} "
          f"(acc={all_methods_ui_d4[best_name_d4]:.3f})")

    draw_best_model_cm(best_name_d4,
                       data_std=data4_std, sequences=sequences4,
                       labels=labels4, users=users4, domain=4)

    # ------------------------------------------------------------------
    # 17.  Final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY – Mean accuracy ± std")
    print("(UI = user-independent, UD = user-dependent)")
    print("=" * 65)

    rows = []
    for domain, res in [
        (1, {
            "edit_UI" : (mean_ed,      std_ed),
            "dtw_UI"  : (mean_dtw,     std_dtw),
            "rf_UI"   : (mean_rf,      std_rf),
            "lstm_UI" : (mean_lstm,    std_lstm),
            "edit_UD" : (mean_ed_ud,   std_ed_ud),
            "dtw_UD"  : (mean_dtw_ud,  std_dtw_ud),
            "rf_UD"   : (mean_rf_ud,   std_rf_ud),
            "lstm_UD" : (mean_lstm_ud, std_lstm_ud),
        }),
        (4, {
            "edit_UI" : (mean_ed4,      std_ed4),
            "dtw_UI"  : (mean_dtw4,     std_dtw4),
            "rf_UI"   : (mean_rf4,      std_rf4),
            "lstm_UI" : (mean_lstm4,    std_lstm4),
            "edit_UD" : (mean_ed4_ud,   std_ed4_ud),
            "dtw_UD"  : (mean_dtw4_ud,  std_dtw4_ud),
            "rf_UD"   : (mean_rf4_ud,   std_rf4_ud),
            "lstm_UD" : (mean_lstm4_ud, std_lstm4_ud),
        }),
    ]:
        for key, (m, s) in res.items():
            method, setting = key.rsplit("_", 1)
            rows.append({"Domain": domain, "Method": method.upper(),
                         "Setting": setting, "Mean": m, "Std": s})

    df_summary = pd.DataFrame(rows)
    df_summary["Result"] = (df_summary["Mean"].map("{:.3f}".format)
                             + " ± "
                             + df_summary["Std"].map("{:.3f}".format))
    pivot = df_summary.pivot_table(
        index=["Domain", "Setting"], columns="Method",
        values="Result", aggfunc="first"
    )
    print(pivot.to_string())
    summary_path = os.path.join(DATA_DIR, "summary_results.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Full results saved → {summary_path}")
    print("\nDone.")
