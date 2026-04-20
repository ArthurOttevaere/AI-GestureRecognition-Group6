"""
MLSM2154 – Artificial Intelligence: Gesture Recognition Project
===============================================================
Phase 1 : Data loading & exploratory analysis
Phase 2 : Pre-processing
            - Per-gesture standardisation (zero mean, unit std per axis)
            - Per-gesture PCA (professor's request): captures intrinsic
              geometry of each individual trajectory; EVR values used as
              additional features for the Random Forest classifier.
            - k-means clustering on standardised 3D coordinates for
              the symbolic encoding required by Edit Distance.
Phase 3 : Baseline methods (DTW + Edit Distance) with k-NN classifier
Phase 4 : Advanced methods (Random Forest, LSTM)
Phase 5 : Cross-validation
            - User-independent : leave-one-user-out  (10 folds)
            - User-dependent   : leave-one-sample-out (100 folds)
          Ablation study: 4 methods x 3 preprocessing conditions
Phase 6 : Statistical tests (Wilcoxon signed-rank, raw p-values +
          Benjamini-Hochberg FDR correction) – user-independent only,
          as required by the course guidelines.

Authors  : Andry Lenny / El Mohcine Mohamed / Ottevaere Arthur
Group    : Group 6
Date     : 2026
"""

import os
import numpy as np
import pandas as pd

from config import DOMAIN1_DIR, DOMAIN4_DIR, DATA_DIR, K_CLUSTERS
from data_loading import load_domain1, load_domain4, print_dataset_info
from visualization import plot_sequence_lengths, plot_gesture_samples
from preprocessing import (standardize_gestures, summarise_per_gesture_pca,
                            cluster_and_encode)
from distance_metrics import dtw_distance, edit_distance
from crossvalidation import crossval_user_independent, crossval_user_dependent
from random_forest import (random_forest_evaluation,
                            random_forest_evaluation_user_dependent)
from lstm_model import lstm_evaluation, lstm_evaluation_user_dependent
from evaluation import draw_best_model_cm
from results import save_fold_results, generate_pvalue_table
from ablation import run_ablation_study


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 0.  Warm-up numba JIT
    # ------------------------------------------------------------------
    print("Warming up numba JIT ...", end=" ", flush=True)
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
    # 2.  Exploratory visualisation
    # ------------------------------------------------------------------
    print("\n=== Exploratory Visualisation ===")
    plot_sequence_lengths(data1, labels1, "Domain 1",
                          save_path=os.path.join(DATA_DIR, "d1_sequence_lengths.png"))
    plot_gesture_samples(data1, labels1, users1, "Domain 1",
                         save_path=os.path.join(DATA_DIR, "d1_gesture_samples.png"))
    plot_sequence_lengths(data4, labels4, "Domain 4",
                          save_path=os.path.join(DATA_DIR, "d4_sequence_lengths.png"))
    plot_gesture_samples(data4, labels4, users4, "Domain 4",
                         save_path=os.path.join(DATA_DIR, "d4_gesture_samples.png"))

    # ------------------------------------------------------------------
    # 3.  Standardisation (per gesture, per axis)
    # ------------------------------------------------------------------
    print("\n=== Standardisation ===")
    data1_std = standardize_gestures(data1)
    data4_std = standardize_gestures(data4)
    print("  Both domains standardised (per-gesture, per-axis).")

    # ------------------------------------------------------------------
    # 4.  Per-gesture PCA analysis (on standardised data)
    #     Reports EVR statistics and planarity distributions.
    #     EVR values are later used as additional RF features (cond. c).
    # ------------------------------------------------------------------
    print("\n=== Per-gesture PCA analysis (standardised data) ===")
    summarise_per_gesture_pca(
        data1_std, "Domain 1",
        save_path=os.path.join(DATA_DIR, "d1_pca_per_gesture.png"))
    summarise_per_gesture_pca(
        data4_std, "Domain 4",
        save_path=os.path.join(DATA_DIR, "d4_pca_per_gesture.png"))

    # ------------------------------------------------------------------
    # 5.  k-means symbolic encoding for Edit Distance (main evaluation)
    #     Fitted once on standardised 3D coordinates before CV, as
    #     explicitly permitted by the course guidelines.
    # ------------------------------------------------------------------
    print(f"\n=== k-means (k={K_CLUSTERS}) on standardised data ===")
    sequences1, _ = cluster_and_encode(data1_std, k=K_CLUSTERS)
    sequences4, _ = cluster_and_encode(data4_std, k=K_CLUSTERS)
    print(f"  Domain 1 - example (first 20 labels): {sequences1[0][:20]}")
    print(f"  Domain 4 - example (first 20 labels): {sequences4[0][:20]}")

    # ------------------------------------------------------------------
    # 6.  Ablation study  (4 methods x 3 preprocessing conditions)
    #     Quantifies the contribution of each preprocessing step.
    #     User-independent setting only.
    # ------------------------------------------------------------------
    print("\n=== Ablation Study - Domain 1 ===")
    run_ablation_study(data1, data1_std, labels1, users1, domain=1)

    print("\n=== Ablation Study - Domain 4 ===")
    run_ablation_study(data4, data4_std, labels4, users4, domain=4)

    # ------------------------------------------------------------------
    # 7.  Main evaluation - Domain 1 - USER-INDEPENDENT
    #     All methods use standardised data + PCA EVR for RF (cond. c).
    # ------------------------------------------------------------------
    print("\n=== Main Evaluation - Domain 1 - User-Independent ===")

    print("\n  Edit Distance:")
    mean_ed1, std_ed1, folds_ed1 = crossval_user_independent(
        sequences1, labels1, users1, edit_distance, k=1)
    print(f"  -> {mean_ed1:.3f} +/- {std_ed1:.3f}")
    save_fold_results(folds_ed1, "edit", "user_independent", 1)

    print("\n  DTW:")
    mean_dtw1, std_dtw1, folds_dtw1 = crossval_user_independent(
        data1_std, labels1, users1, dtw_distance, k=1)
    print(f"  -> {mean_dtw1:.3f} +/- {std_dtw1:.3f}")
    save_fold_results(folds_dtw1, "dtw", "user_independent", 1)

    print("\n  Random Forest (+ PCA EVR):")
    mean_rf1, std_rf1, folds_rf1 = random_forest_evaluation(
        data1_std, labels1, users1, include_pca_evr=True)
    print(f"  -> {mean_rf1:.3f} +/- {std_rf1:.3f}")
    save_fold_results(folds_rf1, "rf", "user_independent", 1)

    print("\n  LSTM:")
    mean_lstm1, std_lstm1, folds_lstm1 = lstm_evaluation(
        data1_std, labels1, users1)
    print(f"  -> {mean_lstm1:.3f} +/- {std_lstm1:.3f}")
    save_fold_results(folds_lstm1, "lstm", "user_independent", 1)

    # ------------------------------------------------------------------
    # 8.  Main evaluation - Domain 1 - USER-DEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Main Evaluation - Domain 1 - User-Dependent ===")

    print("\n  Edit Distance:")
    mean_ed1_ud, std_ed1_ud, folds_ed1_ud = crossval_user_dependent(
        sequences1, labels1, users1, edit_distance, k=1)
    print(f"  -> {mean_ed1_ud:.3f} +/- {std_ed1_ud:.3f}")
    save_fold_results(folds_ed1_ud, "edit", "user_dependent", 1)

    print("\n  DTW:")
    mean_dtw1_ud, std_dtw1_ud, folds_dtw1_ud = crossval_user_dependent(
        data1_std, labels1, users1, dtw_distance, k=1)
    print(f"  -> {mean_dtw1_ud:.3f} +/- {std_dtw1_ud:.3f}")
    save_fold_results(folds_dtw1_ud, "dtw", "user_dependent", 1)

    print("\n  Random Forest (+ PCA EVR):")
    mean_rf1_ud, std_rf1_ud, folds_rf1_ud = \
        random_forest_evaluation_user_dependent(
            data1_std, labels1, users1, include_pca_evr=True)
    print(f"  -> {mean_rf1_ud:.3f} +/- {std_rf1_ud:.3f}")
    save_fold_results(folds_rf1_ud, "rf", "user_dependent", 1)

    print("\n  LSTM:")
    mean_lstm1_ud, std_lstm1_ud, folds_lstm1_ud = \
        lstm_evaluation_user_dependent(data1_std, labels1, users1)
    print(f"  -> {mean_lstm1_ud:.3f} +/- {std_lstm1_ud:.3f}")
    save_fold_results(folds_lstm1_ud, "lstm", "user_dependent", 1)

    # ------------------------------------------------------------------
    # 9.  Statistical tests - Domain 1  (user-independent only)
    # ------------------------------------------------------------------
    print("\n=== Statistical Tests - Domain 1 ===")
    results_ui_d1 = {
        "Edit Distance": folds_ed1,
        "DTW"          : folds_dtw1,
        "RF"           : folds_rf1,
        "LSTM"         : folds_lstm1,
    }
    generate_pvalue_table(results_ui_d1, domain=1)

    # ------------------------------------------------------------------
    # 10.  Confusion matrix - best model - Domain 1
    # ------------------------------------------------------------------
    all_ui_d1    = {"Edit Distance": mean_ed1, "DTW": mean_dtw1,
                    "RF": mean_rf1, "LSTM": mean_lstm1}
    best_name_d1 = max(all_ui_d1, key=all_ui_d1.get)
    print(f"\n  Best model - Domain 1 (UI): {best_name_d1} "
          f"({all_ui_d1[best_name_d1]:.3f})")
    draw_best_model_cm(best_name_d1, data_std=data1_std, sequences=sequences1,
                       labels=labels1, users=users1, domain=1)

    # ==================================================================
    # DOMAIN 4
    # ==================================================================

    # ------------------------------------------------------------------
    # 11.  Main evaluation - Domain 4 - USER-INDEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Main Evaluation - Domain 4 - User-Independent ===")

    print("\n  Edit Distance:")
    mean_ed4, std_ed4, folds_ed4 = crossval_user_independent(
        sequences4, labels4, users4, edit_distance, k=1)
    print(f"  -> {mean_ed4:.3f} +/- {std_ed4:.3f}")
    save_fold_results(folds_ed4, "edit", "user_independent", 4)

    print("\n  DTW:")
    mean_dtw4, std_dtw4, folds_dtw4 = crossval_user_independent(
        data4_std, labels4, users4, dtw_distance, k=1)
    print(f"  -> {mean_dtw4:.3f} +/- {std_dtw4:.3f}")
    save_fold_results(folds_dtw4, "dtw", "user_independent", 4)

    print("\n  Random Forest (+ PCA EVR):")
    mean_rf4, std_rf4, folds_rf4 = random_forest_evaluation(
        data4_std, labels4, users4, include_pca_evr=True)
    print(f"  -> {mean_rf4:.3f} +/- {std_rf4:.3f}")
    save_fold_results(folds_rf4, "rf", "user_independent", 4)

    print("\n  LSTM:")
    mean_lstm4, std_lstm4, folds_lstm4 = lstm_evaluation(
        data4_std, labels4, users4)
    print(f"  -> {mean_lstm4:.3f} +/- {std_lstm4:.3f}")
    save_fold_results(folds_lstm4, "lstm", "user_independent", 4)

    # ------------------------------------------------------------------
    # 12.  Main evaluation - Domain 4 - USER-DEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Main Evaluation - Domain 4 - User-Dependent ===")

    print("\n  Edit Distance:")
    mean_ed4_ud, std_ed4_ud, folds_ed4_ud = crossval_user_dependent(
        sequences4, labels4, users4, edit_distance, k=1)
    print(f"  -> {mean_ed4_ud:.3f} +/- {std_ed4_ud:.3f}")
    save_fold_results(folds_ed4_ud, "edit", "user_dependent", 4)

    print("\n  DTW:")
    mean_dtw4_ud, std_dtw4_ud, folds_dtw4_ud = crossval_user_dependent(
        data4_std, labels4, users4, dtw_distance, k=1)
    print(f"  -> {mean_dtw4_ud:.3f} +/- {std_dtw4_ud:.3f}")
    save_fold_results(folds_dtw4_ud, "dtw", "user_dependent", 4)

    print("\n  Random Forest (+ PCA EVR):")
    mean_rf4_ud, std_rf4_ud, folds_rf4_ud = \
        random_forest_evaluation_user_dependent(
            data4_std, labels4, users4, include_pca_evr=True)
    print(f"  -> {mean_rf4_ud:.3f} +/- {std_rf4_ud:.3f}")
    save_fold_results(folds_rf4_ud, "rf", "user_dependent", 4)

    print("\n  LSTM:")
    mean_lstm4_ud, std_lstm4_ud, folds_lstm4_ud = \
        lstm_evaluation_user_dependent(data4_std, labels4, users4)
    print(f"  -> {mean_lstm4_ud:.3f} +/- {std_lstm4_ud:.3f}")
    save_fold_results(folds_lstm4_ud, "lstm", "user_dependent", 4)

    # ------------------------------------------------------------------
    # 13.  Statistical tests - Domain 4  (user-independent only)
    # ------------------------------------------------------------------
    print("\n=== Statistical Tests - Domain 4 ===")
    results_ui_d4 = {
        "Edit Distance": folds_ed4,
        "DTW"          : folds_dtw4,
        "RF"           : folds_rf4,
        "LSTM"         : folds_lstm4,
    }
    generate_pvalue_table(results_ui_d4, domain=4)

    # ------------------------------------------------------------------
    # 14.  Confusion matrix - best model - Domain 4
    # ------------------------------------------------------------------
    all_ui_d4    = {"Edit Distance": mean_ed4, "DTW": mean_dtw4,
                    "RF": mean_rf4, "LSTM": mean_lstm4}
    best_name_d4 = max(all_ui_d4, key=all_ui_d4.get)
    print(f"\n  Best model - Domain 4 (UI): {best_name_d4} "
          f"({all_ui_d4[best_name_d4]:.3f})")
    draw_best_model_cm(best_name_d4, data_std=data4_std, sequences=sequences4,
                       labels=labels4, users=users4, domain=4)

    # ------------------------------------------------------------------
    # 15.  Final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL SUMMARY - Mean accuracy +/- std")
    print("UI = user-independent  |  UD = user-dependent")
    print("=" * 65)

    summary_rows = []
    for domain, res in [
        (1, {
            ("Edit Distance", "UI"): (mean_ed1,      std_ed1),
            ("DTW",           "UI"): (mean_dtw1,     std_dtw1),
            ("RF",            "UI"): (mean_rf1,      std_rf1),
            ("LSTM",          "UI"): (mean_lstm1,    std_lstm1),
            ("Edit Distance", "UD"): (mean_ed1_ud,   std_ed1_ud),
            ("DTW",           "UD"): (mean_dtw1_ud,  std_dtw1_ud),
            ("RF",            "UD"): (mean_rf1_ud,   std_rf1_ud),
            ("LSTM",          "UD"): (mean_lstm1_ud, std_lstm1_ud),
        }),
        (4, {
            ("Edit Distance", "UI"): (mean_ed4,      std_ed4),
            ("DTW",           "UI"): (mean_dtw4,     std_dtw4),
            ("RF",            "UI"): (mean_rf4,      std_rf4),
            ("LSTM",          "UI"): (mean_lstm4,    std_lstm4),
            ("Edit Distance", "UD"): (mean_ed4_ud,   std_ed4_ud),
            ("DTW",           "UD"): (mean_dtw4_ud,  std_dtw4_ud),
            ("RF",            "UD"): (mean_rf4_ud,   std_rf4_ud),
            ("LSTM",          "UD"): (mean_lstm4_ud, std_lstm4_ud),
        }),
    ]:
        for (method, setting), (m, s) in res.items():
            summary_rows.append({
                "Domain": domain, "Method": method,
                "Setting": setting, "Mean": m, "Std": s,
                "Result": f"{m:.3f} +/- {s:.3f}"
            })

    df_summary = pd.DataFrame(summary_rows)
    pivot = df_summary.pivot_table(
        index=["Domain", "Setting"], columns="Method",
        values="Result", aggfunc="first"
    )
    print(pivot.to_string())
    summary_path = os.path.join(DATA_DIR, "summary_results.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Saved -> {summary_path}")
    print("\nDone.")
