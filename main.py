"""
MLSM2154 — Artificial Intelligence: Gesture Recognition Project
===============================================================
Phase 1 : Data loading & exploratory analysis
Phase 2 : Pre-processing
            - Per-gesture standardisation (zero mean, unit std per axis)
            - Per-gesture PCA denoising (3D -> 2D -> 3D):
                * PCA fitted on the T time-step points of each gesture
                * Projection onto the 2 principal components (2D)
                * Back-projection into the original 3D space
                * The 3 EVR values are also stored as additional RF features.
            - k-means clustering INSIDE each CV fold (rigorous):
                * Fitted on training-set 3D points only per fold
                * Centroids applied to encode both train and test sequences
Phase 3 : Baseline methods (DTW + Edit Distance with k-means quantization)
            - 1-NN classifier for both (KNN_K = 1 enforced; see below)
Phase 4 : Advanced methods
            - Decision Tree (Quinlan 1986, Breiman et al. 1984 CART) with
              GridSearchCV (nested CV, 5-fold inner). Pedagogical baseline
              to demonstrate empirically the bagging gain of Random Forest.
            - Random Forest (Breiman 2001) with GridSearchCV (5-fold inner)
              + per-fold permutation-importance feature selection
              (Strobl et al. 2007; Guyon & Elisseeff 2003).
            - Logistic Regression (Cox 1958; Hosmer et al. 2013), multinomial
              with L2 penalty, GridSearchCV over C (5-fold inner). Linear
              baseline to bound the difficulty of the task.
            - $1 Recognizer 3D (Kratz & Rohs, 2010) with Rodrigues rotation,
              uniform cube scaling, confidence score, and N-best list.
              EVALUATED ON RAW DATA ONLY (Wobbrock 2007 prescription).
Phase 5 : Cross-validation
            - User-independent : leave-one-user-out (10 folds)
            - User-dependent   : leave-one-sample-out (10 folds)
          Ablation study: 6 methods x 3 preprocessing conditions, run in
          BOTH UI AND UD (Iteration 2 fix: best preproc may differ).
Phase 6 : Hyperparameter validation curves (empirical iterative selection)
            - K_CLUSTERS optimum from sensitivity analysis is USED in the
              rest of the pipeline.
            - KNN_K is FORCED to 1 (1-NN) regardless of the validation
              curve outcome (see Major scientific decisions #8).
Phase 7 : Statistical tests
            - Paired Wilcoxon signed-rank test on n=100 paired observations
              (10 gestures x 10 users), one accuracy per (gesture, user) pair
              for each method.
            - Benjamini-Hochberg FDR correction (BH).
            - Pairwise p-value matrix (6x6 = 15 pairs) saved as CSV + heatmap.
Phase 8 : Overfitting diagnostic
            - Per-fold train-acc vs test-acc gap for DT/RF/LR
              (parametric classifiers). DTW/Edit/$1 are 1-NN, so train
              accuracy is 1.0 by construction and not reported.
            - sklearn.model_selection.learning_curve for DT/RF/LR
              on each domain/setting: train score and validation score
              as a function of training set size. Wide gap = high variance
              (overfitting). Reference: Hastie et al. (2009) §7.10.

Outputs are organised under Outputs/ in nested subfolders by type:
    figures/{exploratory, pca_denoising, validation_curves,
             feature_importance, confusion_matrices, statistical_tests}
    tables/{ablation, fold_results, statistical_tests, summary}
    Documentation/  (LaTeX internal pedagogical memo, see plan)

Major scientific decisions
--------------------------
1. $1 Recognizer — 3D adaptation following Kratz & Rohs (2010).
   The original $1 algorithm of Wobbrock, Wilson & Li (2007) is purely 2D
   and provides no empirical validation in 3D. We follow the canonical
   3D extension of Kratz & Rohs (2010) "A $3 Gesture Recognizer", IUI'10:
     - Step 2: rotation around the axis defined by the cross product
                pâ x c, where c is the centroid and pâ is the first
                resampled point. The angle is the arccos of the
                normalised dot product. Rotation applied via Rodrigues'
                formula.
     - Step 3: scaling INSIDE a normalised cube of side l (uniform
                rescaling), avoiding axis-by-axis scaling and the
                division-by-zero issue for quasi-planar gestures.
     - Score:  S = 1 - d / (0.5 * sqrt(3) * l^2), where d is the MSE
                (mean squared point-to-point distance) after GSS
                alignment.  Using MSE is required by Kratz's scoring
                formula; the denominator is 0.5*sqrt(3)*l^2.
     - Templates are preprocessed only ONCE and cached, as specified
       by Wobbrock et al. (2007).
     - Recognize returns a sorted N-best list, allowing kNN with k>1.
     - Golden Section Search (GSS) over the 3 rotation axes (Kratz &
       Rohs 2010, "Search for Minimum Distance at Best Angle") is
       implemented via Numba-JIT functions (_dollar_gss_mse):
       phi = 0.5*(sqrt(5)-1), cutoff = 2 deg, 11 iterations per axis.
     - Scoring heuristic (Kratz & Rohs 2010): top-3 check with epsilon
       thresholds. allow_rejection=False (default) forces 1-Best for
       CV comparability with RF/LR/DT.

2. Wilcoxon n=100. For each method, a 100-vector of accuracies is built,
   one per (gesture, user) pair. Pairs are then compared method-vs-method
   via scipy.stats.wilcoxon (signed-rank). The all-zero-diff degenerate
   case is caught and returns p=1.0 with a warning.

3. Benjamini-Hochberg FDR correction only (no Bonferroni).
   Bonferroni is removed because the pairwise Wilcoxon tests share methods
   across pairs and are therefore not independent — BH is the correct
   procedure under positive dependence (Benjamini & Hochberg 1995, §4).
   The earlier permutation test and Bayesian sign test are also removed
   (redundant with Wilcoxon + BH).

4. LSTM removed. The dataset is too small (~1000 samples) to justify a
   recurrent network. Earlier results were retained from prior versions
   and are no longer relevant.

5. Random Forest hyperparameters selected per fold via GridSearchCV
   (5-fold inner CV). Feature selection done per-fold using permutation
   importance (Breiman 2001; Strobl et al. 2007) with a 95% cumulative
   threshold (Guyon & Elisseeff 2003). NO leakage: importance is fit on
   each outer-fold's training set only.

6. Hyperparameter tuning asymmetry. RF, Decision Tree, and Logistic
   Regression hyperparameters are selected per-fold via GridSearchCV
   (nested CV, inner CV = 5 folds; Varma & Simon 2006; Cawley & Talbot
   2010). DTW and Edit Distance hyperparameters (k-means K) are selected
   once via empirical validation curves on the full user-independent CV,
   then kept fixed for evaluation. This asymmetry is intentional: DTW
   and Edit Distance have at most 2 scalar hyperparameters with a clear
   plateau; tree- and kernel-based classifiers have a combinatorial grid
   that requires per-fold tuning to avoid overfitting. The comparison
   is therefore between best-configured versions of each method, not
   between methods sharing an identical tuning protocol. This limitation
   is acknowledged in the report.

7. K-clustering selected via two empirical validation curves per domain
   (Edit Distance accuracy vs K on user-independent CV):
     - On standardised data (data_std) -> best K used for conditions (b)
       and (c) of the ablation study.
     - On raw data (data_raw) -> best K used for condition (a) only.
   Rationale: k-means distance scales differ between raw and standardised
   spaces (Linde et al. 1980, VQ theory), so the optimal codebook size
   may not transfer across preprocessing conditions.

8. KNN_K = 1 ENFORCED. A validation curve scanning k in {1,3,5,7,9} is
   produced for transparency (saved to Outputs/figures/validation_curves)
   but the rest of the pipeline ALWAYS uses k = 1. This is consistent
   with the gesture recognition literature where 1-NN is the canonical
   baseline:
     - Wobbrock et al. (2007) report results with the single best
       template ($1 design is implicitly 1-NN).
     - Mezari & Maglogiannis (2018) use 1-NN for their commodity-device
       recognizer.
     - Liu et al. (2009) uFlash baseline uses 1-NN.
     - Mitra & Acharya (2007) gesture recognition survey discusses 1-NN
       as the standard non-parametric baseline.
   1-NN gives a transparent, parameter-free baseline that is robust to
   outliers being broken by majority voting with small k.

9. Decision Tree included as pedagogical control. Comparing DT (single
   tree) to RF (bag of trees) empirically demonstrates the value of
   bagging on this dataset (Breiman 2001, Hastie et al. 2009, ch. 15).

10. Logistic Regression (multinomial, L2 penalty, lbfgs solver). Linear
    classifier baseline. References: Cox (1958); Hosmer, Lemeshow &
    Sturdivant (2013); Hastie, Tibshirani & Friedman (2009) §4.4.
    Acts as a linear baseline reference: a linear model on the same feature
    vector.

12. $1 Recognizer evaluated on RAW data ONLY (preprocessing condition
    (a)). Per Wobbrock et al. (2007) and Kratz & Rohs (2010), the $1
    pipeline includes its own internal normalisation (resample, centroid
    translation, indicative-axis rotation, uniform cube scaling).
    Feeding externally-standardised data into $1 alters the geometry
    used by the cross-product rotation step and is methodologically
    inconsistent. The ablation table still reports the three conditions
    for transparency but the main UI/UD evaluation forces (a).

13. Ablation study is run in BOTH user-independent (UI) and
    user-dependent (UD) settings, because the best preprocessing for UI
    is not guaranteed to be the best for UD. Each setting picks its own
    optimum per method.

14. Per-fold three-stage feature selection (replaces previous post-hoc
    Gini selection):
    (a) Variance filter: remove near-constant features (var < 1e-10).
    (b) Correlation-based redundancy removal: for each pair with
        |Pearson r| > 0.99, discard the lower-variance member (adapted
        from Yu & Liu 2004, JMLR). Without this step, highly correlated
        features (e.g. x_range = x_max - x_min; bbox_diag derived from
        bbox_*) dilute each other's unconditional permutation importance
        (Strobl et al. 2008, BMC Bioinformatics). This dilution causes
        the cumulative-importance cut to drop entire correlated clusters,
        destroying classifier performance.
        Note on threshold: While traditional filters often use stricter
        cutoffs (e.g. 0.95), we empirically relaxed this threshold to 0.99.
        This engineering decision preserves a richer pool of geometric
        descriptors (~15-30 features per fold), preventing underfitting
        and ensuring the Random Forest retains enough dimensions to
        effectively model complex non-linear interactions, thereby
        restoring its theoretical advantage over linear models (LR).
    (c) Permutation importance on a 20%-held-out validation split (not
        on the training data itself, which would bias importance toward
        memorised features; Breiman 2001; Strobl et al. 2007).
    Selection is fit on each outer-fold's training set only -> no
    leakage (Ambroise & McLachlan 2002, PNAS).

15. Feature standardisation for LR via StandardScaler inside a
    sklearn Pipeline, fitted on each inner-CV training split only.
    LR (lbfgs solver): gradient magnitudes are scale-dependent;
    scaling ensures balanced gradient steps and reliable
    convergence.  max_iter increased from 2000 to 5000 to
    guarantee convergence across all C values in the grid
    (sklearn ConvergenceWarning fix).

16. Overfitting diagnostic. For each parametric classifier
    (DT, RF, LR) we record train and test accuracy per fold; the
    average gap train-test is reported. sklearn `learning_curve` is
    run on each (domain, setting) producing train/validation score
    curves as a function of training-set size (Hastie et al. 2009 §7.10;
    Domingos 2012).

References
----------
Ambroise, C., & McLachlan, G. (2002). Selection bias in gene extraction
  on the basis of microarray gene-expression data. PNAS, 99 (10),
  6562-6566.
Breiman, L. (2001). Random forests. Machine Learning, 45 (1), 5-32.
Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).
  Classification and regression trees (CART). Wadsworth.
Cawley, G., & Talbot, N. (2010). On over-fitting in model selection
  and subsequent selection bias in performance evaluation. JMLR, 11,
  2079-2107.
Cortes, C., & Vapnik, V. (1995). Support-vector networks.
  Machine Learning, 20, 273-297.
Cox, D. R. (1958). The regression analysis of binary sequences.
  J. Royal Statistical Society B, 20 (2), 215-242.
Domingos, P. (2012). A few useful things to know about machine
  learning. Communications of the ACM, 55 (10), 78-87.
Guyon, I., & Elisseeff, A. (2003). An introduction to variable and
  feature selection. JMLR, 3, 1157-1182.
Hall, M. A. (1999). Correlation-based Feature Selection for Machine
  Learning. PhD thesis, University of Waikato.
Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
  Statistical Learning (2nd ed.). Springer.
Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (2010). A Practical Guide to
  Support Vector Classification. Technical report, NTU.
Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied
  Logistic Regression (3rd ed.). Wiley.
Kratz, S., & Rohs, M. (2010). A $3 gesture recognizer: simple gesture
  recognition for devices equipped with 3D acceleration sensors.
  IUI'10, 341-344.
Kratz, S., & Rohs, M. (2011). Protractor3D: A closed-form solution to
  rotation-invariant 3D gestures. IUI'11, 371-374.
Mezari, A., & Maglogiannis, I. (2018). An easily customized gesture
  recognizer for assisted living using commodity mobile devices.
  Journal of Healthcare Engineering, 2018:3180652.
Mitra, S., & Acharya, T. (2007). Gesture recognition: A survey.
  IEEE Trans. SMC-C, 37 (3), 311-324.
Quinlan, J. R. (1986). Induction of decision trees.
  Machine Learning, 1 (1), 81-106.
Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias
  in random forest variable importance measures: Illustrations,
  sources and a solution. BMC Bioinformatics, 8 (25).
Varma, S., & Simon, R. (2006). Bias in error estimation when using
  cross-validation for model selection. BMC Bioinformatics, 7 (91).
Wobbrock, J.O., Wilson, A.D., & Li, Y. (2007). Gestures without
  libraries, toolkits or training: A $1 recognizer for user interface
  prototypes. UIST'07, 159-168.
Wu, Y., & Huang, T. S. (1999). Vision-based gesture recognition:
  A review. Gesture Workshop, LNAI 1739, 103-115.
Yu, L., & Liu, H. (2004). Efficient Feature Selection via Analysis of
  Relevance and Redundancy. JMLR, 5, 1205-1224.

Authors : Andry Lenny / El Mohcine Mohamed / Ottevaere Arthur
Group   : Group 6
Date    : 2026
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src import utils
from src.config import (
    DOMAIN1_DIR, DOMAIN4_DIR, PCA_N_KEEP, DOLLAR_N,
    DIR_FIG_EXPLORE, DIR_FIG_PCA, DIR_FIG_VC,
    DIR_FIG_LEARNING_CURVES,
    DIR_TBL_SUMMARY, DIR_TBL_OVERFITTING, DIR_DOC,
)
from src.data_loader import load_domain1, load_domain4, print_dataset_info
from src.preprocessing import (
    standardize_gestures, apply_pca_denoising, summarise_pca_denoising,
)
from src.models.baselines import dtw_distance, edit_distance
from src.models.dollar import _dollar_gss_mse
from src.models.parametric import _lc_builder_dt, _lc_builder_rf, _lc_builder_lr
from src.evaluation.crossval import (
    _ui_fold_indices, _ud_fold_indices,
    validation_curve_kclusters, validation_curve_knn,
)
from src.evaluation.metrics import (
    plot_sequence_lengths, plot_all_gesture_classes_2d,
    draw_best_model_cm, plot_learning_curve_method,
)
from src.evaluation.stats import generate_pvalue_table
from src.evaluation.ablation import (
    run_ablation_study, save_overfitting_table,
    _force_dollar_raw, _save_preproc_comparison, _print_and_save_phase,
    METHODS_ORDER,
)


def _data_for(best_entry: dict,
              raw: list, std: list, denoised: list,
              evr: list) -> tuple[list, list | None]:
    """Local helper: map preprocessing condition to (data, evr) tuple."""
    cond = best_entry["condition"]
    if cond == "(a) No preprocessing":
        return raw, None
    if cond == "(b) Standardisation":
        return std, None
    return denoised, evr


def _interactive_menu() -> tuple[list[int], bool]:
    """Display a startup menu and return (domains_to_run, show_plots)."""
    sep = "=" * 60
    print(sep)
    print("  MLSM2154 — Gesture Recognition Pipeline  |  Group 6")
    print(sep)
    print()
    print("  This pipeline runs the full ML experiment:")
    print("    • Data loading & exploratory visualisation")
    print("    • Standardisation + PCA denoising")
    print("    • Baseline methods  (DTW, Edit Distance, $1 3D)")
    print("    • Advanced methods  (Decision Tree, Random Forest, LR)")
    print("    • Cross-validation  (user-independent + user-dependent)")
    print("    • Ablation study, statistical tests, confusion matrices")
    print("    • Overfitting diagnostic & learning curves")
    print()
    print("  Available domains:")
    print("    [1]  Domain 1  (accelerometer gestures, 10 users)")
    print("    [4]  Domain 4  (accelerometer gestures, 10 users)")
    print()

    # --- Domain selection ---
    print(sep)
    print("  Which domain(s) do you want to run?")
    print()
    print("    [1]  Domain 1 only")
    print("    [4]  Domain 4 only")
    print("    [b]  Both domains  (default)")
    print()
    while True:
        raw = input("  Your choice [1 / 4 / b]: ").strip().lower()
        if raw in ("", "b", "both"):
            domains_to_run = [1, 4]
            break
        if raw == "1":
            domains_to_run = [1]
            break
        if raw == "4":
            domains_to_run = [4]
            break
        print("  Invalid choice. Please enter 1, 4, or b.")

    print()

    # --- Display figures ---
    print(sep)
    print("  Display figures interactively on screen?")
    print()
    print("    [y]  Yes — open each figure in a window")
    print("    [n]  No  — save to Outputs/ only  (default, headless)")
    print()
    while True:
        raw = input("  Your choice [y / n]: ").strip().lower()
        if raw in ("", "n", "no"):
            show = False
            break
        if raw in ("y", "yes"):
            show = True
            break
        print("  Invalid choice. Please enter y or n.")

    print()
    domain_label = (
        "Domain 1 only" if domains_to_run == [1]
        else "Domain 4 only" if domains_to_run == [4]
        else "Both domains"
    )
    print(sep)
    print(f"  Running  : {domain_label}")
    print(f"  Figures  : {'displayed on screen' if show else 'saved to Outputs/ (headless)'}")
    print(sep)
    print()
    return domains_to_run, show


def main():
    parser = argparse.ArgumentParser(
        description="Gesture Recognition ML Pipeline — Group 6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Interactive menu (no flags)
  python main.py --domain 1           # Domain 1 only, headless
  python main.py --domain 4           # Domain 4 only, headless
  python main.py --show-plots         # Both domains + display figures
  python main.py --domain 1 --show-plots
        """,
    )
    parser.add_argument(
        "--domain",
        choices=["1", "4", "both"],
        default=None,
        metavar="{1,4,both}",
        help="Which domain(s) to process (default: interactive menu)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=False,
        help="Display figures interactively (default: save only, headless mode)",
    )
    args = parser.parse_args()

    if args.domain is None and not args.show_plots:
        domains_to_run, show = _interactive_menu()
    else:
        show = args.show_plots
        domains_to_run = [1, 4] if (args.domain is None or args.domain == "both") \
            else [int(args.domain)]

    # Logger must be initialised FIRST, before any print()
    sys.stdout = utils.Logger(DIR_DOC)
    sys.stderr = sys.stdout

    # -- 0. Reproducibility & numba warm-up -----------------------------------
    np.random.seed(42)

    print("Warming up numba JIT ...", end=" ", flush=True)
    _d = np.random.randn(10, 3)
    dtw_distance(_d, _d)
    _s = np.zeros(10, dtype=np.int64)
    edit_distance(_s, _s)
    _dp = np.random.randn(DOLLAR_N, 3).astype(np.float64)
    _dollar_gss_mse(_dp, _dp)   # compile _rotate_1axis_nb + _mse_nb + _dollar_gss_mse
    print("done.")

    # -- 1. Load data ---------------------------------------------------------
    if 1 in domains_to_run:
        print("\n=== Loading Domain 1 ===")
        data1, labels1, users1 = load_domain1(DOMAIN1_DIR)
        max_len1 = print_dataset_info(data1, labels1, users1, "Domain 1")

    if 4 in domains_to_run:
        print("\n=== Loading Domain 4 ===")
        data4, labels4, users4 = load_domain4(DOMAIN4_DIR)
        max_len4 = print_dataset_info(data4, labels4, users4, "Domain 4")

    # -- 2. Exploratory visualisation ----------------------------------------
    print("\n=== Exploratory Visualisation ===")
    if 1 in domains_to_run:
        plot_sequence_lengths(data1, labels1, "Domain 1",
                              save_path=os.path.join(DIR_FIG_EXPLORE,
                                                     "d1_sequence_lengths.png"),
                              show=show)
        plot_all_gesture_classes_2d(data1, labels1, users1, "Domain 1",
                                    save_path=os.path.join(DIR_FIG_EXPLORE,
                                                           "d1_gesture_samples.png"),
                                    show=show)
    if 4 in domains_to_run:
        plot_sequence_lengths(data4, labels4, "Domain 4",
                              save_path=os.path.join(DIR_FIG_EXPLORE,
                                                     "d4_sequence_lengths.png"),
                              show=show)
        plot_all_gesture_classes_2d(data4, labels4, users4, "Domain 4",
                                    save_path=os.path.join(DIR_FIG_EXPLORE,
                                                           "d4_gesture_samples.png"),
                                    show=show)

    # -- 3. Standardisation --------------------------------------------------
    print("\n=== Standardisation ===")
    if 1 in domains_to_run:
        data1_std = standardize_gestures(data1)
    if 4 in domains_to_run:
        data4_std = standardize_gestures(data4)
    print("  Both domains standardised (per-gesture, per-axis).")

    # -- 4. PCA denoising ----------------------------------------------------
    print("\n=== Per-gesture PCA denoising analysis ===")
    if 1 in domains_to_run:
        summarise_pca_denoising(data1_std, "Domain 1",
                                 save_path=os.path.join(DIR_FIG_PCA,
                                                        "d1_pca_denoise.png"),
                                 show=show)
    if 4 in domains_to_run:
        summarise_pca_denoising(data4_std, "Domain 4",
                                 save_path=os.path.join(DIR_FIG_PCA,
                                                        "d4_pca_denoise.png"),
                                 show=show)

    print("\n=== Applying PCA denoising (3D -> 2D -> 3D) ===")
    if 1 in domains_to_run:
        data1_denoised, evr1 = apply_pca_denoising(data1_std, n_keep=PCA_N_KEEP)
    if 4 in domains_to_run:
        data4_denoised, evr4 = apply_pca_denoising(data4_std, n_keep=PCA_N_KEEP)
    print("  PCA denoising applied to both domains.")

    # -- 5. Fold indices ------------------------------------------------------
    print("\n=== Generating fold indices (shared across all methods) ===")
    if 1 in domains_to_run:
        folds_ui_1 = _ui_fold_indices(users1)
        folds_ud_1 = _ud_fold_indices(labels1, users1)
        print(f"  Domain 1 - UI: {len(folds_ui_1)} | UD: {len(folds_ud_1)}")
    if 4 in domains_to_run:
        folds_ui_4 = _ui_fold_indices(users4)
        folds_ud_4 = _ud_fold_indices(labels4, users4)
        print(f"  Domain 4 - UI: {len(folds_ui_4)} | UD: {len(folds_ud_4)}")

    # -- 6. Validation curves for K (empirical iterative selection) ----------
    # K_CLUSTERS validation curve: the best K found below IS used in the rest
    # of the pipeline (passed to run_ablation_study / Edit-Distance evals).
    # Two separate scans are run per domain:
    #   - on data_std  -> best K for conditions (b) and (c)
    #   - on data_raw  -> best K for condition (a) only
    # Rationale: k-means distances in raw vs standardised space have different
    # scales, so the optimal codebook size may differ (Linde et al. 1980).
    # kNN K validation curve: the curve is computed and saved for transparency
    # (informative for the report) but the pipeline FORCES k=1 in every
    # downstream evaluation. Justification: 1-NN is the canonical gesture
    # recognition baseline (Wobbrock et al. 2007; Mezari & Maglogiannis 2018;
    # Mitra & Acharya 2007). See module docstring decision #8.
    print("\n=== Validation curves - hyperparameter K ===")
    if 1 in domains_to_run:
        print("  Domain 1 - K_CLUSTERS scan on standardised data (conditions b/c):")
        best_k_clusters_d1 = validation_curve_kclusters(
            data1_std, labels1, users1, folds_ui_1,
            domain=1,
            save_path=os.path.join(DIR_FIG_VC, "d1_vc_kclusters_std.png"),
            show=show)
        print("  Domain 1 - K_CLUSTERS scan on raw data (condition a only):")
        best_k_raw_d1 = validation_curve_kclusters(
            data1, labels1, users1, folds_ui_1,
            domain=1,
            save_path=os.path.join(DIR_FIG_VC, "d1_vc_kclusters_raw.png"),
            show=show)
        print("  Domain 1 - kNN K scan (DTW UI) [informative only, k=1 forced]:")
        _vc_best_knn_d1 = validation_curve_knn(
            data1_std, labels1, users1, folds_ui_1,
            method="dtw", domain=1,
            save_path=os.path.join(DIR_FIG_VC, "d1_vc_knn.png"),
            show=show)
        print(f"  Domain 1 selected: K_CLUSTERS(std)={best_k_clusters_d1}, "
              f"K_CLUSTERS(raw)={best_k_raw_d1}, "
              f"KNN_K=1 (forced; curve optimum was k={_vc_best_knn_d1})")

    if 4 in domains_to_run:
        print("\n  Domain 4 - K_CLUSTERS scan on standardised data (conditions b/c):")
        best_k_clusters_d4 = validation_curve_kclusters(
            data4_std, labels4, users4, folds_ui_4,
            domain=4,
            save_path=os.path.join(DIR_FIG_VC, "d4_vc_kclusters_std.png"),
            show=show)
        print("  Domain 4 - K_CLUSTERS scan on raw data (condition a only):")
        best_k_raw_d4 = validation_curve_kclusters(
            data4, labels4, users4, folds_ui_4,
            domain=4,
            save_path=os.path.join(DIR_FIG_VC, "d4_vc_kclusters_raw.png"),
            show=show)
        print("  Domain 4 - kNN K scan (DTW UI) [informative only, k=1 forced]:")
        _vc_best_knn_d4 = validation_curve_knn(
            data4_std, labels4, users4, folds_ui_4,
            method="dtw", domain=4,
            save_path=os.path.join(DIR_FIG_VC, "d4_vc_knn.png"),
            show=show)
        print(f"  Domain 4 selected: K_CLUSTERS(std)={best_k_clusters_d4}, "
              f"K_CLUSTERS(raw)={best_k_raw_d4}, "
              f"KNN_K=1 (forced; curve optimum was k={_vc_best_knn_d4})")

    # -- 7. Ablation study ----------------------------------------------------
    # K_CLUSTERS = best from validation curve (per domain).
    # KNN_K = 1 forced (decision #8).
    # Ablation now run in BOTH UI and UD: the best preprocessing for UI
    # is not guaranteed to be the best for UD (decision #13).
    if 1 in domains_to_run:
        print("\n=== Ablation Study - Domain 1 - User-Independent ===")
        _, best_prep_d1_ui = run_ablation_study(
            data1, data1_std, data1_denoised, evr1,
            labels1, users1, domain=1, setting="UI",
            k_clusters=best_k_clusters_d1, k_clusters_raw=best_k_raw_d1, knn_k=1)

        print("\n=== Ablation Study - Domain 1 - User-Dependent ===")
        _, best_prep_d1_ud = run_ablation_study(
            data1, data1_std, data1_denoised, evr1,
            labels1, users1, domain=1, setting="UD",
            k_clusters=best_k_clusters_d1, k_clusters_raw=best_k_raw_d1, knn_k=1)

    if 4 in domains_to_run:
        print("\n=== Ablation Study - Domain 4 - User-Independent ===")
        _, best_prep_d4_ui = run_ablation_study(
            data4, data4_std, data4_denoised, evr4,
            labels4, users4, domain=4, setting="UI",
            k_clusters=best_k_clusters_d4, k_clusters_raw=best_k_raw_d4, knn_k=1)

        print("\n=== Ablation Study - Domain 4 - User-Dependent ===")
        _, best_prep_d4_ud = run_ablation_study(
            data4, data4_std, data4_denoised, evr4,
            labels4, users4, domain=4, setting="UD",
            k_clusters=best_k_clusters_d4, k_clusters_raw=best_k_raw_d4, knn_k=1)

    # -- 7b. Force $1 to (a) raw ----------------------------------------------
    if 1 in domains_to_run:
        _force_dollar_raw(best_prep_d1_ui, data1)
        _force_dollar_raw(best_prep_d1_ud, data1)
    if 4 in domains_to_run:
        _force_dollar_raw(best_prep_d4_ui, data4)
        _force_dollar_raw(best_prep_d4_ud, data4)

    # -- 7c. Save best-preproc UI-vs-UD comparison ---------------------------
    if 1 in domains_to_run:
        _save_preproc_comparison(best_prep_d1_ui, best_prep_d1_ud, domain=1)
    if 4 in domains_to_run:
        _save_preproc_comparison(best_prep_d4_ui, best_prep_d4_ud, domain=4)

    # -- 9-10. Read out UI + UD results from the ablation dicts --------------
    if 1 in domains_to_run:
        main_d1_ui = _print_and_save_phase(1, "UI", best_prep_d1_ui)
        main_d1_ud = _print_and_save_phase(1, "UD", best_prep_d1_ud)
    if 4 in domains_to_run:
        main_d4_ui = _print_and_save_phase(4, "UI", best_prep_d4_ui)
        main_d4_ud = _print_and_save_phase(4, "UD", best_prep_d4_ud)

    # -- 11+15. Statistical tests (UI only, per consigne §5) -----------------
    if 1 in domains_to_run:
        print("\n=== Statistical Tests - Domain 1 ===")
        generate_pvalue_table(
            {m: main_d1_ui[m]["gu"] for m in METHODS_ORDER},
            domain=1, show=show)
    if 4 in domains_to_run:
        print("\n=== Statistical Tests - Domain 4 ===")
        generate_pvalue_table(
            {m: main_d4_ui[m]["gu"] for m in METHODS_ORDER},
            domain=4, show=show)

    # -- 12+16. Confusion matrix - best model per domain (UI) ----------------
    if 1 in domains_to_run:
        all_ui_d1 = {m: main_d1_ui[m]["mean"] for m in METHODS_ORDER}
        best_name_d1  = max(all_ui_d1, key=all_ui_d1.get)
        best_entry_d1 = main_d1_ui[best_name_d1]
        print(f"\n  Best model - Domain 1 (UI): {best_name_d1} "
              f"({all_ui_d1[best_name_d1]:.3f})  "
              f"[{best_entry_d1['condition']}]")
        draw_best_model_cm(best_name_d1,
                           best_entry_d1["data"],
                           best_entry_d1["evr"],
                           labels1, users1, folds_ui_1, domain=1, show=show)

    if 4 in domains_to_run:
        all_ui_d4 = {m: main_d4_ui[m]["mean"] for m in METHODS_ORDER}
        best_name_d4  = max(all_ui_d4, key=all_ui_d4.get)
        best_entry_d4 = main_d4_ui[best_name_d4]
        print(f"\n  Best model - Domain 4 (UI): {best_name_d4} "
              f"({all_ui_d4[best_name_d4]:.3f})  "
              f"[{best_entry_d4['condition']}]")
        draw_best_model_cm(best_name_d4,
                           best_entry_d4["data"],
                           best_entry_d4["evr"],
                           labels4, users4, folds_ui_4, domain=4, show=show)

    # -- 17. Overfitting diagnostic ------------------------------------------
    # Train-vs-test gap (parametric classifiers only; 1-NN methods have
    # train acc = 1.0 by construction and are excluded).  References:
    # Hastie et al. (2009) §7.10; Domingos (2012).
    print("\n=== Overfitting analysis: train-vs-test gap ===")
    overfit_rows = []
    domain_mains = []
    if 1 in domains_to_run:
        domain_mains.append((1, [("UI", main_d1_ui), ("UD", main_d1_ud)]))
    if 4 in domains_to_run:
        domain_mains.append((4, [("UI", main_d4_ui), ("UD", main_d4_ud)]))
    for domain, mains in domain_mains:
        for setting_tag, main_dict in mains:
            for m in ["DT", "RF", "LR"]:
                tr_accs = main_dict[m].get("train_accs")
                if not tr_accs:
                    continue
                overfit_rows.append({
                    "Domain"   : domain,
                    "Setting"  : setting_tag,
                    "Method"   : m,
                    "TrainAcc" : float(np.mean(tr_accs)),
                    "TestAcc"  : float(main_dict[m]["mean"]),
                    "Gap"      : float(np.mean(tr_accs)
                                       - main_dict[m]["mean"]),
                })
    overfit_csv = os.path.join(DIR_TBL_OVERFITTING, "overfitting_gap.csv")
    save_overfitting_table(overfit_rows, overfit_csv)

    # -- 17b. Learning curves (parametric classifiers, UI setting) -----------
    # Plots train vs validation accuracy as a function of training-set
    # size.  Wide gap = high variance.  Reference: Hastie et al. (2009)
    # §7.10.  We use the BEST preprocessing for each method on the UI
    # ablation (consistent with how the model would be deployed).
    print("\n=== Learning curves (UI, parametric classifiers) ===")
    LC_BUILDERS = {"DT": _lc_builder_dt, "RF": _lc_builder_rf,
                   "LR": _lc_builder_lr}
    lc_domain_mains = []
    if 1 in domains_to_run:
        lc_domain_mains.append((1, main_d1_ui))
    if 4 in domains_to_run:
        lc_domain_mains.append((4, main_d4_ui))
    for domain, main_dict in lc_domain_mains:
        for m in ["DT", "RF", "LR"]:
            entry = main_dict[m]
            data_lc, evr_lc = _data_for(
                entry,
                data1 if domain == 1 else data4,
                data1_std if domain == 1 else data4_std,
                data1_denoised if domain == 1 else data4_denoised,
                evr1 if domain == 1 else evr4,
            )
            print(f"  Domain {domain} UI - {m} learning curve ...")
            plot_learning_curve_method(
                LC_BUILDERS[m],
                data_lc,
                labels1 if domain == 1 else labels4,
                evr_lc,
                method_name=m, domain=domain, setting="UI",
                save_path=os.path.join(
                    DIR_FIG_LEARNING_CURVES,
                    f"lc_{m.lower()}_d{domain}_ui.png"),
                show=show)

    # -- 18. Final summary table ---------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - Mean accuracy +/- std")
    print("UI = user-independent | UD = user-dependent")
    print("=" * 70)

    summary_rows = []
    summary_domain_mains = []
    if 1 in domains_to_run:
        summary_domain_mains.append((1, {"UI": main_d1_ui, "UD": main_d1_ud}))
    if 4 in domains_to_run:
        summary_domain_mains.append((4, {"UI": main_d4_ui, "UD": main_d4_ud}))
    for domain, mains in summary_domain_mains:
        for setting_tag, main_dict in mains.items():
            for m in METHODS_ORDER:
                entry = main_dict[m]
                summary_rows.append({
                    "Domain"        : domain,
                    "Method"        : m,
                    "Setting"       : setting_tag,
                    "Preprocessing" : entry["condition"],
                    "Mean"          : entry["mean"],
                    "Std"           : entry["std"],
                    "Result"        : f"{entry['mean']:.3f} +/- {entry['std']:.3f}",
                })

    df_summary = pd.DataFrame(summary_rows)
    pivot = df_summary.pivot_table(
        index=["Domain", "Setting"], columns="Method",
        values="Result", aggfunc="first")
    print(pivot.to_string())

    print("\n  Preprocessing selected per method (UI / UD):")
    preproc_domains = []
    if 1 in domains_to_run:
        preproc_domains.append((1, best_prep_d1_ui, best_prep_d1_ud))
    if 4 in domains_to_run:
        preproc_domains.append((4, best_prep_d4_ui, best_prep_d4_ud))
    for domain_id, bp_ui, bp_ud in preproc_domains:
        print(f"\n  Domain {domain_id}:")
        for method in METHODS_ORDER:
            print(f"    {method:<16}: UI = {bp_ui[method]['condition']:<22}"
                  f"  UD = {bp_ud[method]['condition']}")

    summary_path = os.path.join(DIR_TBL_SUMMARY, "summary_results.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Saved -> {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
