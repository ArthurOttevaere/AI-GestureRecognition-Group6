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
    Acts as a lower-bound reference: a linear model on the same feature
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
        |Pearson r| > 0.95, discard the lower-variance member (Yu &
        Liu 2004, JMLR; Hall 1999).  Without this step, correlated
        features (e.g. x_range = x_max - x_min; bbox_diag derived from
        bbox_*) dilute each other's permutation importance (Strobl et
        al. 2007), causing the cumulative-importance cut to drop entire
        correlated clusters and destroy classifier performance.
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

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa - 3D projection
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed
from numba import njit
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ==============================================================================
# KEEPING THE LOGS IN A DEDICATED FILE FOR CONVENIENCE
# ==============================================================================
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # Writes in the terminal
        self.log.write(message)       # Writes in the txt file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Define your precise path here
chemin_log = "./Outputs/Documentation/output_execution.txt"

# Activate dual display (Console + File)
sys.stdout = Logger(chemin_log)
sys.stderr = sys.stdout  # Capture also error messages

# ==============================================================================
# CONFIGURATION
# ==============================================================================

_BASE = os.path.dirname(os.path.abspath(__file__))

DOMAIN1_DIR = os.path.join(_BASE, "GestureData_Mons",
                            "GestureDataDomain1_Mons", "Domain1_csv")
DOMAIN4_DIR = os.path.join(_BASE, "GestureData_Mons",
                            "GestureDataDomain4_Mons")

DOMAIN4_CLASS_NAMES = {
    1: "Cuboid",      2: "Cylinder",    3: "Sphere",
    4: "Rect. Pipe",  5: "Hemisphere",  6: "Cyl. Pipe",
    7: "Pyramid",     8: "Tetrahedron", 9: "Cone",
    10: "Toroid",
}

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

K_CLUSTERS       = 20
PCA_N_KEEP       = 2
RF_N_TREES       = 200
RF_MAX_FEATURES  = "sqrt"
KNN_K            = 1

# Inner CV folds for all GridSearchCV calls. 5-fold is the standard
# minimum for small datasets to keep the bias of error estimation
# acceptable (Varma & Simon 2006, BMC Bioinformatics; Cawley & Talbot
# 2010, JMLR). With ~900 train samples and 10 classes, 5-fold inner CV
# leaves ~180 samples per inner test fold, which is sufficient.
_INNER_CV_FOLDS = 5

# RF GridSearchCV grid (regularised to limit overfitting on this small dataset; Breiman 2001).
RF_GRID = {
    "n_estimators"     : [100, 200],
    "max_depth"        : [5, 10, 15],
    "min_samples_split": [5, 10],
    "min_samples_leaf" : [2, 5],
}
RF_GRID_INNER_CV = _INNER_CV_FOLDS

# Decision Tree GridSearchCV grid (Quinlan 1986; Breiman et al. 1984 CART) -> regularised to limit overfitting on this small dataset. 
DT_GRID = {
    "max_depth"        : [5, 10, 15],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf" : [2, 5, 10],
    "criterion"        : ["gini", "entropy"],
}
DT_GRID_INNER_CV = _INNER_CV_FOLDS

# Logistic Regression GridSearchCV grid (Cox 1958; Hosmer et al. 2013).
# Multinomial L2-penalised LR via the lbfgs solver.  max_iter = 5000:
# lbfgs convergence requires both scaled inputs and sufficient iterations;
# StandardScaler in the Pipeline handles the scaling, and 5000 iterations
# guarantees convergence across all grid cells (sklearn docs).
# Keys use Pipeline prefix "lr__".
LR_GRID = {
    "lr__C"      : [0.01, 0.1, 1.0, 10.0, 100.0],
}
LR_GRID_INNER_CV = _INNER_CV_FOLDS

# Per-fold feature selection (Guyon & Elisseeff 2003, JMLR; Strobl et al.
# 2007, BMC Bioinformatics).  Three-stage pipeline (see _select_features_
# per_fold): variance filter -> correlation deduplication (Yu & Liu 2004,
# JMLR) -> permutation importance on held-out split (Breiman 2001; Ambroise
# & McLachlan 2002, PNAS).  N_REPEATS controls the permutation count.
FEATURE_SELECTION_CUM_THRESHOLD = 0.95 # cumulative importance threshold for selection (Guyon & Elisseeff 2003, JMLR)
FEATURE_SELECTION_N_REPEATS     = 10
# Pearson |r| threshold for correlation-based redundancy removal.
# Features with |r| > FEATURE_CORR_THRESHOLD are considered redundant;
# the member with lower variance is discarded (Yu & Liu 2004, JMLR §4).
FEATURE_CORR_THRESHOLD          = 0.99 # Yu & Liu (2004) use 0.95, but this is relaxed to 0.99 to preserve more features for the RF permutation importance step, which is sensitive to feature count (Strobl et al. 2007).

# Validation-curve scan ranges (Section 5 of instructions)
VC_K_CLUSTERS = [5, 10, 15, 20, 25, 30, 40]
VC_KNN_K      = [1, 3, 5, 7, 9]

# $1 Recognizer hyperparameters (Kratz & Rohs, 2010)
DOLLAR_N         = 64           # number of resampled points
DOLLAR_L         = 1.0          # side of the normalised cube
DOLLAR_SCORE_DENOM = 0.5 * np.sqrt(3.0) * DOLLAR_L ** 2   # MSE normaliser (Kratz & Rohs 2010)
DOLLAR_EPSILON     = 0.60   # confidence threshold for scoring heuristic (Kratz & Rohs 2010);


# ---------------------------------------------------------------------------
# Output folder organisation
# ---------------------------------------------------------------------------
# All artefacts produced by this script are written under Outputs/ in a
# nested structure that groups files by type and purpose.

OUTPUTS_DIR              = os.path.join(_BASE, "Outputs")
DIR_FIG_EXPLORE          = os.path.join(OUTPUTS_DIR, "figures", "exploratory")
DIR_FIG_PCA              = os.path.join(OUTPUTS_DIR, "figures", "pca_denoising")
DIR_FIG_VC               = os.path.join(OUTPUTS_DIR, "figures", "validation_curves")
DIR_FIG_FI               = os.path.join(OUTPUTS_DIR, "figures", "feature_importance")
DIR_FIG_CM               = os.path.join(OUTPUTS_DIR, "figures", "confusion_matrices")
DIR_FIG_STATS            = os.path.join(OUTPUTS_DIR, "figures", "statistical_tests")
DIR_FIG_LEARNING_CURVES  = os.path.join(OUTPUTS_DIR, "figures", "learning_curves")
DIR_TBL_ABLATION         = os.path.join(OUTPUTS_DIR, "tables",  "ablation")
DIR_TBL_FOLDS            = os.path.join(OUTPUTS_DIR, "tables",  "fold_results")
DIR_TBL_STATS            = os.path.join(OUTPUTS_DIR, "tables",  "statistical_tests")
DIR_TBL_SUMMARY          = os.path.join(OUTPUTS_DIR, "tables",  "summary")
DIR_TBL_OVERFITTING      = os.path.join(OUTPUTS_DIR, "tables",  "overfitting")
DIR_TBL_FEAT_SEL         = os.path.join(OUTPUTS_DIR, "tables",  "feature_selection")
DIR_DOC                  = os.path.join(OUTPUTS_DIR, "Documentation")

for _d in [DIR_FIG_EXPLORE, DIR_FIG_PCA, DIR_FIG_VC, DIR_FIG_FI,
           DIR_FIG_CM, DIR_FIG_STATS, DIR_FIG_LEARNING_CURVES,
           DIR_TBL_ABLATION, DIR_TBL_FOLDS, DIR_TBL_STATS, DIR_TBL_SUMMARY,
           DIR_TBL_OVERFITTING, DIR_TBL_FEAT_SEL, DIR_DOC]:
    os.makedirs(_d, exist_ok=True)


# ==============================================================================
# 1.  DATA LOADING
# ==============================================================================

def load_domain1(folder_path: str) -> tuple[list, list, list]:
    """
    Load all Domain 1 CSV files.
    Returns
    -------
    data   : list of np.ndarray (T, 3)
    labels : list of int (0-9)
    users  : list of int (0-9)
    """
    data, labels, users = [], [], []
    pattern = re.compile(r"Subject(\d+)-(\d+)-(\d+)\.csv", re.IGNORECASE)

    for filename in sorted(os.listdir(folder_path)):
        m = pattern.match(filename)
        if m is None:
            continue
        subject  = int(m.group(1)) - 1
        gesture  = int(m.group(2))
        filepath = os.path.join(folder_path, filename)
        df       = pd.read_csv(filepath, header=0,
                               names=["x", "y", "z", "t"])
        coords   = df[["x", "y", "z"]].values.astype(float)
        data.append(coords)
        labels.append(gesture)
        users.append(subject)

    return data, labels, users


def load_domain4(folder_path: str) -> tuple[list, list, list]:
    """Load all Domain 4 plain-text files (no extension)."""
    data, labels, users = [], [], []

    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r") as fh:
                lines = fh.readlines()
        except (UnicodeDecodeError, PermissionError):
            continue

        gesture, subject, data_start = None, None, 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s.lower().startswith("class id"):
                gesture = int(s.split("=")[1].strip())
            elif s.lower().startswith("user id"):
                subject = int(s.split("=")[1].strip()) - 1
            elif s.lower().startswith("<x>"):
                data_start = i + 1
                break

        if gesture is None or subject is None:
            continue

        rows = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]),
                              float(parts[1]),
                              float(parts[2])])
            except ValueError:
                continue

        if not rows:
            continue

        data.append(np.array(rows, dtype=float))
        labels.append(gesture)
        users.append(subject)

    return data, labels, users


def check_completeness(labels: list, users: list,
                        domain_name: str) -> None:
    counts: dict = defaultdict(int)
    for u, g in zip(users, labels):
        counts[(u, g)] += 1
    issues = [(u, g, n) for (u, g), n in counts.items() if n != 10]
    if issues:
        print(f"  [WARNING] {domain_name} - incomplete groups:")
        for u, g, n in sorted(issues):
            print(f"    user={u}, gesture={g} -> {n} rep(s) (expected 10)")
    else:
        print(f"  {domain_name}: {len(labels)} sequences - completeness OK "
              f"({len(set(users))} users x "
              f"{len(set(labels))} gestures x 10 reps)")


def print_dataset_info(data: list, labels: list,
                        users: list, domain_name: str) -> int:
    lengths = [len(seq) for seq in data]
    print(f"\n{'-'*55}")
    print(f"  {domain_name}")
    print(f"{'-'*55}")
    print(f"  Total sequences : {len(data)}")
    print(f"  Subjects        : {sorted(set(users))}")
    print(f"  Gesture classes : {sorted(set(labels))}")
    print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
    check_completeness(labels, users, domain_name)
    return int(max(lengths))


# ==============================================================================
# 2.  EXPLORATORY VISUALISATION
# ==============================================================================

def plot_sequence_lengths(data: list, labels: list,
                           domain_name: str,
                           save_path: str | None = None) -> None:
    gesture_classes = sorted(set(labels))
    lengths_by_class = [
        [len(data[i]) for i, g in enumerate(labels) if g == gc]
        for gc in gesture_classes
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(lengths_by_class,
               tick_labels=[str(g) for g in gesture_classes],
               patch_artist=True)
    ax.set_xlabel("Gesture class")
    ax.set_ylabel("Number of time steps")
    ax.set_title(f"{domain_name} - Sequence lengths per gesture class")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_gesture_samples(data: list, labels: list, users: list,
                          domain_name: str, n_classes: int = 4,
                          n_subjects: int = 3,
                          save_path: str | None = None) -> None:
    gesture_classes = sorted(set(labels))[:n_classes]
    subject_ids     = sorted(set(users))[:n_subjects]
    fig = plt.figure(figsize=(4 * n_subjects, 3.5 * n_classes))
    plot_idx = 1
    for gc in gesture_classes:
        for s in subject_ids:
            samples = [data[i] for i in range(len(data))
                       if labels[i] == gc and users[i] == s]
            ax = fig.add_subplot(n_classes, n_subjects, plot_idx,
                                  projection="3d")
            for seq in samples[:3]:
                ax.plot(seq[:, 0], seq[:, 1], seq[:, 2],
                        alpha=0.6, linewidth=0.8)
                ax.scatter(*seq[0],  color="green", s=12, zorder=5)
                ax.scatter(*seq[-1], color="red",   s=12, zorder=5)
            ax.set_title(f"Gesture {gc} | Subject {s}", fontsize=8)
            ax.set_xlabel("x", fontsize=6)
            ax.set_ylabel("y", fontsize=6)
            ax.set_zlabel("z", fontsize=6)
            ax.tick_params(labelsize=5)
            plot_idx += 1
    plt.suptitle(
        f"{domain_name} - 3D trajectories (green=start, red=end)",
        fontsize=10, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# 3.  PRE-PROCESSING
# ==============================================================================

def standardize_gestures(data: list) -> list:
    standardized = []
    for seq in data:
        mean = np.mean(seq, axis=0)
        std  = np.std(seq,  axis=0)
        std[std == 0] = 1.0
        standardized.append((seq - mean) / std)
    return standardized


def pca_denoise_gesture(sequence: np.ndarray,
                         n_keep: int = PCA_N_KEEP) -> tuple[np.ndarray,
                                                             np.ndarray]:
    pca = PCA(n_components=3)
    projected = pca.fit_transform(sequence)
    projected_truncated          = projected.copy()
    projected_truncated[:, n_keep:] = 0.0
    denoised = pca.inverse_transform(projected_truncated)
    return denoised, pca.explained_variance_ratio_


def apply_pca_denoising(data: list,
                         n_keep: int = PCA_N_KEEP) -> tuple[list, list]:
    denoised, evr_list = [], []
    for seq in data:
        d, evr = pca_denoise_gesture(seq, n_keep)
        denoised.append(d)
        evr_list.append(evr)
    return denoised, evr_list


def summarise_pca_denoising(data_std: list, domain_name: str,
                              n_keep: int = PCA_N_KEEP,
                              save_path: str | None = None) -> None:
    evrs = np.array([pca_denoise_gesture(seq, n_keep)[1]
                     for seq in data_std])

    print(f"\n  Per-gesture PCA denoising - {domain_name}")
    for c in range(evrs.shape[1]):
        print(f"    PC{c+1}: mean EVR = {evrs[:, c].mean():.3f} "
              f"+/- {evrs[:, c].std():.3f}")
    kept_var    = evrs[:, :n_keep].sum(axis=1).mean() * 100
    removed_var = evrs[:, n_keep:].sum(axis=1).mean() * 100
    print(f"    Variance kept   (PC1+PC2): {kept_var:.1f}%")
    print(f"    Variance removed (PC3+): {removed_var:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 3))
    for c in range(evrs.shape[1]):
        ax.hist(evrs[:, c], bins=30, alpha=0.6, label=f"PC{c+1}")
    ax.axvline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("Explained variance ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"{domain_name} - Per-gesture PCA EVR distribution "
                 f"(n_keep={n_keep})")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def encode_with_centroids(data_3d: list,
                           centroids: np.ndarray) -> list:
    sequences = []
    for seq in data_3d:
        diffs  = seq[:, None, :] - centroids[None, :, :]
        dists  = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        sequences.append(labels)
    return sequences


def fit_kmeans_and_encode(train_data: list,
                           test_data: list,
                           k: int = K_CLUSTERS,
                           random_state: int = 42) -> tuple[list, list]:
    all_train_points = np.vstack(train_data)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(all_train_points)
    centroids = kmeans.cluster_centers_

    train_seq = encode_with_centroids(train_data, centroids)
    test_seq  = encode_with_centroids(test_data,  centroids)
    return train_seq, test_seq


# ==============================================================================
# 4.  BASELINE METHODS
# ==============================================================================

@njit(cache=True)
def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two 3D sequences.

    The local cost is the Euclidean distance between 3D points.
    The warping path uses the standard three-move DTW (diagonal, left,
    up), with the diagonal weighted x2 to discourage over-stretching.

    The raw accumulated cost g[I, J] is normalised by (I + J) to make
    the distance comparable across sequence pairs of different lengths.
    This normalisation is not part of the original DTW formulation
    (Sakoe & Chiba, 1978) but is standard practice in gesture
    recognition literature (cf. Mueller, 2007, "Information Retrieval
    for Music and Motion", Ch. 4).

    Returns
    -------
    float
        Normalised DTW distance in [0, +inf).
    """
    I, J = len(seq1), len(seq2)
    g = np.full((I + 1, J + 1), np.inf)
    g[0, 0] = 0.0

    for i in range(1, I + 1):
        for j in range(1, J + 1):
            dx = seq1[i-1, 0] - seq2[j-1, 0]
            dy = seq1[i-1, 1] - seq2[j-1, 1]
            dz = seq1[i-1, 2] - seq2[j-1, 2]
            d  = (dx*dx + dy*dy + dz*dz) ** 0.5
            g[i, j] = min(g[i-1, j]   + d,
                          min(g[i-1, j-1] + 2.0 * d,
                              g[i,   j-1] + d))

    return g[I, J] / (I + J)


@njit(cache=True)
def edit_distance(seq1: np.ndarray, seq2: np.ndarray) -> int:
    L1, L2 = len(seq1), len(seq2)
    mat = np.zeros((L1 + 1, L2 + 1), dtype=np.int64)
    for i in range(L1 + 1):
        mat[i, 0] = i
    for j in range(L2 + 1):
        mat[0, j] = j
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            delta     = 0 if seq1[i-1] == seq2[j-1] else 1
            mat[i, j] = min(mat[i-1, j]   + 1,
                            min(mat[i, j-1]   + 1,
                                mat[i-1, j-1] + delta))
    return mat[L1, L2]


# ==============================================================================
# 4b.  $1 RECOGNIZER -- 3D ADAPTATION FOLLOWING KRATZ & ROHS (2010)
#     [Advanced method -- placed here for implementation proximity to baseline
#      distance functions; classified as advanced per module docstring Phase 4]
# ==============================================================================
# The 2D $1 Recognizer of Wobbrock, Wilson & Li (2007) is extended to 3D
# according to the canonical procedure of Kratz & Rohs (2010), "A $3
# Gesture Recognizer", Proc. IUI '10, pp. 341-344.
#
# Pipeline
# --------
#   1. Resample to N=64 equidistant 3D points (linear interpolation along
#      cumulative arc length).
#   2. Rotate so that the first resampled point lies along the centroid
#      direction.  The rotation axis is the unit vector pâ x c (cross
#      product); the angle is acos((pâ . c) / (||pâ|| ||c||)).  Rotation
#      applied with Rodrigues' formula.  Degenerate case (pâ collinear
#      with c) -> identity rotation.
#      NOTE: rotation must be applied BEFORE translation so that both p0
#      and c are non-zero (centroid not yet at origin).
#   3. Translate centroid to the origin (after rotation).
#   4. Uniformly rescale so that the longest bounding-box edge equals
#      DOLLAR_L (=1.0). This is the "normalised cube of side l" of
#      Kratz & Rohs (2010).  No axis-by-axis scaling: this avoids the
#      division-by-zero issue on quasi-planar gestures.
#
# Score
# -----
#   S = 1 - d / (0.5 * sqrt(3) * l^2)
# where d is the MSE (mean squared point-to-point distance) between the
# preprocessed candidate and the preprocessed template (Kratz & Rohs,
# 2010).  The GSS (Golden Section Search) further refines alignment on
# the 3 axes before computing d.
#
# Templates are preprocessed only once and cached, as required by
# Wobbrock et al. (2007, "For gestures serving as templates, Steps 1-3
# should be carried out once on the raw input points.").
# ==============================================================================

def _dollar_path_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def dollar_resample(points: np.ndarray, n: int = DOLLAR_N) -> np.ndarray:
    """Step 1 of Kratz & Rohs (2010): resample to n equidistant 3D points."""
    total = _dollar_path_length(points)
    if total == 0.0 or len(points) < 2:
        return np.tile(points[0], (n, 1))

    interval   = total / (n - 1)
    D          = 0.0
    new_points = [points[0].copy()]
    pts        = points.copy()

    i = 1
    while i < len(pts) and len(new_points) < n:
        d = float(np.linalg.norm(pts[i] - pts[i - 1]))
        if D + d >= interval:
            frac = (interval - D) / d
            q    = pts[i - 1] + frac * (pts[i] - pts[i - 1])
            new_points.append(q)
            pts = np.insert(pts, i, q, axis=0)
            D = 0.0
        else:
            D += d
        i += 1

    while len(new_points) < n:
        new_points.append(pts[-1].copy())

    return np.array(new_points[:n], dtype=float)


def _dollar_centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def _dollar_translate_to_origin(points: np.ndarray) -> np.ndarray:
    return points - _dollar_centroid(points)


def _rodrigues_rotate(points: np.ndarray,
                       axis: np.ndarray,
                       angle: float) -> np.ndarray:
    """
    Rotate a (T, 3) array of points around a unit axis by `angle` radians,
    using Rodrigues' rotation formula (Kratz & Rohs, 2010).
        v_rot = v cos(t) + (k x v) sin(t) + k (k . v)(1 - cos(t))
    """
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1.0 - c
    kx, ky, kz = axis
    K = np.array([[ 0.0, -kz,  ky],
                  [  kz, 0.0, -kx],
                  [ -ky,  kx, 0.0]])
    R = np.eye(3) * c + K * s + np.outer(axis, axis) * one_minus_c
    return points @ R.T


def _dollar_align_to_indicative_axis(points: np.ndarray) -> np.ndarray:
    """
    Step 3 of Kratz & Rohs (2010): rotate so that the first point pâ aligns
    with the centroid direction. Translation to origin must be applied
    first.
    The rotation axis is pâ x c (unit vector); the angle is the arccos of
    the normalised dot product. Degenerate cases (pâ or c with zero norm,
    or pâ collinear with c) -> identity rotation.
    """
    if len(points) == 0:
        return points
    p1 = points[0]
    c  = _dollar_centroid(points)

    n_p1 = float(np.linalg.norm(p1))
    n_c  = float(np.linalg.norm(c))
    if n_p1 < 1e-12 or n_c < 1e-12:
        return points

    cos_theta = float(np.dot(p1, c) / (n_p1 * n_c))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta     = float(np.arccos(cos_theta))

    cross   = np.cross(p1, c)
    n_cross = float(np.linalg.norm(cross))
    if n_cross < 1e-12 or theta < 1e-9:
        # Degenerate: pâ already collinear with c -> identity.
        return points
    axis = cross / n_cross
    return _rodrigues_rotate(points, axis, theta)


def _dollar_scale_cube(points: np.ndarray,
                        l: float = DOLLAR_L) -> np.ndarray:
    """
    Step 4 (Kratz & Rohs, 2010): uniform rescaling INSIDE a normalised
    cube of side l.  The longest bounding-box edge becomes l. Avoids the
    division-by-zero issue of axis-by-axis scaling on quasi-planar
    gestures.
    """
    extents = points.max(axis=0) - points.min(axis=0)
    max_ext = float(extents.max())
    if max_ext < 1e-12:
        return points
    return points * (l / max_ext)


# -- Golden Section Search (GSS) helpers — Numba JIT --------------------------
# Kratz & Rohs (2010), "Search for Minimum Distance at Best Angle":
#   phi = 0.5*(sqrt(5)-1), angular range [-180°,180°], cutoff = 2°, 11 iters.
# Three functions:
#   _rotate_1axis_nb : rotate (N,3) array around one principal axis
#   _mse_nb          : mean squared Euclidean distance between two (N,3) arrays
#   _dollar_gss_mse  : sequential 3-axis GSS returning the minimum MSE
# All are @njit(cache=True) for maximum speed (called 10^4+ times in CV).

@njit(cache=True)
def _rotate_1axis_nb(pts: np.ndarray, axis_idx: int,
                     theta: float) -> np.ndarray:
    """Rotate (N,3) array around principal axis axis_idx by theta radians."""
    n   = pts.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    c   = np.cos(theta)
    s   = np.sin(theta)
    for i in range(n):
        x = pts[i, 0]
        y = pts[i, 1]
        z = pts[i, 2]
        if axis_idx == 0:           # x-axis
            out[i, 0] = x
            out[i, 1] = y * c - z * s
            out[i, 2] = y * s + z * c
        elif axis_idx == 1:         # y-axis
            out[i, 0] =  x * c + z * s
            out[i, 1] =  y
            out[i, 2] = -x * s + z * c
        else:                       # z-axis
            out[i, 0] = x * c - y * s
            out[i, 1] = x * s + y * c
            out[i, 2] = z
    return out


@njit(cache=True)
def _mse_nb(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared point-to-point distance between two (N,3) arrays."""
    n     = a.shape[0]
    total = 0.0
    for i in range(n):
        dx = a[i, 0] - b[i, 0]
        dy = a[i, 1] - b[i, 1]
        dz = a[i, 2] - b[i, 2]
        total += dx * dx + dy * dy + dz * dz
    return total / n


@njit(cache=True)
def _dollar_gss_mse(cand: np.ndarray, tmpl: np.ndarray) -> float:
    """
    Golden Section Search over 3 rotation axes (Kratz & Rohs 2010).
    Applies GSS sequentially on x, y, z axes; rotates candidate to best
    angle on each axis before moving to the next.
    Parameters: phi = 0.5*(sqrt(5)-1), cutoff = 2 deg = pi/90, 11 iters.
    Returns the minimum MSE after optimal 3-axis alignment.
    """
    phi    = 0.5 * (-1.0 + np.sqrt(5.0))
    cutoff = np.pi / 90.0       # 2 degrees
    n      = cand.shape[0]

    for axis in range(3):
        a_ang = -np.pi
        b_ang =  np.pi
        for _ in range(11):
            if (b_ang - a_ang) <= cutoff:
                break
            c1 = b_ang - phi * (b_ang - a_ang)
            c2 = a_ang + phi * (b_ang - a_ang)

            # Compute d1 (MSE at c1) and d2 (MSE at c2) inline — no alloc
            cos1 = np.cos(c1);  sin1 = np.sin(c1)
            cos2 = np.cos(c2);  sin2 = np.sin(c2)
            d1 = 0.0;  d2 = 0.0
            for i in range(n):
                x = cand[i, 0];  y = cand[i, 1];  z = cand[i, 2]
                if axis == 0:
                    rx1 = x;  ry1 = y*cos1 - z*sin1;  rz1 = y*sin1 + z*cos1
                    rx2 = x;  ry2 = y*cos2 - z*sin2;  rz2 = y*sin2 + z*cos2
                elif axis == 1:
                    rx1 =  x*cos1 + z*sin1;  ry1 = y;  rz1 = -x*sin1 + z*cos1
                    rx2 =  x*cos2 + z*sin2;  ry2 = y;  rz2 = -x*sin2 + z*cos2
                else:
                    rx1 = x*cos1 - y*sin1;  ry1 = x*sin1 + y*cos1;  rz1 = z
                    rx2 = x*cos2 - y*sin2;  ry2 = x*sin2 + y*cos2;  rz2 = z
                ex1 = rx1 - tmpl[i, 0];  ey1 = ry1 - tmpl[i, 1]
                ez1 = rz1 - tmpl[i, 2]
                ex2 = rx2 - tmpl[i, 0];  ey2 = ry2 - tmpl[i, 1]
                ez2 = rz2 - tmpl[i, 2]
                d1 += ex1*ex1 + ey1*ey1 + ez1*ez1
                d2 += ex2*ex2 + ey2*ey2 + ez2*ez2

            if d1 < d2:
                b_ang = c2
            else:
                a_ang = c1

        best_angle = (a_ang + b_ang) * 0.5
        cand = _rotate_1axis_nb(cand, axis, best_angle)

    return _mse_nb(cand, tmpl)


def dollar_preprocess(points: np.ndarray,
                       n: int   = DOLLAR_N,
                       l: float = DOLLAR_L) -> np.ndarray:
    """
    Apply Steps 1-4 of the $3 (Kratz & Rohs, 2010) preprocessing.
    Used once on training templates (cached) and once on each candidate.
    Correct step order (Kratz & Rohs 2010):
      1. Resample to N equidistant points.
      2. Rotate to indicative angle (p0 and centroid c are both non-zero
         here — centroid has NOT yet been moved to origin).
      3. Translate centroid to origin (after rotation so c stays valid).
      4. Scale uniformly to unit cube of side l.
    NOTE: translation must come AFTER rotation. If translation is applied
    first, the centroid becomes (0,0,0) and the indicative-angle formula
    n_c < 1e-12 guard triggers, making the rotation a no-op.
    """
    pts = dollar_resample(points, n)
    pts = _dollar_align_to_indicative_axis(pts)   # step 2: c valid here
    pts = _dollar_translate_to_origin(pts)        # step 3: centroid → 0
    pts = _dollar_scale_cube(pts, l)              # step 4
    return pts


def _dollar_path_distance(a: np.ndarray, b: np.ndarray) -> float:
    """MSE (mean squared point-to-point distance) between two same-length 3D paths.
    Required by Kratz & Rohs (2010) score formula: S = 1 - d/(0.5*sqrt(3)*l^2).
    """
    return float(np.mean(np.sum((a - b) ** 2, axis=1)))


def dollar_score(distance: float, l: float = DOLLAR_L) -> float:
    """
    Confidence score in [0, 1] (Kratz & Rohs, 2010, 3D MSE adaptation):
        S = 1 - d / (0.5 * sqrt(3) * l^2)
    where d is the MSE between the two preprocessed paths.
    Clipped to [0, 1] in case of numerical edge effects.
    """
    s = 1.0 - distance / (0.5 * np.sqrt(3.0) * l ** 2)
    return float(max(0.0, min(1.0, s)))


def dollar_recognize(candidate_pre: np.ndarray,
                      templates_pre: list,
                      template_labels: list,
                      l: float = DOLLAR_L,
                      allow_rejection: bool = False,
                      epsilon: float = DOLLAR_EPSILON
                      ) -> tuple[int, float, list]:
    """
    Step 5 of Kratz & Rohs (2010) recognition: rank all (preprocessed)
    templates against the (preprocessed) candidate by GSS-refined MSE,
    and return:
        (best_label, best_score, ranked_list)
    where ranked_list is a sorted N-best list:
        [(label, distance, score), ...]   sorted by distance ascending.

    allow_rejection=False (default):
        Always return the best-scoring candidate (1-Best, no rejection).
        Use this in cross-validation so the $3 is directly comparable to
        RF/LR/DT on accuracy (force prediction on every sample).

    allow_rejection=True:
        Apply the Kratz & Rohs (2010) scoring heuristic:
          - If best_score > 1.1*epsilon → return best candidate.
          - Elif within top-3, two entries of the same class both have
            score > 0.95*epsilon → return that class.
          - Else → return (-1, 0.0, ranked) meaning "not recognized".
    """
    distances = [
        _dollar_gss_mse(candidate_pre, t) for t in templates_pre
    ]
    order   = np.argsort(distances)
    ranked  = [(template_labels[k], float(distances[k]),
                dollar_score(distances[k], l)) for k in order]
    best_label, _best_d, best_s = ranked[0]

    if not allow_rejection:
        return int(best_label), float(best_s), ranked

    # --- Kratz & Rohs (2010) scoring heuristic ---
    if best_s > 1.1 * epsilon:
        return int(best_label), float(best_s), ranked

    top3 = ranked[:3]
    counts: dict = {}
    for lbl, _d, sc in top3:
        if sc > 0.95 * epsilon:
            counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, cnt in counts.items():
        if cnt >= 2:
            sc_for_lbl = max(sc for l2, _d, sc in top3 if l2 == lbl)
            return int(lbl), float(sc_for_lbl), ranked

    return -1, 0.0, ranked   # gesture not recognized


# ==============================================================================
# 5.  k-NN CLASSIFIER  (generic, distance-based)
# ==============================================================================

def knn_predict(test_item, train_items: list, train_labels: list,
                distance_fn, k: int = KNN_K) -> int:
    distances = np.array([distance_fn(test_item, ref)
                          for ref in train_items])
    k_nearest = np.argsort(distances)[:k]
    k_labels  = [train_labels[idx] for idx in k_nearest]
    return max(set(k_labels), key=k_labels.count)


def knn_predict_from_distances(distances: np.ndarray,
                                train_labels: list,
                                k: int = KNN_K) -> int:
    """
    kNN over a precomputed distance vector. Used by the cached $1
    pipeline (templates preprocessed only once, then plain L2 distance
    on aligned 3D paths).
    """
    k_nearest = np.argsort(distances)[:k]
    k_labels  = [train_labels[idx] for idx in k_nearest]
    return max(set(k_labels), key=k_labels.count)


# ==============================================================================
# 6.  FEATURE EXTRACTION  (RF input)
# ==============================================================================

def extract_features(sequence: np.ndarray,
                      evr: np.ndarray | None = None,
                      feature_mask: list | None = None) -> np.ndarray:
    """
    Fixed-length feature vector for the RF classifier.
    Composition (53 with EVR, 50 without):
      - global per-axis stats (21)
      - kinematics (6)
      - total arc-length (1)
      - bounding box + diagonal (4)
      - 3D curvature mean/std/total (3)
      - per-segment speed and displacement, begin/middle/end (6)
      - per-axis path length (3)
      - PCA EVR (3, optional)

    Parameters
    ----------
    feature_mask : list of str, optional
        If provided, only features whose name appears in this list are
        returned. Computed from RF feature importances post-hoc.
        None = return all features (default).
    """
    features = []

    for i in range(3):
        axis = sequence[:, i]
        features.extend([
            float(np.mean(axis)), float(np.std(axis)),
            float(np.min(axis)),  float(np.max(axis)),
            float(np.max(axis) - np.min(axis)),
            float(pd.Series(axis).skew()),
            float(pd.Series(axis).kurt()),
        ])

    velocity = np.diff(sequence, axis=0)
    speed    = np.linalg.norm(velocity, axis=1)
    features.extend([float(np.mean(speed)), float(np.std(speed)),
                     float(np.max(speed)),  float(np.min(speed))])

    if len(velocity) > 1:
        accel      = np.diff(velocity, axis=0)
        accel_norm = np.linalg.norm(accel, axis=1)
        features.extend([float(np.mean(accel_norm)),
                         float(np.std(accel_norm))])
    else:
        features.extend([0.0, 0.0])

    features.append(float(np.sum(speed)))

    bbox = np.max(sequence, axis=0) - np.min(sequence, axis=0)
    features.extend(bbox.tolist())
    features.append(float(np.linalg.norm(bbox)))

    if len(velocity) > 1:
        v_norm = velocity / (
            np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-8)
        cos_angles = np.clip(
            np.sum(v_norm[:-1] * v_norm[1:], axis=1), -1.0, 1.0)
        angles = np.arccos(cos_angles)
        features.extend([float(np.mean(angles)),
                         float(np.std(angles)),
                         float(np.sum(angles))])
    else:
        features.extend([0.0, 0.0, 0.0])

    T      = len(sequence)
    thirds = [slice(0, T // 3),
              slice(T // 3, 2 * T // 3),
              slice(2 * T // 3, T)]
    for sl in thirds:
        seg = sequence[sl]
        if len(seg) < 2:
            features.append(0.0)
        else:
            seg_vel   = np.diff(seg, axis=0)
            seg_speed = np.linalg.norm(seg_vel, axis=1)
            features.append(float(np.mean(seg_speed)))

    for sl in thirds:
        seg = sequence[sl]
        if len(seg) < 2:
            features.append(0.0)
        else:
            seg_disp = np.linalg.norm(np.diff(seg, axis=0), axis=1)
            features.append(float(np.mean(seg_disp)))

    for i in range(3):
        features.append(
            float(np.sum(np.abs(np.diff(sequence[:, i])))))

    if evr is not None:
        features.extend(evr.tolist())

    feat_array = np.array(features, dtype=float)
    if feature_mask is not None:
        names = feature_names(with_evr=(evr is not None))
        mask_idx = [i for i, n in enumerate(names) if n in feature_mask]
        return feat_array[mask_idx]
    return feat_array


def feature_names(with_evr: bool) -> list:
    names = []
    for ax in ["x", "y", "z"]:
        for stat in ["mean", "std", "min", "max", "range", "skew", "kurt"]:
            names.append(f"{ax}_{stat}")
    names += ["speed_mean", "speed_std", "speed_max", "speed_min",
              "accel_mean", "accel_std", "arc_length",
              "bbox_x", "bbox_y", "bbox_z", "bbox_diag",
              "curv_mean", "curv_std", "curv_total",
              "speed_seg1", "speed_seg2", "speed_seg3",
              "disp_seg1", "disp_seg2", "disp_seg3",
              "path_x", "path_y", "path_z"]
    if with_evr:
        names += ["evr_pc1", "evr_pc2", "evr_pc3"]
    return names


def build_feature_dataset(data: list,
                           evr_list: list | None = None,
                           feature_mask: list | None = None) -> np.ndarray:
    if evr_list is not None:
        return np.array([extract_features(seq, evr, feature_mask)
                         for seq, evr in zip(data, evr_list)])
    return np.array([extract_features(seq, None, feature_mask)
                     for seq in data])


# ==============================================================================
# 7.  CROSS-VALIDATION FOLD INDICES
# ==============================================================================

def _ui_fold_indices(users: list) -> list[tuple[list, list]]:
    unique_users = sorted(set(users))
    folds = []
    for u in unique_users:
        tr = [i for i, usr in enumerate(users) if usr != u]
        te = [i for i, usr in enumerate(users) if usr == u]
        folds.append((tr, te))
    return folds


def _ud_fold_indices(labels: list,
                      users: list) -> list[tuple[list, list, list]]:
    unique_users    = sorted(set(users))
    gesture_classes = sorted(set(labels))

    g2u2idx: dict = {g: {u: [] for u in unique_users}
                     for g in gesture_classes}
    for i, (lbl, usr) in enumerate(zip(labels, users)):
        g2u2idx[lbl][usr].append(i)

    n_folds = min(
        len(g2u2idx[g][u])
        for g in gesture_classes for u in unique_users
    )

    folds = []
    for fold in range(n_folds):
        te = [g2u2idx[g][u][fold]
              for g in gesture_classes for u in unique_users]
        tr = [g2u2idx[g][u][r]
              for g in gesture_classes for u in unique_users
              for r in range(n_folds) if r != fold]
        te_users = [u
                    for g in gesture_classes for u in unique_users]
        assert all(users[te[i]] == te_users[i] for i in range(len(te))), \
            "te / te_users alignment broken in _ud_fold_indices"
        folds.append((tr, te, te_users))
    return folds


# ==============================================================================
# 7b.  PER-(GESTURE, USER) ACCURACY HELPER  -- for Wilcoxon n=100
# ==============================================================================

def _aggregate_gu_accuracy(per_sample_correct: dict,
                            labels: list, users: list) -> np.ndarray:
    """
    Build the n=100 vector of per-(gesture, user) accuracies, ordered by
    (gesture asc, user asc), to align across methods.

    per_sample_correct: dict {sample_idx: 0/1}
    """
    gesture_classes = sorted(set(labels))
    unique_users    = sorted(set(users))
    bucket: dict = {(g, u): [] for g in gesture_classes
                                for u in unique_users}
    for idx, c in per_sample_correct.items():
        bucket[(labels[idx], users[idx])].append(int(c))
    out = []
    for g in gesture_classes:
        for u in unique_users:
            vals = bucket[(g, u)]
            out.append(float(np.mean(vals)) if vals else float("nan"))
    return np.array(out, dtype=float)


# ==============================================================================
# 8.  EVALUATION FUNCTIONS
#     Each returns:
#         mean_acc, std_acc, fold_accs (length=n_folds),
#         gu_acc (length=100, per (gesture, user) pair)
# ==============================================================================

# -- DTW ----------------------------------------------------------------------

def crossval_ui_dtw(data_pca: list, labels: list, users: list,
                     folds: list,
                     knn_k: int = KNN_K
                     ) -> tuple[float, float, list, np.ndarray]:
    fold_accs = []
    correct: dict = {}
    for fold_num, (tr, te) in enumerate(folds):
        tr_items  = [data_pca[i] for i in tr]
        tr_labels = [labels[i]   for i in tr]
        te_items  = [data_pca[i] for i in te]
        te_labels = [labels[i]   for i in te]
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, tr_items, tr_labels,
                                 dtw_distance, knn_k)
            for ts in te_items
        )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        for idx, p in zip(te, preds):
            correct[idx] = int(p == labels[idx])
        u = sorted(set(users))[fold_num]
        print(f"    DTW  (UI) - User {u} held out -> acc = {acc:.3f}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, gu


def crossval_ud_dtw(data_pca: list, labels: list, users: list,
                     folds: list,
                     knn_k: int = KNN_K
                     ) -> tuple[float, float, list, np.ndarray]:
    fold_accs = []
    correct: dict = {}
    for fold_num, (tr, te, te_users) in enumerate(folds):
        tr_users_arr = [users[i] for i in tr]
        te_items  = [data_pca[i] for i in te]
        te_labels = [labels[i]   for i in te]
        preds = []
        for ts, ts_user in zip(te_items, te_users):
            same_user_mask = [j for j, u in enumerate(tr_users_arr)
                              if u == ts_user]
            tr_items_u  = [data_pca[tr[j]] for j in same_user_mask]
            tr_labels_u = [labels[tr[j]]   for j in same_user_mask]
            preds.append(
                knn_predict(ts, tr_items_u, tr_labels_u,
                            dtw_distance, knn_k)
            )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        for idx, p in zip(te, preds):
            correct[idx] = int(p == labels[idx])
        print(f"    DTW  (UD) - Fold {fold_num} -> acc = {acc:.3f}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, gu


# -- Edit Distance ------------------------------------------------------------

def crossval_ui_edit(data_pca: list, labels: list, users: list,
                      folds: list,
                      k_clusters: int = K_CLUSTERS
                      ) -> tuple[float, float, list, np.ndarray]:
    fold_accs = []
    correct: dict = {}
    for fold_num, (tr, te) in enumerate(folds):
        tr_data   = [data_pca[i] for i in tr]
        te_data   = [data_pca[i] for i in te]
        tr_labels = [labels[i]   for i in tr]
        te_labels = [labels[i]   for i in te]
        tr_seq, te_seq = fit_kmeans_and_encode(tr_data, te_data,
                                               k=k_clusters)
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, tr_seq, tr_labels,
                                 edit_distance, KNN_K)
            for ts in te_seq
        )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        for idx, p in zip(te, preds):
            correct[idx] = int(p == labels[idx])
        u = sorted(set(users))[fold_num]
        print(f"    Edit (UI) - User {u} held out -> acc = {acc:.3f}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, gu


def crossval_ud_edit(data_pca: list, labels: list, users: list,
                      folds: list,
                      k_clusters: int = K_CLUSTERS
                      ) -> tuple[float, float, list, np.ndarray]:
    fold_accs = []
    correct: dict = {}
    for fold_num, (tr, te, te_users) in enumerate(folds):
        tr_data   = [data_pca[i] for i in tr]
        te_data   = [data_pca[i] for i in te]
        tr_labels = [labels[i]   for i in tr]
        te_labels = [labels[i]   for i in te]
        tr_users_arr = [users[i] for i in tr]
        tr_seq, te_seq = fit_kmeans_and_encode(tr_data, te_data,
                                               k=k_clusters)
        preds = []
        for ts_seq_item, ts_user in zip(te_seq, te_users):
            same_user_mask = [j for j, u in enumerate(tr_users_arr)
                              if u == ts_user]
            tr_seq_u  = [tr_seq[j]  for j in same_user_mask]
            tr_labels_u = [tr_labels[j] for j in same_user_mask]
            preds.append(
                knn_predict(ts_seq_item, tr_seq_u, tr_labels_u,
                            edit_distance, KNN_K)
            )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        for idx, p in zip(te, preds):
            correct[idx] = int(p == labels[idx])
        print(f"    Edit (UD) - Fold {fold_num} -> acc = {acc:.3f}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, gu


# -- Random Forest with GridSearchCV ------------------------------------------

def _rf_fit_with_grid(X_tr: np.ndarray, y_tr: np.ndarray,
                       grid: dict = RF_GRID,
                       inner_cv: int = RF_GRID_INNER_CV,
                       random_state: int = 42
                       ) -> RandomForestClassifier:
    """
    Fit an RF with hyperparameters tuned via GridSearchCV (inner CV on
    the training fold only -- no test-set leakage).
    """
    base = RandomForestClassifier(max_features=RF_MAX_FEATURES,
                                  random_state=random_state,
                                  n_jobs=-1)
    gs = GridSearchCV(base, param_grid=grid, cv=inner_cv,
                      scoring="accuracy", n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


def crossval_ui_rf(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "",
                    use_grid_search: bool = True,
                    domain: int = 0
                    ) -> tuple[float, float, list, np.ndarray, list]:
    """
    RF user-independent CV with per-fold permutation-importance feature
    selection (Breiman 2001; Strobl et al. 2007; Guyon & Elisseeff 2003;
    Ambroise & McLachlan 2002).

    Returns: (mean_test_acc, std_test_acc, fold_test_accs,
              gu_vector, fold_train_accs).
    """
    X     = build_feature_dataset(data_pca, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    fold_test_accs, fold_train_accs = [], []
    correct: dict = {}
    feat_counter: dict = {n: 0 for n in names}
    for fold_num, (tr, te) in enumerate(folds):
        X_tr_full = X[tr]
        y_tr      = y[tr]
        kept = _select_features_per_fold(X_tr_full, y_tr, names)
        for n in kept:
            feat_counter[n] += 1
        cols = [names.index(n) for n in kept]
        X_tr = X_tr_full[:, cols]
        X_te = X[te][:, cols]
        if use_grid_search:
            clf = _rf_fit_with_grid(X_tr, y_tr)
        else:
            clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                         max_features=RF_MAX_FEATURES,
                                         random_state=42, n_jobs=-1)
            clf.fit(X_tr, y_tr)
        preds_te = clf.predict(X_te)
        preds_tr = clf.predict(X_tr)
        test_acc  = float(np.mean(preds_te == y[te]))
        train_acc = float(np.mean(preds_tr == y_tr))
        fold_test_accs.append(test_acc)
        fold_train_accs.append(train_acc)
        for idx, p in zip(te, preds_te.tolist()):
            correct[idx] = int(p == labels[idx])
        u = sorted(set(users))[fold_num]
        print(f"    RF{tag}   (UI) - User {u} -> test={test_acc:.3f} "
              f"train={train_acc:.3f} (kept {len(kept)}/{len(names)}):\n"
              f"      {kept}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    if domain:
        _save_feat_sel_summary(feat_counter, len(folds), "rf", "ui", domain)
    return (float(np.mean(fold_test_accs)), float(np.std(fold_test_accs)),
            fold_test_accs, gu, fold_train_accs)


def crossval_ud_rf(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "",
                    use_grid_search: bool = True,
                    domain: int = 0
                    ) -> tuple[float, float, list, np.ndarray, list]:
    X     = build_feature_dataset(data_pca, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    fold_test_accs, fold_train_accs = [], []
    correct: dict = {}
    feat_counter: dict = {n: 0 for n in names}
    for fold_num, (tr, te, te_users) in enumerate(folds):
        tr_users_arr = [users[i] for i in tr]
        X_all_tr = X[tr]
        y_all_tr = y[tr]
        kept = _select_features_per_fold(X_all_tr, y_all_tr, names)
        for n in kept:
            feat_counter[n] += 1
        cols = [names.index(n) for n in kept]
        preds_te_list = [None] * len(te)
        per_user_train_accs = []
        for ts_user in set(te_users):
            u_te_pos = [j for j, u in enumerate(te_users) if u == ts_user]
            same_user_mask = [j for j, u in enumerate(tr_users_arr) if u == ts_user]
            X_tr_u = X_all_tr[same_user_mask][:, cols]
            y_tr_u = y_all_tr[same_user_mask]
            X_te_u = X[[te[j] for j in u_te_pos]][:, cols]
            if len(X_tr_u) >= _INNER_CV_FOLDS and use_grid_search:
                clf_u = _rf_fit_with_grid(X_tr_u, y_tr_u)
            else:
                clf_u = RandomForestClassifier(n_estimators=RF_N_TREES,
                                               max_features=RF_MAX_FEATURES,
                                               random_state=42, n_jobs=-1)
                clf_u.fit(X_tr_u, y_tr_u)
            preds_u = clf_u.predict(X_te_u).tolist()
            per_user_train_accs.append(float(np.mean(clf_u.predict(X_tr_u) == y_tr_u)))
            for j, pred in zip(u_te_pos, preds_u):
                preds_te_list[j] = pred
        preds_te = np.array(preds_te_list)
        test_acc  = float(np.mean(preds_te == y[te]))
        train_acc = float(np.mean(per_user_train_accs))
        fold_test_accs.append(test_acc)
        fold_train_accs.append(train_acc)
        for idx, p in zip(te, preds_te.tolist()):
            correct[idx] = int(p == labels[idx])
        print(f"    RF{tag}   (UD) - Fold {fold_num} -> test={test_acc:.3f} "
              f"train={train_acc:.3f} (kept {len(kept)}/{len(names)}):\n"
              f"      {kept}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    if domain:
        _save_feat_sel_summary(feat_counter, len(folds), "rf", "ud", domain)
    return (float(np.mean(fold_test_accs)), float(np.std(fold_test_accs)),
            fold_test_accs, gu, fold_train_accs)


def analyse_rf_feature_importances(data_pca: list, labels: list,
                                     evr_list: list | None,
                                     domain: int,
                                     save_path: str | None = None,
                                     n_repeats: int = 10,
                                     cum_threshold: float = FEATURE_SELECTION_CUM_THRESHOLD
                                     ) -> list:
    """
    POST-HOC visualisation only. Fits one RF on the full dataset and
    plots permutation importance (Breiman 2001; Strobl et al. 2007).
    The actual feature selection used at evaluation time is performed
    per-fold inside the CV (see _select_features_per_fold) on the
    training set only, so there is no leakage.

    The returned list (features whose cumulative permutation importance
    covers `cum_threshold`) is provided for inspection / report writing
    only -- the crossval functions do their own selection per fold.
    """
    X = build_feature_dataset(data_pca, evr_list)
    y = np.array(labels)
    clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                 max_features=RF_MAX_FEATURES,
                                 random_state=42, n_jobs=-1)
    clf.fit(X, y)
    perm = permutation_importance(clf, X, y, n_repeats=n_repeats,
                                    random_state=42, n_jobs=-1)
    importances = np.clip(perm.importances_mean, 0.0, None)
    names = feature_names(with_evr=(evr_list is not None))

    order        = np.argsort(importances)[::-1]
    sorted_names = [names[i] for i in order]
    sorted_imps  = importances[order]

    # Cumulative-importance cut-off line for the plot.
    total = sorted_imps.sum()
    if total > 1e-12:
        cum = np.cumsum(sorted_imps) / total
        cutoff_k = int(np.searchsorted(cum, cum_threshold) + 1)
    else:
        cutoff_k = len(names)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.22 * len(names))))
    bar_colors = ["steelblue"] * len(sorted_names)
    for i in range(cutoff_k, len(sorted_names)):
        bar_colors[i] = "lightgrey"
    ax.barh(range(len(sorted_names))[::-1], sorted_imps, color=bar_colors)
    ax.set_yticks(range(len(sorted_names))[::-1])
    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.set_xlabel("Permutation importance (decrease in accuracy)")
    ax.set_title(
        f"RF permutation importance - Domain {domain}\n"
        f"(coloured = top features capturing {int(cum_threshold*100)} %"
        f" of cumulative importance, k = {cutoff_k})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    kept = sorted_names[:cutoff_k]
    print(f"  RF permutation importance (Domain {domain}, post-hoc): "
          f"{len(kept)}/{len(names)} features cover {int(cum_threshold*100)}"
          f" % of cumulative importance.")
    return kept


def _select_features_per_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                                names: list,
                                cum_threshold: float = FEATURE_SELECTION_CUM_THRESHOLD,
                                n_repeats: int = FEATURE_SELECTION_N_REPEATS,
                                random_state: int = 42,
                                corr_threshold: float = FEATURE_CORR_THRESHOLD
                                ) -> list:
    """
    Per-fold feature selection on the training set ONLY (no leakage;
    Ambroise & McLachlan 2002, PNAS).

    Three-stage pipeline justified by gesture-recognition literature:

    Stage 1 — Variance filter: remove near-constant features (var < 1e-10).
      Near-constant features add noise without information and can
      destabilise correlation estimates (Guyon & Elisseeff 2003, JMLR §3).

    Stage 2 — Correlation-based redundancy removal (Yu & Liu 2004, JMLR;
      Hall 1999, PhD thesis). For each pair with |Pearson r| > corr_threshold,
      remove the lower-variance member. Prevents correlated features from
      diluting each other's permutation importance: if feature A and
      feature B carry the same information, permuting A has little impact
      because B still provides the signal, making A appear unimportant
      even when it genuinely is (Strobl et al. 2007, BMC Bioinformatics).
      Without this step the cumulative-importance cut can discard entire
      correlated clusters, destroying classifier performance.

    Stage 3 — Permutation importance on a held-out 20% validation split
      (Breiman 2001; Strobl et al. 2007; Guyon & Elisseeff 2003). Computing
      importance on the SAME data used to fit the quick RF is biased: the
      RF memorises training noise, making memorised-noise features appear
      important. Using a held-out split gives unbiased importance estimates.
      We keep the smallest set covering cum_threshold of total importance.

    Falls back to all remaining features if any stage yields an empty set
    or the importance vector is degenerate.
    Returns the list of selected feature names preserving original order.
    """
    n_samples = X_tr.shape[0]

    # -- Stage 1: variance filter ------------------------------------------
    variances = np.var(X_tr, axis=0)
    active = np.where(variances > 1e-10)[0]
    if len(active) == 0:
        return list(names)

    # -- Stage 2: correlation-based redundancy removal ---------------------
    X_act = X_tr[:, active]
    if X_act.shape[1] > 1:
        corr = np.corrcoef(X_act.T)
        keep = np.ones(len(active), dtype=bool)
        vars_act = variances[active]
        for i in range(len(active)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(active)):
                if not keep[j]:
                    continue
                if abs(corr[i, j]) > corr_threshold:
                    if vars_act[i] >= vars_act[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break  # i removed; move to next i
        active = active[keep]

    if len(active) == 0:
        return list(names)

    # -- Stage 3: permutation importance on held-out validation split ------
    rng = np.random.RandomState(random_state)
    n_val = max(1, int(0.20 * n_samples))
    val_idx = rng.choice(n_samples, size=n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(n_samples), val_idx)

    if len(tr_idx) < 2:
        return [names[i] for i in active]

    X_sel = X_tr[:, active]
    quick_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                       random_state=random_state)
    quick_rf.fit(X_sel[tr_idx], y_tr[tr_idx])
    perm = permutation_importance(quick_rf,
                                    X_sel[val_idx], y_tr[val_idx],
                                    n_repeats=n_repeats,
                                    random_state=random_state, n_jobs=-1)
    importances = np.clip(perm.importances_mean, 0.0, None)
    total = importances.sum()

    if total < 1e-12:
        return [names[i] for i in active]

    order      = np.argsort(importances)[::-1]
    cum        = np.cumsum(importances[order]) / total
    k          = int(np.searchsorted(cum, cum_threshold) + 1)
    kept_local = sorted(order[:k].tolist())
    final_idx  = sorted(active[kept_local].tolist())
    return [names[i] for i in final_idx]


def _save_feat_sel_summary(feat_counter: dict, n_folds: int,
                            method_tag: str, setting_tag: str,
                            domain: int) -> None:
    """Save a CSV of feature selection frequency across CV folds.
    Columns: feature, n_folds_selected, pct.
    Sorted descending by n_folds_selected.
    """
    rows = sorted(feat_counter.items(), key=lambda x: -x[1])
    feat_df = pd.DataFrame(rows, columns=["feature", "n_folds_selected"])
    feat_df["pct"] = feat_df["n_folds_selected"] / n_folds
    path = os.path.join(
        DIR_TBL_FEAT_SEL,
        f"feat_sel_{method_tag}_{setting_tag}_d{domain}.csv")
    feat_df.to_csv(path, index=False)
    top5 = feat_df.head(5)["feature"].tolist()
    print(f"    Feature selection summary (top-5): {top5}  -> {path}")


# -- Decision Tree with GridSearchCV ------------------------------------------
# Used as a pedagogical baseline to empirically demonstrate the value of the
# bagging procedure in Random Forest. References:
#   Quinlan, J. R. (1986). Induction of decision trees. Mach. Learn., 1, 81-106.
#   Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification
#     and regression trees (CART). Wadsworth.
#   Hastie, Tibshirani, Friedman (2009), §9.2 (single tree) and §15 (RF).

def _dt_fit_with_grid(X_tr: np.ndarray, y_tr: np.ndarray,
                       grid: dict = DT_GRID,
                       inner_cv: int = DT_GRID_INNER_CV,
                       random_state: int = 42
                       ) -> DecisionTreeClassifier:
    """
    Fit a Decision Tree with hyperparameters tuned via GridSearchCV (inner
    CV on the training fold only -- no test-set leakage).
    """
    base = DecisionTreeClassifier(random_state=random_state)
    gs = GridSearchCV(base, param_grid=grid, cv=inner_cv,
                      scoring="accuracy", n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


def crossval_ui_dt(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "",
                    domain: int = 0
                    ) -> tuple[float, float, list, np.ndarray, list]:
    """
    Decision Tree user-independent CV with per-fold permutation-importance
    feature selection. Returns (mean, std, test_accs, gu, train_accs).
    """
    X     = build_feature_dataset(data_pca, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    fold_test_accs, fold_train_accs = [], []
    correct: dict = {}
    feat_counter: dict = {n: 0 for n in names}
    for fold_num, (tr, te) in enumerate(folds):
        X_tr_full = X[tr]
        y_tr      = y[tr]
        kept = _select_features_per_fold(X_tr_full, y_tr, names)
        for n in kept:
            feat_counter[n] += 1
        cols = [names.index(n) for n in kept]
        X_tr = X_tr_full[:, cols]
        X_te = X[te][:, cols]
        clf = _dt_fit_with_grid(X_tr, y_tr)
        preds_te = clf.predict(X_te)
        preds_tr = clf.predict(X_tr)
        test_acc  = float(np.mean(preds_te == y[te]))
        train_acc = float(np.mean(preds_tr == y_tr))
        fold_test_accs.append(test_acc)
        fold_train_accs.append(train_acc)
        for idx, p in zip(te, preds_te.tolist()):
            correct[idx] = int(p == labels[idx])
        u = sorted(set(users))[fold_num]
        print(f"    DT{tag}   (UI) - User {u} -> test={test_acc:.3f} "
              f"train={train_acc:.3f} (kept {len(kept)}/{len(names)}):\n"
              f"      {kept}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    if domain:
        _save_feat_sel_summary(feat_counter, len(folds), "dt", "ui", domain)
    return (float(np.mean(fold_test_accs)), float(np.std(fold_test_accs)),
            fold_test_accs, gu, fold_train_accs)


def crossval_ud_dt(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "",
                    domain: int = 0
                    ) -> tuple[float, float, list, np.ndarray, list]:
    X     = build_feature_dataset(data_pca, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    fold_test_accs, fold_train_accs = [], []
    correct: dict = {}
    feat_counter: dict = {n: 0 for n in names}
    for fold_num, (tr, te, te_users) in enumerate(folds):
        tr_users_arr = [users[i] for i in tr]
        X_all_tr = X[tr]
        y_all_tr = y[tr]
        kept = _select_features_per_fold(X_all_tr, y_all_tr, names)
        for n in kept:
            feat_counter[n] += 1
        cols = [names.index(n) for n in kept]
        preds_te_list = [None] * len(te)
        per_user_train_accs = []
        for ts_user in set(te_users):
            u_te_pos = [j for j, u in enumerate(te_users) if u == ts_user]
            same_user_mask = [j for j, u in enumerate(tr_users_arr) if u == ts_user]
            X_tr_u = X_all_tr[same_user_mask][:, cols]
            y_tr_u = y_all_tr[same_user_mask]
            X_te_u = X[[te[j] for j in u_te_pos]][:, cols]
            if len(X_tr_u) >= _INNER_CV_FOLDS:
                clf_u = _dt_fit_with_grid(X_tr_u, y_tr_u)
            else:
                clf_u = DecisionTreeClassifier(random_state=42)
                clf_u.fit(X_tr_u, y_tr_u)
            preds_u = clf_u.predict(X_te_u).tolist()
            per_user_train_accs.append(float(np.mean(clf_u.predict(X_tr_u) == y_tr_u)))
            for j, pred in zip(u_te_pos, preds_u):
                preds_te_list[j] = pred
        preds_te = np.array(preds_te_list)
        test_acc  = float(np.mean(preds_te == y[te]))
        train_acc = float(np.mean(per_user_train_accs))
        fold_test_accs.append(test_acc)
        fold_train_accs.append(train_acc)
        for idx, p in zip(te, preds_te.tolist()):
            correct[idx] = int(p == labels[idx])
        print(f"    DT{tag}   (UD) - Fold {fold_num} -> test={test_acc:.3f} "
              f"train={train_acc:.3f} (kept {len(kept)}/{len(names)}):\n"
              f"      {kept}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    if domain:
        _save_feat_sel_summary(feat_counter, len(folds), "dt", "ud", domain)
    return (float(np.mean(fold_test_accs)), float(np.std(fold_test_accs)),
            fold_test_accs, gu, fold_train_accs)


# -- Logistic Regression with GridSearchCV ------------------------------------
# Multinomial L2-penalised LR via lbfgs solver. Linear baseline.
# References:
#   Cox, D. R. (1958). The regression analysis of binary sequences.
#   Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied
#     Logistic Regression (3rd ed.). Wiley.
#   Hastie, Tibshirani & Friedman (2009), §4.4 (multinomial LR).
# Gesture-recognition uses: Wu & Huang (1999, §3) list LR among linear
# classifiers for gesture features; it serves as a lower-bound reference
# vs. RF (ensemble) and DT (single tree).

def _lr_fit_with_grid(X_tr: np.ndarray, y_tr: np.ndarray,
                       grid: dict = LR_GRID,
                       inner_cv: int = LR_GRID_INNER_CV,
                       random_state: int = 42
                       ) -> Pipeline:
    """
    Fit a Pipeline(StandardScaler -> multinomial L2 LogisticRegression)
    with C tuned via GridSearchCV (inner CV on training fold only).
    StandardScaler inside the Pipeline ensures lbfgs operates on
    zero-mean unit-variance inputs, which is required for reliable
    convergence (sklearn docs; Hastie et al. 2009, §18.4).
    max_iter=5000 guarantees convergence across all C values in the grid.

    Note on preprocessing conditions: because LR always applies internal
    standardisation, conditions (a) (raw data) and (b) (externally
    standardised data) are functionally equivalent for this classifier.
    This explains why LR may empirically select condition (a) as best on
    some domain/setting combinations, unlike DT and RF which have no
    internal standardisation step.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(solver="lbfgs",
                                       max_iter=5000,
                                       random_state=random_state)),
    ])
    gs = GridSearchCV(pipe, param_grid=grid, cv=inner_cv,
                      scoring="accuracy", n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


def crossval_ui_lr(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "",
                    domain: int = 0
                    ) -> tuple[float, float, list, np.ndarray, list]:
    """
    Logistic Regression user-independent CV with per-fold feature
    selection. Returns (mean, std, test_accs, gu, train_accs).
    """
    X     = build_feature_dataset(data_pca, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    fold_test_accs, fold_train_accs = [], []
    correct: dict = {}
    feat_counter: dict = {n: 0 for n in names}
    for fold_num, (tr, te) in enumerate(folds):
        X_tr_full = X[tr]
        y_tr      = y[tr]
        kept = _select_features_per_fold(X_tr_full, y_tr, names)
        for n in kept:
            feat_counter[n] += 1
        cols = [names.index(n) for n in kept]
        X_tr = X_tr_full[:, cols]
        X_te = X[te][:, cols]
        clf = _lr_fit_with_grid(X_tr, y_tr)
        preds_te = clf.predict(X_te)
        preds_tr = clf.predict(X_tr)
        test_acc  = float(np.mean(preds_te == y[te]))
        train_acc = float(np.mean(preds_tr == y_tr))
        fold_test_accs.append(test_acc)
        fold_train_accs.append(train_acc)
        for idx, p in zip(te, preds_te.tolist()):
            correct[idx] = int(p == labels[idx])
        u = sorted(set(users))[fold_num]
        print(f"    LR{tag}   (UI) - User {u} -> test={test_acc:.3f} "
              f"train={train_acc:.3f} (kept {len(kept)}/{len(names)}):\n"
              f"      {kept}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    if domain:
        _save_feat_sel_summary(feat_counter, len(folds), "lr", "ui", domain)
    return (float(np.mean(fold_test_accs)), float(np.std(fold_test_accs)),
            fold_test_accs, gu, fold_train_accs)


def crossval_ud_lr(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "",
                    domain: int = 0
                    ) -> tuple[float, float, list, np.ndarray, list]:
    X     = build_feature_dataset(data_pca, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    fold_test_accs, fold_train_accs = [], []
    correct: dict = {}
    feat_counter: dict = {n: 0 for n in names}
    for fold_num, (tr, te, te_users) in enumerate(folds):
        tr_users_arr = [users[i] for i in tr]
        X_all_tr = X[tr]
        y_all_tr = y[tr]
        kept = _select_features_per_fold(X_all_tr, y_all_tr, names)
        for n in kept:
            feat_counter[n] += 1
        cols = [names.index(n) for n in kept]
        preds_te_list = [None] * len(te)
        per_user_train_accs = []
        for ts_user in set(te_users):
            u_te_pos = [j for j, u in enumerate(te_users) if u == ts_user]
            same_user_mask = [j for j, u in enumerate(tr_users_arr) if u == ts_user]
            X_tr_u = X_all_tr[same_user_mask][:, cols]
            y_tr_u = y_all_tr[same_user_mask]
            X_te_u = X[[te[j] for j in u_te_pos]][:, cols]
            if len(X_tr_u) >= _INNER_CV_FOLDS:
                clf_u = _lr_fit_with_grid(X_tr_u, y_tr_u)
            else:
                clf_u = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr",     LogisticRegression(solver="lbfgs",
                                                  max_iter=5000,
                                                  random_state=42)),
                ])
                clf_u.fit(X_tr_u, y_tr_u)
            preds_u = clf_u.predict(X_te_u).tolist()
            per_user_train_accs.append(float(np.mean(clf_u.predict(X_tr_u) == y_tr_u)))
            for j, pred in zip(u_te_pos, preds_u):
                preds_te_list[j] = pred
        preds_te = np.array(preds_te_list)
        test_acc  = float(np.mean(preds_te == y[te]))
        train_acc = float(np.mean(per_user_train_accs))
        fold_test_accs.append(test_acc)
        fold_train_accs.append(train_acc)
        for idx, p in zip(te, preds_te.tolist()):
            correct[idx] = int(p == labels[idx])
        print(f"    LR{tag}   (UD) - Fold {fold_num} -> test={test_acc:.3f} "
              f"train={train_acc:.3f} (kept {len(kept)}/{len(names)}):\n"
              f"      {kept}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    if domain:
        _save_feat_sel_summary(feat_counter, len(folds), "lr", "ud", domain)
    return (float(np.mean(fold_test_accs)), float(np.std(fold_test_accs)),
            fold_test_accs, gu, fold_train_accs)


# -- $1 Recognizer (Kratz & Rohs, 2010) with cached templates -----------------

def _dollar_predict_one(cand_pre: np.ndarray,
                         tmpl_pre: list,
                         tmpl_lbl: list,
                         k: int = KNN_K,
                         allow_rejection: bool = False,
                         epsilon: float = DOLLAR_EPSILON) -> int:
    """1-NN prediction via dollar_recognize (includes GSS + optional heuristic).
    Returns the predicted label, or -1 if allow_rejection=True and no gesture
    passes the heuristic thresholds.
    """
    label, _score, _ranked = dollar_recognize(
        cand_pre, tmpl_pre, tmpl_lbl,
        allow_rejection=allow_rejection, epsilon=epsilon)
    return label


def crossval_ui_dollar(data: list, labels: list, users: list,
                        folds: list
                        ) -> tuple[float, float, list, np.ndarray]:
    fold_accs = []
    correct: dict = {}
    # Preprocess all gestures ONCE (cached) -- per Wobbrock et al. (2007).
    pre_all = [dollar_preprocess(seq) for seq in data]
    for fold_num, (tr, te) in enumerate(folds):
        tmpl_pre = [pre_all[i] for i in tr]
        tmpl_lbl = [labels[i]  for i in tr]
        te_pre   = [pre_all[i] for i in te]
        te_lbl   = [labels[i]  for i in te]
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_dollar_predict_one)(c, tmpl_pre, tmpl_lbl, KNN_K)
            for c in te_pre
        )
        acc = float(np.mean([p == t for p, t in zip(preds, te_lbl)]))
        fold_accs.append(acc)
        for idx, p in zip(te, preds):
            correct[idx] = int(p == labels[idx])
        u = sorted(set(users))[fold_num]
        print(f"    $1   (UI) - User {u} held out -> acc = {acc:.3f}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, gu


def crossval_ud_dollar(data: list, labels: list, users: list,
                        folds: list
                        ) -> tuple[float, float, list, np.ndarray]:
    fold_accs = []
    correct: dict = {}
    pre_all = [dollar_preprocess(seq) for seq in data]
    for fold_num, (tr, te, te_users) in enumerate(folds):
        tr_users_arr = [users[i] for i in tr]
        te_pre  = [pre_all[i]  for i in te]
        te_lbl  = [labels[i]   for i in te]
        preds = []
        for c, ts_user in zip(te_pre, te_users):
            same_user_mask = [j for j, u in enumerate(tr_users_arr)
                              if u == ts_user]
            tmpl_pre_u = [pre_all[tr[j]] for j in same_user_mask]
            tmpl_lbl_u = [labels[tr[j]]  for j in same_user_mask]
            preds.append(_dollar_predict_one(c, tmpl_pre_u, tmpl_lbl_u,
                                              KNN_K))
        acc = float(np.mean([p == t for p, t in zip(preds, te_lbl)]))
        fold_accs.append(acc)
        for idx, p in zip(te, preds):
            correct[idx] = int(p == labels[idx])
        print(f"    $1   (UD) - Fold {fold_num} -> acc = {acc:.3f}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, gu


# ==============================================================================
# 8b.  HYPERPARAMETER VALIDATION CURVES
#      Empirical iterative selection (instructions, Section 5).
# ==============================================================================

def validation_curve_kclusters(data_pca: list, labels: list, users: list,
                                 folds: list,
                                 ks: list = VC_K_CLUSTERS,
                                 domain: int = 1,
                                 save_path: str | None = None) -> int:
    """
    Plot Edit-Distance UI accuracy vs k-means K on the user-independent
    folds. Returns the K that maximises mean accuracy.
    """
    means, stds = [], []
    for k in ks:
        accs = []
        for tr, te in folds:
            tr_data  = [data_pca[i] for i in tr]
            te_data  = [data_pca[i] for i in te]
            tr_lbl   = [labels[i]   for i in tr]
            te_lbl   = [labels[i]   for i in te]
            tr_seq, te_seq = fit_kmeans_and_encode(tr_data, te_data, k=k)
            preds = Parallel(n_jobs=-1, prefer="threads")(
                delayed(knn_predict)(ts, tr_seq, tr_lbl,
                                     edit_distance, KNN_K)
                for ts in te_seq
            )
            accs.append(float(np.mean(
                [p == t for p, t in zip(preds, te_lbl)])))
        means.append(float(np.mean(accs)))
        stds.append(float(np.std(accs)))
        print(f"  K={k:>3}: acc = {means[-1]:.3f} +/- {stds[-1]:.3f}")
    best_k = ks[int(np.argmax(means))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3,
                color="steelblue")
    ax.set_xlabel("k-means K (codebook size)")
    ax.set_ylabel("Edit Distance UI accuracy")
    ax.set_title(
        f"Validation curve - K_CLUSTERS - Domain {domain} "
        f"(best K = {best_k})")
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return best_k


def validation_curve_knn(data_pca: list, labels: list, users: list,
                          folds: list,
                          ks: list = VC_KNN_K,
                          method: str = "dtw",
                          domain: int = 1,
                          save_path: str | None = None) -> int:
    """
    Plot kNN UI accuracy vs K with the DTW or Edit distance, on UI folds.
    Returns the K that maximises mean accuracy.
    """
    if method == "dtw":
        dist_fn = dtw_distance
        prep    = lambda tr_d, te_d: (tr_d, te_d)
    else:
        dist_fn = edit_distance
        prep    = lambda tr_d, te_d: fit_kmeans_and_encode(
            tr_d, te_d, k=K_CLUSTERS)

    means, stds = [], []
    for k in ks:
        accs = []
        for tr, te in folds:
            tr_d = [data_pca[i] for i in tr]
            te_d = [data_pca[i] for i in te]
            tr_l = [labels[i]   for i in tr]
            te_l = [labels[i]   for i in te]
            tr_in, te_in = prep(tr_d, te_d)
            preds = Parallel(n_jobs=-1, prefer="threads")(
                delayed(knn_predict)(ts, tr_in, tr_l, dist_fn, k)
                for ts in te_in
            )
            accs.append(float(np.mean(
                [p == t for p, t in zip(preds, te_l)])))
        means.append(float(np.mean(accs)))
        stds.append(float(np.std(accs)))
        print(f"  k={k}: acc = {means[-1]:.3f} +/- {stds[-1]:.3f}")
    best_k = ks[int(np.argmax(means))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3,
                color="darkorange")
    ax.set_xlabel("kNN K")
    ax.set_ylabel(f"{method.upper()} UI accuracy")
    ax.set_title(
        f"Validation curve - kNN K - {method.upper()} - "
        f"Domain {domain} (best k = {best_k})")
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return best_k


# ==============================================================================
# 9.  ABLATION STUDY  -- 4 methods x 3 preprocessing conditions
# ==============================================================================

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
# 10.  STATISTICAL TESTS
#      Paired Wilcoxon signed-rank on n=100 (gesture, user) accuracy pairs.
#      Benjamini-Hochberg FDR correction (BH) only — see decision #3.
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
                           domain: int) -> pd.DataFrame:
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
    plt.show()

    return df_raw


# ==============================================================================
# 11.  CONFUSION MATRICES
# ==============================================================================

def _safe_filename(title: str) -> str:
    for ch in [" ", "|", "(", ")", "-", "/", "+", "->"]:
        title = title.replace(ch, "_")
    return title


def _plot_cm(y_true: list, y_pred: list,
              display_labels: list, title: str) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(DIR_FIG_CM,
                            f"confusion_{_safe_filename(title)}.png")
    plt.savefig(out_path, dpi=150)
    plt.show()


def compute_cm_edit(data_denoised: list, labels: list, users: list,
                     folds: list,
                     title: str = "Confusion matrix - Edit Distance") -> None:
    y_true, y_pred = [], []
    for fold in folds:
        tr, te = fold[0], fold[1]
        tr_data   = [data_denoised[i] for i in tr]
        te_data   = [data_denoised[i] for i in te]
        tr_labels = [labels[i]        for i in tr]
        te_labels = [labels[i]        for i in te]
        tr_seq, te_seq = fit_kmeans_and_encode(tr_data, te_data,
                                               k=K_CLUSTERS)
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, tr_seq, tr_labels,
                                 edit_distance, KNN_K)
            for ts in te_seq
        )
        y_true.extend(te_labels)
        y_pred.extend(preds)
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def compute_cm_dtw(data_denoised: list, labels: list, users: list,
                    folds: list,
                    title: str = "Confusion matrix - DTW") -> None:
    y_true, y_pred = [], []
    for fold in folds:
        tr, te = fold[0], fold[1]
        tr_items  = [data_denoised[i] for i in tr]
        tr_labels = [labels[i]        for i in tr]
        te_items  = [data_denoised[i] for i in te]
        te_labels = [labels[i]        for i in te]
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, tr_items, tr_labels,
                                 dtw_distance, KNN_K)
            for ts in te_items
        )
        y_true.extend(te_labels)
        y_pred.extend(preds)
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def compute_cm_rf(data_denoised: list, labels: list, users: list,
                   folds: list,
                   evr_list: list | None = None,
                   title: str = "Confusion matrix - RF") -> None:
    X     = build_feature_dataset(data_denoised, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    y_true, y_pred = [], []
    for fold in folds:
        tr, te = fold[0], fold[1]
        X_tr_full = X[tr]
        y_tr      = y[tr]
        kept = _select_features_per_fold(X_tr_full, y_tr, names)
        cols = [names.index(n) for n in kept]
        X_tr = X_tr_full[:, cols]
        X_te = X[te][:, cols]
        clf = _rf_fit_with_grid(X_tr, y_tr)
        y_true.extend(y[te].tolist())
        y_pred.extend(clf.predict(X_te).tolist())
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def compute_cm_dt(data_denoised: list, labels: list, users: list,
                   folds: list,
                   evr_list: list | None = None,
                   title: str = "Confusion matrix - DT") -> None:
    X     = build_feature_dataset(data_denoised, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    y_true, y_pred = [], []
    for fold in folds:
        tr, te = fold[0], fold[1]
        X_tr_full = X[tr]
        y_tr      = y[tr]
        kept = _select_features_per_fold(X_tr_full, y_tr, names)
        cols = [names.index(n) for n in kept]
        X_tr = X_tr_full[:, cols]
        X_te = X[te][:, cols]
        clf = _dt_fit_with_grid(X_tr, y_tr)
        y_true.extend(y[te].tolist())
        y_pred.extend(clf.predict(X_te).tolist())
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def compute_cm_lr(data_denoised: list, labels: list, users: list,
                   folds: list,
                   evr_list: list | None = None,
                   title: str = "Confusion matrix - LR") -> None:
    X     = build_feature_dataset(data_denoised, evr_list)
    y     = np.array(labels)
    names = feature_names(with_evr=(evr_list is not None))
    y_true, y_pred = [], []
    for fold in folds:
        tr, te = fold[0], fold[1]
        X_tr_full = X[tr]
        y_tr      = y[tr]
        kept = _select_features_per_fold(X_tr_full, y_tr, names)
        cols = [names.index(n) for n in kept]
        X_tr = X_tr_full[:, cols]
        X_te = X[te][:, cols]
        clf = _lr_fit_with_grid(X_tr, y_tr)
        y_true.extend(y[te].tolist())
        y_pred.extend(clf.predict(X_te).tolist())
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def compute_cm_dollar(data: list, labels: list, users: list,
                       folds: list,
                       title: str = "Confusion matrix - $1") -> None:
    pre_all = [dollar_preprocess(seq) for seq in data]
    y_true, y_pred = [], []
    for fold in folds:
        tr, te = fold[0], fold[1]
        tmpl_pre = [pre_all[i] for i in tr]
        tmpl_lbl = [labels[i]  for i in tr]
        te_pre   = [pre_all[i] for i in te]
        te_lbl   = [labels[i]  for i in te]
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_dollar_predict_one)(c, tmpl_pre, tmpl_lbl, KNN_K)
            for c in te_pre
        )
        y_true.extend(te_lbl)
        y_pred.extend(preds)
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def draw_best_model_cm(best_name: str,
                        data_best: list, evr_best: list | None,
                        labels: list, users: list,
                        folds_ui: list,
                        domain: int) -> None:
    tag = f"User-independent Domain {domain}"
    if best_name == "Edit Distance":
        compute_cm_edit(data_best, labels, users, folds_ui,
                        title=f"Edit Distance {tag}")
    elif best_name == "DTW":
        compute_cm_dtw(data_best, labels, users, folds_ui,
                       title=f"DTW {tag}")
    elif best_name == "DT":
        compute_cm_dt(data_best, labels, users, folds_ui, evr_best,
                       title=f"DT {tag}")
    elif best_name == "RF":
        compute_cm_rf(data_best, labels, users, folds_ui, evr_best,
                      title=f"RF {tag}")
    elif best_name == "LR":
        compute_cm_lr(data_best, labels, users, folds_ui, evr_best,
                       title=f"LR {tag}")
    elif best_name == "$1":
        compute_cm_dollar(data_best, labels, users, folds_ui,
                          title=f"$1 {tag}")


# ==============================================================================
# 11b.  OVERFITTING DIAGNOSTIC  -- train vs test, learning curves
# References:
#   Hastie, Tibshirani & Friedman (2009), "The Elements of Statistical
#     Learning", §7.10 (cross-validation learning curves).
#   Domingos, P. (2012), "A Few Useful Things to Know About ML",
#     Communications of the ACM, 55 (10), 78-87.
# ==============================================================================

def plot_learning_curve_method(estimator_builder, data_pca: list,
                                 labels: list, evr_list: list | None,
                                 method_name: str, domain: int,
                                 setting: str,
                                 save_path: str | None = None) -> None:
    """
    sklearn.model_selection.learning_curve over 5 training sizes
    [0.2, 0.4, 0.6, 0.8, 1.0] of the full dataset, with 5-fold CV.
    Plots train accuracy vs validation accuracy as a function of
    training-set size. A wide vertical gap = high variance (overfitting);
    both low = high bias.

    `estimator_builder()` returns a fresh classifier instance.
    """
    X = build_feature_dataset(data_pca, evr_list)
    y = np.array(labels)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator_builder(),
        X, y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        shuffle=True,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes_abs, train_mean, "o-", color="steelblue",
            label="Train score")
    ax.fill_between(train_sizes_abs,
                     train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="steelblue")
    ax.plot(train_sizes_abs, val_mean, "o-", color="darkorange",
            label="Validation score")
    ax.fill_between(train_sizes_abs,
                     val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="darkorange")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"Learning curve - {method_name} - Domain {domain} ({setting})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def _lc_builder_dt():
    return DecisionTreeClassifier(random_state=42)

def _lc_builder_rf():
    return RandomForestClassifier(n_estimators=RF_N_TREES,
                                   max_features=RF_MAX_FEATURES,
                                   random_state=42, n_jobs=-1)

def _lc_builder_lr():
    return Pipeline([("scaler", StandardScaler()),
                     ("lr",     LogisticRegression(solver="lbfgs",
                                                    max_iter=5000,
                                                    random_state=42))])


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
        flag = "  <-- higher gap" if r["Gap"] > 0.05 else ""
        print(f"   D{int(r['Domain'])}-{r['Setting']:<2} {r['Method']:<3} "
              f"train={r['TrainAcc']:.3f}  test={r['TestAcc']:.3f}  "
              f"gap={r['Gap']:+.3f}{flag}")
    print(f"  Saved -> {csv_path}")
    return df


# ==============================================================================
# 12.  RESULT SAVING
# ==============================================================================

def save_fold_results(fold_accs: list, method: str,
                       setting: str, domain: int) -> None:
    fname = os.path.join(
        DIR_TBL_FOLDS,
        f"results_domain{domain}_{setting}_{method}.csv")
    pd.DataFrame({"accuracy": fold_accs}).to_csv(fname, index=False)
    print(f"  Saved -> {fname}")


# ==============================================================================
# 13.  MAIN
# ==============================================================================

if __name__ == "__main__":

    # -- 0. Reproducibility & numba warm-up -------------------------------
    np.random.seed(42)

    print("Warming up numba JIT ...", end=" ", flush=True)
    _d = np.random.randn(10, 3)
    dtw_distance(_d, _d)
    _s = np.zeros(10, dtype=np.int64)
    edit_distance(_s, _s)
    _dp = np.random.randn(DOLLAR_N, 3).astype(np.float64)
    _dollar_gss_mse(_dp, _dp)   # compile _rotate_1axis_nb + _mse_nb + _dollar_gss_mse
    print("done.")

    # -- 1. Load data ------------------------------------------------------
    print("\n=== Loading Domain 1 ===")
    data1, labels1, users1 = load_domain1(DOMAIN1_DIR)
    max_len1 = print_dataset_info(data1, labels1, users1, "Domain 1")

    print("\n=== Loading Domain 4 ===")
    data4, labels4, users4 = load_domain4(DOMAIN4_DIR)
    max_len4 = print_dataset_info(data4, labels4, users4, "Domain 4")

    # -- 2. Exploratory visualisation -------------------------------------
    print("\n=== Exploratory Visualisation ===")
    plot_sequence_lengths(data1, labels1, "Domain 1",
                          save_path=os.path.join(DIR_FIG_EXPLORE,
                                                 "d1_sequence_lengths.png"))
    plot_gesture_samples(data1, labels1, users1, "Domain 1",
                         save_path=os.path.join(DIR_FIG_EXPLORE,
                                                "d1_gesture_samples.png"))
    plot_sequence_lengths(data4, labels4, "Domain 4",
                          save_path=os.path.join(DIR_FIG_EXPLORE,
                                                 "d4_sequence_lengths.png"))
    plot_gesture_samples(data4, labels4, users4, "Domain 4",
                         save_path=os.path.join(DIR_FIG_EXPLORE,
                                                "d4_gesture_samples.png"))

    # -- 3. Standardisation -----------------------------------------------
    print("\n=== Standardisation ===")
    data1_std = standardize_gestures(data1)
    data4_std = standardize_gestures(data4)
    print("  Both domains standardised (per-gesture, per-axis).")

    # -- 4. PCA denoising -------------------------------------------------
    print("\n=== Per-gesture PCA denoising analysis ===")
    summarise_pca_denoising(data1_std, "Domain 1",
                             save_path=os.path.join(DIR_FIG_PCA,
                                                    "d1_pca_denoise.png"))
    summarise_pca_denoising(data4_std, "Domain 4",
                             save_path=os.path.join(DIR_FIG_PCA,
                                                    "d4_pca_denoise.png"))

    print("\n=== Applying PCA denoising (3D -> 2D -> 3D) ===")
    data1_denoised, evr1 = apply_pca_denoising(data1_std, n_keep=PCA_N_KEEP)
    data4_denoised, evr4 = apply_pca_denoising(data4_std, n_keep=PCA_N_KEEP)
    print("  PCA denoising applied to both domains.")

    # -- 5. Fold indices --------------------------------------------------
    print("\n=== Generating fold indices (shared across all methods) ===")
    folds_ui_1 = _ui_fold_indices(users1)
    folds_ud_1 = _ud_fold_indices(labels1, users1)
    folds_ui_4 = _ui_fold_indices(users4)
    folds_ud_4 = _ud_fold_indices(labels4, users4)
    print(f"  Domain 1 - UI: {len(folds_ui_1)} | UD: {len(folds_ud_1)}")
    print(f"  Domain 4 - UI: {len(folds_ui_4)} | UD: {len(folds_ud_4)}")

    # -- 6. Validation curves for K (empirical iterative selection) -------
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
    print("  Domain 1 - K_CLUSTERS scan on standardised data (conditions b/c):")
    best_k_clusters_d1 = validation_curve_kclusters(
        data1_std, labels1, users1, folds_ui_1,
        domain=1,
        save_path=os.path.join(DIR_FIG_VC, "d1_vc_kclusters_std.png"))
    print("  Domain 1 - K_CLUSTERS scan on raw data (condition a only):")
    best_k_raw_d1 = validation_curve_kclusters(
        data1, labels1, users1, folds_ui_1,
        domain=1,
        save_path=os.path.join(DIR_FIG_VC, "d1_vc_kclusters_raw.png"))
    print("  Domain 1 - kNN K scan (DTW UI) [informative only, k=1 forced]:")
    _vc_best_knn_d1 = validation_curve_knn(
        data1_std, labels1, users1, folds_ui_1,
        method="dtw", domain=1,
        save_path=os.path.join(DIR_FIG_VC, "d1_vc_knn.png"))
    print(f"  Domain 1 selected: K_CLUSTERS(std)={best_k_clusters_d1}, "
          f"K_CLUSTERS(raw)={best_k_raw_d1}, "
          f"KNN_K=1 (forced; curve optimum was k={_vc_best_knn_d1})")

    print("\n  Domain 4 - K_CLUSTERS scan on standardised data (conditions b/c):")
    best_k_clusters_d4 = validation_curve_kclusters(
        data4_std, labels4, users4, folds_ui_4,
        domain=4,
        save_path=os.path.join(DIR_FIG_VC, "d4_vc_kclusters_std.png"))
    print("  Domain 4 - K_CLUSTERS scan on raw data (condition a only):")
    best_k_raw_d4 = validation_curve_kclusters(
        data4, labels4, users4, folds_ui_4,
        domain=4,
        save_path=os.path.join(DIR_FIG_VC, "d4_vc_kclusters_raw.png"))
    print("  Domain 4 - kNN K scan (DTW UI) [informative only, k=1 forced]:")
    _vc_best_knn_d4 = validation_curve_knn(
        data4_std, labels4, users4, folds_ui_4,
        method="dtw", domain=4,
        save_path=os.path.join(DIR_FIG_VC, "d4_vc_knn.png"))
    print(f"  Domain 4 selected: K_CLUSTERS(std)={best_k_clusters_d4}, "
          f"K_CLUSTERS(raw)={best_k_raw_d4}, "
          f"KNN_K=1 (forced; curve optimum was k={_vc_best_knn_d4})")

    # -- 7. Ablation study --------------------------------------------------
    # K_CLUSTERS = best from validation curve (per domain).
    # KNN_K = 1 forced (decision #8).
    # Ablation now run in BOTH UI and UD: the best preprocessing for UI
    # is not guaranteed to be the best for UD (decision #13).
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

    # -- 7b. Force $1 to (a) raw -------------------------------------------
    # Wobbrock et al. (2007) and Kratz & Rohs (2010) prescribe that $1
    # operates on raw input points (the algorithm contains its own
    # internal preprocessing: resample, centroid translation, indicative-
    # axis rotation, uniform cube scaling).  Feeding externally-standardised
    # data into $1 alters the geometry used by the cross-product rotation
    # step and is methodologically inconsistent.  We therefore override
    # the ablation auto-selection for $1 and force condition (a) -- the
    # ablation table is kept for transparency.
    def _force_dollar_raw(best_prep: dict, data_raw_dom: list) -> None:
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

    _force_dollar_raw(best_prep_d1_ui, data1)
    _force_dollar_raw(best_prep_d1_ud, data1)
    _force_dollar_raw(best_prep_d4_ui, data4)
    _force_dollar_raw(best_prep_d4_ud, data4)

    # -- 7c. Save best-preproc UI-vs-UD comparison -------------------------
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

    _save_preproc_comparison(best_prep_d1_ui, best_prep_d1_ud, domain=1)
    _save_preproc_comparison(best_prep_d4_ui, best_prep_d4_ud, domain=4)

    # -- 8. RF permutation-importance plot (post-hoc visualisation only) ---
    # Actual feature selection is per-fold INSIDE the crossval; this plot
    # is informative.
    def _data_for(best_entry: dict,
                  raw: list, std: list, denoised: list,
                  evr: list) -> tuple[list, list | None]:
        cond = best_entry["condition"]
        if cond == "(a) No preprocessing":
            return raw, None
        if cond == "(b) Standardisation":
            return std, None
        return denoised, evr

    # -- 9-10. Read out UI + UD results from the ablation dicts ------------
    METHODS_ORDER  = ["Edit Distance", "DTW", "DT", "RF", "LR", "$1"]
    METHOD_KEY     = {"Edit Distance": "edit", "DTW": "dtw", "DT": "dt",
                       "RF": "rf", "LR": "lr", "$1": "dollar"}

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

    main_d1_ui = _print_and_save_phase(1, "UI", best_prep_d1_ui)
    main_d1_ud = _print_and_save_phase(1, "UD", best_prep_d1_ud)
    main_d4_ui = _print_and_save_phase(4, "UI", best_prep_d4_ui)
    main_d4_ud = _print_and_save_phase(4, "UD", best_prep_d4_ud)

    # -- 11+15. Statistical tests (UI only, per consigne §5) ---------------
    print("\n=== Statistical Tests - Domain 1 ===")
    generate_pvalue_table(
        {m: main_d1_ui[m]["gu"] for m in METHODS_ORDER},
        domain=1)

    print("\n=== Statistical Tests - Domain 4 ===")
    generate_pvalue_table(
        {m: main_d4_ui[m]["gu"] for m in METHODS_ORDER},
        domain=4)

    # -- 12+16. Confusion matrix - best model per domain (UI) --------------
    all_ui_d1 = {m: main_d1_ui[m]["mean"] for m in METHODS_ORDER}
    best_name_d1  = max(all_ui_d1, key=all_ui_d1.get)
    best_entry_d1 = main_d1_ui[best_name_d1]
    print(f"\n  Best model - Domain 1 (UI): {best_name_d1} "
          f"({all_ui_d1[best_name_d1]:.3f})  "
          f"[{best_entry_d1['condition']}]")
    draw_best_model_cm(best_name_d1,
                       best_entry_d1["data"],
                       best_entry_d1["evr"],
                       labels1, users1, folds_ui_1, domain=1)

    all_ui_d4 = {m: main_d4_ui[m]["mean"] for m in METHODS_ORDER}
    best_name_d4  = max(all_ui_d4, key=all_ui_d4.get)
    best_entry_d4 = main_d4_ui[best_name_d4]
    print(f"\n  Best model - Domain 4 (UI): {best_name_d4} "
          f"({all_ui_d4[best_name_d4]:.3f})  "
          f"[{best_entry_d4['condition']}]")
    draw_best_model_cm(best_name_d4,
                       best_entry_d4["data"],
                       best_entry_d4["evr"],
                       labels4, users4, folds_ui_4, domain=4)

    # -- 17. Overfitting diagnostic ---------------------------------------
    # Train-vs-test gap (parametric classifiers only; 1-NN methods have
    # train acc = 1.0 by construction and are excluded).  References:
    # Hastie et al. (2009) §7.10; Domingos (2012).
    print("\n=== Overfitting analysis: train-vs-test gap ===")
    overfit_rows = []
    for domain, mains in [(1, [("UI", main_d1_ui), ("UD", main_d1_ud)]),
                            (4, [("UI", main_d4_ui), ("UD", main_d4_ud)])]:
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

    # -- 17b. Learning curves (parametric classifiers, UI setting) ---------
    # Plots train vs validation accuracy as a function of training-set
    # size.  Wide gap = high variance.  Reference: Hastie et al. (2009)
    # §7.10.  We use the BEST preprocessing for each method on the UI
    # ablation (consistent with how the model would be deployed).
    print("\n=== Learning curves (UI, parametric classifiers) ===")
    LC_BUILDERS = {"DT": _lc_builder_dt, "RF": _lc_builder_rf,
                   "LR": _lc_builder_lr}
    for domain, main_dict in [(1, main_d1_ui), (4, main_d4_ui)]:
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
                    f"lc_{m.lower()}_d{domain}_ui.png"))

    # -- 18. Final summary table ------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - Mean accuracy +/- std")
    print("UI = user-independent | UD = user-dependent")
    print("=" * 70)

    summary_rows = []
    for domain, mains in [(1, {"UI": main_d1_ui, "UD": main_d1_ud}),
                            (4, {"UI": main_d4_ui, "UD": main_d4_ud})]:
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
    for domain_id, bp_ui, bp_ud in [(1, best_prep_d1_ui, best_prep_d1_ud),
                                       (4, best_prep_d4_ui, best_prep_d4_ud)]:
        print(f"\n  Domain {domain_id}:")
        for method in METHODS_ORDER:
            print(f"    {method:<16}: UI = {bp_ui[method]['condition']:<22}"
                  f"  UD = {bp_ud[method]['condition']}")

    summary_path = os.path.join(DIR_TBL_SUMMARY, "summary_results.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Saved -> {summary_path}")
    print("\nDone.")
