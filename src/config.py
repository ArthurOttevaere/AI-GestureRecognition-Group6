import os
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Two dirname calls because config.py lives in src/, one level below project root.
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
FEATURE_CORR_THRESHOLD          = 0.99 # Yu & Liu (2004) use 0.95, but this is relaxed to 0.99 to preserve more features for the RF permutation importance step,
                                       # which is sensitive to feature count (Strobl et al. 2007).

# Validation-curve scan ranges (Section 5 of instructions)
VC_K_CLUSTERS = [5, 10, 15, 20, 25, 30, 40]
VC_KNN_K      = [1, 3, 5, 7, 9]

# $1 Recognizer hyperparameters (Kratz & Rohs, 2010)
DOLLAR_N         = 150           # number of resampled points
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
DIR_DOC                  = os.path.join(OUTPUTS_DIR, "logs")

for _d in [DIR_FIG_EXPLORE, DIR_FIG_PCA, DIR_FIG_VC, DIR_FIG_FI,
           DIR_FIG_CM, DIR_FIG_STATS, DIR_FIG_LEARNING_CURVES,
           DIR_TBL_ABLATION, DIR_TBL_FOLDS, DIR_TBL_STATS, DIR_TBL_SUMMARY,
           DIR_TBL_OVERFITTING, DIR_TBL_FEAT_SEL, DIR_DOC]:
    os.makedirs(_d, exist_ok=True)
