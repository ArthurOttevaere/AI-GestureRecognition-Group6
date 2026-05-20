import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.config import (
    RF_N_TREES, RF_MAX_FEATURES, RF_GRID, RF_GRID_INNER_CV,
    DT_GRID, DT_GRID_INNER_CV, LR_GRID, LR_GRID_INNER_CV,
    FEATURE_SELECTION_CUM_THRESHOLD, FEATURE_SELECTION_N_REPEATS,
    FEATURE_CORR_THRESHOLD, DIR_TBL_FEAT_SEL,
)
from src.features import build_feature_dataset, feature_names


# ==============================================================================
# 9.  RANDOM FOREST WITH GRIDSEARCHCV
# ==============================================================================

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


# ==============================================================================
# 10.  FEATURE SELECTION (per fold, no leakage)
# ==============================================================================

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


# ==============================================================================
# Learning curve estimator builders
# ==============================================================================

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
