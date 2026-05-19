import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 3D projection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.utils.parallel import Parallel, delayed

from src.config import K_CLUSTERS, KNN_K, DIR_FIG_CM
from src.features import build_feature_dataset, feature_names
from src.preprocessing import fit_kmeans_and_encode
from src.models.baselines import dtw_distance, edit_distance, knn_predict
from src.models.dollar import dollar_preprocess, _dollar_predict_one
from src.models.parametric import (
    _rf_fit_with_grid, _dt_fit_with_grid, _lr_fit_with_grid,
    _select_features_per_fold,
)
from src.utils import _safe_filename


# ==============================================================================
# 2.  EXPLORATORY VISUALISATION
# ==============================================================================

def plot_sequence_lengths(data: list, labels: list,
                           domain_name: str,
                           save_path: str | None = None,
                           show: bool = False) -> None:
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
    if show:
        plt.show()


def plot_gesture_samples(data: list, labels: list, users: list,
                          domain_name: str, n_classes: int = 4,
                          n_subjects: int = 3,
                          save_path: str | None = None,
                          show: bool = False) -> None:
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
    if show:
        plt.show()


# ==============================================================================
# 11.  CONFUSION MATRICES
# ==============================================================================

def _plot_cm(y_true: list, y_pred: list,
              display_labels: list, title: str,
              show: bool = False) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(DIR_FIG_CM,
                            f"confusion_{_safe_filename(title)}.png")
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()


def compute_cm_edit(data_denoised: list, labels: list, users: list,
                     folds: list,
                     title: str = "Confusion matrix - Edit Distance",
                     show: bool = False) -> None:
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
    _plot_cm(y_true, y_pred, sorted(set(labels)), title, show=show)


def compute_cm_dtw(data_denoised: list, labels: list, users: list,
                    folds: list,
                    title: str = "Confusion matrix - DTW",
                    show: bool = False) -> None:
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
    _plot_cm(y_true, y_pred, sorted(set(labels)), title, show=show)


def compute_cm_rf(data_denoised: list, labels: list, users: list,
                   folds: list,
                   evr_list: list | None = None,
                   title: str = "Confusion matrix - RF",
                   show: bool = False) -> None:
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
    _plot_cm(y_true, y_pred, sorted(set(labels)), title, show=show)


def compute_cm_dt(data_denoised: list, labels: list, users: list,
                   folds: list,
                   evr_list: list | None = None,
                   title: str = "Confusion matrix - DT",
                   show: bool = False) -> None:
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
    _plot_cm(y_true, y_pred, sorted(set(labels)), title, show=show)


def compute_cm_lr(data_denoised: list, labels: list, users: list,
                   folds: list,
                   evr_list: list | None = None,
                   title: str = "Confusion matrix - LR",
                   show: bool = False) -> None:
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
    _plot_cm(y_true, y_pred, sorted(set(labels)), title, show=show)


def compute_cm_dollar(data: list, labels: list, users: list,
                       folds: list,
                       title: str = "Confusion matrix - $1",
                       show: bool = False) -> None:
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
    _plot_cm(y_true, y_pred, sorted(set(labels)), title, show=show)


def draw_best_model_cm(best_name: str,
                        data_best: list, evr_best: list | None,
                        labels: list, users: list,
                        folds_ui: list,
                        domain: int,
                        show: bool = False) -> None:
    tag = f"User-independent Domain {domain}"
    if best_name == "Edit Distance":
        compute_cm_edit(data_best, labels, users, folds_ui,
                        title=f"Edit Distance {tag}", show=show)
    elif best_name == "DTW":
        compute_cm_dtw(data_best, labels, users, folds_ui,
                       title=f"DTW {tag}", show=show)
    elif best_name == "DT":
        compute_cm_dt(data_best, labels, users, folds_ui, evr_best,
                       title=f"DT {tag}", show=show)
    elif best_name == "RF":
        compute_cm_rf(data_best, labels, users, folds_ui, evr_best,
                      title=f"RF {tag}", show=show)
    elif best_name == "LR":
        compute_cm_lr(data_best, labels, users, folds_ui, evr_best,
                       title=f"LR {tag}", show=show)
    elif best_name == "$1":
        compute_cm_dollar(data_best, labels, users, folds_ui,
                          title=f"$1 {tag}", show=show)


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
                                 save_path: str | None = None,
                                 show: bool = False) -> None:
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
    if show:
        plt.show()
