"""
10. CONFUSION MATRICES
=======================
Confusion matrix computation for all methods, and dispatcher for best model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

from config import DATA_DIR
from classifiers import knn_predict
from features import build_feature_dataset
from lstm_model import prepare_lstm_data, _build_lstm_model
from distance_metrics import dtw_distance, edit_distance


def _save_cm_plot(title: str) -> None:
    """Helper: tighten layout and save confusion-matrix figure."""
    plt.tight_layout()
    safe = (title.replace(" ", "_").replace("|", "")
                 .replace("(", "").replace(")", "")
                 .replace("–", "").replace("/", "_"))
    plt.savefig(os.path.join(DATA_DIR, f"confusion_{safe}.png"), dpi=150)
    plt.show()


def compute_confusion_matrix(
        items: list, labels: list, users: list,
        distance_fn, k: int = 1,
        title: str = "Confusion matrix"
) -> None:
    """
    Leave-one-user-out CV → collect all predictions → confusion matrix.
    Works for both Edit Distance and DTW.
    """
    unique_users = sorted(set(users))
    y_true, y_pred = [], []

    for u in unique_users:
        train_idx = [i for i, usr in enumerate(users) if usr != u]
        test_idx  = [i for i, usr in enumerate(users) if usr == u]

        train_items  = [items[i]  for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_items   = [items[i]  for i in test_idx]
        test_labels  = [labels[i] for i in test_idx]

        fold_preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, train_items, train_labels, distance_fn, k)
            for ts in test_items
        )
        y_true.extend(test_labels)
        y_pred.extend(fold_preds)

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(labels)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    _save_cm_plot(title)


def compute_confusion_matrix_rf(
        data: list, labels: list, users: list,
        title: str = "Confusion matrix – RF"
) -> None:
    """Confusion matrix for Random Forest (user-independent)."""
    X = build_feature_dataset(data)
    y = np.array(labels)
    unique_users = sorted(set(users))
    y_true, y_pred = [], []

    for u in unique_users:
        train_idx = [i for i, usr in enumerate(users) if usr != u]
        test_idx  = [i for i, usr in enumerate(users) if usr == u]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])

        y_true.extend(y[test_idx].tolist())
        y_pred.extend(preds.tolist())

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(labels)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    _save_cm_plot(title)


def compute_confusion_matrix_lstm(
        data: list, labels: list, users: list,
        title: str = "Confusion matrix – LSTM"
) -> None:
    """
    Confusion matrix for LSTM (user-independent leave-one-user-out).
     – dedicated function so LSTM CM can be drawn when LSTM is best.
    """
    min_label   = int(np.min(labels))
    labels_zi   = np.array(labels) - min_label
    num_classes = len(np.unique(labels_zi))

    X, _ = prepare_lstm_data(data, labels_zi)
    unique_users = sorted(set(users))
    y_true, y_pred = [], []

    for u in unique_users:
        train_idx = [i for i, usr in enumerate(users) if usr != u]
        test_idx  = [i for i, usr in enumerate(users) if usr == u]

        y_train = labels_zi[np.array(train_idx)]
        y_test  = labels_zi[np.array(test_idx)]

        model = _build_lstm_model(X.shape[1], num_classes)
        model.fit(X[train_idx], y_train,
                  epochs=30, batch_size=32,
                  validation_split=0.1, verbose=0)

        raw_preds = np.argmax(model.predict(X[test_idx], verbose=0), axis=1)

        # Convert back to original label space for the CM display
        y_true.extend((y_test  + min_label).tolist())
        y_pred.extend((raw_preds + min_label).tolist())

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(labels)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    _save_cm_plot(title)


def draw_best_model_cm(
        best_name: str,
        data_std: list, sequences: list,
        labels: list, users: list,
        domain: int
) -> None:
    """
    Draw the confusion matrix for whichever method achieved the highest
    mean accuracy in the user-independent setting.

    Parameters
    ----------
    best_name  : one of 'Edit Distance', 'DTW', 'RF', 'LSTM'
    data_std   : standardised 3D sequences (used by DTW, RF, LSTM)
    sequences  : cluster-label sequences  (used by Edit Distance)
    labels     : list of int
    users      : list of int
    domain     : 1 or 4  (used in the plot title)
    """
    tag = f"User-independent (Domain {domain})"

    if best_name == "Edit Distance":
        compute_confusion_matrix(
            sequences, labels, users, edit_distance, k=1,
            title=f"Edit Distance – {tag}")

    elif best_name == "DTW":
        compute_confusion_matrix(
            data_std, labels, users, dtw_distance, k=1,
            title=f"DTW – {tag}")

    elif best_name == "RF":
        compute_confusion_matrix_rf(
            data_std, labels, users,
            title=f"RF – {tag}")

    elif best_name == "LSTM":
        compute_confusion_matrix_lstm(
            data_std, labels, users,
            title=f"LSTM – {tag}")

    else:
        print(f"  [WARNING] Unknown method '{best_name}' – skipping CM.")
