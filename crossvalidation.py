"""
9. CROSS-VALIDATION (baseline methods)
========================================
Leave-one-user-out and leave-one-sample-out CV for distance-based classifiers.
"""

import numpy as np
from joblib import Parallel, delayed

from classifiers import knn_predict


def crossval_user_independent(
        items: list, labels: list, users: list,
        distance_fn, k: int = 1
) -> tuple[float, float, list]:
    """
    Leave-one-user-out CV (user-independent setting).

    10 folds: at each fold one user is entirely held out as the test set;
    the model trains on the remaining 9 users.

    Returns
    -------
    mean_acc, std_acc, fold_accs  (10 values)
    """
    unique_users = sorted(set(users))
    fold_accs = []

    for u in unique_users:
        train_idx = [i for i, usr in enumerate(users) if usr != u]
        test_idx  = [i for i, usr in enumerate(users) if usr == u]

        train_items  = [items[i]  for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_items   = [items[i]  for i in test_idx]
        test_labels  = [labels[i] for i in test_idx]

        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, train_items, train_labels, distance_fn, k)
            for ts in test_items
        )

        acc = float(np.mean([p == t for p, t in zip(preds, test_labels)]))
        fold_accs.append(acc)
        print(f"    User {u:2d} held out → accuracy = {acc:.3f}")

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def crossval_user_dependent(
        items: list, labels: list, users: list,
        distance_fn, k: int = 1
) -> tuple[float, float, list]:
    """
    Leave-one-sample-out CV (user-dependent setting).

    For each user and fold f, the f-th repetition of every gesture class
    is held out; the model trains on the remaining repetitions of that user.

    Returns
    -------
    mean_acc, std_acc, fold_accs  (100 values: 10 users × 10 folds)
    """
    unique_users    = sorted(set(users))
    gesture_classes = sorted(set(labels))
    fold_accs = []

    for u in unique_users:
        gesture_to_indices: dict[int, list] = {g: [] for g in gesture_classes}
        for i, (lbl, usr) in enumerate(zip(labels, users)):
            if usr == u:
                gesture_to_indices[lbl].append(i)

        n_folds = min(len(v) for v in gesture_to_indices.values())

        for fold in range(n_folds):
            test_idx  = [gesture_to_indices[g][fold]  for g in gesture_classes]
            train_idx = [
                gesture_to_indices[g][rep]
                for g in gesture_classes
                for rep in range(n_folds) if rep != fold
            ]

            train_items  = [items[i]  for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_items   = [items[i]  for i in test_idx]
            test_labels  = [labels[i] for i in test_idx]

            preds = [knn_predict(ts, train_items, train_labels, distance_fn, k)
                     for ts in test_items]

            acc = float(np.mean([p == t for p, t in zip(preds, test_labels)]))
            fold_accs.append(acc)

        print(f"    User {u:2d} – mean acc over {n_folds} folds = "
              f"{np.mean(fold_accs[-n_folds:]):.3f}")

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs
