"""
7. RANDOM FOREST
=================
User-independent and user-dependent evaluation using Random Forest.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config import RF_N_TREES
from features import build_feature_dataset


def random_forest_evaluation(
        data: list, labels: list, users: list,
        include_pca_evr: bool = True,
        tag: str = ""
) -> tuple[float, float, list]:
    """
    Evaluate Random Forest using leave-one-user-out CV
    (user-independent setting).

    Parameters
    ----------
    include_pca_evr : whether to include per-gesture PCA EVR features
    tag             : short string appended to progress prints
    """
    X = build_feature_dataset(data, include_pca_evr)
    y = np.array(labels)

    unique_users = sorted(set(users))
    accs = []

    for u in unique_users:
        train_idx = [i for i, usr in enumerate(users) if usr != u]
        test_idx  = [i for i, usr in enumerate(users) if usr == u]

        clf = RandomForestClassifier(n_estimators=RF_N_TREES, random_state=42)
        clf.fit(X[train_idx], y[train_idx])

        acc = float(np.mean(clf.predict(X[test_idx]) == y[test_idx]))
        accs.append(acc)
        print(f"    RF{tag} - User {u} -> accuracy = {acc:.3f}")

    return float(np.mean(accs)), float(np.std(accs)), accs


def random_forest_evaluation_user_dependent(
        data: list, labels: list, users: list,
        include_pca_evr: bool = True,
        tag: str = ""
) -> tuple[float, float, list]:
    """
    Evaluate Random Forest using leave-one-sample-out CV
    (user-dependent setting).

    For each user and each fold, one repetition per gesture class is held out.
    Training uses only the other repetitions of the same user.

    Parameters
    ----------
    include_pca_evr : whether to include per-gesture PCA EVR features
    tag             : short string appended to progress prints
    """
    X = build_feature_dataset(data, include_pca_evr)
    y = np.array(labels)

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
            test_idx  = [gesture_to_indices[g][fold] for g in gesture_classes]
            train_idx = [
                gesture_to_indices[g][rep]
                for g in gesture_classes
                for rep in range(n_folds) if rep != fold
            ]

            clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                         random_state=42)
            clf.fit(X[train_idx], y[train_idx])

            acc = float(np.mean(clf.predict(X[test_idx]) == y[test_idx]))
            fold_accs.append(acc)

        print(f"    RF{tag} (dep.) - User {u} -> mean acc = "
              f"{np.mean(fold_accs[-n_folds:]):.3f}")

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs
