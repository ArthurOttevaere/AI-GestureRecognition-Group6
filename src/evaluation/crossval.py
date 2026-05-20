import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.parallel import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    KNN_K, K_CLUSTERS, _INNER_CV_FOLDS,
    RF_N_TREES, RF_MAX_FEATURES,
    VC_K_CLUSTERS, VC_KNN_K,
)
from src.features import build_feature_dataset, feature_names
from src.preprocessing import fit_kmeans_and_encode
from src.models.baselines import dtw_distance, edit_distance, knn_predict
from src.models.dollar import dollar_preprocess, _dollar_predict_one
from src.models.parametric import (
    _rf_fit_with_grid, _dt_fit_with_grid, _lr_fit_with_grid,
    _select_features_per_fold, _save_feat_sel_summary,
)


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


# -- Decision Tree with GridSearchCV ------------------------------------------

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
                                 save_path: str | None = None,
                                 show: bool = False) -> int:
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
    if show:
        plt.show()
    return best_k


def validation_curve_knn(data_pca: list, labels: list, users: list,
                          folds: list,
                          ks: list = VC_KNN_K,
                          method: str = "dtw",
                          domain: int = 1,
                          save_path: str | None = None,
                          show: bool = False) -> int:
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
    if show:
        plt.show()
    return best_k
