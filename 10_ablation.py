"""
10. ABLATION STUDY
===================
Compare all four methods under three preprocessing conditions using
user-independent leave-one-user-out CV.

Conditions
----------
(a) No preprocessing
    DTW and LSTM operate on raw 3D coordinates.
    k-means fitted on raw 3D points -> Edit Distance sequences.
    RF features extracted from raw sequences WITHOUT PCA EVR.

(b) Standardisation only
    Per-gesture zero-mean / unit-std normalisation applied.
    k-means on standardised 3D points -> Edit Distance sequences.
    RF features from standardised sequences WITHOUT PCA EVR.
    DTW and LSTM on standardised sequences.

(c) Standardisation + per-gesture PCA EVR features
    Identical to (b) for DTW, Edit Distance, and LSTM: these methods
    do not use the PCA EVR features.  Only the RF feature vector
    changes, with 3 additional EVR values capturing planarity and
    the 3D-ness of each individual gesture.
"""

import numpy as np
import pandas as pd

from config import K_CLUSTERS, DATA_DIR
from preprocessing import cluster_and_encode
from distance_metrics import dtw_distance, edit_distance
from crossvalidation import crossval_user_independent
from random_forest import random_forest_evaluation
from lstm_model import lstm_evaluation


def run_ablation_study(data_raw: list, data_std: list,
                        labels: list, users: list,
                        domain: int) -> pd.DataFrame:
    """
    Ablation study: 4 methods x 3 preprocessing conditions,
    user-independent leave-one-user-out CV.

    Parameters
    ----------
    data_raw  : list of np.ndarray (T, 3) - raw (no preprocessing)
    data_std  : list of np.ndarray (T, 3) - standardised
    labels    : list of int
    users     : list of int
    domain    : 1 or 4

    Returns
    -------
    DataFrame with columns: Preprocessing, Method, Mean, Std, Note, Result.
    """
    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY | Domain {domain} | User-independent")
    print(f"{'='*65}")

    rows = []

    def _record(cond, method, mean, std, note=""):
        rows.append({"Preprocessing": cond, "Method": method,
                     "Mean": mean, "Std": std, "Note": note})
        note_str = f"  [{note}]" if note else ""
        print(f"    [{cond}] {method}: {mean:.3f} +/- {std:.3f}{note_str}")

    # ------------------------------------------------------------ (a)
    print("\n  (a) No preprocessing")
    seq_raw, _ = cluster_and_encode(data_raw, k=K_CLUSTERS)

    m, s, _ = crossval_user_independent(
        seq_raw, labels, users, edit_distance, k=1)
    _record("(a) No preprocessing", "Edit Distance", m, s)

    m, s, _ = crossval_user_independent(
        data_raw, labels, users, dtw_distance, k=1)
    _record("(a) No preprocessing", "DTW", m, s)

    m, s, _ = random_forest_evaluation(
        data_raw, labels, users, include_pca_evr=False)
    _record("(a) No preprocessing", "RF", m, s)

    m, s, _ = lstm_evaluation(data_raw, labels, users)
    _record("(a) No preprocessing", "LSTM", m, s)

    # ------------------------------------------------------------ (b)
    print("\n  (b) Standardisation only")
    seq_std_abl, _ = cluster_and_encode(data_std, k=K_CLUSTERS)

    m_ed_b, s_ed_b, _ = crossval_user_independent(
        seq_std_abl, labels, users, edit_distance, k=1)
    _record("(b) Standardisation", "Edit Distance", m_ed_b, s_ed_b)

    m_dtw_b, s_dtw_b, _ = crossval_user_independent(
        data_std, labels, users, dtw_distance, k=1)
    _record("(b) Standardisation", "DTW", m_dtw_b, s_dtw_b)

    m_rf_b, s_rf_b, _ = random_forest_evaluation(
        data_std, labels, users, include_pca_evr=False)
    _record("(b) Standardisation", "RF", m_rf_b, s_rf_b)

    m_lstm_b, s_lstm_b, _ = lstm_evaluation(data_std, labels, users)
    _record("(b) Standardisation", "LSTM", m_lstm_b, s_lstm_b)

    # ------------------------------------------------------------ (c)
    print("\n  (c) Standardisation + per-gesture PCA EVR features")

    # DTW, Edit Distance and LSTM: unchanged from (b)
    _record("(c) Std + PCA EVR", "Edit Distance", m_ed_b, s_ed_b,
            note="identical to (b) - EVR not used by this method")
    _record("(c) Std + PCA EVR", "DTW", m_dtw_b, s_dtw_b,
            note="identical to (b) - EVR not used by this method")

    # RF: now WITH per-gesture PCA EVR features
    m_rf_c, s_rf_c, _ = random_forest_evaluation(
        data_std, labels, users, include_pca_evr=True)
    _record("(c) Std + PCA EVR", "RF", m_rf_c, s_rf_c,
            note="3 PCA EVR values added to RF feature vector")

    _record("(c) Std + PCA EVR", "LSTM", m_lstm_b, s_lstm_b,
            note="identical to (b) - EVR not used by this method")

    # Summary table
    df = pd.DataFrame(rows)
    df["Result"] = (df["Mean"].map("{:.3f}".format)
                    + " +/- " + df["Std"].map("{:.3f}".format))
    pivot = df.pivot_table(index="Preprocessing", columns="Method",
                           values="Result", aggfunc="first")
    print(f"\n  Ablation summary - Domain {domain}:")
    print(pivot.to_string())

    import os
    out_path = os.path.join(DATA_DIR, f"ablation_domain{domain}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    return df
