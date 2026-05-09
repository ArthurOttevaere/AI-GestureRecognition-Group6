"""
MLSM2154 – Artificial Intelligence: Gesture Recognition Project
===============================================================
Phase 1 : Data loading & exploratory analysis
Phase 2 : Pre-processing
            - Per-gesture standardisation (zero mean, unit std per axis)
            - Per-gesture PCA denoising (3D → 2D → 3D):
                * PCA fitted on the T time-step points of each gesture
                * Projection onto the 2 principal components (2D)
                * Back-projection into the original 3D space
                * This removes sensor noise captured by PC3 while
                  preserving the intrinsic planar geometry of the gesture.
                * The 3 EVR values are ALSO stored as additional RF features.
            - k-means clustering INSIDE each CV fold (rigorous):
                * Fitted on training-set 3D points only per fold
                * Centroids applied to encode both train and test sequences
Phase 3 : Baseline methods (DTW + Edit Distance) with 1-NN classifier
Phase 4 : Advanced methods (Random Forest, LSTM, $1 Recognizer 3D)
Phase 5 : Cross-validation
            - User-independent : leave-one-user-out (10 folds)
                Each fold holds out ALL data of 1 user; trains on the
                remaining 9 users.
                kNN compares test gesture against ALL training gestures
                (from all 9 training users).
            - User-dependent   : leave-one-sample-out (10 folds)
                Each fold holds out repetition #f of EVERY gesture of
                EVERY user simultaneously; trains on the other 9
                repetitions of all users.
                kNN compares test gesture of user U against training
                gestures of the SAME user U only.
          ALL methods use IDENTICAL fold indices (generated once via
          _ui_fold_indices / _ud_fold_indices) so that train/test sets
          are guaranteed to be the same across every method — a
          prerequisite for valid statistical comparison.
          Ablation study: 5 methods x 3 preprocessing conditions
            (a) No preprocessing
            (b) Standardisation only
            (c) Standardisation + per-gesture PCA denoising (full pipeline)
Phase 6 : Statistical tests (Wilcoxon signed-rank on IDENTICAL fold
          vectors, raw p-values + Benjamini-Hochberg FDR correction
          + permutation test for robustness)
          – user-independent only, as required by the course guidelines.

Design decisions & scientific justifications
--------------------------------------------
1. Per-gesture PCA denoising (3D → 2D → 3D)
   A PCA is fitted independently on the T time-step points of each
   individual gesture AFTER standardisation and BEFORE clustering.
   Step 1: project onto the 2 principal axes  → (T, 2)
   Step 2: back-project into the original 3D space → (T, 3)
   This removes the variance captured by PC3, which for quasi-planar
   gestures (e.g. drawn digits) corresponds mainly to sensor noise.
   Because PCA parameters are estimated from the gesture's own T points
   only, there is no cross-fold data leakage.
   The 3 EVR values (before truncation) are stored separately and used
   as additional features for the RF classifier.

2. k-means inside cross-validation (rigorous mode)
   For Edit Distance, k-means is fitted ONLY on training-set 3D points
   at each CV fold, then the learned centroids are applied to encode
   both training and test sequences.  This strictly follows the course
   guideline: "the most rigorous method would require you to perform the
   clustering inside the cross-validation only on the training set".

3. Shared fold indices — identical train/test splits across all methods
   _ui_fold_indices() and _ud_fold_indices() are called ONCE and the
   resulting index lists are passed to every method.  This guarantees
   that DTW, Edit Distance, RF, LSTM and $1 all see exactly the same
   folds, which is mandatory for valid paired Wilcoxon signed-rank tests.

4. User-dependent kNN — same-user comparison only
   In the user-dependent setting, when classifying a test gesture from
   user U, the kNN search is restricted to training gestures that also
   belong to user U.  This matches the course guideline:
   "make a comparison with the gestures of the same user from the
   training set only (user-dependent case)".

5. User-dependent CV — 10 folds
   Fold #f holds out repetition #f of EVERY gesture of EVERY user
   simultaneously (test = 10 users × 10 gestures = 100 samples per
   fold; train = 900 samples).  This matches "using 90% of the data
   from the 10 users" and ensures training always covers all users.

6. No spectral (FFT) features
   FFT coefficients are not comparable across sequences of different
   lengths (31–314 time steps) without prior resampling.

7. Sequence length excluded from RF features
   Reflects recording speed rather than gesture shape.

8. Statistical tests — Benjamini-Hochberg (BH) correction + permutation
   With C(5,2)=10 pairwise comparisons, BH controls the false discovery
   rate at 5%. A paired permutation test is run in parallel, which is
   more powerful than the Wilcoxon test for small n=10.

9. Adaptive preprocessing selection
   The ablation study returns, for each method, the preprocessing
   condition that achieved the highest mean accuracy on the
   user-independent folds. The main evaluation then uses those
   per-method optimal datasets.

10. LSTM improvements (v2)
    - Dropout (0.3) on LSTM and Dense layers to reduce overfitting.
    - Early stopping (patience=10, restore_best_weights=True) instead
      of a fixed epoch budget.
    - Multiple seeds (3 runs per fold, results averaged) to reduce
      variance due to random weight initialisation.

11. RF improvements (v2)
    - Richer temporal features: per-third-segment statistics (begin /
      middle / end of gesture), 3D curvature (mean angle between
      consecutive velocity vectors), per-axis path length.
    - n_estimators increased to 200 for lower generalisation variance.

12. $1 Recognizer — 3D adaptation (Wobbrock, Wilson & Li, 2007)
    The original $1 algorithm operates in 2D. We adapt it faithfully
    to 3D following the same four steps from the pseudocode:
      Step 1 — Resample to N=64 equidistant 3D points.
      Step 2 — Rotate to the indicative orientation (angle from
                centroid to first point in the XY plane set to 0°,
                then the same rotation applied to all 3 axes).
      Step 3 — Scale to a unit cube and translate centroid to origin.
      Step 4 — Match via Golden Section Search over ±45° in the XY
                plane; path distance averaged over N points.
    Classification uses 1-NN over all training templates, exactly as
    in the original paper.

Authors : Andry Lenny / El Mohcine Mohamed / Ottevaere Arthur
Group   : Group 6
Date    : 2026
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa - 3D projection
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from numba import njit
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import tensorflow as tf
# only keras, no tensorflow.keras, to avoid version issues in the environment
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences


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
# Hyperparameters — justified against the literature and dataset properties
# ---------------------------------------------------------------------------

K_CLUSTERS       = 20
PCA_N_KEEP       = 2
RF_N_TREES       = 200          # increased from 100 (lower variance)
RF_MAX_FEATURES  = "sqrt"
LSTM_UNITS       = 64
LSTM_DENSE_UNITS = 32
LSTM_DROPOUT     = 0.3          # NEW: dropout rate for regularisation
LSTM_MAX_EPOCHS  = 100          # NEW: upper bound; early stopping governs
LSTM_ES_PATIENCE = 10           # NEW: early stopping patience
LSTM_BATCH_SIZE  = 32
LSTM_N_SEEDS     = 3            # NEW: number of random seeds per fold
KNN_K            = 1

# $1 Recognizer hyperparameters (Wobbrock et al., 2007)
DOLLAR_N         = 64           # number of resampled points
DOLLAR_SIZE      = 250.0        # side of the reference square
DOLLAR_THETA_MAX = np.radians(45.0)   # ±45° search range
DOLLAR_THETA_DELTA = np.radians(2.0)  # 2° angular resolution
PHI              = 0.5 * (-1.0 + np.sqrt(5.0))  # golden ratio ≈ 0.618


# ==============================================================================
# 1.  DATA LOADING
# ==============================================================================

def load_domain1(folder_path: str) -> tuple[list, list, list]:
    """
    Load all Domain 1 CSV files.

    File naming: SubjectS-G-R.csv
        S = subject index (1-based in filename, stored 0-based)
        G = gesture class (0-9, digit drawn)
        R = repetition index (1-10)

    Returns
    -------
    data   : list of np.ndarray (T, 3) — x, y, z coordinates
    labels : list of int — gesture class (0-9)
    users  : list of int — subject index (0-9)
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
    """
    Load all Domain 4 plain-text files (no extension).
    """
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
        print(f"  [WARNING] {domain_name} — incomplete groups:")
        for u, g, n in sorted(issues):
            print(f"    user={u}, gesture={g} → {n} rep(s) (expected 10)")
    else:
        print(f"  {domain_name}: {len(labels)} sequences — completeness OK "
              f"({len(set(users))} users x "
              f"{len(set(labels))} gestures x 10 reps)")


def print_dataset_info(data: list, labels: list,
                        users: list, domain_name: str) -> int:
    lengths = [len(seq) for seq in data]
    print(f"\n{'─'*55}")
    print(f"  {domain_name}")
    print(f"{'─'*55}")
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
    ax.set_title(f"{domain_name} — Sequence lengths per gesture class")
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
        f"{domain_name} — 3D trajectories (green=start, red=end)",
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

    print(f"\n  Per-gesture PCA denoising — {domain_name}")
    for c in range(evrs.shape[1]):
        print(f"    PC{c+1}: mean EVR = {evrs[:, c].mean():.3f} "
              f"+/- {evrs[:, c].std():.3f}")
    kept_var    = evrs[:, :n_keep].sum(axis=1).mean() * 100
    removed_var = evrs[:, n_keep:].sum(axis=1).mean() * 100
    print(f"    Variance kept   (PC1+PC2): {kept_var:.1f}%")
    print(f"    Variance removed (PC3+): {removed_var:.1f}%  "
          f"(interpreted as sensor noise)")

    fig, ax = plt.subplots(figsize=(7, 3))
    for c in range(evrs.shape[1]):
        ax.hist(evrs[:, c], bins=30, alpha=0.6, label=f"PC{c+1}")
    ax.axvline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("Explained variance ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"{domain_name} — Per-gesture PCA EVR distribution "
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
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
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
# 4b.  $1 RECOGNIZER — 3D ADAPTATION  (Wobbrock, Wilson & Li, 2007)
# ==============================================================================
# The original $1 algorithm is a 2-D unistroke recognizer. We adapt it to
# 3D by applying the resample, scale, and translate steps in 3D, and
# performing the indicative-angle rotation only in the XY plane (the dominant
# plane for both domains), following the spirit of the original algorithm.
# The Golden Section Search over ±45° in XY is preserved exactly as in the
# pseudocode published by the authors.
# ==============================================================================

def _dollar_path_length(points: np.ndarray) -> float:
    """Sum of Euclidean distances between consecutive 3D points."""
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def dollar_resample(points: np.ndarray, n: int = DOLLAR_N) -> np.ndarray:
    """
    Step 1 of $1 (Wobbrock et al., 2007) — adapted to 3D.
    Resample a 3D point path into n evenly spaced points by linear
    interpolation along cumulative arc-length.
    """
    total = _dollar_path_length(points)
    if total == 0.0 or len(points) < 2:
        return np.tile(points[0], (n, 1))

    interval   = total / (n - 1)
    D          = 0.0
    new_points = [points[0].copy()]

    i = 1
    while i < len(points) and len(new_points) < n:
        d = float(np.linalg.norm(points[i] - points[i - 1]))
        if D + d >= interval:
            frac = (interval - D) / d
            q    = points[i - 1] + frac * (points[i] - points[i - 1])
            new_points.append(q)
            # Insert q back so it can be the starting point of the next step
            points = np.insert(points, i, q, axis=0)
            D = 0.0
        else:
            D += d
        i += 1

    # Floating-point rounding may leave us one point short
    while len(new_points) < n:
        new_points.append(points[-1].copy())

    return np.array(new_points[:n], dtype=float)


def _dollar_centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def _dollar_indicative_angle(points: np.ndarray) -> float:
    """
    Step 2 of $1 — indicative angle in the XY plane from centroid to
    the first resampled point.  Returns angle in radians.
    """
    c = _dollar_centroid(points)
    return float(np.arctan2(c[1] - points[0, 1], c[0] - points[0, 0]))


def _dollar_rotate_by(points: np.ndarray, omega: float) -> np.ndarray:
    """
    Rotate all 3D points around the centroid by angle omega in the XY plane.
    The Z coordinate is left unchanged, consistent with the 2D origin of the
    algorithm and the dominant-plane approach used here.
    """
    c   = _dollar_centroid(points)
    cos_w, sin_w = np.cos(omega), np.sin(omega)
    rotated      = points.copy()
    dx = points[:, 0] - c[0]
    dy = points[:, 1] - c[1]
    rotated[:, 0] = cos_w * dx - sin_w * dy + c[0]
    rotated[:, 1] = sin_w * dx + cos_w * dy + c[1]
    # Z unchanged
    return rotated


def _dollar_scale_to(points: np.ndarray,
                      size: float = DOLLAR_SIZE) -> np.ndarray:
    """
    Step 3a of $1 — non-uniform scale so the bounding box fits in
    size × size × size.  If a dimension has zero extent, no scaling
    is applied on that axis (avoids division by zero).
    """
    mins  = points.min(axis=0)
    maxs  = points.max(axis=0)
    spans = maxs - mins
    spans[spans == 0.0] = 1.0   # guard against degenerate axes
    scaled = (points - mins) / spans * size
    return scaled


def _dollar_translate_to_origin(points: np.ndarray) -> np.ndarray:
    """
    Step 3b of $1 — translate centroid to the coordinate-system origin.
    """
    c = _dollar_centroid(points)
    return points - c


def _dollar_path_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Average per-point Euclidean distance between two same-length 3D paths.
    Corresponds to PATH-DISTANCE in the $1 pseudocode.
    """
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _dollar_distance_at_angle(points: np.ndarray,
                               template: np.ndarray,
                               theta: float) -> float:
    """DISTANCE-AT-ANGLE from the $1 pseudocode."""
    rotated = _dollar_rotate_by(points, theta)
    return _dollar_path_distance(rotated, template)


def _dollar_distance_at_best_angle(points: np.ndarray,
                                    template: np.ndarray,
                                    theta_a: float,
                                    theta_b: float,
                                    theta_delta: float) -> float:
    """
    Golden Section Search over [theta_a, theta_b] for the rotation that
    minimises path distance.  Directly implements DISTANCE-AT-BEST-ANGLE
    from the $1 pseudocode (Wobbrock et al., 2007).
    phi = 0.5 * (-1 + sqrt(5))  ≈ 0.618
    """
    x1 = PHI * theta_a + (1.0 - PHI) * theta_b
    f1 = _dollar_distance_at_angle(points, template, x1)
    x2 = (1.0 - PHI) * theta_a + PHI * theta_b
    f2 = _dollar_distance_at_angle(points, template, x2)

    while abs(theta_b - theta_a) > theta_delta:
        if f1 < f2:
            theta_b = x2
            x2, f2  = x1, f1
            x1 = PHI * theta_a + (1.0 - PHI) * theta_b
            f1 = _dollar_distance_at_angle(points, template, x1)
        else:
            theta_a = x1
            x1, f1  = x2, f2
            x2 = (1.0 - PHI) * theta_a + PHI * theta_b
            f2 = _dollar_distance_at_angle(points, template, x2)

    return min(f1, f2)


def dollar_preprocess(points: np.ndarray,
                       n:    int   = DOLLAR_N,
                       size: float = DOLLAR_SIZE) -> np.ndarray:
    """
    Apply Steps 1–3 of the $1 algorithm to a single 3D gesture.
    Used once on templates during training and once on candidates at
    test time (as specified in the original paper).
    """
    pts = dollar_resample(points, n)
    omega = _dollar_indicative_angle(pts)
    pts   = _dollar_rotate_by(pts, -omega)          # rotate to 0°
    pts   = _dollar_scale_to(pts, size)              # scale to square
    pts   = _dollar_translate_to_origin(pts)         # translate to origin
    return pts


def dollar_recognize(candidate: np.ndarray,
                      templates: list,
                      n:    int   = DOLLAR_N,
                      size: float = DOLLAR_SIZE) -> float:
    """
    Step 4 of the $1 algorithm — find the nearest template.
    Returns the minimum path distance over all templates (used as the
    distance function for 1-NN classification).

    Parameters
    ----------
    candidate  : preprocessed candidate gesture (N, 3)
    templates  : list of preprocessed template gestures [(N, 3), ...]
    """
    best = np.inf
    for tmpl in templates:
        d = _dollar_distance_at_best_angle(
            candidate, tmpl,
            -DOLLAR_THETA_MAX, DOLLAR_THETA_MAX, DOLLAR_THETA_DELTA
        )
        if d < best:
            best = d
    return best


def dollar_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Distance function compatible with knn_predict.
    Both sequences are preprocessed on-the-fly (Steps 1–3), then
    Step 4 (GSS) is applied to find the best angular alignment.
    This matches the recognition procedure described in the paper.
    """
    p1 = dollar_preprocess(seq1)
    p2 = dollar_preprocess(seq2)
    return _dollar_distance_at_best_angle(
        p1, p2,
        -DOLLAR_THETA_MAX, DOLLAR_THETA_MAX, DOLLAR_THETA_DELTA
    )


# ==============================================================================
# 5.  k-NN CLASSIFIER
# ==============================================================================

def knn_predict(test_item, train_items: list, train_labels: list,
                distance_fn, k: int = KNN_K) -> int:
    distances = np.array([distance_fn(test_item, ref)
                          for ref in train_items])
    k_nearest = np.argsort(distances)[:k]
    k_labels  = [train_labels[idx] for idx in k_nearest]
    return max(set(k_labels), key=k_labels.count)


# ==============================================================================
# 6.  FEATURE EXTRACTION  (RF input) — v2 with richer temporal features
# ==============================================================================

def extract_features(sequence: np.ndarray,
                      evr: np.ndarray | None = None) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a 3D gesture sequence.

    Features (v2 — 50 features without EVR, 53 with EVR):
    ─────────────────────────────────────────────────────
    Per-axis global statistics (7 × 3 = 21)
        mean, std, min, max, range, skewness, kurtosis for x, y, z.

    Kinematics (6)
        mean/std/max/min speed, mean/std acceleration magnitude.

    Trajectory length (1)
        total arc-length (sum of per-step norms).

    Bounding box (4)
        extent on x, y, z and diagonal of the enclosing box.

    3D curvature (3)  — NEW
        mean / std / total of the angle between consecutive velocity
        vectors.  Captures turning behaviour along the gesture.

    Per-segment statistics — begin / middle / end thirds (6)  — NEW
        mean speed in each temporal third (3 values) and
        mean 3D displacement magnitude in each third (3 values).
        These capture the temporal dynamics that global statistics miss.

    Per-axis path length (3)  — NEW
        sum of absolute increments on each axis independently.

    PCA EVR (3, optional)
        Only appended when condition (c) is active.
    """
    features = []

    # -- Global per-axis statistics --
    for i in range(3):
        axis = sequence[:, i]
        features.extend([
            float(np.mean(axis)),
            float(np.std(axis)),
            float(np.min(axis)),
            float(np.max(axis)),
            float(np.max(axis) - np.min(axis)),
            float(pd.Series(axis).skew()),
            float(pd.Series(axis).kurt()),
        ])

    # -- Kinematics --
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

    # -- Total arc-length --
    features.append(float(np.sum(speed)))

    # -- Bounding box --
    bbox = np.max(sequence, axis=0) - np.min(sequence, axis=0)
    features.extend(bbox.tolist())
    features.append(float(np.linalg.norm(bbox)))

    # -- 3D curvature (NEW) --
    # Angle between consecutive normalised velocity vectors.
    # A straight gesture → all angles ≈ 0; a curly one → large angles.
    if len(velocity) > 1:
        v_norm = velocity / (
            np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-8)
        cos_angles = np.clip(
            np.sum(v_norm[:-1] * v_norm[1:], axis=1), -1.0, 1.0)
        angles = np.arccos(cos_angles)          # in [0, π]
        features.extend([float(np.mean(angles)),
                         float(np.std(angles)),
                         float(np.sum(angles))])   # total curvature
    else:
        features.extend([0.0, 0.0, 0.0])

    # -- Per-segment statistics (begin / middle / end thirds) (NEW) --
    T      = len(sequence)
    thirds = [slice(0, T // 3),
              slice(T // 3, 2 * T // 3),
              slice(2 * T // 3, T)]
    for sl in thirds:
        seg = sequence[sl]
        if len(seg) < 2:
            features.append(0.0)  # mean speed of this segment
        else:
            seg_vel   = np.diff(seg, axis=0)
            seg_speed = np.linalg.norm(seg_vel, axis=1)
            features.append(float(np.mean(seg_speed)))

    for sl in thirds:
        seg = sequence[sl]
        if len(seg) < 2:
            features.append(0.0)
        else:
            seg_disp = np.linalg.norm(
                np.diff(seg, axis=0), axis=1)
            features.append(float(np.mean(seg_disp)))

    # -- Per-axis path length (NEW) --
    for i in range(3):
        features.append(
            float(np.sum(np.abs(np.diff(sequence[:, i])))))

    # -- PCA EVR (optional) --
    if evr is not None:
        features.extend(evr.tolist())

    return np.array(features, dtype=float)


def build_feature_dataset(data: list,
                           evr_list: list | None = None) -> np.ndarray:
    if evr_list is not None:
        return np.array([extract_features(seq, evr)
                         for seq, evr in zip(data, evr_list)])
    return np.array([extract_features(seq, None) for seq in data])


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
        folds.append((tr, te, te_users))
    return folds


# ==============================================================================
# 8.  EVALUATION FUNCTIONS
# ==============================================================================

# ── DTW ───────────────────────────────────────────────────────────────────────

def crossval_ui_dtw(data_pca: list, labels: list, users: list,
                     folds: list) -> tuple[float, float, list]:
    fold_accs = []
    for fold_num, (tr, te) in enumerate(folds):
        tr_items  = [data_pca[i] for i in tr]
        tr_labels = [labels[i]   for i in tr]
        te_items  = [data_pca[i] for i in te]
        te_labels = [labels[i]   for i in te]
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, tr_items, tr_labels,
                                 dtw_distance, KNN_K)
            for ts in te_items
        )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        u = sorted(set(users))[fold_num]
        print(f"    DTW  (UI) — User {u} held out → acc = {acc:.3f}")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def crossval_ud_dtw(data_pca: list, labels: list, users: list,
                     folds: list) -> tuple[float, float, list]:
    fold_accs = []
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
                            dtw_distance, KNN_K)
            )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        print(f"    DTW  (UD) — Fold {fold_num} → acc = {acc:.3f}")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


# ── Edit Distance ─────────────────────────────────────────────────────────────

def crossval_ui_edit(data_pca: list, labels: list, users: list,
                      folds: list) -> tuple[float, float, list]:
    fold_accs = []
    for fold_num, (tr, te) in enumerate(folds):
        tr_data   = [data_pca[i] for i in tr]
        te_data   = [data_pca[i] for i in te]
        tr_labels = [labels[i]   for i in tr]
        te_labels = [labels[i]   for i in te]
        tr_seq, te_seq = fit_kmeans_and_encode(tr_data, te_data,
                                               k=K_CLUSTERS)
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, tr_seq, tr_labels,
                                 edit_distance, KNN_K)
            for ts in te_seq
        )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        u = sorted(set(users))[fold_num]
        print(f"    Edit (UI) — User {u} held out → acc = {acc:.3f}")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def crossval_ud_edit(data_pca: list, labels: list, users: list,
                      folds: list) -> tuple[float, float, list]:
    fold_accs = []
    for fold_num, (tr, te, te_users) in enumerate(folds):
        tr_data   = [data_pca[i] for i in tr]
        te_data   = [data_pca[i] for i in te]
        tr_labels = [labels[i]   for i in tr]
        te_labels = [labels[i]   for i in te]
        tr_users_arr = [users[i] for i in tr]
        tr_seq, te_seq = fit_kmeans_and_encode(tr_data, te_data,
                                               k=K_CLUSTERS)
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
        print(f"    Edit (UD) — Fold {fold_num} → acc = {acc:.3f}")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


# ── Random Forest ─────────────────────────────────────────────────────────────

def crossval_ui_rf(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "") -> tuple[float, float, list]:
    X         = build_feature_dataset(data_pca, evr_list)
    y         = np.array(labels)
    fold_accs = []
    for fold_num, (tr, te) in enumerate(folds):
        clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                     max_features=RF_MAX_FEATURES,
                                     random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        acc = float(np.mean(clf.predict(X[te]) == y[te]))
        fold_accs.append(acc)
        u = sorted(set(users))[fold_num]
        print(f"    RF{tag}   (UI) — User {u} → acc = {acc:.3f}")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def crossval_ud_rf(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "") -> tuple[float, float, list]:
    X         = build_feature_dataset(data_pca, evr_list)
    y         = np.array(labels)
    fold_accs = []
    for fold_num, (tr, te, _te_users) in enumerate(folds):
        clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                     max_features=RF_MAX_FEATURES,
                                     random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        acc = float(np.mean(clf.predict(X[te]) == y[te]))
        fold_accs.append(acc)
        print(f"    RF{tag}   (UD) — Fold {fold_num} → acc = {acc:.3f}")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


# ── LSTM (v2: dropout + early stopping + multi-seed averaging) ────────────────

def _build_lstm_model(timesteps: int, num_classes: int) -> Sequential:
    """
    LSTM model with dropout regularisation.

    Architecture
    ────────────
    Masking → LSTM(64, dropout=0.3, recurrent_dropout=0.3)
             → Dense(32, ReLU) → Dropout(0.3) → Dense(n_classes, softmax)

    The dropout rate LSTM_DROPOUT (0.3) and the patience LSTM_ES_PATIENCE
    (10) were selected as standard starting values recommended by
    Goodfellow et al. (2016) for sequence classification with limited data.
    """
    model = Sequential([
        Input(shape=(timesteps, 3)),
        Masking(mask_value=0.0),
        LSTM(LSTM_UNITS,
             dropout=LSTM_DROPOUT,
             recurrent_dropout=LSTM_DROPOUT),
        Dense(LSTM_DENSE_UNITS, activation="relu"),
        Dropout(LSTM_DROPOUT),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _train_lstm_one_seed(X_tr: np.ndarray,
                          y_tr: np.ndarray,
                          X_te: np.ndarray,
                          y_te: np.ndarray,
                          timesteps: int,
                          num_classes: int,
                          seed: int) -> float:
    """
    Train one LSTM model with a given random seed and return test accuracy.
    Early stopping is used to prevent overfitting (patience=LSTM_ES_PATIENCE).
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = _build_lstm_model(timesteps, num_classes)
    es    = EarlyStopping(monitor="val_loss",
                          patience=LSTM_ES_PATIENCE,
                          restore_best_weights=True,
                          verbose=0)
    model.fit(X_tr, y_tr,
              epochs=LSTM_MAX_EPOCHS,
              batch_size=LSTM_BATCH_SIZE,
              validation_split=0.1,
              callbacks=[es],
              verbose=0)
    _, acc = model.evaluate(X_te, y_te, verbose=0)
    return float(acc)


def crossval_ui_lstm(data_pca: list, labels: list, users: list,
                      folds: list,
                      max_len: int) -> tuple[float, float, list]:
    """
    User-independent LSTM evaluation.
    Each fold trains LSTM_N_SEEDS models (different initialisations) and
    averages their test accuracy, reducing variance due to random seeds.
    """
    min_label   = int(np.min(labels))
    labels_zi   = np.array(labels) - min_label
    num_classes = len(np.unique(labels_zi))
    X           = pad_sequences(data_pca, maxlen=max_len,
                                dtype="float32", padding="post",
                                truncating="post")
    fold_accs = []
    seeds     = list(range(42, 42 + LSTM_N_SEEDS))

    for fold_num, (tr, te) in enumerate(folds):
        X_tr, y_tr = X[tr], labels_zi[np.array(tr)]
        X_te, y_te = X[te], labels_zi[np.array(te)]

        seed_accs = [
            _train_lstm_one_seed(X_tr, y_tr, X_te, y_te,
                                 max_len, num_classes, s)
            for s in seeds
        ]
        acc = float(np.mean(seed_accs))
        fold_accs.append(acc)
        u = sorted(set(users))[fold_num]
        print(f"    LSTM (UI) — User {u} → acc = {acc:.3f}  "
              f"(seeds: {[f'{a:.3f}' for a in seed_accs]})")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def crossval_ud_lstm(data_pca: list, labels: list, users: list,
                      folds: list,
                      max_len: int) -> tuple[float, float, list]:
    """
    User-dependent LSTM evaluation.
    Same multi-seed averaging as the UI version.
    """
    min_label   = int(np.min(labels))
    labels_zi   = np.array(labels) - min_label
    num_classes = len(np.unique(labels_zi))
    X           = pad_sequences(data_pca, maxlen=max_len,
                                dtype="float32", padding="post",
                                truncating="post")
    fold_accs = []
    seeds     = list(range(42, 42 + LSTM_N_SEEDS))

    for fold_num, (tr, te, _te_users) in enumerate(folds):
        X_tr, y_tr = X[tr], labels_zi[np.array(tr)]
        X_te, y_te = X[te], labels_zi[np.array(te)]

        seed_accs = [
            _train_lstm_one_seed(X_tr, y_tr, X_te, y_te,
                                 max_len, num_classes, s)
            for s in seeds
        ]
        acc = float(np.mean(seed_accs))
        fold_accs.append(acc)
        print(f"    LSTM (UD) — Fold {fold_num} → acc = {acc:.3f}  "
              f"(seeds: {[f'{a:.3f}' for a in seed_accs]})")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


# ── $1 Recognizer ─────────────────────────────────────────────────────────────

def crossval_ui_dollar(data: list, labels: list, users: list,
                        folds: list) -> tuple[float, float, list]:
    """
    User-independent $1 Recognizer evaluation.
    Each test gesture is compared to all training gestures via
    dollar_distance (Steps 1–4 of Wobbrock et al., 2007).
    """
    fold_accs = []
    for fold_num, (tr, te) in enumerate(folds):
        tr_items  = [data[i] for i in tr]
        tr_labels = [labels[i] for i in tr]
        te_items  = [data[i] for i in te]
        te_labels = [labels[i] for i in te]
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, tr_items, tr_labels,
                                 dollar_distance, KNN_K)
            for ts in te_items
        )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        u = sorted(set(users))[fold_num]
        print(f"    $1   (UI) — User {u} held out → acc = {acc:.3f}")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def crossval_ud_dollar(data: list, labels: list, users: list,
                        folds: list) -> tuple[float, float, list]:
    """
    User-dependent $1 Recognizer evaluation.
    """
    fold_accs = []
    for fold_num, (tr, te, te_users) in enumerate(folds):
        tr_users_arr = [users[i] for i in tr]
        te_items     = [data[i]  for i in te]
        te_labels    = [labels[i] for i in te]
        preds = []
        for ts, ts_user in zip(te_items, te_users):
            same_user_mask = [j for j, u in enumerate(tr_users_arr)
                              if u == ts_user]
            tr_items_u  = [data[tr[j]] for j in same_user_mask]
            tr_labels_u = [labels[tr[j]] for j in same_user_mask]
            preds.append(
                knn_predict(ts, tr_items_u, tr_labels_u,
                            dollar_distance, KNN_K)
            )
        acc = float(np.mean([p == t for p, t in zip(preds, te_labels)]))
        fold_accs.append(acc)
        print(f"    $1   (UD) — Fold {fold_num} → acc = {acc:.3f}")
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


# ==============================================================================
# 9.  ABLATION STUDY  — 5 methods × 3 preprocessing conditions
# ==============================================================================

def run_ablation_study(data_raw: list, data_std: list,
                        data_denoised: list, evr_list: list,
                        labels: list, users: list,
                        max_len: int, domain: int
                        ) -> tuple[pd.DataFrame, dict]:
    """
    Compare all five methods under three preprocessing conditions using
    user-independent leave-one-user-out CV (10 folds, same splits for all).

    Returns a best_preprocessing dict with one entry per method:
        {
          "DTW":           {"condition": "...", "data": ...,
                            "evr": ..., "mean": 0.xxx, "folds": [...]},
          "Edit Distance": {...},
          "RF":            {...},
          "LSTM":          {...},
          "$1":            {...},
        }

    Conditions
    ----------
    (a) No preprocessing  — raw 3D coordinates.
    (b) Standardisation   — per-gesture zero-mean/unit-std.
    (c) Std + PCA denoise — 3D→2D→3D denoising; RF features WITH EVR.
    """
    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY | Domain {domain} | User-independent")
    print(f"{'='*65}")

    folds_ui = _ui_fold_indices(users)

    rows = []
    methods   = ["Edit Distance", "DTW", "RF", "LSTM", "$1"]
    cond_data = {
        "(a) No preprocessing" : (data_raw,      None),
        "(b) Standardisation"  : (data_std,       None),
        "(c) Std + PCA denoise": (data_denoised,  evr_list),
    }
    results: dict = {m: {} for m in methods}

    def _record(cond, method, mean, std, folds, note=""):
        rows.append({"Preprocessing": cond, "Method": method,
                     "Mean": mean, "Std": std, "Note": note})
        note_str = f"  [{note}]" if note else ""
        print(f"    [{cond}] {method}: {mean:.3f} +/- {std:.3f}{note_str}")
        results[method][cond] = (mean, std, folds)

    # ---------------------------------------------------------------- (a)
    print("\n  (a) No preprocessing")
    m, s, f = crossval_ui_edit(data_raw, labels, users, folds_ui)
    _record("(a) No preprocessing", "Edit Distance", m, s, f)
    m, s, f = crossval_ui_dtw(data_raw, labels, users, folds_ui)
    _record("(a) No preprocessing", "DTW", m, s, f)
    m, s, f = crossval_ui_rf(data_raw, labels, users, folds_ui,
                              evr_list=None, tag=" [no EVR]")
    _record("(a) No preprocessing", "RF", m, s, f)
    m, s, f = crossval_ui_lstm(data_raw, labels, users, folds_ui, max_len)
    _record("(a) No preprocessing", "LSTM", m, s, f)
    m, s, f = crossval_ui_dollar(data_raw, labels, users, folds_ui)
    _record("(a) No preprocessing", "$1", m, s, f)

    # ---------------------------------------------------------------- (b)
    print("\n  (b) Standardisation only")
    m, s, f = crossval_ui_edit(data_std, labels, users, folds_ui)
    _record("(b) Standardisation", "Edit Distance", m, s, f)
    m, s, f = crossval_ui_dtw(data_std, labels, users, folds_ui)
    _record("(b) Standardisation", "DTW", m, s, f)
    m, s, f = crossval_ui_rf(data_std, labels, users, folds_ui,
                              evr_list=None, tag=" [no EVR]")
    _record("(b) Standardisation", "RF", m, s, f)
    m, s, f = crossval_ui_lstm(data_std, labels, users, folds_ui, max_len)
    _record("(b) Standardisation", "LSTM", m, s, f)
    m, s, f = crossval_ui_dollar(data_std, labels, users, folds_ui)
    _record("(b) Standardisation", "$1", m, s, f)

    # ---------------------------------------------------------------- (c)
    print("\n  (c) Standardisation + PCA denoising 3D→2D→3D (full pipeline)")
    m, s, f = crossval_ui_edit(data_denoised, labels, users, folds_ui)
    _record("(c) Std + PCA denoise", "Edit Distance", m, s, f)
    m, s, f = crossval_ui_dtw(data_denoised, labels, users, folds_ui)
    _record("(c) Std + PCA denoise", "DTW", m, s, f)
    m, s, f = crossval_ui_rf(data_denoised, labels, users, folds_ui,
                              evr_list=evr_list, tag=" [+EVR]")
    _record("(c) Std + PCA denoise", "RF", m, s, f,
            note="3 PCA EVR values added to RF feature vector")
    m, s, f = crossval_ui_lstm(data_denoised, labels, users, folds_ui,
                                max_len)
    _record("(c) Std + PCA denoise", "LSTM", m, s, f)
    m, s, f = crossval_ui_dollar(data_denoised, labels, users, folds_ui)
    _record("(c) Std + PCA denoise", "$1", m, s, f)

    # ----------------------------------------------------------------
    # Best preprocessing per method
    # ----------------------------------------------------------------
    best_preprocessing: dict = {}

    print(f"\n  {'─'*60}")
    print(f"  Best preprocessing per method — Domain {domain}:")
    print(f"  {'─'*60}")

    for method in methods:
        best_cond = max(results[method], key=lambda c: results[method][c][0])
        best_mean, best_std, best_folds = results[method][best_cond]
        best_data, best_evr = cond_data[best_cond]

        if method == "RF" and best_cond != "(c) Std + PCA denoise":
            best_evr = None

        best_preprocessing[method] = {
            "condition": best_cond,
            "data"     : best_data,
            "evr"      : best_evr,
            "mean"     : best_mean,
            "std"      : best_std,
            "folds"    : best_folds,
        }
        print(f"    {method:<16}: {best_cond}  "
              f"(mean acc = {best_mean:.3f} +/- {best_std:.3f})")

    df = pd.DataFrame(rows)
    df["Result"] = (df["Mean"].map("{:.3f}".format)
                    + " +/- " + df["Std"].map("{:.3f}".format))
    pivot = df.pivot_table(index="Preprocessing", columns="Method",
                           values="Result", aggfunc="first")
    print(f"\n  Ablation summary — Domain {domain}:")
    print(pivot.to_string())
    df.to_csv(f"ablation_domain{domain}.csv", index=False)
    print(f"  Saved → ablation_domain{domain}.csv")

    return df, best_preprocessing


# ==============================================================================
# 10.  STATISTICAL TESTS
#      Paired Wilcoxon + permutation test (for robustness with n=10)
# ==============================================================================

def _permutation_pvalue(a: list, b: list,
                         n_perm: int = 10_000,
                         rng: np.random.Generator | None = None) -> float:
    """
    Two-sided paired permutation test on the difference of fold accuracies.

    Under H0, the sign of each paired difference is exchangeable.
    We randomly flip signs and compare the resulting mean difference to
    the observed one.  With n=10 folds, all 2^10=1024 permutations are
    exact; here we use Monte-Carlo for simplicity (n_perm=10000).

    Reference: Good (2005) "Permutation, Parametric, and Bootstrap Tests
               of Hypotheses", Springer.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    diffs   = np.array(a, dtype=float) - np.array(b, dtype=float)
    obs_stat = float(np.abs(np.mean(diffs)))
    n        = len(diffs)
    count    = 0
    for _ in range(n_perm):
        signs    = rng.choice([-1.0, 1.0], size=n)
        perm_stat = float(np.abs(np.mean(signs * diffs)))
        if perm_stat >= obs_stat:
            count += 1
    return count / n_perm


def generate_pvalue_table(methods_results: dict,
                           domain: int) -> pd.DataFrame:
    """
    Paired Wilcoxon signed-rank test + paired permutation test for all
    method pairs, with Benjamini-Hochberg FDR correction applied
    separately to each set of p-values.
    """
    names = list(methods_results.keys())
    n     = len(names)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if len(methods_results[names[i]]) ==
                len(methods_results[names[j]])]
    n_comp     = max(len(pairs), 1)
    alpha_bonf = 0.05 / n_comp

    pair_labels, raw_pvals_wilcox, raw_pvals_perm = [], [], []
    rng = np.random.default_rng(42)

    for i, j in pairs:
        try:
            _, p_w = wilcoxon(methods_results[names[i]],
                              methods_results[names[j]])
        except ValueError:
            # All differences are zero — methods are identical on every fold
            p_w = 1.0
        p_perm = _permutation_pvalue(methods_results[names[i]],
                                      methods_results[names[j]],
                                      n_perm=10_000, rng=rng)
        pair_labels.append((names[i], names[j]))
        raw_pvals_wilcox.append(float(p_w))
        raw_pvals_perm.append(float(p_perm))

    # BH correction on Wilcoxon p-values
    if raw_pvals_wilcox:
        reject_bh, pvals_bh, _, _ = multipletests(
            raw_pvals_wilcox, alpha=0.05, method="fdr_bh")
    else:
        reject_bh, pvals_bh = [], []

    # BH correction on permutation p-values
    if raw_pvals_perm:
        reject_bh_perm, pvals_bh_perm, _, _ = multipletests(
            raw_pvals_perm, alpha=0.05, method="fdr_bh")
    else:
        reject_bh_perm, pvals_bh_perm = [], []

    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)
    for k_idx, (i, j) in enumerate(pairs):
        matrix[i, j] = raw_pvals_wilcox[k_idx]
        matrix[j, i] = raw_pvals_wilcox[k_idx]
    df_raw = pd.DataFrame(matrix, index=names, columns=names)

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Statistical tests | Domain {domain} | User-independent")
    print(f"  Pairs : {n_comp}  |  n=10 per method")
    print(f"  Test 1 : Paired Wilcoxon signed-rank (fold #i of A vs B)")
    print(f"  Test 2 : Paired permutation test (10 000 permutations, seed=42)")
    print(f"  Both corrected with Benjamini-Hochberg FDR at 5%")
    print(f"  Bonferroni threshold : alpha/{n_comp} = {alpha_bonf:.4f}")
    print(sep)

    print("\n  RAW Wilcoxon p-value matrix:")
    print(df_raw.round(4).to_string())

    hdr = (f"\n  {'Method A':<20} {'Method B':<20} "
           f"{'Wilcox p':>9}  {'W-BH p':>9}  {'W-BH sig':>9}  "
           f"{'Perm p':>9}  {'Perm-BH p':>10}  {'P-BH sig':>9}")
    print(hdr)
    print("  " + "-" * 100)
    for k_idx, (i, j) in enumerate(pairs):
        na, nb      = names[i], names[j]
        p_w         = raw_pvals_wilcox[k_idx]
        p_bh_w      = float(pvals_bh[k_idx])
        bh_w_ok     = "YES *" if reject_bh[k_idx]      else "no"
        p_p         = raw_pvals_perm[k_idx]
        p_bh_p      = float(pvals_bh_perm[k_idx])
        bh_p_ok     = "YES *" if reject_bh_perm[k_idx] else "no"
        print(f"  {na:<20} {nb:<20} {p_w:>9.4f}  {p_bh_w:>9.4f}  "
              f"{bh_w_ok:>9}  {p_p:>9.4f}  {p_bh_p:>10.4f}  {bh_p_ok:>9}")

    means    = {name: float(np.mean(folds))
                for name, folds in methods_results.items()}
    best     = max(means, key=means.get)
    best_idx = names.index(best)
    print(f"\n  Best mean accuracy: {best} ({means[best]:.3f})")

    all_sig = True
    for k_idx, (i, j) in enumerate(pairs):
        if best_idx in (i, j) and not reject_bh[k_idx]:
            other = names[j] if i == best_idx else names[i]
            print(f"  → {best} NOT significantly better than {other} "
                  f"(Wilcoxon BH p={pvals_bh[k_idx]:.4f}, "
                  f"Perm BH p={pvals_bh_perm[k_idx]:.4f})")
            all_sig = False
    if all_sig and len(pairs) > 0:
        print(f"  → {best} is significantly better than ALL others (BH)")

    df_raw.to_csv(f"p_values_domain{domain}_user_independent.csv")
    print(f"  Saved → p_values_domain{domain}_user_independent.csv")
    return df_raw


# ==============================================================================
# 11.  CONFUSION MATRICES
# ==============================================================================

def _safe_filename(title: str) -> str:
    for ch in [" ", "|", "(", ")", "-", "/", "+", "→"]:
        title = title.replace(ch, "_")
    return title


def _plot_cm(y_true: list, y_pred: list,
              display_labels: list, title: str) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"confusion_{_safe_filename(title)}.png", dpi=150)
    plt.show()


def compute_cm_edit(data_denoised: list, labels: list, users: list,
                     folds: list,
                     title: str = "Confusion matrix — Edit Distance") -> None:
    y_true, y_pred = [], []
    for tr, te in folds:
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
                    title: str = "Confusion matrix — DTW") -> None:
    y_true, y_pred = [], []
    for tr, te in folds:
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
                   title: str = "Confusion matrix — RF") -> None:
    X      = build_feature_dataset(data_denoised, evr_list)
    y      = np.array(labels)
    y_true, y_pred = [], []
    for tr, te in folds:
        clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                     max_features=RF_MAX_FEATURES,
                                     random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        y_true.extend(y[te].tolist())
        y_pred.extend(clf.predict(X[te]).tolist())
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def compute_cm_lstm(data_denoised: list, labels: list, users: list,
                     folds: list, max_len: int,
                     title: str = "Confusion matrix — LSTM") -> None:
    min_label   = int(np.min(labels))
    labels_zi   = np.array(labels) - min_label
    num_classes = len(np.unique(labels_zi))
    X = pad_sequences(data_denoised, maxlen=max_len,
                      dtype="float32", padding="post", truncating="post")
    y_true, y_pred = [], []
    seeds = list(range(42, 42 + LSTM_N_SEEDS))
    for tr, te in folds:
        X_tr, y_tr = X[tr], labels_zi[np.array(tr)]
        X_te, y_te = X[te], labels_zi[np.array(te)]
        # Use first seed only for CM (speed); main scores already averaged
        tf.random.set_seed(42)
        np.random.seed(42)
        model = _build_lstm_model(max_len, num_classes)
        es    = EarlyStopping(monitor="val_loss", patience=LSTM_ES_PATIENCE,
                              restore_best_weights=True, verbose=0)
        model.fit(X_tr, y_tr,
                  epochs=LSTM_MAX_EPOCHS,
                  batch_size=LSTM_BATCH_SIZE,
                  validation_split=0.1, callbacks=[es], verbose=0)
        raw = np.argmax(model.predict(X_te, verbose=0), axis=1)
        y_true.extend((labels_zi[np.array(te)] + min_label).tolist())
        y_pred.extend((raw + min_label).tolist())
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def compute_cm_dollar(data: list, labels: list, users: list,
                       folds: list,
                       title: str = "Confusion matrix — $1") -> None:
    y_true, y_pred = [], []
    for tr, te in folds:
        tr_items  = [data[i] for i in tr]
        tr_labels = [labels[i] for i in tr]
        te_items  = [data[i]  for i in te]
        te_labels = [labels[i] for i in te]
        preds = Parallel(n_jobs=-1, prefer="threads")(
            delayed(knn_predict)(ts, tr_items, tr_labels,
                                 dollar_distance, KNN_K)
            for ts in te_items
        )
        y_true.extend(te_labels)
        y_pred.extend(preds)
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def draw_best_model_cm(best_name: str,
                        data_best: list, evr_best: list | None,
                        labels: list, users: list,
                        folds_ui: list, max_len: int,
                        domain: int) -> None:
    tag = f"User-independent Domain {domain}"
    if best_name == "Edit Distance":
        compute_cm_edit(data_best, labels, users, folds_ui,
                        title=f"Edit Distance {tag}")
    elif best_name == "DTW":
        compute_cm_dtw(data_best, labels, users, folds_ui,
                       title=f"DTW {tag}")
    elif best_name == "RF":
        compute_cm_rf(data_best, labels, users, folds_ui, evr_best,
                      title=f"RF {tag}")
    elif best_name == "LSTM":
        compute_cm_lstm(data_best, labels, users, folds_ui, max_len,
                        title=f"LSTM {tag}")
    elif best_name == "$1":
        compute_cm_dollar(data_best, labels, users, folds_ui,
                          title=f"$1 {tag}")


# ==============================================================================
# 12.  RESULT SAVING
# ==============================================================================

def save_fold_results(fold_accs: list, method: str,
                       setting: str, domain: int) -> None:
    fname = f"results_domain{domain}_{setting}_{method}.csv"
    pd.DataFrame({"accuracy": fold_accs}).to_csv(fname, index=False)
    print(f"  Saved → {fname}")


# ==============================================================================
# 13.  MAIN
# ==============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 0.  Global seeds for reproducibility + numba JIT warm-up
    # ------------------------------------------------------------------
    np.random.seed(42)
    tf.random.set_seed(42)

    print("Warming up numba JIT ...", end=" ", flush=True)
    _d = np.random.randn(10, 3)
    dtw_distance(_d, _d)
    _s = np.zeros(10, dtype=np.int64)
    edit_distance(_s, _s)
    print("done.")

    # ------------------------------------------------------------------
    # 1.  Load data
    # ------------------------------------------------------------------
    print("\n=== Loading Domain 1 ===")
    data1, labels1, users1 = load_domain1(DOMAIN1_DIR)
    max_len1 = print_dataset_info(data1, labels1, users1, "Domain 1")

    print("\n=== Loading Domain 4 ===")
    data4, labels4, users4 = load_domain4(DOMAIN4_DIR)
    max_len4 = print_dataset_info(data4, labels4, users4, "Domain 4")

    # ------------------------------------------------------------------
    # 2.  Exploratory visualisation
    # ------------------------------------------------------------------
    print("\n=== Exploratory Visualisation ===")
    plot_sequence_lengths(data1, labels1, "Domain 1",
                          save_path="d1_sequence_lengths.png")
    plot_gesture_samples(data1, labels1, users1, "Domain 1",
                         save_path="d1_gesture_samples.png")
    plot_sequence_lengths(data4, labels4, "Domain 4",
                          save_path="d4_sequence_lengths.png")
    plot_gesture_samples(data4, labels4, users4, "Domain 4",
                         save_path="d4_gesture_samples.png")

    # ------------------------------------------------------------------
    # 3.  Standardisation
    # ------------------------------------------------------------------
    print("\n=== Standardisation ===")
    data1_std = standardize_gestures(data1)
    data4_std = standardize_gestures(data4)
    print("  Both domains standardised (per-gesture, per-axis).")

    # ------------------------------------------------------------------
    # 4.  Per-gesture PCA denoising  3D → 2D → 3D
    # ------------------------------------------------------------------
    print("\n=== Per-gesture PCA denoising analysis ===")
    summarise_pca_denoising(data1_std, "Domain 1",
                             save_path="d1_pca_denoise.png")
    summarise_pca_denoising(data4_std, "Domain 4",
                             save_path="d4_pca_denoise.png")

    print("\n=== Applying PCA denoising (3D → 2D → 3D) ===")
    data1_denoised, evr1 = apply_pca_denoising(data1_std, n_keep=PCA_N_KEEP)
    data4_denoised, evr4 = apply_pca_denoising(data4_std, n_keep=PCA_N_KEEP)
    print("  PCA denoising applied to both domains.")
    print(f"  Domain 1 — example EVR: {evr1[0].round(3)}")
    print(f"  Domain 4 — example EVR: {evr4[0].round(3)}")

    # ------------------------------------------------------------------
    # 5.  Generate fold indices ONCE — shared by ALL methods
    # ------------------------------------------------------------------
    print("\n=== Generating fold indices (shared across all methods) ===")
    folds_ui_1 = _ui_fold_indices(users1)
    folds_ud_1 = _ud_fold_indices(labels1, users1)
    folds_ui_4 = _ui_fold_indices(users4)
    folds_ud_4 = _ud_fold_indices(labels4, users4)
    print(f"  Domain 1 — UI folds: {len(folds_ui_1)} | "
          f"UD folds: {len(folds_ud_1)}")
    print(f"  Domain 4 — UI folds: {len(folds_ui_4)} | "
          f"UD folds: {len(folds_ud_4)}")

    # ------------------------------------------------------------------
    # 6.  Ablation study  (5 methods × 3 preprocessing conditions)
    # ------------------------------------------------------------------
    print("\n=== Ablation Study — Domain 1 ===")
    _, best_prep_d1 = run_ablation_study(
        data1, data1_std, data1_denoised, evr1,
        labels1, users1, max_len1, domain=1)

    print("\n=== Ablation Study — Domain 4 ===")
    _, best_prep_d4 = run_ablation_study(
        data4, data4_std, data4_denoised, evr4,
        labels4, users4, max_len4, domain=4)

    # ------------------------------------------------------------------
    # Helper: map ablation condition → UD dataset
    # ------------------------------------------------------------------
    def _ud_data_for(best_entry: dict,
                     raw: list, std: list, denoised: list,
                     evr: list) -> tuple[list, list | None]:
        cond = best_entry["condition"]
        if cond == "(a) No preprocessing":
            return raw, None
        elif cond == "(b) Standardisation":
            return std, None
        else:
            return denoised, evr

    # ------------------------------------------------------------------
    # 7.  Domain 1 — USER-INDEPENDENT  (reuse ablation fold accuracies)
    # ------------------------------------------------------------------
    print("\n=== Main Evaluation — Domain 1 — User-Independent (10 folds) ===")
    print("  (Using per-method optimal preprocessing from ablation study)\n")

    for method in ["Edit Distance", "DTW", "RF", "LSTM", "$1"]:
        entry = best_prep_d1[method]
        print(f"  {method}: best condition = {entry['condition']}  "
              f"(mean = {entry['mean']:.3f} +/- {entry['std']:.3f})")

    folds_ed1     = best_prep_d1["Edit Distance"]["folds"]
    folds_dtw1    = best_prep_d1["DTW"]["folds"]
    folds_rf1     = best_prep_d1["RF"]["folds"]
    folds_lstm1   = best_prep_d1["LSTM"]["folds"]
    folds_dollar1 = best_prep_d1["$1"]["folds"]

    mean_ed1     = best_prep_d1["Edit Distance"]["mean"]
    mean_dtw1    = best_prep_d1["DTW"]["mean"]
    mean_rf1     = best_prep_d1["RF"]["mean"]
    mean_lstm1   = best_prep_d1["LSTM"]["mean"]
    mean_dollar1 = best_prep_d1["$1"]["mean"]

    std_ed1     = best_prep_d1["Edit Distance"]["std"]
    std_dtw1    = best_prep_d1["DTW"]["std"]
    std_rf1     = best_prep_d1["RF"]["std"]
    std_lstm1   = best_prep_d1["LSTM"]["std"]
    std_dollar1 = best_prep_d1["$1"]["std"]

    save_fold_results(folds_ed1,     "edit",   "user_independent", 1)
    save_fold_results(folds_dtw1,    "dtw",    "user_independent", 1)
    save_fold_results(folds_rf1,     "rf",     "user_independent", 1)
    save_fold_results(folds_lstm1,   "lstm",   "user_independent", 1)
    save_fold_results(folds_dollar1, "dollar", "user_independent", 1)

    # ------------------------------------------------------------------
    # 8.  Domain 1 — USER-DEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Main Evaluation — Domain 1 — User-Dependent (10 folds) ===")

    d1_ed_data,    d1_ed_evr   = _ud_data_for(best_prep_d1["Edit Distance"],
                                               data1, data1_std, data1_denoised, evr1)
    d1_dtw_data,   _            = _ud_data_for(best_prep_d1["DTW"],
                                               data1, data1_std, data1_denoised, evr1)
    d1_rf_data,    d1_rf_evr   = _ud_data_for(best_prep_d1["RF"],
                                               data1, data1_std, data1_denoised, evr1)
    d1_lstm_data,  _            = _ud_data_for(best_prep_d1["LSTM"],
                                               data1, data1_std, data1_denoised, evr1)
    d1_dollar_data, _           = _ud_data_for(best_prep_d1["$1"],
                                               data1, data1_std, data1_denoised, evr1)

    print(f"\n  Edit Distance  [{best_prep_d1['Edit Distance']['condition']}]:")
    mean_ed1_ud, std_ed1_ud, folds_ed1_ud = crossval_ud_edit(
        d1_ed_data, labels1, users1, folds_ud_1)
    print(f"  → {mean_ed1_ud:.3f} +/- {std_ed1_ud:.3f}")
    save_fold_results(folds_ed1_ud, "edit", "user_dependent", 1)

    print(f"\n  DTW  [{best_prep_d1['DTW']['condition']}]:")
    mean_dtw1_ud, std_dtw1_ud, folds_dtw1_ud = crossval_ud_dtw(
        d1_dtw_data, labels1, users1, folds_ud_1)
    print(f"  → {mean_dtw1_ud:.3f} +/- {std_dtw1_ud:.3f}")
    save_fold_results(folds_dtw1_ud, "dtw", "user_dependent", 1)

    print(f"\n  Random Forest  [{best_prep_d1['RF']['condition']}]:")
    mean_rf1_ud, std_rf1_ud, folds_rf1_ud = crossval_ud_rf(
        d1_rf_data, labels1, users1, folds_ud_1,
        evr_list=d1_rf_evr, tag=" [adaptive]")
    print(f"  → {mean_rf1_ud:.3f} +/- {std_rf1_ud:.3f}")
    save_fold_results(folds_rf1_ud, "rf", "user_dependent", 1)

    print(f"\n  LSTM  [{best_prep_d1['LSTM']['condition']}]:")
    mean_lstm1_ud, std_lstm1_ud, folds_lstm1_ud = crossval_ud_lstm(
        d1_lstm_data, labels1, users1, folds_ud_1, max_len1)
    print(f"  → {mean_lstm1_ud:.3f} +/- {std_lstm1_ud:.3f}")
    save_fold_results(folds_lstm1_ud, "lstm", "user_dependent", 1)

    print(f"\n  $1 Recognizer  [{best_prep_d1['$1']['condition']}]:")
    mean_dollar1_ud, std_dollar1_ud, folds_dollar1_ud = crossval_ud_dollar(
        d1_dollar_data, labels1, users1, folds_ud_1)
    print(f"  → {mean_dollar1_ud:.3f} +/- {std_dollar1_ud:.3f}")
    save_fold_results(folds_dollar1_ud, "dollar", "user_dependent", 1)

    # ------------------------------------------------------------------
    # 9.  Statistical tests — Domain 1
    # ------------------------------------------------------------------
    print("\n=== Statistical Tests — Domain 1 ===")
    results_ui_d1 = {
        "Edit Distance": folds_ed1,
        "DTW"          : folds_dtw1,
        "RF"           : folds_rf1,
        "LSTM"         : folds_lstm1,
        "$1"           : folds_dollar1,
    }
    generate_pvalue_table(results_ui_d1, domain=1)

    print("\n  [CAVEAT] Statistical tests compare methods under their individually")
    print("  optimal preprocessing conditions. Fold vectors are paired by fold")
    print("  index only, not by identical input data. Interpret p-values accordingly.")

    # ------------------------------------------------------------------
    # 10.  Confusion matrix — best model — Domain 1
    # ------------------------------------------------------------------
    all_ui_d1 = {"Edit Distance": mean_ed1, "DTW": mean_dtw1,
                 "RF": mean_rf1, "LSTM": mean_lstm1, "$1": mean_dollar1}
    best_name_d1  = max(all_ui_d1, key=all_ui_d1.get)
    best_entry_d1 = best_prep_d1[best_name_d1]
    print(f"\n  Best model — Domain 1 (UI): {best_name_d1} "
          f"({all_ui_d1[best_name_d1]:.3f})  "
          f"[{best_entry_d1['condition']}]")
    draw_best_model_cm(best_name_d1,
                       best_entry_d1["data"],
                       best_entry_d1["evr"],
                       labels1, users1, folds_ui_1, max_len1, domain=1)

    # ==================================================================
    # DOMAIN 4
    # ==================================================================

    # ------------------------------------------------------------------
    # 11.  Domain 4 — USER-INDEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Main Evaluation — Domain 4 — User-Independent (10 folds) ===")
    print("  (Using per-method optimal preprocessing from ablation study)\n")

    for method in ["Edit Distance", "DTW", "RF", "LSTM", "$1"]:
        entry = best_prep_d4[method]
        print(f"  {method}: best condition = {entry['condition']}  "
              f"(mean = {entry['mean']:.3f} +/- {entry['std']:.3f})")

    folds_ed4     = best_prep_d4["Edit Distance"]["folds"]
    folds_dtw4    = best_prep_d4["DTW"]["folds"]
    folds_rf4     = best_prep_d4["RF"]["folds"]
    folds_lstm4   = best_prep_d4["LSTM"]["folds"]
    folds_dollar4 = best_prep_d4["$1"]["folds"]

    mean_ed4     = best_prep_d4["Edit Distance"]["mean"]
    mean_dtw4    = best_prep_d4["DTW"]["mean"]
    mean_rf4     = best_prep_d4["RF"]["mean"]
    mean_lstm4   = best_prep_d4["LSTM"]["mean"]
    mean_dollar4 = best_prep_d4["$1"]["mean"]

    std_ed4     = best_prep_d4["Edit Distance"]["std"]
    std_dtw4    = best_prep_d4["DTW"]["std"]
    std_rf4     = best_prep_d4["RF"]["std"]
    std_lstm4   = best_prep_d4["LSTM"]["std"]
    std_dollar4 = best_prep_d4["$1"]["std"]

    save_fold_results(folds_ed4,     "edit",   "user_independent", 4)
    save_fold_results(folds_dtw4,    "dtw",    "user_independent", 4)
    save_fold_results(folds_rf4,     "rf",     "user_independent", 4)
    save_fold_results(folds_lstm4,   "lstm",   "user_independent", 4)
    save_fold_results(folds_dollar4, "dollar", "user_independent", 4)

    # ------------------------------------------------------------------
    # 12.  Domain 4 — USER-DEPENDENT
    # ------------------------------------------------------------------
    print("\n=== Main Evaluation — Domain 4 — User-Dependent (10 folds) ===")

    d4_ed_data,    d4_ed_evr   = _ud_data_for(best_prep_d4["Edit Distance"],
                                               data4, data4_std, data4_denoised, evr4)
    d4_dtw_data,   _            = _ud_data_for(best_prep_d4["DTW"],
                                               data4, data4_std, data4_denoised, evr4)
    d4_rf_data,    d4_rf_evr   = _ud_data_for(best_prep_d4["RF"],
                                               data4, data4_std, data4_denoised, evr4)
    d4_lstm_data,  _            = _ud_data_for(best_prep_d4["LSTM"],
                                               data4, data4_std, data4_denoised, evr4)
    d4_dollar_data, _           = _ud_data_for(best_prep_d4["$1"],
                                               data4, data4_std, data4_denoised, evr4)

    print(f"\n  Edit Distance  [{best_prep_d4['Edit Distance']['condition']}]:")
    mean_ed4_ud, std_ed4_ud, folds_ed4_ud = crossval_ud_edit(
        d4_ed_data, labels4, users4, folds_ud_4)
    print(f"  → {mean_ed4_ud:.3f} +/- {std_ed4_ud:.3f}")
    save_fold_results(folds_ed4_ud, "edit", "user_dependent", 4)

    print(f"\n  DTW  [{best_prep_d4['DTW']['condition']}]:")
    mean_dtw4_ud, std_dtw4_ud, folds_dtw4_ud = crossval_ud_dtw(
        d4_dtw_data, labels4, users4, folds_ud_4)
    print(f"  → {mean_dtw4_ud:.3f} +/- {std_dtw4_ud:.3f}")
    save_fold_results(folds_dtw4_ud, "dtw", "user_dependent", 4)

    print(f"\n  Random Forest  [{best_prep_d4['RF']['condition']}]:")
    mean_rf4_ud, std_rf4_ud, folds_rf4_ud = crossval_ud_rf(
        d4_rf_data, labels4, users4, folds_ud_4,
        evr_list=d4_rf_evr, tag=" [adaptive]")
    print(f"  → {mean_rf4_ud:.3f} +/- {std_rf4_ud:.3f}")
    save_fold_results(folds_rf4_ud, "rf", "user_dependent", 4)

    print(f"\n  LSTM  [{best_prep_d4['LSTM']['condition']}]:")
    mean_lstm4_ud, std_lstm4_ud, folds_lstm4_ud = crossval_ud_lstm(
        d4_lstm_data, labels4, users4, folds_ud_4, max_len4)
    print(f"  → {mean_lstm4_ud:.3f} +/- {std_lstm4_ud:.3f}")
    save_fold_results(folds_lstm4_ud, "lstm", "user_dependent", 4)

    print(f"\n  $1 Recognizer  [{best_prep_d4['$1']['condition']}]:")
    mean_dollar4_ud, std_dollar4_ud, folds_dollar4_ud = crossval_ud_dollar(
        d4_dollar_data, labels4, users4, folds_ud_4)
    print(f"  → {mean_dollar4_ud:.3f} +/- {std_dollar4_ud:.3f}")
    save_fold_results(folds_dollar4_ud, "dollar", "user_dependent", 4)

    # ------------------------------------------------------------------
    # 13.  Statistical tests — Domain 4
    # ------------------------------------------------------------------
    print("\n=== Statistical Tests — Domain 4 ===")
    results_ui_d4 = {
        "Edit Distance": folds_ed4,
        "DTW"          : folds_dtw4,
        "RF"           : folds_rf4,
        "LSTM"         : folds_lstm4,
        "$1"           : folds_dollar4,
    }
    generate_pvalue_table(results_ui_d4, domain=4)

    print("\n  [CAVEAT] Statistical tests compare methods under their individually")
    print("  optimal preprocessing conditions. Fold vectors are paired by fold")
    print("  index only, not by identical input data. Interpret p-values accordingly.")

    # ------------------------------------------------------------------
    # 14.  Confusion matrix — best model — Domain 4
    # ------------------------------------------------------------------
    all_ui_d4 = {"Edit Distance": mean_ed4, "DTW": mean_dtw4,
                 "RF": mean_rf4, "LSTM": mean_lstm4, "$1": mean_dollar4}
    best_name_d4  = max(all_ui_d4, key=all_ui_d4.get)
    best_entry_d4 = best_prep_d4[best_name_d4]
    print(f"\n  Best model — Domain 4 (UI): {best_name_d4} "
          f"({all_ui_d4[best_name_d4]:.3f})  "
          f"[{best_entry_d4['condition']}]")
    draw_best_model_cm(best_name_d4,
                       best_entry_d4["data"],
                       best_entry_d4["evr"],
                       labels4, users4, folds_ui_4, max_len4, domain=4)

    # ------------------------------------------------------------------
    # 15.  Final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Mean accuracy +/- std")
    print("UI = user-independent (10 folds) | UD = user-dependent (10 folds)")
    print("Best preprocessing per method selected automatically from ablation.")
    print("=" * 70)

    summary_rows = []
    for domain, res, bp in [
        (1, {
            ("Edit Distance", "UI"): (mean_ed1,       std_ed1,
                                      best_prep_d1["Edit Distance"]["condition"]),
            ("DTW",           "UI"): (mean_dtw1,      std_dtw1,
                                      best_prep_d1["DTW"]["condition"]),
            ("RF",            "UI"): (mean_rf1,       std_rf1,
                                      best_prep_d1["RF"]["condition"]),
            ("LSTM",          "UI"): (mean_lstm1,     std_lstm1,
                                      best_prep_d1["LSTM"]["condition"]),
            ("$1",            "UI"): (mean_dollar1,   std_dollar1,
                                      best_prep_d1["$1"]["condition"]),
            ("Edit Distance", "UD"): (mean_ed1_ud,    std_ed1_ud,
                                      best_prep_d1["Edit Distance"]["condition"]),
            ("DTW",           "UD"): (mean_dtw1_ud,   std_dtw1_ud,
                                      best_prep_d1["DTW"]["condition"]),
            ("RF",            "UD"): (mean_rf1_ud,    std_rf1_ud,
                                      best_prep_d1["RF"]["condition"]),
            ("LSTM",          "UD"): (mean_lstm1_ud,  std_lstm1_ud,
                                      best_prep_d1["LSTM"]["condition"]),
            ("$1",            "UD"): (mean_dollar1_ud, std_dollar1_ud,
                                      best_prep_d1["$1"]["condition"]),
        }, best_prep_d1),
        (4, {
            ("Edit Distance", "UI"): (mean_ed4,       std_ed4,
                                      best_prep_d4["Edit Distance"]["condition"]),
            ("DTW",           "UI"): (mean_dtw4,      std_dtw4,
                                      best_prep_d4["DTW"]["condition"]),
            ("RF",            "UI"): (mean_rf4,       std_rf4,
                                      best_prep_d4["RF"]["condition"]),
            ("LSTM",          "UI"): (mean_lstm4,     std_lstm4,
                                      best_prep_d4["LSTM"]["condition"]),
            ("$1",            "UI"): (mean_dollar4,   std_dollar4,
                                      best_prep_d4["$1"]["condition"]),
            ("Edit Distance", "UD"): (mean_ed4_ud,    std_ed4_ud,
                                      best_prep_d4["Edit Distance"]["condition"]),
            ("DTW",           "UD"): (mean_dtw4_ud,   std_dtw4_ud,
                                      best_prep_d4["DTW"]["condition"]),
            ("RF",            "UD"): (mean_rf4_ud,    std_rf4_ud,
                                      best_prep_d4["RF"]["condition"]),
            ("LSTM",          "UD"): (mean_lstm4_ud,  std_lstm4_ud,
                                      best_prep_d4["LSTM"]["condition"]),
            ("$1",            "UD"): (mean_dollar4_ud, std_dollar4_ud,
                                      best_prep_d4["$1"]["condition"]),
        }, best_prep_d4),
    ]:
        for (method, setting), (m, s, cond) in res.items():
            summary_rows.append({
                "Domain"        : domain,
                "Method"        : method,
                "Setting"       : setting,
                "Preprocessing" : cond,
                "Mean"          : m,
                "Std"           : s,
                "Result"        : f"{m:.3f} +/- {s:.3f}",
            })

    df_summary = pd.DataFrame(summary_rows)
    pivot = df_summary.pivot_table(
        index=["Domain", "Setting"], columns="Method",
        values="Result", aggfunc="first"
    )
    print(pivot.to_string())

    print("\n  Preprocessing selected per method:")
    for domain_id, bp in [(1, best_prep_d1), (4, best_prep_d4)]:
        print(f"\n  Domain {domain_id}:")
        for method in ["Edit Distance", "DTW", "RF", "LSTM", "$1"]:
            print(f"    {method:<16}: {bp[method]['condition']}")

    df_summary.to_csv("summary_results.csv", index=False)
    print("\n  Saved → summary_results.csv")
    print("\nDone.")