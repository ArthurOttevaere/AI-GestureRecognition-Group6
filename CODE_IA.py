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
            - 1-NN classifier for both
Phase 4 : Advanced methods
            - Random Forest with GridSearchCV (nested CV) + feature importance
            - $1 Recognizer 3D (Kratz & Rohs, 2010) with Rodrigues rotation,
              uniform cube scaling, confidence score, and N-best list
Phase 5 : Cross-validation
            - User-independent : leave-one-user-out (10 folds)
            - User-dependent   : leave-one-sample-out (10 folds)
          Ablation study: 4 methods x 3 preprocessing conditions
Phase 6 : Hyperparameter validation curves (empirical iterative selection)
Phase 7 : Statistical tests
            - Paired Wilcoxon signed-rank test on n=100 paired observations
              (10 gestures x 10 users), one accuracy per (gesture, user) pair
              for each method.
            - Bonferroni correction + Benjamini-Hochberg FDR correction.
            - Pairwise p-value matrix saved as CSV + heatmap.

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
     - Score:  S = 1 - d / (0.5 * sqrt(3) * l), where d is the mean
                point-to-point Euclidean distance after alignment.
                The factor sqrt(3) replaces sqrt(2) of the 2D paper:
                this is the diagonal of the unit cube versus the
                diagonal of the unit square.
     - Templates are preprocessed only ONCE and cached, as specified
       by Wobbrock et al. (2007).
     - Recognize returns a sorted N-best list, allowing kNN with k>1.
     - Golden Section Search refinement (Kratz & Rohs, 2010, sec.
       "Search for Minimum Distance at Best Angle") is NOT implemented
       to limit computational cost.

2. Wilcoxon n=100. For each method, a 100-vector of accuracies is built,
   one per (gesture, user) pair. Pairs are then compared method-vs-method
   via scipy.stats.wilcoxon (signed-rank). The all-zero-diff degenerate
   case is caught and returns p=1.0 with a warning.

3. Bonferroni AND Benjamini-Hochberg FDR corrections are reported.
   The earlier permutation test and Bayesian sign test are removed
   (redundant with Wilcoxon + correction).

4. LSTM removed. The dataset is too small (~1000 samples) to justify a
   recurrent network. Earlier results were retained from prior versions
   and are no longer relevant.

5. Random Forest hyperparameters selected per fold via GridSearchCV
   (3-fold inner CV). Feature importances analysed post-hoc to
   identify and prune uninformative features.

6. Hyperparameter tuning asymmetry. Random Forest hyperparameters are
   selected per-fold via GridSearchCV (nested CV, inner CV = 3 folds).
   DTW and Edit Distance hyperparameters (kNN K, k-means K) are selected
   once via empirical validation curves on the full user-independent CV,
   then kept fixed for evaluation. This asymmetry is intentional: DTW
   and Edit Distance have at most 2 scalar hyperparameters with a clear
   plateau; RF has a combinatorial grid that requires per-fold tuning to
   avoid overfitting. The comparison is therefore between best-configured
   versions of each method, not between methods sharing an identical
   tuning protocol. This limitation is acknowledged in the report.

7. K-clustering / k-NN K selected via empirical validation curves
   (accuracy vs K) on the user-independent CV.

References
----------
Wobbrock, J.O., Wilson, A.D., & Li, Y. (2007). Gestures without
  libraries, toolkits or training: A $1 recognizer for user interface
  prototypes. UIST'07, 159-168.
Kratz, S., & Rohs, M. (2010). A $3 gesture recognizer: simple gesture
  recognition for devices equipped with 3D acceleration sensors.
  IUI'10, 341-344.
Kratz, S., & Rohs, M. (2011). Protractor3D: A closed-form solution to
  rotation-invariant 3D gestures. IUI'11, 371-374.
Mezari, A., & Maglogiannis, I. (2018). An easily customized gesture
  recognizer for assisted living using commodity mobile devices.
  Journal of Healthcare Engineering, 2018:3180652.

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
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
from numba import njit
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


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

# RF GridSearchCV grid (kept compact to limit cost)
RF_GRID = {
    "n_estimators"     : [100, 200],
    "max_depth"        : [None, 20],
    "min_samples_split": [2, 5],
}
RF_GRID_INNER_CV = 3

# Validation-curve scan ranges (Section 5 of instructions)
VC_K_CLUSTERS = [5, 10, 15, 20, 25, 30, 40]
VC_KNN_K      = [1, 3, 5, 7, 9]

# $1 Recognizer hyperparameters (Kratz & Rohs, 2010)
DOLLAR_N         = 64           # number of resampled points
DOLLAR_L         = 1.0          # side of the normalised cube
DOLLAR_SCORE_DENOM = 0.5 * np.sqrt(3.0) * DOLLAR_L   # diagonal/2 of the cube


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
#   2. Translate centroid to the origin.
#   3. Rotate so that the first resampled point lies along the centroid
#      direction.  The rotation axis is the unit vector pâ x c (cross
#      product); the angle is acos((pâ . c) / (||pâ|| ||c||)).  Rotation
#      applied with Rodrigues' formula.  Degenerate case (pâ collinear
#      with c) -> identity rotation.
#   4. Uniformly rescale so that the longest bounding-box edge equals
#      DOLLAR_L (=1.0). This is the "normalised cube of side l" of
#      Kratz & Rohs (2010).  No axis-by-axis scaling: this avoids the
#      division-by-zero issue on quasi-planar gestures.
#
# Score
# -----
#   S = 1 - d / (0.5 * sqrt(3) * l)
# where d is the mean Euclidean point-to-point distance between the
# preprocessed candidate and the preprocessed template (Kratz & Rohs,
# 2010, eq. 3D score).  The factor sqrt(3) replaces sqrt(2) of the 2D
# original (cube diagonal vs square diagonal).
#
# Templates are preprocessed only once and cached, as required by
# Wobbrock et al. (2007, "For gestures serving as templates, Steps 1-3
# should be carried out once on the raw input points.").
#
# Note: the Golden Section Search refinement of Kratz & Rohs (2010) is
# NOT implemented (computational cost vs marginal accuracy gain).
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


def dollar_preprocess(points: np.ndarray,
                       n: int   = DOLLAR_N,
                       l: float = DOLLAR_L) -> np.ndarray:
    """
    Apply Steps 1-4 of the $3 (Kratz & Rohs, 2010) preprocessing.
    Used once on training templates (cached) and once on each candidate.
    Step order: resample -> translate to origin -> rotate to indicative
    angle -> scale to unit cube.
    The second translate-to-origin after scaling is omitted: uniform
    scaling preserves the centroid at the origin.
    """
    pts = dollar_resample(points, n)
    pts = _dollar_translate_to_origin(pts)
    pts = _dollar_align_to_indicative_axis(pts)
    pts = _dollar_scale_cube(pts, l)
    # Second _dollar_translate_to_origin removed: uniform scaling
    # preserves centroid at origin (no displacement).
    return pts


def _dollar_path_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Euclidean distance between two same-length 3D paths."""
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def dollar_score(distance: float, l: float = DOLLAR_L) -> float:
    """
    Confidence score in [0, 1] (Kratz & Rohs, 2010, 3D adaptation):
        S = 1 - d / (0.5 * sqrt(3) * l)
    Clipped to [0, 1] in case of numerical edge effects.
    """
    s = 1.0 - distance / (0.5 * np.sqrt(3.0) * l)
    return float(max(0.0, min(1.0, s)))


def dollar_recognize(candidate_pre: np.ndarray,
                      templates_pre: list,
                      template_labels: list,
                      l: float = DOLLAR_L
                      ) -> tuple[int, float, list]:
    """
    Step 5 of Kratz & Rohs (2010) recognition: rank all (preprocessed)
    templates against the (preprocessed) candidate by mean point-to-point
    distance, and return:
        (best_label, best_score, ranked_list)
    where ranked_list is a sorted N-best list:
        [(label, distance, score), ...]   sorted by distance ascending.
    """
    distances = [
        _dollar_path_distance(candidate_pre, t) for t in templates_pre
    ]
    order   = np.argsort(distances)
    ranked  = [(template_labels[k], float(distances[k]),
                dollar_score(distances[k], l)) for k in order]
    best_label, best_d, best_s = ranked[0]
    return int(best_label), float(best_s), ranked


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
                    feature_mask: list | None = None
                    ) -> tuple[float, float, list, np.ndarray]:
    X         = build_feature_dataset(data_pca, evr_list, feature_mask)
    y         = np.array(labels)
    fold_accs = []
    correct: dict = {}
    for fold_num, (tr, te) in enumerate(folds):
        if use_grid_search:
            clf = _rf_fit_with_grid(X[tr], y[tr])
        else:
            clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                         max_features=RF_MAX_FEATURES,
                                         random_state=42, n_jobs=-1)
            clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        acc   = float(np.mean(preds == y[te]))
        fold_accs.append(acc)
        for idx, p in zip(te, preds.tolist()):
            correct[idx] = int(p == labels[idx])
        u = sorted(set(users))[fold_num]
        print(f"    RF{tag}   (UI) - User {u} -> acc = {acc:.3f}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, gu


def crossval_ud_rf(data_pca: list, labels: list, users: list,
                    folds: list,
                    evr_list: list | None = None,
                    tag: str = "",
                    use_grid_search: bool = True,
                    feature_mask: list | None = None
                    ) -> tuple[float, float, list, np.ndarray]:
    X         = build_feature_dataset(data_pca, evr_list, feature_mask)
    y         = np.array(labels)
    fold_accs = []
    correct: dict = {}
    for fold_num, (tr, te, _te_users) in enumerate(folds):
        if use_grid_search:
            clf = _rf_fit_with_grid(X[tr], y[tr])
        else:
            clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                         max_features=RF_MAX_FEATURES,
                                         random_state=42, n_jobs=-1)
            clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        acc   = float(np.mean(preds == y[te]))
        fold_accs.append(acc)
        for idx, p in zip(te, preds.tolist()):
            correct[idx] = int(p == labels[idx])
        print(f"    RF{tag}   (UD) - Fold {fold_num} -> acc = {acc:.3f}")
    gu = _aggregate_gu_accuracy(correct, labels, users)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, gu


def analyse_rf_feature_importances(data_pca: list, labels: list,
                                     evr_list: list | None,
                                     domain: int,
                                     save_path: str | None = None,
                                     threshold: float = 0.01
                                     ) -> list:
    """
    Fit one RF on the full dataset (no leakage concern: this is post-hoc
    inspection only, not used for evaluation), inspect feature_importances_,
    and return the list of feature names whose importance exceeds the
    threshold. A barplot is saved.
    """
    X = build_feature_dataset(data_pca, evr_list)
    y = np.array(labels)
    clf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                 max_features=RF_MAX_FEATURES,
                                 random_state=42, n_jobs=-1)
    clf.fit(X, y)
    importances = clf.feature_importances_
    names = feature_names(with_evr=(evr_list is not None))

    order = np.argsort(importances)[::-1]
    sorted_names = [names[i] for i in order]
    sorted_imps  = importances[order]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.22 * len(names))))
    ax.barh(range(len(sorted_names))[::-1], sorted_imps,
            color="steelblue")
    ax.set_yticks(range(len(sorted_names))[::-1])
    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.axvline(threshold, color="red", linestyle="--",
               label=f"threshold = {threshold}")
    ax.set_xlabel("Feature importance (Gini)")
    ax.set_title(f"RF feature importances - Domain {domain}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    kept = [names[i] for i, imp in enumerate(importances)
            if imp >= threshold]
    print(f"  RF feature reduction (Domain {domain}): "
          f"{len(kept)}/{len(names)} kept (importance >= {threshold}).")
    return kept


# -- $1 Recognizer (Kratz & Rohs, 2010) with cached templates -----------------

def _dollar_predict_one(cand_pre: np.ndarray,
                         tmpl_pre: list,
                         tmpl_lbl: list,
                         k: int = KNN_K) -> int:
    """1-NN (or kNN) prediction over precomputed distances."""
    dists = np.array([_dollar_path_distance(cand_pre, t) for t in tmpl_pre])
    return knn_predict_from_distances(dists, tmpl_lbl, k=k)


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
                        k_clusters: int = K_CLUSTERS,
                        knn_k: int = KNN_K
                        ) -> tuple[pd.DataFrame, dict]:
    """
    Compare four methods under three preprocessing conditions on UI 10-fold
    CV. Returns a `best_preprocessing` dict per method.

    Conditions
    ----------
    (a) No preprocessing
    (b) Standardisation only
    (c) Standardisation + per-gesture PCA denoising (full pipeline)

    Parameters
    ----------
    k_clusters : int
        k-means K used by the Edit-Distance pipeline (per-domain optimum
        from validation curves).
    knn_k : int
        kNN K used by the DTW pipeline (per-domain optimum from
        validation curves).
    """
    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY | Domain {domain} | User-independent")
    print(f"{'='*65}")

    folds_ui = _ui_fold_indices(users)

    rows = []
    methods   = ["Edit Distance", "DTW", "RF", "$1"]
    cond_data = {
        "(a) No preprocessing" : (data_raw,      None),
        "(b) Standardisation"  : (data_std,       None),
        "(c) Std + PCA denoise": (data_denoised,  evr_list),
    }
    results: dict = {m: {} for m in methods}

    def _record(cond, method, mean, std, folds, gu, note=""):
        rows.append({"Preprocessing": cond, "Method": method,
                     "Mean": mean, "Std": std, "Note": note})
        note_str = f"  [{note}]" if note else ""
        print(f"    [{cond}] {method}: {mean:.3f} +/- {std:.3f}{note_str}")
        results[method][cond] = (mean, std, folds, gu)

    print("\n  (a) No preprocessing")
    m, s, f, gu = crossval_ui_edit(data_raw, labels, users, folds_ui,
                                    k_clusters=k_clusters)
    _record("(a) No preprocessing", "Edit Distance", m, s, f, gu)
    m, s, f, gu = crossval_ui_dtw(data_raw, labels, users, folds_ui,
                                    knn_k=knn_k)
    _record("(a) No preprocessing", "DTW", m, s, f, gu)
    m, s, f, gu = crossval_ui_rf(data_raw, labels, users, folds_ui,
                                   evr_list=None, tag=" [no EVR]")
    _record("(a) No preprocessing", "RF", m, s, f, gu)
    m, s, f, gu = crossval_ui_dollar(data_raw, labels, users, folds_ui)
    _record("(a) No preprocessing", "$1", m, s, f, gu)

    print("\n  (b) Standardisation only")
    m, s, f, gu = crossval_ui_edit(data_std, labels, users, folds_ui,
                                    k_clusters=k_clusters)
    _record("(b) Standardisation", "Edit Distance", m, s, f, gu)
    m, s, f, gu = crossval_ui_dtw(data_std, labels, users, folds_ui,
                                    knn_k=knn_k)
    _record("(b) Standardisation", "DTW", m, s, f, gu)
    m, s, f, gu = crossval_ui_rf(data_std, labels, users, folds_ui,
                                   evr_list=None, tag=" [no EVR]")
    _record("(b) Standardisation", "RF", m, s, f, gu)
    m, s, f, gu = crossval_ui_dollar(data_std, labels, users, folds_ui)
    _record("(b) Standardisation", "$1", m, s, f, gu)

    print("\n  (c) Standardisation + PCA denoising 3D->2D->3D")
    m, s, f, gu = crossval_ui_edit(data_denoised, labels, users, folds_ui,
                                    k_clusters=k_clusters)
    _record("(c) Std + PCA denoise", "Edit Distance", m, s, f, gu)
    m, s, f, gu = crossval_ui_dtw(data_denoised, labels, users, folds_ui,
                                    knn_k=knn_k)
    _record("(c) Std + PCA denoise", "DTW", m, s, f, gu)
    m, s, f, gu = crossval_ui_rf(data_denoised, labels, users, folds_ui,
                                   evr_list=evr_list, tag=" [+EVR]")
    _record("(c) Std + PCA denoise", "RF", m, s, f, gu,
            note="3 PCA EVR values added to RF feature vector")
    m, s, f, gu = crossval_ui_dollar(data_denoised, labels, users, folds_ui)
    _record("(c) Std + PCA denoise", "$1", m, s, f, gu)

    best_preprocessing: dict = {}

    print(f"\n  {'-'*60}")
    print(f"  Best preprocessing per method - Domain {domain}:")
    print(f"  {'-'*60}")

    for method in methods:
        best_cond = max(results[method], key=lambda c: results[method][c][0])
        best_mean, best_std, best_folds, best_gu = results[method][best_cond]
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
            "gu"       : best_gu,
        }
        print(f"    {method:<16}: {best_cond}  "
              f"(mean acc = {best_mean:.3f} +/- {best_std:.3f})")

    df = pd.DataFrame(rows)
    df["Result"] = (df["Mean"].map("{:.3f}".format)
                    + " +/- " + df["Std"].map("{:.3f}".format))
    pivot = df.pivot_table(index="Preprocessing", columns="Method",
                           values="Result", aggfunc="first")
    print(f"\n  Ablation summary - Domain {domain}:")
    print(pivot.to_string())
    df.to_csv(f"ablation_domain{domain}.csv", index=False)
    print(f"  Saved -> ablation_domain{domain}.csv")

    return df, best_preprocessing


# ==============================================================================
# 10.  STATISTICAL TESTS
#      Paired Wilcoxon signed-rank on n=100 (gesture, user) accuracy pairs.
#      Bonferroni + Benjamini-Hochberg FDR corrections.
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
    Bonferroni and Benjamini-Hochberg corrections are reported.
    Saves a square symmetric CSV of raw p-values and a heatmap PNG.

    Parameters
    ----------
    methods_gu : dict[str, np.ndarray]
        method name -> 100-vector of per-(gesture, user) accuracies.
    """
    names = list(methods_gu.keys())
    n     = len(names)

    pairs   = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_comp  = max(len(pairs), 1)
    alpha   = 0.05
    a_bonf  = alpha / n_comp

    raw_pvals = []
    for i, j in pairs:
        raw_pvals.append(_safe_wilcoxon(methods_gu[names[i]],
                                          methods_gu[names[j]]))

    if raw_pvals:
        reject_bh, pvals_bh, _, _ = multipletests(
            raw_pvals, alpha=alpha, method="fdr_bh")
        reject_bf, pvals_bf, _, _ = multipletests(
            raw_pvals, alpha=alpha, method="bonferroni")
    else:
        reject_bh, pvals_bh = [], []
        reject_bf, pvals_bf = [], []

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
    print(f"  Corrections : Bonferroni (alpha = {a_bonf:.4f}) + "
          f"Benjamini-Hochberg FDR @ 5%")
    print(sep)
    print("\n  RAW Wilcoxon p-value matrix (symmetric):")
    print(df_raw.round(4).to_string())

    hdr = (f"\n  {'Method A':<18} {'Method B':<18} "
           f"{'p_raw':>9}  {'p_BH':>9}  {'BH sig':>7}  "
           f"{'p_Bonf':>9}  {'Bonf sig':>9}")
    print(hdr)
    print("  " + "-" * 95)
    for k_idx, (i, j) in enumerate(pairs):
        na, nb  = names[i], names[j]
        p_r     = raw_pvals[k_idx]
        p_bh    = float(pvals_bh[k_idx])
        bh_ok   = "YES *" if reject_bh[k_idx] else "no"
        p_bf    = float(pvals_bf[k_idx])
        bf_ok   = "YES *" if reject_bf[k_idx] else "no"
        print(f"  {na:<18} {nb:<18} {p_r:>9.4f}  {p_bh:>9.4f}  "
              f"{bh_ok:>7}  {p_bf:>9.4f}  {bf_ok:>9}")

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
                  f"(BH p={pvals_bh[k_idx]:.4f}, "
                  f"Bonf p={pvals_bf[k_idx]:.4f})")
            all_sig = False
    if all_sig and len(pairs) > 0:
        print(f"  -> {best} is significantly better than ALL others (BH)")

    csv_path = f"p_values_domain{domain}_user_independent.csv"
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
    heatmap_path = f"p_values_heatmap_domain{domain}.png"
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
    plt.savefig(f"confusion_{_safe_filename(title)}.png", dpi=150)
    plt.show()


def compute_cm_edit(data_denoised: list, labels: list, users: list,
                     folds: list,
                     title: str = "Confusion matrix - Edit Distance") -> None:
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
                    title: str = "Confusion matrix - DTW") -> None:
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
                   title: str = "Confusion matrix - RF") -> None:
    X      = build_feature_dataset(data_denoised, evr_list)
    y      = np.array(labels)
    y_true, y_pred = [], []
    for tr, te in folds:
        clf = _rf_fit_with_grid(X[tr], y[tr])
        y_true.extend(y[te].tolist())
        y_pred.extend(clf.predict(X[te]).tolist())
    _plot_cm(y_true, y_pred, sorted(set(labels)), title)


def compute_cm_dollar(data: list, labels: list, users: list,
                       folds: list,
                       title: str = "Confusion matrix - $1") -> None:
    pre_all = [dollar_preprocess(seq) for seq in data]
    y_true, y_pred = [], []
    for tr, te in folds:
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
    elif best_name == "RF":
        compute_cm_rf(data_best, labels, users, folds_ui, evr_best,
                      title=f"RF {tag}")
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
                          save_path="d1_sequence_lengths.png")
    plot_gesture_samples(data1, labels1, users1, "Domain 1",
                         save_path="d1_gesture_samples.png")
    plot_sequence_lengths(data4, labels4, "Domain 4",
                          save_path="d4_sequence_lengths.png")
    plot_gesture_samples(data4, labels4, users4, "Domain 4",
                         save_path="d4_gesture_samples.png")

    # -- 3. Standardisation -----------------------------------------------
    print("\n=== Standardisation ===")
    data1_std = standardize_gestures(data1)
    data4_std = standardize_gestures(data4)
    print("  Both domains standardised (per-gesture, per-axis).")

    # -- 4. PCA denoising -------------------------------------------------
    print("\n=== Per-gesture PCA denoising analysis ===")
    summarise_pca_denoising(data1_std, "Domain 1",
                             save_path="d1_pca_denoise.png")
    summarise_pca_denoising(data4_std, "Domain 4",
                             save_path="d4_pca_denoise.png")

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
    print("\n=== Validation curves - hyperparameter K ===")
    print("  Domain 1 - K_CLUSTERS scan (Edit Distance UI):")
    best_k_clusters_d1 = validation_curve_kclusters(
        data1_std, labels1, users1, folds_ui_1,
        domain=1, save_path="d1_vc_kclusters.png")
    print("  Domain 1 - kNN K scan (DTW UI):")
    best_knn_k_d1 = validation_curve_knn(
        data1_std, labels1, users1, folds_ui_1,
        method="dtw", domain=1, save_path="d1_vc_knn.png")
    print(f"  Domain 1 selected (on standardised data): "
          f"K_CLUSTERS={best_k_clusters_d1}, KNN_K={best_knn_k_d1}")

    print("\n  Domain 4 - K_CLUSTERS scan (Edit Distance UI):")
    best_k_clusters_d4 = validation_curve_kclusters(
        data4_std, labels4, users4, folds_ui_4,
        domain=4, save_path="d4_vc_kclusters.png")
    print("  Domain 4 - kNN K scan (DTW UI):")
    best_knn_k_d4 = validation_curve_knn(
        data4_std, labels4, users4, folds_ui_4,
        method="dtw", domain=4, save_path="d4_vc_knn.png")
    print(f"  Domain 4 selected (on standardised data): "
          f"K_CLUSTERS={best_k_clusters_d4}, KNN_K={best_knn_k_d4}")

    # Hyperparameters selected per domain via empirical validation curves above.
    # K_CLUSTERS and KNN_K globals are used as defaults only (validation curve scan).

    # -- 7. Ablation study ------------------------------------------------
    print("\n=== Ablation Study - Domain 1 ===")
    _, best_prep_d1 = run_ablation_study(
        data1, data1_std, data1_denoised, evr1,
        labels1, users1, domain=1,
        k_clusters=best_k_clusters_d1,
        knn_k=best_knn_k_d1)

    print("\n=== Ablation Study - Domain 4 ===")
    _, best_prep_d4 = run_ablation_study(
        data4, data4_std, data4_denoised, evr4,
        labels4, users4, domain=4,
        k_clusters=best_k_clusters_d4,
        knn_k=best_knn_k_d4)

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

    # -- 8. RF feature importance analysis (post-hoc, both domains) -------
    print("\n=== RF feature importance analysis ===")
    rf_d1_data, rf_d1_evr = _ud_data_for(best_prep_d1["RF"],
                                          data1, data1_std, data1_denoised, evr1)
    kept_features_d1 = analyse_rf_feature_importances(
        rf_d1_data, labels1, rf_d1_evr, domain=1,
        save_path="d1_rf_feature_importance.png")
    rf_d4_data, rf_d4_evr = _ud_data_for(best_prep_d4["RF"],
                                          data4, data4_std, data4_denoised, evr4)
    kept_features_d4 = analyse_rf_feature_importances(
        rf_d4_data, labels4, rf_d4_evr, domain=4,
        save_path="d4_rf_feature_importance.png")

    if len(kept_features_d1) == 0:
        print("  [WARNING] No features passed importance threshold; "
              "using full feature set.")
        kept_features_d1 = None
    if len(kept_features_d4) == 0:
        print("  [WARNING] No features passed importance threshold; "
              "using full feature set.")
        kept_features_d4 = None

    # -- 9. Domain 1 - UI (reuse ablation fold accuracies) ----------------
    print("\n=== Main Evaluation - Domain 1 - User-Independent ===")
    for method in ["Edit Distance", "DTW", "RF", "$1"]:
        entry = best_prep_d1[method]
        print(f"  {method}: best condition = {entry['condition']}  "
              f"(mean = {entry['mean']:.3f} +/- {entry['std']:.3f})")

    folds_ed1     = best_prep_d1["Edit Distance"]["folds"]
    folds_dtw1    = best_prep_d1["DTW"]["folds"]
    folds_rf1     = best_prep_d1["RF"]["folds"]
    folds_dollar1 = best_prep_d1["$1"]["folds"]

    gu_ed1     = best_prep_d1["Edit Distance"]["gu"]
    gu_dtw1    = best_prep_d1["DTW"]["gu"]
    gu_rf1     = best_prep_d1["RF"]["gu"]
    gu_dollar1 = best_prep_d1["$1"]["gu"]

    mean_ed1, std_ed1         = best_prep_d1["Edit Distance"]["mean"], best_prep_d1["Edit Distance"]["std"]
    mean_dtw1, std_dtw1       = best_prep_d1["DTW"]["mean"],           best_prep_d1["DTW"]["std"]
    mean_rf1, std_rf1         = best_prep_d1["RF"]["mean"],            best_prep_d1["RF"]["std"]
    mean_dollar1, std_dollar1 = best_prep_d1["$1"]["mean"],            best_prep_d1["$1"]["std"]

    save_fold_results(folds_ed1,     "edit",   "user_independent", 1)
    save_fold_results(folds_dtw1,    "dtw",    "user_independent", 1)
    save_fold_results(folds_rf1,     "rf",     "user_independent", 1)
    save_fold_results(folds_dollar1, "dollar", "user_independent", 1)

    # -- 10. Domain 1 - UD ------------------------------------------------
    print("\n=== Main Evaluation - Domain 1 - User-Dependent ===")
    d1_ed_data,    d1_ed_evr   = _ud_data_for(best_prep_d1["Edit Distance"],
                                               data1, data1_std, data1_denoised, evr1)
    d1_dtw_data,   _            = _ud_data_for(best_prep_d1["DTW"],
                                               data1, data1_std, data1_denoised, evr1)
    d1_rf_data,    d1_rf_evr   = _ud_data_for(best_prep_d1["RF"],
                                               data1, data1_std, data1_denoised, evr1)
    d1_dollar_data, _           = _ud_data_for(best_prep_d1["$1"],
                                               data1, data1_std, data1_denoised, evr1)

    print(f"\n  Edit Distance  [{best_prep_d1['Edit Distance']['condition']}]:")
    mean_ed1_ud, std_ed1_ud, folds_ed1_ud, _ = crossval_ud_edit(
        d1_ed_data, labels1, users1, folds_ud_1)
    save_fold_results(folds_ed1_ud, "edit", "user_dependent", 1)

    print(f"\n  DTW  [{best_prep_d1['DTW']['condition']}]:")
    mean_dtw1_ud, std_dtw1_ud, folds_dtw1_ud, _ = crossval_ud_dtw(
        d1_dtw_data, labels1, users1, folds_ud_1)
    save_fold_results(folds_dtw1_ud, "dtw", "user_dependent", 1)

    print(f"\n  Random Forest  [{best_prep_d1['RF']['condition']}]:")
    mean_rf1_ud, std_rf1_ud, folds_rf1_ud, _ = crossval_ud_rf(
        d1_rf_data, labels1, users1, folds_ud_1,
        evr_list=d1_rf_evr, tag=" [adaptive]",
        feature_mask=kept_features_d1)
    save_fold_results(folds_rf1_ud, "rf", "user_dependent", 1)

    print(f"\n  $1 Recognizer  [{best_prep_d1['$1']['condition']}]:")
    mean_dollar1_ud, std_dollar1_ud, folds_dollar1_ud, _ = crossval_ud_dollar(
        d1_dollar_data, labels1, users1, folds_ud_1)
    save_fold_results(folds_dollar1_ud, "dollar", "user_dependent", 1)

    # -- 11. Statistical tests Domain 1 -----------------------------------
    print("\n=== Statistical Tests - Domain 1 ===")
    results_gu_d1 = {
        "Edit Distance": gu_ed1,
        "DTW"          : gu_dtw1,
        "RF"           : gu_rf1,
        "$1"           : gu_dollar1,
    }
    generate_pvalue_table(results_gu_d1, domain=1)

    # -- 12. Confusion matrix - best model - Domain 1 ---------------------
    all_ui_d1 = {"Edit Distance": mean_ed1, "DTW": mean_dtw1,
                 "RF": mean_rf1, "$1": mean_dollar1}
    best_name_d1  = max(all_ui_d1, key=all_ui_d1.get)
    best_entry_d1 = best_prep_d1[best_name_d1]
    print(f"\n  Best model - Domain 1 (UI): {best_name_d1} "
          f"({all_ui_d1[best_name_d1]:.3f})  "
          f"[{best_entry_d1['condition']}]")
    draw_best_model_cm(best_name_d1,
                       best_entry_d1["data"],
                       best_entry_d1["evr"],
                       labels1, users1, folds_ui_1, domain=1)

    # ====================================================================
    # DOMAIN 4
    # ====================================================================

    # -- 13. Domain 4 - UI ------------------------------------------------
    print("\n=== Main Evaluation - Domain 4 - User-Independent ===")
    for method in ["Edit Distance", "DTW", "RF", "$1"]:
        entry = best_prep_d4[method]
        print(f"  {method}: best condition = {entry['condition']}  "
              f"(mean = {entry['mean']:.3f} +/- {entry['std']:.3f})")

    folds_ed4     = best_prep_d4["Edit Distance"]["folds"]
    folds_dtw4    = best_prep_d4["DTW"]["folds"]
    folds_rf4     = best_prep_d4["RF"]["folds"]
    folds_dollar4 = best_prep_d4["$1"]["folds"]

    gu_ed4     = best_prep_d4["Edit Distance"]["gu"]
    gu_dtw4    = best_prep_d4["DTW"]["gu"]
    gu_rf4     = best_prep_d4["RF"]["gu"]
    gu_dollar4 = best_prep_d4["$1"]["gu"]

    mean_ed4, std_ed4         = best_prep_d4["Edit Distance"]["mean"], best_prep_d4["Edit Distance"]["std"]
    mean_dtw4, std_dtw4       = best_prep_d4["DTW"]["mean"],           best_prep_d4["DTW"]["std"]
    mean_rf4, std_rf4         = best_prep_d4["RF"]["mean"],            best_prep_d4["RF"]["std"]
    mean_dollar4, std_dollar4 = best_prep_d4["$1"]["mean"],            best_prep_d4["$1"]["std"]

    save_fold_results(folds_ed4,     "edit",   "user_independent", 4)
    save_fold_results(folds_dtw4,    "dtw",    "user_independent", 4)
    save_fold_results(folds_rf4,     "rf",     "user_independent", 4)
    save_fold_results(folds_dollar4, "dollar", "user_independent", 4)

    # -- 14. Domain 4 - UD ------------------------------------------------
    print("\n=== Main Evaluation - Domain 4 - User-Dependent ===")
    d4_ed_data,    d4_ed_evr   = _ud_data_for(best_prep_d4["Edit Distance"],
                                               data4, data4_std, data4_denoised, evr4)
    d4_dtw_data,   _            = _ud_data_for(best_prep_d4["DTW"],
                                               data4, data4_std, data4_denoised, evr4)
    d4_rf_data,    d4_rf_evr   = _ud_data_for(best_prep_d4["RF"],
                                               data4, data4_std, data4_denoised, evr4)
    d4_dollar_data, _           = _ud_data_for(best_prep_d4["$1"],
                                               data4, data4_std, data4_denoised, evr4)

    print(f"\n  Edit Distance  [{best_prep_d4['Edit Distance']['condition']}]:")
    mean_ed4_ud, std_ed4_ud, folds_ed4_ud, _ = crossval_ud_edit(
        d4_ed_data, labels4, users4, folds_ud_4)
    save_fold_results(folds_ed4_ud, "edit", "user_dependent", 4)

    print(f"\n  DTW  [{best_prep_d4['DTW']['condition']}]:")
    mean_dtw4_ud, std_dtw4_ud, folds_dtw4_ud, _ = crossval_ud_dtw(
        d4_dtw_data, labels4, users4, folds_ud_4)
    save_fold_results(folds_dtw4_ud, "dtw", "user_dependent", 4)

    print(f"\n  Random Forest  [{best_prep_d4['RF']['condition']}]:")
    mean_rf4_ud, std_rf4_ud, folds_rf4_ud, _ = crossval_ud_rf(
        d4_rf_data, labels4, users4, folds_ud_4,
        evr_list=d4_rf_evr, tag=" [adaptive]",
        feature_mask=kept_features_d4)
    save_fold_results(folds_rf4_ud, "rf", "user_dependent", 4)

    print(f"\n  $1 Recognizer  [{best_prep_d4['$1']['condition']}]:")
    mean_dollar4_ud, std_dollar4_ud, folds_dollar4_ud, _ = crossval_ud_dollar(
        d4_dollar_data, labels4, users4, folds_ud_4)
    save_fold_results(folds_dollar4_ud, "dollar", "user_dependent", 4)

    # -- 15. Statistical tests Domain 4 -----------------------------------
    print("\n=== Statistical Tests - Domain 4 ===")
    results_gu_d4 = {
        "Edit Distance": gu_ed4,
        "DTW"          : gu_dtw4,
        "RF"           : gu_rf4,
        "$1"           : gu_dollar4,
    }
    generate_pvalue_table(results_gu_d4, domain=4)

    # -- 16. Confusion matrix - best model - Domain 4 ---------------------
    all_ui_d4 = {"Edit Distance": mean_ed4, "DTW": mean_dtw4,
                 "RF": mean_rf4, "$1": mean_dollar4}
    best_name_d4  = max(all_ui_d4, key=all_ui_d4.get)
    best_entry_d4 = best_prep_d4[best_name_d4]
    print(f"\n  Best model - Domain 4 (UI): {best_name_d4} "
          f"({all_ui_d4[best_name_d4]:.3f})  "
          f"[{best_entry_d4['condition']}]")
    draw_best_model_cm(best_name_d4,
                       best_entry_d4["data"],
                       best_entry_d4["evr"],
                       labels4, users4, folds_ui_4, domain=4)

    # -- 17. Final summary table ------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - Mean accuracy +/- std")
    print("UI = user-independent | UD = user-dependent")
    print("=" * 70)

    summary_rows = []
    for domain, res, bp in [
        (1, {
            ("Edit Distance", "UI"): (mean_ed1,        std_ed1,
                                      best_prep_d1["Edit Distance"]["condition"]),
            ("DTW",           "UI"): (mean_dtw1,       std_dtw1,
                                      best_prep_d1["DTW"]["condition"]),
            ("RF",            "UI"): (mean_rf1,        std_rf1,
                                      best_prep_d1["RF"]["condition"]),
            ("$1",            "UI"): (mean_dollar1,    std_dollar1,
                                      best_prep_d1["$1"]["condition"]),
            ("Edit Distance", "UD"): (mean_ed1_ud,     std_ed1_ud,
                                      best_prep_d1["Edit Distance"]["condition"]),
            ("DTW",           "UD"): (mean_dtw1_ud,    std_dtw1_ud,
                                      best_prep_d1["DTW"]["condition"]),
            ("RF",            "UD"): (mean_rf1_ud,     std_rf1_ud,
                                      best_prep_d1["RF"]["condition"]),
            ("$1",            "UD"): (mean_dollar1_ud, std_dollar1_ud,
                                      best_prep_d1["$1"]["condition"]),
        }, best_prep_d1),
        (4, {
            ("Edit Distance", "UI"): (mean_ed4,        std_ed4,
                                      best_prep_d4["Edit Distance"]["condition"]),
            ("DTW",           "UI"): (mean_dtw4,       std_dtw4,
                                      best_prep_d4["DTW"]["condition"]),
            ("RF",            "UI"): (mean_rf4,        std_rf4,
                                      best_prep_d4["RF"]["condition"]),
            ("$1",            "UI"): (mean_dollar4,    std_dollar4,
                                      best_prep_d4["$1"]["condition"]),
            ("Edit Distance", "UD"): (mean_ed4_ud,     std_ed4_ud,
                                      best_prep_d4["Edit Distance"]["condition"]),
            ("DTW",           "UD"): (mean_dtw4_ud,    std_dtw4_ud,
                                      best_prep_d4["DTW"]["condition"]),
            ("RF",            "UD"): (mean_rf4_ud,     std_rf4_ud,
                                      best_prep_d4["RF"]["condition"]),
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
        for method in ["Edit Distance", "DTW", "RF", "$1"]:
            print(f"    {method:<16}: {bp[method]['condition']}")

    df_summary.to_csv("summary_results.csv", index=False)
    print("\n  Saved -> summary_results.csv")
    print("\nDone.")
