"""
6. FEATURE EXTRACTION
======================
Statistical and kinematic feature extraction from 3D gesture trajectories.

Design decisions
----------------
No FFT features
    FFT coefficients are not comparable across sequences of different
    lengths (31-314 time steps) without prior resampling to a fixed
    length.  They are therefore excluded from the RF feature set.

No sequence length
    Sequence length varies with recording speed and user habits; it is
    not a gesture-shape descriptor and could bias user-dependent results.

Per-gesture PCA EVR (optional)
    Three explained-variance ratios from a PCA fitted on a single gesture's
    time-step points are added as features in ablation condition (c).
    They capture planarity / 3D-ness of each individual trajectory.
"""

import numpy as np
import pandas as pd

from preprocessing import pca_per_gesture


def extract_features(sequence: np.ndarray,
                     include_pca_evr: bool = True) -> np.ndarray:
    """
    Extract statistical and kinematic features from a 3D trajectory.

    Features (total: 32 without PCA, 35 with PCA EVR)
    ---------------------------------------------------
    Per axis (x, y, z)  7 x 3 = 21:
        mean, std, min, max, range, skewness, kurtosis
    Velocity (1st finite differences)  4:
        mean speed, std speed, max speed, min speed
    Acceleration (2nd finite differences)  2:
        mean magnitude, std magnitude
    Path length  1:
        total arc length (sum of inter-frame Euclidean distances)
    Bounding box  4:
        side lengths dx, dy, dz  +  diagonal length
    Per-gesture PCA EVR  3  (only when include_pca_evr=True):
        EVR of PC1, PC2, PC3 - captures planarity / 3D-ness of gesture.

    Parameters
    ----------
    sequence        : np.ndarray (T, 3)
    include_pca_evr : add the 3 PCA EVR values as features

    Returns
    -------
    np.ndarray of shape (32,) or (35,).
    """
    features = []

    # Per-axis statistics
    for i in range(3):
        axis = sequence[:, i]
        features.extend([
            float(np.mean(axis)),
            float(np.std(axis)),
            float(np.min(axis)),
            float(np.max(axis)),
            float(np.max(axis) - np.min(axis)),   # range
            float(pd.Series(axis).skew()),         # skewness
            float(pd.Series(axis).kurt()),         # kurtosis
        ])

    # Velocity (1st finite differences)
    velocity = np.diff(sequence, axis=0)          # (T-1, 3)
    speed    = np.linalg.norm(velocity, axis=1)   # (T-1,)
    features.extend([float(np.mean(speed)), float(np.std(speed)),
                     float(np.max(speed)),  float(np.min(speed))])

    # Acceleration (2nd finite differences)
    if len(velocity) > 1:
        accel      = np.diff(velocity, axis=0)
        accel_norm = np.linalg.norm(accel, axis=1)
        features.extend([float(np.mean(accel_norm)),
                         float(np.std(accel_norm))])
    else:
        features.extend([0.0, 0.0])

    # Path length
    features.append(float(np.sum(speed)))

    # Bounding box
    bbox = np.max(sequence, axis=0) - np.min(sequence, axis=0)
    features.extend(bbox.tolist())
    features.append(float(np.linalg.norm(bbox)))

    # Per-gesture PCA EVR (ablation condition (c) only)
    if include_pca_evr:
        evr = pca_per_gesture(sequence, n_components=3)["evr"]
        features.extend(evr.tolist())

    return np.array(features, dtype=float)


def build_feature_dataset(data: list,
                           include_pca_evr: bool = True) -> np.ndarray:
    """Build a feature matrix (N x D) from a list of sequences."""
    return np.array([extract_features(seq, include_pca_evr)
                     for seq in data])
