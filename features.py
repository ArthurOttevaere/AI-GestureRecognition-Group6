"""
6. FEATURE EXTRACTION
======================
Statistical and kinematic feature extraction from 3D gesture trajectories.
"""

import numpy as np
import pandas as pd


def extract_features(sequence: np.ndarray) -> np.ndarray:
    """
    Extract rich statistical and kinematic features from a 3D trajectory.

    Features
    --------
    Per axis (x, y, z) — 7 × 3 = 21:
        mean, std, min, max, range, skewness, kurtosis
    Velocity (first differences) — 4:
        mean speed, std speed, max speed, min speed
    Acceleration (second differences) — 2:
        mean magnitude, std magnitude
    Global geometry — 5:
        total path length, bounding-box sides (×3), bounding-box diagonal
    Sequence length — 1
    Spectral per axis — 3 × 3 = 9:
        mean, std, max of FFT magnitude coefficients

    Total : 21 + 4 + 2 + 5 + 1 + 9 = 42 features
    """
    features = []

    # ---- Per-axis statistics ----
    for i in range(3):
        axis = sequence[:, i]
        features.extend([
            np.mean(axis),
            np.std(axis),
            np.min(axis),
            np.max(axis),
            np.max(axis) - np.min(axis),   # range
            pd.Series(axis).skew(),         # skewness
            pd.Series(axis).kurt(),         # kurtosis
        ])

    # ---- Velocity ----
    velocity = np.diff(sequence, axis=0)         # (T-1, 3)
    speed    = np.linalg.norm(velocity, axis=1)  # (T-1,)
    features.extend([np.mean(speed), np.std(speed),
                     np.max(speed),  np.min(speed)])

    # ---- Acceleration ----
    if len(velocity) > 1:
        accel      = np.diff(velocity, axis=0)
        accel_norm = np.linalg.norm(accel, axis=1)
        features.extend([np.mean(accel_norm), np.std(accel_norm)])
    else:
        features.extend([0.0, 0.0])

    # ---- Path length ----
    features.append(float(np.sum(speed)))

    # ---- Bounding box ----
    bbox = np.max(sequence, axis=0) - np.min(sequence, axis=0)
    features.extend(bbox.tolist())
    features.append(float(np.linalg.norm(bbox)))

    # ---- Sequence length ----
    features.append(float(len(sequence)))

    # ---- Spectral energy ----
    for i in range(3):
        fft_mag = np.abs(np.fft.rfft(sequence[:, i]))
        features.extend([float(np.mean(fft_mag)),
                         float(np.std(fft_mag)),
                         float(np.max(fft_mag))])

    return np.array(features, dtype=float)


def build_feature_dataset(data: list) -> np.ndarray:
    """Convert a list of sequences into a feature matrix (N × 42)."""
    return np.array([extract_features(seq) for seq in data])
