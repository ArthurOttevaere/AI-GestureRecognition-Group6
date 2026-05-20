import numpy as np
import pandas as pd


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
