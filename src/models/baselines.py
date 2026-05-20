import numpy as np
from numba import njit

from src.config import KNN_K


# ==============================================================================
# 4.  BASELINE METHODS — DTW & Edit Distance
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
