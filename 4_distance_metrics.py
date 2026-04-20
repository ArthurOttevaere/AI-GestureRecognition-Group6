"""
4. BASELINE DISTANCE METRICS
==============================
JIT-compiled DTW and Edit Distance functions.
"""

import numpy as np
from numba import njit


# ------------------------------------------------------------------------------
# 4a. Dynamic Time Warping (DTW)
# ------------------------------------------------------------------------------

@njit(cache=True)
def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute the DTW distance between two 3D gesture sequences.

    Implements the recurrence from the course slides:

        g(1, 1) = d(r1_n, o1)

        g(i, j) = min {  g(i-1, j)   + d(r_i^n, o_j)    [vertical]
                          g(i-1, j-1) + 2·d(r_i^n, o_j)  [diagonal]
                          g(i,   j-1) + d(r_i^n, o_j) }  [horizontal]

        D(R^n, O) = g(I, J) / (I + J)

    where d(a, b) = ||a – b||_2 is the Euclidean distance and the factor
    ×2 on the diagonal follows the course formulation.

    Parameters
    ----------
    seq1 : np.ndarray of shape (I, 3)
    seq2 : np.ndarray of shape (J, 3)

    Returns
    -------
    Normalised DTW distance (scalar).
    """
    I = len(seq1)
    J = len(seq2)

    g = np.full((I + 1, J + 1), np.inf)
    g[0, 0] = 0.0

    for i in range(1, I + 1):
        for j in range(1, J + 1):
            dx = seq1[i - 1, 0] - seq2[j - 1, 0]
            dy = seq1[i - 1, 1] - seq2[j - 1, 1]
            dz = seq1[i - 1, 2] - seq2[j - 1, 2]
            d  = (dx*dx + dy*dy + dz*dz) ** 0.5

            v    = g[i - 1, j    ] + d
            diag = g[i - 1, j - 1] + 2.0 * d
            h    = g[i,     j - 1] + d
            g[i, j] = min(v, min(diag, h))

    return g[I, J] / (I + J)


# ------------------------------------------------------------------------------
# 4b. Edit Distance (Levenshtein) on symbolic sequences
# ------------------------------------------------------------------------------

@njit(cache=True)
def edit_distance(seq1: np.ndarray, seq2: np.ndarray) -> int:
    """
    Compute the edit distance (Levenshtein distance) between two sequences
    of cluster labels obtained after the k-means encoding step.

    Standard DP recurrence:

        ED(i, 0) = i,   ED(0, j) = j
        ED(i, j) = min {  ED(i-1, j)   + 1              [deletion]
                           ED(i,   j-1) + 1              [insertion]
                           ED(i-1, j-1) + δ(s1_i, s2_j) [substitution]  }

    Parameters
    ----------
    seq1, seq2 : array-like of int – cluster-label sequences

    Returns
    -------
    Integer edit distance.
    """
    L1, L2 = len(seq1), len(seq2)

    mat = np.zeros((L1 + 1, L2 + 1), dtype=np.int64)
    for i in range(L1 + 1):
        mat[i, 0] = i
    for j in range(L2 + 1):
        mat[0, j] = j

    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            delta  = 0 if seq1[i - 1] == seq2[j - 1] else 1
            a = mat[i - 1, j    ] + 1
            b = mat[i,     j - 1] + 1
            c = mat[i - 1, j - 1] + delta
            mat[i, j] = min(a, min(b, c))

    return mat[L1, L2]
