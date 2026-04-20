"""
5. k-NN CLASSIFIER
===================
Generic k-Nearest-Neighbour predictor used with any distance function.
"""

import numpy as np


def knn_predict(test_item, train_items: list, train_labels: list,
                distance_fn, k: int = 1) -> int:
    """
    Generic k-Nearest-Neighbour classifier.

    Computes the distance between `test_item` and every item in `train_items`
    using `distance_fn`, then returns the majority label among the k nearest
    neighbours (ties broken by the smallest label).

    Parameters
    ----------
    test_item    : query sequence (np.ndarray or list)
    train_items  : list of training sequences
    train_labels : list of int
    distance_fn  : callable(a, b) -> float
    k            : number of neighbours

    Returns
    -------
    Predicted integer label.
    """
    distances = np.array([distance_fn(test_item, ref) for ref in train_items])
    k_nearest = np.argsort(distances)[:k]
    k_labels  = [train_labels[idx] for idx in k_nearest]
    return max(set(k_labels), key=k_labels.count)
