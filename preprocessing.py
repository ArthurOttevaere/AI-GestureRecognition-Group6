"""
3. PRE-PROCESSING
==================
Standardisation, PCA projection, and k-means cluster encoding.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def standardize_gestures(data: list) -> list:
    """
    Standardise each gesture independently so that each spatial dimension
    has zero mean and unit standard deviation over the gesture's time steps.

    This removes differences in absolute hand position and scale between
    users and recording sessions.  Standardisation is performed per gesture
    (not globally) so that cross-validation folds are not contaminated.

    Parameters
    ----------
    data : list of np.ndarray of shape (T, 3)

    Returns
    -------
    standardized : list of np.ndarray of shape (T, 3)
    """
    standardized = []
    for seq in data:
        mean = np.mean(seq, axis=0)
        std  = np.std(seq,  axis=0)
        std[std == 0] = 1            # avoid division by zero
        standardized.append((seq - mean) / std)
    return standardized


def apply_pca(data: list, n_components: int = 2) -> tuple[list, PCA]:
    """
    Apply PCA to all time-step points across all gestures, then project
    each gesture into the PCA space.

    As permitted by the course guidelines, PCA is fitted on the full dataset
    (train + test pooled) before the cross-validation loop.  This is mentioned
    explicitly in the report.

    Parameters
    ----------
    data         : list of np.ndarray of shape (T, 3)
    n_components : number of principal components to keep (default 2)

    Returns
    -------
    data_pca : list of np.ndarray of shape (T, n_components)
    pca      : fitted sklearn PCA object
    """
    all_points = np.vstack(data)   # shape (N_total_points, 3)

    pca = PCA(n_components=n_components)
    pca.fit(all_points)

    explained = pca.explained_variance_ratio_
    print(f"  PCA ({n_components} components): explained variance = "
          f"{explained.round(3)} | total = {explained.sum():.3f}")

    data_pca = [pca.transform(seq) for seq in data]
    return data_pca, pca


def cluster_and_encode(data_pca: list, k: int = 20,
                       random_state: int = 42) -> tuple[list, KMeans]:
    """
    Fit a k-means clustering on all projected time-step points, then convert
    each gesture from a sequence of PCA coordinates to a sequence of cluster
    labels (integers in 0..k-1).

    As permitted by the course guidelines, clustering is performed once on the
    full dataset before cross-validation.  This is mentioned in the report.

    Parameters
    ----------
    data_pca     : list of np.ndarray of shape (T, n_components)
    k            : number of clusters / symbols
    random_state : random seed for reproducibility

    Returns
    -------
    sequences : list of np.ndarray of int – cluster-label sequences
    kmeans    : fitted KMeans object
    """
    all_points = np.vstack(data_pca)   # shape (N_total, n_components)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    kmeans.fit(all_points)

    sequences = [kmeans.predict(seq) for seq in data_pca]
    return sequences, kmeans
