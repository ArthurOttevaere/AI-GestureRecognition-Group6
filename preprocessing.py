"""
3. PRE-PROCESSING
==================
Standardisation, per-gesture PCA, and k-means cluster encoding.

Design decisions
----------------
Per-gesture PCA
    A PCA is fitted independently on the T time-step points of each
    individual gesture.  Its three explained-variance ratios (EVR) are
    used as additional features for the Random Forest classifier.
    This is entirely separate from the k-means pipeline.

k-means directly on 3D coordinates
    The course guidelines state: "gathering every 3D coordinate of every
    recorded gesture of the training set in one big set of 3D coordinates,
    on which you are going to apply a clustering."  No global PCA reduction
    is applied before k-means; clustering operates in the original 3D space
    (raw or standardised depending on the ablation condition).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from config import DATA_DIR


def standardize_gestures(data: list) -> list:
    """
    Standardise each gesture independently: zero mean, unit std per axis.

    Applied per gesture (not globally) so that cross-validation folds
    are never contaminated - the standardisation parameters of a test
    gesture depend only on that gesture's own time steps.

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
        std[std == 0] = 1.0          # avoid division by zero
        standardized.append((seq - mean) / std)
    return standardized


def pca_per_gesture(sequence: np.ndarray,
                    n_components: int = 3) -> dict:
    """
    Fit a PCA on the T time-step points of a SINGLE gesture sequence.

    This captures the intrinsic geometry of that individual trajectory:
    - Digits (Domain 1) are mostly quasi-planar -> EVR[0] is high.
    - 3D shapes (Domain 4) spread variance more evenly across all axes.

    The three EVR values are used as additional features for the Random
    Forest classifier (ablation condition (c)).  This pipeline is
    completely independent of the k-means pipeline used for Edit Distance.

    Parameters
    ----------
    sequence     : np.ndarray (T, 3)
    n_components : components to keep (fixed at 3 for consistency)

    Returns
    -------
    dict:
        'evr'        : explained variance ratios  - (n_components,)
        'components' : principal axes             - (n_components, 3)
        'projected'  : gesture in PCA space       - (T, n_components)
        'planarity'  : EVR[0] scalar
    """
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(sequence)
    return {
        "evr"       : pca.explained_variance_ratio_,
        "components": pca.components_,
        "projected" : projected,
        "planarity" : float(pca.explained_variance_ratio_[0]),
    }


def summarise_per_gesture_pca(data: list, domain_name: str,
                               save_path: str | None = None) -> None:
    """
    Report and plot the distribution of per-gesture EVR values.

    Called on the standardised dataset.  Standardisation (per axis,
    per gesture) does not alter the relative covariance structure of
    a single gesture sequence, so EVR values computed on standardised
    data faithfully reflect the geometric planarity of each trajectory.
    """
    evrs = np.array([pca_per_gesture(seq)["evr"] for seq in data])

    print(f"\n  Per-gesture PCA - {domain_name}")
    for c in range(evrs.shape[1]):
        print(f"    PC{c+1}: mean EVR = {evrs[:, c].mean():.3f} "
              f"+/- {evrs[:, c].std():.3f}")
    quasi_planar_pct = np.mean(evrs[:, 0] > 0.85) * 100
    print(f"    Quasi-planar gestures (EVR[0] > 0.85): "
          f"{quasi_planar_pct:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 3))
    for c in range(evrs.shape[1]):
        ax.hist(evrs[:, c], bins=30, alpha=0.6, label=f"PC{c+1}")
    ax.set_xlabel("Explained variance ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"{domain_name} - Per-gesture PCA EVR distribution")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def cluster_and_encode(data_3d: list, k: int = 20,
                       random_state: int = 42) -> tuple[list, KMeans]:
    """
    Fit k-means on ALL 3D time-step points pooled across all gestures,
    then convert each gesture to a sequence of integer cluster labels.

    This implements the course guidelines exactly:
        "gathering every 3D coordinate of every recorded gesture of the
         training set in one big set of 3D coordinates, on which you are
         going to apply a clustering (such as a k-means) and record the
         coordinates of the centroid of each cluster. Then, for each
         gesture, you will take each spatial 3D point and replace it with
         the label of the closest centroid."

    No global PCA reduction is applied before k-means; clustering operates
    directly in the original 3D space (raw or standardised).

    Parameters
    ----------
    data_3d      : list of np.ndarray (T, 3)
    k            : number of clusters / alphabet size
    random_state : random seed for reproducibility

    Returns
    -------
    sequences : list of np.ndarray[int] - cluster-label sequences
    kmeans    : fitted KMeans object
    """
    all_points = np.vstack(data_3d)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    kmeans.fit(all_points)
    sequences = [kmeans.predict(seq) for seq in data_3d]
    return sequences, kmeans
