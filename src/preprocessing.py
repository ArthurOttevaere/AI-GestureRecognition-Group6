import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.config import PCA_N_KEEP, K_CLUSTERS


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
                              save_path: str | None = None,
                              show: bool = False) -> None:
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
    if show:
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
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(all_train_points)
    centroids = kmeans.cluster_centers_

    train_seq = encode_with_centroids(train_data, centroids)
    test_seq  = encode_with_centroids(test_data,  centroids)
    return train_seq, test_seq
