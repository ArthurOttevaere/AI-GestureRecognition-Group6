"""
2. EXPLORATORY VISUALISATION
=============================
Plots for gesture sequence lengths and 3D trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa – registers 3D projection


def plot_sequence_lengths(data: list, labels: list,
                          domain_name: str,
                          save_path: str | None = None) -> None:
    """Box-plot of sequence lengths per gesture class."""
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
    ax.set_title(f"{domain_name} – Sequence lengths per gesture class")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_gesture_samples(data: list, labels: list, users: list,
                         domain_name: str, n_classes: int = 4,
                         n_subjects: int = 3,
                         save_path: str | None = None) -> None:
    """
    3D trajectory plots: one row per gesture class, one column per subject.
    Each cell overlays the first 3 repetitions (green = start, red = end).
    """
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

    plt.suptitle(f"{domain_name} – 3D trajectories (green=start, red=end)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
