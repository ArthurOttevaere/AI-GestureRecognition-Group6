# Results Snapshot

This folder contains a frozen snapshot of the outputs produced by the pipeline (see [`methodology.md`](../methodology.md)) on its last full run. It mirrors the structure of `Outputs/` but is committed to the repository so that results are reproducible without re-running the pipeline.

## Structure

```
Results_Snapshot/
├── figures/
│   ├── exploratory/          # Gesture sample plots and sequence-length distributions (Domain 1 & 4)
│   ├── pca_denoising/        # Before/after PCA denoising visualisations
│   ├── validation_curves/    # Hyperparameter sensitivity curves (K_CLUSTERS, KNN_K)
│   ├── feature_importance/   # Random Forest permutation-importance bar charts
│   ├── confusion_matrices/   # Per-method confusion matrices (user-independent)
│   ├── learning_curves/      # Train vs. validation score as a function of training set size
│   └── statistical_tests/    # Benjamini-Hochberg corrected p-value heatmaps
├── tables/
│   ├── fold_results/         # Per-fold accuracy for every method × domain × evaluation setting
│   ├── ablation/             # Ablation study: preprocessing conditions (raw / standardised / PCA)
│   ├── feature_selection/    # Selected feature subsets per method, domain, and evaluation setting
│   └── overfitting/          # Train-vs-test accuracy gap for parametric classifiers (DT, RF, LR)
└── logs/
    └── run_*.txt             # Full console log of the pipeline run that produced this snapshot
```

## Naming conventions

| Token | Meaning |
|-------|---------|
| `d1` / `d4` | Domain 1 / Domain 4 |
| `ui` / `ud` | User-independent (LOSO) / User-dependent (LOSO per user) |
| `rf` / `dt` / `lr` / `svm` | Random Forest / Decision Tree / Logistic Regression / SVM |
| `dtw` / `edit` / `dollar` | DTW 1-NN / Edit-distance 1-NN / $1 Recognizer 3D |

## Regenerating results

Re-run the full pipeline from the project root:

```bash
python main.py
```

Outputs will be written to `Outputs/`. Copy them here manually if you want to update the snapshot.
