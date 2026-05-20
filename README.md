# Gesture Recognition — MLSM2154 Artificial Intelligence

**Group 6** · UCLouvain FUCaM Mons  
**Authors:** Andry Lenny · El Mohcine Mohamed · Ottevaere Arthur  
**Year:** 2026

---

## Overview

This project implements a full machine-learning pipeline for **3D accelerometer gesture recognition**, developed as part of the MLSM2154 Artificial Intelligence course. Given sequences of 3-axis accelerometer data recorded from 10 users performing 10 gesture classes, the pipeline compares six classification methods under rigorous cross-validation, an ablation study over three preprocessing conditions, statistical significance testing, and an overfitting diagnostic.

The pipeline runs end-to-end from raw CSV data to publication-ready figures, tables, and a console log — in a single command.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Interactive Menu](#interactive-menu)
5. [CLI Flags](#cli-flags)
6. [Pipeline Architecture](#pipeline-architecture)
7. [Methods Compared](#methods-compared)
8. [Outputs](#outputs)
9. [Datasets](#datasets)
10. [Documents & Resources](#documents--resources)

---

## Project Structure

```
AI-GestureRecognition-Group6/
│
├── main.py                        # Entry point — interactive menu + CLI
├── requirements.txt               # Python dependencies
├── methodology.md                 # Detailed pipeline methodology & scientific decisions
│
├── src/                           # Source modules
│   ├── config.py                  # All hyperparameters and output paths
│   ├── data_loader.py             # CSV ingestion for Domain 1 and Domain 4
│   ├── preprocessing.py           # Standardisation + PCA denoising
│   ├── features.py                # Feature extraction (geometric descriptors)
│   ├── utils.py                   # Logger and shared utilities
│   ├── models/
│   │   ├── baselines.py           # DTW and Edit Distance (Numba-JIT, 1-NN)
│   │   ├── dollar.py              # $1 Recognizer 3D (Kratz & Rohs 2010)
│   │   └── parametric.py         # Decision Tree, Random Forest, Logistic Regression
│   └── evaluation/
│       ├── crossval.py            # Fold generation (UI / UD) + validation curves
│       ├── ablation.py            # Ablation study (6 methods × 3 conditions × UI/UD)
│       ├── metrics.py             # Visualisations (confusion matrices, learning curves)
│       └── stats.py               # Wilcoxon tests + Benjamini-Hochberg correction
│
├── GestureData_Mons/              # Raw datasets (not committed — see Datasets)
│   ├── GestureDataDomain1_Mons/
│   └── GestureDataDomain4_Mons/
│
├── Outputs/                       # Generated artefacts (gitignored)
│   ├── figures/                   # PNG figures
│   ├── tables/                    # CSV tables
│   └── logs/                      # Full run console logs
│
└── Results_Snapshot/              # Frozen snapshot of the last full run (committed)
    └── README.md                  # Snapshot structure and naming conventions
```

---

## Installation

**Prerequisites:** Python 3.10+ and `pip`.

```bash
# Clone the repository
git clone <repo-url>
cd AI-GestureRecognition-Group6

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

> **macOS Apple Silicon note:** `tensorflow-macos` and `tensorflow-metal` are listed in `requirements.txt`. Install them only if needed; the core pipeline does not require TensorFlow.

---

## Quick Start

```bash
# Interactive menu (recommended for first run)
python main.py

# Headless — Domain 1 only, save figures to Outputs/
python main.py --domain 1

# Headless — Domain 4 only
python main.py --domain 4

# Both domains with figures displayed on screen
python main.py --show-plots

# Domain 1, headless + display figures
python main.py --domain 1 --show-plots
```

All outputs are written to `Outputs/`. A full console log is saved to `Outputs/logs/run_<timestamp>.txt`.

---

## Interactive Menu

When `main.py` is run without flags, it presents a two-step interactive menu:

```
============================================================
  MLSM2154 — Gesture Recognition Pipeline  |  Group 6
============================================================

  This pipeline runs the full ML experiment:
    • Data loading & exploratory visualisation
    • Standardisation + PCA denoising
    • Baseline methods  (DTW, Edit Distance)
    • Advanced methods  (Decision Tree, Random Forest, $1 3D and LR)
    • Cross-validation  (user-independent + user-dependent)
    • Ablation study, statistical tests, confusion matrices
    • Overfitting diagnostic & learning curves

  Available domains:
    [1]  Domain 1  (accelerometer gestures, 10 users)
    [4]  Domain 4  (accelerometer gestures, 10 users)

============================================================
  Which domain(s) do you want to run?

    [1]  Domain 1 only
    [4]  Domain 4 only
    [b]  Both domains  (default)

  Your choice [1 / 4 / b]: _
```

Then:

```
============================================================
  Display figures interactively on screen?

    [y]  Yes — open each figure in a window
    [n]  No  — save to Outputs/ only  (default, headless)

  Your choice [y / n]: _
```

---

## CLI Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--domain` | `1`, `4`, `both` | interactive | Domain(s) to process |
| `--show-plots` | — | `False` | Open figures in a window while running |

---

## Pipeline Architecture

The pipeline runs eight sequential phases on each selected domain:

```
Phase 1  ──  Data loading & exploratory analysis
              └─ CSV ingestion, dataset statistics, gesture sample plots

Phase 2  ──  Pre-processing
              ├─ Per-gesture standardisation (zero mean, unit std per axis)
              ├─ Per-gesture PCA denoising (3D → 2D → 3D)
              └─ k-means clustering inside each CV fold (no leakage)

Phase 3  ──  Baseline methods
              ├─ DTW 1-NN (Numba-JIT)
              └─ Edit Distance 1-NN (k-means quantisation, Numba-JIT)

Phase 4  ──  Advanced methods
              ├─ Decision Tree  (GridSearchCV, 5-fold inner CV)
              ├─ Random Forest  (GridSearchCV + per-fold permutation importance)
              ├─ Logistic Regression  (multinomial L2, GridSearchCV)
              └─ $1 Recognizer 3D  (Kratz & Rohs 2010, GSS, raw data only)

Phase 5  ──  Cross-validation
              ├─ User-independent  (leave-one-user-out, 10 folds)
              └─ User-dependent    (leave-one-sample-out, 10 folds)
              └─ Ablation: 6 methods × 3 preprocessing conditions × UI & UD

Phase 6  ──  Hyperparameter validation curves
              ├─ K_CLUSTERS sensitivity (selected K feeds back into the pipeline)
              └─ KNN_K curve (informative only; k = 1 enforced)

Phase 7  ──  Statistical tests  (UI setting)
              ├─ Paired Wilcoxon signed-rank test (n = 100 pairs per method)
              └─ Benjamini-Hochberg FDR correction (6×6 pairwise matrix)

Phase 8  ──  Overfitting diagnostic
              ├─ Per-fold train-vs-test accuracy gap (DT / RF / LR)
              └─ Learning curves (sklearn.model_selection.learning_curve)
```

> Full scientific justification for every design decision is documented in [methodology.md](methodology.md).

---

## Methods Compared

| ID | Method | Type | Preprocessing |
|----|--------|------|---------------|
| `DTW` | Dynamic Time Warping 1-NN | Non-parametric baseline | Raw / Standardised / PCA |
| `Edit` | Edit Distance 1-NN | Non-parametric baseline | Raw / Standardised / PCA |
| `$1` | $1 Recognizer 3D (Kratz & Rohs 2010) | Template matching | Raw only |
| `DT` | Decision Tree (CART) | Parametric | Best per fold |
| `RF` | Random Forest (Breiman 2001) | Parametric | Best per fold |
| `LR` | Logistic Regression (multinomial L2) | Parametric (linear) | Best per fold |

All methods are evaluated under two cross-validation protocols:

- **User-Independent (UI):** leave-one-user-out — tests generalisation to unseen users.
- **User-Dependent (UD):** leave-one-sample-out — tests recognition for known users.

---

## Outputs

After a run, all artefacts are written to `Outputs/`:

```
Outputs/
├── figures/
│   ├── exploratory/          # Gesture samples, sequence-length distributions
│   ├── pca_denoising/        # Before/after PCA denoising
│   ├── validation_curves/    # K_CLUSTERS and KNN_K sensitivity curves
│   ├── confusion_matrices/   # Best-model confusion matrices
│   ├── learning_curves/      # Train vs. validation score (DT / RF / LR)
│   └── statistical_tests/    # BH-corrected p-value heatmaps
├── tables/
│   ├── ablation/             # 6 methods × 3 conditions × UI & UD
│   ├── fold_results/         # Per-fold accuracy (all methods and domains)
│   ├── feature_selection/    # Selected features per fold
│   ├── overfitting/          # Train-test accuracy gap (DT / RF / LR)
│   ├── statistical_tests/    # Wilcoxon pairwise p-value matrices
│   └── summary/              # Aggregated accuracy (mean ± std)
└── logs/
    └── run_<timestamp>.txt   # Full console log
```

A frozen snapshot of the last full run is committed under [Results_Snapshot/](Results_Snapshot/) — see its [README](Results_Snapshot/README.md) for the naming conventions (`d1`/`d4`, `ui`/`ud`, etc.).

---

## Datasets

The raw data lives in `GestureData_Mons/` (not committed to this repository due to size):

| Domain | Folder | Format | Subjects | Gestures | Samples |
|--------|--------|--------|----------|----------|---------|
| Domain 1 | `GestureDataDomain1_Mons/Domain1_csv/` | CSV, one file per sample | 10 | 10 | ~1 000 |
| Domain 4 | `GestureDataDomain4_Mons/` | CSV, labelled shapes | 10 | 10 | ~1 000 |

Domain 4 gesture classes: Cuboid, Cylinder, Sphere, Rectangular Pipe, Hemisphere, Cylindrical Pipe, Pyramid, Tetrahedron, Cone, Toroid.

**Download the datasets** from the shared Google Drive folder and place it under in the root of the repository:

> [Google Drive — GestureData\_Mons](<https://drive.google.com/drive/folders/1OSOAVIz3M9xSPU-pIogUpd7Wr3gsnYnD?usp=share_link>)

Place the dataset folders at the above paths before running the pipeline.

---

## Documents & Resources

| File | Description |
|------|-------------|
| [methodology.md](methodology.md) | Full pipeline methodology, 16 documented scientific decisions, and all references |
| [Results_Snapshot/README.md](Results_Snapshot/README.md) | Structure and naming conventions for the committed results snapshot |
| [Resources/Documentation_pipeline.pdf](Resources/Documentation_pipeline.pdf) | Pipeline documentation (PDF) |
| [Resources/pipeline_explained.tex](Resources/pipeline_explained.tex) | LaTeX source — pipeline explanation |
| [Resources/results_synthesis.tex](Resources/results_synthesis.tex) | LaTeX source — results synthesis |
| [Resources/MODIFICATIONS_CODE_IA.md](Resources/MODIFICATIONS_CODE_IA.md) | Changelog of code modifications |

---

*MLSM2154 — Artificial Intelligence · Group 6 · 2026*
