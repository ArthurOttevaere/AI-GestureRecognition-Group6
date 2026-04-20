# Gesture Recognition – MLSM2154 Artificial Intelligence

**Authors:** Andry Lenny / El Mohcine Mohamed / Ottevaere Arthur  
**Group:** 6 | **Year:** 2026

---

## Description

This project implements a gesture recognition pipeline on two 3D hand-gesture datasets (Domain 1 and Domain 4). It evaluates four classification methods across two cross-validation settings, with an ablation study quantifying the contribution of each preprocessing step.

**Methods:**

- **Edit Distance** (k-NN on k-means cluster sequences, k=20)
- **DTW** – Dynamic Time Warping (k-NN on standardised 3D sequences)
- **Random Forest** (32 or 35 hand-crafted kinematic features, optionally including per-gesture PCA EVR)
- **LSTM** (deep learning on padded sequences, max length 150)

**Cross-validation settings:**

- **User-independent** – leave-one-user-out (10 folds)
- **User-dependent** – leave-one-sample-out per user (100 folds)

**Ablation study:** 4 methods × 3 preprocessing conditions:

- (a) No preprocessing
- (b) Standardisation only
- (c) Standardisation + per-gesture PCA EVR features (RF only)

Statistical significance is assessed with Wilcoxon signed-rank tests using both Bonferroni correction and Benjamini-Hochberg (BH) FDR correction.

---

## Project Structure

```text
.
├── main.py               # Entry point – runs all phases end-to-end
├── config.py             # Shared constants: data paths, DATA_DIR, hyperparameters
├── data_loading.py       # load_domain1, load_domain4, dataset statistics
├── visualization.py      # Sequence-length box-plots, 3D trajectory plots
├── preprocessing.py      # Standardisation, per-gesture PCA, k-means cluster encoding
├── distance_metrics.py   # DTW and Edit Distance (Numba JIT-compiled)
├── classifiers.py        # Generic k-NN predictor
├── features.py           # 32/35-feature extraction, build_feature_dataset
├── random_forest.py      # Random Forest evaluation (UI & UD)
├── lstm_model.py         # LSTM model factory and evaluation (UI & UD)
├── crossvalidation.py    # CV loops for distance-based methods
├── ablation.py           # Ablation study: 4 methods × 3 preprocessing conditions
├── evaluation.py         # Confusion matrix generation for all methods
├── results.py            # CSV export, Wilcoxon tests, p-value tables (Bonf. + BH)
├── data/                 # Generated outputs (PNGs, CSVs) – auto-created
├── GestureData_Mons/     # Raw gesture data (not included in repo)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

All generated files (plots and CSV results) are saved to the `data/` folder.

### Expected outputs in `data/`

| File | Description |
| ---- | ----------- |
| `d1_sequence_lengths.png` | Sequence length box-plot – Domain 1 |
| `d1_gesture_samples.png` | 3D trajectory samples – Domain 1 |
| `d4_sequence_lengths.png` | Sequence length box-plot – Domain 4 |
| `d4_gesture_samples.png` | 3D trajectory samples – Domain 4 |
| `d1_pca_per_gesture.png` | Per-gesture PCA EVR distribution – Domain 1 |
| `d4_pca_per_gesture.png` | Per-gesture PCA EVR distribution – Domain 4 |
| `ablation_domain{1,4}.csv` | Ablation study results |
| `results_domain{1,4}_{user_independent,user_dependent}_{method}.csv` | Per-fold accuracies |
| `p_values_domain{1,4}_user_independent.csv` | Wilcoxon p-value matrices |
| `confusion_*.png` | Confusion matrix for the best method per domain |
| `summary_results.csv` | Final pivot table: mean ± std for all methods |

---

## Preprocessing Design Decisions

**Per-gesture standardisation**  
Each gesture is normalised independently (zero mean, unit std per axis) to remove differences in absolute hand position and scale between users and sessions. Applied per-gesture — not globally — to avoid cross-validation contamination.

**Per-gesture PCA**  
A PCA is fitted on the T time-step points of each individual gesture. Its three explained-variance ratios (EVR) are used as additional features for the RF classifier (ablation condition (c)). Digits (Domain 1) are mostly quasi-planar (EVR[0] high); 3D shapes (Domain 4) spread variance more evenly. This pipeline is entirely independent of the k-means pipeline.

**k-means directly on 3D coordinates**  
As required by the course guidelines, k-means clusters all 3D coordinates pooled across gestures. No global PCA reduction is applied before clustering; the algorithm operates in the original 3D space (raw or standardised depending on the ablation condition).

**No FFT features**  
Spectral coefficients are not comparable across sequences of different lengths (31–314 time steps) without prior resampling and are therefore excluded.

**No sequence length feature**  
Sequence length reflects recording speed rather than gesture shape and could bias user-dependent evaluation.

---

## Data Loading Instructions

The source datasets are not included in this repository due to size constraints. Please download the gesture data from the link below and place it in the specified directory structure. It is very important that the folder "GestureData_Mons" is placed at the same level as `main.py` and the other source files, as the code relies on relative paths to load the data correctly.

Link : [GestureData_Mons](https://drive.google.com/drive/folders/1MaiFY8ldeWOmCKbuWnQTJgMoWAJeK5of?usp=share_link)

Place the gesture data under:

```text
GestureData_Mons/
├── GestureDataDomain1_Mons/Domain1_csv/    ← Subject{S}-{G}-{R}.csv files
└── GestureDataDomain4_Mons/                ← plain-text files (no extension)
```
