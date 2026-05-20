# MLSM2154 — Artificial Intelligence: Gesture Recognition Project

## Pipeline Overview

1. **Phase 1** — Data loading & exploratory analysis
2. **Phase 2** — Pre-processing
   - Per-gesture standardisation (zero mean, unit std per axis)
   - Per-gesture PCA denoising (3D → 2D → 3D):
     - PCA fitted on the T time-step points of each gesture
     - Projection onto the 2 principal components (2D)
     - Back-projection into the original 3D space
     - The 3 EVR values are also stored as additional RF features
   - k-means clustering INSIDE each CV fold (rigorous):
     - Fitted on training-set 3D points only per fold
     - Centroids applied to encode both train and test sequences
3. **Phase 3** — Baseline methods (DTW + Edit Distance with k-means quantization)
   - 1-NN classifier for both (KNN_K = 1 enforced; see Major decision #8)
4. **Phase 4** — Advanced methods
   - Decision Tree (Quinlan 1986, Breiman et al. 1984 CART) with GridSearchCV (nested CV, 5-fold inner). Pedagogical baseline to demonstrate empirically the bagging gain of Random Forest.
   - Random Forest (Breiman 2001) with GridSearchCV (5-fold inner) + per-fold permutation-importance feature selection (Strobl et al. 2007; Guyon & Elisseeff 2003).
   - Logistic Regression (Cox 1958; Hosmer et al. 2013), multinomial with L2 penalty, GridSearchCV over C (5-fold inner). Linear baseline to bound the difficulty of the task.
   - $1 Recognizer 3D (Kratz & Rohs, 2010) with Rodrigues rotation, uniform cube scaling, confidence score, and N-best list. EVALUATED ON RAW DATA ONLY (Wobbrock 2007 prescription).
5. **Phase 5** — Cross-validation
   - User-independent: leave-one-user-out (10 folds)
   - User-dependent: leave-one-sample-out (10 folds)
   - Ablation study: 6 methods × 3 preprocessing conditions, run in BOTH UI AND UD (Iteration 2 fix: best preproc may differ)
6. **Phase 6** — Hyperparameter validation curves (empirical iterative selection)
   - K_CLUSTERS optimum from sensitivity analysis is USED in the rest of the pipeline
   - KNN_K is FORCED to 1 (1-NN) regardless of the validation curve outcome (see Major decision #8)
7. **Phase 7** — Statistical tests (User-independent setting only)
   - Paired Wilcoxon signed-rank test on n=100 paired observations (10 gestures × 10 users), one accuracy per (gesture, user) pair for each method
   - Benjamini-Hochberg FDR correction (BH)
   - Pairwise p-value matrix (6×6 = 15 pairs) saved as CSV + heatmap
8. **Phase 8** — Overfitting diagnostic
   - Per-fold train-acc vs test-acc gap for DT/RF/LR (parametric classifiers), run in both UI and UD. DTW/Edit/$1 are 1-NN, so train accuracy is 1.0 by construction and not reported.
   - `sklearn.model_selection.learning_curve` for DT/RF/LR on each domain (UI setting only): train score and validation score as a function of training set size. Wide gap = high variance (overfitting). Reference: Hastie et al. (2009) §7.10.

## Outputs

Outputs are organised under `Outputs/` in nested subfolders by type:

```
figures/{exploratory, pca_denoising, validation_curves,
         feature_importance, confusion_matrices, statistical_tests}
tables/{ablation, fold_results, statistical_tests, summary}
logs/
```

## Major Scientific Decisions

1. **$1 Recognizer — 3D adaptation following Kratz & Rohs (2010).** The original $1 algorithm of Wobbrock, Wilson & Li (2007) is purely 2D and provides no empirical validation in 3D. We follow the canonical 3D extension of Kratz & Rohs (2010) "A $3 Gesture Recognizer", IUI'10:
   - Step 2: rotation around the axis defined by the cross product p̂ × c, where c is the centroid and p̂ is the first resampled point. The angle is the arccos of the normalised dot product. Rotation applied via Rodrigues' formula.
   - Step 3: scaling INSIDE a normalised cube of side l (uniform rescaling), avoiding axis-by-axis scaling and the division-by-zero issue for quasi-planar gestures.
   - Score: S = 1 − d / (0.5 × √3 × l²), where d is the MSE (mean squared point-to-point distance) after GSS alignment. Using MSE is required by Kratz's scoring formula; the denominator is 0.5×√3×l².
   - Templates are preprocessed only ONCE and cached, as specified by Wobbrock et al. (2007).
   - Recognize returns a sorted N-best list, allowing kNN with k>1.
   - Golden Section Search (GSS) over the 3 rotation axes (Kratz & Rohs 2010, "Search for Minimum Distance at Best Angle") is implemented via Numba-JIT functions (`_dollar_gss_mse`): φ = 0.5×(√5−1), cutoff = 2°, 11 iterations per axis.
   - Scoring heuristic (Kratz & Rohs 2010): top-3 check with epsilon thresholds. `allow_rejection=False` (default) forces 1-Best for CV comparability with RF/LR/DT.

2. **Wilcoxon n=100.** For each method, a 100-vector of accuracies is built, one per (gesture, user) pair. Pairs are then compared method-vs-method via `scipy.stats.wilcoxon` (signed-rank). The all-zero-diff degenerate case is caught and returns p=1.0 with a warning.

3. **Benjamini-Hochberg FDR correction only (no Bonferroni).** Bonferroni is removed because the pairwise Wilcoxon tests share methods across pairs and are therefore not independent — BH is the correct procedure under positive dependence (Benjamini & Hochberg 1995, §4). The earlier permutation test and Bayesian sign test are also removed (redundant with Wilcoxon + BH).

4. **LSTM removed.** The dataset is too small (~1000 samples) to justify a recurrent network. Earlier results were retained from prior versions and are no longer relevant.

5. **Random Forest hyperparameters selected per fold via GridSearchCV** (5-fold inner CV). Feature selection done per-fold using permutation importance (Breiman 2001; Strobl et al. 2007) with a 95% cumulative threshold (Guyon & Elisseeff 2003). NO leakage: importance is fit on each outer-fold's training set only.

6. **Hyperparameter tuning asymmetry.** RF, Decision Tree, and Logistic Regression hyperparameters are selected per-fold via GridSearchCV (nested CV, inner CV = 5 folds; Varma & Simon 2006; Cawley & Talbot 2010). DTW and Edit Distance hyperparameters (k-means K) are selected once via empirical validation curves on the full user-independent CV, then kept fixed for evaluation. This asymmetry is intentional: DTW and Edit Distance have at most 2 scalar hyperparameters with a clear plateau; tree- and kernel-based classifiers have a combinatorial grid that requires per-fold tuning to avoid overfitting. The comparison is therefore between best-configured versions of each method, not between methods sharing an identical tuning protocol. This limitation is acknowledged in the report.

7. **K-clustering selected via two empirical validation curves per domain** (Edit Distance accuracy vs K on user-independent CV):
   - On standardised data (`data_std`) → best K used for conditions (b) and (c) of the ablation study
   - On raw data (`data_raw`) → best K used for condition (a) only
   - Rationale: k-means distance scales differ between raw and standardised spaces (Linde et al. 1980, VQ theory), so the optimal codebook size may not transfer across preprocessing conditions.

8. **KNN_K = 1 ENFORCED.** A validation curve scanning k in {1,3,5,7,9} is produced for transparency (saved to `Outputs/figures/validation_curves`) but the rest of the pipeline ALWAYS uses k = 1. This is consistent with the gesture recognition literature where 1-NN is the canonical baseline:
   - Wobbrock et al. (2007) report results with the single best template ($1 design is implicitly 1-NN)
   - Mezari & Maglogiannis (2018) use 1-NN for their commodity-device recognizer
   - Liu et al. (2009) uFlash baseline uses 1-NN
   - Mitra & Acharya (2007) gesture recognition survey discusses 1-NN as the standard non-parametric baseline
   - 1-NN gives a transparent, parameter-free baseline that is robust to outliers being broken by majority voting with small k.

9. **Decision Tree included as pedagogical control.** Comparing DT (single tree) to RF (bag of trees) empirically demonstrates the value of bagging on this dataset (Breiman 2001, Hastie et al. 2009, ch. 15).

10. **Logistic Regression** (multinomial, L2 penalty, lbfgs solver). Linear classifier baseline. References: Cox (1958); Hosmer, Lemeshow & Sturdivant (2013); Hastie, Tibshirani & Friedman (2009) §4.4. Acts as a linear baseline reference: a linear model on the same feature vector.

12. **$1 Recognizer evaluated on RAW data ONLY** (preprocessing condition (a)). Per Wobbrock et al. (2007) and Kratz & Rohs (2010), the $1 pipeline includes its own internal normalisation (resample, centroid translation, indicative-axis rotation, uniform cube scaling). Feeding externally-standardised data into $1 alters the geometry used by the cross-product rotation step and is methodologically inconsistent. The ablation table still reports the three conditions for transparency but the main UI/UD evaluation forces (a).

13. **Ablation study is run in BOTH user-independent (UI) and user-dependent (UD) settings**, because the best preprocessing for UI is not guaranteed to be the best for UD. Each setting picks its own optimum per method.

14. **Per-fold three-stage feature selection** (replaces previous post-hoc Gini selection):
    - (a) Variance filter: remove near-constant features (var < 1e-10)
    - (b) Correlation-based redundancy removal: for each pair with |Pearson r| > 0.99, discard the lower-variance member (adapted from Yu & Liu 2004, JMLR). Without this step, highly correlated features (e.g. x_range = x_max − x_min; bbox_diag derived from bbox_*) dilute each other's unconditional permutation importance (Strobl et al. 2008, BMC Bioinformatics). This dilution causes the cumulative-importance cut to drop entire correlated clusters, destroying classifier performance. Note on threshold: While traditional filters often use stricter cutoffs (e.g. 0.95), we empirically relaxed this threshold to 0.99. This engineering decision preserves a richer pool of geometric descriptors (~15–30 features per fold), preventing underfitting and ensuring the Random Forest retains enough dimensions to effectively model complex non-linear interactions, thereby restoring its theoretical advantage over linear models (LR).
    - (c) Permutation importance on a 20%-held-out validation split (not on the training data itself, which would bias importance toward memorised features; Breiman 2001; Strobl et al. 2007).
    - Selection is fit on each outer-fold's training set only → no leakage (Ambroise & McLachlan 2002, PNAS).

15. **Feature standardisation for LR** via `StandardScaler` inside a sklearn `Pipeline`, fitted on each inner-CV training split only. LR (lbfgs solver): gradient magnitudes are scale-dependent; scaling ensures balanced gradient steps and reliable convergence. `max_iter` increased from 2000 to 5000 to guarantee convergence across all C values in the grid (sklearn ConvergenceWarning fix).

16. **Overfitting diagnostic.** For each parametric classifier (DT, RF, LR) we record train and test accuracy per fold (both UI and UD); the average gap train-test is reported. sklearn `learning_curve` is run on each domain (UI setting only) producing train/validation score curves as a function of training-set size (Hastie et al. 2009 §7.10; Domingos 2012).

## References

Ambroise, C., & McLachlan, G. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS*, 99 (10), 6562–6566.

Breiman, L. (2001). Random forests. *Machine Learning*, 45 (1), 5–32.

Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and regression trees (CART)*. Wadsworth.

Cawley, G., & Talbot, N. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079–2107.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273–297.

Cox, D. R. (1958). The regression analysis of binary sequences. *J. Royal Statistical Society B*, 20 (2), 215–242.

Domingos, P. (2012). A few useful things to know about machine learning. *Communications of the ACM*, 55 (10), 78–87.

Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *JMLR*, 3, 1157–1182.

Hall, M. A. (1999). *Correlation-based Feature Selection for Machine Learning*. PhD thesis, University of Waikato.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (2010). *A Practical Guide to Support Vector Classification*. Technical report, NTU.

Kratz, S., & Rohs, M. (2010). A $3 gesture recognizer: simple gesture recognition for devices equipped with 3D acceleration sensors. *IUI'10*, 341–344.

Kratz, S., & Rohs, M. (2011). Protractor3D: A closed-form solution to rotation-invariant 3D gestures. *IUI'11*, 371–374.

Mezari, A., & Maglogiannis, I. (2018). An easily customized gesture recognizer for assisted living using commodity mobile devices. *Journal of Healthcare Engineering*, 2018:3180652.

Mitra, S., & Acharya, T. (2007). Gesture recognition: A survey. *IEEE Trans. SMC-C*, 37 (3), 311–324.

Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1 (1), 81–106.

Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, 8 (25).

Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7 (91).

Wobbrock, J. O., Wilson, A. D., & Li, Y. (2007). Gestures without libraries, toolkits or training: A $1 recognizer for user interface prototypes. *UIST'07*, 159–168.

Wu, Y., & Huang, T. S. (1999). Vision-based gesture recognition: A review. *Gesture Workshop, LNAI 1739*, 103–115.

Yu, L., & Liu, H. (2004). Efficient Feature Selection via Analysis of Relevance and Redundancy. *JMLR*, 5, 1205–1224.

---

*Authors: Andry Lenny / El Mohcine Mohamed / Ottevaere Arthur — Group 6 — 2026*
