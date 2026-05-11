# Final Related Work Positioning Notes

## Core Gap

Recent C-MAPSS studies strongly emphasize deep RUL regression, attention models, robust sensor-failure modeling, classification benchmarks, and explainability. However, fewer studies jointly combine:

1. Continuous RUL regression.
2. Three-stage degradation risk classification.
3. Explicit temporal degradation feature engineering.
4. Feature-group ablation.
5. Bootstrap confidence intervals.
6. SHAP feature-level and category-level interpretation.
7. Deep sequence baseline comparison.

## Safe Novelty Statement

This study does not claim to outperform all deep learning models on C-MAPSS. Instead, it contributes an explainable and reproducible temporal degradation feature framework that is competitive across FD001-FD004 and supports both RUL regression and degradation risk-stage assessment.

## Important Claim Boundary

Do not claim the exploratory hybrid feature-sequence model is the best model. The strongest empirical claim is that engineered temporal feature variants provide robust performance under complex subsets FD002-FD004, while GRU performs best on FD001.

## Recommended Related Work Structure

1. RUL prediction with deep learning and attention.
2. Explainable RUL prediction and SHAP/counterfactual approaches.
3. Classification-based predictive maintenance.
4. Research gap and positioning of this study.
