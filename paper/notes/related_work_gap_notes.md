# Related Work Gap Notes

## Main Positioning

The literature on C-MAPSS RUL prediction is rich in deep learning, attention, transformer, and XAI-based methods. However, many studies focus mainly on continuous RUL regression, often without a dual operational decision task such as normal/warning/critical risk-stage classification.

## Our Gap

This study should be positioned around the combination of:

1. RUL regression across FD001-FD004.
2. Degradation risk-stage classification.
3. Explainable temporal degradation feature engineering.
4. Ablation study over feature groups.
5. Bootstrap confidence intervals.
6. SHAP feature-level and category-level interpretation.
7. Comparison against classical ML, deep sequence models, and exploratory hybrid fusion.

## Claim Boundary

Do not claim that the hybrid model is superior. The strongest claim is that explainable temporal feature engineering is robust, interpretable, and competitive across complex operating conditions.

## Related Work Table Usage

Use `related_work_comparison_draft.csv` as the initial table. Each row must later be refined with exact paper title, venue, dataset subsets, reported metrics, and DOI.
