# IEEE Manuscript Draft Notes

Generated: 2026-05-11 22:11:11

## Proposed Title

Explainable Temporal Feature-Sequence Learning for Remaining Useful Life Prediction and Degradation Risk Assessment

## Abstract Draft

Remaining useful life (RUL) prediction is a central task in predictive maintenance, yet many existing approaches either rely on black-box sequence models or static feature representations that provide limited interpretability. This study presents a reproducible predictive-maintenance framework that combines temporal degradation feature engineering, deep sequence baselines, hybrid feature-sequence learning, degradation risk-stage classification, ablation analysis, bootstrap statistical validation, and SHAP-based explainability. Experiments are conducted on the NASA C-MAPSS turbofan engine benchmark across FD001-FD004. The results demonstrate that temporal rolling statistics and cycle-derived observable features provide strong predictive evidence for RUL estimation, while risk-stage classification supports operational decision-making for normal, warning, and critical degradation states.

## Claimed Contributions

1. A reproducible RUL prediction and risk-stage assessment pipeline on FD001-FD004.
2. A temporal feature engineering strategy using raw sensors, operating settings, lag features, and rolling degradation statistics.
3. A dual-task protocol covering RUL regression and degradation risk-stage classification.
4. A comparison between classical ML, deep sequence baselines, ablation variants, and a hybrid feature-sequence model.
5. SHAP-based interpretation of feature-level and category-level degradation evidence.

## Final RUL Model Comparison

| subset   | family          | model                     |      MAE |    RMSE |       R2 |   NASA_Score |
|:---------|:----------------|:--------------------------|---------:|--------:|---------:|-------------:|
| FD001    | Deep Sequence   | GRU                       |  9.99656 | 13.493  | 0.794572 |      10105.7 |
| FD001    | Best Ablation   | A8_rolling_plus_cycle     | 10.0681  | 15.553  | 0.681973 |      62292.8 |
| FD001    | Classical ML    | CatBoost (CatBoost)       | 10.1112  | 15.5579 | 0.681773 |      61833.5 |
| FD001    | Hybrid Proposed | HybridGRUFeatureAttention | 13.9843  | 18.1798 | 0.627079 |      19275.8 |
| FD002    | Best Ablation   | A5_raw_plus_cycle         | 11.483   | 16.9751 | 0.653237 |     189621   |
| FD002    | Classical ML    | CatBoost (CatBoost)       | 11.501   | 17.2465 | 0.642061 |     230916   |
| FD002    | Hybrid Proposed | HybridGRUFeatureAttention | 14.6207  | 19.4678 | 0.602934 |      81584.6 |
| FD002    | Deep Sequence   | GRU                       | 17.3557  | 21.4338 | 0.51869  |     101084   |
| FD003    | Best Ablation   | A9_full_proposed          |  7.3549  | 12.0842 | 0.763278 |      50177.2 |
| FD003    | Classical ML    | CatBoost (CatBoost)       |  7.39837 | 12.1886 | 0.759169 |      50286.4 |
| FD003    | Hybrid Proposed | HybridGRUFeatureAttention |  8.61362 | 13.1075 | 0.755622 |      15868.1 |
| FD003    | Deep Sequence   | LSTM                      | 10.0728  | 14.1421 | 0.715522 |      19255.6 |
| FD004    | Best Ablation   | A8_rolling_plus_cycle     |  9.55747 | 15.4924 | 0.629844 |     406325   |
| FD004    | Classical ML    | CatBoost (CatBoost)       |  9.60342 | 15.5127 | 0.628873 |     401468   |
| FD004    | Deep Sequence   | GRU                       | 12.6672  | 18.2229 | 0.551902 |     233911   |
| FD004    | Hybrid Proposed | HybridGRUFeatureAttention | 13.2675  | 18.7865 | 0.523752 |     186111   |

## Best RUL Model by Subset

| subset   | family        | model                 |      MAE |    RMSE |       R2 |   NASA_Score |
|:---------|:--------------|:----------------------|---------:|--------:|---------:|-------------:|
| FD001    | Deep Sequence | GRU                   |  9.99656 | 13.493  | 0.794572 |      10105.7 |
| FD002    | Best Ablation | A5_raw_plus_cycle     | 11.483   | 16.9751 | 0.653237 |     189621   |
| FD003    | Best Ablation | A9_full_proposed      |  7.3549  | 12.0842 | 0.763278 |      50177.2 |
| FD004    | Best Ablation | A8_rolling_plus_cycle |  9.55747 | 15.4924 | 0.629844 |     406325   |

## Risk-Stage Classification Results

| subset   | model              |   Accuracy |   Precision_Macro |   Recall_Macro |   F1_Macro |   ROC_AUC_OVR |   PR_AUC_Macro |
|:---------|:-------------------|-----------:|------------------:|---------------:|-----------:|--------------:|---------------:|
| FD001    | LogisticRegression |   0.762065 |          0.751614 |       0.754006 |   0.7527   |      0.897843 |       0.829418 |
| FD002    | LightGBM           |   0.744844 |          0.702359 |       0.743896 |   0.720126 |      0.888623 |       0.785178 |
| FD003    | LogisticRegression |   0.82086  |          0.776194 |       0.811611 |   0.792375 |      0.926048 |       0.864307 |
| FD004    | LightGBM           |   0.7916   |          0.719062 |       0.725841 |   0.721866 |      0.895259 |       0.767252 |

## Ablation Average Performance

| ablation                |   avg_RMSE |   avg_MAE |     avg_R2 |   avg_NASA_Score |   num_features |
|:------------------------|-----------:|----------:|-----------:|-----------------:|---------------:|
| A9_full_proposed        |    15.1018 |   9.63078 |  0.678921  | 187102           |            404 |
| A8_rolling_plus_cycle   |    15.1117 |   9.63683 |  0.678562  | 186884           |            317 |
| A5_raw_plus_cycle       |    15.2728 |   9.78111 |  0.671547  | 194794           |             23 |
| A7_full_without_cycle   |    16.0889 |  11.1026  |  0.63522   | 281405           |            402 |
| A6_rolling_only         |    16.1287 |  11.1268  |  0.633408  | 288331           |            315 |
| A2_raw_sensors_only     |    16.6468 |  11.6219  |  0.610994  | 300304           |             21 |
| A4_raw_plus_operational |    16.6496 |  11.6214  |  0.61087   | 299546           |             24 |
| A3_cycle_only           |    26.4711 |  17.6342  | -0.0201978 | 771420           |              2 |
| A1_operational_only     |    33.6501 |  31.526   | -0.592526  |      1.23223e+06 |              3 |

## Bootstrap Confidence Intervals

| subset   | model    |    RMSE |   RMSE_boot_mean |   RMSE_boot_std |   RMSE_ci95_low |   RMSE_ci95_high |      MAE |   MAE_boot_mean |   MAE_boot_std |   MAE_ci95_low |   MAE_ci95_high |
|:---------|:---------|--------:|-----------------:|----------------:|----------------:|-----------------:|---------:|----------------:|---------------:|---------------:|----------------:|
| FD001    | CatBoost | 15.5579 |          15.5551 |       0.134519  |         15.2913 |          15.8157 | 10.1112  |        10.107   |      0.0972385 |        9.91674 |        10.2965  |
| FD002    | CatBoost | 17.2465 |          17.2445 |       0.091621  |         17.0582 |          17.4116 | 11.501   |        11.4998  |      0.0680651 |       11.3727  |        11.6308  |
| FD003    | CatBoost | 12.1886 |          12.1865 |       0.108729  |         11.9664 |          12.411  |  7.39837 |         7.3985  |      0.0770142 |        7.23895 |         7.54147 |
| FD004    | CatBoost | 15.5127 |          15.5111 |       0.0999661 |         15.3114 |          15.6983 |  9.60342 |         9.60265 |      0.0603932 |        9.49098 |         9.72463 |

## SHAP Category Summary

| subset   | category            |   mean_abs_shap |
|:---------|:--------------------|----------------:|
| FD001    | rolling_mean        |      17.9065    |
| FD001    | cycle_observable    |      11.2038    |
| FD001    | rolling_max         |       9.52162   |
| FD001    | rolling_min         |       9.22621   |
| FD001    | rolling_range       |       4.37462   |
| FD001    | rolling_std         |       3.42837   |
| FD001    | raw_sensor          |       2.04737   |
| FD001    | lag                 |       0.737322  |
| FD001    | delta               |       0         |
| FD001    | operational_setting |       0         |
| FD001    | percentage_change   |       0         |
| FD002    | rolling_max         |      23.6764    |
| FD002    | cycle_observable    |      15.3355    |
| FD002    | rolling_min         |       7.14576   |
| FD002    | rolling_range       |       1.76577   |
| FD002    | raw_sensor          |       0.386611  |
| FD002    | rolling_mean        |       0.282221  |
| FD002    | lag                 |       0.226679  |
| FD002    | rolling_std         |       0.168861  |
| FD002    | delta               |       0         |
| FD002    | operational_setting |       0         |
| FD002    | percentage_change   |       0         |
| FD003    | rolling_mean        |      13.3064    |
| FD003    | rolling_max         |       9.00653   |
| FD003    | cycle_observable    |       8.27833   |
| FD003    | rolling_min         |       6.28576   |
| FD003    | rolling_std         |       5.17937   |
| FD003    | rolling_range       |       4.48137   |
| FD003    | raw_sensor          |       0.981833  |
| FD003    | lag                 |       0.745925  |
| FD003    | percentage_change   |       0.0257027 |
| FD003    | delta               |       0         |
| FD003    | operational_setting |       0         |
| FD004    | rolling_max         |      22.4585    |
| FD004    | cycle_observable    |       9.30757   |
| FD004    | rolling_range       |       6.42172   |
| FD004    | rolling_min         |       5.93669   |
| FD004    | raw_sensor          |       0.395005  |
| FD004    | rolling_mean        |       0.386344  |
| FD004    | lag                 |       0.184728  |
| FD004    | rolling_std         |       0.168664  |
| FD004    | delta               |       0         |
| FD004    | operational_setting |       0         |
| FD004    | percentage_change   |       0         |

## Results Section Draft

The RUL regression results show that temporal degradation features provide a strong predictive representation across all C-MAPSS subsets. The ablation analysis demonstrates that the full proposed representation achieved the best average RMSE, while rolling-plus-cycle variants produced competitive subset-specific performance with fewer features. This indicates that long-range rolling degradation statistics and cycle-derived observable variables are the primary drivers of predictive accuracy. In contrast, operational settings alone and cycle-only representations were insufficient, confirming that sensor-derived temporal degradation patterns are necessary for reliable RUL estimation.

For degradation risk-stage classification, the best models achieved Macro-F1 scores above 0.72 across all subsets and ROC-AUC values close to or above 0.89. Logistic regression performed strongly on FD001 and FD003, suggesting that the engineered temporal representation creates separable degradation states under simpler operating conditions. LightGBM performed best on FD002 and FD004, indicating that nonlinear decision boundaries are more useful under multiple operating conditions and fault modes.

SHAP analysis further supports the interpretability of the framework. Cycle-derived features were consistently important, but rolling sensor statistics dominated several subsets, particularly rolling mean in FD001/FD003 and rolling maximum in FD002/FD004. These findings suggest that smoothed temporal trends and local sensor envelopes encode meaningful degradation evidence beyond raw sensor readings.

## Generated Figure Manifest

- Fig. 1: `fig_ablation_feature_count_vs_rmse.png`
- Fig. 2: `fig_ablation_mae_by_subset.png`
- Fig. 3: `fig_ablation_rmse_by_subset.png`
- Fig. 4: `fig_baseline_average_rmse_rank.png`
- Fig. 5: `fig_baseline_mae_comparison.png`
- Fig. 6: `fig_baseline_rmse_comparison.png`
- Fig. 7: `fig_baseline_training_time_comparison.png`
- Fig. 8: `fig_confusion_matrix_FD001_LogisticRegression.png`
- Fig. 9: `fig_confusion_matrix_FD002_LightGBM.png`
- Fig. 10: `fig_confusion_matrix_FD003_LogisticRegression.png`
- Fig. 11: `fig_confusion_matrix_FD004_LightGBM.png`
- Fig. 12: `fig_deep_sequence_rmse_comparison.png`
- Fig. 13: `fig_final_model_comparison_rmse.png`
- Fig. 14: `fig_hybrid_model_rmse_by_subset.png`
- Fig. 15: `fig_risk_classification_accuracy.png`
- Fig. 16: `fig_risk_classification_average_f1_rank.png`
- Fig. 17: `fig_risk_classification_macro_f1.png`
- Fig. 18: `fig_risk_classification_roc_auc.png`
- Fig. 19: `fig_shap_bar_FD001_CatBoost.png`
- Fig. 20: `fig_shap_bar_FD002_CatBoost.png`
- Fig. 21: `fig_shap_bar_FD003_CatBoost.png`
- Fig. 22: `fig_shap_bar_FD004_CatBoost.png`
- Fig. 23: `fig_shap_feature_category_summary.png`

## Remaining Before Submission

1. Add related-work comparison table with recent C-MAPSS studies.
2. Draw architecture diagram and temporal feature engineering pipeline figure.
3. Polish all figures into IEEE visual style.
4. Write Introduction, Related Work, Methodology, Experiments, Results, and Conclusion.
5. Add reproducibility instructions to README.
