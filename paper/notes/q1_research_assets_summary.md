# Q1/IEEE Research Assets Summary

Generated: 2026-05-11 22:03:43

## Tentative Title

Explainable Temporal Feature Engineering for Remaining Useful Life Prediction and Degradation Risk Assessment in Industrial Predictive Maintenance

## Current Contribution Claim

This project presents a reproducible predictive-maintenance framework that combines temporal degradation feature engineering, multi-subset evaluation on NASA C-MAPSS, RUL regression, risk-stage classification, ablation analysis, bootstrap statistical validation, and SHAP-based interpretability.

## Dataset Audit

| subset   |   train_rows |   test_rows |   train_units |   test_units |   missing_values_train |   missing_values_test |
|:---------|-------------:|------------:|--------------:|-------------:|-----------------------:|----------------------:|
| FD001    |        20631 |       13096 |           100 |          100 |                      0 |                     0 |
| FD002    |        53759 |       33991 |           260 |          259 |                      0 |                     0 |
| FD003    |        24720 |       16596 |           100 |          100 |                      0 |                     0 |
| FD004    |        61249 |       41214 |           249 |          248 |                      0 |                     0 |

## RUL Label Summary

| subset   | split   |   rows |   units |   rul_min |   rul_max |   rul_capped_mean |   normal_count |   warning_count |   critical_count |
|:---------|:--------|-------:|--------:|----------:|----------:|------------------:|---------------:|----------------:|-----------------:|
| FD001    | train   |  20631 |     100 |         0 |       361 |           86.8293 |           8031 |            7500 |             5100 |
| FD001    | test    |  13096 |     100 |         7 |       340 |          108.92   |           7952 |            4258 |              886 |
| FD002    | train   |  53759 |     260 |         0 |       377 |           86.9134 |          20999 |           19500 |            13260 |
| FD002    | test    |  33991 |     259 |         6 |       377 |          107.75   |          20500 |           10905 |             2586 |
| FD003    | train   |  24720 |     100 |         0 |       524 |           93.1432 |          12120 |            7500 |             5100 |
| FD003    | test    |  16596 |     100 |         6 |       483 |          112.314  |          11387 |            4430 |              779 |
| FD004    | train   |  61249 |     249 |         0 |       542 |           92.9852 |          29875 |           18675 |            12699 |
| FD004    | test    |  41214 |     248 |         6 |       553 |          112.614  |          29342 |            9622 |             2250 |

## Feature Engineering Summary

| subset   | split   |   rows |   num_total_columns |   num_model_features |   missing_values |   infinite_values |
|:---------|:--------|-------:|--------------------:|---------------------:|-----------------:|------------------:|
| FD001    | train   |  20631 |                 411 |                  404 |                0 |                 0 |
| FD001    | test    |  13096 |                 411 |                  404 |                0 |                 0 |
| FD002    | train   |  53759 |                 411 |                  404 |                0 |                 0 |
| FD002    | test    |  33991 |                 411 |                  404 |                0 |                 0 |
| FD003    | train   |  24720 |                 411 |                  404 |                0 |                 0 |
| FD003    | test    |  16596 |                 411 |                  404 |                0 |                 0 |
| FD004    | train   |  61249 |                 411 |                  404 |                0 |                 0 |
| FD004    | test    |  41214 |                 411 |                  404 |                0 |                 0 |

## Best RUL Regression Results

| subset   | model    |      MAE |    RMSE |       R2 |   NASA_Score |   train_time_sec |   inference_time_sec |
|:---------|:---------|---------:|--------:|---------:|-------------:|-----------------:|---------------------:|
| FD001    | CatBoost | 10.1112  | 15.5579 | 0.681773 |      61833.5 |          5.29053 |             0.010522 |
| FD002    | CatBoost | 11.501   | 17.2465 | 0.642061 |     230916   |          8.15235 |             0.014324 |
| FD003    | CatBoost |  7.39837 | 12.1886 | 0.759169 |      50286.4 |          5.54016 |             0.041987 |
| FD004    | CatBoost |  9.60342 | 15.5127 | 0.628873 |     401468   |          8.49083 |             0.015753 |

## Best Risk-Stage Classification Results

| subset   | model              |   Accuracy |   Precision_Macro |   Recall_Macro |   F1_Macro |   ROC_AUC_OVR |   PR_AUC_Macro |
|:---------|:-------------------|-----------:|------------------:|---------------:|-----------:|--------------:|---------------:|
| FD001    | LogisticRegression |   0.762065 |          0.751614 |       0.754006 |   0.7527   |      0.897843 |       0.829418 |
| FD002    | LightGBM           |   0.744844 |          0.702359 |       0.743896 |   0.720126 |      0.888623 |       0.785178 |
| FD003    | LogisticRegression |   0.82086  |          0.776194 |       0.811611 |   0.792375 |      0.926048 |       0.864307 |
| FD004    | LightGBM           |   0.7916   |          0.719062 |       0.725841 |   0.721866 |      0.895259 |       0.767252 |

## Ablation Best Results

| subset   | ablation              |   num_features |      MAE |    RMSE |       R2 |   NASA_Score |
|:---------|:----------------------|---------------:|---------:|--------:|---------:|-------------:|
| FD001    | A8_rolling_plus_cycle |            317 | 10.0681  | 15.553  | 0.681973 |      62292.8 |
| FD002    | A5_raw_plus_cycle     |             23 | 11.483   | 16.9751 | 0.653237 |     189621   |
| FD003    | A9_full_proposed      |            404 |  7.3549  | 12.0842 | 0.763278 |      50177.2 |
| FD004    | A8_rolling_plus_cycle |            317 |  9.55747 | 15.4924 | 0.629844 |     406325   |

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

## Ablation Improvement Summary

| subset   |   proposed_RMSE | reference        |   reference_RMSE |   absolute_RMSE_reduction |   relative_RMSE_reduction_percent |
|:---------|----------------:|:-----------------|-----------------:|--------------------------:|----------------------------------:|
| FD001    |         15.5643 | raw_sensors_only |          16.9285 |                  1.36412  |                           8.05817 |
| FD001    |         15.5643 | cycle_only       |          23.3843 |                  7.81994  |                          33.441   |
| FD001    |         15.5643 | rolling_only     |          16.2372 |                  0.67291  |                           4.14424 |
| FD002    |         17.2362 | raw_sensors_only |          18.9455 |                  1.70929  |                           9.02212 |
| FD002    |         17.2362 | cycle_only       |          25.2505 |                  8.01428  |                          31.7391  |
| FD002    |         17.2362 | rolling_only     |          18.8157 |                  1.57942  |                           8.39419 |
| FD003    |         12.0842 | raw_sensors_only |          14.2213 |                  2.13713  |                          15.0277  |
| FD003    |         12.0842 | cycle_only       |          26.752  |                 14.6679   |                          54.829   |
| FD003    |         12.0842 | rolling_only     |          12.8041 |                  0.719923 |                           5.6226  |
| FD004    |         15.5224 | raw_sensors_only |          16.4921 |                  0.969644 |                           5.87946 |
| FD004    |         15.5224 | cycle_only       |          30.4976 |                 14.9752   |                          49.1029  |
| FD004    |         15.5224 | rolling_only     |          16.6578 |                  1.13538  |                           6.81589 |

## Classification Average Statistical Summary

| model              |   mean_Accuracy |   std_Accuracy |   mean_F1_Macro |   std_F1_Macro |   mean_ROC_AUC_OVR |   std_ROC_AUC_OVR |
|:-------------------|----------------:|---------------:|----------------:|---------------:|-------------------:|------------------:|
| LogisticRegression |        0.76862  |      0.0385625 |        0.741682 |      0.0391385 |           0.900386 |         0.017909  |
| LightGBM           |        0.771605 |      0.0394526 |        0.739238 |      0.0331771 |           0.895526 |         0.0118179 |
| CatBoost           |        0.778251 |      0.0383389 |        0.737339 |      0.0363327 |           0.897446 |         0.013014  |
| RandomForest       |        0.782302 |      0.036896  |        0.735784 |      0.0317332 |           0.895723 |         0.0121599 |
| XGBoost            |        0.771936 |      0.0381404 |        0.735326 |      0.0315837 |           0.895579 |         0.0127002 |
| ExtraTrees         |        0.78426  |      0.0351067 |        0.733067 |      0.0312559 |           0.893205 |         0.011585  |

## SHAP Feature Category Summary

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

## Generated Figures

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
- Fig. 12: `fig_risk_classification_accuracy.png`
- Fig. 13: `fig_risk_classification_average_f1_rank.png`
- Fig. 14: `fig_risk_classification_macro_f1.png`
- Fig. 15: `fig_risk_classification_roc_auc.png`
- Fig. 16: `fig_shap_bar_FD001_CatBoost.png`
- Fig. 17: `fig_shap_bar_FD002_CatBoost.png`
- Fig. 18: `fig_shap_bar_FD003_CatBoost.png`
- Fig. 19: `fig_shap_bar_FD004_CatBoost.png`
- Fig. 20: `fig_shap_feature_category_summary.png`

## Manuscript-Ready Results Draft

The experimental results demonstrate that the proposed temporal feature representation provides strong predictive information for both continuous RUL estimation and discrete degradation risk assessment. For RUL regression, CatBoost achieved the lowest RMSE across all C-MAPSS subsets, indicating robust compatibility between gradient-boosted decision trees and the engineered temporal degradation features. The risk-stage classification task further showed that the same feature representation supports operational decision making, with Logistic Regression performing strongly on simpler subsets and LightGBM performing better on more complex multi-condition subsets. SHAP analysis confirmed that cycle-derived features and rolling sensor statistics contributed substantially to RUL prediction, supporting the value of temporal degradation modeling beyond raw sensor readings.

## Remaining Work Before Submission

1. Add deep sequence baseline: GRU/LSTM/TCN.
2. Add proposed hybrid model: sequence encoder + engineered temporal features.
3. Add architecture diagram and feature pipeline diagram.
4. Compare with previous C-MAPSS studies.
5. Write full IEEE manuscript sections.
