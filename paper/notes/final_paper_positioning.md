# Final Paper Positioning

## Recommended Title

An Explainable Temporal Degradation Feature Framework for Remaining Useful Life Prediction and Risk Assessment of Industrial Systems

## Core Claim

This paper should be positioned as an explainable temporal feature engineering and predictive-maintenance framework, not as a hybrid deep-learning model paper.

## Main Contributions

1. A reproducible predictive-maintenance framework is developed for both remaining useful life prediction and degradation risk-stage assessment using NASA C-MAPSS FD001–FD004 datasets.

2. A temporal degradation feature representation is designed by integrating raw sensor measurements, operational settings, cycle-derived variables, lag features, and rolling degradation statistics.

3. A comprehensive evaluation is conducted across classical machine learning, deep sequence baselines, hybrid feature-sequence learning, and feature-ablation variants.

4. A dual-task evaluation protocol is introduced, covering continuous RUL regression and discrete degradation risk-stage classification for normal, warning, and critical states.

5. Bootstrap confidence intervals and SHAP-based explainability are provided to assess performance stability and identify the dominant temporal degradation patterns contributing to RUL prediction.

## Final Abstract Draft

Remaining useful life (RUL) prediction is a critical task in predictive maintenance, yet many existing approaches either rely on black-box sequence models or use static feature representations with limited interpretability. This study proposes an explainable temporal degradation feature framework for RUL prediction and degradation risk-stage assessment using multivariate industrial sensor time-series data. The framework integrates raw sensor measurements, operating conditions, cycle-derived variables, lag features, and rolling degradation statistics to represent temporal degradation behavior. Experiments are conducted on the NASA C-MAPSS turbofan engine benchmark across FD001–FD004. The proposed feature representation is evaluated against classical machine learning models, deep sequence baselines, hybrid feature-sequence learning, and multiple ablation variants. Results show that temporal feature engineering provides robust RUL prediction across complex operating conditions, achieving the best performance on three of four subsets, while a GRU sequence baseline performs best on FD001. Risk-stage classification further demonstrates the operational usefulness of the representation, achieving Macro-F1 scores above 0.72 and ROC-AUC values close to or above 0.89 across all subsets. Bootstrap confidence intervals confirm performance stability, and SHAP analysis reveals that cycle-derived variables and rolling sensor statistics are the dominant contributors to RUL prediction. These findings demonstrate that explainable temporal degradation features can provide accurate and interpretable predictive-maintenance decision support.

## Key Result Interpretation

The RUL regression results indicate that no single model family dominates all C-MAPSS subsets. The GRU sequence baseline achieved the lowest RMSE on FD001, suggesting that sequential dynamics are highly informative under the simpler single-condition setting. However, engineered temporal feature variants achieved the best results on FD002, FD003, and FD004, which involve more complex operating conditions and fault modes. This finding suggests that explicit temporal degradation features provide robust and compact representations when the operating environment becomes more heterogeneous.

The ablation analysis further confirms the contribution of temporal feature engineering. The full proposed feature representation achieved the best average RMSE across all subsets, while rolling-plus-cycle and raw-plus-cycle variants produced competitive subset-specific performance with fewer features. Compared with raw sensor features, the proposed representation reduced RMSE by 5.88% to 15.03%. Compared with cycle-only features, it reduced RMSE by 31.74% to 54.83%. These improvements demonstrate that neither operational age nor raw sensor readings alone are sufficient for reliable RUL estimation; instead, temporal degradation patterns encoded by rolling statistics provide critical predictive information.

For degradation risk-stage classification, the engineered temporal features supported effective discrimination among normal, warning, and critical states. Logistic regression achieved the best Macro-F1 on FD001 and FD003, whereas LightGBM performed best on FD002 and FD004. This suggests that the proposed feature representation can yield linearly separable degradation stages under simpler conditions, while nonlinear decision boundaries become more useful under multiple operating conditions and fault modes.

SHAP analysis provides additional interpretability. Cycle-derived variables were consistently influential, reflecting the relationship between operational age and degradation progression. However, rolling sensor statistics dominated several subsets, especially rolling mean features in FD001 and FD003 and rolling maximum features in FD002 and FD004. This indicates that smoothed temporal trends and local sensor envelopes encode meaningful degradation evidence beyond raw sensor values.

## Important Warning

Do not claim that the hybrid feature-sequence model is the best model. In the current experiment, it is an exploratory fusion baseline and does not outperform the best ablation or deep sequence models.

## Next Required Work

1. Build related-work comparison table.
2. Create architecture diagram.
3. Create temporal feature pipeline diagram.
4. Polish result figures.
5. Write full IEEE manuscript.
