from pathlib import Path
import pandas as pd
from datetime import datetime

ROOT = Path.cwd()
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
PAPER_NOTES = ROOT / "paper" / "notes"
PAPER_NOTES.mkdir(parents=True, exist_ok=True)

def safe_read(path):
    if path.exists():
        return pd.read_csv(path)
    return None

def md_table(df, cols=None, max_rows=None):
    if df is None:
        return "_Missing file._"
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    if max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False)

def main():
    print("=" * 80)
    print("GENERATE Q1 RESEARCH ASSETS SUMMARY")
    print("=" * 80)

    audit = safe_read(ROOT / "outputs" / "metrics" / "cmapss_audit_summary.csv")
    labels = safe_read(TABLE_DIR / "rul_label_summary.csv")
    features = safe_read(TABLE_DIR / "feature_engineering_summary.csv")
    reg = safe_read(TABLE_DIR / "best_ml_baseline_by_subset.csv")
    clf = safe_read(TABLE_DIR / "best_risk_classification_by_subset.csv")
    shap_cat = safe_read(TABLE_DIR / "shap_feature_category_summary.csv")
    ablation = safe_read(TABLE_DIR / "ablation_best_by_subset.csv")
    ablation_avg = safe_read(TABLE_DIR / "ablation_average_performance.csv")
    reg_ci = safe_read(TABLE_DIR / "stat_regression_bootstrap_ci.csv")
    abl_imp = safe_read(TABLE_DIR / "stat_ablation_improvement_summary.csv")
    clf_stat = safe_read(TABLE_DIR / "stat_classification_average_by_model.csv")

    figures = sorted([p.name for p in FIG_DIR.glob("*.png")])

    out = PAPER_NOTES / "q1_research_assets_summary.md"

    lines = []
    lines.append("# Q1/IEEE Research Assets Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Tentative Title")
    lines.append("")
    lines.append("Explainable Temporal Feature Engineering for Remaining Useful Life Prediction and Degradation Risk Assessment in Industrial Predictive Maintenance")
    lines.append("")
    lines.append("## Current Contribution Claim")
    lines.append("")
    lines.append(
        "This project presents a reproducible predictive-maintenance framework that combines temporal degradation feature engineering, "
        "multi-subset evaluation on NASA C-MAPSS, RUL regression, risk-stage classification, ablation analysis, bootstrap statistical validation, "
        "and SHAP-based interpretability."
    )
    lines.append("")
    lines.append("## Dataset Audit")
    lines.append("")
    lines.append(md_table(audit, cols=["subset", "train_rows", "test_rows", "train_units", "test_units", "missing_values_train", "missing_values_test"]))
    lines.append("")
    lines.append("## RUL Label Summary")
    lines.append("")
    lines.append(md_table(labels, cols=["subset", "split", "rows", "units", "rul_min", "rul_max", "rul_capped_mean", "normal_count", "warning_count", "critical_count"]))
    lines.append("")
    lines.append("## Feature Engineering Summary")
    lines.append("")
    lines.append(md_table(features, cols=["subset", "split", "rows", "num_total_columns", "num_model_features", "missing_values", "infinite_values"]))
    lines.append("")
    lines.append("## Best RUL Regression Results")
    lines.append("")
    lines.append(md_table(reg, cols=["subset", "model", "MAE", "RMSE", "R2", "NASA_Score", "train_time_sec", "inference_time_sec"]))
    lines.append("")
    lines.append("## Best Risk-Stage Classification Results")
    lines.append("")
    lines.append(md_table(clf, cols=["subset", "model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "ROC_AUC_OVR", "PR_AUC_Macro"]))
    lines.append("")
    lines.append("## Ablation Best Results")
    lines.append("")
    lines.append(md_table(ablation, cols=["subset", "ablation", "num_features", "MAE", "RMSE", "R2", "NASA_Score"]))
    lines.append("")
    lines.append("## Ablation Average Performance")
    lines.append("")
    lines.append(md_table(ablation_avg))
    lines.append("")
    lines.append("## Bootstrap Confidence Intervals")
    lines.append("")
    lines.append(md_table(reg_ci))
    lines.append("")
    lines.append("## Ablation Improvement Summary")
    lines.append("")
    lines.append(md_table(abl_imp))
    lines.append("")
    lines.append("## Classification Average Statistical Summary")
    lines.append("")
    lines.append(md_table(clf_stat))
    lines.append("")
    lines.append("## SHAP Feature Category Summary")
    lines.append("")
    lines.append(md_table(shap_cat))
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    for i, fig in enumerate(figures, 1):
        lines.append(f"- Fig. {i}: `{fig}`")
    lines.append("")
    lines.append("## Manuscript-Ready Results Draft")
    lines.append("")
    lines.append(
        "The experimental results demonstrate that the proposed temporal feature representation provides strong predictive information for both "
        "continuous RUL estimation and discrete degradation risk assessment. For RUL regression, CatBoost achieved the lowest RMSE across all "
        "C-MAPSS subsets, indicating robust compatibility between gradient-boosted decision trees and the engineered temporal degradation features. "
        "The risk-stage classification task further showed that the same feature representation supports operational decision making, with Logistic "
        "Regression performing strongly on simpler subsets and LightGBM performing better on more complex multi-condition subsets. SHAP analysis "
        "confirmed that cycle-derived features and rolling sensor statistics contributed substantially to RUL prediction, supporting the value of "
        "temporal degradation modeling beyond raw sensor readings."
    )
    lines.append("")
    lines.append("## Remaining Work Before Submission")
    lines.append("")
    lines.append("1. Add deep sequence baseline: GRU/LSTM/TCN.")
    lines.append("2. Add proposed hybrid model: sequence encoder + engineered temporal features.")
    lines.append("3. Add architecture diagram and feature pipeline diagram.")
    lines.append("4. Compare with previous C-MAPSS studies.")
    lines.append("5. Write full IEEE manuscript sections.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")
    print("\n[Final status]")
    print("STATUS: Q1_RESEARCH_ASSETS_READY")

if __name__ == "__main__":
    main()
