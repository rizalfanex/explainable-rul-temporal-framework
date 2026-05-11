from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path.cwd()
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
PAPER_NOTES = ROOT / "paper" / "notes"
PAPER_NOTES.mkdir(parents=True, exist_ok=True)

def read(path):
    return pd.read_csv(path) if path.exists() else None

def md(df, cols=None):
    if df is None:
        return "_Missing._"
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    return df.to_markdown(index=False)

def main():
    print("=" * 80)
    print("GENERATE IEEE MANUSCRIPT DRAFT NOTES")
    print("=" * 80)

    final_cmp = read(TABLE_DIR / "final_model_comparison_rul.csv")
    final_best = read(TABLE_DIR / "final_best_rul_model_by_subset.csv")
    clf = read(TABLE_DIR / "best_risk_classification_by_subset.csv")
    ablation_avg = read(TABLE_DIR / "ablation_average_performance.csv")
    ci = read(TABLE_DIR / "stat_regression_bootstrap_ci.csv")
    shap = read(TABLE_DIR / "shap_feature_category_summary.csv")
    figs = sorted([p.name for p in FIG_DIR.glob("*.png")])

    out = PAPER_NOTES / "ieee_manuscript_draft_notes.md"

    lines = []
    lines.append("# IEEE Manuscript Draft Notes")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Proposed Title")
    lines.append("")
    lines.append("Explainable Temporal Feature-Sequence Learning for Remaining Useful Life Prediction and Degradation Risk Assessment")
    lines.append("")
    lines.append("## Abstract Draft")
    lines.append("")
    lines.append(
        "Remaining useful life (RUL) prediction is a central task in predictive maintenance, yet many existing approaches either rely on black-box sequence models or static feature representations that provide limited interpretability. This study presents a reproducible predictive-maintenance framework that combines temporal degradation feature engineering, deep sequence baselines, hybrid feature-sequence learning, degradation risk-stage classification, ablation analysis, bootstrap statistical validation, and SHAP-based explainability. Experiments are conducted on the NASA C-MAPSS turbofan engine benchmark across FD001-FD004. The results demonstrate that temporal rolling statistics and cycle-derived observable features provide strong predictive evidence for RUL estimation, while risk-stage classification supports operational decision-making for normal, warning, and critical degradation states."
    )
    lines.append("")
    lines.append("## Claimed Contributions")
    lines.append("")
    lines.append("1. A reproducible RUL prediction and risk-stage assessment pipeline on FD001-FD004.")
    lines.append("2. A temporal feature engineering strategy using raw sensors, operating settings, lag features, and rolling degradation statistics.")
    lines.append("3. A dual-task protocol covering RUL regression and degradation risk-stage classification.")
    lines.append("4. A comparison between classical ML, deep sequence baselines, ablation variants, and a hybrid feature-sequence model.")
    lines.append("5. SHAP-based interpretation of feature-level and category-level degradation evidence.")
    lines.append("")
    lines.append("## Final RUL Model Comparison")
    lines.append("")
    lines.append(md(final_cmp, cols=["subset", "family", "model", "MAE", "RMSE", "R2", "NASA_Score"]))
    lines.append("")
    lines.append("## Best RUL Model by Subset")
    lines.append("")
    lines.append(md(final_best, cols=["subset", "family", "model", "MAE", "RMSE", "R2", "NASA_Score"]))
    lines.append("")
    lines.append("## Risk-Stage Classification Results")
    lines.append("")
    lines.append(md(clf, cols=["subset", "model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "ROC_AUC_OVR", "PR_AUC_Macro"]))
    lines.append("")
    lines.append("## Ablation Average Performance")
    lines.append("")
    lines.append(md(ablation_avg))
    lines.append("")
    lines.append("## Bootstrap Confidence Intervals")
    lines.append("")
    lines.append(md(ci))
    lines.append("")
    lines.append("## SHAP Category Summary")
    lines.append("")
    lines.append(md(shap))
    lines.append("")
    lines.append("## Results Section Draft")
    lines.append("")
    lines.append(
        "The RUL regression results show that temporal degradation features provide a strong predictive representation across all C-MAPSS subsets. The ablation analysis demonstrates that the full proposed representation achieved the best average RMSE, while rolling-plus-cycle variants produced competitive subset-specific performance with fewer features. This indicates that long-range rolling degradation statistics and cycle-derived observable variables are the primary drivers of predictive accuracy. In contrast, operational settings alone and cycle-only representations were insufficient, confirming that sensor-derived temporal degradation patterns are necessary for reliable RUL estimation."
    )
    lines.append("")
    lines.append(
        "For degradation risk-stage classification, the best models achieved Macro-F1 scores above 0.72 across all subsets and ROC-AUC values close to or above 0.89. Logistic regression performed strongly on FD001 and FD003, suggesting that the engineered temporal representation creates separable degradation states under simpler operating conditions. LightGBM performed best on FD002 and FD004, indicating that nonlinear decision boundaries are more useful under multiple operating conditions and fault modes."
    )
    lines.append("")
    lines.append(
        "SHAP analysis further supports the interpretability of the framework. Cycle-derived features were consistently important, but rolling sensor statistics dominated several subsets, particularly rolling mean in FD001/FD003 and rolling maximum in FD002/FD004. These findings suggest that smoothed temporal trends and local sensor envelopes encode meaningful degradation evidence beyond raw sensor readings."
    )
    lines.append("")
    lines.append("## Generated Figure Manifest")
    lines.append("")
    for i, f in enumerate(figs, 1):
        lines.append(f"- Fig. {i}: `{f}`")
    lines.append("")
    lines.append("## Remaining Before Submission")
    lines.append("")
    lines.append("1. Add related-work comparison table with recent C-MAPSS studies.")
    lines.append("2. Draw architecture diagram and temporal feature engineering pipeline figure.")
    lines.append("3. Polish all figures into IEEE visual style.")
    lines.append("4. Write Introduction, Related Work, Methodology, Experiments, Results, and Conclusion.")
    lines.append("5. Add reproducibility instructions to README.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")

    print("\n[Final status]")
    print("STATUS: MANUSCRIPT_DRAFT_NOTES_READY")

if __name__ == "__main__":
    main()
