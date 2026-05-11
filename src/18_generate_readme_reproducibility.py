from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path.cwd()
TABLE_DIR = ROOT / "outputs" / "tables"
PAPER_NOTES = ROOT / "paper" / "notes"
PAPER_NOTES.mkdir(parents=True, exist_ok=True)

def safe_read(path):
    if path.exists():
        return pd.read_csv(path)
    return None

def md_table(df, cols=None):
    if df is None:
        return "_Missing file._"
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    return df.to_markdown(index=False)

def main():
    print("=" * 80)
    print("GENERATE FINAL README AND REPRODUCIBILITY GUIDE")
    print("=" * 80)

    final_best = safe_read(TABLE_DIR / "final_best_rul_model_by_subset.csv")
    clf_best = safe_read(TABLE_DIR / "best_risk_classification_by_subset.csv")
    ablation_avg = safe_read(TABLE_DIR / "ablation_average_performance.csv")
    ci = safe_read(TABLE_DIR / "stat_regression_bootstrap_ci.csv")

    readme = []
    readme.append("# Explainable Temporal Degradation Feature Framework")
    readme.append("")
    readme.append("This repository contains a reproducible predictive-maintenance research pipeline for Remaining Useful Life (RUL) prediction and degradation risk-stage assessment using the NASA C-MAPSS turbofan engine degradation benchmark.")
    readme.append("")
    readme.append("## Recommended Paper Title")
    readme.append("")
    readme.append("**An Explainable Temporal Degradation Feature Framework for Remaining Useful Life Prediction and Risk Assessment of Industrial Systems**")
    readme.append("")
    readme.append("## Research Scope")
    readme.append("")
    readme.append("The project evaluates temporal degradation features for:")
    readme.append("")
    readme.append("- RUL regression")
    readme.append("- Risk-stage classification: normal, warning, critical")
    readme.append("- Feature ablation analysis")
    readme.append("- Bootstrap statistical validation")
    readme.append("- SHAP-based explainability")
    readme.append("- Deep sequence baseline comparison")
    readme.append("")
    readme.append("## Dataset")
    readme.append("")
    readme.append("NASA C-MAPSS turbofan engine degradation dataset:")
    readme.append("")
    readme.append("- FD001")
    readme.append("- FD002")
    readme.append("- FD003")
    readme.append("- FD004")
    readme.append("")
    readme.append("The dataset files should be placed in:")
    readme.append("")
    readme.append("```text")
    readme.append("data/raw/cmapss/")
    readme.append("```")
    readme.append("")
    readme.append("Required files:")
    readme.append("")
    readme.append("```text")
    readme.append("train_FD001.txt, test_FD001.txt, RUL_FD001.txt")
    readme.append("train_FD002.txt, test_FD002.txt, RUL_FD002.txt")
    readme.append("train_FD003.txt, test_FD003.txt, RUL_FD003.txt")
    readme.append("train_FD004.txt, test_FD004.txt, RUL_FD004.txt")
    readme.append("```")
    readme.append("")
    readme.append("## Environment")
    readme.append("")
    readme.append("```powershell")
    readme.append("conda activate main")
    readme.append("python src\\00_check_environment.py")
    readme.append("```")
    readme.append("")
    readme.append("Core environment:")
    readme.append("")
    readme.append("- Python 3.11")
    readme.append("- PyTorch CUDA")
    readme.append("- scikit-learn")
    readme.append("- XGBoost")
    readme.append("- LightGBM")
    readme.append("- CatBoost")
    readme.append("- SHAP")
    readme.append("")
    readme.append("## Pipeline")
    readme.append("")
    readme.append("Run each step sequentially:")
    readme.append("")
    readme.append("```powershell")
    for i in range(0, 19):
        script_map = {
            0: "00_check_environment.py",
            1: "01_data_audit.py",
            2: "02_build_rul_labels.py",
            3: "03_feature_engineering.py",
            4: "04_train_ml_baselines.py",
            5: "05_generate_baseline_figures.py",
            6: "06_train_risk_classification.py",
            7: "07_generate_risk_classification_figures.py",
            8: "08_explain_shap_rul.py",
            9: "09_ablation_rul_features.py",
            10: "10_statistical_validation.py",
            11: "11_generate_q1_research_assets.py",
            12: "12_deep_sequence_baselines.py",
            13: "13_hybrid_feature_sequence_model.py",
            14: "14_generate_final_ieee_assets.py",
            15: "15_generate_manuscript_draft.py",
            16: "16_create_related_work_draft.py",
            17: "17_generate_method_diagrams.py",
            18: "18_generate_readme_reproducibility.py",
        }
        readme.append(f"python src\\{script_map[i]}")
    readme.append("```")
    readme.append("")
    readme.append("Or run the prepared PowerShell pipelines:")
    readme.append("")
    readme.append("```powershell")
    readme.append("powershell -ExecutionPolicy Bypass -File run_overnight_pipeline.ps1")
    readme.append("powershell -ExecutionPolicy Bypass -File run_final_q1_pipeline.ps1")
    readme.append("```")
    readme.append("")
    readme.append("## Main Results")
    readme.append("")
    readme.append("### Best RUL Model by Subset")
    readme.append("")
    readme.append(md_table(final_best, cols=["subset", "family", "model", "MAE", "RMSE", "R2", "NASA_Score"]))
    readme.append("")
    readme.append("### Best Risk-Stage Classification Model by Subset")
    readme.append("")
    readme.append(md_table(clf_best, cols=["subset", "model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "ROC_AUC_OVR", "PR_AUC_Macro"]))
    readme.append("")
    readme.append("### Ablation Average Performance")
    readme.append("")
    readme.append(md_table(ablation_avg))
    readme.append("")
    readme.append("### Bootstrap Confidence Intervals")
    readme.append("")
    readme.append(md_table(ci))
    readme.append("")
    readme.append("## Important Claim Boundary")
    readme.append("")
    readme.append("This repository should be positioned as an explainable temporal degradation feature framework. The hybrid feature-sequence model is included as an exploratory fusion baseline and should not be claimed as the best-performing method.")
    readme.append("")
    readme.append("## Key Outputs")
    readme.append("")
    readme.append("Tables:")
    readme.append("")
    readme.append("```text")
    readme.append("outputs/tables/final_model_comparison_rul.csv")
    readme.append("outputs/tables/final_best_rul_model_by_subset.csv")
    readme.append("outputs/tables/ablation_rul_results.csv")
    readme.append("outputs/tables/stat_regression_bootstrap_ci.csv")
    readme.append("outputs/tables/shap_feature_category_summary.csv")
    readme.append("```")
    readme.append("")
    readme.append("Figures:")
    readme.append("")
    readme.append("```text")
    readme.append("outputs/figures/fig_framework_architecture.png")
    readme.append("outputs/figures/fig_temporal_feature_pipeline.png")
    readme.append("outputs/figures/fig_final_model_comparison_rmse.png")
    readme.append("outputs/figures/fig_ablation_rmse_by_subset.png")
    readme.append("outputs/figures/fig_shap_feature_category_summary.png")
    readme.append("```")
    readme.append("")
    readme.append("## Reproducibility")
    readme.append("")
    readme.append("```powershell")
    readme.append("conda env export > environment_main.yml")
    readme.append("pip freeze > requirements.txt")
    readme.append("```")
    readme.append("")
    readme.append("## Generated")
    readme.append("")
    readme.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    (ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    repro = []
    repro.append("# Reproducibility Guide")
    repro.append("")
    repro.append("## 1. Activate environment")
    repro.append("")
    repro.append("```powershell")
    repro.append("conda activate main")
    repro.append("```")
    repro.append("")
    repro.append("## 2. Verify environment")
    repro.append("")
    repro.append("```powershell")
    repro.append("python src\\00_check_environment.py")
    repro.append("```")
    repro.append("")
    repro.append("## 3. Prepare dataset")
    repro.append("")
    repro.append("Place NASA C-MAPSS files into:")
    repro.append("")
    repro.append("```text")
    repro.append("data/raw/cmapss/")
    repro.append("```")
    repro.append("")
    repro.append("## 4. Reproduce major results")
    repro.append("")
    repro.append("```powershell")
    repro.append("powershell -ExecutionPolicy Bypass -File run_overnight_pipeline.ps1")
    repro.append("powershell -ExecutionPolicy Bypass -File run_final_q1_pipeline.ps1")
    repro.append("python src\\17_generate_method_diagrams.py")
    repro.append("python src\\18_generate_readme_reproducibility.py")
    repro.append("```")
    repro.append("")
    repro.append("## 5. Main result files")
    repro.append("")
    repro.append("```text")
    repro.append("outputs/tables/final_model_comparison_rul.csv")
    repro.append("outputs/tables/final_best_rul_model_by_subset.csv")
    repro.append("outputs/tables/best_risk_classification_by_subset.csv")
    repro.append("outputs/tables/ablation_average_performance.csv")
    repro.append("outputs/tables/stat_regression_bootstrap_ci.csv")
    repro.append("outputs/tables/shap_feature_category_summary.csv")
    repro.append("```")
    repro.append("")
    repro.append("## 6. Paper notes")
    repro.append("")
    repro.append("```text")
    repro.append("paper/notes/final_paper_positioning.md")
    repro.append("paper/notes/q1_research_assets_summary.md")
    repro.append("paper/notes/ieee_manuscript_draft_notes.md")
    repro.append("paper/notes/reproducibility_guide.md")
    repro.append("```")

    (PAPER_NOTES / "reproducibility_guide.md").write_text("\n".join(repro), encoding="utf-8")

    print("Saved: README.md")
    print(f"Saved: {PAPER_NOTES / 'reproducibility_guide.md'}")
    print("STATUS: README_REPRODUCIBILITY_READY")

if __name__ == "__main__":
    main()
