# Reproducibility Guide

## 1. Activate environment

```powershell
conda activate main
```

## 2. Verify environment

```powershell
python src\00_check_environment.py
```

## 3. Prepare dataset

Place NASA C-MAPSS files into:

```text
data/raw/cmapss/
```

## 4. Reproduce major results

```powershell
powershell -ExecutionPolicy Bypass -File run_overnight_pipeline.ps1
powershell -ExecutionPolicy Bypass -File run_final_q1_pipeline.ps1
python src\17_generate_method_diagrams.py
python src\18_generate_readme_reproducibility.py
```

## 5. Main result files

```text
outputs/tables/final_model_comparison_rul.csv
outputs/tables/final_best_rul_model_by_subset.csv
outputs/tables/best_risk_classification_by_subset.csv
outputs/tables/ablation_average_performance.csv
outputs/tables/stat_regression_bootstrap_ci.csv
outputs/tables/shap_feature_category_summary.csv
```

## 6. Paper notes

```text
paper/notes/final_paper_positioning.md
paper/notes/q1_research_assets_summary.md
paper/notes/ieee_manuscript_draft_notes.md
paper/notes/reproducibility_guide.md
```