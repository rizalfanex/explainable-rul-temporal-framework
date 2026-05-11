from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
TABLE_DIR = ROOT / "outputs" / "tables"
PAPER_TABLE_DIR = ROOT / "paper" / "tables"
PAPER_NOTES_DIR = ROOT / "paper" / "notes"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
PAPER_NOTES_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "ref_id": "R1",
        "year": 2025,
        "study_focus": "Deep learning RUL prediction",
        "method_family": "CNN / attention / feature fusion",
        "dataset": "NASA C-MAPSS",
        "tasks": "RUL regression",
        "uses_fd001_fd004": "Partial / to verify",
        "uses_risk_classification": "No / unclear",
        "uses_ablation": "To verify",
        "uses_xai": "No / unclear",
        "reported_strength": "Deep feature extraction and attention-based representation",
        "limitation_gap": "Often emphasizes prediction accuracy more than decision-stage classification and feature-level interpretability",
        "relation_to_our_work": "Our work adds dual-task RUL + risk-stage evaluation, temporal feature ablation, bootstrap CI, and SHAP category analysis",
        "source_hint": "Sensors 2025 dual-attention C-MAPSS RUL"
    },
    {
        "ref_id": "R2",
        "year": 2025,
        "study_focus": "XAI impact in RUL prediction",
        "method_family": "Preprocessing + model complexity + LIME/SHAP/L2X",
        "dataset": "RUL predictive maintenance datasets / C-MAPSS context",
        "tasks": "RUL regression explanation",
        "uses_fd001_fd004": "To verify",
        "uses_risk_classification": "No",
        "uses_ablation": "Modeling/preprocessing analysis",
        "uses_xai": "Yes",
        "reported_strength": "Studies how preprocessing and model complexity influence explanation quality",
        "limitation_gap": "May focus on explanation quality rather than a complete dual-task operational decision framework",
        "relation_to_our_work": "Our work combines SHAP with RUL regression, risk-stage classification, feature ablation, and statistical validation",
        "source_hint": "Applied Soft Computing 2025 XAI RUL preprocessing/model complexity"
    },
    {
        "ref_id": "R3",
        "year": 2025,
        "study_focus": "Deep learning aero-engine RUL",
        "method_family": "Attention GRU / deep sequence learning",
        "dataset": "NASA C-MAPSS",
        "tasks": "RUL regression",
        "uses_fd001_fd004": "Partial / to verify",
        "uses_risk_classification": "No / unclear",
        "uses_ablation": "To verify",
        "uses_xai": "Limited / unclear",
        "reported_strength": "Deep sequence modeling for aero-engine RUL",
        "limitation_gap": "Deep models can be less transparent and may not include decision-stage classification",
        "relation_to_our_work": "Our deep sequence baseline is compared against explainable temporal feature engineering and risk-stage assessment",
        "source_hint": "Scientific Reports 2025 deep learning prognostic approach"
    },
    {
        "ref_id": "R4",
        "year": 2026,
        "study_focus": "Classification-based predictive maintenance",
        "method_family": "Machine-learning classification",
        "dataset": "NASA C-MAPSS",
        "tasks": "Failure / PdM classification",
        "uses_fd001_fd004": "To verify",
        "uses_risk_classification": "Yes / classification framing",
        "uses_ablation": "To verify",
        "uses_xai": "No / unclear",
        "reported_strength": "Wide-scope classification benchmark for aircraft engine maintenance",
        "limitation_gap": "Classification may not jointly evaluate continuous RUL regression and SHAP-based feature interpretation",
        "relation_to_our_work": "Our work combines RUL regression with risk-stage classification and interpretable temporal degradation features",
        "source_hint": "AI EDAM 2026 classification-based PdM C-MAPSS"
    },
    {
        "ref_id": "Ours",
        "year": 2026,
        "study_focus": "Explainable temporal degradation feature framework",
        "method_family": "Temporal feature engineering + ML + sequence baseline + SHAP",
        "dataset": "NASA C-MAPSS FD001-FD004",
        "tasks": "RUL regression + risk-stage classification",
        "uses_fd001_fd004": "Yes",
        "uses_risk_classification": "Yes",
        "uses_ablation": "Yes",
        "uses_xai": "Yes, SHAP feature and category analysis",
        "reported_strength": "Dual-task evaluation, ablation, bootstrap CI, SHAP, deep sequence comparison, reproducible pipeline",
        "limitation_gap": "Hybrid neural fusion is exploratory and not claimed as best-performing",
        "relation_to_our_work": "Proposed study",
        "source_hint": "This project"
    }
]

df = pd.DataFrame(rows)

out1 = TABLE_DIR / "related_work_comparison_draft.csv"
out2 = PAPER_TABLE_DIR / "related_work_comparison_draft.csv"
out3 = PAPER_NOTES_DIR / "related_work_gap_notes.md"

df.to_csv(out1, index=False)
df.to_csv(out2, index=False)

notes = """# Related Work Gap Notes

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
"""

out3.write_text(notes, encoding="utf-8")

print(f"Saved: {out1}")
print(f"Saved: {out2}")
print(f"Saved: {out3}")
print("STATUS: RELATED_WORK_DRAFT_READY")
