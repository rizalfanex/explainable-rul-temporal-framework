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
        "ref_id": "Lin2024_CATA_TCN",
        "year": 2024,
        "paper_title": "Channel attention and temporal attention based temporal convolutional network: A dual attention framework for remaining useful life prediction of aircraft engines",
        "venue": "Reliability Engineering & System Safety",
        "method_family": "TCN + channel attention + temporal attention",
        "dataset": "C-MAPSS and real flight data",
        "task": "RUL regression",
        "reported_scope": "C-MAPSS RUL prediction with attention-based temporal modeling",
        "xai_or_interpretability": "Attention-based feature/time weighting, but not full SHAP category analysis",
        "risk_stage_classification": "No",
        "ablation_or_validation": "Hyperparameter and method component evaluation reported",
        "main_strength": "Models sensor-channel importance and key temporal degradation points using a dual-attention TCN framework",
        "limitation_gap": "Focuses on RUL regression; does not jointly formulate degradation risk-stage classification or bootstrap uncertainty around model performance",
        "relation_to_our_work": "Our work uses explicit temporal degradation features, dual RUL + risk-stage tasks, SHAP feature-category interpretation, and bootstrap CI",
        "source_url": "https://www.sciencedirect.com/science/article/abs/pii/S147403462400020X"
    },
    {
        "ref_id": "Elsherif2025_SciRep_CAELSTM",
        "year": 2025,
        "paper_title": "A deep learning-based prognostic approach for predicting turbofan engine degradation and remaining useful life",
        "venue": "Scientific Reports",
        "method_family": "Convolutional autoencoder + attention-based LSTM",
        "dataset": "NASA C-MAPSS",
        "task": "RUL regression",
        "reported_scope": "Deep prognostic model for turbofan degradation and RUL prediction",
        "xai_or_interpretability": "Limited; attention improves sequence representation but SHAP-style explanation is not the central contribution",
        "risk_stage_classification": "No",
        "ablation_or_validation": "Compared with prior deep learning methods",
        "main_strength": "Uses autoencoder-like representation and attention-based sequence processing for RUL",
        "limitation_gap": "Mainly focuses on continuous RUL prediction and deep learning performance; lacks a separate operational risk-stage task",
        "relation_to_our_work": "Our study compares deep sequence baselines against explainable engineered temporal features and adds risk-stage classification",
        "source_url": "https://www.nature.com/articles/s41598-025-09155-z"
    },
    {
        "ref_id": "Wang2025_Sensors_DualAttention",
        "year": 2025,
        "paper_title": "A Deep-Learning Method for Remaining Useful Life Prediction of Aero-Engines Based on Dual Attention",
        "venue": "Sensors",
        "method_family": "Dual attention + CNN/channel attention",
        "dataset": "NASA C-MAPSS",
        "task": "RUL regression",
        "reported_scope": "Dual-attention deep learning for RUL prediction",
        "xai_or_interpretability": "Attention mechanism provides internal weighting but not necessarily post-hoc SHAP analysis",
        "risk_stage_classification": "No",
        "ablation_or_validation": "Deep model evaluation reported",
        "main_strength": "Combines CNN-based feature learning with attention to improve RUL estimation",
        "limitation_gap": "Emphasizes prediction accuracy; operational risk-stage classification and bootstrap confidence intervals are not central",
        "relation_to_our_work": "Our framework emphasizes transparent temporal feature groups, ablation, risk-stage classification, and SHAP",
        "source_url": "https://www.mdpi.com/1424-8220/25/2/497"
    },
    {
        "ref_id": "Ozcan2025_SciRep_InterpretableEnsemble",
        "year": 2025,
        "paper_title": "Interpretable ensemble remaining useful life prediction enables dynamic maintenance scheduling for aircraft engines",
        "venue": "Scientific Reports",
        "method_family": "Interpretable ensemble ML with SHAP",
        "dataset": "NASA C-MAPSS FD001-FD004",
        "task": "RUL regression",
        "reported_scope": "Strong FD001/FD003 performance and competitive FD002/FD004 results; SHAP highlights critical sensors and operational cycles",
        "xai_or_interpretability": "Yes, SHAP",
        "risk_stage_classification": "No",
        "ablation_or_validation": "Ensemble comparisons reported",
        "main_strength": "Combines ensemble learning with SHAP interpretability for dynamic maintenance scheduling",
        "limitation_gap": "Does not jointly evaluate a normal/warning/critical degradation risk-stage classification layer",
        "relation_to_our_work": "Closest to our XAI direction; our distinction is explicit temporal feature ablation, dual-task risk-stage classification, and bootstrap CI",
        "source_url": "https://www.nature.com/articles/s41598-025-23473-2"
    },
    {
        "ref_id": "Yu2025_HybridCNNBiLSTM_DualAttention",
        "year": 2025,
        "paper_title": "Remaining useful life prediction based on hybrid CNN-BiLSTM with dual attention mechanism",
        "venue": "Engineering Applications of Artificial Intelligence / Elsevier source",
        "method_family": "Hybrid CNN + BiLSTM + dual attention",
        "dataset": "NASA C-MAPSS",
        "task": "RUL regression",
        "reported_scope": "Parallel deep architecture for aircraft engine RUL",
        "xai_or_interpretability": "Attention-based weighting, but not SHAP-centered",
        "risk_stage_classification": "No",
        "ablation_or_validation": "Model comparison reported",
        "main_strength": "Addresses serial CNN-LSTM information loss through a parallel dual-attention architecture",
        "limitation_gap": "Deep model remains comparatively opaque and does not include risk-stage decision classification",
        "relation_to_our_work": "Our work provides a transparent temporal feature framework and evaluates deep baselines rather than relying only on deep fusion",
        "source_url": "https://www.sciencedirect.com/science/article/pii/S0142061525007008"
    },
    {
        "ref_id": "Andringa2025_CounterfactualRUL",
        "year": 2025,
        "paper_title": "Counterfactual explanations for remaining useful life prediction",
        "venue": "Engineering Applications of Artificial Intelligence / Elsevier source",
        "method_family": "Bayesian LSTM + counterfactual explanations",
        "dataset": "Predictive maintenance / aviation RUL context",
        "task": "RUL prediction with explainability",
        "reported_scope": "Uses counterfactual explanations to improve transparency in maintenance",
        "xai_or_interpretability": "Yes, counterfactual explanations",
        "risk_stage_classification": "No",
        "ablation_or_validation": "Interpretability and prediction comparison reported",
        "main_strength": "Focuses on explainable maintenance decisions using counterfactuals",
        "limitation_gap": "Counterfactual focus differs from feature-group ablation and SHAP category-level temporal degradation explanation",
        "relation_to_our_work": "Our work contributes complementary SHAP-based temporal feature/category interpretation and dual-task evaluation",
        "source_url": "https://www.sciencedirect.com/science/article/pii/S1566253525000454"
    },
    {
        "ref_id": "Xue2025_SciRep_MaintenanceThreshold",
        "year": 2025,
        "paper_title": "Predictive maintenance programs for aircraft engines based on remaining useful life prediction",
        "venue": "Scientific Reports",
        "method_family": "RUL-based maintenance threshold strategy",
        "dataset": "C-MAPSS",
        "task": "Maintenance decision / thresholding from RUL",
        "reported_scope": "Uses C-MAPSS to compute optimal maintenance thresholds and improve mission availability",
        "xai_or_interpretability": "Not central",
        "risk_stage_classification": "Threshold-oriented maintenance decision, but not three-stage ML classification",
        "ablation_or_validation": "Maintenance strategy comparison reported",
        "main_strength": "Connects RUL prediction to maintenance decision thresholds",
        "limitation_gap": "Focuses on threshold strategy rather than systematic temporal feature engineering, ML ablation, and SHAP explanation",
        "relation_to_our_work": "Our normal/warning/critical stage design provides a machine-learning risk layer complementary to threshold-based maintenance",
        "source_url": "https://www.nature.com/articles/s41598-025-19957-w"
    },
    {
        "ref_id": "Tang2026_PLOS_PartialSensorFailure",
        "year": 2026,
        "paper_title": "A deep learning framework for remaining useful life prediction of turbofan engines with partial sensor failure",
        "venue": "PLOS One",
        "method_family": "LSTM-GAN / generative regression under sensor failure",
        "dataset": "NASA C-MAPSS",
        "task": "RUL regression under partial sensor failure",
        "reported_scope": "Robust RUL prediction when sensor data are partially missing or damaged",
        "xai_or_interpretability": "Not central",
        "risk_stage_classification": "No",
        "ablation_or_validation": "Baseline comparisons under sensor-failure settings",
        "main_strength": "Addresses robustness to partial sensor failure using missing-parameter generation and RUL prediction",
        "limitation_gap": "Targets sensor failure robustness rather than explainable temporal feature groups and operational risk-stage classification",
        "relation_to_our_work": "Our work focuses on interpretable temporal degradation evidence and dual-task decision support; sensor failure robustness is future work",
        "source_url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0347312"
    },
    {
        "ref_id": "Ozcan2026_AIEDAM_ClassificationPdM",
        "year": 2026,
        "paper_title": "Predictive maintenance in aircraft engine maintenance using the C-MAPSS dataset: performance comparison and evaluation of machine learning classification algorithms",
        "venue": "AI EDAM",
        "method_family": "10 ML classifiers + 3 deep learning models",
        "dataset": "NASA C-MAPSS",
        "task": "Short-term binary failure classification",
        "reported_scope": "Predicts whether an engine will fail within the next 30 cycles; compares 10 ML and 3 DL models",
        "xai_or_interpretability": "No / not central",
        "risk_stage_classification": "Binary classification, not three-stage normal/warning/critical",
        "ablation_or_validation": "Wide-scope classification benchmark",
        "main_strength": "Broad classification benchmark for aircraft engine predictive maintenance",
        "limitation_gap": "Does not jointly perform continuous RUL regression and explainable feature-category SHAP analysis",
        "relation_to_our_work": "Our work extends the decision layer to three risk stages and combines it with RUL regression and temporal-feature interpretation",
        "source_url": "https://www.cambridge.org/core/journals/ai-edam/article/predictive-maintenance-in-aircraft-engine-maintenance-using-the-cmapss-dataset-performance-comparison-and-evaluation-of-machine-learning-classification-algorithms/93B83E8496618EC0DAE786CFEFF93909"
    },
    {
        "ref_id": "Lan2026_DeepResidualAttention",
        "year": 2026,
        "paper_title": "Engine remaining useful life prediction method based on deep residual network and attention mechanism",
        "venue": "Discover Computing",
        "method_family": "Deep residual network + attention",
        "dataset": "C-MAPSS FD001-FD004",
        "task": "RUL regression",
        "reported_scope": "Reports RMSE values for FD001-FD004 and average RMSE across all four subsets",
        "xai_or_interpretability": "Attention-based interpretability, but not SHAP-based category analysis",
        "risk_stage_classification": "No",
        "ablation_or_validation": "Ablation on attention and residual components",
        "main_strength": "Strong deep learning performance and generalization across all four C-MAPSS subsets",
        "limitation_gap": "Does not include a separate risk-stage classification task and does not focus on transparent temporal feature engineering",
        "relation_to_our_work": "Our work provides an interpretable feature-engineering alternative with dual-task evaluation and bootstrap CI",
        "source_url": "https://link.springer.com/article/10.1007/s10791-026-10000-8"
    },
    {
        "ref_id": "Ours2026",
        "year": 2026,
        "paper_title": "An Explainable Temporal Degradation Feature Framework for Remaining Useful Life Prediction and Risk Assessment of Industrial Systems",
        "venue": "This study",
        "method_family": "Temporal feature engineering + ML + deep sequence baseline + SHAP",
        "dataset": "NASA C-MAPSS FD001-FD004",
        "task": "RUL regression + normal/warning/critical risk-stage classification",
        "reported_scope": "Full reproducible pipeline with ablation, bootstrap CI, SHAP, deep sequence comparison, and diagrams",
        "xai_or_interpretability": "Yes, SHAP feature-level and category-level interpretation",
        "risk_stage_classification": "Yes, three-stage classification",
        "ablation_or_validation": "Yes, feature-group ablation and bootstrap confidence intervals",
        "main_strength": "Combines RUL prediction, risk-stage decision support, temporal feature ablation, statistical validation, and explainability",
        "limitation_gap": "Hybrid neural fusion is exploratory and not claimed as best-performing",
        "relation_to_our_work": "Proposed work",
        "source_url": "This project"
    }
]

df = pd.DataFrame(rows)

out1 = TABLE_DIR / "related_work_comparison_final.csv"
out2 = PAPER_TABLE_DIR / "related_work_comparison_final.csv"
out3 = PAPER_NOTES_DIR / "related_work_final_notes.md"

df.to_csv(out1, index=False)
df.to_csv(out2, index=False)

notes = []
notes.append("# Final Related Work Positioning Notes")
notes.append("")
notes.append("## Core Gap")
notes.append("")
notes.append("Recent C-MAPSS studies strongly emphasize deep RUL regression, attention models, robust sensor-failure modeling, classification benchmarks, and explainability. However, fewer studies jointly combine:")
notes.append("")
notes.append("1. Continuous RUL regression.")
notes.append("2. Three-stage degradation risk classification.")
notes.append("3. Explicit temporal degradation feature engineering.")
notes.append("4. Feature-group ablation.")
notes.append("5. Bootstrap confidence intervals.")
notes.append("6. SHAP feature-level and category-level interpretation.")
notes.append("7. Deep sequence baseline comparison.")
notes.append("")
notes.append("## Safe Novelty Statement")
notes.append("")
notes.append("This study does not claim to outperform all deep learning models on C-MAPSS. Instead, it contributes an explainable and reproducible temporal degradation feature framework that is competitive across FD001-FD004 and supports both RUL regression and degradation risk-stage assessment.")
notes.append("")
notes.append("## Important Claim Boundary")
notes.append("")
notes.append("Do not claim the exploratory hybrid feature-sequence model is the best model. The strongest empirical claim is that engineered temporal feature variants provide robust performance under complex subsets FD002-FD004, while GRU performs best on FD001.")
notes.append("")
notes.append("## Recommended Related Work Structure")
notes.append("")
notes.append("1. RUL prediction with deep learning and attention.")
notes.append("2. Explainable RUL prediction and SHAP/counterfactual approaches.")
notes.append("3. Classification-based predictive maintenance.")
notes.append("4. Research gap and positioning of this study.")
notes.append("")

out3.write_text("\n".join(notes), encoding="utf-8")

print(f"Saved: {out1}")
print(f"Saved: {out2}")
print(f"Saved: {out3}")
print("STATUS: RELATED_WORK_FINAL_READY")
