from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
TABLE_DIR = ROOT / "outputs" / "tables"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIG_DIR = ROOT / "outputs" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = TABLE_DIR / "ml_risk_classification_results.csv"

def plot_metric_bar(df, metric, filename, title, ylabel):
    pivot = df.pivot(index="model", columns="subset", values=metric)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = FIG_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_confusion_matrix(cm_path, out_path, title):
    cm = pd.read_csv(cm_path, index_col=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values)

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels([c.replace("pred_", "") for c in cm.columns], rotation=30, ha="right")
    ax.set_yticklabels([i.replace("true_", "") for i in cm.index])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    print("=" * 80)
    print("GENERATE RISK-STAGE CLASSIFICATION FIGURES")
    print("=" * 80)

    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Missing result file: {RESULT_PATH}")

    df = pd.read_csv(RESULT_PATH)

    best = df.loc[df.groupby("subset")["F1_Macro"].idxmax()].copy()
    best = best.sort_values("subset")
    best_out = TABLE_DIR / "best_risk_classification_by_subset.csv"
    best.to_csv(best_out, index=False)

    print("\n[Best risk-stage classifier by subset]")
    print(best[["subset", "model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "ROC_AUC_OVR", "PR_AUC_Macro"]].to_string(index=False))
    print(f"Saved: {best_out}")

    rank_df = df.copy()
    rank_df["F1_Macro_rank"] = rank_df.groupby("subset")["F1_Macro"].rank(method="dense", ascending=False).astype(int)
    rank_df["Accuracy_rank"] = rank_df.groupby("subset")["Accuracy"].rank(method="dense", ascending=False).astype(int)
    rank_df = rank_df.sort_values(["subset", "F1_Macro_rank"])
    rank_out = TABLE_DIR / "risk_classification_ranked_results.csv"
    rank_df.to_csv(rank_out, index=False)
    print(f"Saved: {rank_out}")

    avg_rank = rank_df.groupby("model")[["F1_Macro_rank", "Accuracy_rank"]].mean().sort_values("F1_Macro_rank")
    avg_rank_out = TABLE_DIR / "risk_classification_average_rank.csv"
    avg_rank.to_csv(avg_rank_out)
    print(f"Saved: {avg_rank_out}")

    plot_metric_bar(
        df,
        metric="F1_Macro",
        filename="fig_risk_classification_macro_f1.png",
        title="Risk-Stage Classification Comparison by Macro-F1",
        ylabel="Macro-F1"
    )

    plot_metric_bar(
        df,
        metric="Accuracy",
        filename="fig_risk_classification_accuracy.png",
        title="Risk-Stage Classification Comparison by Accuracy",
        ylabel="Accuracy"
    )

    plot_metric_bar(
        df,
        metric="ROC_AUC_OVR",
        filename="fig_risk_classification_roc_auc.png",
        title="Risk-Stage Classification Comparison by ROC-AUC",
        ylabel="ROC-AUC, One-vs-Rest"
    )

    ax = avg_rank["F1_Macro_rank"].sort_values(ascending=False).plot(kind="barh", figsize=(9, 5))
    ax.set_title("Average Macro-F1 Rank Across C-MAPSS Subsets")
    ax.set_xlabel("Average Rank, Lower is Better")
    ax.set_ylabel("Model")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    rank_fig = FIG_DIR / "fig_risk_classification_average_f1_rank.png"
    plt.savefig(rank_fig, dpi=300)
    plt.close()
    print(f"Saved: {rank_fig}")

    for _, row in best.iterrows():
        subset = row["subset"]
        model = row["model"]
        cm_path = METRICS_DIR / f"{subset}_{model}_confusion_matrix.csv"
        out_path = FIG_DIR / f"fig_confusion_matrix_{subset}_{model}.png"

        if cm_path.exists():
            plot_confusion_matrix(
                cm_path,
                out_path,
                title=f"Confusion Matrix: {subset} - {model}"
            )
        else:
            print(f"Missing confusion matrix: {cm_path}")

    print("\n[Final status]")
    print("STATUS: RISK_CLASSIFICATION_FIGURES_READY")

if __name__ == "__main__":
    main()
