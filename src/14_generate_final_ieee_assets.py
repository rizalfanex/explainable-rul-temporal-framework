from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
PAPER_TABLE_DIR = ROOT / "paper" / "tables"
PAPER_FIG_DIR = ROOT / "paper" / "figures"

PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

def read_if_exists(path):
    return pd.read_csv(path) if path.exists() else None

def main():
    print("=" * 80)
    print("GENERATE FINAL IEEE ASSETS")
    print("=" * 80)

    baseline = read_if_exists(TABLE_DIR / "best_ml_baseline_by_subset.csv")
    ablation = read_if_exists(TABLE_DIR / "ablation_best_by_subset.csv")
    deep = read_if_exists(TABLE_DIR / "deep_sequence_best_by_subset.csv")
    hybrid = read_if_exists(TABLE_DIR / "hybrid_rul_results.csv")
    clf = read_if_exists(TABLE_DIR / "best_risk_classification_by_subset.csv")
    reg_ci = read_if_exists(TABLE_DIR / "stat_regression_bootstrap_ci.csv")
    shap_cat = read_if_exists(TABLE_DIR / "shap_feature_category_summary.csv")

    rows = []

    if baseline is not None:
        for _, r in baseline.iterrows():
            rows.append({
                "subset": r["subset"],
                "family": "Classical ML",
                "model": f"CatBoost ({r['model']})",
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "R2": r["R2"],
                "NASA_Score": r["NASA_Score"],
            })

    if ablation is not None:
        for _, r in ablation.iterrows():
            rows.append({
                "subset": r["subset"],
                "family": "Best Ablation",
                "model": r["ablation"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "R2": r["R2"],
                "NASA_Score": r["NASA_Score"],
            })

    if deep is not None:
        for _, r in deep.iterrows():
            rows.append({
                "subset": r["subset"],
                "family": "Deep Sequence",
                "model": r["model"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "R2": r["R2"],
                "NASA_Score": r["NASA_Score"],
            })

    if hybrid is not None:
        for _, r in hybrid.iterrows():
            rows.append({
                "subset": r["subset"],
                "family": "Hybrid Proposed",
                "model": r["model"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "R2": r["R2"],
                "NASA_Score": r["NASA_Score"],
            })

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(["subset", "RMSE"]).reset_index(drop=True)
    out = TABLE_DIR / "final_model_comparison_rul.csv"
    comparison.to_csv(out, index=False)
    comparison.to_csv(PAPER_TABLE_DIR / "final_model_comparison_rul.csv", index=False)
    print(f"Saved: {out}")

    if not comparison.empty:
        pivot = comparison.pivot_table(index="model", columns="subset", values="RMSE", aggfunc="first")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        ax = pivot.plot(kind="bar", figsize=(13, 7))
        ax.set_title("Final RUL Model Comparison Across C-MAPSS Subsets")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig_path = FIG_DIR / "fig_final_model_comparison_rmse.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved: {fig_path}")

    if clf is not None:
        clf.to_csv(PAPER_TABLE_DIR / "best_risk_classification_by_subset.csv", index=False)
    if reg_ci is not None:
        reg_ci.to_csv(PAPER_TABLE_DIR / "regression_bootstrap_ci.csv", index=False)
    if shap_cat is not None:
        shap_cat.to_csv(PAPER_TABLE_DIR / "shap_feature_category_summary.csv", index=False)

    figure_manifest = pd.DataFrame({
        "figure_file": sorted([p.name for p in FIG_DIR.glob("*.png")])
    })
    figure_manifest.to_csv(TABLE_DIR / "final_figure_manifest.csv", index=False)

    print("\n[Best RUL model per subset from final comparison]")
    if not comparison.empty:
        best = comparison.loc[comparison.groupby("subset")["RMSE"].idxmin()].sort_values("subset")
        best.to_csv(TABLE_DIR / "final_best_rul_model_by_subset.csv", index=False)
        print(best[["subset", "family", "model", "MAE", "RMSE", "R2", "NASA_Score"]].to_string(index=False))

    print("\n[Final status]")
    print("STATUS: FINAL_IEEE_ASSETS_READY")

if __name__ == "__main__":
    main()
