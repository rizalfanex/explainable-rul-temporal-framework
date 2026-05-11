from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = TABLE_DIR / "ml_baseline_rul_results.csv"

def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Missing result file: {RESULT_PATH}")

    df = pd.read_csv(RESULT_PATH)

    print("=" * 80)
    print("GENERATE BASELINE FIGURES AND TABLES")
    print("=" * 80)

    best = df.loc[df.groupby("subset")["RMSE"].idxmin()].copy()
    best = best.sort_values("subset")
    best_out = TABLE_DIR / "best_ml_baseline_by_subset.csv"
    best.to_csv(best_out, index=False)

    print("\n[Best baseline by subset]")
    print(best[["subset", "model", "MAE", "RMSE", "R2", "NASA_Score", "train_time_sec", "inference_time_sec"]].to_string(index=False))
    print(f"Saved: {best_out}")

    rank_df = df.copy()
    rank_df["RMSE_rank"] = rank_df.groupby("subset")["RMSE"].rank(method="dense", ascending=True).astype(int)
    rank_df["MAE_rank"] = rank_df.groupby("subset")["MAE"].rank(method="dense", ascending=True).astype(int)
    rank_df = rank_df.sort_values(["subset", "RMSE_rank"])
    rank_out = TABLE_DIR / "ml_baseline_ranked_results.csv"
    rank_df.to_csv(rank_out, index=False)
    print(f"Saved: {rank_out}")

    pivot_rmse = df.pivot(index="model", columns="subset", values="RMSE")
    pivot_rmse = pivot_rmse.loc[pivot_rmse.mean(axis=1).sort_values().index]

    ax = pivot_rmse.plot(kind="bar", figsize=(12, 6))
    ax.set_title("RUL Regression Baseline Comparison by RMSE")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    rmse_fig = FIG_DIR / "fig_baseline_rmse_comparison.png"
    plt.savefig(rmse_fig, dpi=300)
    plt.close()
    print(f"Saved: {rmse_fig}")

    pivot_mae = df.pivot(index="model", columns="subset", values="MAE")
    pivot_mae = pivot_mae.loc[pivot_mae.mean(axis=1).sort_values().index]

    ax = pivot_mae.plot(kind="bar", figsize=(12, 6))
    ax.set_title("RUL Regression Baseline Comparison by MAE")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    mae_fig = FIG_DIR / "fig_baseline_mae_comparison.png"
    plt.savefig(mae_fig, dpi=300)
    plt.close()
    print(f"Saved: {mae_fig}")

    pivot_time = df.pivot(index="model", columns="subset", values="train_time_sec")
    pivot_time = pivot_time.loc[pivot_time.mean(axis=1).sort_values().index]

    ax = pivot_time.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Training Time Comparison Across ML Baselines")
    ax.set_xlabel("Model")
    ax.set_ylabel("Training Time (seconds)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    time_fig = FIG_DIR / "fig_baseline_training_time_comparison.png"
    plt.savefig(time_fig, dpi=300)
    plt.close()
    print(f"Saved: {time_fig}")

    avg_rank = rank_df.groupby("model")[["RMSE_rank", "MAE_rank"]].mean().sort_values("RMSE_rank")
    avg_rank_out = TABLE_DIR / "ml_baseline_average_rank.csv"
    avg_rank.to_csv(avg_rank_out)

    ax = avg_rank["RMSE_rank"].sort_values(ascending=False).plot(kind="barh", figsize=(9, 5))
    ax.set_title("Average RMSE Rank Across C-MAPSS Subsets")
    ax.set_xlabel("Average Rank, Lower is Better")
    ax.set_ylabel("Model")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    rank_fig = FIG_DIR / "fig_baseline_average_rmse_rank.png"
    plt.savefig(rank_fig, dpi=300)
    plt.close()
    print(f"Saved: {rank_fig}")
    print(f"Saved: {avg_rank_out}")

    print("\n[Final status]")
    print("STATUS: BASELINE_FIGURES_READY")

if __name__ == "__main__":
    main()
