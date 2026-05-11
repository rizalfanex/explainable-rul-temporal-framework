from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
TABLE_DIR = ROOT / "outputs" / "tables"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIG_DIR = ROOT / "outputs" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
BOOTSTRAPS = 1000
SEED = 42

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=BOOTSTRAPS, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))

    vals = np.asarray(vals)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "ci95_low": float(np.percentile(vals, 2.5)),
        "ci95_high": float(np.percentile(vals, 97.5)),
    }

def load_regression_predictions(subset, model):
    path = METRICS_DIR / f"{subset}_{model}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    return df

def load_classification_results():
    path = TABLE_DIR / "ml_risk_classification_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing classification result file: {path}")
    return pd.read_csv(path)

def regression_bootstrap():
    print("\n[Regression bootstrap CI]")
    baseline_path = TABLE_DIR / "best_ml_baseline_by_subset.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing best baseline table: {baseline_path}")

    best_df = pd.read_csv(baseline_path)
    rows = []

    for _, row in best_df.iterrows():
        subset = row["subset"]
        model = row["model"]

        pred_df = load_regression_predictions(subset, model)
        y_true = pred_df["y_true"].values
        y_pred = pred_df["y_pred"].values

        rmse_ci = bootstrap_ci(y_true, y_pred, rmse, seed=SEED)
        mae_ci = bootstrap_ci(y_true, y_pred, mae, seed=SEED + 1)

        out = {
            "subset": subset,
            "model": model,
            "RMSE": row["RMSE"],
            "RMSE_boot_mean": rmse_ci["mean"],
            "RMSE_boot_std": rmse_ci["std"],
            "RMSE_ci95_low": rmse_ci["ci95_low"],
            "RMSE_ci95_high": rmse_ci["ci95_high"],
            "MAE": row["MAE"],
            "MAE_boot_mean": mae_ci["mean"],
            "MAE_boot_std": mae_ci["std"],
            "MAE_ci95_low": mae_ci["ci95_low"],
            "MAE_ci95_high": mae_ci["ci95_high"],
        }
        rows.append(out)

        print(
            f"{subset} {model}: RMSE={row['RMSE']:.4f} "
            f"CI95=({rmse_ci['ci95_low']:.4f}, {rmse_ci['ci95_high']:.4f})"
        )

    out_df = pd.DataFrame(rows)
    out_path = TABLE_DIR / "stat_regression_bootstrap_ci.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

def ablation_improvement_summary():
    print("\n[Ablation improvement summary]")
    path = TABLE_DIR / "ablation_rul_results.csv"
    if not path.exists():
        print(f"Skipping ablation improvement because missing: {path}")
        return

    df = pd.read_csv(path)

    rows = []
    for subset in sorted(df["subset"].unique()):
        sub = df[df["subset"] == subset].copy()
        full = sub[sub["ablation"] == "A9_full_proposed"]
        raw = sub[sub["ablation"] == "A2_raw_sensors_only"]
        cycle = sub[sub["ablation"] == "A3_cycle_only"]
        rolling = sub[sub["ablation"] == "A6_rolling_only"]

        if full.empty:
            continue

        full_rmse = float(full["RMSE"].iloc[0])

        def improvement_against(name, ref_df):
            if ref_df.empty:
                return None
            ref_rmse = float(ref_df["RMSE"].iloc[0])
            return {
                "subset": subset,
                "proposed_RMSE": full_rmse,
                "reference": name,
                "reference_RMSE": ref_rmse,
                "absolute_RMSE_reduction": ref_rmse - full_rmse,
                "relative_RMSE_reduction_percent": ((ref_rmse - full_rmse) / ref_rmse) * 100.0,
            }

        for name, ref_df in [
            ("raw_sensors_only", raw),
            ("cycle_only", cycle),
            ("rolling_only", rolling),
        ]:
            item = improvement_against(name, ref_df)
            if item is not None:
                rows.append(item)

    out_df = pd.DataFrame(rows)
    out_path = TABLE_DIR / "stat_ablation_improvement_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"Saved: {out_path}")

def classification_summary():
    print("\n[Classification statistical summary]")
    df = load_classification_results()

    best = df.loc[df.groupby("subset")["F1_Macro"].idxmax()].copy()
    best = best.sort_values("subset")

    avg = (
        df.groupby("model")
        .agg(
            mean_Accuracy=("Accuracy", "mean"),
            std_Accuracy=("Accuracy", "std"),
            mean_F1_Macro=("F1_Macro", "mean"),
            std_F1_Macro=("F1_Macro", "std"),
            mean_ROC_AUC_OVR=("ROC_AUC_OVR", "mean"),
            std_ROC_AUC_OVR=("ROC_AUC_OVR", "std"),
        )
        .reset_index()
        .sort_values("mean_F1_Macro", ascending=False)
    )

    best_out = TABLE_DIR / "stat_classification_best_by_subset.csv"
    avg_out = TABLE_DIR / "stat_classification_average_by_model.csv"

    best.to_csv(best_out, index=False)
    avg.to_csv(avg_out, index=False)

    print("\nBest by subset:")
    print(best[["subset", "model", "Accuracy", "F1_Macro", "ROC_AUC_OVR", "PR_AUC_Macro"]].to_string(index=False))
    print("\nAverage by model:")
    print(avg.to_string(index=False))

    print(f"Saved: {best_out}")
    print(f"Saved: {avg_out}")

def main():
    print("=" * 80)
    print("STATISTICAL VALIDATION")
    print("=" * 80)

    regression_bootstrap()
    ablation_improvement_summary()
    classification_summary()

    print("\n[Final status]")
    print("STATUS: STATISTICAL_VALIDATION_READY")

if __name__ == "__main__":
    main()
