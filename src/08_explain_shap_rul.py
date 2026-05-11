from pathlib import Path
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
FEATURE_DIR = ROOT / "data" / "processed" / "features"
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"
FIG_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MODEL_NAME = "CatBoost"
TARGET = "RUL_capped"
SEED = 42
MAX_EXPLAIN_ROWS = 3000
TOP_K = 25

def load_feature_columns():
    path = TABLE_DIR / "model_feature_columns.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature column file: {path}")
    cols = path.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in cols if c.strip()]
    if not cols:
        raise ValueError("Feature column list is empty.")
    return cols

def feature_category(feature_name: str) -> str:
    if feature_name.startswith("op_setting_"):
        return "operational_setting"
    if feature_name.startswith("sensor_") and "_roll" not in feature_name and "_lag" not in feature_name and "_delta" not in feature_name and "_pct" not in feature_name:
        return "raw_sensor"
    if feature_name.startswith("cycle_"):
        return "cycle_observable"
    if "_lag1" in feature_name:
        return "lag"
    if "_delta1" in feature_name:
        return "delta"
    if "_pct_change1" in feature_name:
        return "percentage_change"
    if "_roll" in feature_name and "_mean" in feature_name:
        return "rolling_mean"
    if "_roll" in feature_name and "_std" in feature_name:
        return "rolling_std"
    if "_roll" in feature_name and "_min" in feature_name:
        return "rolling_min"
    if "_roll" in feature_name and "_max" in feature_name:
        return "rolling_max"
    if "_roll" in feature_name and "_range" in feature_name:
        return "rolling_range"
    return "other"

def sample_dataframe(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=seed).copy()

def plot_top_shap_bar(top_df: pd.DataFrame, subset: str):
    plot_df = top_df.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    ax.set_title(f"Top SHAP Features for RUL Prediction ({subset}, CatBoost)")
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = FIG_DIR / f"fig_shap_bar_{subset}_CatBoost.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def explain_subset(subset: str, feature_cols: list):
    print(f"\n[{subset}] Loading model and features")

    model_path = MODEL_DIR / f"{subset}_{MODEL_NAME}_rul_capped.joblib"
    test_path = FEATURE_DIR / f"{subset}_test_features.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test feature file: {test_path}")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X = test_df[feature_cols].astype("float32")
    y = test_df[TARGET].astype("float32")

    explain_df = sample_dataframe(test_df, MAX_EXPLAIN_ROWS, SEED)
    X_explain = explain_df[feature_cols].astype("float32")

    print(f"Explain rows: {len(X_explain)}")
    print("Computing SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    shap_array = np.asarray(shap_values)
    if shap_array.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape for regression: {shap_array.shape}")

    mean_abs = np.abs(shap_array).mean(axis=0)

    shap_df = pd.DataFrame({
        "subset": subset,
        "model": MODEL_NAME,
        "target": TARGET,
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
    })

    shap_df["category"] = shap_df["feature"].apply(feature_category)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = np.arange(1, len(shap_df) + 1)

    top_df = shap_df.head(TOP_K).copy()
    plot_top_shap_bar(top_df, subset)

    per_subset_out = TABLE_DIR / f"shap_top_features_{subset}_CatBoost.csv"
    top_df.to_csv(per_subset_out, index=False)
    print(f"Saved: {per_subset_out}")

    # Save local explanation sample for traceability.
    local_out = METRICS_DIR / f"{subset}_CatBoost_shap_values_sample.csv"
    local_df = pd.DataFrame(shap_array, columns=feature_cols)
    local_df.insert(0, "subset", subset)
    local_df.insert(1, "sample_index", np.arange(len(local_df)))
    local_df.to_csv(local_out, index=False)
    print(f"Saved: {local_out}")

    return shap_df

def main():
    print("=" * 80)
    print("SHAP EXPLAINABILITY - RUL REGRESSION - CATBOOST")
    print("=" * 80)

    feature_cols = load_feature_columns()
    print(f"Loaded model features: {len(feature_cols)}")

    all_shap = []

    for subset in SUBSETS:
        shap_df = explain_subset(subset, feature_cols)
        all_shap.append(shap_df)

    all_df = pd.concat(all_shap, axis=0, ignore_index=True)

    all_out = TABLE_DIR / "shap_feature_importance_rul_all.csv"
    all_df.to_csv(all_out, index=False)
    print(f"\nSaved: {all_out}")

    top_all = all_df.sort_values(["subset", "rank"]).groupby("subset").head(TOP_K)
    top_out = TABLE_DIR / "shap_top_features_rul.csv"
    top_all.to_csv(top_out, index=False)
    print(f"Saved: {top_out}")

    category_summary = (
        all_df.groupby(["subset", "category"])["mean_abs_shap"]
        .sum()
        .reset_index()
        .sort_values(["subset", "mean_abs_shap"], ascending=[True, False])
    )

    category_out = TABLE_DIR / "shap_feature_category_summary.csv"
    category_summary.to_csv(category_out, index=False)
    print(f"Saved: {category_out}")

    # Category-level figure
    pivot = category_summary.pivot(index="category", columns="subset", values="mean_abs_shap").fillna(0)
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop(columns=["mean"])

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("SHAP Importance by Feature Category for RUL Prediction")
    ax.set_xlabel("Feature Category")
    ax.set_ylabel("Summed Mean Absolute SHAP Value")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    category_fig = FIG_DIR / "fig_shap_feature_category_summary.png"
    plt.savefig(category_fig, dpi=300)
    plt.close()
    print(f"Saved: {category_fig}")

    print("\n[Top 10 SHAP features per subset]")
    for subset in SUBSETS:
        print(f"\n{subset}")
        print(
            top_all[top_all["subset"] == subset]
            [["rank", "feature", "category", "mean_abs_shap"]]
            .head(10)
            .to_string(index=False)
        )

    print("\n[Final status]")
    print("STATUS: SHAP_RUL_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
