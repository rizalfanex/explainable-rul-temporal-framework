from pathlib import Path
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
FEATURE_DIR = ROOT / "data" / "processed" / "features"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEED = 42
TARGET = "RUL_capped"

def nasa_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0
    )
    return float(np.sum(score))

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, 125)

    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "R2": round(float(r2_score(y_true, y_pred)), 6),
        "NASA_Score": round(float(nasa_score(y_true, y_pred)), 6),
    }

def load_feature_metadata():
    path = TABLE_DIR / "feature_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {path}")
    return pd.read_csv(path)

def get_feature_sets(meta_df):
    by_cat = {}
    for category in sorted(meta_df["category"].unique()):
        by_cat[category] = meta_df.loc[meta_df["category"] == category, "feature"].tolist()

    op = by_cat.get("operational_setting", [])
    raw = by_cat.get("raw_sensor", [])
    cycle = by_cat.get("cycle_observable", [])
    lag = by_cat.get("lag", [])
    delta = by_cat.get("delta", [])
    pct = by_cat.get("percentage_change", [])
    roll_mean = by_cat.get("rolling_mean", [])
    roll_std = by_cat.get("rolling_std", [])
    roll_min = by_cat.get("rolling_min", [])
    roll_max = by_cat.get("rolling_max", [])
    roll_range = by_cat.get("rolling_range", [])

    rolling_all = roll_mean + roll_std + roll_min + roll_max + roll_range
    short_term = lag + delta + pct

    feature_sets = {
        "A1_operational_only": op,
        "A2_raw_sensors_only": raw,
        "A3_cycle_only": cycle,
        "A4_raw_plus_operational": raw + op,
        "A5_raw_plus_cycle": raw + cycle,
        "A6_rolling_only": rolling_all,
        "A7_full_without_cycle": op + raw + short_term + rolling_all,
        "A8_rolling_plus_cycle": rolling_all + cycle,
        "A9_full_proposed": op + raw + cycle + short_term + rolling_all,
    }

    clean_sets = {}
    for name, cols in feature_sets.items():
        seen = set()
        clean = []
        for c in cols:
            if c not in seen:
                clean.append(c)
                seen.add(c)
        clean_sets[name] = clean

    return clean_sets

def build_model():
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=False
    )

def run_ablation_for_subset(subset, feature_sets):
    train_path = FEATURE_DIR / f"{subset}_train_features.csv"
    test_path = FEATURE_DIR / f"{subset}_test_features.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df[TARGET].astype("float32")
    y_test = test_df[TARGET].astype("float32")

    rows = []

    for ablation_name, cols in feature_sets.items():
        if len(cols) == 0:
            print(f"  Skipping {ablation_name}: no features")
            continue

        X_train = train_df[cols].astype("float32")
        X_test = test_df[cols].astype("float32")

        print(f"  Training {subset} | {ablation_name} | features={len(cols)}")

        model = build_model()

        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train

        start_pred = time.perf_counter()
        pred = model.predict(X_test)
        inference_time = time.perf_counter() - start_pred

        metrics = evaluate(y_test, pred)

        row = {
            "subset": subset,
            "target": TARGET,
            "model": "CatBoost",
            "ablation": ablation_name,
            "num_features": int(len(cols)),
            **metrics,
            "train_time_sec": round(float(train_time), 6),
            "inference_time_sec": round(float(inference_time), 6),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        rows.append(row)

        model_path = MODEL_DIR / f"{subset}_CatBoost_{ablation_name}_rul.joblib"
        joblib.dump(
            {
                "model": model,
                "features": cols,
                "ablation": ablation_name,
                "target": TARGET,
            },
            model_path
        )

        pred_out = pd.DataFrame({
            "subset": subset,
            "unit_id": test_df["unit_id"].values,
            "cycle": test_df["cycle"].values,
            "ablation": ablation_name,
            "y_true": y_test.values,
            "y_pred": np.clip(pred, 0, 125),
        })
        pred_out.to_csv(TABLE_DIR.parent / "metrics" / f"{subset}_{ablation_name}_ablation_predictions.csv", index=False)

        print(
            f"    MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, "
            f"R2={row['R2']:.4f}, NASA={row['NASA_Score']:.2f}, "
            f"train={row['train_time_sec']:.2f}s"
        )

    return rows

def generate_figures(results_df):
    pivot = results_df.pivot(index="ablation", columns="subset", values="RMSE")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    ax = pivot.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Ablation Study: RMSE Across Temporal Feature Sets")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out_path = FIG_DIR / "fig_ablation_rmse_by_subset.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    pivot_mae = results_df.pivot(index="ablation", columns="subset", values="MAE")
    pivot_mae = pivot_mae.loc[pivot_mae.mean(axis=1).sort_values().index]

    ax = pivot_mae.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Ablation Study: MAE Across Temporal Feature Sets")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out_path = FIG_DIR / "fig_ablation_mae_by_subset.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    avg_df = (
        results_df.groupby("ablation")
        .agg(
            avg_RMSE=("RMSE", "mean"),
            avg_MAE=("MAE", "mean"),
            avg_R2=("R2", "mean"),
            avg_NASA_Score=("NASA_Score", "mean"),
            num_features=("num_features", "mean")
        )
        .reset_index()
        .sort_values("avg_RMSE")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(avg_df["num_features"], avg_df["avg_RMSE"])

    for _, row in avg_df.iterrows():
        label = row["ablation"].replace("_", " ")
        ax.annotate(label, (row["num_features"], row["avg_RMSE"]), fontsize=7)

    ax.set_title("Feature Count vs Average RMSE")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Average RMSE Across Subsets")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = FIG_DIR / "fig_ablation_feature_count_vs_rmse.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    avg_out = TABLE_DIR / "ablation_average_performance.csv"
    avg_df.to_csv(avg_out, index=False)
    print(f"Saved: {avg_out}")

def main():
    print("=" * 80)
    print("ABLATION STUDY - TEMPORAL FEATURE SETS FOR RUL REGRESSION")
    print("=" * 80)

    meta_df = load_feature_metadata()
    feature_sets = get_feature_sets(meta_df)

    print("\n[Feature sets]")
    for name, cols in feature_sets.items():
        print(f"{name}: {len(cols)} features")

    all_rows = []

    for subset in SUBSETS:
        print(f"\n[{subset}]")
        subset_rows = run_ablation_for_subset(subset, feature_sets)
        all_rows.extend(subset_rows)
        pd.DataFrame(all_rows).to_csv(TABLE_DIR / "ablation_rul_results_partial.csv", index=False)

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values(["subset", "RMSE", "MAE"]).reset_index(drop=True)

    out_csv = TABLE_DIR / "ablation_rul_results.csv"
    out_json = TABLE_DIR / "ablation_rul_results.json"

    results_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    best = results_df.loc[results_df.groupby("subset")["RMSE"].idxmin()].copy()
    best = best.sort_values("subset")
    best_out = TABLE_DIR / "ablation_best_by_subset.csv"
    best.to_csv(best_out, index=False)

    generate_figures(results_df)

    print("\n[Best ablation per subset by RMSE]")
    print(best[["subset", "ablation", "num_features", "MAE", "RMSE", "R2", "NASA_Score"]].to_string(index=False))

    print("\n[Final status]")
    print("STATUS: ABLATION_RUL_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
